"""Compare the two Experiment 03/04 backends on a two-structure training batch."""

import argparse
from copy import deepcopy
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import yaml
from ase.io import read
from graph_pes.atomic_graph import AtomicGraph, to_batch
from graph_pes.config.shared import (
    TorchConfig,
    instantiate_config_from_dict,
    parse_model,
    resolve_config_dict,
)
from graph_pes.config.training import TrainingConfig
from graph_pes.scripts.utils import configure_general_options
from graph_pes.utils.lammps import deploy_model

from nmr_helpers.openequivariance_tp import (
    OpenEquivarianceMessageTensorProduct,
    _cuda_kernel,
)


def force_gradient(model, batch):
    model.train()
    prediction = model.predict(batch, ["energy", "forces"])
    force_mse = (prediction["forces"] - batch.properties["forces"]).square().mean()
    force_mse.backward()
    gradients = {
        name: parameter.grad.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    return {key: value.detach().cpu() for key, value in prediction.items()}, gradients


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Run the OpenEquivariance check on a CUDA compute node")
        print("CUDA:", torch.version.cuda, torch.cuda.get_device_name(0))
        print("OpenEquivariance:", version("openequivariance"))

    config_dir = Path("configs/tensor")
    baseline = yaml.safe_load((config_dir / "03-EF-Pretrain-TensorBackbone-Batch2.yaml").read_text())
    accelerated = yaml.safe_load(
        (config_dir / "04-EF-Pretrain-TensorBackbone-OpenEquivariance.yaml").read_text()
    )
    expected = deepcopy(baseline)
    model_key = "+nmr_helpers.tensor_ef_pretraining.TensorCompatibleEFNequIP"
    expected["model"]["nequip"][model_key]["use_openequivariance"] = True
    expected["wandb"]["name"] = "04-EF-Pretrain-TensorBackbone-OpenEquivariance"
    if accelerated != expected:
        raise AssertionError("Experiment 04 differs from 03 beyond the backend and run name")

    models = []
    frames = read("data/train.xyz", ":2")
    if args.device == "cpu":
        # Keep both species and periodic geometry for a quick local check.
        frames = [atoms[list((atoms.numbers == 14).nonzero()[0][:8])
                        + list((atoms.numbers == 8).nonzero()[0][:16])]
                  for atoms in frames]
    graphs = [
        AtomicGraph.from_ase(
            atoms, cutoff=6.0, property_mapping=baseline["data"]["train"]["property_map"]
        )
        for atoms in frames
    ]
    for raw in (baseline, accelerated):
        resolved = resolve_config_dict(raw, TrainingConfig)
        general = resolved["general"]
        configure_general_options(TorchConfig(**general["torch"]), general["seed"])
        _, config = instantiate_config_from_dict(resolved, TrainingConfig)
        model = parse_model(config.model)
        model.pre_fit_all_components(graphs)
        models.append(model)
    ref, fast = models
    assert sum(isinstance(layer.message_tensor_product, OpenEquivarianceMessageTensorProduct)
               for layer in fast.models["nequip"].layers) == 4

    ref_state, fast_state = ref.state_dict(), fast.state_dict()
    assert ref_state.keys() == fast_state.keys()
    for key, value in ref_state.items():
        if torch.is_tensor(value):
            torch.testing.assert_close(value, fast_state[key], rtol=0, atol=0, msg=key)
    fast.load_state_dict(ref_state, strict=True)
    ref.load_state_dict(fast_state, strict=True)
    print("Matched initial weights and checkpoint keys", flush=True)

    with TemporaryDirectory() as directory:
        export = Path(directory) / "model.pt"
        torch.save(fast.eval(), export)
        restored = torch.load(export, map_location="cpu", weights_only=False)
        deploy_model(restored, Path(directory) / "lammps_model.pt")

    batch = to_batch(graphs).to(args.device)
    ref_prediction, ref_gradients = force_gradient(ref.to(args.device), batch)
    fast_prediction, fast_gradients = force_gradient(fast.to(args.device), batch)
    tol = dict(rtol=3e-3, atol=2e-3) if args.device == "cuda" else dict(rtol=2e-5, atol=1e-7)
    torch.testing.assert_close(ref_prediction["forces"], fast_prediction["forces"], **tol)
    energy_atol = 0.03 if args.device == "cuda" else 1e-3
    torch.testing.assert_close(
        ref_prediction["energy"], fast_prediction["energy"], rtol=0, atol=energy_atol
    )
    assert ref_gradients.keys() == fast_gradients.keys() and ref_gradients
    baseline_gradient = torch.cat([x.ravel() for x in ref_gradients.values()])
    accelerated_gradient = torch.cat([x.ravel() for x in fast_gradients.values()])
    gradient_norm = baseline_gradient.norm()
    assert gradient_norm.isfinite() and gradient_norm > 0
    relative_error = (accelerated_gradient - baseline_gradient).norm() / gradient_norm
    assert relative_error <= (5e-3 if args.device == "cuda" else 2e-5), relative_error
    print(f"Force-loss gradient relative L2 difference: {relative_error:.3g}")

    if args.device == "cuda":
        assert _cuda_kernel.cache_info().currsize > 0
        print("CUDA kernels, energy/force parity, and force-loss backward passed.")
    else:
        print("CPU fallback, serialisation, deployment, and force-loss backward passed.")


if __name__ == "__main__":
    main()
