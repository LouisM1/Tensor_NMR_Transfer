"""Compare the two Experiment 03/04 backends on a two-structure training batch."""

import argparse
import os
import sys
from copy import deepcopy
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

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
    load_e3nn_state_dict,
)

from nmr_helpers.openequivariance_compatibility import canonical_e3nn_model


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
    parser.add_argument("--checkpoint", type=Path, help="Compare a legacy Experiment 04 checkpoint with its canonical e3nn conversion")
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

    frames = read("data/train.xyz", ":2")
    if args.device == "cpu":
        frames = [atoms[list((atoms.numbers == 14).nonzero()[0][:8])
                        + list((atoms.numbers == 8).nonzero()[0][:16])]
                  for atoms in frames]
    graphs = [AtomicGraph.from_ase(atoms, cutoff=6.0,
                property_mapping=baseline["data"]["train"]["property_map"])
              for atoms in frames]
    if args.checkpoint:
        fast = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        ref = canonical_e3nn_model(fast)
    else:
        models = []
        for raw in (baseline, accelerated):
            resolved = resolve_config_dict(raw, TrainingConfig)
            general = resolved["general"]
            configure_general_options(TorchConfig(**general["torch"]), general["seed"])
            _, config = instantiate_config_from_dict(resolved, TrainingConfig)
            model = parse_model(config.model)
            model.pre_fit_all_components(graphs)
            models.append(model)
        ref, fast = models
        load_e3nn_state_dict(fast, ref.state_dict(), strict=True)
        canonical_state = canonical_e3nn_model(fast).state_dict()
        assert ref.state_dict().keys() == canonical_state.keys()
        for key, value in ref.state_dict().items():
            if torch.is_tensor(value):
                torch.testing.assert_close(value, canonical_state[key], rtol=0, atol=0, msg=key)
        print("Matched initial e3nn function and parameter values", flush=True)
    assert sum(isinstance(layer.message_tensor_product, OpenEquivarianceMessageTensorProduct)
               for layer in fast.models["nequip"].layers) == 4
    cpu_prediction = fast.eval().predict(to_batch(graphs), ["energy", "forces"])
    cpu_prediction = {key:value.detach().cpu() for key,value in cpu_prediction.items()}

    with TemporaryDirectory() as directory:
        export = Path(directory) / "model.pt"
        torch.save(fast.eval(), export)
        restored = torch.load(export, map_location="cpu", weights_only=False)
        deploy_model(restored, Path(directory) / "lammps_model.pt")

    batch = to_batch(graphs).to(args.device)
    ref_prediction, ref_gradients = force_gradient(ref.to(args.device), batch)
    fast_prediction, fast_gradients = force_gradient(fast.to(args.device), batch)
    tol = dict(rtol=3e-3, atol=2e-3) if args.device == "cuda" else dict(rtol=3e-5, atol=3e-6)
    torch.testing.assert_close(ref_prediction["forces"], fast_prediction["forces"], **tol)
    # Evaluation and force-training use different autograd graph settings;
    # float32 force derivatives can differ by a few micro-eV/Angstrom.
    cpu_tol = tol if args.device == "cuda" else dict(rtol=3e-5, atol=3e-6)
    torch.testing.assert_close(cpu_prediction["forces"], fast_prediction["forces"], **cpu_tol)
    energy_atol = 0.03 if args.device == "cuda" else 1e-3
    torch.testing.assert_close(
        ref_prediction["energy"], fast_prediction["energy"], rtol=0, atol=energy_atol
    )
    torch.testing.assert_close(cpu_prediction["energy"], fast_prediction["energy"], rtol=0, atol=energy_atol)
    # Canonical export reparameterises final radial output rows. Transform the
    # associated gradients back into the canonical coordinates before comparison.
    if args.checkpoint:
        for index, layer in enumerate(fast.models["nequip"].layers):
            signs = layer.message_tensor_product._oeq_weight_to_e3nn.detach().cpu()
            mlp = layer.weight_generator.components[0][1]
            prefix = f"models.nequip.layers.{index}.weight_generator.components.0.1.linear_layers.{len(mlp.linear_layers)-1}."
            for suffix in ("weight", "bias"):
                key = prefix + suffix
                if key in fast_gradients:
                    fast_gradients[key] *= signs[:,None] if suffix == "weight" else signs
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
