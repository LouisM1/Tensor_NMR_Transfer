"""Compare the two Experiment 03/04 backends on a two-structure training batch."""

import argparse
import hashlib
import json
import os
import sys
from copy import deepcopy
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

# Needed by deterministic CUDA matmuls; set before any CUDA context is used.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

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


def captured_prediction(model, batch):
    """Observe component energies and layer outputs without changing the result."""
    components, features, handles = {}, {}, []
    def component_hook(name):
        def capture(module, args, output):
            components[name] = output["local_energies"].detach().to("cpu", dtype=torch.float64)
        return capture
    def feature_hook(index):
        def capture(module, args, output):
            features[str(index)] = output.detach().cpu().clone()
        return capture
    for name, component in model.models.items():
        handles.append(component.register_forward_hook(component_hook(name)))
    for index, layer in enumerate(model.models["nequip"].layers):
        handles.append(layer.register_forward_hook(feature_hook(index)))
    try:
        prediction = model.predict(batch, ["energy", "forces"])
    finally:
        for handle in handles:
            handle.remove()
    return prediction, components, features


def force_gradient(model, batch):
    model.train()
    model.zero_grad(set_to_none=True)
    prediction, components, features = captured_prediction(model, batch)
    force_mse = (prediction["forces"] - batch.properties["forces"]).square().mean()
    force_mse.backward()
    gradients = {
        name: parameter.grad.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    return ({key:value.detach().cpu() for key,value in prediction.items()},
            gradients, components, features)


def energy_summary(prediction, components, batch):
    assignment = batch.batch
    if assignment is None:
        assignment = torch.zeros(len(batch.Z), dtype=torch.long)
    assignment = assignment.cpu()
    counts = torch.bincount(assignment).double()
    sums = {name:torch.stack([values[assignment == index].sum()
                            for index in range(len(counts))])
            for name,values in components.items()}
    stable = torch.stack(list(sums.values())).sum(0)
    normal = prediction["energy"].detach().cpu().double().reshape(-1)
    return {
        "atoms_per_structure": counts.tolist(),
        "energy_eV": normal.tolist(),
        "component_sums_float64_eV": {name:values.tolist() for name,values in sums.items()},
        "component_sum_float64_total_eV": stable.tolist(),
        "normal_minus_float64_sum_eV": (normal-stable).tolist(),
        "normal_minus_float64_sum_meV_per_atom": (1000*(normal-stable)/counts).tolist(),
    }


def comparison_diagnostics(a, b, a_components, b_components, a_features, b_features, batch):
    left = energy_summary(a,a_components,batch)
    right = energy_summary(b,b_components,batch)
    counts = torch.tensor(left["atoms_per_structure"], dtype=torch.float64)
    delta = a["energy"].double().reshape(-1)-b["energy"].double().reshape(-1)
    stable_delta = (torch.tensor(left["component_sum_float64_total_eV"], dtype=torch.float64)
                    -torch.tensor(right["component_sum_float64_total_eV"], dtype=torch.float64))
    force_delta = (a["forces"].double()-b["forces"].double())
    return {
        "energy_difference_eV": delta.tolist(),
        "energy_difference_meV_per_atom": (1000*delta/counts).tolist(),
        "float64_component_sum_difference_eV": stable_delta.tolist(),
        "float64_component_sum_difference_meV_per_atom": (1000*stable_delta/counts).tolist(),
        "forces_max_abs_eV_per_A": float(force_delta.abs().max()),
        "forces_RMSE_difference_eV_per_A": float(force_delta.square().mean().sqrt()),
        "component_local_energy_max_abs_eV": {name:float((a_components[name]-b_components[name]).abs().max())
                                              for name in a_components},
        "layer_features": {name:{
            "max_abs":float((a_features[name]-b_features[name]).abs().max()),
            "relative_L2":float((a_features[name]-b_features[name]).norm()/a_features[name].norm().clamp_min(1e-30))}
            for name in a_features},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--checkpoint", type=Path, help="Compare a legacy Experiment 04 checkpoint with its canonical e3nn conversion")
    parser.add_argument("--matmul-precision", choices=("highest", "high"), default="highest")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--report", type=Path, help="Save all diagnostic values and every check, including failures")
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
    # Both branches get the same explicit verification settings, including the
    # legacy branch which previously inherited PyTorch defaults. Training config
    # and acceptance thresholds are unchanged.
    torch.set_float32_matmul_precision(args.matmul_precision)
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(args.deterministic)
    settings = {
        "torch": torch.__version__, "device":args.device,
        "float32_matmul_precision":torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32":torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32":torch.backends.cudnn.allow_tf32,
        "deterministic_algorithms":torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config":os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "tf32_environment_override":os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"),
    }
    print("VERIFICATION_SETTINGS",json.dumps(settings),flush=True)
    cpu_batch = to_batch(graphs)
    cpu_prediction, cpu_components, cpu_features = captured_prediction(fast.eval(), cpu_batch)
    cpu_prediction = {key:value.detach().cpu() for key,value in cpu_prediction.items()}

    with TemporaryDirectory() as directory:
        export = Path(directory) / "model.pt"
        torch.save(fast.eval(), export)
        restored = torch.load(export, map_location="cpu", weights_only=False)
        deploy_model(restored, Path(directory) / "lammps_model.pt")

    batch = to_batch(graphs).to(args.device)
    ref_prediction, ref_gradients, ref_components, ref_features = force_gradient(ref.to(args.device), batch)
    fast_prediction, fast_gradients, fast_components, fast_features = force_gradient(fast.to(args.device), batch)
    diagnostics = {
        "cpu":energy_summary(cpu_prediction,cpu_components,cpu_batch),
        "e3nn_device":energy_summary(ref_prediction,ref_components,cpu_batch),
        "oeq_device":energy_summary(fast_prediction,fast_components,cpu_batch),
        "e3nn_vs_oeq":comparison_diagnostics(ref_prediction,fast_prediction,ref_components,fast_components,ref_features,fast_features,cpu_batch),
        "cpu_vs_oeq":comparison_diagnostics(cpu_prediction,fast_prediction,cpu_components,fast_components,cpu_features,fast_features,cpu_batch),
    }
    print("COMPATIBILITY_DIAGNOSTICS",json.dumps(diagnostics,sort_keys=True),flush=True)
    checks = []
    def check(name, function):
        try:
            function()
        except AssertionError as exc:
            checks.append({"name":name,"passed":False,"error":str(exc)})
            print(f"FAIL {name}: {exc}",flush=True)
        else:
            checks.append({"name":name,"passed":True})
            print(f"PASS {name}",flush=True)
    tol = dict(rtol=3e-3, atol=2e-3) if args.device == "cuda" else dict(rtol=3e-5, atol=3e-6)
    check("e3nn_vs_oeq_forces",lambda:torch.testing.assert_close(ref_prediction["forces"],fast_prediction["forces"],**tol))
    # Evaluation and force-training use different autograd graph settings;
    # float32 force derivatives can differ by a few micro-eV/Angstrom.
    cpu_tol = tol if args.device == "cuda" else dict(rtol=3e-5, atol=3e-6)
    check("cpu_vs_oeq_forces",lambda:torch.testing.assert_close(cpu_prediction["forces"],fast_prediction["forces"],**cpu_tol))
    energy_atol = 0.03 if args.device == "cuda" else 1e-3
    check("e3nn_vs_oeq_total_energy",lambda:torch.testing.assert_close(ref_prediction["energy"],fast_prediction["energy"],rtol=0,atol=energy_atol))
    check("cpu_vs_oeq_total_energy",lambda:torch.testing.assert_close(cpu_prediction["energy"],fast_prediction["energy"],rtol=0,atol=energy_atol))
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
    threshold = 5e-3 if args.device == "cuda" else 2e-5
    def check_gradients():
        assert relative_error <= threshold, relative_error
    check("force_loss_parameter_gradients",check_gradients)
    print(f"Force-loss gradient relative L2 difference: {relative_error:.3g}",flush=True)
    if args.device == "cuda":
        def check_kernel():
            assert _cuda_kernel.cache_info().currsize > 0
        check("native_cuda_kernel_used",check_kernel)
    record = {"checkpoint":str(args.checkpoint) if args.checkpoint else None,
              "checkpoint_sha256":hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() if args.checkpoint else None,
              "settings":settings,"diagnostics":diagnostics,"checks":checks,
              "force_loss_gradient_relative_L2":float(relative_error),
              "energy_atol_eV":energy_atol,"energy_rtol":0,
              "force_gradient_relative_L2_threshold":threshold,
              "passed":all(item["passed"] for item in checks)}
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(record,indent=2)+"\n")
        print(f"Diagnostic report: {args.report}",flush=True)
    if not record["passed"]:
        failed = [item["name"] for item in checks if not item["passed"]]
        raise AssertionError("Compatibility verification failed: "+", ".join(failed))

    if args.device == "cuda":
        assert _cuda_kernel.cache_info().currsize > 0
        print("CUDA kernels, energy/force parity, and force-loss backward passed.")
    else:
        print("CPU fallback, serialisation, deployment, and force-loss backward passed.")


if __name__ == "__main__":
    main()
