"""Verify Experiments 05/06 and make two bounded CPU optimizer updates each.

Production configs/datasets/source weights are retained. W&B and post-fit
scoring are disabled only in this check. Test-set labels are never accessed.
"""

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/symmetric-transfer-mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/symmetric-transfer-cache")
os.environ["WANDB_MODE"] = "disabled"
sys.dont_write_bytecode = True

import torch
import nmr_helpers  # graph-pes registers the legacy e3nn constants loader.
import yaml
from ase.io import read
from e3nn import o3
from e3nn.io import CartesianTensor
from pytorch_lightning import Callback
from graph_pes.atomic_graph import AtomicGraph, to_batch
from graph_pes.config.shared import (TorchConfig, instantiate_config_from_dict,
                                     parse_loss, parse_model, resolve_config_dict)
from graph_pes.config.training import TrainingConfig
from graph_pes.data.datasets import file_dataset
import graph_pes.scripts.train as training
from graph_pes.scripts.utils import configure_general_options
from graph_pes.training.callbacks import SaveBestModel
from graph_pes.training.opt import Optimizer
from graph_pes.utils.sampling import SequenceSampler
from graph_pes.utils.nn import PerElementParameter, count_used_parameters
from nmr_helpers.elora import PathELoRA
from nmr_helpers.shared_head_training import train_from_config
from nmr_helpers.symmetric_transfer import BACKBONE_PARTS, from_energy_force_model

CONFIGS = ["05-SymmetricTensor-FullFineTune-N100", "06-SymmetricTensor-ELoRA-N100-R4"]
SOURCE = ROOT / "graph-pes-results/04-EF-Pretrain-TensorBackbone-OpenEquivariance-9265688/model-e3nn.pt"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def equal_states(a, b):
    if torch.is_tensor(a):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for key in a:
            equal_states(a[key],b[key])
    elif isinstance(a,(list,tuple)):
        assert type(a) is type(b) and len(a)==len(b)
        for left,right in zip(a,b):
            equal_states(left,right)
    else:
        assert a==b


def make_model(raw):
    resolved = resolve_config_dict(deepcopy(raw), TrainingConfig)
    configure_general_options(TorchConfig(**resolved["general"]["torch"]), resolved["general"]["seed"])
    _, config = instantiate_config_from_dict(resolved, TrainingConfig)
    return parse_model(config.model), config


def is_expected_trainable(name, elora):
    return not elora or name.startswith(("tensor_readout.", "scaler.")) or ".A." in name or ".B." in name


def symmetry(model, atoms):
    model.eval()
    basis = CartesianTensor("ij=ji")
    graph = AtomicGraph.from_ase(atoms, cutoff=6.0)
    with torch.no_grad():
        reference = basis.to_cartesian(model(graph)["tensor"])
    errors = []
    for rotation in (o3.rand_matrix(), torch.diag(torch.tensor([-1.0, 1.0, 1.0]))):
        rotated = atoms.copy()
        rotated.positions = atoms.positions @ rotation.numpy().T
        rotated.set_cell(atoms.cell.array @ rotation.numpy().T, scale_atoms=False)
        with torch.no_grad():
            actual = basis.to_cartesian(model(AtomicGraph.from_ase(rotated, cutoff=6.0))["tensor"])
        expected = rotation @ reference @ rotation.T
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=3e-3)
        errors.append(float((actual-expected).abs().max()))
    return errors


class TrainingAudit(Callback):
    def __init__(self, elora, indices, fit_graphs):
        self.elora, self.indices, self.fit_graphs = elora, indices, fit_graphs

    def on_train_start(self, trainer, task):
        model = task.model
        dataset = trainer.train_dataloader.dataset
        assert len(dataset) == 100
        assert list(map(int, dataset.graphs.structures.indices)) == self.indices
        valid = trainer.val_dataloaders
        if isinstance(valid, (tuple, list)):
            valid = valid[0]
        assert len(valid.dataset) == 50
        assert bool(model._has_been_pre_fit)
        self.before = {name:p.detach().clone() for name,p in model.named_parameters()}
        self.wrappers = {name:module for name,module in model.named_modules() if isinstance(module, PathELoRA)}
        assert len(self.wrappers) == (12 if self.elora else 0)
        for name, parameter in model.named_parameters():
            assert parameter.requires_grad == is_expected_trainable(name, self.elora), name
        y, z = to_batch(self.fit_graphs).properties["tensor"], to_batch(self.fit_graphs).Z
        for element in (8, 14):
            values = y[z == element]
            torch.testing.assert_close(model.scaler.offsets[element,0], values[:,0].mean())
            torch.testing.assert_close(model.scaler.scales[element,0], values[:,0].std(unbiased=False))
            torch.testing.assert_close(model.scaler.scales[element,1], values[:,1:].square().mean().sqrt())
        self.trainable_parameters = count_used_parameters(model, only_learnable=True)
        self.steps = 0

    def on_before_optimizer_step(self, trainer, task, optimizer):
        for name, parameter in task.model.named_parameters():
            if parameter.grad is not None:
                assert torch.isfinite(parameter.grad).all(), name
            if not parameter.requires_grad:
                assert parameter.grad is None, name
        for name, wrapper in self.wrappers.items():
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in [*wrapper.A, *wrapper.B]), name
            assert any(torch.count_nonzero(p.grad) for p in wrapper.B), name
            if self.steps:
                assert any(torch.count_nonzero(p.grad) for p in wrapper.A), name
        self.steps += 1

    def on_train_end(self, trainer, task):
        assert self.steps == trainer.global_step == 2
        after = dict(task.model.named_parameters())
        if self.elora:
            for name, parameter in after.items():
                if not parameter.requires_grad:
                    assert torch.equal(parameter, self.before[name]), name
            for prefix in self.wrappers:
                for factor in ("A", "B"):
                    assert any(not torch.equal(p,self.before[name]) for name,p in after.items()
                               if name.startswith(prefix+"."+factor+".")), (prefix,factor)
        else:
            for index in range(4):
                assert any(not torch.equal(p,self.before[name]) for name,p in after.items()
                           if name.startswith(f"layers.{index}.")), index
        assert any(not torch.equal(p,self.before[name]) for name,p in after.items()
                   if name.startswith("tensor_readout."))
        self.losses = {}
        for split in ("train", "valid"):
            assert torch.isfinite(trainer.callback_metrics[f"{split}/loss/total"])
            self.losses[split] = float(trainer.callback_metrics[f"{split}/loss/total"])
        for element in ("Si", "O"):
            for block in ("iso", "aniso"):
                assert torch.isfinite(trainer.callback_metrics[f"valid/metrics/{element}_{block}_rmse"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or Path(tempfile.mkdtemp(prefix="symmetric-transfer-smoke-"))
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float32)
    source_hash = sha(SOURCE)
    source = torch.load(SOURCE, map_location="cpu", weights_only=False).models["nequip"]
    paths = [ROOT / f"configs/tensor/{name}.yaml" for name in CONFIGS]
    configs = [yaml.safe_load(path.read_text()) for path in paths]
    baseline_path = ROOT / "configs/tensor/01-SymmetricTensor-Scratch-N100.yaml"
    baseline_hash = sha(baseline_path)
    baseline = yaml.safe_load(baseline_path.read_text())
    indices, selections = [], []
    for config in [baseline, *configs]:
        assert config["data"] == baseline["data"] and config["loss"] == baseline["loss"]
        assert config["data"]["test"] is None
        fitting = deepcopy(config["fitting"])
        fitting["optimizer"]["lr"] = .003
        assert fitting == baseline["fitting"]
        dataset = file_dataset(**config["data"]["train"], cutoff=6.0, pre_transform=False)
        selected = list(map(int, dataset.graphs.structures.indices))
        assert len(dataset) == len(set(selected)) == 100
        selections.append(dataset)
        indices.append(selected)
    assert indices[0] == indices[1] == indices[2]
    graphs = list(SequenceSampler(selections[0].graphs).sample_at_most(4))
    assert all(set(graph.properties) == {"tensor"} for graph in graphs)
    fitting_batch = to_batch(graphs)
    assert fitting_batch.properties["tensor"].shape[1] == 6
    models, parsed, rng_states = [], [], []
    for config in configs:
        model, parsed_config = make_model(config)
        assert not bool(model._has_been_pre_fit)
        for part in BACKBONE_PARTS:
            # ELoRA adds only wrappers/factors; its .base states must still match.
            actual = getattr(model, part).state_dict()
            if len(models) == 1:
                actual = {key.replace(".base.", "."):value for key,value in actual.items()
                          if ".A." not in key and ".B." not in key}
            equal_states(getattr(source, part).state_dict(), actual)
        assert torch.equal(model.scaler.offsets, torch.zeros_like(model.scaler.offsets))
        assert torch.equal(model.scaler.scales, torch.ones_like(model.scaler.scales))
        assert not hasattr(model, "energy_readout")
        models.append(model); parsed.append(parsed_config)
        rng_states.append(torch.random.get_rng_state())
    equal_states(models[0].tensor_readout.state_dict(), models[1].tensor_readout.state_dict())
    assert torch.equal(rng_states[0], rng_states[1])
    # Hash rejection happens before any backbone copying.
    bad, _ = make_model(baseline)
    before = deepcopy(bad.state_dict())
    try:
        from_energy_force_model(bad, SOURCE, expected_sha256="0"*64)
    except ValueError as error:
        assert "SHA256 mismatch" in str(error)
    else:
        raise AssertionError("A wrong source hash must be rejected")
    equal_states(before,bad.state_dict())
    for model in models:
        model.pre_fit_all_components(graphs)
    with torch.no_grad():
        initial = [model.eval()(graphs[0])["tensor"] for model in models]
    assert initial[0].shape == (len(graphs[0].Z),6)
    torch.testing.assert_close(initial[0],initial[1],rtol=2e-5,atol=1e-5)
    gradient_norms = []
    for model,config in zip(models,parsed):
        model.train(); captured = {}
        def capture(module, args, tensor):
            tensor.retain_grad(); captured["features"] = tensor
        handle = model.layers[-1].register_forward_hook(capture)
        prediction = model(graphs[0])
        value = parse_loss(config.loss)(model,graphs[0],prediction).loss_value
        value.backward();handle.remove()
        grad = captured["features"].grad
        scalar_channels = model.layers[-1].irreps_out.count("0e")
        norms = [float(grad[:,:scalar_channels].norm()),float(grad[:,scalar_channels:].norm())]
        assert min(norms)>0 and all(torch.isfinite(torch.tensor(norms)))
        gradient_norms.append(norms)
    print("PASS: exact source copying, matched N100 subsets/head/RNG, zero-adapter parity and 0e/2e gradients.",flush=True)

    # Use the production launcher's stable parameter ordering for the smoke.
    launcher = (ROOT/"train.sbatch").read_text()
    exec(launcher[launcher.index("def ordered_optimizer"):launcher.index('if sys.argv[1] !=')],globals())
    original_factory = training.trainer_from_config
    original_tester = training.test_with_lightning
    records = []
    for index,(name,raw,config) in enumerate(zip(CONFIGS,configs,parsed)):
        elora = index == 1
        probe = deepcopy(raw)
        probe["general"].update(root_dir=str(output), run_id=name)
        probe["wandb"] = None
        probe["fitting"]["max_n_pre_fit"] = 4
        probe["fitting"]["trainer_kwargs"].update(
            accelerator="cpu", max_epochs=1, limit_train_batches=8,
            limit_val_batches=2, enable_model_summary=False,
        )
        for split in ("train", "valid"):
            probe["data"][split]["pre_transform"] = False
        audit = TrainingAudit(elora, indices[0], graphs)
        def factory(config, output_dir):
            trainer = original_factory(config, output_dir)
            trainer.callbacks.append(audit)
            for callback in trainer.callbacks:
                if isinstance(callback, SaveBestModel):
                    callback.try_to_deploy = False  # Energy/LAMMPS export is unrelated to NMR.
            return trainer
        training.trainer_from_config = factory
        training.test_with_lightning = lambda *args, **kwargs: None
        try:
            train_from_config(probe)
        finally:
            training.trainer_from_config = original_factory
            training.test_with_lightning = original_tester
        directory = output/name
        best = torch.load(directory/"checkpoints/best.ckpt", map_location="cpu", weights_only=False)
        last = torch.load(directory/"checkpoints/last.ckpt", map_location="cpu", weights_only=False)
        saved = torch.load(directory/"model.pt", map_location="cpu", weights_only=False).eval()
        expected = {key[6:]:value for key,value in best["state_dict"].items() if key.startswith("model.")}
        equal_states(expected, saved.state_dict())
        assert best["global_step"] == last["global_step"] == 2
        for state in last["optimizer_states"][0]["state"].values():
            if "exp_avg" in state:
                assert torch.isfinite(state["exp_avg"]).all()
        for param_name,param in saved.named_parameters():
            assert param.requires_grad == is_expected_trainable(param_name,elora),param_name
        reconstructed,_ = make_model(raw)
        reconstructed.load_state_dict(expected,strict=True)
        for parameter in reconstructed.parameters():
            if isinstance(parameter,PerElementParameter):
                parameter.register_elements([8,14])
        rebuilt_optimizer = config.fitting.get_optimizer()(reconstructed)
        assert [group["param_names"] for group in rebuilt_optimizer.param_groups] == [
            group["param_names"] for group in last["optimizer_states"][0]["param_groups"]]
        rebuilt_optimizer.load_state_dict(last["optimizer_states"][0])
        assert rebuilt_optimizer.state
        reconstructed.eval()
        with torch.no_grad():
            reloaded = saved(graphs[0])["tensor"]
            rebuilt = reconstructed(graphs[0])["tensor"]
        assert reloaded.shape == initial[0].shape and torch.isfinite(reloaded).all()
        torch.testing.assert_close(reloaded,rebuilt,rtol=2e-5,atol=3e-4)
        atoms = read(ROOT/raw["data"]["train"]["path"], index=indices[0][0])
        symmetry_errors = symmetry(saved,atoms)
        records.append({
            "config":str(paths[index].relative_to(ROOT)),"config_sha256":sha(paths[index]),
            "method":"ELoRA" if elora else "full fine-tuning",
            "trainable_parameters":audit.trainable_parameters,"optimizer_steps":2,
            "adapter_operators":12 if elora else 0,"learning_rate":raw["fitting"]["optimizer"]["lr"],
            "final_feature_0e_2e_gradient_norms":gradient_norms[index],
            "rotation_reflection_max_abs_errors_ppm":symmetry_errors,
            "frozen_parameters_unchanged":True if elora else None,
            "adapter_factors_updated":True if elora else None,
            "all_four_backbone_layers_updated":True if not elora else None,
            "fresh_readout_updated":True,"model_export_matches_best":True,
            "state_and_optimizer_reload":True,"finite_train_validation_loss":True,
            "temporary_smoke_losses":audit.losses,
        })
        print(f"PASS {name}: two optimizer steps, correct freeze mask, tensor symmetry and checkpoint/optimizer reload.",flush=True)
    assert sha(SOURCE) == source_hash and sha(baseline_path) == baseline_hash
    report = {
        "passed":True,"source_checkpoint":str(SOURCE.relative_to(ROOT)),"source_sha256":source_hash,
        "matched_n100_with_experiment01":True,"training_subset_indices_zero_based":indices[0],
        "training_structures":100,"validation_structures":50,
        "training_file_sha256":sha(ROOT/baseline["data"]["train"]["path"]),
        "validation_file_sha256":sha(ROOT/baseline["data"]["valid"]["path"]),
        "matched_fresh_readout_and_rng":True,"zero_initialized_adapters_preserve_initial_predictions":True,
        "source_hash_guard":True,"fresh_shielding_scaler":True,"exact_backbone_state_copy":True,
        "no_EF_head_scale_offset_optimizer_or_prefit_flag_transferred":True,
        "held_out_test_accessed":False,"runs":records,
        "torch":torch.__version__,"temporary_outputs":str(output),
        "smoke_overrides":{
            "device":"cpu","epochs":1,"training_batches":8,"validation_batches":2,
            "pre_fit_structures":4,"pre_transform":False,"wandb":False,
            "post_fit_scoring":False,"LAMMPS_deployment":False,
        },
    }
    (output/"smoke-results.json").write_text(json.dumps(report,indent=2)+"\n")
    print(f"ALL SYMMETRIC TRANSFER CHECKS PASSED: {output/'smoke-results.json'}",flush=True)


if __name__ == "__main__":
    main()
