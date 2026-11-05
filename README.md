# Tensor NMR Transfer

Workspace for predicting symmetric magnetic shielding tensors and comparing direct training, full fine-tuning and ELoRA transfer from energy/force pretraining.

Bootstrapped on 3 November 2026 from the working `NequIP_experiments` project. No training jobs have been submitted and no tensor-specific model implementation is implied by this setup.

## Start locally

```bash
cd /Users/louismcauliffe/code/research/Tensor_NMR_Transfer
source .venv/bin/activate
python scripts/check_environment.py
```

In your editor, select `.venv/bin/python` as the interpreter and notebook kernel environment. The environment has its own installed packages and activation/command paths. It shares the normal base Python installation, but does not depend on the old project's `.venv`.

`requirements-lock.txt` records the exact installed package versions. The initial environment was copied from the known-working installation into a fresh virtual environment so that local package changes, if any, are retained. For package-version reconstruction on a compatible Mac:

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements-lock.txt
```

Use those reconstruction commands only when intentionally replacing/rebuilding the environment. A package reinstall reproduces the recorded versions, not any unrecorded source edits inside installed packages. The current copied environment is the verified baseline.

## Layout

| Directory/file | Purpose |
| --- | --- |
| `data/` | Independent copies of raw data, train/valid/test partitions, split indices and scalar-labelled copies. Original `ms` tensors are retained. |
| `configs/tensor/` | Tensor NMR and compatible E/F configurations, kept together. |
| `configs/reference_scalar/` | Existing scalar/E/F configurations, kept together as references. |
| `nmr_helpers/` | Existing importable compatibility, scalar-loss, transfer, ELoRA and training helpers. |
| `analysis_helpers/reference_scalar/` | Existing analysis routines, retained as references with project roots updated. |
| `scripts/reference_scalar/` | Existing scalar smoke checks, adapted to the new reference-config locations. |
| `scripts/check_environment.py` | Local imports, dependency versions, checkpoint inference and tensor-conversion checks. |
| `pretrained/Kavanagh-Med-9059603/` | Original bootstrap checkpoint location; currently absent. E/F weights are not required by the direct-training pilot. |
| `graph-pes-results/` | New training outputs; initially empty. |
| `experiments/` | New experiment notebooks and write-ups; initially empty. |
| `comparisons/` | New comparison outputs; initially empty. |
| `logs/` | New job logs; initially empty. |
| `provenance/` | Symmetric-data manifests and short CPU training audits. |
| `train.sbatch` | Existing ARC launcher, with a new default job name and updated example config path. |
| `TENSOR_NMR_REQUIREMENTS.local.md` | Local short-term requirements checklist; deliberately ignored by Git. |

The repository tracks the datasets, code, configs, dependency specifications and provenance. `.venv`, generated job logs, model outputs, local HTML work logs and the temporary requirements checklist are ignored by Git. Historical NMR run directories, plots and notebooks remain in `NequIP_experiments`; they were not brought into this clean workspace.

## What still needs implementation

The inherited `nmr_helpers` are the working **scalar** baseline. In particular, the transfer and ELoRA helpers currently enforce a scalar target, and the source E/F checkpoint has a scalar-only final interaction layer. They must be extended and verified before symmetric-tensor production training. The first direct-training tensor baseline is now implemented in `configs/tensor/01-SymmetricTensor-Scratch-N100.yaml`; E/F transfer and ELoRA still require tensor-specific extensions.

Keep derived symmetric-tensor labels separate from the existing data. Verify basis conversion, rotations, tensor-safe scaling/offsets and the exact transferred/fresh parameter boundary before running the new experiments. The local requirements checklist specifies the planned goals and sequence.

Scalar reference analysis expects historical result caches that are not copied here. Reference configs that resume older NMR runs likewise require those external checkpoints; use them as templates, not as an instruction to reproduce the old experiment catalogue. The original E/F source checkpoint references have been updated to `pretrained/`.

## ARC

The Mac `.venv` is not portable to ARC. The inherited launcher uses the existing Linux environment:

`/data/oums-atom-elec-ml/trin4011/envs/graph-pes-arc`

When ready, sync this project separately to `/data/oums-atom-elec-ml/trin4011/Tensor_NMR_Transfer/`, excluding `.venv/` and `.git/`. Submit from that directory so relative config/data paths and results stay inside the new project. No remote directory or Git remote has been created as part of this local bootstrap.

The launcher retains its existing resource defaults and training overrides; review them or override them at submission for each new experiment. No experiment should be launched solely because a scalar reference config exists.

## Plot conventions

Use plain descriptive titles, clear axes and legends, and RMSE/MAE/R² annotations on parity panels. Avoid question-based titles, promotional subtitles and prose footnotes.

## Symmetric shielding pilot

New files in `data/nmr_symm/{raw,train,valid,test}.xyz` retain all existing arrays and metadata and add:

- `MS_Symm_QM`: nine row-major Cartesian entries of `(ms + ms.T) / 2`, in ppm.
- `MS_Symm_Irreps`: six orthonormal e3nn coefficients (`0e + 2e`), in ppm. The scalar coefficient is `sqrt(3) * sigma_iso`; the other five represent the symmetric traceless tensor. The e3nn version and basis convention are recorded in `data/nmr_symm/manifest.json`.

The extractor refuses existing outputs. Reproduce only into a fresh output directory:

```bash
.venv/bin/python -m nmr_helpers.extract_ms_symmetric --output-dir data/nmr_symm_new
```

The first config is `configs/tensor/01-SymmetricTensor-Scratch-N100.yaml`: direct training from random initialisation, 100 seed-42 training structures, all 50 validation structures, four layers, 6 Å cutoff, 64 radial features, and 128 channels per requested hidden irrep type (`0e,0o,1e,1o,2e,2o`). The output is exactly `0e + 2e`; pruning retains the required output features. Not every requested hidden type is reachable in every early layer.

AdamW starts at 0.003, with weight decay 0.01, batch size 1, accumulation 4, ReduceLROnPlateau (patience 25, factor 0.8), early-stopping patience 300, and a 3,000-epoch ceiling. This is an initial recipe, not an optimised tensor model. Test data are excluded from training/configuration selection.

The loss averages four normalised contributions equally: Si isotropic, Si anisotropic, O isotropic, O anisotropic. Its fixed denominators come from the full 800-structure training pool and are additional information beyond the N100 fitting labels. Learnable model offsets/scales are pre-fitted on the selected N100 only. The model has one scalar offset per element, one scalar scale and one shared five-component anisotropic scale per element; scaling therefore preserves rotations. The callback reports separate Si/O isotropic and anisotropic RMSE in ppm. Anisotropic RMSE is `sqrt(mean(||Q_pred-Q_ref||_F² / 5))`.

Run locally (this starts production training and W&B logging):

```bash
.venv/bin/python scripts/train_symmetric.py
```

On ARC, the existing launcher accepts the new YAML:

```bash
sbatch train.sbatch configs/tensor/01-SymmetricTensor-Scratch-N100.yaml 01-SymmetricTensor-Scratch-N100
```

The W&B project is `tensor-nmr-transfer`, under the existing Oxford entity. Review/override ARC resources at submission as usual. No production training was launched during setup. The saved short CPU smoke audit is in `provenance/symmetric-setup/`.

The initial setup audit used learning rate 0.001. The current YAML uses 0.003; a further bounded CPU check at that rate completed two optimiser updates with finite validation and reloaded predictions. See `provenance/symmetric-setup/lr003-smoke-results.json`. These checks do not establish convergence or GPU memory requirements.

## Tensor-compatible E/F pretraining

Config: `configs/tensor/02-EF-Pretrain-TensorBackbone.yaml`.

This trains a fresh E/F source on all 800 training structures and the existing 50 validation structures. Only `QM_energy` and `QM_forces` are mapped into graphs; no shielding targets or held-out test structures are used. The network matches the current symmetric-tensor pilot: four interaction layers, 6 Å cutoff, 64 radial features, identical element ordering, 128 copies of the six requested hidden irrep types, and final features `128x0e + 128x2e`.

`nmr_helpers/tensor_ef_pretraining.py` provides `TensorCompatibleEFNequIP`. It inherits NequIP's energy summation, scaling and conservative force calculation, but takes its embedding/edge/message-passing modules from the same constructor used by the symmetric-tensor baseline. This guarantees matching backbone layout instead of relying on an unpruned energy model whose final features would differ from the running tensor pilot.

A standard scalar linear energy head would ignore the final `2e` outputs. The task-specific energy head therefore concatenates the 128 scalar features with 128 rotational invariants, one for each rank-two channel: `mean(component**2)` across its five components. A 256→128→1 SiLU MLP produces each atomic energy. These squared norms are invariant under rotations/reflections, and let the E/F objective train the final anisotropic channels. This is a deliberate change to the E/F readout, not a claim that unmodified NequIP trains all final tensor features. Forces are differentiated from total energy, not separately predicted.

For later transfer, copy the element embedding, initial node embedding, edge embedding and all four interaction layers. Start the NMR linear readout and its tensor-safe scales/offsets afresh. Discard the E/F invariant MLP, energy scale and energy offset. The strict state-copy check is part of the smoke audit; the existing scalar-only production transfer and ELoRA helpers still need extension for this source. Gradient flow establishes that the features are trainable by E/F, not that they will necessarily improve NMR accuracy.

The initial recipe uses AdamW at 0.003, weight decay 0.01, batch size 1 with accumulation 4, a 1,000-epoch ceiling, early-stopping patience 300 and ReduceLROnPlateau (patience 25, factor 0.8). Energy loss is MSE of per-atom-normalised structure energy error, weighted 1000; force-component MSE is weighted 100. Those loss choices follow the original E/F recipe; they are starting settings, not optimised weights. Pre-fitting uses the fitting set only. With fixed SiO2 composition, individual elemental energy offsets are not uniquely identifiable; their combined contribution is what the data constrain.

On ARC, from this project's root:

```bash
sbatch train.sbatch configs/tensor/02-EF-Pretrain-TensorBackbone.yaml 02-EF-Pretrain-TensorBackbone
```

The inherited launcher supplies GPU/batch/stopping overrides. Review GPU allocation and wall time at submission; larger E/F training can require resuming from `checkpoints/last.ckpt`. A time limit is distinct from convergence. No production E/F job was launched during local setup.

Run the bounded CPU check with `.venv/bin/python scripts/smoke_ef_pretraining.py`. It checks backbone compatibility, energy/force predictions, E/F gradients to both final irrep blocks, sampled rotation/reflection behaviour, two optimizer updates, saved weights and deployment. Its temporary overrides and results are recorded under `provenance/ef-pretraining-setup/`. It does not establish GPU memory requirements, convergence or NMR transfer accuracy.

## Experiment 04: OpenEquivariance for E/F pretraining

Experiment 04 repeats the fresh Experiment 03 recipe in float32 with batch
size 2 and accumulation 2. Only the four message tensor products use
OpenEquivariance on CUDA. CPU inference and TorchScript deployment use e3nn.
A persistent convention buffer is now saved with each wrapper; ordinary parameter names and shapes remain unchanged.

Copy `run-exp04.sbatch`, `train.sbatch`, the Experiment 04 YAML and the
updated E/F and OpenEquivariance helpers to ARC. From the project root on HTC:

```bash
mkdir -p logs
sbatch run-exp04.sbatch
```

This requests eight hours on one L40S in `htc/short`, with eight CPUs and
64,000 MB of host RAM. The compute job creates a temporary Python overlay that
uses the existing Conda packages and installs OpenEquivariance 0.3.0 within
that overlay. This release avoids the autocast API missing from our PyTorch
2.6 runtime. Setup time counts against the same allocation. The job imports
OpenEquivariance before preparing data, then starts training directly. If the node needs a specific CUDA module, pass
its full name in `TENSOR_NMR_CUDA_MODULE` at submission. Otherwise the
launcher looks for a matching module when `nvcc` is absent.

Check the output and error logs for environment setup and kernel compilation,
then confirm the saved `train-config.yaml` has batch size and accumulation
both set to 2. Compare steady-state time per optimiser update and peak memory
with Experiment 03. The optional `scripts/smoke_openequivariance.py` can
compare CUDA energy, force and gradient parity in a separate setup.


### Experiment 04 compatibility repair — 5 November 2026

The original OEQ 0.3.0 CUDA function and e3nn 0.4.4 CPU fallback differed in three Clebsch–Gordan coupling signs. The GPU training used one consistent convention; recovering that function does not require retraining. The historical run remains a recipe speed comparison, with a different initial function from the e3nn baseline at the same unmapped weights.

`nmr_helpers/openequivariance_tp.py` now records the external-weight convention. Untagged legacy Experiment 04 Python objects and state checkpoints recover the original OEQ convention automatically. CPU and TorchScript apply that mapping; CUDA maps from it into the installed OEQ coefficient tables. Path normalisation and coefficient compatibility are checked before a CUDA kernel is used. New wrappers use the e3nn convention on both devices. The saved convention is guarded by the pinned e3nn 0.4.4 version; upgrading that dependency requires an explicit migration. When importing an untagged e3nn baseline state into wrappers, use `load_e3nn_state_dict` explicitly; untagged legacy OEQ states and canonical e3nn states cannot otherwise be distinguished.

The original model/checkpoints are retained. Separate corrected exports are:

- `graph-pes-results/04-EF-Pretrain-TensorBackbone-OpenEquivariance-9265688/model-e3nn.pt`: ordinary e3nn model, appropriate as the source for symmetric tensor backbone transfer.
- `graph-pes-results/04-EF-Pretrain-TensorBackbone-OpenEquivariance-9265688/model-e3nn-lammps.pt`: corrected TorchScript deployment.
- `model-e3nn.json` alongside them: source/export hashes and conversion description.

The original `lammps_model.pt` retains its old compiled CPU fallback. Updating Python helpers does not rewrite that file; use `model-e3nn-lammps.pt` for deployment.

`nmr_helpers/openequivariance_compatibility.py` folds the convention signs into each radial MLP's final output weights and bias, then replaces the wrapper with ordinary e3nn. This is a deterministic reparameterisation; no fitting data or targets enter the conversion. Converted models need a fresh optimizer; the original Adam moments are not converted. Resume original E/F optimizer state with the convention-aware wrapper instead. Do not combine the new loader with the old notebook sign hooks; those hooks have been removed from the Experiment 04 notebook.

To create another export under new output names:

```bash
.venv/bin/python scripts/convert_exp04_checkpoint.py graph-pes-results/04-EF-Pretrain-TensorBackbone-OpenEquivariance-9265688/model.pt --output /tmp/experiment04-e3nn.pt
```

The converter refuses existing outputs. Local checks are reproducible with:

```bash
.venv/bin/python scripts/smoke_openequivariance.py --device cpu
.venv/bin/python scripts/smoke_openequivariance.py --device cpu --checkpoint graph-pes-results/04-EF-Pretrain-TensorBackbone-OpenEquivariance-9265688/model.pt
.venv/bin/python scripts/check_exp04_compatibility.py --report /tmp/experiment04-compatibility.json
```

The regression checks include force-loss parameter gradients, legacy and tagged state loading, model round-trips, eager/TorchScript predictions, agreement with archived validation/test predictions, idempotent canonical conversion and strict copying of the four-layer backbone into the symmetric tensor model. Native CUDA has not been retested locally: this Mac has no CUDA, and ARC SSH authentication was unavailable. On an ARC CUDA allocation with OEQ 0.3.0 installed, run both smoke commands with `--device cuda`; they compare CPU/GPU energies and forces and the gradients used in force training.


The existing ARC launcher also has a verification-only mode. After copying the updated helpers, scripts, YAML and `run-exp04.sbatch` to the existing ARC project, submit from that directory:

```bash
sbatch --clusters=htc --partition=short --time=00:30:00 --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64000M --gres=gpu:l40s:1 --job-name=04-OEQ-compat --export=ALL,TENSOR_NMR_VERIFY_ONLY=1 run-exp04.sbatch
```

This uses the working job-local OEQ overlay, checks both new-model and legacy-checkpoint CUDA parity, then exits without launching training. It does not alter the original run. The local repair audit is in `provenance/experiment-04-compatibility/repair-validation.json`.
