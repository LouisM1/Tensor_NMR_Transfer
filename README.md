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
| `configs/pretraining/` | New tensor-compatible E/F pretraining configurations. Initially empty. |
| `configs/tensor/` | New tensor NMR configurations. Initially empty. |
| `configs/reference_scalar/` | Existing scalar/E/F configurations, kept together as references. |
| `nmr_helpers/` | Existing importable compatibility, scalar-loss, transfer, ELoRA and training helpers. |
| `analysis_helpers/reference_scalar/` | Existing analysis routines, retained as references with project roots updated. |
| `scripts/reference_scalar/` | Existing scalar smoke checks, adapted to the new reference-config locations. |
| `scripts/check_environment.py` | Local imports, dependency versions, checkpoint inference and tensor-conversion checks. |
| `pretrained/Kavanagh-Med-9059603/` | Original E/F source model and its saved metadata/checkpoints. |
| `graph-pes-results/` | New training outputs; initially empty. |
| `experiments/` | New experiment notebooks and write-ups; initially empty. |
| `comparisons/` | New comparison outputs; initially empty. |
| `logs/` | New job logs; initially empty. |
| `provenance/` | Bootstrap source hashes and environment verification. |
| `train.sbatch` | Existing ARC launcher, with a new default job name and updated example config path. |
| `TENSOR_NMR_REQUIREMENTS.local.md` | Local short-term requirements checklist; deliberately ignored by Git. |

Data, pretrained weights, `.venv`, job logs and new model outputs are kept locally and ignored by Git. The repository tracks code, config references, dependency specifications and provenance. Historical NMR run directories, plots and notebooks remain in `NequIP_experiments`; they were not brought into this clean workspace.

## What still needs implementation

The inherited `nmr_helpers` are the working **scalar** baseline. In particular, the transfer and ELoRA helpers currently enforce a scalar target, and the source E/F checkpoint has a scalar-only final interaction layer. They must be extended and verified before symmetric-tensor production training. The new `configs/tensor/` folder intentionally contains no pretend-ready tensor config.

Keep derived symmetric-tensor labels separate from the existing data. Verify basis conversion, rotations, tensor-safe scaling/offsets and the exact transferred/fresh parameter boundary before running the new experiments. The local requirements checklist specifies the planned goals and sequence.

Scalar reference analysis expects historical result caches that are not copied here. Reference configs that resume older NMR runs likewise require those external checkpoints; use them as templates, not as an instruction to reproduce the old experiment catalogue. The original E/F source checkpoint references have been updated to `pretrained/`.

## ARC

The Mac `.venv` is not portable to ARC. The inherited launcher uses the existing Linux environment:

`/data/oums-atom-elec-ml/trin4011/envs/graph-pes-arc`

When ready, sync this project separately to `/data/oums-atom-elec-ml/trin4011/Tensor_NMR_Transfer/`, excluding `.venv/` and `.git/`. Submit from that directory so relative config/data paths and results stay inside the new project. No remote directory or Git remote has been created as part of this local bootstrap.

The launcher retains its existing resource defaults and training overrides; review them or override them at submission for each new experiment. No experiment should be launched solely because a scalar reference config exists.

## Plot conventions

Use plain descriptive titles, clear axes and legends, and RMSE/MAE/R² annotations on parity panels. Avoid question-based titles, promotional subtitles and prose footnotes.
