# AGENTS.md — Operational Best Practices

This document defines the working rules, technical guardrails, and end-to-end research workflow for this repository. It is intentionally **operational and evergreen**: do not add transient task lists, current run statuses, or progress notes here (those live in Obsidian and `provenance/`).

> **Portability note:** Sections 1–6 apply across sibling NMR transfer repositories (including `Foundation_NMR_Transfer` and future cloud-compute setups). Only **Section 7** is specific to this repository's stack and paths.

---

## 1. Project Context

- **Scientific goal:** Predict per-atom solid-state NMR tensors in amorphous SiO₂ (e.g. symmetric magnetic shielding `0e + 2e` and electric field gradient `2e`) using equivariant graph neural networks, and evaluate how effectively pretraining on energies and forces (in-house or foundation models) reduces the number of expensive DFT-NMR calculations required.
- **Core comparison:** Direct training from scratch vs. transfer methods (frozen head, last interaction layer + head, path-wise ELoRA, full fine-tuning) across nested training-set sizes (`N = 10, 50, 100, 200, 800`).
- **Execution model:** Develop and validate code locally on a feature/experiment branch → run production GPU training on Oxford ARC (or cloud GPUs when model scale/wall-time requires it) → copy run outputs back locally → run reproducible analysis and figures locally → log results to Git and Obsidian.

---

## 2. Git, Branching & Artifact Policy

- **Branch discipline:** Always work on a dedicated experiment or feature branch. Never commit directly to `main`.
- **Committing:** Commit logical, self-contained milestones automatically (e.g. validated helper/config changes, smoke receipts, pulled training outputs, executed analysis notebooks/figures) with concise, descriptive commit messages.
- **Pushing:** **Always ask the user before running `git push`.**
- **What goes into Git:**
  - **Track:** source code, YAML configs, SLURM/compute launchers, `provenance/` manifests and smoke receipts, completed run outputs copied back from compute (see Section 7), analysis notebooks (with executed outputs), prediction caches, metric CSVs, and exported figures.
  - **Do not track:** `.venv/`, `__pycache__/`, SLURM `.out`/`.err` logs, W&B local scratch caches, `.env*`, local HTML scratch logs, and temporary local checklists.
- **Immutability:** Never overwrite raw datasets, completed production run directories, or historical model exports. Write derived targets, converted checkpoints, and re-runs to fresh paths with SHA-256 provenance manifests.

---

## 3. End-to-End Workflow: Local → Compute → Local Analysis

### Stage A — Local Development & Pre-Flight Checks
1. **Use the repo's own environment** (`.venv/bin/python`), never system Python or another project's virtual environment. Confirm dependencies match the lockfile before making changes.
2. **Pass all correctness gates locally on CPU before any GPU job:**
   - **Input & split integrity:** Verify dataset and source-checkpoint SHA-256 hashes, exact nested subset membership (`N10 ⊂ N50 ⊂ N100 ⊂ N200 ⊂ N800`), and `test: null` in every training config.
   - **Freeze masks & parameter counts:** Verify `requires_grad` for every named parameter, confirm zero gradients on frozen parameters, and confirm expected modules actually update after optimizer steps.
   - **Selected-`N` pre-fitting:** Confirm learned scales/offsets are initialised strictly from the selected `N` training structures (never the full pool, validation, or test).
   - **O(3) equivariance:** Verify predicted symmetric tensors transform as $S_{\text{rot}} = R S R^\top$ under random rotations and inversion applied consistently to atomic positions and periodic cells.
   - **Export & resume round-trip:** Run a bounded multi-step smoke fit, verify `model.pt` matches `best.ckpt` bit-for-bit, confirm finite weights/losses/gradients, and exercise the exact production resume code path (optimizer state, scheduler, early-stopping counters, best-validation threshold).

### Stage B — Offloading to Remote Compute (ARC Default; Cloud Ready)
1. **Sync code & configs to ARC:**
   - Use `rsync` from the local project root to the remote project root on `/data`, excluding `.venv/`, `.git/`, and `__pycache__/`.
   - Never run training, heavy preprocessing, or long `wandb sync` jobs on ARC login nodes (login nodes enforce a 1-hour CPU limit and run a different OS image).
2. **Dry-run & smoke-gate every production sweep:**
   - Validate SLURM requests first with `sbatch --test-only`.
   - Gate every multi-run Slurm array behind a short **CUDA smoke job** (`--dependency=afterok:<SMOKE_ID> --kill-on-invalid-dep=yes`) that writes a fingerprinted `smoke-results.json` receipt. Array tasks must verify the receipt fingerprint matches current code/configs/data before starting.
   - Record submitted job IDs in `provenance/` (guard against accidental duplicate submissions with `noclobber`).
3. **Node hygiene & logging:**
   - Point transient caches (`XDG_CACHE_HOME`, `MPLCONFIGDIR`, `TORCH_EXTENSIONS_DIR`) to node-local `$TMPDIR` (cleaned up on exit); keep checkpoints, metrics, and W&B files on persistent `/data` storage.
   - Always log durable per-epoch CSV metrics locally alongside W&B so a network or W&B hiccup never loses training history.
   - If W&B runs offline, upload from a compute node (e.g. a dedicated sync job), never the login node.
4. **Wall-time & checkpoint resumption:**
   - Treat wall-time expiry as a clean pause, not a failure. Resume from `last.ckpt` using the repo's verified resume script — never re-initialise scalers or lose Adam moments/early-stopping counters.
   - Distinguish **incomplete/timed-out** runs clearly (e.g. move into `incomplete/` and exclude from headline plots until finished) from converged runs.
5. **Cloud GPU portability (when scale exceeds ARC):**
   - Keep training, validation, and evaluation scripts scheduler-agnostic: use relative project paths, explicit `--device`, self-contained manifests, and receipt verification so the exact same Python entrypoints run unchanged on cloud GPU instances.

### Stage C — Pull Results Back & Analyse Locally
1. **Copy run outputs back** from remote compute into the local results folder (`graph-pes-results/` or repo equivalent) using `rsync`, preserving configs, checkpoints, exported `model.pt`, `epoch-metrics.csv`, and completion audits.
2. **Run reproducible evaluation locally:**
   - Validate every run's config, source checkpoint SHA-256, and `model.pt == best.ckpt` parity *before* reading held-out test targets.
   - Cache per-atom predictions (`.npz`) stamped with `(model_sha256, data_sha256, method, n, split, device)` so downstream notebooks and figure regenerations never silently mix stale caches or repeat inference unnecessarily.
   - Keep analysis notebooks deterministic and reproducible from saved predictions; no model fitting, calibration, or checkpoint selection happens during test evaluation.
3. **Commit & document:** Commit the pulled run outputs, analysis notebook, prediction caches, metrics CSVs, and figures to the active branch, then update Obsidian (Section 6).

---

## 4. Scientific, Data & Physics Guardrails

- **Strict test-set isolation:** Held-out test structures (`150` structures) are evaluated *only* on fixed, validation-selected recipes. Never use test scores to tune learning rates, pick seeds, choose checkpoints, or adjust loss weights.
- **Aligned splits & independent starts:**
  - Maintain aligned `800 / 50 / 150` (train / valid / test) partitions across E/F and NMR tasks.
  - Every `(method, N)` run starts independently from the same verified pretrained source checkpoint (pinned by SHA-256) with a fresh head and fresh optimizer. Never warm-start a larger-`N` NMR run from a smaller-`N` NMR checkpoint.
- **Tensor targets & equivariant scaling:**
  - **Magnetic shielding (`0e + 2e`, 6 components):** Symmetrise $S = (\text{ms} + \text{ms}^\top)/2$ in absolute ppm (no chemical-shift referencing). Account for basis normalisation: the `0e` irrep coefficient is $\sqrt{3}\,\sigma_{\text{iso}}$, and the five `2e` coefficients are an orthonormal representation of the traceless symmetric tensor $Q = S - \sigma_{\text{iso}} I$.
  - **EFG (`2e`, 5 components):** Symmetric and traceless; no `0e` channel and no anisotropic offset.
  - **Equivariant affine scaling:** Per-element scalar offset on `0e` only; one shared scale per element across all five components of a `2e` block; zero fixed anisotropic offset.
  - **Loss balancing:** Normalise element (Si/O) and irrep-block (`0e`/`2e`) contributions explicitly using fixed training-pool statistics so neither atom stoichiometry nor tensor dimension dominates gradients.
- **Methodological honesty:**
  - Report single-seed results explicitly as single-seed; do not overclaim small differences that could sit within seed-to-seed variance.
  - Report Si and O metrics separately alongside element-balanced scores.
  - Retain all atoms in headline metrics — never silently clip, drop outliers, or align anisotropy signs.
  - Distinguish practical convergence (e.g. LR decay plateau) from formal early stopping, and record best/final epochs and stopping reasons.

---

## 5. Evaluation, Metrics & Plotting Standards

### Physical NMR Parameters (Soprano)
- Always instantiate a **fresh `ase.Atoms` object** per tensor set when calling Soprano getters so cached diagonalisations never leak across tensors.
- Compute DFT references directly from original stored `ms` / `efg` tensors and assert agreement with targets reconstructed from irreps.
- **Headline shielding metrics:**
  - Isotropic shielding $\sigma_{\text{iso}}$ (ppm).
  - **Absolute Haeberlen anisotropy $|\Delta\sigma|$** (ppm) and **asymmetry $\eta$** (dimensionless) as primary tensor-shape metrics. *Why:* near $\eta \approx 1$, tiny errors swap which principal axis is labelled $zz$ and flip the sign of signed $\Delta\sigma$, inflating RMSE by 4–8×. Keep Maryland **skew $\kappa$** and signed $\Delta\sigma$ (with sign-flip counts) as secondary diagnostics.
- **Headline EFG metrics:** $|C_q|$ (MHz, primarily $^{17}\text{O}$ since $^{29}\text{Si}$ has spin $1/2$), $\eta_Q$, and $V_{zz}$.
- **Element-balanced standardised MSE:**
  $$\text{Balanced MSE} = \frac{1}{2}\left(\frac{\text{MSE}_{\text{Si}}}{\text{Var}_{\text{Si}}} + \frac{\text{MSE}_{\text{O}}}{\text{Var}_{\text{O}}}\right)$$
  using DFT variances computed strictly on the full 800-structure training pool (a value of `1.0` equals predicting the training-pool mean).

### Figures & Tables
- **Titles & copy:** Plain, descriptive titles; essential axis labels with physical units; clean legends. Avoid question-based titles, promotional subtitles, and prose footnotes.
- **Consistent method palette & markers across studies:**
  | Method | Label | Hex colour | Marker |
  | --- | --- | --- | --- |
  | `full` | Full fine-tuning | `#7548A3` | `D` |
  | `elora` | ELoRA | `#008C83` | `o` |
  | `last` | Last layer + NMR head | `#377BB5` | `s` |
  | `head` | NMR head only | `#7D858D` | `v` |
  | `scratch` | Direct training | `#D18B35` | `^` |
- **Learning curves:** Logarithmic x-axis (`N` ticks at `10, 50, 100, 200, 800`) and logarithmic y-axis; 3-panel layout per parameter (`Silicon`, `Oxygen`, `Balanced error`).
- **Parity plots:** Equal aspect ratio, dashed identity line, tiny circular scatter points (`s=6`, `alpha=0.19`, no outlines, `rasterized=True`), and a compact annotation box showing `RMSE`, `MAE`, and `R²`.
- **Exports:** Save every publication figure as `.png` (high DPI) and `.pdf` (plus `.svg` for learning curves), alongside machine-readable `.csv` metric tables.

---

## 6. Obsidian Vault Documentation

- **Vault path:** `/Users/louismcauliffe/Library/CloudStorage/GoogleDrive-lmcauliffe04@gmail.com/My Drive/Louis' Vault`
- **When to update:** Automatically update the vault after completing a major training run, evaluation/analysis, or methodological decision.
- **What to update:**
  1. **Experiment note** in `Papers & Research/Experiments/<category>/` (match config numbering; include date, tags, repo location, *Hypothesis*, *Outline*, *Results* with exact metrics/tables and embedded figures if generated, *Conclusion*, *Next hypothesis*, and links to the notebook & results folder).
  2. **Index table** in `Papers & Research/Experiments/Experiments contents.md` (concise one-sentence summary).
  3. **Daily/milestone row** in `Research Log.md` linking to the experiment note.
- **Writing style:** Clear, direct, scientifically cautious prose — state exact numbers, limitations (e.g. single-seed, untuned LR, timed-out runs), and avoid hype or filler.

---

## 7. Repository-Specific Configuration (`Tensor_NMR_Transfer`)

> *When copying `AGENTS.md` to another repo (e.g. `Foundation_NMR_Transfer`), replace only this section.*

### Environment & Compute Paths
- **Local Python:** `.venv/bin/python` (verified via `.venv/bin/python scripts/check_environment.py` against `requirements-lock.txt`).
- **Pinned core stack:** `graph-pes==1.0.2`, `e3nn==0.4.4`, `pytorch-lightning==2.5.1`, `soprano==0.11.4`.
- **Sibling repo boundary:** Because `graph-pes 1.0.2` pins `e3nn 0.4.4` while `nequip` foundation models require `e3nn >= 0.5.6`, foundation-model work lives in `../Foundation_NMR_Transfer`. Keep in-house `graph-pes` backbone, E/F, shielding, and EFG work in this repository.
- **ARC cluster:** Oxford ARC HTC (`--clusters=htc --partition=short`, default `1× L40S`, `8 CPUs`, `64000M` RAM, up to `12:00:00` on `short`).
- **ARC paths:**
  - Project root: `/data/oums-atom-elec-ml/trin4011/Tensor_NMR_Transfer/`
  - Conda env: `/data/oums-atom-elec-ml/trin4011/envs/graph-pes-arc` (loaded after `module purge && module load Anaconda3/2025.06-1`)
- **W&B:** entity `trin4011-university-of-oxford`, project `tensor-nmr-transfer` (HTC compute nodes reach `api.wandb.ai`; offline runs can be uploaded via `sbatch sync-wandb07.sbatch`).

### Key Repository Layout
| Path | Role |
| --- | --- |
| `data/` & `data/nmr_symm/` | Raw/split datasets (`800/50/150`) and derived symmetric tensor targets (`MS_Symm_QM`, `MS_Symm_Irreps`). Never overwrite existing files. |
| `configs/tensor/` | Production YAML configs for tensor experiments (`configs/reference_scalar/` is archival reference only). |
| `nmr_helpers/` | Importable model, loss, transfer, ELoRA, OEQ compatibility, and training-fix modules. |
| `analysis_helpers/` | Soprano parameter extraction, anisotropic diagnostics, and learning-curve plotting modules. |
| `scripts/` | Environment checks, smoke tests, checkpoint converters, sweep validators, and evaluators. |
| `provenance/` | Data manifests, smoke receipts, compatibility audits, and submission records. |
| `graph-pes-results/` | Pulled production run outputs, checkpoints, `model.pt`, and completion audits (committed to Git). |
| `experiments/` | Executed analysis notebooks and cached evaluation data/figures. |
| `train.sbatch`, `experiment07.sbatch`, `submit-experiment07.sh`, `sync-wandb07.sbatch`, `run-exp04.sbatch` | ARC SLURM launchers and submission wrappers. |

### Known `graph-pes 1.0.2` & OpenEquivariance Pitfalls (Mandatory Safeguards)
1. **Deterministic optimizer ordering:** `graph-pes 1.0.2` builds parameter groups from unordered sets, which breaks Adam state on resume. Always patch `Optimizer.__call__` with `ordered_optimizer` (used in `train.sbatch` and `nmr_helpers/experiment07.py`).
2. **Early-stopping state on resume:** Always call `nmr_helpers.training_fixes.install_training_fixes()` so `EarlyStoppingWithLogging` saves and restores Lightning's `wait_count` and `best_score`.
3. **Checkpoint resume metadata:** When resuming a `graph-pes` checkpoint:
   - Restore `PerElementParameter` element registries (`register_elements([8, 14])`) without re-fitting.
   - Restore `SaveBestModel.best_val_loss` from the checkpoint's `ModelCheckpoint` best score so a worse resumed epoch never overwrites `model.pt`.
   - Disable `SaveBestModel.try_to_deploy` for NMR runs (no LAMMPS energy deployment needed).
4. **Shared ARC env graph cache collision:** `locache 4.0.2` writes beside installed package files. Always install `install_job_local_graph_cache()` (via `CurveAudit` or equivalent) before running concurrent Slurm array tasks against the shared Conda env.
5. **Batching override in `train.sbatch`:** `train.sbatch` forces `batch_size=1, accumulate_grad_batches=4` unless `TENSOR_NMR_CONFIG_BATCHING=1` is exported. Always set `TENSOR_NMR_CONFIG_BATCHING=1` when configs control their own batching.
6. **Pickled checkpoint loading:** Always `import nmr_helpers` before `torch.load(..., weights_only=False)` so legacy module aliases resolve cleanly.
7. **OpenEquivariance 0.3.0 vs `e3nn 0.4.4`:**
   - Pin `openequivariance==0.3.0` on PyTorch 2.6 (newer versions call a missing autocast API) inside a job-local overlay on `$TMPDIR`.
   - OEQ 0.3.0 and `e3nn 0.4.4` differ in Clebsch–Gordan signs on `(1,2,2)`, `(2,1,2)`, and `(2,2,1)`. Always use a converted ordinary-e3nn export (`model-e3nn.pt`, checked by `symmetric_transfer.py`) as the source for NMR transfer — never a raw unconverted OEQ checkpoint.
   - When comparing sub-meV/atom E/F energies across devices, enforce explicit `highest` float32 matmul precision (TF32 disabled) and deterministic algorithms so reduction precision does not mask real discrepancies.
