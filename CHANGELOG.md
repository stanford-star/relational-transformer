# Changelog

All notable changes to the `relational-transformer` package.

## 1.9.0

The first release of the current line. `1.1.0` was the ICLR-submission
snapshot; everything below happened between it and here. The package is a
library with no CLI, the `rt` namespace is organized into subpackages, and the
native data engine ships as a prebuilt stable-ABI extension.

### Breaking

- **No CLI.** `rt.cli`, `rt.config` and the `tyro` config objects are gone. A
  run is a script that calls a library entry point (`rt.train`, `rt.eval`,
  `rt.preprocess`, `rt.data.mlock`), and every entry point takes all of its
  arguments explicitly — no defaults that hide a choice. Released values live in
  [`examples/`](examples/).
- **Package layout.** The flat `src/rt/*.py` modules were reorganized into
  `rt.model` (`net`, `checkpoints`, `legacy`), `rt.data` (`resolve`, `datasets`,
  `tasks`, `stage`, `mlock`), `rt.eval` (`evaluator`, `metrics`, `relbench`,
  `legacy`), `rt.train` (`_train`, `muon`, `swa`) and `rt.preprocess`
  (`_preprocess`, `embed`, `legacy`). Modules no longer collide with the
  functions they export (`_train`, `_eval`, `_preprocess`).
- **The compiled extension is `rt.rustler`** (was `rt._rustler`).
- **`scripts/` is two files.** `pretrain.py`, `eval.py`, `preprocess.py` and
  `mlock_recipe.py` became library entry points driven from
  [`examples/`](examples/), and the two slurm wrappers went with their CLIs;
  `baseline.py` and `ctx_viz.py` remain.
- **`rel2tab` follows the new data API**: a featurizer config takes
  `db_task_list` and `splits` where it took an `eval_recipe` name, `embedder`
  where it took `embedding_model`, and `Rel2TabModel.predict` lost
  `bool_as_num` along with the boolean semantic type.
- **Data is a local directory.** `pre_dir` and `db_task_list` are local paths;
  nothing is fetched from the Hub at run time. Task lists ship with the
  preprocessed data.
- **`requires-python = ">=3.11,<3.14"`.** The 3.11 floor lets consumers that pin
  3.11 resolve the package; the 3.14 ceiling is because `rt` assumes fork-based
  multiprocessing and the Rust `Sampler` does not pickle.
- Removed config knobs with no consumers, the `bool_as_num` flag (booleans are
  folded in the model), `train_only_fallback`, `eval.csv_out_dir`, and
  `eval.tasks` / `eval.task_type` subset selection.

### Added

- **Legacy architectures and eval loops.** `rt.model.legacy` (RT-v1,
  RT-PluRel) and `rt.eval.legacy` reproduce the published leaderboard numbers,
  with `rt.preprocess.legacy` for the RT-v1 boolean-typed preprocessing.
- **Reproducing the paper.** [`reproduce/`](reproduce/) is the pipeline behind
  every figure and table of the RT-J paper, with the cluster it was run on
  stripped out: six stages, each enumerating its jobs explicitly and running
  them sequentially unless you supply a launcher. The per-task context grid
  (`reproduce/tune/tuned_configs.json`), the default-vs-tuned table
  (`reproduce/valtest/results.json`) and the reduced per-figure series
  (`reproduce/series/`) are committed, so the published numbers are checkable
  without re-running anything.
- **Bring your own database.** [`byod/colab.ipynb`](byod/colab.ipynb) runs the
  whole flow end to end on a user database (with a bundled demo DuckDB).
- **Checkpoint loading from the Hub**: `RelationalTransformer.from_pretrained`.
- **Context ensembling and tuning in eval**: separate `val_ensemble_size` /
  `test_ensemble_size`, per-member `context_seed` mixing, the tuned context
  config (`ctx_size`, `local_ctx_size`, `bfs_width`, `prefer_latest`) as a
  searched grid, and ensemble-curve logging.
- **Training**: Muon or AdamW, fp32 master weights, cosine/linear decay with
  `lr_warmup_steps` / `lr_decay_steps`, SWA (`swa_momentum=None` turns it off),
  `delta_finetune` toward the pretrained weights, `grad_accum`,
  `early_stop_after_steps`, atomic `best_*` and `latest.safetensors` publishing,
  Huber and L1 regression losses, and `loss_fn` selection per head.
- **Preemption and resume**: ranks are signalled, state is fsynced before an
  atomic rename, a resumed run continues the data stream instead of re-seeding
  it, and an eval is forced at the resume step.
- **Multi-node and DDP**: `torchrun`/`srun` DDP eval with gather and ensemble
  broadcast, `examples/ddp_check.py` as a cluster sanity check, and a
  one-eval-shaped-batch memory guard before training starts.
- **`rt.data.stage`**: stages `pre_dir` onto node-local storage before it is
  mmapped; worker tensors are shared by descriptor so nothing leaks into
  `/dev/shm`.
- **`CITATION.cff`, `SECURITY.md`, `LICENSE` (MIT)**, a
  GitHub Actions CI workflow (wheel build + tests in a plain venv, ruff
  lint/format) and a release workflow that builds wheels and an sdist on a
  `v*` tag and publishes them with PyPI trusted publishing.
- **Optional-dependency extras**: `dev` (maturin, pre-commit, pytest, ruff) is
  what the tree needs to test and lint itself; `baselines` carries what
  `rel2tab/` needs, and `research` the cluster-submission and plotting tooling.
  A plain `pip install relational-transformer` installs none of them.
- **Docs**: [`downloads.md`](docs/downloads.md),
  [`preprocess.md`](docs/preprocess.md), [`inference.md`](docs/inference.md),
  [`train.md`](docs/train.md), [`baselines.md`](docs/baselines.md),
  [`context-visualization.md`](docs/context-visualization.md).

### Changed

- **`abi3-py311`**: one stable-ABI wheel covers every CPython from 3.11 up.
- **Sampler (`rustler`)**: BFS children are drawn without rejection sampling;
  `p2f` expansion is bounded at `2x bfs_width` draws with no scan fallback; a
  node with no `p2f` edges draws nothing instead of panicking; `prefer_latest`
  orders Tier 2 as well as Tier 1; column statistics come from the train period
  only; key-only tables are visible to the model; the identifier-column policy
  triggers on measured identifiability and is selectable; no boolean semantic
  type leaves the sampler (booleans are z-scored numbers end to end);
  `remove_columns` is scoped to the target's horizon and honored for every task
  kind.
- **Temporal correctness**: task rows in context are strictly past; a
  same-horizon task row is invisible rather than merely unquotable; `db_cutoff`
  accepts an explicit timestamp and picks the split whose timestamp trims the
  database.
- **NaN cells are expected, not stale data**: unified skip semantics between
  `pre` and `fly`, no imputation in the forward pass, and constant columns are
  never normalized by a zero standard deviation.
- **Preprocessing** encodes in chunks, so GPU memory does not scale with the
  database size; raw files are verified by size; recommendation (formerly
  `link_prediction`) tasks are skipped, label tables included.
- **Hub fetches** are bulk and cache-first — one snapshot per pattern set rather
  than one request per database.
- **Logging**: plain `key: value` records with log-friendly progress prints in
  place of tqdm bars, per-task metric curves, per-split eval metrics with
  val-only selection, and one wandb run per attempt grouped by `run_id`.
- **Removed from the model**: the full-attention block (it did not help) and the
  numeric feature augmentation experiment (reverted).
- **Style**: ruff lint and format across `src`, `examples`, `tests`, `byod`;
  imports are eager and at the top of their file; no `from __future__ import
  annotations`.
- `relbench` is imported lazily in `rt.eval.relbench` and is not a declared
  dependency; a failed lazy import explains itself.
- Site-specific NCCL and cluster branches were removed from the library.

### Fixed

- **`pip install` of the published distribution could not complete**:
  `huggingface-hub` was unbounded, so a resolver took hub 2.0 and backtracked
  `sentence-transformers` to 2.2.2 -> `tokenizers` 0.10.3, a 2021 sdist that
  needs a Rust toolchain. Capped at `huggingface-hub<2`.
- `beartype` is no longer a runtime dependency; nothing under `src/rt` imports
  it.
- Data-split regressions after the package reorganization.
- A dead embedding worker fails the job instead of hanging it.
- A database whose files will not load is dropped instead of aborting the run.
- `get_tasks` rejects a task that can resolve to nothing.
- Manifest schemas are closed, so a used field cannot silently vanish.
- Timestamp ids are colon-free (wandb and cargo both reject `:`).
- The wandb attempt id carries the slurm step, so attempts inside one held
  allocation do not collide.
- Eval memory: mask intermediates halved, eval memory claimed before training,
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` at the entry points, eval
  loader workers kept alive between passes, and OMP threads budgeted per rank.
- Tests that catch a stale compiled wheel the git checks cannot see.

## 1.1.0 — 2026-07-20

The ICLR-submission snapshot: flat `src/rt` modules, a `tyro` CLI under
`scripts/`, and the `rel2tab` tabular-baseline package. Preserved as the
`rt-v1` branch.
