# Public API

What outside code may depend on, and what it may not. RT is a library driven
from a script, so "the API" is larger than the symbols `rt` exports: it includes
the on-disk layouts a caller reads and writes, and the selection behaviour a
caller leans on without naming it. This page writes both down, so that the next
refactor knows what it is allowed to break.

The honest state of affairs: `rt/__init__.py` exports **only**
`RelationalTransformer`. Everything else below is reached by importing a
submodule — `rt.data`, `rt.eval`, `rt.train`, `rt.preprocess` — and is therefore
*de-facto* public API that nothing declares. The submodule `__all__` lists are
the closest thing to a declaration there is. Treat the entries on this page as
the supported surface regardless of where they are exported from; treat anything
else, including a name that happens to be in a submodule's `__all__` but is not
listed here, as internal.

The examples in [`examples/`](../examples/) are the intended entry points, and
`rt.train.main` / `rt.eval.main` are what they call. A caller that drives RT as a
library — a benchmark harness, say — uses the narrower surface below instead.

## Symbols

Every entry point takes its arguments explicitly and defines no defaults that
hide a choice; the released values live in `examples/`.

### `rt.RelationalTransformer`

The model. Constructed directly:

```python
RelationalTransformer(num_blocks, d_model, d_text, num_heads, d_ff, compile,
                      materialize_attn_masks, loss_fn="huber")
```

or, for anything already trained, through the classmethod — which is what a
caller normally wants:

```python
RelationalTransformer.from_pretrained(model_id_or_path, *, device="cpu",
                                      compile=False, revision=None,
                                      **model_kwargs)
```

`model_id_or_path` is a local `.safetensors` file, a local directory, or a Hub
model spec. The model dims (`num_blocks`, `d_model`, `d_text`, `num_heads`,
`d_ff`) come from a `config.json` resolved beside the checkpoint, and missing
dims raise rather than defaulting — so a bare `.safetensors` with no sibling
`config.json` needs them passed as keyword arguments. Casting the loaded module
(`.to(torch.bfloat16)`) is supported and is how inference is run.

### `rt.data.get_tasks(pre_dir, db_task_list, splits) -> list[Task]`

Resolves `(db, task)` pairs against `<pre_dir>/<db>/meta.json` and returns opaque
`Task` objects to hand to an evaluator. `db_task_list` is either a list of
pairs or a path to a JSON file of pairs (the curated lists are vendored in the
package; see `rt.data.get_mixture` below). `splits` is the
tuple of split names to resolve. A pair naming a task that `meta.json` does not
carry is **skipped, not an error**; a task whose `meta.json` entry is stale
raises.

`Task` is a pass-through token. Its fields are internal.

### `rt.data.get_mixture(collection, name) -> list[tuple[str, str]]`

The curated `(db, task)` mixtures, vendored in the package rather than fetched
with the data. `collection` is the preprocessed collection the mixture belongs
to (`the-join`, `relbench`, `plurel`) and `name` the mixture within it; an
unknown pair asserts. `rt.data.get_mixture_path(collection, name)` returns the
vendored JSON file's path instead of its contents, for an argument that wants a
path, and `rt.data.list_mixtures()` returns every `(collection, name)` pair.

| collection | mixtures |
| --- | --- |
| `the-join` | `rt-j` (the RT-J pretraining mixture, 13243 pairs), `all` (same set), `forecast` (4098), `autocomplete` (9145) |
| `relbench` | `forecast` (the 21-task benchmark), `autocomplete` (13), `all` (34) |
| `plurel` | `rt-plurel-train` (86211 pairs over 1900 dbs), `all` (116088), `autocomplete` (116088), `forecast` (empty) |

### `rt.eval.build_evaluator(tasks, pre_dir, *, ...)`

Returns an `Evaluator` for one context configuration. All of `embedder`,
`d_text`, `device`, `ctx_size_list`, `local_ctx_size`, `bfs_width`, `num_walks`,
`walk_length`, `tokens_per_gpu`, `items_per_task`, `num_workers`,
`context_seed`, `prefer_latest`, `shuffle_seed`, `mmap_populate`,
`prefetch_factor`, `vector_db_path` and `db_cutoff` are keyword-only and
required; `legacy_boolean`, `global_rank`, `local_rank`, `world_size` and `ddp`
default.

The one method on the result that is public is the generator:

```python
evaluator.evaluate_raw(nets_with_prefix, eval_ctx_size_list_to_use,
                       with_node_idxs=False)
```

`nets_with_prefix` is a list of `(net, prefix)`. It yields, **on global rank 0
only**, one tuple per `(task, ctx_size)`:

```
(task, ctx_size, labels, preds_by_prefix, num_labels)
```

plus a trailing `node_idxs` when `with_node_idxs=True`. `preds_by_prefix` is
keyed by the prefixes passed in. Three properties of this are load-bearing and
not visible in the signature:

- **One build answers for many context sizes.** An evaluator builds contexts at
  the largest size in `eval_ctx_size_list_to_use` and scores every smaller size
  off a prefix of the same build. Walking sizes is cheap; walking
  `(local_ctx_size, bfs_width, prefer_latest)` is not, because each needs its own
  evaluator and its own build.
- **Rows are not yielded in table order**, and not every row is necessarily
  covered. `node_idxs` is the only way back to a table row — see
  `table_info.json` below.
- `evaluate_raw` calls `net.eval()` on each net it is given and runs under
  inference mode. It does not restore the previous training mode.

### `rt.eval.member_context_seed(context_seed, member) -> int`

Mixes a base seed with an ensemble member index. Pure, stable, and the only
supported way to derive per-member context seeds: reimplementing the mix
elsewhere would silently diverge from RT's own ensembling the moment the mix
changes.

### `rt.train.main(*, ...)`

One fit. Keyword-only, no defaults worth relying on, and long — see the
signature in [`src/rt/train/_train.py`](../src/rt/train/_train.py) and the
released values in [`examples/train.py`](../examples/train.py). Its behavioural
contract is the substantial part and is documented below.

### `rt.preprocess.{one, embed_dataset, update_meta_with_embeddings}`

```python
one(*, dataset, out_dir, embedder, batch_size, skip_tasks, embed,
    upload_repo, public, revision) -> None
embed_dataset(pre_dataset_dir, embedder, batch_size) -> int   # returns d_text
update_meta_with_embeddings(pre_dataset_dir, embedder, d_text) -> None
```

`one` runs `resolve → rustler → embed` and writes `<out_dir>/<name>/`. The two
embedding functions are separable from it on purpose: a caller that wants to
cache or reuse text embeddings calls `one(..., embed=False)`, puts
`text_emb_<embedder>.bin` in place itself, and then calls
`update_meta_with_embeddings` so `meta.json` declares it. That split is
supported, and so is the file layout it depends on (below).

`one`'s `dataset` is a dataset directory in relbench format, local or a Hub
spec. The manifest schema rustler parses is **`deny_unknown_fields`**: a stray
key stops preprocessing rather than being ignored. A caller that assembles a
dataset directory itself is depending on that schema, which is a real coupling
and the least pinned-down thing on this page.

## Contracts no signature captures

These are what will silently break an external caller, so they are the ones to
treat as API.

### `resume.pt`

`rt.train.main` writes `resume.pt` into the **run directory**,
`<out_root>/<entity or "no-entity">/<project>/<run_id>/`, and reads it back from
exactly there. Consequences a caller must plan for:

- **The run directory is the resume key.** A run relaunched with the same
  `out_root`, `entity`, `project` and `run_id` resumes from step N; point it at a
  fresh directory and it silently restarts at step 0 — not an error, just a
  wasted run. A harness that submits preemptible jobs must therefore name the
  directory after something stable across a requeue (a job id, not a process id)
  and must not let two concurrent runs of the same configuration share one.
- **`resume.pt` wins over `load_ckpt_path`.** The warm start is loaded only when
  no `resume.pt` exists. This is what makes a requeue correct, and it means a
  caller cannot change the warm start of a run that has already written one.
- It is written on a timer (`resume_save_mins`), at every eval, on `SIGTERM` /
  `SIGUSR1`, and once more at the end, via a temp file plus `os.replace`, so a
  reader never sees a partial file.
- It carries fp32 master weights, optimizer and scheduler state, the SWA state,
  the step, and the best-so-far trackers. It is **not** a model checkpoint: load
  weights from the `.safetensors` artifacts, not from here.
- Resuming asserts loudly rather than coping. A `resume.pt` written without fp32
  masters, or with `swa_momentum` set when this run has it `None` (or the
  reverse), or with a pre-per-kind best tracker, crashes. Changing `swa_momentum`
  on a resumed run is not supported.

### In-loop validation and checkpoint selection

Selection happens **inside the fit**, on validation only, and a caller does not
get to do it afterwards.

- Every `eval_freq` steps the run evaluates the configured `eval_splits` and
  calls `consider()`. Selection reads `val` only — a `test` split evaluated
  alongside never picks a checkpoint.
- The metrics are fixed: `BEST_METRICS = [("clf", "auroc", max), ("reg",
  "nmae", min)]`. Selection is by the **mean** validation metric over the tasks
  of that type, maximized for clf and minimized for reg.
- The step-0 eval counts only when `can_select_init_model=True`.
- At the end the run publishes, per task type `tt` in `{clf, reg}`:
  `best_live_<tt>.safetensors`, `best_swa_<tt>.safetensors`, and
  `best_<tt>.safetensors` — the better of the two nets on val. Alongside them,
  `latest.safetensors` and `latest_swa.safetensors` always point at the most
  recent step of each net, which is the right artifact for a run that scored
  nothing.
- Each published checkpoint carries its step in the safetensors metadata under
  the key `"step"`. Reading it back is the supported way to recover which step
  validation chose.
- **A `best_*` name can legitimately be absent.** The step-0 eval runs before the
  SWA tracker has averaged anything, so `swa_steps=0.safetensors` is never
  written; if step 0 is also the best val score the run ever sees, no
  `best_swa_<tt>` is ever written. With `eval_live=False` there is then no
  `best_<tt>` either. A caller must handle the missing file — it means
  validation never improved on the warm start, which is information, not a bug.
- With `keep_all_ckpts=False` the per-step checkpoints are pruned to those a
  best-so-far still points at. The current weights remain reachable through
  `latest*.safetensors` and `resume.pt`.
- The run writes `config.json` into the run directory, naming the embedder,
  `d_text` and the model dims, so `RelationalTransformer.from_pretrained` works
  on any checkpoint in that directory.

### Two nets, and what `consider()` does

When `swa_momentum` is set, `rt.train.main` tracks **two** nets: the live net and
an EMA of the weights (`SwaState`, `momentum=swa_momentum`). `eval_live=False`
scores only the SWA net, and is asserted against `swa_momentum=None`, since that
combination would score nothing.

`consider(metrics, step)` walks every `(kind, context configuration)` pair, with
`kind` in `{live, swa}`, and:

- updates `best[tt][kind]` when that net's mean val metric for task type `tt`
  improves, taking the best over context configurations at that step;
- also tracks a per-`(kind, configuration)` best;
- returns `True` when **either** net, on **any** configuration, improved.

That return value is what refreshes `improved_at`, which is what
`early_stop_after_steps` measures patience from. So patience is refreshed when
either net improves, and a caller cannot ask for patience on one net alone. The
names `best_live_*`, `best_swa_*` and `best_*` follow directly from the two
kinds, and the `swa/` prefix in the metric keys is how the two are told apart in
the metrics dict.

### Preprocessed directory layout

A `pre_dir` is a **local directory**, always. Nothing is fetched at run time; a
`pre_dir` that does not exist raises rather than downloading. One subdirectory
per database, so a directory you produced and a collection you downloaded are
interchangeable. Per database, the files a reader may rely on:

```
<pre_dir>/<db>/meta.json           relational + task metadata, embedding registry
<pre_dir>/<db>/table_info.json     per-table node index information
<pre_dir>/<db>/column_index.json
<pre_dir>/<db>/nodes.rkyv
<pre_dir>/<db>/offsets.rkyv
<pre_dir>/<db>/p2f_adj.rkyv
<pre_dir>/<db>/text.json                   the text strings, in order
<pre_dir>/<db>/text_emb_<embedder>.bin     their embeddings
```

`CORE_FILES` and `METADATA_FILES` in `rt.data` name the first set
programmatically. The `.rkyv` files are rustler's own format and are internal:
read them only through the RT dataloaders.

The two text files are a documented pair, because the embedding cache split above
depends on them: `text.json` is a JSON list of strings, and
`text_emb_<embedder>.bin` is their embeddings as **bfloat16, row-major
`(len(text.json), d_text)`** with no header — which is why `d_text` can be
recovered from the file size. `meta.json`'s `text_embeddings` entry registers the
file, and `update_meta_with_embeddings` is what writes that entry. A database is
complete only when `meta.json` declares embeddings and the files it names exist.

**`table_info.json`** is the supported way to map a prediction back to a table
row, and it is the contract behind `evaluate_raw`'s `node_idxs`. It maps a table
key to information including `node_idx_offset`, the global rustler node index of
that table's first row, so

```
row_position = node_idx - info[key]["node_idx_offset"]
```

The key is `"<table>:Db"` when the split was ingested as one table, and
`"<table>:Train"` / `":Val"` / `":Test"` when the splits were ingested
separately — a reader must handle both forms. Because the key is a node index and
not a row position, per-row predictions stay aligned no matter what order the
evaluator yields rows in.

## Not API

Anything under `rt.*` not listed above, including: `Task`'s fields, the `.rkyv`
formats, `rt.eval`'s private helpers (a caller wanting a seed offset reads
`table_info.json`), `SwaState`, `rt.train`'s internal closures, the contents of
`resume.pt`, and the `rt.*.legacy` modules. `rt.train.main` and `rt.eval.main`
are public but their argument lists are long and move with the recipe; pin a
version if you call them from outside.
