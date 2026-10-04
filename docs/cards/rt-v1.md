---
license: cc-by-4.0
tags:
  - relational-deep-learning
  - relational-databases
  - relational-data
  - tabular
  - tabular-classification
  - tabular-regression
  - foundation-model
  - in-context-learning
  - relbench
  - relational-transformer
  - legacy
datasets:
  - stanford-star/relbench-v1
  - stanford-star/relbench-preprocessed
metrics:
  - roc_auc
  - mae
pipeline_tag: tabular-classification
library_name: pytorch
---

# RT-v1 (`rt-v1`)

The checkpoints of the **first** Relational Transformer paper,
[Relational Transformer: Toward Zero-Shot Foundation Models for Relational Data
(arXiv:2510.06377)](https://arxiv.org/abs/2510.06377) (ICLR 2026). RT predicts
directly over relational databases — tables linked by foreign keys — using
relational attention over columns, rows and primary/foreign-key links.

> **This release is superseded.** The current model is
> [`stanford-star/rt-j`](https://huggingface.co/stanford-star/rt-j): a single
> checkpoint that handles every task zero-shot, where this release ships one
> checkpoint per task. Start there unless you specifically want to reproduce
> the ICLR 2026 paper. The intermediate release is
> [`stanford-star/rt-plurel`](https://huggingface.co/stanford-star/rt-plurel).

- Code: <https://github.com/stanford-star/relational-transformer> —
  `main` is RT-J; this paper's original implementation is the
  [`rt-v1`](https://github.com/stanford-star/relational-transformer/tree/rt-v1)
  branch.
- Paper: [arXiv:2510.06377](https://arxiv.org/abs/2510.06377)
- Running these checkpoints from `main`:
  [`docs/inference.md`](https://github.com/stanford-star/relational-transformer/blob/main/docs/inference.md#legacy-checkpoints-rt-v1-rt-plurel)

## Model

| | |
|---|---|
| Blocks | 12 |
| `d_model` | 256 |
| Heads | 8 |
| `d_ff` | 1024 |
| Text embedder | `sentence-transformers/all-MiniLM-L12-v2` (`d_text` 384) |
| Format | PyTorch `.pt` state dicts |

Each block applies relational attention at four levels — `col`, `feat`, `nbr`,
`full` — followed by a SwiGLU FFN, with RMSNorm throughout. Cell values are
encoded per semantic type (`number`, `text`, `datetime`, `boolean`) and decoded
by matching per-type heads; classification targets are read from the
BCE-trained `boolean` head. This is the **RT-v1 architecture**, which differs
from RT-J's — it is kept verbatim in `rt.model.legacy.v1.V1Transformer` on
`main` and is state-dict compatible with the files here.

## Checkpoints

59 `.pt` files, one per (database, task), in four families:

| prefix | count | what it is |
|---|---|---|
| `pretrain_<db>_<task>.pt` | 21 | pretrained with `<db>` **held out** — the zero-shot result |
| `contd-pretrain_<db>_<task>.pt` | 21 | continued pretraining on `<db>` with `<task>` held out |
| `finetune-from-contd-pretrain_<db>_<task>.pt` | 13 | fine-tuned on `<task>`, from the continued-pretraining init |
| `finetune-from-pretrain_<db>_<task>.pt` | 4 | fine-tuned on `<task>`, from the plain-pretraining init |

All of `pretrain` and `contd-pretrain` are **in-context**: no checkpoint was
ever trained on the target task's database (`pretrain`) or on the target task
(`contd-pretrain`).

### RelBench leaderboard checkpoints (added 2026-06)

These files back the RT entries on the
[RelBench leaderboard](https://relbench.stanford.edu/leaderboard/). Protocols
follow the paper, with one change: regression best-checkpoint selection uses
val nMAE (MAE / train-split std, ddof=1) — the leaderboard metric — rather than
R². Evaluation is the full official test split (AUROC / nMAE).

- `pretrain_rel-event_<task>.pt` — leave-`rel-event`-out pretraining (50k
  steps), per-task best. `rel-event` was not covered in the original release;
  these produce the RT zero-shot `rel-event` cells.
- `contd-pretrain_rel-event_<task>.pt` — continued pretraining on the other
  `rel-event` tasks, from the matching pretrain checkpoint (2^12+1 steps).
- `finetune-from-{pretrain,contd-pretrain}_<db>_<task>.pt` — the fine-tuned
  checkpoint behind each replicated "RT | pretrained + fine-tuned" cell. The
  board takes the per-task best over the two inits (init treated as a
  hyperparameter), so the file present is the winning init for that task.
  Cells without a file here are the paper's own pretrain-init fine-tuning
  numbers.

## Usage

There is **no CLI**. RT is a library; a run is a script that calls it. Start
from [`examples/`](https://github.com/stanford-star/relational-transformer/tree/main/examples)
and edit the call.

Load one checkpoint:

```python
from rt.model.legacy.v1 import V1Transformer

model = V1Transformer.from_pretrained("pretrain_rel-f1_driver-top3.pt")
```

Reproduce the paper's zero-shot numbers across the 21 RelBench forecast tasks
with [`examples/eval_legacy.py`](https://github.com/stanford-star/relational-transformer/blob/main/examples/eval_legacy.py),
which picks the right per-task checkpoint and writes RelBench leaderboard
submission directories:

```python
from examples.eval_legacy import eval_v1

eval_v1()
```

**These nets need the legacy preprocessing.** `eval_legacy.py` reads
`data/relbench-preprocessed/legacy`, which is the `legacy/` subdirectory of
[`stanford-star/relbench-preprocessed`](https://huggingface.co/datasets/stanford-star/relbench-preprocessed):
RelBench re-preprocessed with the RT-v1-era boolean-typing rules, where binary
targets and a few database columns get a real Boolean semantic type instead of
being z-scored numbers. The regular RT-J `pre_dir` will not do.

```bash
pixi run hf download stanford-star/relbench-preprocessed --repo-type dataset \
  --include "legacy/*" --local-dir data/relbench-preprocessed

pixi run python examples/eval_legacy.py
```

`eval_v1` uses the paper's context configuration: `ctx_size` 1024,
`local_ctx_size` 1024, one BFS neighbourhood around the seed with `bfs_width`
256, no random-walk tier (`num_walks=0`), `prefer_latest` false.

`pre_dir` is always a **local directory**; nothing is fetched on demand.

## Intended use

- Reproducing the ICLR 2026 paper and the RT entries it backs on the RelBench
  leaderboard.
- A baseline for research on relational foundation models.

For new work, use [`rt-j`](https://huggingface.co/stanford-star/rt-j) instead.

## Limitations and out-of-scope use

- **Superseded.** RT-J is one checkpoint for all tasks and scores higher; this
  release needs a different checkpoint per task.
- **One checkpoint per (database, task).** There is no single general model
  here, and the zero-shot claim holds only for the checkpoint whose name names
  the held-out database.
- Binary classification and scalar regression over entities only. Not
  multiclass, not link prediction, not recommendation, not generation.
- Requires the legacy boolean-typed preprocessing described above.
- Metrics reproduce the paper within noise **except RT-v1 on `rel-avito`**,
  which degrades for sampler-level reasons outside these configurations.
- Text is consumed only through frozen `all-MiniLM-L12-v2` embeddings. The
  preprocessed repositories ship embeddings and no readable strings, so
  switching embedder means re-running preprocessing from the raw collection.
- Trained on public databases; carries whatever biases and errors those
  contain, has no calibration guarantees, and must not be used unexamined for
  decisions about people.

## Licence

These weights are released under **CC BY 4.0** (see `LICENSE`) — attribution
only, commercial use permitted.

The licence covers the released weights. It does not extend to the training
data, which carries its own terms:
[`stanford-star/relbench-v1`](https://huggingface.co/datasets/stanford-star/relbench-v1)
is **CC BY-SA 4.0**, and its databases are built from third-party sources that
keep their own licences. RT's source code is MIT.

## Related

- Models: [rt-j](https://huggingface.co/stanford-star/rt-j) (current) ·
  [rt-plurel](https://huggingface.co/stanford-star/rt-plurel)
- Datasets: [relbench](https://huggingface.co/datasets/stanford-star/relbench-v1) ·
  [relbench-preprocessed](https://huggingface.co/datasets/stanford-star/relbench-preprocessed)
- Code: <https://github.com/stanford-star/relational-transformer>

## Citation

Cite this paper for these checkpoints:

```bibtex
@inproceedings{ranjan2026relationaltransformer,
    title={{Relational Transformer:} Toward Zero-Shot Foundation Models for Relational Data},
    author={Rishabh Ranjan and Valter Hudovernik and Mark Znidar and Charilaos Kanatsoulis and Roshan Upendra and Mahmoud Mohammadi and Joe Meyer and Tom Palczewski and Carlos Guestrin and Jure Leskovec},
    booktitle={The Fourteenth International Conference on Learning Representations},
    year={2026}
}
```

If you use the current model, cite RT-J instead — see
[`stanford-star/rt-j`](https://huggingface.co/stanford-star/rt-j).
