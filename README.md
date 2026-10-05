# Relational Transformer (RT)

This repo is the official implementation
of the **Relational Transformer (RT)** architecture
and the recipes to pretrain, fine-tune, inference and test-time scale
**Relational Foundation Models (RFMs)** based on RT.

The following results are fully reproducible from here:  
🥇 **RT-J** is the #1 in-context model on the [RelBench leaderboard](https://star-project.stanford.edu/relbench/leaderboard) at the time of submission ([in-context](examples/icl/), [fine-tuned](examples/finetune/)).  
🥇 **RT-PluRel** is #1 on the combined system + model leaderboard on [RelArena-alpha](https://star-project.stanford.edu/relarena-alpha) at the time of submission ([in-context](examples/icl/), [fine-tuned](examples/finetune/)).

This repo is linked to the following papers in the
[Stanford Tabular and Relational (STAR) project](https://star-project.stanford.edu) ecosystem:

| Paper | Venue | Implementation |
|---|---|---|
| [RT-J: Large-Scale Pretraining of Relational Transformers for Context-Efficient Predictions](https://star-project.stanford.edu/rt-j) | NeurIPS 2026 | RT-J recipes + paper experiments: this repo. |
| [PluRel: Synthetic Data unlocks Scaling Laws for Relational Foundation Models](https://arxiv.org/abs/2602.04029) | ICML 2026 | Synthetic data generation + paper experiments: [`stanford-star/plurel`](https://github.com/stanford-star/plurel). RT-PluRel recipes: this repo.|
| [Relational Transformer: Toward Zero-Shot Foundation Models for Relational Data](https://arxiv.org/abs/2510.06377) | ICLR 2026 | Legacy architecture + paper experiments: this repo @ [`rt-v1`](https://github.com/stanford-star/relational-transformer/tree/rt-v1) |

Models and datasets (including preprocessed versions) are available on
[Hugging Face](https://huggingface.co/stanford-star).

## Installation

```bash
pip install relational-transformer
```

This installs the `rt` package, including the Rust sampler `rt.rustler`,
_without any dependency on Rust_.

## Quickstart

The [Quickstart notebook](notebooks/quickstart.ipynb)
([open in Colab](https://colab.research.google.com/github/stanford-star/relational-transformer/blob/main/notebooks/quickstart.ipynb))
evaluates the released RT-J checkpoint on a RelBench task, `rel-f1/driver-dnf`,
with no training: download one preprocessed database, predict every test row
from its sampled context, score the AUROC. A few minutes on a GPU runtime.

## Bring your own database

The [BYOD notebook](notebooks/byod.ipynb)
([open in Colab](https://colab.research.google.com/github/stanford-star/relational-transformer/blob/main/notebooks/byod.ipynb))
preprocesses your database and predicts tasks you define in SQL with a released
checkpoint. For more than a first look, [`examples/`](examples/README.md) runs
the full recipes on your own data in RelBench format: in-context prediction
with a frozen checkpoint, or per-task fine-tuning.

## Development

We use [pixi](https://pixi.sh) to manage one self-contained
environment (Python, PyTorch + CUDA, Rust, and all dependencies), built on first use.

```bash
git clone https://github.com/stanford-star/relational-transformer.git
cd relational-transformer
pixi run pytest                              # the test suite
pixi run python -m examples.eval.plan        # or examples.pretrain.phases, ...
```

## Examples

[`examples/`](examples/README.md) holds the recipes behind the released
checkpoints and leaderboard entries, with the notes on how each part works.

| | what |
|---|---|
| [`preprocess/`](examples/preprocess/) | RelBench-format databases -> RT's tensor format |
| [`pretrain/`](examples/pretrain/) | RT-J from scratch: PluRel, then the Join |
| [`eval/`](examples/eval/) | a checkpoint on every RelBench task; context knobs, tuning, ensembling |
| [`icl/`](examples/icl/) | leaderboard entries: RT-J and RT-PluRel in-context |
| [`finetune/`](examples/finetune/) | leaderboard entries: RT, RT-J and RT-PluRel fine-tuned |

## Citation

If you use this repo, please cite the following papers:

```bibtex
@inproceedings{ranjan2026rtj,
    title={{RT-J}: Large-Scale Pretraining of Relational Transformers for Context-Efficient Predictions},
    author={Rishabh Ranjan and Vignesh Kothapalli and Harshvardhan Agarwal and Charilaos Kanatsoulis and Roshan Upendra and Tom Palczewski and Carlos Guestrin and Jure Leskovec},
    booktitle={The Fortieth Annual Conference on Neural Information Processing Systems},
    year={2026}
}

@inproceedings{ranjan2026relational,
    title={Relational Transformer: Toward Zero-Shot Foundation Models for Relational Data},
    author={Rishabh Ranjan and Valter Hudovernik and Mark Znidar and Charilaos Kanatsoulis and Roshan Upendra and Mahmoud Mohammadi and Joe Meyer and Tom Palczewski and Carlos Guestrin and Jure Leskovec},
    booktitle={The Fourteenth International Conference on Learning Representations},
    year={2026}
}
```

Additionally, if you use **RT-PluRel** please also cite:

```bibtex
@inproceedings{kothapalli2026plurel,
    title={{PluRel}: Synthetic Data unlocks Scaling Laws for Relational Foundation Models},
    author={Vignesh Kothapalli and Rishabh Ranjan and Valter Hudovernik and Vijay Prakash Dwivedi and Johannes Hoffart and Carlos Guestrin and Jure Leskovec},
    booktitle={Forty-third International Conference on Machine Learning},
    year={2026}
}
```

## License

MIT, see [LICENSE](LICENSE). The released checkpoints and datasets carry their
own licences; "the Join" is assembled from third-party databases that keep
theirs.
