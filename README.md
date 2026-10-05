# Relational Transformer (RT)

This repo is the official implementation
of the **Relational Transformer (RT)** architecture
and the recipes to pretrain, fine-tune, inference and test-time scale
**Relational Foundation Models (RFMs)** based on RT.

The following results are fully reproducible from here:  
🥇 **RT-J** is the #1 in-context model on the [RelBench leaderboard](https://star-project.stanford.edu/relbench/leaderboard) at the time of submission.  
🥇 **RT-PluRel** is #1 on the combined system + model leaderboard on [RelArena-alpha](https://star-project.stanford.edu/relarena-alpha) at the time of submission.

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
_without_ any Rust dependency.

## Quickstart

The [notebook](examples/byod/colab.ipynb)
([open in Colab](https://colab.research.google.com/github/stanford-star/relational-transformer/blob/main/examples/byod/colab.ipynb))
runs a released RT-J checkpoint end to end on a bundled toy database: define
tasks in SQL, preprocess, predict, score. Swap in your own database to go from
there.

## Bring your own database

The [quickstart notebook](examples/byod/colab.ipynb) is the first step: it
preprocesses your database and predicts your tasks with a released checkpoint.
For more than a first look, [`examples/`](examples/README.md) runs the full
recipes on your own data in RelBench format: in-context prediction with a
frozen checkpoint, or per-task fine-tuning.

## Development

We use [pixi](https://pixi.sh) to manage one self-contained
environment (Python, PyTorch + CUDA, Rust, and all dependencies), built on first use.

```bash
git clone https://github.com/stanford-star/relational-transformer.git
cd relational-transformer
pixi run pytest                              # the test suite
pixi run python -m examples.eval.plan        # or examples.pretrain.phases, ...
```

## Documentation

| Guide | Description |
|---|---|
| [Downloads](docs/downloads.md) | Bulk-download raw data, preprocessed data, and checkpoints from HuggingFace |
| [Preprocess](docs/preprocess.md) | Convert RelBench-format databases into RT's on-disk format |
| [Inference](docs/inference.md) | Run a trained checkpoint; evaluate, engineer, tune, and ensemble contexts |
| [Pretrain](docs/train.md) | Train RT from scratch, single-GPU to multi-node |
| [Context visualization](docs/ctx_viz.md) | Inspect the contexts sampled for each row |
| [Examples](examples/README.md) | Preprocess, pretrain, evaluate, in-context and fine-tune, on RelBench or your own data |

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
