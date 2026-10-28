# Relational Transformer (RT)

[![Website](https://img.shields.io/badge/website-RT--J-8C1515?logo=googlechrome&logoColor=white)](https://star-project.stanford.edu/rt-j)
[![arXiv](https://img.shields.io/badge/arXiv-2610.17867-B31B1B?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2610.17867)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-stanford--star-yellow)](https://huggingface.co/stanford-star)
[![PyPI](https://img.shields.io/pypi/v/relational-transformer?logo=pypi&logoColor=white)](https://pypi.org/project/relational-transformer/)
[![Python](https://img.shields.io/pypi/pyversions/relational-transformer?logo=python&logoColor=white)](https://pypi.org/project/relational-transformer/)
[![Quickstart in Colab](https://img.shields.io/badge/Colab-Quickstart-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/stanford-star/relational-transformer/blob/main/notebooks/quickstart.ipynb)
[![BYOD in Colab](https://img.shields.io/badge/Colab-BYOD-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/stanford-star/relational-transformer/blob/main/notebooks/byod.ipynb)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue)](https://github.com/stanford-star/relational-transformer/blob/main/LICENSE)

This repo is the official implementation
of the **Relational Transformer (RT)** architecture
and the recipes to pretrain, fine-tune, run inference with and test-time scale
**Relational Foundation Models (RFMs)** based on RT.

The current state of the repo reflects our NeurIPS 2026 work, [RT-J](https://star-project.stanford.edu/rt-j), whose pretraining recipe is shown below:

[![RT-J pretraining recipe](https://star-project.stanford.edu/assets/img/rt-j/recipe.png)](https://star-project.stanford.edu/rt-j)

This repo is linked to the following papers in the
[Stanford Tabular and Relational (STAR) project](https://star-project.stanford.edu) ecosystem:

| Paper | Venue | Implementation |
|---|---|---|
| [RT-J: Large-Scale Pretraining of Relational Transformers for Context-Efficient Predictions](https://star-project.stanford.edu/rt-j) | NeurIPS 2026 | RT-J recipes + paper experiments: this repo. |
| [PluRel: Synthetic Data unlocks Scaling Laws for Relational Foundation Models](https://arxiv.org/abs/2602.04029) | ICML 2026 | Synthetic data generation + paper experiments: [`stanford-star/plurel`](https://github.com/stanford-star/plurel). RT-PluRel recipes: this repo.|
| [Relational Transformer: Toward Zero-Shot Foundation Models for Relational Data](https://arxiv.org/abs/2510.06377) | ICLR 2026 | Legacy architecture + paper experiments: this repo @ [`rt-v1`](https://github.com/stanford-star/relational-transformer/tree/rt-v1) |

Models and datasets (including preprocessed versions) are available on
[Hugging Face](https://huggingface.co/stanford-star).
See also our repo on [agentic task curation for the Join corpus](https://github.com/stanford-star/agentic-task-gen).

## Installation

```bash
pip install relational-transformer
```

This installs the `rt` package, including the Rust sampler `rt.rustler`,
_without any dependency on Rust_.

RT needs Python 3.10+ and PyTorch 2.6+ (for `flex_attention`). If the CUDA build
pip picks for PyTorch does not match your driver, install PyTorch first following
[pytorch.org](https://pytorch.org/get-started/locally/), then RT.

## Quickstart

The [Quickstart notebook](https://github.com/stanford-star/relational-transformer/blob/main/notebooks/quickstart.ipynb)
([open in Colab](https://colab.research.google.com/github/stanford-star/relational-transformer/blob/main/notebooks/quickstart.ipynb))
evaluates the released RT-J checkpoint on a RelBench task, `rel-f1/driver-dnf`,
with no training: download one preprocessed database, predict every test row
from its sampled context, score the AUROC (0.826 on an A100). A few minutes on
an A100 or L4 runtime.

## Bring your own data (BYOD)

RT runs on any database in RelBench format: Parquet tables, a `manifest.yaml` naming
their keys and time columns, and labelled splits per task. See RelBench's
[BYOD guide](https://github.com/stanford-star/relbench/blob/main/byod/README.md) for the format.
The [BYOD notebook](https://github.com/stanford-star/relational-transformer/blob/main/notebooks/byod.ipynb)
([open in Colab](https://colab.research.google.com/github/stanford-star/relational-transformer/blob/main/notebooks/byod.ipynb))
builds one from a DuckDB database with RelBench, with tasks defined in SQL, then
preprocesses it and predicts with a released checkpoint. For more than a first look,
[`examples/`](https://github.com/stanford-star/relational-transformer/blob/main/examples/README.md)
runs the full recipes on it: in-context prediction with a frozen checkpoint, or
per-task fine-tuning.

## Development

We use [pixi](https://pixi.sh) to manage one self-contained
environment (Python, PyTorch + CUDA, Rust, and all dependencies), built on first use.
The environment is defined for Linux x86_64 only.

```bash
git clone https://github.com/stanford-star/relational-transformer.git
cd relational-transformer
pixi run python -m examples.eval.plan        # or examples.pretrain.phases, ...
```

## Examples

[`examples/`](https://github.com/stanford-star/relational-transformer/blob/main/examples/README.md) holds the recipes behind the released
checkpoints and leaderboard entries, with the notes on how each part works.

| | what |
|---|---|
| [`preprocess/`](https://github.com/stanford-star/relational-transformer/tree/main/examples/preprocess) | RelBench-format databases -> RT's tensor format |
| [`pretrain/`](https://github.com/stanford-star/relational-transformer/tree/main/examples/pretrain) | RT-PluRel pretraining on PluRel, then RT-J continued pretraining on the Join |
| [`eval/`](https://github.com/stanford-star/relational-transformer/tree/main/examples/eval) | evaluate a checkpoint on every RelBench task; context knobs, tuning, ensembling |
| [`icl/`](https://github.com/stanford-star/relational-transformer/tree/main/examples/icl) | leaderboard entries: RT-J and RT-PluRel in-context |
| [`finetune/`](https://github.com/stanford-star/relational-transformer/tree/main/examples/finetune) | leaderboard entries: RT, RT-J and RT-PluRel fine-tuned |

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

Apache-2.0, see [LICENSE](https://github.com/stanford-star/relational-transformer/blob/main/LICENSE). The released checkpoints and datasets carry their
own licences; "the Join" is assembled from third-party databases that keep
theirs.
