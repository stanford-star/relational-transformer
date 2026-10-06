# Relational Transformer (RT)

[![Website](https://img.shields.io/badge/website-RT--J-8C1515?logo=googlechrome&logoColor=white)](https://star-project.stanford.edu/rt-j)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-stanford--star-yellow)](https://huggingface.co/stanford-star)
[![PyPI](https://img.shields.io/pypi/v/relational-transformer?logo=pypi&logoColor=white)](https://pypi.org/project/relational-transformer/)
[![Python](https://img.shields.io/pypi/pyversions/relational-transformer?logo=python&logoColor=white)](https://pypi.org/project/relational-transformer/)
[![Quickstart in Colab](https://img.shields.io/badge/Colab-Quickstart-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/stanford-star/relational-transformer/blob/main/notebooks/quickstart.ipynb)
[![BYOD in Colab](https://img.shields.io/badge/Colab-BYOD-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/stanford-star/relational-transformer/blob/main/notebooks/byod.ipynb)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](https://github.com/stanford-star/relational-transformer/blob/main/LICENSE)

This repo is the official implementation
of the **Relational Transformer (RT)** architecture
and the recipes to pretrain, fine-tune, run inference with and test-time scale
**Relational Foundation Models (RFMs)** based on RT.

The following results are fully reproducible from here:  
🥇 **RT-J** is the #1 in-context model on the [RelBench leaderboard](https://star-project.stanford.edu/relbench/leaderboard) at the time of submission ([repro](https://github.com/stanford-star/relational-transformer/tree/main/examples/icl)).  
🥇 **RT-PluRel** is #1 on the combined system + model leaderboard on [RelArena-alpha](https://star-project.stanford.edu/relarena-alpha) at the time of submission ([repro](https://github.com/stanford-star/relational-transformer/tree/main/examples/finetune)).

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

Requires Python 3.10+. Wheels are prebuilt for Linux (x86_64, aarch64) and
macOS; elsewhere pip builds from source, which needs Rust. Running the model
needs an NVIDIA GPU from the Ampere generation or newer (e.g. A100, L4, H100);
older GPUs such as Colab's free T4 are not supported.

## Quickstart

The [Quickstart notebook](https://github.com/stanford-star/relational-transformer/blob/main/notebooks/quickstart.ipynb)
([open in Colab](https://colab.research.google.com/github/stanford-star/relational-transformer/blob/main/notebooks/quickstart.ipynb))
evaluates the released RT-J checkpoint on a RelBench task, `rel-f1/driver-dnf`,
with no training: download one preprocessed database, predict every test row
from its sampled context, score the AUROC (0.826 on an A100). A few minutes on
an A100 or L4 runtime.

## Bring your own database (BYOD)

The [BYOD notebook](https://github.com/stanford-star/relational-transformer/blob/main/notebooks/byod.ipynb)
([open in Colab](https://colab.research.google.com/github/stanford-star/relational-transformer/blob/main/notebooks/byod.ipynb))
preprocesses your database and predicts tasks you define in SQL with a released
checkpoint. For more than a first look, [`examples/`](https://github.com/stanford-star/relational-transformer/blob/main/examples/README.md) runs
the full recipes on your own data in RelBench format: in-context prediction
with a frozen checkpoint, or per-task fine-tuning. That format is a directory of
Parquet tables with a `manifest.yaml` naming each table's primary key, foreign
keys and time column, plus one directory of labelled splits per task; the BYOD
notebook writes one, and
[`examples/preprocess`](https://github.com/stanford-star/relational-transformer/tree/main/examples/preprocess#run-it-on-your-own-data)
shows the layout.

## Development

We use [pixi](https://pixi.sh) to manage one self-contained
environment (Python, PyTorch + CUDA, Rust, and all dependencies), built on first use.
The environment is defined for Linux x86_64 only.

```bash
git clone https://github.com/stanford-star/relational-transformer.git
cd relational-transformer
pixi run pytest                              # the test suite
pixi run python -m examples.eval.plan        # or examples.pretrain.phases, ...
```

`tests/test_smoke.py`, a short training run on a GPU, is skipped unless
`data/relbench-preprocessed/rel-f1` exists; fetching it is described under
[Data](https://github.com/stanford-star/relational-transformer/blob/main/examples/README.md#data).

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

MIT, see [LICENSE](https://github.com/stanford-star/relational-transformer/blob/main/LICENSE). The released checkpoints and datasets carry their
own licences; "the Join" is assembled from third-party databases that keep
theirs.
