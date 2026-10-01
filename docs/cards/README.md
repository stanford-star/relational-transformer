# Hub cards

The cards published on
[`huggingface.co/stanford-star`](https://huggingface.co/stanford-star), kept
here so the text is reviewable and has a history. Each file is the `README.md`
of the repository it is named after, front matter included.

## Models

| file | Hub repository | licence |
|---|---|---|
| [`rt-j.md`](rt-j.md) | [`stanford-star/rt-j`](https://huggingface.co/stanford-star/rt-j) | CC BY 4.0 |
| [`rt-plurel.md`](rt-plurel.md) | [`stanford-star/rt-plurel`](https://huggingface.co/stanford-star/rt-plurel) | CC BY 4.0 (`paper/` stays MIT) |
| [`rt-v1.md`](rt-v1.md) | [`stanford-star/rt-v1`](https://huggingface.co/stanford-star/rt-v1) | CC BY 4.0 |

Released weights are permissive: the share-alike obligations of the upstream
data do not reach a model trained on it, and each card scopes its licence to
the weights explicitly.

## Datasets

| file | Hub repository | licence |
|---|---|---|
| [`the-join-preprocessed.md`](the-join-preprocessed.md) | [`stanford-star/the-join-preprocessed`](https://huggingface.co/datasets/stanford-star/the-join-preprocessed) | CC BY-SA 4.0 |
| [`relbench-preprocessed.md`](relbench-preprocessed.md) | [`stanford-star/relbench-preprocessed`](https://huggingface.co/datasets/stanford-star/relbench-preprocessed) | CC BY-SA 4.0 |
| [`plurel-preprocessed.md`](plurel-preprocessed.md) | [`stanford-star/plurel-preprocessed`](https://huggingface.co/datasets/stanford-star/plurel-preprocessed) | CC BY 4.0 |
| [`relbench-raw.md`](relbench-raw.md) | [`stanford-star/relbench-raw`](https://huggingface.co/datasets/stanford-star/relbench-raw) | CC BY 4.0, packaging only |

Share-alike where the data is third-party and inherits it, CC BY 4.0 where the
data is ours or where the licence covers only the packaging; each card explains
which case it is in.

## Changing a card

Edit the file here and upload it as that repository's `README.md`
(`repo_type="model"` for the model cards, `"dataset"` for the dataset cards):

```python
from huggingface_hub import HfApi

HfApi().upload_file(
    path_or_fileobj="docs/cards/rt-j.md",
    path_in_repo="README.md",
    repo_id="stanford-star/rt-j",
    repo_type="model",
    commit_message="Update the model card",
)
```

Every number and code snippet on a card is checked against the artifact before
upload, and the uploaded file is fetched back and compared byte for byte.
