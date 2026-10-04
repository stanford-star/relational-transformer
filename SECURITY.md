# Security policy

## Reporting a vulnerability

Please do not open a public issue for a security problem. Report it privately
through GitHub's
[security advisory form](https://github.com/stanford-star/relational-transformer/security/advisories/new)
and we will respond on the advisory.

## Scope

This is research software for relational prediction. It loads model weights with
[safetensors](https://github.com/huggingface/safetensors) and reads preprocessed
datasets from local directories; it does not execute untrusted code from
checkpoints.

Two things are worth knowing when you run it on data you did not produce:

- A preprocessed dataset directory is read with memory mapping and is trusted.
  Only point `pre_dir` at data you built yourself or downloaded from a source
  you trust.
- The preprocessing pipeline reads raw databases and the text embedder downloads
  model weights from HuggingFace on first use.

## Supported versions

Fixes land on `main` and in the next release on PyPI. There are no long-term
support branches.
