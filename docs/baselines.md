# Baselines

`rel2tab` tabular baselines run through the **same eval path** as RT
([inference.md](inference.md)) and produce the same RelBench submissions, so they
are directly comparable. A baseline is a `(featurizer, predictor)` pair: each
task's in-context training labels (and optional features) are fed to a tabular
predictor, and the result is scored with RelBench's own leaderboard evaluator.

```bash
pip install "relational-transformer[baselines]"
python scripts/baseline.py --featurizer entity --predictor ridge \
  --pre-dir data/relbench-preprocessed \
  --db-task-list "$(python -c 'import rt.data; print(rt.data.get_mixture_path("relbench", "forecast"))')" \
  --out-dir baseline_out
```

- **Featurizers** (`--featurizer`): `global`, `entity`, `rt` (RT embeddings —
  pass a checkpoint with `--rt-ckpt`).
- **Predictors** (`--predictor`): `mean`, `linear`, `ridge`, `xgboost`.

The `global`/`entity` featurizers with the `mean`/`linear`/`ridge` predictors
need no GPU (only the `rt` featurizer runs a model). `--out-dir` is a valid
RelBench submission directory, scored and re-validatable exactly like RT's eval
output. The context flags (`--ctx-size`, `--local-ctx-size`, `--bfs-width`, …)
match `examples/eval.py` — see [context engineering](inference.md#context-engineering).
