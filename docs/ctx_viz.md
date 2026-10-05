# Context visualization

RT predicts from the context sampled around each row. `examples/ctx_viz.py`
serves a web UI to look at those contexts: pick a database, task and row, set
the sampler knobs (`local_ctx_size`, `bfs_width`, ...), and see what the model
attends over.

```bash
pixi run python examples/ctx_viz.py --pre-root data/relbench-preprocessed
```

Then open the printed URL.
