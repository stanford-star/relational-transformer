# Leaderboard ensemble

RT-J's RelBench leaderboard submission and the paper's tuned-and-ensembled
per-task table: for every task, the **top-4 context configurations** by
validation score (from [`../tune`](../tune)) each run with **4 context seeds**
on the **full official test split**. The 16 raw predictions are averaged per
row, then denormalized (regression) or sigmoided (classification) into
prediction CSVs scored by RelBench's own evaluator.

```bash
python -m reproduce.leaderboard.plan     # 84 units: 21 tasks x 4 configs, one GPU each
python -m reproduce.leaderboard.reduce   # -> CSVs under $RT_SHARE, ../series/leaderboard/top4x4.json

# package for submission
python -m relbench.submit "$RT_SHARE/leaderboard/preds" --out "$RT_SHARE/leaderboard/rt-j-icl.zip"
```

A unit is [`../enscurve/run.py`](../enscurve/run.py) — the same ensemble runner
— at `n_seeds=4` on the full split, so it resumes per seed and a lost job costs
one full-test pass at most.

`reduce.py` refuses a task whose four units have not all finished their four
seeds, asserts every unit's configuration against `tuned_configs_rt-j.json`
along with its checkpoint, seeds and protocol, and `_emit_and_score`'s
alignment guard proves the node-index join against RelBench's ground truth on
every task (the `|dy|` / `cls=` figures it prints). A unit records the
checkpoint it was produced from, and the configurations file records the grid
stem it was ranked from, so units and configurations from two different
checkpoints cannot be averaged together.

`reduce.py` needs only the finished units and RelBench's own task tables, no
GPU.

`plan.jobs` and `reduce.main` take the checkpoint, the task list, the
configurations file and the output directory as arguments; this module's
`__main__` passes the paper's RT-J values.
[`pipelines/icl`](../../pipelines/icl) is the entry point that runs this stage
for either released in-context checkpoint, and documents the expected scores.
