from pathlib import Path

from reproduce import config
from reproduce.launch import Job, describe, run_sequential

IVF_THRESHOLD = 50_000


def _tasks():
    from rt.data import get_tasks

    return get_tasks(config.pre_dir(), config.db_task_list(), ("test",))


def featurized(db: str, subdir: str, tables: list[str]) -> bool:
    feat_dir = Path(config.share()) / "features" / db / subdir
    return all((feat_dir / f"{t}_meta.json").exists() for t in tables)


def jobs() -> list[Job]:
    share, pre_dir, raw_dir = config.share(), config.pre_dir(), config.raw_dir()
    tasks = _tasks()
    dbs = sorted({t.db_name for t in tasks})
    tables = {db: sorted(t.table_name for t in tasks if t.db_name == db) for db in dbs}
    out = []

    for db in dbs:
        if featurized(db, "sql_features", tables[db]):
            continue
        out.append(
            Job(
                name=f"featurize-sql-{db}",
                target="reproduce.baselines.featurize_sql:featurize_db",
                args={
                    "db": db,
                    "db_task_list": config.db_task_list(),
                    "pre_dir": pre_dir,
                    "raw_dir": raw_dir,
                    "features_root": f"{share}/features",
                    "relbench_cache_dir": f"{share}/relbench-cache",
                },
            )
        )

    for task in tasks:
        db, table = task.db_name, task.table_name
        if featurized(db, "rdblearn_features", [table]):
            continue
        out.append(
            Job(
                name=f"featurize-rdblearn-{db}-{table}",
                target="reproduce.baselines.featurize_rdblearn:featurize_table",
                args={
                    "db": db,
                    "table": table,
                    "task_type": task.task_type,
                    "pre_dir": pre_dir,
                    "raw_dir": raw_dir,
                    "features_root": f"{share}/features",
                    "relbench_cache_dir": f"{share}/relbench-cache",
                    "max_depth": 2,
                    "max_train_samples": 1000,
                },
            )
        )

    for db in dbs:
        if featurized(db, "rt_features", tables[db]):
            continue
        out.append(
            Job(
                name=f"featurize-rt-{db}",
                target="reproduce.baselines.featurize_rt:featurize_db",
                args={
                    "db": db,
                    "db_task_list": config.db_task_list(),
                    "pre_dir": pre_dir,
                    "features_root": f"{share}/features",
                    "ckpt": config.ckpt(),
                    "local_ctx_size": 256,
                    "bfs_width": 32,
                    "shuffle_seed": 0,
                    "context_seed": 0,
                    "db_cutoff": None,
                    "batch_size": 1024,
                },
            )
        )

    for subdir, root in [
        ("rdblearn_features", f"{share}/vector_db/rdblearn"),
        ("rt_features", f"{share}/vector_db/rt"),
    ]:
        if all(
            (Path(root) / t.db_name / f"{t.table_name}.index").exists() for t in tasks
        ):
            continue
        if not all(featurized(db, subdir, tables[db]) for db in dbs):
            print(f"{subdir}: not every table is featurized yet, no index build")
            continue
        out.append(
            Job(
                name=f"vector-db-{subdir.removesuffix('_features')}",
                target="reproduce.baselines.build_vector_db:build_all",
                args={
                    "db_task_list": config.db_task_list(),
                    "pre_dir": pre_dir,
                    "features_root": f"{share}/features",
                    "features_subdir": subdir,
                    "vector_db_root": root,
                    "ivf_threshold": IVF_THRESHOLD,
                },
            )
        )
    return out


if __name__ == "__main__":
    plan = jobs()
    describe(plan)
    run_sequential(plan)
