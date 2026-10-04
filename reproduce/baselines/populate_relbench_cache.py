import os

from reproduce.baselines.featurize_rdblearn import rdb_dataset
from rt.data import get_mixture


def main() -> None:
    assert os.environ.get("RELBENCH_CACHE_DIR"), "set RELBENCH_CACHE_DIR"
    from relbench.datasets import get_dataset
    from relbench.tasks import get_task, get_task_names

    pairs = get_mixture("relbench", "forecast")
    for db in sorted({db for db, _ in pairs}):
        ds = get_dataset(db, download=True)
        rb_db = ds.get_db(upto_test_timestamp=False)
        print(f"{db}: {len(rb_db.table_dict)} tables materialized", flush=True)
        for task_name in get_task_names(db):
            task = get_task(db, task_name, download=True)
            for split in ("train", "val", "test"):
                task.get_table(split)
            print(f"{db}/{task_name}: extracted", flush=True)
        dataset = rdb_dataset(db)
        print(f"{db}: rdblearn tasks {sorted(dataset.tasks)}", flush=True)
    for db, task_name in pairs:
        task = get_task(db, task_name, download=True)
        for split in ("train", "val", "test"):
            n = len(task.get_table(split).df)
            print(f"{db}/{task_name}/{split}: {n} rows", flush=True)
    print("cache populated", flush=True)


if __name__ == "__main__":
    main()
