from rt.data import db_task_list, plurel_train_db_task_list, write_db_task_list

OUT_DIR = "data/db-task-lists"

if __name__ == "__main__":
    write_db_task_list(
        db_task_list("data/the-join-preprocessed"),
        f"{OUT_DIR}/rt-j.json",
    )
    write_db_task_list(
        plurel_train_db_task_list(
            "data/plurel-preprocessed", raw_dir="data/plurel", num_dbs=1900
        ),
        f"{OUT_DIR}/rt-plurel-train.json",
    )
    write_db_task_list(
        db_task_list("data/relbench-preprocessed", kinds=("forecast",)),
        f"{OUT_DIR}/relbench-forecast.json",
    )
