from rt.data import mlock_main

PRE_DIR = "data/the-join-preprocessed"

if __name__ == "__main__":
    mlock_main(
        db_task_list=f"{PRE_DIR}/db-task-lists/rt-j.json",
        pre_dir=PRE_DIR,
        embedder_ref="all-MiniLM-L12-v2",
        workers=32,
    )
