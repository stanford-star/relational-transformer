from rt.data import get_mixture_path, mlock_main

PRE_DIR = "data/the-join-preprocessed"

if __name__ == "__main__":
    mlock_main(
        db_task_list=str(get_mixture_path("the-join", "rt-j")),
        pre_dir=PRE_DIR,
        embedder_ref="all-MiniLM-L12-v2",
        workers=32,
    )
