from rt.preprocess import ls, many, one, upload


def preprocess_one_dataset() -> None:
    one(
        dataset="stanford-star/relbench/rel-f1",
        out_dir="data/relbench-preprocessed",
        embedder="all-MiniLM-L12-v2",
        batch_size=1024,
        skip_tasks=False,
        embed=True,
        upload_repo=None,
        public=False,
        revision=None,
    )


def preprocess_a_collection() -> None:
    many(
        repo="stanford-star/the-join",
        out_dir="data/the-join-preprocessed",
        shard=0,
        num_shards=1,
        skip_existing=True,
        embedder="all-MiniLM-L12-v2",
        batch_size=1024,
        skip_tasks=False,
        embed=True,
        upload_repo=None,
        public=False,
        revision=None,
    )


def list_a_collection() -> None:
    ls(repo="stanford-star/the-join", revision=None)


def upload_result() -> None:
    upload(
        pre_dir="data/relbench-preprocessed/rel-f1",
        repo="your-org/your-preprocessed",
        bulk=False,
        public=False,
    )


if __name__ == "__main__":
    preprocess_one_dataset()
