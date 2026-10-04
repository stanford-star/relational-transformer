from datetime import datetime

from rt.data import get_mixture_path
from rt.eval import main


def evaluate(pre_dir: str, out_root: str, checkpoint: str, run_id: str) -> None:
    main(
        load_ckpt_path=checkpoint,
        embedder="all-MiniLM-L12-v2",
        d_text=384,
        num_blocks=12,
        d_model=512,
        num_heads=8,
        d_ff=2048,
        splits=["test"],
        db_task_list=str(get_mixture_path("relbench", "forecast")),
        pre_dir=pre_dir,
        tokens_per_gpu=2**18,
        num_workers=2,
        prefetch_factor=2,
        num_walks=10_000,
        walk_length=20,
        val_items_per_task=None,
        test_items_per_task=10_000_000,
        mmap_populate=True,
        shuffle_seed=0,
        context_seed=0,
        vector_db_path=None,
        db_cutoff=None,
        ctx_size_list=[8192],
        lcs_bw_pl_grid=[(256, 32, True)],
        val_ensemble_size=1,
        test_ensemble_size=1,
        run_id=run_id,
        run_name=None,
        targets={},
        project="rt-eval",
        entity=None,
        out_root=out_root,
        wandb_disabled=True,
    )


if __name__ == "__main__":
    evaluate(
        pre_dir="data/relbench-preprocessed",
        out_root="~/ckpts",
        checkpoint="stanford-star/rt-j",
        run_id=f"{datetime.now():%y-%m-%d_%H-%M-%S}",  # noqa: DTZ005
    )
