from datetime import datetime
from pathlib import Path

import pytest
import torch

from rt.train import main as train

PRE_DIR = "data/relbench-preprocessed"


def smoke(
    pre_dir: str, out_root: str, run_id: str, total_steps: int, compile: bool
) -> None:
    train(
        embedder="all-MiniLM-L12-v2",
        d_text=384,
        num_blocks=1,
        d_model=64,
        num_heads=2,
        d_ff=128,
        compile=compile,
        materialize_attn_masks=True,
        loss_fn="huber",
        load_ckpt_path=None,
        db_task_list=[("rel-f1", "driver-dnf")],
        train_splits=["train"],
        pre_dir=pre_dir,
        stage_dir=None,
        tokens_per_gpu=256,
        num_workers=0,
        prefetch_factor=None,
        ctx_size_list=[128],
        local_ctx_size_list=[64],
        bfs_width_list=[8],
        prefer_latest_list=[True],
        num_walks=100,
        walk_length=4,
        mask_prob_max=0.0,
        items_per_task=8,
        delta_finetune=False,
        optimizer="muon",
        lr=1e-3,
        wd=0.0,
        lr_warmup_steps=1,
        lr_decay_steps=0,
        grad_norm_max=1.0,
        total_bs=2,
        total_steps=total_steps,
        early_stop_after_steps=None,
        can_select_init_model=False,
        swa_momentum=0.9,
        seed=0,
        mmap_populate=False,
        timeout_per_item=60.0,
        eval_freq=None,
        keep_all_ckpts=False,
        vector_db_path=None,
        db_cutoff=None,
        resume_save_mins=60.0,
        eval_splits=["val"],
        eval_db_task_list=[("rel-f1", "driver-dnf")],
        eval_pre_dir=pre_dir,
        eval_tokens_per_gpu=256,
        eval_num_workers=0,
        eval_prefetch_factor=None,
        eval_num_walks=100,
        eval_walk_length=4,
        eval_items_per_task=4,
        eval_ctx_size_list=[128],
        eval_mmap_populate=False,
        eval_shuffle_seed=0,
        eval_context_seed=0,
        eval_ensemble_size=1,
        eval_vector_db_path=None,
        eval_lcs_bw_pl_grid=[(64, 8, True)],
        run_id=run_id,
        targets={},
        project="smoke",
        wandb_entity=None,
        run_name=None,
        wandb_disabled=True,
        out_root=out_root,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="training needs a GPU")
@pytest.mark.skipif(
    not Path(PRE_DIR).is_dir(), reason=f"no preprocessed data at {PRE_DIR}"
)
def test_smoke(tmp_path):
    run_id = f"{datetime.now():%y-%m-%d_%H-%M-%S}"  # noqa: DTZ005
    smoke(
        pre_dir=PRE_DIR,
        out_root=str(tmp_path),
        run_id=run_id,
        total_steps=3,
        compile=False,
    )
    out = tmp_path / "no-wandb-entity" / "smoke" / run_id
    assert (out / "params.json").is_file(), "the run's arguments are its record"
    assert (out / "resume.pt").is_file(), (
        "a finished run must leave a resumable checkpoint"
    )
