from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from rt.data import process_batch
from rt.rel2tab.featurizer import Featurizer


@dataclass
class RTFeaturizerConfig:
    """Config for RTFeaturizer.

    Fully self-contained: builds its own RT model, loads checkpoint, and
    creates samplers for all eval tasks.
    """

    # RT model params
    embedder: str
    d_text: int
    num_blocks: int
    d_model: int
    num_heads: int
    d_ff: int
    compile: bool
    materialize_attn_masks: bool
    load_ckpt_path: str | None

    # Sampler params
    ctx_size: int
    bfs_width: int
    db_task_list: str
    splits: tuple[str, ...]
    pre_dir: str
    shuffle_seed: int
    context_seed: int
    # See rt.train.main's vector_db_path.
    vector_db_path: str | None

    def build(self, device):
        return RTFeaturizer(
            embedder=self.embedder,
            d_text=self.d_text,
            num_blocks=self.num_blocks,
            d_model=self.d_model,
            num_heads=self.num_heads,
            d_ff=self.d_ff,
            compile=self.compile,
            materialize_attn_masks=self.materialize_attn_masks,
            load_ckpt_path=self.load_ckpt_path,
            device=device,
            db_task_list=self.db_task_list,
            splits=self.splits,
            pre_dir=self.pre_dir,
            ctx_size=self.ctx_size,
            bfs_width=self.bfs_width,
            shuffle_seed=self.shuffle_seed,
            context_seed=self.context_seed,
            vector_db_path=self.vector_db_path,
            db=None,
        )


class RTFeaturizer(Featurizer, nn.Module):
    """Build local contexts and produce masked-token embeddings from a RelationalTransformer.

    Fully self-contained: creates its own RT model, loads checkpoint, and
    builds samplers for all eval tasks.
    """

    def __init__(
        self,
        embedder,
        d_text,
        num_blocks,
        d_model,
        num_heads,
        d_ff,
        compile,
        materialize_attn_masks,
        load_ckpt_path,
        device,
        db_task_list,
        splits,
        pre_dir,
        ctx_size,
        bfs_width,
        shuffle_seed,
        context_seed,
        vector_db_path,
        db,
    ):
        super().__init__()

        from rt.model import RelationalTransformer

        self.rt_model = RelationalTransformer(
            num_blocks=num_blocks,
            d_model=d_model,
            d_text=d_text,
            num_heads=num_heads,
            d_ff=d_ff,
            compile=compile,
            materialize_attn_masks=materialize_attn_masks,
        )
        if load_ckpt_path is not None:
            from rt.model import load_model

            raw = load_model(Path(load_ckpt_path).expanduser())
            state_dict = {k.removeprefix("_orig_mod."): v for k, v in raw.items()}
            self.rt_model.load_state_dict(state_dict)
        self.rt_model.to(device).to(torch.bfloat16)
        self.rt_model.requires_grad_(False)
        self.rt_model.eval()

        from rt.data import RustlerDataset, get_tasks

        all_tasks = get_tasks(pre_dir, db_task_list, splits)
        if db is not None:
            all_tasks = [t for t in all_tasks if db in t.db_name]

        self._samplers = {}
        for task in all_tasks:
            ds = RustlerDataset(
                tasks=[task],
                pre_dir=pre_dir,
                global_rank=0,
                local_rank=0,
                world_size=1,
                local_ctx_size_list=[ctx_size],
                bfs_width_list=[bfs_width],
                num_walks=0,
                walk_length=0,
                prefer_latest_list=[False],
                mask_prob_max=0.0,
                embedder=embedder,
                d_text=d_text,
                shuffle_seed=shuffle_seed,
                context_seed=context_seed,
                items_per_task=0,
                quiet=True,
                ignore_data_errors=False,
                mmap_populate=False,
                legacy_boolean=False,
                timeout_per_item=3600.0,
                vector_db_path=vector_db_path,
                db_cutoff=None,
            )
            self._samplers[task] = ds.sampler

    def compute_features(self, task, node_idxs, device, batch_size):
        sampler = self._samplers[task]
        lc_tup = sampler.batch_for_nodes_py(
            node_idxs=node_idxs.cpu().tolist(),
            dataset_idx=0,
            ctx_size=sampler.local_ctx_size,
        )
        lc_batch = process_batch(lc_tup, sampler.d_text)
        lc_batch.pop("batch_mask", None)

        N = node_idxs.shape[0]
        chunks = []
        with torch.inference_mode():
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                real_bs = end - start
                chunk = {
                    k: v[start:end].to(device, non_blocking=True)
                    for k, v in lc_batch.items()
                }
                # Pad last chunk to full batch_size to avoid recompilation
                if real_bs < batch_size:
                    pad_size = batch_size - real_bs
                    chunk = {
                        k: torch.cat(
                            [
                                v,
                                torch.zeros(
                                    pad_size,
                                    *v.shape[1:],
                                    dtype=v.dtype,
                                    device=v.device,
                                ),
                            ]
                        )
                        for k, v in chunk.items()
                    }
                embeddings = self.rt_model(
                    chunk, return_embeddings=True
                )  # (batch_size, S_max, d_model)
                is_targets = chunk["is_targets"][:real_bs]  # (real_bs, S_max)
                chunks.append(embeddings[:real_bs][is_targets])  # (real_bs, d_model)
        return torch.cat(chunks, dim=0)

    def featurize(self, train_labels, train_f2ps, target_f2p, train_feats, test_feat):
        return train_feats, train_labels, test_feat
