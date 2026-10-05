import time

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from rt.data import EvalDataset, RustlerDataset
from rt.progress import fmt_duration, log


class Evaluator:
    def __init__(
        self,
        *,
        tasks,
        pre_dir,
        eval_bs,
        ctx_size_list,
        items_per_task,
        num_workers,
        prefetch_factor,
        persistent_workers,
        local_ctx_size,
        bfs_width,
        num_walks,
        walk_length,
        prefer_latest,
        mmap_populate,
        legacy_boolean,
        embedder,
        d_text,
        shuffle_seed,
        context_seed,
        vector_db_path,
        db_cutoff,
        global_rank,
        local_rank,
        world_size,
        ddp,
        device,
    ):
        self.tasks = [t for t in tasks if "synthetic" not in t.db_name]
        self.eval_splits = sorted({t.split for t in self.tasks if t.split})
        self.ctx_size_list = ctx_size_list
        self.eval_bs = eval_bs
        self.items_per_task = items_per_task
        self.global_rank = global_rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.ddp = ddp
        self.device = device

        max_eval_ctx_size = max(ctx_size_list)

        self.eval_loaders = {}
        self.eval_loader_iters = {}

        init_tic = time.time()
        prefetch_time = 0.0
        total_items = 0

        for eval_task in self.tasks:
            rustler_dataset = RustlerDataset(
                tasks=[eval_task],
                pre_dir=pre_dir,
                global_rank=global_rank,
                local_rank=local_rank,
                world_size=world_size,
                local_ctx_size_list=[local_ctx_size],
                bfs_width_list=[bfs_width],
                num_walks=num_walks,
                walk_length=walk_length,
                prefer_latest_list=[prefer_latest],
                mask_prob_max=0.0,
                embedder=embedder,
                d_text=d_text,
                shuffle_seed=shuffle_seed,
                context_seed=context_seed,
                items_per_task=items_per_task,
                quiet=True,
                ignore_data_errors=False,
                mmap_populate=mmap_populate,
                legacy_boolean=legacy_boolean,
                timeout_per_item=3600.0,
                vector_db_path=vector_db_path,
                db_cutoff=db_cutoff,
            )
            total_items += rustler_dataset.num_items
            eval_dataset = EvalDataset(
                rustler_dataset=rustler_dataset,
                eval_bs=eval_bs,
                eval_ctx_size=max_eval_ctx_size,
            )
            self.eval_loaders[eval_task] = DataLoader(
                eval_dataset,
                batch_size=None,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor if num_workers > 0 else None,
                persistent_workers=persistent_workers,
                multiprocessing_context="fork" if num_workers > 0 else None,
                pin_memory=True,
                in_order=True,
            )
            _prefetch_tic = time.time()
            self.eval_loader_iters[eval_task] = iter(self.eval_loaders[eval_task])
            prefetch_time += time.time() - _prefetch_tic
        if local_rank == 0:
            log(
                eval_tasks_loaded=len(self.tasks),
                total_items=total_items,
                elapsed=fmt_duration(time.time() - init_tic),
                prefetch_time=fmt_duration(prefetch_time),
            )

    def mem_guard(self, nets):
        if not self.eval_loader_iters:
            return
        eval_task = next(iter(self.eval_loader_iters))
        modes = [net.training for net in nets]
        with torch.inference_mode():
            batch = next(self.eval_loader_iters[eval_task])
            batch.pop("batch_mask")
            for net in nets:
                net.eval()
                net.predict(batch, [max(self.ctx_size_list)], self.device, eval_task)
        del batch
        self.eval_loader_iters[eval_task] = iter(self.eval_loaders[eval_task])
        for net, was_training in zip(nets, modes, strict=True):
            net.train(was_training)

    def evaluate_raw(
        self, nets_with_prefix, eval_ctx_size_list_to_use, with_node_idxs=False
    ):
        device = self.device
        ddp = self.ddp
        world_size = self.world_size
        global_rank = self.global_rank

        for net, _ in nets_with_prefix:
            net.eval()

        t_load = t_predict = t_gather = t_post = 0.0
        t_call = time.perf_counter()

        def _sync():
            if device.startswith("cuda"):
                torch.cuda.synchronize(device)

        with torch.inference_mode():
            for eval_task, eval_loader_iter in self.eval_loader_iters.items():
                eval_loader = self.eval_loaders[eval_task]

                n_batches = len(eval_loader.dataset)
                if self.items_per_task is not None:
                    n_batches = min(
                        n_batches,
                        max(1, self.items_per_task // self.eval_bs // world_size),
                    )

                preds_per_prefix_per_ctx = {
                    prefix: {ctx: [] for ctx in eval_ctx_size_list_to_use}
                    for _, prefix in nets_with_prefix
                }
                num_labels_per_ctx = {ctx: [] for ctx in eval_ctx_size_list_to_use}
                labels = []
                batch_masks = []
                node_idxs_acc = []
                for _ in range(n_batches):
                    _t = time.perf_counter()
                    batch = next(eval_loader_iter)
                    t_load += time.perf_counter() - _t

                    batch_mask = batch.pop("batch_mask")

                    for eval_ctx_size in eval_ctx_size_list_to_use:
                        tb = {k: v[:, :eval_ctx_size] for k, v in batch.items()}
                        tb_is_targets = tb["is_targets"]
                        tb_target_col = torch.full(
                            (tb_is_targets.shape[0], 1),
                            -1,
                            dtype=tb["col_name_idxs"].dtype,
                        )
                        tb_target_node = tb_target_col.clone()
                        tb_bidxs, tb_sidxs = tb_is_targets.nonzero(as_tuple=True)
                        tb_target_col[tb_bidxs, 0] = tb["col_name_idxs"][
                            tb_bidxs, tb_sidxs
                        ]
                        tb_target_node[tb_bidxs, 0] = tb["node_idxs"][
                            tb_bidxs, tb_sidxs
                        ]
                        is_label_cell = (
                            tb["is_task_nodes"]
                            & ~tb["is_padding"]
                            & (tb["col_name_idxs"] == tb_target_col)
                            & (tb["node_idxs"] != tb_target_node)
                        )
                        num_labels_per_ctx[eval_ctx_size].append(
                            is_label_cell.sum(dim=1).to(torch.int64)
                        )

                    _t = time.perf_counter()
                    for net, prefix in nets_with_prefix:
                        preds_by_ctx = net.predict(
                            batch,
                            eval_ctx_size_list_to_use,
                            device,
                            eval_task,
                        )
                        for ctx_size, yhat in preds_by_ctx.items():
                            assert yhat.size(0) == batch_mask.size(0)
                            preds_per_prefix_per_ctx[prefix][ctx_size].append(yhat)
                    _sync()
                    t_predict += time.perf_counter() - _t

                    vals = batch["number_values"].squeeze(-1)
                    y = (vals * batch["is_targets"].to(vals.dtype)).sum(dim=1)
                    assert y.size(0) == batch_mask.size(0)
                    labels.append(y)
                    batch_masks.append(batch_mask)
                    if with_node_idxs:
                        nidx = (
                            batch["node_idxs"].to(torch.int64)
                            * batch["is_targets"].to(torch.int64)
                        ).sum(dim=1)
                        assert nidx.size(0) == batch_mask.size(0)
                        node_idxs_acc.append(nidx)

                self.eval_loader_iters[eval_task] = iter(eval_loader)

                _t = time.perf_counter()
                labels_cat = torch.cat(labels, dim=0).to(device)
                masks_cat = torch.cat(batch_masks, dim=0).to(device)
                if ddp:
                    labels_gathered = torch.empty(
                        labels_cat.size(0) * world_size,
                        dtype=labels_cat.dtype,
                        device=device,
                    )
                    masks_gathered = torch.empty(
                        masks_cat.size(0) * world_size,
                        dtype=masks_cat.dtype,
                        device=device,
                    )
                    dist.all_gather_single(labels_gathered, labels_cat.contiguous())
                    dist.all_gather_single(masks_gathered, masks_cat.contiguous())
                else:
                    labels_gathered = labels_cat
                    masks_gathered = masks_cat

                _sync()
                t_gather += time.perf_counter() - _t
                _t = time.perf_counter()
                if global_rank == 0:
                    labels_np = labels_gathered[masks_gathered].float().cpu().numpy()

                node_idxs_np = None
                if with_node_idxs:
                    nidx_cat = torch.cat(node_idxs_acc, dim=0).to(device)
                    if ddp:
                        nidx_gathered = torch.empty(
                            nidx_cat.size(0) * world_size,
                            dtype=nidx_cat.dtype,
                            device=device,
                        )
                        dist.all_gather_single(nidx_gathered, nidx_cat.contiguous())
                    else:
                        nidx_gathered = nidx_cat
                    if global_rank == 0:
                        node_idxs_np = nidx_gathered[masks_gathered].cpu().numpy()

                for eval_ctx_size in eval_ctx_size_list_to_use:
                    nlabels_cat = torch.cat(
                        num_labels_per_ctx[eval_ctx_size], dim=0
                    ).to(device)
                    if ddp:
                        nlabels_gathered = torch.empty(
                            nlabels_cat.size(0) * world_size,
                            dtype=nlabels_cat.dtype,
                            device=device,
                        )
                        dist.all_gather_single(
                            nlabels_gathered, nlabels_cat.contiguous()
                        )
                    else:
                        nlabels_gathered = nlabels_cat

                    preds_by_prefix_np = {}
                    for _, prefix in nets_with_prefix:
                        preds = torch.cat(
                            preds_per_prefix_per_ctx[prefix][eval_ctx_size], dim=0
                        ).to(device)
                        if ddp:
                            preds_gathered = torch.empty(
                                preds.size(0) * world_size,
                                dtype=preds.dtype,
                                device=device,
                            )
                            dist.all_gather_single(preds_gathered, preds.contiguous())
                            preds = preds_gathered
                        if global_rank == 0:
                            preds_by_prefix_np[prefix] = (
                                preds[masks_gathered].float().cpu().numpy()
                            )

                    if global_rank == 0:
                        num_labels_np = nlabels_gathered[masks_gathered].cpu().numpy()
                        out = (
                            eval_task,
                            eval_ctx_size,
                            labels_np,
                            preds_by_prefix_np,
                            num_labels_np,
                        )
                        if with_node_idxs:
                            out = out + (node_idxs_np,)
                        yield out
                t_post += time.perf_counter() - _t

        if self.local_rank == 0:
            log(
                eval_phase_secs=1,
                total=f"{time.perf_counter() - t_call:.0f}",
                load=f"{t_load:.0f}",
                predict=f"{t_predict:.0f}",
                gather=f"{t_gather:.0f}",
                post=f"{t_post:.0f}",
            )
