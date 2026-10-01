import time

import torch
from torch import nn
from tqdm.auto import tqdm


def _fmt(secs):
    m, s = divmod(int(secs), 60)
    return f"{m}m{s:02d}s"


class Rel2TabModel(nn.Module):
    def __init__(self, featurizer, predictor, featurize_batch_size):
        super().__init__()
        self.featurizer = featurizer
        self.predictor = predictor
        self.featurize_batch_size = featurize_batch_size

    def _extract_task_nodes(self, batch):
        """Find unique task nodes in the batch (vectorized on GPU).

        Returns (item_idxs, positions, node_idxs, labels, is_target, f2ps) — all 1D
        tensors of length N (one entry per unique task node across the batch).
        """
        B = batch["is_targets"].shape[0]
        is_targets = batch["is_targets"]
        node_idxs = batch["node_idxs"]
        col_name_idxs = batch["col_name_idxs"]

        b_idxs, s_idxs = is_targets.nonzero(as_tuple=True)
        if len(b_idxs) == 0:
            dev = node_idxs.device
            empty_long = torch.empty(0, device=dev, dtype=torch.long)
            empty_float = torch.empty(0, device=dev, dtype=torch.float)
            empty_bool = torch.empty(0, device=dev, dtype=torch.bool)
            return (
                empty_long,
                empty_long,
                empty_long,
                empty_float,
                empty_bool,
                empty_long,
            )
        target_col = col_name_idxs[b_idxs[0], s_idxs[0]]
        target_node_per_b = torch.full(
            (B,), -1, device=node_idxs.device, dtype=node_idxs.dtype
        )
        target_node_per_b[b_idxs] = node_idxs[b_idxs, s_idxs]

        is_label_cell = (
            batch["is_task_nodes"]
            & ~batch["is_padding"]
            & (col_name_idxs == target_col)
        )
        lc_b, lc_s = is_label_cell.nonzero(as_tuple=True)
        lc_node = node_idxs[lc_b, lc_s]
        lc_label = batch["number_values"][lc_b, lc_s, 0]
        lc_is_target = lc_node == target_node_per_b[lc_b]
        lc_f2p = batch["f2p_nbr_idxs"][lc_b, lc_s]

        return lc_b, lc_s, lc_node, lc_label, lc_is_target, lc_f2p

    def _predict_per_ctx(
        self,
        eval_ctx_sizes,
        true_bs,
        task_type,
        item_idxs,
        positions,
        labels,
        is_target,
        f2ps,
        features,
    ):
        """Run predictor for each (ctx_size, batch_item). Returns {ctx_size: (true_bs,)}."""
        preds = {ctx: torch.zeros(true_bs) for ctx in eval_ctx_sizes}
        default = 0.5 if task_type == "clf" else 0.0
        has_feats = features is not None

        # Collect all work items: featurize first, then predict (possibly batched).
        work_items = []  # list of (ctx, b, train_features, train_labels, test_features)

        for b in range(true_bs):
            b_mask = item_idxs == b
            if not b_mask.any():
                for ctx in eval_ctx_sizes:
                    preds[ctx][b] = default
                continue

            b_positions = positions[b_mask]
            b_labels = labels[b_mask]
            b_is_target = is_target[b_mask]
            b_f2ps = f2ps[b_mask]
            b_feats = features[b_mask.to(features.device)] if has_feats else None

            target_idx = b_is_target.nonzero(as_tuple=True)[0]
            if len(target_idx) == 0:
                for ctx in eval_ctx_sizes:
                    preds[ctx][b] = default
                continue
            target_idx = target_idx[0].item()
            target_f2p = b_f2ps[target_idx]
            test_feat = b_feats[target_idx] if has_feats else None

            for ctx in eval_ctx_sizes:
                visible = b_positions < ctx
                if not visible[target_idx]:
                    preds[ctx][b] = default
                    continue
                train_mask = visible & ~b_is_target
                train_labels_ctx = b_labels[train_mask]
                train_f2ps = b_f2ps[train_mask]
                train_feats = (
                    b_feats[train_mask] if has_feats and train_mask.any() else None
                )
                ft, lt, tt = self.featurizer.featurize(
                    train_labels_ctx,
                    train_f2ps,
                    target_f2p,
                    train_feats,
                    test_feat,
                )
                work_items.append((ctx, b, ft, lt, tt))

        if not work_items:
            return preds

        # Use batch predict if the predictor supports it (e.g. TabICL).
        if hasattr(self.predictor, "predict_batch"):
            batch_inputs = [
                (ft, lt, tt, task_type) for _ctx, _b, ft, lt, tt in work_items
            ]
            results = self.predictor.predict_batch(batch_inputs)
            for (ctx, b, _ft, _lt, _tt), pred_val in zip(work_items, results):
                preds[ctx][b] = pred_val
        else:
            for ctx, b, ft, lt, tt in work_items:
                preds[ctx][b] = self.predictor.predict(ft, lt, tt, task_type)

        return preds

    def predict(self, batch, eval_ctx_sizes, device, task):
        """Eval-mode predictions at multiple context sizes.

        Returns (bs,) per ctx. Rustler lays real rows at indices 0..true_bs-1
        and leaves the rest as phantoms (batch_mask=False), so we fill only
        the real prefix and keep phantom slots at 0 — caller filters them via
        batch_mask after gather.
        """
        bs = batch["is_targets"].size(0)
        true_bs = int(batch["is_targets"].any(dim=1).sum().item())
        task_type = task.task_type

        # 1. Extract task nodes
        tic = time.time()
        item_idxs, positions, node_idxs, labels, is_target, f2ps = (
            self._extract_task_nodes(batch)
        )
        N = item_idxs.shape[0]
        t_extract = time.time() - tic

        if N == 0:
            return {ctx: torch.zeros(bs) for ctx in eval_ctx_sizes}

        # 2. Featurize
        tic = time.time()
        features = self.featurizer.compute_features(
            task, node_idxs, device, self.featurize_batch_size
        )
        t_featurize = time.time() - tic

        # 3. Predict (dense preds for the real prefix, padded to bs).
        tic = time.time()
        dense = self._predict_per_ctx(
            eval_ctx_sizes,
            true_bs,
            task_type,
            item_idxs.cpu(),
            positions.cpu(),
            labels.cpu().float(),
            is_target.cpu(),
            f2ps.cpu(),
            features,
        )
        preds = {}
        for ctx, p in dense.items():
            full = torch.zeros(bs)
            full[:true_bs] = p
            preds[ctx] = full
        t_predict = time.time() - tic

        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            tqdm.write(
                f"    rel2tab: {N} task nodes |"
                f" extract \033[1m{_fmt(t_extract)}\033[0m"
                f"  featurize \033[1m{_fmt(t_featurize)}\033[0m"
                f"  predict \033[1m{_fmt(t_predict)}\033[0m"
            )

        return preds

    def forward(self, batch, return_embeddings):
        raise NotImplementedError(
            "Use predict() for eval. Rel2TabModel is not trainable."
        )
