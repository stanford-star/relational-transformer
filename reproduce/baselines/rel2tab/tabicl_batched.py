import threading
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

CLF_CHECKPOINT = "tabicl-classifier-v2-20260212.ckpt"
REG_CHECKPOINT = "tabicl-regressor-v2-20260212.ckpt"
HF_REPO = "jingang/TabICL"


_local = threading.local()
_patch_installed = False
_patch_lock = threading.Lock()


def _per_item_ssmax(ssmax_layer, q, n_tensor):
    from tabicl._model.ssmax import QASSMaxMLP, SSMax, SSMaxMLP

    logn = torch.log(n_tensor.clamp(min=1).to(q.dtype)).reshape(-1, 1)
    flat_bs, nh, _, hs = q.shape

    if isinstance(ssmax_layer, SSMax):
        s = ssmax_layer.scales.view(1, nh, 1, 1)
        return q * (s * logn.view(-1, 1, 1, 1))
    if isinstance(ssmax_layer, SSMaxMLP):
        out = ssmax_layer.mlp(logn)
        if ssmax_layer.elementwise:
            scales = out.view(flat_bs, nh, 1, hs)
        else:
            scales = out.view(flat_bs, nh, 1, 1)
        return q * scales
    if isinstance(ssmax_layer, QASSMaxMLP):
        if ssmax_layer.elementwise:
            base = ssmax_layer.base_mlp(logn).view(flat_bs, nh, 1, hs)
        else:
            base = ssmax_layer.base_mlp(logn).view(flat_bs, nh, 1, 1)
        modulation = 1 + torch.tanh(ssmax_layer.query_mlp(q))
        return q * (base * modulation)
    raise TypeError(f"unknown SSMax layer type: {type(ssmax_layer).__name__}")


def _install_attention_patch():
    global _patch_installed
    with _patch_lock:
        if _patch_installed:
            return
        from tabicl._model import attention as _attn_mod
        from tabicl._model.layers import MultiheadAttentionBlock
        from torch.nn import functional as F

        orig_block_forward = MultiheadAttentionBlock.forward
        orig_sdpa = _attn_mod.sdpa_with_flattened_batch

        def patched_block_forward(
            self,
            q,
            k=None,
            v=None,
            cached_kv=None,
            key_padding_mask=None,
            attn_mask=None,
            train_size=None,
            rope=None,
            need_kv=False,
        ):
            m = getattr(_local, "mask", None)
            if m is not None and key_padding_mask is None and cached_kv is None:
                if train_size is not None:
                    eff_k_len = train_size
                elif k is not None:
                    eff_k_len = k.shape[-2]
                else:
                    eff_k_len = q.shape[-2]
                if eff_k_len == m.shape[-1]:
                    batch_shape = q.shape[:-2]
                    extra = len(batch_shape) - 1
                    view_shape = [m.shape[0]] + [1] * extra + [m.shape[1]]
                    key_padding_mask = m.view(*view_shape).expand(
                        *batch_shape, m.shape[1]
                    )
            return orig_block_forward(
                self,
                q,
                k,
                v,
                cached_kv=cached_kv,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
                train_size=train_size,
                rope=rope,
                need_kv=need_kv,
            )

        def patched_sdpa(q, k, v, attn_mask=None, dropout_p=0.0, ssmax_layer=None):
            m = getattr(_local, "mask", None)
            real_lens = getattr(_local, "real_lens", None)
            if m is None or real_lens is None:
                return orig_sdpa(q, k, v, attn_mask, dropout_p, ssmax_layer)

            q_shape = q.shape
            q_flat = q.reshape(-1, *q.shape[-3:])
            k_flat = k.reshape(-1, *k.shape[-3:])
            v_flat = v.reshape(-1, *v.shape[-3:])
            am = (
                None
                if attn_mask is None
                else attn_mask.reshape(-1, *attn_mask.shape[-3:])
            )

            if ssmax_layer is not None:
                src_len = k_flat.size(-2)
                if src_len == m.shape[-1]:
                    flat_bs = q_flat.shape[0]
                    B = real_lens.shape[0]
                    multiplier = flat_bs // B
                    if flat_bs != B * multiplier:
                        raise RuntimeError(f"flat_bs={flat_bs} not a multiple of B={B}")
                    flat_real_n = real_lens.repeat_interleave(multiplier).to(
                        q_flat.device
                    )
                    q_flat = _per_item_ssmax(ssmax_layer, q_flat, flat_real_n)
                else:
                    q_flat = ssmax_layer(q_flat, src_len)

            out = F.scaled_dot_product_attention(q_flat, k_flat, v_flat, am, dropout_p)
            return out.view(q_shape)

        MultiheadAttentionBlock.forward = patched_block_forward
        _attn_mod.sdpa_with_flattened_batch = patched_sdpa
        _patch_installed = True


@contextmanager
def _padded_forward(mask, real_lens):
    prev_mask = getattr(_local, "mask", None)
    prev_lens = getattr(_local, "real_lens", None)
    _local.mask = mask
    _local.real_lens = real_lens
    try:
        yield
    finally:
        _local.mask = prev_mask
        _local.real_lens = prev_lens


class TabICLBatchedPredictor:
    def __init__(
        self, max_batch_size, min_bin_size, softmax_temperature, checkpoint_dir, device
    ):
        self.max_batch_size = max_batch_size
        self.min_bin_size = min_bin_size
        self.softmax_temperature = softmax_temperature
        self.checkpoint_dir = Path(checkpoint_dir).expanduser()

        self._device = device
        self._clf_model = None
        self._reg_model = None
        self._clf_inference_config = None
        self._reg_inference_config = None

        _install_attention_patch()

    def _build_inference_config(self):
        from tabicl import InferenceConfig

        cfg = InferenceConfig()
        cfg.update_from_dict(
            {
                "COL_CONFIG": {
                    "device": self._device,
                    "use_amp": False,
                    "use_fa3": False,
                    "verbose": False,
                    "offload": "auto",
                    "disk_offload_dir": None,
                },
                "ROW_CONFIG": {
                    "device": self._device,
                    "use_amp": False,
                    "use_fa3": False,
                    "verbose": False,
                },
                "ICL_CONFIG": {
                    "device": self._device,
                    "use_amp": False,
                    "use_fa3": False,
                    "verbose": False,
                },
            }
        )
        return cfg

    def _load_model(self, filename):
        from tabicl._model.tabicl import TabICL

        path = self.checkpoint_dir / filename
        assert path.exists(), (
            f"TabICL checkpoint {path} not found; fetch it once with "
            f"`python -m reproduce.baselines.fetch_tabicl`"
        )
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        model = TabICL(**ckpt["config"])
        model.load_state_dict(ckpt["state_dict"])
        model.eval().to(self._device)
        return model

    def _ensure_clf(self):
        if self._clf_model is None:
            self._clf_model = self._load_model(CLF_CHECKPOINT)
            self._clf_inference_config = self._build_inference_config()

    def _ensure_reg(self):
        if self._reg_model is None:
            self._reg_model = self._load_model(REG_CHECKPOINT)
            self._reg_inference_config = self._build_inference_config()

    @staticmethod
    def _trivial_pred(train_features, train_labels, test_features, task_type):
        if len(train_labels) < 2:
            return 0.5 if task_type == "clf" else 0.0

        X_train = train_features.float()
        y_train = train_labels.float()

        if not torch.isfinite(y_train).any():
            return 0.5 if task_type == "clf" else 0.0
        if not torch.isfinite(test_features).any():
            return (
                0.5
                if task_type == "clf"
                else float(y_train[torch.isfinite(y_train)].mean().item())
            )

        first_row = X_train[0]
        if (X_train == first_row).all():
            finite_y = y_train[torch.isfinite(y_train)]
            if finite_y.numel() == 0:
                return 0.5 if task_type == "clf" else 0.0
            return float(finite_y.mean().item())

        if task_type == "clf":
            y_int = (y_train > 0).long()
            unique_count = int(torch.unique(y_int).numel())
            if unique_count < 2:
                return float(y_int[0].item())
        return None

    @staticmethod
    def _preprocess_none_batched(X_train, x_test, real_lens):
        threshold = 4.0
        eps = 1e-6
        clip_min, clip_max = -100.0, 100.0

        device = X_train.device
        dtype = X_train.dtype
        B, T, _H = X_train.shape

        arange_T = torch.arange(T, device=device).unsqueeze(0)
        real_mask = (arange_T < real_lens.unsqueeze(-1)).unsqueeze(-1)
        n = real_lens.to(dtype).view(B, 1)
        eps_t = torch.tensor(eps, device=device, dtype=dtype)

        masked_X = X_train * real_mask
        mean_cs = masked_X.sum(dim=1) / n
        diffs_cs = (X_train - mean_cs.unsqueeze(1)) * real_mask
        var_cs = (diffs_cs**2).sum(dim=1) / n
        scale_cs = torch.sqrt(var_cs) + eps

        X_scaled = ((X_train - mean_cs.unsqueeze(1)) / scale_cs.unsqueeze(1)).clamp(
            clip_min, clip_max
        )
        x_test_scaled = ((x_test - mean_cs) / scale_cs).clamp(clip_min, clip_max)

        masked_Xs = X_scaled * real_mask
        mean_or1 = masked_Xs.sum(dim=1) / n
        diffs_or1 = (X_scaled - mean_or1.unsqueeze(1)) * real_mask
        var_or1 = (diffs_or1**2).sum(dim=1) / (n - 1)
        std_or1 = torch.maximum(torch.sqrt(var_or1), eps_t)

        lower_or1 = mean_or1 - threshold * std_or1
        upper_or1 = mean_or1 + threshold * std_or1

        outlier_mask = (X_scaled < lower_or1.unsqueeze(1)) | (
            X_scaled > upper_or1.unsqueeze(1)
        )
        valid_mask = real_mask & ~outlier_mask

        valid_count = valid_mask.sum(dim=1).to(dtype)

        sum_clean = (X_scaled * valid_mask).sum(dim=1)
        mean_or2 = torch.where(
            valid_count > 0,
            sum_clean / valid_count.clamp(min=1),
            torch.full_like(sum_clean, float("nan")),
        )
        diffs_or2 = (X_scaled - mean_or2.unsqueeze(1)) * valid_mask
        sq_diffs2 = (diffs_or2**2).sum(dim=1)
        var_or2 = torch.where(
            valid_count > 1,
            sq_diffs2 / (valid_count - 1).clamp(min=1),
            torch.full_like(sq_diffs2, float("nan")),
        )
        std_or2 = torch.maximum(torch.sqrt(var_or2), eps_t)

        lower_bounds = mean_or2 - threshold * std_or2
        upper_bounds = mean_or2 + threshold * std_or2

        def _soft_clip(x, lo, hi):
            x = torch.maximum(-torch.log1p(x.abs()) + lo, x)
            x = torch.minimum(torch.log1p(x.abs()) + hi, x)
            return x

        X_out = _soft_clip(
            X_scaled, lower_bounds.unsqueeze(1), upper_bounds.unsqueeze(1)
        )
        x_test_out = _soft_clip(x_test_scaled, lower_bounds, upper_bounds)
        return X_out, x_test_out

    @staticmethod
    def _standardize_y_batched(y_train, real_lens):
        device = y_train.device
        dtype = y_train.dtype
        B, T = y_train.shape

        arange_T = torch.arange(T, device=device).unsqueeze(0)
        real_mask = arange_T < real_lens.unsqueeze(-1)
        n = real_lens.to(dtype)

        masked_y = y_train * real_mask
        y_mean = masked_y.sum(dim=1) / n
        diffs = (y_train - y_mean.unsqueeze(-1)) * real_mask
        var = (diffs**2).sum(dim=1) / n
        raw_std = torch.sqrt(var)
        y_std = torch.where(raw_std == 0, torch.ones_like(raw_std), raw_std)
        y_n = (y_train - y_mean.unsqueeze(-1)) / y_std.unsqueeze(-1)
        return y_n, y_mean, y_std

    @staticmethod
    def _bin_size(n_train, min_bin_size):
        n = max(n_train, min_bin_size)
        return 1 << (n - 1).bit_length()

    @staticmethod
    def _zero_pad(rows, target_n):
        n = rows.shape[0]
        if n == target_n:
            return rows
        pad = torch.zeros(
            (target_n - n, *rows.shape[1:]), dtype=rows.dtype, device=rows.device
        )
        return torch.cat([rows, pad], dim=0)

    def predict_batch(self, work_items):
        n = len(work_items)
        results = [None] * n

        groups = defaultdict(list)

        for i, (tf, tl, xf, tt) in enumerate(work_items):
            tf = tf.float()
            xf = xf.float()
            col_means = torch.nan_to_num(
                torch.nanmean(tf, dim=0, keepdim=True), nan=0.0
            )
            tf = torch.where(torch.isnan(tf), col_means.expand_as(tf), tf)
            xf = torch.where(torch.isnan(xf), col_means.squeeze(0), xf)
            tl = torch.nan_to_num(tl.float(), nan=0.0)

            triv = self._trivial_pred(tf, tl, xf, tt)
            if triv is not None:
                results[i] = triv
                continue

            n_train, d = tf.shape
            bin_size = self._bin_size(n_train, self.min_bin_size)

            if tt == "clf":
                y_int = (tl > 0).long()
                num_classes = max(int(y_int.max().item()) + 1, 2)
                key = (tt, bin_size, num_classes, d)
                groups[key].append((i, n_train, tf, y_int.float(), xf))
            else:
                key = (tt, bin_size, None, d)
                groups[key].append((i, n_train, tf, tl, xf))

        for key, items in groups.items():
            tt, bin_size, num_classes, d = key

            if tt == "clf":
                self._ensure_clf()
                model = self._clf_model
                cfg = self._clf_inference_config
            else:
                self._ensure_reg()
                model = self._reg_model
                cfg = self._reg_inference_config

            for chunk_start in range(0, len(items), self.max_batch_size):
                chunk = items[chunk_start : chunk_start + self.max_batch_size]
                bs = len(chunk)

                X_raw_stack = torch.stack(
                    [self._zero_pad(it[2], bin_size) for it in chunk]
                ).to(self._device, non_blocking=True)
                y_raw_stack = torch.stack(
                    [self._zero_pad(it[3], bin_size) for it in chunk]
                ).to(self._device, non_blocking=True)
                x_test_raw = torch.stack([it[4] for it in chunk]).to(
                    self._device, non_blocking=True
                )

                pad_mask = torch.zeros(
                    (bs, bin_size), dtype=torch.bool, device=self._device
                )
                real_lens = torch.empty(bs, dtype=torch.long, device=self._device)
                for j, it in enumerate(chunk):
                    real_n = it[1]
                    real_lens[j] = real_n
                    if real_n < bin_size:
                        pad_mask[j, real_n:] = True

                X_train_stack, x_test_stack = self._preprocess_none_batched(
                    X_raw_stack, x_test_raw, real_lens
                )

                if tt == "clf":
                    y_train_stack = y_raw_stack
                    y_means = None
                    y_stds = None
                else:
                    y_train_stack, y_means, y_stds = self._standardize_y_batched(
                        y_raw_stack, real_lens
                    )

                X_full = torch.cat([X_train_stack, x_test_stack.unsqueeze(1)], dim=1)

                with torch.no_grad(), _padded_forward(pad_mask, real_lens):
                    if tt == "clf":
                        logits = model(
                            X=X_full,
                            y_train=y_train_stack,
                            return_logits=True,
                            softmax_temperature=self.softmax_temperature,
                            inference_config=cfg,
                        )
                        probs = torch.softmax(
                            logits.float() / self.softmax_temperature, dim=-1
                        )
                        pred_np = (
                            probs[:, 0, 1].detach().cpu().numpy().astype(np.float64)
                        )
                    else:
                        means = model.predict_stats(
                            X=X_full,
                            y_train=y_train_stack,
                            output_type="mean",
                            inference_config=cfg,
                        )
                        pred_np = means[:, 0].float().detach().cpu().numpy()
                        pred_np = pred_np.astype(np.float64)
                        pred_np = pred_np * y_stds.detach().cpu().numpy().astype(
                            np.float64
                        )
                        pred_np = pred_np + y_means.detach().cpu().numpy().astype(
                            np.float64
                        )

                assert np.isfinite(pred_np).all(), (
                    f"{int((~np.isfinite(pred_np)).sum())} of {bs} TabICL predictions "
                    f"are not finite ({tt}, bin {bin_size})"
                )
                for j, item in enumerate(chunk):
                    results[item[0]] = float(pred_np[j])

        return results
