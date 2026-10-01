from functools import partial

import torch
import torch.nn.functional as F
from einops import rearrange
from einops._torch_specific import allow_ops_in_compiled_graph
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.flex_attention import flex_attention

from rt.model.legacy._common import (
    load_legacy_checkpoint,
    make_block_mask,
    predict,
)

allow_ops_in_compiled_graph()
flex_attention = torch.compile(flex_attention)

PLUREL_HUB_REPO = "stanford-star/rt-plurel"
PLUREL_SYNTH_CKPT = "paper/synthetic-pretrain_rdb_1024_size_4b.pt"


class MaskedAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // self.num_heads
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)

        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, block_mask):
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        q = rearrange(q, "b s (h d) -> b h s d", h=self.num_heads)
        k = rearrange(k, "b s (h d) -> b h s d", h=self.num_heads)
        v = rearrange(v, "b s (h d) -> b h s d", h=self.num_heads)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if block_mask is None:
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                x = F.scaled_dot_product_attention(q, k, v)
        else:
            x = flex_attention(q, k, v, block_mask=block_mask)

        x = rearrange(x, "b h s d -> b s (h d)")
        x = self.wo(x)
        return x


class FFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()

        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class RelationalBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()

        self.attn_types = ["col", "feat", "nbr"]

        self.norms = nn.ModuleDict(
            {lvl: nn.RMSNorm(d_model) for lvl in self.attn_types + ["ffn"]}
        )

        self.attns = nn.ModuleDict()
        for lvl in self.attn_types:
            self.attns[lvl] = MaskedAttention(d_model, num_heads)

        self.ffn = FFN(d_model, d_ff)

    def forward(self, x, block_masks):
        for attn in self.attn_types:
            x = x + self.attns[attn](self.norms[attn](x), block_mask=block_masks[attn])

        x = x + self.ffn(self.norms["ffn"](x))
        return x


class PluRelTransformer(nn.Module):
    def __init__(self, num_blocks, d_model, d_text, num_heads, d_ff):
        super().__init__()

        self.enc_dict = nn.ModuleDict(
            {
                "number": nn.Linear(1, d_model, bias=True),
                "text": nn.Linear(d_text, d_model, bias=True),
                "datetime": nn.Linear(1, d_model, bias=True),
                "col_name": nn.Linear(d_text, d_model, bias=True),
                "boolean": nn.Linear(1, d_model, bias=True),
            }
        )
        self.dec_dict = nn.ModuleDict(
            {
                "number": nn.Linear(d_model, 1, bias=True),
                "text": nn.Linear(d_model, d_text, bias=True),
                "datetime": nn.Linear(d_model, 1, bias=True),
                "boolean": nn.Linear(d_model, 1, bias=True),
            }
        )
        self.norm_dict = nn.ModuleDict(
            {
                "number": nn.RMSNorm(d_model),
                "text": nn.RMSNorm(d_model),
                "datetime": nn.RMSNorm(d_model),
                "col_name": nn.RMSNorm(d_model),
                "boolean": nn.RMSNorm(d_model),
            }
        )
        self.mask_embs = nn.ParameterDict(
            {
                t: nn.Parameter(torch.randn(d_model))
                for t in ["number", "text", "datetime", "boolean"]
            }
        )
        self.blocks = nn.ModuleList(
            [RelationalBlock(d_model, num_heads, d_ff) for i in range(num_blocks)]
        )
        self.norm_out = nn.RMSNorm(d_model)
        self.d_model = d_model

    @classmethod
    def from_pretrained(cls, filename, *, repo_id=PLUREL_HUB_REPO, device="cpu"):
        return load_legacy_checkpoint(cls, repo_id, filename, device=device)

    def forward(self, batch):
        batch = {**batch, "boolean_values": batch["number_values"]}
        node_idxs = batch["node_idxs"]
        f2p_nbr_idxs = batch["f2p_nbr_idxs"]
        col_name_idxs = batch["col_name_idxs"]
        table_name_idxs = batch["table_name_idxs"]
        is_padding = batch["is_padding"]
        is_targets = batch["is_targets"]

        batch_size, seq_len = node_idxs.shape
        device = node_idxs.device

        pad = (~is_padding[:, :, None]) & (~is_padding[:, None, :])

        same_node = node_idxs[:, :, None] == node_idxs[:, None, :]

        kv_in_f2p = (node_idxs[:, None, :, None] == f2p_nbr_idxs[:, :, None, :]).any(-1)

        q_in_f2p = (node_idxs[:, :, None, None] == f2p_nbr_idxs[:, None, :, :]).any(-1)

        same_col_table = (col_name_idxs[:, :, None] == col_name_idxs[:, None, :]) & (
            table_name_idxs[:, :, None] == table_name_idxs[:, None, :]
        )

        attn_masks = {
            "feat": (same_node | kv_in_f2p) & pad,
            "nbr": q_in_f2p & pad,
            "col": same_col_table & pad,
        }

        for lvl, mask in attn_masks.items():
            attn_masks[lvl] = mask.contiguous()

        mbm = partial(
            make_block_mask, batch_size=batch_size, seq_len=seq_len, device=device
        )
        block_masks = {lvl: mbm(attn_mask) for lvl, attn_mask in attn_masks.items()}

        x = 0
        x = x + (
            self.norm_dict["col_name"](
                self.enc_dict["col_name"](batch["col_name_values"])
            )
            * (~is_padding)[..., None]
        )

        for i, t in enumerate(["number", "text", "datetime", "boolean"]):
            x = x + (
                self.norm_dict[t](self.enc_dict[t](batch[t + "_values"]))
                * ((batch["sem_types"] == i) & ~is_targets & ~is_padding)[..., None]
            )
            x = x + (
                self.mask_embs[t]
                * ((batch["sem_types"] == i) & is_targets & ~is_padding)[..., None]
            )

        for i, block in enumerate(self.blocks):
            x = block(x, block_masks)

        x = self.norm_out(x)

        yhat_out = {"number": None, "text": None, "datetime": None, "boolean": None}

        B, _S, _ = x.shape
        sem_types = batch["sem_types"]
        masks = is_targets.bool()

        loss_per_seq = x.new_zeros(B)

        for i, t in enumerate(["number", "text", "datetime", "boolean"]):
            yhat = self.dec_dict[t](x)
            y = batch[f"{t}_values"]
            sem_type_mask = (sem_types == i) & masks

            if not sem_type_mask.any():
                if t in yhat_out:
                    loss_per_seq = loss_per_seq + (yhat.sum() * 0.0)
                    yhat_out[t] = yhat
                continue

            if t in ("number", "datetime"):
                loss_t = F.huber_loss(yhat, y, reduction="none").mean(-1)
            elif t == "boolean":
                loss_t = F.binary_cross_entropy_with_logits(
                    yhat, (y > 0).float(), reduction="none"
                ).mean(-1)
            elif t == "text":
                raise ValueError("masking text not supported")

            loss_per_seq = loss_per_seq + (loss_t * sem_type_mask).sum(dim=1)

            if t in yhat_out:
                yhat_out[t] = yhat

        masks_per_seq = masks.sum(dim=1).float().clamp(min=1)
        loss_per_seq = loss_per_seq / masks_per_seq
        loss_out = loss_per_seq.mean()

        return loss_out, yhat_out

    predict = predict
