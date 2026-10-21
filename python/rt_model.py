"""Port of the paper-era official Relational Transformer (PE/rt/model.py, commit 2a899c8).

Module structure and parameter names are IDENTICAL to the official
``RelationalTransformer`` (enc_dict, dec_dict, norm_dict, mask_embs,
blocks.N.norms / blocks.N.attns / blocks.N.ffn, norm_out), so state_dicts are
interchangeable in both directions (``load_state_dict(strict=True)``).

The only changes (both numerical-environment changes, not operational ones):

1. FlexAttention (``torch.compile(flex_attention)`` + ``create_block_mask``,
   model.py:14-17, 43, 89-101) is replaced by dense masked attention:
   ``scores.masked_fill(~mask, -inf) -> softmax -> rows with no allowed key set to 0``.
   That is exactly FlexAttention's semantics, including its handling of fully
   masked query rows (output 0, torch 2.6 ``_inductor/kernel/flex_attention.py:422-427``;
   spec §6.1). The default scale 1/sqrt(d_head) is kept (model.py:43).
2. ``nn.RMSNorm`` gets an explicit ``eps`` (default 2**-7). The official class
   uses ``eps=None``, which under the pinned torch 2.6.0 with bf16 tensors means
   eps = bf16 epsilon = 2**-7 (spec §7.2). We run in float32, so we pin it.

Everything else (embedding gates, pre-norm residual blocks in the order
col -> feat -> nbr -> full -> FFN, SwiGLU, decoders, the loss, the
"touch unused decoders" trick) is copied line by line.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

PAPER_EPS = 2.0**-7

# Order inside a block since MAIN 7f891d1 (model.py:83).
ATTN_ORDER = ["col", "feat", "nbr", "full"]


def dense_masked_attention(q, k, v, mask):
    """Masked SDPA with FlexAttention semantics.

    q, k, v: (B, H, S, Dh); mask: (B, S, S) bool, shared by all heads (H=None, model.py:96).
    Returns (B, H, S, Dh). A query row whose mask row is all False outputs exactly 0.
    """
    scale = 1.0 / math.sqrt(q.shape[-1])  # flex_attention default scale 1/sqrt(E)
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale  # (B, H, S, S)
    m = mask[:, None, :, :]  # broadcast over heads
    row_any = m.any(-1, keepdim=True)  # (B, 1, S, 1)
    scores = scores.masked_fill(~m, float("-inf"))
    # Fully masked rows: give softmax a finite dummy row so neither the forward nor the
    # backward pass produces NaN, then zero the probabilities (FlexAttention: output 0).
    scores = scores.masked_fill(~row_any, 0.0)
    probs = torch.softmax(scores, dim=-1) * row_any
    return torch.matmul(probs, v)


class MaskedAttention(nn.Module):
    """model.py:20-47 with dense masked attention instead of flex_attention."""

    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads

        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask):
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        q = rearrange(q, "b s (h d) -> b h s d", h=self.num_heads)
        k = rearrange(k, "b s (h d) -> b h s d", h=self.num_heads)
        v = rearrange(v, "b s (h d) -> b h s d", h=self.num_heads)

        if mask is None:
            # model.py:39-41 (never taken in RT: every layer has a mask, model.py:204-206)
            x = F.scaled_dot_product_attention(q, k, v)
        else:
            x = dense_masked_attention(q, k, v, mask)

        x = rearrange(x, "b h s d -> b s (h d)")
        x = self.wo(x)
        return x


class FFN(nn.Module):
    """model.py:50-59 (SwiGLU, no biases)."""

    def __init__(self, d_model, d_ff):
        super().__init__()

        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class RelationalBlock(nn.Module):
    """model.py:62-86. Registration order feat, nbr, col, full (state_dict order);
    execution order col, feat, nbr, full."""

    def __init__(self, d_model, num_heads, d_ff, eps=PAPER_EPS):
        super().__init__()

        self.norms = nn.ModuleDict(
            {l: nn.RMSNorm(d_model, eps=eps) for l in ["feat", "nbr", "col", "full", "ffn"]}
        )
        self.attns = nn.ModuleDict(
            {l: MaskedAttention(d_model, num_heads) for l in ["feat", "nbr", "col", "full"]}
        )
        self.ffn = FFN(d_model, d_ff)

    def forward(self, x, masks, trace=None, prefix="", disabled=()):
        for l in ATTN_ORDER:
            if l not in disabled:
                x = x + self.attns[l](self.norms[l](x), mask=masks[l])
            if trace is not None:
                trace.append((f"{prefix}{l}", x))
        x = x + self.ffn(self.norms["ffn"](x))
        if trace is not None:
            trace.append((f"{prefix}ffn", x))
        return x


def build_attn_masks(batch):
    """Dense boolean masks, verbatim from model.py:164-195."""
    node_idxs = batch["node_idxs"]
    f2p_nbr_idxs = batch["f2p_nbr_idxs"]
    col_name_idxs = batch["col_name_idxs"]
    table_name_idxs = batch["table_name_idxs"]
    is_padding = batch["is_padding"]

    # Padding mask for attention pairs (allow only non-pad -> non-pad)
    pad = (~is_padding[:, :, None]) & (~is_padding[:, None, :])  # (B, S, S)

    # cells in the same node
    same_node = node_idxs[:, :, None] == node_idxs[:, None, :]  # (B, S, S)

    # kv index is among q's foreign -> primary neighbors
    kv_in_f2p = (node_idxs[:, None, :, None] == f2p_nbr_idxs[:, :, None, :]).any(-1)  # (B, S, S)

    # q index is among kv's primary -> foreign neighbors (reverse relation)
    q_in_f2p = (node_idxs[:, :, None, None] == f2p_nbr_idxs[:, None, :, :]).any(-1)  # (B, S, S)

    # Same column AND same table
    same_col_table = (col_name_idxs[:, :, None] == col_name_idxs[:, None, :]) & (
        table_name_idxs[:, :, None] == table_name_idxs[:, None, :]
    )  # (B, S, S)

    attn_masks = {
        "feat": (same_node | kv_in_f2p) & pad,
        "nbr": q_in_f2p & pad,
        "col": same_col_table & pad,
        "full": pad,
    }
    for l in attn_masks:
        attn_masks[l] = attn_masks[l].contiguous()
    return attn_masks


class RelationalTransformer(nn.Module):
    """model.py:104-267. Same constructor arguments (+ eps), same state_dict, same forward."""

    def __init__(self, num_blocks, d_model, d_text, num_heads, d_ff, eps=PAPER_EPS):
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
                "number": nn.RMSNorm(d_model, eps=eps),
                "text": nn.RMSNorm(d_model, eps=eps),
                "datetime": nn.RMSNorm(d_model, eps=eps),
                "col_name": nn.RMSNorm(d_model, eps=eps),
                "boolean": nn.RMSNorm(d_model, eps=eps),
            }
        )
        self.mask_embs = nn.ParameterDict(
            {t: nn.Parameter(torch.randn(d_model)) for t in ["number", "text", "datetime", "boolean"]}
        )
        self.blocks = nn.ModuleList(
            [RelationalBlock(d_model, num_heads, d_ff, eps=eps) for i in range(num_blocks)]
        )
        self.norm_out = nn.RMSNorm(d_model, eps=eps)
        self.d_model = d_model

        # Analysis hooks (not part of the official class; no effect on parameters):
        #   self.trace = []                -> forward appends ("x0", x), ("block{i}.{l}", x), ("xFinal", x)
        #   self.disabled_layers = {"nbr"} -> skip those attention sublayers (inference-time ablation)
        self.trace = None
        self.disabled_layers = ()

    def forward(self, batch):
        is_padding = batch["is_padding"]

        # model.py:164-206 (dense masks; no block-mask conversion needed)
        attn_masks = build_attn_masks(batch)

        # model.py:208-224
        x = 0
        x = x + (
            self.norm_dict["col_name"](self.enc_dict["col_name"](batch["col_name_values"]))
            * (~is_padding)[..., None]
        )

        for i, t in enumerate(["number", "text", "datetime", "boolean"]):
            x = x + (
                self.norm_dict[t](self.enc_dict[t](batch[t + "_values"]))
                * ((batch["sem_types"] == i) & ~batch["masks"] & ~is_padding)[..., None]
            )
            x = x + (
                self.mask_embs[t]
                * ((batch["sem_types"] == i) & batch["masks"] & ~is_padding)[..., None]
            )

        trace = self.trace
        if trace is not None:
            trace.append(("x0", x))

        # model.py:226-229
        for i, block in enumerate(self.blocks):
            x = block(x, attn_masks, trace=trace, prefix=f"block{i}.", disabled=self.disabled_layers)

        x = self.norm_out(x)
        if trace is not None:
            trace.append(("xFinal", x))

        # model.py:231-267
        loss_out = x.new_zeros(())
        yhat_out = {"number": None, "text": None, "datetime": None, "boolean": None}

        B, S, _ = x.shape
        sem_types = batch["sem_types"]  # (B,S) ints 0..3
        masks = batch["masks"].bool()  # (B,S) where to train

        for i, t in enumerate(["number", "text", "datetime", "boolean"]):
            yhat = self.dec_dict[t](x)  # (B,S, D_t)
            y = batch[f"{t}_values"]  # (B,S, D_y)
            sem_type_mask = (sem_types == i) & masks  # (B,S) mask for this type

            if not sem_type_mask.any():
                if t in yhat_out:
                    # still touch the param to avoid unused param error
                    loss_out = loss_out + (yhat.sum() * 0.0)
                    yhat_out[t] = yhat
                continue

            if t in ("number", "datetime"):
                loss_t = F.huber_loss(yhat, y, reduction="none").mean(-1)
            elif t == "boolean":
                loss_t = F.binary_cross_entropy_with_logits(yhat, (y > 0).float(), reduction="none").mean(-1)
            elif t == "text":
                raise ValueError("masking text not supported")

            # masked sum for this type
            loss_out = loss_out + (loss_t * sem_type_mask).sum()

            if t in yhat_out:
                yhat_out[t] = yhat

        loss_out = loss_out / masks.sum()

        return loss_out, yhat_out


def set_rmsnorm_eps(module: nn.Module, eps: float = PAPER_EPS) -> int:
    """Set eps on every nn.RMSNorm inside `module` (used to pin the official class to 2**-7)."""
    n = 0
    for m in module.modules():
        if isinstance(m, nn.RMSNorm):
            m.eps = eps
            n += 1
    return n


def clean_state_dict(sd: dict) -> dict:
    """Strip wrappers added by torch.compile (`_orig_mod.`) and DDP (`module.`), main.py:209, 356,
    in any nesting order (e.g. `_orig_mod.module.` and `module._orig_mod.`)."""
    prefixes = ("_orig_mod.", "module.")
    out = {}
    for k, v in sd.items():
        while k.startswith(prefixes):
            k = k[len(next(p for p in prefixes if k.startswith(p))) :]
        out[k] = v
    return out
