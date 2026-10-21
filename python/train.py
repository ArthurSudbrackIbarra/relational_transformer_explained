"""Leave-one-DB-out pretraining, zero-shot evaluation and fine-tuning of the MINI Relational Transformer.

This mirrors the paper-era training script (PE/rt/main.py, commit 2a899c8; spec §8.5-§8.6, §9) at
the scale of our four dummy databases. Contexts come from scripts/export-contexts.ts, i.e. from the
site's own TypeScript sampler (fly.rs mirror), so the model is trained on exactly what the site shows.

What is identical to main.py (only the scale differs, BUILD_PLAN §0.4)
  * model: rt_model.RelationalTransformer (state_dict-identical port of PE/rt/model.py), MINI config
    num_blocks=4, d_model=64, d_text=384, num_heads=4, d_ff=256, RMSNorm eps 2**-7 (spec §7.2);
    init = PyTorch default after seed_everything(0) (main.py:98, example_*.py seed=0);
  * loss: the model's own loss (Huber δ=1 / BCE-with-logits on the ONE masked cell, spec §8.2);
  * one item list over all training tasks, shuffled per epoch with seed epoch+seed, no balancing
    (fly.rs:257-297, 324-328; spec §4.14); context PRNG seed epoch + seed_node_idx + seed (fly.rs:371-375);
    batch b takes items (b·B + i) mod N, the last batch wraps (fly.rs:332-355);
  * pretraining optimizer: AdamW(lr 1e-3, betas (0.9, 0.999), eps 1e-8, wd 0.1, ONE param group) +
    OneCycleLR(max_lr 1e-3, total_steps = max_steps, pct_start 0.2, anneal 'linear', defaults
    cycle_momentum=True, div_factor 25, final_div_factor 1e4) (main.py:188-204, spec §9.2);
  * step order: forward → zero_grad → backward → total grad norm → clip to 1.0 → opt.step → lrs.step
    (main.py:394-424);
  * eval at step 0, every eval_freq steps and at powers of two (main.py:373-375), BEFORE the update of
    that step; max_steps = k·eval_freq + 1 so the last multiple is evaluated (50_001 in the scripts);
  * zero-shot evaluation on the held-out DB's forecast tasks, val and test, fixed epoch-0 contexts
    (main.py:146-168); metrics: AUROC on logits (sklearn roc_auc_score(int(y>0), logit)), R² in
    z-space (sklearn r2_score) (main.py:283-337, spec §8.5);
  * checkpoint selection PER TARGET TASK: best val → keep that step's test metric and weights
    (main.py:377-392, spec §9.1);
  * fine-tuning: fresh AdamW, constant lr 1e-4, wd 0, grad clip 1.0, target task Train split only,
    from the pretrain-best checkpoint of that task, and from random init with the same seed
    ("untrained RT", spec §9.2).

What differs (numerical environment / scale, all reported in results.json)
  * float32 on CPU (paper: pure bf16 on 8×A100); single process (no DDP, so no wrapped duplicates
    in eval, spec §8.6); batch 32 (paper 32/GPU × 8 = 256); L = 128, w = 16 (paper 1024, 256).
  * attention kernel: rt_model's dense masked attention is swapped, after an equivalence check, for
    torch's fused F.scaled_dot_product_attention with the same boolean mask. Same maths (masked
    softmax, fully masked rows → 0 like FlexAttention); ~5× faster on CPU. See _install_fast_attention.
  * steps: 4,001 (≈22 min on this CPU, ≈0.3 s/step with evals; paper: 50,001 steps); fine-tuning 1,025
    steps (paper: 32,769).

Outputs
  python/data/runs/<run>/{state.pt, best-*.pt}         resumable state (saved every 2 min), selected checkpoints
  public/data/training/{curves,results,schedule,dataset}.json
  public/data/weights/{init,pretrain-churn,pretrain-sales,pretrain-rating,pretrain-verified,pretrain-final,
                       finetune-churn,finetune-sales}.{json,bin}

Usage (every command is resumable; --budget-min bounds one invocation's wall time, exit code 3 =
"not finished, run again"):
  python/.venv/bin/python python/train.py bench
  python/.venv/bin/python python/train.py all --threads 6 --pretrain-steps 4001 --pretrain-eval-freq 100 \
      --finetune-steps 1025 --finetune-eval-freq 64 --budget-min 8.5  # repeat until it prints ALL DONE
  python/.venv/bin/python python/train.py pretrain --pretrain-run pretrain-sqrtlr --pretrain-lr 3.5355e-4  # the LR-fix run
  python/.venv/bin/python python/train.py pretrain | finetune --task shop/customer-churn --init pretrained
  python/.venv/bin/python python/train.py export                      # JSON + weights from finished runs
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

# sklearn (→ scipy) must be imported BEFORE torch on this machine (torch first loads the system
# libstdc++, which lacks the CXXABI miniconda's scipy needs). If a caller imported torch first, fall
# back to exact equivalents of the two metrics main.py uses (main.py:328-337).
try:
    from sklearn.metrics import r2_score, roc_auc_score  # noqa: I001
except ImportError:  # pragma: no cover - environment dependent

    def roc_auc_score(y_true, y_score):
        """AUROC = Mann-Whitney U / (n_pos·n_neg) with average ranks for ties (= sklearn for binary y)."""
        y = np.asarray(y_true).astype(bool)
        s = np.asarray(y_score, dtype=np.float64)
        order = np.argsort(s, kind="mergesort")
        ranks = np.empty(len(s))
        ss = s[order]
        i = 0
        while i < len(ss):
            j = i
            while j + 1 < len(ss) and ss[j + 1] == ss[i]:
                j += 1
            ranks[order[i : j + 1]] = (i + j) / 2 + 1
            i = j + 1
        n_pos, n_neg = y.sum(), (~y).sum()
        return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))

    def r2_score(y_true, y_pred):
        y = np.asarray(y_true, dtype=np.float64)
        p = np.asarray(y_pred, dtype=np.float64)
        return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import rt_model  # noqa: E402
from export_weights import export_state_dict, infer_config  # noqa: E402
from rt_model import PAPER_EPS, RelationalTransformer  # noqa: E402

CTX_DIR = HERE / "data" / "contexts"
RUNS_DIR = HERE / "data" / "runs"
EMB_DIR = ROOT / "public" / "data" / "embeddings"
TRAINING_OUT = ROOT / "public" / "data" / "training"
WEIGHTS_OUT = ROOT / "public" / "data" / "weights"

MINI = dict(num_blocks=4, d_model=64, d_text=384, num_heads=4, d_ff=256)
ATTENTION_INFO: dict = {}  # filled by _install_fast_attention() in main()
SEED = 0  # example_*.py: seed=0 → seed_everything(0) (main.py:21-29, 98)
BATCH = 32  # per-GPU batch of the scripts (example_pretrain.py:19)
TARGET_FORECAST = ["shop/customer-churn", "shop/product-sales"]
# Shop autocomplete tasks: zero-shot evaluated and selected too (our extension: the paper's eval_tasks
# are the held-out DB's forecast tasks only, example_pretrain.py:18). Their seeds are ALL review rows
# (split ignored, data.py:77-81, spec §2.4), so selection uses the even review_ids ("Val") and the
# reported number the odd ones ("Test"): the contexts are unchanged, only the metric is computed on two
# disjoint halves, so the selected checkpoint is never scored on the seeds that selected it.
TARGET_AUTOCOMPLETE = ["shop/reviews.rating", "shop/reviews.verified"]
TARGET_ALL = TARGET_FORECAST + TARGET_AUTOCOMPLETE
SHORT = {"shop/customer-churn": "churn", "shop/product-sales": "sales", "shop/reviews.rating": "rating",
         "shop/reviews.verified": "verified"}
AUTOCOMPLETE_HALVES = "Val = even review_id, Test = odd review_id (seed position; every review row is a seed)"

# Pretraining hyperparameters (example_pretrain.py; spec §9.2). Steps are scaled down.
PRETRAIN = dict(lr=1e-3, wd=0.1, max_grad_norm=1.0, eval_freq=200, eval_pow2=True)
# Fine-tuning (example_finetune.py: lr 1e-4 constant, wd 0, clip 1.0). eval_pow2 on for the Fig. 3 analogue.
FINETUNE = dict(lr=1e-4, wd=0.0, max_grad_norm=1.0, eval_freq=64, eval_pow2=True)
SAVE_EVERY_SECONDS = 120  # resumable state (state.pt) is also written every 2 minutes, not only at the budget
# Version of the pretraining DBs (src/engine/db/generators/{school,racing,forum}.ts). 2 = persistent self labels.
DATA_VERSION = 2
DATA_VERSION_NOTE = ("Data version 2: the pretraining DB generators make entity behaviour persistent (lag-1 label "
                     "correlation ≥ 0.4 and EntityMean beating the Train mean on every pretraining forecast task, see "
                     "labelPersistence), as in real RelBench data. ShopDB is unchanged (db.test.ts checks its SHA-256).")


# ──────────────────────────────────────────────────────────────────────────────
# Attention kernel
# ──────────────────────────────────────────────────────────────────────────────
def _sdpa_masked_attention(q, k, v, mask):
    """Same contract as rt_model.dense_masked_attention: (B,H,S,Dh) q/k/v, (B,S,S) bool mask shared
    by all heads; scale 1/sqrt(Dh); fully masked query rows output exactly 0 (FlexAttention)."""
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None, :, :])


def _install_fast_attention() -> dict:
    """Swap rt_model.dense_masked_attention for the fused SDPA kernel after checking, on random data
    with fully masked rows, that outputs AND gradients agree to ≤ 1e-5 in both grad and no-grad mode.
    Falls back to the dense reference if the check fails (e.g. a torch version that returns NaN rows)."""
    g = torch.Generator().manual_seed(123)
    B, H, S, D = 3, 4, 64, 16
    q, k, v = (torch.randn(B, H, S, D, generator=g, requires_grad=True) for _ in range(3))
    mask = torch.rand(B, S, S, generator=g) < 0.15
    mask[:, :7, :] = False  # empty rows (padding queries, empty neighbour rows)
    mask[:, 7, :] = True
    ref = rt_model.dense_masked_attention(q, k, v, mask)
    out = _sdpa_masked_attention(q, k, v, mask)
    up = torch.randn(ref.shape, generator=g)
    g_ref = torch.autograd.grad(ref, (q, k, v), up)
    g_out = torch.autograd.grad(out, (q, k, v), up)
    with torch.inference_mode():
        out_inf = _sdpa_masked_attention(q.detach(), k.detach(), v.detach(), mask)
    d_fwd = float((out - ref).detach().abs().max())
    d_inf = float((out_inf - ref.detach()).abs().max())
    d_grad = max(float((a - b).abs().max()) for a, b in zip(g_out, g_ref))
    finite = bool(torch.isfinite(out).all() and all(torch.isfinite(x).all() for x in g_out))
    ok = finite and max(d_fwd, d_inf, d_grad) <= 1e-5
    if ok:
        rt_model.dense_masked_attention = _sdpa_masked_attention  # MaskedAttention.forward looks it up at call time
    return {"kernel": "F.scaled_dot_product_attention" if ok else "rt_model.dense_masked_attention",
            "maxAbsDiffForward": d_fwd, "maxAbsDiffInference": d_inf, "maxAbsDiffGrad": d_grad, "finite": finite}


# ──────────────────────────────────────────────────────────────────────────────
# Data: exported contexts → official batch dict
# ──────────────────────────────────────────────────────────────────────────────
class Contexts:
    """All context sets of python/data/contexts (scripts/export-contexts.ts) + the MiniLM table."""

    def __init__(self, ctx_dir: Path = CTX_DIR):
        self.dir = ctx_dir
        self.index = json.loads((ctx_dir / "index.json").read_text())
        self.L = self.index["L"]
        self.tasks = self.index["tasks"]
        self.task_by_id = {t["id"]: t for t in self.tasks}
        # Frozen text embeddings (spec §3.6): per-DB text index → string → MiniLM row.
        emb_meta = json.loads((EMB_DIR / "minilm-l12-v2.json").read_text())
        strings = emb_meta["strings"]
        dim = emb_meta["dim"]
        flat = np.fromfile(EMB_DIR / "minilm-l12-v2.f32", dtype="<f4").reshape(len(strings), dim)
        row_of = {s: i for i, s in enumerate(strings)}
        db_names = list(self.index["dbs"].keys())
        mats, self.emb_offset = [], {}
        off = 0
        for d in db_names:
            text = self.index["dbs"][d]["text"]
            missing = [s for s in text if s not in row_of]
            if missing:
                raise SystemExit(f"[data] {len(missing)} strings of {d} have no embedding (re-run export-strings + embed.py): {missing[:3]}")
            mats.append(flat[[row_of[s] for s in text]])
            self.emb_offset[d] = off
            off += len(text)
        self.zero_row = off  # padding col_name_values / text_values are 0⃗ (fly.rs:65-85)
        self.emb = torch.from_numpy(np.concatenate(mats + [np.zeros((1, dim), np.float32)]).astype(np.float32))
        self.task_offset = np.array([self.emb_offset[t["db"]] for t in self.tasks], dtype=np.int64)
        self._sets = {}

    def set(self, name: str) -> "ContextSet":
        if name not in self._sets:
            if name not in self.index["sets"]:
                raise KeyError(f"no context set {name!r}; available: {list(self.index['sets'])}")
            self._sets[name] = ContextSet(self, name, self.index["sets"][name])
        return self._sets[name]


class ContextSet:
    def __init__(self, ctxs: Contexts, name: str, desc: dict):
        self.ctxs, self.name, self.desc = ctxs, name, desc
        self.n = desc["n"]
        self.items_per_epoch = desc["itemsPerEpoch"]
        self.epochs = desc["epochs"]
        path = ctxs.dir / desc["file"]
        self.a = {
            k: np.memmap(path, dtype=np.dtype(v["dtype"]).newbyteorder("<"), mode="r", offset=v["offset"], shape=tuple(v["shape"]))
            for k, v in desc["arrays"].items()
        }

    def batch(self, idx) -> dict:
        """Official batch dict (fly.rs:27-44 → data.py:116-154), float32 on CPU, for context indices idx."""
        idx = np.asarray(idx, dtype=np.int64)
        a = self.a
        sem = torch.from_numpy(a["sem_types"][idx].astype(np.int64))
        vals = torch.from_numpy(np.ascontiguousarray(a["values"][idx], dtype=np.float32))
        flags = torch.from_numpy(a["flags"][idx].astype(np.uint8))
        is_padding = (flags & 4) != 0
        off = torch.from_numpy(self.ctxs.task_offset[a["task"][idx]])[:, None]
        col = torch.from_numpy(a["col_name_idxs"][idx].astype(np.int64))
        txt = torch.from_numpy(a["text_idxs"][idx].astype(np.int64))
        col_rows = torch.where(is_padding, self.ctxs.zero_row, off + col)
        txt_rows = torch.where(txt < 0, self.ctxs.zero_row, off + txt)
        zero = torch.zeros_like(vals)
        return {
            "node_idxs": torch.from_numpy(a["node_idxs"][idx].astype(np.int64)),
            "f2p_nbr_idxs": torch.from_numpy(a["f2p_nbr_idxs"][idx].astype(np.int64)),
            "table_name_idxs": torch.from_numpy(a["table_name_idxs"][idx].astype(np.int64)),
            "col_name_idxs": col,
            "col_name_values": self.ctxs.emb[col_rows],
            "sem_types": sem,
            "number_values": torch.where(sem == 0, vals, zero)[..., None],
            "text_values": self.ctxs.emb[txt_rows],
            "datetime_values": torch.where(sem == 2, vals, zero)[..., None],
            "boolean_values": torch.where(sem == 3, vals, zero)[..., None],
            "masks": (flags & 1) != 0,
            "is_targets": (flags & 2) != 0,
            "is_padding": is_padding,
            "is_task_nodes": (flags & 8) != 0,
        }

    def context_index(self, epoch: int, item: int) -> int:
        """Contexts are stored epoch-major. Epochs beyond the exported ones reuse epoch mod E."""
        return (epoch % self.epochs) * self.items_per_epoch + item


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation (main.py:219-344)
# ──────────────────────────────────────────────────────────────────────────────
def predict_set(net, cs: ContextSet, target_type: str, drop_labels: bool = False):
    """One forward pass over a fixed eval set. Returns (prediction at the masked cell, stored label,
    per-context loss) as float64 arrays in context order. The per-context loss is the model's own loss
    for its one masked cell (Huber δ=1 on z / BCE-with-logits on 1{z>0}, model.py:238-265, spec §8.2).

    drop_labels (diagnostic only, not part of the official protocol): every visible cell of the target
    column other than the masked target (= the in-context labels, spec §2.5) is turned into padding, i.e.
    removed from the context, like the paper's "no labels" context ablation (Table 4, spec §11.2)."""
    net.eval()
    preds, labels = [], []
    tcol = cs.ctxs.task_by_id[cs.desc["tasks"][0]]["targetColNameIdx"]
    with torch.inference_mode():
        for s in range(0, cs.n, BATCH):
            b = cs.batch(np.arange(s, min(s + BATCH, cs.n)))
            if drop_labels:
                lab = (b["col_name_idxs"] == tcol) & ~b["is_targets"] & ~b["is_padding"]
                b["is_padding"] = b["is_padding"] | lab
            _, yhat = net(b)
            t = b["is_targets"]
            if target_type == "binary":
                preds.append(yhat["boolean"][t].flatten())
                labels.append(b["boolean_values"][t].flatten())
            else:
                preds.append(yhat["number"][t].flatten())
                labels.append(b["number_values"][t].flatten())
    p = torch.cat(preds).double()
    y = torch.cat(labels).double()
    if target_type == "binary":
        loss = F.binary_cross_entropy_with_logits(p, (y > 0).double(), reduction="none")
    else:
        loss = F.huber_loss(p, y, reduction="none")
    return p.numpy(), y.numpy(), loss.numpy()


def metric_of(target_type: str, p, y) -> dict:
    """AUROC on logits (binary) / R² in z-space (regression), main.py:328-337."""
    if target_type == "binary":
        return {"metric": "auroc", "value": float(roc_auc_score([int(v > 0) for v in y], p))}
    return {"metric": "r2", "value": float(r2_score(y, p))}


def eval_set(net, cs: ContextSet, target_type: str, drop_labels: bool = False) -> dict:
    """Metric and mean loss over the whole set."""
    p, y, loss = predict_set(net, cs, target_type, drop_labels)
    return {"loss": float(loss.mean()), "n": cs.n, **metric_of(target_type, p, y)}


def autocomplete_halves(cs: ContextSet) -> dict:
    """Index arrays of the two disjoint halves of an autocomplete eval set (see TARGET_AUTOCOMPLETE).
    Seeds are listed in node order and every row of the table is a seed, so position = review_id."""
    seeds = np.asarray(cs.a["seed_node"], dtype=np.int64)
    if cs.n > 1 and not (np.diff(seeds) == 1).all():
        raise SystemExit(f"[eval] {cs.name}: seeds are not consecutive rows; the even/odd split needs row ids")
    pos = np.arange(cs.n)
    return {"Val": pos[pos % 2 == 0], "Test": pos[pos % 2 == 1]}


def eval_sets_for(ctxs: Contexts, kind: str) -> list[tuple[str, str, str, str]]:
    """(set name, task id, split, target type) for 'target' (shop forecast val/test), 'autocomplete'
    (shop autocomplete, split 'All') and 'diag' (pretraining DBs' forecast Val)."""
    out = []
    for name, d in ctxs.index["sets"].items():
        if kind == "target" and name.startswith("eval/") and d["split"] in ("Val", "Test"):
            pass
        elif kind == "autocomplete" and name.startswith("eval/") and d["split"] == "All":
            pass
        elif kind == "diag" and name.startswith("diag/"):
            pass
        else:
            continue
        tid = d["tasks"][0]
        out.append((name, tid, d["split"], ctxs.task_by_id[tid]["targetType"]))
    return out


def eval_named_set(net, ctxs: Contexts, name: str, tid: str, split: str, tt: str, drop_labels: bool = False) -> dict:
    """{"<tid>/<split>": result}; an autocomplete set ('All') also yields its Val/Test halves."""
    cs = ctxs.set(name)
    p, y, loss = predict_set(net, cs, tt, drop_labels)
    res = {f"{tid}/{split}": {"loss": float(loss.mean()), "n": cs.n, **metric_of(tt, p, y)}}
    if split == "All":
        for half, idx in autocomplete_halves(cs).items():
            res[f"{tid}/{half}"] = {"loss": float(loss[idx].mean()), "n": int(len(idx)), **metric_of(tt, p[idx], y[idx])}
    return res


def evaluate(net, ctxs: Contexts, kinds=("target",)) -> dict:
    res = {}
    for kind in kinds:
        for name, tid, split, tt in eval_sets_for(ctxs, kind):
            res.update(eval_named_set(net, ctxs, name, tid, split, tt))
    return res


def entity_mean_baseline(ctxs: Contexts) -> dict:
    """EntityMean (paper App. I p.23) on the same eval seeds: AUROC / R² of the precomputed means."""
    out = {}
    for name, tid, split, tt in eval_sets_for(ctxs, "target") + eval_sets_for(ctxs, "diag"):
        cs = ctxs.set(name)
        y = np.asarray(cs.a["label"], dtype=np.float64)
        m = np.asarray(cs.a["entity_mean"], dtype=np.float64)
        npast = np.asarray(cs.a["entity_n_past"])
        r = {"n": cs.n, "fracWithPastLabels": float((npast > 0).mean()), "meanPastLabels": float(npast.mean())}
        if tt == "binary":
            r.update(metric="auroc", value=float(roc_auc_score((y > 0).astype(int), m)))
        else:
            r.update(metric="r2", value=float(r2_score(y, m)))
        out[f"{tid}/{split}"] = r
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Runs
# ──────────────────────────────────────────────────────────────────────────────
def new_model() -> RelationalTransformer:
    """seed_everything(0) then RelationalTransformer(...) (main.py:98, 170-176): PyTorch default init."""
    torch.manual_seed(SEED)
    return RelationalTransformer(**MINI, eps=PAPER_EPS)


def is_eval_step(step: int, eval_freq: int, pow2: bool) -> bool:
    """main.py:373-375 (`steps & (steps - 1) == 0` is also true for step 0)."""
    return (eval_freq is not None and step % eval_freq == 0) or (pow2 and step & (step - 1) == 0)


def r5(x: float) -> float:
    return float(f"{x:.5g}")


class Run:
    """A resumable training run (pretraining or one fine-tuning run)."""

    def __init__(self, name: str, train_set: str, max_steps: int, hp: dict, eval_kinds: tuple, select_tasks: list[str],
                 lr_schedule: bool, init_from: str | None, extra_eval_every: int | None = None,
                 base_kinds: tuple = ("target",)):
        self.name = name
        self.dir = RUNS_DIR / name
        self.train_set_name = train_set
        self.max_steps = max_steps
        self.hp = hp
        self.eval_kinds = eval_kinds
        self.select_tasks = select_tasks
        self.lr_schedule = lr_schedule
        self.init_from = init_from
        self.extra_eval_every = extra_eval_every  # eval_kinds beyond base_kinds: every n-th regular eval only
        self.base_kinds = base_kinds  # evaluated at every eval (everything checkpoint selection needs)

    @property
    def done(self) -> bool:
        p = self.dir / "state.pt"
        if not p.exists():
            return False
        return torch.load(p, map_location="cpu", weights_only=False)["steps"] >= self.max_steps

    def build(self):
        net = new_model()
        if self.init_from is not None:
            sd = torch.load(self.init_from, map_location="cpu", weights_only=True)
            net.load_state_dict(sd, strict=True)  # main.py:177-180; optimizer is NOT restored (fresh AdamW)
        # main.py:188-195: ONE param group (weight decay on every parameter, norms and biases included)
        opt = torch.optim.AdamW(net.parameters(), lr=self.hp["lr"], weight_decay=self.hp["wd"], betas=(0.9, 0.999), eps=1e-8)
        lrs = None
        if self.lr_schedule:  # main.py:197-204 (OneCycleLR defaults: cycle_momentum, div 25, final_div 1e4)
            lrs = torch.optim.lr_scheduler.OneCycleLR(
                opt, max_lr=self.hp["lr"], total_steps=self.max_steps, pct_start=0.2, anneal_strategy="linear",
                cycle_momentum=True, base_momentum=0.85, max_momentum=0.95, div_factor=25.0, final_div_factor=1e4,
            )
        return net, opt, lrs

    def run(self, ctxs: Contexts, deadline: float) -> bool:
        """Train until max_steps or the deadline. Returns True when finished."""
        self.dir.mkdir(parents=True, exist_ok=True)
        net, opt, lrs = self.build()
        state_path = self.dir / "state.pt"
        if state_path.exists():
            st = torch.load(state_path, map_location="cpu", weights_only=False)
            net.load_state_dict(st["model"])
            opt.load_state_dict(st["opt"])
            if lrs is not None:
                lrs.load_state_dict(st["lrs"])
            steps, log, best = st["steps"], st["log"], st["best"]
            if log.get("initFrom") != self.init_from or log.get("maxSteps") != self.max_steps or log.get("hp", {}).get("lr") != self.hp["lr"]:
                raise SystemExit(f"[{self.name}] stored run has initFrom={log.get('initFrom')}, maxSteps={log.get('maxSteps')}, "
                                 f"lr={log.get('hp', {}).get('lr')}; requested {self.init_from}, {self.max_steps}, {self.hp['lr']}. "
                                 f"Delete {self.dir} to restart it.")
            print(f"[{self.name}] resumed at step {steps}/{self.max_steps}")
        else:
            steps = 0
            log = {"step": [], "loss": [], "lr": [], "beta1": [], "gradNorm": [], "evals": [], "epochsReused": False,
                   "wallSeconds": 0.0, "trainSet": self.train_set_name, "maxSteps": self.max_steps, "hp": self.hp,
                   "lrSchedule": "OneCycleLR" if self.lr_schedule else "constant", "initFrom": self.init_from,
                   "attention": dict(ATTENTION_INFO), "threads": torch.get_num_threads()}
            best = {}
            torch.save(net.state_dict(), self.dir / "init.pt")
        if steps >= self.max_steps:
            return True

        ts = ctxs.set(self.train_set_name)
        N = ts.items_per_epoch
        n_batches = math.ceil(N / BATCH)  # len(loader) = ceil(N / (B·W)), W = 1 (fly.rs:332-334)
        t_start = time.time()
        n_regular_evals = sum(1 for e in log["evals"])

        def save():
            log["wallSeconds"] += time.time() - t_start_save[0]
            t_start_save[0] = time.time()
            st = {"model": net.state_dict(), "opt": opt.state_dict(), "lrs": lrs.state_dict() if lrs else None,
                  "steps": steps, "log": log, "best": best}
            torch.save(st, self.dir / "state.tmp")
            os.replace(self.dir / "state.tmp", state_path)

        t_start_save = [t_start]
        t_last_save = [t_start]
        t_last = [t_start]
        perm_epoch, perm = -1, None
        while steps < self.max_steps:
            # ── evaluation BEFORE the update of this step (main.py:373-392) ──
            if is_eval_step(steps, self.hp["eval_freq"], self.hp["eval_pow2"]):
                t0 = time.time()
                kinds = self.base_kinds
                if self.extra_eval_every and (n_regular_evals % self.extra_eval_every == 0 or steps + 1 >= self.max_steps):
                    kinds = self.eval_kinds
                elif not self.extra_eval_every:
                    kinds = self.eval_kinds
                res = evaluate(net, ctxs, kinds)
                n_regular_evals += 1
                entry = {"step": steps, "metrics": {k: r5(v["value"]) for k, v in res.items()},
                         "loss": {k: r5(v["loss"]) for k, v in res.items()}}
                log["evals"].append(entry)
                msg = []
                for tid in self.select_tasks:
                    val, test = res.get(f"{tid}/Val"), res.get(f"{tid}/Test")
                    if val is None:
                        continue
                    msg.append(f"{SHORT.get(tid, tid)} val {val['value']:.4f} test {test['value']:.4f}")
                    b = best.get(tid)
                    if b is None or val["value"] > b["val"]:  # strict '>' like main.py:383
                        best[tid] = {"step": steps, "val": val["value"], "test": test["value"]}
                        if f"{tid}/All" in res:
                            best[tid]["all"] = res[f"{tid}/All"]["value"]
                        torch.save(net.state_dict(), self.dir / f"best-{SHORT.get(tid, tid.replace('/', '_'))}.pt")
                others = [f"{k} {v['value']:.3f}" for k, v in res.items() if not any(k.startswith(t) for t in self.select_tasks)]
                print(f"[{self.name}] step {steps:5d} eval ({time.time() - t0:.1f}s): " + "; ".join(msg) +
                      (("  | " + ", ".join(others)) if others else ""), flush=True)

            # ── one training step (main.py:394-424) ──
            epoch = steps // n_batches  # shuffle_py(int(steps / len(loader))) (main.py:370)
            b_idx = steps % n_batches
            if epoch != perm_epoch:
                perm = np.random.default_rng(epoch + SEED).permutation(N)  # StdRng(epoch + seed) shuffle (fly.rs:324-328)
                perm_epoch = epoch
                if epoch >= ts.epochs:
                    log["epochsReused"] = True
            items = perm[(b_idx * BATCH + np.arange(BATCH)) % N]  # wrap-around last batch (fly.rs:345-350)
            batch = ts.batch([ts.context_index(epoch, int(i)) for i in items])
            net.train()
            loss, _ = net(batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=self.hp["max_grad_norm"])
            lr_used = opt.param_groups[0]["lr"]
            beta1_used = opt.param_groups[0]["betas"][0]
            opt.step()
            if lrs is not None:
                lrs.step()
            log["step"].append(steps)
            log["loss"].append(r5(loss.item()))
            log["lr"].append(float(f"{lr_used:.6g}"))
            log["beta1"].append(float(f"{beta1_used:.6g}"))
            log["gradNorm"].append(r5(float(grad_norm)))
            steps += 1
            if steps % 100 == 0:
                recent = log["loss"][-100:]
                now = time.time()
                print(f"[{self.name}] step {steps:5d}  loss(avg100) {sum(recent) / len(recent):.4f}  lr {lr_used:.2e}  "
                      f"beta1 {beta1_used:.3f}  {(now - t_last[0]) / 100:.3f} s/step (evals included)", flush=True)
                t_last[0] = now
            if time.time() > deadline and steps < self.max_steps:
                save()
                print(f"[{self.name}] budget reached at step {steps}; state saved (run again to resume)")
                return False
            # periodic checkpoint so an interrupted (killed) process resumes close to where it stopped;
            # batches are a pure function of the step (per-epoch permutation), so a resumed run is the same run
            if time.time() - t_last_save[0] > SAVE_EVERY_SECONDS and steps < self.max_steps:
                save()
                t_last_save[0] = time.time()
        save()
        torch.save(net.state_dict(), self.dir / "final.pt")
        print(f"[{self.name}] finished {steps} steps; best: {json.dumps(best)}")
        return True


def pretrain_run(args) -> Run:
    hp = {**PRETRAIN, "eval_freq": args.pretrain_eval_freq, "lr": args.pretrain_lr}
    return Run(args.pretrain_run, "pretrain", args.pretrain_steps, hp,
               eval_kinds=("target", "autocomplete", "diag"), select_tasks=TARGET_ALL, lr_schedule=True,
               init_from=None, extra_eval_every=3, base_kinds=("target", "autocomplete"))


def finetune_runs(args) -> list[Run]:
    runs = []
    for tid in TARGET_FORECAST:
        for init in ("pretrained", "untrained"):
            init_from = str(RUNS_DIR / args.pretrain_run / f"best-{SHORT[tid]}.pt") if init == "pretrained" else None
            runs.append(Run(f"finetune-{SHORT[tid]}-{init}", f"finetune/{tid}", args.finetune_steps,
                            {**FINETUNE, "eval_freq": args.finetune_eval_freq}, eval_kinds=("target",), select_tasks=[tid],
                            lr_schedule=False, init_from=init_from))
    return runs


# ──────────────────────────────────────────────────────────────────────────────
# Commands
# ──────────────────────────────────────────────────────────────────────────────
def cmd_bench(args, ctxs: Contexts):
    """Measure the wall time of one training step (batch build + forward + backward + AdamW)."""
    net = new_model()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.1)
    ts = ctxs.set("pretrain")
    rng = np.random.default_rng(0)
    times, tb = [], []
    for i in range(args.bench_steps + 5):
        t0 = time.time()
        b = ts.batch(rng.integers(0, ts.n, BATCH))
        t1 = time.time()
        loss, _ = net(b)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if i >= 5:
            times.append(time.time() - t0)
            tb.append(t1 - t0)
    t0 = time.time()
    r = evaluate(net, ctxs, ("target", "autocomplete", "diag"))
    t_eval_all = time.time() - t0
    t0 = time.time()
    evaluate(net, ctxs, ("target",))
    t_eval_target = time.time() - t0
    out = {"stepSeconds": float(np.median(times)), "batchBuildSeconds": float(np.median(tb)),
           "evalAllSeconds": t_eval_all, "evalTargetSeconds": t_eval_target, "threads": torch.get_num_threads()}
    print(json.dumps(out, indent=1))
    budget = 15 * 60
    n_evals = 30
    steps = int((budget - n_evals * t_eval_all * 0.6) / out["stepSeconds"])
    print(f"[bench] ≈{steps} steps fit in 15 min (with ~{n_evals} evals)")
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    (RUNS_DIR / "bench.json").write_text(json.dumps(out, indent=1))


def cmd_all(args, ctxs: Contexts) -> int:
    deadline = time.time() + args.budget_min * 60
    runs = [pretrain_run(args)] + finetune_runs(args)
    for r in runs:
        if r.done:
            continue
        if r.init_from is not None and not Path(r.init_from).exists():
            raise SystemExit(f"{r.name}: missing init checkpoint {r.init_from}")
        finished = r.run(ctxs, deadline)
        if not finished:
            return 3
        if time.time() > deadline - 30:
            print("[all] budget nearly exhausted; run again to continue")
            return 3
    cmd_export(args, ctxs)
    print("ALL DONE")
    return 0


def _load_state(name):
    p = RUNS_DIR / name / "state.pt"
    if not p.exists():
        return None
    return torch.load(p, map_location="cpu", weights_only=False)


def onecycle_reference(total_steps: int, points: int = 201) -> dict:
    """lr/β1 of the PAPER schedule (OneCycleLR, max_lr 1e-3, total 50,001 steps; spec §9.2), sampled."""
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([p], lr=1e-3, betas=(0.9, 0.999))
    lrs = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=total_steps, pct_start=0.2, anneal_strategy="linear")
    want = set(np.unique(np.linspace(0, total_steps - 1, points).round().astype(int)).tolist())
    out = {"step": [], "lr": [], "beta1": []}
    for s in range(total_steps):
        if s in want:
            out["step"].append(s)
            out["lr"].append(float(f"{opt.param_groups[0]['lr']:.6g}"))
            out["beta1"].append(float(f"{opt.param_groups[0]['betas'][0]:.6g}"))
        opt.step()
        lrs.step()
    return out


def cmd_export(args, ctxs: Contexts):
    """Write public/data/training/*.json and public/data/weights/* from the finished runs."""
    TRAINING_OUT.mkdir(parents=True, exist_ok=True)
    main_run = args.pretrain_run
    pre = _load_state(main_run)
    if pre is None:
        raise SystemExit("no pretraining run yet")
    bench = json.loads((RUNS_DIR / "bench.json").read_text()) if (RUNS_DIR / "bench.json").exists() else None

    # ── curves.json ─────────────────────────────────────────────────────────
    curves = {"mainPretrainRun": args.pretrain_run,
              "note": "loss = the model's own training loss of each optimizer step (mean over the batch's masked cells); "
                      "lr/beta1 = values USED by that step's update; evals happen BEFORE the update of their step "
                      "(main.py:373-424). Metrics: AUROC on logits (binary), R² in z-space (regression).",
              "runs": {}}
    pretrain_names = sorted(p.name for p in RUNS_DIR.iterdir() if p.is_dir() and p.name.startswith("pretrain"))
    run_names = pretrain_names + [r.name for r in finetune_runs(args)]
    states = {}
    for name in run_names:
        st = _load_state(name)
        if st is None:
            continue
        states[name] = st
        lg = st["log"]
        entry = {k: lg[k] for k in ("step", "loss", "gradNorm", "evals")}
        if name.startswith("pretrain"):
            entry["lr"], entry["beta1"] = lg["lr"], lg["beta1"]
            entry["main"] = name == main_run
        entry.update(maxSteps=lg["maxSteps"], lrSchedule=lg["lrSchedule"], hp=lg["hp"], trainSet=lg["trainSet"],
                     epochsReused=lg["epochsReused"], completedSteps=st["steps"], best=st["best"],
                     wallSeconds=round(lg["wallSeconds"], 1))
        curves["runs"][name] = entry
    (TRAINING_OUT / "curves.json").write_text(json.dumps(curves, separators=(",", ":")))

    # ── schedule.json ───────────────────────────────────────────────────────
    lg = pre["log"]
    schedule = {
        "note": "OneCycleLR(max_lr=1e-3, total_steps=max_steps, pct_start=0.2, anneal_strategy='linear'); defaults "
                "div_factor=25 (start 4e-5), final_div_factor=1e4 (end 4e-9), cycle_momentum=True (β1 0.95→0.85→0.95) "
                "(main.py:197-204; spec §9.2). 'mini' = our pretraining run (lr/β1 used by each update); "
                "'paper' = the same scheduler with the paper's 50,001 steps, sampled.",
        "mini": {"totalSteps": lg["maxSteps"], "step": lg["step"], "lr": lg["lr"], "beta1": lg["beta1"]},
        "paper": {"totalSteps": 50001, **onecycle_reference(50001)},
    }
    (TRAINING_OUT / "schedule.json").write_text(json.dumps(schedule, separators=(",", ":")))

    # ── dataset.json ────────────────────────────────────────────────────────
    idx = ctxs.index
    sets = {}
    for name, d in idx["sets"].items():
        sets[name] = {"contexts": d["n"], "epochs": d["epochs"], "itemsPerEpoch": d["itemsPerEpoch"], "tasks": d["tasks"],
                      "split": d["split"], "tokens": d["stats"]["tokens"], "realTokens": d["stats"]["realTokens"],
                      "meanRealTokens": round(d["stats"]["meanReal"], 2), "meanSelfLabels": round(d["stats"]["meanSelfLabels"], 2),
                      "meanOtherLabels": round(d["stats"]["meanOtherLabels"], 2)}
    pset = idx["sets"]["pretrain"]
    pretrain_tasks = [t for t in idx["tasks"] if t["id"] in pset["tasks"]]
    seen_ctx = min(pre["steps"], lg["maxSteps"]) * BATCH
    dataset = {
        "L": idx["L"], "w": idx["w"], "batchSize": BATCH, "generator": idx["generator"],
        "pretrain": {
            "dbs": idx["pretrainDbs"], "heldOut": idx["targetDb"], "numTasks": len(pretrain_tasks),
            "tasks": [{"id": t["id"], "kind": t["kind"], "targetType": t["targetType"],
                       "seeds": t["seeds"]["Train"] if t["kind"] == "forecast" else t["seeds"]["Train"]} for t in pretrain_tasks],
            "itemsPerEpoch": pset["itemsPerEpoch"], "exportedEpochs": pset["epochs"], "exportedContexts": pset["n"],
            "steps": lg["maxSteps"], "contextsSeen": seen_ctx, "tokensSeen": seen_ctx * idx["L"],
            "epochsSeen": round(seen_ctx / pset["itemsPerEpoch"], 2),
        },
        "paper": {"steps": 50001, "batch": 256, "L": 1024, "contextsSeen": 50001 * 256, "tokensSeen": 50001 * 256 * 1024,
                  "note": "tokens = steps × 256 × 1024 ≈ 13.1B (slides: '13B tokens')"},
        "sets": sets,
    }
    (TRAINING_OUT / "dataset.json").write_text(json.dumps(dataset, indent=1))

    # ── results.json ────────────────────────────────────────────────────────
    evals = lg["evals"]
    step0 = next(e for e in evals if e["step"] == 0)
    last = evals[-1]
    em = entity_mean_baseline(ctxs)
    zero_shot = {}
    for tid in TARGET_ALL:
        if tid not in pre["best"]:
            continue
        b = pre["best"][tid]
        sel = next(e for e in evals if e["step"] == b["step"])
        auto = tid in TARGET_AUTOCOMPLETE
        z = {
            "metric": "auroc" if ctxs.task_by_id[tid]["targetType"] == "binary" else "r2",
            "pretrainBest": {"step": b["step"], "val": r5(b["val"]), "test": r5(b["test"])},
            "pretrainFinal": {"step": last["step"], "val": last["metrics"][f"{tid}/Val"], "test": last["metrics"][f"{tid}/Test"]},
            "untrainedInit": {"step": 0, "val": step0["metrics"][f"{tid}/Val"], "test": step0["metrics"][f"{tid}/Test"]},
            "selectedStepOtherMetrics": {k: v for k, v in sel["metrics"].items() if not k.startswith(tid)},
        }
        if auto:
            # no entity, so no EntityMean (paper App. I defines it for forecast tasks only)
            z.update(kind="autocomplete", entityMean=None, split=AUTOCOMPLETE_HALVES,
                     all={"pretrainBest": sel["metrics"].get(f"{tid}/All"), "pretrainFinal": last["metrics"].get(f"{tid}/All"),
                          "untrainedInit": step0["metrics"].get(f"{tid}/All")})
        else:
            z.update(kind="forecast",
                     entityMean={"val": r5(em[f"{tid}/Val"]["value"]), "test": r5(em[f"{tid}/Test"]["value"]),
                                 "fracSeedsWithPastLabels": {"val": round(em[f"{tid}/Val"]["fracWithPastLabels"], 3),
                                                             "test": round(em[f"{tid}/Test"]["fracWithPastLabels"], 3)}})
        zero_shot[tid] = z
    # autocomplete + in-distribution diagnostics at the selected steps / final
    def at(step, key):
        for e in evals:
            if e["step"] == step and key in e["metrics"]:
                return e["metrics"][key]
        return None

    def last_with(key):
        for e in reversed(evals):
            if key in e["metrics"]:
                return {"step": e["step"], "value": e["metrics"][key]}
        return None

    extra = {}
    for name, tid, split, tt in eval_sets_for(ctxs, "autocomplete") + eval_sets_for(ctxs, "diag"):
        key = f"{tid}/{split}"
        vals = [(e["step"], e["metrics"][key]) for e in evals if key in e["metrics"]]
        best_step, best_val = max(vals, key=lambda x: x[1]) if vals else (None, None)
        extra[key] = {"metric": "auroc" if tt == "binary" else "r2", "init": at(0, key), "final": last_with(key),
                      "bestOverSteps": {"step": best_step, "value": best_val},
                      "entityMean": r5(em[key]["value"]) if key in em else None}
    finetune = {}
    for r in finetune_runs(args):
        st = states.get(r.name)
        if st is None:
            continue
        tid = r.select_tasks[0]
        ev = st["log"]["evals"]
        b = st["best"][tid]
        finetune[r.name] = {
            "task": tid, "init": "pretrain-best (" + SHORT[tid] + ")" if "pretrained" in r.name else "random (seed 0, same as pretraining init)",
            "steps": st["log"]["maxSteps"], "completedSteps": st["steps"],
            "best": {"step": b["step"], "val": r5(b["val"]), "test": r5(b["test"])},
            "step0": {"val": ev[0]["metrics"][f"{tid}/Val"], "test": ev[0]["metrics"][f"{tid}/Test"]},
            "final": {"step": ev[-1]["step"], "val": ev[-1]["metrics"][f"{tid}/Val"], "test": ev[-1]["metrics"][f"{tid}/Test"]},
        }
    results = {
        "note": "Zero-shot = no gradient step on ShopDB (leave-one-DB-out pretraining on school+racing+forum). "
                "Checkpoint selection per target task on ShopDB VAL (paper protocol, spec §9.1); 'test' is the test "
                "metric of the selected step. AUROC (binary) on logits; R² (regression) in z-space. EntityMean = mean "
                "of the entity's past labels, fallback Train mean (paper App. I). The two autocomplete tasks "
                "(reviews.rating, reviews.verified) are our extension (the paper evaluates forecast tasks only): all "
                "303 review rows are seeds, " + AUTOCOMPLETE_HALVES + ". Single seed, tiny eval sets (churn val/test "
                "71/74 seeds, sales 30/30, reviews 152/151): expect large variance.",
        "config": {**MINI, "eps": PAPER_EPS, "L": idx["L"], "w": idx["w"], "batch": BATCH, "dtype": "float32", "device": "cpu",
                   "threads": lg.get("threads"), "attention": lg.get("attention")},
        "pretrain": {"steps": lg["maxSteps"], "evalFreq": lg["hp"]["eval_freq"], "wallSeconds": round(lg["wallSeconds"], 1),
                     "stepSeconds": bench["stepSeconds"] if bench else None, "epochsReused": lg["epochsReused"]},
        "mainPretrainRun": main_run,
        "pretrainRuns": {
            n: {"peakLr": st["log"]["hp"]["lr"], "steps": st["log"]["maxSteps"], "completedSteps": st["steps"],
                "best": {t: {k: (r5(v) if isinstance(v, float) else v) for k, v in b.items()} for t, b in st["best"].items()},
                "final": {k: v for k, v in st["log"]["evals"][-1]["metrics"].items() if k.startswith(tuple(TARGET_ALL))},
                "wallSeconds": round(st["log"]["wallSeconds"], 1)}
            for n, st in states.items() if n.startswith("pretrain")},
        "pretrainRunsNote": (
            "'pretrain' uses the paper's peak LR 1e-3 and is the main run. 'pretrain-sqrtlr' is the ONE fix tried after "
            "diagnosing that ShopDB zero-shot peaks early and then decays while the pretraining loss keeps falling "
            "(specialisation to the 3 source DBs): Adam square-root LR scaling for our 8× smaller batch, peak "
            "1e-3·sqrt(32/256) = 3.54e-4, everything else identical. It did not improve ShopDB VAL for either task, so the "
            "paper-LR run is kept (selection on val, spec §9.1)." if "pretrain-sqrtlr" in states else
            "'pretrain' uses the paper's peak LR 1e-3 (OneCycleLR) and is the only pretraining run on data version 2. "
            "On data version 1 a second run with Adam square-root LR scaling for our 8× smaller batch (peak 3.54e-4) did "
            "not improve ShopDB VAL (see previousDataVersion), so it was not repeated."),
        "dataVersion": DATA_VERSION,
        "dataVersionNote": DATA_VERSION_NOTE,
        "previousDataVersion": _previous_data_version(),
        "zeroShot": zero_shot,
        "extraZeroShotAndDiagnostics": extra,
        "finetune": finetune,
        "checkpoints": {},
    }

    # ── weights ─────────────────────────────────────────────────────────────
    WEIGHTS_OUT.mkdir(parents=True, exist_ok=True)
    cfg = infer_config(pre["model"], MINI["num_heads"], PAPER_EPS)
    ckpts = {
        "init": (RUNS_DIR / main_run / "init.pt", {"step": 0, "run": main_run, "selection": "random init: torch.manual_seed(0) + PyTorch defaults"}),
        "pretrain-churn": (RUNS_DIR / main_run / "best-churn.pt", {"run": main_run, "selection": "best ShopDB customer-churn VAL AUROC", **_sel(pre, "shop/customer-churn")}),
        "pretrain-sales": (RUNS_DIR / main_run / "best-sales.pt", {"run": main_run, "selection": "best ShopDB product-sales VAL R²", **_sel(pre, "shop/product-sales")}),
        "pretrain-rating": (RUNS_DIR / main_run / "best-rating.pt", {"run": main_run, "selection": "best ShopDB reviews.rating VAL R² (even review_ids)", **_sel(pre, "shop/reviews.rating")}),
        "pretrain-verified": (RUNS_DIR / main_run / "best-verified.pt", {"run": main_run, "selection": "best ShopDB reviews.verified VAL AUROC (even review_ids)", **_sel(pre, "shop/reviews.verified")}),
        "pretrain-final": (RUNS_DIR / main_run / "final.pt", {"run": main_run, "step": lg["maxSteps"], "selection": "last step",
                                                             "note": "the site's default for school/racing/forum tasks; those DBs were SEEN in pretraining, so it is not zero-shot there"}),
    }
    for tid in TARGET_FORECAST:
        st = states.get(f"finetune-{SHORT[tid]}-pretrained")
        if st is not None:
            ckpts[f"finetune-{SHORT[tid]}"] = (RUNS_DIR / f"finetune-{SHORT[tid]}-pretrained" / f"best-{SHORT[tid]}.pt",
                                              {"run": f"finetune-{SHORT[tid]}-pretrained", "selection": f"best ShopDB {tid} VAL",
                                               **_sel(st, tid)})
    total = 0
    for name, (path, meta) in ckpts.items():
        if not path.exists():
            print(f"[export] skip {name}: {path} missing")
            continue
        sd = torch.load(path, map_location="cpu", weights_only=True)
        meta = {"name": name, "trainedBy": "python/train.py", "config": "MINI", **meta}
        j, b = export_state_dict(sd, WEIGHTS_OUT, name, cfg, meta)
        total += j.stat().st_size + b.stat().st_size
        results["checkpoints"][name] = meta
    results["checkpoints"]["totalBytes"] = total
    stats_path = HERE / "data" / "label-stats.json"  # npx tsx python/label_stats.ts
    if stats_path.exists():
        results["labelPersistence"] = json.loads(stats_path.read_text())
    results["diagnostics"] = diagnostics(ctxs, {k: v[0] for k, v in ckpts.items() if k != "init"} | {"init": ckpts["init"][0]})
    (TRAINING_OUT / "results.json").write_text(json.dumps(results, indent=1))
    print(f"[export] training JSON → {TRAINING_OUT}; weights → {WEIGHTS_OUT} ({total / 1e6:.2f} MB)")
    print(json.dumps({"zeroShot": zero_shot, "finetune": finetune}, indent=1))


def diagnostics(ctxs: Contexts, ckpts: dict) -> dict:
    """(1) In-context-label ablation of each checkpoint on the ShopDB eval sets (forecast val/test,
    autocomplete All + halves); (2) zero-shot of untrained models with other init seeds (how much of the
    step-0 metric is chance)."""
    out = {"note": "withoutLabels: every visible cell of the target column except the masked one is removed "
                   "(turned into padding) — an inference-time analogue of the paper's 'no labels' context "
                   "ablation (Table 4; the paper re-samples contexts, we drop tokens). For forecast tasks these are "
                   "the self (and other-entity) task-row labels; for autocomplete tasks the other rows' values of "
                   "the target column. untrainedInitSeeds: zero-shot of PyTorch-default inits with "
                   "torch.manual_seed(s); seed 0 is our 'init'.",
           "inContextLabelAblation": {}, "untrainedInitSeeds": {}}
    sets = eval_sets_for(ctxs, "target") + eval_sets_for(ctxs, "autocomplete")
    for name, path in ckpts.items():
        if not Path(path).exists():
            continue
        net = new_model()
        net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        ab = {}
        for sn, tid, split, tt in sets:
            w = eval_named_set(net, ctxs, sn, tid, split, tt)
            wo = eval_named_set(net, ctxs, sn, tid, split, tt, drop_labels=True)
            for k in w:
                ab[k] = {"with": r5(w[k]["value"]), "withoutLabels": r5(wo[k]["value"])}
        out["inContextLabelAblation"][name] = ab
    for seed in range(0, 7):
        torch.manual_seed(seed)
        net = RelationalTransformer(**MINI, eps=PAPER_EPS)
        r = {}
        for sn, tid, split, tt in sets:
            r.update({k: r5(v["value"]) for k, v in eval_named_set(net, ctxs, sn, tid, split, tt).items()})
        out["untrainedInitSeeds"][str(seed)] = r
    return out


def _previous_data_version() -> dict | None:
    """Headline numbers of the run on data version 1 (snapshot of its results.json), kept for the record:
    before the generators made pretraining labels persistent (python/README.md)."""
    p = HERE / "data" / "old" / "runs-data-v1" / "public-data" / "results.json"
    if not p.exists():
        return None
    old = json.loads(p.read_text())
    return {
        "dataVersion": 1,
        "note": "Pretraining DBs before the persistence fix: EntityMean R² on Val was −0.77 (school course-submissions) "
                "and −0.43 (forum post-votes), racing driver-dnf had lag-1 correlation −0.04. 3,001 pretraining steps. "
                "ShopDB (target) and its eval contexts are identical in both versions.",
        "pretrainSteps": old["pretrain"]["steps"],
        "zeroShot": {t: {k: z[k] for k in ("metric", "pretrainBest", "pretrainFinal", "untrainedInit", "entityMean") if k in z}
                     for t, z in old["zeroShot"].items()},
        "finetune": {n: {k: f[k] for k in ("best", "step0", "final")} for n, f in old.get("finetune", {}).items()},
    }


def _sel(st, tid):
    b = st["best"][tid]
    out = {"step": b["step"], "val": r5(b["val"]), "test": r5(b["test"])}
    if "all" in b:
        out["all"] = r5(b["all"])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["bench", "all", "pretrain", "finetune", "export", "baselines"])
    ap.add_argument("--budget-min", type=float, default=8.5, help="wall-clock budget of this invocation (minutes)")
    ap.add_argument("--pretrain-steps", type=int, default=4001)
    ap.add_argument("--pretrain-lr", type=float, default=PRETRAIN["lr"], help="peak LR of OneCycleLR (paper 1e-3)")
    ap.add_argument("--pretrain-run", default="pretrain", help="run name (python/data/runs/<name>); fine-tuning and "
                    "export use this run's checkpoints")
    ap.add_argument("--pretrain-eval-freq", type=int, default=200)
    ap.add_argument("--finetune-steps", type=int, default=1025)
    ap.add_argument("--finetune-eval-freq", type=int, default=64)
    ap.add_argument("--task", default="shop/customer-churn")
    ap.add_argument("--init", choices=["pretrained", "untrained"], default="pretrained")
    ap.add_argument("--bench-steps", type=int, default=20)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    att = _install_fast_attention()
    ATTENTION_INFO.update(att)
    print(f"[train] torch {torch.__version__}, threads {torch.get_num_threads()}, attention kernel: {att['kernel']} "
          f"(max |Δ| vs dense reference: fwd {att['maxAbsDiffForward']:.1e}, grad {att['maxAbsDiffGrad']:.1e})")
    ctxs = Contexts()
    if args.cmd == "bench":
        cmd_bench(args, ctxs)
    elif args.cmd == "all":
        sys.exit(cmd_all(args, ctxs))
    elif args.cmd == "pretrain":
        sys.exit(0 if pretrain_run(args).run(ctxs, time.time() + args.budget_min * 60) else 3)
    elif args.cmd == "finetune":
        r = next(r for r in finetune_runs(args) if r.select_tasks[0] == args.task and r.name.endswith(args.init))
        sys.exit(0 if r.run(ctxs, time.time() + args.budget_min * 60) else 3)
    elif args.cmd == "export":
        cmd_export(args, ctxs)
    elif args.cmd == "baselines":
        print(json.dumps(entity_mean_baseline(ctxs), indent=1))


if __name__ == "__main__":
    main()
