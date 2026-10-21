"""Parity checks for the RT model port, stage 1 (random weights).

(a) python/rt_model.py == the OFFICIAL paper-era class (PE/rt/model.py, FlexAttention) on a
    random but structurally valid synthetic batch (B=2, S=128): rows with parents, rows whose
    parents are outside the context, rows with no children in context (empty neighbour rows),
    text cells, a padding tail, masked number / boolean (and one extra datetime) cells.
    Same state_dict in both; every nn.RMSNorm pinned to eps = 2**-7. Reports max |Δ| of every
    decoder output and of the loss, and of all parameter gradients.
(b) Dumps the fixture consumed by src/engine/model/forward.test.ts:
      tests/fixtures/parity-random-weights.{json,bin}   weights in the site's export format
      tests/fixtures/parity-random.{json,bin}           tokens (Token fields), the text table and its
                                                        embeddings, the official dense masks, and
                                                        expected x0, x after every sublayer,
                                                        xFinal, yhat at all positions, loss.

Usage:  python/.venv/bin/python python/parity.py [--official PATH/TO/rt/model.py] [--seed 0]
(The official model.py path defaults to $RT_OFFICIAL_MODEL_PY or the session scratchpad copy.)
"""

from __future__ import annotations

import argparse
import base64
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from export_weights import export_state_dict, infer_config  # noqa: E402
from rt_model import PAPER_EPS, RelationalTransformer, build_attn_masks, set_rmsnorm_eps  # noqa: E402

DEFAULT_OFFICIAL = (
    "/tmp/claude-1004/-home-ibarra-Documents-Mestrado-Relational-Transformer/"
    "155ecce6-1f67-4870-9609-2c9740593feb/scratchpad/upstream/rt-paper-era/rt/model.py"
)

MINI = dict(num_blocks=4, d_model=64, d_text=384, num_heads=4, d_ff=256)
REPORT: dict = {}  # machine-readable summary → --report (python/data/parity-report.json)
SEM = {"number": 0, "text": 1, "datetime": 2, "boolean": 3}
MAX_F2P = 5


# ──────────────────────────────────────────────────────────────────────────────
# Official model import
# ──────────────────────────────────────────────────────────────────────────────
def load_official(path: str):
    """Import PE/rt/model.py as a module. It runs torch.compile(flex_attention) at import (works on
    CPU via inductor). If compilation is unavailable, fall back to the eager flex_attention and
    create_block_mask(_compile=False) — same maths, slower."""
    spec = importlib.util.spec_from_file_location("rt_official_model", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mode = "compiled flex_attention (inductor, CPU)"
    try:
        # distinct q/k/v buffers: the CPU lowering rejects q is k is v
        q, k, v = (torch.randn(2, 4, 128, 16) for _ in range(3))
        bm = mod._make_block_mask(torch.ones(2, 128, 128, dtype=torch.bool), 2, 128, "cpu")
        mod.flex_attention(q, k, v, block_mask=bm)
    except Exception as e:  # pragma: no cover - environment dependent
        from torch.nn.attention import flex_attention as fa

        print(f"[parity] compiled flex_attention failed ({type(e).__name__}: {e}); using eager fallback")
        mod.flex_attention = fa.flex_attention
        orig = fa.create_block_mask

        def _make_block_mask(mask, batch_size, seq_len, device):
            def _mod(b, h, q_idx, kv_idx):
                return mask[b, q_idx, kv_idx]

            return orig(mask_mod=_mod, B=batch_size, H=None, Q_LEN=seq_len, KV_LEN=seq_len, device=device, _compile=False)

        mod._make_block_mask = _make_block_mask
        mode = "eager flex_attention (fallback)"
    return mod, mode


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic, structurally valid relational context
# ──────────────────────────────────────────────────────────────────────────────
class Text:
    """Per-DB string table, interned in first-seen order like pre.rs:406-437, 595-597."""

    def __init__(self):
        self.strings: list[str] = []
        self.idx: dict[str, int] = {}

    def __call__(self, s: str) -> int:
        if s not in self.idx:
            self.idx[s] = len(self.strings)
            self.strings.append(s)
        return self.idx[s]


SCHEMA = [
    # (table, table_type, n_rows, parent tables (FK order), [(col, sem)], is_task)
    ("customers", "Db", 10, [], [("country", "text"), ("age", "number"), ("is_premium", "boolean"), ("signup_date", "datetime")], False),
    ("orders", "Db", 30, ["customers", "products"], [("order_time", "datetime"), ("quantity", "number"), ("channel", "text")], False),
    ("products", "Db", 8, [], [("category", "text"), ("price", "number"), ("name", "text")], False),
    ("reviews", "Db", 20, ["customers", "products"], [("review_time", "datetime"), ("rating", "number"), ("verified", "boolean"), ("review_text", "text")], False),
    ("customer-churn", "Train", 24, ["customers"], [("timestamp", "datetime"), ("churn", "boolean")], True),
    ("customer-churn", "Val", 8, ["customers"], [("timestamp", "datetime"), ("churn", "boolean")], True),
]
VOCAB = {
    "country": ["BR", "US", "DE", "JP", "IN", "FR"],
    "channel": ["web", "app", "store"],
    "category": ["electronics", "books", "home", "toys", "sports"],
    "name": ["Wireless Mouse", "Desk Lamp", "Novel", "Puzzle", "Yoga Mat", "Kettle", "Headphones", "Board Game"],
    "review_text": ["Great product, works as described.", "Terrible, broke after a day.", "Okay for the price.", "Would buy again!"],
}


def build_db(rng: np.random.Generator):
    text = Text()
    nodes = []  # dicts
    offsets = {}
    nxt = 0
    for tname, ttype, n, parents, cols, is_task in SCHEMA:
        tidx = text(tname)
        for c, _ in cols:
            text(f"{c} of {tname}")
        offsets[(tname, ttype)] = nxt
        nxt += n
    # boolean z-score constants per column (true > 0 > false), like a 0/1 z-score
    bool_vals = {}
    for tname, ttype, n, parents, cols, is_task in SCHEMA:
        for c, s in cols:
            if s == "boolean":
                p = rng.uniform(0.25, 0.75)
                sd = np.sqrt(p * (1 - p))
                bool_vals[(tname, c)] = ((1 - p) / sd, -p / sd)
    for tname, ttype, n, parents, cols, is_task in SCHEMA:
        off = offsets[(tname, ttype)]
        for r in range(n):
            f2p = []
            for ptab in parents:
                pn = [x for x in SCHEMA if x[0] == ptab][0][2]
                f2p.append(offsets[(ptab, "Db")] + int(rng.integers(pn)))
            cells = []
            for c, s in cols:
                if not is_task and s != "datetime" and rng.random() < 0.12:
                    continue  # null cell -> no token (spec §3.5)
                cell = dict(col=c, col_idx=text(f"{c} of {tname}"), sem=SEM[s], number=0.0, text_idx=0, datetime=0.0, boolean=0.0, class_idx=-1)
                if s == "number":
                    cell["number"] = float(np.float32(rng.standard_normal()))
                    cell["raw"] = round(float(rng.uniform(1, 100)), 2)
                elif s == "datetime":
                    cell["datetime"] = float(np.float32(rng.standard_normal()))
                    cell["raw"] = int(1.7e12 + rng.integers(0, 30_000_000_000))
                elif s == "boolean":
                    t, f = bool_vals[(tname, c)]
                    b = bool(rng.random() < 0.5)
                    cell["boolean"] = float(np.float32(t if b else f))
                    cell["raw"] = b
                else:
                    v = VOCAB[c][int(rng.integers(len(VOCAB[c])))]
                    ti = text(v)
                    cell["text_idx"] = ti
                    cell["class_idx"] = ti
                    cell["raw"] = v
                cells.append(cell)
            nodes.append(dict(node_idx=off + r, table=tname, table_type=ttype, table_idx=text.idx[tname], row_pos=r, f2p=f2p, cells=cells, is_task=is_task))
    children = {n["node_idx"]: [] for n in nodes}
    for n in nodes:
        for p in n["f2p"]:
            children[p].append(n["node_idx"])
    return text, {n["node_idx"]: n for n in nodes}, children


def sample_context(rng, nodes, children, seed_node, target_col, S, max_real):
    """BFS-like walk (parents first, then children), emitting every cell of each visited row.
    Not the fly.rs sampler — only needs to be a structurally valid context for the model."""
    order, seen, frontier = [], set(), [seed_node]
    depth = {seed_node: 0}
    while frontier and sum(len(nodes[n]["cells"]) for n in order) < max_real:
        n = frontier.pop(0)
        if n in seen:
            continue
        seen.add(n)
        order.append(n)
        # Parents (F->P) with priority. Deeper rows sometimes keep their parents out of the context
        # (as when L cuts the walk), so masks must also handle parent ids absent from the sequence.
        skip_parents = depth[n] >= 2 and rng.random() < 0.4
        for p in nodes[n]["f2p"]:
            if p not in seen and not skip_parents:
                depth.setdefault(p, depth[n] + 1)
                frontier.insert(0, p)
        kids = [c for c in children[n] if c not in seen]
        rng.shuffle(kids)
        for c in kids[:4]:  # a few children (P->F)
            depth.setdefault(c, depth[n] + 1)
            frontier.append(c)
    toks = []
    for n in order:
        node = nodes[n]
        for cell in node["cells"]:
            if len(toks) >= max_real:
                break  # may cut a row (fly.rs:503-509)
            is_target = n == seed_node and cell["col"] == target_col
            toks.append(
                dict(
                    pos=len(toks),
                    isPadding=False,
                    nodeIdx=n,
                    f2pNbrIdxs=(node["f2p"] + [-1] * MAX_F2P)[:MAX_F2P],
                    tableNameIdx=node["table_idx"],
                    colNameIdx=cell["col_idx"],
                    sem=cell["sem"],
                    numberValue=cell["number"],
                    textIdx=cell["text_idx"],
                    datetimeValue=cell["datetime"],
                    booleanValue=cell["boolean"],
                    mask=is_target,
                    isTarget=is_target,
                    isTaskNode=node["is_task"] or is_target,
                    classValueIdx=cell["class_idx"],
                    tableName=node["table"],
                    tableType=node["table_type"],
                    colName=cell["col"],
                    rowPos=node["row_pos"],
                    raw=cell["raw"],
                    depth=depth[n],
                )
            )
    while len(toks) < S:  # padding defaults (fly.rs:65-84)
        toks.append(
            dict(pos=len(toks), isPadding=True, nodeIdx=-1, f2pNbrIdxs=[-1] * MAX_F2P, tableNameIdx=0, colNameIdx=0, sem=0,
                 numberValue=0.0, textIdx=0, datetimeValue=0.0, booleanValue=0.0, mask=False, isTarget=False, isTaskNode=False,
                 classValueIdx=-1, tableName="", tableType=None, colName="", rowPos=-1, raw=None, depth=-1)
        )
    assert sum(t["isTarget"] for t in toks) == 1, "target cell must be in context"
    return toks


def to_batch(seqs, emb: np.ndarray):
    """Token lists -> the official batch dict (fly.rs:462-500, data.py:116-154), float32."""
    B, S = len(seqs), len(seqs[0])
    D = emb.shape[1]
    out = {
        "node_idxs": np.full((B, S), -1, np.int32),
        "f2p_nbr_idxs": np.full((B, S, MAX_F2P), -1, np.int32),
        "table_name_idxs": np.zeros((B, S), np.int32),
        "col_name_idxs": np.zeros((B, S), np.int32),
        "class_value_idxs": np.full((B, S), -1, np.int32),
        "col_name_values": np.zeros((B, S, D), np.float32),
        "sem_types": np.zeros((B, S), np.int32),
        "number_values": np.zeros((B, S, 1), np.float32),
        "text_values": np.zeros((B, S, D), np.float32),
        "datetime_values": np.zeros((B, S, 1), np.float32),
        "boolean_values": np.zeros((B, S, 1), np.float32),
        "masks": np.zeros((B, S), bool),
        "is_targets": np.zeros((B, S), bool),
        "is_task_nodes": np.zeros((B, S), bool),
        "is_padding": np.ones((B, S), bool),
    }
    for b, toks in enumerate(seqs):
        for i, t in enumerate(toks):
            if t["isPadding"]:
                continue
            out["node_idxs"][b, i] = t["nodeIdx"]
            out["f2p_nbr_idxs"][b, i] = t["f2pNbrIdxs"]
            out["table_name_idxs"][b, i] = t["tableNameIdx"]
            out["col_name_idxs"][b, i] = t["colNameIdx"]
            out["class_value_idxs"][b, i] = t["classValueIdx"]
            out["col_name_values"][b, i] = emb[t["colNameIdx"]]
            out["sem_types"][b, i] = t["sem"]
            out["number_values"][b, i, 0] = t["numberValue"]
            out["text_values"][b, i] = emb[t["textIdx"]]  # non-text cells: text[0]'s embedding (pre.rs:559, fly.rs:476-478)
            out["datetime_values"][b, i, 0] = t["datetimeValue"]
            out["boolean_values"][b, i, 0] = t["booleanValue"]
            out["masks"][b, i] = t["mask"]
            out["is_targets"][b, i] = t["isTarget"]
            out["is_task_nodes"][b, i] = t["isTaskNode"]
            out["is_padding"][b, i] = False
    return {k: torch.from_numpy(v) for k, v in out.items()}


def slice_batch(batch, b):
    return {k: v[b : b + 1] for k, v in batch.items()}


def pack_mask(m: np.ndarray) -> str:
    return base64.b64encode(np.packbits(m.astype(np.uint8).ravel(), bitorder="little").tobytes()).decode()


# ──────────────────────────────────────────────────────────────────────────────
def stage_random(args) -> bool:
    """Stage 1: random weights, synthetic context (see module docstring)."""
    # NB: do not raise torch.set_num_threads above the default here: with 8 threads the compiled CPU
    # FlexAttention kernel of torch 2.12 aborts ("malloc(): unaligned tcache chunk") on this machine.
    rng = np.random.default_rng(args.seed)
    S = 128  # B = 2 sequences (seq0, seq1)

    # ── synthetic DB + two contexts ─────────────────────────────────────────
    text, nodes, children = build_db(rng)
    review_seed = [n for n in nodes.values() if n["table"] == "reviews" and any(c["col"] == "rating" for c in n["cells"])][0]["node_idx"]
    churn_seed = [n for n in nodes.values() if n["table"] == "customer-churn" and n["table_type"] == "Val"][0]["node_idx"]
    seq0 = sample_context(rng, nodes, children, review_seed, "rating", S, max_real=101)
    seq1 = sample_context(rng, nodes, children, churn_seed, "churn", S, max_real=93)
    # Exercise the datetime loss path too: one extra masked datetime cell in seq 0 (multi-cell
    # masking is the Table-14 ablation, not the default; the model handles any number of masks).
    extra = next(t for t in seq0 if not t["isPadding"] and t["sem"] == SEM["datetime"] and not t["isTarget"] and t["nodeIdx"] != review_seed)
    extra["mask"] = True
    seqs = [seq0, seq1]

    # synthetic unit-norm "embeddings" for every string of the text table (stage 1: random)
    emb = rng.standard_normal((len(text.strings), MINI["d_text"])).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb.astype(np.float32)
    batch = to_batch(seqs, emb)

    # ── models ────────────────────────────────────────────────────────────────
    official_mod, mode = load_official(args.official)
    torch.manual_seed(args.seed)
    official = official_mod.RelationalTransformer(**MINI)
    n_norms = set_rmsnorm_eps(official, PAPER_EPS)
    # Perturb the default init so the test can catch layer/norm mix-ups: γ ~ U(0.5, 1.5) for every
    # RMSNorm (all-ones γ would hide a wrong-norm bug) and sharper attention (wq, wk × 3).
    with torch.no_grad():
        g = torch.Generator().manual_seed(args.seed + 1)
        for name, p in official.named_parameters():
            if p.ndim == 1 and ("norm" in name):
                p.copy_(torch.rand(p.shape, generator=g) + 0.5)
            if name.endswith(("wq.weight", "wk.weight")):
                p.mul_(3.0)
    official.eval()
    sd = official.state_dict()

    port = RelationalTransformer(**MINI, eps=PAPER_EPS)
    assert list(port.state_dict().keys()) == list(sd.keys()), "state_dict keys/order differ"
    port.load_state_dict(sd, strict=True)
    port.eval()
    print(f"[parity] official: {args.official}\n[parity] attention: {mode}; RMSNorms pinned to eps=2**-7: {n_norms}")
    print(f"[parity] params: {sum(p.numel() for p in port.parameters()):,}; text table: {len(text.strings)} strings")

    # ── (a) official vs port ─────────────────────────────────────────────────
    with torch.no_grad():
        t0 = time.time()
        loss_o, yhat_o = official(batch)
        t1 = time.time()
        port.trace = []
        loss_p, yhat_p = port(batch)
        trace = dict(port.trace)
        port.trace = None
    report = {"attention": mode, "official_seconds": round(t1 - t0, 3)}
    for t in ["number", "text", "datetime", "boolean"]:
        report[f"maxAbsDiff_yhat_{t}"] = float((yhat_o[t] - yhat_p[t]).abs().max())
    report["loss_official"] = float(loss_o)
    report["loss_port"] = float(loss_p)
    report["absDiff_loss"] = float(abs(loss_o - loss_p))
    ok = all(v <= 1e-5 for k, v in report.items() if k.startswith(("maxAbsDiff", "absDiff")))

    if not args.no_grad_check:
        try:
            official.zero_grad()
            port.zero_grad()
            official(batch)[0].backward()
            port(batch)[0].backward()
            gd = 0.0
            for (n1, p1), (n2, p2) in zip(official.named_parameters(), port.named_parameters()):
                assert n1 == n2
                if p1.grad is None and p2.grad is None:
                    continue
                g1 = p1.grad if p1.grad is not None else torch.zeros_like(p1)
                g2 = p2.grad if p2.grad is not None else torch.zeros_like(p2)
                gd = max(gd, float((g1 - g2).abs().max()))
            report["maxAbsDiff_grad"] = gd
            ok = ok and gd <= 1e-5
        except Exception as e:  # pragma: no cover
            # FlexAttention has no CPU backward; at least check that the port's backward is NaN-free
            # despite fully masked attention rows (padding queries, empty neighbour rows).
            report["grad_check_vs_official"] = f"skipped: {type(e).__name__}: {str(e).splitlines()[0][:120]}"
            port.zero_grad()
            port(batch)[0].backward()
            finite = all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in port.parameters())
            report["port_grads_finite"] = finite
            ok = ok and finite
            port.zero_grad()

    print("[parity] (a) official vs port:", json.dumps(report, indent=1))
    print("[parity] (a)", "PASS" if ok else "FAIL")
    REPORT["random"] = {**report, "contexts": len(seqs), "pass": ok}

    # ── (b) fixture ──────────────────────────────────────────────────────────
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = infer_config(sd, MINI["num_heads"], PAPER_EPS)
    export_state_dict(sd, out, "parity-random-weights", cfg, {"name": "parity-random", "seed": args.seed,
                      "init": "pytorch default (torch.manual_seed(seed)) + RMSNorm γ~U(0.5,1.5) + wq,wk×3"})

    chunks: list[np.ndarray] = []
    off = [0]

    def put(a) -> dict:
        a = np.asarray(a.detach().cpu() if isinstance(a, torch.Tensor) else a, dtype=np.float32).ravel()
        ref = {"offset": off[0], "length": int(a.size)}
        chunks.append(a)
        off[0] += int(a.size)
        return ref

    masks_all = build_attn_masks(batch)
    sublayer_names = [f"block{i}.{l}" for i in range(MINI["num_blocks"]) for l in ["col", "feat", "nbr", "full", "ffn"]]
    sequences = []
    for b, toks in enumerate(seqs):
        with torch.no_grad():
            loss_b, _ = port(slice_batch(batch, b))  # per-sequence loss (what the TS forward returns)
        target = next(i for i, t in enumerate(toks) if t["isTarget"])
        head = {0: "number", 2: "datetime", 3: "boolean"}[toks[target]["sem"]]
        sequences.append({
            "description": ["review row, target 'rating' (number) + one extra masked datetime cell",
                            "customer-churn Val row, target 'churn' (boolean)"][b],
            "tokens": toks,
            "numReal": sum(not t["isPadding"] for t in toks),
            "targetPos": target,
            "masks": {l: pack_mask(masks_all[l][b].numpy()) for l in ["col", "feat", "nbr", "full"]},
            "expected": {
                "x0": put(trace["x0"][b]),
                "sublayers": {n: put(trace[n][b]) for n in sublayer_names},
                "xFinal": put(trace["xFinal"][b]),
                "yhat": {t: put(yhat_p[t][b]) for t in ["number", "datetime", "boolean"]},
                "loss": float(loss_b),
                "prediction": float(yhat_p[head][b, target, 0]),
                "predictionHead": head,
            },
        })
    emb_ref = put(emb)
    np.concatenate(chunks).astype("<f4").tofile(out / "parity-random.bin")
    fixture = {
        "format": "rt-parity-v1",
        "meta": {
            "generator": "python/parity.py",
            "seed": args.seed,
            "torch": torch.__version__,
            "dtype": "float32",
            "official": "PE/rt/model.py (commit 2a899c8), " + mode,
            "embeddings": "synthetic random unit vectors (stage 1), NOT MiniLM",
            "officialVsPort": report,
            "note": "Values are float32 refs {offset,length} into parity-random.bin (elements). "
                    "Masks are packbits(bitorder='little') of the dense S×S official masks, base64.",
        },
        "weights": "parity-random-weights",
        "config": cfg,
        "S": S,
        "text": text.strings,
        "embeddings": {**emb_ref, "dim": MINI["d_text"]},
        "sublayerOrder": sublayer_names,
        "sequences": sequences,
        "batchLoss": float(loss_p),
    }
    (out / "parity-random.json").write_text(json.dumps(fixture))
    size = sum(c.size for c in chunks) * 4
    print(f"[parity] (b) wrote {out/'parity-random.json'}, {out/'parity-random.bin'} ({size/1e6:.2f} MB), "
          f"{out/'parity-random-weights.json'}, .bin")
    for b, s in enumerate(sequences):
        print(f"[parity]     seq {b}: real={s['numReal']} target={s['targetPos']} loss={s['expected']['loss']:.6f} "
              f"pred({s['expected']['predictionHead']})={s['expected']['prediction']:.6f}")
    return ok



# ──────────────────────────────────────────────────────────────────────────────
# Stage 2: TRAINED weights on REAL contexts built by the TypeScript engine
# ──────────────────────────────────────────────────────────────────────────────
WEIGHTS_DIR = ROOT / "public" / "data" / "weights"
# pipeline.pickDefaultSeed(shop/customer-churn, Val) = node 2406 (customer-churn:Val[32], the showcase
# customer; asserted in src/engine/pipeline.test.ts). It is the context every chapter shows by default.
DEFAULT_TRAINED_SEED = 2406
TRAINED_CHECKPOINTS = ["init", "pretrain-churn", "pretrain-sales", "pretrain-rating", "pretrain-verified", "pretrain-final",
                       "finetune-churn", "finetune-sales"]
# Extra trained fixtures (besides the default churn one, which keeps its own copy of the weights):
# (fixture name, context set, checkpoint). The context is the set's seed with the most visible self
# labels (ties → first). Their weights are the shipped public/data/weights/<checkpoint> files (sha256
# recorded), so the fixture holds only the context and the activations.
EXTRA_TRAINED_FIXTURES = [
    ("parity-trained-sales", "eval/shop/product-sales/Val", "pretrain-sales"),  # number head, ShopDB
    ("parity-trained-forum", "diag/forum/post-votes/Val", "pretrain-final"),  # a pretraining DB
]


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _compare(official, port, batch) -> dict:
    """Official vs port on `batch`, run in chunks of exactly B = 2 sequences: torch 2.12's inductor
    fails to compile the CPU FlexAttention kernel for some batch sizes (e.g. B = 16: C++ error
    "cur_qSplitSize8 was not declared"), while B = 2 (the stage-1 shape) compiles. Sequences do not
    interact inside the model, so chunking changes nothing but the per-chunk loss average, which is
    compared chunk by chunk. A single sequence is duplicated to B = 2."""
    B = batch["node_idxs"].shape[0]
    idx = list(range(B)) + ([B - 1] if B % 2 else [])
    r = {f"maxAbsDiff_yhat_{t}": 0.0 for t in ["number", "text", "datetime", "boolean"]}
    r.update(maxAbsDiff_target_boolean=0.0, maxAbsDiff_target_number=0.0, absDiff_loss=0.0)
    for c in range(0, len(idx), 2):
        sub = {k: v[idx[c : c + 2]] for k, v in batch.items()}
        with torch.no_grad():
            lo, yo = official(sub)
            lp, yp = port(sub)
        for t in ["number", "text", "datetime", "boolean"]:
            r[f"maxAbsDiff_yhat_{t}"] = max(r[f"maxAbsDiff_yhat_{t}"], float((yo[t] - yp[t]).abs().max()))
        tgt = sub["is_targets"]
        r["maxAbsDiff_target_boolean"] = max(r["maxAbsDiff_target_boolean"], float((yo["boolean"][tgt] - yp["boolean"][tgt]).abs().max()))
        r["maxAbsDiff_target_number"] = max(r["maxAbsDiff_target_number"], float((yo["number"][tgt] - yp["number"][tgt]).abs().max()))
        r["absDiff_loss"] = max(r["absDiff_loss"], float(abs(lo - lp)))
    with torch.no_grad():
        r["loss"] = float(port(batch)[0])
    return r


def stage_trained(args) -> bool:
    """(a) official PE/rt/model.py == python/rt_model.py with every exported TRAINED checkpoint on a
    batch of real ShopDB contexts (exported by scripts/export-contexts.ts, i.e. built by the TS engine);
    (b) dump tests/fixtures/parity-trained.{json,bin} (+ parity-trained-weights.*): the default
    shop/customer-churn Val context with the pretrain-churn checkpoint, every intermediate activation of
    the port, for src/engine/model/forward.test.ts (TS == PyTorch ≤ 1e-4)."""
    from export_weights import load_exported
    from train import Contexts  # numpy and torch are already imported: see train.py import note

    ctxs = Contexts()
    # A mixed batch of real contexts: both ShopDB forecast tasks (Val/Test), both ShopDB autocomplete tasks
    # and the forecast Val sets of the three pretraining DBs, so both loss paths (BCE, Huber), both
    # read-outs and every DB's schema are exercised. B = 28.
    picks = [("eval/shop/customer-churn/Val", 4), ("eval/shop/customer-churn/Test", 2), ("eval/shop/product-sales/Val", 4),
             ("eval/shop/product-sales/Test", 2), ("eval/shop/reviews.verified/All", 2), ("eval/shop/reviews.rating/All", 2),
             ("diag/school/student-dropout/Val", 2), ("diag/school/course-submissions/Val", 2),
             ("diag/racing/driver-dnf/Val", 2), ("diag/racing/driver-points/Val", 2),
             ("diag/forum/user-engagement/Val", 2), ("diag/forum/post-votes/Val", 2)]
    parts = []
    for name, k in picks:
        cs = ctxs.set(name)
        idx = np.linspace(0, cs.n - 1, k).round().astype(int)
        parts.append(cs.batch(idx))
    batch = {key: torch.cat([p[key] for p in parts]) for key in parts[0]}

    official_mod, mode = load_official(args.official)
    report = {"attention": mode, "batch": {"B": int(batch["node_idxs"].shape[0]), "S": int(batch["node_idxs"].shape[1]),
                                           "sets": {n: k for n, k in picks}}, "checkpoints": {}}
    ok = True
    for ck in TRAINED_CHECKPOINTS:
        prefix = WEIGHTS_DIR / ck
        if not (WEIGHTS_DIR / f"{ck}.json").exists():
            report["checkpoints"][ck] = "missing"
            print(f"[parity] (a) {ck}: missing, skipped")
            continue
        sd, cfg, meta = load_exported(prefix)
        official = official_mod.RelationalTransformer(num_blocks=cfg["nBlocks"], d_model=cfg["dModel"], d_text=cfg["dText"],
                                                      num_heads=cfg["nHeads"], d_ff=cfg["dFF"])
        set_rmsnorm_eps(official, cfg["eps"])
        official.load_state_dict(sd, strict=True)
        official.eval()
        port = RelationalTransformer(num_blocks=cfg["nBlocks"], d_model=cfg["dModel"], d_text=cfg["dText"],
                                     num_heads=cfg["nHeads"], d_ff=cfg["dFF"], eps=cfg["eps"])
        port.load_state_dict(sd, strict=True)
        port.eval()
        r = _compare(official, port, batch)
        good = all(v <= 1e-5 for k, v in r.items() if k.startswith(("maxAbsDiff", "absDiff")))
        ok = ok and good
        report["checkpoints"][ck] = r
        print(f"[parity] (a) {ck:15s} official vs port: max|Δ yhat| bool {r['maxAbsDiff_yhat_boolean']:.2e} "
              f"number {r['maxAbsDiff_yhat_number']:.2e} text {r['maxAbsDiff_yhat_text']:.2e}  |Δ loss| {r['absDiff_loss']:.2e}  "
              f"{'PASS' if good else 'FAIL'}")

    REPORT["trained"] = {**report, "contexts": int(batch["node_idxs"].shape[0]), "pass": ok}

    # ── (b) fixtures: the default churn Val context + pretrain-churn (own weights copy), and extra ones ──
    REPORT["fixtures"] = {}
    ok = dump_trained_fixture(ctxs, official_mod, mode, report, args, "parity-trained", "eval/shop/customer-churn/Val",
                              args.trained_seed, args.trained_ckpt, own_weights=True) and ok
    for name, set_name, ck in EXTRA_TRAINED_FIXTURES:
        cs = ctxs.set(set_name)
        seed = int(np.asarray(cs.a["seed_node"])[int(np.argmax(np.asarray(cs.a["n_self_labels"])))])
        ok = dump_trained_fixture(ctxs, official_mod, mode, report, args, name, set_name, seed, ck, own_weights=False) and ok
    return ok


def dump_trained_fixture(ctxs, official_mod, mode, report, args, name: str, set_name: str, seed: int, ck: str,
                         own_weights: bool) -> bool:
    """Dump tests/fixtures/<name>.{json,bin}: one exported context of `set_name` (seed node `seed`) with
    checkpoint `ck`, every intermediate activation of the port, and official-vs-port on that context.
    own_weights: also copy the weights to <name>-weights.{json,bin}; else the TS test loads the shipped
    public/data/weights/<ck> (its sha256 is recorded here)."""
    from export_weights import load_exported

    cs = ctxs.set(set_name)
    task_id = cs.desc["tasks"][0]
    split = cs.desc["split"]
    hits = np.nonzero(np.asarray(cs.a["seed_node"]) == seed)[0]
    if len(hits) != 1:
        raise SystemExit(f"seed {seed} not in {set_name}")
    ci = int(hits[0])
    one = cs.batch([ci])
    sd, cfg, meta = load_exported(WEIGHTS_DIR / ck)
    port = RelationalTransformer(num_blocks=cfg["nBlocks"], d_model=cfg["dModel"], d_text=cfg["dText"],
                                 num_heads=cfg["nHeads"], d_ff=cfg["dFF"], eps=cfg["eps"])
    port.load_state_dict(sd, strict=True)
    port.eval()
    official = official_mod.RelationalTransformer(num_blocks=cfg["nBlocks"], d_model=cfg["dModel"], d_text=cfg["dText"],
                                                  num_heads=cfg["nHeads"], d_ff=cfg["dFF"])
    set_rmsnorm_eps(official, cfg["eps"])
    official.load_state_dict(sd, strict=True)
    official.eval()
    fixture_vs_official = _compare(official, port, one)
    with torch.no_grad():
        port.trace = []
        loss, yhat = port(one)
        trace = dict(port.trace)
        port.trace = None
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if own_weights:
        export_state_dict(sd, out, f"{name}-weights", cfg, {**meta, "copiedFrom": f"public/data/weights/{ck}"})

    chunks: list[np.ndarray] = []
    off = [0]

    def put(a) -> dict:
        a = np.asarray(a.detach().cpu() if isinstance(a, torch.Tensor) else a, dtype=np.float32).ravel()
        ref = {"offset": off[0], "length": int(a.size)}
        chunks.append(a)
        off[0] += int(a.size)
        return ref

    sublayer_names = [f"block{i}.{l}" for i in range(cfg["nBlocks"]) for l in ["col", "feat", "nbr", "full", "ffn"]]
    target = int(cs.a["target_pos"][ci])
    head = "boolean" if ctxs.task_by_id[task_id]["targetType"] == "binary" else "number"
    masks_all = build_attn_masks(one)
    a = cs.a
    fixture = {
        "format": "rt-parity-trained-v1",
        "meta": {
            "generator": "python/parity.py --stage trained",
            "torch": torch.__version__,
            "dtype": "float32",
            "official": "PE/rt/model.py (commit 2a899c8), " + mode,
            "embeddings": "real all-MiniLM-L12-v2 vectors (public/data/embeddings)",
            "context": f"exported by scripts/export-contexts.ts (TS engine: preprocess + sampleContext), "
                       f"set {set_name}, epoch 0, rngSeed = 0 + seed_node + 0 (fly.rs:371-375)",
            "officialVsPort": fixture_vs_official,
            "officialVsPortCheckpoints": report,
            "note": "Expected values are float32 refs {offset,length} into the fixture .bin (elements). Token "
                    "arrays are the exported batch layout (values: number/datetime/boolean slot of the cell's "
                    "own sem type; flags bit0 masks, bit1 is_targets, bit2 is_padding, bit3 is_task_nodes).",
        },
        "dbName": task_id.split("/")[0],
        "taskId": task_id,
        "split": split,
        "seedNode": seed,
        "L": int(ctxs.L),
        "w": int(ctxs.index["w"]),
        "rngSeed": 0,
        "checkpoint": ck,
        "checkpointSha256": _sha256(WEIGHTS_DIR / f"{ck}.bin"),
        "weights": f"{name}-weights" if own_weights else None,
        "config": cfg,
        "S": int(ctxs.L),
        "targetPos": target,
        "batch": {
            "nodeIdxs": a["node_idxs"][ci].astype(int).tolist(),
            "f2pNbrIdxs": a["f2p_nbr_idxs"][ci].astype(int).tolist(),
            "tableNameIdxs": a["table_name_idxs"][ci].astype(int).tolist(),
            "colNameIdxs": a["col_name_idxs"][ci].astype(int).tolist(),
            "semTypes": a["sem_types"][ci].astype(int).tolist(),
            "values": [float(v) for v in np.asarray(a["values"][ci], dtype=np.float32)],
            "textIdxs": a["text_idxs"][ci].astype(int).tolist(),
            "flags": a["flags"][ci].astype(int).tolist(),
        },
        "masks": {l: pack_mask(masks_all[l][0].numpy()) for l in ["col", "feat", "nbr", "full"]},
        "sublayerOrder": sublayer_names,
        "expected": {
            "x0": put(trace["x0"][0]),
            "sublayers": {n: put(trace[n][0]) for n in sublayer_names},
            "xFinal": put(trace["xFinal"][0]),
            "yhat": {t: put(yhat[t][0]) for t in ["number", "datetime", "boolean"]},
            "loss": float(loss),
            "prediction": float(yhat[head][0, target, 0]),
            "predictionHead": head,
            "label": float(a["label"][ci]),
        },
    }
    np.concatenate(chunks).astype("<f4").tofile(out / f"{name}.bin")
    (out / f"{name}.json").write_text(json.dumps(fixture))
    good = all(v <= 1e-5 for k, v in fixture_vs_official.items() if k.startswith(("maxAbsDiff", "absDiff")))
    REPORT["fixtures"][name] = {"set": set_name, "seedNode": seed, "checkpoint": ck, **fixture_vs_official, "pass": good}
    print(f"[parity] (b) {name}: {set_name} seed {seed}, target pos {target}, label z {fixture['expected']['label']:.4f}, "
          f"{head} {fixture['expected']['prediction']:.6f}, loss {fixture['expected']['loss']:.6f}; official vs port "
          f"max|Δ yhat| {max(v for k, v in fixture_vs_official.items() if k.startswith('maxAbsDiff')):.2e} {'PASS' if good else 'FAIL'}")
    print(f"[parity] (b) wrote {out / (name + '.json')}, {out / (name + '.bin')}" + (f", {out / (name + '-weights')}.{{json,bin}}" if own_weights else ""))
    return good


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--official", default=os.environ.get("RT_OFFICIAL_MODEL_PY", DEFAULT_OFFICIAL))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "tests" / "fixtures"))
    ap.add_argument("--no-grad-check", action="store_true")
    ap.add_argument("--stage", choices=["random", "trained", "all"], default="random",
                    help="random: stage 1 (synthetic context, random weights); trained: stage 2 (real exported "
                         "ShopDB contexts, trained checkpoints from public/data/weights); all: both")
    ap.add_argument("--trained-seed", type=int, default=DEFAULT_TRAINED_SEED)
    ap.add_argument("--trained-ckpt", default="pretrain-churn")
    ap.add_argument("--report", default=str(HERE / "data" / "parity-report.json"),
                    help="machine-readable summary (read by python/write_verification.py)")
    args = ap.parse_args()
    ok = True
    if args.stage == "all":
        # Each stage runs in its OWN process. In one process, stage 1's gradient check (FlexAttention has no
        # CPU backward → exception inside torch.compile) leaves dynamo running flex_attention eagerly for the
        # rest of the process, so stage 2 would silently compare against the eager kernel (bit-identical to
        # the port's dense attention, |Δ| = 0) instead of the compiled one it reports.
        import subprocess
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            for st in ("random", "trained"):
                rep = Path(tmp) / f"{st}.json"
                cmd = [sys.executable, str(Path(__file__).resolve()), "--stage", st, "--official", args.official,
                       "--seed", str(args.seed), "--out", args.out, "--trained-seed", str(args.trained_seed),
                       "--trained-ckpt", args.trained_ckpt, "--report", str(rep)]
                if args.no_grad_check:
                    cmd.append("--no-grad-check")
                code = subprocess.run(cmd).returncode
                ok = ok and code == 0
                if rep.exists():
                    sub = json.loads(rep.read_text())
                    REPORT.update({k: v for k, v in sub.items() if k not in ("stage", "ok", "torch", "official", "date")})
                else:
                    ok = False
    if args.stage == "random":
        ok = stage_random(args) and ok
    if args.stage == "trained":
        ok = stage_trained(args) and ok
    REPORT.update(stage=args.stage, ok=ok, torch=torch.__version__, official=args.official,
                  date=time.strftime("%Y-%m-%d %H:%M:%S %z"))
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(REPORT, indent=1))
    print(f"[parity] report → {args.report}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
