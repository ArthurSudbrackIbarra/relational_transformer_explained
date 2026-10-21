"""state_dict <-> the site's weight format (read by src/engine/model/weights.ts).

Format ("rt-weights-v1"):
  <name>.bin   float32 little-endian; every tensor's values concatenated in state_dict order (C order)
  <name>.json  {"format": "rt-weights-v1",
                "config": {"dModel", "nHeads", "dFF", "nBlocks", "dText", "eps"},
                "meta":   {...provenance...},
                "tensors": {"<state_dict name>": {"shape": [...], "offset": <elements>, "length": <elements>}}}

Tensor names are the PyTorch state_dict names of PE/rt/model.py (identical in python/rt_model.py).

CLI:
  python python/export_weights.py CKPT.pt OUT_DIR NAME --heads 4 [--eps 0.0078125] [--meta '{"step": 123}']
  (d_model, d_ff, d_text and the number of blocks are read from the tensor shapes; the number of
  heads and eps are not recoverable from shapes and must be given.)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rt_model import PAPER_EPS, clean_state_dict  # noqa: E402

FORMAT = "rt-weights-v1"


def infer_config(sd: dict, num_heads: int, eps: float = PAPER_EPS) -> dict:
    """RTConfig (camelCase, as in src/engine/types.ts) from tensor shapes + heads + eps."""
    d_model, d_text = sd["enc_dict.text.weight"].shape
    d_ff = sd["blocks.0.ffn.w1.weight"].shape[0]
    blocks = {int(m.group(1)) for k in sd if (m := re.match(r"blocks\.(\d+)\.", k))}
    n_blocks = max(blocks) + 1
    if blocks != set(range(n_blocks)):
        raise ValueError(f"non-contiguous block indices {sorted(blocks)}")
    if d_model % num_heads:
        raise ValueError(f"d_model {d_model} not divisible by heads {num_heads}")
    return {
        "dModel": int(d_model),
        "nHeads": int(num_heads),
        "dFF": int(d_ff),
        "nBlocks": int(n_blocks),
        "dText": int(d_text),
        "eps": float(eps),
    }


def export_state_dict(sd: dict, out_dir, name: str, config: dict, meta: dict | None = None):
    """Write OUT_DIR/NAME.bin and OUT_DIR/NAME.json. Returns (json_path, bin_path)."""
    sd = clean_state_dict(sd)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tensors = {}
    chunks = []
    offset = 0
    for k, v in sd.items():  # state_dict order = module registration order
        arr = v.detach().to(torch.float32).cpu().contiguous().numpy().astype("<f4", copy=False).ravel()
        tensors[k] = {"shape": [int(s) for s in v.shape], "offset": offset, "length": int(arr.size)}
        chunks.append(arr)
        offset += int(arr.size)
    bin_path = out_dir / f"{name}.bin"
    json_path = out_dir / f"{name}.json"
    np.concatenate(chunks).astype("<f4").tofile(bin_path)
    manifest = {"format": FORMAT, "config": config, "meta": meta or {}, "tensors": tensors}
    json_path.write_text(json.dumps(manifest, indent=1))
    return json_path, bin_path


def load_exported(prefix) -> tuple[dict, dict, dict]:
    """Reverse direction: PREFIX.{json,bin} -> (state_dict of float32 tensors, config, meta).
    PREFIX may also be given with its .json/.bin extension. (Suffixes are appended, not swapped
    with Path.with_suffix, so names containing dots such as "ckpt.step1000" work.)"""
    prefix = str(prefix)
    if prefix.endswith((".json", ".bin")):
        prefix = prefix.rsplit(".", 1)[0]
    manifest = json.loads(Path(prefix + ".json").read_text())
    if manifest.get("format", FORMAT) != FORMAT:
        raise ValueError(f"unsupported format {manifest.get('format')}")
    flat = np.fromfile(prefix + ".bin", dtype="<f4")
    sd = {}
    for k, t in manifest["tensors"].items():
        a = flat[t["offset"] : t["offset"] + t["length"]]
        if a.size != t["length"]:
            raise ValueError(f"{k}: truncated .bin")
        sd[k] = torch.from_numpy(a.astype(np.float32).reshape(t["shape"]).copy())
    return sd, manifest["config"], manifest.get("meta", {})


def model_from_exported(prefix):
    """Instantiate python/rt_model.RelationalTransformer from an exported checkpoint."""
    from rt_model import RelationalTransformer

    sd, cfg, meta = load_exported(prefix)
    net = RelationalTransformer(
        num_blocks=cfg["nBlocks"],
        d_model=cfg["dModel"],
        d_text=cfg["dText"],
        num_heads=cfg["nHeads"],
        d_ff=cfg["dFF"],
        eps=cfg["eps"],
    )
    net.load_state_dict(sd, strict=True)
    return net, cfg, meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt", help="torch .pt file holding a state_dict (or {'model'|'state_dict': sd})")
    ap.add_argument("out_dir")
    ap.add_argument("name")
    ap.add_argument("--heads", type=int, required=True)
    ap.add_argument("--eps", type=float, default=PAPER_EPS)
    ap.add_argument("--meta", default="{}", help="JSON object merged into meta")
    args = ap.parse_args()
    obj = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    if isinstance(obj, dict) and ("model" in obj or "state_dict" in obj) and not any(k.startswith("enc_dict") for k in obj):
        obj = obj.get("model", obj.get("state_dict"))
    sd = clean_state_dict(obj)
    cfg = infer_config(sd, args.heads, args.eps)
    meta = {"source": str(args.ckpt), **json.loads(args.meta)}
    j, b = export_state_dict(sd, args.out_dir, args.name, cfg, meta)
    print(f"wrote {j} and {b} ({sum(v.numel() for v in sd.values()):,} params, config {cfg})")


if __name__ == "__main__":
    main()
