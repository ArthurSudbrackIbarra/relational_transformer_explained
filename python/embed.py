"""Embed every string of the dummy DBs with the frozen RT text encoder (BUILD_PLAN §4 step 2).

Mirrors PE/rt/embed.py (spec §3.6): SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")
and its default ``encode`` (Transformer -> mean Pooling -> Normalize, max_seq_length = 128), so every
vector is L2-unit-norm and 384-d. Differences, by design:
  * device = CPU and float32 output (paper era: fp32 on cuda:0, stored as raw bf16; spec §3.6-3.7);
  * no torch.compile (same numerics, only slower).

Inputs : python/data/strings.json        (written by scripts/export-strings.ts)
Outputs: public/data/embeddings/minilm-l12-v2.f32   float32 little-endian, N x 384, row-major
         public/data/embeddings/minilm-l12-v2.json  {model, dim, count, strings} (same order as rows)
         public/data/embeddings/pca2d.json          2-D PCA of the column phrases only

Usage: python/.venv/bin/python python/embed.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L12-v2"
DIM = 384

ROOT = Path(__file__).resolve().parent.parent
STRINGS_JSON = ROOT / "python" / "data" / "strings.json"
OUT_DIR = ROOT / "public" / "data" / "embeddings"


def pca_2d(x: np.ndarray) -> np.ndarray:
    """Top-2 principal-component scores of the rows of x (centred SVD), with a fixed sign convention."""
    xc = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(xc.astype(np.float64), full_matrices=False)
    comps = vt[:2]
    # make the largest-magnitude loading of each component positive (deterministic orientation)
    for i in range(2):
        if comps[i, np.argmax(np.abs(comps[i]))] < 0:
            comps[i] = -comps[i]
    return xc @ comps.T


def main() -> None:
    data = json.loads(STRINGS_JSON.read_text(encoding="utf-8"))
    strings: list[str] = data["strings"]
    assert len(strings) == data["count"] == len(set(strings)), "strings must be unique"

    model = SentenceTransformer(MODEL_NAME, device="cpu")
    assert model.max_seq_length == 128, model.max_seq_length  # sentence_bert_config.json (spec §3.6)
    emb = model.encode(strings, batch_size=256, convert_to_numpy=True, show_progress_bar=False)
    emb = np.asarray(emb, dtype=np.float32)

    # ── checks: shape, finite, unit norm (Normalize module) ────────────────────────────────
    assert emb.shape == (len(strings), DIM), emb.shape
    assert np.isfinite(emb).all()
    norms = np.linalg.norm(emb.astype(np.float64), axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), (norms.min(), norms.max())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    f32_path = OUT_DIR / "minilm-l12-v2.f32"
    emb.astype("<f4").tofile(f32_path)
    meta = {"model": MODEL_NAME, "dim": DIM, "count": len(strings), "strings": strings}
    (OUT_DIR / "minilm-l12-v2.json").write_text(json.dumps(meta, ensure_ascii=False) + "\n", encoding="utf-8")

    # ── 2-D PCA of the column phrases ("<col> of <table>") of all four DBs ──────────────────
    index = {s: i for i, s in enumerate(strings)}
    phrases = data["columnPhrases"]
    xy = pca_2d(emb[[index[p["text"]] for p in phrases]])
    pca = [
        {
            "text": p["text"],
            "db": p["db"],
            "table": p["table"],
            "col": p["col"],
            "x": round(float(x), 5),
            "y": round(float(y), 5),
        }
        for p, (x, y) in zip(phrases, xy)
    ]
    (OUT_DIR / "pca2d.json").write_text(json.dumps(pca, ensure_ascii=False) + "\n", encoding="utf-8")

    # ── verify the written file round-trips ─────────────────────────────────────────────────
    back = np.fromfile(f32_path, dtype="<f4").reshape(-1, DIM)
    assert back.shape[0] == len(strings) and np.array_equal(back, emb)

    sizes = {p.name: p.stat().st_size for p in sorted(OUT_DIR.iterdir())}
    print(f"embedded {len(strings)} strings -> {f32_path} ({emb.nbytes} bytes)")
    print(f"  norms: min {norms.min():.7f} max {norms.max():.7f}")
    print(f"  pca2d: {len(pca)} column phrases")
    print(f"  files: {sizes}, total {sum(sizes.values())} bytes")


if __name__ == "__main__":
    main()
