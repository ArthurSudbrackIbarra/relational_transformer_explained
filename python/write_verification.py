"""Compose public/data/verification.json (read by the "How this was built" page) from the outputs of
the verification steps. Nothing is computed here; every number comes from one of these files:

  python/data/parity-report.json      python/parity.py --stage all   (official PE/rt/model.py vs python/rt_model.py)
  python/data/ts-parity.json          npx tsx python/ts_parity.ts    (TS forward vs PyTorch on every fixture)
  python/data/sampler-crosscheck.json python/sampler_crosscheck.py   (independent Python pre.rs + fly.rs + masks)
  python/data/vitest.json             npx vitest run --reporter=json --outputFile=python/data/vitest.json
  public/data/weights/*.json|bin      the shipped checkpoints (sha256 computed here)

Usage: python/.venv/bin/python python/write_verification.py
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = HERE / "data"
WEIGHTS = ROOT / "public" / "data" / "weights"
OUT = ROOT / "public" / "data" / "verification.json"


def _max_diff(d: dict) -> float:
    return max((float(v) for k, v in d.items() if k.startswith(("maxAbsDiff", "absDiff")) and isinstance(v, (int, float))), default=0.0)


def main():
    parity = json.loads((DATA / "parity-report.json").read_text())
    ts = json.loads((DATA / "ts-parity.json").read_text())
    sampler = json.loads((DATA / "sampler-crosscheck.json").read_text())
    vitest = json.loads((DATA / "vitest.json").read_text())

    trained = parity["trained"]
    per_ckpt = {name: _max_diff(r) for name, r in trained["checkpoints"].items() if isinstance(r, dict)}
    fixtures = {name: _max_diff(r) for name, r in parity.get("fixtures", {}).items()}
    official_vs_port = max([_max_diff(parity["random"]), *per_ckpt.values(), *fixtures.values()])

    results = json.loads((ROOT / "public" / "data" / "training" / "results.json").read_text())
    meta = results.get("checkpoints", {})
    checkpoints = []
    for j in sorted(WEIGHTS.glob("*.json")):
        name = j.stem
        b = j.with_suffix(".bin")
        m = meta.get(name, {})
        checkpoints.append({"name": name, "bytes": j.stat().st_size + b.stat().st_size,
                            "sha256": hashlib.sha256(b.read_bytes()).hexdigest(),
                            "step": m.get("step"), "selection": m.get("selection"),
                            "officialVsPortMaxAbsDiff": per_ckpt.get(name)})

    out = {
        "official_vs_port_max_abs_diff": official_vs_port,
        "ts_vs_torch_max_abs_diff": ts["maxAbsDiff"],
        "n_contexts": trained["contexts"],
        "checkpoints": [c["name"] for c in checkpoints],
        "sampler_crosscheck": {"contexts": sampler["contexts"], "mismatches": sampler["mismatches"],
                               "note": "independent Python re-implementation of pre.rs+fly.rs"},
        "tests": vitest["numPassedTests"],
        "date": time.strftime("%Y-%m-%d"),
        "details": {
            "note": "official = PE/rt/model.py (paper-era commit 2a899c8, FlexAttention on CPU, RMSNorm eps pinned to 2^-7); "
                    "port = python/rt_model.py (dense masked attention); TS = src/engine/model/forward.ts. n_contexts = real "
                    "exported contexts (all four DBs) on which every checkpoint was compared official vs port; the random-weight "
                    "stage adds 2 synthetic contexts and the trained fixtures 3 real ones (each compared at every activation, TS "
                    "vs PyTorch). All float32 on CPU.",
            "officialVsPort": {"randomWeights": _max_diff(parity["random"]), "perCheckpoint": per_ckpt,
                               "trainedFixtures": fixtures, "attention": trained.get("attention")},
            "tsVsTorch": ts["fixtures"],
            "samplerCrosscheck": {**sampler, "how": "python/sampler_dump.ts dumps the raw DBs and the TS preprocess/sampler/"
                                  "mask outputs (per task and split: 8 + 12 random seeds, at (L, w) = (128, 16), (512, 3), "
                                  "(64, 1), (1024, 256)); python/sampler_crosscheck.py rebuilds them from the Rust/Python "
                                  "sources with a bit-exact port of the site's PRNG and compares every token, visit order "
                                  "and mask."},
            "testsTotal": vitest["numTotalTests"],
            "testsFailed": vitest["numFailedTests"],
            "testFiles": vitest["numTotalTestSuites"],
            "checkpoints": checkpoints,
            "torch": parity.get("torch"),
        },
    }
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "details"}, indent=1))
    print(f"→ {OUT}")


if __name__ == "__main__":
    main()
