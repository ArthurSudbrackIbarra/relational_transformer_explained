# Relational Transformer Explained

An interactive website that walks through every step of the **Relational Transformer** (Ranjan et al., *Relational Transformer: Toward Zero-Shot Foundation Models for Relational Data*, ICLR 2026, [arXiv 2510.06377](https://arxiv.org/abs/2510.06377)). It runs a real, scaled-down RT in your browser on small synthetic relational databases.

Every operation mirrors the paper-era official code ([snap-stanford/relational-transformer](https://github.com/snap-stanford/relational-transformer) @ `2a899c8`). Only the size is reduced:

|  | paper | this site |
|---|---|---|
| d_model / heads / blocks | 256 / 8 / 12 | 64 / 4 / 4 |
| context length L / BFS width w | 1024 / 256 | 128 / 16 |
| data | RelBench | 4 synthetic DBs (ShopDB held out for zero-shot) |

## Run it

```bash
npm install
npm run dev        # http://localhost:5173
npm test           # engine + parity tests
npm run build      # static site in dist/
```

## How it is verified

- **Model:** the TypeScript forward pass (`src/engine/model`) matches a PyTorch port (`python/rt_model.py`). That port matches the **official `rt/model.py` class**, loaded with the same weights, to ≤ 3e-6 (`python/parity.py`, `public/data/verification.json`).
- **Preprocessing and sampler:** the code mirroring `pre.rs` and `fly.rs` reproduces thousands of contexts, token for token, against an independent Python re-implementation of the Rust code.
- **Spec:** every behaviour depicted is documented, with citations, in `research/RT_TECHNICAL_SPEC.md`.

## Questions & answers

The site has a Q&A page (`/#/qa`). Each answer is one file in `src/qa/entries/`, written in the chapters' style and
checked against the paper-era code. In Claude Code, `/rtquestion <question>` (skill in `.claude/skills/rtquestion/`)
researches a new question against the official repository, adds the entry, verifies the build and pushes it.

## Layout

- `src/engine/`: pure-TS simulation engine (data generators, preprocessing, sampler, masks, model forward).
- `src/chapters/`: the 17 interactive chapters.
- `src/qa/`: the Questions & answers page and its entries.
- `src/components/`: shared UI and visualization components.
- `python/`: embeddings (all-MiniLM-L12-v2), training of the mini RT, parity checks. See `python/README.md`.
- `public/data/`: precomputed embeddings, trained weights, training curves.
- `research/`: technical spec and research reports.
- `docs/`: build plan and chapter briefs.
- `src/styles/fonts/libron/`: [Libron](https://github.com/nicoverbruggen/libron) (SIL OFL 1.1), the reading-mode font.
