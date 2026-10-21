# 05 — Web research sweep: Relational Transformer (RT), arXiv 2510.06377, ICLR 2026

Scope: web sources (author pages, slides, arXiv version history, OpenReview metadata, HF cards, follow-ups, third-party analyses, background docs) to find technical details that are *not obvious from the camera-ready paper*, plus discrepancies. Every item carries a citation. Local copies of downloaded material live in
`/tmp/claude-1004/-home-ibarra-Documents-Mestrado-Relational-Transformer/155ecce6-1f67-4870-9609-2c9740593feb/scratchpad/web/` (PDF->text via `pdftotext`; file names given below).

Conventions: `v1/v2/v3` = arXiv versions. "Paper" without qualifier = v3 = ICLR camera-ready = the user's local PDF (verified: `pdftotext` of `/home/ibarra/Documents/Mestrado/Relational_Transformer/Relational_Transformer.pdf` is byte-identical in text to arXiv v3, and the OpenReview camera-ready PDF differs from v3 only by the arXiv stamp line; see `web/local.raw.txt`, `web/rt_v3.raw.txt`, `web/rt_openreview.raw.txt`).
Code paths: `PE/` = paper-era snapshot `.../scratchpad/upstream/rt-paper-era`, `MAIN/` = current main `.../scratchpad/upstream/rt-official`.

---

## (a) Facts about RT found on the web (beyond / around the paper)

### A1. Publication history and what changed between arXiv versions

Source: arXiv abs pages (`web/arxiv_abs_v{1,2,3}.html`), https://arxiv.org/abs/2510.06377v1, ...v2, ...v3; PDFs `web/rt_v{1,2,3}.pdf`.

| Version | Date (UTC) | arXiv comment | Pages | Headline numbers |
|---|---|---|---|---|
| v1 | Tue 7 Oct 2025 18:51 | "preprint; under review" | 23 | abstract: **94%** of supervised AUROC; intro: 90.8% -> 94.4% with continued pretraining; Gemma3-27B "10^5x more inference FLOPS" (v1 p.2-3) |
| v2 | Wed 22 Oct 2025 22:13 | "preprint; under review" | 23 | **93%**; 90.3% -> 93.1%; "**10^7x** more inference FLOPS" (v2 p.3) |
| v3 | Mon 2 Mar 2026 07:22 | "Accepted to ICLR 2026" | 26 | 93%; 90.3% -> 93.1%; "**10^5x** more inference FLOPs" (v3 p.3) |

Concrete v1 -> v2 changes (sentence diff, `web/sent_v1.txt` vs `web/sent_v2.txt`):
- RT's own result columns were re-run: Table 1 zero-shot mean AUROC for RT went 70.1 -> 69.7 (unseen DB) and 72.9 -> 71.9 (continued pretraining); several per-task RT entries changed (e.g. rel-f1 driver-dnf 82.0 -> 81.2). Regression table RT values also changed (e.g. Table 12 driver-pos 54.7 -> 52.4). Baselines unchanged.
- Affiliation "SAP" -> "SAP Labs LLC"; footnote "Work done, in part, as an intern at SAP Labs LLC"; acknowledgement "We thank SAP Labs LLC".
- Timeline coincidence (NOT verified as causal): the code repo got commit `7f891d1` "fix in attention order and sampling" on 2025-10-08 (one day after v1), changing `RelationalBlock.forward` loop order from `["feat","nbr","col","full"]` to `["col","feat","nbr","full"]` (`git -C MAIN show 7f891d1 -- rt/model.py`), plus sampler fixes `66c4247` (2025-10-09 "temporal filter then do bfs subsampling") and `801a32b` (2025-10-16 "fixed bfs width sampling"). v2 (Oct 22) has new RT numbers. The ICLR-submission commit is `eeccc2b` (2025-10-06 "iclr submission").

Concrete v1/v2 -> v3 changes (camera-ready; compare `web/rt_v1.raw.txt` vs `web/rt_v3.raw.txt`):
- Abstract: contributions go from 3 to 4 items — "(i) incorporates task specification via **task table prompting**" is added; "harnesses task-table context" -> "leverages task context".
- Terminology: "Task table integration" -> "**Task table prompting**"; "Context sampling" paragraph -> "**Context window for zero-shot prompting**" (v3 §3.1 p.4).
- Design principle (i) now reads "effectively capture relational structure **while preserving the natural symmetries of relational data**" (v3 §3 p.4).
- **Attention equation corrected.** v1/v2 §3.2 (v1 p.5): `Attention(Q,K,V;M) = Softmax((QK^T ⊙ M)/sqrt(d_K)) V` (element-wise product — which would set masked logits to 0, not exclude them). v3 §3.2 (p.5): `SDPA(Q,K,V;M) = Softmax(Mask(QK^T;M)/sqrt(d_k)) V` with `Mask(A;M)_ij = A_ij if M_ij=1, −∞ if M_ij=0`. The project page still shows the old ⊙M form (https://star-project.stanford.edu/rt/). The code (FlexAttention `mask_mod`) implements the −∞ semantics (see B3).
- Positional-encoding paragraph rewritten as "**Permutation invariance for tables, rows and columns**" (v3 p.5) and the discussion formerly in v1 App. C.1 (p.16) moved to new **App. E "Symmetry and Expressive Power"** (v3 p.19-20). Content (both versions): removing column attention does not reduce expressivity (global heads can emulate it via table/column-name embeddings); feature attention is the *only* mechanism that explicitly groups cells into rows; neighbor attention can be simulated by feature + global attention; without full attention information propagates one hop per layer.
- New in v3: §2.3 "Zero-shot relational learning"; §4.4 "Relational feature extraction" (Table 3: MLP on the masked-token representation *before the decoder layer*); §4.5 "Cost-quality trade-off" (Fig. 4; RT evaluated at context 256/512/768/1024 after pretraining at 1024); **Fig. 6 = real context window** (rel-hm/item-sales, L=512, w=128; task 39 rows x 3 cols, transactions 21x5, article 14x25, customer 7x7; p.15); App. B expanded related work; App. C "Suitability of RelBench" (Jaccard/fuzzy column-name similarity tables); **App. H.1 / Table 15 attention-order ablation** (p.24); **App. I.1 EntityMean definition** with fallback to global train mean (p.23).
- EntityMean wording: v1/v2 "mean of past target-entity labels **in the context window**"; v3 "mean of the past labels for the target entity", fallback "global mean of the target column on the training set" (v3 §4.1 p.7, App. I.1 p.23).
- Regression table: v3 Table 2 has the v2 values (RT unseen-DB mean R² 22.7; continued-pretrain 26.4).

Table 15 (v3 p.24, zero-shot; C/F/N = column/feature/neighbor; "Parallel" = the three applied in parallel and summed; full attention not part of the permutation) — useful to state honestly that order is not critical:

| order | CFN | CNF | FCN | FNC | NCF | NFC | Parallel |
|---|---|---|---|---|---|---|---|
| mean AUROC | 70.0 | 70.6 | 70.0 | 69.8 | 70.2 | 69.5 | 69.3 |
| mean R² | 23.0 | 22.5 | 23.2 | 22.5 | 22.1 | 23.0 | 23.6 |

### A2. Venues, talks, identifiers

- ICLR 2026 poster. OpenReview forum `rpPtgMC5s9`, **Submission 9833**, "Published: 26 Jan 2026", primary area "foundation or frontier models, including LLMs", keywords "foundation models, relational deep learning, relational data, transformer", TL;DR "A novel architecture for relational data that shows strong zero-shot abilities on unseen datasets after pre-training." Invitations include `Rebuttal_Revision` and `Camera_Ready_Revision` (so there *was* a rebuttal revision). Source: Wayback snapshot http://web.archive.org/web/20260414194902/https://openreview.net/forum?id=rpPtgMC5s9 (`web/or_rpPt_wayback.html`); https://aiconfpaper.com/paper/iclr-2026-rpPtgMC5s9.
- Poster session: Sat 25 Apr 2026, 10:30–13:00 (-03), poster #823 Pavilion 3 (https://iclr.cc/virtual/2026/poster/10007116; author post on X mirrored at https://nitter.cf/_rishabhranjan_).
- Also at "Greeks in AI 2026" (OpenReview `eyvi5OwhyC`, Submission 196, published 2 Jun 2026; https://openreview.net/forum?id=eyvi5OwhyC).
- EurIPS 2025 "AI for Tabular Data" workshop: **Oral** per author's post (5 Dec 2025, nitter mirror) and ICLR slide 1 ("Oral @ AI for Tabular Data (AI4TD) Workshop, NeurIPS 2025"); Mark Žnidar's site calls it "Spotlight" (https://mark-znidar.github.io/). Minor inconsistency.
- The OpenReview-era abstract (ICLR submission, shown on papers.cool and aiconfpaper) is the v1 abstract ("94%", 3 contributions): https://papers.cool/venue/rpPtgMC5s9@OpenReview.
- Talks: Stanford Graph Learning Workshop, 14 Oct 2025, "Relbench V2 and Zero-shot predictions with Relational Transformer" (R. Ranjan, video linked) https://snap.stanford.edu/graphlearning-workshop-2025/ ; Temporal Graph Learning reading group, 19 Feb 2026 — slides published by ICLR as https://iclr.cc/media/iclr-2026/Slides/10007116.pdf (PDF title "2026-02-19__tgl", 59 pages; `web/slides_iclr.pdf`, `web/slides.raw.txt`); UCSC CSE colloquium 28 Jan 2026 by C. Kanatsoulis (https://events.ucsc.edu/event/cse-colloquium-towards-relational-foundation-models-zero-shot-forecasting-over-relational-databases/); YC ML Meetup at Startup School (M. Žnidar; listed on https://star-project.stanford.edu/).
- Project page: https://star-project.stanford.edu/rt/ (has an interactive "Relational Attention explorer" widget "reproducing the four masks").
- Code: originally `github.com/snap-stanford/relational-transformer`, now redirects to `github.com/stanford-star/relational-transformer`. README: the ICLR paper's implementation is the **`rt-v1` branch/tag**; `main` is PluRel/RT-J (https://github.com/stanford-star/relational-transformer). In the local clone, tag `rt-v1` = `8d83590` (2026-06-27), i.e. paper-era code plus leaderboard tooling commits after `2a899c8`.
- Paper checkpoints on HF: `stanford-star/rt-v1` — "one .pt file per (database, task). `pretrain_<db>_<task>.pt` is pretrained with `<db>` held out, `contd-pretrain_*` continues pretraining on `<db>` with the task held out, `finetune-from-*` is fine-tuned on the task. 12 blocks, d_model 256, 8 heads, d_ff 1024; text embedded with all-MiniLM-L12-v2." Preprocessed data: HF dataset `stanford-star/relbench-preprocessed` (`legacy/` subdir for v1). Source: https://huggingface.co/stanford-star/rt-v1/raw/main/README.md.

### A3. Details from the authors' slides (ICLR media / TGL talk, 19 Feb 2026) not stated in the paper

Source: https://iclr.cc/media/iclr-2026/Slides/10007116.pdf (`web/slides.raw.txt`; text has ligature drop-outs, e.g. "Rel tion l" = "Relational").
- Architecture figure stack (bottom to top): **Column Attention -> Feature Attention -> Neighbor Attention -> Full Attention -> MLP**, "Relational Transformer Block (repeated N times)", then "Per-datatype decoder num|text|bool|time". Input side: "∀ cells: Per-datatype encoder num|text|bool|time **+** Frozen LM embedding for '<column> of <table>'". Context is drawn as a "**Set of cell values without positional information**". Task slide: "Task Table Prompting: Task is specified by adding new table to the DB (Model treats task table as any other table)"; "Task rows provide in-context labels"; "Target row with masked cell".
- Efficiency slide: "Masked FlashAttention kernels using FlexAttention — Block sparsity in attention masks — **With column-major serialization of tokens**". NOTE: I did not find any column-major reordering in the paper-era sampler: `PE/rustler/src/fly.rs:455-470` writes all cells of each visited row consecutively (row-by-row in BFS visit order). Treat "column-major serialization" as a slide illustration of block sparsity (or a later optimization), not as paper-era behaviour (unverified either way beyond this grep).
- Pretraining slide: "Leave-one-DB-out on RelBench; Masked token prediction objective; Combined training on classification and regression: classification loss BinaryCrossEntropyLoss, regression loss HuberLoss; **Each run sees ~20 tasks from 6 DBs** including forecasting tasks ('predict future derived quantity') and autocomplete tasks ('predict future column value'); **22M parameters, 13B tokens, 2 hrs on 8xA100, 1024 ctx_len**". (13B ≈ 50k steps x 256 x 1024 = 13.1B; consistent with paper §4.1.)
- Zero-shot slide: vs Gemma3-27B, RT has "1,000x fewer parameters, 10-100x lower context requirements, **100,000x fewer inference FLOPs**, but 13% better AUROC" — agrees with v1/v3 (10^5), contradicts v2 and the project page (10^7).
- "LLMs are unable to do regression (worse than the GlobalMean baseline)"; LLM serialization example: JSON nested "add_transactions" lists, "context ~10k tokens" — from Wydmuch et al. "Tackling prediction tasks in relational databases with LLMs".
- Ablation slides: "Column attention helps to adapt to new column distributions; Feature attention helps to adapt to new entity types."

### A4. Project page (star-project.stanford.edu/rt) — claims and how they differ from the paper

Source: https://star-project.stanford.edu/rt/
- Affiliations differ from the PDF: page lists "Valter Hudovernik — Kumo AI", "Mark Znidar — University of Oxford" (paper v3 footnote: "Work done while at Stanford University").
- Says context "n cells (**1024 in the released configuration**)".
- Value encoding table: numeric/boolean per-column z-score; datetime seconds, global z-score; text frozen encoder. Token embedding `x = W_d r + W E_schema(c,t)`, masked value term replaced by learned mask vector. (Same as paper; note the code additionally applies RMSNorm after each Linear — see established facts.)
- Output section: "boolean datatype — Binary classification ... BCE"; "**numeric & datetime** datatype — Regression ... Huber loss" (paper §3.3 only mentions numeric; code also applies Huber to datetime).
- Still shows the v1 `⊙ M` attention formula and the v2 "10^7x" FLOPs claim (both superseded in v3).
- Supervised FT summary (Table 7 in their numbering): RT pretrained 77.2 AUROC (vs RelGNN/RelLLM 77.1, RDL-GNN 75.4); 33.2 R² (vs Griffin 19.5, RDL-GNN 14.3).
- Attention-layer ablation (zero-shot R²): none 22.8, −col 11.9, −feat 18.8, −nbr 22.0, −full 23.7.

### A5. Discrepancies / subtleties that matter for a faithful simulation

1. **Alg. 2 (paper p.26) vs code vs LLaMA.** Alg. 2 writes `X ← X + Norm(MHA(X; M))` (norm applied to the sub-layer *output*). The paper also says "Other architectural details (normalization, activations, etc.) follow the design choices of LLaMA" (§3.2 p.5). LLaMA's "Pre-normalization [GPT3]... we normalize the **input** of each transformer sub-layer, instead of normalizing the output. We use the RMSNorm" (Touvron et al. 2023 §2.2, https://arxiv.org/abs/2302.13971; `web/llama.txt:262-265`). The code is pre-norm: `x = x + self.attns[l](self.norms[l](x), ...)` (`PE/rt/model.py:84`). So Alg. 2 is inconsistent with both its own LLaMA claim and the code; the website should show pre-norm and can footnote Alg. 2. This is unchanged across v1–v3 (v1 p.23, v3 p.26).
2. **FFN width.** LLaMA uses SwiGLU hidden size `2/3·4d` (LLaMA §2.2). RT uses d_ff = 1024 = 4·256 (paper §4.1 "with hidden dimension 1024"; `PE/scripts/example_pretrain.py:35`). So "LLaMA-style" refers to SwiGLU form, not its 2/3 sizing.
3. **"MiniLMv2".** Paper: "we use the MiniLMv2 [41] model from SentenceTransformers" with [41] = Wang et al., "MiniLMv2: Multi-head self-attention relation distillation", ACL Findings 2021 (v3 p.6, ref list p.13). Code uses `sentence-transformers/all-MiniLM-L12-v2` (`PE/rt/embed.py:15,38`). That model's base is `microsoft/MiniLM-L12-H384-uncased` (config `_name_or_path`, model card "We used the pretrained microsoft/MiniLM-L12-H384-uncased"), whose own card says "MiniLMv1-L12-H384-uncased" and cites the **MiniLM (v1)** paper arXiv 2002.10957 (https://huggingface.co/microsoft/MiniLM-L12-H384-uncased). The "-v2" in `all-MiniLM-L12-v2` is the sentence-transformers release version, not MiniLMv2. Minor citation inaccuracy in the paper; the model actually used is unambiguous.
4. **BFS width w.** Paper never states w for the main runs; Fig. 6 uses w=128 (p.15); HF `rt-plurel` card lists "Max BFS width 128"; paper-era example scripts use `max_bfs_width=256` (`PE/scripts/example_pretrain.py:21`, `example_finetune.py:22`, `example_contd_pretrain.py:23`). Unverified which value produced the paper tables.
5. **Batch size.** Paper: batch 256 (§4.1). Script: `batch_size=32` (`PE/scripts/example_pretrain.py:19`) — consistent with 8 GPUs x 32 = 256 if per-device (inference, not verified). Script also has `max_grad_norm=1.0` (line 26), not mentioned in the paper.
6. **Fully-masked attention rows output zero.** In the code, a query can have *no* allowed key (e.g. neighbor attention for a cell whose row has no child rows in the context). FlexAttention's reference semantics apply `torch._safe_softmax` and set the logsumexp of fully masked rows to −inf, so the output row is **0** (not NaN) (`web/flex_hop_v28.py:249-256`, PyTorch v2.8 `torch/_higher_order_ops/flex_attention.py`; default kernel option `ROWS_GUARANTEED_SAFE=False`, `web/flex_v28.py:1107`). RT-J's paper states this explicitly: "Queries with no accessible keys receive a zero attention contribution" (RT-J App. F, https://arxiv.org/abs/2610.17867). With bias-free `wo` (`PE/rt/model.py:28`), such a sub-layer adds exactly 0 to the residual stream. The website's neighbor-attention step should show this.
7. **Attention scale.** Paper-era code calls `flex_attention(q, k, v, block_mask=block_mask)` with no `scale` (`PE/rt/model.py:43`) -> default `1/sqrt(E)` with E = head_dim = 256/8 = 32 (PyTorch docs: "If none, the default value is set to 1/sqrt(E)", https://docs.pytorch.org/docs/2.8/nn.attention.flex_attention.html). A `block_mask is None` branch using `F.scaled_dot_product_attention` under the FLASH backend exists (`PE/rt/model.py:39-41`), but in the normal forward the "full" layer receives a real BlockMask built from the padding mask (`attn_masks["full"] = pad`, `PE/rt/model.py:189-194`), so all four layers go through FlexAttention; padding *query* rows are all-False and therefore output 0 (A5.6).
8. **Block masks.** Built once per batch from a dense boolean `[B, S, S]` tensor via `create_block_mask(mask_mod=lambda b,h,q,kv: mask[b,q,kv], B=batch, H=None, Q_LEN=S, KV_LEN=S, _compile=True)` (`PE/rt/model.py:89-101`), default `BLOCK_SIZE=128` (PyTorch docs). Heads broadcast (H=None) => all 8 heads share each mask.
9. **Parameter count (derived, not from the web).** From the code shapes: per block 4 attentions x 4·256² = 1,048,576 + SwiGLU 3·256·1024 = 786,432 + 5 RMSNorm weights x 256 ≈ 1.836M; x12 ≈ 22.04M; plus encoders (2 x Linear(384,256)+b ≈ 197k; 3 x Linear(1,256)+b) and decoders (Linear(256,384)+b ≈ 98.7k, 3 x Linear(256,1)+b) ≈ 22.3M total, matching "about 22M trainable parameters" (paper §4.1).
10. **Throughput consistency.** Paper "8 batches/second or 2M tokens/second" ≈ 8 x 256 x 1024 = 2.1M tokens/s (paper §4.1 p.6).

### A6. Third-party analyses of RT (useful for the "limitations" part of the site)

- **OpenRFM: Dissecting Relational In-Context Learning** (Chen et al., arXiv 2606.04320, 3 Jun 2026; https://arxiv.org/abs/2606.04320, https://arxiv.org/html/2606.04320). Claims: RT performs "relation-level ICL"; its signal "arrives only when its BFS walk reaches another row of the task table, whose label becomes part of the context"; define label reachability ρ(T) = fraction of seeds whose walk reaches another task-table row; RT "excels on high-ρ tasks and collapses on low-ρ ones". Kernel-regression reading: ŷ⋆ ≈ Σ_{i∈S} α(x⋆, x_i) y_i (under stated linearisation assumptions). PluRel-synthetic-only pretraining shows a "lazy, frozen-feature regime" (label shuffling / feature randomisation barely matter), in-distribution pretraining a "feature-learning" regime. Failure examples include rel-trial study-outcome and study-adverse. This is their analysis, not the RT authors'.
- **Tony Kwok, "Testing an Open Relational Foundation Model on My Laptop"** (Aug 2026, https://tonykwokch.com/writing/testing-an-open-relational-foundation-model): using the RT-PluRel checkpoint, rel-f1 driver-top3 continued-pretrained RT 0.844 AUROC vs XGBoost 0.855, RelGT 0.804; rel-trial study-outcome zero-shot 0.515, continued 0.590 vs XGBoost 0.718; at the val-selected threshold zero-shot RT predicted positive for every test row. Fixed-seed, non-apples-to-apples (author's caveat). Note this is RT-PluRel, not the paper checkpoints.
- **alphaXiv overview** (https://www.alphaxiv.org/overview/2510.06377): stresses that "zero-shot" does not mean "no labels in the DB" — past task-table labels of the same entity can be in the context; 69.7% mean zero-shot AUROC = 90.3% of supervised RT mean 77.2%; Gemma3-27B 64.6%; Gemma pretraining exposure "Maybe".
- The paper itself (v3 Table 12 p.23) reports the count of in-context label cells (self vs other entities) in 1024-cell contexts; the self-label ablation collapses zero-shot AUROC 70.1 -> 53.8 and R² 22.8 -> −5.5 (Table 4 p.9).

### A7. Do NOT confuse RT (paper) with RT-PluRel / RT-J (later work, same repo)

Sources: RT-J paper arXiv 2610.17867 (13 Oct 2026, "40th NeurIPS", `web/rtj.txt`), https://star-project.stanford.edu/rt-j/, PluRel arXiv 2602.04029 (ICML 2026, `web/plurel.txt`), HF cards, and code `MAIN/src/rt/model/net.py`, `MAIN/src/rt/model/legacy/{v1,plurel}.py`.

| | RT (ICLR 2026 paper) | RT-PluRel (PluRel, ICML 2026) | RT-J (NeurIPS 2026) |
|---|---|---|---|
| size | 12 blocks, d=256, 8 heads (dh=32), d_ff=1024, ~22M | HF card: 12 / 256 / 8 / 1024, ctx 1024, BFS width 128 (https://huggingface.co/stanford-star/rt-plurel); RT-J paper says its RT-PluRel is 86M (so the RT-J-recipe RT-PluRel is the large one) | 12 blocks, d=512, 8 heads (dh=64), d_ff=2048; 85,562,530 params (RT-J App. F); site says 85M, abstract 86M |
| attentions per block | col, feat, nbr, **full** (`PE/rt/model.py:84`) | code: **col, feat, nbr only** (`legacy/plurel.py`, `attn_types = ["col","feat","nbr"]`); PluRel paper text still describes "three structured attention layers followed by a bidirectional attention layer" (`web/plurel.txt`) — paper/code mismatch | col, feat, nbr only (`MAIN/src/rt/model/net.py:111`; RT-J App. F "applies column, feature, and neighbor attention sequentially") |
| QK-norm | no | yes, RMSNorm per head on Q and K ("To reduce such sensitivity [to initialization], we applied QK-Norm", PluRel §3) | yes (`net.py:41-42,63-64`) |
| extra attention tricks | none | none | queries scaled by learned per-head α_h · log(max(#valid keys,1)), softmax scale 1/dh (`net.py:66-80`; RT-J eq. 13); output gate g = 2σ(W_G z) (`net.py:84-85`; eq. 16) |
| init | default PyTorch | default | zero-init of W_O, W_G, FFN W2 (`net.py:48,52,97`) |
| context | 1024 cells, BFS (Alg. 1) | 1024 | L ∈ {512..8192}, local ctx ℓ ∈ {128..8192}, BFS width b ∈ {8..256}, recency ρ ∈ {0,1} sampled during pretraining; random-walk retriever W=10,000 walks of length K=20; only the target cell masked (RT-J §5, App. G) |
| data | RelBench v1, leave-one-DB-out | PluRel synthetic DBs (+ optional real continued pretraining) | PluRel phase then "the Join": 639 web-crawled DBs, 6,023 LLM-generated forecasting tasks, ~11K autocomplete tasks; "11x more total tokens, up to 8x larger contexts, 4x larger model, ~70x more pretraining FLOPs" than RT (RT-J §1) |

Third-party explainer warning: RelativeDB's "How the Relational Transformer predicts over a database" (1 Aug 2026, https://relativedb.com/research/relational-transformer) describes the **RT-J** model (d_model 512, QK-norm, zero-init residual branches, "three attentions in sequence ... feat, nbr, col", sampler up to 8192 cells with random walks) under the name "Relational Transformer". Its order "feat, nbr, col" also contradicts the RT-J code (`col, feat, nbr`). Do not use it as a source for the ICLR model.

### A8. Small errors in secondary sources (so we don't copy them)
- STAR publications page lists RT as "arXiv:2505.10960" (https://star-project.stanford.edu/publications/); 2505.10960 is actually Relational Graph Transformer (RelGT) — paper ref [11].
- emergentmind topic pages mix v1 numbers (94%, 90.8%, 94.4%) and conflate RT with the unrelated graph "Relational Transformer" of Diao & Loynd 2022 (https://www.emergentmind.com/topics/relational-attention-transformers).
- HF `rt-plurel` card describes the architecture as "treating rows across linked tables as a sequence via BFS-ordered context sampling" with no mention of QK-norm or the dropped full attention.

---

## (b) Background facts needed to explain RT

### B1. RelBench (Robinson et al., NeurIPS 2024 D&B; arXiv 2407.20060; `web/relbench.txt`; https://relbench.stanford.edu/)
- A predictive task is defined by a **training table** "with columns for Entity ID, seed time, and target labels. The seed time indicates at which time the target is to be predicted, filtering future data." (RelBench §2). RT's "task table" is this training table treated as an ordinary table (RT §2.2 "Task tables"; RT §3.1).
- **Temporal splits:** "models trained on rows up to VAL_TIMESTAMP, validated on the rows between VAL_TIMESTAMP and TEST_TIMESTAMP, and tested on the rows after TEST_TIMESTAMP. Our implementation carefully hides data after TEST_TIMESTAMP during inference" (RelBench §2). Temporal neighbor sampling: "all nodes within the sampled subgraph appear before the seed time" (RelBench §4).
- v1: 7 DBs, 30 tasks (51 tables, 103,466,370 rows, 489 cols). Tasks per DB: amazon 7, avito 4, event 3, f1 3, hm 3, stack 5, trial 5. Tables per DB: amazon 3, avito 8, event 5, **f1 9**, hm 3, stack 7, trial 15 (RelBench Table 1).
- rel-f1: 74,063 rows, 67 cols, start 1950-05-13, **val 2005-01-01, test 2010-01-01**; tasks: "driver-position: Predict the average finishing position of each driver all races in the next 2 months"; "driver-dnf: ... if they will DNF (did not finish) a race in the next 1 month"; "driver-top3: ... if they will qualify in the top-3 for a race in the next 1 month" (RelBench App. A). rel-hm: user-churn "no transactions in the next week", item-sales "sum of prices of the associated transactions in the next week".
- Metrics: entity classification AUROC, regression MAE (RT reports R² instead), recommendation MAP (relbench.stanford.edu).
- RelBench v2 (Jan 2026, arXiv 2602.12606): 4 new DBs (rel-salt, rel-ratebeer, rel-arxiv, rel-mimic), 40 new tasks incl. 28 **autocomplete** tasks "predict existing columns", plus CTU (71), 4DBInfer (7), TGB (12) integrations (relbench.stanford.edu; RelBench v2 announcement mirrored at https://nitter.cf/_rishabhranjan_).

### B2. Relational Deep Learning (Fey et al., ICML 2024 position paper, arXiv 2312.04615; `web/rdl.txt`)
- **Relational Entity Graph**: "a node for each entity in each table, and edges are defined via foreign-primary key connections between entities"; **schema graph**: table-level graph with edges L ∪ L^{-1} (inverse links added); **time-consistent computation graph** = the sampled training example (RDL §3.1-3.3).
- A predictive task "is specified and added to the database by introducing an additional training table" (RDL Fig. 1). RT inherits this idea, but tokenizes cells instead of rows and replaces GNN message passing with masked attention (neighbor attention "acts analogously to message-passing", RT §3.2).

### B3. FlexAttention (PyTorch; RT ref [8] Dong et al. arXiv 2412.05496)
- `mask_mod(b, h, q_idx, kv_idx) -> bool` ("returns True if this position should participate"); `create_block_mask(mask_mod, B, H, Q_LEN, KV_LEN, ...)` builds a `BlockMask`; FlexAttention "can then use BlockMask to take advantage of the sparsity" by skipping fully-masked blocks; default `BLOCK_SIZE=128`; partially-masked blocks apply the mask element-wise, fully-unmasked blocks skip it; masks that change every batch should be built once at the start of the forward and reused across layers (https://pytorch.org/blog/flexattention/; https://docs.pytorch.org/docs/2.8/nn.attention.flex_attention.html). RT builds 4 BlockMasks per batch and reuses them in all 12 blocks (`PE/rt/model.py:198-205,227`).
- Masked entries behave as score = −∞ (blog: `torch.where(mask, score, -inf)`); fully-masked rows -> output 0 via `_safe_softmax` (A5.6).
- Default softmax scale 1/sqrt(head_dim) (docs).

### B4. LLaMA design choices (Touvron et al. 2023, arXiv 2302.13971 §2.2; `web/llama.txt:255-275`)
- Pre-normalization with RMSNorm on each sub-layer input; SwiGLU FFN with 2/3·4d hidden; RoPE. RT keeps pre-norm RMSNorm + SwiGLU, uses 4d hidden, and has **no positional encoding at all** (RT §3.1 "RT does not incorporate positional encodings").
- RMSNorm: `y_i = x_i / RMS(x) * γ_i`, `RMS(x) = sqrt(ε + mean(x²))`; `torch.nn.RMSNorm(normalized_shape, eps=None, elementwise_affine=True)`, default ε = `torch.finfo(x.dtype).eps`, γ initialised to ones (https://docs.pytorch.org/docs/2.8/generated/torch.nn.RMSNorm.html). RT uses `nn.RMSNorm(d_model)` with defaults (`PE/rt/model.py:72,134-138,150`).
- SwiGLU (Shazeer 2020, arXiv 2002.05202 eq. 6; `web/glu.txt:56-70`): `FFN_SwiGLU(x, W, V, W2) = (Swish_1(xW) ⊗ xV) W2`, biases omitted; Swish_1 = SiLU = x·σ(x). RT: `w2(silu(w1 x) * w3 x)`.

### B5. all-MiniLM-L12-v2 (text and schema encoder)
Sources: https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2 (`modules.json`, `sentence_bert_config.json`, `config.json`, `1_Pooling/config.json`, README; `web/minilm/`).
- Pipeline modules: `Transformer` -> `Pooling` (**mean pooling**, `pooling_mode_mean_tokens: true`, dim 384) -> `Normalize` (L2 to unit length). So `model.encode(...)` returns **unit-norm 384-d vectors** even with `normalize_embeddings` left at default (the Normalize module is part of the model). RT-paper-era code calls `self.model.encode(text_list, batch_size=..., convert_to_numpy=True, ...)` with no extra normalization flag and stores bf16 (`PE/rt/embed.py:14-31,71`).
- BERT architecture: 12 layers, hidden 384, 12 heads, intermediate 1536, GELU, vocab 30,522, max_position_embeddings 512; tokenizer uncased (`do_lower_case: true` in tokenizer_config); **max_seq_length 128** tokens (longer text truncated); ~33.4M params (HF API `safetensors.total = 33,360,512`).
- Fine-tuned contrastively from `microsoft/MiniLM-L12-H384-uncased` on ~1.17B sentence pairs, 100k steps, batch 1024, lr 2e-5, seq len 128 (model card).
- Mean pooling formula (card): `sum(token_emb * mask) / clamp(sum(mask), min=1e-9)`.

### B6. Losses (PyTorch)
- Huber: `l = 0.5 (x−y)²` if `|x−y| < δ`, else `δ(|x−y| − 0.5δ)`; **default δ = 1.0** (equivalent to SmoothL1 at δ=1) (https://docs.pytorch.org/docs/2.8/generated/torch.nn.HuberLoss.html). RT code: `F.huber_loss(yhat, y, reduction="none")` (`PE/rt/model.py:251`) -> δ = 1 on the z-scored target.
- BCE-with-logits on target 1{r>0} (`PE/rt/model.py:253`); for a boolean column with 0 < μ < 1, the z-score r > 0 iff v = 1, so the label is the original boolean.

### B7. Neighbouring models mentioned by RT
- Griffin (Wang et al., ICML 2025, arXiv 2505.05568): graph-centric RDB FM; cross-attention module + enhanced MPNN; pretrained text encoder for categorical, pretrained float encoder/decoder for numerical; single-table masked-cell pretraining then joint SFT (https://arxiv.org/html/2505.05568v2). RT §6: "Griffin aggregates at the row level before GNN-based propagation."
- KumoRFM: "cannot operate in zero-shot settings, and its technical details remain undisclosed" (RT §6).

---

## (c) Could not verify / not found

1. **OpenReview reviews, author rebuttals, meta-review, decision text** for ICLR submission 9833 (`rpPtgMC5s9`): every OpenReview endpoint (forum page, api/api2 `/notes?forum=`, PDF) returned "Challenge verification required" (`web/or_eyvi.json`), Exa/WebFetch got only the challenge page, and the Wayback snapshots (2026-03-26, 2026-04-14) contain only the server-rendered main note; replies are loaded client-side and no archived API response exists. Only metadata (submission number, `Rebuttal_Revision` invitation, publication date, keywords, TL;DR) is confirmed. Any architecture/sampling/normalization clarifications made in the rebuttal remain unknown.
2. The full X/Twitter thread (only tweet "1/n" of 30 Oct 2025 is mirrored on nitter/vanlett); the Google-Drive copy of the TGL slides (assumed identical to the ICLR-hosted "2026-02-19__tgl" PDF, not checked); talk videos (Stanford Graph Learning Workshop video exists, not watched).
3. Whether the v1 -> v2 metric changes were caused by commits `7f891d1`/`66c4247`/`801a32b` (only the timeline is established).
4. Which `max_bfs_width` (128 vs 256) and per-device batch produced the published tables.
5. "Column-major serialization of tokens" (slides) — not present in paper-era `fly.rs` as far as inspected.
6. The exact value of the Gemma-vs-RT FLOPs ratio (10^5 in v1/v3/slides, 10^7 in v2/project page); no derivation is given anywhere I found.
7. No HuggingFace model card for the paper checkpoints beyond the short `stanford-star/rt-v1` README; no Papers-with-Code entry found (HF papers page exists: https://huggingface.co/papers/2510.06377). alphaXiv discussion comments require login (connector not authorized).
