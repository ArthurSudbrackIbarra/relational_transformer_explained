# 04 — Paper-era vs current main (RT-J): keeping the website faithful to the ICLR 2026 model

Path abbreviations used in every citation:

- `PE/` = paper-era snapshot (commit `2a899c8`, 2026-02-04): `/tmp/claude-1004/-home-ibarra-Documents-Mestrado-Relational-Transformer/155ecce6-1f67-4870-9609-2c9740593feb/scratchpad/upstream/rt-paper-era/`
- `CM/` = current main (HEAD `def35f9`, 2026-10-18): `/tmp/claude-1004/-home-ibarra-Documents-Mestrado-Relational-Transformer/155ecce6-1f67-4870-9609-2c9740593feb/scratchpad/upstream/rt-official/`
- `paper p.N` = page N of `/home/ibarra/Documents/Mestrado/Relational_Transformer/Relational_Transformer.pdf` (pages located with `pdftotext`)

---

## 0. TL;DR for the website builders

1. **`CM/src/rt/model/legacy/v1.py` (`V1Transformer`) is the same network as `PE/rt/model.py` (`RelationalTransformer`).** Every layer, init, mask, encoder/decoder, the loss and the block order match line for line. Only two things differ, and both are about the batch interface, not the math: v1 reads `batch["is_targets"]` where PE reads `batch["masks"]`, and it sets `boolean_values := number_values` (`CM/src/rt/model/legacy/v1.py:134,140`). In PE the sampler writes `masks` equal to `is_targets` (`PE/rustler/src/fly.rs:492-496`), so the first difference does nothing. The checkpoint dims are `num_blocks=12, d_model=256, d_text=384, num_heads=8, d_ff=1024` (`CM/src/rt/model/legacy/_common.py:7-13`), the same as the PE pretraining script (`PE/scripts/example_pretrain.py:31-35`).
2. **`CM/src/rt/model/net.py` is RT-J, a different architecture.** Our site must not show any of these RT-J features:
   - QK-RMSNorm
   - a learned per-head scale times `log(kv_size)`
   - attention scale `1/head_dim`
   - a sigmoid output gate
   - zero-init of `wo`, `w2` and the decoders
   - no full-attention layer
   - no boolean type
   - a cell re-sort by column
   - per-sequence loss averaging
   - selectable huber/l1/bce loss
   - d_model 512
3. **The current sampler (`CM/rustler/src/fly.rs`) and preprocessor (`CM/rustler/src/pre.rs`) differ heavily from PE.** Even the "legacy RT-v1 eval" in CM runs through the new sampler, so it does not reproduce the PE context algorithm. For sampling and preprocessing, use **only the PE Rust code and paper Alg. 1**.
4. **The ICLR model's own code and data path has known "pre-fix" behaviours.** If we want to be 100% faithful to the paper model, we should depict them (optionally with a footnote):
   - z-score statistics are computed over whole DB tables, and global datetime stats over all tables including val/test (the paper text says "training split")
   - same-timestamp task rows of other entities are visible with their labels
   - task-table P→F children are never width-subsampled
   - the context is cut mid-row at `seq_len`

---

## 1. Architecture: legacy v1 vs paper-era vs RT-J (task item 1)

### 1.1 `legacy/v1.py` == paper-era `rt/model.py` (confirmed)

I diffed the two files (after renaming `l`→`lvl`). The only non-cosmetic hunks:

| Aspect | Paper-era `PE/rt/model.py` | Legacy `CM/src/rt/model/legacy/v1.py` | Semantic effect |
|---|---|---|---|
| Training-mask key | `batch["masks"]` (`:219,223,236`) | `batch["is_targets"]` (`:140,183,187,199`) | None. PE sampler sets `masks[i] = is_target` (`PE/rustler/src/fly.rs:492-496`) |
| Boolean values | `batch["boolean_values"]` from sampler (`PE/rt/data.py:146-148`) | `batch = {**batch, "boolean_values": batch["number_values"]}` (`:134`) | Adapter for the CM sampler, which puts boolean values into `number_values` (`CM/rustler/src/fly.rs:1529-1532`) |
| Block-mask helper | local `_make_block_mask` (`PE:89-101`) | imported `make_block_mask` (`CM/src/rt/model/legacy/_common.py:17-29`) | Identical body |
| Loading | none | `from_pretrained` from HF `stanford-star/rt-v1` (`v1.py:20,129-131`; `_common.py:46-55`) | — |
| `predict` | in `PE/rt/main.py:283-289` | `_common.py:32-43` | Both take `yhat_dict["boolean"]` for clf and `["number"]` for reg at the target position |

Everything else is identical:

- **MHA.** `wq/wk/wv/wo` are `Linear(d,d,bias=False)`. The input is split into heads, then `flex_attention(q,k,v,block_mask)` runs with its default scale `1/sqrt(head_dim)`. Unmasked SDPA is the fallback when `block_mask is None` (PE `:20-47`, v1 `:23-50`).
- **FFN.** SwiGLU, `w2(silu(w1 x) * w3 x)`, with no biases (PE `:50-59`, v1 `:53-62`).
- **Block.** It owns 5 `nn.RMSNorm(d_model)` (default eps) and 4 attentions. The forward is pre-norm residual: `for l in ["col","feat","nbr","full"]: x = x + attn_l(norm_l(x))`, then `x = x + ffn(norm_ffn(x))` (PE `:62-86`, v1 `:65-84`).
- **Masks** (PE `:165-191`, v1 `:145-162`):
  - `feat = (same_node | kv_in_q.f2p) & pad`
  - `nbr = (q in kv.f2p) & pad`
  - `col = (same col_name_idx & same table_name_idx) & pad`
  - `full = pad`
- **Input embedding** (PE `:208-224`, v1 `:172-188`): `x = RMSNorm_colname(Linear_colname(colname_emb))·[not pad]`, plus for each type t: `RMSNorm_t(Linear_t(value_t))·[type==t & not masked & not pad]` + `mask_emb_t·[type==t & masked & not pad]`. The mask embeddings are `nn.Parameter(torch.randn(d_model))` for number, text, datetime and boolean.
- **Output.** `norm_out` RMSNorm feeds 4 linear decoders. All decoders have `bias=True`: `Linear(d,1)` for number, datetime and boolean, and `Linear(d,384)` for text (PE `:124-131`).
- **Loss** (PE `:238-265`, v1 `:201-226`):
  - Huber (default delta = 1) for number and datetime.
  - `BCE_with_logits(yhat, (y>0))` for boolean.
  - Masking a text cell raises `ValueError`.
  - Total = Σ over masked cells / `masks.sum()`, summed over the whole batch (one global mean, not per-sequence).
  - A type absent from the batch contributes `yhat.sum()*0`.
- **Encoder init** is the PyTorch default for `nn.Linear` (no custom init anywhere in PE).

Note on Alg. 2 (paper p.26): it writes `X ← X + Norm(MHA(X; M))`, i.e. norm after the sublayer. The code is pre-norm, `X + MHA(Norm(X))`. The website should show the **code** form. The paper itself says other details "follow the design choices of LLaMA" (paper p.5), and LLaMA is pre-norm.

Note on the column mask: the paper defines `M_column[q,k] = 1{Col(k)=Col(q)}` (paper p.5). The code also requires the same `table_name_idx`. This is redundant in practice, because `col_name_idx` already indexes the string `"<col> of <table>"` (`PE/rustler/src/pre.rs:432-437`).

Attention-order history: the ICLR submission commit `eeccc2b` used order `["feat","nbr","col","full"]`. Commit `7f891d1` ("fix in attention order and sampling", 2025-10-08) changed it to `["col","feat","nbr","full"]` (`git show 7f891d1 -- rt/model.py`). That is the PE order, and it matches paper Alg. 2 and §3.2 (column → feature → neighbor → full).

### 1.2 What the current `net.py` (RT-J) changes — do **not** depict

| # | RT-J feature (CM `src/rt/model/net.py`) | Paper model |
|---|---|---|
| 1 | Custom `RMSNorm(dim, eps=1e-6)`, computed in fp32 then cast back (`:17-28`) | `nn.RMSNorm(d_model)`, default eps (`PE:72,134-150`) |
| 2 | **QK-norm**: per-head `q_norm`, `k_norm` RMSNorm on head_dim (`:41-42,63-64`) | none |
| 3 | **Log-length scaling**: `q = q * scale_h * log(max(kv_size,1))`, with a learnable per-head `scale` init 1 (`:50,66-70`). `kv_size` = number of attendable keys per query per mask (`:364-367`, or `_kv_sizes` `:153-215`) | none |
| 4 | Softmax scale `1/head_dim` (`:79`) | flex default `1/sqrt(head_dim)` |
| 5 | **Output gate** `attn_out *= 2*sigmoid(W_g x)`, with `W_g` zero-init (gate = 1 at init) (`:51-52,83-84`) | none |
| 6 | Zero-init `wo` (`:48`), FFN `w2` (`:97`), and all decoders' weight and bias (`:272-274`) | PyTorch default init |
| 7 | **No full attention**: `attn_types = ["col","feat","nbr"]` (`:111`). The CHANGELOG says "the full-attention block (it did not help)" was removed (`git show 4727f21^:CHANGELOG.md`, line 173) | 4 attention layers incl. full |
| 8 | **No boolean type**: `SEM_TYPE_NAMES = ["number","text","datetime"]` (`:218`). There is no boolean encoder, decoder or mask_emb (`:240-265`). Booleans are z-scored numbers end to end | boolean encoder `Linear(1,d)`, boolean RMSNorm, boolean mask_emb, boolean decoder with BCE |
| 9 | Cells are **re-sorted by `col_name_idx`** (stable argsort, padding last) before everything (`:326-344`). This only helps block sparsity; outputs come back in sorted order (`:491,500-502`) | sequence order as sampled |
| 10 | Optional on-the-fly `mask_mod` closures instead of materialized S×S masks (`:346-429`). Same semantics | materialized boolean masks |
| 11 | Loss: per-sequence mean over masked cells, then mean over the batch (`:461-489`). `loss_fn ∈ {huber, l1, bce}` applies to every head (`:234-237,470-478`). BCE is weighted by `y != 0` (`:475-478`). Per-type losses are also returned (`:480-481`) | global sum / global count; Huber for number and datetime, BCE for boolean |
| 12 | `forward(batch, return_embeddings)` returns 4 outputs (`:316,453-454,491`). `predict` always uses the `number` head (`:501`) | returns `(loss, yhat)`; clf uses the boolean head |
| 13 | Dims: `num_blocks=12, d_model=512, num_heads=8, d_ff=2048, loss_fn="huber"` (`CM/examples/pretrain/phases.py:27-33`) | 12 / 256 / 8 / 1024 |
| 14 | `flex_attention` compiled with `dynamic=False` (`:14`) | `torch.compile(flex_attention)` |

`CM/src/rt/model/legacy/plurel.py` (RT-PluRel) sits between the two. It adds QK-norm (`nn.RMSNorm(head_dim)`), drops full attention and uses per-sequence loss averaging (`diff v1.py plurel.py`). It keeps the boolean head and has no log-scaling or gate. It is not the paper model either.

"RT" in `CM/examples/finetune` is **RT-J architecture trained from random init** (`CM/examples/finetune/plan.py:12-14`; README: "`rt` trains the whole model from random init"). It is not the ICLR model.

### 1.3 A runtime detail useful for the simulation: fully masked attention rows

With the paper masks, many query rows are all-False:

- `nbr` for any cell whose row has no children in the context (e.g. every task row and leaf row);
- every padding query, through `pad`.

FlexAttention returns **0** for a fully masked row, not NaN. The PyTorch test added in PR #133373 asserts `out[:, :, M:, :].sum() == 0` for fully-masked rows ([android mirror of PR #133373](https://android.googlesource.com/platform/external/pytorch/+/629bd6f718%5E%21)). So for such a cell the residual update of that attention sub-layer is `wo·0 = 0`, and the cell passes through unchanged. The simulator should do the same: no NaN, no uniform attention.

---

## 2. Sampler and preprocessing: PE vs CM (task item 2)

### 2.1 PE sampler semantics (the one to animate). `PE/rustler/src/fly.rs`

- **Items (seed rows).** These are all nodes of the task table's split whose cells include the target column (`:287-296`). The split is chosen in Python: key `"{table}:Db"` for autocomplete, else `"{table}:Train|Val|Test"` (`PE/rt/data.py:71-84`).
- **RNG.** `StdRng::seed_from_u64(epoch + seed_node_idx + seed)` per sequence (`:371-375`). Training shuffles items per epoch (`:324-328`). Batches are contiguous slices of the shuffled list (`:336-355`).
- **Frontier**:
  - `f2p_ftr` is a **stack** (LIFO). It starts with `(0, seed)` and is always popped first, so F→P links are followed immediately (`:366,378-380`).
  - Otherwise the sampler picks the smallest depth `d` with a non-empty `p2f_ftr[d]` and swap-removes a uniformly random element (`:381-396`).
  - A visited node is skipped (`:399-402`).
- **On visiting a node:**
  1. Push all F→P parents with depth+1. There is **no temporal filter on F→P** (`:406-408`).
  2. Scan P→F edges (children) (`:415-437`):
     - Keep only Db-table children, or task-table children of the **same task table as the seed** (`:417`).
     - **Temporal filter**: skip if the child's timestamp > the seed's timestamp. Equality is kept (`<=`). The filter only applies when both timestamps exist (`:422`).
     - Db children go to a temp list. Task-table children go straight to `p2f_ftr[depth+1]` and are **never subsampled** (`:426-436`).
  3. **Width bound.** If there are more than `max_bfs_width` Db children, keep a uniform random subset of size `max_bfs_width` (`rand::seq::index::sample`, without replacement) (`:439-452`).
  4. **Emit cells.** Emit all of the node's cells in stored column order (`:454-506`). Columns in `columns_to_drop` are skipped when the node is the seed, or when `node.timestamp == seed.timestamp`. Note `None == None` counts as equal (`:457-461`).
  5. Each emitted cell records:
     - `node_idx`
     - up to 5 `f2p_nbr_idxs`, padded with −1 (`MAX_F2P_NBRS=5`, `:25,464-467`)
     - `table_name_idx`, `col_name_idx`
     - `col_name_values` = text embedding of the column-name string
     - `sem_type`
     - one value per type
     - `is_target = (node is seed) && (col == target)`, with `masks = is_target` (`:492-496`)
     - `is_task_node`
     - `is_padding = false`
  6. **Stop as soon as `seq_len` cells are written, even mid-row** (`:503-509`). Unused positions stay `is_padding=true` (`:82`).
- **Defaults.** In the PE scripts: `seq_len=1024`, `max_bfs_width=256` (`PE/scripts/example_pretrain.py:21,31`). The paper's Fig. 6 example uses L=512, w=128 (paper p.15).
- History inside PE (all before `2a899c8`):
  - `7f891d1` briefly made task-row edges strictly past (`>=`).
  - `66c4247` moved the temporal filter before the width subsampling and reverted to `>`.
  - `801a32b` "fixed bfs width sampling" restored Db children from other tables, which `66c4247` had accidentally dropped (`git show 66c4247 801a32b -- rustler/src/fly.rs`).

### 2.2 CM sampler (`CM/rustler/src/fly.rs`): behavioural differences (do NOT depict)

1. **Target cell at position 0.** `cells_to_add` starts with the target cell (`:1131-1137`). The cell is skipped again inside its own row (`:1741-1743`). In PE the target sits wherever its column falls inside the seed row.
2. **Multi-seed context.**
   - First a local BFS around the target with budget `local_ctx_size` (`:1140-1159`).
   - Then more task-table "seed" rows, each with its own local BFS (`:1205-1231`). These seeds are ranked by random-walk visit counts: `num_walks` walks of length `walk_length` (`:1065-1078,1380-1484`). Optionally ranked by recency (`prefer_latest`, `:1091-1111`) or by FAISS vector-DB neighbours (`:1162-1203`).
   - Then random fallback rows of the task table (`:1239-1340`).
   - PE has exactly one seed and one BFS.
3. **Temporal filter on both directions.** `edge_visible` is applied to F→P and P→F edges (`:1617-1622,1626-1636`). The bound is `min(target_ts, cutoff)` (`:1878-1884`). **Same-horizon task rows are invisible**, i.e. any task row with timestamp == the target's (`:1886-1900,1724-1726`). PE allows equal timestamps.
4. **Budget is in whole rows.** The local BFS stops before adding the row that would reach `local_ctx_size` (`:1608-1611`). The final fill stops at `ctx_len` (`:1747-1749,1353-1355`).
5. **Width sampling.** A partial Fisher–Yates draws over *all* visible P→F edges, with at most `2*bfs_width` draws (`DRAW_BUDGET=2`, `:76,1645-1670`). Eligibility (Db or same table) is checked *after* drawing, so fewer than `bfs_width` may be kept. Task-table children are subsampled too. P→F lists are timestamp-sorted, enabling `partition_point` (`CM/rustler/src/pre.rs:964-969`).
6. **Depth-aware revisit.** A node can be re-expanded if it is reached again at a smaller depth (`visited_at_depth`, `:1600-1604,1613`).
7. **Random multi-cell masking.**
   - Per sequence, `mask_prob = U(0,1) * mask_prob_max` (`:1041`).
   - Each non-target cell is marked `is_target` with probability `mask_prob` (`:1540-1545`).
   - This corresponds to the paper's Table 14 ablation (paper p.24), which is **not** in the PE code. PE masks only the target cell.
   - Eval uses `mask_prob_max=0.0` (`CM/src/rt/eval/evaluator.py:74`).
8. **Booleans emitted as Number** unless `legacy_boolean`. Boolean values go into `number_values` (`:1525-1532`).
9. **NaN handling.** NaN cells are skipped (`:1501-1509`). Seeds with a NaN or missing label are rejected (`:1019-1036,1916-1931`).
10. **Dropped columns** apply to the target row, or same-timestamp rows **only if the target has a timestamp** (`:1734-1739`). PE also drops them on `None == None`.
11. **Batching and robustness.**
    - Training draws items uniformly **with replacement** per step from a seed derived from `context_seed+step` (`:932-954`).
    - `items_per_task` subsampling (`:842-879`).
    - Per-item deadlines and retries (`:958-990`).
    - Extra outputs: `timestamps`, `seed_node_idxs`, `bfs_depths`, `batch_mask` (`:92-97,1550-1555`).
12. **The "legacy" eval path does not use the PE sampler.** CM evaluates `V1Transformer` with `ctx_size=1024, local_ctx_size=1024, num_walks=0, bfs_width=256, prefer_latest=False` through the new sampler (`CM/examples/eval/legacy.py:4-20`; `CM/src/rt/eval/legacy.py:48-70`). Points 1, 3, 4, 5 and 10 therefore still apply, plus point 2's random fallback rows. So CM's legacy eval does not reproduce the PE context.

### 2.3 Preprocessing: PE `PE/rustler/src/pre.rs` (paper model) vs CM `CM/rustler/src/pre.rs`

**PE semantics (depict these):**

- **Inputs.** RelBench parquet with `pkey_col`, `fkey_col_to_pkey_table` and `time_col` read from parquet metadata (`:118-150`).
- **Per-dataset hard-coded casts.** Task labels are cast to Boolean (e.g. `rel-f1 driver-dnf did_not_finish`) (`:208-289`). Some columns are binarized against the first value (`:20-32,179-201`).
- **Node indices.** Each table's rows get consecutive node indices (`node_idx_offset`, `:291-306`).
- **FK resolution.** An FK value `v` is resolved as `parent_offset + v`, i.e. **the key must equal the row number** (`:477`).
- **Primary-key column** → no cell (`:439-442`).
- **FK column** → no cell. It produces instead (`:444-533`):
  - an `f2p_nbr_idxs` entry,
  - an F→P edge carrying the parent's timestamp,
  - a P→F edge on the parent, carrying the child's timestamp and table type (Db/Train/Val/Test).
- **Row timestamp** comes from the time column in **seconds**. Quirk: it is set only inside the FK loop, so a table with a time column but no FK gets no node timestamp (`:481-493`).
- **Cell encoding** (`:535-612`). Nulls and NaN numbers produce no cell.
  - Boolean → `(b − μ_c)/σ_c`, `SemType::Boolean`, stored in `boolean_values` (`:553-564`).
  - Ints and floats → `(v − μ_c)/σ_c`, `SemType::Number` (`:565-583`).
  - Datetime → `(ns − μ_T)/σ_T`, with **one global μ_T, σ_T in nanoseconds over every datetime column** (`:584-594,342-376`). The z-score is unit-invariant, so this matches the paper's "seconds".
  - String → index into a shared text vocabulary, `SemType::Text`, `class_value_idx = text_idx` (`:595-605`).
- **Column stats.**
  - Number: mean and sample std (ddof=1), NaN dropped, std 0→1 (`:330-341`).
  - Boolean: mean and std(ddof=1) with **no zero-std guard**, so a constant boolean column gives 0/0 = NaN (`:320-329`).
  - Datetime: population std over all datetime values (`:375-376`).
- **Stats scope.** Each table computes its own stats, **including all rows of DB tables**. Then Val/Test task tables copy the Train task table's stats (`:383-396`). The global datetime stats include Val/Test task tables too (`:316-373`).
- **Vocabulary.** Column-name strings `"<col> of <table>"` and table names share one vocabulary with text values (`:424-437`).
- **Outputs**: `nodes.rkyv`, `offsets.rkyv`, `p2f_adj.rkyv`, `text.json`, `text_map.json`, `column_index.json`, `table_info.json` (`:619-698`). The text embeddings come from `all-MiniLM-L12-v2` (`PE/rt/embed.py:14-31`), stored as bf16 (`:71-72`).
  - That model's pipeline ends in a `Normalize` module ([modules.json](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2/raw/main/modules.json): Transformer → Pooling → Normalize). So every text and column-name vector is L2-unit-norm before `Linear(384,d)`.

**CM differences (do not depict):**

- **Manifest-driven.** YAML manifests for tables and tasks replace parquet metadata (`:40-135,361-451`). The per-dataset casts moved to Python, in `CM/src/rt/preprocess/legacy/main.py:15-58`. Recommendation tasks are skipped (`:406-408`).
- **Booleans → Float64 0/1 → z-scored `SemType::Number`** (`:811`). The current preprocessor never emits `SemType::Boolean`. So rebuilding "legacy" data with CM's `pre.rs` loses the boolean typing the paper model expects. The `legacy_boolean` flag (`fly.rs:1526`) only helps with data built by an older preprocessor (inference: the HF `legacy/` tree predates internal commit `970167c`, see §5).
- **Train-period stats (temporal-leakage fix, internal PR #4 / `8030aa8`).**
  - Val/Test tables are skipped when computing stats (`:547-550`).
  - DB tables are filtered to rows ≤ `val_timestamp` when the manifest provides one (`:534-572`).
  - The global datetime stats use Welford over the same rows (`:609-631,640-644`).
- **Zero-std guard for booleans** (`:579-580`).
- **`ensure_emittable`.** Drops constant non-structural columns from DB tables. Adds a synthetic `identifier ~ N(0,1)` column when rows are poorly identifiable; the policy comes from `RT_IDENTIFIER_POLICY` (`:205-332`).
- **More input types.** FK values may be lists (multi-parent) (`:168-174,745-796`). Datetime-typed FK columns are skipped (`:728-731`). More dtypes are accepted: Int8…UInt64, Duration→number, Binary/List→text, Categorical→String, Date→Datetime (`:176-203,586-608,810-885`).
- **P→F adjacency sorted by timestamp** (`:964-969`). `meta.json` is written; `text_map.json` is not (`:978-1000`).

---

## 3. README, examples README, notebooks and the mini-shop DB (task item 3)

### 3.1 READMEs

- `CM/README.md:24-28`: the repo hosts RT-J (NeurIPS 2026), RT-PluRel (ICML 2026) and, *for the ICLR paper*, "Legacy architecture + paper experiments: this repo @ `rt-v1`". Tag `rt-v1` = `8d83590`. It differs from `2a899c8` only by an added `timestamps` output and two leaderboard scripts (`git diff 2a899c8 rt-v1 -- rt/ rustler/`), so it has the same model.
- `CM/README.md:43-50`: the Quickstart evaluates **RT-J** (not the ICLR model) on `rel-f1/driver-dnf` and reports AUROC 0.826.
- `CM/examples/README.md:28-68`: describes the data on HF `stanford-star`, with pinned dataset and checkpoint revisions. `relbench-preprocessed` is pinned at `1016626…` (2026-08-08).
- `CM/examples/README.md:70-78`: describes `ctx_viz.py` (see §4).

### 3.2 `quickstart.ipynb`

`python3 -I` dump of the cells:

- **Cell 4.** `snapshot_download("stanford-star/relbench-preprocessed", allow_patterns="rel-f1/*")`, then `RelationalTransformer.from_pretrained("stanford-star/rt-j")` in bf16. An Ampere+ GPU is asserted.
- **Cell 6.** `build_evaluator(... ctx_size_list=[8192], local_ctx_size=256, bfs_width=32, prefer_latest=True, num_walks=10_000, walk_length=20 ...)`. Probabilities = `sigmoid(out)`.
- **Cell 8.** `roc_auc_score(labels > 0, probs)`.

All of these knobs are RT-J-only.

### 3.3 `byod.ipynb`: the mini-shop demo

- **Database** `CM/notebooks/mini-shop.duckdb`, inspected read-only with `python3 -I` + duckdb 1.5.3. The file was first added in `8b2618b` as `examples/byod/mini-shop.duckdb`.

  | table | rows | columns (DuckDB type) | keys |
  |---|---|---|---|
  | `customers` | 1500 | `customer_id` BIGINT, `region` VARCHAR, `signup_year` INTEGER | PK `customer_id` 0..1499 |
  | `products` | 150 | `product_id` BIGINT, `category` VARCHAR, `price` DOUBLE | PK `product_id` 0..149 |
  | `transactions` | 48 200 | `transaction_id` INTEGER, `customer_id` BIGINT, `product_id` INTEGER, `timestamp` TIMESTAMP, `quantity` INTEGER, `amount` DOUBLE | PK `transaction_id`; FK `customer_id`→customers, `product_id`→products; time col `timestamp` |

  Data shape (my queries):
  - `region` ∈ {east, north, south, west}, 375 each, cycling by id.
  - `signup_year` ∈ 2016–2019, 375 each.
  - `category` ∈ {beauty, books, electronics, grocery, home, toys}, 25 each, cycling.
  - `price` 1.49–40.94.
  - Transactions run from 2020-01-01 to 2021-12-20 (~2 000/month). `quantity` is 1–5.
  - **`amount == quantity × price` exactly for all 48 200 rows.**
  - 1498 of 1500 customers transact, between 1 and 202 transactions each.
  - The DB has no declared constraints. Keys are declared only in the manifest the notebook writes (byod cell 7): `customers{pkey}`, `products{pkey}`, `transactions{pkey, time_col:"timestamp", fkeys:{customer_id:customers, product_id:products}}`.
- **Tasks** are point-in-time SQL (byod cell 4).
  - Cutoffs: `2020-01-01 + d days` for `d ∈ range(360, 692, 28)`, giving 12 cutoffs from 2020-12-26 to 2021-10-30.
  - `eligible` = customers with ≥1 transaction before the cutoff.
  - **`customer-churn`** (binary classification): `churn = (no transaction in (cutoff, cutoff+28d])`. Positive rate ≈ 0.407 (17 847 rows).
  - **`customer-spend`** (regression): `spend = COALESCE(sum(amount) in (cutoff, cutoff+28d], 0)`. Mean 75.17, sd 105.69, 40.7 % zeros.
  - Splits (cell 7): train = timestamp < `cuts[-2]` (2021-10-02), val = `cuts[-2]`, test = `cuts[-1]` (2021-10-30).
  - Task manifest: `entity_table=customers, entity_col=customer_id, target_col, task_type ∈ {binary_classification, regression}, time_col=timestamp`.
- **Pipeline (cells 7–13).**
  - `rt.preprocess.one(...)` with `all-MiniLM-L12-v2`.
  - RT-J `build_evaluator` with the same RT-J knobs as the quickstart.
  - `out` is mapped back via `node_idxs − table_info["<task>:Test"].node_idx_offset`.
  - clf: `sigmoid(out)`. reg: de-normalize as `out * train.std(ddof=1) + train.mean()`.
  - Reported results: AUROC 0.8082, MAE 60.11.
- **Key rule (cell 14)**, which matches PE `pre.rs:477`. "each primary key must be `0..n-1` in the order its rows are written, and foreign keys must use those numbers". Otherwise references silently attach to the wrong rows.

**How a PE-faithful dummy DB would tokenize (for our website, inferred from the PE `pre.rs`):**

- **`customers` row** → 2 cells: `region` (Text) and `signup_year` (Number). The PK gives no cell. The row gets no timestamp: there is no time column, and the timestamp is only set in the FK loop anyway.
- **`products` row** → `category` (Text), `price` (Number).
- **`transactions` row** → `timestamp` (DateTime), `quantity` (Number), `amount` (Number). Plus 2 f2p neighbours. The row timestamp is set.
- **Task row** (customer-churn) → `timestamp` (DateTime) + `churn` (**Boolean** after cast; the target). `customer_id` gives no cell, only an F→P link. Its timestamp drives the temporal filter.
- **Suggestion.** mini-shop has no boolean *feature* column. Add one to exercise the boolean encoder on a non-target cell, e.g. `customers.is_premium`. Do not make it constant (see the PE zero-std NaN quirk). Optionally add a free-text column so the text pathway shows up.

---

## 4. `examples/ctx_viz.py`: what it visualizes and how (task item 4)

- **Architecture.**
  - A single-file Python HTTP server, `ThreadingHTTPServer` (`CM/examples/ctx_viz.py:714-898`), with an inline HTML/JS app at `:1008-2713`.
  - It uses **D3 v7** from d3js.org (`:1022`) and CSS design tokens with a light/dark theme (`:1024-1070`).
  - The server caches `RustlerDataset` objects in an LRU and samples one context per request (`:1-13,212-252`).
- **Payload** (`_build_context_payload`, `:212-570`).
  - Per token: node, table (from `table_info` offsets), column short name, sem type, decoded value, f2p list, timestamp, `seed_node_idx`, `bfs_depth`, flags (padding, task node, primary target, masked feature).
  - Groupings by node, by seed, and by (table, column).
  - An F→P edge list among context rows.
  - Per-table counts and stats: padding %, sem-type counts, #nodes/#tables/#edges/#seeds, masked-feature count.
- **Tabs** (`:1661-1666`): Graph, Sequence, By Row, By Column, Attention Masks, Raw Tokens.
  - **Graph** (`:2462-2680`): a deterministic **radial-shell layout**. The primary target sits at the centre, the other seeds on a ring, and each seed's BFS nodes in concentric shells by depth (overlay text `:1671`). Nodes are coloured by table; zoom and drag; click a node to see its row.
  - **Sequence** (`:2090-2199`): an SVG "tape" with one 1-px bar per position, coloured by table, with the target outlined. Hover or click jumps to the token.
  - **By Row / By Column** (`:2200-2337`): cells grouped per node, or per (table, column) with their sem types.
  - **Attention Masks** (`:2384-2460`): recomputes FEAT/NBR/COL masks in JS, mirroring the model. Positions are reordered by `col_name_idx` and drawn as S×S pixel canvases (q rows × kv columns). Each mask has a one-line caption ("my row + the rows I point to", "rows that point to me", "other cells from this column").
  - **Raw** (`:2338-2383`): a token table.
- **Controls** (`:1565-1660`): pre-root, DB, task table, split, target column, drop columns, `ctx_size`, `local_ctx_size`, `bfs_width`, `num_walks`, `walk_length`, `prefer_latest`, `item_idx`, seeds, `mask_prob_max`.
- **Caveat for us.** It visualizes the **RT-J** sampler and masks: three masks, multi-seed shells, `local_ctx_size`, walks. For the paper model, add the 4th **full** mask and use single-seed BFS shells. Drop `local_ctx_size`, walks, `prefer_latest` and `mask_prob_max`, and do not reorder by column. The PE model keeps sampled order; reordering would only be a display choice.
- **Paper-era analogue.** `PE/scripts/visualize_rustler.py` (186 lines) draws a networkx/matplotlib graph of a context, coloured by table, with target and task nodes flagged (`:1-60`, `seq_len=1024, max_bfs_width=256` at `:170-173`).
- **Ideas worth reusing:**
  - radial BFS shells around the seed
  - sequence tape coloured by table
  - pixel-canvas mask matrices next to the row/column views
  - hover-linking between views
  - per-table legend with node and cell counts
  - light/dark tokens

---

## 5. Post-Feb-2026 history relevant to faithfulness (task item 5)

The public history jumps from `2a899c8` (2026-02-04) to README edits (`bbb49a8`, `421478c`, Jun 2026), then:

- **`238d8ba`** (2026-06-27) "leaderboard-submission tooling + timestamps". It adds a `timestamps` output to the PE sampler and an `RT_SAVE_ALL_STEPS` flag (`git diff 2a899c8 rt-v1`). No change to model or sampling semantics. This is the `rt-v1` tag.
- **`8b2618b`** (2026-07-09) "Release cleaned-up RT codebase". It replaces the repo with the RT-J line (fly.rs +2310 lines, model.py rewritten). "The pre-release code remains available at the rt-v1 tag."
- **`4af9ff1`** (2026-09-30) "Relational Transformer 1.9.0". Its CHANGELOG lists the semantic changes from the ICLR snapshot (`git show 4727f21^:CHANGELOG.md`, lines 147-181):
  - full attention removed;
  - booleans are z-scored numbers, with no boolean sem type;
  - "task rows in context are strictly past; a same-horizon task row is invisible";
  - BFS children drawn without rejection, `p2f` bounded at 2×`bfs_width` draws;
  - `remove_columns` scoped to the target's horizon;
  - NaN cells skipped, constant columns never divided by a zero std;
  - identifier-column policy; key-only tables visible.

  Note: the CHANGELOG calls `1.1.0` "the ICLR-submission snapshot, preserved as the `rt-v1` branch" (line 203-207).
- **`358bfd3`** (2026-09-30) "pre-fix normalization caveat" and **`5fef314`** (2026-10-01) "drop the normalization write-up". These docs describe internal PR #4 / commit `8030aa8`, "rustler: column stats from the train period only" (merged 2026-10-01). "before it, column statistics and the global datetime statistics were computed over the validation and test rows too". The published preprocessed trees keep the old leaky normalization. Measured impact on RT-J is within ±0.51 points (mean −0.045 AUROC, −0.035 nMAE over 7 tasks). The RT-v1 checkpoints are explicitly "not covered" and "orphaned from the current preprocessor"; "Pin the revision … to reproduce RT-v1 results" (`git show 358bfd3`, relbench-preprocessed card). The same diff references internal commit `970167c` (2026-09-09), "rustler: no boolean sem type in sampler output; bools are z-scored numbers end to end".
  - **Implication.** The PE code is exactly this "leaky" version (`PE/rustler/src/pre.rs:316-396`). Paper §3.1 says stats come from "the training split" (paper p.4-5). That statement holds for task tables (Val/Test copy Train) but not for DB tables or the global datetime stats. A 100%-faithful simulation of the *paper code* computes stats over the full DB tables.
- `5fef314` also removes a note that the RT-J-era BFS child sampler "used to draw indices with replacement until it filled its quota; it now draws without replacement". That concerns the RT-J line; PE always used `index::sample` without replacement (`PE/rustler/src/fly.rs:440`).
- Other later commits (`b806960` time-column dtype warning, `fdd42da` GIL release, `0485763` ensemble seeds, `035fec9` relicense to Apache-2.0) do not touch paper semantics.
- **Pre-Feb fixes already inside PE** (good to know, no action needed):
  - `e55bf0d`: zero-weight touch of unused decoders.
  - `7f891d1`: attention order → col, feat, nbr, full; `is_target` no longer restricted by sem type.
  - `77bdb9b`: `MAX_F2P_NBRS` assert.
  - `66c4247`: temporal filter before width subsampling.
  - `801a32b`: restore Db children from other tables.

---

## 6. Paper vs PE-code discrepancies the site should handle explicitly

| Topic | Paper | PE code | Suggested site behaviour |
|---|---|---|---|
| Block form | `X + Norm(MHA(X))` (Alg. 2, p.26) | pre-norm `X + MHA(Norm(X))` (`PE/rt/model.py:83-85`) | show the code; footnote the paper notation |
| Normalization stats | "training split" (p.4-5) | full DB tables; val/test task tables copy Train stats; global datetime over all tables (`pre.rs:316-396`) | show the code version; footnote the later fix |
| Datetime unit | seconds (p.5) | nanoseconds (`pre.rs:586`) | identical after z-scoring; mention only |
| Width bound | applies to all P→F neighbours (Alg. 1, p.15) | only Db-table children are subsampled; task-table children are all kept (`fly.rs:426-452`) | follow the code |
| Temporal filter | `T(q) ≤ T(s)` on P→F (Alg. 1) | same (`>` skip), P→F only; rows without timestamps are always allowed (`fly.rs:422`) | follow both (they agree); show same-time task rows as visible |
| Stopping | "stops when total cells reaches L" (p.16) | truncates mid-row at `seq_len` (`fly.rs:503-509`) | show the truncation |
| Column mask | same column | same `col_name_idx` AND same table (`model.py:181-183`) | equivalent, since the column id is table-qualified |
| Masked cells | MTP; Table 14 ablates P(mask)>0 (p.24) | only the target cell is masked (`fly.rs:496`) | default = target only; P(mask) is an ablation, not the default |
| Booleans | z-score like numbers, BCE on `1{r>0}` (p.4,6) | Boolean sem type, z-scored; BCE on `y>0` (`pre.rs:553-564`, `model.py:252-255`) | consistent |
