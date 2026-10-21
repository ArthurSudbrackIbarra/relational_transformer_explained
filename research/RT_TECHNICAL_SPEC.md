# Relational Transformer (RT), ICLR 2026: technical specification for a faithful re-implementation

This document specifies the Relational Transformer as published in "Relational Transformer: Toward Zero-Shot Foundation Models for Relational Data" (Ranjan et al., arXiv 2510.06377v3, ICLR 2026). It is written for engineers re-implementing a faithful miniature RT in TypeScript for an educational website that runs on a tiny dummy database.

**Ground truth.** "What actually happens" means the paper-era code: the official repository at commit `2a899c8` (2026-02-04). It matches the camera-ready paper. The paper's wording is given wherever it differs. Every non-trivial fact carries a citation.

---

## Citation keys

| key | meaning |
|---|---|
| `PE/<file>:<lines>` | Paper-era code snapshot, commit `2a899c8` (local copy `.../scratchpad/upstream/rt-paper-era`). Files: `rt/model.py`, `rt/data.py`, `rt/main.py`, `rt/tasks.py`, `rt/embed.py`, `rustler/src/{fly.rs,pre.rs,common.rs}`, `scripts/example_*.py`, `README.md`. Line numbers are 1-indexed. |
| `MAIN <commit>` | Official repo history (local clone `.../scratchpad/upstream/rt-official`). Current `main` is the later "RT-J" line and **not** paper behaviour. Tag `rt-v1` = `8d83590` = `2a899c8` + leaderboard tooling only (`git diff 2a899c8 rt-v1` touches only README, a `timestamps` output, `RT_SAVE_ALL_STEPS` and two predict scripts). |
| `paper p.N §X` | Printed page N of `/home/ibarra/Documents/Mestrado/Relational_Transformer/Relational_Transformer.pdf` (arXiv v3, 2 Mar 2026, "Published as a conference paper at ICLR 2026"; PDF page = printed page). |
| `Alg.1`, `Alg.2` | Paper Algorithm 1 (p.15) and Algorithm 2 (p.26). |
| URL | Web source. |
| `[verified locally]` | Checked by a small script in `.../scratchpad/specchecks/` (torch 2.12 CPU, Node 24), not by running repository code. |

Version facts you can rely on:

- `PE/rustler/src/{fly,pre,common}.rs` are byte-identical to commit `801a32b` (2025-10-16, "fixed bfs width sampling"). `git diff 801a32b 2a899c8 -- rustler/` is empty.
- `pre.rs` has not changed since the ICLR-submission commit `eeccc2b` (2025-10-06).
- The block order `col, feat, nbr, full` comes from `MAIN 7f891d1` (2025-10-08, "fix in attention order and sampling"). Before that it was `feat, nbr, col, full`.
- Pinned environment: `torch==2.6.0` (`PE/pixi.lock:203`).

---

## 0. One-page overview of the pipeline

```
 RelBench-style relational DB (tables, PK/FK, time columns)  +  task tables (train/val/test parquet)
        │  (§1, §2)
        ▼
 PREPROCESSING (pre.rs, once per DB)                                                     (§3)
   • every row of every table (task tables included) → one "node" with a global index
   • PK column → nothing; FK column → F→P edge (+ reverse P→F edge on the parent); no token
   • every other non-null, non-NaN cell → one cell record with a sem_type (Number/Text/DateTime/Boolean)
   • numbers & booleans: per-column z-score; datetimes: one global z-score; text: index into one string table
   • column-name phrase "<col> of <table>" interned; MiniLM-L12-v2 embeds all strings (384-d, unit norm)
        │
        ▼
 CONTEXT SAMPLING (fly.rs, per training/eval example)                                     (§4)
   seed row s (a task row, or a DB row for autocomplete) → bounded-width BFS:
   • F→P parents: LIFO stack, always followed first, never time-filtered, never subsampled
   • P→F children: only children with ts ≤ ts(s); DB children subsampled to ≤ w per visited row;
     children from the seed's own task table (any split) all kept; other task tables skipped
   • next P→F row = uniform pick among the lowest-depth frontier bucket
   • emit every cell of each visited row (drop leakage columns), stop at exactly L cells; pad the rest
   • exactly ONE masked cell: the seed row's target cell
        │  L tokens (L = 1024), each: node idx, ≤5 parent idxs, table idx, column idx,
        │  column-name embedding, sem_type, value(s), mask flag, padding flag            (§5)
        ▼
 INPUT EMBEDDING  x_i = RMSNorm_col(W_col·E("<col> of <table>") + b)
                       + [unmasked] RMSNorm_t(W_t·value + b_t)  or  [masked] m_t (learned vector)
        ▼
 12 × RT BLOCK (pre-norm, residual)                                                      (§6, §7)
   x += MHA_col (RMSNorm(x); same column)          ┐ 8 heads, d_head 32, scale 1/√32,
   x += MHA_feat(RMSNorm(x); own row + parents)    │ no positional encodings,
   x += MHA_nbr (RMSNorm(x); child rows)           │ boolean masks AND non-padding
   x += MHA_full(RMSNorm(x); all non-padding)      ┘
   x += SwiGLU(RMSNorm(x))      (256 → 1024 → 256, no biases)
        ▼
 RMSNorm_out → per-type linear decoders (number, datetime, boolean → 1 scalar; text → 384)     (§8)
   loss on the masked cell only: Huber(δ=1) on z-score (number/datetime), BCE-with-logits on 1{z>0} (boolean)
   prediction = decoder output at the masked position: logit (binary) or z-score (regression)
   metrics: AUROC on logits (= on sigmoid probabilities); R² in z-space (= raw-scale R²)
        ▼
 TRAINING REGIMES (§9): leave-one-DB-out pretraining on all tasks of 6 DBs (50k steps, batch 256)
   → zero-shot on the held-out DB's forecasting tasks; optional continued pretraining; fine-tuning.
```

What it is in one sentence: a 22.3M-parameter, 12-block, d=256 transformer over **cell tokens**. Structure enters **only** through four boolean attention masks; there is no positional encoding. Every prediction task is masked-cell prediction on a "seed" row whose context is gathered by a temporal, bounded-width BFS over primary–foreign-key links (paper p.4-6 §3; `PE/rt/model.py`; `PE/rustler/src/fly.rs`).

---

## 1. Relational-DB concepts as RT uses them

### 1.1 Tables, rows, cells, columns

- An RDB is a set of tables. A table is a set of rows; a row has one cell per column (paper p.3 §2.1).
- **Feature columns** are what RT tokenizes. The paper says "columns that contain numeric, text, and datetime information" (p.3 §2.1). The code also has a Boolean type (`PE/rustler/src/common.rs:38-44`).
- **ID columns** "define how rows are uniquely identified and connected across tables" (p.3 §2.1). These are the primary key (PK) and foreign keys (FK).
- **Cell = token.** "A cell is represented by (v, c, t), where v is the cell value, c is the column name, and t is the table name" (p.4 §3.1). Only non-missing feature cells become tokens: "all its non-missing feature cells (i.e., cells not from primary- or foreign-key columns)" (p.16 App. A).

### 1.2 Primary and foreign keys

- The **PK column produces nothing**: no token and no edge (`PE/rustler/src/pre.rs:439-442`).
- **Each FK column produces no token.** For each non-null FK value it produces a link (`pre.rs:444-533`):
  1. the parent's global node index is appended to the child's `f2p_nbr_idxs` (`pre.rs:479`);
  2. an F→P `Edge{parent, parent_table_name_idx, Db, parent_ts}` is appended to the child's `f2p_edges` (`pre.rs:515-521`);
  3. a P→F `Edge{child, child_table_name_idx, child_table_type, child_ts}` is appended to `p2f_adj[parent]` (`pre.rs:523-529`).
- **FK value = parent row position.** The parent node is `offset(parent_table:Db) + value` (`pre.rs:452-459, 477`). RelBench re-indexes PKs to 0..n−1 in row order and maps FKs accordingly. A demo DB must therefore use PK = row position, with FK values pointing at those positions. The authors later spelled this out as "each primary key must be `0..n-1` in the order its rows are written" (MAIN `notebooks/byod.ipynb` cell 14; see also MAIN `9f0e027`).
- Only `Int64` FK values are accepted; anything else panics (`pre.rs:472-476`). Null FKs are skipped (`pre.rs:462-464`).
- FK and PK **column names are never used** by the model. RT "does not incorporate the names of primary-key and foreign-key columns" and "does not disambiguate between different foreign key columns" (paper p.10 §7 Limitations). The code keeps only the ordered list of parent node ids.
- At most **5 non-null FK values per row** can be represented: `MAX_F2P_NBRS = 5` (`fly.rs:25`), batch layout `[B,S,5]` (`data.py:143`). The limit counts non-null values, not FK columns, because null FKs are skipped before the push (`pre.rs:462-464` before `pre.rs:479`). The check `assert!(node.f2p_nbr_idxs.len() <= MAX_F2P_NBRS)` (`fly.rs:464`, added in MAIN `77bdb9b`) sits inside the per-cell emission loop, after the leakage-drop test (`fly.rs:455-464`). It therefore fires only when such a row is visited **and** emits at least one cell; a zero-cell row with 6 FKs, or one never reached, does not panic. Keep ≤5 FK columns per table in the demo.

### 1.3 F→P and P→F links

- An **F→P link** goes from a row holding an FK to the row it references (its parent). A **P→F link** is the reverse, from a parent to each referencing child (paper p.3 §2.1).
- A row has at most #FK-columns parents but an unbounded number of children. That is why sampling treats the two directions asymmetrically (paper p.16 App. A).
- Task-table rows are always children (they have an FK to the entity), never parents (`pre.rs:444-533`; task tables have no PK). The structural reason: an FK's parent is always looked up as `(<table>, Db)` (`pre.rs:452-458`) and the F→P edge is hard-coded `table_type: TableType::Db` (`pre.rs:518`), so only DB rows can be parents.

### 1.4 Temporal rows

- Each table may declare one `time_col`, read from parquet metadata (`pre.rs:145-150`).
- **The time column is an ordinary datetime feature cell**, z-scored like any other datetime (`pre.rs:584-594`). It is also the source of the row's integer `timestamp`, used only by the sampler.
- `node.timestamp` is in **seconds** (`ns / 1_000_000_000` as `i32`). **It is set only inside the FK loop** (`pre.rs:481-493`). A row whose table has a time column but no non-null FK therefore gets `timestamp = None`. Examples: an FK-less dimension table with a time column, or a row whose FKs are all null. A row whose time cell is null also gets `None` (`.get(r)` returns `None`, `pre.rs:488`), so the temporal filter always lets it through (`fly.rs:422` needs `edge.timestamp.is_some()`).
- **Integer range and resolution.** `node.timestamp = trunc(ns / 1e9)`, then cast with `as i32` (`pre.rs:489`; parent ts likewise at `pre.rs:510`; fields are `Option<i32>`, `common.rs:28, 53`). The `as` cast wraps silently, so dates after 2038-01-19T03:14:07Z or before 1901-12-13 become wrong values and silently break the temporal filter and the same-timestamp drop rule (`fly.rs:422, 458`). Dummy dates must lie in 1901-12-14 .. 2038-01-18. Sampler comparisons have 1-second resolution: times within the same second compare equal. Datetime z-scores are unaffected, because they use the f64 ns value (`pre.rs:586`).
- The P→F edge carries the **child's** timestamp (`pre.rs:523-528`). The F→P edge carries the parent's timestamp, but the sampler never reads it (`fly.rs:406-408`).
- Temporal leakage rule, as stated in the paper: "modeling is conditioned only on rows that were created prior to the target row" (p.3 §2.1). The implemented rule is narrower (§4.6).

### 1.5 Global node indexing

- All rows of all tables of one DB, task tables of all splits included, share one integer index space.
- Order: DB parquet files sorted by path, then task parquet files `tasks/<task>/{test,train,val}.parquet` sorted by path (so test < train < val) (`pre.rs:79-92`).
- `node_idx = node_idx_offset(table) + row_position` (`pre.rs:291-306, 466, 537`).
- `table_info.json` maps `"<table>:<Db|Train|Val|Test>"` → `{node_idx_offset, num_nodes}` (`pre.rs:648-664`).
- Two-dash task directories (recommendation/link-prediction tasks such as `user-item-purchase`) are skipped by counting dashes (`pre.rs:70, 93-95`). RT cannot do link prediction (paper p.10 §7).

---

## 2. Task tables

### 2.1 Forecasting task tables

- A task table "stores the forecasting labels, together with foreign keys linking to the relevant entities (e.g., user IDs or item IDs) and a timestamp specifying the prediction horizon" (paper p.4 §2.2).
- In RT, "for each task, we attach a dedicated task table to the database. Only one task table is active at a time … Rows from the task table serve as seed rows … from the model's perspective task tables are treated as ordinary relational tables" (p.4 §3.1). The paper calls this "task table prompting".
- RelBench schema of a task table: `time_col` (e.g. `date` or `timestamp`), one FK `entity_col → entity_table`, the target column, and `pkey_col = None`. Example: rel-f1 `driver-dnf` = `date` (time), `driverId` (FK → drivers), `did_not_finish` (RelBench `tasks/f1.py`, v1.1.0).
- **After preprocessing, a task row has exactly 2 tokens**: the timestamp cell (DateTime) and the label cell (Boolean or Number). The FK produces only `f2p_nbr_idxs = [entity]` plus edges (`pre.rs:444-613`).
- `is_task_node = true` for every task-table row (`pre.rs:468, 539`). This flag is **not used by the model** (only `PE/scripts/visualize_rustler.py:22,53` reads it).

### 2.2 Splits and timestamps

These come from RelBench (`relbench/base/task_base.py:102-142` at v1.1.0, https://github.com/snap-stanford/relbench/blob/v1.1.0/relbench/base/task_base.py).

- **train**: timestamps `val_timestamp − Δ, val_timestamp − 2Δ, …` back to the DB's minimum timestamp.
- **val**: `val_timestamp` (… up to `num_eval_timestamps` windows, default 1).
- **test**: `test_timestamp` (default a single timestamp).
- The label of a row (entity e, time t) aggregates the future window (t, t+Δ] (RelBench `task_base.py:40-43` docstring; e.g. "if they will DNF … in the next 1 month", RelBench paper App. A, arXiv 2407.20060). Example cutoffs: rel-f1 val 2005-01-01 / test 2010-01-01 (RelBench `datasets/f1.py`).

Consequences in RT:

- Each split is its own table `("<task>", Train|Val|Test)` with its own offsets. **All three splits share one `table_name_idx` and one `col_name_idx` per column**, because the name string is the task name in all three (`pre.rs:152-174, 424-437`).
- **Test labels are present on disk.** RelBench caches `{split}.parquet` unmasked, and `pre.rs` reads it directly (`pre.rs:87-91`). The sampler masks only the seed's own target cell (`fly.rs:492-496`), and evaluation reads labels from the batch (`main.py:283-288`).

### 2.3 Seeds and targets

- A **seed** ("item") is every row of the requested table whose cells contain the target column, i.e. whose label is non-null and non-NaN (`fly.rs:287-296`).
- The table key is `"<table>:Db"` if such a DB table exists (autocomplete), else `"<table>:<Train|Val|Test>"` (forecasting) (`PE/rt/data.py:77-81`).
- The **target column** is identified by `column_index["<target_col> of <table>"]` (`data.py:31-43, 86-87`).
- The **target cell** is the seed row's cell in the target column. It is the one masked token (§4.10).

### 2.4 Autocomplete tasks

- "Masked cell prediction on feature columns that already exist in the database … sampling starts from the masked database row" (paper p.19 App. D).
- The seed is a DB row with `table_type = Db`. Every task-table child then fails the "seed's task table" test, so **no task rows appear in an autocomplete context** (`fly.rs:417`).
- In-context labels are the **visible target-column values of other rows of the same DB table** reached by the BFS.
- **Leakage columns** `columns_to_drop` are given per task (`PE/rt/tasks.py:30-88`). Example: rel-f1 `results.position` drops `statusId, positionOrder, points, laps, milliseconds, fastestLap, rank`. They are removed as tokens for the seed row and for every row with the same timestamp as the seed (§4.9). The paper says "identify potential sources of information leakage and discard these columns on the fly" (p.19 App. D).
- **Split is ignored for autocomplete.** data.py always picks `"<table>:Db"`, so the seeds are all rows of the table from every time period. No cutoff is applied (`data.py:77-81`).

### 2.5 How labels become in-context labels (forecasting)

- Path: seed task row → (F→P stack) entity row at depth 1 → the entity's P→F children. These children are:
  - DB rows, time-filtered and subsampled to w;
  - **every task row of the same task table, any split, with ts ≤ ts(seed)**, never subsampled.

  All of them go into depth bucket 2 (`fly.rs:406-452`).
- These rows' label cells are **unmasked**. They are the "in-context labels". "While task rows provide 'in-context labels', our setting is not few-shot as explicit subgraph-label pairs are not required" (paper p.2).
- Other entities' task rows are reached deeper, e.g. entity → transaction → product → another transaction → another customer → that customer's task rows. Paper Table 12 (p.23) counts these per 1024-cell context (computed with the pre-`66c4247` sampler; see §11.3 provenance).
- Visibility by seed split, which follows from the shared `table_name_idx` and the `≤` filter:
  - **Train seed**: train rows with ts ≤ its own ts.
  - **Val seed**: train rows, plus val rows at the same `val_timestamp` (other entities).
  - **Test seed**: train rows, val rows, and test rows at the same `test_timestamp` (other entities).
- **Same-horizon leakage (paper-era behaviour).** Because the filter is `child_ts > seed_ts → skip`, other entities' task rows at the **same** timestamp as the seed (same split, same horizon) pass, labels visible (`fly.rs:422`). The ICLR-submission code `eeccc2b` excluded equal-timestamp task rows. `66c4247` removed that exclusion, and the final paper-era code keeps it removed. Later RT-J re-excluded them as "same-horizon task rows" (MAIN `4af9ff1` CHANGELOG; MAIN `rustler/src/fly.rs:1724, 1886-1900`). See §10 and §11 for why this barely matters numerically (Table 4: removing all other-entity labels gives 70.6 vs 70.1 AUROC). Caveat: Table 4 is unchanged since arXiv v1 (2025-10-07), so it was produced before `66c4247`, when the public sampler (`eeccc2b`) still excluded same-horizon task rows. It therefore does not measure the effect of the same-horizon labels that the released sampler shows (§13 #1).

### 2.6 Task inventory (paper experiments)

From `PE/rt/tasks.py`; matches paper App. D Tables 8-9 (p.19). Each tuple is `(db, table, target, leakage_cols)`.

- **Forecast classification (12)**: amazon user-churn, item-churn; hm user-churn; stack user-badge, user-engagement; avito user-visits, user-clicks; event user-ignore, user-repeat; trial study-outcome; f1 driver-dnf, driver-top3 (`tasks.py:3-16`).
- **Forecast regression (9)**: hm item-sales; amazon user-ltv, item-ltv; stack post-votes; trial site-success, study-adverse; event user-attendance; f1 driver-position; avito ad-ctr (`tasks.py:18-28`).
- **Autocomplete classification (7)** and **autocomplete regression (8)**: `tasks.py:30-88`.
- **The loss/decoder is chosen by the sem_type of the target cell**, not by the task list (`model.py:238-258`). Labels that RelBench stores as int 0/1 are cast to Boolean in preprocessing (`pre.rs:208-289`), e.g. `did_not_finish`, `churn`. A 0/1 integer column not in that list stays a Number and gets Huber.
- Quirk: rel-event `user-attendance` is listed as regression (`tasks.py:25`; `main.py:235`) but cast to Boolean (`pre.rs:272-278`). It is therefore trained with BCE. rel-event is never evaluated (paper p.6).
- **Supported target types**: binary (Boolean → BCE) and regression (Number → Huber). Datetime targets would work (Huber, `model.py:250-251`) but are never used. **Text or categorical targets are unsupported**: `raise ValueError("masking text not supported")` (`model.py:256-257`; paper p.6 §3.3 "we only mask cells in boolean or numeric columns"). Multi-class, ranking, recommendation and link prediction are out of scope (paper p.10 §7).

---

## 3. Preprocessing (`PE/rustler/src/pre.rs`, `PE/rt/embed.py`)

### 3.1 Inputs

- One parquet file per DB table (`db/*.parquet`) and per task split (`tasks/<task>/{train,val,test}.parquet`).
- Key-value metadata on each file: `pkey_col`, `fkey_col_to_pkey_table`, `time_col` (`pre.rs:107-150`).
- **Hard requirements** (otherwise `pre.rs` panics):
  - Each parquet carries all three metadata keys as JSON (`metadata.get(...).unwrap()`, `pre.rs:118, 124, 145`). `pkey_col` and `time_col` must be `null` or a string. `fkey_col_to_pkey_table` must be an object, `{}` if there are no FKs (`null` hits `_ => panic!()`, `pre.rs:125-143`).
  - Each task that has Val/Test needs a Train split, because Val/Test copy its stats (`col_stats_map.get(table_name).unwrap()`, `pre.rs:392`).
  - FK parents are always looked up as `<table>:Db` (`pre.rs:452-458, 518`), so only DB rows can be parents (§1.3).
- rel-avito FK targets are rewritten: `UserID → UserInfo`, `AdID → AdsInfo` (`pre.rs:133-139`).
- Hard-coded per-dataset transforms (`pre.rs:177-289`):
  - Some task labels are cast to Boolean: `cast_col_to_bool`, where nonzero → true (`pre.rs:61-65`).
  - Some DB columns are binarized as "equals the value of row 0" by `make_column_boolean` (`pre.rs:21-32`). The comment says "first non-null", but the code uses row 0. Example: stack `postLinks.LinkTypeId` "1 vs 3", trial `has_dmc` "t vs f" (paper Table 8 p.19).
  - rel-amazon `product.category` keeps only the first list element (`pre.rs:216-227`).
  - rel-stack drops `posts.AcceptedAnswerId` (`pre.rs:238-240`).
  - rel-event: `drop_nulls`; a `dummy` row-index column is added to the key-only `user_friends` table (`pre.rs:265-271`).
  - None of this is needed for a dummy DB. Just declare dtypes directly.

### 3.2 Type inference (by physical dtype, per value; `pre.rs:543-611`)

| dtype | sem_type (code) | stored value | other slots |
|---|---|---|---|
| `Boolean` | Boolean (3) | `boolean_values = ((b?1:0) − μ)/σ` | number 0, text idx 0, datetime 0, class −1 (`pre.rs:553-564`) |
| `UInt32`, `Int32`, `Int64`, `Float32`, `Float64` (cast to f64) | Number (0) | `number_values = (v − μ)/σ` | (`pre.rs:565-583`) |
| `Datetime` (must be ns) | DateTime (2) | `datetime_values = (t_ns − μ_T)/σ_T` | (`pre.rs:584-594`) |
| `String` | Text (1) | `text_values = class_value_idx = text_id(value)` | (`pre.rs:595-605`) |
| any other dtype with a non-null value | — | `panic!()` | (`pre.rs:606-611`) |

- `SemType` enum order: Number=0, Text=1, DateTime=2, Boolean=3 (`common.rs:38-44`). The model iterates `["number","text","datetime","boolean"]` with these indices (`model.py:216, 238`).
- **No semantic inference.** Categoricals and strings are Text. Integer IDs that are not declared PK/FK are Numbers and get z-scored. Example: `AccountId: 12827220.0` in the paper's serialization (p.25).
- `class_value_idx` is carried to the batch but **never used by the paper-era model** (`fly.rs:471`; no use in `model.py`/`main.py`).

### 3.3 Normalization: exact formulas

Statistics are computed **per (table, split) and per column**, over **all rows of that table file** (`pre.rs:316-373`). Then `Val`/`Test` task tables **overwrite** their stats with the same task's `Train` stats (`pre.rs:383-396`). There is no clipping anywhere.

**Number** (`pre.rs:330-341`):
```
x  = column as f64, nulls dropped, NaN dropped
μ  = mean(x)                         (0.0 if x is empty)
σ  = sample std, ddof = 1            (1.0 if undefined, i.e. n ≤ 1)
if σ == 0: σ = 1
z  = (v − μ) / σ                     (pre.rs:569; a ±inf result panics, pre.rs:570-575)
```

**Boolean** (`pre.rs:320-329`):
```
x  = column cast to f64 (true=1, false=0), nulls dropped
μ  = mean(x)                         (0.0 if empty)
σ  = sample std, ddof = 1            (0.0 if undefined)   ← NO zero guard
z  = ((b ? 1 : 0) − μ) / σ           (pre.rs:554-555)
```
- With 0 < μ < 1: true → (1−μ)/σ > 0 and false → −μ/σ < 0. So **z > 0 ⇔ true**. This is what the BCE target `1{z>0}` and the AUROC label `int(z>0)` rely on (`model.py:253-255`; `main.py:336`; paper p.6 §3.3 "BCE(1{r > 0}, r′)").
- **Edge case**: a constant boolean column (σ=0), or one with a single non-null value (σ undefined → 0 via `unwrap_or(0.0)`, `pre.rs:323`), gives (b−μ)/0 = 0/0 = **NaN** for every value in that table file, because every value equals μ exactly. ±inf occurs only when a Val/Test task split reuses a constant Train column's stats (`pre.rs:383-396`) and holds the other value, e.g. Train all false, Val true → (1−0)/0 = +inf. That still becomes NaN after the encoder's RMSNorm (inf/inf).
  - Unlike the Number branch, which panics on inf (`pre.rs:570-575`), the Boolean branch has no NaN/inf check (`pre.rs:553-564`), so the NaN goes into the batch.
  - Masking does not remove it. Gating is multiplicative (`model.py:217-223`) and NaN·0 = NaN, so the NaN enters that token's input even when the cell is the masked target.
  - Full attention then spreads it to every non-padding token of the sequence. Because the loss multiplies per-position losses by 0/1 masks and sums (`model.py:246, 260`), the whole per-GPU batch loss becomes NaN.
  - The code does not guard this (later fixed in RT-J: MAIN `4af9ff1` CHANGELOG "constant columns are never normalized by a zero standard deviation"). **A faithful demo DB must not contain constant boolean columns, and every task's Train split must contain both label classes.**
- Worked value: a label with 25% positives has μ=0.25 and σ=√(0.25·0.75·n/(n−1)) ≈ 0.433 for large n (sample std, ddof=1, `pre.rs:321-323`). So true → (1−0.25)/0.433 ≈ **+1.732** (=√3) and false → −0.25/0.433 ≈ **−0.577** (=−1/√3). Only a 50/50 label gives σ ≈ 0.5 (true ≈ +1, false ≈ −1). Small n changes σ through the n/(n−1) factor: n=1000 with 250 true gives σ=0.43323 and +1.7312/−0.5771; n=8 with 2 true gives σ=0.4629 and +1.620/−0.540. Do not hard-code these in a UI; compute them from the data.

**Datetime: one global pair per database** (`pre.rs:342-366, 375-376`):
```
over EVERY Datetime column of EVERY table in the DB (all DB tables AND Train/Val/Test task tables):
    collect t_ns (i64 nanoseconds as f64), nulls/NaN dropped
μ_T = Σt / n
σ_T = sqrt(Σt²/n − μ_T²)              ← POPULATION std (ddof = 0), no zero guard
z   = (t_ns − μ_T) / σ_T              (pre.rs:586)
```
- The paper says "Convert to seconds and normalize globally" (p.5 §3.1). The code uses nanoseconds. **The z-score is identical** because the transform is affine. A TS implementation may use seconds or milliseconds.

**Text**: no normalization. The value is an index into the text table, embedded by the frozen encoder (§3.6).

### 3.4 Which rows feed the statistics: paper vs code

- **Paper**: "µc and σc are the column mean and standard deviation computed on the training split"; "global mean and standard deviation of timestamps in the training split" (p.4-5 §3.1).
- **Code**:
  - **Task-table labels**: Train-split stats, reused for Val and Test ✔ (`pre.rs:383-396`).
  - **DB tables**: stats over **all rows of the full DB**, including rows after the val/test timestamps. No cutoff is read anywhere: no `cutoff`, `val_timestamp` or `upto` in `PE/rt`, `PE/rustler/src` or `PE/scripts`.
  - **Datetime**: global over every table and split.
- The authors later documented this as the "pre-fix" (leaky) normalization and fixed it in RT-J. The fix is internal PR #4 / commit `8030aa8` "column stats from the train period only", described in MAIN `358bfd3` (`docs/cards/relbench-preprocessed.md` parent revision). They measured ≤0.51-point changes on RT-J and state that its effect on RT-v1 is unmeasured. MAIN `5fef314` then dropped the write-up.
- A faithful simulation of **the paper code** computes DB-table stats over all rows. It may offer a toggle labelled "RT-J fix" (§10 #3).

### 3.5 Missing values

- `null` → **no cell at all** (`pre.rs:552`). Float `NaN` → no cell (`pre.rs:566-568`).
- Missing values are **neither imputed nor masked**; the row simply has fewer tokens. This matches "non-missing feature cells" (paper p.4 §3.1, p.16 App. A).
- Row-header fields (`node_idx`, `table_name_idx`, `is_task_node`) are set even for null cells (`pre.rs:537-541`).

### 3.6 Text dictionary, column-name phrase, text encoder

- **One string table per DB** (`text_to_idx`, `pre.rs:406`). It holds:
  - table names (`pre.rs:424-427, 495-498`);
  - every column phrase **`"<col> of <table>"`** (`pre.rs:432-437`), for every column including PK/FK columns, which get indices but never produce cells;
  - every distinct string cell value (`pre.rs:596-597`).

  Identical strings share one index and one embedding.
- `column_index.json` maps `"<col> of <table>"` → index (`pre.rs:641-645`). Task targets and drop columns are resolved through it (`data.py:31-43`).
- **The table name enters the token only through the phrase.** Paper examples: "price of product", "age of user" (p.5 §3.1). For task tables the phrase is e.g. `"churn of user-churn"` or `"date of driver-dnf"`. `table_name_idx` is used only as an integer, in the column mask and the sampler's task-table test (`model.py:181-183`; `fly.rs:417`).
- **Encoder**: `SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")`, `encode(text_list, batch_size=8192, convert_to_numpy=True)`, stored as raw bf16 `[len(text), 384]` (`PE/rt/embed.py:14-31, 70-72`).
  - At paper time it ran in fp32 on `cuda:0`; bf16 inference was introduced later, in `7d564f7` (2025-11-03) (`git show 801a32b:rt/embed.py`).
  - Module pipeline: Transformer → mean Pooling → **Normalize**, so vectors are **L2-unit-norm** (https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2/raw/main/modules.json; `1_Pooling/config.json`: `pooling_mode_mean_tokens: true`). `max_seq_length = 128` (`sentence_bert_config.json`).
  - BERT: 12 layers, hidden 384, uncased tokenizer.
  - The paper calls it "MiniLMv2 [41] … 384 dimensional embeddings" (p.6 §4.1). The "-v2" in the model name is the sentence-transformers release, not MiniLMv2 (§10 #16).
  - Browser/Node port: `Xenova/all-MiniLM-L12-v2` (transformers.js ONNX, https://huggingface.co/Xenova/all-MiniLM-L12-v2). Use `{pooling:'mean', normalize:true}` **and truncate inputs to 128 tokens** to match exactly. sentence-transformers truncates at `max_seq_length = 128` (https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2/raw/main/sentence_bert_config.json), but the tokenizer's own limit is `model_max_length = 512` (`tokenizer_config.json` of both repos), so a JS pipeline without the 128 cap gives different embeddings for strings longer than 128 wordpieces. Short dummy strings are unaffected.
- **Text cells are embedded from the raw value string only**, with no column context (`embed.py:65-68`; `pre.rs:596-597`). Column context comes from the separate column-name term.
- **Non-text cells store text index 0** (`pre.rs:559, 578, 589`), so the sampler copies the embedding of `text[0]` (some table name) into their `text_values` (`fly.rs:484-486`). The model gates that term out by sem_type (`model.py:216-220`). In a re-implementation, zeros are equivalent.

### 3.7 Precision

- Per-cell values are stored as f32 (`common.rs:57-60`) and converted to **bf16** (round-to-nearest-even) when the batch is built (`fly.rs:482-490`). Labels used in the loss and metrics are bf16 too.
- Example: 0.3333 → 0.333984375 [verified locally]. A TS simulation in float64 is didactically fine; mention bf16 in the UI.

### 3.8 Storage layout (only if you mirror it)

`nodes.rkyv` + `offsets.rkyv`, `p2f_adj.rkyv`, `text.json`, `text_map.json`, `column_index.json`, `table_info.json`, `text_emb_<model>.bin` (`pre.rs:619-698`). A TS demo only needs in-memory equivalents:

```ts
interface Edge { node: number; tableNameIdx: number; tableType: 'Db'|'Train'|'Val'|'Test'; ts: number|null }
interface Cell { colNameIdx: number; sem: 0|1|2|3; num: number; textIdx: number; dt: number; bool: number }
interface Node { nodeIdx: number; tableNameIdx: number; isTaskNode: boolean; ts: number|null;
                 f2pNbrIdxs: number[]; f2pEdges: Edge[]; cells: Cell[] }     // cells in DataFrame column order
// p2fAdj[parent] : Edge[]   (edge.ts = CHILD's timestamp; edge.tableType = CHILD's table type)
```

---

## 4. Context-window sampling (`PE/rustler/src/fly.rs`, Alg. 1)

### 4.1 Parameters

| name | meaning | paper runs | source |
|---|---|---|---|
| `L` = `seq_len` | number of cell tokens per context | 1024 | `PE/scripts/example_pretrain.py:31`; paper p.6 §4.1 |
| `w` = `max_bfs_width` | max **DB** children kept per visited row | 256 in all scripts (paper Fig. 6 uses L=512, w=128) | `example_pretrain.py:21`; p.15 Fig. 6 |
| seed row | the row whose target cell is predicted | — | `fly.rs:362` |
| `target_column` | `col_name_idx` of `"<target> of <table>"` | — | `data.py:86` |
| `columns_to_drop` | `col_name_idx` list of leakage columns | `[]` for all forecasting tasks | `tasks.py` |
| RNG seed | `epoch + seed_node_idx + seed` | `seed=0` | `fly.rs:371-375` |

### 4.2 RNG

- `StdRng::seed_from_u64(epoch + seed_node_idx + seed)` per sequence, with wrapping u64 adds (`fly.rs:371-375`). This is ChaCha12 in rand 0.9 (https://docs.rs/rand/0.9.1/rand/rngs/struct.StdRng.html).
- A context is a deterministic function of (epoch, seed row, seed). Training contexts change every epoch. Eval contexts are fixed (epoch=0, seed=0) (§8.6).
- **Bit-exact RNG parity is not required for didactic fidelity.** Any seeded PRNG gives the same distribution.

### 4.3 Exact algorithm (mirrors `fly.rs:357-511`)

```ts
function sampleContext(seed: number, L: number, w: number, targetCol: number, dropCols: number[],
                       nodes: Node[], p2fAdj: Edge[][], rng: Rng): Token[] {
  const S = nodes[seed];
  const visited = new Array(nodes.length).fill(false);  // fly.rs:364 (whole DB)
  const f2pStack: [number, number][] = [[0, seed]];        // fly.rs:366  (depth, node) LIFO STACK
  const buckets: number[][] = [];                          // fly.rs:368  buckets[d] = P→F frontier at depth d
  const out: Token[] = [];
  const push = (d: number, n: number) => { while (buckets.length <= d) buckets.push([]); buckets[d].push(n); };

  while (true) {
    // ---- select next row (fly.rs:378-397)
    let depth: number, n: number;
    if (f2pStack.length > 0) {
      [depth, n] = f2pStack.pop()!;                        // F→P rows have absolute priority, LIFO
    } else {
      const d = buckets.findIndex(b => b.length > 0);      // smallest depth label with candidates
      if (d === -1) break;                                 // frontier exhausted → rest is padding
      const b = buckets[d];
      const r = rng.int(b.length);                         // uniform in [0, len)
      [b[r], b[b.length - 1]] = [b[b.length - 1], b[r]];   // swap-remove
      n = b.pop()!; depth = d;
    }
    if (visited[n]) continue;                              // fly.rs:399-401 (checked at POP time)
    visited[n] = true;
    const node = nodes[n];

    // ---- expand frontier BEFORE emitting cells (fly.rs:406-452)
    for (const e of node.f2pEdges) f2pStack.push([depth + 1, e.node]);     // ALL parents; no time filter; no visited test
    const dbKids: number[] = [];
    for (const e of p2fAdj[n]) {
      if (e.tableType !== 'Db' && e.tableNameIdx !== S.tableNameIdx) continue;   // fly.rs:417 other task tables
      if (e.ts !== null && S.ts !== null && e.ts > S.ts) continue;              // fly.rs:422 keep ts ≤ seed ts
      if (e.tableType === 'Db') { dbKids.push(e.node); continue; }              // fly.rs:426-429
      push(depth + 1, e.node);                                                    // fly.rs:431-436 task rows: never subsampled
    }
    const keep = dbKids.length > w ? rng.sampleWithoutReplacement(dbKids.length, w)   // fly.rs:439-443
                                   : [...dbKids.keys()];
    for (const i of keep) push(depth + 1, dbKids[i]);                            // fly.rs:445-452

    // ---- emit cells (fly.rs:454-506)
    for (const c of node.cells) {                                                 // stored column order
      if (dropCols.includes(c.colNameIdx) && (node.nodeIdx === seed || node.ts === S.ts)) continue; // null===null is TRUE (Rust Option ==)
      const isTarget = node.nodeIdx === seed && c.colNameIdx === targetCol;     // fly.rs:492-494
      out.push(makeToken(node, c, /*mask=*/isTarget));                            // fly.rs:462-500
      if (out.length >= L) return out;                                            // fly.rs:502-509: may cut a row
    }
  }
  return out;                                                                     // caller pads to L
}
```

### 4.4 Selection rule, in words

1. **F→P first, LIFO.** If the parent stack is non-empty, pop its last element (`fly.rs:378-379`). There is no randomness here. A row's parents are pushed in FK-column order, so the **last FK is visited first**. Each parent's own parents are pushed on top, so FK chains are followed depth-first and completely before any P→F row. Paper: "F→P links are immediately followed" (p.16 App. A). Alg. 1 instead writes `RandomSelect(R)` among F→P rows (§10 #5).
2. **Otherwise P→F, by lowest depth label.** Take the smallest `d` with a non-empty bucket and remove a uniformly random element. Paper: "unvisited rows at the same depth from the seed row are sampled uniformly" (p.16).
3. **Visited check at pop time.** Duplicates may sit in the frontier. Popping a visited row is a no-op, but it still consumes an RNG draw if it came from a bucket.

### 4.5 Depth labels vs "HopDistance"

- Alg. 1's `argmin_{r∈F} HopDistance(r, s)` is implemented as **the depth label carried by the frontier entry**: +1 per hop, F→P or P→F (`fly.rs:407, 436, 451`).
- It is the length of the discovery path, not a true shortest-path distance. The first pop wins, and the row is expanded with that entry's depth.
- F→P rows ignore depth for *selection* (the stack has priority). Their depth still sets the bucket of *their* children.
- Example: seed task row d0 → entity d1 → entity's children in bucket 2 → a child transaction visited from bucket 2 pushes its parents at d3 (visited immediately via the stack) → those parents' children go to bucket 4, and the transaction's own P→F children go to bucket 3 (`fly.rs:436, 451` push to `p2f_ftr[depth + 1]`). Selection always takes the smallest non-empty bucket (`fly.rs:381-391`), so bucket 4 is drawn from only when buckets 2 **and** 3 are both empty.

### 4.6 Temporal filter: exact semantics (`fly.rs:421-424`)

- It applies **only to P→F children**. The comparison is child timestamp vs **seed** timestamp, never vs the current row.
- A child is skipped iff **both timestamps exist and `child_ts > seed_ts`**. Equal timestamps are **kept** (`≤`), matching Alg. 1 `T(q) ≤ T(s)` (p.15).
- **A child with no timestamp always passes.** Its table may have no time column, or its time value may be null.
- **If the seed has no timestamp, the filter is disabled for the whole context.** This happens for autocomplete seeds in tables without a time column or without FKs (§1.4).
- **F→P parents are never time-filtered** (`fly.rs:406-408`). Paper: "all parent rows (F→P) are always included" (p.4 §3.1).
- There is **no global cutoff**: no DB rows are dropped by date in preprocessing or sampling (§10 #10). RT-J later added `cutoff_timestamps` (MAIN `rustler/src/fly.rs:336, 1005, 1878-1884`).
- History (verified with `git show`/`git diff` in MAIN):
  - `eeccc2b` (2025-10-06, ICLR submission): time filter **before** width subsampling; equal-ts task rows excluded (`edge.timestamp == seed_node.timestamp && edge.table_type != Db → continue`); `is_target` restricted to Number/Text/Boolean cells, so DateTime cells could never be masked.
  - `7f891d1` (2025-10-08): DB children subsampled *before* the time filter (post-sample check on `db_node.timestamp`); task rows with ts `>=` seed excluded; `is_target` sem-type restriction removed.
  - `77bdb9b` (2025-10-08): adds the `≤ MAX_F2P_NBRS` assert (§1.2).
  - `66c4247` (2025-10-09): filter-before-subsample restored, with a single `>` test (equal-ts task rows now kept); bug: the table test `edge.table_name_idx != seed_node.table_name_idx → continue` also dropped every DB child of another table.
  - `801a32b` (2025-10-16): bug fixed. This is the final paper-era sampler. Semantically it equals `eeccc2b` minus the equal-ts task-row exclusion, minus the `is_target` sem-type restriction, plus the `77bdb9b` assert (`git diff eeccc2b 801a32b -- rustler/src/fly.rs`).

### 4.7 Width bound w (`fly.rs:426-452`)

- Applied **per visited row, only to its DB children that passed the time filter**. If more than w remain, keep a uniform random subset of size w without replacement (`rand::seq::index::sample`, https://docs.rs/rand/0.9.1/rand/seq/index/fn.sample.html).
- It is applied **after** the temporal filter, matching Alg. 1's order (filter, then `RandomSample(N, w)`). This was already the order in `eeccc2b`; `7f891d1` reversed it for one day and `66c4247` restored it (§4.6 History).
- **Task-table children (the in-context label rows) are never subsampled** (`fly.rs:431-436`). Alg. 1 subsamples all P→F neighbours (§10 #7).
- Children are pushed even if already visited; they are skipped when popped.
- **Already-visited children count toward w.** `db_p2f_ftr` collects every DB child that passes the table test (`fly.rs:417`) and the time test (`fly.rs:422`), with no `visited[...]` check (`fly.rs:415-429`). That includes rows already visited, such as the child row through which the BFS reached this parent (e.g. a product reached via F→P from order o1 has o1 among its DB children). The uniform w-subsample is drawn over all of them (`fly.rs:439-443`), and visited ones are discarded at pop time, so a row can end up with fewer than w new children.

### 4.8 Row expansion order vs cell emission

- A row's frontier is expanded (`fly.rs:406-452`) **before** its cells are emitted (`fly.rs:454-506`). This only affects RNG consumption order.

### 4.9 Cell emission and leakage columns

- **Every cell of the visited row is emitted, in stored order.** Stored order is the DataFrame column order minus PK, FK, null and NaN cells (`pre.rs:429, 535-613`).
- **Drop rule** (`fly.rs:457-461`): skip a cell if its column is in `columns_to_drop` **and** (the row is the seed **or** `row.timestamp == seed.timestamp`).
  - Rust `Option` equality makes `None == None` **true**. With a timestamp-less seed, the drop therefore applies to every timestamp-less row carrying that column.
  - Drop indices are `"<col> of <seed table>"`, so only rows of the seed's table can match.
  - Leakage columns in rows with *other* timestamps stay visible.
  - Example: in rel-f1 `results.position`, other drivers' results in the **same race** (same date) lose `points`, `positionOrder`, …, but keep their visible `position` values. RT-J later added `target.timestamp.is_some() &&` (MAIN `fly.rs:1734-1737`).

### 4.10 Masking: exactly one masked cell

- `is_target = (row == seed) && (col == target_column)`, and `masks = is_target` (`fly.rs:492-496`).
- **There is no random masking anywhere** in the paper-era sampler or model. Training and evaluation build sequences identically.
- The masked cell **stays a token**. Its column-name embedding and sem_type are kept; only its value embedding is replaced by the learned `m_t` (`model.py:216-224`). The raw label is still in the batch tensor and serves as the loss/metric target.
- **All other cells are visible**, including other rows' values in the target column. These are the in-context labels (other task rows; other DB rows for autocomplete).
- The paper's multi-cell masking with P(mask) ∈ {0.2, 0.4} (Table 14, p.24) is an **ablation that is not in the released code**. Main results are P(mask)=0. The 0.0 column (70.1/22.8) equals the "none" baselines of Tables 4/11.
- Pre-submission commits (`ab17c23`…`6cae3ef`) had a cosine-decayed `mask_prob` sampler. It was removed at `eeccc2b`.

### 4.11 Stopping, truncation, padding

- **Stop after exactly L tokens, even mid-row** (`fly.rs:502-509`). The paper says "once a row is selected, all of its non-missing feature cells are added" (p.4) and "stops when the total number of cells reaches the context length" (p.16). The code can cut the last row (§10 #8).
- **Or stop when the frontier is empty.** The remaining positions stay as padding (`fly.rs:387-389`). There is no iteration limit.
- **Padding token defaults** (`fly.rs:65-85`):
  - `node_idx = −1`, `f2p_nbr_idxs = [−1,−1,−1,−1,−1]`;
  - `table_name_idx = 0`, `col_name_idx = 0`, `class_value_idx = −1`;
  - `col_name_values = 0⃗`, `text_values = 0⃗`, `sem_type = 0`, all scalar values 0;
  - `masks = is_targets = is_task_nodes = false`, `is_padding = true`.

### 4.12 Token order

- The seed row is visited first (initial stack entry `(0, seed)`), so **its cells, including the masked target, are tokens 0..k−1** (`fly.rs:366`).
- After that, rows come in visit order; within a row, cells come in stored column order.
- **Order has no effect on the model** (no positional encodings, §6.7). It matters only for which rows fit before L and where truncation happens.
- The authors' slides mention "column-major serialization of tokens" for block sparsity (https://iclr.cc/media/iclr-2026/Slides/10007116.pdf). That is not in the paper-era sampler, which emits row by row. RT-J re-sorts by column inside the model (MAIN `src/rt/model/net.py:326-344`). Do not depict it as paper behaviour.

### 4.13 `f2p_nbr_idxs` layout

- Shape `[B, S, 5]`, i32, padded with −1 (`fly.rs:69, 465-467`; `data.py:143`).
- **Every cell of a row repeats the row's full parent list**, in FK-column order. Duplicates are possible if two FKs hit the same parent.
- The list contains **all parents, whether or not they are in the context**. A parent influences the masks only if some of its cells are present (§6).

### 4.14 Seeds, shuffling, batching (`fly.rs:257-355`, `main.py`)

- `items` = concatenation over all tasks of the qualifying rows in node order (`fly.rs:257-297`). During pretraining **one item list spans all training tasks and DBs** (`main.py:123-136`).
  - There is **no task balancing**: a task's share is proportional to its number of seeds.
  - A batch mixes DBs, tasks, classification and regression.
- Shuffle: `shuffle_py(epoch)` sets `self.epoch = epoch` and shuffles with `StdRng(epoch + seed)` (`fly.rs:324-328`). Training calls it with `int(steps/len(loader))` before each pass (`main.py:370`).
- Batches: `len = ceil(N/(B·W))`. Batch b on rank k takes items `(b·B·W + k·B + i) mod N`; the last batch wraps around (`fly.rs:332-355`). `true_batch_size` counts the real ones; training ignores it (`main.py:401`), eval neutralizes the rest (§8.6), except on ranks where the usize subtraction underflows (§8.6).
- Per-GPU `B = 32` × 8 GPUs = global batch 256 (`example_pretrain.py:19`; `PE/README.md:128` `--nproc_per_node=8`; paper p.6 "batch size of 256").
- DataLoader `batch_size=None`: each Rust call returns one whole batch (`main.py:137-144`; `data.py:116-154`).

### 4.15 Alg. 1 (paper) vs `fly.rs` (code)

| aspect | paper (Alg. 1 p.15, App. A p.16, §3.1 p.4) | code |
|---|---|---|
| choice among F→P-added rows | `RandomSelect(R)` | LIFO pop, deterministic (`fly.rs:378-379`) |
| choice among P→F rows | min HopDistance, uniform | min depth label, uniform (`fly.rs:381-396`) |
| temporal filter | `T(q) ≤ T(s)` on P→F | same; only when both timestamps exist (`fly.rs:422`) |
| width bound | `RandomSample(N, w)` on all P→F | DB children only; task rows never subsampled (`fly.rs:426-452`) |
| task tables | "only one task table is active at a time" | only the seed's task table, **any split** (`fly.rs:417`) |
| stop | `while |C| < L` | hard stop at exactly L tokens; last row may be partial (`fly.rs:502-509`) |
| cells of a row | non-missing feature cells | same, minus `columns_to_drop` for seed/same-ts rows |
| masked cells | the target cell | exactly one: the seed's target cell |

### 4.16 Edge cases a demo can hit

- **Rows with zero cells** (key-only link tables) are visited and expand the frontier but emit nothing (`fly.rs:454-506`).
- **A row with more than 5 non-null FK values** makes the sampler panic (`fly.rs:464`), but only when that row is visited and emits at least one cell (§1.2). Keep ≤5 FK columns per table.
- **A seed without a timestamp** disables the temporal filter and widens the drop rule (`None==None`).
- **A constant boolean column** produces NaN values that poison the whole batch loss, even when the NaN cell is masked (§3.3).
- **Dates outside 1901-12-14 .. 2038-01-18** wrap the i32 sampler timestamp silently (§1.4).
- **A null/NaN label** means the row is not a seed. **A NaN feature** produces no cell.
- **The frontier may empty before L** (small DB), leaving padding. This is common in a tiny demo.

---

## 5. Token representation

### 5.1 Per-token fields (batch dict, `fly.rs:27-44, 462-500`; reshaped in `data.py:116-154`)

| field | shape / dtype | value for a cell | padding | used by model? |
|---|---|---|---|---|
| `node_idxs` | [B,S] i32 | global row index | −1 | yes: masks (`model.py:168-178`) |
| `f2p_nbr_idxs` | [B,S,5] i32 | row's parent node idxs, −1-padded | −1×5 | yes: feat/nbr masks |
| `table_name_idxs` | [B,S] i32 | interned table name; Train/Val/Test of a task share it | 0 | yes: col mask (`model.py:181-183`) |
| `col_name_idxs` | [B,S] i32 | interned `"<col> of <table>"` | 0 | yes: col mask |
| `col_name_values` | [B,S,384] bf16 | MiniLM embedding of `"<col> of <table>"` | 0⃗ | yes: input embedding |
| `sem_types` | [B,S] i32 | 0 number, 1 text, 2 datetime, 3 boolean | 0 | yes: gating, loss |
| `number_values` | [B,S,1] bf16 | z-score, or 0 | 0 | yes (gated) |
| `text_values` | [B,S,384] bf16 | MiniLM embedding of the string (non-text cells: embedding of text[0]) | 0⃗ | yes (gated) |
| `datetime_values` | [B,S,1] bf16 | global z-score, or 0 | 0 | yes (gated) |
| `boolean_values` | [B,S,1] bf16 | z-scored 0/1, or 0 | 0 | yes (gated); label `y>0` |
| `masks` | [B,S] bool | = `is_targets` | false | yes: input + loss |
| `is_targets` | [B,S] bool | seed row ∧ target column | false | eval readout (`main.py:283-288`) |
| `is_task_nodes` | [B,S] bool | `node.is_task_node ∨ col == target` (`fly.rs:498-499`) | false | **no** |
| `class_value_idxs` | [B,S] i32 | text idx for text cells, else −1 | −1 | **no** |
| `is_padding` | [B,S] bool | false | true | yes |
| `true_batch_size` | int | real (non-wrapped) sequences in batch | — | eval only |

**Ids are per-DB index spaces.** Node, table, column and text ids are assigned per database, because `pre.rs` runs once per `cli.db_name`: node offsets come from `num_rows_sum` (`pre.rs:302`) and `text_to_idx` is created fresh (`pre.rs:406`). One batch can hold sequences from different DBs, because items from every dataset go into one `items` vec (`fly.rs:257-297`) that is shuffled together (`fly.rs:324-328`). Equal ids in different sequences can therefore mean different rows or columns. This is harmless because every mask compares ids only within one sequence, `[:, :, None] == [:, None, :]` per batch row b (`model.py:165-183`). A batch-level visualization must not compare ids across sequences.

### 5.2 Input embedding: exact formula (`model.py:208-224`)

For token i with `pad_i = is_padding[i]`, `t_i = sem_types[i]` and `μask_i = masks[i]`:

```
x_i =  [¬pad_i] · RMSNorm_colname( W_colname · c_i + b_colname )                       c_i ∈ R^384 (col_name_values)
     + Σ_{t ∈ {number,text,datetime,boolean}}
         [t_i = t ∧ ¬μask_i ∧ ¬pad_i] · RMSNorm_t( W_t · v_i^t + b_t )                  v^number, v^datetime, v^boolean ∈ R^1; v^text ∈ R^384
       + [t_i = t ∧  μask_i ∧ ¬pad_i] · m_t                                             m_t ∈ R^256, learned
```

- Encoders: `number`, `datetime`, `boolean` are `nn.Linear(1, 256, bias=True)`; `text` and `col_name` are `nn.Linear(384, 256, bias=True)` (`model.py:115-123`).
- **Each encoder output passes through its own RMSNorm** (`norm_dict`, 5 RMSNorms: number, text, datetime, col_name, boolean; `model.py:132-140, 210-218`).
- **Mask vectors `m_t` are added raw**, not normalized: `nn.Parameter(torch.randn(256))` per type (`model.py:141-146, 221-224`).
- **Exactly one value term per non-padding token**: either its type's encoded value, or that type's mask vector if masked.
- A **padding token's input is the zero vector**.
- Paper form: `x = W_d r + W E_schema(c,t)`, "For masked cells, the value embedding W_d r is replaced with a learned mask vector m_d" (p.5 §3.1). The code adds biases and an RMSNorm after each projection (§10 #2).
- Teaching note (follows from the formula): a scalar value v enters as `RMSNorm(v·w + b)`. Before the RMSNorm gain, all numbers of one type lie on a one-parameter curve on (approximately, because of ε) the sphere of radius √256. In the input vector x, the column is identified **only** by the column-name term and the value only by this curve. Separately, the integer column and table ids decide which tokens attend to each other in column attention (`model.py:181-189`); no embedding is involved there.

### 5.3 The masked target token

- It has the target column's name embedding, sem_type (number or boolean), its node and parent ids, and `m_t` instead of a value.
- It therefore knows **which column of which table, of which type, in which row** it must predict. That row is linked to the entity through `f2p_nbr_idxs`.
- **The sampler does not blank the label.** The token still stores the true z-scored label in its `number_values`/`boolean_values` slot, because values are written for every emitted cell, target included (`fly.rs:482-490`); only `is_targets`/`masks` are set for it (`fly.rs:492-496`). The input embedding ignores the value through the `~masks` gate (`model.py:219`). It is read only as the loss target (`model.py:240`) and the eval label (`main.py:285, 288`). Replacing it with any **finite** placeholder leaves ŷ unchanged; a NaN/inf placeholder would not, because NaN·0 = NaN (§3.3). A simulator should show that the label sits in the token arrays but cannot affect the prediction.

---

## 6. Relational attention

### 6.1 Multi-head attention (`model.py:20-47`)

```
Q = h·W_qᵀ,  K = h·W_kᵀ,  V = h·W_vᵀ          (h = RMSNorm_l(x); W_* ∈ R^{256×256}, no bias)
split heads: "b s (h d) -> b h s d", 8 heads, d_head = 32      (head j uses dims [32j, 32j+32))
O_head[q] = Σ_k softmax_k( Mask(Q_head[q]·K_head[k]; M) / √32 ) · V_head[k]
           Mask(A;M)_qk = A_qk if M[q,k] = 1, −∞ otherwise                                 (paper p.5 §3.2)
out = concat_heads(O) · W_oᵀ                    (W_o no bias)
```

- Implemented with `flex_attention(q, k, v, block_mask)`, default scale `1/√E` with E = 32 (https://docs.pytorch.org/docs/2.8/nn.attention.flex_attention.html; `model.py:43`) [verified locally: matches masked softmax with 1/√d].
- **All 8 heads share each mask** (`create_block_mask(..., H=None)`, `model.py:89-101`). Masks are built once per forward from dense `[B,S,S]` booleans and reused in all 12 blocks (`model.py:164-206, 226-227`).
- Attention is **bidirectional**: there is no causal mask.
- The `block_mask is None` SDPA/Flash branch (`model.py:39-41`) is never taken, because all four layers always get a mask (`model.py:204-206`).
- **Fully masked query rows output exactly 0**, not NaN and not uniform. This happens for padding queries, and in `nbr` for any cell whose row has no children in the context. This is confirmed from the pinned torch v2.6.0 source (`pyproject.toml:17` `torch = "==2.6.0"`) for both paths:
  - Compiled Triton kernel (used here, because `flex_attention = torch.compile(flex_attention)`, `model.py:17`): `torch/_inductor/kernel/flex_attention.py:422-427`, "`# [Note] Handle fully masked out rows: # Li will be the sum(e^(-inf)) == 0.0 for masked out rows`", then `l_i = tl.where(l_i == 0.0, 1, l_i)` and `acc = acc / l_i[:, None]`. `acc` stays 0, so the output is 0.
  - Eager/dense reference: `torch/_higher_order_ops/flex_attention.py:242-249`, `masked_rows = torch.all(post_mod_scores == -float("inf"), dim=-1)` and `torch._safe_softmax(post_mod_scores, dim=-1)`.

  Also [verified locally on torch 2.12 CPU: max |out| = 0.0, no NaN]. With bias-free `W_o` the residual update is exactly 0 and the token passes through unchanged.

### 6.2 The four masks: paper notation and code

Paper (p.5 §3.2): Col(i) is the column of token i, Row(i) its row, and OutLinks(i) the set of rows pointed to by the FKs of Row(i).

| layer | paper mask | code (`model.py:165-191`) | what it lets flow |
|---|---|---|---|
| **column** | `1{Col(k) = Col(q)}` | `col_name_idx[q]==col_name_idx[k] ∧ table_name_idx[q]==table_name_idx[k] ∧ pad` | between all cells of the same column of the same table. "helps model the distribution of values in each column" (p.5). **The masked label cell sees every visible label cell of its task (all splits).** |
| **feature** | `1{Row(k)=Row(q) ∨ Row(k) ∈ OutLinks(q)}` | `(node[q]==node[k] ∨ node[k] ∈ f2p[q]) ∧ pad` | from own row and **parent** rows into each cell. "equivalent to row-wise attention after joining each table with its parent tables" (p.5). Child → parent direction only. |
| **neighbor** | `1{Row(q) ∈ OutLinks(k)}` | `node[q] ∈ f2p[k] ∧ pad` | from **child** rows into the parent's cells. "aggregate signals from its child rows … analogously to message-passing in GNNs" (p.5). Exactly the transpose of the parent part of the feature mask. |
| **full** | `1` | `pad` | all non-padding pairs. "confers the expressive power of standard Transformers" (p.5). |

`pad[q,k] = ¬is_padding[q] ∧ ¬is_padding[k]` (`model.py:165`).

Naming: the paper also calls the **full** layer "**global**": in §3.2's list "four attention types: column, feature, neighbor, and global" (p.5), in App. E (p.19-20, "global attention can emulate it") and in App. J (p.26, "a global attention layer"). The bullet title "Full attention", Alg. 2 (`MHA(X; M^full)`, p.26) and the code (key `"full"`, `model.py:186-191`) use "full". It is the same layer, with mask = `pad` (`model.py:190`). There are four layers, not five.

Implementation gotchas:

- **The `∧ pad` is essential.** Padding has `node_idx = −1` and parent lists are −1-padded, so without it `kv_in_f2p` would match padding keys to every query's empty parent slots (`model.py:171-173, 187`).
- The column test also requires the same table (`model.py:181-183`). That is redundant, because `col_name_idx` already encodes `"<col> of <table>"` (`pre.rs:432-437`), but it is cheap to mirror.
- **Self-attention is allowed** in column, feature and full (the token's own row and column). Neighbor contains self only if a row's FK points at that same row, which does not occur in RelBench-style data.
- **Every non-padding token has at least one allowed key** in column, feature and full (itself). Only neighbor rows can be empty.
- **Cross-split sharing**: Train/Val/Test task rows share `table_name_idx` and `col_name_idx`, so label cells of all splits are in one column-attention group (`pre.rs:424-437`).

### 6.3 Worked example (toy DB, exact output of an independent re-implementation)

The DB (PK = row position; `orders.order_time` and the task `timestamp` are time columns):

- `customers(customer_id PK, country:str, age:float, is_premium:bool)`: c0 = (BR, 34, true), c1 = (US, 51, false), c2 = (BR, null, false).
- `products(product_id PK, category:str, price:float)`: p0 = (books, 12), p1 = (toys, 30).
- `orders(order_id PK, customer_id FK, product_id FK, order_time, quantity:int)`: o0 = (c0, p0, 2024-01-10, 1), o1 = (c0, p1, 02-05, 2), o2 = (c1, p0, 02-20, 1), o3 = (c0, p0, 03-15, 3), o4 = (c2, p1, 01-25, 1).
- Task `customer-churn(timestamp, customer_id FK, churn:bool)`:
  - Train: (01-01, c0, F), (01-01, c1, T), (02-01, c0, F), (02-01, c2, T).
  - Val: (03-01, c0, **T**), (03-01, c1, F).
  - Test: (04-01, c0, F), (04-01, c2, T).

Node offsets in glob order: customers 0-2, orders 3-7, products 8-9, churn:Test 10-11, churn:Train 12-15, churn:Val 16-17.

Statistics:

- `age` μ=42.5, σ=12.0208; c2 has no age cell.
- `is_premium` μ=1/3, σ=0.5774, so true → 1.1547 and false → −0.5774.
- `churn` uses Train stats μ=0.5, σ=0.5774, so ±0.8660 for every split.
- Datetime global over 13 datetime cells: μ_T = 2024-02-12T18:27Z, σ_T = 30.08 days (population std).

Sampling, with seed = Val row 0 (#16, c0 @ 2024-03-01), L=24, w=1:

1. Pop `(0,#16)` from the stack. Push parent `(1,c0)`. Emit `timestamp of customer-churn`, then `churn of customer-churn` **[MASK]**.
2. Pop `(1,c0)` (stack). Its children are o0, o1, o3, #10, #12, #14, #16.
   - o3 (03-15) and #10 (Test 04-01) are **future**, so they are dropped.
   - Task rows #12, #14, #16 all go into bucket 2 (#16 is the seed itself, kept by `≤`, later skipped as visited).
   - DB kids [o0, o1] exceed w=1, so a random subsample keeps o1. **o0 is never reached.**
   - Emit country, age, is_premium.
3. Bucket 2 → #14 (Train 02-01, c0). Emit its timestamp and **churn=false** (self label). Its parent c0 is already visited.
4. Bucket 2 → o1. Push its parents (c0, p1). p1 is visited via the stack; p1's children: o1 (visited later) and o4 → bucket 4.
5. Bucket 2 → #16 (visited, skip). Bucket 2 → #12 (Train 01-01, c0, churn=false).
6. Bucket 4 → o4. Its parents c2 and p1 go on the stack. c2's children: #11 (Test, future, dropped), #15 (Train 02-01, c2) and o4 → bucket 6.
7. Bucket 6 → #15. This is an **other-entity label** (c2, churn=true). Then the frontier is empty, so positions 19-23 are padding.

Tokens:

```
 0 churn:Val[0]#16   timestamp of customer-churn  (2) 2024-03-01   f2p=[0]
 1 churn:Val[0]#16   churn of customer-churn      (3) [MASK]       f2p=[0]
 2 customers[0]#0    country of customers         (1) "BR"         f2p=[]
 3 customers[0]#0    age of customers             (0) 34           f2p=[]
 4 customers[0]#0    is_premium of customers      (3) true         f2p=[]
 5 churn:Train[2]#14 timestamp of customer-churn  (2) 2024-02-01   f2p=[0]
 6 churn:Train[2]#14 churn of customer-churn      (3) false        f2p=[0]
 7 orders[1]#4       order_time of orders         (2) 2024-02-05   f2p=[0,9]
 8 orders[1]#4       quantity of orders           (0) 2            f2p=[0,9]
 9 products[1]#9     category of products         (1) "toys"       f2p=[]
10 products[1]#9     price of products            (0) 30           f2p=[]
11 churn:Train[0]#12 timestamp of customer-churn  (2) 2024-01-01   f2p=[0]
12 churn:Train[0]#12 churn of customer-churn      (3) false        f2p=[0]
13 orders[4]#7       order_time of orders         (2) 2024-01-25   f2p=[2,9]
14 orders[4]#7       quantity of orders           (0) 1            f2p=[2,9]
15 customers[2]#2    country of customers         (1) "BR"         f2p=[]
16 customers[2]#2    is_premium of customers      (3) false        f2p=[]
17 churn:Train[3]#15 timestamp of customer-churn  (2) 2024-02-01   f2p=[2]
18 churn:Train[3]#15 churn of customer-churn      (3) true         f2p=[2]
19-23 PAD
```

Masks (rows = query, cols = key, `#` = allowed):

```
        column                     feature                    neighbor
    012345678901234567890123   012345678901234567890123   012345678901234567890123
 0  #....#.....#.....#......   #####...................   ........................
 1  .#....#.....#.....#.....   #####...................   ........................
 2  ..#............#........   ..###...................   ##...####..##...........
 3  ...#....................   ..###...................   ##...####..##...........
 4  ....#...........#.......   ..###...................   ##...####..##...........
 5  #....#.....#.....#......   ..#####.................   ........................
 6  .#....#.....#.....#.....   ..#####.................   ........................
 7  .......#.....#..........   ..###..####.............   ........................
 8  ........#.....#.........   ..###..####.............   ........................
 9  .........#..............   .........##.............   .......##....##.........
10  ..........#.............   .........##.............   .......##....##.........
11  #....#.....#.....#......   ..###......##...........   ........................
12  .#....#.....#.....#.....   ..###......##...........   ........................
13  .......#.....#..........   .........##..####.......   ........................
14  ........#.....#.........   .........##..####.......   ........................
15  ..#............#........   ...............##.......   .............##..##.....
16  ....#...........#.......   ...............##.......   .............##..##.....
17  #....#.....#.....#......   ...............####.....   ........................
18  .#....#.....#.....#.....   ...............####.....   ........................
19-23 all '.' (padding)        all '.'                    all '.'
full: rows 0-18 = '#' on keys 0-18, '.' on 19-23; rows 19-23 all '.'
```

Reading the masks:

- **Column**: the masked target (row 1) sees the label cells 6, 12 and 18. Two are its own past labels (c0: false, false) and one belongs to another entity (c2: true). Every timestamp cell (0, 5, 11, 17) sees all task timestamps. `order_time` sees only `order_time`.
- **Feature**: the target (row 1) sees its own row (0, 1) and its parent c0 (2-4). Order o1 (7, 8) sees itself, c0 (2-4) and p1 (9, 10). Customer cells see only their own row.
- **Neighbor**: c0's cells (2-4) see their children in context: #16 (0, 1), #14 (5, 6), o1 (7, 8) and #12 (11, 12). p1 (9, 10) sees o1 and o4. **Rows 0, 1, 5-8, 11-14, 17, 18 have an empty neighbor row**, so the neighbor sub-layer adds 0 to them.

The same seed with w=2 instead of 1 also reaches o0 and p0. It then hits L=24 **in the middle of row c1**: only `country of customers` is emitted, `age` and `is_premium` are cut. Its bucket also contained Val row #17 (c1, same timestamp 2024-03-01, label visible), which is the same-horizon case from §2.5.

### 6.4 How information can reach the masked cell (teaching interpretation, not a paper claim)

This follows from the masks.

- **Block 1, column layer**: the target directly reads the values of every visible label cell of the task (self and other entities). It cannot yet tell which belong to the same entity.
- **Feature layer**: the target and each past task row read their parent entity's cells, so all of them now carry information about "which entity am I about".
- **Neighbor layer**: entity cells read their children: the task rows with labels, plus transactions. This is how per-entity aggregates form.
- **Full layer**: unrestricted mixing.
- Stacking 12 blocks composes these hops. The paper's App. E (p.19-20) discusses expressivity:
  - column attention can be emulated by full attention;
  - feature attention is the only mechanism that explicitly groups cells into rows;
  - neighbor attention can be simulated by feature + full;
  - without full attention information moves one hop per layer.

### 6.5 No positional encodings, and permutation invariance

- "RT does not incorporate positional encodings … Relational Attention layers … are invariant to the ordering of tables, rows and columns" (paper p.5 §3.1).
- In code, `x` depends only on value, type and column-name embeddings (`model.py:208-224`), and the masks depend only on ids (`model.py:165-191`). **Permuting the token order permutes the outputs identically** (permutation equivariance; the prediction at the target is invariant).
- Order matters only inside the sampler (which rows make it in, and truncation).
- Paper App. E: "the architecture is invariant to permutations of rows, columns, and tables" (p.19).

### 6.6 History of the attention equation and order

- arXiv v1/v2 wrote `Softmax((QKᵀ ⊙ M)/√d_K)` (element-wise product). v3 corrected this to the −∞ masking above (v3 p.5). The project page still shows the old form (https://star-project.stanford.edu/rt/).
- Block order: `feat, nbr, col, full` at the ICLR submission (`eeccc2b`), then `col, feat, nbr, full` from `7f891d1` on. That matches Alg. 2 (p.26), paper Fig. 1 (p.2) and the slides.
- Table 15 (p.24) shows the order barely matters (§11).

---

## 7. The RT block, depth, final norm, parameters

### 7.1 Block (`model.py:62-86`)

```
for l in ["col", "feat", "nbr", "full"]:
    x = x + MHA_l( RMSNorm_l(x), mask_l )        # pre-norm residual, 4 distinct attention modules
x = x + FFN( RMSNorm_ffn(x) )
FFN(h) = W2 · ( SiLU(W1·h) ⊙ (W3·h) )             # W1,W3: 256→1024, W2: 1024→256, no biases (model.py:50-59)
```

- **5 RMSNorms per block** (feat, nbr, col, full, ffn), each with its own weight γ ∈ R^256, initialized to 1 (`model.py:71-73`).
- **12 blocks** (`example_pretrain.py:32`; paper p.6 §4.1 "12 layer transformer with hidden dimension 256 and 8 attention heads … gated MLPs with SiLU … hidden dimension 1024").
- `d_ff = 1024 = 4·d`. LLaMA's 2/3 sizing is not used.
- After the last block: `x = RMSNorm_out(x)` (`model.py:150, 229`).
- **Paper Alg. 2 (p.26) writes post-sublayer norm** `X ← X + Norm(MHA(X; M))`. The **code is pre-norm** `X + MHA(Norm(X))`, consistent with "follow the design choices of LLaMA" (p.5 §3.2), since LLaMA is pre-norm (Touvron et al. 2023 §2.2, https://arxiv.org/abs/2302.13971). **Implement and show pre-norm**, and footnote Alg. 2 (§10 #1).
- There is no dropout, no attention bias, no positional encoding, no QK-norm and no attention-output gate (`PE/rt/model.py`). The only multiplicative gate is SwiGLU's W3 branch inside the FFN (`model.py:59` `self.w2(F.silu(self.w1(x)) * self.w3(x))`; paper p.6 §4.1 "We use gated MLPs with SiLU activation, as found in the LLaMA architecture"). Do not import RT-J features such as its sigmoid attention-output gate (Appendix C).

### 7.2 RMSNorm

- `nn.RMSNorm(256)` with defaults: `y_i = x_i / sqrt(mean_j x_j² + ε) · γ_i` (https://docs.pytorch.org/docs/2.8/generated/torch.nn.RMSNorm.html).
- **ε at the pinned torch 2.6.0 with bf16 tensors = 2⁻⁷ = 0.0078125.** In 2.6.0, `rms_norm` is `CompositeImplicitAutograd` (`native_functions.yaml`). Its C++ body sets `eps_val = std::numeric_limits<at::scalar_value_type<scalar_t>::type>::epsilon()`, which is bf16 epsilon for bf16 input (pytorch v2.6.0 `aten/src/ATen/native/layer_norm.cpp:287-292`, https://github.com/pytorch/pytorch/blob/v2.6.0/aten/src/ATen/native/layer_norm.cpp). The model and inputs are bf16 (`main.py:186-187`).
- **The same ε applies under `torch.compile`** (`main.py:209`, `compile_=True`). torch 2.6.0 registers no Python decomposition or ref for `aten.rms_norm`: grep finds no `rms_norm` in v2.6.0 `torch/_decomp/decompositions.py`, `torch/_refs/__init__.py`, `torch/_refs/nn/functional/__init__.py` or `torch/_inductor/decomposition.py`. So the `CompositeImplicitAutograd: rms_norm_symint` C++ body (`native_functions.yaml:3306-3308`; `layer_norm.cpp:287-303`) is traced. `nn.RMSNorm` defaults to `eps=None` and calls `F.rms_norm` → `torch.rms_norm(input, normalized_shape, weight, eps)` (`torch/nn/functional.py:2915-2929`).
  - Order inside that body: input upcast to fp32 (`toOpMathType`, `layer_norm.cpp:295`), normalized with ε = 2⁻⁷ added in fp32, cast back to bf16 (`.type_as(input)`, `:299`), then multiplied by γ in bf16 (`result.mul(weight)`, `:302`).
- Newer torch (e.g. 2.12) uses fp32 epsilon 1.19e-7 for bf16 inputs [verified locally]. For a float64 TS simulation, use `eps = 2**-7` to mirror the paper environment. The difference matters mostly at init, e.g. the column-name projection has mean(x²) ≈ 2e-3 < 2⁻⁷.
- `RMSNorm(0⃗) = 0⃗`, so padding stays exactly zero through every block (attention rows fully masked → 0; FFN(0) = 0).

### 7.3 Parameter count: 22,336,131

The total is the sum of the module shapes in `model.py`. The paper says "about 22M" (p.6) and "22.3M" (p.26 App. I.4).

| component | params |
|---|---|
| per block: 4 attentions × 4 × 256² (no bias) | 1,048,576 |
| per block: SwiGLU 3 × 256 × 1024 (no bias) | 786,432 |
| per block: 5 RMSNorm × 256 | 1,280 |
| 12 blocks | 22,035,456 |
| encoders: 3 × Linear(1,256)+b = 1,536; 2 × Linear(384,256)+b = 197,120 | 198,656 |
| decoders: 3 × Linear(256,1)+b = 771; Linear(256,384)+b = 98,688 | 99,459 |
| 5 input RMSNorms | 1,280 |
| 4 mask embeddings | 1,024 |
| norm_out | 256 |
| **total** | **22,336,131** |

The frozen MiniLM (~33.4M) is not part of the model; embeddings are precomputed.

### 7.4 Initialization

- PyTorch defaults throughout. `nn.Linear` weights and biases ~ U(−1/√fan_in, 1/√fan_in); RMSNorm γ = 1; `m_t ~ N(0, I)` (`model.py:141-146`). There is no custom init in PE.
- The torch seed is set by `seed_everything(seed=0)` (`main.py:21-29, 98`).

---

## 8. Decoders, losses, predictions, metrics

### 8.1 Decoders (`model.py:124-131, 238-239`)

- `number`, `datetime`, `boolean`: `Linear(256, 1, bias=True)`. `text`: `Linear(256, 384, bias=True)`.
- They are applied to `RMSNorm_out(x)` at **every** position, but only the masked position is used.
- "The decoder to select for final prediction depends on the task type, or equivalently on the datatype of the masked cell" (paper p.6 §3.3).
- **One encoder and one decoder per data type, shared by all columns, tables and databases** (`model.py:115-131`: one `nn.Linear` per type key in `enc_dict`/`dec_dict`). There are no per-column heads. `yhat = self.dec_dict[t](x)` is applied to all positions regardless of column (`model.py:239`). So the number head emits one scalar that is read as a z-score in the target column's own (μ_c, σ_c) (each numeric column is normalized with its own stats, `pre.rs:569`), and §8.4 de-normalizes it with that column's Train stats.
- All four decoders run on every position. Any decoder whose type has no masked cell in the current batch gets a zero-weight "touch" (`loss_out += yhat.sum() * 0.0`), so DDP sees no unused parameters, and its `yhat_out[t]` is still filled (`model.py:238-248`). For example, a batch whose targets are all boolean touches the number and datetime decoders too. The text decoder is the only one that is **always** in that case, because text is never masked, so it is never trained.

### 8.2 Loss (`model.py:231-265`)

```
loss = Σ_i [masks_i] · ℓ_i  /  Σ_i masks_i                  (sum over the whole per-GPU batch, B·S positions)
ℓ_i = Huber_δ=1( ŷ_i^number,   y_i^number )      if sem_i = 0
      Huber_δ=1( ŷ_i^datetime, y_i^datetime )    if sem_i = 2
      BCEWithLogits( ŷ_i^boolean, 1{y_i^boolean > 0} )   if sem_i = 3
      error "masking text not supported"          if sem_i = 1
Huber_1(d) = 0.5 d²  if |d| < 1  else |d| − 0.5            (F.huber_loss default δ = 1.0)
BCEWithLogits(z, y) = max(z,0) − z·y + log(1 + e^{−|z|})
```

- Targets y are the **z-scored** values from the batch, bf16-rounded. Huber operates in z-space.
- One masked cell per sequence, so the loss is the plain mean over the 32 sequences of a GPU batch. DDP averages gradients over 8 GPUs, giving the mean over 256.
- **There is no per-task, per-type or class weighting**, and no label smoothing (`main.py:394-424`).
- Paper: "HuberLoss(r, r′) for regression and binary cross-entropy loss BCE(1{r > 0}, r′) … The overall loss is the mean over all masked cells in the batch. This formulation is used in both pretraining and fine-tuning" (p.6 §3.3). The paper does not mention datetime; the code applies Huber to it.
- [verified locally: `F.huber_loss(3,0) = 2.5` ⇒ δ=1; `BCEWithLogits(0,1) = ln 2`.]

### 8.3 Reading predictions (`main.py:283-293`)

- Binary task: `ŷ = yhat["boolean"][is_targets]`, a **raw logit**. Label: `y = boolean_values[is_targets]`, then `int(y > 0)`.
- Regression task: `ŷ = yhat["number"][is_targets]`, a **z-score**. Label: `number_values[is_targets]`.
- Which head to read is decided in eval by a **hard-coded list of regression task names** (`main.py:228-241`); everything else is read as classification.

### 8.4 Post-processing for display

- **Probability** = σ(logit). The authors' later leaderboard script applies `torch.sigmoid` (`git show rt-v1:scripts/predict_stage1.py:99`).
- **Regression in original units** = `ŷ·σ_train + μ_train`, with the Train-split label stats (sample std, ddof=1). This is what the authors do in `rt-v1:scripts/predict_stage2.py:44-68`.

### 8.5 Metrics (`main.py:328-337`)

- **AUROC** = `sklearn.roc_auc_score(int(y>0), logit)`. It is computed on logits with **no sigmoid**; this is identical to AUROC on probabilities because σ is monotone.
- **R²** = `sklearn.r2_score(y_z, ŷ_z)` in z-space. It equals raw-unit R², since both sums of squares scale by 1/σ². ȳ is the mean of the evaluated labels, so "Global mean baseline is 0.0" (Table 2 caption p.8).
- There is **no ensembling**: one context and one forward pass per seed ("a single forward pass", abstract p.1). Context-seed ensembling exists only in RT-J (MAIN `0485763`).

### 8.6 Evaluation protocol (`main.py:146-168, 219-344`)

- **Data and contexts**:
  - Eval datasets are built with `seed=0` and `shuffle_py(0)`, so the item order and every BFS draw are **fixed across evaluations** (`main.py:158-160`).
  - At most `max_eval_steps = 40` batches per rank (`main.py:299-300`; scripts set 40). With 32 × 8 GPUs that is **≤ 10,240 seeds per split per task**, a fixed random subset. The paper does not state this for RT.
- **Wrapped duplicates in the last batch** are neutralized by `masks = is_targets = False` and `is_padding = True` for rows ≥ `true_batch_size` (`main.py:277-279`). **This fails under multi-rank DDP when the usize subtraction underflows.**
  - `true_batch_size = B.min(N − k·B − b·B·W)` is computed in `usize` (`fly.rs:337-341`). The extension is built in release mode (`data.py:4` `MaturinSettings(release=True)`; README:31 `maturin develop --uv --release`). `rustler/Cargo.toml` sets no `overflow-checks`, and Cargo's release default is off. So on the last batch, any rank whose start index `b·B·W + k·B` exceeds N wraps to a huge number, and `min` gives `true_batch_size = 32`.
  - That rank's 32 wrapped sequences (`j % N`, `fly.rs:347-350`) are not neutralized and enter AUROC/R² a second time. `assert yhat.size(0) == true_batch_size` (`main.py:290`) still passes.
  - This affects small val/test splits that the eval reaches in full (N ≤ 10,240 with 8 ranks). With B=32, W=8: N=300 adds 192 duplicates, N=522 adds 224, N=700 adds 64, N=1000 adds 0 [checked by hand from `fly.rs:333-341`].
  - A single-process demo (W=1) is unaffected. Do not claim exact de-duplication for the paper's numbers.
- **Forward**: `net.eval()` + `torch.inference_mode()`. The math is the same as training (no dropout).
- **Eval cadence**: step 0 and every `eval_freq` steps, plus powers of two when `eval_pow2` (`main.py:373-375`).

---

## 9. Training regimes

### 9.1 Pretraining: leave-one-DB-out

- "We pretrain separately for each target dataset on all tasks from all other datasets" (p.6 §4.1).
  - Script for target rel-amazon: `train_tasks = [t for t in all_tasks if t[0] != "rel-amazon"]` (forecast + autocomplete of the 6 other DBs; forecast tasks: Train split of their task table; autocomplete tasks: **all rows of the DB table** via the `:Db` key, `data.py:77-81`); `eval_tasks` = rel-amazon's forecasting tasks, val and test (`example_pretrain.py:17-18`). The training dataset is built at `main.py:123-136`, the eval loaders at `main.py:146-168`.
  - In code this is **29-33 training tasks per run**: `tasks.py` defines 36 tasks (12 forecast clf + 9 forecast reg + 7 autocomplete clf + 8 autocomplete reg; per DB: amazon 6, hm 3, stack 4, avito 4, event 5, trial 7, f1 7). Holding one DB out leaves amazon 30, hm 33, stack 32, avito 32, trial 29, f1 29. The slides' "~20 tasks from 6 DBs" is approximate.
  - rel-event tasks are in the pretraining mix, but rel-event is never evaluated, "as it has been found to have temporal leakage issues" (p.6).
- **What is masked**: one target cell per sequence (§4.10). The input is the BFS context around the seed, including in-context labels.
- **Task mixing**: one shuffled item list over all train tasks. Proportional to seed counts, no balancing (§4.14). The slides say "~20 tasks from 6 DBs … 22M parameters, 13B tokens, 2 hrs on 8xA100" (https://iclr.cc/media/iclr-2026/Slides/10007116.pdf).
- **Checkpoint selection, per target task**: at every eval, for each (db, task), if val > best val, store the **test metric of that same step** and save `<db>_<task>_best.pt` (`main.py:377-392`). The printed `best_test_metrics` are the reported numbers (`main.py:441-448`). "We select the best validation checkpoint separately for each target task as we found significant overfitting during pretraining" (p.6). **Target-task validation labels are used for model selection.**

### 9.2 Hyperparameters (scripts; matches paper p.6 unless noted)

| | pretrain (`example_pretrain.py`) | continued pretrain (`example_contd_pretrain.py`) | fine-tune (`example_finetune.py`) |
|---|---|---|---|
| init | PyTorch default | leave-one-DB-out best ckpt for the target task | leave-one-DB-out ckpt (paper; see note) |
| train tasks | all tasks of other 6 DBs | target DB's other tasks (forecast + autocomplete), target task excluded (`:17-19`) | target task only (`:18`) |
| optimizer | AdamW(β=(0.9,0.999), eps=1e-8, fused), wd 0.1 | AdamW, wd 0 | AdamW, wd 0 |
| LR | peak 1e-3, OneCycleLR(pct_start=0.2, linear) | 1e-4 constant | 1e-4 constant |
| steps | 50,001 | 4,097 (2¹²+1) | 32,769 (2¹⁵+1) ("33k", p.6) |
| batch | 32/GPU × 8 = 256 | 256 | 256 |
| L, w | 1024, 256 | 1024, 256 | 1024, 256 |
| grad clip | global L2 norm 1.0 | 1.0 | 1.0 |
| eval | every 1000 + powers of 2 | every 100 + powers of 2 | every 1000 (`eval_pow2=False`) |
| time (8×A100) | ≈2 h | ≈15 min | ≈1.5 h (`PE/README.md:126-145`) |

Optimizer and schedule details:

- **Exact LR schedule** (OneCycleLR defaults `div_factor=25`, `final_div_factor=1e4`, `cycle_momentum=True`; `main.py:197-204`) [verified locally]:

  | step | 0 | 1k | 5k | ≈10k | 20k | 30k | 40k | 50k |
  |---|---|---|---|---|---|---|---|---|
  | lr | 4.0e-5 | 1.36e-4 | 5.2e-4 | 1.0e-3 | 7.5e-4 | 5.0e-4 | 2.5e-4 | 4.0e-9 |
  | β1 | 0.95 | 0.94 | 0.90 | 0.85 | 0.875 | 0.90 | 0.925 | 0.95 |

  So the warmup starts at 4e-5, not 0. The decay ends at 4e-9. β1 is cycled, overriding the coded 0.9. The paper says "linear warmup from zero for the first 20% … linear decay to zero" (p.6) (§10 #13).
- **Weight decay applies to all parameters**: one param group, including norms, biases and mask embeddings (`main.py:188-195`).
- Train step order (`main.py:394-424`): forward → `zero_grad` → backward → total grad norm → clip to 1.0 → `opt.step()` → `lrs.step()`.
- **Precision**: the whole model is cast to bf16 (`main.py:186-187`). There is no autocast, no fp32 master weights and no GradScaler. AdamW states are in the parameter dtype. Model and FlexAttention are `torch.compile`d (`model.py:17`; `main.py:209`). The paper says "8×A100 GPUs at BFloat16 precision … 8 batches/second or 2M tokens/second" (p.6).
- **Fine-tuning init caveat**: the script's `load_ckpt_path` points to `ckpts/rel-amazon_user-churn/...`, which no script writes; the commented alternative is the pretrain checkpoint (`example_finetune.py:12-13`). Fig. 3's caption ("Target datasets and tasks are unseen during pretraining", p.7) and the rt-v1 README ("Cells without a file here are the paper's own pretrain-init fine-tuning numbers") point to **pretrain init** for the paper.
  - The script also has `save_ckpt_dir=None`, so no best checkpoint is saved; selection is done from logs.
  - **History of the init path.** The ICLR-submission script fine-tuned from the **continued-pretraining** checkpoint: `eeccc2b:scripts/fine_tune.py:12` `load_ckpt_path="ckpts/contd_rel-amazon_user-churn/rel-amazon_user-churn_val_best.pt"`, which is exactly what the `eeccc2b` contd script wrote (`save_ckpt_dir="ckpts/contd_rel-amazon_user-churn/"`, `{db}_{table}_{split}_best.pt` naming). README:140-143 still reads that way ("Finetune from best checkpoint obtained above", directly after the continued-pretrain bullet). `f4d440c` (2025-10-20) switched to `ckpts/rel-amazon_user-churn/..._val_best.pt`, and `23ede2e` (2025-11-21) to the current `..._best.pt`. No script writes either path. The evidence on the paper's init is therefore mixed. Fig. 3's caption still favours pretrain init for Fig. 3.
  - The quoted rt-v1 README sentence (`rt-v1:README.md:93`) is about **RelBench-leaderboard** cells. The HF `finetune-from-*` checkpoints (13 `finetune-from-contd-pretrain_*`, 4 `finetune-from-pretrain_*`; https://huggingface.co/api/models/stanford-star/rt-v1) follow a leaderboard protocol: best of two inits, val-nMAE selection for regression, full official test split. Do not present them as the paper's fine-tuned models.
  - **Fresh optimizer at every stage.** Checkpoints hold only the model `state_dict` (`main.py:356-357`). Loading restores weights, and a new AdamW is then built (`main.py:177-195`). Continued pretraining and fine-tuning therefore start with fresh AdamW moments and a constant LR of 1e-4 from step 0 with no warmup (`lr_schedule=False`, `example_contd_pretrain.py:27`, `example_finetune.py:26`).
  - Paper: "For all methods, the best checkpoint is selected based on validation set performance" (p.20 App. F.1).
- **"Untrained RT"** = the same fine-tuning with random init (`main.py:177-180`).

### 9.3 Continued pretraining ("Yes" columns of Tables 1/2)

- Starts from the leave-one-DB-out checkpoint selected for the target task. It trains on the target DB's other tasks, including autocomplete over **all** rows of those tables (no cutoff, §2.4).
- It evaluates the still-unseen target task (`example_contd_pretrain.py:12-20`).
- **Step 0 is a selection candidate.** The loop evaluates at `steps = 0`, before any update (`main.py:373-375`; `example_contd_pretrain.py:9-10` `eval_freq=100, eval_pow2=True`), and best-val starts at `-inf` (`main.py:380-385`). That step-0 model is the unchanged leave-one-DB-out checkpoint, evaluated on the same fixed contexts. If continued pretraining never improves target-task val, RT(Yes) therefore equals RT(No). Table 1 (p.8) shows this for rel-f1 driver-dnf (81.2 / 81.2) and driver-top3 (89.3 / 89.3).
- The target task's **train rows** can still appear in its val/test contexts as in-context labels (§2.5).

### 9.4 Zero-shot inference: what actually happens

1. Build the target DB (preprocessing as in §3), including the task table with its train/val/test splits.
2. For each val/test seed row of the target task, sample a context (§4). Same code, epoch=0, seed=0.
3. Run a forward pass with frozen weights. The masked cell is the seed's label. Past labels of the same task (train rows; val rows for test seeds; same-horizon rows of other entities) are visible.
4. Read the boolean logit or number z-score at the masked position (§8.3).
5. "Zero-shot" = **no gradient updates on the target DB ("No") or target task ("Yes")**. Checkpoint selection uses target-task val labels (§9.1).
6. Definition in the paper: "predicting new targets on a new RDB with a new schema, without weight updates. Information about the new RDB is available only at inference time, solely through the context window" (p.4 §2.3).

### 9.5 RT as feature extractor (Table 3, p.8 §4.4)

- "We use the masked token representation before the decoder layer from a pretrained RT and fine-tune (FT) only a 2-layer MLP over it."
- In code terms: the 256-d output of `norm_out` at the target position (`model.py:229, 239`). RT-J exposes this as `return_embeddings=True` (MAIN `src/rt/model/net.py:450-454`).
- **This path is not in the paper-era code**, and the MLP details are unspecified.

---

## 10. Paper-vs-code discrepancies and known caveats

| # | topic | paper says | paper-era code does | what to show |
|---|---|---|---|---|
| 1 | block normalization | `X ← X + Norm(MHA(X;M))` (Alg. 2 p.26) | pre-norm `x + MHA(RMSNorm(x))` (`model.py:83-85`) | code; footnote Alg. 2 |
| 2 | token embedding | `x = W_d r + W E_schema(c,t)` (p.5) | bias + **RMSNorm after every projection**; raw `m_t` for masked (`model.py:115-146, 208-224`) | code |
| 3 | normalization stats | "computed on the training split" (p.4-5) | task labels: Train ✔; **DB tables: all rows**; datetime: global over all tables and splits, population std (`pre.rs:316-396`); later called leaky and fixed in RT-J (MAIN `358bfd3`, `5fef314`) | code; optional "RT-J fix" toggle |
| 4 | datetime unit | seconds (p.5) | nanoseconds (`pre.rs:586`) | identical z-score; mention only |
| 5 | F→P selection | `RandomSelect(R)` (Alg. 1) | LIFO stack, deterministic (`fly.rs:378-379`) | code |
| 6 | P→F selection | min HopDistance | min depth label of the frontier entry (`fly.rs:381-396`) | code |
| 7 | width bound | all P→F neighbours (Alg. 1) | DB children only; task-table children never subsampled (`fly.rs:426-452`) | code |
| 8 | row completeness | "all of its non-missing feature cells are added" (p.4) | hard stop at L, may cut a row (`fly.rs:502-509`) | code |
| 9 | temporal filter | "rows with timestamps later than the seed row are excluded" (p.4) | only P→F children; children without ts always pass; seed without ts → no filter; parents never filtered (`fly.rs:406-424`) | code |
| 10 | DB cutoff | "database rows only up to the training cutoff timestamps" (p.6) | no cutoff anywhere; autocomplete seeds from all periods (`data.py:77-81`) | code; note |
| 11 | same-horizon labels | not discussed | other entities' task rows at the seed's timestamp are visible (`fly.rs:422`); excluded in `eeccc2b` and in RT-J | code; highlight |
| 12 | column mask | `Col(k)=Col(q)` | same col idx AND same table idx (`model.py:181-183`) | equivalent |
| 13 | LR schedule | warmup from 0, decay to 0 (p.6) | 4e-5 → 1e-3 → 4e-9; β1 cycled 0.95↔0.85; grad clip 1.0 (not in paper) | code |
| 14 | masking | MTP; Table 14 masks extra cells | only the target cell; P(mask) sampler not released (`fly.rs:492-496`) | target only |
| 15 | masked types | "only mask cells in boolean or numeric columns" (p.6) | datetime would work (Huber); text raises (`model.py:250-258`) | as code |
| 16 | text encoder | "MiniLMv2 [41]" (p.6) | `sentence-transformers/all-MiniLM-L12-v2` (MiniLM-L12-H384 base), fp32 at paper time, stored bf16 (`embed.py`) | name the real model |
| 17 | feature columns | numeric, text, datetime (p.3) | + Boolean sem type (`common.rs:38-44`) | 4 types |
| 18 | eval set | "test set performance" | fixed ≤10,240-seed subset per split (`main.py:299-300`); under 8-rank DDP, on the last batch any rank whose start index exceeds N gets a wrapped (usize-underflowed) `true_batch_size` of 32, so its 32 wrapped duplicates are counted twice for small splits (`fly.rs:337-341`; §8.6) | note; single-process demo unaffected |
| 19 | regression decoder | numeric only (p.6) | Huber also for datetime | note |
| 20 | precision | "BFloat16 precision" | pure bf16 weights, inputs, optimizer state; RMSNorm ε = 2⁻⁷ (torch 2.6) | note |
| 21 | fine-tune script | 33k steps, best val ckpt | 32,769 steps; saves nothing; `eval_pow2=False` cannot give Fig. 3's log-x points; init path not produced by other scripts (`example_finetune.py`) | follow paper text |
| 22 | user-attendance | (not evaluated) | listed as regression, cast to Boolean, trained with BCE (`tasks.py:25`; `pre.rs:272-278`) | irrelevant for demo |
| 23 | attention order | col→feat→nbr→full (Alg. 2) | same since `7f891d1`; the ICLR submission used feat→nbr→col→full | show col→feat→nbr→full |
| 24 | attention equation | v1/v2: `⊙ M`; v3: −∞ masking | −∞ semantics via FlexAttention | v3 |
| 25 | headline numbers | Table 1 RT "No" mean 69.7 | ablation "none" columns say 70.1 (Tables 4/11/13/14) and Table 15 CFN 70.0. Ablation "none" = the v1-era main results (arXiv v1, 2025-10-07, before the sampler fixes); headline Tables 1/2 were re-run after the sampler fixes (§13 #1) | quote Tables 1/2 for headlines |
| 26 | "93%" | abstract "averaging 93% of fully supervised AUROC" | = continued-pretraining 71.9/77.2. Unseen-DB zero-shot is 69.7/77.2 = 90.3% (p.2) | be precise |

---

## 11. Ablations and findings worth teaching (paper numbers)

Provenance (§13 #1): Tables 4, 5, 10, 11, 12, 13 and 14 are unchanged since arXiv v1 (2025-10-07), so they predate the released sampler and attention order. Tables 1 (re-run for v2), 2, 3 and 15 (v3) are later.

### 11.1 Headline zero-shot results

**Table 1** (p.8): zero-shot test AUROC (%) over 10 binary tasks. Random = 50.

| | Gemma3 4B | 12B | 27B | EntityMean | Griffin (No) | **RT (No)** | RelLLM (Yes) | Griffin (Yes) | **RT (Yes)** |
|---|---|---|---|---|---|---|---|---|---|
| mean | 63.3 | 62.4 | 64.6 | 66.7 | 64.0 | **69.7** | 63.2 | 67.3 | **71.9** |

RT (No) per task: amazon item-churn 70.9, user-churn 64.0; avito user-clicks 59.5, user-visits 61.8; f1 driver-dnf 81.2, driver-top3 89.3; hm user-churn 62.8; stack user-badge 80.1, user-engage 75.7; trial study-out 51.8.

**Table 2** (p.8): zero-shot R² (%) over 8 regression tasks.

| | EntityMean | Griffin (No) | RT (No) | Griffin (Yes) | RT (Yes) |
|---|---|---|---|---|---|
| mean | 19.3 | 8.9 | 22.7 | 10.1 | 26.4 |

EntityMean beats RT on item-ltv (54.2 vs 32.5) and post-votes (43.7 vs 33.9). LLMs score R² < −9 on most regression tasks (Table 16 p.26).

Context for these numbers:

- "Zero-shot … 90.3% of full supervised learning AUROC, and rises to 93.1% with continued pretraining … Gemma3-27B achieves only 83.7% … despite taking 10⁵× more inference FLOPs" (p.2-3).
- Supervised RT (Table 10, p.21): 77.2 AUROC, 33.2 R².

### 11.2 Context ablations (Table 4 p.9; per task in Table 11 p.22)

| ablated | zero-shot AUROC | zero-shot R² | fine-tuned AUROC | fine-tuned R² |
|---|---|---|---|---|
| none | 70.1 | 22.8 | 77.2 | 33.2 |
| col/table names shuffled | 69.5 | 20.5 | 77.5 | 33.2 |
| **self labels removed** | **53.8** | **−5.5** | 77.1 | 26.7 |
| other-entity labels removed | 70.6 | 22.9 | 77.4 | 31.0 |

- "Zero-shot transfer is driven primarily by the model's ability to leverage past labels of the target entity" (p.9 §5.1). RT learns more than averaging, since it beats EntityMean (p.9).
- Fig. 5 (p.9): pretrained RT still transfers positively without self labels.
- After full fine-tuning, context ablations matter little, except that self labels still help regression (p.9).

### 11.3 In-context label counts (Table 12 p.23)

Mean ± sd per 1024-cell context, as target-entity labels / other-entity labels:

- f1 driver-dnf 19±14 / 0±0
- f1 driver-top3 17±11 / 0
- hm item-sales 39±13 / 0±3
- stack user-engagement 16±10 / 10±10
- stack post-votes 16±10 / 4±14
- amazon user-churn 4±4 / 1±5
- **trial study-outcome 0±0 / 0±0** (zero-shot AUROC there is ≈ random: 51.8)
- **trial study-adverse 0 / 0** (R² 2.6)

**Provenance.** Table 12 is unchanged since arXiv v1 (2025-10-07; v1 Table 9 has identical rows, e.g. "rel-stack user-engagement 16 ± 10 10 ± 10 3±1", "rel-hm item-sales 39 ± 13 0±3 1±1"). It was therefore sampled before `66c4247` (2025-10-09). At that date the public sampler excluded task rows with ts == seed_ts (`eeccc2b:rustler/src/fly.rs:428-431` explicit `==` exclusion; `7f891d1` `>=`). The released sampler keeps them (`fly.rs:422`, `>`), so same-horizon task rows of other entities are included (§2.5) and other-entity counts from a faithful simulation can be higher. Show Table 12 as the paper's numbers, not as the released sampler's expected output.

Labels refer to "unmasked cells from the target column" (caption). Third-party analysis (OpenRFM, arXiv 2606.04320, https://arxiv.org/abs/2606.04320) frames this as "label reachability". That is their interpretation, not the authors'.

### 11.4 Attention-layer ablations (Table 5 p.10, R²; Table 13 p.23, AUROC)

Parameter count is kept constant by adding blocks.

| removed | zero-shot R² | FT R² | zero-shot AUROC | FT AUROC |
|---|---|---|---|---|
| none | 22.8 | 33.2 | 70.1 | 77.2 |
| column | **11.9** | 31.2 | 69.8 | 77.3 |
| feature | 18.8 | 32.2 | 70.6 | 76.7 |
| neighbor | 22.0 | 31.4 | 70.8 | 76.5 |
| full | 23.7 | 32.7 | 71.1 | 77.0 |

- "Column attention has the highest impact on zero-shot transfer … on fine-tuning … feature and neighbor attention is more significant. Removing full attention has the least impact" (p.9 §5.2).
- Classification differences are minor (p.9). The slides add: "Column attention helps to adapt to new column distributions; Feature attention helps to adapt to new entity types."

### 11.5 Attention order (Table 15 p.24, zero-shot)

| order | CFN (released) | CNF | FCN | FNC | NCF | NFC | Parallel (sum) |
|---|---|---|---|---|---|---|---|
| AUROC | 70.0 | 70.6 | 70.0 | 69.8 | 70.2 | 69.5 | 69.3 |
| R² | 23.0 | 22.5 | 23.2 | 22.5 | 22.1 | 23.0 | 23.6 |

"The specific order of relational attention layers is not critical" (p.23 App. H.1).

### 11.6 Multi-cell masking during pretraining (Table 14 p.24; ablation, not the default)

| P(mask) | 0.0 (default) | 0.2 | 0.4 | no pretraining |
|---|---|---|---|---|
| zero-shot AUROC / R² | 70.1 / 22.8 | 69.8 / 16.9 | 69.3 / 15.3 | — |
| fine-tuned AUROC / R² | 77.2 / 33.2 | 77.3 / 35.6 | 76.7 / 34.4 | 76.1 / 28.4 |

### 11.7 Learning efficiency, feature extraction, cost

- **Fig. 3 (p.7)**: pretrained RT is best overall during fine-tuning.
  - There is a small dip in the earliest fine-tuning steps, conjectured to be a transition "from a zero-shot, in-context-labels- and semantics-driven regime to a data-driven prediction regime".
  - Untrained RT overtakes RDL-GNN after ~2 (clf) / ≈100 (reg) steps (p.7 §4.2).
  - Pretrained RT reaches the second-best baseline's performance in "10–100× fewer steps" (p.3).
- **Table 3 (p.8)**: a 2-layer MLP on frozen RT(P) embeddings reaches 79.1 AUROC / 46.2 R², vs full FT 80.7 / 41.9, untrained RT 78.0 / 32.5 and untrained GNN 77.9 / 10.9.
  - **Task subset.** Table 3 averages over only 5 classification tasks (amazon item-churn, user-churn; hm user-churn; stack user-badge, user-engagement) and 4 regression tasks (amazon item-ltv, user-ltv; hm item-sales; stack post-votes) (p.8).
  - Fig. 3 likewise averages over 5 clf / 4 reg tasks "which do not show overfitting" (titles and caption, p.7).
  - These are not the 10/8 tasks of Tables 1/2/10, so do not compare "Full FT 80.7 / 41.9" directly with Table 10's 77.2 / 33.2.
- **Fig. 4 (p.8-9)**: zero-shot quality degrades gracefully with shorter inference contexts (256/512/768/1024): "a drop of only 4% absolute AUROC and 7% absolute R² with 25% lower time and 20% lower memory" (p.8 §4.5).
- **Table 10 (p.21)**: supervised fine-tuning.
  - AUROC: RT pretrained 77.2 (untrained 76.1) vs RelGNN 77.1, RelLLM 77.1, RelGT 76.2, RDL-GNN 75.4, Griffin 75.0 / 73.2.
  - R²: RT pretrained 33.2 (untrained 28.4) vs Griffin 22.6 / 19.5, RDL-GNN 14.3.
- **Ranked drivers of zero-shot transfer (p.10 §7)**:
  1. forecasting from the target entity's past task rows;
  2. column attention;
  3. feature attention (same and parent tables);
  4. schema semantics from table/column names;
  5. in-context learning from other entities' task rows.
- **Schema heterogeneity (App. C p.17-18)**: column-name Jaccard overlap across DBs is ≤ 4.3%. So transfer is not explained by lexical overlap.

---

## 12. Invariants a faithful simulation must respect

**Data and preprocessing**

- [ ] A token is a non-null, non-NaN cell of a **non-PK, non-FK** column. PK and FK columns never become tokens.
- [ ] FK values are parent **row positions**; every FK becomes a parent id in `f2p_nbr_idxs` (≤ 5 per row) and a P→F edge on the parent carrying the **child's** timestamp and table type.
- [ ] Four sem types with codes Number=0, Text=1, DateTime=2, Boolean=3. The time column is a DateTime **token**.
- [ ] Numbers: per-column z-score, sample std, σ=0 or undefined → 1. Booleans: per-column z-score of 0/1, so true > 0 and false < 0; **no σ=0 guard, so never use a constant boolean column**. Datetimes: **one global** z-score per DB with population std.
- [ ] DB-table stats over **all rows** (paper-era); task-table Val/Test reuse **Train** stats.
- [ ] Column phrase is exactly `"<col> of <table>"`. Task splits share table and column ids. The text encoder is all-MiniLM-L12-v2, mean-pooled and unit-normalized (384-d).
- [ ] `node.timestamp` exists only for rows that have ≥1 non-null FK, a declared time column, **and a non-null time value in that row** (`pre.rs:481-493`). It is whole seconds cast to i32 (wraps outside 1901-12-14 .. 2038-01-18, §1.4).

**Sampling**

- [ ] The seed's own row is emitted first. **Exactly one masked token**: the seed's target cell. Every other cell is visible, including other rows' labels.
- [ ] F→P parents: LIFO stack with absolute priority, never time-filtered, never subsampled.
- [ ] P→F children: skip other task tables; skip only if both timestamps exist and `child_ts > seed_ts` (equal timestamps kept).
- [ ] Width bound w applies to **DB children only**, after the time filter, uniformly without replacement. Task-row children are all kept.
- [ ] Next P→F row: uniform among the **lowest non-empty depth bucket**. The visited check happens at pop time.
- [ ] Leakage columns are dropped for the seed row and for rows with `ts == seed_ts` (null == null counts as equal).
- [ ] Stop at **exactly L tokens** (may cut a row) or when the frontier empties; pad the rest with the padding defaults.

**Model**

- [ ] No positional encoding. **Permuting tokens permutes outputs**; the target prediction is unchanged.
- [ ] Input = `RMSNorm(Lin(colname))` + (`RMSNorm(Lin_t(value))` or raw `m_t` if masked). Padding input = 0.
- [ ] Masks: col = same column & same table; feat = same row ∨ key row ∈ query's parents; nbr = query row ∈ key's parents; full = all. **All ANDed with non-padding on both sides.** Shared by all 8 heads.
- [ ] Softmax scale 1/√32. **A fully masked query row outputs 0** (no NaN, no uniform weights).
- [ ] Block = 4 pre-norm residual attentions in the order **col → feat → nbr → full**, then pre-norm SwiGLU (256→1024→256), no biases. 12 blocks, then final RMSNorm (ε = 2⁻⁷ at paper settings).
- [ ] Decoders are linear with bias. Read the output **only at the masked position**: boolean logit for binary tasks, number z-score for regression.

**Loss and metrics**

- [ ] Loss = Huber(δ=1) in z-space for number/datetime, BCE-with-logits against `1{z>0}` for boolean, averaged over masked cells. No weighting. Text targets are unsupported.
- [ ] AUROC on logits ≡ on probabilities, and R² in z-space ≡ raw units, **mathematically**. In the paper code both are computed on bf16 labels and bf16 predictions: labels are bf16-rounded z-scores (`fly.rs:482, 490`; `data.py:127`), predictions come from a pure-bf16 model (`main.py:186-187, 284-288`), and both are cast to float only after concatenation (`main.py:328-337`). So R² is computed against bf16-quantized z-labels (8 significant bits, about 2-3 decimal digits), and bf16 logits produce many tied scores, which `roc_auc_score` credits as half. Expect small differences in a float64 demo. Display uses σ(logit) and `ẑ·σ_train + μ_train`.

**Regimes**

- [ ] Zero-shot = no weight updates on the target DB/task. In-context labels (past task rows) are present. Checkpoints are chosen by target-task validation.
- [ ] Pretraining mixes tasks in proportion to seed counts, with one masked cell per sequence and a global batch of 256.

---

## 13. Open questions / not verifiable

1. **Which code produced the published numbers (now dated from the arXiv v1/v2/v3 PDFs).** arXiv v1 was posted Tue 7 Oct 2025 18:51 UTC, v2 Wed 22 Oct 2025, v3 Mon 2 Mar 2026 (arXiv abs page). The sampler/order fixes are `7f891d1` (10-08), `66c4247` (10-09) and `801a32b` (10-16).
   - **Tables 4, 5, 10, 11, 12, 13 and 14 are unchanged since v1**, where they were Tables 3, 4, 7, 8, 9, 10 and 11. v1's Table 1 RT(No) per-task column (70.2, 63.9, 59.5, 61.8, 82.0, 89.1, 62.8, 80.0, 77.1, 54.5; mean 70.1) is exactly the v3 Table 11 "none" column (p.22). These tables therefore predate `7f891d1`, `66c4247` and `801a32b`, and were not produced with the released sampler or attention order. The public code at that date (`eeccc2b`) used `feat→nbr→col→full` (`eeccc2b:rt/model.py:83`) and excluded same-timestamp task rows.
   - **Table 1 was re-run for v2**: RT No/Yes 70.1/72.9 → 69.7/71.9. v2 Table 1 equals v3 Table 1.
   - **Table 2 was re-run for v3**. The v1/v2 RT(No) column (33.2, 36.4, 4.5, 54.7, 14.0, 32.4, 5.2, 2.1; mean 22.8) equals the ablation "none" column (v3 Table 5, p.10). v3 Table 2 (p.8) has 32.5, 36.9, …, mean 22.7. RT(Yes) changed per task while its mean stayed 26.4.
   - **Tables 3, 6, 7 and 15 are new in v3** (no feature-extractor, Jaccard or attention-order table in v1/v2).
   - So only Tables 1 (v2), 2, 3 and 15 (v3) postdate `801a32b`. That they were produced with exactly `801a32b` is plausible but still not provable.
2. **w for the tables.** The scripts say 256 (`example_pretrain.py:21`). Fig. 6 (p.15) uses w = 128 together with L = 512, i.e. an illustration setting different from the experimental L = 1024 (p.6 "We pretrain RT for 50k steps at a context length of 1024"), so it is weak evidence for 128. The paper text gives no w for the experiments. (The HF RT-PluRel card, a different model, lists 128.)
3. **Ablation implementations are not released**: column/table-name shuffling, self/other label removal, P(mask) > 0 multi-cell masking, attention-layer removal with "increased number of blocks" (exact block counts unknown), the parallel-attention variant, and the Table 3 MLP probe (width, activation, training hyperparameters). Pre-submission `mask_prob` code (cosine-decayed, `6cae3ef`) exists, but whether Table 14 used it is unknown.
4. ~~RMSNorm ε under `torch.compile` at torch 2.6.~~ **Resolved from source**: v2.6.0 has no Python decomposition/ref for `rms_norm`, so the compiled graph traces the C++ composite and ε = 2⁻⁷ for bf16 (§7.2). Not executed on 2.6.
5. ~~FlexAttention fully-masked rows on the compiled Triton kernel in torch 2.6.~~ **Resolved from source**: v2.6.0 `torch/_inductor/kernel/flex_attention.py:422-427` sets `l_i = 1` for fully masked rows, so the output is 0; the eager path uses `_safe_softmax` (§6.1). Not executed on GPU.
6. **Dtypes of some autocomplete targets** (rel-avito `IsUserLoggedOn`, rel-event `not_interested`). They decide BCE vs Huber. Not checked against parquet files.
7. **Whether RelBench's downloaded DB is the full DB.** It contains post-test rows (needed for test labels). This follows from RelBench caching code, not checked byte-by-byte.
8. **OpenReview discussion** (submission 9833, forum `rpPtgMC5s9`). Reviews and rebuttal were inaccessible (challenge page). Any clarifications made there are unknown.
9. **Bit-exact contexts** need Rust's ChaCha12 `StdRng`, `random_range` and `index::sample`. These are not reproduced here and are unnecessary for teaching.
10. **Fig. 3's plotted curves** need eval at powers of 2, but `example_finetune.py` has `eval_pow2=False`. The exact fine-tuning configuration behind the figure is inferred, not verified.

---

## Appendix A: TS-ready model forward and loss (mirrors `model.py:153-267`)

```ts
const D = 256, H = 8, DH = 32, DFF = 1024, DTEXT = 384, NBLK = 12, EPS = 2 ** -7;
type Vec = Float64Array;
const rmsnorm = (x: Vec, g: Vec) => { let ms = 0; for (const v of x) ms += v * v; ms /= x.length;
  const r = 1 / Math.sqrt(ms + EPS); return x.map((v, i) => v * r * g[i]); };
const linear = (W: Vec[] /*out×in*/, x: Vec, b?: Vec) => Float64Array.from(W, (row, o) => dot(row, x) + (b ? b[o] : 0));
const silu = (z: number) => z / (1 + Math.exp(-z));

function buildMasks(tok: Token[]) {            // tok[i]: {node, f2p[5], tableIdx, colIdx, pad}
  const S = tok.length, M = { col: [], feat: [], nbr: [], full: [] } as Record<string, boolean[][]>;
  for (let q = 0; q < S; q++) for (const k of ['col','feat','nbr','full']) M[k][q] = new Array(S);
  for (let q = 0; q < S; q++) for (let k = 0; k < S; k++) {
    const pad = !tok[q].pad && !tok[k].pad;
    M.col[q][k]  = pad && tok[q].colIdx === tok[k].colIdx && tok[q].tableIdx === tok[k].tableIdx;
    M.feat[q][k] = pad && (tok[q].node === tok[k].node || tok[q].f2p.includes(tok[k].node));
    M.nbr[q][k]  = pad && tok[k].f2p.includes(tok[q].node);
    M.full[q][k] = pad;
  }
  return M;
}

function embed(t: Token, P: Params): Vec {
  if (t.pad) return new Float64Array(D);
  let x = rmsnorm(linear(P.enc.col_name.W, t.colNameEmb, P.enc.col_name.b), P.norm.col_name);
  const type = ['number', 'text', 'datetime', 'boolean'][t.sem];
  const v = type === 'text' ? t.textEmb : Float64Array.of(t[type]);      // scalar z-score as length-1 vector
  const term = t.mask ? P.maskEmb[type] : rmsnorm(linear(P.enc[type].W, v, P.enc[type].b), P.norm[type]);
  return add(x, term);
}

function mha(X: Vec[], M: boolean[][], A: AttnParams): Vec[] {
  const Q = X.map(x => linear(A.wq, x)), K = X.map(x => linear(A.wk, x)), V = X.map(x => linear(A.wv, x));
  return X.map((_, q) => {
    const o = new Float64Array(D);
    for (let h = 0; h < H; h++) {
      const lo = h * DH; const s: number[] = []; let mx = -Infinity;
      for (let k = 0; k < X.length; k++) { s[k] = M[q][k] ? dotRange(Q[q], K[k], lo, lo + DH) / Math.sqrt(DH) : -Infinity; mx = Math.max(mx, s[k]); }
      if (mx === -Infinity) continue;                                  // fully masked → head output 0
      let Z = 0; for (let k = 0; k < s.length; k++) { s[k] = Math.exp(s[k] - mx); Z += s[k]; }
      for (let k = 0; k < s.length; k++) if (s[k] > 0) for (let d = lo; d < lo + DH; d++) o[d] += (s[k] / Z) * V[k][d];
    }
    return linear(A.wo, o);
  });
}

function forward(tok: Token[], P: Params) {
  const M = buildMasks(tok);
  let X = tok.map(t => embed(t, P));
  for (const B of P.blocks) {
    for (const l of ['col', 'feat', 'nbr', 'full'] as const) {
      const Hn = X.map(x => rmsnorm(x, B.norm[l]));
      const O = mha(Hn, M[l], B.attn[l]);
      X = X.map((x, i) => add(x, O[i]));
    }
    X = X.map(x => { const h = rmsnorm(x, B.norm.ffn);
      const a = linear(B.w1, h), c = linear(B.w3, h);
      return add(x, linear(B.w2, a.map((z, j) => silu(z) * c[j]))); });
  }
  X = X.map(x => rmsnorm(x, P.normOut));
  const yhat = { number: X.map(x => linear(P.dec.number.W, x, P.dec.number.b)[0]),
                 datetime: X.map(x => linear(P.dec.datetime.W, x, P.dec.datetime.b)[0]),
                 boolean: X.map(x => linear(P.dec.boolean.W, x, P.dec.boolean.b)[0]) };   // text decoder exists, unused
  let L = 0, n = 0;
  tok.forEach((t, i) => { if (!t.mask) return; n++;
    if (t.sem === 0) L += huber(yhat.number[i] - t.number);
    else if (t.sem === 2) L += huber(yhat.datetime[i] - t.datetime);
    else if (t.sem === 3) L += bceLogits(yhat.boolean[i], t.boolean > 0 ? 1 : 0);
    else throw new Error('masking text not supported'); });
  return { loss: L / n, yhat, X };
}
const huber = (d: number) => Math.abs(d) < 1 ? 0.5 * d * d : Math.abs(d) - 0.5;
const bceLogits = (z: number, y: number) => Math.max(z, 0) - z * y + Math.log1p(Math.exp(-Math.abs(z)));
```

Init for a demo with random weights (PyTorch defaults):

- `Linear` W and b ~ U(−1/√fan_in, 1/√fan_in);
- RMSNorm γ = 1;
- `maskEmb[t]` ~ N(0,1)^256.

Real trained weights exist (HF `stanford-star/rt-v1`, `pretrain_<db>_<task>.pt`, https://huggingface.co/stanford-star/rt-v1), but they need MiniLM embeddings and paper-era preprocessing to be meaningful.

## Appendix B: preprocessing and sampler reference

- Preprocessing: §3.2-3.6 formulas, plus a mirror of `pre.rs` loops. Order: tables → columns in DataFrame order → rows. PK skip; FK → edges; else cell.
- Sampler: §4.3 pseudo-code.
- An independent Node implementation of both, plus the masks, which produced §6.3 (not derived by running repository code): `/tmp/claude-1004/-home-ibarra-Documents-Mestrado-Relational-Transformer/155ecce6-1f67-4870-9609-2c9740593feb/scratchpad/specchecks/toy_rt.mjs` (w=2 variant) and `toy_rt_w1.mjs` (w=1, the one shown).

## Appendix C: do NOT depict these (later RT-J / RT-PluRel features in the same repo)

From MAIN `src/rt/model/net.py`, `src/rt/model/legacy/plurel.py`, `rustler/src/fly.rs` at HEAD `def35f9`, and the RT-J paper arXiv 2610.17867:

- **Model**: QK-RMSNorm; learned per-head scale × log(#valid keys); softmax scale 1/d_head; sigmoid output gate; zero-init of W_o/W2/decoders; **no full attention**; **no boolean type**; cell re-sort by column; per-sequence loss averaging; selectable huber/l1/bce; d_model 512 / d_ff 2048 (85.6M params).
- **Sampler**: target cell placed at position 0; multi-seed contexts with random-walk retrieval; `local_ctx_size`; recency preference; temporal filter on both directions with a cutoff; same-horizon task rows invisible; whole-row budgets; partial Fisher–Yates draws over all children with a 2×w draw budget; depth-aware revisits; random multi-cell masking `mask_prob_max`; train-period-only stats; identifier-column policy.
- `CM/src/rt/model/legacy/v1.py` **is** the paper network (line-for-line equal to `PE/rt/model.py` except batch-key adapters). But CM's "legacy eval" runs it through the new sampler, so it does **not** reproduce paper contexts.
- Third-party explainers describing "the Relational Transformer" with d_model 512 / QK-norm / 3 attentions describe RT-J (e.g. https://relativedb.com/research/relational-transformer).

## Appendix D: key source index

| what | where |
|---|---|
| model, masks, embedding, loss | `PE/rt/model.py:20-267` |
| sampler | `PE/rustler/src/fly.rs:242-511` |
| preprocessing, stats, edges | `PE/rustler/src/pre.rs:67-699` |
| node/edge/sem types | `PE/rustler/src/common.rs:6-67` |
| split → table key, column index | `PE/rt/data.py:31-108` |
| training/eval loop, metrics | `PE/rt/main.py:66-452` |
| task list | `PE/rt/tasks.py:1-107` |
| text embedding | `PE/rt/embed.py:12-72` |
| hyperparameters | `PE/scripts/example_{pretrain,contd_pretrain,finetune}.py` |
| attention-order fix | MAIN `7f891d1` |
| sampler history | MAIN `eeccc2b`, `7f891d1`, `77bdb9b`, `66c4247`, `801a32b` |
| normalization caveat | MAIN `358bfd3`, `5fef314` |
| paper method | p.3-6 (§2-3), Alg. 1 p.15, App. A p.15-16, App. D p.19, App. E p.19-20, Alg. 2 p.26 |
| paper results | Tables 1-2 p.8, Table 3 p.8, Tables 4-5 p.9-10, Table 10 p.21, Tables 11-15 p.22-24 |

---

## 14. Verification log

Every correction proposed by the adversarial verifiers was re-checked against PE code, MAIN git history, the v3 PDF, the arXiv v1/v2 text, torch v2.6.0 source (fetched from GitHub) and the HF configs/API. Result: **28 accepted, 0 rejected**. Five were accepted with an amendment.

| # | lens | section | verdict | reason |
|---|---|---|---|---|
| 1 | model | 7.1 | ACCEPTED | `model.py:59` is a SwiGLU gate and p.6 says "gated MLPs"; reworded to "no attention-output gate". |
| 2 | model | 6.1, 13 #5 | ACCEPTED | v2.6.0 `_inductor/kernel/flex_attention.py:422-427` (`l_i = tl.where(l_i == 0.0, 1, l_i)`) and `_higher_order_ops/flex_attention.py:242-249` (`_safe_softmax`) confirmed; #5 marked resolved. |
| 3 | model | 7.2, 13 #4 | ACCEPTED | `layer_norm.cpp:289` bf16 epsilon and `native_functions.yaml:3306-3308` confirmed; no `rms_norm` in `_decomp`, `_refs`, `_refs/nn/functional` or `_inductor/decomposition.py` at v2.6.0; #4 marked resolved. |
| 4 | model | 6.2 | ACCEPTED | p.5 list, App. E p.19-20 and App. J p.26 say "global"; the bullet, Alg. 2 and `model.py:186-191` say "full". |
| 5 | model | 5.3 | ACCEPTED (amended) | `fly.rs:482-496`, `model.py:219, 240`, `main.py:285, 288` confirmed; narrowed "any placeholder" to "any finite placeholder" (NaN·0 = NaN). |
| 6 | model | 8.1 | ACCEPTED | `model.py:238-248` touches every decoder whose type has no masked cell and still fills `yhat_out`. |
| 7 | model | 5.2 | ACCEPTED | Column identity also enters structurally through `same_col_table` (`model.py:181-189`). |
| 8 | model | 8.1 | ACCEPTED | One `nn.Linear` per type (`model.py:115-131`) applied at all positions (`model.py:239`). |
| 9 | model | 5.1 | ACCEPTED | `pre.rs:302, 406` (per-DB spaces), `fly.rs:257-297, 324-328` (mixed batches), `model.py:165-183` (per-sequence comparisons). |
| 10 | data | 3.3 | ACCEPTED (major) | ddof=1 std of a 25%-positive 0/1 column is ≈0.433, not 0.5 (`pre.rs:321-323`); values recomputed (+1.732/−0.577; n=8: +1.620/−0.540). |
| 11 | data | 3.3, 4.16 | ACCEPTED (amended) | 0/0 = NaN within a table file, and multiplicative gating cannot remove it (`model.py:217-223`). Kept ±inf for Val/Test splits that reuse constant Train stats (`pre.rs:383-396`), and noted that the whole per-GPU batch loss becomes NaN (`model.py:246, 260`). |
| 12 | data | 1.2, 4.16 | ACCEPTED | Null FKs are skipped before the push (`pre.rs:462-464` vs `479`); the assert is inside the per-cell loop after the drop test (`fly.rs:455-464`). |
| 13 | data | 4.5 | ACCEPTED | The smallest non-empty bucket wins (`fly.rs:381-391`), and the transaction's own children go to bucket 3 (`fly.rs:436, 451`). |
| 14 | data | 4.6, 4.7 | ACCEPTED | `git show eeccc2b/7f891d1/66c4247/801a32b` and `git diff eeccc2b 801a32b` match the proposed history; added the `77bdb9b` assert. |
| 15 | data | 4.7 | ACCEPTED | `fly.rs:415-429` has no `visited` test before `db_p2f_ftr.push`; the sample is drawn at `fly.rs:439-443`. |
| 16 | data | 1.4 | ACCEPTED | `(v / 1_000_000_000) as i32` at `pre.rs:489, 510`; `Option<i32>` at `common.rs:28, 53`; Rust `as` truncates or wraps silently. |
| 17 | data | 3.1, 1.3 | ACCEPTED | `pre.rs:118/124/145` unwraps, `133-139` avito rename, `392` Train-stats unwrap, `452-458, 518` Db-only parents; added that `fkey_col_to_pkey_table` must be an object (`pre.rs:125-143`). |
| 18 | data | 3.6 | ACCEPTED | Fetched configs: `sentence_bert_config.json` max_seq_length 128; `tokenizer_config.json` model_max_length 512 (sentence-transformers and Xenova). |
| 19 | data | 12 | ACCEPTED | `.get(r)` returns `None` for a null time cell (`pre.rs:488`); the filter needs `is_some()` (`fly.rs:422`). |
| 20 | training | 10 #25, 13 #1 | ACCEPTED (major) | v1/v2 text and v3 PDF compared: v1 Table 1 RT(No) = v3 Table 11 none; v1/v2 Table 2 RT(No) = v3 Table 5 none; v2 Table 1 = v3; v1 Tables 9/10/11 = v3 12/13/14 per task. Dates from the arXiv abs page and `git log`. |
| 21 | training | 11.3 | ACCEPTED | v1 Table 9 rows = v3 Table 12; `eeccc2b:rustler/src/fly.rs:428-431` excluded equal-ts task rows; released `fly.rs:422` uses `>`. |
| 22 | training | 9.2 | ACCEPTED (amended) | `eeccc2b:scripts/fine_tune.py:12` loads the contd checkpoint that the `eeccc2b` contd script writes; HF API lists 13 contd-init / 4 pretrain-init fine-tunes; `main.py:356-357, 177-195` confirmed. Amended: the current `_best.pt` path comes from `23ede2e` (2025-11-21), while `f4d440c` set `_val_best.pt`; kept the Fig. 3 caption as evidence for pretrain init. |
| 23 | training | 9.3 | ACCEPTED | Step-0 eval and the `-inf` start are confirmed (`main.py:373-385`; `example_contd_pretrain.py:9-10`); Table 1 shows rel-f1 81.2/81.2 and 89.3/89.3. |
| 24 | training | 8.6, 10 #18, 4.14 | ACCEPTED | `usize` subtraction (`fly.rs:337-341`), release build (`data.py:4`, README:31), no overflow-checks in `Cargo.toml`; 192/224/64/0 recomputed by hand. |
| 25 | training | 9.1 | ACCEPTED | `data.py:77-81` `:Db` key; `tasks.py` has 36 tasks (12/9/7/8), giving 29-33 per run (recomputed); slides say "~20 tasks from 6 DBs". |
| 26 | training | 12 | ACCEPTED (amended) | bf16 labels and predictions confirmed (`fly.rs:482, 490`; `data.py:127`; `main.py:186-187, 284-288, 328-337`); precision reworded to 8 significant bits (≈2-3 digits). |
| 27 | training | 11.7 | ACCEPTED | Table 3 rows (p.8) are 5 clf + 4 reg tasks; Fig. 3 titles (p.7) say "mean over 5/4 downstream tasks". |
| 28 | training | 13 #2 | ACCEPTED | Fig. 6 caption (p.15) gives L=512 and w=128; p.6 gives context 1024; script `example_pretrain.py:21` `max_bfs_width=256` (line 21, not 23). |

Follow-on edits made for consistency: §2.5 now notes that Tables 4 and 12 predate `66c4247` (from #20 and #21); §4.14 references the underflow (#24); §11 opens with a provenance note (#20).
