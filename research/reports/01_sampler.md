# 01 — The RT context-window sampler (Algorithm 1 / App. A), exactly as implemented

Scope: the paper-era sampler `rustler` (Rust), how it builds one context window of `seq_len` cell tokens around a seed row, plus the preprocessing that produces the data it reads.
Abbreviations used in citations:

- `PE/` = paper-era snapshot `.../scratchpad/upstream/rt-paper-era` (commit `2a899c8`, 2026-02-04). Files: `PE/rustler/src/fly.rs` (sampler), `PE/rustler/src/pre.rs` (preprocessing), `PE/rustler/src/common.rs` (shared structs), `PE/rt/data.py`, `PE/rt/main.py`, `PE/rt/tasks.py`, `PE/rt/model.py`, `PE/rt/embed.py`.
- `MAIN/` = current main `.../scratchpad/upstream/rt-official` (`def35f9`, 2026-10-18). This is the later "RT-J" version and **not** paper behaviour.
- `paper p.X` = Relational_Transformer.pdf (arXiv 2510.06377v3, ICLR 2026).

**Version verdict.** `PE/rustler/src/{fly,pre,common}.rs` are byte-identical to commit `801a32b` (2025-10-16, "fixed bfs width sampling"). `git diff 801a32b 2a899c8 -- rustler/` is empty; between those commits only `rt/embed.py` changed. `pre.rs` and `common.rs` have not changed since the ICLR submission commit `eeccc2b` (2025-10-06). So the paper-era sampler is the last one released before the camera-ready (v3, 2 Mar 2026). The git log cannot tell us whether every published number was re-run after the 2025-10-08..16 sampler fixes (§9).

---

## 1. Data the sampler loads, and its on-disk format

### 1.1 Files (`~/scratch/pre/<db>/`)

| file | content | writer | reader |
|---|---|---|---|
| `nodes.rkyv` | One rkyv-serialized `Node` per row, concatenated back to back, in global node order | pre.rs:676-683 | fly.rs:259-261 (mmap), `get_node` fly.rs:514-520 |
| `offsets.rkyv` | `Offsets{offsets: Vec<i64>}`: byte offsets, starting with 0. Length is `num_nodes+1`, and node `i` is bytes `[off[i], off[i+1])` | pre.rs:668, 681, 687-690 | fly.rs:267-273, 515-517 |
| `p2f_adj.rkyv` | One rkyv `Adj{adj: Vec<Vec<Edge>>}`. `adj[p]` lists every child row whose FK points at row `p` (P→F adjacency) | pre.rs:411-413, 523-529, 693-696 | fly.rs:275-277, `get_p2f_edges` fly.rs:522-526 |
| `text.json` | JSON list of strings. Table names, `"<col> of <table>"` strings and string cell values all share **one** interned vocabulary (`text_to_idx`) | pre.rs:406, 424-437, 595-597, 624-630 | `PE/rt/embed.py:55-59` |
| `text_emb_<model>.bin` | Raw bf16 matrix `[len(text.json), d_text]`, row-major (`np.tofile`). Model is `sentence-transformers/all-MiniLM-L12-v2`, d_text = 384 | embed.py:14-16, 70-72 | fly.rs:263-265, `get_text_emb` fly.rs:528-532 |
| `table_info.json` | `{"<table>:<Db\|Train\|Val\|Test>": {node_idx_offset, num_nodes}}` | pre.rs:650-664 | data.py:71-84 |
| `column_index.json` | `{"<col> of <table>": text_idx}` | pre.rs:407, 434-436, 642-645 | data.py:17-43 |
| `text_map.json` | reverse map string→idx (not used by the sampler) | pre.rs:633-639 | — |

### 1.2 Global node (row) indexing

- Parquet files are read in this order: `db/*.parquet` sorted by path, then `tasks/*/*.parquet` sorted by path (pre.rs:79-92). Within a task directory, `test` therefore comes before `train`, which comes before `val`. Each table gets `node_idx_offset = running row count` (pre.rs:302, 305). A node index is global across all tables of one database, task tables included.
- A task file whose path has exactly **two more dashes** than the dataset path is skipped (pre.rs:93-95). In RelBench this drops the two-dash task names, e.g. `user-item-purchase`, which are the recommendation / link-prediction tasks. RT does not support those (paper p.10, Limitations).
- Table type comes from the path. `db/` → `TableType::Db`. For `tasks/<name>/{train,val,test}.parquet`, the table name is the directory name and the type is `Train`/`Val`/`Test` (pre.rs:152-174; enum common.rs:14-20).
- Table metadata comes from the parquet key-value metadata: `pkey_col`, `fkey_col_to_pkey_table`, `time_col` (pre.rs:107-150). For rel-avito, task FKs `UserID`/`AdID` are remapped to `UserInfo`/`AdsInfo` (pre.rs:133-139).

### 1.3 `Node` (one row) — `common.rs:48-62`

```
Node { is_task_node: bool, node_idx: i32, f2p_nbr_idxs: Vec<i32>, f2p_edges: Vec<Edge>,
       timestamp: Option<i32>, table_name_idx: i32,
       col_name_idxs, sem_types, number_values, text_values, datetime_values, boolean_values, class_value_idx  // parallel per-cell arrays
}
```

How pre.rs fills these (loop over tables, then over columns in DataFrame order, then over rows; pre.rs:415-615):

- `table_name_idx` = interned table name (pre.rs:424-427). A task's Train, Val and Test tables use the same name string, so they share **one** `table_name_idx` and the same `col_name_idx` per column. This matters for §6.
- `col_name_idx` = interned `"<col> of <table>"` (pre.rs:432-437). Every column is interned, PK and FK columns included, so they appear in `column_index.json`.
- **PK column** → skipped. It produces no cells (pre.rs:439-442).
- **FK column** → produces **no cell**, only edges (pre.rs:444-533; see §1.4 and §8). Null FK values are skipped (pre.rs:462-464).
- **Other columns** → one cell per non-null value (pre.rs:535-613). Each cell pushes to *all* parallel arrays, with zeros in the slots of the other types:
  - Boolean → `SemType::Boolean`, `boolean_values = ((b?1:0) - mean)/std`, i.e. booleans are **z-scored** too (pre.rs:553-564). The std comes from `std(ddof=1)`, defaulting to 0.0 for an empty column, with **no** guard for std = 0 (pre.rs:320-328). A constant boolean column would therefore produce NaN.
  - UInt32/Int32/Int64/Float32/Float64 → cast to f64 (pre.rs:543-550). NaN is skipped, so no cell (pre.rs:566-568). Otherwise `SemType::Number`, `number_values = (v-mean)/std` (pre.rs:565-583). If std is 0 or undefined it is set to 1 (pre.rs:330-340).
  - Datetime (must be ns) → `SemType::DateTime`, `datetime_values = (ns - dt_mean)/dt_std`. These are **global** statistics over every datetime column of every table: all DB tables and all task splits, computed in nanoseconds with population std (pre.rs:342-366, 375-376, 584-594). The **time column itself is an ordinary datetime cell**; it is not removed.
  - String → `SemType::Text`, `text_values = class_value_idx = intern(value)` (pre.rs:595-605). For non-text cells `class_value_idx = -1` and `text_values = 0`.
- **Column statistics** (pre.rs:316-397) are computed per table file over **all rows of that file**. Val/Test task tables reuse the Train table's statistics (pre.rs:383-396). DB tables therefore use whole-database statistics, which includes the val/test periods. The paper says "computed on the training split" (paper p.4-5, §3.1); for DB tables that does not hold in this code. MAIN docs later call this the "pre-fix, leaky normalization" (MAIN commit `358bfd3`, docs/downloads.md diff).
- `SemType` enum order is Number=0, Text=1, DateTime=2, Boolean=3 (common.rs:38-44). The sampler emits `sem_type as i32` (fly.rs:479-480). model.py iterates `["number","text","datetime","boolean"]` with these same indices (model.py:208-224, 239).

### 1.4 `Edge` and adjacency — `common.rs:22-34`

`Edge { node_idx: i32, table_name_idx: i32, table_type: TableType, timestamp: Option<i32> }`

For each non-null FK value `v` in row `r` of table T, with child node `c = T.offset + r` (pre.rs:460-530):

- Parent node `p = offset(ptable, Db) + v` (pre.rs:452-459, 477). The FK value is used directly as the 0-based **row position** of the parent table, so RelBench's consecutive 0..n-1 primary keys are assumed. Parents are always looked up as **Db** tables.
- `c.f2p_nbr_idxs.push(p)` (pre.rs:479).
- `c.f2p_edges.push(Edge{p, intern(ptable), Db, parent_ts})`. Here `parent_ts` is the parent row's own time-column value in seconds (pre.rs:495-521). **fly.rs never reads `f2p_edges[*].timestamp`.**
- `p2f_adj[p].push(Edge{c, T.table_name_idx, T.table_type, child_ts})` (pre.rs:523-529). `child_ts` is the child row's time column in seconds, or None if T has no time column or the value is null.
- The per-row order of `f2p_edges` follows FK-column order in the DataFrame, because the outer loop runs over columns (pre.rs:429).

### 1.5 `Node.timestamp` — subtle rule

`node.timestamp` is assigned **only inside the FK loop** (pre.rs:481-493): `Some(ns/1e9 as i32)` if the table has a `time_col`, otherwise None. Consequences:

- A row with **no non-null FK** keeps `timestamp = None`, even if its table has a time column (`Node::default()`, pre.rs:408-410).
- Task rows always have an FK to their entity and a `timestamp` time column, so they always carry a timestamp.
- Timestamps are seconds as i32. The `as i32` cast truncates outside 1901..2038.
- If a row has only a PK and null values, `node_idx`/`table_name_idx` keep their default 0. It has zero cells, so it emits nothing (edge case only).

### 1.6 Loading in the sampler (`Sampler::new`, fly.rs:242-314)

Inputs are a list of `dataset_tuples (db_name, node_idx_offset, num_nodes)`, one per *task*, plus `target_columns[i]` and `columns_to_drop[i]`, given as `col_name_idx` values. data.py builds them (data.py:59-108). The table to enumerate is `"<table>:Db"` if that key exists (an autocomplete task on a DB table), otherwise `"<table>:<Train|Val|Test>"` (data.py:77-82). The target and drop indices are `column_index["<col> of <table>"]` (data.py:86-92). For each task the sampler mmaps `nodes.rkyv`, `text_emb_*.bin` and `p2f_adj.rkyv`, and reads `offsets` into memory (fly.rs:258-285). A database is mmapped once per task.

---

## 2. The BFS, exactly (fly.rs `seq`, lines 357-511)

### 2.1 State per sequence

```
visited  = [false; num_nodes]                     // fly.rs:364, num_nodes = offsets.len()-1, whole DB incl. task tables
f2p_ftr  = [(0, seed)]                            // fly.rs:366   a STACK of (depth, node)
p2f_ftr  = []                                     // fly.rs:368   p2f_ftr[d] = Vec<node> ("bucket" at depth d)
rng      = StdRng::seed_from_u64(epoch + seed_node_idx + seed)   // fly.rs:371-375, wrapping u64 adds
seq_i    = 0
```

`StdRng` in rand 0.9.1 (Cargo.lock) is ChaCha12 ("The current algorithm used is the ChaCha block cipher with 12 rounds", https://docs.rs/rand/0.9.1/rand/rngs/struct.StdRng.html). A context is therefore **deterministic** for a given (epoch, seed row, seed).

### 2.2 Selecting the next row (fly.rs:378-397)

1. **If `f2p_ftr` is non-empty: `pop()` the last element (LIFO).** No randomness. F→P rows have absolute priority over any P→F row, whatever their depth. Because it is a stack, the FK chain is followed depth-first: the *last* FK of the row is visited first, then that parent's own parents, and so on.
2. Otherwise, find the smallest `d` with `p2f_ftr[d]` non-empty (`depth_choices[0]`). If there is none, **return**: the frontier is empty and the rest of the sequence stays padding. Otherwise draw `r = rng.random_range(0..len)`, swap `bucket[r]` with the last element and pop it. This is a uniform random pick among the frontier rows with the minimum *depth label*.

### 2.3 Visiting a row (fly.rs:399-452)

```
if visited[n]: continue            // checked at POP time; duplicates may sit in the frontier
visited[n] = true
for e in node.f2p_edges:           // ALL parents, no temporal filter, no visited check
    f2p_ftr.push((depth+1, e.node_idx))
db_p2f = []
for e in p2f_adj[n]:               // all children of n
    if e.table_name_idx != seed.table_name_idx && e.table_type != Db: continue   // L417: task rows only of the SEED's task table
    if e.timestamp.is_some() && seed.timestamp.is_some() && e.timestamp > seed.timestamp: continue  // L422: keep <=
    if e.table_type == Db: db_p2f.push(e.node_idx); continue                    // DB children: collected for subsampling
    p2f_ftr[depth+1].push(e.node_idx)                                          // task-table children: ALL kept, never subsampled
idxs = db_p2f.len() > max_bfs_width ? rand::seq::index::sample(rng, db_p2f.len(), max_bfs_width) : 0..db_p2f.len()  // L439-443
for i in idxs: p2f_ftr[depth+1].push(db_p2f[i])                                 // L445-452
```

- `index::sample` returns `amount` distinct indices in fully shuffled random order (https://docs.rs/rand/0.9.1/rand/seq/index/fn.sample.html). The result is a uniform subset without replacement.
- The width bound **w = `max_bfs_width`** (256 in every example script: `PE/scripts/example_pretrain.py:23`, `example_finetune.py:22`, `example_contd_pretrain.py:23`; paper Fig. 6 uses L=512, w=128) applies **per visited row, to its DB children only**. Task-table children are never subsampled. In practice these are the in-context label rows: every task row of the same task (any split) that references the visited entity and passes the time filter.
- The width bound applies **after** the temporal filter ("temporal filter then bfs subsampling", commit `66c4247`). This matches paper Alg. 1 (p.15): filter N by time, then RandomSample(N, w).
- Children are pushed even if already visited; they are skipped when popped (fly.rs:399-401).
- A row's frontier is expanded **before** its cells are emitted (L406-452 come before L454-506). This only affects RNG consumption order.

### 2.4 Temporal filter: precise semantics

- The comparison is **child-edge timestamp vs. the seed row's timestamp**, never vs. the current row (fly.rs:422). The edge timestamp is the child row's own time-column value (pre.rs:481-492, 523-528).
- A child is dropped only if **both** are Some **and** `child_ts > seed_ts` (strict >). Equal timestamps are **kept**, i.e. `<=`, matching paper Alg. 1 `T(q) ≤ T(s)` (p.15).
- A child **without** a timestamp, because its table has no time column or the value is null, **always passes**.
- If the **seed has no timestamp**, the filter is **disabled entirely** for that sequence. This can happen with an autocomplete seed in a table without a time column or without FKs (§1.5).
- **F→P parents are never time-filtered** (fly.rs:406-408). This matches the paper: "all parent rows (F→P) are always included" (p.4 §3.1; p.16 App. A).
- **No global cutoff** exists in the paper-era sampler. Only the seed's own timestamp is used. MAIN later added per-task `cutoff_timestamps` (MAIN fly.rs:336, 1005, 1878-1884).

### 2.5 "HopDistance" vs. the code's `depth`

The paper's `argmin_{r∈F} HopDistance(r,s)` (Alg. 1, p.15) is implemented as **the depth label carried by the frontier entry**. That label is the length of the discovery path: +1 per F→P hop (fly.rs:407) and +1 per P→F hop (fly.rs:436, 451). It is not a true shortest-path distance:

- A node may sit in several buckets. The *first pop* wins (visited check), and the node is then expanded with that entry's depth.
- F→P rows ignore depth for selection entirely, because the stack has priority. They still carry a depth that determines the bucket of their own P→F children.
- Example: seed task row depth 0 → entity depth 1 → entity's children go into bucket 2. A child (e.g. a transaction) visited from bucket 2 immediately pulls its parents (depth 3) through the stack, and their children go into bucket 4. Bucket 4 is only drawn from once bucket 2 is empty.

### 2.6 Cell emission, stop condition, truncation (fly.rs:454-510)

- Cells come out in the row's stored order (`col_name_idxs`), which is DataFrame column order minus PK/FK/null/NaN cells (pre.rs:429, 535-613).
- A cell is skipped if `col ∈ columns_to_drop` **and** (`node is the seed` **or** `node.timestamp == seed.timestamp`) (fly.rs:457-461). The Option equality makes **None == None true**, so with a timestamp-less seed the drop applies to every timestamp-less row carrying that column. Drop indices are `"<col> of <seed table>"`, so only rows of the seed's own table can match. MAIN later added `target_node.timestamp.is_some() &&` (MAIN fly.rs:1734-1737).
- After each emitted cell, `seq_i += 1`; if `seq_i >= seq_len` the cell loop and the BFS loop both break (fly.rs:502-509). **The last row can therefore be truncated mid-row.** The paper says "once a row is selected, all of its non-missing feature cells are added" (p.4) and "the traversal stops when the total number of cells reaches the context length" (p.16). The code stops exactly at L cells, which may cut a row.
- `assert!(node.f2p_nbr_idxs.len() <= MAX_F2P_NBRS)` with `MAX_F2P_NBRS = 5` (fly.rs:25, 464). A row with more than 5 non-null FKs would panic the sampler.
- There is no limit on BFS iterations. If the frontier is exhausted before L, the remaining tokens are padding.

### 2.7 Paper Alg. 1 vs code

| aspect | paper (Alg. 1 p.15, App. A p.16, §3.1 p.4) | code (fly.rs) |
|---|---|---|
| choice among F→P-added rows | `RandomSelect(R)` | **LIFO stack pop, deterministic** (L378-379) |
| choice among P→F rows | min HopDistance, uniform random | min **depth label**, uniform random (L381-396) |
| temporal filter | `T(q) ≤ T(s)` on P→F neighbours | same (`>` skipped), only when both timestamps exist (L422) |
| width bound w | `RandomSample(N, w)` on all P→F neighbours | **DB children only**; task-table children (in-context labels) never subsampled (L426-452) |
| task tables | "only one task table is active at a time" (p.4) | task-table children kept only if their `table_name_idx` equals the seed's (L417). Any split passes |
| stop | `while |C| < L` | hard stop at L tokens, last row may be partial (L502-509) |
| visited | mark at pop | same (L399-402) |
| cells of a row | non-missing feature cells (not PK/FK) | same, minus `columns_to_drop` (L457-461) |

---

## 3. Token order inside a sequence and padding

- The seed row is always visited first (initial stack entry `(0, seed)`, fly.rs:366), so **its cells are tokens 0..k-1**, including the masked target cell.
- After that, rows appear in **visit order** as defined by §2.2. Within a row, cells follow the row's stored column order (§2.6).
- Positions `[seq_i, seq_len)` keep the defaults from `Vecs::new` (fly.rs:65-85). See the padding column below.
- The model uses **no positional encoding** (paper p.5 "Permutation invariance"; model.py builds `x` only from value and column-name embeddings, lines 208-224). Order therefore matters only for truncation and for which rows make it in.

---

## 4. Per-token fields (fly.rs:27-44, 462-500; reshaped in data.py:116-154)

Final tensor shapes per batch: `[B,S]` unless noted; `S = seq_len`, `B = batch_size`.

| field | dtype / shape | value for a cell token | padding default | used by paper model? |
|---|---|---|---|---|
| `node_idxs` | i32 [B,S] | global row index of the cell's row (L462) | -1 | yes (feat/nbr masks, model.py:168-178) |
| `f2p_nbr_idxs` | i32 [B,S,5] | the row's parent global indices, in FK-column order, -1-padded to 5 (L464-467). Lists **all** FK parents, whether or not they are in the context | -1 | yes (model.py:171-178) |
| `table_name_idxs` | i32 | interned table name; Train/Val/Test of a task share it (L469) | 0 | yes (col mask, model.py:181-183) |
| `col_name_idxs` | i32 | interned `"<col> of <table>"` (L470) | 0 | yes (col mask) |
| `class_value_idxs` | i32 | text idx for Text cells, else -1 (L471; pre.rs:563,582,593,604) | -1 | **no** |
| `col_name_values` | bf16 [B,S,384] | MiniLM embedding of `"<col> of <table>"` (L472-477) | zeros | yes |
| `sem_types` | i32 | 0 Number, 1 Text, 2 DateTime, 3 Boolean (L479-480) | 0 | yes |
| `number_values` | bf16 [B,S,1] | z-scored number, or 0 (L482) | 0 | yes |
| `text_values` | bf16 [B,S,384] | embedding of `text_values[cell]`. **For non-text cells this is index 0, i.e. the embedding of whatever string was interned first, not zeros** (L484-486; pre.rs:559,578,589). Harmless, because the model gates by sem_type | zeros | yes (gated) |
| `datetime_values` | bf16 [B,S,1] | globally z-scored ns, or 0 (L488) | 0 | yes |
| `boolean_values` | bf16 [B,S,1] | **z-scored** 0/1, so true is >0 and false is <0, or 0 (L490) | 0 | yes; label = `y>0` (model.py:253-255; main.py:336) |
| `masks` | bool | `= is_targets` (L496) | false | yes (mask embedding + loss) |
| `is_targets` | bool | seed row AND col == target column (L492-494) | false | eval only (main.py:284-288) |
| `is_task_nodes` | bool | `node.is_task_node || col == target_column` (L498-499). For autocomplete this flags the target column in every row of that table | false | **no** |
| `is_padding` | bool | false (L500) | true | yes |
| `true_batch_size` | int (per batch) | number of non-wrapped items (§7) | — | eval |

Value precision: values are stored f32 in nodes.rkyv and cast to **bf16** at sampling time (fly.rs:482-490).

---

## 5. Masking

- **Exactly one masked token per sequence: the seed row's target cell.** `masks[i] = is_targets[i] = (row == seed) && (col == target_column)` (fly.rs:492-496). There is no random masking anywhere in the paper-era sampler or model. Training and evaluation build sequences the same way.
  - The paper's multi-cell masking (Table 14, p.24, P(mask) ∈ {0.2, 0.4}) is an ablation that is **not** in the released paper-era code. Its default is P(mask)=0. MAIN later added `set_mask_prob_max_py` (MAIN fly.rs:475).
- **Other cells of the target column are visible.** This covers other task rows of the same task (forecasting in-context labels) and other rows of the same DB table (autocomplete). This is the "in-context labels" mechanism (paper p.2, p.22-23 Table 12, p.25 "In-context labels").
- The seed's target cell is **kept** as a token: its `sem_type` and column-name embedding stay, and the value embedding is swapped for the learned `mask_emb[type]` (model.py:208-224). The loss is computed only on `masks` (model.py:236-265).
- `columns_to_drop` (leakage columns, defined per autocomplete task in tasks.py:35-85, e.g. rel-f1 `results.position` drops `statusId, positionOrder, points, laps, milliseconds, fastestLap, rank`) are **removed as tokens entirely**, not masked. This happens only for the seed row and for rows whose timestamp equals the seed's (fly.rs:457-461), with the None==None caveat from §2.6. The same leakage columns in rows with other timestamps stay visible. All forecasting tasks use `[]` (tasks.py:3-28).
- In the ICLR-submission version (`eeccc2b`), `is_target` also required sem_type ∈ {Number, Text, Boolean}. Commit `7f891d1` removed that condition (§9).

---

## 6. Task tables (forecasting) and which split's rows are visible

- A task table row has a `timestamp` (datetime cell plus node timestamp), an FK to the entity (edge only, no cell) and the label (cell). Example: rel-f1 `driver-dnf` rows yield 2 tokens, `"timestamp of driver-dnf"` and `"did_not_finish of driver-dnf"`. Some label columns are cast to bool in preprocessing (pre.rs:208-289).
- **Seeds are task rows of the requested split.** `items` = every row of `"<task>:<Split>"` whose target cell exists, i.e. a non-null and non-NaN label (fly.rs:287-296; data.py:77-84).
- **How in-context labels are reached:** seed task row → (F→P stack) entity row at depth 1 → entity's P→F children. These include DB rows (time-filtered, subsampled to w) and **every task row of the same task table, any split, with `ts ≤ seed_ts`** (fly.rs:415-437). All of them go into bucket 2. Other entities' task rows are reached deeper: entity → DB child → its other parent → that parent's children → another entity (via F→P) → that entity's task rows. Paper Table 12 (p.23) reports 0-10 such "other entity labels" per 1024-cell context on average.
- Task rows of **other** tasks on the same database are excluded (fly.rs:417; paper p.4 "Only one task table is active at a time").
- **Split visibility** follows from the shared `table_name_idx` (pre.rs:424-427) plus the `≤ seed_ts` filter. RelBench builds train timestamps backward from `val_timestamp - Δ`, and val/test timestamps forward from `val_timestamp`/`test_timestamp` with step Δ (relbench `BaseTask._get_table`, https://github.com/snap-stanford/relbench/blob/main/relbench/base/task_base.py):
  - Train seed: sees only Train rows with ts ≤ its own ts. Val/Test rows are later.
  - Val seed: sees Train rows and Val rows with ts ≤ seed_ts.
  - Test seed: sees Train, Val and Test rows with ts ≤ seed_ts.
  - **Surprise:** because the comparison is `<=`, task rows of *other entities* at the **same timestamp as the seed** (same prediction horizon, in the same split, labels unmasked) are allowed in the context. For a Test seed those are other test examples' ground-truth labels. The ICLR-submission code excluded them (`edge.timestamp == seed.timestamp && table_type != Db → continue`, eeccc2b fly.rs, §9). `7f891d1` kept them excluded (`>=`). `66c4247` switched everything to `>`, which admits them. MAIN later re-excludes them as "same-horizon task rows" (MAIN fly.rs:1724, 1886-1900). The seed's own entity cannot contribute a second row at the same timestamp, because RelBench task rows are unique per (entity, timestamp).
- Test labels come straight from `tasks/<task>/test.parquet` (pre.rs:87-91, 167-172). RelBench's `get_table("test")` masks targets by default; the raw parquet read here evidently contains them, since items require a non-null target and eval computes AUROC/R² from `boolean_values`/`number_values` at `is_targets` (main.py:283-288, 331-337).
- **Autocomplete tasks** (target in a DB table): the seed is a DB row with `table_type = Db`. Every task-table child then fails L417, because task names differ from DB table names, so **no task rows appear at all**. Other rows of the same table, which carry visible target-column values, are the in-context labels. `split` is ignored for these tasks: data.py:77-80 picks `"<table>:Db"` whenever it exists, so "train" seeds are **all rows of the table, at any time**. The paper says "we pretrain and fine-tune only on the training splits ... using database rows only up to the training cutoff timestamps" (p.6 §4.1). The code applies no such cutoff to autocomplete seeds, only the per-seed `≤ seed_ts` filter. This matters in the continued-pretraining setting, where the target DB's autocomplete tasks are trained on (`example_contd_pretrain.py:17-19`). MAIN later added `cutoff_timestamps`.

---

## 7. Seed enumeration, shuffling, batching, sharding

- **Enumeration:** `items` is the concatenation, over the task list, of qualifying rows in node order (fly.rs:257-297). In pretraining, the training sampler holds all train tasks at once (main.py:123-136), so **batches mix tasks and DBs**. There is no per-task reweighting; a task's share is proportional to its row count.
- **Shuffle:** `shuffle_py(epoch)` sets `self.epoch = epoch` and shuffles `items` with `StdRng::seed_from_u64(epoch + seed)` (fly.rs:324-328). Training calls `shuffle_py(int(steps/len(loader)))` before each pass (main.py:370). Each eval sampler is built with `seed=0` and `shuffle_py(0)` is called once (main.py:149-160), so the eval order and every eval context are fixed. Since the per-sequence RNG seed is `epoch + node + seed`, training contexts change every epoch.
- **Length:** `len = ceil(N / (batch_size * world_size))` (fly.rs:332-334).
- **Batch `b` on rank `k`:** item `j = b*bs*ws + k*bs + i` for `i < bs`, taken `mod N` so it **wraps around** (fly.rs:347-351). `true_batch_size = min(bs, N - k*bs - b*bs*ws)` (fly.rs:337-341). This is a usize subtraction, so it can underflow on a rank's final batch. The crate is built `--release`, where overflow wraps and the result is effectively `bs`.
- **Eval:** rows `≥ true_batch_size` get `masks = is_targets = False` and `is_padding = True` (main.py:273-279). Eval stops after `max_eval_steps = 40` batches (main.py:299-300; scripts set 40). With bs=32 and 8 GPUs, metrics are therefore computed on a **fixed random subset of at most 40·32·8 = 10,240 seeds per split**. Training does not neutralize wrapped duplicates; it only pops `true_batch_size` (main.py:401).
- **Sizes:** per-GPU `batch_size=32` × 8 GPUs = 256, matching paper p.6. `seq_len=1024`, `max_bfs_width=256` (scripts). The DataLoader uses `batch_size=None`, so each `__getitem__(batch_idx)` returns one whole Rust batch (data.py:116-154; main.py:137-144).

---

## 8. F→P links: representation for task rows and DB rows

Task rows and DB rows are handled the same way (pre.rs:444-533). An FK column becomes, per non-null value:

1. A parent id in `f2p_nbr_idxs`. It is shipped per token, up to 5 entries padded with -1, and the model's feature and neighbour masks consume it (model.py:171-178, 186-188).
2. An `Edge` in `f2p_edges`. The BFS walks it via the stack (fly.rs:406-408).
3. A reverse `Edge` in `p2f_adj[parent]` carrying the child's table name, table type and timestamp. The BFS walks it via the buckets (fly.rs:410-452).

A task row → entity is simply one FK edge of a row whose `table_type ∈ {Train, Val, Test}`. Task rows are never parents. RT does not distinguish between multiple FK columns (paper p.10 Limitations); the code only keeps the ordered list of parent ids, and FK column names never become tokens.

---

## 9. Git history of the sampler (rt-official)

| commit | date | change to sampling (fly.rs) |
|---|---|---|
| `eeccc2b` "iclr submission" | 2025-10-06 | Temporal filter before subsampling. DB children `≤ seed_ts`. **Task-table children require `< seed_ts`** (same-timestamp task rows excluded). DB children subsampled to `subsample_p2f_edges`. `is_target` limited to Number/Text/Boolean. Model attention order `feat, nbr, col, full`. |
| `7f891d1` "fix in attention order and sampling" | 2025-10-08 | Model order → `col, feat, nbr, full` (model.py:83). Sampler: DB children collected unfiltered, **subsampled first, then** filtered by the child node's timestamp (`>` dropped). Task children `>= seed_ts` dropped (strict <). `is_target` for any sem type. |
| `77bdb9b` | 2025-10-08 | Rename `MAX_F2P_NBR` → `MAX_F2P_NBRS`; add `assert!(len ≤ 5)`. |
| `66c4247` "temporal filter then bfs subsampling" | 2025-10-09 | Rename to `max_bfs_width`. Single filter `> seed_ts` for all children (task rows now `≤`, same-timestamp allowed) **before** subsampling. **Bug:** the table-name check was applied to DB edges too, so only children from the seed's own table survived. |
| `801a32b` "fixed bfs width sampling" | 2025-10-16 | Fix: table-name check only for non-Db edges (fly.rs:417). **= final paper-era sampler.** Unchanged through `2a899c8` (2026-02-04). |
| `238d8ba` | 2026-06-27 | Adds a `timestamps` output field (post-paper). |
| `4af9ff1` (1.9.0) and later | 2026-09-30.. | Rewrite (MAIN): same-horizon task-row exclusion (MAIN fly.rs:1724, 1898-1900), cutoff timestamps, mask probability, booleans as numbers by default (`legacy_boolean`, MAIN fly.rs:1526), train-period-only statistics (internal `8030aa8`, cited in MAIN `358bfd3`; object not in this clone). **Not paper behaviour.** |

---

## 10. Pseudo-code mirroring the Rust (TypeScript-ready)

```ts
// ===== Preprocessed structures (pre.rs) =====
type SemType = 0 | 1 | 2 | 3;               // Number, Text, DateTime, Boolean
type TableType = 'Db' | 'Train' | 'Val' | 'Test';
interface Edge { node: number; tableNameIdx: number; tableType: TableType; ts: number | null }
interface Node {
  isTaskNode: boolean; nodeIdx: number; tableNameIdx: number;
  ts: number | null;                         // seconds; ONLY set if row has >=1 non-null FK and table has time col
  f2pNbrIdxs: number[];                      // parents, FK-column order (<=5 else Rust panics)
  f2pEdges: Edge[];                          // same parents as Edges (type always 'Db')
  cells: { colNameIdx: number; sem: SemType; num: number; textIdx: number; dt: number; bool: number; classIdx: number }[];
}
// p2fAdj[p] = Edge[] of children (edge.ts = CHILD's timestamp, edge.tableType = child's type)

// ===== Sampler.seq (fly.rs:357-511) =====
function sampleSequence(seed: number, epoch: number, globalSeed: number, L: number, w: number,
                        targetCol: number, dropCols: number[], nodes: Node[], p2fAdj: Edge[][]): Token[] {
  const rng = new SeededRng(epoch + seed + globalSeed);    // Rust: ChaCha12 StdRng (bit-exactness not required for a demo)
  const visited = new Array(nodes.length).fill(false);
  const f2pStack: [number, number][] = [[0, seed]];        // (depth, node) — LIFO
  const buckets: number[][] = [];                          // buckets[d] = P->F frontier at depth d
  const S = nodes[seed];
  const out: Token[] = [];
  const push = (d: number, n: number) => { while (buckets.length <= d) buckets.push([]); buckets[d].push(n); };

  while (true) {
    let depth: number, n: number;
    if (f2pStack.length > 0) {
      [depth, n] = f2pStack.pop()!;                        // F->P first, last-pushed first
    } else {
      const d = buckets.findIndex(b => b.length > 0);
      if (d === -1) break;                                 // frontier exhausted -> rest is padding
      const b = buckets[d];
      const r = rng.int(0, b.length);                      // uniform in [0, len)
      [b[r], b[b.length - 1]] = [b[b.length - 1], b[r]];
      n = b.pop()!; depth = d;
    }
    if (visited[n]) continue;                              // visited test at pop time
    visited[n] = true;
    const node = nodes[n];

    for (const e of node.f2pEdges) f2pStack.push([depth + 1, e.node]);   // no time filter, no visited test

    const dbKids: number[] = [];
    for (const e of p2fAdj[n]) {
      if (e.tableType !== 'Db' && e.tableNameIdx !== S.tableNameIdx) continue;   // only seed's task table (any split)
      if (e.ts !== null && S.ts !== null && e.ts > S.ts) continue;               // keep <= seed time
      if (e.tableType === 'Db') { dbKids.push(e.node); continue; }
      push(depth + 1, e.node);                                                    // task rows: never subsampled
    }
    const idxs = dbKids.length > w ? rng.sampleWithoutReplacement(dbKids.length, w)
                                   : [...dbKids.keys()];
    for (const i of idxs) push(depth + 1, dbKids[i]);

    for (const c of node.cells) {                                                 // stored column order
      if (dropCols.includes(c.colNameIdx) && (node.nodeIdx === seed || node.ts === S.ts)) continue; // null===null is true (as in Rust)
      const isTarget = node.nodeIdx === seed && c.colNameIdx === targetCol;
      out.push({
        nodeIdx: node.nodeIdx,
        f2pNbrIdxs: [...node.f2pNbrIdxs, -1, -1, -1, -1, -1].slice(0, 5),
        tableNameIdx: node.tableNameIdx, colNameIdx: c.colNameIdx, classValueIdx: c.classIdx,
        colNameEmb: textEmb(c.colNameIdx), semType: c.sem,
        number: c.num, textEmb: textEmb(c.textIdx) /* idx 0 for non-text */, datetime: c.dt, boolean: c.bool,
        mask: isTarget, isTarget, isTaskNode: node.isTaskNode || c.colNameIdx === targetCol, isPadding: false,
      });
      if (out.length >= L) return out;                     // may cut the current row mid-way
    }
  }
  return out;                                              // caller pads to L
}

// padding token: nodeIdx -1, f2p [-1x5], tableNameIdx 0, colNameIdx 0, classValueIdx -1, zero embeddings,
//                semType 0, values 0, mask/isTarget/isTaskNode false, isPadding true

// ===== Sampler.batch / shuffle (fly.rs:324-355) =====
function shuffle(items: Item[], epoch: number, globalSeed: number) { shuffleInPlace(items, new SeededRng(epoch + globalSeed)); }
function batch(b: number, rank: number, ws: number, bs: number, items: Item[]) {
  const trueBs = Math.min(bs, items.length - rank * bs - b * bs * ws);
  return { seqs: range(bs).map(i => sampleSequence(items[(b * bs * ws + rank * bs + i) % items.length].node, ...)), trueBs };
}
// items = rows of the task table (split) [or whole DB table for autocomplete] that HAVE a target cell (non-null, non-NaN)
```

---

## 11. Worked toy trace (for the demo)

Schema:

- `customers(customer_id PK, age:num, country:text)`, no time column.
- `products(product_id PK, price:num, category:text)`, no time column.
- `orders(order_id PK, customer_id FK, product_id FK, order_time:datetime [time col], amount:num)`.
- Task `cust-churn/{train,val,test}(timestamp [time col], customer_id FK, churn:bool)`.

Seed: test row `T = (2024-07-01, cust0, churn=?)`. Take w = 2.

1. Pop `(0,T)` from the stack. Push `(1,cust0)`. T has no children. Emit `timestamp of cust-churn` (DateTime) and `churn of cust-churn` (Boolean, **mask=true**). 2 tokens.
2. Pop `(1,cust0)`. Its children are orders o1 (05-03), o2 (06-20), o5 (08-15) and churn rows tr1 (01-01, Train), tr2 (04-01, Train), T (07-01, Test). o5 fails (08-15 > 07-01). The churn rows go straight into bucket 2: [tr1, tr2, T]. dbKids [o1, o2] (≤ w) are appended to bucket 2. Emit `age of customers`, `country of customers`. 4 tokens.
3. Stack empty → bucket 2, random pick, say o2. Push `(3,cust0)` and `(3,prodB)` (FK order customer_id, product_id). Emit `order_time of orders`, `amount of orders`.
4. Pop `(3,prodB)` (LIFO). prodB's children are o2, o3 (cust1, 02-11) and o6 (cust2, 09-01). o6 fails the time filter. Bucket 4 += [o2, o3]. Emit `price`, `category`.
5. Pop `(3,cust0)` → visited, skip. Random draw from bucket 2 → T → visited, skip (an RNG draw is still consumed). Draw tr2 → visit, push `(3,cust0)`, emit its 2 cells with **churn visible** (self label). The stack pop of cust0 → visited.
6. … bucket 2 drains (tr1, o1, then o1's product prodA, whose children go into bucket 4). Then bucket 4: o3 → stack `(5,cust1)`, `(5,prodB)`. prodB is visited; cust1 is new. cust1's children are its orders and its churn rows, **including cust1's own Test row at 2024-07-01 (same timestamp → passes `<=`)**. These go into bucket 6. The process continues until L tokens are emitted.

---

## 12. Edge cases checklist

- A seed without a timestamp → no temporal filtering at all, and the drop rule applies to every timestamp-less row of the seed's table.
- Children without timestamps always pass the filter. Parents are never filtered.
- Rows with zero feature cells (e.g. pure link tables) are visited and expand the frontier but emit **no tokens**. rel-event `user_friends` is given a `dummy` row-index column (pre.rs:268-271), presumably to avoid this.
- A row with more than 5 non-null FKs → panic (fly.rs:464).
- Duplicate frontier entries are expected; each one still costs an RNG draw when picked from a bucket.
- The final row may be truncated. If the frontier empties early, the sequence is padded.
- NaN or null target → the row is not a seed. NaN/null cells → not tokens.
- A constant boolean column → NaN values (std = 0 is not guarded for booleans, pre.rs:320-328).
- Wrapped items in the last batch: neutralized in eval, duplicated in training.
- A database file is mmapped once per task, and `visited` is sized to the whole database (fly.rs:364).

## 13. Notes for a faithful TypeScript demo

- Reproducing the algorithm exactly needs: a LIFO F→P stack with priority; depth buckets with uniform random pick; the `≤ seed_ts` filter on children only, skipped when either timestamp is null; width bound on DB children only; all same-task rows from any split; one masked cell (the seed's target); truncation at L; padding.
- Bit-exact parity with Rust's ChaCha12 `StdRng`, `random_range` and `index::sample` is not needed for didactic correctness. Say so in the UI ("any seeded PRNG; the real code uses ChaCha12").
- Normalization to show: per-column z-score for numbers and booleans; one global z-score over all datetime columns; Val/Test task tables reuse Train statistics; DB-table statistics use all rows.
- When showing a Val/Test seed, highlight that same-timestamp task rows of other entities can enter (paper-era behaviour). If the demo offers a "fixed" toggle, label it as MAIN behaviour.
