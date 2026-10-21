# 02 — RT preprocessing pipeline (paper-era code, commit 2a899c8)

Scope: how a RelBench database plus its task tables becomes the node/cell graph that RT's sampler reads. Everything below was checked against code. Where the paper and the code disagree, both are given.

## 0. Conventions and sources

- **PE** = paper-era snapshot `…/scratchpad/upstream/rt-paper-era` (commit `2a899c8`, 2026-02-04). `pre.rs` has not changed since the ICLR submission commit `eeccc2b` (2025-10-06). `git log 2a899c8 -- rustler/src/pre.rs` lists only `ab17c23` and `eeccc2b`, and `git diff eeccc2b 2a899c8 --stat` does not touch `pre.rs` or `common.rs`.
- **OFF** = current main `…/scratchpad/upstream/rt-official` (RT-J line). It is used only to explain later fixes.
- Paper = `Relational_Transformer.pdf` (arXiv 2510.06377v3). Page numbers are the printed page numbers.
- RelBench = v1.1.0 source on GitHub (`https://github.com/snap-stanford/relbench/blob/v1.1.0/...`).
- Crate versions from `PE/rustler/Cargo.lock`: polars-core 0.46.0 (l.1628-1629), half 2.5.0 (l.882-883), rkyv 0.8.10 (l.2352-2353).

## 1. Pipeline overview

The data-prep steps come from README.md:35-63.

1. `python scripts/download_relbench.py` calls `get_dataset(name, download=True)` and `get_task(name, task, download=True)` for every dataset and every task (scripts/download_relbench.py:4-11). Files land in `~/.cache/relbench/<db>/{db/*.parquet, tasks/<task>/{train,val,test}.parquet}`, which you symlink as `~/scratch/relbench` (README.md:43-47).
2. `cargo run --release -- pre <db>` dispatches `Command::Pre` to `pre::main` (main.rs:13-24). It reads `~/scratch/relbench/<db>` (pre.rs:68) and writes `~/scratch/pre/<db>/` (pre.rs:619-620):
   - `text.json`: the deduplicated string table (pre.rs:622-630).
   - `text_map.json`: string → index (pre.rs:632-639).
   - `column_index.json`: `"<col> of <table>"` → index (pre.rs:641-645).
   - `table_info.json`: `"<table>:<Db|Train|Val|Test>"` → `{node_idx_offset, num_nodes}` (pre.rs:648-664).
   - `nodes.rkyv`: one rkyv-archived `Node` per row, concatenated (pre.rs:676-683).
   - `offsets.rkyv`: byte offsets, N+1 entries starting at 0 (pre.rs:668, 681, 686-690).
   - `p2f_adj.rkyv`: one `Vec<Edge>` of children per node (pre.rs:692-696).
3. `python -m rt.embed <db>` runs all-MiniLM-L12-v2 over `text.json` and writes `text_emb_all-MiniLM-L12-v2.bin` as a raw bf16 `[len(text.json), 384]` matrix (embed.py:55-72).

Data structures (common.rs):
- `TableInfo{node_idx_offset:i32, num_nodes:i32}` (l.6-10).
- `TableType{Db,Train,Val,Test}` (l.14-20).
- `Edge{node_idx, table_name_idx, table_type, timestamp:Option<i32>}` (l.24-29).
- `SemType{Number=0, Text=1, DateTime=2, Boolean=3}` (l.38-44). The order matches model.py's loop `["number","text","datetime","boolean"]` (model.py:215, 239).
- `Node{is_task_node, node_idx, f2p_nbr_idxs:Vec<i32>, f2p_edges:Vec<Edge>, timestamp:Option<i32>, table_name_idx, col_name_idxs, sem_types, number_values:Vec<f32>, text_values:Vec<i32>, datetime_values:Vec<f32>, boolean_values:Vec<f32>, class_value_idx:Vec<i32>}` (l.48-62). The seven per-cell vectors are parallel, with one entry per cell.

## 2. What RelBench hands to `pre.rs`

- **Parquet key-value metadata.** RelBench `Table.save` stores `fkey_col_to_pkey_table`, `pkey_col` and `time_col`, each JSON-encoded (RelBench `base/table.py`, save/load). `pre.rs` parses exactly these three keys (pre.rs:118-150): `pkey_col` is null or a string, `fkey_col_to_pkey_table` is an object, and `time_col` is null or a string.
- **Keys are row positions.** `Database.reindex_pkeys_and_fkeys()` (RelBench `base/database.py`) does three things:
  - It sorts each table that has a `time_col` by that column.
  - It replaces each pkey with its row position as nullable `Int64` (`pd.RangeIndex(len).astype("Int64")`).
  - It remaps each FK by left merge, so a dangling FK becomes `pd.NA`.

  `pre.rs` relies on this: an FK value *is* the parent's row number (`pnode_idx = ptable_offset + val`, pre.rs:477). Only `AnyValue::Int64` is accepted (pre.rs:472-476). OFF commit `9f0e027` confirms it: "The preprocessor read every foreign-key value as the parent's row position and never looked at `pkey`. That is right for RelBench, whose keys are 0..n-1 in row order".
- **The cached DB is the full DB.** RelBench `get_db` loads `cache_dir/db`, or builds it, reindexes it and saves it there, and applies `db.upto(test_timestamp)` only in memory afterwards (RelBench `base/dataset.py`, get_db). The parquet files `pre.rs` reads therefore contain **all rows, including rows after the val and test timestamps**.
- **Task parquet files keep their labels, including test.** `get_table(split)` caches `{split}.parquet` unmasked and strips the target only on return, by default only for test (RelBench `base/task_base.py`, get_table/_mask_input_cols). `evaluate()` itself reloads `get_table("test", mask_input_cols=False)` (RelBench `base/task_entity.py`). `pre.rs` reads the file directly, so **test labels are present** and RT uses them for its test AUROC/R² (main.py:283-288, 331-337).
- **Task table columns.** RelBench entity tasks build a `Table(df, fkey_col_to_pkey_table={entity_col: entity_table}, pkey_col=None, time_col=<time col>)`. For example, rel-f1 `driver-dnf` has columns `date` (time), `driverId` (FK → drivers) and `did_not_finish` (int 0/1), in that SELECT order, with a 30-day timedelta (RelBench `tasks/f1.py`).

## 3. Discovering tables; the global node index

- **Glob order.** DB tables come from `glob("<ds>/db/*.parquet").sorted()`, then task tables from `glob("<ds>/tasks/*/*.parquet").sorted()` (pre.rs:79-92). Paths sort component-wise, so within a task the order is `test`, `train`, `val`.
- **Recommendation tasks are skipped by counting dashes.** `dashes` = number of `-` in the dataset path (pre.rs:70). Any parquet path with exactly `dashes+2` dashes is skipped (pre.rs:93-95). This drops two-dash task directories such as `user-item-purchase`, `user-ad-visit` and `post-post-related` (the link-prediction tasks), and keeps one-dash tasks such as `user-churn`. OFF CHANGELOG 1.9.0 later describes this as a deliberate rule: "recommendation (formerly `link_prediction`) tasks are skipped" (`git show 4af9ff1:CHANGELOG.md` l.111).
- **Names and types.**
  - A DB table's name is the file stem, with type `Db`.
  - A task table's name is the parent directory, i.e. the task name such as `driver-dnf`. Its type is `Train`, `Val` or `Test` from the stem; anything else panics (pre.rs:152-174).
  - The map key is `(name, type)` (pre.rs:175).
- **Offsets.** One global index space covers DB and task rows. `node_idx_offset` is the running sum of row counts in glob order (pre.rs:291-306); `num_rows_sum` is the total. `global node idx = node_idx_offset(table) + row position r` (pre.rs:466, 537).
- **`table_info.json`.** The key is `format!("{}:{:?}", name, type)`, for example `"drivers:Db"`, `"driver-dnf:Train"`, `"driver-dnf:Val"`, `"driver-dnf:Test"`. The value is `{node_idx_offset, num_nodes = df.height()}` (pre.rs:650-660).
- **`node_vec`.** It holds one `Node::default()` per global row (pre.rs:408-410), and `p2f_adj` holds one empty Vec per row (pre.rs:411-413). A row whose only columns are pkey/FK (a "key-only" table) still gets a node, but that node has **zero cells** (see §5). Such nodes contribute no tokens; the sampler still traverses through them, because fly.rs follows their edges at l.406-452.

## 4. Hard-coded per-dataset transforms (pre.rs:177-289)

| DB | table | transform | line |
|---|---|---|---|
| rel-stack | postLinks | `LinkTypeId` → bool via `make_column_boolean` | 179-181 |
| rel-trial | studies | `has_dmc` → bool via `make_column_boolean` | 187-189 |
| rel-trial | eligibilities | `adult`, `child`, `older_adult` → bool via `make_column_boolean` | 197-201 |
| rel-amazon | user-churn, item-churn (task) | `churn` cast to Boolean | 210-215 |
| rel-amazon | product | `category` (list) → first element as a string (`XXX: takes only the first element`) | 216-227 |
| rel-stack | user-engagement / user-badge (task) | `contribution` / `WillGetBadge` cast to Boolean | 232-237 |
| rel-stack | posts | **drop `AcceptedAnswerId`** (an undeclared ID column) | 238-240 |
| rel-trial | study-outcome (task) | `outcome` cast to Boolean | 244-246 |
| rel-f1 | driver-dnf / driver-top3 (task) | `did_not_finish` / `qualifying` cast to Boolean | 249-256 |
| rel-hm | user-churn (task) | `churn` cast to Boolean | 259-261 |
| rel-event | event_attendees | `drop_nulls` (drops any row with any null) | 265-267 |
| rel-event | user_friends | `drop_nulls`, then **add a UInt32 row-index column `dummy`** | 268-271 |
| rel-event | user-repeat, user-ignore, **user-attendance** (task) | drop `index`, cast `target` to Boolean | 272-278 |
| rel-avito | user-visits / user-clicks (task) | `num_click` cast to Boolean | 282-289 |
| rel-avito | all tables | FK `UserID`→`UserInfo`, `AdID`→`AdsInfo` override | 133-139 |

Notes on the table:

- `cast_col_to_bool` is a polars `cast(Boolean)` (pre.rs:61-65), so a nonzero number becomes true.
- `make_column_boolean` casts to string and sets `value == value_of_row_0` (pre.rs:21-32):
  - The comment says "first non-null", but the code uses `ca.get(0)`, which panics if row 0 is null. OFF's later legacy transform uses the first non-null value instead (`OFF/src/rt/preprocess/legacy/main.py:42-51`).
  - Nulls stay null, because comparison propagates null.
  - So "true" means "equals whatever value row 0 has". Paper Table 8 (p.19) lists LinkTypeId "1 vs 3" and has_dmc/adult/child "t vs f".
- **Surprise: `user-attendance` is a regression task in rt/tasks.py:25, yet its `target` is cast to Boolean** (pre.rs:272-277). Its cells therefore get `SemType::Boolean`. In pretraining it gets the boolean head and the BCE loss (model.py:252-256), effectively "attends ≥1 event". OFF's legacy `CAST_TO_BOOL` lists only `user-repeat` and `user-ignore` for rel-event (`OFF/src/rt/preprocess/legacy/main.py:15-28`). The paper never evaluates rel-event (p.6 §4.1), but rel-event tasks are in the pretraining mix (tasks.py:11,14,25,63,87).
- The `dummy` column exists because `user_friends` is key-only (FKs `user`, `friend` → users, no time column, per RelBench `datasets/event.py`). Without it the rows would have no cells. OFF CHANGELOG 1.9.0 later lists "key-only tables are visible to the model" as a change (`git show 4af9ff1:CHANGELOG.md` l.98).
- The rel-amazon `category` uses `AnyValue::to_string()`. polars 0.46 formats `AnyValue::String(v)` as `"\"{v}\""` (polars-core rs-0.46.0 `src/fmt.rs`), so the stored text keeps **literal double quotes**, e.g. `"Books"` with the quotes. An empty list would panic on `.next().unwrap()`.
- The avito override is a no-op on RelBench v1.1.0, which already maps `UserID`→`UserInfo` and `AdID`→`AdsInfo` (RelBench `datasets/avito.py`).

## 5. Column → cell conversion (question 1)

For each table, in df column order (pre.rs:429), the rules below apply. If `--skip-db` is set, DB tables are skipped entirely (pre.rs:416-422; the default is false, pre.rs:57-58).

1. `table_name_idx = text_id(table_name)` (pre.rs:424-427). Train, Val and Test share the task name, so they share the same `table_name_idx`.
2. For every column, primary key included, `col_name_idx = text_id("<col> of <table>")`. It is also appended to `column_name_to_idx` the first time the string is inserted (pre.rs:432-437). So PK and FK column names *are* embedded and appear in `column_index.json`, although they never become cells.
3. **The PK column becomes nothing**: no cell and no edge (pre.rs:439-442). Task tables have `pkey_col=None`, so nothing is skipped there.
4. **An FK column becomes an edge, not a cell** (pre.rs:444-533). For each row r:
   - A null value is skipped (l.462-464). Otherwise:
   - set `node.is_task_node = type != Db`, `node.node_idx`, `node.table_name_idx` (l.466-470);
   - `pnode = offset(ptable:Db) + val` (l.452-459, 477);
   - push `pnode` to `f2p_nbr_idxs` (l.479);
   - set `node.timestamp` = own time-column value in **seconds** (`ns / 1e9`, integer division, then `as i32`), or None (l.481-493);
   - push the F→P `Edge{pnode, text_id(ptable), Db, parent's own time-col seconds}` to `f2p_edges` (l.495-521);
   - push the P→F `Edge{node_idx, own table_name_idx, own TableType, own timestamp}` to `p2f_adj[pnode]` (l.523-529).
5. **Every other column becomes one cell per non-missing row** (pre.rs:535-612). This includes the **time column**, which becomes a **datetime cell**, and non-key integer ID columns, which become **number cells** (§6).
   - The node header fields are set for every row, even when the value is null (l.537-541).
   - `AnyValue::Null` → no cell (l.552).
   - Float NaN → no cell (l.566-568).
   - So **missing values are dropped from the row; they are neither imputed nor masked**. This matches the paper's "all of its non-missing feature cells … (i.e., cells not from primary- or foreign-key columns)" (p.4 §3.1; p.16 App. A).

**Consequence: `node.timestamp` is only assigned inside the FK branch** (pre.rs:493). A row of a table that has a `time_col` but no non-null FK gets `timestamp = None`. Examples are rel-event `users` (time col `joinedAt`, no FKs, per RelBench `datasets/event.py`) and any FK-less table with a time column. This matters for the sampler's temporal filter (§12).

## 6. Type inference: physical dtype → SemType (question 2)

The type is chosen by polars `AnyValue` per value. There is no semantic inference (pre.rs:543-611).

| polars dtype (after §4 transforms) | SemType | value stored | other value slots |
|---|---|---|---|
| `Boolean` | Boolean (3) | `boolean_values = (b?1:0 − μ)/σ` | number 0.0, text 0, datetime 0.0, class_value_idx −1 (l.553-564) |
| `UInt32`, `Int32`, `Int64`, `Float32`, `Float64` (converted to f64, l.543-550) | Number (0) | `number_values = (v − μ)/σ` | l.565-583 |
| `Datetime(ns)` | DateTime (2) | `datetime_values = (t_ns − μ_T)/σ_T` | l.584-594 |
| `String` | Text (1) | `text_values = text_id(value)` | **`class_value_idx = text_id(value)`** (l.595-605) |
| any other dtype with a non-null value (Int8/16, UInt8/16/64, Date, Duration, Categorical, List, Binary…) | — | `panic!()` | l.606-611 |

What this means in practice:

- **Categorical and string columns are text.** Each value is embedded by MiniLM as the raw string only, with no column context (embed.py:65-68). The `class_value_idx` slot equals the text index for text cells and −1 otherwise. It is carried through the sampler (fly.rs:471) but **never used by the paper-era model**: there are no references in model.py or main.py. Text cells are never masked: `"masking text not supported"` (model.py:257-258), and the paper says the same (p.6 §3.3).
- **Integer IDs that are not declared keys are ordinary numbers** and get z-scored. Examples are rel-f1 `results.statusId`, `number` and `rank`, and rel-stack `users.AccountId` (shown as `12827220.0` in paper p.25). The only exception is the hand-dropped `posts.AcceptedAnswerId` (pre.rs:238-240). OFF later added an "identifier-column policy" (`git show 4af9ff1:CHANGELOG.md` l.98-99).
- **Booleans** are native `Boolean` columns, or columns converted by §4. A 0/1 integer column that is not listed in §4 stays a **Number**. Paper Table 8 marks rel-avito `SearchInfo.IsUserLoggedOn` and rel-event `event_interest.not_interested` with "Pos./Neg. values N/A". The code does not cast them, so if they are integers in RelBench, these "classification" autocomplete tasks are trained as Number/Huber. Their dtype could not be verified here (no data on disk).

## 7. Normalization (question 3)

Statistics are computed **per (table, type) and per column position**. The loop is `for table in table_map.values_mut() { for col in table.df.iter() … push ColStat }` (pre.rs:316-373). There is no clipping anywhere.

### Number columns (pre.rs:330-341)
```
x = column cast to f64, drop nulls, drop NaN
μ = mean(x), or 0.0 if empty
σ = sample std (ddof=1) of x, or 1.0 if None (n ≤ 1, per polars 0.46 VarState::finalize: weight <= ddof → None)
if σ == 0.0: σ = 1.0
cell value r = (v − μ) / σ        (pre.rs:569)
if r is ±inf: panic              (pre.rs:570-575; only possible if v itself is ±inf)
```

### Boolean columns (pre.rs:320-329)
```
x = column cast to f64 (true=1, false=0), drop nulls      (no NaN filter)
μ = mean(x), or 0.0
σ = sample std (ddof=1), or **0.0** if None             (no zero guard!)
r = ((b ? 1 : 0) − μ) / σ                              (pre.rs:554-555)
```
- With 0<μ<1, true maps to `(1−μ)/σ > 0` and false to `−μ/σ < 0`. So **`y > 0` ⇔ true**, which is the BCE target `(y>0).float()` (model.py:253-255) and the AUROC label `int(x>0)` (main.py:336). The paper writes `BCE(1{r > 0}, r′)` (p.6 §3.3).
- Edge cases differ from numbers. A constant column has σ=0, which gives NaN for the matching value (0/0) and ±inf for the other. A single non-null value gives σ=None→0.0, which gives NaN. **Nothing guards this**; there is no inf/NaN check in the boolean branch. OFF fixed it later with `unwrap_or(1.0)` and `if std==0 {1}` (OFF `rustler/src/pre.rs:577-580`; CHANGELOG 1.9.0 l.107-109, "constant columns are never normalized by a zero standard deviation").
- Worked value: a train label column with 25% positives has μ=0.25 and σ=0.5, so true → 1.5 and false → −0.5.

### Datetime columns (one global pair) (pre.rs:342-366, 375-376)
```
for every Datetime column of EVERY table in table_map (all DB tables + Train + Val + Test task tables):
    x = physical i64 nanoseconds as f64, drop nulls, drop NaN
    dt_cnt += len(x); dt_sum += Σx; dt_sum_sq += Σx²
μ_T = dt_sum / dt_cnt
σ_T = sqrt(dt_sum_sq/dt_cnt − μ_T²)          ← POPULATION std (ddof=0), no zero guard
r = (t_ns − μ_T) / σ_T                       (pre.rs:586)
```
- `assert!(unit == Nanoseconds)` (l.343, 585). The ColStat pushed for datetime columns is a dummy `{0,0}` (l.362-365).
- z-scores do not depend on the unit, so "seconds" in the paper (p.5) and nanoseconds give the same r, up to f64 rounding.
- `node.timestamp` (seconds, i32) is separate. It is used only by the sampler's temporal filter, never as a model input.

### Which rows feed the statistics
- **DB tables:** **all rows of the full RelBench DB** (§2), including rows after `val_timestamp` and `test_timestamp`. No cutoff is read anywhere in PE: there are no occurrences of `cutoff`, `val_timestamp`, `test_timestamp` or `upto` in `rt/`, `rustler/src/` or `scripts/`.
- **Task tables:** the number and boolean stats of `Val` and `Test` are **overwritten with the stats of the same task's `Train` table**, by name lookup (pre.rs:383-396). **Task labels are therefore normalized with train-split statistics**, as the paper says.
- **Datetime:** global over everything. That includes the timestamp column of the Val and Test task tables, and all DB datetime cells of every period.
- **Paper claim** (p.4-5 §3.1): "µc and σc are the column mean and standard deviation computed on the training split" and "global mean and standard deviation of timestamps in the training split". In the code this holds **only** for task-table number and boolean columns. It does **not** hold for DB columns or datetime. See §13 for the authors' own acknowledgement.

### Precision downstream (fly.rs:482-490)
- Per-cell f32 values are converted to **bf16** (`half::bf16::from_f32`, round-to-nearest-even) when the sampler builds a batch. data.py then reinterprets them as `torch.bfloat16` (data.py:120-127, 146-148).
- bf16 has 8 significand bits. At |z|∈[1,2) the datetime grid is 2⁻⁷σ_T. With σ_T ≈ 3 years that is about 8.5 days, so nearby dates collapse to the same input. Labels used in the loss are bf16 too.

## 8. Text: dictionary, embeddings, column names (question 4)

- **One global dictionary for the whole DB:** `text_to_idx: HashMap<String,i32>`, where `text_id(s)` returns the existing index or `len` (pre.rs:406). It holds:
  - every table name (pre.rs:424-427, 495-498);
  - every column phrase `"<col> of <table>"` (pre.rs:432-437);
  - every distinct string cell value (pre.rs:596-597).

  All of them are deduplicated against each other. The indices follow HashMap iteration order over `table_map`, which is deterministic because `BuildHasherDefault<DefaultHasher>` is used (pre.rs:75), but it is not alphabetical. The semantics do not depend on the order.
- `text.json` is a JSON array where index i holds string i (pre.rs:624-630). `text_map.json` is the inverse (pre.rs:633-639). `column_index.json` is a `{"<col> of <table>": idx}` map built only from **newly inserted** column phrases (pre.rs:407, 434-436, 642-645). data.py resolves task target and drop columns through it: `get_column_index(col, table, db)` looks up `f"{column_name} of {table_name}"` (data.py:31-43).
- **Embedding** (embed.py):
  - `SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")`, in bf16 weights on CUDA and fp32 on CPU since commit 7d564f7 (embed.py:14-21).
  - `encode(text_list, batch_size=8192, convert_to_numpy=True)` (embed.py:24-31, 37).
  - The output is `np.stack(...).astype(bfloat16).tofile(...)` with no header (embed.py:70-72).
  - The model's module list is Transformer → Pooling → **Normalize** (HF `all-MiniLM-L12-v2/modules.json`), so vectors are unit-L2-norm before the bf16 cast.
  - `max_seq_length = 128` (HF `sentence_bert_config.json`), so long texts are truncated to 128 word-pieces.
  - d_text is 384.
- **Lookup at sample time:** `get_text_emb(idx) = emb[idx*384 : (idx+1)*384]` (fly.rs:528-532).
  - Each cell receives `col_name_values = emb[col_name_idx]` (fly.rs:472-477) and `text_values = emb[text_values[cell]]` (fly.rs:484-486).
  - Non-text cells store `text_values = 0` (pre.rs:559, 578, 589), so they carry the embedding of `text.json[0]`, which is some table name. The model zeroes it out with the `sem_types==text` mask (model.py:215-220).
- **Table names are not separately embedded as inputs.** `table_name_idx` is used only as an integer:
  - for the column-attention mask, same `col_name_idx` AND same `table_name_idx` (model.py:181-183);
  - for the sampler's task-table filter (fly.rs:417).

  The table name enters the token only through the phrase `"<col> of <table>"`, for example `"churn of user-churn"`, `"date of driver-dnf"` or `"dob of drivers"`. The paper describes this as "price of product" (p.5).

## 9. Task tables (question 5)

- Each RelBench task becomes three tables: `("<task>", Train|Val|Test)`. Their `table_info.json` keys are `"<task>:Train"`, `"<task>:Val"` and `"<task>:Test"`. Each split has its own `node_idx_offset` and `num_nodes` and occupies a contiguous block of global node ids after all DB tables (§3).
- **Columns of each task row:**
  - The **timestamp** (`time_col`, e.g. `date` or `timestamp`) is a **datetime cell** with column phrase `"<time col> of <task>"`. It is also the source of `node.timestamp`, in seconds.
  - The **entity id** (the only FK) produces no cell. It produces `f2p_nbr_idxs=[entity node]` and an F→P edge, and it registers a P→F edge on the entity whose `table_type` is Train, Val or Test (pre.rs:523-529).
  - The **label** is a boolean cell for classification tasks after the casts in §4, or a number cell for regression. It is normalized with the **Train** split's μ and σ (pre.rs:383-396).
  - A typical task node therefore has exactly 2 cells: `[timestamp(datetime), label]`.
- `is_task_node = true` for every row of a task table (pre.rs:468, 539).
- **Test labels are present** (§2). The sampler masks only the seed's own target cell: `is_target = (node_idx == seed) && (col_name_idx == target_col)`, and `masks = is_target` (fly.rs:492-496).
- The splits share `table_name_idx` and `col_name_idx`. Column attention therefore links label cells across Train, Val and Test, and the sampler's task-row rule `edge.table_name_idx == seed.table_name_idx` admits rows of **any split of the same task** (fly.rs:415-419).
- **Seeds:** `RelationalDataset` maps `split` to "Train", "Val" or "Test" and reads `table_info["<task>:<Split>"]` (data.py:63-83). The sampler then keeps only the nodes in `[offset, offset+num_nodes)` whose `col_name_idxs` contain the target column (fly.rs:287-296). A row with a null or NaN label has no target cell, so it is **not a seed**.

## 10. Autocomplete tasks (question 6)

- No preprocessing is specific to autocomplete. The target is an ordinary feature column of a DB table, and it is z-scored with **full-table** stats (§7). Some columns are binarized first (§4).
- The task tuples are `(db, table, target_col, leakage_cols)` (tasks.py:30-88). For a DB table, data.py prefers the key `"<table>:Db"` **regardless of the requested split** (data.py:77-81). So train, val and test seeds for autocomplete are **all non-null rows of the table, from every time period**.
- In PE, autocomplete tasks are only used for training:
  - `eval_tasks` are forecasting only (example_pretrain.py:17-18; paper p.6: "We evaluate only on the forecasting tasks").
  - Continued pretraining on the target DB includes that DB's autocomplete tasks over **all** rows (example_contd_pretrain.py:90-92). For rel-amazon that means `review.verified` and `review.rating`, including rows from the val and test periods.
- Leakage columns (`columns_to_drop`) are not removed in preprocessing. The sampler skips them for the seed row and for **any node whose `timestamp` equals the seed's** (fly.rs:454-461). Because `None == None`, a seed with no timestamp (§5) drops those columns from every timestamp-less node in the context. Paper App. D (p.19): "identify potential sources of information leakage and discard these columns on the fly".
- The target column's other rows stay visible (unmasked). They act as in-context labels. `is_task_nodes` is set true for any cell in the target column (fly.rs:498-499), but the model never reads `is_task_nodes`; only `scripts/visualize_rustler.py:22,53` does.

## 11. f2p neighbour lists; maximum foreign keys (question 7)

- `f2p_nbr_idxs` holds the **global node indices of the parent rows**, one per **non-null** FK column, in df column order (pre.rs:444-479). If two FKs point to the same parent row, the list has duplicates.
- `f2p_edges` holds the same parents as `Edge` structs: parent table name idx, `Db`, and the parent's own time-col seconds. That parent timestamp is not used by the paper-era sampler.
- `p2f_adj[parent]` holds the child edges, carrying the **child's** timestamp and the child's table type.
- **The maximum is 5**: `const MAX_F2P_NBRS = 5` (fly.rs:25), `assert!(node.f2p_nbr_idxs.len() <= MAX_F2P_NBRS)` (fly.rs:464, added in commit 77bdb9b). The batch tensor is `[B, S, 5]` padded with −1 (fly.rs:69; data.py:143). `pre.rs` itself does not limit the count.
- Every cell of a row repeats the row's full `f2p_nbr_idxs` (fly.rs:465-467). The attention masks use it: `kv_in_f2p` / `q_in_f2p` (model.py:168-176). The paper describes this as Row(k) ∈ OutLinks(q) (p.5).

## 12. Temporal aspects (question 8)

- **No DB cutoff in preprocessing.** RelBench cutoffs exist, e.g. rel-f1 val 2005-01-01 / test 2010-01-01 (RelBench `datasets/f1.py`); rel-avito 2015-05-08 / 2015-05-14 (`datasets/avito.py`); rel-event 2012-11-21 / 2012-11-29 (`datasets/event.py`). `pre.rs` never reads them, and no row is dropped by time. Post-test rows are in `nodes.rkyv` and influence the normalization statistics (§7).
- **Time is cut only per seed, at sampling time** (fly.rs:415-437). A P→F child edge is skipped when `edge.timestamp.is_some() && seed.timestamp.is_some() && edge.timestamp > seed.timestamp` (l.422). That means:
  - **equal timestamps are allowed**;
  - children without a timestamp are always allowed;
  - a seed without a timestamp disables the filter entirely.
- F→P parents are always followed with no time check (l.406-408).
- Task-table children are admitted only from the seed's own task (l.417), but from any split (§9).
- Paper (p.6 §4.1): "We pretrain and fine-tune only on the training splits of the tasks, using database rows only up to the training cutoff timestamps". The code has no explicit DB cutoff. For train forecasting seeds, the per-seed filter implies "≤ seed time ≤ val_timestamp − Δ" for P→F-reached rows, but the code never enforces "≤ train cutoff" for F→P parents, for timestamp-less rows, or for autocomplete seeds.
- **Same-timestamp task rows.** Rows of other entities at the **same** timestamp as the seed (e.g. other test rows at `test_timestamp`) pass the `>` test, and their labels are unmasked in context. OFF later made task rows "strictly past" so that "a same-horizon task row is invisible" (`git show 4af9ff1:CHANGELOG.md` l.103-104; OFF `rustler/src/fly.rs:1724-1726, 1893-1900`, `same_horizon_task_row`). Paper Table 12 (p.23) reports nonzero "Other entity labels" for several tasks. The sampler is covered in detail by another report; it is mentioned here because it depends on how `pre.rs` stores timestamps.

## 13. The normalization caveat (OFF commits 358bfd3, 5fef314)

- **What was wrong.** The parent revision of 358bfd3 (`git show 358bfd3^:docs/cards/relbench-preprocessed.md`, l.101-117) says that `rustler` PR #4 (commit `8030aa8`, "column stats from the train period only", merged 2026-10-01) fixed temporal leakage. Before it, "per-column z-scoring statistics and the global datetime statistics were computed over the validation and test rows as well as the train period, so training and inference inputs were normalized with statistics that had seen the future". The parent of 5fef314 (`reproduce/README.md` l.184-190) says the statistics "used to be computed over all rows of every table, which let the validation and test periods set the scale that training inputs were normalized with".
- **Size of the change.**
  - On the data: "34 of the 50 database tables are time-indexed" and "27 of 40 numeric columns shift by more than 0.1 of their old standard deviation, the largest by 1.43".
  - On results, measured with RT-J, one seed and 7 tasks: mean −0.045 AUROC and −0.035 nMAE, every task within 0.51, mixed sign (`358bfd3^:docs/downloads.md` l.102-121).
  - The `legacy/` tree, which feeds RT-v1 (the paper model), "changes under the fix as well". Its effect on RT-v1 "Nobody has measured".
- **What the two commits did.** 358bfd3 rewrote the cards into neutral wording ("Since PR #4 … the statistics come from the train period alone"). 5fef314 "drop[ped] the normalization write-up from public docs", reducing it to "pin the revision, expect a few tenths".
- **This matches PE's `pre.rs` exactly.** DB column stats use all rows (pre.rs:316-341), and the datetime accumulators include every DB row and the Val/Test task tables (pre.rs:342-357). The boolean zero-std issue is a separate, unguarded edge case (§7).
- **The fix in OFF `rustler/src/pre.rs`, for contrast:**
  - Val/Test task tables are skipped in the stats loop (l.547-549).
  - DB tables that have a time column are filtered to `time ≤ val_timestamp` before computing stats (l.534-542, 551-570).
  - The boolean std defaults to 1 and is guarded against 0 (l.577-580).
  - Datetime uses Welford, still population std, with `dt_std = 1.0` if `cnt ≤ 1` (l.619-626, 640-644).
  - Tables **without** a time column still use all rows, as they must.

## 14. Paper vs code: summary of discrepancies

| topic | paper | PE code |
|---|---|---|
| µc, σc | "computed on the training split" (p.4) | DB columns: all rows of the full DB. Task labels: Train split ✔. Sample std (ddof=1). Numbers: σ=0→1. Booleans: σ=0 not guarded. |
| datetime µT, σT | "timestamps in the training split" (p.5) | every datetime cell of every table and split (DB + Train + Val + Test). Population std. |
| DB cutoff | rows "only up to the training cutoff timestamps" (p.6) | no cutoff in preprocessing; only the per-seed `>` filter on P→F edges at sampling time |
| feature columns | "numeric, text, and datetime" (p.3) | also boolean; the time column is a datetime cell; non-key integer IDs are numbers; strings/categoricals are text |
| missing values | "non-missing feature cells" (p.4, p.16) | ✔ null and NaN dropped (no cell) |
| PK/FK | not cells (p.16) | ✔ PK ignored; FK → `f2p_nbr_idxs` + edges |
| text | frozen encoder (p.5) | MiniLM-L12-v2, unit-normalized, 128-token truncation, stored bf16 |
| user-attendance | regression task (tasks.py:25) | label cast to Boolean, so BCE in pretraining |
| autocomplete splits | "sampling starts from the masked database row" (p.19) | all rows of the table for every split (data.py:77-81) |

## 15. TypeScript re-implementation (mirrors pre.rs semantics)

```ts
// ---------- types (common.rs) ----------
enum SemType { Number = 0, Text = 1, DateTime = 2, Boolean = 3 }
type TableType = "Db" | "Train" | "Val" | "Test";
interface Edge { node_idx: number; table_name_idx: number; table_type: TableType; timestamp: number | null } // seconds
interface Node {
  is_task_node: boolean; node_idx: number; table_name_idx: number;
  timestamp: number | null;            // seconds (i32), set ONLY in the FK branch
  f2p_nbr_idxs: number[]; f2p_edges: Edge[];
  col_name_idxs: number[]; sem_types: SemType[];
  number_values: number[]; text_values: number[]; datetime_values: number[]; boolean_values: number[];
  class_value_idx: number[];           // text idx for Text cells, -1 otherwise (unused by model)
}
type DType = "Boolean" | "Int64" | "Int32" | "UInt32" | "Float32" | "Float64" | "Datetime" | "String";
interface RawTable {
  name: string; type: TableType;            // DB: file stem; task: task dir name + split
  pkey: string | null; fkeys: Record<string, string>; tcol: string | null;
  cols: string[]; dtypes: Record<string, DType>;
  rows: (number | bigint | boolean | string | null)[][]; // Datetime as bigint nanoseconds; FK/PK as row positions
}
// ---------- 1. order + offsets (pre.rs:79-95, 291-306) ----------
// DB tables sorted by name, then for each task (sorted): test, train, val. Skip 2-dash task names.
let off = 0; for (const t of tables) { t.offset = off; off += t.rows.length; }
const key = (t) => `${t.name}:${t.type}`;   // table_info.json key (pre.rs:652)
// ---------- 2. column stats (pre.rs:310-396) ----------
const mean = (a) => a.length ? a.reduce((s, x) => s + x, 0) / a.length : null;
const stdSample = (a) => { if (a.length <= 1) return null;          // polars std(1): None when n <= ddof
  const m = mean(a); return Math.sqrt(a.reduce((s, x) => s + (x - m) ** 2, 0) / (a.length - 1)); };
let dtCnt = 0, dtSum = 0, dtSumSq = 0;
for (const t of tables) t.stats = t.cols.map((c, j) => {           // ALL rows, ALL tables incl. Val/Test
  const v = t.rows.map(r => r[j]).filter(x => x !== null);
  switch (t.dtypes[c]) {
    case "Boolean": { const f = v.map(b => b ? 1 : 0);
      return { mean: mean(f) ?? 0, std: stdSample(f) ?? 0 }; }      // NOTE: no zero guard
    case "Int64": case "Int32": case "UInt32": case "Float32": case "Float64": {
      const f = v.map(Number).filter(x => !Number.isNaN(x));
      let s = stdSample(f) ?? 1; if (s === 0) s = 1; return { mean: mean(f) ?? 0, std: s }; }
    case "Datetime": for (const x of v) { const f = Number(x); dtCnt++; dtSum += f; dtSumSq += f * f; }
      return { mean: 0, std: 0 };
    default: return { mean: 0, std: 0 };
  }
});
const dtMean = dtSum / dtCnt;
const dtStd = Math.sqrt(dtSumSq / dtCnt - dtMean * dtMean);       // population std, no zero guard
for (const t of tables) if (t.type === "Val" || t.type === "Test")
  t.stats = tables.find(x => x.name === t.name && x.type === "Train")!.stats;
// ---------- 3. text dictionary (pre.rs:406, 424-437, 495-498, 596-597) ----------
const text = new Map<string, number>(); const columnIndex = new Map<string, number>();
const tid = (s: string) => { if (!text.has(s)) text.set(s, text.size); return text.get(s)!; };
// ---------- 4. nodes + edges (pre.rs:408-615) ----------
const nodes: Node[] = Array.from({ length: off }, () => emptyNode());
const p2f: Edge[][] = Array.from({ length: off }, () => []);
for (const t of tables) {                                          // Rust: HashMap order (only affects text ids)
  const tni = tid(t.name);
  t.cols.forEach((c, j) => {
    const phrase = `${c} of ${t.name}`;
    if (!text.has(phrase)) { columnIndex.set(phrase, text.size); }
    const cni = tid(phrase);
    if (c === t.pkey) return;                                       // PK: nothing
    if (c in t.fkeys) {                                             // FK: edges only
      const p = tables.find(x => x.name === t.fkeys[c] && x.type === "Db")!;
      t.rows.forEach((r, ri) => { const v = r[j]; if (v === null) return;
        const n = nodes[t.offset + ri]; n.is_task_node = t.type !== "Db"; n.node_idx = t.offset + ri; n.table_name_idx = tni;
        const pidx = p.offset + Number(v); n.f2p_nbr_idxs.push(pidx);
        const ts = t.tcol ? toSec(r[t.cols.indexOf(t.tcol)]) : null; n.timestamp = ts;
        const pts = p.tcol ? toSec(p.rows[Number(v)][p.cols.indexOf(p.tcol)]) : null;
        n.f2p_edges.push({ node_idx: pidx, table_name_idx: tid(p.name), table_type: "Db", timestamp: pts });
        p2f[pidx].push({ node_idx: n.node_idx, table_name_idx: tni, table_type: t.type, timestamp: ts });
      });
      return;
    }
    const st = t.stats[j];
    t.rows.forEach((r, ri) => {
      const n = nodes[t.offset + ri]; n.is_task_node = t.type !== "Db"; n.node_idx = t.offset + ri; n.table_name_idx = tni;
      const v = r[j]; if (v === null) return;                       // missing -> no cell
      const push = (sem, num = 0, txt = 0, dt = 0, bool = 0, cls = -1) => {
        n.sem_types.push(sem); n.col_name_idxs.push(cni); n.number_values.push(num);
        n.text_values.push(txt); n.datetime_values.push(dt); n.boolean_values.push(bool); n.class_value_idx.push(cls); };
      switch (t.dtypes[c]) {
        case "Boolean": push(SemType.Boolean, 0, 0, 0, ((v ? 1 : 0) - st.mean) / st.std); break;
        case "Datetime": push(SemType.DateTime, 0, 0, (Number(v) - dtMean) / dtStd); break;
        case "String": { const id = tid(v as string); push(SemType.Text, 0, id, 0, 0, id); break; }
        default: { const x = Number(v); if (Number.isNaN(x)) return;  // NaN -> no cell
          const z = (x - st.mean) / st.std; if (!Number.isFinite(z)) throw Error("inf"); push(SemType.Number, z); }
      }
    });
  });
}
// toSec(ns: bigint|null) = ns === null ? null : Number(ns / 1_000_000_000n)  (trunc toward 0, then i32)
// Sampler later rounds every value to bf16 (round-to-nearest-even) before the model sees it.
```

## 16. Worked example (output of the mirror script)

The script is `…/scratchpad/scripts/pre_demo.mjs`. It defines a tiny DB:

- `customers{customer_id PK, name, age, is_premium}` — 3 rows; Bruno's `age` is null.
- `products{product_id PK, category, price}` — 2 rows; Toys' `price` is NaN.
- `orders{order_id PK, customer_id FK, product_id FK, order_date time, quantity}` — 5 rows.
- Task `customer-churn{timestamp, customer_id FK, churn}` — Train 4 rows, Val 2 rows, Test 2 rows.

Results:

- **Offsets:** customers 0, orders 3, products 8, customer-churn:Test 10, :Train 12, :Val 16.
- **Stats:**
  - `age`: μ=42.5, σ=12.0208 (n=2, so the null is excluded); Ana → −0.7071, Carla → 0.7071, Bruno → no `age` cell.
  - `is_premium`: μ=1/3, σ=0.57735; true → 1.1547, false → −0.5774.
  - `quantity`: μ=2.4, σ=1.6733.
  - `price`: one non-NaN value, so σ=None→1, and Books → 0.0. Toys has no `price` cell.
  - `churn` (Train): μ=0.25, σ=0.5; true → 1.5, false → −0.5, reused for Val and Test.
- **Datetime:** 13 cells (5 orders, 4 train, 2 val, 2 test task rows). μ_T = 2024-01-29T22:09Z, σ_T = 24.33 days (population std).
- **Example nodes:**
  - order 3: `f2p=[0,9]`, `timestamp=1704412800`, cells `[order_date z=−1.0244, quantity z=−0.239]`.
  - train task row 12: `f2p=[0]`, cells `[timestamp z=−1.1888, churn z=−0.5]`.
  - customers: `timestamp=null`, because the table has no FK.

## 17. Open items / unverified

- The dtypes of rel-avito `IsUserLoggedOn` and rel-event `not_interested` (Boolean vs Int64) were not checked against actual parquet files. They decide whether those "classification" autocomplete tasks use BCE or Huber.
- That RelBench's downloaded `db.zip` is the full DB rests on RelBench's caching code: `get_db` saves before `upto`, and building test labels needs post-test rows. It was not checked byte-by-byte.
