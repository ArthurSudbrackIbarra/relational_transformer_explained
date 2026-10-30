# Chapter briefs

Companion to `docs/BUILD_PLAN.md`. Each brief tells a chapter author what to teach, which interactive figures to build, which spec sections back it, which official code excerpts to show (`CodeRef`), the mandatory "Paper vs code" callouts, quiz ideas, and pitfalls.

## Conventions for every chapter

- **Code paths.** `PE/<path>` = paper-era official code at `<scratchpad>/upstream/rt-paper-era/<path>`.
- **Copying code.** Copy `CodeRef` snippets **verbatim** from the file. Use the exact line range, re-checked by opening the file. Keep snippets ≤ ~40 lines; trim with `// …` only between whole lines.
- **Running example.** ShopDB, task `customer-churn`, the store's current seed (default: the showcase customer's Val row). Refer to concrete things ("customer #12, Ana from BR") using live data, never hard-coded values that can drift.
- **Live state.** Every number shown comes from the live engine, through the hooks in `src/state/hooks.ts`. If weights are still `init` (untrained), say so with a small badge from `useAssets().weightsKind`.
- **Scale.** Wherever model sizes or L/w appear, render `<ScaleNote/>`. Paper: d=256, 8 heads, 12 blocks, L=1024, w=256. Mini: d=64, 4 heads, 4 blocks, L=128, w=16.
- **Prose.**
  - Clear, concrete, warm, precise. Short paragraphs. Define every term on first use (`<Term>`).
  - Avoid hype and filler ("delve", "crucial", "seamlessly", "it's worth noting"). Avoid em-dash chains.
  - Explain *why* a design choice exists, citing the paper's stated rationale.
  - Teach the latest correct version and skip the rest: no paper errata, arXiv version history, provenance of numbers (which run or version produced a table) or numerical edge cases that never arise in normal use (NaN, σ = 0, bf16 rounding, kernel internals). Mention a paper-vs-code difference only when it changes how RT behaves, in one or two plain sentences.
- **Interactivity.** At least one hands-on figure per chapter. Hover cross-highlighting between views where it makes sense. Animations use `motion/react`, respect reduced motion and expose a `StepPlayer` when an algorithm runs.
- **Files.**
  - A chapter lives in `src/chapters/<slug>/`: `index.tsx` plus local components.
  - Only add shared components if truly generic. Coordinate by adding new files rather than editing shared ones. If a shared component needs a small backward-compatible prop, it's OK to add it.
- **Font sizes (user requirement).** Never use small text: prose 18px, UI text ≥ 15px, captions/table cells ≥ 14px, dense chips/labels ≥ 13px, SVG/canvas labels ≥ 12px. Make room instead of shrinking.
- **Quality.** It must typecheck and look polished in light/dark at 1440px and 390px.

---

## 1. `overview`: The big picture

**Goal.** Motivation and a map. In two minutes the learner knows what RT predicts, what goes in, what comes out, and the three key ideas.

**Sections.**
1. *Why relational data is different.* Multiple tables, keys, time; prior approaches (tabular models with feature engineering, schema-specific GNNs, LLM serialization) and their limits (paper p.1-2 §1).
2. *RT in one picture.* A large animated figure (re-imagining paper Fig. 1), stepping through the live running example:
   - ShopDB tables → the task row (customer, timestamp, churn = ?) → BFS context (rows light up) → cells become tokens (chips fly into a strip) → 4 relational attention types (brief pulses) → decoder → probability.
   - Each stage is clickable and links to its chapter.
3. *Three innovations* (paper p.2 "Our contribution"):
   - cell-level tokenization (cells are tokens built from value, column name and table name);
   - task-table prompting;
   - relational attention (column, feature, neighbor + full).
4. *What "zero-shot" means here* (p.4 §2.3): new DB, new schema, no weight updates, information only via the context window.
5. *Headline results* (Stat tiles): 22M params; mean zero-shot AUROC 69.7 on unseen DBs vs Gemma3-27B 64.6 (Table 1); "90.3% / 93.1% of supervised AUROC" with the precise definitions (spec §10 #26).
6. *Paper vs our miniature*: config table and the ScaleNote; what is identical (every operation) and what is not (size, data, bf16 → float32, RNG).
7. *Course map*: chapter cards.

**Spec.** §0, §9.4, §10 #26, §11.1. No CodeRef needed except optional model.py:104-151 (the whole model is ~260 lines).

**Quiz.** "What is a token in RT?" (a cell); "Which information does RT get about a new database at test time?" (only through the context window).

---

## 2. `database`: The relational database

**Goal.** Fluency with tables, rows, cells, PK/FK, F→P vs P→F, time columns, feature vs key columns, as RT uses them.

**Sections and figures.**
1. *ShopDB schema.*
   - `SchemaDiagram` of the 4 tables: column lists with dtype glyphs; PK key icon, FK link icon, time-column clock icon.
   - FK arrows from child to parent (F→P), labelled with the FK column name.
   - Sidebar text: what a table/row/cell/column is (p.3 §2.1).
2. *Browse the data.* `TableView` with tabs per table, showing raw values with ∅ for nulls. Clicking a row selects it.
3. *Links.*
   - For the selected row, show its **parents** (F→P: "this order points to customer 12 and product 3") and **children** (P→F: "customer 12 is referenced by 17 orders and 4 reviews").
   - Mini `EntityGraph` of the selected row ± 1 hop.
   - Explain the asymmetry: a row has at most #FK parents but unboundedly many children (p.16 App. A). This motivates the sampler later.
4. *Feature columns vs key columns.*
   - PK and FK columns **never become tokens**; they only define structure (spec §1.2).
   - The model never sees FK column names and cannot tell two FKs to the same table apart (paper Limitations p.10).
   - Show a row and grey out its key columns.
5. *Time.*
   - Time columns (order_time, review_time); customers/products have none.
   - The leakage principle: to predict at time t, use only rows created before t (p.3 §2.1).
   - Preview: RT's actual rule is narrower (child rows only, ≤ seed time; chapter 6).
   - A timeline strip of all orders of the selected customer.
6. *The graph view.* The DB as a relational entity graph (one node per row, edges for FK links), shown for a neighbourhood. Paper App. B: RDL frames DBs as graphs; RT keeps cell-level tokens instead.

**Spec.** §1 (all). **CodeRef.** `PE/rustler/src/pre.rs` PK skip and FK edge creation (≈439-533, pick the most readable ~30 lines: PK `continue`; FK → `f2p_nbr_idxs.push`, F→P edge, P→F edge).

**Callouts.**
- note: "FK value = parent row position" (§1.2).
- insight: why the model ignores key names.

**Quiz.** Which of these columns become tokens? Given an order row, what are its F→P links?

---

## 3. `tasks`: Predictive tasks and task tables

**Goal.** Understand masked-token prediction, forecasting vs autocomplete, and how a task becomes a *table* attached to the DB.

**Sections and figures.**
1. *Every task is "fill in the masked cell"* (p.3 §2.2 MTP).
   - Animation: a cell in a table gets a "?" and the model must predict it.
   - Autocomplete: the missing value lives in an existing column (`reviews.rating`).
   - Forecasting: the value does not exist yet and must be computed from the future.
2. *Building `customer-churn`, step by step* (`StepPlayer`):
   - (a) pick timestamps every Δ=30 days going back from the val timestamp (RelBench style, spec §2.2);
   - (b) for one customer and one timestamp t, highlight the window (t, t+30d] on the customer's order timeline;
   - (c) the label = 1 iff no order in the window;
   - (d) a row (timestamp, customer_id, churn) is appended;
   - (e) repeat for all active customers.
   - Show the resulting task table (`TableView`) with its FK to customers.
3. *Splits on a timeline.* Train rows (timestamps before val), Val rows at 2025-03-01, Test rows at 2025-04-15, drawn as a timeline with row counts. All three splits are the same logical table (same table name and column names, spec §2.2). Preview of why that matters (in-context labels can come from train rows).
4. *Regression and autocomplete.*
   - `product-sales` (units in the next 30 days).
   - Autocomplete `reviews.rating` and `reviews.verified`, with leakage columns: `review_text` is dropped for rating because it reveals the answer (spec §2.4, §4.9). Analogy: rel-amazon's review_text/summary (`PE/rt/tasks.py`).
5. *Task-table prompting* (p.4 §3.1).
   - The task table is attached to the database as an **ordinary table** with an FK to the entity. Its rows are the seed rows. Only one task table is active at a time.
   - Figure: schema diagram with the task table (magenta) added.
6. *What the model sees of a task row*: exactly 2 tokens, the timestamp cell and the label cell. The FK produces only a link (spec §2.1).

**Spec.** §2 (all), §4.9. **CodeRef.**
- `PE/rt/tasks.py` (forecast task tuples + one autocomplete tuple with leakage columns).
- `PE/rt/data.py:77-86` (task table key `:Db` vs `:<split>`, target column index).
- `PE/rustler/src/fly.rs` seeds = rows having the target column (≈287-296).

**Callouts.** paper-vs-code: autocomplete ignores the split and uses rows from all periods (spec §2.4, §10 #10).

**Quiz.** Is "product-sales" autocomplete or forecasting? Why is review_text dropped for the rating task?

---

## 4. `preprocessing`: From rows to cells

**Goal.** Exactly how values are encoded before the model.

**Sections and figures.**
1. *One index space.*
   - Every row of every table, task splits included, becomes a node with a global index.
   - Figure: a horizontal bar of offsets coloured by table, in glob order (DB tables by name, then task splits test < train < val), from live `pdb.tableInfo` (spec §1.5).
2. *Which cells become tokens.* Pick any row → see which of its cells become cells: PK/FK → none; null/NaN → none; others → one cell each (spec §3.5, `pre.rs:543-611`).
3. *Four semantic types* (spec §3.2).
   - dtype → sem type table with glyphs; codes Number=0, Text=1, DateTime=2, Boolean=3 (`common.rs:38-44`).
   - No semantic inference: strings are Text; non-key integer ids would be Numbers.
4. *Normalization, interactively.*
   - Pick a column → histogram of raw values and z-scores, with live μ and σ.
   - **Booleans.** Show the computed true/false z values for `is_premium` and the churn label, and explain why "true ⇔ z > 0" (spec §3.3).
   - **Datetimes.** One global μ_T and σ_T shared by every datetime column of every table. Show three datetime columns on one axis.
   - Task labels use **Train** statistics for all splits.
5. *Pick a cell, see its encoding.* Click a cell anywhere → panel: raw value → sem type → stored value(s) → which embedding slot it will use.

**Spec.** §1.5, §3 (all), §10 #3, #4. **CodeRef.**
- `PE/rustler/src/pre.rs:316-341` (number/boolean stats);
- `pre.rs:342-366` (datetime);
- `pre.rs:383-396` (Val/Test reuse Train stats);
- `pre.rs:553-605` (per-type cell values);
- `PE/rustler/src/common.rs:38-44`.

**Callouts.**
- paper-vs-code (§3.4): paper says "training split"; code uses all DB rows for DB-table stats; task labels use Train.

**Quiz.** What is the z-value of `true` if 25% are true (≈ +1.73)? Do datetime columns get per-column statistics?

---

## 5. `semantics`: Text and schema semantics

**Goal.** How names and strings enter the model through a frozen text encoder, and why it enables transfer across schemas.

**Sections and figures.**
1. *The phrase "<col> of <table>".*
   - Every cell carries the embedding of its column phrase, e.g. "price of products", "churn of customer-churn" (p.5 §3.1; spec §3.6).
   - Live list of ShopDB phrases.
2. *A real 384-d vector.* `VectorStrip` (384 cells, wrapped) of the selected phrase's actual MiniLM-L12-v2 embedding, with norm = 1 shown (mean pooling + L2 normalize).
3. *Similarity across databases.* Cosine-similarity heatmap among column phrases of ShopDB vs School/Racing/Forum (computed live from the store).
   - Hover → the pair and its cosine.
   - Highlight the semantically close pairs: "churn of customer-churn" ~ "dropout of student-dropout"? (Compute; do not assume.)
4. *2-D map* of all column phrases (from `pca2d.json`), coloured by DB. Pan, hover.
5. *Text cells.* String values are embedded from the raw string only, with no column context (spec §3.6). Example: "BR" as a country.
6. *Why it matters.*
   - Paper Table 4: shuffling column/table names lowers zero-shot (69.5 vs 70.1 AUROC, 20.5 vs 22.8 R²).
   - Column-name overlap across RelBench DBs is tiny (Jaccard ≤ 4.3%, App. C), so transfer is not lexical copying.
7. *The encoder is frozen.* Embeddings are precomputed once (`embed.py`). The model learns only linear projections on top.

**Spec.** §3.6, §10 #16, §11.2, §11.7 (App. C). **CodeRef.**
- `PE/rt/embed.py:14-31` (SentenceTransformer encode);
- `pre.rs:424-437` (interning table names and column phrases);
- `PE/rt/data.py:31-43` (`"<col> of <table>"` lookup).

**Callouts.** none required. Name the encoder as the code loads it, `sentence-transformers/all-MiniLM-L12-v2`.

**Quiz.** Does the model see FK column names? (no) What text is embedded for a text cell? (the raw value only)

---

## 6. `sampling`: Building the context window (Algorithm 1). THE CENTREPIECE

**Goal.** Watch the exact BFS of `fly.rs` build the context for the live seed, one decision at a time.

**Layout.**
- Large two-pane figure (like paper Fig. 6): left `EntityGraph` of visited/frontier rows (radial by depth, seed in the centre); right the context as per-table mini tables filling up.
- Below: the token strip filling to L.
- A control bar: seed picker (Val/Test/Train rows of the task), L (16–256), w (1–64), RNG seed, `StepPlayer` driven by `SampledContext.events`.

**State panels**, updated per event:
- **F→P stack** (LIFO; top highlighted);
- **P→F buckets** by depth label (chips per row; the uniform pick is animated, showing the draw);
- **visited set** count;
- **tokens used / L** meter.

**Event narration.** A caption that explains each event in plain words, e.g.:
- "Popped customer 12 from the stack (depth 1). Parents always come first, LIFO."
- "Order 311 is in the future relative to the seed (2025-03-14 > 2025-03-01): skipped."
- "Customer 12 has 23 order/review children that pass the time filter. Keep a uniform random 16 of them (w = 16). Task rows are never subsampled."
- "Context is full at L = 128: row cut mid-way."

**Sections.**
1. *Why sample?* The whole DB does not fit. Relevant information is concentrated within a few hops (p.4 §3.1, App. A).
2. *The algorithm.* Alg. 1 as written in the paper (rendered as pseudo-code) next to the code's actual rules (spec §4.4-4.11). Interactive step-through.
3. *Four rules worth remembering*:
   - (1) parents first, always, never filtered;
   - (2) children only if not later than the seed;
   - (3) at most w DB children per row, while all task-table children (the label rows) are kept;
   - (4) stop at exactly L cells.
4. *In-context labels.*
   - Highlight label cells in the context: **self labels** (past task rows of the same customer) vs **other-entity labels**.
   - Explain splits visibility (Val seed sees train rows + same-horizon Val rows of other customers, spec §2.5).
   - Paper Table 12 for scale.
5. *Experiment.* Sliders for L and w with live counts (tokens, rows per table, self labels, other labels, padding). Note Fig. 4: shorter context degrades gracefully.
6. *Alg. 1 vs code*: the full comparison table from spec §4.15.

**Spec.** §4 (all), §2.5, §11.3. **CodeRef.**
- `PE/rustler/src/fly.rs:357-399` (setup + selection);
- `fly.rs:400-452` (expansion: parents, time filter, task-table rule, width bound);
- `fly.rs:454-511` (emission, single mask, stop at L).

**Callouts (paper-vs-code).**
- F→P is LIFO, not RandomSelect;
- the width bound applies to DB children only;
- a row can be cut at L;
- same-horizon labels of other entities are visible in the released code (spec §10 #5, #7, #8, #11).

**Quiz.** Which rows are never subjected to the time filter? If a customer has 40 past orders and w=16, how many orders can enter via that customer?

---

## 7. `tokens`: Cells become tokens

**Goal.** The exact per-token record the model receives, and two key properties: one masked target, no positions.

**Sections and figures.**
1. *The batch dictionary.*
   - `TokenTable` of the live context: pos, node_idx, f2p_nbr_idxs[5], table_name_idx, col_name_idx, sem_type, number/datetime/boolean value, text idx, mask, is_padding (spec §5.1).
   - Hover a token → highlight in TokenStrip and in a mini table view of its row.
   - Explain each field and whether the model uses it (is_task_nodes and class_value_idxs are unused).
2. *Rows are not tokens, cells are.* A row of 4 cells = 4 tokens sharing node_idx and f2p list. Every cell repeats its row's full parent list (−1-padded to 5).
3. *The single masked token.*
   - Exactly one `mask = true`: the seed row's target cell.
   - It keeps its column-name embedding and type; only its value is replaced by a learned vector (chapter 8).
   - **Demo:** "The true label is still stored in the token arrays." A toggle changes the hidden stored label (true ↔ false); the prediction is identical, because the value is gated out (spec §5.3).
4. *No positional encodings.*
   - **Demo:** "Shuffle tokens" button permutes the token order, reruns the forward live and shows the prediction is identical (to float precision), while the token order visibly changes (spec §6.5).
   - Explain why: the embeddings ignore position, and the masks depend on ids, not positions.
   - Note: order matters only to the sampler (which rows fit before L).
5. *Padding.* Padded positions have node_idx −1, attend to nothing, and stay exactly zero.

**Spec.** §5.1, §5.3, §4.10-4.13, §6.5. **CodeRef.**
- `PE/rustler/src/fly.rs:462-500` (writing the fields);
- `PE/rt/data.py:116-154` (reshaping the batch);
- `PE/rustler/src/fly.rs:65-85` (padding defaults; check the lines).

**Quiz.** How many tokens have mask = true? Does shuffling token order change the prediction?

---

## 8. `embedding`: The token embedding

**Goal.** Compute x₀ for any token, with real numbers.

**Sections and figures.**
1. *The formula* (MathBlock, spec §5.2):

   x_i = RMSNorm_col(W_col·c_i + b_col) + [RMSNorm_t(W_t·v_i + b_t) or m_t if masked], with padding → 0.

   Then the paper's simplified form and a paper-vs-code callout (no bias, no RMSNorm in the paper, §10 #2).
2. *Pick a token → step through.*
   - (1) column phrase → 384-d MiniLM vector;
   - (2) W_col·c + b (64-d);
   - (3) RMSNorm;
   - (4) the value path for its type: a scalar z → W_t·z + b → RMSNorm; or text: 384-d → Linear → RMSNorm; or masked: m_t;
   - (5) sum = x₀.
   - All as `VectorStrip`s with live numbers from the trained mini model.
3. *RMSNorm explained* (spec §7.2): y = x / √(mean(x²)+ε) · γ, with ε = 2⁻⁷. Interactive: drag a 2-D vector and see its RMS normalization. Note RMSNorm(0) = 0.
4. *One encoder per type.*
   - Numbers of all columns share W_number. Column identity comes only from the column term (and from the column mask later).
   - Figure: sweep z from −3 to 3 for the number encoder and plot the value term's trajectory (2-D PCA of the 64-d outputs). It is a 1-D curve, approximately on a sphere (spec §5.2 teaching note).
5. *The mask vector.* m_t per type, learned, added without normalization. The masked token knows "which column, which table, which type" but not the value.
6. *Padding.* Zero vector.

**Spec.** §5.2-5.3, §7.2, §10 #2. **CodeRef.** `PE/rt/model.py:115-146` (encoders, norms, mask embs); `model.py:208-224` (embedding computation).

**Quiz.** Which part of x₀ tells the model the column? What replaces the value for the masked cell?

---

## 9. `masks`: Relational attention masks

**Goal.** The four masks, computed for the live context, and what each lets flow.

**Sections and figures.**
1. *Attention with a mask.*
   - SDPA formula with −∞ masking (paper p.5; MathBlock).
2. *The four masks side by side.*
   - Four `MaskMatrix` views (S×S) of the live context, with axes grouped by row/table colour bands.
   - Definitions in paper notation and code form (spec §6.2 table).
   - Toggle "hide padding".
3. *"Who can I see?"* Hover/click a query token:
   - in each of the 4 panels, the allowed keys light up in a compact per-table view of the context;
   - in a mini graph;
   - in the TokenStrip.
   - The same visual idea as paper Fig. 1 (right).
   - Preset buttons: "the masked target", "a customer cell", "an order cell", "a product cell".
4. *What each layer is for* (paper wording + interpretation):
   - **column**: the distribution of a column; the target sees all visible labels of its task, all splits;
   - **feature**: own row + parents, like a join;
   - **neighbor**: parent reads its children, GNN-like aggregation; rows with no children in context get an empty row and a zero update;
   - **full**: everything.
5. *Mask counts.* A small bar chart of allowed pairs per mask (density) for the live context.
6. *Details that matter.*
   - Padding AND (why it is needed: −1 ids);
   - the column test also checks the table;
   - all heads share the mask;
   - masks are built once and reused by every block.
7. *Worked example from the spec* (§6.3): the 24-token toy masks printed as ASCII-like grids, with the explanation. Optional: re-render with our MaskMatrix.

**Spec.** §6.1-6.3, §6.6, §10 #12, #24. **CodeRef.** `PE/rt/model.py:164-206` (mask construction and block masks); `model.py:89-101` (`_make_block_mask`).

**Callouts.** paper: "global" = "full" (§6.2 naming); neighbor is the exact transpose of feature's parent part.

**Quiz.** Can a customer cell attend to its orders in the feature layer? (No: neighbor layer.) Which mask lets the target read past labels directly? (Column.)

---

## 10. `attention`: Inside one attention sublayer

**Goal.** Follow the numbers through one masked multi-head attention, then see attention patterns of the trained mini model everywhere.

**Sections and figures.**
1. *Choose*: block (1–4), layer (col/feat/nbr/full), head (1–4), query token (default the target).
2. *Step-through* (`StepPlayer`, 7 steps), with live tensors from `ForwardResult.blocks[b].attn[l]`:
   - (1) residual stream x entering the sublayer;
   - (2) h = RMSNorm_l(x);
   - (3) Q, K, V = h·W (64×64, no bias), head slice of 16 dims;
   - (4) scores q·k / √16 for all keys (bar chart over keys, grouped by table);
   - (5) mask → −∞ (masked bars greyed);
   - (6) softmax → probabilities (bars; sum = 1, or all zero if fully masked);
   - (7) weighted sum of V, concat heads, W_o, residual add (VectorStrips, with the update norm).
3. *All heads, all layers.* Grid of `Heatmap`s (4 layers × 4 heads) for the selected block. Click → focus. Row/column labels by token, colour bands by table.
4. *Where does the target look?*
   - For the target query, a ranked list of the keys with the largest weights per layer/head, linked to the context table view.
   - Interpretation is clearly labelled as such (the model is a small synthetic-data replica).
5. *Implementation note.* FlexAttention block-sparse kernels in the paper vs our dense loop. Mathematically the same (spec §6.1). Fully masked query rows output exactly 0.

**Spec.** §6.1, §6.2, §7.1. **CodeRef.** `PE/rt/model.py:20-47` (MaskedAttention).

**Quiz.** What is the scale factor in the paper model? (1/√32.) In ours? (1/√16.)

---

## 11. `block`: The RT block and the residual stream

**Goal.** The full block, the stack, and how information accumulates in the target token.

**Sections and figures.**
1. *Block diagram* (animated SVG): x → [RMSNorm → MHA_col] + → [RMSNorm → MHA_feat] + → [RMSNorm → MHA_nbr] + → [RMSNorm → MHA_full] + → [RMSNorm → SwiGLU] + → x′. Click a box to inspect it (links to chapter 10 for attention).
2. *Pre-norm vs the paper's Alg. 2.*
   - Paper Alg. 2 writes X ← X + Norm(MHA(X; M)) (post-sublayer norm).
   - The code is X ← X + MHA(Norm(X); M) (pre-norm, LLaMA-style).
   - Show both formulas; the site follows the code (spec §7.1, §10 #1).
3. *SwiGLU.* w2(SiLU(w1·h) ⊙ w3·h), 64 → 256 → 64 in ours (256 → 1024 → 256 in the paper), no biases. Interactive SiLU curve plus the gate. Live activations of the target token (VectorStrip of 256 gated units).
4. *The stack.* 4 blocks in ours, 12 in the paper, then a final RMSNorm. Parameter count: live computed for the mini model vs 22,336,131 for the paper (spec §7.3 table).
5. *The residual stream of the target token.*
   - A chart of ‖update‖ for each of the 20 sublayers (4 blocks × 5);
   - the target's representation after each sublayer as VectorStrips;
   - optional cosine similarity to the final representation.
6. *How information can reach the target* (spec §6.4, explicitly labelled "interpretation").
   - Column (labels) → feature (parent entity) → neighbor (entity aggregates its children) → full.
   - With 4 blocks, multi-hop paths compose. Interactive path tracer: pick a source token (e.g. a past label) and show which layers/blocks give a direct edge to the target.
7. *Expressivity notes* (paper App. E).

**Spec.** §7 (all), §6.4, §10 #1, #23. **CodeRef.** `PE/rt/model.py:50-86` (FFN + RelationalBlock); `model.py:147-150, 226-229` (blocks + norm_out).

**Quiz.** In what order are the four attentions applied? Is the norm applied before or after each sublayer in the code?

---

## 12. `decoding`: Decoders, loss and prediction

**Goal.** From the final vector to a probability or a number, and how the loss is computed.

**Sections and figures.**
1. *Final RMSNorm, then four linear decoders at every position.* Show the 4 head outputs for the target (number, text (384-d, never used because text is never masked), datetime, boolean); only the target's type is read (spec §8.1).
2. *Binary.*
   - Logit z → σ(z) = probability. Interactive sigmoid with the live logit marked.
   - Label = 1{z_label > 0} because booleans are z-scored (chapter 4 link).
3. *Regression.* Predicted z → de-normalized ŷ = z·σ_train + μ_train. Show the live product-sales prediction when the task is switched.
4. *Loss with numbers.*
   - BCE-with-logits formula with the live values;
   - Huber δ=1 curve with the live residual marked;
   - mean over masked cells of the batch (one per context).
5. *Metrics.*
   - AUROC on logits (= on probabilities, σ is monotone) and R² in z-space (= raw units).
   - Live computed on the mini model's val predictions over all val seeds of the task (compute in a web-worker-free loop; cache).
   - A ROC curve figure plus a calibration-free explanation.
6. *One decoder per type, shared across all columns, tables, DBs.* The number head's output is read in the target column's own z-space.

**Spec.** §8 (all), §10 #15, #19. **CodeRef.**
- `PE/rt/model.py:229-267` (decoders + loss);
- `PE/rt/main.py:283-293` (readout at is_targets);
- `PE/rt/main.py:328-337` (AUROC/R²).

**Quiz.** Is the sigmoid needed to compute AUROC? Which loss is used for a number target?

---

## 13. `end-to-end`: One prediction, start to end

**Goal.** One page that runs a single prediction from the choice of task to the final number, showing every matrix of the forward pass and the weights that produce it. The earlier chapters each open one step; this one connects them on the live seed.

**Sections and figures** (one per stage, ids in brackets):
0. *The pass at a glance* [`map`]. Seven stages with what flows between them (shapes) and the learned weights each uses, switchable between this site's sizes and the paper's (`parameterShapes`); live results per stage. The shared *followed token* (the store's `selectedPos`, the masked target by default) is outlined in every matrix.
1. *Pick a task and a seed row* [`question`]. Database / task / split / seed pickers that write the global store; the seed row column by column: which cells become tokens, PK (nothing), FKs (links), nulls, leakage columns, the masked target and its hidden label; what comes out (logit or z-score) and which checkpoint is loaded.
2. *Sample the context* [`sample`]. A replay of the sampler's event trace one visited row per step (skipped pops counted in the narration): parents pushed, children filtered by time / other task tables, task rows kept, width bound, cells emitted, the stop at L; the token strip filling up.
3. *Write the tokens* [`tokens`]. The batch dictionary (`TokenTable`), and the two roads its fields take: ids → masks, values → embedding; three fields unread by model.py.
4. *Embed every token* [`embed`]. Text lookups (`col_name_values`, `text_values`, including E(text[0]) for non-text cells), the column-name projection as a matrix product, its RMSNorm, the value term per type with the mask vectors, and x₀.
5. *Build the four masks* [`masks`]. The four S × S masks, hover-linked, with the condition that allows each pair.
6. *Run the blocks* [`blocks`]. One player over every operation of every block (4 × (4 × 7 + 4) = 128 steps): RMSNorm, Q/K/V, per-head scores, mask, softmax, weighted sum, W_o + residual; SwiGLU W₁/W₃, gate, W₂ + residual. Matrix products use the textbook layout (B above C, A to its left); hover any entry for its dot product.
7. *Decode and read the answer* [`output`]. norm_out, the three scalar decoders at every position (the text head runs in the official code but is never read), the read-out at the target, σ or de-normalization, the hidden label and the training loss.
8. *The same pass at the paper's size* [`scale`].

**Correctness.** Every matrix is the engine's capture. Every number written out next to a figure is recomputed from the operands on screen (`src/chapters/end-to-end/math.ts`), and `math.test.ts` checks each re-derivation against the capture on the trained checkpoints for a binary and a regression task, at every block, layer and head.

**Spec.** §0, §2.1-§2.5, §3.3-§3.6, §4.3-§4.11, §5.1-§5.3, §6.1-§6.2, §7.1-§7.3, §8.1-§8.5, §10 #1, #2, #5. **CodeRef.** `PE/rustler/src/fly.rs:404-452`, `fly.rs:462-500`; `PE/rt/model.py:208-224`, `164-191`, `62-86`, `20-47`, `226-229`; `PE/rt/main.py:283-293`.

**Callouts (paper-vs-code).** The embedding's biases and RMSNorms (§10 #2); pre-norm vs Alg. 2 (§10 #1); Alg. 1 vs the sampler (§10 #5, link to chapter 6).

**Quiz.** How many cells does one pass predict? Which fields decide who attends to whom? What runs after column attention? What does the decoder output at the masked cell?

---

## 14. `training`: Pretraining, zero-shot, fine-tuning

**Goal.** How RT is trained, what "zero-shot" means operationally, and our real training run.

**Sections and figures.**
1. *Masked token prediction pretraining.*
   - Each training context has one masked cell (the seed's target).
   - Tasks from many DBs are mixed in one shuffled list, proportional to seed counts, no balancing (spec §4.14, §9.1).
   - Animation: a batch of contexts from School/Racing/Forum.
2. *Leave-one-DB-out.*
   - Paper: pretrain on 6 RelBench DBs, test on the 7th.
   - Ours: pretrain on School + Racing + Forum, evaluate on ShopDB, which is never seen.
   - Diagram with the 4 DB cards.
3. *Our training run* (from `public/data/training/*.json`):
   - loss curve;
   - LR and β1 schedules (OneCycle: warmup from max_lr/25, peak at 20%, decay to max_lr/(25·1e4); β1 cycles 0.95↔0.85), as two separate charts (never dual axis);
   - zero-shot val metrics vs step with the selected best-val checkpoint marked.
4. *Checkpoint selection.*
   - Best val **per target task** (paper protocol; uses target-task val labels, spec §9.1).
   - Step 0 can win in continued pretraining (spec §9.3), as in Table 1's rel-f1 rows.
5. *Results on ShopDB*: zero-shot vs EntityMean baseline vs untrained vs fine-tuned (AUROC for churn, R² for sales), with the paper's numbers alongside for context (Tables 1, 2, 10). Honest commentary on what our tiny synthetic run shows and does not show.
6. *Fine-tuning curves.* Pretrained vs random init, a Fig. 3 analogue with a log-x axis.
7. *Hyperparameters.* Paper vs mini table (spec §9.2): AdamW β, wd 0.1, peak LR 1e-3, OneCycle 20% warmup, 50k steps, batch 256, grad clip 1.0, bf16, ≈2 h on 8×A100. Ours: CPU, float32, our step counts.

**Spec.** §9 (all), §11.1, §11.7, §10 #13. **CodeRef.**
- `PE/rt/main.py:186-204` (optimizer + OneCycleLR);
- `main.py:394-424` (train step with grad clip);
- `main.py:373-392` (eval + best-val checkpointing);
- `PE/scripts/example_pretrain.py` (whole file; it is short).

**Callouts.**
- note: the paper's multi-cell masking P(mask) is an ablation not in the released code (Table 14).

**Quiz.** Which DB is never seen during pretraining in our run? Why is selection by target-task val labels still "zero-shot" in the paper's sense?

---

## 15. `playground`: Zero-shot playground and ablations

**Goal.** Let the learner poke the model and build intuition, mirroring the paper's ablations.

**Layout.**
- Left: pick task (churn / sales / autocomplete), split, and seed, with a searchable list of customers or products.
- Centre: the context (compact table view, editable label cells) and the prediction (big Stat with probability or value, and a history of the last predictions as a small bar chart).
- Right: ablation toggles.

**Ablations** (each recomputes live; show Δ prediction):
- *Edit in-context labels.* Click a visible past label (self or other) to flip it (bool) or edit it (number), mirroring an "edit cell" action. The edited value is z-scored with the same stats.
- *Remove self labels* / *Remove other-entity labels*: drop those task-row label tokens from the context, like paper Table 4. State how we implement it: the label cells are removed from the token list before the forward. The exact implementation in the paper is unreleased (spec §13 #3).
- *Shuffle column names*: permute the column-phrase embeddings among columns (Table 4 analogue). Clearly an inference-time emulation.
- *Context length L*: 16–256 (Fig. 4 analogue).
- *Disable attention layer types* (col/feat/nbr/full) at **inference time**. A prominent callout: the paper's Table 5/13 ablations **retrain** a model without that layer; ours just skip a trained layer, so effects will be larger and are not comparable.
- *Checkpoint*: init (untrained) / pretrained (zero-shot) / fine-tuned.

**Batch mode.** Run the current ablation over all val seeds and report AUROC/R² vs the unablated model (cache results), next to the paper's corresponding numbers (spec §11.2, §11.4).

**Spec.** §11.2-11.5, §13 #3, §9.4. **CodeRef.** none required (optional `fly.rs:492-496` for the mask).

**Quiz.** Which context component drives zero-shot transfer most in the paper? (Self labels.)

---

## 16. `results`: What the paper found

**Goal.** A faithful, browsable summary of the paper's empirical findings and its caveats.

**Sections and figures** (data typed into `src/chapters/results/data.ts` from spec §11 and the paper PDF tables; double-check every number against the PDF):
1. *Zero-shot classification* (Table 1): grouped bar chart per task, Gemma 4B/12B/27B, EntityMean, Griffin, RT (No), RelLLM/Griffin/RT (Yes). Filter by group. Mean row highlighted.
2. *Zero-shot regression* (Table 2 + Table 16 note that LLMs give R² < −9).
3. *Learning efficiency* (Fig. 3 description, Table 10 supervised results).
4. *Context ablations* (Table 4/11) and *attention ablations* (Table 5/13), *attention order* (Table 15), *multi-cell masking* (Table 14), *feature extraction* (Table 3, with the 5/4-task subset caveat), *cost-quality* (Fig. 4 numbers).
5. *The ranked drivers of zero-shot transfer* (p.10 §7).
6. *Limitations* (p.10): no link prediction/recommendation; ignores PK/FK names; cannot disambiguate multiple FKs to the same table; future: better cell encoders.
7. *Where the code differs from the paper*: only the few differences that change what RT computes or what its context holds (spec §10), one card each with citations and a link to the chapter that shows it.
8. *Glossary* (all terms) and *References* (paper, code, RelBench, FlexAttention, LLaMA, MiniLM).

**Spec.** §10, §11, §13.

---

## 17. `about`: How this simulation was built and verified

**Sections.**
- *Pipeline*: data generation (deterministic), preprocessing/sampler mirrors, MiniLM embeddings precomputed with sentence-transformers, PyTorch port, CPU training, TS forward.
- *Verification*:
  - the parity results (TS vs PyTorch port vs the official `model.py` class), read from `tests/fixtures` / a small `public/data/verification.json` if present;
  - the test suites;
  - the spec and its adversarial verification log.
- *What is scaled down or different*: size, data, float32, RNG, dense attention instead of FlexAttention.
- *Sources and licenses*: paper (CC BY 4.0 per arXiv? only state what you can verify, else just link); official code (MIT at paper time, Apache-2.0 now); MiniLM model card.
- How to run locally (`npm install && npm run dev`) and how to retrain (`python/README.md`).
