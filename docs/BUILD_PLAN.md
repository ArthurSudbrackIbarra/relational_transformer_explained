# Build plan: "Relational Transformer Explained"

An interactive website that walks a learner through **every step** of the Relational Transformer (RT; Ranjan et al., ICLR 2026, arXiv 2510.06377v3). It runs a **real, faithful miniature RT** in the browser on dummy relational databases.

## 0. Prime directive: faithfulness

1. **Ground truth** is `research/RT_TECHNICAL_SPEC.md` (the "spec"; § numbers below refer to it). It documents the paper-era official code (commit `2a899c8`), with citations. Read the sections relevant to your task **before** writing code or prose.
2. **Never invent behaviour.** If the spec does not say it, do not depict it. If you are unsure, say so in the UI ("not specified by the paper or code") rather than guess.
3. **Where paper and code differ, show the code** and add a "Paper vs code" callout citing both (spec §10).
4. **Scale is the only thing we change.** The paper model has d=256, 8 heads, 12 blocks, L=1024, w=256, trained on RelBench. Ours has d=64, 4 heads, 4 blocks, L=128, w=16, trained on synthetic DBs. Every *operation* is identical. The UI must state this wherever numbers appear: use the `<ScaleNote/>` component.
5. **Do not mix in RT-J** (the later NeurIPS 2026 model in the same repo; spec Appendix C).
6. Every factual claim about RT in prose carries a `<Cite>`: spec section, paper page, or code `file:line`.

## 1. Repository layout

```
/                       Vite + React 19 + TS app (npm run dev / build / test)
  index.html
  src/
    main.tsx, App.tsx           router + shell
    styles/index.css            Tailwind v4 + design tokens (CSS variables, light/dark)
    engine/                     PURE TypeScript (no React). Runs in browser AND Node (tsx).
      types.ts                  shared contracts — DO NOT change without coordination
      rng.ts                    seeded PRNG
      db/                       dummy DB generators + RelBench-style task tables
        generators/{shop,school,racing,forum}.ts
        tasks.ts                forecast + autocomplete task construction
        registry.ts             getDatabase(name), listDatabases(), table color slots
      preprocess.ts             mirror of pre.rs (spec §1.5, §3)
      sampler.ts                mirror of fly.rs with event trace (spec §4)
      masks.ts                  the four relational attention masks (spec §6.2)
      embeddings.ts             EmbeddingStore (precomputed MiniLM vectors)
      model/
        ops.ts                  linear, rmsnorm, silu, softmax, etc. (Float32Array)
        weights.ts              load .bin + manifest → RTWeights
        forward.ts              forward pass with full capture (spec §5.2, §6, §7, §8)
        loss.ts                 huber, bceWithLogits
      metrics.ts                AUROC, R², sigmoid, de-normalization
      pipeline.ts               convenience glue (db → pdb → context → forward)
      *.test.ts                 vitest unit tests
    state/                      zustand store + memoized hooks
    components/                 shared UI + visualization components
    chapters/                   one folder per chapter (see §6)
  public/data/
    embeddings/minilm-l12-v2.{f32,json}   + pca2d.json
    weights/<ckpt>.{bin,json}
    training/*.json            curves, eval results, schedule
  scripts/                      Node scripts (run with `npx tsx scripts/<x>.ts`)
    export-strings.ts           all strings of all DBs → python/data/strings.json
    export-contexts.ts          training/eval contexts → python/data/*.bin
  python/                       PyTorch side (venv at python/.venv, already has torch + sentence-transformers)
    embed.py                    MiniLM-L12-v2 embeddings → public/data/embeddings
    rt_model.py                 port of official rt/model.py (dense masked attention instead of FlexAttention)
    train.py                    leave-one-DB-out pretraining, fine-tuning; checkpoint selection on val
    export_weights.py           state_dict → public/data/weights
    parity.py                   (a) our port == official model.py class; (b) dumps fixtures for the TS parity test
  tests/fixtures/               parity fixtures consumed by vitest
  research/                     spec + raw research reports (read-only reference)
  docs/BUILD_PLAN.md            this file
```

Official code for reference (read-only): `/tmp/claude-1004/-home-ibarra-Documents-Mestrado-Relational-Transformer/155ecce6-1f67-4870-9609-2c9740593feb/scratchpad/upstream/rt-paper-era` (paper era) and `.../upstream/rt-official` (current main, RT-J; do not depict).

## 2. Engine contracts

`src/engine/types.ts` defines every shared type. The key functions:

```ts
// db/registry.ts
getDatabase(name: 'shop'|'school'|'racing'|'forum'): RawDatabase   // deterministic, cached
listDatabases(): { name, title, role: 'target'|'pretrain' }[]
tableColorSlot(dbName, tableName): number                          // 1..8 categorical slot (task tables share one slot)

// preprocess.ts
preprocess(db: RawDatabase): PreprocessedDB                        // includes ALL task tables of the DB (all splits)
// spec §1.5: node order = DB tables sorted by name, then task splits sorted "test" < "train" < "val" per task (task dirs sorted by name).

// sampler.ts
sampleContext(pdb: PreprocessedDB, p: SampleParams): SampledContext
listSeeds(pdb, task: TaskDef, split: Split): number[]              // forecast: rows of "<task>:<split>" with non-null target; autocomplete: all rows of "<table>:Db" with target cell (split ignored, spec §2.4)
targetColNameIdx(pdb, task): number; dropColNameIdxs(pdb, task): number[]

// masks.ts
buildMasks(tokens: Token[]): MaskSet                               // spec §6.2 incl. padding AND

// embeddings.ts
interface EmbeddingStore { dim: number; get(text: string): Float32Array; has(text: string): boolean }
loadEmbeddingStore(): Promise<EmbeddingStore>                      // browser: fetch public/data/embeddings; node: fs

// model/
loadWeights(ckpt: string): Promise<RTWeights>
initWeights(config: RTConfig, seed: number): RTWeights             // PyTorch-default-style init (for "untrained")
forward(tokens: Token[], pdb: PreprocessedDB, store: EmbeddingStore, w: RTWeights, opts?: ForwardOptions): ForwardResult

// metrics.ts
auroc(labels: number[], scores: number[]): number; r2(y: number[], yhat: number[]): number
sigmoid(z); denormalize(z, stats)
```

### Mini model config (`MINI_CONFIG`)

`{ dModel: 64, nHeads: 4, dFF: 256, nBlocks: 4, dText: 384, eps: 2 ** -7 }`. Defaults: `L = 128`, `w = 16`, eval `rngSeed = 0`. Paper: `{256, 8, 1024, 12, 384}`, L=1024, w=256.

### Numeric conventions

- Float32Array everywhere in the model; z-scores computed in float64 then stored.
- Datetimes are epoch ms in raw data. Node `ts` = `Math.trunc(ms / 1000)` (whole seconds, i32 semantics). Dates must lie in 1901-12-14 .. 2038-01-18 (all our DBs are 2023–2025).
- The paper environment rounds values to bf16. We use float32 and say so in a `<ScaleNote/>`.

## 3. The four dummy databases

All generators are **seeded and deterministic**: same output on every load, in browser and Node. Rules from the spec:

- PK = row position 0..n−1 (§1.2).
- At most 5 FKs per table.
- No constant boolean column, and every task's Train split must contain both classes (§3.3, NaN hazard).
- Include some nulls (missing values produce no token).
- Labels must carry learnable, *transferable* signal. The paper finds self-label autocorrelation is the main driver, so make entity behaviour persistent over time (Markov regimes, slowly drifting popularity).

Time span 2024-01-01 .. 2025-06-30 for all DBs. Forecast tasks: Δ = 30 days unless noted. `valTimestamp` = 2025-03-01, `testTimestamp` = 2025-04-15. Train timestamps are `val − kΔ` for k=1,2,… while ≥ DB min timestamp + Δ (RelBench style, spec §2.2). Val split = rows at valTimestamp, Test split = rows at testTimestamp. Entities included at time t: those with ≥1 activity row before t (RelBench-like "active" filter). Labels aggregate the window (t, t+Δ].

### 3.1 ShopDB (`shop`): TARGET DB, held out from pretraining (used in chapters 2–12, 14)

| table | pk | fks | timeCol | feature columns |
|---|---|---|---|---|
| customers | customer_id | — | null | name (str), country (str: BR/US/DE/JP/IN/FR), age (float, ~10% null), is_premium (bool, ~30% true), signup_date (datetime) |
| products | product_id | — | null | name (str, e.g. "Wireless Mouse"), category (str: electronics/books/home/toys/sports), price (float) |
| orders | order_id | customer_id→customers, product_id→products | order_time | order_time (datetime), quantity (int 1–5), channel (str: web/app/store) |
| reviews | review_id | customer_id→customers, product_id→products | review_time | review_time (datetime), rating (int 1–5), review_text (str from ~25 templates correlated with rating), verified (bool ~75% true) |

Sizes: ~80 customers, ~30 products, ~1,200 orders, ~300 reviews.

Process: each customer has a Markov active/inactive monthly regime (stay-active ≈ 0.85, re-activate ≈ 0.15). While active, orders ~ Poisson(λ_c), λ_c ~ Gamma, premium higher. Product popularity is Zipf-like and drifts slowly. Rating depends on a product-quality latent plus noise.

Tasks:
- `customer-churn` (forecast, **binary**): churn = 1 iff the customer has no order in (t, t+30d].
- `product-sales` (forecast, **regression**): sales = number of units ordered in (t, t+30d].
- `reviews` / target `rating` (autocomplete, **regression**), columnsToDrop `['review_text']`.
- `reviews` / target `verified` (autocomplete, **binary**), columnsToDrop `[]`.

### 3.2 SchoolDB (`school`): pretraining

- students(student_id; program str, age float, scholarship bool, enrolled_on datetime)
- courses(course_id; title str, department str, credits int, difficulty float)
- submissions(submission_id; student_id→students, course_id→courses; submitted_at datetime [timeCol], grade float 0–100 (some null), late bool)

Tasks: `student-dropout` (binary: no submission in next 30d), `course-submissions` (regression: #submissions in next 30d), autocomplete `submissions.grade` (regression, drop `[]`), `submissions.late` (binary).

### 3.3 RacingDB (`racing`): pretraining

- drivers(driver_id; name str, nationality str, birth_year int)
- teams(team_id; name str, country str)
- races(race_id; circuit str, race_date datetime [timeCol], laps int). No FK, so node ts = null (a nice edge case, spec §1.4).
- results(result_id; driver_id→drivers, team_id→teams, race_id→races; date datetime [timeCol], grid int, position int (null when DNF), points float, dnf bool)

Races every ~2 weeks. Tasks (Δ = 30d, valTimestamp 2025-03-01 like the other DBs): `driver-dnf` (binary: any DNF in window), `driver-points` (regression: points in window), autocomplete `results.position` (regression, drop `['points','dnf']`).

### 3.4 ForumDB (`forum`): pretraining

- users(user_id; display_name str, location str, reputation float, joined datetime)
- posts(post_id; owner_id→users; created datetime [timeCol], title str, tag str, body_length int)
- votes(vote_id; user_id→users, post_id→posts; created datetime [timeCol], is_upvote bool)

Tasks: `user-engagement` (binary: 1 iff the user posts or votes in the next 30d; note the polarity is the opposite of churn), `post-votes` (regression: upvotes in next 30d), autocomplete `votes.is_upvote` (binary).

### 3.5 Table colour slots (categorical palette order, see §5)

- Each DB assigns slots 1..4 to its DB tables in schema order.
- **All task tables use slot 5 (magenta)**, so "task rows / labels" read consistently everywhere.
- ShopDB: customers=1 blue, products=2 orange, orders=3 aqua, reviews=7 violet, tasks=5 magenta.

## 4. Data / training pipeline (offline; outputs are committed to `public/data`)

1. `npx tsx scripts/export-strings.ts` → `python/data/strings.json` (every table name, every `"<col> of <table>"` phrase incl. task tables, every string cell value, across all 4 DBs).
2. `python/.venv/bin/python python/embed.py` → `public/data/embeddings/minilm-l12-v2.f32` (float32, N×384, row-major), `.json` (string list, same order) and `pca2d.json` (2-D PCA of the column phrases, for the semantics chapter). Model: `SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')`, default `encode` (mean pooling + normalize, max_seq_length 128).
3. `npx tsx scripts/export-contexts.ts` → for each DB/task/split and several epochs (rngSeed = epoch + seedNode + 0, like fly.rs), sampled contexts in the batch layout (`node_idxs`, `f2p_nbr_idxs[5]`, `table_name_idxs`, `col_name_idxs`, `sem_types`, value scalars, `text_idx`, `masks`, `is_padding`) as binary files + a JSON index. Python maps text indices to embeddings through the per-DB text lists.
4. `python/train.py`:
   - **Leave-one-DB-out pretraining** on all tasks of school+racing+forum (forecast Train split + autocomplete all rows), one masked cell per context.
   - AdamW(wd=0.1), OneCycleLR(max_lr 1e-3, pct_start 0.2, anneal linear, cycle_momentum, div_factor 25, final_div 1e4), grad clip 1.0.
   - Eval zero-shot on shop forecast tasks (val/test) every N steps; keep the best-val checkpoint **per target task** (paper protocol, spec §9.1).
   - Then **fine-tune** on each shop forecast task (constant LR 1e-4, wd 0) from (a) pretrained and (b) random init, logging curves (Fig. 3 analogue).
   - Also log the EntityMean baseline (spec Table 1/2 baseline).
   - float32, CPU (the GPU is too old for FlexAttention); RMSNorm eps = 2⁻⁷.
5. `python/export_weights.py` → `public/data/weights/{init,pretrain-churn,pretrain-sales,finetune-churn,finetune-sales}.{bin,json}`, plus `public/data/training/{curves,results,schedule}.json`.
6. `python/parity.py`:
   - Loads our trained weights into the **official** `RelationalTransformer` class from the paper-era `rt/model.py` (FlexAttention, CPU; patch RMSNorm eps to 2⁻⁷) and checks its outputs match `rt_model.py` on real exported contexts.
   - Dumps `tests/fixtures/parity-*.json` (tokens, embeddings lookups, x0, per-block outputs, final yhat at target) so `src/engine/model/forward.test.ts` asserts that the TS forward matches PyTorch (max abs diff ≤ 1e-4).
   - The site shows this verification on the "How this was built" page.

## 5. Design system

**Look.** Calm, scholarly, modern: think distill.pub × Observable. Generous whitespace, one accent, data is the loudest thing. Light and dark themes (toggle, plus `prefers-color-scheme`).

**Type.** READABILITY FIRST (user feedback: small text is hard to read). Minimum sizes, no exceptions:
- prose 18px (`--prose-size`), body/UI text ≥ 15px, secondary UI text / captions / table cells ≥ 14px, dense labels and chips ≥ 13px,
- SVG chart tick/axis labels ≥ 12px (prefer 13px), never anything below 12px anywhere (including canvas labels and badges).
- If something does not fit, give it more room, wrap it, abbreviate with a tooltip, or show fewer items, but never shrink the text below these floors.
- UI and prose: system sans (`system-ui, -apple-system, "Segoe UI", sans-serif`). Prose 17px/1.65, max ~70ch.
- Values, ids, code: `ui-monospace, "SF Mono", Menlo, monospace`, with tabular numbers in tables.
- Math: KaTeX.

**Tokens** (CSS variables in `src/styles/index.css`, light/dark; from the dataviz reference palette):
- surfaces: `--page` `#f9f9f7`/`#0d0d0d`, `--surface` `#fcfcfb`/`#1a1a19`, `--surface-2` (one step off)
- ink: `--ink` `#0b0b0b`/`#f0efec`, `--ink-2` `#52514e`/`#c3c2b7`, `--muted` `#898781`
- lines: `--grid` `#e1e0d9`/`#2c2c2a`, `--axis` `#c3c2b7`/`#383835`, `--border` rgba(11,11,11,.10)/rgba(255,255,255,.10)
- categorical `--cat-1..8` (light/dark): blue `#2a78d6`/`#3987e5`, orange `#eb6834`/`#d95926`, aqua `#1baf7a`/`#199e70`, yellow `#eda100`/`#c98500`, magenta `#e87ba4`/`#d55181`, green `#008300`/`#008300`, violet `#6250d6`/`#9085e9`, red `#e34948`/`#e66767`. **Fixed order, never cycled.**
- sequential blue ramp `--seq-100..700` (`#cde2fb` … `#0d366b`) for attention weights and magnitudes.
- diverging blue↔red with gray midpoint (`#f0efec`/`#383835`) for signed vector values.
- `--accent` = `--cat-7` violet, for UI chrome (active nav, focus rings, primary buttons).
- **Target / masked cell**: a dashed accent outline plus a "?" glyph, everywhere.

**Encoding rules.**
- **Table identity = categorical colour** (§3.5), shown as a coloured swatch or left border; text stays in ink tokens.
- **Sem type = glyph**, not colour: number `#`, text `Aa`, datetime clock icon, boolean toggle icon. Always rendered by `<SemBadge/>`.
- **Attention layers** are identified by name label + icon. Mask matrices use one sequential hue.
- Status colours (good/warn/critical) only for verification badges and warnings, with icon + label.
- Follow the dataviz skill rules: 2px lines, ≥8px markers, hairline grids, a legend for ≥2 series, tooltips on hover, no dual axes, text never in series colour.

**Motion.** `motion/react`. Animations explain causality (cells flowing into tokens, BFS frontier growing, attention weights forming). Respect `prefers-reduced-motion` (instant transitions). Step-through controls everywhere an algorithm runs: play/pause, step forward/back, speed, reset (`<StepPlayer/>`).

**Shared components** (`src/components/`). Build these first; chapters must reuse them.
- Shell: `Layout` (left chapter sidebar with progress, collapsible on mobile; top bar with theme toggle and global "simulation controls" popover), `ChapterHeader` (number, title, lede, "you will learn"), `Section`, `ChapterNav` (prev/next).
- Prose helpers: `Callout` (variants: `note`, `paper-vs-code`, `warning`, `insight`), `Cite` (inline superscript chip: spec §, paper p., code file:line; hover shows full reference), `CodeRef` (collapsible official-code excerpt with file:line header; snippets are copied verbatim from the paper-era code), `Math` / `MathBlock` (KaTeX), `ScaleNote`, `Glossary` term tooltip (`<Term id="seed-row">`), `Quiz` (1–3 multiple-choice checks per chapter, explanation on answer).
- Data views:
  - `SchemaDiagram`: tables as cards with columns, PK/FK/time icons, FK arrows (F→P), table colour.
  - `TableView`: virtualized-ish rows, cell highlight API, sem badges, null shown as "∅", raw vs z-scored toggle.
  - `EntityGraph`: d3-force graph of rows, edges F→P with arrows, node colour by table, highlight API.
  - `TokenStrip`: sequence of token chips coloured by table, target "?" chip, padding greyed.
  - `TokenTable`: the batch-dict fields per token (§5.1).
  - `VectorStrip`: a d-dim vector as diverging colour cells with hover values.
  - `Heatmap`: canvas S×S or H×S×S with hover readout, row/col highlighting, optional labels.
  - `MaskMatrix`: Heatmap specialized for boolean masks.
  - `StepPlayer`.
  - `LineChart` and `BarChart` (simple SVG, dataviz-compliant, tooltips).
  - `Stat` tile.

## 6. Chapters (routes `/#/<slug>`)

Each chapter: `ChapterHeader` → 3–6 `Section`s of prose and interactive figure → `CodeRef`s → "Paper vs code" callouts where relevant → `Quiz` → `ChapterNav`.

- Prose: didactic, concrete and intuitive. Use the running example (ShopDB, task customer-churn, a specific customer). Explain *why* each design choice exists (cite the paper's rationale). No filler.
- **Every chapter reads live simulation state from the store**, so the same seed, context and tokens flow through all chapters.

1. **`overview`: The big picture.** Why relational foundation models are hard. RT's three ideas (cell tokens, task-table prompting, relational attention). An animated end-to-end pipeline on the running example: DB → task row → context → tokens → 4 attention types → prediction. Paper vs mini scale table. A map of the course.
2. **`database`: The relational database.** ShopDB schema diagram; table browser; PK/FK; feature vs key columns; F→P and P→F links (hover a row → highlight parents/children); temporal rows and time columns; the "entity graph" view (§1).
3. **`tasks`: Predictive tasks and task tables.** Masked-token prediction framing; forecasting vs autocomplete.
   - Animated construction of the `customer-churn` task table: timestamps every Δ, the window (t, t+Δ] and label derivation from future orders.
   - Train/val/test splits on a timeline; the regression task `product-sales`; autocomplete tasks and leakage columns.
   - "Task table prompting": the task table is attached as an ordinary table with an FK to the entity (§2).
4. **`preprocessing`: From rows to cells.** Global node indexing (table_info offsets in glob order); which cells become tokens (PK/FK → structure only; nulls → nothing); sem-type inference table.
   - z-scoring: per-column number stats (sample std); booleans → ±values (true > 0; computed from data); one global datetime z-score (population std).
   - Interactive "pick a cell → see its encoding".
   - Paper-vs-code callout on which rows feed the statistics (§3.3, §3.4).
5. **`semantics`: Text and schema semantics.** The frozen MiniLM-L12-v2 encoder: `"<col> of <table>"` phrases; real 384-d vectors (VectorStrip); cosine-similarity matrix of column phrases across the 4 DBs; 2-D PCA map; text cell values embedded raw (no column context). Why names matter (Table 4 ablation: shuffled names hurt zero-shot) (§3.6).
6. **`sampling`: Building the context window (Algorithm 1).** The centrepiece animation.
   - Step-by-step BFS on the live seed: F→P LIFO stack (absolute priority), P→F depth buckets with uniform pick, temporal filter (child ts > seed ts → skip; equal kept), width bound w on DB children only (task rows never subsampled), visited check at pop, cell emission, hard stop at L (may cut a row), padding.
   - Synchronized graph view and table view (like paper Fig. 6).
   - Controls: seed, L, w, RNG seed.
   - In-context labels highlighted (self vs other entities, same-horizon note).
   - Alg. 1 vs code table (§4).
7. **`tokens`: Cells become tokens.** The batch dict (§5.1) per token, with `TokenTable` and hover cross-highlighting to graph and table. f2p_nbr_idxs; the single masked target. "The label is still in the array but gated out": a demo that changes the hidden label and shows the prediction does not change. Permutation-invariance demo: shuffle tokens, rerun live, identical prediction (§6.5).
8. **`embedding`: The token embedding.** Formula breakdown with live numbers:
   - `RMSNorm(W_col·E(phrase)+b)` + `RMSNorm(W_t·value+b)`, or the raw mask vector m_t; padding → 0.
   - Interactive slider: change a numeric value and watch the value term move along its 1-D curve (projected to 2-D).
   - RMSNorm explained (ε = 2⁻⁷ note).
   - Paper-vs-code callout (the paper omits the bias and the RMSNorm) (§5.2).
9. **`masks`: Relational attention masks.** The four masks (column, feature, neighbor, full) as matrices for the live context, plus "who can I see?" mode: hover any query token → allowed keys light up in the TokenStrip, table view and graph (like paper Fig. 1 right).
   - Definitions in paper notation and code; the padding AND; empty neighbor rows → zero update.
   - What information each lets flow (GNN analogy for neighbor).
   - Worked example (§6.2, §6.3).
10. **`attention`: Inside one attention sublayer.** Pick block, layer, head and query token. Show h = RMSNorm(x), Q, K, V slices, scores q·k/√d_head, mask → −∞, softmax, weighted sum of V, concat heads, W_o, residual add, all with real numbers from the trained mini model. Attention-weight heatmaps for every layer and head (§6.1).
11. **`block`: The RT block and the residual stream.**
   - Block diagram (col→feat→nbr→full→SwiGLU, each pre-norm residual); SwiGLU internals; 4 blocks stacked; final RMSNorm.
   - Residual-stream view of the target token: the norm of each sublayer's update per block.
   - "How information reaches the target" interactive path view, clearly labelled as interpretation (§6.4).
   - Paper Alg. 2 vs code callout (pre-norm) (§7).
12. **`decoding`: Decoders, loss and prediction.**
   - norm_out → all four type decoders run at every position; only the target's type is read.
   - Binary: logit → σ → probability. Regression: z → de-normalize with Train stats.
   - Loss computation with numbers: BCE on 1{z>0}, Huber δ=1, mean over masked cells.
   - Metrics: AUROC on logits, R² in z-space (§8).
13. **`training`: Pretraining, zero-shot, fine-tuning.**
   - Masked-token-prediction pretraining with one masked cell per context; leave-one-DB-out with our 4 DBs.
   - Real training curves from our run; the OneCycle LR and β1 schedule (spec §9.2); best-val checkpoint selection per task.
   - Zero-shot results on ShopDB vs the EntityMean baseline; fine-tuning curves pretrained vs untrained (Fig. 3 analogue).
   - Paper hyperparameter table vs mini (§9).
14. **`playground`: Zero-shot playground and ablations.** Pick any ShopDB seed and run live. Edit in-context labels; remove self labels / other labels; shuffle column names; change L (Fig. 4 analogue); disable attention-layer types at inference.
   - Clearly label that the paper's layer ablations **retrain** the model; ours are inference-time.
   - Show the prediction and the attention change, side by side with the paper's ablation numbers (§11).
15. **`results`: What the paper found.** Interactive charts of Tables 1, 2, 4, 5, 10, 13, 15 (from spec §11). The ranked drivers of zero-shot transfer. Limitations (§7 of paper). All paper-vs-code discrepancies (spec §10) as a browsable list. Glossary. References.
16. **`about`: How this simulation was built and verified.** Pipeline, the parity tests (TS == PyTorch port == official model.py), what is scaled down, sources and licenses.

## 7. Global state (`src/state/`)

`useSim` (zustand), persisted in localStorage where sensible (wrapped in try/catch):

```
dbName='shop', taskId='shop/customer-churn', split='Val', seedNode (default: a Val seed with several past labels),
L=128, w=16, rngSeed=0, ckpt='auto' (auto = pretrain-<task> for zero-shot), selectedPos (token), block, layer, head,
disabledLayers=[], theme
```

Memoized hooks:

```
useDatabase()        → RawDatabase
usePreprocessed()    → PreprocessedDB
useContextWindow()   → SampledContext
useMasks()           → MaskSet
useForward()         → ForwardResult | null       (null while weights/embeddings load)
useAssets()          → { store, weights, status }
```

## 8. Quality gates

- `npm run typecheck`, `npm test`, `npm run build` must pass.
- Engine unit tests:
  - preprocessing stats on the spec §6.3 toy DB (μ, σ, ±0.866 churn values, population-std datetime);
  - sampler invariants (one target, seed first, F→P priority, time filter, width bound, L cut, padding);
  - masks equal the spec §6.3 printed masks for the spec's token list;
  - forward parity with the PyTorch fixtures.
- Visual QA: Playwright screenshots of every chapter in light/dark at 1440px and 390px; no console errors.
- Faithfulness review: each chapter's prose and visuals are checked against the spec by an adversarial reviewer.
