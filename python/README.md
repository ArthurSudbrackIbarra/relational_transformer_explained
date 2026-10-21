# python/: the PyTorch side of the simulation

Everything the website shows about training (curves, zero-shot results, checkpoints) is produced
offline by the scripts below. Contexts are built by the site's own TypeScript engine, so the model is
trained and evaluated on exactly the token sequences the chapters display. Ground truth for every
semantic is `research/RT_TECHNICAL_SPEC.md` (spec §), which cites the paper-era official code
(`PE/...`, commit `2a899c8`).

| file | role |
|---|---|
| `rt_model.py` | state_dict-identical port of `PE/rt/model.py` (dense masked attention instead of FlexAttention; RMSNorm eps pinned to 2⁻⁷) |
| `embed.py` | all-MiniLM-L12-v2 embeddings of every string → `public/data/embeddings/` |
| `train.py` | leave-one-DB-out pretraining, zero-shot evaluation, EntityMean baseline, fine-tuning, export |
| `export_weights.py` | state_dict ↔ the site's weight format (`public/data/weights/<name>.{json,bin}`) |
| `parity.py` | official `model.py` == `rt_model.py`; dumps the fixtures for `src/engine/model/forward.test.ts` |
| `label_stats.ts` | label persistence of every forecast task (window-to-window correlation, EntityMean per split) → `results.json → labelPersistence` |
| `ts_parity.ts` | TS forward vs PyTorch on every parity fixture (numbers for `verification.json`) |
| `sampler_dump.ts` + `sampler_crosscheck.py` | independent Python re-implementation of pre.rs + fly.rs + the masks, compared token by token with the TS engine |
| `write_verification.py` | composes `public/data/verification.json` from the outputs above + the vitest report |
| `data/` | (git-ignored) `strings.json`, exported contexts, training runs, verification intermediates |

## Environment

- Python venv: `python/.venv` (torch 2.12 CPU, sentence-transformers, einops, numpy, scikit-learn).
- Node 24 + npm for the TypeScript scripts (`npx tsx ...`).
- Everything runs on CPU in float32. The paper ran pure bf16 on 8×A100 (spec §9.2); this machine's GPU
  is too old for FlexAttention.
- Machine-specific: import scikit-learn (scipy) **before** torch in the same process (torch otherwise
  loads the system libstdc++, which lacks the CXXABI that miniconda's scipy needs). `train.py` does this
  and falls back to exact re-implementations of AUROC / R² if it cannot.
- Run the official FlexAttention model (`parity.py`) with torch's default thread count: with 8 threads
  the compiled CPU FlexAttention kernel of torch 2.12 aborts on this machine.

## Reproduce everything

Run all commands from the repository root.

```bash
# 1. strings → embeddings (re-run whenever a generator changes; db.test.ts checks coverage)
npx tsx scripts/export-strings.ts
python/.venv/bin/python python/embed.py

# 2. label persistence of the forecast tasks (→ python/data/label-stats.json → results.json)
npx tsx python/label_stats.ts

# 3. contexts (TS engine → python/data/contexts, ~800 MB, ~50 s)
npx tsx scripts/export-contexts.ts --pretrain-epochs 11 --finetune-epochs 104

# 4. measure the step time on this machine
python/.venv/bin/python python/train.py bench

# 5. pretraining (~19 min) + 4 fine-tuning runs (~5 min each) + export (resumable: state.pt is saved
#    every 2 min and at the budget, so a killed run continues where it stopped; repeat until it prints
#    "ALL DONE"; exit code 3 = budget reached). Batches are a pure function of the step, so a resumed
#    run is the same run. OMP_WAIT_POLICY=PASSIVE + 6 threads: much faster than 8 spinning threads
#    when other processes share the CPU.
OMP_WAIT_POLICY=PASSIVE python/.venv/bin/python python/train.py all --threads 6 --pretrain-steps 4001 \
    --pretrain-eval-freq 100 --finetune-steps 1025 --finetune-eval-freq 64 --budget-min 100

# 6. verification: official model.py == port (8 checkpoints, 28 real contexts of all four DBs) and the
#    fixtures; TS == PyTorch; independent sampler cross-check; all tests; verification.json
python/.venv/bin/python python/parity.py --stage all
npx tsx python/ts_parity.ts
npx tsx python/sampler_dump.ts python/data/sampler-dump.json
python/.venv/bin/python -I python/sampler_crosscheck.py python/data/sampler-dump.json --json python/data/sampler-crosscheck.json
npx vitest run --reporter=default --reporter=json --outputFile=python/data/vitest.json
python/.venv/bin/python python/write_verification.py
```

`train.py export` re-writes the JSON outputs and weights from the finished runs without training.
`train.py baselines` prints the EntityMean baseline on every evaluation set.

## What each step does

### `scripts/export-contexts.ts`

For every database it runs `getDatabase → preprocess` (pre.rs mirror) and, per seed,
`sampleContext` (fly.rs mirror) at **L = 128, w = 16**, with the PRNG seed
`epoch + seed_node_idx + seed`, `seed = 0` (fly.rs:371-375, spec §4.2). Sets:

| set | content |
|---|---|
| `pretrain` | all tasks of school + racing + forum (leave-one-DB-out for target `shop`, spec §9.1): forecast tasks → Train split; autocomplete → **all rows** of the DB table (data.py:77-81, spec §2.4). One item list over all tasks, no task balancing (spec §4.14). Stored epoch-major in item order, one context per item per epoch. |
| `eval/shop/<task>/<Val\|Test>` | zero-shot evaluation, forecast tasks, epoch 0 (main.py:146-168) |
| `eval/shop/<table>.<col>/All` | shop autocomplete tasks, epoch 0. All 303 review rows are seeds (split ignored, spec §2.4); `train.py` scores the even review_ids as "Val" (checkpoint selection) and the odd ones as "Test" (reported), plus "All" |
| `diag/<db>/<task>/Val` | forecast Val splits of the pretraining DBs (in-distribution sanity check only) |
| `finetune/shop/<task>` | target task Train split, many epochs (example_finetune.py:18) |

Batch layout per token (spec §5.1), compact: int16 ids (`node_idxs`, `f2p_nbr_idxs[5]`,
`table_name_idxs`, `col_name_idxs`, `text_idxs`), uint8 `sem_types` and `flags`
(bit0 `masks`, bit1 `is_targets`, bit2 `is_padding`, bit3 `is_task_nodes`), and ONE float32 `values`
array: `number_values = values·[sem=0]`, `datetime_values = values·[sem=2]`,
`boolean_values = values·[sem=3]` (pre.rs stores 0 in the other slots; the exporter asserts it).
`text_idxs` is the per-DB text index (0 for non-text cells as in pre.rs, −1 for padding);
`index.json` holds every DB's text list, so Python maps index → string → MiniLM row of
`public/data/embeddings/`. Per context it also stores the task, seed, epoch, target position, label,
the EntityMean prediction and in-context label counts.

### `train.py`

Mirrors `PE/rt/main.py` (see the module docstring for the line-by-line list):

- **Model**: `rt_model.RelationalTransformer(num_blocks=4, d_model=64, d_text=384, num_heads=4,
  d_ff=256, eps=2⁻⁷)`, init = PyTorch defaults after `torch.manual_seed(0)` (`seed_everything(0)`).
- **Pretraining**: batch 32; AdamW(lr 1e-3, betas (0.9, 0.999), eps 1e-8, wd 0.1, one param group);
  OneCycleLR(max_lr 1e-3, total_steps = max_steps, pct_start 0.2, linear, cycle_momentum, div 25,
  final_div 1e4); grad clip 1.0; the model's own loss. Each epoch the item list is shuffled with seed
  `epoch + 0` and batch b takes items `(b·32 + i) mod N` (fly.rs:324-355).
- **Evaluation** at step 0, every `eval_freq` steps and at powers of two, *before* that step's
  update (main.py:373-375): AUROC on logits for binary tasks, R² in z-space for regression
  (main.py:283-337). Checkpoint selection **per target task on ShopDB val** (main.py:377-392,
  spec §9.1): the test metric reported is the one of the selected step. We select on all four ShopDB
  tasks: the two forecast tasks (paper protocol) and, as an extension (the paper's `eval_tasks` are
  forecast tasks only, example_pretrain.py:18), the two autocomplete tasks on their even/odd halves.
- **EntityMean** (paper App. I): mean of the entity's past labels of the same task table (any split,
  strictly earlier timestamp); fallback = Train-split mean. Computed by the exporter on the same seeds.
- **Fine-tuning** (spec §9.2): target task Train split only; fresh AdamW, constant lr 1e-4, wd 0,
  clip 1.0; from the pretrain-best checkpoint of that task and from the same random init
  ("untrained RT"); eval at powers of two and every `eval_freq` steps; best val kept.
- **Attention kernel**: training swaps `rt_model.dense_masked_attention` for
  `F.scaled_dot_product_attention` with the same boolean mask after checking on random data (with
  fully masked rows) that outputs and gradients agree to ≤ 1e-5. Same maths, ~5× faster on CPU. The
  exported weights are verified against the official FlexAttention class by `parity.py`.

Outputs:

| file | content |
|---|---|
| `public/data/training/curves.json` | per run: train loss, grad norm (and lr, β1 for pretraining) per step; every evaluation |
| `public/data/training/results.json` | zero-shot val/test per target task, all four ShopDB tasks (pretrain-best, pretrain-final, untrained init, EntityMean for forecast tasks), selected steps, in-distribution metrics, fine-tuning results, checkpoint provenance, in-context-label ablation, `labelPersistence` |
| `public/data/training/schedule.json` | lr / β1 per step of our run, and the paper's 50,001-step OneCycle schedule (sampled) |
| `public/data/training/dataset.json` | #contexts, #tasks, #tokens per set; tokens seen vs the paper's ≈13B |
| `public/data/weights/*.{json,bin}` | `init`, `pretrain-churn`, `pretrain-sales`, `pretrain-rating`, `pretrain-verified`, `pretrain-final`, `finetune-churn`, `finetune-sales` (8 × 2.1 MB) |

### `parity.py`

- `--stage random` (stage 1): synthetic context, random weights; dumps `tests/fixtures/parity-random*`.
- `--stage trained` (stage 2): (a) loads every exported checkpoint into the **official**
  `PE/rt/model.py` class (FlexAttention, CPU; RMSNorms pinned to 2⁻⁷) and into `rt_model.py`, and
  compares all decoder outputs and the loss on a batch of 28 real contexts (both ShopDB forecast tasks,
  val and test, both ShopDB autocomplete tasks, and the forecast Val sets of SchoolDB, RacingDB and
  ForumDB); (b) dumps `tests/fixtures/parity-trained*`: the default `shop/customer-churn` Val context
  (seed node 2406, `pipeline.pickDefaultSeed`) with `pretrain-churn` (own weights copy), a
  `shop/product-sales` Val context with `pretrain-sales` (number head) and a `forum/post-votes` Val
  context with `pretrain-final` (a pretraining DB), each with every activation. `forward.test.ts`
  rebuilds each context with the TS engine, checks it token by token against the export, and checks
  TS forward == PyTorch (≤ 1e-4). The run's summary goes to `python/data/parity-report.json`.

## The pretraining data: persistent self labels (data version 2)

The paper's main finding is that zero-shot transfer is driven by the target entity's own past labels
(self labels; removing them drops zero-shot R² from 22.8 to −5.5, Table 4, spec §11.2). The first
version of our pretraining DBs did not reward using them for regression: EntityMean R² (predict the
mean of the entity's past labels) was −0.77 on school course-submissions and −0.43 on forum
post-votes, and racing driver-dnf labels had no persistence (window-to-window correlation −0.04).
The generators were changed so that entity behaviour is persistent, as in real RelBench data
(rel-stack post-votes: EntityMean R² 43.7). ShopDB is untouched (`db.test.ts` checks its SHA-256).

| DB | change |
|---|---|
| SchoolDB | Every course runs in every regular term (before: many fall-only / spring-only courses, so a course's monthly count flipped to 0 every other term). Each course has a persistent log-normal popularity that drifts slowly from term to term and drives enrolment; coursework is two assignments per credit, evenly spaced; most students join in January 2024; six summer courses. 4,277 submissions (was 2,271). |
| RacingDB | Δ = 30 days (RelBench rel-f1 driver-dnf: "in the next 1 month") and the shared cutoffs val 2025-03-01 / test 2025-04-15 (was Δ = 60 days, val 2025-02-14): 12 train timestamps instead of 5. A driver's reliability propensity is the main DNF driver (logit −2.8 + 2.0·driver + 1.2·team), so DNFs cluster on the same drivers; less race-day noise; the calendar runs mid-January 2024 to 1 December 2024, then from 12 January 2025, so every 30-day window holds 1–3 races (1 only in December 2024). |
| ForumDB | Votes are generated per post: a post receives votes at rate appeal × attention, attention decaying slowly with age (e-folding 250 days) to a 40 % floor, independent of how many other posts exist (before, a fixed vote budget was spread over a growing pool of posts, so every post's count shrank over time). Heavy-tailed appeal (quality + a post-specific factor). 180 users, 198 posts, 3,587 votes (was 100 / 307 / 2,235). |

Label persistence now (`npx tsx python/label_stats.ts`; `results.json → labelPersistence`). lag-1 =
Pearson correlation of an entity's labels in consecutive Train windows; EntityMean = AUROC (binary)
or R² (regression) of the EntityMean baseline:

| task | lag-1 | EntityMean Train | Val | Test |
|---|---|---|---|---|
| shop/customer-churn (unchanged) | 0.47 | 0.69 | 0.71 | 0.77 |
| shop/product-sales (unchanged) | 0.69 | 0.45 | 0.76 | 0.75 |
| school/student-dropout | 0.65 | 0.75 | 0.80 | 0.85 |
| school/course-submissions | 0.73 (was 0.51) | 0.37 (was −0.30) | 0.86 (was −0.77) | 0.89 |
| racing/driver-dnf | 0.42 (was −0.04) | 0.80 (was 0.54) | 0.72 | 0.98 |
| racing/driver-points | 0.66 | 0.43 (was 0.24) | 0.17 | 0.39 |
| forum/user-engagement | 0.46 | 0.67 | 0.72 | 0.60 |
| forum/post-votes | 0.73 (was 0.40) | 0.48 (was −0.60) | 0.41 (was −0.43) | 0.62 |

`db.test.ts` asserts lag-1 ≥ 0.4 for every pretraining forecast task, EntityMean R² > 0.3 (regression)
or AUROC > 0.6 (binary) on Train. Over 8–16 other generator seeds the same processes give lag-1
0.37 ± 0.10 for driver-dnf (the realised seed is at 0.42; a 2-race binary window is intrinsically
noisy), 0.65 ± 0.05 for course-submissions and 0.75 ± 0.07 for post-votes; EntityMean R² on Train
0.35 ± 0.08 (course-submissions) and 0.32 ± 0.11 (post-votes). Val/Test sets are small (22–172
rows), so their EntityMean numbers move a lot between seeds (driver-points Val 0.17 here).

The pretraining item list grew from 10,521 to 16,220 items per epoch (forecast Train rows 3,621,
autocomplete rows 12,599: proportional to seed counts, no balancing, spec §4.14).

## Results (data version 2; single seed, CPU float32)

Pretraining: 4,001 steps × 32 contexts × 128 tokens ≈ 16.4M tokens (≈7.9 epochs of 16,220 items,
10 tasks of school + racing + forum), 1,156 s wall time (paper: 50,001 × 256 × 1024 ≈ 13.1B tokens).
The training loss (mean of 100 steps) goes from 0.447 to 0.308.

Zero-shot on ShopDB (never seen in pretraining). The checkpoint is selected per task on ShopDB val, and
"test" is the test metric at that step (spec §9.1). Eval sets are tiny (churn 71/74 seeds, sales 30/30,
reviews 152/151), so differences of a few points are noise. EntityMean exists only for forecast tasks
(autocomplete rows have no entity history).

| task (metric) | untrained init | EntityMean | RT pretrain-best (step) | RT pretrain-final (4000) | data v1 pretrain-best |
|---|---|---|---|---|---|
| customer-churn (AUROC) val / test | 0.690 / 0.755 | 0.713 / 0.769 | **0.749 / 0.779** (2500) | 0.723 / 0.791 | 0.745 / 0.771 |
| product-sales (R²) val / test | −0.377 / −0.276 | **0.756 / 0.748** | 0.609 / 0.585 (600) | 0.573 / 0.549 | 0.465 / 0.558 |
| reviews.rating (R²) even / odd ids | −0.353 / −0.345 | – | **0.111 / 0.096** (400) | −0.018 / 0.151 | – |
| reviews.verified (AUROC) even / odd ids | 0.585 / 0.547 | – | **0.635 / 0.584** (2700) | 0.599 / 0.558 | – |

How to read this:
- **Churn: RT ≥ EntityMean** (val +0.036, test +0.010), as in the paper where zero-shot RT is close to
  or above EntityMean on classification. **The untrained init's 0.69 / 0.75 churn AUROC is chance**: with
  init seeds 1–6 it is 0.31–0.66 val (`results.json → diagnostics.untrainedInitSeeds`).
- **Sales: RT is far above init and better than on data v1, but still below EntityMean** (val 0.61 vs
  0.76, test 0.58 vs 0.75). Making the pretraining labels persistent raised the zero-shot sales val R²
  from 0.47 to 0.61 (and from 0.18 to 0.57 at the final step), but the mini model does not learn to
  copy the entity's past values as well as the hand-made EntityMean. The paper's full RT also stays
  below EntityMean on 2 of its 8 zero-shot regression tasks (rel-amazon item-ltv 32.5 vs 54.2, rel-stack
  post-votes 33.9 vs 43.7; Table 2 p.8, spec §11.1), and we train on 1/800 of its tokens.
- **Autocomplete tasks** transfer weakly: reviews.rating R² 0.10 (init −0.35), reviews.verified AUROC
  0.58–0.64. The verified AUROC is inside the range that untrained inits reach by chance (0.42–0.65 on
  the even ids), so it is not evidence of transfer.
- **What transfers is reading in-context labels.** Removing the visible target-column cells
  (`diagnostics.inContextLabelAblation`) drops pretrain-churn from 0.749 to 0.544 val AUROC,
  pretrain-sales from R² 0.609 to 0.052, pretrain-rating from 0.111 to 0.004 and pretrain-verified from
  0.635 to 0.388. This is the paper's main finding (self labels drive zero-shot, Table 4, spec §11.2).
- **Zero-shot peaks early for sales and rating** (steps 600 and 400) while the pretraining loss keeps
  falling and the in-distribution metrics keep improving: forum post-votes val R² −0.43 → 0.51 (above its
  EntityMean 0.41), racing driver-dnf val AUROC 0.54 → 0.78 (EntityMean 0.72), school course-submissions
  val R² −0.82 → 0.82 (EntityMean 0.86) (`extraZeroShotAndDiagnostics`). Churn and verified peak late
  (2500, 2700). The paper reports the same overfitting to the source DBs (p.6) and answers it with
  per-task best-val selection, which we use: every ShopDB task gets its own checkpoint.

Fine-tuning (1,025 steps, lr 1e-4 constant, best val; `finetune-*` in `results.json`):

| task | pretrained init: best val / its test (step) | untrained init: best val / its test (step) |
|---|---|---|
| customer-churn (AUROC) | 0.775 / 0.717 (192) | 0.753 / 0.663 (448) |
| product-sales (R²) | 0.854 / 0.679 (576) | 0.859 / 0.491 (1024) |

These are Fig. 3 analogues. Sales: the pretrained model starts at val R² 0.61 where the untrained one
starts at −0.38; at step 16 it is 0.60 vs 0.26 and at step 64 0.68 vs 0.55, and the curves meet around
step 128. The untrained model reaches the same best val later (step 1024) with a much worse test R²
(0.49 vs 0.68). Churn: the pretrained model starts at the AUROC the untrained one reaches after ~64
steps, and its best val is higher. Both overfit later (test drops after step ~128–256). The untrained
runs are identical to data version 1 (same ShopDB, same init).

Data version 1, for the record (`results.json → previousDataVersion`): churn 0.745 / 0.771 (step 1100),
sales 0.465 / 0.558 (step 64, then decaying to 0.18 / 0.06 at the end); fine-tuning from pretrain-sales
0.865 / 0.587. A run with Adam square-root LR scaling (peak 3.54e-4) did not help on v1 and was not
repeated.

Parity (`public/data/verification.json`): official `model.py` (compiled FlexAttention) == `rt_model.py`
on all 8 checkpoints and 28 real contexts of all four DBs, max |Δ| ≤ 1.8e-6 (random-weight stage
included); TS == PyTorch on the 4 fixtures, max |Δ| ≤ 2.7e-6 at every stage; the independent sampler
cross-check rebuilds 2,144 contexts with 0 mismatches; 206 tests pass. `parity.py --stage all` runs its
two stages in separate processes: in one process, stage 1's failed FlexAttention backward on CPU makes
dynamo run flex_attention eagerly afterwards, which matched the port bit for bit (|Δ| = 0) and hid
which kernel stage 2 compared against.
