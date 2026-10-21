# 03 — Training, fine-tuning and evaluation of the Relational Transformer (paper-era code)

Scope: what `rt/main.py`, `rt/data.py`, `rt/tasks.py`, `scripts/example_*.py` and `README.md` in the
paper-era snapshot do, checked against the paper (arXiv 2510.06377v3, ICLR 2026). Where the training
behaviour depends on the Rust sampler or preprocessor (`rustler/src/fly.rs`, `rustler/src/pre.rs`),
those lines are cited too.

Path abbreviations (all line numbers are 1-indexed):

- `PE/` = `/tmp/claude-1004/-home-ibarra-Documents-Mestrado-Relational-Transformer/155ecce6-1f67-4870-9609-2c9740593feb/scratchpad/upstream/rt-paper-era/` (commit `2a899c8`, 2026-02-04)
- `CUR/` = `/tmp/claude-1004/-home-ibarra-Documents-Mestrado-Relational-Transformer/155ecce6-1f67-4870-9609-2c9740593feb/scratchpad/upstream/rt-official/` (current main, RT-J era)
- "paper p.N" = page N of the 26-page PDF `/home/ibarra/Documents/Mestrado/Relational_Transformer/Relational_Transformer.pdf`

Version notes:
- The authors' README on current main points to the git tag **`rt-v1`** as the paper's code (`CUR/README.md:28`). The tag `rt-v1` = `8d83590` sits 4 commits after `2a899c8`. Those 4 commits only add a `timestamps` output field, an `RT_SAVE_ALL_STEPS` env switch, and two leaderboard prediction scripts (`git diff 2a899c8 rt-v1`). The training and eval logic is the same.
- In the ICLR-submission commit `eeccc2b` (2025-10-06), the block order was `["feat","nbr","col","full"]`. Commit `7f891d1` (2025-10-08, "fix in attention order and sampling") changed it to `["col","feat","nbr","full"]`, the order in paper Alg. 2 (p.26) and `PE/rt/model.py:83`.
- Pre-submission commits `ab17c23`…`6cae3ef` (Jul–Aug 2025) had a multi-cell masking sampler (`mask_prob`). The submission commit removed it (see §2.6).

---

## 1. Model config defaults, parameter count, precision, compilation

### 1.1 Hyper-parameters (identical in all three example scripts)

| arg | value | source |
|---|---|---|
| `num_blocks` | 12 | `PE/scripts/example_pretrain.py:32` (same at contd:34, finetune:33) |
| `d_model` | 256 | `example_pretrain.py:33` |
| `num_heads` | 8 (so head dim = 32) | `example_pretrain.py:34` |
| `d_ff` | 1024 (SwiGLU hidden) | `example_pretrain.py:35` |
| `d_text` | 384 | `example_pretrain.py:30` |
| `embedding_model` | `all-MiniLM-L12-v2` | `example_pretrain.py:29` |
| `seq_len` (context length, in cells) | 1024 | `example_pretrain.py:31` |
| `max_bfs_width` (w in Alg. 1) | 256 | `example_pretrain.py:21` |
| `batch_size` | 32 per GPU; README runs `torchrun --nproc_per_node=8`, so the global batch is **256** | `example_pretrain.py:19`, `PE/README.md:128` |
| `num_workers` | 8 | `example_pretrain.py:20` |
| `max_grad_norm` | 1.0 | `example_pretrain.py:26` |
| `seed` | 0 | `example_pretrain.py:15` |

`main()` has no defaults. Every argument is passed explicitly and `strictfire` exposes it as a CLI (`PE/rt/main.py:66-97, 454-457`).
Paper p.6 §4.1 "Architecture details" gives the same numbers: 12 layers, hidden 256, 8 heads, gated SiLU MLP with hidden 1024, MiniLMv2 384-d. Fig. 6 (p.15) uses L=512 and w=128 for visualisation only. Training uses 1024 and 256.

### 1.2 Parameter count: 22,336,131

`main.py:182-184` prints `sum(p.numel() for p in net.parameters())`. I re-implemented the module shapes of `PE/rt/model.py` independently, without importing repo code (script at `/tmp/claude-1004/-home-ibarra-Documents-Mestrado-Relational-Transformer/155ecce6-1f67-4870-9609-2c9740593feb/scratchpad/scripts/paramcount.py`). It gives:

| component | params |
|---|---|
| per block: 4 attentions × (wq, wk, wv, wo; 256×256, no bias) | 1,048,576 |
| per block: SwiGLU FFN w1, w3 (256→1024) + w2 (1024→256), no bias | 786,432 |
| per block: 5 RMSNorm weights (feat, nbr, col, full, ffn) | 1,280 |
| **per block total** | **1,836,288** |
| 12 blocks | 22,035,456 |
| encoders: number/datetime/boolean Linear(1,256)+b = 512 each; text/col_name Linear(384,256)+b = 98,560 each | 198,656 |
| decoders: number/datetime/boolean Linear(256,1)+b = 257 each; text Linear(256,384)+b = 98,688 | 99,459 |
| 5 input RMSNorms (`norm_dict`) | 1,280 |
| 4 mask embeddings (`mask_embs`, 256 each) | 1,024 |
| `norm_out` | 256 |
| **total** | **22,336,131** |

This matches "about 22M" (paper p.6) and "22.3M" (paper p.26 App. I.4). The frozen MiniLM text encoder is not part of the model. Text embeddings are precomputed offline (`PE/rt/embed.py:141-179`) and read from a memory-mapped bf16 file (`PE/rustler/src/fly.rs:263-265, 528-532`). The 98,688-parameter **text decoder is never trained**: text cells are never masked, and masking one would raise (`PE/rt/model.py:256-257`). It only gets a zero-gradient "touch" so DDP does not complain about unused parameters (`model.py:243-248`, added in commit `e55bf0d`).

### 1.3 Precision: pure bf16, not autocast

- `net = net.to(device); net = net.to(torch.bfloat16)` (`main.py:186-187`). All weights are bf16. There is no `torch.autocast`, no fp32 master copy and no GradScaler anywhere in `main.py`.
- Inputs arrive as bf16. Rust writes `bf16` (`fly.rs:33-38, 482-490`) and Python reinterprets the bits (`PE/rt/data.py:120-127`).
- AdamW (`fused=True`) keeps its state in the parameter dtype. I checked locally with torch 2.12: `exp_avg`/`exp_avg_sq` are bf16 and `step` is fp32.
- A consequence (my inference from a local test, not something the authors state): with round-to-nearest bf16 weights, an Adam step of size ≈lr=1e-4 is rounded away for weights of magnitude ≳0.03. For example, 0.0600586 stays unchanged after one step, while 0.01 moves by one ulp. Small-magnitude weights still update.
- The paper says "8×A100 GPUs at BFloat16 precision" (p.6 §4.1).
- `nn.RMSNorm(d_model)` is built with the default `eps=None`, which resolves to `torch.finfo(dtype).eps`. For bf16 that is **0.0078125**. I checked this locally; it matters only if the site prints RMSNorm numerics.

### 1.4 torch.compile and FlexAttention

- `flex_attention = torch.compile(flex_attention)` at import (`PE/rt/model.py:17`). Block masks come from `create_block_mask(..., _compile=True)` (`model.py:89-101`). The mask is rebuilt on every forward from four dense boolean (B,S,S) masks (`model.py:164-206`).
- The whole model is compiled with `net = torch.compile(net, dynamic=False, disable=not compile_)` (`main.py:209`), with `compile_=True` in all scripts. Compilation is applied **after** DDP wrapping (`main.py:206-209`).
- Dynamo settings: `cache_size_limit = 16`, `compiled_autograd = compile_ if ddp else False`, `optimize_ddp = True`, `torch.set_num_threads(1)` (`main.py:117-121`).
- The `SDPBackend.FLASH_ATTENTION` branch in `MaskedAttention` runs only when `block_mask is None` (`model.py:39-43`). That never happens, because all four masks are always passed (`model.py:204-206`).
- Pinned environment: `torch==2.6.0`, `scikit-learn<1.6.0`, Python 3.12 (`PE/pyproject.toml:3,17,25`).
- Distributed: DDP over NCCL when `LOCAL_RANK` is set (`main.py:100-111, 206-207`).
- Not present anywhere in the paper-era code: gradient checkpointing, gradient accumulation, dropout, EMA, mixed-precision scaler.

---

## 2. Pretraining

### 2.1 Task inventory (`PE/rt/tasks.py`)

Each tuple is `(db, table, target_column, leakage_columns_to_drop)` (`tasks.py:1`).

- `forecast_clf_tasks`, 12 tuples (`tasks.py:3-16`): amazon user-churn, hm user-churn, stack user-badge, amazon item-churn, stack user-engagement, avito user-visits, avito user-clicks, event user-ignore, trial study-outcome, f1 driver-dnf, event user-repeat, f1 driver-top3.
- `forecast_reg_tasks`, 9 tuples (`tasks.py:18-28`): hm item-sales, amazon user-ltv, amazon item-ltv, stack post-votes, trial site-success, trial study-adverse, event user-attendance, f1 driver-position, avito ad-ctr.
- `autocomplete_clf_tasks`, 7 tuples (`tasks.py:30-64`): avito SearchInfo.IsUserLoggedOn, stack postLinks.LinkTypeId, amazon review.verified, trial studies.has_dmc, trial eligibilities.adult (drops 7 leakage cols), trial eligibilities.child (drops 7), event event_interest.not_interested (drops `interested`).
- `autocomplete_reg_tasks`, 8 tuples (`tasks.py:66-88`): amazon review.rating (drops review_text, summary), f1 results.position (drops 7), f1 qualifying.position, trial studies.enrollment, f1 constructor_results.points, f1 constructor_standings.position (drops wins, points), hm transactions.price, event users.birthyear.
- `all_tasks` = all 36 tuples (`tasks.py:90-95`). `forecast_tasks` = 21 tuples (`tasks.py:97`).

These match paper App. D Tables 8–9 (p.19).

### 2.2 Leave-one-DB-out: what is trained and what is evaluated

`example_pretrain.py:17-18` uses `rel-amazon` as the held-out DB:

```python
train_tasks = [t for t in all_tasks if t[0] != "rel-amazon"]      # forecast + autocomplete of the 6 other DBs
eval_tasks  = [t for t in forecast_tasks if t[0] == "rel-amazon"] # the 4 amazon forecasting tasks
```

- Training uses only the `"train"` split of each task (`main.py:123-127`).
- **rel-event tasks are included in pretraining** for every other held-out DB, because `all_tasks` contains them. The paper only says it does not *evaluate* on rel-event (p.6: "We skip rel-event, as it has been found to have temporal leakage issues").
- "We pretrain separately for each target dataset on all tasks from all other datasets" (p.6). That means one pretraining run per held-out DB, evaluating all of that DB's forecasting tasks (val and test) during the run.

### 2.3 How items and batches are formed (task mixing)

1. **Items per task** (`PE/rt/data.py:63-94`, `fly.rs:257-297`). For each task tuple, `table_info_key` is `"<table>:Db"` if the table is a DB table (autocomplete tasks), else `"<table>:Train|Val|Test"` (forecast tasks) (`data.py:77-81`). Every node in `[node_idx_offset, node_idx_offset+num_nodes)` becomes an item if it **has a non-missing cell in the target column** (`fly.rs:287-296`). The target column is identified by the column-name index of the string `"<target_column> of <table>"` (`data.py:31-43`).
2. **One global item list** across all tasks and DBs (`fly.rs:256-297`), shuffled with `StdRng(epoch + seed)` (`fly.rs:324-328`). The shuffle is re-run at the start of every pass through the loader with `epoch = int(steps / len(loader))` (`main.py:370`). There is **no task balancing or temperature sampling**. Tasks are mixed in proportion to their item counts, and a single batch freely mixes DBs, tasks, and classification with regression.
3. **Per-rank slicing.** `len = ceil(N_items / (batch_size·world_size))` (`fly.rs:332-334`). Batch `b` on rank `r` takes items `b·B·W + r·B + i` for i < B, modulo N, so the last batch wraps around (`fly.rs:336-355`). Each sequence is built independently from its seed item (`fly.rs:357-511`). The sequence RNG is `StdRng(epoch + seed_node_idx + seed)` (`fly.rs:371-375`), so the same seed row gets a **different BFS subsample every epoch**.
4. During training, `true_batch_size` is popped and ignored (`main.py:401`). Wrapped duplicates in the last batch are trained on normally.
5. `DataLoader(batch_size=None, num_workers=8, persistent_workers=False, pin_memory=True, in_order=True)` (`main.py:137-144`). Each `__getitem__` returns a whole pre-batched dict from Rust (`data.py:116-154`).

Batch dict produced by the sampler (`fly.rs:141-206`, reshaped in `data.py:133-152`), with B=32 and S=1024:

| key | shape / dtype | meaning |
|---|---|---|
| `node_idxs` | (B,S) i32 | row id of each cell's row (−1 = pad) |
| `f2p_nbr_idxs` | (B,S,5) i32 | row ids of the row's F→P parents (−1 = none; `MAX_F2P_NBRS=5`, `fly.rs:25`) |
| `table_name_idxs`, `col_name_idxs` | (B,S) i32 | text-vocab ids of the table name and of "`<col> of <table>`" |
| `col_name_values` | (B,S,384) bf16 | frozen MiniLM embedding of "`<col> of <table>`" |
| `sem_types` | (B,S) i32 | 0 number, 1 text, 2 datetime, 3 boolean (`PE/rustler/src/common.rs:38-44`) |
| `number_values`, `datetime_values`, `boolean_values` | (B,S,1) bf16 | z-scored values (0 where the type does not apply) |
| `text_values` | (B,S,384) bf16 | MiniLM embedding of the cell text |
| `masks` | (B,S) bool | cells whose value is hidden and that are trained on |
| `is_targets` | (B,S) bool | the seed row's target cell |
| `is_task_nodes` | (B,S) bool | cell belongs to a task-table row, or to a target-column cell (`fly.rs:498-499`); **not used by model or training** |
| `class_value_idxs` | (B,S) i32 | text id for text cells; **not used by model or training** |
| `is_padding` | (B,S) bool | pad positions |
| `true_batch_size` | int | number of real (non-wrapped) sequences |

`is_task_nodes` is only read by `scripts/visualize_rustler.py:22,53`, for colouring. `class_value_idxs` is not read anywhere in `rt/`. I grepped `PE/rt` and `PE/scripts` to confirm.

### 2.4 What is masked during pretraining: exactly one cell per sequence

```rust
// PE/rustler/src/fly.rs:492-496
let is_target = seed_node_idx == node.node_idx && node.col_name_idxs[cell_i] == target_column;
slices.is_targets[seq_i] = is_target;
slices.masks[seq_i] = is_target;
```

- `masks == is_targets`. The only masked cell is the target cell of the seed row. **No other cells are masked in the released paper-era code** (no BERT-style random masking).
- The seed row is visited first (`fly.rs:366, 378-379`) and its cells are written first, so the target cell is effectively always in the context.
- The masked cell keeps its column-name embedding and its datatype. The input uses the type-specific mask vector `mask_embs[t]` instead of the value embedding (`model.py:216-224`), so the model knows *which* column and *which datatype* it must predict.
- All other cells are unmasked, **including the target-column cells of other task-table rows** (past labels of the same entity and labels of other entities). These are the "in-context labels" (paper p.2, p.23 Table 12).
- Which task tables enter the context: a P→F edge into a task table is followed only if that task table is the seed's own table (`fly.rs:416-419`: skip if `edge.table_name_idx != seed_node.table_name_idx && edge.table_type != Db`). This implements "Only one task table is active at a time" (paper p.4).
- Leakage columns (`columns_to_drop`) are removed from the seed row and from any row with the same timestamp as the seed (`fly.rs:457-461`). This is how autocomplete leakage is handled (paper p.19 App. D). A Rust detail: `None == None` is true, so for a seed without a timestamp, the drop also applies to every timestamp-less row in the context.

**Table 14 (multi-cell masking, P(mask) ∈ {0, 0.2, 0.4}, paper p.24)** is an ablation. Its caption: "Masked cells contribute to the loss. The target cell is always masked. Other cells are masked with probability P(mask)". The **main results correspond to P(mask)=0.0**: the 0.0 column (70.1 AUROC, 22.8 R²) equals the "none" column of Tables 4/11 (p.9, p.22). The released code implements only P(mask)=0. The ablation sampler is not in the repo.

For historical context only: before the submission (commit `6cae3ef`, `rustler/src/fly.rs:356-361, 516-521` at that commit), masking was `mask = (Number && seed row) || (Number && rng.random_bool(mask_prob))`. Here `mask_prob = 0.5 · gain`, and `gain` decays as a cosine to 0 over `cos_steps = max_steps − zero_mask_prob_steps` (`rt/main.py:72-73, 143` at that commit). There were also flags `mask_db_cells` / `mask_task_cells`. This code is gone from the paper-era snapshot, and it is not known whether Table 14 used it.

### 2.5 Loss (from `model.py:231-265`, restated for training)

```
x   = norm_out(blocks(x0))                   # (B,S,256)
for t, i in [(number,0),(text,1),(datetime,2),(boolean,3)]:
    yhat_t = dec_t(x)                        # (B,S,1) or (B,S,384); computed at ALL positions
    m_t    = (sem_types == i) & masks
    number/datetime: l = huber_loss(yhat_t, y_t, delta=1.0)        # on z-scored values
    boolean:         l = BCEWithLogits(yhat_t, 1[y_t > 0])        # y_t is the z-scored boolean
    loss += sum(l * m_t)
loss = loss / masks.sum()                    # = mean over the 32 target cells of this rank
```

- Huber δ is the PyTorch default, 1.0 (`F.huber_loss(..., reduction="none")` at `model.py:251`; default verified locally).
- Booleans are z-scored like numbers in preprocessing (`PE/rustler/src/pre.rs:553-557`: `(1.0 or 0.0 − mean)/std`), so `y > 0` recovers the original truth value: true → (1−μ)/σ > 0, false → −μ/σ < 0.
- With one masked cell per sequence, the per-rank loss is the plain average over 32 sequences. A sequence's term is BCE if its target is boolean and Huber if its target is numeric. **There is no per-task or per-type loss weighting.** DDP averages gradients over the 8 ranks, giving the mean over the 256 sequences of the global batch. Training loss is all-reduced (AVG) for logging only (`main.py:426-427`).
- The loss function is chosen by the **sem_type of the target cell**, which comes from the parquet dtype after the per-dataset casts in `pre.rs:208-289`. It is not chosen by the clf/reg list in `tasks.py`. Notable cast: `rel-event/user-attendance` (listed as a regression task, `tasks.py:25` and `main.py:235`) has its `target` cast to Boolean (`pre.rs:272-278`). In pretraining it is therefore trained with BCE as "attended ≥ 1". It is never evaluated in the paper.
- This matches paper p.6 §3.3: "HuberLoss(r, r′) for regression and binary cross-entropy loss BCE(1{r > 0}, r′) … The overall loss is the mean over all masked cells in the batch. This formulation is used in both pretraining and fine-tuning."

### 2.6 Optimizer, LR schedule, steps, clipping

```python
# main.py:188-204
opt = optim.AdamW(net.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.999), eps=1e-8, fused=True)
if lr_schedule:
    lrs = optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=max_steps,
                                        pct_start=0.2, anneal_strategy="linear")
```

Pretraining values (`example_pretrain.py:23-27`): `lr=1e-3`, `wd=0.1`, `lr_schedule=True`, `max_grad_norm=1.0`, `max_steps=50_001`.

- **Weight decay applies to every parameter**: one param group, including RMSNorm weights, biases and `mask_embs`.
- **Initialization** is PyTorch default (Linear: Kaiming-uniform; RMSNorm weight = 1; `mask_embs = torch.randn(d_model)` ~ N(0,1), `model.py:141-146`). There is no custom init. `seed_everything(seed)` sets python, numpy and torch seeds (`main.py:21-29, 98`).

**The OneCycleLR defaults the code relies on** (PyTorch docs for 2.6: `pct_start=0.3` overridden to 0.2, `anneal_strategy` overridden to linear, `cycle_momentum=True`, `base_momentum=0.85`, `max_momentum=0.95`, `div_factor=25`, `final_div_factor=1e4`; https://docs.pytorch.org/docs/2.6/generated/torch.optim.lr_scheduler.OneCycleLR.html). I simulated the exact call with local torch 2.12; the script is `/tmp/claude-1004/-home-ibarra-Documents-Mestrado-Relational-Transformer/155ecce6-1f67-4870-9609-2c9740593feb/scratchpad/scripts/onecycle.py`.

| step | lr | AdamW β1 |
|---|---|---|
| 0 | 4.00e-5 (= 1e-3/25) | 0.950 |
| 1,000 | 1.36e-4 | 0.940 |
| 5,000 | 5.20e-4 | 0.900 |
| ≈10,000 (end of warm-up, 0.2·50,001 − 1 = 9,999.2) | 1.00e-3 | 0.850 |
| 20,000 | 7.50e-4 | 0.875 |
| 30,000 | 5.00e-4 | 0.900 |
| 40,000 | 2.50e-4 | 0.925 |
| 50,000 | 4.0e-9 (= 4e-5/1e4) | 0.950 |

So in the actual code:
- Warm-up is linear from **4e-5, not from 0**. The paper says "linear warmup from zero for the first 20%" (p.6).
- Decay is linear to **4e-9**, which is effectively zero.
- Because `cycle_momentum=True` and AdamW has `betas`, **β1 is cycled 0.95→0.85→0.95**, inversely to the LR. This overrides the `betas=(0.9, …)` passed to AdamW. The paper does not mention it. β2 stays 0.999 and eps 1e-8.
- `lrs.step()` is called once per optimizer step (`main.py:421-422`).

**Gradient clipping** by global L2 norm, after the DDP all-reduce:

```python
# main.py:413-418
grad_norm = get_total_norm([p.grad for p in net.parameters() if p.grad is not None])
clip_grads_with_norm_(net.parameters(), max_norm=max_grad_norm, total_norm=grad_norm)
```

The paper does not mention clipping.

**Steps and budget.** Paper: "We pretrain RT for 50k steps at a context length of 1024, with a batch size of 256" (p.6). That is 50,000 × 256 = 12.8M sequences ≈ 13.1M target cells. The code uses `max_steps = 50_001` so that an evaluation still fires at step 50,000: the eval check runs at the top of the loop with the current step count (`main.py:369-376`). Throughput claim: "8 batches/second or 2M tokens/second" (p.6), consistent with 8 × 256 × 1024 ≈ 2.1M tokens/s.

### 2.7 Evaluation cadence and checkpoint selection during pretraining

`example_pretrain.py:8-14`: `eval_splits=["val","test"]`, `eval_freq=1000`, `eval_pow2=True`, `max_eval_steps=40`, `save_ckpt_dir="ckpts/leave_rel-amazon"`.

- Eval fires when `steps % eval_freq == 0` or `(steps & (steps-1)) == 0` (`main.py:373-375`). That is steps 0, 1, 2, 4, …, 1024, 2048, …, 32768 plus every multiple of 1000. **Step 0 is always evaluated**, before any update: random init for pretraining, the loaded checkpoint otherwise.
- For each `(db, table)` in the val metrics: if `val_metric > best_val` (strictly), store it, record the **test metric of the same step** as `best_test_metrics[(db, table)]`, and save `"{db}_{table}_best.pt"`. Otherwise save `"steps={steps}.pt"` (`main.py:377-392, 346-358`). This is "the best validation checkpoint … separately for each target task as we found significant overfitting during pretraining due to limited pretraining tasks" (p.6). The metric is always higher-is-better: AUROC or R² (`main.py:379`).
- At the end, rank 0 prints `best_test_metrics` (`main.py:441-448`). **These are the reported numbers.**
- The HF checkpoints `pretrain_<dataset>_<task>.pt` are these per-task best files (`PE/README.md:90-92`).

---

## 3. Continued pretraining and fine-tuning

### 3.1 Continued pretraining (the "Yes" columns of Tables 1/2, p.8)

`PE/scripts/example_contd_pretrain.py`:
- `load_ckpt_path="ckpts/leave_rel-amazon/rel-amazon_user-churn_best.pt"`. This is the leave-one-DB-out checkpoint selected *for this target task* (line 12).
- `train_tasks = [t for t in all_tasks if t[0]=="rel-amazon" and t[1] != "user-churn"]` (lines 17-19): every other amazon forecasting task (item-churn, user-ltv, item-ltv) plus amazon autocomplete (review.verified, review.rating), train split.
- `eval_tasks = [("rel-amazon","user-churn","churn",[])]` (line 20).
- `lr=1e-4`, `wd=0.0`, `lr_schedule=False` (constant LR, no warm-up), `max_grad_norm=1.0`, `max_steps=2**12+1 = 4097`, `eval_freq=100`, `eval_pow2=True` (lines 9-10, 25-29).
- Best-val checkpointing goes to `ckpts/contd_rel-amazon_user-churn/` (line 13).
- README: "takes about 15 minutes on 8xA100" (`PE/README.md:131-138`). HF names: `contd-pretrain_<dataset>_<task>.pt` (`README.md:92`).
- The target task is still never trained on. Its own train rows can still appear in val/test contexts as in-context labels (§5).
- The paper does not give contd-pretraining hyper-parameters. The above come only from the script.

### 3.2 Fine-tuning (Fig. 3, Figs. 7–8, Tables 3 and 10)

`PE/scripts/example_finetune.py`:
- `train_tasks = eval_tasks = [("rel-amazon","user-churn","churn",[])]` (lines 18-19).
- `lr=1e-4`, `wd=0.0`, `lr_schedule=False`, `max_grad_norm=1.0`, `max_steps=2**15+1 = 32,769` (lines 25-28). The paper says "33k steps" (p.6) and "up to 32k fine-tuning steps (8M training examples, including repetitions)" (Fig. 3, p.7). 32,768 × 256 = 8.39M.
- `eval_freq=1000`, `eval_pow2=False` (lines 9-10). With these settings there are no evaluations at 1, 2, 4, … steps, but Fig. 3 plots such points on a log axis. The plotted curves presumably used `eval_pow2=True`. That is an inference: the script's value differs from what the figure needs.
- `save_ckpt_dir=None` (line 14). The whole best-val block is skipped (`main.py:377`), so **no best checkpoint is saved and `best_test_metrics` stays empty** in this script. Best-val selection has to be done from the wandb logs: `auc/...` and `r2/.../val|test` are logged at every eval (`main.py:339-341`). The paper says "For all methods, the best checkpoint is selected based on validation set performance" (p.20 App. F.1).
- Initialization: `load_ckpt_path="ckpts/rel-amazon_user-churn/rel-amazon_user-churn_best.pt"` (line 12). That directory is produced by neither of the other two scripts (`ckpts/leave_rel-amazon/…`, `ckpts/contd_rel-amazon_user-churn/…`). The commented alternative is `~/scratch/rt_ckpts/pretrain_rel-amazon_user-churn.pt` (line 13), the leave-one-DB-out checkpoint. The README says "Finetune from best checkpoint obtained above" (i.e. after contd-pretrain, `README.md:140-145`). Fig. 3's caption says "Target datasets and tasks are unseen during pretraining", which points to the leave-one-DB-out init. The `rt-v1` README later says the leaderboard fine-tuned files come from the better of the two inits, and that "Cells without a file here are the paper's own pretrain-init fine-tuning numbers" (`git show rt-v1:README.md`). **Paper fine-tuning = pretrain (leave-one-DB-out) init**.
- "Untrained RT" = the same script with `load_ckpt_path=None`, i.e. PyTorch default init (`main.py:177-180`).
- Fine-tuning has exactly the same objective, sampler and loss as pretraining: one masked target cell, in-context labels visible (paper p.6: "This formulation is used in both pretraining and fine-tuning").
- Paper p.6: "fine-tune for 33k steps with the same context length, batch size and optimizer as above, but with a constant learning rate of 10−4 and no weight decay". About 1.5 h on 8×A100.

### 3.3 Summary table

| | pretrain | contd-pretrain | fine-tune |
|---|---|---|---|
| init | PyTorch default | leave-one-DB-out best ckpt (per target task) | leave-one-DB-out (paper) / contd (README) |
| train tasks | all tasks of the other DBs (forecast + autocomplete) | all tasks of the target DB except the target task | the target task only |
| lr | 1e-3 peak, OneCycle linear (4e-5 → 1e-3 at 20% → 4e-9), β1 cycled 0.95↔0.85 | 1e-4 constant | 1e-4 constant |
| wd | 0.1 | 0.0 | 0.0 |
| steps | 50,001 | 4,097 | 32,769 |
| eval | every 1000 + powers of 2 | every 100 + powers of 2 | every 1000 |
| ckpt | per-task best val | per-task best val | none saved (`save_ckpt_dir=None`) |
| clip | 1.0 | 1.0 | 1.0 |
| global batch | 256 (32 × 8 GPUs) | 256 | 256 |

---

## 4. Evaluation and zero-shot inference

### 4.1 Eval data

- One `RelationalDataset` per `(task, split)` with `split ∈ {"val","test"}` (`main.py:146-168`), so the seed rows are the task table's Val/Test rows (`data.py:64-82`).
- Built with **`seed=0`** and `sampler.shuffle_py(0)` (`main.py:158-160`). The item order and every sequence's BFS RNG (`epoch=0`, `seed=0`) are therefore fixed: every evaluation sees **exactly the same contexts**.
- The loader uses `persistent_workers=True` (`main.py:161-168`). After each evaluation the iterator is recreated (`main.py:302`), so the next evaluation restarts from batch 0.
- **Subset size.** `max_eval_steps=40` (all scripts) stops after 40 batches per rank (`main.py:299-300`). With 8 GPUs that is at most 40 × 32 × 8 = **10,240 seed rows per split per task**, a fixed random subset chosen by the seed-0 shuffle. Smaller splits are evaluated in full. The paper never states that RT val/test metrics are computed on a ≤10,240-row subset. It only says this for EntityMean ("we report the metrics on a subsampled test set", p.23 App. I.1). The `rt-v1` README later stresses that the leaderboard numbers are on the "full official test split", which implies the paper numbers were not.
- Padding of the last batch: the sampler wraps around. Rows ≥ `true_batch_size` are neutralised with `masks=False`, `is_targets=False`, `is_padding=True` (`main.py:273-279`), so a fully padded sequence has no target and only attends to nothing.
- Edge case (my reading): `true_batch_size` is a `usize` subtraction (`fly.rs:337-341`). On ranks whose slice of the final batch starts past the end of the item list, it wraps under `--release` (no overflow checks; `PE/rustler/Cargo.toml` sets no profile). That gives `true_batch_size = 32`, so those ranks would score wrapped duplicate rows. This only matters for splits smaller than the 40-batch budget.

### 4.2 Forward pass and readout

```python
# main.py:281-293
loss, yhat_dict = net(batch)                      # same forward as training, under inference_mode
if task_type == "clf":
    yhat = yhat_dict["boolean"][batch["is_targets"]]            # raw LOGIT, shape (true_batch_size, 1)
    y    = batch["boolean_values"][batch["is_targets"]].flatten() # z-scored boolean
elif task_type == "reg":
    yhat = yhat_dict["number"][batch["is_targets"]]             # prediction in z-score space
    y    = batch["number_values"][batch["is_targets"]].flatten()  # z-scored label
assert yhat.size(0) == true_batch_size
```

- The eval runs `net.eval()` and `torch.inference_mode()` (`main.py:221-222`). There is no dropout, so `eval()` does not change the computation.
- The same `forward` runs, and the loss is still computed. The eval loss is logged as `loss/{db}/{table}/{split}` (`main.py:316-326`).
- `task_type` comes from a **hard-coded list of table names**: `item-sales, user-ltv, item-ltv, post-votes, site-success, study-adverse, user-attendance, driver-position, ad-ctr` are "reg", everything else is "clf" (`main.py:228-241`). Only forecasting tasks are evaluated.
- The prediction is read **only at the `is_targets` position**, one per sequence. The decoder outputs at all other positions are ignored.
- With DDP, predictions and labels are gathered across ranks (`main.py:308-311`, `all_gather_nd` at `main.py:32-63`).

### 4.3 Metrics

```python
# main.py:328-337
preds  = cat(preds).float().cpu().numpy();  labels = cat(labels).float().cpu().numpy()
reg: metric = sklearn.metrics.r2_score(labels, preds)
clf: labels = [int(x > 0) for x in labels]; metric = sklearn.metrics.roc_auc_score(labels, preds)
```

- **AUROC is computed on raw logits. No sigmoid is applied.** AUROC is rank-based and sigmoid is monotone, so the score is identical. If the site shows probabilities, the correct conversion is σ(logit). The authors' later leaderboard script applies `torch.sigmoid` when it needs probabilities (`git show rt-v1:scripts/predict_stage1.py`, the "pred = probability in [0,1] (sigmoid applied)" line in its docstring and the `torch.sigmoid(yhat["boolean"]…)` line).
- **R² is computed in normalized (z-score) space, with no de-normalization.** Labels and predictions share the same affine map. Regression targets in task tables are z-scored with the **Train-split** mean/std (sample std, ddof=1, `pre.rs:334-340`), and Val/Test task tables reuse the Train stats (`pre.rs:384-396`). Under y′=(y−μ)/σ and ŷ′=(ŷ−μ)/σ, both SS_res and SS_tot scale by 1/σ², so **R² is invariant** and the reported R² equals raw-scale R². To show real-unit predictions, de-normalize with ŷ·σ_train + μ_train. The authors do exactly this in `rt-v1:scripts/predict_stage2.py` ("Regression predictions are in normalized (z-score) space — denormalize to original scale using the train-split target stats").
- R²'s ȳ is the mean of the evaluated labels (sklearn), so "Global mean baseline is 0.0" (Table 2 caption, p.8).
- Labels pass through bf16 (`fly.rs:482, 490`), so they carry about 3 significant digits.
- **No ensembling** in the paper-era code: one context per seed row, one forward pass ("a single forward pass", abstract p.1). Context-seed ensembling (`run_ensemble`, `val_ensemble_size`/`test_ensemble_size`) exists only in the later RT-J code (`CUR/src/rt/eval/_eval.py:90-111, 211-235`; commit `0485763`).
- The later RT-J code reports MAE for regression (`CUR/src/rt/eval/metrics.py:4-7`), not R². The paper-era code and paper use R².

### 4.4 Checkpoint selection

Covered in §2.7. One point matters for zero-shot claims: the **"zero-shot" numbers in Tables 1/2 are the test metrics at the pretraining step with the best validation metric on the target task**. Target-task validation labels are used for early stopping and model selection, but there are no gradient updates on the target DB ("No") or the target task ("Yes"). Paper p.6: "To pick the best checkpoints, we use the validation task splits and validation cutoff timestamps".

---

## 5. How zero-shot inference differs mechanically from training

| aspect | training step | zero-shot / eval |
|---|---|---|
| seed rows | train-split task rows (forecast) or all DB rows with the target present (autocomplete) | val/test task rows of the unseen task |
| sampler | same Rust BFS (`fly.rs:357-511`), `epoch` = pass index, `seed=0` | same code, `epoch=0`, `seed=0`, fixed order |
| masking | target cell of the seed row only | identical (target cell only) |
| in-context labels | other rows of the same task table with timestamp ≤ seed's, unmasked | identical. For val/test seeds these include **train-split rows** (and val rows for test seeds) of the same task |
| forward | `net.train()` + autograd | `net.eval()` + `inference_mode`, identical math |
| loss | backpropagated | computed and logged only |
| output used | none (only the loss) | logit / z-score at `is_targets` |
| weights | updated (AdamW + clip) | frozen |

Details:
- **Temporal filter.** A P→F edge (to a DB row or a task row) is skipped if `edge.timestamp > seed.timestamp` (`fly.rs:421-424`). **Equal timestamps are allowed.** For a val/test seed this means other entities' task rows *of the same split at the same timestamp* can enter the context unmasked, if the BFS reaches them (≥4 hops for entity tasks).
  - The submission-time code `7f891d1` had used `>=` for task edges, and `66c4247` relaxed it to `>`.
  - The current RT-J sampler explicitly excludes these "same-horizon task rows" (`CUR/rustler/src/fly.rs:1724, 1886-1900`).
  - The paper's ablation shows removing *all* other-entity labels barely changes zero-shot (Table 4 p.9: 70.6 vs 70.1 AUROC; 22.9 vs 22.8 R²).
- **BFS width bound.** w=256 subsampling is applied **only to P→F edges into DB tables**. P→F edges into the active task table (the in-context label rows) are all queued without subsampling (`fly.rs:426-452`). Alg. 1 (p.15) subsamples all P→F neighbours.
- If the seed or the edge has no timestamp, no temporal filter is applied (`fly.rs:422`).
- **DB cutoff not enforced for autocomplete seeds** (my reading). Autocomplete items are all rows of the DB table (`data.py:77-81`, `fly.rs:287-296`), with no train-cutoff filter. The paper says "using database rows only up to the training cutoff timestamps" (p.6). This matters for contd-pretraining on the target DB. The later RT-J sampler adds per-task `cutoff_timestamps` (`CUR/rustler/src/fly.rs:336, 1005, 1878-1884`).
- **Normalization statistics** (preprocessing, used identically in training and eval):
  - DB-table numeric and boolean columns use mean/std over the **whole table as stored on disk** (`pre.rs:316-341`). Only task tables use Train-split stats (`pre.rs:384-396`).
  - Datetimes use one **global** mean/std over every datetime cell of every table, task tables of all splits included. The values are raw nanoseconds (`pre.rs:342-366, 375-376, 584-586`); z-scoring makes the unit irrelevant.
  - The paper says column stats are "computed on the training split" (p.4-5 §3.1).
  - The authors later documented this as "pre-fix, leaky normalization", fixed by `rustler` PR #4 (`8030aa8`) in the RT-J line. The measured effect on RT-J was ≤0.51 points with mixed sign (`git show 358bfd3`, `docs/cards/relbench-preprocessed.md`). They note "Nobody has measured the effect on RT-v1".

---

## 6. Other forward/training details not visible in `model.py`

1. **Train step order** (`main.py:394-424`): `net.train()` → fetch batch → move it to the GPU (`non_blocking=True`) → `loss, _ = net(batch)` → `opt.zero_grad(set_to_none=True)` → `loss.backward()` → total grad norm → clip to 1.0 → `opt.step()` → `lrs.step()` (if scheduled) → `steps += 1` → log loss, lr, epochs, grad_norm (`main.py:428-437`).
2. There is no gradient checkpointing, no accumulation, no autocast, no EMA, no loss weighting by task or type, no label smoothing and no class re-weighting.
3. `is_task_nodes` and `class_value_idxs` are produced by the sampler and reshaped (`data.py:137, 141`), but the model and the training loop never use them.
4. `masks` and `is_targets` are identical in this sampler. The model uses `masks` for input masking and the loss (`model.py:219, 223, 236`). `main.py` uses `is_targets` for readout. RT-J's re-implementation of the paper model uses `is_targets` for both (`CUR/src/rt/model/legacy/v1.py:140, 183-187, 199`).
5. The text decoder and the `text`/`datetime` mask embeddings exist but are never used for loss in practice, because no text or datetime targets exist. Datetime masking would work (Huber), text masking raises.
6. Epoch counter: `epochs = steps / len(loader)` is logged. The sampler is reshuffled with the integer epoch whenever the inner loop exhausts the loader (`main.py:369-372, 433`).
7. The eval-time loss `loss/...` uses the same normalization `/ masks.sum()`. Fully padded wrap rows have `masks=False` and do not count.

---

## 7. Table 3, "RT as feature extractor" (paper p.8 §4.4)

- Paper: "We use the masked token representation before the decoder layer from a pretrained RT and fine-tune (FT) only a 2-layer MLP over it for downstream tasks on unseen datasets." The comparison is with untrained GNN, untrained RT and full FT.
- **The paper-era code has no feature-extraction or MLP-probe path.** I grepped all of `PE/` and searched git history for probe, return_emb and feature-extract terms. Nothing.
- Code-level meaning of "before the decoder layer": every decoder is applied directly to `x = self.norm_out(x)` (`model.py:229, 239`). So the representation is the **256-d output of `norm_out` (after the last block and the final RMSNorm) at the `is_targets` position**, the masked target cell, whose input token was `colname term + mask_emb_t`.
- The later RT-J model exposes exactly this tensor as `return_embeddings=True` (`CUR/src/rt/model/net.py:450-454`: `x = self.norm_out(x); if return_embeddings: return x`). That supports the reading, but it is not paper-era evidence.
- MLP width, activation and training hyper-parameters for Table 3 are **not specified** anywhere (paper or code).

---

## 8. Pseudo-code mirroring the code

```python
# ---------- preprocessing facts the training relies on (pre.rs) ----------
# numeric/bool cell r = (v - mean_col) / std_col   (std ddof=1; task tables: Train stats, also used for Val/Test)
# bool: v in {1.0, 0.0} before z-scoring;  datetime: r = (ns - mean_all_dt) / std_all_dt
# missing/NaN cells: not tokens at all; PK columns: not tokens; FK columns: become f2p links, not tokens
# task-table rows: cells = timestamp (datetime) + target (bool/number); table name = task name ("user-churn")

def main(train_tasks, eval_tasks, lr, wd, lr_schedule, max_steps, eval_freq, eval_pow2, max_eval_steps, ...):
    seed_everything(seed)                                                     # main.py:21-29
    train_ds = RelationalDataset([(db, tbl, col, "train", drop) for ...], B=32, S=1024, w=256, seed=seed)
    eval_ds  = {(db, tbl, split): RelationalDataset([(db, tbl, col, split, drop)], ..., seed=0)
                for (db, tbl, col, drop) in eval_tasks for split in ["val", "test"]}
    for ds in eval_ds.values(): ds.sampler.shuffle_py(0)                      # fixed order + fixed contexts

    net = RelationalTransformer(12, 256, 384, 8, 1024)
    if load_ckpt_path: net.load_state_dict(torch.load(load_ckpt_path))
    net = net.cuda().to(torch.bfloat16)                                       # pure bf16
    opt = AdamW(net.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.999), eps=1e-8, fused=True)
    if lr_schedule:   # pretraining only
        sched = OneCycleLR(opt, max_lr=lr, total_steps=max_steps, pct_start=0.2, anneal_strategy="linear")
        # => lr 4e-5 -> 1e-3 (linear, 20%) -> 4e-9 (linear); beta1 0.95 -> 0.85 -> 0.95
    net = torch.compile(DDP(net), dynamic=False)

    steps, best_val, best_test = 0, {}, {}
    while steps < max_steps:
        train_ds.sampler.shuffle_py(steps // len(loader))                     # new item order + new BFS draws
        for batch in loader:                                                  # each item = full (32, 1024) batch
            if steps % eval_freq == 0 or (eval_pow2 and steps & (steps - 1) == 0):
                val, test = evaluate(net)
                if save_ckpt_dir:
                    for task, m in val.items():
                        if m > best_val.get(task, -inf):
                            best_val[task], best_test[task] = m, test[task]
                            save(f"{db}_{tbl}_best.pt")
                        else:
                            save(f"steps={steps}.pt")
            net.train()
            loss, _ = net(batch)                                              # 1 masked cell / sequence
            opt.zero_grad(set_to_none=True); loss.backward()
            g = total_norm(grads); clip_grads_with_norm_(params, 1.0, g)
            opt.step()
            if lr_schedule: sched.step()
            steps += 1
            if steps >= max_steps: break
    print(best_test)                                                          # reported numbers

@torch.inference_mode()
def evaluate(net):
    net.eval()
    for (db, tbl, split), loader in eval_loaders.items():
        task_type = "reg" if tbl in REG_TABLE_NAMES else "clf"
        preds, labels = [], []
        for b_idx, batch in enumerate(loader):
            n = batch.pop("true_batch_size")
            batch["masks"][n:] = False; batch["is_targets"][n:] = False; batch["is_padding"][n:] = True
            loss, yhat = net(batch)
            t = batch["is_targets"]                                           # exactly one True per real row
            if task_type == "clf": p, y = yhat["boolean"][t], batch["boolean_values"][t]
            else:                  p, y = yhat["number"][t],  batch["number_values"][t]
            preds.append(p.flatten()); labels.append(y.flatten())
            if b_idx + 1 >= max_eval_steps: break                             # 40 batches per rank
        p, y = all_gather(cat(preds)), all_gather(cat(labels))
        metric = r2_score(y, p) if task_type == "reg" else roc_auc_score((y > 0).astype(int), p)  # p = logits
    return metrics  # {"val": {...}, "test": {...}}

# ---------- loss inside RelationalTransformer.forward (model.py:231-265) ----------
def rt_loss(x_out, batch):                       # x_out = norm_out(last block output), (B, S, 256)
    total = 0
    for i, t in enumerate(["number", "text", "datetime", "boolean"]):
        yhat = dec[t](x_out)
        m = (batch["sem_types"] == i) & batch["masks"]
        if not m.any(): total += 0.0 * yhat.sum(); continue
        if t in ("number", "datetime"): l = huber(yhat, batch[f"{t}_values"], delta=1.0)
        elif t == "boolean":            l = bce_with_logits(yhat, (batch["boolean_values"] > 0).float())
        else: raise ValueError("masking text not supported")
        total += (l.mean(-1) * m).sum()
    return total / batch["masks"].sum()
```

---

## 9. Discrepancies and surprises (paper vs code)

1. **LR warm-up starts at 4e-5, not 0, and decays to 4e-9.** These are OneCycleLR defaults `div_factor=25` and `final_div_factor=1e4`. The paper says "linear warmup from zero … linear decay to zero" (p.6).
2. **AdamW β1 is cycled 0.95→0.85→0.95** by OneCycleLR's default `cycle_momentum=True` during pretraining. It overrides the coded β1=0.9. The paper does not mention it.
3. **Gradient clipping at global norm 1.0** is not in the paper.
4. **Pure bf16 parameters and AdamW states** (`.to(torch.bfloat16)`, no autocast and no fp32 master weights). The paper only says "BFloat16 precision". RMSNorm eps is therefore 2⁻⁷.
5. **Only the target cell is masked** in the released code (`masks = is_target`). Multi-cell masking with P(mask) is only a Table 14 ablation. The main results are P(mask)=0. A cosine-decayed `mask_prob=0.5` variant existed only before the submission.
6. **Batch size 256 = 32/GPU × 8 GPUs**, with no gradient accumulation. A different GPU count changes the effective batch.
7. **Eval on a fixed subset of ≤10,240 rows per split** (`max_eval_steps=40` × 32 × 8 GPUs), with deterministic contexts. The paper does not state this for RT.
8. **AUROC on logits and R² in z-space.** Both are mathematically identical to probability-space and raw-unit metrics. De-normalization is only needed for display.
9. **"Zero-shot" uses target-task validation labels for checkpoint selection** across pretraining steps, which the paper acknowledges. The target-task train rows are visible in context as in-context labels.
10. **Same-timestamp task rows are allowed in context** (`>` filter, `fly.rs:422`). Later RT-J excludes "same-horizon task rows".
11. **BFS width bound applies only to DB children.** Task-table children (label rows) are never subsampled (`fly.rs:426-452`). Alg. 1 subsamples all P→F.
12. **Normalization stats for DB tables span the whole table**, not the training split as the paper says (p.4-5). The authors later called this leaky and fixed it in RT-J. Task-table stats do come from Train.
13. **Autocomplete seeds are not restricted to the training cutoff** (all DB rows with the target present). The paper says DB rows are used only up to the training cutoff.
14. `example_finetune.py` saves no checkpoint and never fills `best_test_metrics` (`save_ckpt_dir=None`). Its `eval_pow2=False` cannot produce the log-x learning curves of Fig. 3. Its `load_ckpt_path` points to a directory no script writes.
15. `rel-event/user-attendance` is cast to Boolean in preprocessing, so pretraining treats it as binary, although `tasks.py`/`main.py` list it as regression. rel-event is used in pretraining but never evaluated.
16. Table 1 RT "No" mean AUROC is 69.7. The ablation tables' "none" baseline is 70.1 (Tables 4/11/13/14), and Table 15's CFN (the released order) is 70.0. The ablations evidently used other runs or checkpoints. The website should quote Tables 1/2 for headline numbers.
17. Block order: the ICLR-submission commit used feat→nbr→col→full. The released and camera-ready order is col→feat→nbr→full (commit `7f891d1`). Table 15 shows the order barely matters (p.24).

## 10. What the website must show to be faithful

- Per training example: **one** seed row, **one** masked cell (the target), shown as `colname-embedding term + mask_emb[type]`. Every other cell is visible, including past labels in task-table rows (marked as "in-context labels").
- Output: `norm_out` → type decoder at the target position only.
  - Boolean: a logit, with loss BCE(logit, 1[z>0]) and displayed probability σ(logit).
  - Number: a z-score, with loss Huber(δ=1) against the z-scored label and display value ŷ·σ_train+μ_train.
- Batch: 256 independent sequences, from mixed tasks and DBs in pretraining. Loss = plain mean over the 256 target cells, with no weighting.
- Optimizer panel: AdamW (wd 0.1 pretrain / 0 fine-tune), clip 1.0, OneCycle LR curve (4e-5 → 1e-3 at step ≈10k → ~0 at 50k) with the β1 curve. Fine-tune and contd-pretrain use a flat 1e-4.
- Zero-shot panel: no gradient, the same sampler and the same masking. Metric = AUROC over logits or R² over z-scores. "Best val step" selection picks the test number.
