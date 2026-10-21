/**
 * Verbatim excerpts of the paper-era official code (snap-stanford/relational-transformer @2a899c8),
 * extracted programmatically with their exact line numbers. Do not edit by hand.
 */

/** rustler/src/fly.rs:287-296. One item per row of every task that has a target cell (spec §4.14). */
export const FLY_ITEMS_287_296 =
  '            for j in node_idx_offset..node_idx_offset + num_nodes {\n                let node = get_node(&datasets[i], j);\n                // skip the node if target column was removed during preprocessing\n                if node.col_name_idxs.iter().any(|&c| c == target) {\n                    items.push(Item {\n                        dataset_idx: i as i32,\n                        node_idx: j,\n                    });\n                }\n            }';

/** rustler/src/fly.rs:324-328. shuffle_py: one shuffle of the whole item list per epoch, seeded with epoch + seed (spec §4.14). */
export const FLY_SHUFFLE_324_328 =
  '    fn shuffle_py(&mut self, epoch: u64) {\n        self.epoch = epoch;\n        let mut rng = StdRng::seed_from_u64(epoch.wrapping_add(self.seed));\n        self.items.shuffle(&mut rng);\n    }';

/** rustler/src/fly.rs:336-355. A batch takes consecutive items of the shuffled list; the last batch wraps around (spec §4.14). */
export const FLY_BATCH_336_355 =
  '    fn batch(&self, batch_idx: usize) -> Vecs {\n        let true_batch_size = self.batch_size.min(\n            self.items.len()\n                - self.rank * self.batch_size\n                - batch_idx * self.batch_size * self.world_size,\n        );\n\n        let mut vecs = Vecs::new(self.batch_size, self.seq_len, true_batch_size, self.d_text);\n        vecs.chunks_exact_mut(self.seq_len, self.d_text)\n            .enumerate()\n            .for_each(|(i, slices)| {\n                let j =\n                    batch_idx * self.batch_size * self.world_size + self.rank * self.batch_size + i;\n                // when self.batch_size > true_batch_size, this will wrap around\n                let j = j % self.items.len();\n                let item = &self.items[j];\n                self.seq(item, slices);\n            });\n        vecs\n    }';

/** rustler/src/fly.rs:492-496. Exactly one masked cell: the seed row's target cell (spec §4.10). */
export const FLY_MASK_492_496 =
  '                let is_target = seed_node_idx == node.node_idx\n                    && node.col_name_idxs[cell_i] == target_column;\n                slices.is_targets[seq_i] = is_target;\n\n                slices.masks[seq_i] = is_target;';

/** rt/main.py:123-136. One RelationalDataset over every training task, all on the "train" split (spec §9.1). */
export const MAIN_DATASET_123_136 =
  '    dataset = RelationalDataset(\n        tasks=[\n            (db_name, table_name, target_column, "train", columns_to_drop)\n            for db_name, table_name, target_column, columns_to_drop in train_tasks\n        ],\n        batch_size=batch_size,\n        seq_len=seq_len,\n        rank=rank,\n        world_size=world_size,\n        max_bfs_width=max_bfs_width,\n        embedding_model=embedding_model,\n        d_text=d_text,\n        seed=seed,\n    )';

/** rt/main.py:186-204. bf16 cast, AdamW with one parameter group, OneCycleLR (spec §9.2). */
export const MAIN_OPT_186_204 =
  '    net = net.to(device)\n    net = net.to(torch.bfloat16)\n    opt = optim.AdamW(\n        net.parameters(),\n        lr=lr,\n        weight_decay=wd,\n        betas=(0.9, 0.999),\n        eps=1e-8,\n        fused=True,\n    )\n\n    if lr_schedule:\n        lrs = optim.lr_scheduler.OneCycleLR(\n            opt,\n            max_lr=lr,\n            total_steps=max_steps,\n            pct_start=0.2,\n            anneal_strategy="linear",\n        )';

/** rt/main.py:369-371. Re-shuffle at every pass over the loader (spec §4.14). */
export const MAIN_SHUFFLE_369_371 =
  '    while steps < max_steps:\n        loader.dataset.sampler.shuffle_py(int(steps / len(loader)))\n        loader_iter = iter(loader)';

/** rt/main.py:373-392. Evaluate at step 0, every eval_freq steps and at powers of two; keep the best val checkpoint per task (spec §9.1). */
export const MAIN_EVAL_SELECT_373_392 =
  '            if (eval_freq is not None and steps % eval_freq == 0) or (\n                eval_pow2 and steps & (steps - 1) == 0\n            ):\n                metrics = evaluate(net)\n                if save_ckpt_dir is not None:\n                    for (db_name, table_name), metric in metrics["val"].items():\n                        # Eval metric is always higher is better (auc, r2)\n                        best_metric = best_val_metrics.get(\n                            (db_name, table_name), -float("inf")\n                        )\n                        if metric > best_metric:\n                            best_val_metrics[(db_name, table_name)] = metric\n                            best_test_metrics[(db_name, table_name)] = metrics["test"][(db_name, table_name)]\n                            checkpoint(\n                                best=True,\n                                db_name=db_name,\n                                table_name=table_name\n                            )\n                        else:\n                            checkpoint()';

/** rt/main.py:394-424. One optimizer step: forward, backward, clip the global grad norm to 1.0, AdamW, scheduler (spec §9.2). */
export const MAIN_STEP_394_424 =
  '            net.train()\n\n            tic = time.time()\n            try:\n                batch = next(loader_iter)\n            except StopIteration:\n                break\n            batch.pop("true_batch_size")\n            for k in batch:\n                batch[k] = batch[k].to(device, non_blocking=True)\n            toc = time.time()\n            load_time = toc - tic\n            if rank == 0:\n                wandb.log({"load_time": load_time}, step=steps)\n\n            loss, _yhat_dict = net(batch)\n            opt.zero_grad(set_to_none=True)\n            loss.backward()\n\n            grad_norm = get_total_norm(\n                [p.grad for p in net.parameters() if p.grad is not None]\n            )\n            clip_grads_with_norm_(\n                net.parameters(), max_norm=max_grad_norm, total_norm=grad_norm\n            )\n\n            opt.step()\n            if lr_schedule:\n                lrs.step()\n\n            steps += 1';

/** rt/main.py:441-448. The reported numbers: the test metric recorded at each task's best val step (spec §9.1). */
export const MAIN_BEST_441_448 =
  '    # Print best test metrics\n    if rank == 0:\n        print("\\n" + "="*80)\n        print("Best test metrics:")\n        print("="*80)\n        for (db_name, table_name), metric in best_test_metrics.items():\n            print(f"{db_name}/{table_name}/test: {metric:.4f}")\n        print("="*80 + "\\n")';

/** rt/model.py:250-267. Huber for numbers, BCE against 1{y>0} for booleans, mean over the masked cells of the batch (spec §8.2). */
export const MODEL_LOSS_250_267 =
  '            if t in ("number", "datetime"):\n                loss_t = F.huber_loss(yhat, y, reduction="none").mean(-1)\n            elif t == "boolean":\n                loss_t = F.binary_cross_entropy_with_logits(\n                    yhat, (y > 0).float(), reduction="none"\n                ).mean(-1)\n            elif t == "text":\n                raise ValueError("masking text not supported")\n\n            # masked sum for this type\n            loss_out = loss_out + (loss_t * sem_type_mask).sum()\n\n            if t in yhat_out:\n                yhat_out[t] = yhat\n\n        loss_out = loss_out / masks.sum()\n\n        return loss_out, yhat_out';

/** scripts/example_pretrain.py:1-36. The leave-one-DB-out pretraining script for target rel-amazon (spec §9.2). */
export const PRETRAIN_SCRIPT_1_36 =
  'from rt.main import main\nfrom rt.tasks import all_tasks, forecast_tasks\n\nif __name__ == "__main__":\n    main(\n        # misc\n        project="rt",\n        eval_splits=["val", "test"],\n        eval_freq=1_000,\n        eval_pow2=True,\n        max_eval_steps=40,\n        load_ckpt_path=None,\n        save_ckpt_dir="ckpts/leave_rel-amazon",\n        compile_=True,\n        seed=0,\n        # data\n        train_tasks=[t for t in all_tasks if t[0] != "rel-amazon"],\n        eval_tasks=[t for t in forecast_tasks if t[0] == "rel-amazon"],\n        batch_size=32,\n        num_workers=8,\n        max_bfs_width=256,\n        # optimization\n        lr=1e-3,\n        wd=0.1,\n        lr_schedule=True,\n        max_grad_norm=1.0,\n        max_steps=50_001,\n        # model\n        embedding_model="all-MiniLM-L12-v2",\n        d_text=384,\n        seq_len=1024,\n        num_blocks=12,\n        d_model=256,\n        num_heads=8,\n        d_ff=1024,\n    )';

/** scripts/example_finetune.py:1-37. The fine-tuning script (spec §9.2, §10 #21). */
export const FINETUNE_SCRIPT_1_37 =
  'from rt.main import main\nfrom rt.tasks import all_tasks, forecast_tasks\n\nif __name__ == "__main__":\n    main(\n        # misc\n        project="rt",\n        eval_splits=["val", "test"],\n        eval_freq=1_000,\n        eval_pow2=False,\n        max_eval_steps=40,\n        load_ckpt_path="ckpts/rel-amazon_user-churn/rel-amazon_user-churn_best.pt",\n        # load_ckpt_path="~/scratch/rt_ckpts/pretrain_rel-amazon_user-churn.pt",\n        save_ckpt_dir=None,\n        compile_=True,\n        seed=0,\n        # data\n        train_tasks=[("rel-amazon", "user-churn", "churn", [])],\n        eval_tasks=[("rel-amazon", "user-churn", "churn", [])],\n        batch_size=32,\n        num_workers=8,\n        max_bfs_width=256,\n        # optimization\n        lr=1e-4,\n        wd=0.0,\n        lr_schedule=False,\n        max_grad_norm=1.0,\n        max_steps=2**15 + 1,\n        # model\n        embedding_model="all-MiniLM-L12-v2",\n        d_text=384,\n        seq_len=1024,\n        num_blocks=12,\n        d_model=256,\n        num_heads=8,\n        d_ff=1024,\n    )';
