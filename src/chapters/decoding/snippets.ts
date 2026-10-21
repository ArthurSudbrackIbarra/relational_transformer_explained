/**
 * Verbatim excerpts of the paper-era official code (snap-stanford/relational-transformer @2a899c8),
 * extracted programmatically with their exact line numbers. Do not edit by hand.
 */

/** rt/model.py:124-131. dec_dict: one Linear per sem type, shared by every column (spec §8.1). */
export const MODEL_DEC_DICT_124_131 =
  '        self.dec_dict = nn.ModuleDict(\n            {\n                "number": nn.Linear(d_model, 1, bias=True),\n                "text": nn.Linear(d_model, d_text, bias=True),\n                "datetime": nn.Linear(d_model, 1, bias=True),\n                "boolean": nn.Linear(d_model, 1, bias=True),\n            }\n        )';

/** rt/model.py:226-229. The blocks, then the final RMSNorm norm_out (spec §7.1). */
export const MODEL_BLOCKS_NORM_OUT_226_229 =
  '        for i, block in enumerate(self.blocks):\n            x = block(x, block_masks)\n\n        x = self.norm_out(x)';

/** rt/model.py:231-248. Every decoder runs on every position; unused types get a zero-weight touch (spec §8.1). */
export const MODEL_DECODE_231_248 =
  '        loss_out = x.new_zeros(())\n        yhat_out = {"number": None, "text": None, "datetime": None, "boolean": None}\n\n        B, S, _ = x.shape\n        sem_types = batch["sem_types"]  # (B,S) ints 0..3\n        masks = batch["masks"].bool()  # (B,S) where to train\n\n        for i, t in enumerate(["number", "text", "datetime", "boolean"]):\n            yhat = self.dec_dict[t](x)  # (B,S, D_t)\n            y = batch[f"{t}_values"]  # (B,S, D_y)\n            sem_type_mask = (sem_types == i) & masks  # (B,S) mask for this type\n\n            if not sem_type_mask.any():\n                if t in yhat_out:\n                    # still touch the param to avoid unused param error\n                    loss_out = loss_out + (yhat.sum() * 0.0)\n                    yhat_out[t] = yhat\n                continue';

/** rt/model.py:250-267. Huber for number/datetime, BCE against 1{y>0} for boolean, mean over masked cells (spec §8.2). */
export const MODEL_LOSS_250_267 =
  '            if t in ("number", "datetime"):\n                loss_t = F.huber_loss(yhat, y, reduction="none").mean(-1)\n            elif t == "boolean":\n                loss_t = F.binary_cross_entropy_with_logits(\n                    yhat, (y > 0).float(), reduction="none"\n                ).mean(-1)\n            elif t == "text":\n                raise ValueError("masking text not supported")\n\n            # masked sum for this type\n            loss_out = loss_out + (loss_t * sem_type_mask).sum()\n\n            if t in yhat_out:\n                yhat_out[t] = yhat\n\n        loss_out = loss_out / masks.sum()\n\n        return loss_out, yhat_out';

/** rt/main.py:228-241. Which head eval reads: a hard-coded list of regression task names (spec §8.3). */
export const MAIN_TASK_TYPE_228_241 =
  '                if table_name in [\n                    "item-sales",\n                    "user-ltv",\n                    "item-ltv",\n                    "post-votes",\n                    "site-success",\n                    "study-adverse",\n                    "user-attendance",\n                    "driver-position",\n                    "ad-ctr",\n                ]:\n                    task_type = "reg"\n                else:\n                    task_type = "clf"';

/** rt/main.py:283-293. Read the prediction and the label at is_targets (spec §8.3). */
export const MAIN_READOUT_283_293 =
  '                    if task_type == "clf":\n                        yhat = yhat_dict["boolean"][batch["is_targets"]]\n                        y = batch["boolean_values"][batch["is_targets"]].flatten()\n                    elif task_type == "reg":\n                        yhat = yhat_dict["number"][batch["is_targets"]]\n                        y = batch["number_values"][batch["is_targets"]].flatten()\n\n                    assert yhat.size(0) == true_batch_size\n                    assert y.size(0) == true_batch_size\n\n                    pred = yhat.flatten()';

/** rt/main.py:328-337. R² in z-space, AUROC on raw logits (spec §8.5). */
export const MAIN_METRICS_328_337 =
  '                    preds = torch.cat(preds, dim=0).float().cpu().numpy()\n                    labels = torch.cat(labels, dim=0).float().cpu().numpy()\n\n                    if task_type == "reg":\n                        metric_name = "r2"\n                        metric = r2_score(labels, preds)\n                    elif task_type == "clf":\n                        metric_name = "auc"\n                        labels = [int(x > 0) for x in labels]\n                        metric = roc_auc_score(labels, preds)';

/** rt/main.py:188-195. One AdamW parameter group: weight decay on every parameter (spec §9.2). */
export const MAIN_ADAMW_188_195 =
  '    opt = optim.AdamW(\n        net.parameters(),\n        lr=lr,\n        weight_decay=wd,\n        betas=(0.9, 0.999),\n        eps=1e-8,\n        fused=True,\n    )';
