/**
 * Verbatim excerpts of the paper-era official code (snap-stanford/relational-transformer @2a899c8),
 * extracted programmatically with their exact line numbers. Do not edit by hand.
 */

/** rt/model.py:82-86. RelationalBlock.forward: one residual update per attention layer (spec §7.1). */
export const MODEL_BLOCK_FWD_82_86 = "    def forward(self, x, block_masks):\n        for l in [\"col\", \"feat\", \"nbr\", \"full\"]:\n            x = x + self.attns[l](self.norms[l](x), block_mask=block_masks[l])\n        x = x + self.ffn(self.norms[\"ffn\"](x))\n        return x";

/** rt/model.py:208-214. The column-name term of the input embedding reads col_name_values (spec §5.2). */
export const MODEL_COLNAME_TERM_208_214 = "        x = 0\n        x = x + (\n            self.norm_dict[\"col_name\"](\n                self.enc_dict[\"col_name\"](batch[\"col_name_values\"])\n            )\n            * (~is_padding)[..., None]\n        )";

/** rustler/src/fly.rs:469-477. The sampler stores the column id (for the masks) and the column-phrase embedding (for the input) separately (spec §5.1). */
export const FLY_COLNAME_469_477 = "                slices.table_name_idxs[seq_i] = node.table_name_idx.into();\n                slices.col_name_idxs[seq_i] = node.col_name_idxs[cell_i].into();\n                slices.class_value_idxs[seq_i] = node.class_value_idx[cell_i].into();\n                slices.col_name_values[seq_i * self.d_text..(seq_i + 1) * self.d_text]\n                    .copy_from_slice(get_text_emb(\n                        dataset,\n                        slices.col_name_idxs[seq_i],\n                        self.d_text,\n                    ));";

/** rustler/src/fly.rs:492-496. Exactly one masked cell: the seed row's target column (spec §4.10). */
export const FLY_TARGET_492_496 = "                let is_target = seed_node_idx == node.node_idx\n                    && node.col_name_idxs[cell_i] == target_column;\n                slices.is_targets[seq_i] = is_target;\n\n                slices.masks[seq_i] = is_target;";

/** rt/main.py:328-337. Evaluation metrics: R² on z-scores, AUROC on logits with labels int(z > 0) (spec §8.5). */
export const MAIN_METRICS_328_337 = "                    preds = torch.cat(preds, dim=0).float().cpu().numpy()\n                    labels = torch.cat(labels, dim=0).float().cpu().numpy()\n\n                    if task_type == \"reg\":\n                        metric_name = \"r2\"\n                        metric = r2_score(labels, preds)\n                    elif task_type == \"clf\":\n                        metric_name = \"auc\"\n                        labels = [int(x > 0) for x in labels]\n                        metric = roc_auc_score(labels, preds)";
