/**
 * Verbatim excerpts of the paper-era official code (snap-stanford/relational-transformer @2a899c8),
 * extracted programmatically from rt/model.py with their exact line numbers. Do not edit by hand.
 */

/** rt/model.py:30-47. MaskedAttention.forward: the mask goes straight into flex_attention (spec §6.1). */
export const MODEL_ATTN_FWD_30_47 = "    def forward(self, x, block_mask):\n        q = self.wq(x)\n        k = self.wk(x)\n        v = self.wv(x)\n\n        q = rearrange(q, \"b s (h d) -> b h s d\", h=self.num_heads)\n        k = rearrange(k, \"b s (h d) -> b h s d\", h=self.num_heads)\n        v = rearrange(v, \"b s (h d) -> b h s d\", h=self.num_heads)\n\n        if block_mask is None:\n            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):\n                x = F.scaled_dot_product_attention(q, k, v)\n        else:\n            x = flex_attention(q, k, v, block_mask=block_mask)\n\n        x = rearrange(x, \"b h s d -> b s (h d)\")\n        x = self.wo(x)\n        return x";

/** rt/model.py:82-86. RelationalBlock.forward: each layer gets its own block mask (spec §7.1). */
export const MODEL_BLOCK_FWD_82_86 = "    def forward(self, x, block_masks):\n        for l in [\"col\", \"feat\", \"nbr\", \"full\"]:\n            x = x + self.attns[l](self.norms[l](x), block_mask=block_masks[l])\n        x = x + self.ffn(self.norms[\"ffn\"](x))\n        return x";

/** rt/model.py:89-101. _make_block_mask: H=None, so all heads share the mask (spec §6.1). */
export const MODEL_MAKE_BLOCK_MASK_89_101 = "def _make_block_mask(mask, batch_size, seq_len, device):\n    def _mod(b, h, q_idx, kv_idx):\n        return mask[b, q_idx, kv_idx]\n\n    return create_block_mask(\n        mask_mod=_mod,\n        B=batch_size,\n        H=None,\n        Q_LEN=seq_len,\n        KV_LEN=seq_len,\n        device=device,\n        _compile=True,\n    )";

/** rt/model.py:164-191. The four boolean masks from ids (spec §6.2). */
export const MODEL_MASKS_164_191 = "        # Padding mask for attention pairs (allow only non-pad -> non-pad)\n        pad = (~is_padding[:, :, None]) & (~is_padding[:, None, :])  # (B, S, S)\n\n        # cells in the same node\n        same_node = node_idxs[:, :, None] == node_idxs[:, None, :]  # (B, S, S)\n\n        # kv index is among q's foreign -> primary neighbors\n        kv_in_f2p = (node_idxs[:, None, :, None] == f2p_nbr_idxs[:, :, None, :]).any(\n            -1\n        )  # (B, S, S)\n\n        # q index is among kv's primary -> foreign neighbors (reverse relation)\n        q_in_f2p = (node_idxs[:, :, None, None] == f2p_nbr_idxs[:, None, :, :]).any(\n            -1\n        )  # (B, S, S)\n\n        # Same column AND same table\n        same_col_table = (col_name_idxs[:, :, None] == col_name_idxs[:, None, :]) & (\n            table_name_idxs[:, :, None] == table_name_idxs[:, None, :]\n        )  # (B, S, S)\n\n        # Final boolean masks (apply padding once here)\n        attn_masks = {\n            \"feat\": (same_node | kv_in_f2p) & pad,\n            \"nbr\": q_in_f2p & pad,\n            \"col\": same_col_table & pad,\n            \"full\": pad,\n        }";

/** rt/model.py:193-206. Masks converted to block masks once per forward pass (spec §6.1). */
export const MODEL_BLOCK_MASKS_193_206 = "        # Make them contiguous for better kernel performance\n        for l in attn_masks:\n            attn_masks[l] = attn_masks[l].contiguous()\n\n        # Convert to block masks\n        make_block_mask = partial(\n            _make_block_mask,\n            batch_size=batch_size,\n            seq_len=seq_len,\n            device=device,\n        )\n        block_masks = {\n            l: make_block_mask(attn_mask) for l, attn_mask in attn_masks.items()\n        }";

/** rt/model.py:226-227. The same block masks are passed to every block (spec §6.1). */
export const MODEL_BLOCKS_LOOP_226_227 = "        for i, block in enumerate(self.blocks):\n            x = block(x, block_masks)";

