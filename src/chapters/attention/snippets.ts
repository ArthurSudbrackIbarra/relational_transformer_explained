/**
 * Verbatim excerpts of the paper-era official code (snap-stanford/relational-transformer @2a899c8),
 * extracted programmatically from rt/model.py with their exact line numbers. Do not edit by hand.
 */

/** rt/model.py:14-17. Imports FlexAttention and compiles it once (spec §6.1). */
export const MODEL_FLEX_14_17 = "from torch.nn.attention.flex_attention import create_block_mask, flex_attention\n\nallow_ops_in_compiled_graph()\nflex_attention = torch.compile(flex_attention)";

/** rt/model.py:20-47. MaskedAttention: four bias-free projections, the head split, flex_attention, W_o (spec §6.1). */
export const MODEL_MASKED_ATTENTION_20_47 = "class MaskedAttention(nn.Module):\n    def __init__(self, d_model, num_heads):\n        super().__init__()\n        self.num_heads = num_heads\n\n        self.wq = nn.Linear(d_model, d_model, bias=False)\n        self.wk = nn.Linear(d_model, d_model, bias=False)\n        self.wv = nn.Linear(d_model, d_model, bias=False)\n        self.wo = nn.Linear(d_model, d_model, bias=False)\n\n    def forward(self, x, block_mask):\n        q = self.wq(x)\n        k = self.wk(x)\n        v = self.wv(x)\n\n        q = rearrange(q, \"b s (h d) -> b h s d\", h=self.num_heads)\n        k = rearrange(k, \"b s (h d) -> b h s d\", h=self.num_heads)\n        v = rearrange(v, \"b s (h d) -> b h s d\", h=self.num_heads)\n\n        if block_mask is None:\n            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):\n                x = F.scaled_dot_product_attention(q, k, v)\n        else:\n            x = flex_attention(q, k, v, block_mask=block_mask)\n\n        x = rearrange(x, \"b h s d -> b s (h d)\")\n        x = self.wo(x)\n        return x";

/** rt/model.py:71-79. RelationalBlock.__init__: one RMSNorm and one MaskedAttention per layer (spec §7.1). */
export const MODEL_NORMS_ATTNS_71_79 = "        self.norms = nn.ModuleDict(\n            {l: nn.RMSNorm(d_model) for l in [\"feat\", \"nbr\", \"col\", \"full\", \"ffn\"]}\n        )\n        self.attns = nn.ModuleDict(\n            {\n                l: MaskedAttention(d_model, num_heads)\n                for l in [\"feat\", \"nbr\", \"col\", \"full\"]\n            }\n        )";

/** rt/model.py:82-86. RelationalBlock.forward: pre-norm residual, order col → feat → nbr → full (spec §7.1). */
export const MODEL_BLOCK_FWD_82_86 = "    def forward(self, x, block_masks):\n        for l in [\"col\", \"feat\", \"nbr\", \"full\"]:\n            x = x + self.attns[l](self.norms[l](x), block_mask=block_masks[l])\n        x = x + self.ffn(self.norms[\"ffn\"](x))\n        return x";

/** rt/model.py:89-101. _make_block_mask: H=None, so every head shares the mask (spec §6.1). */
export const MODEL_MAKE_BLOCK_MASK_89_101 = "def _make_block_mask(mask, batch_size, seq_len, device):\n    def _mod(b, h, q_idx, kv_idx):\n        return mask[b, q_idx, kv_idx]\n\n    return create_block_mask(\n        mask_mod=_mod,\n        B=batch_size,\n        H=None,\n        Q_LEN=seq_len,\n        KV_LEN=seq_len,\n        device=device,\n        _compile=True,\n    )";

