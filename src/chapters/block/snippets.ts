/**
 * Verbatim excerpts of the paper-era official code (snap-stanford/relational-transformer @2a899c8),
 * extracted programmatically with their exact line numbers (scratchpad gen_snippets.py). Do not edit by hand.
 */

/** rt/model.py:50-59. FFN: SwiGLU with three bias-free linear maps (spec §7.1). */
export const MODEL_FFN_50_59 = "class FFN(nn.Module):\n    def __init__(self, d_model, d_ff):\n        super().__init__()\n\n        self.w1 = nn.Linear(d_model, d_ff, bias=False)\n        self.w2 = nn.Linear(d_ff, d_model, bias=False)\n        self.w3 = nn.Linear(d_model, d_ff, bias=False)\n\n    def forward(self, x):\n        return self.w2(F.silu(self.w1(x)) * self.w3(x))";

/** rt/model.py:62-86. RelationalBlock: five RMSNorms, four MaskedAttentions, one FFN; pre-norm residual forward (spec §7.1). */
export const MODEL_BLOCK_62_86 = "class RelationalBlock(nn.Module):\n    def __init__(\n        self,\n        d_model,\n        num_heads,\n        d_ff,\n    ):\n        super().__init__()\n\n        self.norms = nn.ModuleDict(\n            {l: nn.RMSNorm(d_model) for l in [\"feat\", \"nbr\", \"col\", \"full\", \"ffn\"]}\n        )\n        self.attns = nn.ModuleDict(\n            {\n                l: MaskedAttention(d_model, num_heads)\n                for l in [\"feat\", \"nbr\", \"col\", \"full\"]\n            }\n        )\n        self.ffn = FFN(d_model, d_ff)\n\n    def forward(self, x, block_masks):\n        for l in [\"col\", \"feat\", \"nbr\", \"full\"]:\n            x = x + self.attns[l](self.norms[l](x), block_mask=block_masks[l])\n        x = x + self.ffn(self.norms[\"ffn\"](x))\n        return x";

/** rt/model.py:147-150. RelationalTransformer.__init__: the block stack and the final RMSNorm (spec §7.1). */
export const MODEL_STACK_147_150 = "        self.blocks = nn.ModuleList(\n            [RelationalBlock(d_model, num_heads, d_ff) for i in range(num_blocks)]\n        )\n        self.norm_out = nn.RMSNorm(d_model)";

/** rt/model.py:226-229. RelationalTransformer.forward: run every block with the same block masks, then norm_out (spec §7.1). */
export const MODEL_LOOP_226_229 = "        for i, block in enumerate(self.blocks):\n            x = block(x, block_masks)\n\n        x = self.norm_out(x)";

/** scripts/example_pretrain.py:28-35. Paper model hyperparameters in the pretraining script (spec §7.1). */
export const PRETRAIN_MODEL_28_35 = "        # model\n        embedding_model=\"all-MiniLM-L12-v2\",\n        d_text=384,\n        seq_len=1024,\n        num_blocks=12,\n        d_model=256,\n        num_heads=8,\n        d_ff=1024,";
