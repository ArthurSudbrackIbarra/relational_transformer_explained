/**
 * Verbatim excerpts of the paper-era official code (snap-stanford/relational-transformer @2a899c8),
 * generated from the local snapshot by line range and re-checked against the files. Do not edit by hand.
 */

/** rt/model.py:14-17 */
export const MODEL_FLEX_IMPORT_14_17 = "from torch.nn.attention.flex_attention import create_block_mask, flex_attention\n\nallow_ops_in_compiled_graph()\nflex_attention = torch.compile(flex_attention)";

/** rt/model.py:30-47 */
export const MODEL_ATTN_FWD_30_47 = "    def forward(self, x, block_mask):\n        q = self.wq(x)\n        k = self.wk(x)\n        v = self.wv(x)\n\n        q = rearrange(q, \"b s (h d) -> b h s d\", h=self.num_heads)\n        k = rearrange(k, \"b s (h d) -> b h s d\", h=self.num_heads)\n        v = rearrange(v, \"b s (h d) -> b h s d\", h=self.num_heads)\n\n        if block_mask is None:\n            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):\n                x = F.scaled_dot_product_attention(q, k, v)\n        else:\n            x = flex_attention(q, k, v, block_mask=block_mask)\n\n        x = rearrange(x, \"b h s d -> b s (h d)\")\n        x = self.wo(x)\n        return x";

/** rt/model.py:71-73 */
export const MODEL_BLOCK_NORMS_71_73 = "        self.norms = nn.ModuleDict(\n            {l: nn.RMSNorm(d_model) for l in [\"feat\", \"nbr\", \"col\", \"full\", \"ffn\"]}\n        )";

/** rt/model.py:89-101 */
export const MODEL_MAKE_BLOCK_MASK_89_101 = "def _make_block_mask(mask, batch_size, seq_len, device):\n    def _mod(b, h, q_idx, kv_idx):\n        return mask[b, q_idx, kv_idx]\n\n    return create_block_mask(\n        mask_mod=_mod,\n        B=batch_size,\n        H=None,\n        Q_LEN=seq_len,\n        KV_LEN=seq_len,\n        device=device,\n        _compile=True,\n    )";

/** rt/main.py:186-187 */
export const MAIN_BF16_186_187 = "    net = net.to(device)\n    net = net.to(torch.bfloat16)";

/** rustler/src/fly.rs:371-375 */
export const FLY_RNG_371_375 = "        let mut rng = StdRng::seed_from_u64(\n            self.epoch\n                .wrapping_add(seed_node_idx as u64)\n                .wrapping_add(self.seed),\n        );";

/** rustler/src/fly.rs:390-395 */
export const FLY_PICK_390_395 = "                    let depth = depth_choices[0];\n                    let r = rng.random_range(0..p2f_ftr[depth].len());\n                    let l = p2f_ftr[depth].len();\n                    p2f_ftr[depth].swap(r, l - 1);\n                    let node_idx = p2f_ftr[depth].pop().unwrap();\n                    (depth, node_idx)";

/** rustler/src/fly.rs:439-443 */
export const FLY_WIDTH_439_443 = "            let idxs = if db_p2f_ftr.len() > self.max_bfs_width {\n                index::sample(&mut rng, db_p2f_ftr.len(), self.max_bfs_width).into_vec()\n            } else {\n                (0..db_p2f_ftr.len()).collect::<Vec<_>>()\n            };";

/** README.md:155-160 */
export const README_BIBTEX_155_160 = "@inproceedings{ranjan2025relationaltransformer,\n    title={{Relational Transformer:} Toward Zero-Shot Foundation Models for Relational Data}, \n    author={Rishabh Ranjan and Valter Hudovernik and Mark Znidar and Charilaos Kanatsoulis and Roshan Upendra and Mahmoud Mohammadi and Joe Meyer and Tom Palczewski and Carlos Guestrin and Jure Leskovec},\n    booktitle={The Fourteenth International Conference on Learning Representations},\n    year={2026}\n}";
