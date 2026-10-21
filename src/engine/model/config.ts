/**
 * Model configurations and the exact parameter inventory of the RT.
 *
 * Every name and shape below is the PyTorch state_dict of the paper-era
 * `RelationalTransformer` (PE/rt/model.py:104-151), so weights exported from
 * PyTorch can be loaded by name with no renaming (spec §7.3).
 */
import type { RTConfig } from '../types';

/** The miniature model this site trains and runs (BUILD_PLAN §2). Only the scale differs from the paper. */
export const MINI_CONFIG: RTConfig = {
  dModel: 64,
  nHeads: 4,
  dFF: 256,
  nBlocks: 4,
  dText: 384,
  eps: 2 ** -7,
};

/**
 * The paper model: d=256, 8 heads, d_ff=1024, 12 blocks (paper p.6 §4.1; `example_pretrain.py:32`).
 * d_text=384 = all-MiniLM-L12-v2. ε = 2⁻⁷ = bf16 epsilon under the pinned torch 2.6.0 (spec §7.2).
 */
export const PAPER_CONFIG: RTConfig = {
  dModel: 256,
  nHeads: 8,
  dFF: 1024,
  nBlocks: 12,
  dText: 384,
  eps: 2 ** -7,
};

/** Context length and BFS width of the mini setting (BUILD_PLAN §2). */
export const DEFAULT_L = 128;
export const DEFAULT_W = 16;
/** Paper settings: L=1024 (p.6), w=max_bfs_width=256 (`example_pretrain.py:21`, spec §13 #2). */
export const PAPER_L = 1024;
export const PAPER_W = 256;

/**
 * Sem-type keys in SemType-code order. model.py iterates `enumerate(["number", "text", "datetime", "boolean"])`
 * and compares `sem_types == i` (model.py:216-219, 238-241), so index = SemType code (common.rs:38-44).
 */
export const SEM_KEYS = ['number', 'text', 'datetime', 'boolean'] as const;
export type SemKey = (typeof SEM_KEYS)[number];

/** Keys of enc_dict / norm_dict in module-registration order (model.py:115-140). */
export const ENC_KEYS = ['number', 'text', 'datetime', 'col_name', 'boolean'] as const;
/** Keys of dec_dict in registration order (model.py:124-131). */
export const DEC_KEYS = ['number', 'text', 'datetime', 'boolean'] as const;
/** Block-internal ModuleDict registration order (model.py:71-79). NOT the execution order (ATTN_ORDER). */
export const BLOCK_NORM_KEYS = ['feat', 'nbr', 'col', 'full', 'ffn'] as const;
export const BLOCK_ATTN_KEYS = ['feat', 'nbr', 'col', 'full'] as const;
/**
 * state_dict order of `mask_embs`. model.py:141-146 builds `nn.ParameterDict({t: ... for t in
 * ["number", "text", "datetime", "boolean"]})` from a PLAIN dict, and `ParameterDict.update` iterates
 * `sorted(parameters.items())` for a non-OrderedDict mapping (torch/nn/modules/container.py,
 * ParameterDict.update), so the keys are registered alphabetically. (ModuleDict does NOT sort, so
 * enc_dict / dec_dict / norm_dict keep their list order.) Order only matters for serialization;
 * tensors are always looked up by name.
 */
export const MASK_EMB_STATE_DICT_KEYS = ['boolean', 'datetime', 'number', 'text'] as const;

export function dHead(cfg: RTConfig): number {
  if (cfg.dModel % cfg.nHeads !== 0) throw new Error(`dModel ${cfg.dModel} not divisible by nHeads ${cfg.nHeads}`);
  return cfg.dModel / cfg.nHeads;
}

/**
 * Ordered list of every parameter tensor with its shape, in the same order as
 * `RelationalTransformer.state_dict()` (registration order of model.py:115-150; mask_embs sorted,
 * see MASK_EMB_STATE_DICT_KEYS). Checked against the official class's state_dict in
 * forward.test.ts. nn.Linear weights are [out, in].
 */
export function parameterShapes(cfg: RTConfig): [string, number[]][] {
  const { dModel: d, dText, dFF, nBlocks } = cfg;
  const out: [string, number[]][] = [];
  // enc_dict (model.py:115-123)
  for (const k of ENC_KEYS) {
    const fanIn = k === 'text' || k === 'col_name' ? dText : 1;
    out.push([`enc_dict.${k}.weight`, [d, fanIn]], [`enc_dict.${k}.bias`, [d]]);
  }
  // dec_dict (model.py:124-131)
  for (const k of DEC_KEYS) {
    const fanOut = k === 'text' ? dText : 1;
    out.push([`dec_dict.${k}.weight`, [fanOut, d]], [`dec_dict.${k}.bias`, [fanOut]]);
  }
  // norm_dict (model.py:132-140)
  for (const k of ENC_KEYS) out.push([`norm_dict.${k}.weight`, [d]]);
  // mask_embs (model.py:141-146), alphabetical: nn.ParameterDict sorts plain-dict keys
  for (const k of MASK_EMB_STATE_DICT_KEYS) out.push([`mask_embs.${k}`, [d]]);
  // blocks (model.py:147-149, 71-80)
  for (let b = 0; b < nBlocks; b++) {
    for (const l of BLOCK_NORM_KEYS) out.push([`blocks.${b}.norms.${l}.weight`, [d]]);
    for (const l of BLOCK_ATTN_KEYS)
      for (const p of ['wq', 'wk', 'wv', 'wo']) out.push([`blocks.${b}.attns.${l}.${p}.weight`, [d, d]]);
    out.push(
      [`blocks.${b}.ffn.w1.weight`, [dFF, d]],
      [`blocks.${b}.ffn.w2.weight`, [d, dFF]],
      [`blocks.${b}.ffn.w3.weight`, [dFF, d]],
    );
  }
  out.push(['norm_out.weight', [d]]); // model.py:150
  return out;
}

/** Total parameter count. PAPER_CONFIG → 22,336,131 (spec §7.3). */
export function paramCount(cfg: RTConfig): number {
  let n = 0;
  for (const [, shape] of parameterShapes(cfg)) n += shape.reduce((a, b) => a * b, 1);
  return n;
}
