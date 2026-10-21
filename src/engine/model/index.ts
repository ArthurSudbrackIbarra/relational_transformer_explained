/** Public API of the RT model engine (spec §5-§8). */
export {
  MINI_CONFIG,
  PAPER_CONFIG,
  DEFAULT_L,
  DEFAULT_W,
  PAPER_L,
  PAPER_W,
  SEM_KEYS,
  ENC_KEYS,
  DEC_KEYS,
  MASK_EMB_STATE_DICT_KEYS,
  dHead,
  parameterShapes,
  paramCount,
  type SemKey,
} from './config';
export {
  linear,
  linearVec,
  rmsnorm,
  rmsnormRow,
  silu,
  allRows,
  add,
  l2norm,
  maxAbsDiff,
  maskRows,
  maskedAttentionCore,
  type MaskRows,
} from './ops';
export { huber, bceWithLogits, lossTerms, meanLoss, type LossTerm, type DecoderOutputs } from './loss';
export {
  WEIGHTS_FORMAT,
  loadWeights,
  parseWeights,
  serializeWeights,
  validateWeights,
  initWeights,
  tensor,
  type WeightsManifest,
} from './weights';
export {
  forward,
  forwardWithMasks,
  embedTokens,
  scalarValueTerm,
  textEmbedder,
  type RTForwardResult,
  type TextEmbedder,
} from './forward';
