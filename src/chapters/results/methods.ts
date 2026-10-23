/**
 * Visual identity of every method in the paper's comparison tables (dataviz: color follows the
 * entity, never its rank; fixed slots, never cycled; composite encoding past the palette).
 *
 *   family  → hue (categorical slot)
 *   zero-shot tables: circle = target DB unseen ("No"), square = continued pretraining ("Yes");
 *             Gemma3 sizes ("Maybe") share the aqua hue and differ by shape;
 *   supervised / feature tables: hollow = untrained (or cannot be pretrained), filled = pretrained.
 * EntityMean, a non-neural baseline, is drawn in the neutral slot 0.
 */
import type { ShapeName } from '@/components/viz/shapes';

export interface MethodStyle {
  id: string;
  /** Legend / tooltip label. */
  label: string;
  /** Categorical slot 0..8 (0 = neutral gray). */
  slot: number;
  shape: ShapeName;
  hollow?: boolean;
  /** What the method is, in one line (tooltip). */
  blurb: string;
}

const M = (id: string, label: string, slot: number, shape: ShapeName, hollow: boolean, blurb: string): MethodStyle => ({
  id,
  label,
  slot,
  shape,
  hollow,
  blurb,
});

export const METHODS: Record<string, MethodStyle> = {
  // zero-shot tables (1, 2, 16)
  gemma4b: M(
    'gemma4b',
    'Gemma3 4B',
    3,
    'triangle',
    false,
    'LLM prompted with the text-serialized context subgraph (App. I.2). Pretraining data unknown: “Maybe”.',
  ),
  gemma12b: M(
    'gemma12b',
    'Gemma3 12B',
    3,
    'hexagon',
    false,
    'LLM prompted with the text-serialized context subgraph (App. I.2). Pretraining data unknown: “Maybe”.',
  ),
  gemma27b: M(
    'gemma27b',
    'Gemma3 27B',
    3,
    'diamond',
    false,
    'LLM prompted with the text-serialized context subgraph (App. I.2). Pretraining data unknown: “Maybe”.',
  ),
  entityMean: M(
    'entityMean',
    'EntityMean',
    0,
    'circle',
    false,
    'Non-neural: the mean of the target entity’s past labels, else the global training mean (App. I.1).',
  ),
  griffinNo: M(
    'griffinNo',
    'Griffin (No)',
    2,
    'circle',
    false,
    'Row-level graph foundation model, 22M parameters, same pretraining data as RT; target DB unseen.',
  ),
  rtNo: M('rtNo', 'RT (No)', 7, 'circle', false, 'RT, leave-one-DB-out: the target database was never seen.'),
  relllmYes: M('relllmYes', 'RelLLM (Yes)', 1, 'square', false, 'GNN encoders + LLM, retrieval-augmented; own prompt construction. Target DB seen.'),
  griffinYes: M('griffinYes', 'Griffin (Yes)', 2, 'square', false, 'Griffin with continued pretraining on the target DB (other tasks).'),
  rtYes: M('rtYes', 'RT (Yes)', 7, 'square', false, 'RT with continued pretraining on the target DB’s other tasks; the target task is still unseen.'),
  // supervised (Table 10)
  rdlGnn: M('rdlGnn', 'RDL-GNN', 4, 'triangle', true, 'Schema-specific GNN over rows (cannot be pretrained).'),
  relGnn: M('relGnn', 'RelGNN', 6, 'triangle-down', true, 'Schema-specific GNN with composite message passing (cannot be pretrained).'),
  relGt: M('relGt', 'RelGT', 8, 'cross', true, 'Relational graph transformer, schema-specific (cannot be pretrained).'),
  relllm: M('relllm', 'RelLLM', 1, 'circle', false, 'GNN encoders + LLM. Table 10 marks it “pretrained? Yes” and prints “nan” for R².'),
  griffinU: M('griffinU', 'Griffin, untrained', 2, 'circle', true, 'Griffin fine-tuned from random initialization.'),
  griffinP: M('griffinP', 'Griffin, pretrained', 2, 'circle', false, 'Griffin fine-tuned from its pretrained checkpoint.'),
  rtU: M('rtU', 'RT, untrained', 7, 'circle', true, 'RT fine-tuned from random initialization.'),
  rtP: M('rtP', 'RT, pretrained', 7, 'circle', false, 'RT fine-tuned from the pretrained checkpoint.'),
  // feature extraction (Table 3)
  gnnU3: M('gnnU3', 'MLP on GNN (untrained)', 4, 'triangle', true, 'A 2-layer MLP trained on frozen embeddings of an untrained GNN.'),
  rtU3: M('rtU3', 'MLP on RT (untrained)', 7, 'circle', true, 'A 2-layer MLP trained on frozen embeddings of an untrained RT.'),
  rtP3: M(
    'rtP3',
    'MLP on RT (pretrained)',
    7,
    'circle',
    false,
    'A 2-layer MLP trained on frozen embeddings of a pretrained RT: the masked token’s representation before the decoder.',
  ),
  fullFt3: M('fullFt3', 'Full fine-tuning (RT)', 7, 'square', false, 'The whole RT fine-tuned.'),
};

/** Table 10 column id → method id. */
export const T10_METHOD: Record<string, string> = {
  rdlGnn: 'rdlGnn',
  relGnn: 'relGnn',
  relGt: 'relGt',
  relllm: 'relllm',
  griffinNo: 'griffinU',
  griffinYes: 'griffinP',
  rtNo: 'rtU',
  rtYes: 'rtP',
};

/** Table 3 column id → method id. */
export const T3_METHOD: Record<string, string> = { gnnU: 'gnnU3', rtU: 'rtU3', rtP: 'rtP3', fullFt: 'fullFt3' };
