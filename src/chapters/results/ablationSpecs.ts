/**
 * Configurations of the ablation explorer for each ablation family of the paper.
 */
import type { AblationSpec } from './Ablation';
import { TABLE11, TABLE14, TABLE15, TABLE5_13 } from './data';

export const CONTEXT_SPEC: AblationSpec = {
  table: TABLE11,
  variants: [
    { id: 'none', label: 'none', short: 'none' },
    { id: 'colNames', label: 'column and table names shuffled', short: 'Names shuffled' },
    { id: 'selfLabels', label: 'self labels removed', short: 'Self labels' },
    { id: 'otherLabels', label: 'other-entity labels removed', short: 'Other labels' },
  ],
  settings: ['zs', 'ft'],
  source: { clf: { label: 'Table 11', page: 22 }, reg: { label: 'Table 11', page: 22 } },
  refName: 'none',
  describe: (v) =>
    v.id === 'colNames' ? 'column and table names shuffled' : v.id === 'selfLabels' ? 'self labels removed' : 'other-entity labels removed',
};

export const ATTENTION_SPEC: AblationSpec = {
  table: TABLE5_13,
  variants: [
    { id: 'none', label: 'none', short: 'none' },
    { id: 'col', label: 'column attention removed', short: 'Column' },
    { id: 'feat', label: 'feature attention removed', short: 'Feature' },
    { id: 'nbr', label: 'neighbor attention removed', short: 'Neighbor' },
    { id: 'full', label: 'full attention removed', short: 'Full' },
  ],
  settings: ['zs', 'ft'],
  source: { clf: { label: 'Table 13', page: 23 }, reg: { label: 'Table 5', page: 10 } },
  refName: 'none',
  describe: (v) => `${v.short.toLowerCase()} attention removed`,
};

export const ORDER_SPEC: AblationSpec = {
  table: TABLE15,
  variants: [
    { id: 'CFN', label: 'CFN (RT’s order)', short: 'CFN' },
    { id: 'CNF', label: 'CNF', short: 'CNF' },
    { id: 'FCN', label: 'FCN', short: 'FCN' },
    { id: 'FNC', label: 'FNC', short: 'FNC' },
    { id: 'NCF', label: 'NCF', short: 'NCF' },
    { id: 'NFC', label: 'NFC', short: 'NFC' },
    { id: 'Parallel', label: 'parallel (outputs summed)', short: 'Parallel' },
  ],
  settings: ['zs'],
  source: { clf: { label: 'Table 15', page: 24 }, reg: { label: 'Table 15', page: 24 } },
  refName: 'CFN',
  describe: (v) => (v.id === 'Parallel' ? 'the three layers in parallel vs CFN' : `order ${v.id} vs CFN`),
};

export const MASKING_SPEC: AblationSpec = {
  table: TABLE14,
  variants: [
    { id: 'p0', label: 'P(mask) = 0.0 (target only)', short: 'P = 0.0' },
    { id: 'p02', label: 'P(mask) = 0.2', short: 'P = 0.2' },
    { id: 'p04', label: 'P(mask) = 0.4', short: 'P = 0.4' },
    { id: 'np', label: 'no pretraining', short: 'No pretraining' },
  ],
  settings: ['zs', 'ft'],
  source: { clf: { label: 'Table 14', page: 24 }, reg: { label: 'Table 14', page: 24 } },
  refName: 'P(mask) = 0.0',
  describe: (v) => (v.id === 'np' ? 'no pretraining vs P(mask) = 0.0' : `pretraining with ${v.label} vs 0.0`),
};
