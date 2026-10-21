/**
 * The paper's published ablation numbers that this chapter compares against (typed from the PDF,
 * arXiv 2510.06377v3; each value checked against the printed table). Means only: per-task values
 * live in Tables 11 and 13 (results chapter).
 *
 * Provenance (spec §11 intro, §13 #1): Tables 4, 5, 11 and 13 are unchanged since arXiv v1, so their
 * "none" column (70.1 / 22.8) predates the released sampler; the headline Table 1/2 means are 69.7 / 22.7.
 */
import type { AttnLayer } from '@/engine/types';

/** One row of Table 4 (p.9): mean AUROC (10 binary tasks) and R² (8 regression tasks), in %. */
export interface ContextAblationRow {
  zsClf: number;
  zsReg: number;
  ftClf: number;
  ftReg: number;
}

/** Table 4 (p.9), "Mean AUROC (%) and R² (%) on ablating context window components". */
export const TABLE4: Record<'none' | 'colNames' | 'selfLabels' | 'otherLabels', ContextAblationRow> = {
  none: { zsClf: 70.1, zsReg: 22.8, ftClf: 77.2, ftReg: 33.2 },
  colNames: { zsClf: 69.5, zsReg: 20.5, ftClf: 77.5, ftReg: 33.2 },
  selfLabels: { zsClf: 53.8, zsReg: -5.5, ftClf: 77.1, ftReg: 26.7 },
  otherLabels: { zsClf: 70.6, zsReg: 22.9, ftClf: 77.4, ftReg: 31.0 },
};

/**
 * Attention-layer ablations: models RETRAINED without that layer, parameter count kept constant
 * by adding blocks (p.9 §5.2). R² from Table 5 (p.10), AUROC from Table 13 (p.23). Means in %.
 */
export const LAYER_ABLATION: Record<'none' | AttnLayer, ContextAblationRow> = {
  none: { zsClf: 70.1, zsReg: 22.8, ftClf: 77.2, ftReg: 33.2 },
  col: { zsClf: 69.8, zsReg: 11.9, ftClf: 77.3, ftReg: 31.2 },
  feat: { zsClf: 70.6, zsReg: 18.8, ftClf: 76.7, ftReg: 32.2 },
  nbr: { zsClf: 70.8, zsReg: 22.0, ftClf: 76.5, ftReg: 31.4 },
  full: { zsClf: 71.1, zsReg: 23.7, ftClf: 77.0, ftReg: 32.7 },
};

/** Headline zero-shot means, Table 1 (p.8, AUROC) and Table 2 (p.8, R²), RT "No" column. */
export const HEADLINE = { zsClf: 69.7, zsReg: 22.7 } as const;

/** Fig. 4 (p.8-9): what was varied, and the paper's own summary (the figure prints no table). */
export const FIG4 = {
  contexts: [256, 512, 768, 1024],
  pretrainedAt: 1024,
  quote:
    'RT’s performance degrades gracefully, showing a drop of only 4% absolute AUROC and 7% absolute R² with 25% lower time and 20% lower memory.',
} as const;

/** Table 12 (p.23): mean ± sd in-context labels per 1024-cell context, a few rows for scale. */
export const TABLE12_EXAMPLES: { task: string; self: string; other: string }[] = [
  { task: 'rel-f1 driver-dnf', self: '19 ± 14', other: '0 ± 0' },
  { task: 'rel-stack user-engagement', self: '16 ± 10', other: '10 ± 10' },
  { task: 'rel-amazon user-churn', self: '4 ± 4', other: '1 ± 5' },
  { task: 'rel-trial study-outcome', self: '0 ± 0', other: '0 ± 0' },
];

/** Ranked drivers of zero-shot transfer, p.10 §7 (quoted in order). */
export const DRIVERS: string[] = [
  'time-series forecasting from past task rows for the target entity',
  'column attention with the help of Relational Attention to generalize to new column distributions',
  'feature attention, both in the same table and in parent tables, with the help of Relational Attention to generalize to new entity types',
  'schema semantics from table and column names',
  'in-context learning from task rows for other entities',
];

/** The paper metric that matches a task type: mean zero-shot AUROC (10 binary tasks) or R² (8 regression tasks). */
export function paperMetric(binary: boolean): { key: 'zsClf' | 'zsReg'; name: string; tasks: string } {
  return binary ? { key: 'zsClf', name: 'AUROC', tasks: '10 binary tasks' } : { key: 'zsReg', name: 'R²', tasks: '8 regression tasks' };
}

/** A paper number with a typographic minus. */
export function paperNum(v: number): string {
  return (v < 0 ? '−' : '') + Math.abs(v).toFixed(1);
}

/** "AUROC 70.1 → 53.8" (mean zero-shot, the metric of the live task type). */
export function paperArrow(none: ContextAblationRow, row: ContextAblationRow, binary: boolean): string {
  const m = paperMetric(binary);
  return `${m.name} ${paperNum(none[m.key])} → ${paperNum(row[m.key])}`;
}

/** Table 10 (p.21): supervised fine-tuning on the full training set, mean AUROC / R² (%), RT pretrained or not. */
export const TABLE10 = {
  pretrained: { clf: 77.2, reg: 33.2 },
  untrained: { clf: 76.1, reg: 28.4 },
} as const;
