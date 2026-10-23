/**
 * The paper's published numbers (Ranjan et al., ICLR 2026, arXiv 2510.06377v3), typed from the PDF
 * (printed page = PDF page) and checked against `pdftotext -layout` of every page involved.
 * Spec: research/RT_TECHNICAL_SPEC.md §11 (numbers), §13 #1 (provenance), §10 #25-26.
 *
 * Conventions
 *  - Values are percentages exactly as printed (AUROC % or R² %), one decimal.
 *  - `null` = no number in the paper (Table 10 prints "nan" for RelLLM R²; Table 16 prints "−" for RelLLM).
 *  - `LT9` = the paper prints "< −9" (Table 16, LLM regression).
 *  - `mean` is the mean row AS PRINTED. Recomputed means are only ever shown labeled as such.
 *  - Row order is the paper's row order for that table.
 *
 * Nothing in this file is computed: it is a transcription. Any derived number on the page is computed
 * from these arrays at render time and labeled as computed.
 */

// ─────────────────────────────────────────────────────────────────────────────
// Tasks
// ─────────────────────────────────────────────────────────────────────────────

export type TaskKey =
  // binary classification (10)
  | 'amazon/item-churn'
  | 'amazon/user-churn'
  | 'avito/user-clicks'
  | 'avito/user-visits'
  | 'f1/driver-dnf'
  | 'f1/driver-top3'
  | 'hm/user-churn'
  | 'stack/user-badge'
  | 'stack/user-engagement'
  | 'trial/study-outcome'
  // regression (8)
  | 'amazon/item-ltv'
  | 'amazon/user-ltv'
  | 'avito/ad-ctr'
  | 'f1/driver-position'
  | 'hm/item-sales'
  | 'stack/post-votes'
  | 'trial/site-success'
  | 'trial/study-adverse';

export interface TaskInfo {
  key: TaskKey;
  /** RelBench database, e.g. "rel-amazon". */
  db: string;
  /** Task name as printed in Tables 1/2 (abbreviated there: "user-engage", "study-out", "driver-pos", …). */
  task: string;
  kind: 'clf' | 'reg';
}

const T = (key: TaskKey, db: string, task: string, kind: 'clf' | 'reg'): TaskInfo => ({ key, db, task, kind });

export const TASKS: Record<TaskKey, TaskInfo> = {
  'amazon/item-churn': T('amazon/item-churn', 'rel-amazon', 'item-churn', 'clf'),
  'amazon/user-churn': T('amazon/user-churn', 'rel-amazon', 'user-churn', 'clf'),
  'avito/user-clicks': T('avito/user-clicks', 'rel-avito', 'user-clicks', 'clf'),
  'avito/user-visits': T('avito/user-visits', 'rel-avito', 'user-visits', 'clf'),
  'f1/driver-dnf': T('f1/driver-dnf', 'rel-f1', 'driver-dnf', 'clf'),
  'f1/driver-top3': T('f1/driver-top3', 'rel-f1', 'driver-top3', 'clf'),
  'hm/user-churn': T('hm/user-churn', 'rel-hm', 'user-churn', 'clf'),
  'stack/user-badge': T('stack/user-badge', 'rel-stack', 'user-badge', 'clf'),
  'stack/user-engagement': T('stack/user-engagement', 'rel-stack', 'user-engage', 'clf'),
  'trial/study-outcome': T('trial/study-outcome', 'rel-trial', 'study-out', 'clf'),
  'amazon/item-ltv': T('amazon/item-ltv', 'rel-amazon', 'item-ltv', 'reg'),
  'amazon/user-ltv': T('amazon/user-ltv', 'rel-amazon', 'user-ltv', 'reg'),
  'avito/ad-ctr': T('avito/ad-ctr', 'rel-avito', 'ad-ctr', 'reg'),
  'f1/driver-position': T('f1/driver-position', 'rel-f1', 'driver-pos', 'reg'),
  'hm/item-sales': T('hm/item-sales', 'rel-hm', 'item-sales', 'reg'),
  'stack/post-votes': T('stack/post-votes', 'rel-stack', 'post-votes', 'reg'),
  'trial/site-success': T('trial/site-success', 'rel-trial', 'site-succ', 'reg'),
  'trial/study-adverse': T('trial/study-adverse', 'rel-trial', 'study-adv', 'reg'),
};

/** Row order of Tables 1, 11, 13, 14, 15 (classification). */
export const CLF_ORDER: TaskKey[] = [
  'amazon/item-churn',
  'amazon/user-churn',
  'avito/user-clicks',
  'avito/user-visits',
  'f1/driver-dnf',
  'f1/driver-top3',
  'hm/user-churn',
  'stack/user-badge',
  'stack/user-engagement',
  'trial/study-outcome',
];

/** Row order of Tables 2, 5, 11, 14, 15, 16 (regression). */
export const REG_ORDER: TaskKey[] = [
  'amazon/item-ltv',
  'amazon/user-ltv',
  'avito/ad-ctr',
  'f1/driver-position',
  'hm/item-sales',
  'stack/post-votes',
  'trial/site-success',
  'trial/study-adverse',
];

/** "amazon · item-churn" (the paper drops "rel-" in some tables). */
export function taskLabel(k: TaskKey): string {
  const t = TASKS[k];
  return `${t.db.replace(/^rel-/, '')} · ${t.task}`;
}

// ─────────────────────────────────────────────────────────────────────────────
// Generic table shape
// ─────────────────────────────────────────────────────────────────────────────

/** The paper prints "< −9" (Table 16). */
export const LT9 = 'lt-9' as const;
export type Cell = number | null | typeof LT9;

export interface PaperTable<C extends string> {
  /** "Table 1" */
  id: string;
  /** Printed page. */
  page: number;
  caption: string;
  cols: readonly C[];
  rows: { task: TaskKey; v: Cell[] }[];
  /** The mean row as printed. */
  mean: Cell[];
}

// ─────────────────────────────────────────────────────────────────────────────
// Table 1 (p.8): zero-shot test AUROC (%), 10 binary tasks. Random/majority = 50.0
// ─────────────────────────────────────────────────────────────────────────────

export const T1_COLS = ['gemma4b', 'gemma12b', 'gemma27b', 'entityMean', 'griffinNo', 'rtNo', 'relllmYes', 'griffinYes', 'rtYes'] as const;
export type T1Col = (typeof T1_COLS)[number];

export const TABLE1: PaperTable<T1Col> = {
  id: 'Table 1',
  page: 8,
  caption:
    'Zero-shot test AUROC (%) for 10 binary classification tasks. Higher is better. Random/majority baseline is 50.0. The target task is never seen during pretraining.',
  cols: T1_COLS,
  rows: [
    { task: 'amazon/item-churn', v: [62.1, 55.0, 42.1, 73.0, 69.0, 70.9, 64.1, 71.9, 73.3] },
    { task: 'amazon/user-churn', v: [58.1, 54.7, 50.5, 64.4, 62.3, 64.0, 60.1, 64.1, 66.1] },
    { task: 'avito/user-clicks', v: [54.5, 59.5, 59.8, 44.7, 45.9, 59.5, 62.3, 45.9, 60.9] },
    { task: 'avito/user-visits', v: [60.1, 57.9, 62.7, 60.7, 60.7, 61.8, 56.2, 62.2, 62.6] },
    { task: 'f1/driver-dnf', v: [56.2, 54.6, 75.8, 75.4, 57.7, 81.2, 71.8, 57.7, 81.2] },
    { task: 'f1/driver-top3', v: [84.6, 90.5, 91.4, 85.0, 82.5, 89.3, 70.6, 81.8, 89.3] },
    { task: 'hm/user-churn', v: [59.8, 47.1, 48.7, 64.4, 60.2, 62.8, 56.0, 60.4, 63.3] },
    { task: 'stack/user-badge', v: [79.1, 79.8, 80.0, 66.2, 73.5, 80.1, 62.1, 82.3, 81.1] },
    { task: 'stack/user-engagement', v: [65.9, 67.8, 78.0, 83.5, 77.5, 75.7, 69.5, 89.4, 86.9] },
    { task: 'trial/study-outcome', v: [52.6, 57.4, 57.2, 50.0, 51.0, 51.8, 59.0, 57.2, 54.6] },
  ],
  mean: [63.3, 62.4, 64.6, 66.7, 64.0, 69.7, 63.2, 67.3, 71.9],
};

// ─────────────────────────────────────────────────────────────────────────────
// Table 2 (p.8) + Table 16 (p.26): zero-shot test R² (%), 8 regression tasks. Global mean = 0.0
// Table 16 repeats Table 2 and adds the Gemma3 columns ("< −9" on most tasks) and RelLLM ("−").
// ─────────────────────────────────────────────────────────────────────────────

export const T2_COLS = ['gemma4b', 'gemma12b', 'gemma27b', 'entityMean', 'griffinNo', 'rtNo', 'griffinYes', 'rtYes'] as const;
export type T2Col = (typeof T2_COLS)[number];

export const TABLE2: PaperTable<T2Col> = {
  id: 'Table 2 + Table 16',
  page: 8,
  caption:
    'Zero-shot test R² (%) for 8 regression tasks. Higher is better. Global mean baseline is 0.0. The Gemma3 columns come from Table 16 (p.26); “< −9” is printed there.',
  cols: T2_COLS,
  rows: [
    { task: 'amazon/item-ltv', v: [LT9, LT9, LT9, 54.2, 20.1, 32.5, 20.1, 32.2] },
    { task: 'amazon/user-ltv', v: [LT9, LT9, LT9, 19.9, 20.6, 36.9, 24.4, 38.3] },
    { task: 'avito/ad-ctr', v: [LT9, LT9, -8.2, 3.4, 2.4, 4.5, 2.4, 8.0] },
    { task: 'f1/driver-position', v: [35.2, 43.4, 52.4, 38.2, -0.7, 52.4, 4.6, 58.7] },
    { task: 'hm/item-sales', v: [LT9, LT9, LT9, 1.8, 2.7, 14.0, 2.5, 30.9] },
    { task: 'stack/post-votes', v: [LT9, LT9, LT9, 43.7, 27.4, 33.9, 27.1, 35.0] },
    { task: 'trial/site-success', v: [LT9, LT9, LT9, -6.4, 1.4, 4.5, 2.6, 5.1] },
    { task: 'trial/study-adverse', v: [LT9, LT9, -7.1, -0.5, -2.5, 2.6, -2.5, 3.1] },
  ],
  mean: [LT9, LT9, LT9, 19.3, 8.9, 22.7, 10.1, 26.4],
};

// ─────────────────────────────────────────────────────────────────────────────
// Table 10 (p.21): supervised fine-tuning (full training set, until convergence, best val checkpoint)
// ─────────────────────────────────────────────────────────────────────────────

export const T10_COLS = ['rdlGnn', 'relGnn', 'relGt', 'relllm', 'griffinNo', 'griffinYes', 'rtNo', 'rtYes'] as const;
export type T10Col = (typeof T10_COLS)[number];

export interface Table10 {
  clf: PaperTable<T10Col> & { size: string[] };
  reg: PaperTable<T10Col> & { size: string[] };
}

export const TABLE10: Table10 = {
  clf: {
    id: 'Table 10',
    page: 21,
    caption: 'Supervised fine-tuning, AUROC (%) for 10 binary classification tasks. Random/majority baseline is 50.0.',
    cols: T10_COLS,
    size: ['4.7M', '3.9M', '3.4M', '1.4M', '2.6M', '87K', '59K', '12K', '11K', '1.4K'],
    rows: [
      { task: 'amazon/user-churn', v: [70.7, 71.0, 70.4, 71.9, 70.0, 69.4, 70.5, 70.8] },
      { task: 'hm/user-churn', v: [69.4, 70.9, 69.3, 70.5, 68.3, 68.0, 69.9, 70.5] },
      { task: 'stack/user-badge', v: [88.9, 89.0, 86.3, 89.6, 87.0, 87.0, 88.5, 88.7] },
      { task: 'stack/user-engagement', v: [90.6, 90.8, 90.5, 91.2, 89.8, 90.4, 90.0, 90.2] },
      { task: 'amazon/item-churn', v: [82.8, 82.6, 82.5, 83.4, 81.1, 79.9, 83.2, 83.4] },
      { task: 'avito/user-visits', v: [66.1, 66.2, 66.8, 67.0, 65.0, 62.6, 65.0, 65.2] },
      { task: 'avito/user-clicks', v: [63.1, 68.2, 68.3, 66.7, 63.0, 64.7, 63.6, 59.0] },
      { task: 'trial/study-outcome', v: [68.6, 71.2, 68.6, 71.0, 68.9, 64.6, 68.6, 68.2] },
      { task: 'f1/driver-dnf', v: [72.5, 75.3, 75.9, 77.2, 74.5, 66.7, 78.7, 84.2] },
      { task: 'f1/driver-top3', v: [80.9, 85.7, 83.5, 82.2, 82.5, 78.7, 82.7, 91.9] },
    ],
    mean: [75.4, 77.1, 76.2, 77.1, 75.0, 73.2, 76.1, 77.2],
  },
  reg: {
    id: 'Table 10',
    page: 21,
    caption: 'Supervised fine-tuning, R² (%) for 8 regression tasks. Global-mean baseline is 0.0. RelLLM prints “nan”.',
    cols: T10_COLS,
    size: ['5.5M', '4.7M', '2.7M', '2.5M', '150K', '43K', '7.5K', '5.1K'],
    rows: [
      { task: 'hm/item-sales', v: [21.8, 22.1, 22.6, null, 31.1, 30.4, 45.7, 39.0] },
      { task: 'amazon/user-ltv', v: [21.9, 17.9, 17.5, null, 30.7, 32.9, 47.9, 47.4] },
      { task: 'amazon/item-ltv', v: [3.7, 3.5, 3.4, null, 23.4, 25.2, 31.5, 36.8] },
      { task: 'stack/post-votes', v: [17.9, 12.2, 13.1, null, 41.8, 42.7, 37.1, 36.5] },
      { task: 'trial/site-success', v: [4.0, -9.5, -28.8, null, 6.8, -2.4, -8.8, 6.4] },
      { task: 'trial/study-adverse', v: [18.8, 19.7, 17.0, null, 11.2, 18.2, 41.3, 43.4] },
      { task: 'f1/driver-position', v: [7.6, 20.7, 12.4, null, 29.9, 0.6, 33.7, 51.6] },
      { task: 'avito/ad-ctr', v: [18.3, 15.6, 18.4, null, 5.9, 8.4, -1.5, 4.5] },
    ],
    mean: [14.3, 12.8, 9.4, null, 22.6, 19.5, 28.4, 33.2],
  },
};

// ─────────────────────────────────────────────────────────────────────────────
// Table 3 (p.8): RT as feature extractor. A 2-layer MLP on frozen embeddings (MLP-only FT) vs full FT.
// Only 5 classification and 4 regression tasks.
// ─────────────────────────────────────────────────────────────────────────────

export const T3_COLS = ['gnnU', 'rtU', 'rtP', 'fullFt'] as const;
export type T3Col = (typeof T3_COLS)[number];

export const TABLE3: { clf: PaperTable<T3Col>; reg: PaperTable<T3Col> } = {
  clf: {
    id: 'Table 3',
    page: 8,
    caption: 'RT as feature extractor, AUROC (%). GNN and RT(U) are untrained, RT(P) is pretrained. 5 classification tasks.',
    cols: T3_COLS,
    rows: [
      { task: 'amazon/item-churn', v: [79.6, 79.9, 81.6, 83.4] },
      { task: 'amazon/user-churn', v: [67.6, 67.0, 68.5, 70.8] },
      { task: 'hm/user-churn', v: [67.5, 68.6, 69.3, 70.5] },
      { task: 'stack/user-badge', v: [85.5, 86.9, 87.0, 88.7] },
      { task: 'stack/user-engagement', v: [89.1, 87.8, 89.3, 90.2] },
    ],
    mean: [77.9, 78.0, 79.1, 80.7],
  },
  reg: {
    id: 'Table 3',
    page: 8,
    caption: 'RT as feature extractor, R² (%). 4 regression tasks.',
    cols: T3_COLS,
    rows: [
      { task: 'amazon/item-ltv', v: [1.6, 43.3, 56.5, 36.8] },
      { task: 'amazon/user-ltv', v: [13.6, 31.8, 46.8, 47.9] },
      { task: 'hm/item-sales', v: [14.1, 17.5, 38.8, 45.7] },
      { task: 'stack/post-votes', v: [14.1, 37.4, 42.6, 37.1] },
    ],
    mean: [10.9, 32.5, 46.2, 41.9],
  },
};

// ─────────────────────────────────────────────────────────────────────────────
// Ablation tables. Each has a zero-shot block and (mostly) a fine-tuned block.
// ─────────────────────────────────────────────────────────────────────────────

export interface AblationTable {
  id: string;
  page: number;
  caption: string;
  /** Column ids of one block; the first is the reference ("none", "0.0", "CFN"). */
  cols: readonly string[];
  /** Per-task values: zero-shot block and fine-tuned block (same column order; FT may have extra cols). */
  clf?: { rows: { task: TaskKey; zs: number[]; ft?: number[] }[]; mean: { zs: number[]; ft?: number[] } };
  reg?: { rows: { task: TaskKey; zs: number[]; ft?: number[] }[]; mean: { zs: number[]; ft?: number[] } };
}

/** Table 4 (p.9): mean AUROC / R² when ablating context components. Order: zero-shot clf, reg, fine-tuned clf, reg. */
export const TABLE4 = {
  id: 'Table 4',
  page: 9,
  rows: [
    { id: 'none', zsClf: 70.1, zsReg: 22.8, ftClf: 77.2, ftReg: 33.2 },
    { id: 'colNames', zsClf: 69.5, zsReg: 20.5, ftClf: 77.5, ftReg: 33.2 },
    { id: 'selfLabels', zsClf: 53.8, zsReg: -5.5, ftClf: 77.1, ftReg: 26.7 },
    { id: 'otherLabels', zsClf: 70.6, zsReg: 22.9, ftClf: 77.4, ftReg: 31.0 },
  ],
} as const;

/** Table 11 (p.22): per-task context ablations. Columns: none, col names, self labels, other labels. */
export const TABLE11: AblationTable = {
  id: 'Table 11',
  page: 22,
  caption:
    'Ablation of context construction: column and table names shuffled, past labels of the target entity removed (self labels), labels of other entities removed. Zero-shot and after fine-tuning with the same modified contexts.',
  cols: ['none', 'colNames', 'selfLabels', 'otherLabels'],
  clf: {
    rows: [
      { task: 'amazon/item-churn', zs: [70.2, 71.0, 48.1, 72.2], ft: [83.4, 83.3, 83.4, 83.2] },
      { task: 'amazon/user-churn', zs: [63.9, 63.9, 55.2, 64.2], ft: [70.8, 70.4, 70.8, 70.6] },
      { task: 'avito/user-clicks', zs: [59.5, 58.5, 55.0, 59.7], ft: [59.0, 60.8, 61.0, 60.1] },
      { task: 'avito/user-visits', zs: [61.8, 61.2, 49.9, 62.1], ft: [65.2, 65.1, 64.4, 65.3] },
      { task: 'f1/driver-dnf', zs: [82.0, 81.7, 50.3, 82.0], ft: [84.2, 84.3, 83.6, 84.3] },
      { task: 'f1/driver-top3', zs: [89.1, 86.5, 74.3, 89.1], ft: [91.9, 91.8, 89.7, 91.7] },
      { task: 'hm/user-churn', zs: [62.8, 60.0, 54.5, 62.9], ft: [70.5, 70.6, 70.2, 70.7] },
      { task: 'stack/user-badge', zs: [80.0, 79.2, 54.8, 82.4], ft: [88.7, 88.9, 88.8, 88.8] },
      { task: 'stack/user-engagement', zs: [77.1, 80.1, 41.9, 77.2], ft: [90.2, 90.1, 90.2, 90.2] },
      { task: 'trial/study-outcome', zs: [54.5, 53.2, 54.5, 54.6], ft: [68.2, 69.0, 68.8, 68.9] },
    ],
    mean: { zs: [70.1, 69.5, 53.8, 70.6], ft: [77.2, 77.5, 77.1, 77.4] },
  },
  reg: {
    rows: [
      { task: 'amazon/item-ltv', zs: [33.2, 33.0, -2.8, 33.1], ft: [36.8, 34.7, 28.0, 35.1] },
      { task: 'amazon/user-ltv', zs: [36.4, 33.6, -5.7, 36.1], ft: [47.4, 47.2, 21.9, 37.6] },
      { task: 'avito/ad-ctr', zs: [4.5, 5.7, -3.6, 4.5], ft: [4.5, 4.9, -5.2, 4.3] },
      { task: 'f1/driver-position', zs: [54.7, 50.3, -31.9, 54.7], ft: [51.6, 52.3, 54.7, 54.1] },
      { task: 'hm/item-sales', zs: [14.0, 9.2, -2.8, 14.7], ft: [39.0, 39.5, 33.6, 39.3] },
      { task: 'stack/post-votes', zs: [32.4, 27.5, -0.7, 32.8], ft: [36.5, 35.9, 34.9, 37.2] },
      { task: 'trial/site-success', zs: [5.2, 3.2, 1.5, 5.1], ft: [6.4, 8.3, 8.7, 2.5] },
      { task: 'trial/study-adverse', zs: [2.1, 1.3, 2.1, 2.1], ft: [43.4, 42.4, 37.0, 38.1] },
    ],
    mean: { zs: [22.8, 20.5, -5.5, 22.9], ft: [33.2, 33.2, 26.7, 31.0] },
  },
};

/** Tables 5 (p.10, R²) and 13 (p.23, AUROC): one attention layer type removed (more blocks keep the parameter count). */
export const TABLE5_13: AblationTable = {
  id: 'Tables 5 and 13',
  page: 10,
  caption:
    'Ablation of the relational attention layers: column, feature, neighbor or full attention absent. Total parameter count kept constant by increasing the number of layers.',
  cols: ['none', 'col', 'feat', 'nbr', 'full'],
  clf: {
    rows: [
      { task: 'amazon/item-churn', zs: [70.2, 60.8, 73.0, 70.8, 71.5], ft: [83.4, 83.3, 83.1, 82.2, 83.2] },
      { task: 'amazon/user-churn', zs: [63.9, 63.2, 62.8, 62.9, 63.1], ft: [70.8, 70.7, 70.4, 69.3, 70.3] },
      { task: 'avito/user-clicks', zs: [59.5, 63.0, 62.0, 58.7, 61.5], ft: [59.0, 62.1, 64.9, 62.4, 63.3] },
      { task: 'avito/user-visits', zs: [61.8, 60.9, 62.9, 62.7, 60.6], ft: [65.2, 65.6, 65.3, 64.6, 65.0] },
      { task: 'f1/driver-dnf', zs: [82.0, 79.4, 77.1, 81.4, 81.9], ft: [84.2, 83.8, 82.2, 81.1, 82.2] },
      { task: 'f1/driver-top3', zs: [89.1, 87.5, 85.1, 88.0, 89.3], ft: [91.9, 92.0, 85.9, 90.4, 90.2] },
      { task: 'hm/user-churn', zs: [62.8, 66.1, 65.8, 65.6, 64.4], ft: [70.5, 69.8, 70.1, 69.4, 70.1] },
      { task: 'stack/user-badge', zs: [80.0, 79.4, 82.0, 79.7, 81.2], ft: [88.7, 89.2, 88.9, 88.1, 88.8] },
      { task: 'stack/user-engagement', zs: [77.1, 78.2, 80.9, 79.1, 83.1], ft: [90.2, 90.0, 89.5, 89.2, 90.0] },
      { task: 'trial/study-outcome', zs: [54.5, 59.4, 54.8, 58.6, 54.8], ft: [68.2, 66.6, 66.5, 68.3, 67.1] },
    ],
    mean: { zs: [70.1, 69.8, 70.6, 70.8, 71.1], ft: [77.2, 77.3, 76.7, 76.5, 77.0] },
  },
  reg: {
    rows: [
      { task: 'amazon/item-ltv', zs: [33.2, 5.6, 29.8, 33.2, 44.9], ft: [36.8, 34.5, 34.7, 32.9, 33.3] },
      { task: 'amazon/user-ltv', zs: [36.4, 6.5, 30.4, 34.6, 35.0], ft: [47.4, 47.7, 45.6, 46.3, 46.4] },
      { task: 'avito/ad-ctr', zs: [4.5, -1.4, 9.2, 8.5, 5.3], ft: [4.5, -7.1, 16.6, 2.1, 10.6] },
      { task: 'f1/driver-position', zs: [54.7, 36.9, 37.6, 49.4, 50.1], ft: [51.6, 51.4, 42.0, 48.9, 39.5] },
      { task: 'hm/item-sales', zs: [14.0, 11.0, 4.9, 10.5, 13.9], ft: [39.0, 38.1, 34.8, 37.0, 39.6] },
      { task: 'stack/post-votes', zs: [32.4, 28.7, 29.2, 31.2, 30.8], ft: [36.5, 37.2, 35.9, 36.9, 36.4] },
      { task: 'trial/site-success', zs: [5.2, 4.7, 7.9, 6.7, 4.9], ft: [6.4, 4.7, 8.8, 6.7, 7.9] },
      { task: 'trial/study-adverse', zs: [2.1, 3.0, 1.6, 1.8, 5.1], ft: [43.4, 43.0, 39.0, 40.3, 48.4] },
    ],
    mean: { zs: [22.8, 11.9, 18.8, 22.0, 23.7], ft: [33.2, 31.2, 32.2, 31.4, 32.7] },
  },
};

/**
 * Table 14 (p.24): pretraining with multi-cell masking. Zero-shot columns P(mask) = 0.0, 0.2, 0.4;
 * fine-tuned columns 0.0, 0.2, 0.4 and NP (no pretraining). Stored with a 4th zero-shot slot = null-free
 * by keeping NP only in `ft` (the zero-shot block has three columns).
 */
export const TABLE14: AblationTable = {
  id: 'Table 14',
  page: 24,
  caption:
    'Pretraining with multi-cell masking. The target cell is always masked; other cells are masked with probability P(mask) and contribute to the loss. NP = no pretraining.',
  cols: ['p0', 'p02', 'p04', 'np'],
  clf: {
    rows: [
      { task: 'amazon/item-churn', zs: [70.2, 70.5, 72.1], ft: [83.4, 83.0, 82.9, 83.2] },
      { task: 'amazon/user-churn', zs: [63.9, 62.6, 62.9], ft: [70.8, 70.7, 70.6, 70.5] },
      { task: 'avito/user-clicks', zs: [59.5, 61.5, 61.1], ft: [59.0, 62.3, 62.2, 63.6] },
      { task: 'avito/user-visits', zs: [61.8, 63.0, 63.2], ft: [65.2, 65.5, 65.5, 65.0] },
      { task: 'f1/driver-dnf', zs: [82.0, 80.9, 76.7], ft: [84.2, 81.7, 77.7, 78.7] },
      { task: 'f1/driver-top3', zs: [89.1, 89.8, 87.6], ft: [91.9, 91.0, 91.2, 82.7] },
      { task: 'hm/user-churn', zs: [62.8, 60.5, 62.3], ft: [70.5, 69.9, 69.9, 69.9] },
      { task: 'stack/user-badge', zs: [80.0, 79.1, 77.9], ft: [88.7, 88.3, 88.7, 88.5] },
      { task: 'stack/user-engagement', zs: [77.1, 75.0, 73.6], ft: [90.2, 90.1, 90.0, 90.0] },
      { task: 'trial/study-outcome', zs: [54.5, 55.1, 55.2], ft: [68.2, 70.2, 68.2, 68.6] },
    ],
    mean: { zs: [70.1, 69.8, 69.3], ft: [77.2, 77.3, 76.7, 76.1] },
  },
  reg: {
    rows: [
      { task: 'amazon/item-ltv', zs: [33.2, 9.3, 13.2], ft: [36.8, 37.0, 30.9, 31.5] },
      { task: 'amazon/user-ltv', zs: [36.4, 25.8, 16.4], ft: [47.4, 49.0, 49.6, 47.9] },
      { task: 'avito/ad-ctr', zs: [4.5, 8.0, 10.8], ft: [4.5, 3.6, 1.9, -1.5] },
      { task: 'f1/driver-position', zs: [54.7, 46.8, 42.4], ft: [51.6, 50.1, 47.7, 33.7] },
      { task: 'hm/item-sales', zs: [14.0, 10.0, 6.2], ft: [39.0, 50.7, 53.5, 45.7] },
      { task: 'stack/post-votes', zs: [32.4, 33.5, 32.3], ft: [36.5, 39.9, 38.5, 37.1] },
      { task: 'trial/site-success', zs: [5.2, 1.4, 3.0], ft: [6.4, 7.1, 6.6, -8.8] },
      { task: 'trial/study-adverse', zs: [2.1, 0.5, -1.8], ft: [43.4, 47.3, 46.8, 41.3] },
    ],
    mean: { zs: [22.8, 16.9, 15.3], ft: [33.2, 35.6, 34.4, 28.4] },
  },
};

/** Table 15 (p.24): order of the relational attention layers, zero-shot only. C/F/N = column/feature/neighbor. */
export const TABLE15: AblationTable = {
  id: 'Table 15',
  page: 24,
  caption:
    'Different orders of the relational attention layers (zero-shot). The letters give the order in which column (C), feature (F) and neighbor (N) attention are applied. Parallel: the three are applied in parallel and their outputs summed.',
  cols: ['CFN', 'CNF', 'FCN', 'FNC', 'NCF', 'NFC', 'Parallel'],
  clf: {
    rows: [
      { task: 'amazon/item-churn', zs: [70.1, 72.4, 70.8, 69.4, 71.1, 70.9, 70.5] },
      { task: 'amazon/user-churn', zs: [63.2, 64.3, 64.2, 63.7, 63.9, 64.1, 63.8] },
      { task: 'avito/user-clicks', zs: [58.7, 60.9, 59.6, 61.5, 58.8, 59.3, 62.3] },
      { task: 'avito/user-visits', zs: [61.6, 62.1, 61.3, 61.5, 61.1, 62.2, 62.2] },
      { task: 'f1/driver-dnf', zs: [79.6, 81.5, 81.4, 80.7, 81.2, 80.0, 79.9] },
      { task: 'f1/driver-top3', zs: [87.9, 88.9, 88.1, 88.1, 90.0, 88.6, 86.8] },
      { task: 'hm/user-churn', zs: [63.2, 62.5, 63.4, 63.3, 63.6, 63.5, 62.5] },
      { task: 'stack/user-badge', zs: [79.6, 79.7, 81.2, 79.3, 79.8, 79.6, 80.5] },
      { task: 'stack/user-engagement', zs: [80.5, 78.5, 74.3, 76.3, 80.9, 73.3, 77.0] },
      { task: 'trial/study-outcome', zs: [55.8, 55.3, 55.5, 54.3, 51.6, 53.4, 47.5] },
    ],
    mean: { zs: [70.0, 70.6, 70.0, 69.8, 70.2, 69.5, 69.3] },
  },
  reg: {
    rows: [
      { task: 'amazon/item-ltv', zs: [31.6, 30.7, 30.9, 29.9, 30.0, 30.9, 32.2] },
      { task: 'amazon/user-ltv', zs: [35.6, 35.1, 38.3, 37.3, 34.5, 35.1, 38.1] },
      { task: 'avito/ad-ctr', zs: [5.7, 5.7, 6.5, 4.9, 3.4, 4.7, 4.9] },
      { task: 'f1/driver-position', zs: [54.8, 52.8, 57.2, 53.0, 55.8, 55.6, 53.9] },
      { task: 'hm/item-sales', zs: [17.5, 16.5, 14.6, 16.5, 18.5, 15.9, 18.8] },
      { task: 'stack/post-votes', zs: [32.4, 32.1, 32.2, 31.5, 29.5, 31.1, 34.6] },
      { task: 'trial/site-success', zs: [3.9, 4.5, 3.3, 5.2, 4.2, 8.0, 4.0] },
      { task: 'trial/study-adverse', zs: [2.1, 2.7, 2.7, 1.5, 1.2, 2.5, 2.3] },
    ],
    mean: { zs: [23.0, 22.5, 23.2, 22.5, 22.1, 23.0, 23.6] },
  },
};

// ─────────────────────────────────────────────────────────────────────────────
// Table 12 (p.23): in-context labels per 1024-cell context (mean ± sd, rounded to integers)
// ─────────────────────────────────────────────────────────────────────────────

export interface LabelCountRow {
  db: string;
  task: string;
  kind: 'clf' | 'reg';
  self: [number, number];
  other: [number, number];
  unique: [number, number];
}

const LC = (
  db: string,
  task: string,
  kind: 'clf' | 'reg',
  self: [number, number],
  other: [number, number],
  unique: [number, number],
): LabelCountRow => ({
  db,
  task,
  kind,
  self,
  other,
  unique,
});

/** Row order as printed (it differs from Tables 1/2). */
export const TABLE12: LabelCountRow[] = [
  LC('rel-amazon', 'user-churn', 'clf', [4, 4], [1, 5], [2, 3]),
  LC('rel-hm', 'user-churn', 'clf', [6, 4], [0, 0], [1, 0]),
  LC('rel-stack', 'user-badge', 'clf', [7, 6], [3, 6], [1, 1]),
  LC('rel-amazon', 'item-churn', 'clf', [8, 7], [2, 6], [2, 3]),
  LC('rel-stack', 'user-engagement', 'clf', [16, 10], [10, 10], [3, 1]),
  LC('rel-avito', 'user-visits', 'clf', [2, 2], [0, 0], [1, 0]),
  LC('rel-avito', 'user-clicks', 'clf', [1, 1], [0, 0], [0, 1]),
  LC('rel-trial', 'study-outcome', 'clf', [0, 0], [0, 0], [0, 0]),
  LC('rel-f1', 'driver-dnf', 'clf', [19, 14], [0, 0], [1, 0]),
  LC('rel-f1', 'driver-top3', 'clf', [17, 11], [0, 0], [1, 0]),
  LC('rel-hm', 'item-sales', 'reg', [39, 13], [0, 3], [1, 1]),
  LC('rel-amazon', 'user-ltv', 'reg', [4, 4], [1, 5], [2, 3]),
  LC('rel-amazon', 'item-ltv', 'reg', [9, 8], [2, 6], [2, 3]),
  LC('rel-stack', 'post-votes', 'reg', [16, 10], [4, 14], [2, 2]),
  LC('rel-trial', 'site-success', 'reg', [1, 2], [0, 1], [1, 1]),
  LC('rel-trial', 'study-adverse', 'reg', [0, 0], [0, 0], [0, 0]),
  LC('rel-f1', 'driver-position', 'reg', [14, 10], [0, 0], [1, 0]),
  LC('rel-avito', 'ad-ctr', 'reg', [1, 1], [0, 0], [0, 1]),
];

// ─────────────────────────────────────────────────────────────────────────────
// Figures and prose numbers
// ─────────────────────────────────────────────────────────────────────────────

/** Fig. 3 (p.7): the zero-shot starting points printed on the plot (mean over 5 clf / 4 reg tasks). */
export const FIG3_POINTS = {
  clf: { rtPretrained: 70.8, griffinPretrained: 68.5 },
  reg: { rtPretrained: 29.0, griffinPretrained: 17.7 },
} as const;

/** Fig. 4 (p.8-9): the settings and the stated summary. The figure prints no table. */
export const FIG4 = {
  rtContexts: [256, 512, 768, 1024],
  griffinFanoutHops: ['10:1', '10:2', '20:2'],
  stated: { aurocDrop: 4, r2Drop: 7, lessTime: 25, lessMemory: 20 },
} as const;

/** p.2 §1: "90.3% of full supervised learning AUROC … 93.1% with continued pretraining … Gemma3-27B 83.7%". */
export const STATED_RATIOS = { rtNo: 90.3, rtYes: 93.1, gemma27b: 83.7 } as const;

/** App. C.2 (p.17): the largest exact column-name Jaccard overlap between two RelBench DBs (amazon–hm). */
export const MAX_JACCARD = 0.043;

// ─────────────────────────────────────────────────────────────────────────────
// Provenance (spec §13 #1, which compares the arXiv v1/v2/v3 PDFs)
// ─────────────────────────────────────────────────────────────────────────────

/** arXiv v1's Table 1 RT (No) column, as quoted by spec §13 #1 (CLF_ORDER). v1 mean: 70.1. */
export const V1_TABLE1_RTNO = [70.2, 63.9, 59.5, 61.8, 82.0, 89.1, 62.8, 80.0, 77.1, 54.5] as const;
/** arXiv v1/v2's Table 2 RT (No) column, as quoted by spec §13 #1 (REG_ORDER). Mean: 22.8. */
export const V1_TABLE2_RTNO = [33.2, 36.4, 4.5, 54.7, 14.0, 32.4, 5.2, 2.1] as const;
/** v1 → v2 change of Table 1's RT means (spec §13 #1). */
export const TABLE1_V1_MEANS = { rtNo: 70.1, rtYes: 72.9 } as const;
