/**
 * The few published numbers this chapter quotes, copied from the paper PDF (arXiv 2510.06377v3,
 * ICLR 2026) and checked against it page by page. They are context for our run, never inputs to it.
 */

/** Table 1 (p.8): mean zero-shot test AUROC (%) over 10 binary tasks. */
export const T1_MEAN = { entityMean: 66.7, griffinNo: 64.0, rtNo: 69.7, griffinYes: 67.3, rtYes: 71.9 } as const;

/** Table 2 (p.8): mean zero-shot test R² (%) over 8 regression tasks. */
export const T2_MEAN = { entityMean: 19.3, griffinNo: 8.9, rtNo: 22.7, griffinYes: 10.1, rtYes: 26.4 } as const;

/** Table 10 (p.21): supervised fine-tuning, mean over the same 10 / 8 tasks. */
export const T10_MEAN = { auroc: { rtPretrained: 77.2, rtUntrained: 76.1 }, r2: { rtPretrained: 33.2, rtUntrained: 28.4 } } as const;

/** Fig. 3 (p.7): the zero-shot (first) point of each curve, mean over 5 clf / 4 reg tasks that do not overfit. */
export const FIG3_ZERO_SHOT = { auroc: { rt: 70.8, griffin: 68.5 }, r2: { rt: 29.0, griffin: 17.7 } } as const;

/** Per-database task counts in rt/tasks.py (forecast + autocomplete), spec §9.1. */
export const RELBENCH_DBS: { name: string; forecast: number; autocomplete: number; evaluated: boolean }[] = [
  { name: 'rel-amazon', forecast: 4, autocomplete: 2, evaluated: true },
  { name: 'rel-avito', forecast: 3, autocomplete: 1, evaluated: true },
  { name: 'rel-event', forecast: 3, autocomplete: 2, evaluated: false },
  { name: 'rel-f1', forecast: 3, autocomplete: 4, evaluated: true },
  { name: 'rel-hm', forecast: 2, autocomplete: 1, evaluated: true },
  { name: 'rel-stack', forecast: 3, autocomplete: 1, evaluated: true },
  { name: 'rel-trial', forecast: 3, autocomplete: 4, evaluated: true },
];

/** Paper / scripts: OneCycleLR defaults (main.py:197-204, spec §9.2). */
export const ONECYCLE = { maxLr: 1e-3, pctStart: 0.2, divFactor: 25, finalDivFactor: 1e4, beta1Max: 0.95, beta1Min: 0.85 } as const;
