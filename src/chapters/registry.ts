/**
 * Chapter registry (BUILD_PLAN §6). The single source of truth for chapter
 * order, numbers, titles, one-line summaries and lazy loaders. The sidebar,
 * the landing page's course map, ChapterHeader and ChapterNav all read it.
 *
 * Routes are `/#/<slug>`. Each chapter lives in `src/chapters/<slug>/index.tsx`
 * and default-exports its page component; it is code-split via `load()`.
 */
import { lazy, type ComponentType, type LazyExoticComponent } from 'react';

export type ChapterPart = 'foundations' | 'context' | 'model' | 'learning';

export const PARTS: { id: ChapterPart; title: string; blurb: string }[] = [
  { id: 'foundations', title: 'Data', blurb: 'The database, the task, and how rows become cells.' },
  { id: 'context', title: 'Context', blurb: 'Sampling a context window and turning it into tokens.' },
  { id: 'model', title: 'Model', blurb: 'Embedding, relational attention, blocks and decoders.' },
  { id: 'learning', title: 'Learning & results', blurb: 'Training regimes, live experiments and the paper’s findings.' },
];

export interface ChapterMeta {
  slug: string;
  /** 1-based chapter number. */
  number: number;
  /** Full title (ChapterHeader). */
  title: string;
  /** Short title for the sidebar. */
  short: string;
  /** One-line summary (course map, NotFound suggestions, meta). */
  summary: string;
  part: ChapterPart;
  /** Planned topics (BUILD_PLAN §6), shown by the placeholder page. */
  outline: string[];
  load: () => Promise<{ default: ComponentType }>;
}

export const CHAPTERS: ChapterMeta[] = [
  {
    slug: 'overview',
    number: 1,
    title: 'The big picture',
    short: 'The big picture',
    summary: 'Why relational foundation models are hard, and RT’s three ideas in one end-to-end pass.',
    part: 'foundations',
    outline: [
      'Why relational foundation models are hard',
      'RT’s three ideas: cell tokens, task-table prompting, relational attention',
      'An animated end-to-end pipeline on the running example',
      'Paper scale vs mini scale',
      'A map of the course',
    ],
    load: () => import('./overview/index.tsx'),
  },
  {
    slug: 'database',
    number: 2,
    title: 'The relational database',
    short: 'The relational database',
    summary: 'ShopDB’s tables, primary and foreign keys, time columns and the entity graph.',
    part: 'foundations',
    outline: [
      'The ShopDB schema',
      'Primary keys, foreign keys, feature vs key columns',
      'F→P and P→F links',
      'Temporal rows and time columns',
      'The entity-graph view',
    ],
    load: () => import('./database/index.tsx'),
  },
  {
    slug: 'tasks',
    number: 3,
    title: 'Predictive tasks and task tables',
    short: 'Tasks and task tables',
    summary: 'Forecasting and autocomplete as masked-cell prediction; task tables, windows and splits.',
    part: 'foundations',
    outline: [
      'Masked-token prediction as the universal task',
      'Building the customer-churn task table: timestamps, windows, labels',
      'Train / val / test splits on a timeline',
      'Regression tasks and autocomplete tasks with leakage columns',
      'Task-table prompting: the task is just another table',
    ],
    load: () => import('./tasks/index.tsx'),
  },
  {
    slug: 'preprocessing',
    number: 4,
    title: 'From rows to cells',
    short: 'From rows to cells',
    summary: 'Global node indexing, which cells become tokens, sem types and z-scoring.',
    part: 'foundations',
    outline: [
      'Global node indexing and table offsets',
      'Which cells become tokens (keys and nulls do not)',
      'Semantic types from physical dtypes',
      'Number, boolean and datetime normalization',
      'Pick a cell, see its encoding',
    ],
    load: () => import('./preprocessing/index.tsx'),
  },
  {
    slug: 'semantics',
    number: 5,
    title: 'Text and schema semantics',
    short: 'Text and schema semantics',
    summary: 'The frozen MiniLM encoder, “<col> of <table>” phrases and why names matter.',
    part: 'foundations',
    outline: [
      'The frozen all-MiniLM-L12-v2 encoder',
      'Column phrases “<col> of <table>”',
      'Cosine similarity and a 2-D map of column phrases',
      'Text cell values are embedded raw',
      'Why names matter: the shuffled-names ablation',
    ],
    load: () => import('./semantics/index.tsx'),
  },
  {
    slug: 'sampling',
    number: 6,
    title: 'Building the context window',
    short: 'The context window',
    summary: 'Algorithm 1 step by step: the bounded-width, temporal BFS that gathers L cells.',
    part: 'context',
    outline: [
      'F→P parents first (LIFO stack), then P→F depth buckets',
      'Temporal filter and the width bound w',
      'Visited checks, cell emission and the hard stop at L',
      'In-context labels: self vs other entities',
      'Algorithm 1 vs the released code',
    ],
    load: () => import('./sampling/index.tsx'),
  },
  {
    slug: 'tokens',
    number: 7,
    title: 'Cells become tokens',
    short: 'Cells become tokens',
    summary: 'The batch dictionary per token, the single masked target and permutation invariance.',
    part: 'context',
    outline: [
      'The batch dictionary, field by field',
      'Parent lists (f2p_nbr_idxs)',
      'The single masked target, and why its stored label cannot leak',
      'Shuffle the tokens: identical prediction',
    ],
    load: () => import('./tokens/index.tsx'),
  },
  {
    slug: 'embedding',
    number: 8,
    title: 'The token embedding',
    short: 'The token embedding',
    summary: 'Column-name term plus value term, each through its own RMSNorm; the mask vector.',
    part: 'model',
    outline: [
      'Column-name term and value term',
      'The learned mask vector for the target',
      'Move a value, watch its embedding move',
      'RMSNorm and its epsilon',
      'Paper vs code: biases and norms',
    ],
    load: () => import('./embedding/index.tsx'),
  },
  {
    slug: 'masks',
    number: 9,
    title: 'Relational attention masks',
    short: 'Relational attention masks',
    summary: 'Column, feature, neighbor and full masks on the live context: who can see whom.',
    part: 'model',
    outline: [
      'The four masks as matrices',
      '“Who can I see?”: hover any token',
      'Definitions in paper notation and in code',
      'Padding, empty neighbor rows and zero updates',
      'What information each mask lets flow',
    ],
    load: () => import('./masks/index.tsx'),
  },
  {
    slug: 'attention',
    number: 10,
    title: 'Inside one attention sublayer',
    short: 'Inside one attention sublayer',
    summary: 'RMSNorm, Q/K/V, masked softmax, heads, output projection and the residual add.',
    part: 'model',
    outline: [
      'Pick a block, layer, head and query token',
      'Scores, masking, softmax and the weighted sum',
      'Concatenating heads and the output projection',
      'Attention maps for every layer and head',
    ],
    load: () => import('./attention/index.tsx'),
  },
  {
    slug: 'block',
    number: 11,
    title: 'The RT block and the residual stream',
    short: 'The block & residual stream',
    summary: 'Four pre-norm attentions and a SwiGLU per block; how information reaches the target.',
    part: 'model',
    outline: [
      'col → feat → nbr → full → SwiGLU, each pre-norm residual',
      'SwiGLU internals and the final RMSNorm',
      'The residual stream of the target token',
      'How information can reach the target (interpretation)',
      'Paper Algorithm 2 vs code',
    ],
    load: () => import('./block/index.tsx'),
  },
  {
    slug: 'decoding',
    number: 12,
    title: 'Decoders, loss and prediction',
    short: 'Decoders, loss, prediction',
    summary: 'Per-type decoders, logits and z-scores, Huber and BCE losses, AUROC and R².',
    part: 'model',
    outline: [
      'Four decoders at every position; only the target is read',
      'Binary: logit → probability. Regression: z-score → original units',
      'The loss, with numbers',
      'Metrics: AUROC on logits, R² in z-space',
    ],
    load: () => import('./decoding/index.tsx'),
  },
  {
    slug: 'training',
    number: 13,
    title: 'Pretraining, zero-shot, fine-tuning',
    short: 'Pretraining & fine-tuning',
    summary: 'Leave-one-DB-out pretraining, the OneCycle schedule, zero-shot results and fine-tuning.',
    part: 'learning',
    outline: [
      'Masked-cell pretraining, leave-one-database-out',
      'Training curves and the learning-rate schedule',
      'Best-validation checkpoint per task',
      'Zero-shot vs EntityMean; fine-tuning from pretrained vs random init',
    ],
    load: () => import('./training/index.tsx'),
  },
  {
    slug: 'playground',
    number: 14,
    title: 'Zero-shot playground and ablations',
    short: 'Playground & ablations',
    summary: 'Run any ShopDB seed live; edit labels, shuffle names, shrink L, switch layers off.',
    part: 'learning',
    outline: [
      'Pick any ShopDB seed and run live',
      'Edit or remove in-context labels',
      'Shuffle column names; change L',
      'Disable attention layers at inference (the paper retrains)',
    ],
    load: () => import('./playground/index.tsx'),
  },
  {
    slug: 'results',
    number: 15,
    title: 'What the paper found',
    short: 'What the paper found',
    summary: 'Interactive versions of the paper’s tables, the drivers of transfer, and limitations.',
    part: 'learning',
    outline: [
      'Headline zero-shot results',
      'Context and attention ablations',
      'The ranked drivers of zero-shot transfer',
      'Limitations, and every paper-vs-code discrepancy',
    ],
    load: () => import('./results/index.tsx'),
  },
  {
    slug: 'about',
    number: 16,
    title: 'How this simulation was built and verified',
    short: 'How this was built',
    summary: 'The pipeline, the parity tests against PyTorch and the official model, and what is scaled down.',
    part: 'learning',
    outline: [
      'The pipeline: built once offline, run live in your browser',
      'Verification: TypeScript == PyTorch port == official model.py, sampler cross-check, tests, the spec log',
      'What is scaled down, and what is different',
      'Sources and licenses',
      'Run it locally and rebuild the data',
    ],
    load: () => import('./about/index.tsx'),
  },
];

const BY_SLUG = new Map(CHAPTERS.map((c) => [c.slug, c]));

/** Chapter metadata by slug (undefined for unknown slugs). */
export function getChapter(slug: string | undefined | null): ChapterMeta | undefined {
  return slug ? BY_SLUG.get(slug) : undefined;
}

/** Previous / next chapters in course order. */
export function getNeighbors(slug: string): { prev?: ChapterMeta; next?: ChapterMeta } {
  const i = CHAPTERS.findIndex((c) => c.slug === slug);
  if (i < 0) return {};
  return { prev: CHAPTERS[i - 1], next: CHAPTERS[i + 1] };
}

const lazyCache = new Map<string, LazyExoticComponent<ComponentType>>();

/** Stable React.lazy component for a chapter (one per slug, so Suspense never remounts needlessly). */
export function getChapterComponent(slug: string): LazyExoticComponent<ComponentType> | undefined {
  const meta = BY_SLUG.get(slug);
  if (!meta) return undefined;
  let c = lazyCache.get(slug);
  if (!c) {
    c = lazy(meta.load);
    lazyCache.set(slug, c);
  }
  return c;
}

/** Warm a chapter's chunk (e.g. on hover of a nav link). Errors are ignored. */
export function prefetchChapter(slug: string): void {
  const meta = BY_SLUG.get(slug);
  if (meta) meta.load().catch(() => {});
}
