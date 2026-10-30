/**
 * Glossary of RT terms used by <Term id="…"> tooltips and the results
 * chapter's glossary list. Every definition is written from
 * research/RT_TECHNICAL_SPEC.md (paper-era code @2a899c8) and cites it.
 * Do not add behavior the spec does not state.
 */

export type CiteKind = 'spec' | 'paper' | 'code' | 'url';

export interface GlossaryEntry {
  /** Display name. */
  term: string;
  /** One or two precise sentences. Plain text. */
  def: string;
  /** Where it is documented. */
  cite: { kind: CiteKind; ref: string }[];
  /** Related term ids. */
  see?: string[];
}

export const GLOSSARY = {
  'relational-database': {
    term: 'Relational database',
    def: 'A set of tables whose rows are linked through primary and foreign keys. RT reads it as a graph of rows, each row holding feature cells.',
    cite: [{ kind: 'paper', ref: 'p.3 §2.1' }, { kind: 'spec', ref: '§1.1' }],
  },
  'feature-column': {
    term: 'Feature column',
    def: 'Any column that is not a primary or foreign key. RT turns its non-missing cells into tokens (number, text, datetime or boolean).',
    cite: [{ kind: 'spec', ref: '§1.1' }, { kind: 'paper', ref: 'p.16 App. A' }],
    see: ['cell-token', 'sem-type'],
  },
  'primary-key': {
    term: 'Primary key (PK)',
    def: 'The column that identifies a row. It produces no token and no link. In RelBench-style data its value is the row’s position, 0 to n − 1.',
    cite: [{ kind: 'spec', ref: '§1.2' }, { kind: 'code', ref: 'rustler/src/pre.rs:439-442' }],
  },
  'foreign-key': {
    term: 'Foreign key (FK)',
    def: 'A column holding the position of a parent row in another table. It produces no token, only links: the child records its parent (F→P), and the parent gets a link back to the child (P→F).',
    cite: [{ kind: 'spec', ref: '§1.2' }, { kind: 'code', ref: 'rustler/src/pre.rs:444-533' }],
    see: ['f2p-link', 'p2f-link'],
  },
  'f2p-link': {
    term: 'F→P link',
    def: 'The link from a row that holds a foreign key to the row it references, its parent. The sampler always follows these links first, with no time filter and no width limit.',
    cite: [{ kind: 'spec', ref: '§1.3' }, { kind: 'spec', ref: '§4.4' }],
  },
  'p2f-link': {
    term: 'P→F link',
    def: 'The reverse of an F→P link: from a parent row to each child row that references it. A row can have a great many children, so the sampler filters them by time and keeps at most w of them.',
    cite: [{ kind: 'spec', ref: '§1.3' }, { kind: 'spec', ref: '§4.7' }],
  },
  'time-column': {
    term: 'Time column',
    def: 'The one datetime column a table may declare. Its cells are ordinary datetime tokens, and its value also gives the row the timestamp that the sampler reads.',
    cite: [{ kind: 'spec', ref: '§1.4' }],
    see: ['node-timestamp'],
  },
  'node-timestamp': {
    term: 'Row timestamp',
    def: 'The row’s time-column value in whole seconds. Only the sampler reads it, for its temporal filter.',
    cite: [{ kind: 'spec', ref: '§1.4' }, { kind: 'code', ref: 'rustler/src/pre.rs:481-493' }],
  },
  'global-node-index': {
    term: 'Global node index',
    def: 'One integer id for every row of a database, task-table rows included: the offset of the row’s table plus its position in that table. Database tables come first, in alphabetical order, then the task splits (test, train, val).',
    cite: [{ kind: 'spec', ref: '§1.5' }, { kind: 'code', ref: 'rustler/src/pre.rs:79-92' }],
  },
  'cell-token': {
    term: 'Cell token',
    def: 'One non-missing feature cell, represented by its value, its column name and its table name. RT’s input sequence is made of these, so a row becomes one token per feature value it holds.',
    cite: [{ kind: 'paper', ref: 'p.4 §3.1' }, { kind: 'spec', ref: '§1.1' }],
  },
  'sem-type': {
    term: 'Semantic type',
    def: 'Number (0), Text (1), DateTime (2) or Boolean (3), read off the column’s storage type: integers and floats are numbers, strings are text. Each type has its own encoder, decoder and mask vector.',
    cite: [{ kind: 'spec', ref: '§3.2' }, { kind: 'code', ref: 'rustler/src/common.rs:38-44' }],
  },
  'z-score': {
    term: 'z-score normalization',
    def: 'Numbers and booleans are standardized per column as (v − μ)/σ. Datetimes share one mean and one standard deviation across the whole database. Task labels use statistics from the Train split, database columns use all their rows.',
    cite: [{ kind: 'spec', ref: '§3.3' }, { kind: 'spec', ref: '§3.4' }],
  },
  'column-phrase': {
    term: 'Column phrase',
    def: 'The string “<column> of <table>”, e.g. “age of customers”. Its MiniLM embedding is how a token learns which column and table it belongs to.',
    cite: [{ kind: 'spec', ref: '§3.6' }, { kind: 'code', ref: 'rustler/src/pre.rs:432-437' }],
    see: ['text-encoder'],
  },
  'text-encoder': {
    term: 'Text encoder',
    def: 'The frozen sentence-transformers/all-MiniLM-L12-v2 model (384-d, mean pooled, unit norm). It embeds column phrases, table names and text cell values once, ahead of time.',
    cite: [{ kind: 'spec', ref: '§3.6' }, { kind: 'code', ref: 'rt/embed.py:14-31' }],
  },
  'task-table': {
    term: 'Task table',
    def: 'A table that holds the prediction task: a timestamp, a foreign key to the entity and the label. It is attached to the database as an ordinary table, with Train, Val and Test splits.',
    cite: [{ kind: 'paper', ref: 'p.4 §2.2' }, { kind: 'spec', ref: '§2.1' }],
    see: ['task-table-prompting'],
  },
  'task-table-prompting': {
    term: 'Task-table prompting',
    def: 'Giving RT a task by attaching its task table to the database. Its rows are the seeds, and the same task’s past rows, reached through the entity, bring visible labels into the context.',
    cite: [{ kind: 'paper', ref: 'p.4 §3.1' }, { kind: 'spec', ref: '§2.1' }],
  },
  'forecasting-task': {
    term: 'Forecasting task',
    def: 'Predict an entity’s label over a future window (t, t+Δ] from data up to t, e.g. whether a customer places no order in the next 30 days.',
    cite: [{ kind: 'spec', ref: '§2.2' }],
  },
  'autocomplete-task': {
    term: 'Autocomplete task',
    def: 'Predict a masked cell of an existing database column. The seed is a database row and no task rows enter the context. Columns that would give the answer away are dropped while sampling.',
    cite: [{ kind: 'paper', ref: 'p.19 App. D' }, { kind: 'spec', ref: '§2.4' }],
    see: ['leakage-columns'],
  },
  'leakage-columns': {
    term: 'Leakage columns',
    def: 'Columns that would give away the answer of an autocomplete task (columns_to_drop). Their cells are skipped for the seed row and for rows with the seed’s timestamp.',
    cite: [{ kind: 'spec', ref: '§4.9' }, { kind: 'code', ref: 'rustler/src/fly.rs:457-461' }],
  },
  'seed-row': {
    term: 'Seed row',
    def: 'The row whose target cell is predicted. Its cells come first in the context, and the BFS starts from it.',
    cite: [{ kind: 'spec', ref: '§2.3' }, { kind: 'spec', ref: '§4.12' }],
  },
  'target-cell': {
    term: 'Masked target cell',
    def: 'The single masked token of a context: the seed row’s label cell. It keeps its column name and type, but its value embedding is replaced by the learned mask vector.',
    cite: [{ kind: 'spec', ref: '§4.10' }, { kind: 'spec', ref: '§5.3' }],
    see: ['mask-vector'],
  },
  'mask-vector': {
    term: 'Mask vector mₜ',
    def: 'A learned vector for each sem type. It takes the place of the masked cell’s value term and is added as it is, with no RMSNorm.',
    cite: [{ kind: 'spec', ref: '§5.2' }, { kind: 'code', ref: 'rt/model.py:141-146' }],
  },
  'in-context-label': {
    term: 'In-context label',
    def: 'A visible label cell in the context: another row of the same task table (the target entity’s past rows or other entities’ rows), or for autocomplete, other rows of the target column.',
    cite: [{ kind: 'spec', ref: '§2.5' }, { kind: 'paper', ref: 'p.2' }],
    see: ['self-label', 'same-horizon'],
  },
  'self-label': {
    term: 'Self label',
    def: 'An in-context label of the target entity itself, from its past task rows. Removing them causes the largest zero-shot drop of any ablation in the paper.',
    cite: [{ kind: 'spec', ref: '§11.2' }, { kind: 'paper', ref: 'p.9 Table 4' }],
  },
  'same-horizon': {
    term: 'Same-horizon rows',
    def: 'Other entities’ task rows with the same timestamp as the seed. The sampler skips only rows later than the seed, so these rows stay and their labels can appear in the context.',
    cite: [{ kind: 'spec', ref: '§2.5' }, { kind: 'code', ref: 'rustler/src/fly.rs:422' }],
  },
  'context-window': {
    term: 'Context window (L)',
    def: 'The L cell tokens the sampler gathers around the seed (L = 1024 in the paper). Sampling stops at exactly L tokens, even in the middle of a row, and pads the rest if it runs out of rows first.',
    cite: [{ kind: 'spec', ref: '§4.1' }, { kind: 'spec', ref: '§4.11' }],
  },
  'bfs-width': {
    term: 'BFS width bound (w)',
    def: 'At most w database children are kept for each visited row, drawn at random after the temporal filter. Task-table children are never subsampled. The paper uses w = 256.',
    cite: [{ kind: 'spec', ref: '§4.7' }, { kind: 'code', ref: 'rustler/src/fly.rs:426-452' }],
  },
  'depth-label': {
    term: 'Depth label',
    def: 'The number of hops from the seed to a row waiting in the sampler’s frontier (+1 per F→P or P→F hop). The next child row is drawn at random from the shallowest non-empty depth.',
    cite: [{ kind: 'spec', ref: '§4.5' }],
  },
  'temporal-filter': {
    term: 'Temporal filter',
    def: 'A P→F child row is skipped when its timestamp is later than the seed’s. Equal timestamps pass, and parents are never filtered.',
    cite: [{ kind: 'spec', ref: '§4.6' }, { kind: 'code', ref: 'rustler/src/fly.rs:421-424' }],
  },
  padding: {
    term: 'Padding token',
    def: 'A filler position, flagged by is_padding, used when fewer than L cells were sampled. Its input is the zero vector, and every mask excludes it.',
    cite: [{ kind: 'spec', ref: '§4.11' }, { kind: 'spec', ref: '§6.2' }],
  },
  'relational-attention': {
    term: 'Relational attention',
    def: 'Multi-head attention restricted by a boolean mask built from the tokens’ table, column and row ids. The paper’s three relational layers are column, feature and neighbor attention; each block adds a full attention layer after them.',
    cite: [{ kind: 'paper', ref: 'p.5 §3.2' }, { kind: 'spec', ref: '§6.2' }],
  },
  'column-attention': {
    term: 'Column attention',
    def: 'A token attends to every token of the same column of the same table. The masked label sees every visible label cell of its task.',
    cite: [{ kind: 'spec', ref: '§6.2' }, { kind: 'code', ref: 'rt/model.py:181-183' }],
  },
  'feature-attention': {
    term: 'Feature attention',
    def: 'A token attends to its own row and to its parent rows (the rows its FKs point to): row-wise attention after joining each table with its parents.',
    cite: [{ kind: 'spec', ref: '§6.2' }, { kind: 'paper', ref: 'p.5 §3.2' }],
  },
  'neighbor-attention': {
    term: 'Neighbor attention',
    def: 'A parent’s tokens attend to the tokens of its child rows in the context, much like message passing in a graph neural network. A row with no children in the context gets a zero update.',
    cite: [{ kind: 'spec', ref: '§6.2' }, { kind: 'spec', ref: '§6.1' }],
  },
  'full-attention': {
    term: 'Full attention',
    def: 'Every non-padding token attends to every other one. The paper also calls it “global” attention.',
    cite: [{ kind: 'spec', ref: '§6.2' }, { kind: 'code', ref: 'rt/model.py:186-191' }],
  },
  rmsnorm: {
    term: 'RMSNorm',
    def: 'y = x / √(mean(x²) + ε) · γ, with ε = 2⁻⁷ in RT. There is one before every sublayer, one after each input projection and one at the very end.',
    cite: [{ kind: 'spec', ref: '§7.2' }],
  },
  'pre-norm': {
    term: 'Pre-norm residual',
    def: 'Each sublayer computes x + f(RMSNorm(x)): it reads a normalized copy of x, and its raw output is added back to x.',
    cite: [{ kind: 'spec', ref: '§7.1' }, { kind: 'code', ref: 'rt/model.py:83-85' }],
  },
  swiglu: {
    term: 'SwiGLU feed-forward',
    def: 'FFN(h) = W₂(SiLU(W₁h) ⊙ W₃h) with no biases, 4× hidden width (256 → 1024 → 256 in the paper).',
    cite: [{ kind: 'spec', ref: '§7.1' }, { kind: 'code', ref: 'rt/model.py:50-59' }],
  },
  'permutation-invariance': {
    term: 'Permutation invariance',
    def: 'RT has no positional encodings. Shuffling the tokens reorders the outputs the same way, so the prediction at the target stays the same.',
    cite: [{ kind: 'spec', ref: '§6.5' }, { kind: 'paper', ref: 'p.5 §3.1' }],
    see: ['permutation-equivariance'],
  },
  'permutation-equivariance': {
    term: 'Permutation equivariance',
    def: 'Reordering the input tokens reorders the outputs in the same way: each token’s final hidden state moves with it, unchanged. RT has this property because each input vector depends only on the token’s own fields, and the masks depend only on ids.',
    cite: [{ kind: 'spec', ref: '§6.5' }, { kind: 'code', ref: 'rt/model.py:165-224' }],
    see: ['permutation-invariance'],
  },
  'zero-shot': {
    term: 'Zero-shot',
    def: 'Predicting on a database (or task) the model was never trained on, with no weight updates. The new data enters only through the context window. The checkpoint is still chosen on the target task’s validation split.',
    cite: [{ kind: 'paper', ref: 'p.4 §2.3' }, { kind: 'spec', ref: '§9.4' }],
  },
  'leave-one-db-out': {
    term: 'Leave-one-DB-out pretraining',
    def: 'Pretrain on all tasks of every other database, then evaluate zero-shot on the one held out. For each target task, the checkpoint with the best validation score is kept.',
    cite: [{ kind: 'spec', ref: '§9.1' }, { kind: 'paper', ref: 'p.6 §4.1' }],
  },
  'entity-mean': {
    term: 'EntityMean',
    def: 'Baseline: the mean of the target entity’s past labels, falling back to the global training mean of the target column.',
    cite: [{ kind: 'paper', ref: 'p.23 App. I.1' }],
  },
  huber: {
    term: 'Huber loss',
    def: 'The loss for number targets, computed in z-space with δ = 1: 0.5d² if |d| < 1, else |d| − 0.5.',
    cite: [{ kind: 'spec', ref: '§8.2' }],
  },
  bce: {
    term: 'BCE with logits',
    def: 'The loss for boolean targets, against 1{z > 0}, where z is the z-scored label (z > 0 exactly when the label is true).',
    cite: [{ kind: 'spec', ref: '§8.2' }, { kind: 'spec', ref: '§3.3' }],
  },
  auroc: {
    term: 'AUROC',
    def: 'Area under the ROC curve, the metric for binary tasks. RT computes it on raw logits; sigmoid probabilities would give the same value.',
    cite: [{ kind: 'spec', ref: '§8.5' }],
  },
  r2: {
    term: 'R²',
    def: 'Coefficient of determination, the metric for regression tasks. Computed in z-space, it equals R² in raw units. Always predicting the global mean scores 0.',
    cite: [{ kind: 'spec', ref: '§8.5' }],
  },
} satisfies Record<string, GlossaryEntry>;

export type GlossaryId = keyof typeof GLOSSARY;

export function getGlossaryEntry(id: string): GlossaryEntry | undefined {
  return (GLOSSARY as Record<string, GlossaryEntry>)[id];
}
