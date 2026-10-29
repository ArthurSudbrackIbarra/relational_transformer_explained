/**
 * The final quiz that unlocks the certificate: one question per stage of the course, each answered by
 * the chapters it points to (their text is the source of every answer and explanation). The reader
 * answers all of them, then submits; a pass needs PASS_MARK of the questions. After a failed attempt
 * the page names the chapters to review but not the right answers.
 */

export interface FinalQuestion {
  /** Stable id (kept when the wording changes). */
  id: string;
  /** Chapter slugs that teach the answer, shown as "review" links. */
  chapters: string[];
  prompt: string;
  /** Exactly four choices; the page shuffles them for every attempt. */
  choices: [string, string, string, string];
  /** Index of the right choice in `choices`. */
  answer: number;
  /** Why, shown in the review once the reader has passed. */
  why: string;
}

export const FINAL_QUIZ: FinalQuestion[] = [
  {
    id: 'cell-tokens',
    chapters: ['overview'],
    prompt: 'What does one token of the Relational Transformer stand for?',
    choices: [
      'One cell of the database: a value, with the names of its column and table',
      'One row of a table, with all of its columns',
      'One whole table, summarized as a single vector',
      'One word of a column name',
    ],
    answer: 0,
    why: 'RT represents each database cell as a token, the triple (value, column, table). A row with k non-missing feature cells becomes k tokens.',
  },
  {
    id: 'parent-link',
    chapters: ['database'],
    prompt: 'An order row holds the foreign key customer_id. What is the customer row to that order?',
    choices: [
      'Its parent: the order has an F→P link up to the customer',
      'Its child: the customer has an F→P link down to the order',
      'A row of the same table, linked by a shared primary key',
      'Nothing: foreign keys are deleted before the model sees the data',
    ],
    answer: 0,
    why: 'A link from a foreign key to the primary key it references is an F→P link. The row a foreign key points to is the parent; the rows that point to a row are its children (P→F). Foreign keys are not deleted: they become links, never tokens.',
  },
  {
    id: 'task-table',
    chapters: ['tasks'],
    prompt: 'How does RT receive a forecasting question such as “will this customer churn?”',
    choices: [
      'As a task table attached to the database, whose label cell is masked in the seed row',
      'As a natural-language prompt that describes the task',
      'As a separate output head, trained for that task alone',
      'As a SQL query that the model runs on the database',
    ],
    answer: 0,
    why: 'Task-table prompting: the labels live in a task table (a foreign key to the entity, a timestamp and the label) attached to the database. To the model it is an ordinary table; the seed row is special only because its label cell is the masked token.',
  },
  {
    id: 'which-cells',
    chapters: ['preprocessing'],
    prompt:
      'A row has a primary key, a foreign key, two feature columns with values and one feature column that is null. How many cell tokens can it contribute?',
    choices: [
      '2: the keys become a link or nothing, and the null produces nothing',
      '3: the null becomes a placeholder token',
      '4: every column except the primary key',
      '5: one token per column',
    ],
    answer: 0,
    why: 'The primary key produces nothing, a foreign key becomes a link to its parent row (f2p_nbr_idxs), and a missing value produces no cell, no placeholder and no imputation. Only the two present feature values become cells.',
  },
  {
    id: 'column-phrase',
    chapters: ['semantics'],
    prompt: 'How does a cell token tell the model which column of which table it belongs to?',
    choices: [
      'Through the phrase “<column> of <table>”, embedded by a frozen sentence encoder',
      'Through a learned embedding for each column id',
      'Through the column’s position in its table',
      'Through a description of the column written by the authors',
    ],
    answer: 0,
    why: 'The column and table names are joined into “<column_name> of <table_name>” and embedded by all-MiniLM-L12-v2, which stays frozen. The column ids only steer the attention masks. That is how names carry meaning to a database the model has never seen.',
  },
  {
    id: 'temporal-filter',
    chapters: ['sampling'],
    prompt: 'Which rows does the released sampler skip because of time?',
    choices: [
      'P→F child rows whose timestamp is later than the seed row’s',
      'Parent rows whose timestamp is later than the seed row’s',
      'Every row more than a year older than the seed row',
      'None: the sampler ignores timestamps',
    ],
    answer: 0,
    why: 'Parents are followed first and never filtered. A child row is skipped only when both timestamps exist and the child’s is later than the seed’s; equal timestamps, and rows without one, pass.',
  },
  {
    id: 'shuffle',
    chapters: ['tokens'],
    prompt: 'You shuffle the tokens of a context and run the model again. What happens to the prediction at the masked cell?',
    choices: [
      'It stays the same: RT has no positional encodings, and its masks compare ids, not positions',
      'It changes slightly, because the positional encodings change',
      'It changes completely, because the sampler’s order is part of the input',
      'The model cannot run until the tokens are sorted by table again',
    ],
    answer: 0,
    why: 'Each token’s input vector comes from its own fields only, and the four masks compare row, column, table and padding ids. Shuffling moves every output along with its token and leaves each one unchanged.',
  },
  {
    id: 'mask-vector',
    chapters: ['embedding', 'tokens'],
    prompt: 'How does the masked target cell enter the model without revealing its label?',
    choices: [
      'Its value term is replaced by a trainable mask vector for its type; its column term and ids stay',
      'The whole token is removed from the context',
      'Its value is set to zero and passed through the usual value encoder',
      'Its value is replaced by a random value from the same column',
    ],
    answer: 0,
    why: 'The masked token keeps its column phrase, its type and its row ids, so the model knows which cell to fill. Only its value term changes, to the learned mask vector of its type. The true label stays in the batch, where only the loss reads it.',
  },
  {
    id: 'feature-attention',
    chapters: ['masks'],
    prompt: 'Which attention layer limits a cell to the cells of its own row and of its parent rows?',
    choices: ['Feature attention', 'Column attention', 'Neighbor attention', 'Full attention'],
    answer: 0,
    why: 'Feature attention allows Row(k) = Row(q) or Row(k) ∈ OutLinks(q), so information flows from parent to child, like a join. Neighbor attention is the reverse (a parent reads its children), column attention reads the same column of the same table, and full attention reads every non-padding token.',
  },
  {
    id: 'blocked-key',
    chapters: ['attention', 'masks'],
    prompt: 'Inside an attention head, what happens to a key that the mask blocks for a query?',
    choices: [
      'Its score is set to −∞ before the softmax, so it gets exactly zero weight',
      'Its score is halved before the softmax',
      'Its value vector is swapped for the mask vector',
      'It is removed from the context for every later layer',
    ],
    answer: 0,
    why: 'The mask keeps a score where M[q, k] = 1 and writes −∞ elsewhere, and e^−∞ = 0. A blocked key gets exactly zero weight, and a query with no allowed key gets an update of exactly zero. Every layer applies its own mask to the whole context again.',
  },
  {
    id: 'block-order',
    chapters: ['block'],
    prompt: 'What does one RT block run, in order?',
    choices: [
      'Column, feature, neighbor and full attention, then a SwiGLU feed-forward network, each adding its output to the residual stream',
      'One full attention layer, then a feed-forward network',
      'Full attention first, then column, feature and neighbor attention, each followed by a LayerNorm',
      'The four attention layers in parallel, averaged, then one RMSNorm',
    ],
    answer: 0,
    why: 'Five sublayers, always column, feature, neighbor, full, then the gated feed-forward network (SwiGLU). Each has its own RMSNorm and weights and computes x ← x + f(RMSNorm(x)): pre-norm, as the code runs it.',
  },
  {
    id: 'decoder',
    chapters: ['decoding'],
    prompt: 'After the last block, how does RT turn the context into its answer?',
    choices: [
      'A final RMSNorm, then the decoder of the masked cell’s data type, read at the masked position',
      'A separate decoder for each task, applied to the average of all tokens',
      'The text decoder writes the answer word by word',
      'A softmax over every value stored in the database',
    ],
    answer: 0,
    why: 'norm_out, then four linear decoders (number, text, datetime, boolean) run at every position, but only one output is read: the head that matches the masked cell’s type, at the masked position.',
  },
  {
    id: 'shape',
    chapters: ['end-to-end'],
    prompt: 'A context has L = 128 tokens and the model width is d = 64. What shape is the matrix that enters the first block?',
    choices: [
      '128 × 64: one row per token, one column per model dimension',
      '64 × 64: one row and one column per model dimension',
      '128 × 128: one row and one column per token',
      '128 × 384: one column per dimension of the sentence encoder',
    ],
    answer: 0,
    why: 'The embedding turns each token into d numbers, so the blocks read an L × d matrix: 128 × 64 here, 1,024 × 256 in the paper. L × L is the shape of the masks and attention scores, and the 384-dimensional MiniLM vectors are projected to d before the blocks.',
  },
  {
    id: 'zero-shot',
    chapters: ['training', 'overview'],
    prompt:
      'The paper’s zero-shot setting predicts “on a new RDB with a new schema, without weight updates”. Where does the model get its information about the new database?',
    choices: [
      'Only from the context window sampled at prediction time',
      'From a short fine-tuning run on a few labeled rows of the new database',
      'From pretraining on the new database’s schema, without its labels',
      'From a text description of the database given with each question',
    ],
    answer: 0,
    why: '“Information about the new RDB is available only at inference time, solely through the context window.” To test this, the paper pretrains on every task of the other databases and leaves the target database out (leave-one-database-out), as this site does with ShopDB.',
  },
  {
    id: 'top-driver',
    chapters: ['playground', 'results'],
    prompt: 'Which source of information does the paper rank first among the drivers of zero-shot transfer?',
    choices: [
      'The target entity’s own past labels, from earlier rows of the task table',
      'The column and table names of the new database',
      'Text values embedded by the sentence encoder',
      'The number of tables in the database',
    ],
    answer: 0,
    why: 'The paper lists the drivers “in order of importance”, and the first is “time-series forecasting from past task rows for the target entity”: the masked cell reads the entity’s earlier labels through column attention, since they share its column.',
  },
];

/** Share of right answers needed to pass. */
export const PASS_MARK = 0.8;

/** Right answers needed to pass a quiz of `total` questions (12 of 15). */
export function passScore(total: number): number {
  return Math.ceil(total * PASS_MARK - 1e-9);
}

/** Number of right answers; `picked[i]` is the index into FINAL_QUIZ[i].choices, or null. */
export function gradeQuiz(picked: readonly (number | null)[], quiz: readonly FinalQuestion[] = FINAL_QUIZ): number {
  return quiz.reduce((n, q, i) => n + (picked[i] === q.answer ? 1 : 0), 0);
}
