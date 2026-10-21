/** Steps of the "RT in one picture" figure and the chapters that open each one up. */

export const STEPS = ['db', 'task', 'context', 'tokens', 'col', 'feat', 'nbr', 'full', 'predict'] as const;
export type StepId = (typeof STEPS)[number];

export const STEP_CHAPTER_SLUGS: Record<StepId, string[]> = {
  db: ['database'],
  task: ['tasks'],
  context: ['sampling'],
  tokens: ['preprocessing', 'semantics', 'tokens', 'embedding'],
  col: ['masks', 'attention', 'block'],
  feat: ['masks', 'attention', 'block'],
  nbr: ['masks', 'attention', 'block'],
  full: ['masks', 'attention', 'block'],
  predict: ['decoding'],
};
