/**
 * The paper-vs-code differences that change what RT computes or what its context holds, chosen from
 * research/RT_TECHNICAL_SPEC.md §10. "Code" is the paper-era code at commit 2a899c8; this site follows it.
 */
import type { CiteKind } from '@/components/glossary';

export interface Ref {
  kind: CiteKind;
  ref: string;
}

export interface Discrepancy {
  topic: string;
  paper: string;
  paperRefs: Ref[];
  code: string;
  codeRefs: Ref[];
  /** Chapter slug where the site covers it. */
  chapter: string;
}

const P = (ref: string): Ref => ({ kind: 'paper', ref });
const C = (ref: string): Ref => ({ kind: 'code', ref });
const S = (ref: string): Ref => ({ kind: 'spec', ref });

export const DISCREPANCIES: Discrepancy[] = [
  {
    topic: 'Where the norm sits in a block',
    paper: 'Algorithm 2 normalizes after each sublayer: X ← X + Norm(MHA(X; M)).',
    paperRefs: [P('p.26 Alg. 2')],
    code: 'Pre-norm: each sublayer reads a normalized copy of its input, x + MHA(RMSNorm(x)), and the feed-forward layer works the same way.',
    codeRefs: [C('rt/model.py:83-85')],
    chapter: 'block',
  },
  {
    topic: 'Which rows feed the normalization statistics',
    paper: 'Values are normalized with statistics “computed on the training split”.',
    paperRefs: [P('p.4-5 §3.1')],
    code: 'Only the task labels use training-split statistics. Database columns use all their rows, and datetimes share one z-score across all tables and splits.',
    codeRefs: [C('rustler/src/pre.rs:316-396'), S('§3.4')],
    chapter: 'preprocessing',
  },
  {
    topic: 'How the sampler picks rows',
    paper:
      'Algorithm 1 picks a random parent from the frontier, takes the child with the smallest hop distance, and caps the children of every row at width w.',
    paperRefs: [P('p.15 Alg. 1')],
    code: 'Parents come off a stack, in a fixed order and before any child. Children are drawn at random from the shallowest non-empty depth, and only database rows are capped at w; task rows never are.',
    codeRefs: [C('rustler/src/fly.rs:378-396'), C('rustler/src/fly.rs:426-452')],
    chapter: 'sampling',
  },
  {
    topic: 'Labels from the seed’s own date',
    paper: 'Not discussed. The paper only says that “rows with timestamps later than the seed row are excluded”.',
    paperRefs: [P('p.4 §3.1')],
    code: 'Rows with the seed’s own timestamp pass that filter, so other entities’ labels for the very date being predicted can sit in the context.',
    codeRefs: [C('rustler/src/fly.rs:422'), S('§2.5')],
    chapter: 'sampling',
  },
  {
    topic: 'How many cells are masked',
    paper: 'Masked token prediction, and Table 14 also masks extra cells with probability P(mask).',
    paperRefs: [P('p.6 §3.3'), P('p.24 Table 14')],
    code: 'Only the target cell is ever masked, one per context.',
    codeRefs: [C('rustler/src/fly.rs:492-496')],
    chapter: 'training',
  },
];
