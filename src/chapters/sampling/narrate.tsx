/**
 * Plain-language narration of each replay step, plus which lines of the paper's Algorithm 1
 * (p.15) and of the condensed fly.rs loop (spec §4.3) the step corresponds to.
 * Every sentence is generated from the recorded events of the live trace.
 */
import type { ReactNode } from 'react';
import type { PreprocessedDB, RawDatabase, SampledContext, TaskDef } from '@/engine/types';
import { iterationEvents, plural, rowName, singular, tsDate, type Iteration, type ReplayState, type Step } from './trace';

/** Paper Algorithm 1 (p.15), transcribed. `id` is used for highlighting. */
export const ALG1: { id: string; indent: number; text: string; comment?: string }[] = [
  { id: 'init', indent: 0, text: 'C ← {}, F ← {s}', comment: 'F is the frontier of rows to explore' },
  { id: 'while', indent: 0, text: 'while |C| < L ∧ F ≠ {} do' },
  { id: 'c-select', indent: 1, text: '/* select a row to explore; R is the set of candidates */' },
  { id: 'R-f2p', indent: 1, text: 'R ← {r ∈ F | r was added via an F→P link}', comment: 'F→P linked rows' },
  { id: 'if-empty', indent: 1, text: 'if R = {} then' },
  { id: 'argmin', indent: 2, text: 'R ← argmin_{r∈F} HopDistance(r, s)', comment: 'rows closest to s' },
  { id: 'random', indent: 1, text: 'r ← RandomSelect(R)', comment: 'pick a row at random' },
  { id: 'remove', indent: 1, text: 'F ← F \\ {r}', comment: 'remove row from frontier' },
  { id: 'visited', indent: 1, text: 'if r has been visited then continue else mark r as visited' },
  { id: 'c-visit', indent: 1, text: '/* visit row */' },
  { id: 'cells', indent: 1, text: 'C ← C ∪ C(r)', comment: 'add cells to context' },
  { id: 'f2p', indent: 1, text: 'F ← F ∪ N_F→P(r)', comment: 'add F→P neighbors to frontier' },
  { id: 'time', indent: 1, text: 'N ← {q ∈ N_P→F(r) | T(q) ≤ T(s)}', comment: 'filter P→F neighbors by time' },
  { id: 'sample', indent: 1, text: 'N ← RandomSample(N, w)', comment: 'pick ≤ w P→F neighbors at random' },
  { id: 'p2f', indent: 1, text: 'F ← F ∪ N', comment: 'add P→F neighbors to frontier' },
  { id: 'return', indent: 0, text: 'return C' },
];

/** The released loop, condensed from fly.rs:357-511 (spec §4.3). Not verbatim: see the CodeRefs. */
export const CODE_LINES: { id: string; indent: number; text: string; lines: string }[] = [
  { id: 'init', indent: 0, text: 'stack ← [(0, seed)];  buckets ← [];  visited ← ∅', lines: '364-368' },
  { id: 'loop', indent: 0, text: 'loop:', lines: '376' },
  { id: 'pop', indent: 1, text: 'if stack ≠ ∅:  (d, r) ← stack.pop()   // LIFO, no randomness', lines: '378-379' },
  { id: 'bucket', indent: 1, text: 'else:  d ← lowest non-empty bucket (none left → return)', lines: '381-390' },
  { id: 'draw', indent: 1, text: '       r ← swap_remove(bucket[d], uniform index)', lines: '391-394' },
  { id: 'visited', indent: 1, text: 'if visited[r]: continue;  visited[r] ← true', lines: '399-402' },
  { id: 'parents', indent: 1, text: 'for each parent p of r:  stack.push((d+1, p))   // no filter', lines: '406-408' },
  { id: 'children', indent: 1, text: 'for each child q of r:', lines: '415' },
  { id: 'other-task', indent: 2, text: 'skip q if it is in another task table than the seed’s', lines: '417' },
  { id: 'time', indent: 2, text: 'skip q if T(q), T(s) both exist and T(q) > T(s)', lines: '421-424' },
  { id: 'route', indent: 2, text: 'DB row → collect;  task row → bucket[d+1]   // never subsampled', lines: '426-436' },
  { id: 'width', indent: 1, text: 'if more than w DB rows: keep a uniform random w;  push to bucket[d+1]', lines: '439-452' },
  { id: 'drop', indent: 1, text: 'for each cell of r:  skip leakage cells (seed / same-ts rows)', lines: '455-461' },
  { id: 'emit', indent: 2, text: 'emit a token;  masked iff r = seed and column = target', lines: '462-500' },
  { id: 'stop', indent: 2, text: 'if tokens = L:  stop, even mid-row', lines: '502-509' },
];

export interface Narration {
  text: ReactNode;
  alg: string[];
  code: string[];
}

const B = ({ children }: { children: ReactNode }) => <strong className="font-semibold text-ink">{children}</strong>;

function list(items: string[], max = 3): string {
  if (items.length <= max) return items.length > 1 ? `${items.slice(0, -1).join(', ')} and ${items[items.length - 1]}` : items.join('');
  return `${items.slice(0, max).join(', ')} and ${items.length - max} more`;
}

export interface NarrateInput {
  step: Step;
  iters: Iteration[];
  ctx: SampledContext;
  pdb: PreprocessedDB;
  db: RawDatabase;
  task: TaskDef;
  before: ReplayState;
  state: ReplayState;
}

export function narrate({ step, iters, ctx, pdb, db, task, before, state }: NarrateInput): Narration {
  const name = (n: number) => rowName(db, pdb, n);
  const seed = pdb.nodes[ctx.params.seedNode];
  const L = ctx.params.L;
  const w = ctx.params.w;

  if (step.kind === 'start') {
    return {
      text: (
        <>
          The frontier starts with one entry: the <B>seed row</B> {name(seed.nodeIdx).long}, on the F→P stack at depth 0. Press play to run the loop one
          decision at a time.
        </>
      ),
      alg: ['init'],
      code: ['init'],
    };
  }

  if (step.kind === 'end') {
    const e = ctx.events.find((x) => x.kind === 'frontier-empty');
    const n = e && e.kind === 'frontier-empty' ? e.tokens : ctx.numReal;
    return {
      text: (
        <>
          The stack and every bucket are empty after <B>{plural(n, 'token')}</B>: there is nothing left to visit, so fly.rs returns.{' '}
          {n < L ? (
            <>
              Positions {n}–{L - 1} stay <B>padding</B> ({plural(L - n, 'token')}).
            </>
          ) : null}
        </>
      ),
      alg: ['while', 'return'],
      code: ['bucket'],
    };
  }

  const it = iters[step.iter];
  const ie = iterationEvents(ctx, it);
  const nm = name(it.node);
  const d = it.depth;

  // ── sentences per phase ────────────────────────────────────────────────
  const selectText = (): ReactNode => {
    if (ie.pop.from === 'stack') {
      const left = before.stack.length - 1;
      if (it.node === seed.nodeIdx && it.iter === 0)
        return (
          <>
            Pop the seed row <B>{nm.long}</B> from the F→P stack (depth 0). Its cells, including the masked target, will be the first tokens.
          </>
        );
      return (
        <>
          Pop <B>{nm.long}</B> from the F→P stack (depth {d}). Parent rows always come first, last in, first out
          {left > 0 ? `; ${plural(left, 'more entry', 'more entries')} ${left === 1 ? 'waits' : 'wait'} below it` : ''}.
        </>
      );
    }
    const size = ie.pop.bucketSize ?? 0;
    return (
      <>
        The stack is empty, so draw from the lowest non-empty depth bucket: depth {d} holds {plural(size, 'entry', 'entries')}. A uniform draw
        picks index <span className="num">{ie.pop.draw}</span> of 0–{size - 1}: <B>{nm.long}</B>.
      </>
    );
  };

  const skipText = (): ReactNode => (
    <>
      {' '}
      It was already visited, so the loop skips it: it adds no cells and pushes nothing.
    </>
  );

  const visitText = (): ReactNode => {
    const ps = ie.parents.map((p) => name(p.node).short);
    const already = ie.parents.filter((p) => before.visitedDepth.has(p.node)).map((p) => name(p.node).short);
    const top = ie.parents[ie.parents.length - 1];
    const topVisited = top ? before.visitedDepth.has(top.node) : false;
    if (ps.length === 0)
      return (
        <>
          Visit <B>{nm.short}</B> at depth {d}. It holds no foreign key, so there are no parents to push.
        </>
      );
    return (
      <>
        Visit <B>{nm.short}</B> at depth {d} and push {ps.length === 1 ? 'its parent' : `its ${ps.length} parents`} onto the stack at depth {d + 1}:{' '}
        {ps.join(', then ')}
        {ps.length > 1 ? (topVisited ? ' (on top, so it is popped next)' : ' (on top, so it is popped and visited next)') : ''}. Parents are never
        time-filtered or subsampled.
        {already.length > 0 ? ` ${list(already)} ${already.length === 1 ? 'is' : 'are'} already visited and will be skipped when popped.` : ''}
      </>
    );
  };

  const filterText = (): ReactNode => {
    const cand = ie.subsample ? ie.subsample.candidates.length : ie.dbKids.length;
    const total = ie.future.length + ie.otherTask.length + ie.taskKids.length + cand;
    const parts: ReactNode[] = [];
    if (ie.future.length)
      parts.push(
        <span key="f">
          <B>{ie.future.length}</B> {ie.future.length === 1 ? 'is' : 'are'} later than the seed ({tsDate(seed.ts)}) and skipped
        </span>,
      );
    if (ie.otherTask.length)
      parts.push(
        <span key="o">
          <B>{ie.otherTask.length}</B> belong{ie.otherTask.length === 1 ? 's' : ''} to another task table and {ie.otherTask.length === 1 ? 'is' : 'are'} skipped
        </span>,
      );
    if (ie.taskKids.length)
      parts.push(
        <span key="t">
          <B>{ie.taskKids.length}</B> {task.kind === 'forecast' ? task.name : 'task'} {ie.taskKids.length === 1 ? 'row passes' : 'rows pass'} and{' '}
          {ie.taskKids.length === 1 ? 'goes' : 'all go'} to bucket {d + 1}, because task rows are never subsampled
          {ie.taskKids.some((e) => e.node === seed.nodeIdx) ? ' (one of them is the seed row itself, already visited)' : ''}
        </span>,
      );
    if (cand)
      parts.push(
        <span key="c">
          <B>{cand}</B> DB {cand === 1 ? 'row passes' : 'rows pass'} the time filter
        </span>,
      );
    return (
      <>
        <B>{nm.short}</B> has {plural(total, 'child row')} (P→F).{' '}
        {parts.map((p, i) => (
          <span key={i}>
            {i > 0 ? (i === parts.length - 1 ? '; and ' : '; ') : ''}
            {p}
          </span>
        ))}
        .
      </>
    );
  };

  const widthText = (): ReactNode => {
    const cand = ie.subsample ? ie.subsample.candidates : ie.dbKids.map((e) => e.node);
    const visitedCand = cand.filter((c) => before.visitedDepth.has(c)).length;
    const keptVisited = ie.dbKids.filter((e) => before.visitedDepth.has(e.node)).length;
    const isAre = (n: number) => (n === 1 ? 'is' : 'are');
    const visitedText = (): string => {
      if (visitedCand === 0) return '';
      if (!ie.subsample) return ` ${visitedCand} of them ${isAre(visitedCand)} already visited and will be skipped when drawn.`;
      const cands = visitedCand === 1 ? '1 of the candidates is already visited; it still competed' : `${visitedCand} of the candidates are already visited; they still competed`;
      const kept =
        keptVisited === 0
          ? visitedCand === 1
            ? ' It was not kept.'
            : ' None of them was kept.'
          : keptVisited === visitedCand
            ? ` ${visitedCand === 1 ? 'It was' : 'All of them were'} kept and will be skipped when drawn.`
            : ` ${keptVisited} of them ${keptVisited === 1 ? 'was' : 'were'} kept and will be skipped when drawn.`;
      return ` ${cands} for the w places.${kept}`;
    };
    const kinds = [...new Set(cand.map((c) => singular(pdb.nodes[c].tableName)))];
    return (
      <>
        {ie.subsample ? (
          <>
            {cand.length} {list(kinds)} children passed, more than <B>w = {w}</B>: keep a uniform random {w} of them, without replacement, and push them to
            bucket {d + 1}.
          </>
        ) : (
          <>
            {cand.length} {list(kinds)} {cand.length === 1 ? 'child' : 'children'} passed, no more than w = {w}: all go to bucket {d + 1}.
          </>
        )}
        {visitedText()}
      </>
    );
  };

  const emitText = (): ReactNode => {
    const k = ie.emits.length;
    const target = ie.emits.find((e) => e.isTarget);
    const node = pdb.nodes[it.node];
    const notDropped = node.cells.length - ie.drops.length;
    return (
      <>
        {k > 0 ? (
          <>
            Emit {nm.short}’s {k === 1 ? 'cell' : `${k} cells`} as {k === 1 ? `token ${ie.emits[0].pos}` : `tokens ${ie.emits[0].pos}–${ie.emits[k - 1].pos}`}.
          </>
        ) : (
          <>{nm.short} emits no cells.</>
        )}
        {target ? (
          <>
            {' '}
            The {target.colName} cell is the single <B>masked target</B> (?).
          </>
        ) : null}
        {ie.drops.length ? (
          <>
            {' '}
            {list(ie.drops.map((x) => x.colName))} {ie.drops.length === 1 ? 'is a leakage column' : 'are leakage columns'} of this task: dropped (seed row or
            same timestamp as the seed).
          </>
        ) : null}
        {ie.truncated ? (
          k < notDropped ? (
            <>
              {' '}
              The context is <B>full at L = {L}</B>: the row is cut after {k} of its {notDropped} cells, and sampling stops.
            </>
          ) : (
            <>
              {' '}
              The context is <B>full at L = {L}</B>, exactly at the end of this row: sampling stops.
            </>
          )
        ) : null}
      </>
    );
  };

  switch (step.kind) {
    case 'select':
      return {
        text: (
          <>
            {selectText()}
            {it.skipped ? skipText() : null}
          </>
        ),
        alg: ie.pop.from === 'stack' ? ['R-f2p', 'random', 'remove', ...(it.skipped ? ['visited'] : [])] : ['if-empty', 'argmin', 'random', 'remove', ...(it.skipped ? ['visited'] : [])],
        code: ie.pop.from === 'stack' ? ['pop', ...(it.skipped ? ['visited'] : [])] : ['bucket', 'draw', ...(it.skipped ? ['visited'] : [])],
      };
    case 'visit':
      return { text: visitText(), alg: ['visited', 'f2p'], code: ['visited', 'parents'] };
    case 'filter':
      return { text: filterText(), alg: ['time'], code: ['children', 'other-task', 'time', 'route'] };
    case 'width':
      return { text: widthText(), alg: ['sample', 'p2f'], code: ['width'] };
    case 'emit':
      return { text: emitText(), alg: ['cells', ...(ie.truncated ? ['while'] : [])], code: ['drop', 'emit', ...(ie.truncated ? ['stop'] : [])] };
    case 'row': {
      const alg = ie.pop.from === 'stack' ? ['R-f2p', 'random', 'remove', 'visited'] : ['if-empty', 'argmin', 'random', 'remove', 'visited'];
      const code = ie.pop.from === 'stack' ? ['pop', 'visited'] : ['bucket', 'draw', 'visited'];
      if (it.skipped)
        return {
          text: (
            <>
              {selectText()}
              {skipText()}
            </>
          ),
          alg,
          code,
        };
      return {
        text: (
          <>
            {selectText()}{' '}
            {ie.parents.length ? `It pushes ${plural(ie.parents.length, 'parent')} onto the stack. ` : ''}
            {it.hasChildren
              ? `Children: ${ie.future.length} too late, ${ie.otherTask.length} in another task table, ${ie.taskKids.length} task rows kept, ${ie.dbKids.length} DB rows pushed${ie.subsample ? ` (w = ${w} of ${ie.subsample.candidates.length})` : ''}. `
              : ''}
            {ie.emits.length ? `Emits ${plural(ie.emits.length, 'token')}.` : ''}
            {ie.truncated ? <> Context full at L = {L}.</> : null}
          </>
        ),
        alg: [...alg, 'cells', 'f2p', ...(it.hasChildren ? ['time', 'sample', 'p2f'] : [])],
        code: [...code, 'parents', ...(it.hasChildren ? ['children', 'other-task', 'time', 'route', 'width'] : []), 'drop', 'emit', ...(ie.truncated ? ['stop'] : [])],
      };
    }
    default:
      return { text: null, alg: [], code: [] };
  }
}

/** fly.rs line range of the main code line of a step (for the narration's code chip). */
export function codeChip(code: string[]): string | null {
  const first = CODE_LINES.find((c) => c.id === code[0]);
  const last = [...CODE_LINES].reverse().find((c) => code.includes(c.id));
  if (!first || !last) return null;
  const a = first.lines.split('-')[0];
  const b = last.lines.split('-').pop()!;
  return a === b ? a : `${a}-${b}`;
}
