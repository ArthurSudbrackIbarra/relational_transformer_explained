/**
 * "Four rules worth remembering", each with a live example drawn from the current trace and a
 * "show me" button that jumps the replay (above) to the decision.
 */
import { Clock, Crosshair, Layers, Ruler, Scissors } from 'lucide-react';
import { useMemo, type ReactNode } from 'react';
import { Button, Callout, Cite } from '@/components';
import { useExplore } from './explore';
import { useTrace } from './player';
import { cutRow, iterationEvents, plural, rowName, tsDate, type Iteration } from './trace';

function Card({ n, icon, title, rule, cite, example, paper, code, onShow, showLabel = 'Show me' }: {
  n: number;
  icon: ReactNode;
  title: string;
  rule: ReactNode;
  cite: ReactNode;
  example: ReactNode;
  /** Paper vs code (BUILD_PLAN §0.3): what the paper says, and how the code differs. */
  paper: ReactNode;
  code: ReactNode;
  /** Receives the card element (offered back as "Back to rule n" in the figure). */
  onShow?: (card: HTMLElement) => void;
  showLabel?: string;
}) {
  return (
    <article className="card flex min-w-0 flex-col px-4 py-4">
      <div className="flex items-center gap-2.5">
        <span className="inline-flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-accent-soft text-[15.5px] font-semibold text-accent-text">{n}</span>
        <h3 className="text-[18px] leading-snug font-semibold text-ink">{title}</h3>
        <span aria-hidden="true" className="ml-auto text-muted">
          {icon}
        </span>
      </div>
      <p className="mt-2 text-[17px] leading-[1.55] text-ink">
        {rule} {cite}
      </p>
      <Callout variant="paper-vs-code" title="Paper vs code" className="!mb-0 !mt-3 !px-3 !py-2.5">
        <p className="!mb-1 text-[16px] leading-[1.5] text-ink-2">
          <span className="font-semibold text-ink">Paper: </span>
          {paper}
        </p>
        <p className="text-[16px] leading-[1.5] text-ink">
          <span className="font-semibold">Code: </span>
          {code}
        </p>
      </Callout>
      <div className="mt-3 flex-1 rounded-lg bg-surface-2 px-3 py-2.5 text-[16px] leading-snug text-ink-2">
        <div className="eyebrow mb-1 !text-[13px]">In this context</div>
        {example}
      </div>
      {onShow && (
        <div className="mt-3">
          <Button
            size="sm"
            variant="secondary"
            data-show-me=""
            icon={<Crosshair size={13} />}
            onClick={(e: React.MouseEvent<HTMLElement>) => {
              const card = e.currentTarget.closest('article');
              if (card) onShow(card as HTMLElement);
            }}
          >
            {showLabel}
          </Button>
        </div>
      )}
    </article>
  );
}

export function RuleCards() {
  const { ctx, pdb, db, task } = useExplore();
  const { iters, jumpTo } = useTrace();
  const name = (n: number) => rowName(db, pdb, n).short;
  const seed = pdb.nodes[ctx.params.seedNode];

  const facts = useMemo(() => {
    const all = iters.map((it) => ({ it, ie: iterationEvents(ctx, it) }));
    const parentPushes = all.reduce((s, x) => s + x.ie.parents.length, 0);
    const stackPops = all.filter((x) => x.it.from === 'stack').length;
    const stackSkips = all.filter((x) => x.it.from === 'stack' && x.it.skipped).length;
    // a row that pushed ≥ 2 parents, preferring one whose top parent was not yet visited (so the pop that
    // follows really visits it; otherwise that pop is skipped, fly.rs:399-401)
    const seen = new Set<number>();
    let multiParent: (typeof all)[number] | undefined;
    let multiParentTopVisited: (typeof all)[number] | undefined;
    for (const x of all) {
      if (x.it.skipped) continue;
      if (x.ie.parents.length >= 2) {
        const top = x.ie.parents[x.ie.parents.length - 1].node;
        if (!seen.has(top) && !multiParent) multiParent = x;
        if (seen.has(top) && !multiParentTopVisited) multiParentTopVisited = x;
      }
      seen.add(x.it.node);
    }
    const anyParent = all.find((x) => !x.it.skipped && x.ie.parents.length >= 1);
    const future = all.reduce((s, x) => s + x.ie.future.length, 0);
    const firstFuture = all.find((x) => x.ie.future.length > 0);
    const subs = all.filter((x) => x.ie.subsample);
    const biggest = subs.reduce<(typeof subs)[number] | undefined>((b, x) => (!b || x.ie.subsample!.candidates.length > b.ie.subsample!.candidates.length ? x : b), undefined);
    const taskKids = all.reduce((s, x) => s + x.ie.taskKids.length, 0);
    const firstTask = all.find((x) => x.ie.taskKids.length > 0);
    return { parentPushes, stackPops, stackSkips, multiParent: multiParent ?? multiParentTopVisited, topVisited: !multiParent && !!multiParentTopVisited, anyParent, future, firstFuture, subs, biggest, taskKids, firstTask };
  }, [iters, ctx]);

  const cut = useMemo(() => cutRow(ctx, ctx.params.dropColNameIdxs, pdb), [ctx, pdb]);
  const ended = ctx.events.find((e) => e.kind === 'frontier-empty');

  const at = (n: number, it: Iteration | undefined, kind: string) => (card: HTMLElement) => {
    if (!it) return;
    jumpTo((s) => s.iter === it.iter && s.kind === kind, { label: `rule ${n}`, el: card });
  };

  const mp = facts.multiParent ?? facts.anyParent;
  const mpParents = mp ? mp.ie.parents.map((p) => name(p.node)) : [];

  return (
    <div className="wide my-8 grid min-w-0 items-start gap-4 md:grid-cols-2">
      <Card
        n={1}
        icon={<Layers size={17} />}
        title="Parents first, always, never filtered"
        rule={
          <>
            Whenever the F→P stack is non-empty, its top is popped: no randomness, no time filter, no width bound, and the last foreign key pushed is
            followed first.
          </>
        }
        cite={
          <>
            <Cite kind="code" ref="rustler/src/fly.rs:378-379" />
            <Cite kind="code" ref="rustler/src/fly.rs:406-408" />
            <Cite kind="spec" ref="§4.4" />
          </>
        }
        paper={
          <>
            among rows added via an F→P link, <span className="font-mono">r ← RandomSelect(R)</span> <Cite kind="paper" ref="p.15 Alg. 1" />; “F→P links are
            immediately followed” <Cite kind="paper" ref="p.16 App. A" />.
          </>
        }
        code={
          <>
            no random pick: a LIFO pop. Each parent’s own parents go on top, so foreign-key chains are followed depth-first, completely, before any child
            row <Cite kind="spec" ref="§4.4" />.
          </>
        }
        example={
          <>
            {plural(facts.parentPushes, 'parent push', 'parent pushes')}, {plural(facts.stackPops, 'stack pop')} ({facts.stackSkips} of them already visited).
            {mp && mpParents.length >= 2 ? (
              <>
                {' '}
                {name(mp.it.node)} pushed {mpParents.join(' then ')}; {mpParents[mpParents.length - 1]} was on top, so it was popped first
                {facts.topVisited && facts.multiParent ? ' (already visited, so that pop was skipped)' : ' and visited'}.
              </>
            ) : mp ? (
              <>
                {' '}
                {name(mp.it.node)} pushed {mpParents.join(', ')}.
              </>
            ) : null}
          </>
        }
        onShow={mp ? at(1, mp.it, 'visit') : undefined}
      />
      <Card
        n={2}
        icon={<Clock size={17} />}
        title="Children only if not later than the seed"
        rule={
          <>
            A P→F child is skipped only when both timestamps exist and the child’s is later than the seed’s. Equal timestamps pass, rows without a timestamp
            pass, and the comparison is always against the <em>seed</em>, never the current row.
          </>
        }
        cite={
          <>
            <Cite kind="code" ref="rustler/src/fly.rs:421-424" />
            <Cite kind="spec" ref="§4.6" />
          </>
        }
        paper={
          <>
            the prose says “rows with timestamps later than the seed row are excluded” <Cite kind="paper" ref="p.4 §3.1" />, and training uses database rows
            “only up to the training cutoff timestamps” <Cite kind="paper" ref="p.6 §4.1" />; Algorithm 1 itself applies <span className="font-mono">T(q) ≤ T(s)</span> only
            to P→F neighbors <Cite kind="paper" ref="p.15 Alg. 1" />.
          </>
        }
        code={
          <>
            the same scope as Algorithm 1 (parents are never tested), but the test needs both timestamps: a child without one passes, and a seed without one
            disables the filter <Cite kind="code" ref="rustler/src/fly.rs:406-424" />. No global cutoff exists <Cite kind="spec" ref="§10" />.
          </>
        }
        example={
          seed.ts === null ? (
            <>This seed has no timestamp, so the filter is disabled for the whole context.</>
          ) : facts.firstFuture ? (
            <>
              {plural(facts.future, 'child row')} skipped as too late. First one: {name(facts.firstFuture.ie.future[0].node)} at{' '}
              {tsDate(facts.firstFuture.ie.future[0].childTs)}, after the seed’s {tsDate(seed.ts)}.
            </>
          ) : (
            <>No child of a visited row was later than the seed ({tsDate(seed.ts)}).</>
          )
        }
        onShow={facts.firstFuture ? at(2, facts.firstFuture.it, 'filter') : undefined}
      />
      <Card
        n={3}
        icon={<Ruler size={17} />}
        title="At most w DB children per row; task rows never subsampled"
        rule={
          <>
            If more than w of a row’s DB children pass the time filter, a uniform random w of them is kept (without replacement). Children from the seed’s
            task table that pass the time filter bypass the bound: all of them enter the frontier.
          </>
        }
        cite={
          <>
            <Cite kind="code" ref="rustler/src/fly.rs:421-452" />
            <Cite kind="spec" ref="§4.7" />
          </>
        }
        paper={
          <>
            <span className="font-mono">N ← RandomSample(N, w)</span> on all P→F neighbors <Cite kind="paper" ref="p.15 Alg. 1" />.
          </>
        }
        code={
          <>
            only DB children are subsampled, and children that are already visited still count toward w <Cite kind="spec" ref="§4.7" />.
          </>
        }
        example={
          <>
            {facts.biggest ? (
              <>
                The bound fired {plural(facts.subs.length, 'time')}; the largest cut: {name(facts.biggest.it.node)} had {facts.biggest.ie.subsample!.candidates.length}{' '}
                DB children after the time filter, {ctx.params.w} kept.
              </>
            ) : (
              <>No row had more than w = {ctx.params.w} DB children that passed the filter.</>
            )}{' '}
            {task.kind === 'forecast'
              ? `${plural(facts.taskKids, `${task.name} row`)} entered the frontier without any subsampling.`
              : 'Autocomplete: no task rows are ever pushed (the seed is a DB row).'}
          </>
        }
        onShow={facts.biggest ? at(3, facts.biggest.it, 'width') : facts.firstTask ? at(3, facts.firstTask.it, 'filter') : undefined}
      />
      <Card
        n={4}
        icon={<Scissors size={17} />}
        title="Stop at exactly L cells"
        rule={<>The loop breaks as soon as the L-th token is written, even in the middle of a row. If the frontier empties first, the rest is padding.</>}
        cite={
          <>
            <Cite kind="code" ref="rustler/src/fly.rs:502-509" />
            <Cite kind="spec" ref="§4.11" />
          </>
        }
        paper={
          <>
            “Once a row is selected, all of its non-missing feature cells are added” <Cite kind="paper" ref="p.4 §3.1" />, and the traversal “stops when the total
            number of cells reaches the context length” <Cite kind="paper" ref="p.16 App. A" />.
          </>
        }
        code={
          <>
            the last row can be partial: the loop does not finish the row it is writing <Cite kind="code" ref="rustler/src/fly.rs:502-509" />.
          </>
        }
        example={
          ended && ended.kind === 'frontier-empty' ? (
            <>
              The frontier emptied after {plural(ended.tokens, 'token')}: {plural(ctx.params.L - ended.tokens, 'padding position')}.
            </>
          ) : cut ? (
            <>
              Full at L = {ctx.params.L}: {name(cut.node)} was cut after {cut.emitted} of its {cut.total} cells.
            </>
          ) : (
            <>Full at L = {ctx.params.L}, exactly at the end of a row (nothing cut this time).</>
          )
        }
        onShow={(card) => jumpTo((s) => s.kind !== 'start' && s.end === ctx.events.length, { label: 'rule 4', el: card })}
        showLabel="Show the last step"
      />
    </div>
  );
}
