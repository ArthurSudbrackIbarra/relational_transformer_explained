/**
 * Stage 2: the context window, built by the sampler one row at a time.
 *
 * A replay of the engine's event trace (src/engine/sampler.ts, a mirror of rustler/src/fly.rs:357-511):
 * each step is one visited row, with the state of the frontier after it and the cells it added.
 * Pops of rows that were already visited are skipped by fly.rs (fly.rs:399-401) and are counted in
 * the narration rather than shown as steps. Nothing is re-decided here; the trace helpers
 * (../sampling/trace.ts) only read the recorded decisions.
 */
import { useEffect, useMemo, useRef } from 'react';
import { Cite, Figure, StepPlayer, TableSwatch, TokenStrip, cx, useStepPlayer } from '@/components';
import { cutRow, iterationEvents, parseIterations, replay, rowName } from '../sampling/trace';
import { int, plural } from './format';
import { useE2E } from './live';

export function SampleStage() {
  const live = useE2E();
  const { ctx, db, pdb, slots, setFocus, focus } = live;
  const L = ctx.params.L;
  const iters = useMemo(() => parseIterations(ctx.events), [ctx]);
  const visits = useMemo(() => iters.filter((it) => !it.skipped), [iters]);
  // step 0 = the start (the seed alone on the stack); step s = after the s-th visited row
  const player = useStepPlayer(visits.length + 1, { intervalMs: 1100 });
  const s = Math.min(player.index, visits.length);
  const last = s === visits.length;
  const upto = s === 0 ? 0 : last ? ctx.events.length : visits[s - 1].end;
  const st = useMemo(() => replay(ctx, upto), [ctx, upto]);
  const cur = s > 0 ? visits[s - 1] : null;
  const ie = cur ? iterationEvents(ctx, cur) : null;
  const skippedBefore = useMemo(() => {
    if (!cur) return 0;
    const prev = s > 1 ? visits[s - 2].iter : -1;
    return cur.iter - prev - 1;
  }, [cur, s, visits]);
  const emittedNow = useMemo(() => new Set(ie?.emits.map((e) => e.pos) ?? []), [ie]);
  const cut = useMemo(() => cutRow(ctx, ctx.params.dropColNameIdxs, pdb), [ctx, pdb]);
  const notYet = useMemo(() => {
    const out: number[] = [];
    for (const t of ctx.tokens) if (!t.isPadding && t.pos >= st.emitted) out.push(t.pos);
    return out;
  }, [ctx, st.emitted]);

  // keep the newest visited row in view inside the log (scroll the list only)
  const logRef = useRef<HTMLOListElement | null>(null);
  useEffect(() => {
    const el = logRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [s]);

  const name = (node: number) => rowName(db, pdb, node);
  const swatch = (node: number) => <TableSwatch slot={slots[pdb.nodes[node]?.tableName ?? ''] ?? 0} />;
  const depthOf = (node: number) => st.visitedDepth.get(node);

  return (
    <Figure
      wide
      title="Sampling the context, one row at a time"
      caption={
        <>
          A replay of the sampler’s recorded decisions for this seed. Each step visits one row and appends its cells to the context; the strip
          at the bottom fills up to L = {L}. Click a token in the strip to follow it on the rest of the page.
        </>
      }
    >
      <StepPlayer
        {...player}
        unit="row"
        label="Play the sampler row by row"
        formatCounter={(i, n) => (i === 0 ? `start · ${int(n - 1)} rows` : `row ${i} / ${int(n - 1)}`)}
        describe={(i) =>
          i === 0 ? 'The seed row waits alone on the F→P stack.' : i === visits.length ? 'The context is complete.' : `Visit ${name(visits[i - 1].node).short}.`
        }
      />
      <div className="mt-4 grid grid-cols-1 gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <div className="min-w-0">
          <div className="mb-1.5 text-[15.5px] font-semibold text-ink">Rows visited, in order ({int(st.visitedOrder.length)})</div>
          <ol ref={logRef} className="scroll-quiet !m-0 grid grid-cols-1 max-h-[300px] content-start gap-1 overflow-y-auto rounded-xl border border-border bg-surface !p-1.5" style={{ listStyle: 'none' }}>
            {st.visitedOrder.length === 0 && <li className="!m-0 px-2 py-1.5 text-[15px] text-muted">No row visited yet.</li>}
            {st.visitedOrder.map((node, k) => {
              const nm = name(node);
              const n = st.emittedByNode.get(node)?.length ?? 0;
              const isCur = cur?.node === node;
              return (
                <li key={node} className={cx('!m-0 flex items-center gap-2 rounded-lg px-2 py-1 text-[15px]', isCur ? 'bg-accent-soft' : '')}>
                  <span className="num w-[26px] shrink-0 text-right text-muted">{k + 1}</span>
                  {swatch(node)}
                  <span className={cx('min-w-0 flex-1 truncate', isCur ? 'font-semibold text-accent-text' : 'text-ink')}>{nm.long}</span>
                  <span className="num shrink-0 text-muted">d{depthOf(node)}</span>
                  <span className="num w-[62px] shrink-0 text-right text-ink-2">{plural(n, 'cell')}</span>
                </li>
              );
            })}
          </ol>
        </div>
        <div className="grid grid-cols-1 min-w-0 content-start gap-3">
          <div className="rounded-xl bg-surface-2 px-4 py-3 text-[15.5px] leading-snug text-ink-2" aria-live="polite">
            {!cur ? (
              <p className="!m-0">
                The frontier starts with the seed row alone on the F→P stack, at depth 0 <Cite kind="code" ref="rustler/src/fly.rs:366" />. Rows come off
                the stack first; only when it is empty is a row drawn from the lowest non-empty P→F depth bucket <Cite kind="code" ref="rustler/src/fly.rs:378-397" />.
              </p>
            ) : (
              <>
                <p className="!m-0">
                  <span className="font-semibold text-ink">{name(cur.node).long}</span>, depth {cur.depth}:{' '}
                  {ie!.pop.from === 'stack' ? (
                    <>popped from the F→P stack (parents come first, last in, first out)</>
                  ) : (
                    <>
                      drawn uniformly from the depth-{cur.depth} P→F bucket{ie!.pop.bucketSize ? `, one of ${ie!.pop.bucketSize} entries` : ''}
                    </>
                  )}
                  {skippedBefore > 0 && <>, after {plural(skippedBefore, 'pop')} of rows already visited, which are skipped</>}.
                </p>
                <ul className="!mt-2 !mb-0 grid grid-cols-1 gap-1 !pl-5">
                  {ie!.parents.length > 0 && (
                    <li className="!m-0">
                      {plural(ie!.parents.length, 'parent')} pushed onto the stack: {ie!.parents.map((e) => name(e.node).short).join(', ')}{' '}
                      <Cite kind="code" ref="rustler/src/fly.rs:406-408" />.
                    </li>
                  )}
                  {(ie!.future.length > 0 || ie!.otherTask.length > 0) && (
                    <li className="!m-0">
                      Children skipped: {ie!.future.length > 0 && <>{plural(ie!.future.length, 'row')} later than the seed</>}
                      {ie!.future.length > 0 && ie!.otherTask.length > 0 && ', '}
                      {ie!.otherTask.length > 0 && <>{plural(ie!.otherTask.length, 'row')} of another task table</>}{' '}
                      <Cite kind="code" ref="rustler/src/fly.rs:415-424" />.
                    </li>
                  )}
                  {ie!.taskKids.length > 0 && (
                    <li className="!m-0">
                      {plural(ie!.taskKids.length, 'task row')} (labels) pushed to depth {cur.depth + 1}, all of them: the width bound does not apply{' '}
                      <Cite kind="code" ref="rustler/src/fly.rs:431-436" />.
                    </li>
                  )}
                  {ie!.subsample ? (
                    <li className="!m-0">
                      {plural(ie!.subsample.candidates.length, 'database child', 'database children')} passed the time filter; a uniform {ie!.subsample.kept.length}{' '}
                      of them (w = {ie!.subsample.w}) pushed to depth {cur.depth + 1} <Cite kind="code" ref="rustler/src/fly.rs:439-452" />.
                    </li>
                  ) : (
                    ie!.dbKids.length > 0 && (
                      <li className="!m-0">
                        {plural(ie!.dbKids.length, 'database child', 'database children')} pushed to depth {cur.depth + 1} (no more than w ={' '}
                        {ctx.params.w}, so all kept) <Cite kind="code" ref="rustler/src/fly.rs:439-452" />.
                      </li>
                    )
                  )}
                  <li className="!m-0">
                    {ie!.emits.length > 1 ? (
                      <>
                        Its {plural(ie!.emits.length, 'cell')} become tokens {ie!.emits[0].pos}–{ie!.emits[ie!.emits.length - 1].pos}
                        {ie!.emits.some((e) => e.isTarget) && ', including the masked target'}
                      </>
                    ) : ie!.emits.length === 1 ? (
                      <>
                        One cell of it becomes token {ie!.emits[0].pos}
                        {ie!.emits[0].isTarget && ', the masked target'}
                      </>
                    ) : (
                      'No cell of it is emitted'
                    )}
                    {ie!.drops.length > 0 && <>; {plural(ie!.drops.length, 'leakage cell')} dropped</>} <Cite kind="code" ref="rustler/src/fly.rs:454-500" />.
                  </li>
                  {ie!.truncated && (
                    <li className="!m-0">
                      {cut && cut.node === cur.node
                        ? `The context is full at L = ${L} after ${cut.emitted} of this row’s ${cut.total} cells, so the sampler stops mid-row`
                        : `The context is full at L = ${L} with this row’s last cell, so the sampler stops`}{' '}
                      <Cite kind="code" ref="rustler/src/fly.rs:502-509" />.
                    </li>
                  )}
                </ul>
              </>
            )}
            {last && st.finished?.kind === 'empty' && (
              <p className="!mt-2 !mb-0">
                The frontier ran out with {int(st.finished.tokens)} tokens: the remaining {int(L - st.finished.tokens)} positions are padding{' '}
                <Cite kind="spec" ref="§4.11" />.
              </p>
            )}
          </div>
          <div className="grid grid-cols-3 gap-2 text-center">
            <Counter label="tokens" value={`${int(st.emitted)} / ${int(L)}`} />
            <Counter label="on the F→P stack" value={int(st.stack.length)} />
            <Counter label="in P→F buckets" value={int(st.buckets.reduce((a, b) => a + b.length, 0))} />
          </div>
        </div>
      </div>
      <div className="mt-4">
        <div className="mb-1.5 flex flex-wrap items-baseline justify-between gap-x-3 text-[15.5px]">
          <span className="font-semibold text-ink">The context window, position by position</span>
          <span className="text-[14.5px] text-muted">accent ring: cells added at this step · faded: not sampled yet</span>
        </div>
        <TokenStrip
          tokens={ctx.tokens}
          tableSlots={slots}
          variant="cells"
          highlight={emittedNow}
          muted={notYet}
          selected={focus}
          onSelect={(p) => setFocus(p)}
          label="Context window tokens, in sampling order"
        />
      </div>
    </Figure>
  );
}

function Counter({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-border bg-surface px-2 py-2">
      <div className="num text-[18px] font-semibold text-ink">{value}</div>
      <div className="text-[14px] leading-tight text-muted">{label}</div>
    </div>
  );
}
