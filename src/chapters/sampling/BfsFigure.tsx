/**
 * The chapter's centerpiece: fly.rs building the live context, one decision at a time.
 *
 *   controls (split, seed, L, w, RNG seed; local to this chapter unless applied)
 *   StepPlayer + narration (driven by SampledContext.events; sticky on wide screens)
 *   the Alg. 1 / fly.rs lines of the current step, quoted
 *   radial BFS graph │ decision detail, F→P stack, P→F buckets (on narrow screens the stack and buckets
 *   come first, as a scrollable strip above the graph)
 *   the context as per-table mini tables
 *   progress meter │ token strip filling up to L
 */
import { ArrowDown, Dice5, Globe2, Ruler, RotateCcw, Shuffle, SkipForward } from 'lucide-react';
import { memo, useCallback, useMemo, useState } from 'react';
import { Button, Cite, IconButton, NumberField, Segmented, Select, Slider, StepPlayer, TokenStrip, cx } from '@/components';
import type { Split } from '@/engine/types';
import { BfsGraph } from './BfsGraph';
import { ContextTables } from './ContextTables';
import { DecisionPanel } from './DecisionPanel';
import { useExplore } from './explore';
import { BucketsPanel, CountersPanel, StackPanel } from './FrontierPanels';
import { renderAlgText } from './AlgorithmPanel';
import { ALG1, CODE_LINES, codeChip } from './narrate';
import { usePlayer } from './player';
import { cutRow, rowName, tsDate } from './trace';
import { useNarration } from './useNarration';

/**
 * Seed labels, most telling part first so a narrow select still shows it:
 * "customer 35 · 2025-03-01 · Val row 32 · #2406" (task row) or "review 151 · 2024-11-20 · #1547" (DB row).
 */
function useSeedOptions() {
  const { seeds, pdb, db } = useExplore();
  return useMemo(
    () =>
      seeds.map((s) => {
        const n = pdb.nodes[s];
        const nm = rowName(db, pdb, s);
        const parent = n.f2pNbrIdxs[0];
        const date = n.ts !== null ? ` · ${tsDate(n.ts)}` : '';
        const label =
          n.tableType !== 'Db' && parent !== undefined
            ? `${rowName(db, pdb, parent).short}${date} · ${n.tableType} row ${n.rowPos} · #${s}`
            : `${nm.short}${date} · #${s}`;
        return { value: s, label, short: nm.short };
      }),
    [seeds, pdb, db],
  );
}

export const SamplerControls = memo(function SamplerControls({ compact = false }: { compact?: boolean }) {
  const ex = useExplore();
  const options = useSeedOptions();
  const MAX = 500;
  const shown = options.length > MAX ? [...options.slice(0, MAX), ...options.filter((o) => o.value === ex.seedNode && options.indexOf(o) >= MAX)] : options;
  return (
    <div className="grid gap-3">
      <div className={cx('grid gap-x-4 gap-y-3', compact ? 'sm:grid-cols-2' : 'sm:grid-cols-2 lg:items-end lg:grid-cols-[auto_minmax(0,1.6fr)_minmax(0,1fr)_minmax(0,1fr)_110px]')}>
        {!compact && (
          <div className="flex flex-col gap-1">
            <span className="text-[14.5px] font-medium text-ink-2">Seed split</span>
            <Segmented<Split>
              label="Seed split"
              value={ex.split}
              onChange={(s) => ex.set({ split: s, seedNode: undefined })}
              disabled={ex.task.kind === 'autocomplete'}
              options={[
                { value: 'Train', label: 'Train' },
                { value: 'Val', label: 'Val' },
                { value: 'Test', label: 'Test' },
              ]}
            />
          </div>
        )}
        {!compact && (
          <div className="flex min-w-0 items-end gap-1.5">
            <Select
              className="min-w-0 flex-1"
              label={`Seed row (${ex.seeds.length.toLocaleString('en-US')} ${ex.task.kind === 'autocomplete' ? 'rows' : `${ex.split} rows`})`}
              value={String(ex.seedNode)}
              onChange={(e) => ex.set({ seedNode: Number(e.target.value) })}
            >
              {shown.map((o) => (
                <option key={o.value} value={o.value}>
                  {o.label}
                </option>
              ))}
            </Select>
            <IconButton
              label="Random seed row"
              size="sm"
              className="border border-border-strong"
              onClick={() => {
                if (ex.seeds.length) ex.set({ seedNode: ex.seeds[Math.floor(Math.random() * ex.seeds.length)] });
              }}
            >
              <Shuffle size={14} />
            </IconButton>
          </div>
        )}
        <Slider label="Context length L" value={Math.min(ex.L, 256)} min={16} max={256} step={8} onChange={(v) => ex.set({ L: v })} format={() => ex.L} hint="Paper: 1024" />
        <Slider label="BFS width w" value={Math.min(ex.w, 64)} min={1} max={64} onChange={(v) => ex.set({ w: v })} format={() => ex.w} hint="Paper scripts: 256" />
        {!compact && (
          <NumberField label="RNG seed" value={ex.rngSeed} min={0} max={9999} onChange={(v) => ex.set({ rngSeed: Math.max(0, Math.min(9999, Math.round(v))) })} hint="fly.rs seed" />
        )}
      </div>
      {ex.isLocal && (
        <div className="flex flex-wrap items-center gap-x-3 gap-y-2 rounded-lg border border-dashed border-accent bg-accent-soft px-3 py-2 text-[15px] text-ink">
          <span>
            <strong className="font-semibold">Local to this chapter.</strong> The rest of the course still uses{' '}
            {ex.localKeys
              .map((k) =>
                k === 'seedNode' ? `seed #${ex.global.seedNode}` : k === 'split' ? `the ${ex.global.split} split` : k === 'rngSeed' ? `RNG seed ${ex.global.rngSeed}` : `${k} = ${ex.global[k]}`,
              )
              .join(', ')}
            .
          </span>
          <span className="ml-auto flex flex-wrap gap-2">
            <Button size="sm" variant="ghost" icon={<RotateCcw size={13} />} onClick={ex.reset}>
              Back to course settings
            </Button>
            <Button size="sm" variant="primary" icon={<Globe2 size={13} />} onClick={ex.applyGlobally}>
              Use in every chapter
            </Button>
          </span>
        </div>
      )}
    </div>
  );
});

/** The text of the Alg. 1 and fly.rs lines of the current step, quoted inside the figure. */
const ActiveLines = memo(function ActiveLines({ alg, code }: { alg: string[]; code: string[] }) {
  const CAP = 4;
  const algLines = ALG1.map((l, i) => ({ ...l, n: i + 1 })).filter((l) => alg.includes(l.id));
  const codeLines = CODE_LINES.filter((l) => code.includes(l.id));
  if (algLines.length === 0 && codeLines.length === 0) return null;
  const more = (n: number) =>
    n > CAP ? <li className="text-[14px] text-muted">+ {n - CAP} more {n - CAP === 1 ? 'line' : 'lines'}, all lit in the side-by-side panel above</li> : null;
  return (
    <div className="grid min-w-0 gap-x-5 gap-y-2 rounded-xl border border-border bg-surface-2/50 px-3 py-2.5 md:grid-cols-2" aria-label="Lines of the current step">
      <div className="min-w-0">
        <div className="mb-1 flex items-baseline gap-2 text-[14px] font-semibold text-ink-2">
          Paper, Algorithm 1 <Cite kind="paper" ref="p.15 Alg. 1" />
        </div>
        <ol className="m-0 grid list-none gap-0.5 p-0 font-mono text-[15px] leading-snug text-ink">
          {algLines.slice(0, CAP).map((l) => (
            <li key={l.id} className="flex min-w-0 gap-2">
              <span className="w-[1.75rem] shrink-0 text-right text-[14px] text-muted">{l.n}</span>
              <span className={cx('min-w-0', l.text.startsWith('/*') && 'text-muted italic')}>{renderAlgText(l.text)}</span>
            </li>
          ))}
          {more(algLines.length)}
        </ol>
      </div>
      <div className="min-w-0">
        <div className="mb-1 flex items-baseline gap-2 text-[14px] font-semibold text-ink-2">
          Code, fly.rs (condensed) <Cite kind="code" ref={`rustler/src/fly.rs:${codeLines[0]?.lines.split('-')[0] ?? 357}-${codeLines[codeLines.length - 1]?.lines.split('-').pop() ?? 511}`} />
        </div>
        <ol className="m-0 grid list-none gap-0.5 p-0 font-mono text-[15px] leading-snug text-ink">
          {codeLines.slice(0, CAP).map((l) => (
            <li key={l.id} className="flex min-w-0 gap-2">
              <span className="w-[4.5rem] shrink-0 text-right text-[14px] whitespace-nowrap text-muted">{l.lines}</span>
              <span className="min-w-0">{l.text}</span>
            </li>
          ))}
          {more(codeLines.length)}
        </ol>
      </div>
    </div>
  );
});

export function BfsFigure() {
  const ex = useExplore();
  const { ctx, pdb, db, task, slots } = ex;
  const pl = usePlayer();
  const { step, state, before, layout, player, granularity, setGranularity, iters, cursor } = pl;
  const narration = useNarration();
  const [hover, setHover] = useState<number | null>(null);

  const it = step.iter >= 0 ? iters[step.iter] : undefined;
  const current = it ? it.node : step.kind === 'start' ? ctx.params.seedNode : null;

  // rows that entered the frontier in this step (pop-in + accent link)
  const fresh = useMemo(() => {
    const s = new Set<number>();
    for (const ln of layout.nodes.values()) if (ln.firstEvent >= step.start && ln.firstEvent < step.end) s.add(ln.node);
    return s;
  }, [layout, step]);

  const candidates = useMemo(() => {
    const e = ctx.events[step.start];
    if ((step.kind === 'select' || step.kind === 'row') && e && e.kind === 'pop' && e.from === 'bucket') return new Set(before.buckets[e.depth] ?? []);
    return undefined;
  }, [ctx, step, before]);

  const freshTokens = useMemo(() => {
    const s = new Set<number>();
    for (let i = step.start; i < step.end; i++) {
      const e = ctx.events[i];
      if (e.kind === 'emit') s.add(e.pos);
    }
    return s;
  }, [ctx, step]);

  const cut = useMemo(() => cutRow(ctx, ctx.params.dropColNameIdxs, pdb), [ctx, pdb]);
  const tokenCol = useCallback((pos: number) => ctx.tokens[pos]?.colName, [ctx]);

  const stripTokens = useMemo(() => {
    const real = ctx.tokens.slice(0, state.emitted);
    return state.finished?.kind === 'empty' ? [...real, ...ctx.tokens.slice(ctx.numReal)] : real;
  }, [ctx, state.emitted, state.finished]);

  const stripHighlight = useMemo(() => {
    const s = new Set<number>(freshTokens);
    if (hover !== null) for (const p of state.emittedByNode.get(hover) ?? []) s.add(p);
    return s;
  }, [freshTokens, hover, state.emittedByNode]);

  const chip = codeChip(narration.code);
  const algIdx = [...new Set(narration.alg.map((id) => ALG1.findIndex((l) => l.id === id) + 1).filter((x) => x > 0))].sort((a, b) => a - b);
  const iterCount = iters.length;

  const common = { db, pdb, slots, state, before, step, ctx, hover, onHover: setHover };

  // Keyboard anywhere in the figure (not only inside the player): Space, ← →, Home / End. The player's own
  // handler runs first and marks the event as handled.
  const onFigureKey = (e: React.KeyboardEvent<HTMLDivElement>) => {
    if (e.defaultPrevented || e.altKey || e.ctrlKey || e.metaKey) return;
    const t = e.target as HTMLElement;
    const tag = t.tagName;
    if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA' || t.isContentEditable || t.getAttribute('role') === 'radio') return;
    const p = player;
    if (e.key === ' ' && tag !== 'BUTTON') {
      e.preventDefault();
      p.setPlaying(!p.playing);
    } else if (e.key === 'ArrowRight') {
      e.preventDefault();
      p.step(1);
    } else if (e.key === 'ArrowLeft') {
      e.preventDefault();
      p.step(-1);
    } else if (e.key === 'Home') {
      e.preventDefault();
      p.setIndex(0);
    } else if (e.key === 'End') {
      e.preventDefault();
      p.setIndex(p.count - 1);
    }
  };

  const back = pl.origin;

  return (
    <div className="wide my-8 min-w-0">
      {/* tabIndex −1: a click anywhere in the figure keeps keyboard focus here, so the shortcuts keep working */}
      <div className="card px-3 py-4 outline-none sm:px-5 sm:py-5" role="group" aria-label="Sampler replay figure" tabIndex={-1} onKeyDown={onFigureKey}>
        <div className="mb-4 flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
          <h3 className="text-[17.5px] font-semibold text-ink">
            fly.rs, live: the context of {rowName(db, pdb, ctx.params.seedNode).long}
          </h3>
          <span className="text-[15px] text-ink-2">
            {db.name} · {task.id.split('/')[1]} · L = {ctx.params.L} · w = {ctx.params.w}
          </span>
        </div>

        <SamplerControls />

        {/* scroll target of "Show me" (a static marker: the sticky player itself moves) */}
        <div ref={pl.figureRef} aria-hidden="true" className="h-0" />
        <div className="mt-4 mb-2 flex flex-wrap items-center justify-between gap-2">
          <div className="flex flex-wrap items-center gap-2">
            <Segmented
              label="Replay granularity"
              value={granularity}
              onChange={setGranularity}
              options={[
                { value: 'decisions', label: 'Each decision', title: 'select, visit, filter, width, emit' },
                { value: 'rows', label: 'Each row', title: 'one loop iteration per step' },
              ]}
            />
            <div className="flex flex-wrap items-center gap-1" role="group" aria-label="Skip ahead">
              <span className="mr-0.5 text-[14.5px] text-ink-2">Next:</span>
              <Button size="sm" variant="ghost" icon={<SkipForward size={13} />} disabled={!pl.canSkip('row')} onClick={() => pl.skip('row')} title="Next loop iteration (the next pop)">
                row
              </Button>
              <Button size="sm" variant="ghost" icon={<Dice5 size={13} />} disabled={!pl.canSkip('draw')} onClick={() => pl.skip('draw')} title="Next uniform draw from a depth bucket">
                bucket draw
              </Button>
              <Button size="sm" variant="ghost" icon={<Ruler size={13} />} disabled={!pl.canSkip('width')} onClick={() => pl.skip('width')} title="Next time the width bound keeps w of more children">
                width cut
              </Button>
            </div>
          </div>
          <span className="num text-[14.5px] text-ink-2">
            {step.kind === 'start' ? 'before the first pop' : step.kind === 'end' ? 'after the last pop' : `loop iteration ${step.iter + 1} of ${iterCount}`}
            <span className="ml-3 hidden text-[14px] text-muted xl:inline">Space play/pause · ← → step · Home / End</span>
          </span>
        </div>
        {back && (
          <div className="mb-2 flex justify-end">
            <Button
              size="sm"
              variant="secondary"
              iconRight={<ArrowDown size={13} />}
              onClick={() => {
                // back to the card's own "Show me" button, centered, with focus on it
                const target = back.el.querySelector<HTMLElement>('[data-show-me]') ?? back.el;
                pl.clearOrigin();
                const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
                target.scrollIntoView({ block: 'center', behavior: reduce ? 'auto' : 'smooth' });
                target.focus({ preventScroll: true });
              }}
            >
              Back to {back.label}
            </Button>
          </div>
        )}

        {/* the player and its narration stay in view while the figure scrolls (wide screens) */}
        <div className="z-20 lg:sticky lg:top-[calc(var(--topbar-h)+8px)]">
          <div className="rounded-xl border border-border bg-surface shadow-[var(--shadow-card)]">
            <StepPlayer {...player} unit={granularity === 'rows' ? 'step' : 'decision'} label="BFS replay controls" className="!rounded-b-none !border-0 !bg-transparent" />
            {/* the narration: announced when the reader steps, silent while it autoplays (one sentence every second would flood a screen reader) */}
            <div className="mx-3 flex flex-col gap-1.5 border-t border-grid pt-2 pb-2.5 text-[17px] leading-snug text-ink" aria-live={player.playing ? 'off' : 'polite'}>
              <div>{narration.text}</div>
              {(algIdx.length > 0 || chip) && (
                <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-[14px] text-muted">
                  {algIdx.length > 0 && (
                    <span>
                      Alg. 1 line{algIdx.length > 1 ? 's' : ''} {algIdx.join(', ')} <Cite kind="paper" ref="p.15 Alg. 1" />
                    </span>
                  )}
                  {chip && (
                    <span>
                      code <Cite kind="code" ref={`rustler/src/fly.rs:${chip}`} />
                    </span>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
        <div className="mt-2">
          <ActiveLines alg={narration.alg} code={narration.code} />
        </div>

        <div className="mt-4 grid min-w-0 gap-4 lg:grid-cols-12">
          {/* frontier state: a scrollable strip above the graph on narrow screens, the right column on wide ones */}
          <div className="order-1 min-w-0 lg:order-none lg:col-span-6 lg:col-start-7 lg:row-start-2">
            <div className="grid gap-3 sm:grid-cols-2" aria-label="Frontier state">
              <div className="min-w-0">
                <StackPanel {...common} />
              </div>
              <div className="min-w-0">
                <BucketsPanel {...common} />
              </div>
            </div>
          </div>
          <div className="order-2 min-w-0 lg:order-none lg:col-span-6 lg:col-start-1 lg:row-span-2 lg:row-start-1">
            <BfsGraph
              layout={layout}
              state={state}
              pdb={pdb}
              db={db}
              slots={slots}
              seed={ctx.params.seedNode}
              current={current}
              fresh={fresh}
              stepStart={step.start}
              cursor={cursor}
              candidates={candidates}
              cut={state.finished?.kind === 'full' && cut ? cut.node : null}
              hover={hover}
              onHover={setHover}
              onSelect={pl.jumpToNode}
            />
          </div>
          <div className="order-3 min-w-0 lg:order-none lg:col-span-6 lg:col-start-7 lg:row-start-1">
            <div className="scroll-quiet max-h-[340px] min-h-[120px] overflow-y-auto rounded-xl border border-border bg-surface-2/50 px-3 py-2.5" aria-label="Decision detail">
              <DecisionPanel step={step} iters={iters} ctx={ctx} db={db} pdb={pdb} task={task} slots={slots} before={before} state={state} hover={hover} onHover={setHover} />
            </div>
          </div>
        </div>

        <div className="mt-5">
          <div className="mb-1.5 flex flex-wrap items-baseline justify-between gap-2">
            <div className="text-[15px] font-semibold text-ink">The context, table by table</div>
            <div className="text-[14px] text-muted">Only the rows visited so far; struck-through cells produced no token.</div>
          </div>
          <ContextTables
            db={db}
            pdb={pdb}
            task={task}
            slots={slots}
            state={state}
            ctx={ctx}
            cursor={cursor}
            seed={ctx.params.seedNode}
            current={current}
            freshTokens={freshTokens}
            hover={hover}
            onHover={setHover}
            cut={cut}
            tokenCol={tokenCol}
            maxHeight={520}
            className="min-[1700px]:grid-cols-2"
          />
        </div>

        <div className="mt-5 grid min-w-0 gap-4 md:grid-cols-12">
          <div className="min-w-0 md:col-span-4">
            <CountersPanel state={state} ctx={ctx} slots={slots} hover={hover} onHover={setHover} />
          </div>
          <div className="min-w-0 md:col-span-8">
            <div className="mb-2 flex flex-wrap items-baseline justify-between gap-2">
              <div className="text-[15px] font-semibold text-ink">Token strip, in emission order</div>
              <div className="num text-[14px] text-muted">
                {state.emitted} / {ctx.params.L}
              </div>
            </div>
            {stripTokens.length > 0 ? (
              <TokenStrip
                tokens={stripTokens}
                tableSlots={slots}
                variant="cells"
                highlight={stripHighlight}
                onHover={(p) => setHover(p === null ? null : ctx.tokens[p]?.isPadding ? null : (ctx.tokens[p]?.nodeIdx ?? null))}
                label="Context tokens emitted so far"
              />
            ) : (
              <p className="text-[15px] text-muted">No tokens yet.</p>
            )}
          </div>
        </div>
      </div>
      <p className="mx-auto mt-3 max-w-[var(--prose-max)] text-[16px] leading-snug text-ink-2">
        Each step replays a decision recorded by the site’s port of <span className="font-mono">fly.rs</span>, so the graph, the buckets and the tokens show the
        sampler’s real state. Hover over anything to find the same row in the other views, or click a row in the graph to jump to its visit.
      </p>
    </div>
  );
}
