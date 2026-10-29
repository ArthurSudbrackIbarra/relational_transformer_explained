/**
 * "What is scaled down or different": one card per difference with the paper's setting, this site's,
 * and whether it changes the mathematics, plus a live audit of what the random numbers decide in the
 * current context (the only place where Rust's ChaCha12 and this site's PRNG lead to different contexts).
 */
import { motion, useReducedMotion } from 'motion/react';
import { Dices, RotateCcw } from 'lucide-react';
import { useMemo, useState, type ReactNode } from 'react';
import { Button, Cite, SVG_TYPE, cx } from '@/components';
import { MINI_CONFIG, PAPER_CONFIG, paramCount, PAPER_L, PAPER_W, DEFAULT_L, DEFAULT_W } from '@/engine/model';
import { getContext, makeSampleParams } from '@/engine/pipeline';
import { useContextWindow, usePreprocessed, useSampleParams, useSeedNode, useSimParams, useTask } from '@/state/hooks';
import { useElementSize } from '@/components/viz/useSize';
import type { SamplerEvent } from '@/engine/types';
import { int, sci, useDataset, useResults } from './data';

type Verdict = 'same' | 'rounding' | 'draws' | 'values';

const VERDICT: Record<Verdict, { label: string; tone: string }> = {
  same: { label: 'same operations', tone: 'border-accent text-accent-text' },
  rounding: { label: 'rounding only', tone: 'border-accent text-accent-text' },
  draws: { label: 'different random draws', tone: 'border-border-strong text-ink' },
  values: { label: 'different numbers learned', tone: 'border-border-strong text-ink' },
};

function DiffCard({ title, paper, ours, verdict, children }: { title: string; paper: ReactNode; ours: ReactNode; verdict: Verdict; children: ReactNode }) {
  const v = VERDICT[verdict];
  return (
    <div className="card flex min-w-0 flex-col px-4 py-3.5">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h3 className="!m-0 text-[18px] font-semibold text-ink">{title}</h3>
        <span className={cx('rounded-full border px-2.5 py-px text-[14px] font-semibold', v.tone)}>{v.label}</span>
      </div>
      <dl className="mt-2.5 grid grid-cols-[auto_minmax(0,1fr)] gap-x-3 gap-y-1 text-[15.5px] leading-snug">
        <dt className="text-ink-2">Paper</dt>
        <dd className="!m-0 text-ink">{paper}</dd>
        <dt className="text-ink-2">This site</dt>
        <dd className="!m-0 text-ink">{ours}</dd>
      </dl>
      <div className="mt-2.5 text-[16px] leading-[1.55] text-ink-2">{children}</div>
    </div>
  );
}

export function DifferenceCards() {
  const { data: ds } = useDataset();
  const { data: res } = useResults();
  const tok = ds ? ds.paper.tokensSeen / ds.pretrain.tokensSeen : null;
  const sets = ds ? Object.entries(ds.sets).filter(([k]) => k.startsWith('eval/shop/')) : [];
  const churnVal = ds?.sets['eval/shop/customer-churn/Val']?.contexts;
  const churnTest = ds?.sets['eval/shop/customer-churn/Test']?.contexts;
  return (
    <div className="wide my-6 grid gap-4 md:grid-cols-2">
      <DiffCard
        title="Model size"
        verdict="same"
        paper={
          <>
            d = {PAPER_CONFIG.dModel}, {PAPER_CONFIG.nHeads} heads, {PAPER_CONFIG.nBlocks} blocks, {int(paramCount(PAPER_CONFIG))} parameters
          </>
        }
        ours={
          <>
            d = {MINI_CONFIG.dModel}, {MINI_CONFIG.nHeads} heads, {MINI_CONFIG.nBlocks} blocks, {int(paramCount(MINI_CONFIG))} parameters
          </>
        }
      >
        The same modules, with narrower tensors and fewer blocks. Both parameter counts are computed here from the module shapes of{' '}
        <span className="font-mono text-[15px]">model.py</span>, and the paper’s count matches its “22.3M” <Cite kind="spec" ref="§7.3" />{' '}
        <Cite kind="paper" ref="p.26 App. I.4" />.
      </DiffCard>
      <DiffCard
        title="Context size"
        verdict="same"
        paper={
          <>
            L = {PAPER_L}, w = {PAPER_W}
          </>
        }
        ours={
          <>
            L = {DEFAULT_L}, w = {DEFAULT_W}
          </>
        }
      >
        The sampler is the same algorithm at any (L, w); the cross-check above runs it at four settings, the paper’s among them{' '}
        <Cite kind="spec" ref="§4.1" />. This site’s databases are small, so a short context already reaches the seed’s neighborhood.
      </DiffCard>
      <DiffCard
        title="Data"
        verdict="values"
        paper={<>RelBench: 7 databases, pretraining on the 6 not under test</>}
        ours={<>4 synthetic databases, pretraining on the 3 that are not ShopDB</>}
      >
        Same protocol: leave one database out <Cite kind="paper" ref="p.6 §4.1" /> <Cite kind="spec" ref="§9.1" />. The learned weights and every
        metric are this site’s own, and they are never plotted on one axis with the paper’s tables.
      </DiffCard>
      <DiffCard
        title="Precision"
        verdict="rounding"
        paper={
          <>
            bfloat16 weights and optimizer state <Cite kind="code" ref="rt/main.py:186-187" />, cell values and text embeddings{' '}
            <Cite kind="code" ref="rt/embed.py:70-72" /> <Cite kind="spec" ref="§3.7" />; RMSNorm ε = 2⁻⁷
          </>
        }
        ours={<>float32 throughout, embeddings included; RMSNorm ε pinned to 2⁻⁷</>}
      >
        float32 keeps 24 significant bits where bfloat16 keeps 8, so the site’s numbers are closer to the exact result. ε is pinned so that RMSNorm
        divides by the same √(mean x² + ε) as in the paper’s runs <Cite kind="spec" ref="§7.2" />.
      </DiffCard>
      <DiffCard
        title="Random numbers"
        verdict="draws"
        paper={<>Rust StdRng (ChaCha12), seeded with epoch + seed row + seed</>}
        ours={<>xoshiro128** with the same seed formula, drawn at the same points</>}
      >
        Wherever the sampler draws (which child rows to keep, which frontier entry to expand), the two generators pick different rows, so a given
        context is not bit-identical to Rust’s. The distribution over contexts is the same <Cite kind="spec" ref="§4.2" />. The audit below shows
        what the draws decide.
      </DiffCard>
      <DiffCard
        title="Attention kernel"
        verdict="rounding"
        paper={<>compiled FlexAttention with block masks</>}
        ours={<>dense masked softmax (PyTorch port); per-query loops over allowed keys (browser)</>}
      >
        All three compute the same function. The scores are scaled by 1/√d_head, the softmax runs over the allowed keys only, and a query with no
        allowed key outputs 0 <Cite kind="spec" ref="§6.1" />. Only the order of floating-point additions differs, and the figure below checks it on
        the live context.
      </DiffCard>
      <DiffCard
        title="Training budget"
        verdict="values"
        paper={
          <>
            50,001 steps × 256 contexts × 1024 tokens on 8×A100, about 2 hours <Cite kind="paper" ref="p.6 §4.1" />
          </>
        }
        ours={
          ds && res?.pretrain ? (
            <>
              {int(ds.pretrain.steps)} steps × {ds.batchSize} × {ds.L} tokens on a CPU, {Math.round(res.pretrain.wallSeconds / 60)} minutes
            </>
          ) : (
            '…'
          )
        }
      >
        {tok ? (
          <>
            The paper’s model sees about {int(tok)} times more tokens ({sci(ds!.paper.tokensSeen, 1)} vs {sci(ds!.pretrain.tokensSeen, 1)}). The
            optimizer, schedule shape and selection rule are the same <Cite kind="spec" ref="§9.2" />.
          </>
        ) : (
          '…'
        )}
      </DiffCard>
      <DiffCard
        title="Evaluation sets"
        verdict="values"
        paper={<>at most 10,240 seed rows per split and task (40 batches × 32 seeds × 8 GPUs)</>}
        ours={
          churnVal !== undefined ? (
            <>
              {churnVal} / {churnTest} customer-churn rows (val / test), {sets.length} ShopDB sets in all
            </>
          ) : (
            '…'
          )
        }
      >
        The official code stops evaluating after 40 batches per GPU, so it scores a fixed subset of each split <Cite kind="spec" ref="§8.6" />.
        This site’s splits are tiny and evaluated in full, so a gap of a few AUROC points between two checkpoints is within noise.
      </DiffCard>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// What the random numbers decide (live)
// ─────────────────────────────────────────────────────────────────────────────

type PopEvent = Extract<SamplerEvent, { kind: 'pop' }>;
type SubEvent = Extract<SamplerEvent, { kind: 'db-children-subsampled' }>;

interface Draw {
  i: number;
  node: number;
  size: number;
  depth: number;
  /** The drawn row had already been visited: the draw was consumed, nothing was expanded (spec §4.4). */
  skipped: boolean;
}

/** One bar per frontier draw: how many candidates the PRNG chose from. */
function DrawBars({ draws, nodeName }: { draws: Draw[]; nodeName: (n: number) => string }) {
  const [ref, { width }] = useElementSize<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null);
  const H = 120;
  const top = 18;
  const bottom = 22;
  const left = 34;
  const maxSize = Math.max(2, ...draws.map((d) => d.size));
  const plotW = Math.max(40, width - left - 4);
  const bw = draws.length ? plotW / draws.length : plotW;
  const y = (v: number) => top + (H - top - bottom) * (1 - v / maxSize);
  const hv = hover !== null ? draws[hover] : null;
  const fill = (d: Draw) => (d.skipped ? 'none' : d.size > 1 ? 'var(--accent)' : 'var(--quiet-bar)');
  return (
    <div ref={ref} className="min-w-0">
      {width > 0 && (
        <svg width={width} height={H} role="img" aria-label="Number of candidates at each draw from the P→F frontier" onPointerLeave={() => setHover(null)}>
          {[0, maxSize].map((t) => (
            <g key={t}>
              <line x1={left} x2={width - 4} y1={y(t)} y2={y(t)} stroke="var(--grid)" strokeWidth={1} />
              <text x={left - 6} y={y(t)} dy="0.35em" textAnchor="end" fontSize={SVG_TYPE.tick} fill="var(--ink-2)" className="tabular">
                {t}
              </text>
            </g>
          ))}
          {draws.map((d, k) => {
            const x0 = left + k * bw;
            const h = Math.max(2, y(0) - y(d.size));
            return (
              <g key={d.i} onPointerEnter={() => setHover(k)}>
                <rect x={x0} y={top - 4} width={bw} height={H - top - bottom + 4} fill={hover === k ? 'var(--surface-3)' : 'transparent'} />
                <rect
                  x={x0 + Math.min(1, bw * 0.15)}
                  y={y(0) - h}
                  width={Math.max(1, bw - Math.min(2, bw * 0.3))}
                  height={h}
                  rx={Math.min(2, bw / 4)}
                  fill={fill(d)}
                  stroke={d.skipped ? 'var(--ink-2)' : 'none'}
                  strokeWidth={d.skipped ? 1.5 : 0}
                />
              </g>
            );
          })}
          <text x={left} y={H - 5} fontSize={SVG_TYPE.tick} fill="var(--ink-2)">
            draw 1
          </text>
          <text x={width - 4} y={H - 5} textAnchor="end" fontSize={SVG_TYPE.tick} fill="var(--ink-2)">
            draw {draws.length}
          </text>
        </svg>
      )}
      <div className="mt-1 min-h-[44px] text-[15px] leading-snug text-ink-2" aria-live="polite">
        {hv ? (
          <>
            Draw {hover! + 1}: {hv.size === 1 ? 'only one candidate' : <>one of {hv.size} candidates</>} at depth {hv.depth} →{' '}
            <span className="text-[14.5px] text-ink">{nodeName(hv.node)}</span>
            {hv.skipped ? ', already visited, so nothing was added.' : '.'}
          </>
        ) : (
          <>Hover a bar to see which row the draw picked.</>
        )}
      </div>
    </div>
  );
}

export function RandomnessAudit() {
  const reduce = useReducedMotion();
  const ctx = useContextWindow();
  const pdb = usePreprocessed();
  const task = useTask();
  const seed = useSeedNode();
  const params = useSampleParams();
  const { rngSeed: seedTerm } = useSimParams();
  const [offset, setOffset] = useState(0);
  const [showAll, setShowAll] = useState(false);

  const audit = useMemo(() => {
    const ev = ctx.events;
    const draws: Draw[] = [];
    let stackPops = 0;
    ev.forEach((e, i) => {
      if (e.kind !== 'pop') return;
      const p = e as PopEvent;
      if (p.from === 'stack') stackPops++;
      else draws.push({ i, node: p.node, size: p.bucketSize ?? 1, depth: p.depth, skipped: ev[i + 1]?.kind === 'skip-visited' });
    });
    const subs = (ev.filter((e) => e.kind === 'db-children-subsampled') as SubEvent[]).slice().sort((a, b) => b.candidates.length - a.candidates.length);
    const real = draws.filter((d) => d.size > 1);
    const sizes = real.map((d) => d.size).sort((a, b) => a - b);
    return {
      draws,
      subs,
      stackPops,
      real: real.length,
      skipped: draws.filter((d) => d.skipped).length,
      median: sizes.length ? sizes[Math.floor(sizes.length / 2)] : 0,
      largest: sizes.length ? sizes[sizes.length - 1] : 0,
    };
  }, [ctx]);

  const other = useMemo(() => {
    if (offset === 0) return null;
    const p = makeSampleParams(pdb, task, seed, params.L, params.w, seedTerm + offset);
    const c = getContext(pdb, p);
    const key = (t: { nodeIdx: number; colNameIdx: number }) => `${t.nodeIdx}:${t.colNameIdx}`;
    const a = new Set(ctx.tokens.filter((t) => !t.isPadding).map(key));
    const b = c.tokens.filter((t) => !t.isPadding).map(key);
    const shared = b.filter((k) => a.has(k)).length;
    const rowsA = new Set(ctx.visited.map((v) => v.node));
    const sharedRows = c.visited.filter((v) => rowsA.has(v.node)).length;
    return { shared, n: b.length, sharedRows, rows: c.visited.length, term: seedTerm + offset, prng: p.rngSeed };
  }, [offset, pdb, task, seed, params, seedTerm, ctx]);

  const nodeName = (n: number) => {
    const node = pdb.nodes[n];
    return node ? `${node.tableName}[${node.rowPos}]` : `#${n}`;
  };
  const maxSub = Math.max(1, ...audit.subs.map((s) => s.candidates.length));
  const SHOW = 5;
  const subsShown = showAll ? audit.subs : audit.subs.slice(0, SHOW);

  return (
    <div className="min-w-0" style={{ ['--quiet-bar' as string]: 'color-mix(in srgb, var(--muted) 62%, transparent)' }}>
      <div className="grid gap-3 lg:grid-cols-2">
        <div className="min-w-0 rounded-lg border border-border bg-surface-2 px-3.5 py-3">
          <div className="text-[16px] font-semibold text-ink">Width bound: which children to keep</div>
          <p className="mt-1 mb-2.5 text-[15.5px] leading-snug text-ink-2">
            {audit.subs.length === 0 ? (
              <>No row in this context had more than w = {params.w} database children, so the sampler never subsampled.</>
            ) : (
              <>
                {audit.subs.length} {audit.subs.length === 1 ? 'row had' : 'rows had'} more than w = {params.w} database children. For each, the
                PRNG chose which {params.w} to keep <Cite kind="code" ref="rustler/src/fly.rs:439-443" />.
              </>
            )}
          </p>
          <ul className="grid gap-2 !p-0" style={{ listStyle: 'none' }}>
            {subsShown.map((s, i) => (
              <li key={s.from} className="!m-0 !pl-0">
                <div className="flex items-baseline justify-between gap-3">
                  <span className="min-w-0 truncate text-[14.5px] text-ink">{nodeName(s.from)}</span>
                  <span className="num shrink-0 text-[14.5px] text-ink-2">
                    kept {s.kept.length} of {s.candidates.length}
                  </span>
                </div>
                <span className="mt-1 flex h-2.5 overflow-hidden rounded-full bg-surface-3" aria-hidden="true">
                  <motion.span
                    className="h-full rounded-l-full bg-accent"
                    initial={reduce ? false : { width: 0 }}
                    animate={{ width: `${(100 * s.kept.length) / maxSub}%` }}
                    transition={{ duration: 0.5, delay: reduce ? 0 : i * 0.04 }}
                  />
                  <span className="h-full" style={{ width: `${(100 * (s.candidates.length - s.kept.length)) / maxSub}%`, background: 'var(--quiet-bar)' }} />
                </span>
              </li>
            ))}
          </ul>
          {audit.subs.length > SHOW && (
            <Button size="sm" variant="ghost" className="mt-2" onClick={() => setShowAll((v) => !v)}>
              {showAll ? 'Show fewer' : `Show all ${audit.subs.length}`}
            </Button>
          )}
          {audit.subs.length > 0 && (
            <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-[14px] text-ink-2">
              <span className="inline-flex items-center gap-1.5">
                <span aria-hidden="true" className="h-2.5 w-4 rounded-sm bg-accent" /> kept
              </span>
              <span className="inline-flex items-center gap-1.5">
                <span aria-hidden="true" className="h-2.5 w-4 rounded-sm" style={{ background: 'var(--quiet-bar)' }} /> dropped by the draw
              </span>
            </div>
          )}
        </div>
        <div className="min-w-0 rounded-lg border border-border bg-surface-2 px-3.5 py-3">
          <div className="text-[16px] font-semibold text-ink">Frontier: which child row to expand next</div>
          <p className="mt-1 mb-2.5 text-[15.5px] leading-snug text-ink-2">
            {audit.draws.length} draws from the P→F frontier, uniform over the rows at the smallest depth{' '}
            <Cite kind="code" ref="rustler/src/fly.rs:390-395" />. {audit.real} were real choices
            {audit.real > 0 && (
              <>
                {' '}
                (median <span className="num text-ink">{audit.median}</span> candidates, largest{' '}
                <span className="num text-ink">{audit.largest}</span>)
              </>
            )}
            {audit.skipped > 0 && <>; {audit.skipped} picked an already visited row and added nothing</>}. The {audit.stackPops} pops from the
            F→P parent stack used no randomness <Cite kind="spec" ref="§4.4" />.
          </p>
          {audit.draws.length > 0 && <DrawBars draws={audit.draws} nodeName={nodeName} />}
          {audit.draws.length > 0 && (
            <div className="mt-1 flex flex-wrap gap-x-4 gap-y-1 text-[14px] text-ink-2">
              <span className="inline-flex items-center gap-1.5">
                <span aria-hidden="true" className="h-3 w-2.5 rounded-sm bg-accent" /> a real choice
              </span>
              <span className="inline-flex items-center gap-1.5">
                <span aria-hidden="true" className="h-3 w-2.5 rounded-sm" style={{ background: 'var(--quiet-bar)' }} /> one candidate
              </span>
              {audit.skipped > 0 && (
                <span className="inline-flex items-center gap-1.5">
                  <span aria-hidden="true" className="h-3 w-2.5 rounded-sm border-[1.5px] border-ink-2" /> already visited
                </span>
              )}
            </div>
          )}
        </div>
      </div>
      <div className="mt-4 flex flex-wrap items-center gap-3">
        <Button size="sm" icon={<Dices size={15} />} onClick={() => setOffset((o) => o + 1)}>
          Draw with another PRNG seed
        </Button>
        {offset > 0 && (
          <Button size="sm" variant="ghost" icon={<RotateCcw size={15} />} onClick={() => setOffset(0)}>
            Back to the live seed
          </Button>
        )}
      </div>
      <p className="mt-2.5 mb-0 text-[15.5px] leading-snug text-ink-2" aria-live="polite">
        {other ? (
          <>
            With the seed term {other.term} instead of {seedTerm} (PRNG seed {other.prng} instead of {params.rngSeed}), {other.sharedRows} of{' '}
            {other.rows} rows and {other.shared} of {other.n} cells also appear in the live context. The rules are the same and only the picks
            differ, as they would with a different generator such as Rust’s. Your global settings are unchanged.
          </>
        ) : (
          <>
            The live context uses PRNG seed {params.rngSeed} = seed row {seed} + seed term {seedTerm} (epoch 0). A different generator, like Rust’s,
            would make different picks at exactly these points. Try another seed here; your global settings stay as they are.
          </>
        )}
      </p>
    </div>
  );
}
