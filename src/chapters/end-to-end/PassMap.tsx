/**
 * The whole pass at a glance: the seven stages of the page with what goes in and what comes out,
 * the learned weights each stage uses, and the live result for the current seed. Shapes are derived
 * from the model configs (this site's MINI_CONFIG / loaded weights, or the paper's PAPER_CONFIG with
 * L = 1024, w = 256); weight counts from parameterShapes (the exact model.py inventory, spec §7.3).
 */
import { ArrowDown, Boxes, Gauge, Grid3x3, Layers, Sigma, Table2, Waypoints, type LucideIcon } from 'lucide-react';
import { useMemo, useState, type ReactNode } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Figure, Segmented, cx } from '@/components';
import { PAPER_CONFIG, PAPER_L, PAPER_W, parameterShapes } from '@/engine/model/config';
import { ATTN_ORDER, type RTConfig } from '@/engine/types';
import { fmtNum, fmtSigned, int, pct } from './format';
import { useE2E } from './live';

interface StageSpec {
  id: string;
  title: string;
  icon: LucideIcon;
  io: (c: Dims) => ReactNode;
  weights: (c: RTConfig) => number | null;
  note: ReactNode;
}

interface Dims {
  L: number;
  w: number;
  d: number;
  H: number;
  dh: number;
  dFF: number;
  dText: number;
  n: number;
}

function countWhere(c: RTConfig, test: (name: string) => boolean): number {
  let n = 0;
  for (const [name, shape] of parameterShapes(c)) if (test(name)) n += shape.reduce((a, b) => a * b, 1);
  return n;
}

const Sh = ({ children }: { children: ReactNode }) => <span className="num whitespace-nowrap text-ink">{children}</span>;

export function PassMap() {
  const live = useE2E();
  const { ctx, task, cfg, numReal, masks, target, pred, tokenLabels, db, pdb, seedNode } = live;
  const [mode, setMode] = useState<'ours' | 'paper'>('ours');
  const { pathname } = useLocation();
  const dims: Dims =
    mode === 'ours'
      ? { L: ctx.params.L, w: ctx.params.w, d: cfg.dModel, H: cfg.nHeads, dh: cfg.dModel / cfg.nHeads, dFF: cfg.dFF, dText: cfg.dText, n: cfg.nBlocks }
      : { L: PAPER_L, w: PAPER_W, d: PAPER_CONFIG.dModel, H: PAPER_CONFIG.nHeads, dh: PAPER_CONFIG.dModel / PAPER_CONFIG.nHeads, dFF: PAPER_CONFIG.dFF, dText: PAPER_CONFIG.dText, n: PAPER_CONFIG.nBlocks };
  const conf = mode === 'ours' ? cfg : PAPER_CONFIG;
  const allowed = useMemo(() => {
    const S = ctx.tokens.length;
    const out: Record<string, number> = {};
    for (const l of ATTN_ORDER) {
      let n = 0;
      for (let i = 0; i < S * S; i++) n += masks[l][i];
      out[l] = n;
    }
    return out;
  }, [ctx, masks]);
  const seed = pdb.nodes[seedNode];
  const targetTok = target >= 0 ? ctx.tokens[target] : null;

  const stages: StageSpec[] = [
    {
      id: 'question',
      title: 'Pick a task and a seed row',
      icon: Table2,
      io: () => (
        <>
          a seed row → <Sh>1</Sh> masked cell
        </>
      ),
      weights: () => null,
      note: (
        <>
          {task.id.split('/')[1]} · {seed ? (seed.tableType !== 'Db' ? `${seed.tableName}:${seed.tableType}[${seed.rowPos}]` : `${seed.tableName}[${seed.rowPos}]`) : ''} ·{' '}
          <code>{task.targetCol}</code> = ?
        </>
      ),
    },
    {
      id: 'sample',
      title: 'Sample the context',
      icon: Waypoints,
      io: (c) => (
        <>
          BFS over rows (w = <Sh>{int(c.w)}</Sh>) → <Sh>L = {int(c.L)}</Sh> cells
        </>
      ),
      weights: () => null,
      note: (
        <>
          {int(ctx.visited.length)} rows of {db.title.split(' ')[0]} → {int(numReal)} cells
        </>
      ),
    },
    {
      id: 'tokens',
      title: 'Write the tokens',
      icon: Boxes,
      io: (c) => (
        <>
          15 arrays per context: <Sh>{int(c.L)}</Sh>, <Sh>{int(c.L)} × 5</Sh> or <Sh>{int(c.L)} × {c.dText}</Sh>
        </>
      ),
      weights: () => null,
      note: targetTok ? <>1 masked token, position {target} ({tokenLabels[target]})</> : null,
    },
    {
      id: 'embed',
      title: 'Embed every token',
      icon: Sigma,
      io: (c) => (
        <>
          <Sh>
            2 × {int(c.L)} × {c.dText}
          </Sh>{' '}
          → x₀ <Sh>{int(c.L)} × {c.d}</Sh>
        </>
      ),
      weights: (c) => countWhere(c, (n) => n.startsWith('enc_dict.') || n.startsWith('norm_dict.') || n.startsWith('mask_embs.')),
      note: <>{new Set(ctx.tokens.slice(0, numReal).map((t) => t.colNameIdx)).size} distinct column phrases</>,
    },
    {
      id: 'masks',
      title: 'Build the four masks',
      icon: Grid3x3,
      io: (c) => (
        <>
          ids → <Sh>4 × {int(c.L)} × {int(c.L)}</Sh> booleans
        </>
      ),
      weights: () => null,
      note: <>allowed pairs: {ATTN_ORDER.map((l) => `${l} ${int(allowed[l])}`).join(' · ')}</>,
    },
    {
      id: 'blocks',
      title: `Run the ${dims.n} blocks`,
      icon: Layers,
      io: (c) => (
        <>
          <Sh>{c.n} × 5</Sh> sublayers, {c.H} heads of {c.dh}, SwiGLU {c.d}→{c.dFF}→{c.d}; x stays <Sh>{int(c.L)} × {c.d}</Sh>
        </>
      ),
      weights: (c) => countWhere(c, (n) => n.startsWith('blocks.')),
      note: <>col → feat → nbr → full → SwiGLU, in every block</>,
    },
    {
      id: 'output',
      title: 'Decode and read the answer',
      icon: Gauge,
      io: (c) => (
        <>
          RMSNorm → ŷ <Sh>{int(c.L)} × 3</Sh> → <Sh>1</Sh> number
        </>
      ),
      weights: (c) => countWhere(c, (n) => n.startsWith('dec_dict.') || n === 'norm_out.weight'),
      note: pred ? (
        pred.taskKind === 'binary' ? (
          <>
            logit {fmtSigned(pred.output)} → p = {pct(pred.probability!)} (label {String(pred.label.raw)})
          </>
        ) : (
          <>
            z {fmtSigned(pred.output)} → {fmtNum(pred.value!, 2)} (label {String(pred.label.raw)})
          </>
        )
      ) : (
        <>loading the weights…</>
      ),
    },
  ];
  const total = countWhere(conf, () => true);

  return (
    <Figure
      wide
      title="The whole pass at a glance"
      controls={
        <Segmented
          label="Sizes"
          value={mode}
          onChange={setMode}
          options={[
            { value: 'ours', label: 'This site’s model' },
            { value: 'paper', label: 'The paper’s model' },
          ]}
        />
      }
      caption={
        <>
          Shapes of what flows between stages, and the learned weights each stage uses ({int(total)} in all for{' '}
          {mode === 'ours' ? 'this site’s model' : 'the paper’s model'}, including the text decoder that is never read). Click a stage to jump to it.
          The live column follows the seed chosen in stage 1.
        </>
      }
    >
      <ol className="!m-0 grid grid-cols-1 gap-1.5 !p-0" style={{ listStyle: 'none' }}>
        {stages.map((s, i) => {
          const Icon = s.icon;
          const wn = s.weights(conf);
          return (
            <li key={s.id} className="!m-0 !p-0">
              <Link
                to={`${pathname}?section=${s.id}`}
                data-plain
                onClick={(e) => {
                  e.preventDefault();
                  document.getElementById(s.id)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }}
                className="group grid grid-cols-1 gap-x-4 gap-y-1 rounded-xl border border-border bg-surface px-3.5 py-2.5 transition-colors hover:border-accent md:grid-cols-[minmax(0,1.05fr)_minmax(0,1.25fr)_minmax(0,1.15fr)] md:items-center"
              >
                <span className="flex min-w-0 items-center gap-2.5">
                  <span className="num flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-accent-soft text-[14.5px] font-semibold text-accent-text">{i + 1}</span>
                  <Icon aria-hidden="true" size={16} className="shrink-0 text-ink-2" />
                  <span className="text-[16.5px] font-semibold text-ink group-hover:text-accent-text">{s.title}</span>
                </span>
                <span className="text-[15px] leading-snug text-ink-2">
                  {s.io(dims)}
                  {wn !== null && <span className="text-muted"> · {int(wn)} weights</span>}
                </span>
                <span className={cx('text-[15px] leading-snug', mode === 'ours' ? 'text-ink-2' : 'text-muted')}>
                  {mode === 'ours' ? s.note : 'live values: switch to this site’s model'}
                </span>
              </Link>
              {i < stages.length - 1 && (
                <div className="flex justify-start py-0.5 pl-[22px]" aria-hidden="true">
                  <ArrowDown size={14} className="text-axis" />
                </div>
              )}
            </li>
          );
        })}
      </ol>
    </Figure>
  );
}
