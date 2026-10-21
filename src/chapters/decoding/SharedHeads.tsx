/**
 * Step 6: one decoder per sem type, shared by every column, table and database (model.py:115-131;
 * spec §8.1). The same output value means different things for different target columns, because
 * each column is read in its own (μ, σ) (pre.rs:569, spec §8.4). Stats are computed live by the
 * preprocessing mirror for all four databases; the decoders' sizes come from the parameter inventory
 * of model/config.ts.
 */
import { useEffect, useMemo, useState } from 'react';
import { Figure, ScaleNote, SemBadge, Slider, StatusBadge } from '@/components';
import { getDatabase, listDatabases } from '@/engine/db/registry';
import { MINI_CONFIG, PAPER_CONFIG, parameterShapes } from '@/engine/model/config';
import { tensor } from '@/engine/model/weights';
import { getPreprocessed } from '@/engine/pipeline';
import { targetStats } from '@/engine/sampler';
import { SemType, type ColumnStats, type RTConfig, type TaskDef } from '@/engine/types';
import { fmt, fmtS, int, taskLabel, useAssetsFor, type Lane } from './live';
import { ratioStats, sigmoid } from './math';

interface Row {
  db: string;
  dbTitle: string;
  role: 'target' | 'pretrain';
  task: TaskDef;
  stats: ColumnStats;
}

function useAllTaskStats(): Row[] | null {
  const [rows, setRows] = useState<Row[] | null>(null);
  useEffect(() => {
    let dead = false;
    const out: Row[] = [];
    const dbs = listDatabases();
    let i = 0;
    const next = () => {
      if (dead) return;
      if (i >= dbs.length) {
        setRows(out);
        return;
      }
      const info = dbs[i++];
      const db = getDatabase(info.name as 'shop');
      const pdb = getPreprocessed(db);
      for (const t of db.tasks) {
        try {
          out.push({
            db: info.name,
            dbTitle: info.title,
            role: info.role,
            task: t,
            stats: targetStats(pdb, t),
          });
        } catch {
          /* no stats: skip */
        }
      }
      window.setTimeout(next, 0);
    };
    const id = window.setTimeout(next, 60);
    return () => {
      dead = true;
      window.clearTimeout(id);
    };
  }, []);
  return rows;
}

function fmtUnits(v: number): string {
  const a = Math.abs(v);
  return fmt(v, a >= 100 ? 0 : a >= 10 ? 1 : 2);
}

function HeadName({ binary }: { binary: boolean }) {
  return (
    <span className="inline-flex items-center gap-1.5">
      <SemBadge sem={binary ? SemType.Boolean : SemType.Number} size="sm" />
      <span className="font-mono text-ink">{binary ? 'boolean' : 'number'}</span>
    </span>
  );
}

/** What decoder output `s` means for one target column. */
function Meaning({ r, s }: { r: Row; s: number }) {
  if (r.task.targetType === 'binary')
    return (
      <>
        p({r.task.targetCol} = true) = <span className="num font-semibold">{fmt(sigmoid(s), 3)}</span>
        <div className="text-[14px] text-ink-2">
          labels stored as true ↦ {fmtS((1 - r.stats.mean) / r.stats.std, 2)}, false ↦ {fmtS(-r.stats.mean / r.stats.std, 2)}
        </div>
      </>
    );
  return (
    <>
      {r.task.targetCol} = <span className="num font-semibold">{fmtUnits(s * r.stats.std + r.stats.mean)}</span>
      <div className="text-[14px] text-ink-2">
        {fmtS(s, 1)} × {fmt(r.stats.std, 2)} + {fmt(r.stats.mean, 2)}
      </div>
    </>
  );
}

export function SharedHeadsFigure() {
  const rows = useAllTaskStats();
  const [s, setS] = useState(1);
  return (
    <Figure
      wide
      title="One output value, read in each target column’s own scale"
      controls={
        <Slider
          label="Decoder output"
          value={s}
          min={-2}
          max={3}
          step={0.1}
          onChange={setS}
          format={(v) => fmtS(v, 1)}
          className="w-full max-w-[340px]"
        />
      }
      caption={
        <>
          Every target of every task in the four databases. The same decoder serves all the rows of one type: its output is a z-score of the target
          column (number rows) or a logit (boolean rows). Statistics are recomputed live from each database’s preprocessing.
        </>
      }
    >
      {!rows ? (
        <div className="rounded-lg border border-dashed border-border-strong px-4 py-8 text-center text-[16px] text-ink-2">
          Preprocessing the four databases…
        </div>
      ) : (
        <>
          <table className="hidden w-full border-collapse text-[15px] md:table">
            <thead>
              <tr className="border-b border-grid text-left text-ink-2">
                <th scope="col" className="py-2 pr-3 font-semibold">
                  Database
                </th>
                <th scope="col" className="py-2 pr-3 font-semibold">
                  Target column
                </th>
                <th scope="col" className="py-2 pr-3 font-semibold">
                  Head read
                </th>
                <th scope="col" className="py-2 pr-3 text-right font-semibold">
                  μ
                </th>
                <th scope="col" className="py-2 pr-3 text-right font-semibold">
                  σ
                </th>
                <th scope="col" className="py-2 font-semibold">
                  Output {fmtS(s, 1)} means
                </th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.task.id} className="border-b border-grid/70 align-top last:border-0">
                  <td className="py-2 pr-3 text-ink-2">
                    {r.dbTitle.split(' ')[0]}
                    {r.role === 'target' && <div className="text-[14px] text-accent-text">held out</div>}
                  </td>
                  <td className="py-2 pr-3">
                    <span className="font-mono text-ink">{taskLabel(r.task)}</span>
                    <div className="text-[14px] text-ink-2">
                      {r.task.kind}, {r.task.kind === 'forecast' ? 'Train stats' : 'all rows'}
                    </div>
                  </td>
                  <td className="py-2 pr-3">
                    <HeadName binary={r.task.targetType === 'binary'} />
                  </td>
                  <td className="num py-2 pr-3 text-right text-ink">{fmt(r.stats.mean, 3)}</td>
                  <td className="num py-2 pr-3 text-right text-ink">{fmt(r.stats.std, 3)}</td>
                  <td className="py-2 text-ink">
                    <Meaning r={r} s={s} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <ul className="!m-0 grid gap-2 !p-0 md:hidden" style={{ listStyle: 'none' }}>
            {rows.map((r) => (
              <li key={r.task.id} className="!m-0 rounded-lg border border-border bg-surface px-3 py-2.5 text-[15px]">
                <div className="flex flex-wrap items-center justify-between gap-x-3 gap-y-1">
                  <span className="font-mono text-ink">{taskLabel(r.task)}</span>
                  <HeadName binary={r.task.targetType === 'binary'} />
                </div>
                <div className="text-[14px] text-ink-2">
                  {r.dbTitle.split(' ')[0]}
                  {r.role === 'target' ? ' (held out)' : ''} · {r.task.kind} · μ = {fmt(r.stats.mean, 3)}, σ = {fmt(r.stats.std, 3)}
                </div>
                <div className="mt-1 text-ink">
                  <Meaning r={r} s={s} />
                </div>
              </li>
            ))}
          </ul>
        </>
      )}
    </Figure>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// The decoders' sizes, mini vs paper
// ─────────────────────────────────────────────────────────────────────────────

function decoderParams(cfg: RTConfig) {
  const out: Record<string, number> = {};
  for (const [name, shape] of parameterShapes(cfg)) {
    const m = /^dec_dict\.(\w+)\./.exec(name);
    if (m) out[m[1]] = (out[m[1]] ?? 0) + shape.reduce((a, b) => a * b, 1);
  }
  return out;
}

export function DecoderSizes() {
  const mini = decoderParams(MINI_CONFIG);
  const paper = decoderParams(PAPER_CONFIG);
  const keys = ['number', 'text', 'datetime', 'boolean'];
  const shape = (cfg: RTConfig, k: string) => `Linear(${cfg.dModel} → ${k === 'text' ? cfg.dText : 1})`;
  return (
    <div className="my-6">
      <div className="card px-4 py-3">
        <table className="w-full border-collapse text-[15px]">
          <thead>
            <tr className="border-b border-grid text-left text-ink-2">
              <th scope="col" className="py-1.5 pr-3 font-semibold">
                Decoder
              </th>
              <th scope="col" className="py-1.5 pr-3 font-semibold">
                Paper model
              </th>
              <th scope="col" className="py-1.5 font-semibold">
                This site
              </th>
            </tr>
          </thead>
          <tbody>
            {keys.map((k) => (
              <tr key={k} className="border-b border-grid/70 align-top last:border-0">
                <td className="py-2 pr-3">
                  <span className="inline-flex items-center gap-1.5">
                    <SemBadge sem={keys.indexOf(k) as SemType} size="sm" />
                    <span className="font-mono text-ink">{k}</span>
                  </span>
                </td>
                <td className="py-2 pr-3">
                  <div className="font-mono text-[14px] text-ink-2">{shape(PAPER_CONFIG, k)}</div>
                  <div className="num text-ink">{int(paper[k])} params</div>
                </td>
                <td className="py-2">
                  <div className="font-mono text-[14px] text-ink-2">{shape(MINI_CONFIG, k)}</div>
                  <div className="num text-ink">{int(mini[k])} params</div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        <div className="mt-1.5 text-[14px] text-ink-2">Each count includes the bias: d·out + out.</div>
      </div>
      <ScaleNote paper="d = 256" ours="d = 64" className="mt-3">
        The decoders differ only in their input width d. The text head’s output width is the MiniLM size, 384, in both.
      </ScaleNote>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// The text head never learns: compare the checkpoint with its initialisation
// ─────────────────────────────────────────────────────────────────────────────

export function TextHeadCheck({ lane }: { lane: Lane }) {
  const init = useAssetsFor('init');
  const w = lane.assets.weights;
  const w0 = init.weights;
  const check = useMemo(() => {
    if (!w || !w0 || lane.assets.fallback || init.fallback) return null;
    try {
      const tw = ratioStats(tensor(w0, 'dec_dict.text.weight'), tensor(w, 'dec_dict.text.weight'));
      const tb = ratioStats(tensor(w0, 'dec_dict.text.bias'), tensor(w, 'dec_dict.text.bias'));
      return {
        tw,
        tb,
        n: tensor(w, 'dec_dict.text.weight').length + tensor(w, 'dec_dict.text.bias').length,
      };
    } catch {
      return null;
    }
  }, [w, w0, lane.assets.fallback, init.fallback]);
  if (!check) return null;
  if (lane.ckpt === 'init')
    return <StatusBadge status="info">The loaded checkpoint is the initialisation itself, so there is nothing to compare.</StatusBadge>;
  const uniform = check.tw.maxDev < 1e-4 && Math.abs(check.tb.mean - check.tw.mean) < 1e-4;
  return (
    <div className="rounded-lg border border-border bg-surface px-4 py-3 text-[16px] leading-relaxed text-ink">
      <div className="mb-1 flex flex-wrap items-center gap-2">
        <span className="font-semibold">Checked on the loaded checkpoint ({lane.ckpt})</span>
        {uniform ? (
          <StatusBadge status="good">no gradient ever reached the text head</StatusBadge>
        ) : (
          <StatusBadge status="warning">not a uniform scaling</StatusBadge>
        )}
      </div>
      {uniform ? (
        <>
          Each of the {int(check.n)} text-decoder parameters equals its initial value times{' '}
          <span className="num font-semibold">{fmt(check.tw.mean, 4)}</span> (largest deviation {check.tw.maxDev.toExponential(0)}). A uniform shrink
          is exactly what weight decay alone does: AdamW multiplies every parameter by (1 − lr·wd) at each step, and with a zero gradient nothing else
          moves it. For comparison, the boolean head’s trained weights are no rescaling of their initial values at all: that head learned.
        </>
      ) : (
        <>The text head’s parameters are not a uniform rescaling of their initial values in this checkpoint.</>
      )}
    </div>
  );
}
