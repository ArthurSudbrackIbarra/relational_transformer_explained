/**
 * "What zero-shot means here":
 *  - LeaveOneOut: which databases the miniature was pretrained on and which one it never saw
 *    (paper p.6 §4.1 leave-one-DB-out; spec §9.1), read from the DB registry;
 *  - AskTheModel: pick any seed row of the current task split and run the SAME frozen weights on its
 *    context (spec §9.4). One forward pass per pick (~60-70 ms), capture off, cached by input, so it
 *    never touches the shared forward cache or the global seed unless the reader asks for it.
 */
import { ArrowDown, ArrowRight, Dices, Lock, StepBack, StepForward } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import { Button, IconButton, Select, TableSwatch, formatNumber, formatPct } from '@/components';
import { cx } from '@/components/ui/cx';
import { getDatabase, listDatabases, tableColorSlot } from '@/engine/db/registry';
import { MINI_CONFIG, forwardWithMasks, paramCount, textEmbedder } from '@/engine/model';
import {
  getContext,
  getMasks,
  inContextLabels,
  makeSampleParams,
  readPrediction,
  type InContextLabels,
  type PredictionReadout,
} from '@/engine/pipeline';
import { useShallow } from 'zustand/react/shallow';
import type { DbName } from '@/state/store';
import { useSim } from '@/state/store';
import { useSeeds } from '@/state/hooks';
import type { OverviewLive } from './live';
import { int, rowLabel } from './live';

// ─────────────────────────────────────────────────────────────────────────────
// Leave one database out
// ─────────────────────────────────────────────────────────────────────────────

function DbCard({ name, held }: { name: DbName; held?: boolean }) {
  const db = getDatabase(name);
  return (
    <div className={cx('min-w-0 rounded-xl border px-3.5 py-3', held ? 'border-accent bg-accent-soft' : 'border-border bg-surface')}>
      <div className="text-[16.5px] font-semibold text-ink">{db.title}</div>
      <div className="mt-1.5 flex flex-wrap gap-x-3 gap-y-1">
        {db.tables.map((t) => (
          <TableSwatch key={t.schema.name} slot={tableColorSlot(name, t.schema.name)} label={t.schema.name} />
        ))}
      </div>
      <div className="mt-1.5 text-[14.5px] leading-snug text-ink-2">
        tasks: {db.tasks.map((t) => (t.kind === 'autocomplete' ? t.id.split('/')[1] : t.name)).join(', ')}
      </div>
    </div>
  );
}

export function LeaveOneOut({ weightsKind }: { weightsKind: OverviewLive['assets']['weightsKind'] }) {
  const dbs = listDatabases();
  const pre = dbs.filter((d) => d.role === 'pretrain');
  const target = dbs.find((d) => d.role === 'target');
  const Arrow = ({ label }: { label: string }) => (
    <div className="flex items-center justify-center gap-2 text-[14.5px] text-ink-2 lg:flex-col">
      <ArrowRight aria-hidden="true" size={22} className="hidden text-ink-2 lg:block" />
      <ArrowDown aria-hidden="true" size={22} className="text-ink-2 lg:hidden" />
      <span className="text-center leading-snug lg:max-w-[120px]">{label}</span>
    </div>
  );
  return (
    <div className="grid items-center gap-4 lg:grid-cols-[minmax(0,1.25fr)_auto_minmax(0,0.8fr)_auto_minmax(0,1.1fr)]">
      <div className="grid gap-2.5">
        <div className="eyebrow">Pretraining databases</div>
        {pre.map((d) => (
          <DbCard key={d.name} name={d.name as DbName} />
        ))}
      </div>
      <Arrow label="masked-cell prediction on all their tasks" />
      <div className="rounded-xl border border-border-strong bg-surface-2 px-4 py-4 text-center">
        <Lock aria-hidden="true" size={20} className="mx-auto text-ink-2" />
        <div className="mt-1.5 text-[17px] font-semibold text-ink">Mini RT</div>
        <div className="num mt-0.5 text-[15px] text-ink-2">{int(paramCount(MINI_CONFIG))} parameters</div>
        <div className="mt-1.5 text-[14.5px] leading-snug text-ink-2">
          {weightsKind === 'init' ? 'untrained weights loaded' : 'weights frozen after pretraining'}
        </div>
      </div>
      <Arrow label="no gradient step, only a context" />
      <div className="grid gap-2.5">
        <div className="eyebrow">Held out: never seen in training</div>
        {target && <DbCard name={target.name as DbName} held />}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Ask the frozen model about another seed
// ─────────────────────────────────────────────────────────────────────────────

interface Answer {
  seed: number;
  pred: PredictionReadout | null;
  labels: InContextLabels;
  numReal: number;
}

const answerCache = new Map<string, Answer>();

export function AskTheModel({ live }: { live: OverviewLive }) {
  const { db, pdb, task, assets } = live;
  const seeds = useSeeds();
  const { L, w, rngSeed } = useSim(useShallow((s) => ({ L: s.L, w: s.w, rngSeed: s.rngSeed })));
  const setSeedNode = useSim((s) => s.setSeedNode);
  const [pick, setPick] = useState<number>(live.seedNode);
  useEffect(() => setPick(live.seedNode), [live.seedNode]);
  const seed = seeds.includes(pick) ? pick : live.seedNode;

  const options = useMemo(
    () =>
      seeds.map((s) => {
        const n = pdb.nodes[s];
        const ent = task.kind === 'forecast' && n?.f2pNbrIdxs.length ? n.f2pNbrIdxs[0] : null;
        return {
          seed: s,
          label: ent !== null ? rowLabel(db, pdb, ent).long : rowLabel(db, pdb, s).long,
        };
      }),
    [seeds, pdb, db, task],
  );

  const key = `${task.id}|${assets.ckpt}|${seed}|${L}|${w}|${rngSeed}`;
  const isLive = seed === live.seedNode;
  const [answer, setAnswer] = useState<Answer | null>(() => answerCache.get(key) ?? null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (isLive) return;
    const hit = answerCache.get(key);
    if (hit) {
      setAnswer(hit);
      return;
    }
    if (assets.status !== 'ready' || !assets.store || !assets.weights) return;
    setBusy(true);
    const store = assets.store;
    const weights = assets.weights;
    const t = window.setTimeout(() => {
      const ctx = getContext(pdb, makeSampleParams(pdb, task, seed, L, w, rngSeed));
      const fr = forwardWithMasks(ctx.tokens, getMasks(ctx), textEmbedder(pdb, store), weights, { capture: false });
      const a: Answer = {
        seed,
        pred: readPrediction(fr, ctx, pdb, task),
        labels: inContextLabels(pdb, ctx, task),
        numReal: ctx.numReal,
      };
      answerCache.set(key, a);
      if (answerCache.size > 200) answerCache.delete(answerCache.keys().next().value as string);
      setAnswer(a);
      setBusy(false);
    }, 30);
    return () => {
      window.clearTimeout(t);
      setBusy(false);
    };
  }, [key, isLive, assets, pdb, task, seed, L, w, rngSeed]);

  const shown: Answer | null = isLive
    ? { seed, pred: live.pred, labels: live.labels, numReal: live.ctx.numReal }
    : answer && answer.seed === seed
      ? answer
      : null;
  const idx = seeds.indexOf(seed);
  const go = (d: number) => setPick(seeds[(idx + d + seeds.length) % seeds.length]);
  const random = () => {
    if (seeds.length < 2) return;
    let r = seed;
    while (r === seed) r = seeds[Math.floor(Math.random() * seeds.length)];
    setPick(r);
  };
  const pred = shown?.pred ?? null;
  const self = shown ? (task.kind === 'forecast' ? shown.labels.self : shown.labels.other) : [];
  const selfSorted = [...self].sort((a, b) => (a.ts ?? 0) - (b.ts ?? 0));
  const slot = live.slots[task.name] ?? 5;

  return (
    <div>
      <div className="flex flex-wrap items-end gap-x-3 gap-y-2">
        <Select
          label={
            task.kind === 'forecast'
              ? `${task.entityTable ?? 'Entity'} (${seeds.length} ${pdb.nodes[seed]?.tableType ?? ''} rows)`
              : `Seed row (${seeds.length})`
          }
          value={seed}
          onChange={(e) => setPick(Number(e.target.value))}
          className="min-w-0 max-w-full"
        >
          {options.map((o) => (
            <option key={o.seed} value={o.seed}>
              {o.label}
            </option>
          ))}
        </Select>
        <div className="flex items-center gap-1">
          <IconButton label="Previous seed" onClick={() => go(-1)}>
            <StepBack size={16} />
          </IconButton>
          <IconButton label="Next seed" onClick={() => go(1)}>
            <StepForward size={16} />
          </IconButton>
          <Button size="sm" variant="secondary" icon={<Dices size={15} />} onClick={random}>
            Random
          </Button>
        </div>
      </div>

      <div className="mt-4 grid gap-4 md:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <div className="rounded-xl border border-border bg-surface px-4 py-3.5">
          <div className="text-[15px] text-ink-2">Zero-shot prediction</div>
          {!pred || busy ? (
            <div className="mt-2 text-[16px] text-ink-2">
              {assets.status === 'error' ? 'The weights failed to load.' : 'Running the forward pass…'}
            </div>
          ) : pred.taskKind === 'binary' ? (
            <>
              <div className="mt-1 flex items-baseline gap-2">
                <span className="text-[30px] leading-tight font-semibold text-ink">{formatPct(pred.probability ?? NaN)}</span>
                <span className="text-[15.5px] text-ink-2">P({task.targetCol} = true)</span>
              </div>
              <div className="relative mt-2 h-2.5 rounded-full bg-grid" aria-hidden="true">
                <div className="absolute inset-y-0 left-0 rounded-full bg-accent" style={{ width: `${(pred.probability ?? 0) * 100}%` }} />
                <div className="absolute -top-1 -bottom-1 left-1/2 w-px bg-ink-2" />
              </div>
              <div className="mt-2.5 text-[15.5px] text-ink-2">
                logit {formatNumber(pred.output, 3)} · true label <span className="font-mono text-ink">{String(pred.label.raw)}</span>
              </div>
            </>
          ) : (
            <>
              <div className="mt-1 flex items-baseline gap-2">
                <span className="text-[30px] leading-tight font-semibold text-ink">{formatNumber(pred.value ?? NaN, 2)}</span>
                <span className="text-[15.5px] text-ink-2">predicted {task.targetCol}</span>
              </div>
              <div className="mt-2.5 text-[15.5px] text-ink-2">
                z {formatNumber(pred.output, 3)} · true value <span className="font-mono text-ink">{formatNumber(Number(pred.label.raw), 2)}</span>
              </div>
            </>
          )}
        </div>
        <div className="rounded-xl border border-border bg-surface px-4 py-3.5">
          <div className="text-[15px] text-ink-2">What the context gave it</div>
          {shown ? (
            <>
              <div className="mt-1 text-[16.5px] text-ink">
                {int(shown.numReal)} cell tokens, {int(self.length)} visible {task.targetCol}{' '}
                {task.kind === 'forecast'
                  ? `${self.length === 1 ? 'label' : 'labels'} of this ${task.entityTable?.replace(/s$/, '') ?? 'entity'}`
                  : 'values of other rows'}
                {task.kind === 'forecast' && shown.labels.other.length > 0 && ` and ${int(shown.labels.other.length)} of other entities`}
              </div>
              {task.targetType === 'binary' && selfSorted.length > 0 && (
                <div className="mt-2.5">
                  <div className="flex flex-wrap gap-1" aria-label="Visible labels, oldest first">
                    {selfSorted.map((l) => {
                      const on = l.raw === true || l.raw === 1;
                      return (
                        <span
                          key={l.pos}
                          title={`${l.tableType} row ${l.rowPos}: ${String(l.raw)}`}
                          className="inline-block h-3.5 w-3.5 rounded-full border-2"
                          style={{
                            borderColor: `var(--cat-${slot})`,
                            background: on ? `var(--cat-${slot})` : 'var(--surface)',
                          }}
                        />
                      );
                    })}
                  </div>
                  <div className="mt-1.5 text-[14.5px] text-ink-2">
                    oldest first · filled = true ({self.filter((l) => l.raw === true || l.raw === 1).length}), hollow = false (
                    {self.filter((l) => !(l.raw === true || l.raw === 1)).length})
                  </div>
                </div>
              )}
            </>
          ) : (
            <div className="mt-2 text-[16px] text-ink-2">…</div>
          )}
        </div>
      </div>
      <div className="mt-3 flex flex-wrap items-center gap-x-3 gap-y-2 text-[15px] text-ink-2">
        {isLive ? (
          <span>This is the course’s running example.</span>
        ) : (
          <>
            <span>Same weights, a different context.</span>
            <Button size="sm" variant="secondary" onClick={() => setSeedNode(seed)}>
              Make this the running example for the whole course
            </Button>
          </>
        )}
      </div>
    </div>
  );
}
