/**
 * Replay of the checkpoint selection of main.py:373-392 on our logged evaluations (curves.json):
 * at step 0, every eval_freq steps and every power of two, evaluate; for each target task, if the
 * val metric beats the best so far (strict >, starting at −∞), record that step's TEST metric and
 * save "<db>_<task>_best.pt". The reported number is the recorded test metric (main.py:441-448).
 */
import { motion, useReducedMotion } from 'motion/react';
import { Save } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import { Cite, Segmented, StepPlayer, useStepPlayer } from '@/components';
import { TrainChart } from './charts';
import { fx, int, mainRun, replaySelection, type Curves } from './data';

const TASKS = [
  { id: 'shop/customer-churn', label: 'customer-churn', metric: 'AUROC', ext: false },
  { id: 'shop/product-sales', label: 'product-sales', metric: 'R²', ext: false },
  { id: 'shop/reviews.rating', label: 'reviews.rating', metric: 'R²', ext: true },
  { id: 'shop/reviews.verified', label: 'reviews.verified', metric: 'AUROC', ext: true },
];

export function SelectionFigure({ curves, initialTask }: { curves: Curves | null; initialTask?: string }) {
  const run = mainRun(curves);
  const [task, setTask] = useState(TASKS.some((t) => t.id === initialTask) ? initialTask! : TASKS[0].id);
  const [xs, setXs] = useState<'linear' | 'log'>('linear');
  const meta = TASKS.find((t) => t.id === task)!;
  const ev = useMemo(() => replaySelection(run, task), [run, task]);
  const player = useStepPlayer(ev.length, { intervalMs: 450, initial: Math.max(0, ev.length - 1) });
  const reduce = useReducedMotion();
  useEffect(() => {
    player.setIndex(ev.length - 1);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [task, ev.length]);

  if (!run || ev.length === 0) return <div className="wide my-8 card px-5 py-6 text-[16px] text-ink-2">Loading curves.json…</div>;

  const cur = ev[Math.min(player.index, ev.length - 1)];
  const last = ev[ev.length - 1];
  const saves = ev.slice(0, player.index + 1).filter((e) => e.improved);
  const fmt = (v: number) => fx(v, 3);
  const xPlot = xs === 'log' ? (s: number) => s + 1 : undefined;
  const ckptName = `${task.replace('/', '_')}_best.pt`;
  const done = player.index >= ev.length - 1;

  return (
    <figure className="wide my-8 min-w-0">
      <div className="card px-4 py-4 sm:px-5 sm:py-5">
        <div className="mb-3 flex flex-wrap items-end gap-x-4 gap-y-3">
          <div className="mr-auto text-[17px] font-semibold text-ink">Replaying the checkpoint selection</div>
          <Segmented
            label="Target task"
            value={task}
            onChange={setTask}
            options={TASKS.map((t) => ({
              value: t.id,
              label: `${t.label}${t.ext ? '*' : ''}`,
              title: t.ext ? 'our extension: the paper selects forecast tasks only' : undefined,
            }))}
          />
          <Segmented
            label="Step axis"
            value={xs}
            onChange={setXs}
            options={[
              { value: 'linear', label: 'linear' },
              { value: 'log', label: 'log' },
            ]}
          />
        </div>
        <StepPlayer {...player} unit="eval" label="Selection replay" describe={undefined} />
        <div className="mt-4 grid gap-5 lg:grid-cols-[minmax(0,1.7fr)_minmax(0,1fr)]">
          <TrainChart
            title={`ShopDB ${meta.label}: zero-shot ${meta.metric} at each evaluation`}
            subtitle="the model has never seen ShopDB; no gradient comes from these numbers"
            series={[
              { id: 'val', label: `val ${meta.metric} (used to select)`, slot: 1, points: ev.map((e) => ({ x: e.step, y: e.val })), dots: true },
              {
                id: 'test',
                label: `test ${meta.metric} (recorded)`,
                slot: 2,
                dashed: true,
                points: ev.map((e) => ({ x: e.step, y: e.test })),
                dots: true,
              },
              { id: 'best', label: 'best val so far', slot: 1, raw: true, points: ev.map((e) => ({ x: e.step, y: e.bestVal })), hidden: true },
            ]}
            clipX={cur.step}
            xLabel={xs === 'log' ? 'step + 1 (log)' : 'pretraining step'}
            xScale={xs}
            xPlot={xPlot}
            xTicks={xs === 'log' ? [1, 10, 100, 1000, 4000].filter((t) => t <= last.step + 1).map((t) => t - 1) : undefined}
            formatX={(s) => (xs === 'log' ? int(s + 1) : int(s))}
            xDomain={xs === 'log' ? [1, last.step + 1] : [0, last.step]}
            yLabel={meta.metric}
            formatY={fmt}
            highlights={[{ x: cur.bestStep, y: cur.bestVal, slot: 1, label: done ? `selected: step ${int(cur.bestStep)}` : `best so far` }]}
            height={300}
          />
          <div className="min-w-0">
            <div className="grid grid-cols-2 gap-2">
              <div className="rounded-lg bg-surface-2 px-3 py-2">
                <div className="text-[14px] text-ink-2">evaluation at step</div>
                <div className="num text-[20px] font-semibold text-ink">{int(cur.step)}</div>
                <div className="num text-[14.5px] text-ink-2">
                  val {fmt(cur.val)} · test {fmt(cur.test)}
                </div>
              </div>
              <div className="rounded-lg bg-surface-2 px-3 py-2">
                <div className="text-[14px] text-ink-2">best val so far</div>
                <div className="num text-[20px] font-semibold text-ink">{fmt(cur.bestVal)}</div>
                <div className="num text-[14.5px] text-ink-2">
                  step {int(cur.bestStep)} · test {fmt(cur.bestTest)}
                </div>
              </div>
            </div>
            <motion.div
              key={`${task}-${cur.i}`}
              initial={reduce || !cur.improved ? false : { scale: 0.97, opacity: 0.4 }}
              animate={{ scale: 1, opacity: 1 }}
              className={`mt-2 rounded-lg border px-3 py-2 text-[15px] leading-snug ${cur.improved ? 'border-accent bg-accent-soft text-ink' : 'border-border text-ink-2'}`}
              aria-live="polite"
            >
              {cur.improved ? (
                <span className="inline-flex items-start gap-2">
                  <Save aria-hidden="true" size={16} className="mt-0.5 shrink-0 text-accent-text" />
                  <span>
                    {cur.i === 0 ? 'First evaluation: anything beats −∞.' : `${fmt(cur.val)} > ${fmt(ev[Math.max(0, ev.indexOf(cur) - 1)].bestVal)}.`}{' '}
                    Save <code className="break-all">{ckptName}</code> and record test = {fmt(cur.test)}.
                  </span>
                </span>
              ) : (
                <span>
                  {fmt(cur.val)} ≤ {fmt(cur.bestVal)}: not better, the recorded test stays {fmt(cur.bestTest)}.
                </span>
              )}
            </motion.div>
            <div className="mt-3 text-[15px] font-semibold text-ink">Saves so far: {saves.length}</div>
            <ol className="scroll-quiet !mt-1 !mb-0 max-h-[150px] overflow-y-auto !pl-0 text-[14.5px]" style={{ listStyle: 'none' }}>
              {[...saves].reverse().map((e) => (
                <li key={e.i} className="!m-0 flex justify-between gap-3 border-b border-grid/70 py-0.5 !pl-0 last:border-0">
                  <span className="num text-ink">step {int(e.step)}</span>
                  <span className="num text-ink-2">
                    val {fmt(e.val)} → test {fmt(e.test)}
                  </span>
                </li>
              ))}
            </ol>
          </div>
        </div>
        {done && (
          <div className="mt-4 grid gap-2 sm:grid-cols-3">
            <div className="rounded-lg border border-accent bg-surface px-3 py-2">
              <div className="text-[14px] text-ink-2">reported (best val step {int(last.bestStep)})</div>
              <div className="num text-[20px] font-semibold text-ink">test {fmt(last.bestTest)}</div>
            </div>
            <div className="rounded-lg border border-border px-3 py-2">
              <div className="text-[14px] text-ink-2">last step ({int(last.step)}) instead</div>
              <div className="num text-[20px] font-semibold text-ink">test {fmt(last.test)}</div>
            </div>
            <div className="rounded-lg border border-border px-3 py-2">
              <div className="text-[14px] text-ink-2">best test over all evals (not allowed)</div>
              <div className="num text-[20px] font-semibold text-ink">test {fmt(Math.max(...ev.map((e) => e.test)))}</div>
            </div>
          </div>
        )}
      </div>
      <figcaption className="mx-auto mt-3 max-w-[var(--prose-max)] text-[16px] leading-snug text-ink-2">
        {int(ev.length)} evaluations at step 0, every {int(run.hp?.eval_freq)} steps and at every power of two, before that step’s update{' '}
        <Cite kind="code" ref="rt/main.py:373-375" />. The rule is a strict “greater than”, starting from −∞{' '}
        <Cite kind="code" ref="rt/main.py:380-385" />.
        {meta.ext && ' * Our extension: the paper evaluates forecast tasks only; here “val” is the even review_ids and “test” the odd ones.'}
      </figcaption>
    </figure>
  );
}
