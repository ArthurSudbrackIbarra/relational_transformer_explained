/**
 * Step 2 figure: the boolean head's output at the target is a logit ẑ; σ(ẑ) is the probability shown
 * to a person (spec §8.4). The hidden label is the stored z-scored boolean, and 1{z > 0} recovers
 * true/false because booleans are z-scored with 0 < μ < 1 (spec §3.3; model.py:253-255; main.py:336).
 */
import { useEffect, useState } from 'react';
import { Figure, Slider } from '@/components';
import { CurveChart, Readout } from './charts';
import { LaneHeader } from './LaneHeader';
import { fmt, fmtS, int, rawLabelText, type Lane } from './live';
import { sigmoid } from './math';

export function BinaryFigure({ lane }: { lane: Lane }) {
  const { pred, ctx, stats, task } = lane;
  const z = pred?.output ?? NaN;
  const p = pred?.probability ?? NaN;
  const tok = ctx.tokens[ctx.targetPos];
  const storedZ = tok?.booleanValue ?? NaN;
  const R = Number.isFinite(z) ? Math.max(6, Math.ceil(Math.abs(z) + 1.5)) : 6;
  const [tryZ, setTryZ] = useState(0);
  useEffect(() => {
    if (Number.isFinite(z)) setTryZ(Math.round(z * 20) / 20);
  }, [z]);
  const showGhost = Number.isFinite(z) && Math.abs(tryZ - z) > 0.03;
  const truePos = stats ? (1 - stats.mean) / stats.std : NaN;
  const falseNeg = stats ? -stats.mean / stats.std : NaN;
  const nTrue = stats ? Math.round(stats.mean * stats.count) : 0;

  return (
    <Figure
      wide
      title="From logit to probability"
      caption={
        <>
          The curve is σ(z) = 1 / (1 + exp(−z)). The filled dot is the boolean head’s output at the masked cell; the slider moves a hollow dot to
          explore the curve. A logit of 0 is a probability of exactly 0.5.
        </>
      }
    >
      <LaneHeader lane={lane} />
      {!pred ? (
        <div className="rounded-lg border border-dashed border-border-strong px-4 py-8 text-center text-[16px] text-ink-2">
          Running the forward pass…
        </div>
      ) : (
        <div className="grid gap-6 lg:grid-cols-[minmax(0,1.25fr)_minmax(0,1fr)] lg:gap-8">
          <div className="min-w-0">
            <CurveChart
              title="Sigmoid of the logit"
              series={[
                {
                  id: 'sig',
                  label: 'σ(z)',
                  color: 'var(--ink-2)',
                  fn: sigmoid,
                },
              ]}
              xDomain={[-R, R]}
              yDomain={[0, 1]}
              xLabel="logit z (boolean head output)"
              yLabel="σ(z)"
              height={300}
              hlines={[{ y: 0.5, label: 'p = 0.5' }]}
              marks={[
                ...(showGhost
                  ? [
                      {
                        x: tryZ,
                        y: sigmoid(tryZ),
                        kind: 'ghost' as const,
                        label: `p = ${fmt(sigmoid(tryZ), 2)}`,
                        guides: true,
                      },
                    ]
                  : []),
                {
                  x: z,
                  y: p,
                  kind: 'live' as const,
                  label: `ẑ = ${fmtS(z, 2)} → p = ${fmt(p, 2)}`,
                  guides: true,
                },
              ]}
              readout={(x) => (
                <>
                  z = <span className="num text-ink">{fmtS(x, 2)}</span> → σ(z) = <span className="num text-ink">{fmt(sigmoid(x), 3)}</span>
                </>
              )}
            />
            <Slider
              label="Try another logit"
              value={tryZ}
              min={-R}
              max={R}
              step={0.05}
              onChange={setTryZ}
              format={(v) => (
                <>
                  z = {fmtS(v, 2)} <span className="text-ink-2">→ p = {fmt(sigmoid(v), 3)}</span>
                </>
              )}
              className="mt-3 w-full max-w-[420px]"
            />
          </div>
          <div className="min-w-0">
            <div className="rounded-lg border border-border bg-surface px-3 py-1">
              <Readout label="Boolean head output at the target (a logit)" value={fmtS(z, 4)} strong />
              <Readout
                label={<>Probability σ(ẑ) that {task.targetCol} = true</>}
                value={fmt(p, 4)}
                note="For display only: the loss and the metric both use the logit itself."
              />
              <Readout
                label="Hidden label stored in the target token"
                value={`${fmtS(storedZ, 4)}`}
                note={
                  <>
                    The raw value is <span className="text-ink">{rawLabelText(tok?.raw ?? null)}</span>; it sits in{' '}
                    <span className="font-mono">boolean_values</span> but is gated out of the input.
                  </>
                }
              />
              <Readout label="Label used by loss and metric: 1{z > 0}" value={String(pred.label.binary ?? '–')} />
            </div>
            {stats && (
              <div className="mt-4 rounded-lg bg-surface-2 px-3 py-2.5 text-[15.5px] leading-snug text-ink-2">
                <div className="mb-1 font-semibold text-ink">
                  Why 1{'{'}z &gt; 0{'}'} is the label
                </div>
                The <span className="text-ink">{task.targetCol}</span> column was z-scored with its{' '}
                {task.kind === 'forecast' ? 'Train-split' : 'whole-table'} statistics: μ = <span className="num text-ink">{fmt(stats.mean, 4)}</span>{' '}
                (the share of true among {int(stats.count)} labels, {int(nTrue)} true) and σ ={' '}
                <span className="num text-ink">{fmt(stats.std, 4)}</span> (sample std). So true ↦ (1 − μ)/σ ={' '}
                <span className="num text-ink">{fmtS(truePos, 3)}</span> and false ↦ −μ/σ = <span className="num text-ink">{fmtS(falseNeg, 3)}</span>:
                the sign alone says which class it is.
              </div>
            )}
          </div>
        </div>
      )}
    </Figure>
  );
}
