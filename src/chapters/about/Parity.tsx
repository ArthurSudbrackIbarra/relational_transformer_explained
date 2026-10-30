/**
 * The three implementations of the RT forward pass and the measured agreement between them, read from
 * public/data/verification.json (python/write_verification.py) and training/results.json.
 */
import { ArrowDown, Cpu, FileCode2, Globe } from 'lucide-react';
import type { ReactNode } from 'react';
import { ChartFrame, StatusBadge, cx } from '@/components';
import { bytes, int, maxCheckpointDiff, sci, useResults, useVerification, type VerificationJson } from './data';
import { LogDotChart } from './LogDotChart';

function ImplCard({ icon, name, file, lines, tone }: { icon: ReactNode; name: string; file: string; lines: ReactNode[]; tone: 'official' | 'site' }) {
  return (
    <div className="card flex h-full min-w-0 flex-col px-4 py-3.5" style={tone === 'official' ? { borderColor: 'var(--accent-ring)' } : undefined}>
      <div className="flex items-start gap-2.5">
        <span className="inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-surface-2 text-ink-2">{icon}</span>
        <div className="min-w-0">
          <div className="text-[17px] leading-tight font-semibold text-ink">{name}</div>
          <div className="mt-0.5 font-mono text-[14px] break-words text-ink-2">{file}</div>
        </div>
      </div>
      {tone === 'official' && (
        <span className="mt-2 inline-flex w-fit rounded-full border border-accent px-2 text-[13.5px] font-semibold text-accent-text">the reference</span>
      )}
      <ul className="mt-2.5 grid gap-1 !p-0 text-[15.5px] leading-snug text-ink-2" style={{ listStyle: 'none' }}>
        {lines.map((l, i) => (
          <li key={i} className="!m-0 flex gap-2 !pl-0">
            <span aria-hidden="true" className="mt-[0.55em] h-1 w-1 shrink-0 rounded-full bg-muted" />
            <span>{l}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

function DeltaPill({ value, loading }: { value: number | null; loading: boolean }) {
  return (
    <span className="relative inline-flex rounded-full border border-accent bg-surface px-3 py-0.5 text-[15px] font-semibold whitespace-nowrap text-ink tabular">
      {loading ? '…' : value === null ? '–' : `|Δ| ≤ ${sci(value, 1)}`}
    </span>
  );
}

/** Desktop connector: a bracket from the center of one card to the center of the next, with the bound. */
function Bracket({ value, what, loading }: { value: number | null; what: ReactNode; loading: boolean }) {
  return (
    <div className="flex min-w-0 flex-col items-center">
      <div aria-hidden="true" className="h-5 w-full rounded-b-xl border-2 border-t-0 border-dashed" style={{ borderColor: 'var(--accent)' }} />
      <span aria-hidden="true" className="h-3 w-0 border-l-2 border-dashed" style={{ borderColor: 'var(--accent)' }} />
      <DeltaPill value={value} loading={loading} />
      <div className="mt-1.5 max-w-[300px] text-center text-[15px] leading-snug text-ink-2">{what}</div>
    </div>
  );
}

/** Mobile connector between two stacked cards. */
function StackEdge({ value, what, loading }: { value: number | null; what: ReactNode; loading: boolean }) {
  return (
    <div className="flex flex-col items-center gap-1.5 py-2 text-center">
      <span aria-hidden="true" className="h-3 w-0 border-l-2 border-dashed" style={{ borderColor: 'var(--accent)' }} />
      <DeltaPill value={value} loading={loading} />
      <div className="max-w-[300px] text-[15px] leading-snug text-ink-2">{what}</div>
      <ArrowDown aria-hidden="true" size={18} className="text-accent" />
    </div>
  );
}

export function ThreeImplementations() {
  const { data: v, error } = useVerification();
  const { data: res } = useResults();
  const loading = !v && !error;
  const nCk = v?.details.checkpoints.length ?? v?.checkpoints.length;
  const nTsFix = v ? Object.keys(v.details.tsVsTorch).length : null;
  const sdpa = res?.config?.attention;
  const torchVer = v?.details.torch;
  const cards = [
    <ImplCard
      key="official"
      tone="official"
      icon={<FileCode2 size={17} aria-hidden="true" />}
      name="Official model"
      file="rt/model.py @2a899c8"
      lines={[
        <>The paper-era class; only RMSNorm ε is pinned (below)</>,
        <>Compiled FlexAttention with block masks</>,
        <>Run as the reference by python/parity.py{torchVer ? <>, torch {torchVer.split('+')[0]}</> : null}</>,
      ]}
    />,
    <ImplCard
      key="port"
      tone="site"
      icon={<Cpu size={17} aria-hidden="true" />}
      name="PyTorch port"
      file="python/rt_model.py"
      lines={[<>Same module names, so state_dicts load both ways</>, <>Dense masked attention instead of FlexAttention</>, <>Trains the model on CPU</>]}
    />,
    <ImplCard
      key="engine"
      tone="site"
      icon={<Globe size={17} aria-hidden="true" />}
      name="Browser engine"
      file="src/engine/model/forward.ts"
      lines={[<>Plain TypeScript on Float32Array</>, <>Softmax over each query’s allowed keys</>, <>Runs every forward pass on this site</>]}
    />,
  ];
  const e1 = {
    value: v?.official_vs_port_max_abs_diff ?? null,
    what: (
      <>
        same weights in both. The largest gap is with random weights; on {nCk ?? '…'} trained checkpoints × {v?.n_contexts ?? '…'} real
        contexts from all four databases it is ≤ {v ? sci(maxCheckpointDiff(v)) : '…'}
      </>
    ),
  };
  const e2 = {
    value: v?.ts_vs_torch_max_abs_diff ?? null,
    what: <>every activation of {nTsFix ?? '…'} fixtures: embedding, all 20 sublayers, decoders, loss</>,
  };
  return (
    <div className="min-w-0">
      {/* desktop: three cards in a row, brackets underneath */}
      <div className="hidden md:block">
        <div className="grid grid-cols-3 gap-4">{cards}</div>
        <div className="grid grid-cols-6">
          <div className="col-span-2 col-start-2 pr-1.5">
            <Bracket {...e1} loading={loading} />
          </div>
          <div className="col-span-2 col-start-4 pl-1.5">
            <Bracket {...e2} loading={loading} />
          </div>
        </div>
      </div>
      {/* mobile: a vertical chain */}
      <div className="md:hidden">
        {cards[0]}
        <StackEdge {...e1} loading={loading} />
        {cards[1]}
        <StackEdge {...e2} loading={loading} />
        {cards[2]}
      </div>
      {error && (
        <div className="mt-3">
          <StatusBadge status="warning">verification.json could not be loaded: {error}</StatusBadge>
        </div>
      )}
      {sdpa && (
        <p className="mt-5 mb-0 text-[15.5px] leading-snug text-ink-2">
          For speed, the training run replaced the port’s dense attention with{' '}
          <span className="font-mono text-[15px]">F.scaled_dot_product_attention</span> and the same boolean mask. Before the switch, the two were
          compared on random data, fully masked rows included, and they agree to {sci(sdpa.maxAbsDiffForward)} on outputs and {sci(sdpa.maxAbsDiffGrad)} on gradients (
          <span className="font-mono text-[15px]">results.json → config.attention</span>).
        </p>
      )}
    </div>
  );
}

/** Official vs port, per exported checkpoint (dot plot + table twin with provenance and sha256). */
export function CheckpointParity() {
  const { data: v, error } = useVerification();
  if (error) return <StatusBadge status="warning">verification.json could not be loaded: {error}</StatusBadge>;
  if (!v) return <p className="text-[16px] text-ink-2">Loading the verification report…</p>;
  const cks = v.details.checkpoints;
  const rows = [
    ...cks.map((c) => ({
      id: c.name,
      label: c.name,
      value: c.officialVsPortMaxAbsDiff ?? v.details.officialVsPort.perCheckpoint[c.name] ?? NaN,
      extra: [
        { k: 'step', v: c.step === null ? '–' : int(c.step) },
        { k: 'files (.bin + .json)', v: bytes(c.bytes) },
      ],
    })),
    { id: 'random', label: 'random weights', plain: true, value: v.details.officialVsPort.randomWeights, extra: [], sepBefore: true },
  ].filter((r) => Number.isFinite(r.value));
  return (
    <ChartFrame
      title="Official model.py vs the PyTorch port, per checkpoint"
      subtitle={<>largest |Δ| over all decoder outputs and the loss, {v.n_contexts} contexts per checkpoint</>}
      table={<CheckpointTable v={v} />}
    >
      <LogDotChart
        rows={rows}
        thresholds={[{ value: 1e-5, label: 'asserted ≤ 10⁻⁵' }]}
        valueName="|official − port|"
        ariaLabel="Largest absolute difference between the official model and the port for each checkpoint, log scale"
        domainExp={[-8, -4]}
      />
    </ChartFrame>
  );
}

function CheckpointTable({ v }: { v: VerificationJson }) {
  return (
    <table className="w-full min-w-[640px] border-collapse text-[15px]">
      <thead>
        <tr className="border-b border-grid text-left text-ink-2">
          <th className="py-1.5 pr-3 font-semibold">checkpoint</th>
          <th className="py-1.5 pr-3 font-semibold">how it was selected</th>
          <th className="py-1.5 pr-3 text-right font-semibold">step</th>
          <th className="py-1.5 pr-3 text-right font-semibold">max |Δ|</th>
          <th className="py-1.5 font-semibold">SHA-256 of the .bin</th>
        </tr>
      </thead>
      <tbody>
        {v.details.checkpoints.map((c) => (
          <tr key={c.name} className="border-b border-grid/70 align-top">
            <td className="py-1.5 pr-3 font-mono text-ink">{c.name}</td>
            <td className="py-1.5 pr-3 text-ink-2">{c.selection ?? '–'}</td>
            <td className="num py-1.5 pr-3 text-right">{c.step === null ? '–' : int(c.step)}</td>
            <td className="num py-1.5 pr-3 text-right">{sci(c.officialVsPortMaxAbsDiff, 1)}</td>
            <td className="py-1.5 font-mono text-[14px] text-ink-2" title={c.sha256}>
              {c.sha256.slice(0, 16)}…
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

/** TS vs PyTorch per fixture, from verification.json (the CI numbers; the live run above re-measures one). */
export function TsFixtureTable() {
  const { data: v } = useVerification();
  if (!v) return null;
  const entries = Object.entries(v.details.tsVsTorch);
  return (
    <div className="scroll-quiet overflow-x-auto">
      <table className="w-full min-w-[620px] border-collapse text-[15px]">
        <thead>
          <tr className="border-b border-grid text-left text-ink-2">
            <th className="py-1.5 pr-3 font-semibold">fixture</th>
            <th className="py-1.5 pr-4 text-right font-semibold">max |TS − PyTorch|</th>
            <th className="py-1.5 pr-3 font-semibold">at stage</th>
            <th className="py-1.5 pr-3 font-semibold">weights</th>
            <th className="py-1.5 text-right font-semibold">contexts</th>
          </tr>
        </thead>
        <tbody>
          {entries.map(([name, r]) => (
            <tr key={name} className="border-b border-grid/70">
              <td className="py-1.5 pr-3 font-mono whitespace-nowrap text-ink">{name}</td>
              <td className="num py-1.5 pr-4 text-right whitespace-nowrap text-ink">{sci(r.maxAbsDiff, 1)}</td>
              <td className="py-1.5 pr-3 font-mono text-ink-2">{r.worstStage ?? '–'}</td>
              <td className={cx('py-1.5 pr-3 whitespace-nowrap text-ink-2', r.checkpoint && 'font-mono')}>{r.checkpoint ?? 'random'}</td>
              <td className="num py-1.5 text-right">{r.contexts}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="mt-2 mb-0 text-[14.5px] text-ink-2">
        Measured by <span className="font-mono">python/ts_parity.ts</span>; report written {v.date}.
      </p>
    </div>
  );
}
