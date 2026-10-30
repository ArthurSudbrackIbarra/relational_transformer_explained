/**
 * "Run the parity test in your browser": the same checks as src/engine/model/forward.test.ts (trained
 * fixtures), executed live on the reader's machine.
 *
 *   1. load a fixture written by python/parity.py (tests/fixtures/parity-trained*.{json,bin})
 *   2. hash the shipped checkpoint (public/data/weights/<ckpt>.bin) and compare with the fixture's sha256
 *   3. rebuild the exported context with the TS engine (preprocess + sampler) and compare every token field
 *   4. rebuild the four masks and compare them with the official dense masks of the fixture
 *   5. run the TS forward pass
 *   6. compare x0, every sublayer output, the final norm, the decoders, the loss and the prediction
 *
 * Nothing is precomputed: every number in this figure is measured when the reader presses "Run".
 */
import { motion, useReducedMotion } from 'motion/react';
import { CircleCheck, CircleDashed, CircleX, LoaderCircle, Play, RotateCcw } from 'lucide-react';
import { useEffect, useMemo, useRef, useState } from 'react';
import { Button, cx, Select, StatusBadge } from '@/components';
import { buildMasks } from '@/engine/masks';
import { loadEmbeddingStore } from '@/engine/embeddings';
import { forward, maxAbsDiff } from '@/engine/model';
import { baseUrl } from '@/engine/model/io';
import { loadCheckpoint, simulate, type CheckpointName } from '@/engine/pipeline';
import { ATTN_ORDER, SemType, type Split } from '@/engine/types';
import type { DbName } from '@/engine/db/registry';
import { resolveCheckpoint, useSeedNode, useSimParams } from '@/state/hooks';
import { int, sci, ulp32 } from './data';
import { LogDotChart, type LogDotRow } from './LogDotChart';

// ─────────────────────────────────────────────────────────────────────────────
// Fixtures (URLs only; the files are fetched when the reader asks)
// ─────────────────────────────────────────────────────────────────────────────

const URLS = import.meta.glob('/tests/fixtures/parity-trained*.{json,bin}', { query: '?url', import: 'default', eager: true }) as Record<string, string>;

interface FixtureFiles {
  name: string;
  json: string;
  bin: string;
}

const FIXTURES: FixtureFiles[] = (() => {
  const out: FixtureFiles[] = [];
  for (const [path, url] of Object.entries(URLS)) {
    const m = /\/(parity-trained(?:-[a-z]+)?)\.json$/.exec(path);
    if (!m || m[1].endsWith('-weights')) continue;
    const bin = URLS[path.replace(/\.json$/, '.bin')];
    if (bin) out.push({ name: m[1], json: url, bin });
  }
  // the default context first
  return out.sort((a, b) => (a.name === 'parity-trained' ? -1 : b.name === 'parity-trained' ? 1 : a.name.localeCompare(b.name)));
})();

interface Ref {
  offset: number;
  length: number;
}

interface TrainedFixture {
  meta: Record<string, unknown> & { torch?: string; official?: string; generator?: string };
  dbName?: string;
  taskId: string;
  split: Split;
  seedNode: number;
  L: number;
  w: number;
  rngSeed: number;
  checkpoint: string;
  checkpointSha256: string;
  S: number;
  targetPos: number;
  batch: {
    nodeIdxs: number[];
    f2pNbrIdxs: number[][];
    tableNameIdxs: number[];
    colNameIdxs: number[];
    semTypes: number[];
    values: number[];
    textIdxs: number[];
    flags: number[];
  };
  masks: Record<string, string>;
  sublayerOrder: string[];
  expected: {
    x0: Ref;
    sublayers: Record<string, Ref>;
    xFinal: Ref;
    yhat: { number: Ref; datetime: Ref; boolean: Ref };
    loss: number;
    prediction: number;
    predictionHead: string;
    label: number;
  };
}

const fixtureJsonCache = new Map<string, Promise<TrainedFixture>>();
function loadFixtureJson(f: FixtureFiles): Promise<TrainedFixture> {
  let p = fixtureJsonCache.get(f.name);
  if (!p) {
    p = fetch(f.json).then((r) => {
      if (!r.ok) throw new Error(`fetch ${f.name}.json: HTTP ${r.status}`);
      return r.json() as Promise<TrainedFixture>;
    });
    p.catch(() => fixtureJsonCache.delete(f.name));
    fixtureJsonCache.set(f.name, p);
  }
  return p;
}

function unpackMask(b64: string, S: number): Uint8Array {
  const raw = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
  const out = new Uint8Array(S * S);
  for (let i = 0; i < S * S; i++) out[i] = (raw[i >> 3] >> (i & 7)) & 1;
  return out;
}

async function sha256Hex(buf: ArrayBuffer): Promise<string | null> {
  const subtle = globalThis.crypto?.subtle;
  if (!subtle) return null;
  const d = await subtle.digest('SHA-256', buf);
  return [...new Uint8Array(d)].map((b) => b.toString(16).padStart(2, '0')).join('');
}

const DB_TITLE: Record<string, string> = { shop: 'ShopDB', school: 'SchoolDB', racing: 'RacingDB', forum: 'ForumDB' };

function fixtureLabel(fx: TrainedFixture): string {
  const db = DB_TITLE[fx.dbName ?? 'shop'] ?? fx.dbName;
  const task = fx.taskId.split('/')[1];
  return `${db} ${task} · ${fx.split} · seed node ${fx.seedNode} · ${fx.checkpoint}`;
}

// ─────────────────────────────────────────────────────────────────────────────
// The run
// ─────────────────────────────────────────────────────────────────────────────

type StepId = 'fixture' | 'weights' | 'context' | 'masks' | 'forward' | 'compare';
type StepState = 'pending' | 'running' | 'done' | 'fail';

const STEPS: { id: StepId; title: string }[] = [
  { id: 'fixture', title: 'Load the PyTorch reference' },
  { id: 'weights', title: 'Check the weights are the shipped checkpoint' },
  { id: 'context', title: 'Rebuild the context with the TypeScript sampler' },
  { id: 'masks', title: 'Rebuild the four attention masks' },
  { id: 'forward', title: 'Run the TypeScript forward pass' },
  { id: 'compare', title: 'Compare every activation' },
];

interface StageDiff {
  key: string;
  diff: number;
  refMax: number;
  n: number;
}

interface RunResult {
  fx: TrainedFixture;
  steps: Record<StepId, { state: StepState; detail?: string }>;
  sha?: { got: string | null; want: string };
  tokens?: { n: number; fields: number; mismatches: number };
  masks?: { entries: number; differing: number; perLayer: Record<string, number> };
  ms?: number;
  stages?: StageDiff[];
  prediction?: { ts: number; torch: number; head: string };
  loss?: { ts: number; torch: number };
  error?: string;
}

const TOL = 1e-4; // the test threshold in forward.test.ts

function emptySteps(): RunResult['steps'] {
  return Object.fromEntries(STEPS.map((s) => [s.id, { state: 'pending' as StepState }])) as RunResult['steps'];
}

export function BrowserParity() {
  const reduce = useReducedMotion();
  const sim = useSimParams();
  const liveSeed = useSeedNode();
  const [fixtures, setFixtures] = useState<Record<string, TrainedFixture>>({});
  const [choice, setChoice] = useState(FIXTURES[0]?.name ?? '');
  const [run, setRun] = useState<RunResult | null>(null);
  const [busy, setBusy] = useState(false);
  const alive = useRef(true);

  useEffect(() => {
    alive.current = true;
    Promise.all(FIXTURES.map((f) => loadFixtureJson(f).then((fx) => [f.name, fx] as const))).then(
      (pairs) => alive.current && setFixtures(Object.fromEntries(pairs)),
      () => {},
    );
    return () => {
      alive.current = false;
    };
  }, []);

  const files = FIXTURES.find((f) => f.name === choice);

  const start = async () => {
    if (!files || busy) return;
    setBusy(true);
    const pause = () => new Promise<void>((r) => window.setTimeout(r, reduce ? 0 : 260));
    let cur: RunResult | null = null;
    const update = (patch: (r: RunResult) => RunResult) => {
      if (!cur) return;
      cur = patch(cur);
      if (alive.current) setRun(cur);
    };
    const mark = (id: StepId, state: StepState, detail?: string) => update((r) => ({ ...r, steps: { ...r.steps, [id]: { state, detail } } }));
    try {
      // 1. fixture
      const fxJson = await loadFixtureJson(files);
      cur = { fx: fxJson, steps: emptySteps() };
      setRun(cur);
      mark('fixture', 'running');
      const binBuf = await fetch(files.bin).then((r) => {
        if (!r.ok) throw new Error(`fetch ${files.name}.bin: HTTP ${r.status}`);
        return r.arrayBuffer();
      });
      const bin = new Float32Array(binBuf);
      const ref = (r: Ref) => bin.subarray(r.offset, r.offset + r.length);
      const fx = fxJson;
      mark('fixture', 'done', `${int(binBuf.byteLength / 1024)} kB of float32 activations written by ${String(fx.meta.generator ?? 'python/parity.py')}`);
      await pause();

      // 2. weights
      mark('weights', 'running');
      const ckpt = await loadCheckpoint(fx.checkpoint as CheckpointName);
      const binUrl = `${baseUrl()}data/weights/${fx.checkpoint}.bin`;
      const wBuf = await fetch(binUrl).then((r) => (r.ok ? r.arrayBuffer() : Promise.reject(new Error(`fetch ${binUrl}: HTTP ${r.status}`))));
      const got = await sha256Hex(wBuf);
      update((r) => ({ ...r, sha: { got, want: fx.checkpointSha256 } }));
      if (ckpt.fallback) {
        mark('weights', 'fail', `checkpoint ${fx.checkpoint} could not be loaded`);
        throw new Error('weights unavailable');
      }
      if (got === null) mark('weights', 'done', 'SHA-256 is not available in this browser context; weights loaded but not hashed');
      else if (got !== fx.checkpointSha256) mark('weights', 'fail', `SHA-256 differs: the checkpoint was re-exported after this fixture`);
      else mark('weights', 'done', `public/data/weights/${fx.checkpoint}.bin · SHA-256 ${got.slice(0, 12)}… matches the fixture`);
      await pause();

      // 3. context
      mark('context', 'running');
      const s = simulate({
        dbName: (fx.dbName ?? 'shop') as DbName,
        taskId: fx.taskId,
        split: fx.split,
        seedNode: fx.seedNode,
        L: fx.L,
        w: fx.w,
        rngSeed: fx.rngSeed,
      });
      const b = fx.batch;
      let mism = 0;
      let fields = 0;
      s.ctx.tokens.forEach((t, i) => {
        const own = t.sem === SemType.Number ? t.numberValue : t.sem === SemType.DateTime ? t.datetimeValue : t.sem === SemType.Boolean ? t.booleanValue : 0;
        const flags = (t.mask ? 1 : 0) | (t.isTarget ? 2 : 0) | (t.isPadding ? 4 : 0) | (t.isTaskNode ? 8 : 0);
        const checks = [
          t.nodeIdx === b.nodeIdxs[i],
          JSON.stringify(t.f2pNbrIdxs) === JSON.stringify(b.f2pNbrIdxs[i]),
          t.tableNameIdx === b.tableNameIdxs[i],
          t.colNameIdx === b.colNameIdxs[i],
          t.sem === b.semTypes[i],
          Math.fround(own) === b.values[i],
          (t.isPadding ? -1 : t.textIdx) === b.textIdxs[i],
          flags === b.flags[i],
        ];
        fields += checks.length;
        mism += checks.filter((c) => !c).length;
      });
      const lengthOk = s.ctx.tokens.length === fx.S && s.ctx.targetPos === fx.targetPos;
      update((r) => ({ ...r, tokens: { n: s.ctx.tokens.length, fields, mismatches: mism + (lengthOk ? 0 : 1) } }));
      mark(
        'context',
        mism === 0 && lengthOk ? 'done' : 'fail',
        `${s.ctx.tokens.length} tokens × 8 fields = ${int(fields)} values compared, ${mism} differ · target at position ${s.ctx.targetPos}`,
      );
      await pause();

      // 4. masks
      mark('masks', 'running');
      const m = buildMasks(s.ctx.tokens);
      const perLayer: Record<string, number> = {};
      let differing = 0;
      for (const l of ATTN_ORDER) {
        const want = unpackMask(fx.masks[l], fx.S);
        let d = 0;
        for (let i = 0; i < want.length; i++) if (want[i] !== m[l][i]) d++;
        perLayer[l] = d;
        differing += d;
      }
      const entries = ATTN_ORDER.length * fx.S * fx.S;
      update((r) => ({ ...r, masks: { entries, differing, perLayer } }));
      mark('masks', differing === 0 ? 'done' : 'fail', `${int(entries)} mask entries (4 × ${fx.S} × ${fx.S}), ${differing} differ from the official dense masks`);
      await pause();

      // 5. forward
      mark('forward', 'running');
      await new Promise((r) => window.setTimeout(r, 30)); // let the spinner paint
      const store = await loadEmbeddingStore();
      const t0 = performance.now();
      const res = forward(s.ctx.tokens, s.pdb, store, ckpt.weights);
      const ms = performance.now() - t0;
      update((r) => ({ ...r, ms }));
      mark('forward', 'done', `${ms.toFixed(0)} ms on this device, every activation captured`);
      await pause();

      // 6. compare
      mark('compare', 'running');
      const e = fx.expected;
      const stages: StageDiff[] = [];
      const absMax = (a: Float32Array) => a.reduce((mx, v) => Math.max(mx, Math.abs(v)), 0);
      const push = (key: string, got: Float32Array, want: Float32Array) => stages.push({ key, diff: maxAbsDiff(got, want), refMax: absMax(want), n: want.length });
      push('x0', res.embedding.x0, ref(e.x0));
      res.blocks.forEach((blk, bi) => {
        for (const l of ATTN_ORDER) push(`block${bi}.${l}`, blk.attn[l].xOut, ref(e.sublayers[`block${bi}.${l}`]));
        push(`block${bi}.ffn`, blk.ffn.xOut, ref(e.sublayers[`block${bi}.ffn`]));
      });
      push('xFinal', res.xFinal, ref(e.xFinal));
      for (const h of ['number', 'datetime', 'boolean'] as const) push(`yhat.${h}`, res.yhat[h], ref(e.yhat[h]));
      stages.push({ key: 'loss', diff: Math.abs(res.loss - e.loss), refMax: Math.abs(e.loss), n: 1 });
      stages.push({ key: 'prediction', diff: Math.abs(res.prediction - e.prediction), refMax: Math.abs(e.prediction), n: 1 });
      const worst = stages.reduce((a, c) => (c.diff > a.diff ? c : a), stages[0]);
      const pass = stages.every((st) => st.diff <= TOL);
      update((r) => ({
        ...r,
        stages,
        prediction: { ts: res.prediction, torch: e.prediction, head: e.predictionHead },
        loss: { ts: res.loss, torch: e.loss },
      }));
      mark('compare', pass ? 'done' : 'fail', `${stages.length} comparisons · largest |Δ| = ${sci(worst.diff, 2)} at ${worst.key}`);
    } catch (err) {
      update((r) => ({ ...r, error: err instanceof Error ? err.message : String(err) }));
    } finally {
      if (alive.current) setBusy(false);
    }
  };

  const rows: LogDotRow[] = useMemo(
    () =>
      (run?.stages ?? []).map((st, i, arr) => {
        const grp = (k: string) => (k.startsWith('block') ? k.slice(0, 6) : k === 'x0' ? 'in' : 'out');
        return {
          id: st.key,
          label: st.key,
          value: st.diff,
          ref: st.refMax > 0 ? ulp32(st.refMax) : undefined,
          sepBefore: i > 0 && grp(arr[i - 1].key) !== grp(st.key),
          extra: [
            { k: 'largest |value|', v: st.refMax.toPrecision(4) },
            { k: 'values compared', v: int(st.n) },
            ...(st.refMax > 0 ? [{ k: 'in float32 steps', v: (st.diff / ulp32(st.refMax)).toFixed(1) }] : []),
          ],
        };
      }),
    [run?.stages],
  );

  const fx = fixtures[choice];
  const isLive =
    !!fx &&
    sim.taskId === fx.taskId &&
    sim.split === fx.split &&
    liveSeed === fx.seedNode &&
    sim.L === fx.L &&
    sim.w === fx.w &&
    sim.rngSeed === fx.rngSeed &&
    resolveCheckpoint(sim.taskId, sim.ckpt) === fx.checkpoint;
  const done = run && STEPS.every((s) => run.steps[s.id].state === 'done');
  const failed = run && STEPS.some((s) => run.steps[s.id].state === 'fail');

  if (FIXTURES.length === 0) {
    return <StatusBadge status="warning">No parity fixtures found in tests/fixtures. Run python/parity.py --stage trained.</StatusBadge>;
  }

  return (
    <div className="min-w-0">
      <div className="flex flex-wrap items-end gap-x-4 gap-y-3">
        <Select
          label="Fixture"
          value={choice}
          onChange={(e) => {
            setChoice(e.target.value);
            setRun(null);
          }}
          disabled={busy}
          className="min-w-0 flex-1 basis-[300px]"
        >
          {FIXTURES.map((f) => (
            <option key={f.name} value={f.name}>
              {fixtures[f.name] ? fixtureLabel(fixtures[f.name]) : f.name}
            </option>
          ))}
        </Select>
        <Button variant="primary" onClick={start} disabled={busy || !fx} icon={busy ? <LoaderCircle size={16} className="animate-spin" /> : run ? <RotateCcw size={16} /> : <Play size={16} />}>
          {busy ? 'Running…' : run ? 'Run again' : 'Run the parity test'}
        </Button>
      </div>
      {fx && (
        <p className="mt-3 mb-0 text-[16px] leading-snug text-ink-2">
          Written by <span className="font-mono text-[15px] text-ink">{String(fx.meta.generator ?? 'python/parity.py')}</span> with torch{' '}
          <span className="num text-ink">{String(fx.meta.torch ?? '?')}</span> in float32 on CPU.{' '}
          {isLive ? (
            <strong className="text-ink">This is the context and checkpoint every chapter shows right now.</strong>
          ) : (
            <>It is not the context of your current simulation settings, and it does not need to be: the test only needs a real context.</>
          )}
        </p>
      )}

      <ol className="mt-5 grid gap-2 !p-0" style={{ listStyle: 'none' }} aria-live="polite">
        {STEPS.map((s, i) => {
          const st = run?.steps[s.id] ?? { state: 'pending' as StepState };
          return (
            <motion.li
              key={s.id}
              layout={!reduce}
              className={cx(
                '!m-0 flex items-start gap-3 rounded-lg border px-3.5 py-2.5 !pl-3.5 transition-colors',
                st.state === 'running' ? 'border-accent bg-accent-soft' : 'border-border bg-surface',
              )}
            >
              <span className="mt-[2px] shrink-0">
                {st.state === 'done' ? (
                  <CircleCheck size={19} style={{ color: 'var(--good)' }} aria-label="passed" />
                ) : st.state === 'fail' ? (
                  <CircleX size={19} style={{ color: 'var(--critical)' }} aria-label="failed" />
                ) : st.state === 'running' ? (
                  <LoaderCircle size={19} className="animate-spin text-accent" aria-label="running" />
                ) : (
                  <CircleDashed size={19} className="text-axis" aria-label="pending" />
                )}
              </span>
              <span className="min-w-0 flex-1">
                <span className="block text-[16.5px] font-semibold text-ink">
                  <span className="num mr-2 text-muted">{i + 1}</span>
                  {s.title}
                </span>
                {st.detail && <span className="mt-0.5 block text-[15px] leading-snug break-words text-ink-2">{st.detail}</span>}
              </span>
            </motion.li>
          );
        })}
      </ol>

      {run?.error && (
        <div className="mt-3">
          <StatusBadge status="critical">The run stopped: {run.error}</StatusBadge>
        </div>
      )}

      {run?.stages && (
        <motion.div initial={reduce ? false : { opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.35 }} className="mt-6">
          <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
            <div className="text-[17px] font-semibold text-ink">Largest |TS − PyTorch| at each stage</div>
            {done && <StatusBadge status="good">All {run.stages.length} stages within 10⁻⁴</StatusBadge>}
            {failed && <StatusBadge status="critical">Some checks failed</StatusBadge>}
          </div>
          <div className="mt-1.5 mb-2 flex flex-wrap gap-x-5 gap-y-1 text-[15px] text-ink-2">
            <span className="inline-flex items-center gap-2">
              <svg width="14" height="14" aria-hidden="true">
                <circle cx="7" cy="7" r="5" fill="var(--accent)" />
              </svg>
              largest |Δ| over all values of the stage
            </span>
            <span className="inline-flex items-center gap-2">
              <svg width="10" height="16" aria-hidden="true">
                <line x1="5" x2="5" y1="2" y2="14" stroke="var(--muted)" strokeWidth="2" strokeLinecap="round" />
              </svg>
              one float32 step at the stage’s largest |value|
            </span>
          </div>
          <LogDotChart
            rows={rows}
            thresholds={[{ value: TOL, label: 'test tolerance 10⁻⁴' }]}
            valueName="max |TS − PyTorch|"
            refName="one float32 step"
            ariaLabel="Largest absolute difference between the TypeScript and PyTorch activations at each stage, log scale"
          />
          {run.prediction && (
            <div className="mt-4 grid gap-3 sm:grid-cols-2">
              <div className="rounded-lg border border-border bg-surface-2 px-3.5 py-2.5">
                <div className="text-[15px] text-ink-2">
                  Prediction at the masked cell ({run.prediction.head === 'boolean' ? 'logit' : 'z-score'})
                </div>
                <div className="num mt-1 text-[16px] text-ink">
                  TypeScript <span className="font-semibold">{run.prediction.ts.toPrecision(9)}</span>
                </div>
                <div className="num text-[16px] text-ink">
                  PyTorch&nbsp;&nbsp;&nbsp; <span className="font-semibold">{run.prediction.torch.toPrecision(9)}</span>
                </div>
              </div>
              {run.loss && (
                <div className="rounded-lg border border-border bg-surface-2 px-3.5 py-2.5">
                  <div className="text-[15px] text-ink-2">Loss on the masked cell</div>
                  <div className="num mt-1 text-[16px] text-ink">
                    TypeScript <span className="font-semibold">{run.loss.ts.toPrecision(9)}</span>
                  </div>
                  <div className="num text-[16px] text-ink">
                    PyTorch&nbsp;&nbsp;&nbsp; <span className="font-semibold">{run.loss.torch.toPrecision(9)}</span>
                  </div>
                </div>
              )}
            </div>
          )}
        </motion.div>
      )}
      {!run && (
        <p className="mt-4 mb-0 text-[15.5px] text-ink-2">
          The run downloads one fixture file of float32 activations and reuses the embeddings the site already loaded. Its six steps are the
          trained-weights block of <span className="font-mono text-[15px]">src/engine/model/forward.test.ts</span>, run on your device.
        </p>
      )}
    </div>
  );
}
