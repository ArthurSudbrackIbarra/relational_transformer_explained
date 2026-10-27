/**
 * The pipeline map: what was built once offline (Node + Python, CPU) and what runs in the reader's browser
 * on every change, with the files that connect the two. Every stage's detail panel shows live numbers:
 * from the engine (databases, preprocessing, context, forward pass), from the shipped files
 * (embeddings, dataset.json, results.json, verification.json) or measured on the spot (SHA-256 of the
 * regenerated ShopDB, forward-pass time).
 */
import { motion, useReducedMotion } from 'motion/react';
import { ArrowDown, ArrowRight, Database, FileJson, Gauge, Hash, LoaderCircle } from 'lucide-react';
import { useEffect, useMemo, useState, type ReactNode } from 'react';
import { Button, Cite, StatusBadge, StepPlayer, cx, useStepPlayer } from '@/components';
import { getDatabase, listDatabases } from '@/engine/db/registry';
import { forward, MINI_CONFIG } from '@/engine/model';
import { useAssets, useContextWindow, useDatabase, usePrediction, usePreprocessed, useSeedNode, useTask } from '@/state/hooks';
import { bytes, int, sci, useDataset, useRawSource, useResults, useVerification } from './data';

type StageId = 'gen' | 'embed' | 'contexts' | 'train' | 'verify' | 'r-gen' | 'r-pre' | 'r-sample' | 'r-forward';

interface StageMeta {
  id: StageId;
  row: 'build' | 'run';
  n: string;
  title: string;
  where: string;
  files: string[];
  /** Same code as these stages (cross-highlighted). */
  twins?: StageId[];
  produces?: string;
}

const STAGES: StageMeta[] = [
  { id: 'gen', row: 'build', n: 'B1', title: 'Generate the databases', where: 'TypeScript · Node', files: ['src/engine/db/generators/*.ts'], twins: ['r-gen'] },
  {
    id: 'embed',
    row: 'build',
    n: 'B2',
    title: 'Embed every string',
    where: 'Node → Python',
    files: ['scripts/export-strings.ts', 'python/embed.py'],
    produces: 'embeddings',
  },
  {
    id: 'contexts',
    row: 'build',
    n: 'B3',
    title: 'Sample training contexts',
    where: 'TypeScript · Node',
    files: ['scripts/export-contexts.ts'],
    twins: ['r-pre', 'r-sample'],
  },
  { id: 'train', row: 'build', n: 'B4', title: 'Train on CPU', where: 'Python · PyTorch', files: ['python/train.py', 'python/rt_model.py'], produces: 'weights' },
  {
    id: 'verify',
    row: 'build',
    n: 'B5',
    title: 'Verify',
    where: 'Python + Node',
    files: ['python/parity.py', 'python/ts_parity.ts', 'python/sampler_crosscheck.py', 'vitest'],
    produces: 'verification',
  },
  { id: 'r-gen', row: 'run', n: 'R1', title: 'Regenerate ShopDB', where: 'TypeScript · browser', files: ['src/engine/db/registry.ts'], twins: ['gen'] },
  { id: 'r-pre', row: 'run', n: 'R2', title: 'Preprocess', where: 'TypeScript · browser', files: ['src/engine/preprocess.ts'], twins: ['contexts'] },
  { id: 'r-sample', row: 'run', n: 'R3', title: 'Sample the context', where: 'TypeScript · browser', files: ['src/engine/sampler.ts', 'src/engine/masks.ts'], twins: ['contexts'] },
  { id: 'r-forward', row: 'run', n: 'R4', title: 'Forward pass', where: 'TypeScript · browser', files: ['src/engine/model/forward.ts'] },
];

const ORDER: StageId[] = STAGES.map((s) => s.id);
const byId = (id: StageId) => STAGES.find((s) => s.id === id)!;

// ─────────────────────────────────────────────────────────────────────────────
// Cards
// ─────────────────────────────────────────────────────────────────────────────

function StageCard({ s, active, twin, onSelect, onHover }: { s: StageMeta; active: boolean; twin: boolean; onSelect: () => void; onHover: (h: boolean) => void }) {
  const reduce = useReducedMotion();
  return (
    <button
      type="button"
      onClick={onSelect}
      onPointerEnter={() => onHover(true)}
      onPointerLeave={() => onHover(false)}
      onFocus={() => onHover(true)}
      onBlur={() => onHover(false)}
      aria-pressed={active}
      className={cx(
        'relative flex h-full min-w-0 flex-col rounded-xl border bg-surface px-3 py-2.5 text-left transition-colors',
        active ? 'border-transparent' : twin ? 'border-accent' : 'border-border hover:border-border-strong',
      )}
    >
      {active && (
        <motion.span
          layoutId="pipeline-active"
          aria-hidden="true"
          className="pointer-events-none absolute inset-[-1px] rounded-xl border-2 border-accent"
          style={{ background: 'var(--accent-soft)' }}
          transition={reduce ? { duration: 0 } : { type: 'spring', stiffness: 420, damping: 34 }}
        />
      )}
      <span className="relative flex items-center gap-2">
        <span className="num text-[14px] font-semibold text-muted">{s.n}</span>
        {s.twins && s.twins.length > 0 && (
          <span className="rounded-full border border-dashed border-border-strong px-1.5 text-[13px] font-medium text-ink-2" title="The same TypeScript code runs offline and in your browser">
            shared code
          </span>
        )}
      </span>
      <span className="relative mt-1 block text-[16px] leading-snug font-semibold text-ink">{s.title}</span>
      <span className="relative mt-auto block pt-1 text-[14px] text-ink-2">{s.where}</span>
    </button>
  );
}

function Arrow({ dir = 'right' }: { dir?: 'right' | 'down' }) {
  return (
    <span aria-hidden="true" className="flex items-center justify-center text-axis">
      {dir === 'right' ? <ArrowRight size={18} className="hidden md:block" /> : null}
      <ArrowDown size={18} className={dir === 'right' ? 'md:hidden' : ''} />
    </span>
  );
}

const ARTIFACTS: { id: string; from: StageId; file: string; label: string }[] = [
  { id: 'embeddings', from: 'embed', file: 'embeddings/minilm-l12-v2.{f32,json}', label: 'text embeddings' },
  { id: 'weights', from: 'train', file: 'weights/*.{bin,json} + training/*.json', label: 'checkpoints and training logs' },
  { id: 'verification', from: 'verify', file: 'verification.json', label: 'the verification report' },
];

// ─────────────────────────────────────────────────────────────────────────────
// Detail panels (live)
// ─────────────────────────────────────────────────────────────────────────────

function KV({ k, v }: { k: ReactNode; v: ReactNode }) {
  return (
    <div className="flex min-w-0 items-baseline justify-between gap-3 border-b border-grid/70 py-1 last:border-0">
      <span className="text-[15px] text-ink-2">{k}</span>
      <span className="num text-right text-[15px] text-ink">{v}</span>
    </div>
  );
}

function DbTable() {
  const rows = useMemo(
    () =>
      listDatabases().map((d) => {
        const db = getDatabase(d.name);
        return { d, tables: db.tables.map((t) => ({ name: t.schema.name, n: t.rows.length })) };
      }),
    [],
  );
  return (
    <ul className="mt-1 grid gap-0 !p-0" style={{ listStyle: 'none' }}>
      {rows.map(({ d, tables }) => (
        <li key={d.name} className="!m-0 border-b border-grid/70 py-2 !pl-0 last:border-0">
          <div className="flex flex-wrap items-baseline gap-x-2.5">
            <span className="text-[16px] font-semibold text-ink">{d.title.split(' ')[0]}</span>
            <span
              className={cx(
                'rounded-full border px-2 text-[13.5px] font-medium',
                d.role === 'target' ? 'border-accent text-accent-text' : 'border-border text-ink-2',
              )}
            >
              {d.role === 'target' ? 'held out: zero-shot target' : 'pretraining'}
            </span>
          </div>
          <div className="mt-1 flex flex-wrap gap-x-3 gap-y-0.5 text-[15px] text-ink-2">
            {tables.map((t) => (
              <span key={t.name} className="whitespace-nowrap">
                {t.name} <span className="num text-ink">{int(t.n)}</span>
              </span>
            ))}
          </div>
        </li>
      ))}
    </ul>
  );
}

/** SHA-256 of JSON.stringify(getDatabase('shop')) in this browser vs the hash pinned in db.test.ts. */
function ShopHash() {
  const { data: testSrc } = useRawSource('dbTest');
  const [hash, setHash] = useState<string | null | undefined>(undefined);
  useEffect(() => {
    let alive = true;
    const subtle = globalThis.crypto?.subtle;
    if (!subtle) {
      setHash(null);
      return;
    }
    const t = window.setTimeout(() => {
      subtle.digest('SHA-256', new TextEncoder().encode(JSON.stringify(getDatabase('shop')))).then((d) => {
        if (alive) setHash([...new Uint8Array(d)].map((b) => b.toString(16).padStart(2, '0')).join(''));
      });
    }, 30);
    return () => {
      alive = false;
      window.clearTimeout(t);
    };
  }, []);
  const pinned = testSrc ? /SHOP_SHA256\s*=\s*'([0-9a-f]{64})'/.exec(testSrc)?.[1] ?? null : undefined;
  return (
    <div className="mt-3 rounded-lg border border-border bg-surface-2 px-3.5 py-2.5">
      <div className="flex items-center gap-2 text-[15.5px] font-semibold text-ink">
        <Hash aria-hidden="true" size={15} className="text-muted" /> Determinism check, run now
      </div>
      <div className="mt-1 text-[15px] leading-snug text-ink-2">
        SHA-256 of ShopDB regenerated in your browser:{' '}
        <span className={cx('text-[14.5px] break-all text-ink', hash && 'font-mono')}>{hash === undefined ? 'computing…' : hash === null ? 'unavailable here' : hash}</span>
      </div>
      <div className="mt-0.5 text-[15px] leading-snug text-ink-2">
        Pinned in <span className="font-mono text-[14.5px]">src/engine/db/db.test.ts</span>:{' '}
        <span className={cx('text-[14.5px] break-all text-ink', pinned && 'font-mono')}>{pinned === undefined ? 'loading…' : pinned ?? 'not found'}</span>
      </div>
      {hash && pinned && (
        <div className="mt-1.5">
          <StatusBadge status={hash === pinned ? 'good' : 'warning'}>{hash === pinned ? 'identical: same rows as the training run saw' : 'different hashes'}</StatusBadge>
        </div>
      )}
    </div>
  );
}

function ForwardTimer() {
  const assets = useAssets();
  const ctx = useContextWindow();
  const pdb = usePreprocessed();
  const [ms, setMs] = useState<number | null>(null);
  const [busy, setBusy] = useState(false);
  const run = () => {
    if (!assets.store || !assets.weights || busy) return;
    setBusy(true);
    window.setTimeout(() => {
      const times: number[] = [];
      for (let i = 0; i < 3; i++) {
        const t0 = performance.now();
        forward(ctx.tokens, pdb, assets.store!, assets.weights!, { capture: false });
        times.push(performance.now() - t0);
      }
      times.sort((a, b) => a - b);
      setMs(times[1]);
      setBusy(false);
    }, 20);
  };
  return (
    <div className="mt-3 flex flex-wrap items-center gap-3">
      <Button size="sm" onClick={run} disabled={busy || assets.status !== 'ready'} icon={busy ? <LoaderCircle size={15} className="animate-spin" /> : <Gauge size={15} />}>
        Time one forward pass
      </Button>
      <span className="text-[15.5px] text-ink-2">
        {ms === null ? (
          'median of three runs on this device, no capture'
        ) : (
          <>
            <span className="num font-semibold text-ink">{ms.toFixed(0)} ms</span> for {ctx.tokens.length} tokens (median of three runs)
          </>
        )}
      </span>
    </div>
  );
}

function Detail({ id }: { id: StageId }) {
  const s = byId(id);
  const db = useDatabase();
  const pdb = usePreprocessed();
  const ctx = useContextWindow();
  const task = useTask();
  const seed = useSeedNode();
  const assets = useAssets();
  const pred = usePrediction();
  const { data: ds } = useDataset();
  const { data: res } = useResults();
  const { data: ver } = useVerification();

  let body: ReactNode = null;
  switch (id) {
    case 'gen':
      body = (
        <>
          <p>
            Four small relational databases are generated by seeded code, so every run produces the same rows. They replace RelBench, which the paper
            uses <Cite kind="paper" ref="p.6 §4.1" />. ShopDB is the held-out target; the other three are pretraining data.
          </p>
          <DbTable />
        </>
      );
      break;
    case 'embed':
      body = (
        <>
          <p>
            Every string the model can meet (table names, “&lt;column&gt; of &lt;table&gt;” phrases, text cell values) is collected once and embedded
            with the frozen <span className="font-mono text-[16px]">all-MiniLM-L12-v2</span> sentence encoder, mean-pooled and L2-normalized, as the
            official preprocessing does <Cite kind="spec" ref="§3.6" /> <Cite kind="code" ref="rt/embed.py:12-72" />. The browser only looks vectors
            up; it never runs the encoder.
          </p>
          <KV k="strings embedded" v={assets.store ? int(assets.store.strings.length) : 'loading…'} />
          <KV k="dimension" v={assets.store ? assets.store.dim : '…'} />
          <KV k="file size (float32)" v={assets.store ? bytes(assets.store.strings.length * assets.store.dim * 4) : '…'} />
        </>
      );
      break;
    case 'contexts':
      body = (
        <>
          <p>
            The training data are built by the same TypeScript preprocessing and sampler that run in your browser, so the model was trained on
            exactly the kind of token sequences the chapters display. The PRNG seed of each context is epoch + seed row + 0, the formula of{' '}
            <span className="font-mono text-[16px]">fly.rs</span> <Cite kind="code" ref="rustler/src/fly.rs:371-375" />.
          </p>
          {ds ? (
            <>
              <KV k="pretraining databases (target held out)" v={`${ds.pretrain.dbs.join(', ')} · ${ds.pretrain.heldOut} held out`} />
              <KV k="pretraining tasks" v={ds.pretrain.numTasks} />
              <KV k="contexts exported" v={`${int(ds.pretrain.exportedContexts)} (${ds.pretrain.exportedEpochs} epochs × ${int(ds.pretrain.itemsPerEpoch)})`} />
              <KV k="L, w" v={`${ds.L}, ${ds.w}`} />
            </>
          ) : (
            <KV k="dataset.json" v="loading…" />
          )}
        </>
      );
      break;
    case 'train':
      body = (
        <>
          <p>
            Training mirrors <span className="font-mono text-[16px]">rt/main.py</span>: leave-one-database-out pretraining with AdamW and the OneCycle
            schedule, then checkpoint selection on ShopDB validation per task <Cite kind="spec" ref="§9.1" /> <Cite kind="spec" ref="§9.2" />. It ran in
            float32 on a CPU. Chapter 14 shows the curves.
          </p>
          {res?.pretrain && ds ? (
            <>
              <KV k="steps × batch" v={`${int(res.pretrain.steps)} × ${ds.batchSize}`} />
              <KV k="wall time" v={`${int(res.pretrain.wallSeconds)} s (${(res.pretrain.stepSeconds * 1000).toFixed(0)} ms per step)`} />
              <KV k="tokens seen" v={`${sci(ds.pretrain.tokensSeen, 2)} (paper ${sci(ds.paper.tokensSeen, 2)})`} />
              <KV k="device, dtype" v={`${res.config?.device ?? '?'}, ${res.config?.dtype ?? '?'}`} />
            </>
          ) : (
            <KV k="results.json" v="loading…" />
          )}
        </>
      );
      break;
    case 'verify':
      body = (
        <>
          <p>
            Four independent checks, each written to <span className="font-mono text-[16px]">public/data/verification.json</span>, which the next
            sections of this chapter read.
          </p>
          {ver ? (
            <>
              <KV k="official model.py vs port" v={`|Δ| ≤ ${sci(ver.official_vs_port_max_abs_diff)}`} />
              <KV k="TypeScript vs PyTorch" v={`|Δ| ≤ ${sci(ver.ts_vs_torch_max_abs_diff)}`} />
              <KV k="sampler cross-check" v={`${int(ver.sampler_crosscheck.contexts)} contexts, ${ver.sampler_crosscheck.mismatches} mismatches`} />
              <KV k="tests" v={`${int(ver.details.testsTotal - ver.details.testsFailed)} / ${int(ver.details.testsTotal)} passed`} />
            </>
          ) : (
            <KV k="verification.json" v="loading…" />
          )}
        </>
      );
      break;
    case 'r-gen':
      body = (
        <>
          <p>
            When the page loads, your browser runs the same generator code as step B1 and gets the same rows. The current database is{' '}
            <strong>{db.title}</strong>.
          </p>
          {db.tables.map((t) => (
            <KV key={t.schema.name} k={t.schema.name} v={`${int(t.rows.length)} rows`} />
          ))}
          {db.name === 'shop' && <ShopHash />}
        </>
      );
      break;
    case 'r-pre':
      body = (
        <>
          <p>
            Rows become graph nodes and cells, with z-scored values and text indices, following <span className="font-mono text-[16px]">pre.rs</span>{' '}
            line by line <Cite kind="spec" ref="§3" />. Task tables are included as ordinary tables, one per split <Cite kind="spec" ref="§1.5" />.
          </p>
          <KV k="nodes (rows of all tables and task splits)" v={int(pdb.nodes.length)} />
          <KV k="table files" v={Object.keys(pdb.tableInfo).length} />
          <KV k="strings in this database’s text table" v={int(pdb.text.length)} />
        </>
      );
      break;
    case 'r-sample':
      body = (
        <>
          <p>
            The bounded BFS of <span className="font-mono text-[16px]">fly.rs</span> collects the context around the seed row for the task{' '}
            {task.id}, then the four attention masks are built from the token fields{' '}
            <Cite kind="spec" ref="§4.3" /> <Cite kind="code" ref="rt/model.py:164-191" />.
          </p>
          <KV k="seed node" v={seed} />
          <KV k="real tokens / L" v={`${ctx.numReal} / ${ctx.tokens.length}`} />
          <KV k="rows visited" v={ctx.visited.length} />
          <KV k="sampler decisions traced" v={int(ctx.events.length)} />
        </>
      );
      break;
    case 'r-forward':
      body = (
        <>
          <p>
            The embedding, {MINI_CONFIG.nBlocks} blocks of column, feature, neighbor and full attention plus SwiGLU, the final norm and the decoders,
            as in{' '}
            <span className="font-mono text-[16px]">rt/model.py</span> <Cite kind="code" ref="rt/model.py:153-267" />. It reads the embeddings (B2)
            and the checkpoint (B4).
          </p>
          <KV k="checkpoint" v={assets.ckpt} />
          <KV k="weights" v={assets.status === 'ready' ? (assets.weightsKind === 'init' ? 'untrained (init)' : 'trained') : assets.status} />
          <KV
            k="prediction at the masked cell"
            v={
              pred
                ? pred.taskKind === 'binary'
                  ? `logit ${pred.output.toFixed(3)} → p = ${pred.probability?.toFixed(3)}`
                  : `z ${pred.output.toFixed(3)} → ${pred.value?.toFixed(2)}`
                : '…'
            }
          />
          <ForwardTimer />
        </>
      );
      break;
  }

  return (
    <div className="min-w-0">
      <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
        <span className="num text-[15px] font-semibold text-muted">{s.n}</span>
        <h3 className="!m-0 text-[19px] font-semibold text-ink">{s.title}</h3>
        <span className="text-[15px] text-ink-2">{s.row === 'build' ? 'built once, offline' : 'runs in your browser on every change'}</span>
      </div>
      <div className="mt-1.5 flex flex-wrap gap-1.5">
        {s.files.map((f) => (
          <span key={f} className="rounded-md border border-border bg-surface-2 px-1.5 font-mono text-[14px] text-ink-2">
            {f}
          </span>
        ))}
      </div>
      <div className="prose mt-3 max-w-none text-[17px] [&_p]:mb-3">{body}</div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// The figure
// ─────────────────────────────────────────────────────────────────────────────

const DESCRIBE: Record<StageId, string> = {
  gen: 'Seeded generators write four synthetic databases.',
  embed: 'Every string is embedded once with all-MiniLM-L12-v2.',
  contexts: 'The TypeScript sampler writes the training contexts.',
  train: 'PyTorch trains the port of model.py on CPU and exports checkpoints.',
  verify: 'Parity, sampler cross-check and tests write verification.json.',
  'r-gen': 'Your browser regenerates the same database.',
  'r-pre': 'Rows become nodes and cells (pre.rs mirror).',
  'r-sample': 'The context window is sampled around the seed (fly.rs mirror).',
  'r-forward': 'The forward pass runs with the shipped weights and embeddings.',
};

export function PipelineMap() {
  const player = useStepPlayer(ORDER.length, { intervalMs: 1800 });
  const active = ORDER[player.index];
  const [hover, setHover] = useState<StageId | null>(null);
  const twinOf = (id: StageId) => (hover ? byId(hover).twins?.includes(id) ?? false : false);
  const select = (id: StageId) => {
    player.setPlaying(false);
    player.setIndex(ORDER.indexOf(id));
  };
  const build = STAGES.filter((s) => s.row === 'build');
  const run = STAGES.filter((s) => s.row === 'run');
  const activeArtifact = ARTIFACTS.find((a) => a.from === active)?.id;

  const rowGrid = 'grid gap-1.5 md:grid-cols-[repeat(4,minmax(0,1fr)_20px)_minmax(0,1fr)]';

  return (
    <div className="min-w-0">
      <div className="eyebrow mb-2 flex items-center gap-2">
        <Database aria-hidden="true" size={14} /> Built once, offline
      </div>
      <div className={rowGrid}>
        {build.map((s, i) => (
          <div key={s.id} className="contents">
            <StageCard s={s} active={active === s.id} twin={twinOf(s.id)} onSelect={() => select(s.id)} onHover={(h) => setHover(h ? s.id : null)} />
            {i < build.length - 1 && <Arrow />}
          </div>
        ))}
      </div>

      <div className="my-3 rounded-xl border border-dashed border-border-strong bg-surface-2 px-3.5 py-2.5">
        <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
          <FileJson aria-hidden="true" size={16} className="text-muted" />
          <span className="font-mono text-[15px] font-semibold text-ink">public/data/</span>
          <span className="text-[15px] text-ink-2">committed files that connect the two halves</span>
        </div>
        <div className="mt-2 flex flex-wrap gap-2">
          {ARTIFACTS.map((a) => (
            <span
              key={a.id}
              className={cx(
                'inline-flex flex-wrap items-baseline gap-x-1.5 rounded-lg border px-2.5 py-1 transition-colors',
                activeArtifact === a.id || (active === 'r-forward' && a.id !== 'verification') ? 'border-accent bg-accent-soft' : 'border-border bg-surface',
              )}
            >
              <span className="num text-[14px] font-semibold text-muted">{byId(a.from).n}</span>
              <span className="font-mono text-[14.5px] text-ink">{a.file}</span>
              <span className="text-[14.5px] text-ink-2">{a.label}</span>
            </span>
          ))}
        </div>
      </div>

      <div className="eyebrow mb-2 flex items-center gap-2">
        <Gauge aria-hidden="true" size={14} /> Runs in your browser, on every change
      </div>
      <div className={rowGrid}>
        {run.map((s, i) => (
          <div key={s.id} className="contents">
            <StageCard s={s} active={active === s.id} twin={twinOf(s.id)} onSelect={() => select(s.id)} onHover={(h) => setHover(h ? s.id : null)} />
            {i < run.length - 1 && <Arrow />}
          </div>
        ))}
        <Arrow />
        <PredictionCard />
      </div>

      <StepPlayer {...player} className="mt-4" unit="stage" label="Walk through the pipeline" describe={(i) => DESCRIBE[ORDER[i]]} />

      <div className="mt-4 rounded-xl border border-border bg-surface px-4 py-3.5" aria-live="polite">
        <Detail id={active} />
      </div>
    </div>
  );
}

function PredictionCard() {
  const pred = usePrediction();
  const assets = useAssets();
  return (
    <div className="flex h-full min-w-0 flex-col rounded-xl border border-border bg-surface-2 px-3 py-2.5">
      <span className="num text-[14px] font-semibold text-muted">out</span>
      <span className="mt-1 block text-[16px] leading-snug font-semibold text-ink">Prediction</span>
      <span className="num mt-auto block pt-1 text-[15px] text-ink">
        {pred ? (pred.taskKind === 'binary' ? `p = ${pred.probability?.toFixed(3)}` : `${pred.value?.toFixed(2)}`) : '…'}
      </span>
      {assets.weightsKind === 'init' && <span className="mt-1 text-[13.5px] text-ink-2">untrained weights</span>}
    </div>
  );
}
