/**
 * Figures for "The encoder is frozen":
 *  1. a step-through diagram of where the text vectors come from (offline, once: embed.py) and how a
 *     forward pass uses them (a row lookup, then the first learned layer), including where the
 *     gradient stops;
 *  2. the learned part, live: W_col·e + b → RMSNorm for the chapter's focus phrase, computed with the
 *     engine's ops on the loaded checkpoint (model.py:209-213).
 */
import { ArrowDown, ArrowRight, X } from 'lucide-react';
import { motion, useReducedMotion } from 'motion/react';
import { useEffect, useMemo, useRef, type ReactNode } from 'react';
import { StatusBadge, StepPlayer, VectorStrip, cx, useStepPlayer } from '@/components';
import { axis, vec } from '../shapes';
import type { EmbeddingStore } from '@/engine/embeddings';
import { MINI_CONFIG, PAPER_CONFIG, linearVec, paramCount, parameterShapes, rmsnormRow, tensor } from '@/engine/model';
import type { RTConfig } from '@/engine/types';
import type { AssetsState } from '@/state/hooks';
import { BreakId, Phrase, useMedia } from './bits';
import { ENCODER, cosine, f3, int, norm, type Catalog } from './data';

// ─────────────────────────────────────────────────────────────────────────────
// 1. Step-through diagram
// ─────────────────────────────────────────────────────────────────────────────

interface Node {
  id: string;
  title: string;
  sub: ReactNode;
  zone: 'offline' | 'online';
  learned?: boolean;
  frozen?: boolean;
}

interface StepDef {
  active: string[];
  /** connector index that pulses (between node i and i+1) */
  edge?: number;
  backward?: boolean;
  text: ReactNode;
}

export function FrozenPipeline({ nStrings, dbLabel, d }: { nStrings: number; dbLabel: string; d: number }) {
  const reduce = useReducedMotion();
  const nodes: Node[] = [
    { id: 'strings', title: 'text.json', sub: `${int(nStrings)} strings of ${dbLabel}`, zone: 'offline' },
    { id: 'encoder', title: 'all-MiniLM-L12-v2', sub: `${(ENCODER.parameters / 1e6).toFixed(1)}M parameters, frozen`, zone: 'offline', frozen: true },
    { id: 'table', title: 'text_emb_all-MiniLM-L12-v2.bin', sub: `${int(nStrings)} × ${ENCODER.hidden} table`, zone: 'offline', frozen: true },
    { id: 'lookup', title: 'Sampler lookup', sub: 'copy row col_name_idx (Rust data loader)', zone: 'online' },
    { id: 'linear', title: 'W·e + b', sub: `Linear(384 → ${d}), learned`, zone: 'online', learned: true },
    { id: 'norm', title: 'RMSNorm', sub: 'learned gain γ', zone: 'online', learned: true },
    { id: 'term', title: 'Column term', sub: `${d} numbers, added to x`, zone: 'online' },
  ];
  const steps: StepDef[] = [
    {
      active: ['strings'],
      text: (
        <>
          Preprocessing writes one string table per database: table names, column phrases and string values. For {dbLabel} it holds{' '}
          {int(nStrings)} strings.
        </>
      ),
    },
    {
      active: ['encoder'],
      edge: 0,
      text: <>embed.py runs the sentence encoder over the whole list, once. This is the only time the encoder ever runs.</>,
    },
    {
      active: ['table'],
      edge: 1,
      text: (
        <>
          The vectors are written to disk as one raw [{int(nStrings)}, 384] array (bf16 in the paper’s code, float32 on this site). From here
          on, a string is just a row number.
        </>
      ),
    },
    {
      active: ['lookup'],
      edge: 2,
      text: (
        <>
          While it builds a batch, the sampler copies, for every token, row <code>col_name_idx</code> into <code>col_name_values</code>, and row{' '}
          <code>text_idx</code> into <code>text_values</code> (the value string for a text cell, the unused <code>text[0]</code> for every other
          cell). A memory copy, no neural network.
        </>
      ),
    },
    {
      active: ['linear'],
      edge: 3,
      text: <>The model’s first learned layer maps the 384 frozen numbers to d = {d} (enc_dict["col_name"], with a bias).</>,
    },
    {
      active: ['norm', 'term'],
      edge: 5,
      text: <>RMSNorm with a learned gain gives the column term, which is added to the token’s value term to form its input vector.</>,
    },
    {
      active: ['term', 'norm', 'linear'],
      backward: true,
      text: (
        <>
          Training sends the loss gradient back through RMSNorm and the linear layer, updating γ, W and b. It stops at the lookup: the table is
          plain data, so the encoder’s vectors never change.
        </>
      ),
    },
  ];
  const player = useStepPlayer(steps.length, { intervalMs: 2200 });
  const st = steps[player.index];
  const narrow = useMedia('(max-width: 767px)');
  const nodeEls = useRef<Record<string, HTMLDivElement | null>>({});
  const firstRun = useRef(true);
  // narrow screens: bring the active box into view when the step changes (not on the first render)
  useEffect(() => {
    if (firstRun.current) {
      firstRun.current = false;
      return;
    }
    if (!narrow) return;
    const el = nodeEls.current[st.active[0]];
    if (!el) return;
    const r = el.getBoundingClientRect();
    const topbar = 56 + 170; // the app bar plus the sticky player above the diagram
    if (r.top < topbar || r.bottom > window.innerHeight - 8) {
      window.scrollTo({ top: window.scrollY + r.top - (topbar + window.innerHeight - r.height) / 2, behavior: reduce ? 'auto' : 'smooth' });
    }
  }, [player.index, narrow, reduce, st.active]);
  const offline = nodes.filter((n) => n.zone === 'offline');
  const online = nodes.filter((n) => n.zone === 'online');

  const renderChain = (list: Node[], offset: number) => (
    <div className="flex flex-col items-stretch gap-1 md:flex-row md:items-stretch">
      {list.map((n, k) => {
        const i = offset + k;
        const on = st.active.includes(n.id);
        return (
          <div key={n.id} className="contents">
            <motion.div
              ref={(el: HTMLDivElement | null) => {
                nodeEls.current[n.id] = el;
              }}
              animate={{ scale: on && !reduce ? 1.03 : 1 }}
              transition={{ duration: 0.25 }}
              className={cx(
                'relative min-w-0 flex-1 rounded-lg border px-3 py-2.5 transition-colors',
                on ? 'border-accent bg-accent-soft' : 'border-border bg-surface',
                n.frozen && 'border-dashed',
              )}
            >
              <div className={cx('text-[15.5px] leading-tight font-semibold text-ink [overflow-wrap:anywhere]', n.id === 'table' || n.id === 'strings' ? 'font-mono' : '')}>
                <BreakId text={n.title} />
              </div>
              <div className="mt-1 text-[14px] leading-snug text-ink-2">{n.sub}</div>
              {n.learned && <span className="mt-1.5 inline-flex rounded bg-surface-3 px-1.5 text-[13px] text-ink">trained</span>}
              {n.frozen && <span className="mt-1.5 inline-flex rounded border border-dashed border-border-strong px-1.5 text-[13px] text-ink-2">fixed</span>}
            </motion.div>
            {k < list.length - 1 && (
              <Connector active={st.edge === i} backward={!!st.backward && i >= 4} stop={!!st.backward && i === 3} reduce={!!reduce} />
            )}
          </div>
        );
      })}
    </div>
  );

  return (
    <div className="flex flex-col">
      {/* below md the chain stacks into a tall column: the player goes on top and stays in view */}
      <div className="sticky top-[calc(var(--topbar-h)+4px)] z-20 order-1 -mx-1 mb-3 rounded-xl border border-border bg-surface px-1 shadow-[var(--shadow-pop)] md:static md:order-2 md:mx-0 md:mt-4 md:mb-0 md:rounded-none md:border-0 md:px-0 md:shadow-none">
        <StepPlayer {...player} unit="step" describe={() => <span className="text-[16px] leading-snug text-ink">{st.text}</span>} label="Frozen-encoder walkthrough" />
      </div>
      <div className="order-2 grid gap-1 md:order-1">
        <Zone title="Offline, once per database">{renderChain(offline, 0)}</Zone>
        <div className="flex items-center justify-center gap-3">
          <Connector active={st.edge === 2} backward={false} reduce={!!reduce} big vertical />
          <span className="text-[14.5px] text-ink-2">rows read by index</span>
        </div>
        <Zone title="Every batch: the sampler, then the model’s forward pass (training and inference)">{renderChain(online, 3)}</Zone>
      </div>
    </div>
  );
}

function Zone({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="min-w-0 rounded-xl border border-border bg-surface-2/60 p-2.5">
      <div className="eyebrow mb-2 px-1">{title}</div>
      {children}
    </div>
  );
}

function Connector({
  active,
  backward,
  reduce,
  stop,
  big,
  vertical,
}: {
  active: boolean;
  backward: boolean;
  reduce: boolean;
  stop?: boolean;
  big?: boolean;
  vertical?: boolean;
}) {
  return (
    <div className={cx('relative flex shrink-0 items-center justify-center', big ? 'h-10 w-10' : stop ? 'h-9 md:h-auto md:w-8' : 'h-7 md:h-auto md:w-6')} aria-hidden="true">
      {stop ? (
        <span
          className="inline-flex h-7 w-7 items-center justify-center rounded-full border-2 border-accent bg-surface text-accent-text"
          title="the gradient stops here"
          aria-label="the gradient stops here"
        >
          <X size={16} strokeWidth={2.6} />
        </span>
      ) : (
        <>
          <ArrowDown size={big ? 22 : 18} className={cx(!vertical && 'md:hidden', active || backward ? 'text-accent' : 'text-axis', backward && 'rotate-180')} />
          {!vertical && (
            <ArrowRight size={18} className={cx('hidden md:block', active || backward ? 'text-accent' : 'text-axis', backward && 'rotate-180')} />
          )}
          {(active || backward) && !reduce && (
            <motion.span
              key={`${active}-${backward}`}
              className="absolute h-2 w-2 rounded-full bg-accent"
              initial={{ opacity: 0, scale: 0.5 }}
              animate={{ opacity: [0, 1, 0], scale: [0.5, 1.2, 0.5] }}
              transition={{ duration: 1.1, repeat: Infinity }}
            />
          )}
        </>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. The learned part, live
// ─────────────────────────────────────────────────────────────────────────────

function textParams(cfg: RTConfig): number {
  let n = 0;
  for (const [name, shape] of parameterShapes(cfg)) {
    if (/^(enc_dict|norm_dict)\.(col_name|text)\./.test(name)) n += shape.reduce((a, b) => a * b, 1);
  }
  return n;
}

export function LearnedProjection({ phrase, store, assets, catalog }: { phrase: string; store: EmbeddingStore; assets: AssetsState; catalog: Catalog | null }) {
  const w = assets.weights;
  const res = useMemo(() => {
    if (!w || !store.has(phrase)) return null;
    const d = w.config.dModel;
    const W = tensor(w, 'enc_dict.col_name.weight');
    const b = tensor(w, 'enc_dict.col_name.bias');
    const g = tensor(w, 'norm_dict.col_name.weight');
    // the column term of one phrase: RMSNorm_γ(W·e + b) (model.py:209-213)
    const term = (text: string) => {
      const e = store.get(text);
      const pre = linearVec(e, W, d, b);
      const out = new Float32Array(d);
      rmsnormRow(pre, 0, d, g, w.config.eps, out, 0);
      return { e, pre, out };
    };
    const me = term(phrase);
    // partners: the closest on-token phrase of another database, the closest task label of another
    // database, and another phrase of the same table (all chosen live by cosine of the frozen vectors)
    const info = catalog?.byText.get(phrase);
    const partners: { why: string; text: string }[] = [];
    if (catalog && info) {
      const others = catalog.phrases.filter((p) => p.db !== info.db && p.onToken && store.has(p.text));
      const byCos = others.map((p) => ({ p, c: cosine(me.e, store.get(p.text)) })).sort((x, y) => y.c - x.c);
      if (byCos[0]) partners.push({ why: 'closest phrase of another database', text: byCos[0].p.text });
      const lab = byCos.find((x) => x.p.role === 'label' && x.p.text !== byCos[0]?.p.text);
      if (lab) partners.push({ why: 'closest task label of another database', text: lab.p.text });
      const mate = catalog.byDb[info.db].find((p) => p.table === info.table && p.onToken && p.text !== phrase && store.has(p.text));
      if (mate) partners.push({ why: 'another column of the same table', text: mate.text });
    }
    const rows = partners.map((pt) => {
      const t = term(pt.text);
      return { ...pt, before: cosine(me.e, t.e), after: cosine(me.out, t.out) };
    });
    let ms = 0;
    for (let i = 0; i < d; i++) ms += me.out[i] * me.out[i];
    return { ...me, d, shape: w.tensors['enc_dict.col_name.weight'].shape, rows, rms: Math.sqrt(ms / d) };
  }, [w, store, phrase, catalog]);

  const mini = { text: textParams(MINI_CONFIG), total: paramCount(MINI_CONFIG) };
  const paper = { text: textParams(PAPER_CONFIG), total: paramCount(PAPER_CONFIG) };

  return (
    <div>
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <span className="text-[16px] text-ink">
          Column term of <Phrase text={phrase} strong />
        </span>
        {assets.status === 'ready' && assets.weightsKind === 'trained' && <StatusBadge status="good">trained mini model ({assets.ckpt})</StatusBadge>}
        {assets.status === 'ready' && assets.weightsKind === 'init' && <StatusBadge status="warning">untrained weights: a random initialization</StatusBadge>}
        {assets.status === 'loading' && <StatusBadge status="info">loading weights…</StatusBadge>}
      </div>
      {res ? (
        <>
          <div className="grid items-start gap-4 lg:grid-cols-[minmax(0,1.3fr)_minmax(0,1fr)_minmax(0,1fr)]">
            <div className="min-w-0">
              <div className="mb-1 text-[15px] text-ink-2">
                e: 384 frozen numbers · ‖e‖ = <span className="num text-ink">{norm(res.e).toFixed(3)}</span>
              </div>
              <VectorStrip values={res.e} columns={48} cellHeight={9} showReadout={false} shape={vec(`e = E(“${phrase}”), the frozen embedding of the column phrase`, axis.text(res.e.length))} />
            </div>
            <div className="min-w-0">
              <div className="mb-1 text-[15px] text-ink-2">
                W·e + b, W of shape [{res.shape.join(', ')}]
              </div>
              <VectorStrip values={res.pre} columns={32} cellHeight={16} showNorm={false} shape={vec('W·e + b: the phrase projected to the model width by enc_dict["col_name"]', axis.model(res.d))} />
            </div>
            <div className="min-w-0">
              <div className="mb-1 text-[15px] text-ink-2">
                RMSNorm → column term ({res.d}-d) · RMS <span className="num text-ink">{res.rms.toFixed(2)}</span>
              </div>
              <VectorStrip values={res.out} columns={32} cellHeight={16} showNorm={false} shape={vec('the column term: RMSNorm_col(W·e + b)', axis.model(res.d))} />
            </div>
          </div>

          {res.rows.length > 0 && (
            <div className="mt-5 rounded-lg border border-border px-4 py-3">
              <div className="mb-2 text-[15.5px] font-semibold text-ink">Does the learned map keep the encoder’s similarities?</div>
              <div className="grid gap-2.5 sm:hidden">
                {res.rows.map((r) => (
                  <div key={r.text} className="rounded-md border border-grid/70 px-3 py-2">
                    <Phrase text={r.text} />
                    <div className="text-[14px] text-muted">{r.why}</div>
                    <div className="mt-1 flex flex-wrap gap-x-4 text-[15px] text-ink-2">
                      <span>
                        cos, frozen e <span className="num text-ink">{f3(r.before)}</span>
                      </span>
                      <span>
                        cos, column term <span className="num font-semibold text-ink">{f3(r.after)}</span>
                      </span>
                    </div>
                  </div>
                ))}
              </div>
              <div className="scroll-quiet hidden overflow-x-auto sm:block">
                <table className="w-full border-collapse text-[15px]">
                  <caption className="sr-only">Cosine of the frozen vectors and of the column terms, for three partner phrases</caption>
                  <thead>
                    <tr className="border-b border-grid text-left text-ink-2">
                      <th scope="col" className="py-1.5 pr-3 font-semibold">
                        Partner phrase
                      </th>
                      <th scope="col" className="py-1.5 pr-3 text-right font-semibold whitespace-nowrap">
                        cos, frozen e
                      </th>
                      <th scope="col" className="py-1.5 text-right font-semibold whitespace-nowrap">
                        cos, column term
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {res.rows.map((r) => (
                      <tr key={r.text} className="border-b border-grid/70 align-top">
                        <td className="py-1.5 pr-3">
                          <Phrase text={r.text} />
                          <div className="text-[14px] text-muted">{r.why}</div>
                        </td>
                        <td className="num py-1.5 pr-3 text-right text-ink">{f3(r.before)}</td>
                        <td className="num py-1.5 text-right font-semibold text-ink">{f3(r.after)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="mt-2 !mb-0 text-[14.5px] leading-snug text-ink-2">
                The encoder fixes where every phrase starts. On the way into the token, training can reshape that geometry only through this one
                affine map (W, b) and the gain γ, the same for every phrase of every database. The right column is what the model actually adds to
                each token as its column term{assets.weightsKind === 'init' ? ' (here with untrained weights)' : ''}.
              </p>
            </div>
          )}
        </>
      ) : (
        <div className="flex min-h-[90px] items-center justify-center rounded-lg bg-surface-2 text-[15px] text-ink-2">
          {assets.status === 'error' ? `Weights unavailable (${assets.error ?? 'error'})` : 'Loading the checkpoint…'}
        </div>
      )}

      <div className="mt-5 grid gap-2.5 sm:grid-cols-3">
        <div className="rounded-lg border border-dashed border-border-strong px-3.5 py-3">
          <div className="text-[15px] text-ink-2">Text encoder (pretrained, frozen: never updated by RT)</div>
          <div className="num mt-1 text-[21px] font-semibold text-ink">{int(ENCODER.parameters)}</div>
          <div className="text-[14px] text-muted">
            {ENCODER.layers} layers, hidden {ENCODER.hidden}, {ENCODER.heads} heads
          </div>
        </div>
        <div className="rounded-lg border border-border px-3.5 py-3">
          <div className="text-[15px] text-ink-2">Learned parameters that read the text vectors, this site</div>
          <div className="num mt-1 text-[21px] font-semibold text-ink">{int(mini.text)}</div>
          <div className="text-[14px] text-muted">
            of {int(mini.total)} in all (col_name and text projections + norms)
          </div>
        </div>
        <div className="rounded-lg border border-border px-3.5 py-3">
          <div className="text-[15px] text-ink-2">Same, paper model</div>
          <div className="num mt-1 text-[21px] font-semibold text-ink">{int(paper.text)}</div>
          <div className="text-[14px] text-muted">of {int(paper.total)} in all</div>
        </div>
      </div>
    </div>
  );
}
