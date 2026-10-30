/**
 * Figure "A real 384-d vector": the stored all-MiniLM-L12-v2 embedding of any embedded string, the
 * WordPiece tokens the encoder saw (generated offline from the real tokenizer), the pipeline
 * Transformer → mean pooling → Normalize (spec §3.6), and a comparison strip showing that the cosine
 * of two unit vectors is the sum of their 384 element-wise products.
 */
import { motion, useReducedMotion } from 'motion/react';
import { useMemo, useState } from 'react';
import { VectorStrip, cx } from '@/components';
import { axis, vec } from '../shapes';
import type { EmbeddingStore } from '@/engine/embeddings';
import { Phrase, Readout, Wordpieces } from './bits';
import { ENCODER, cosine, f3, int, norm, wordpiecesOf, type Catalog } from './data';
import { StringPicker } from './StringPicker';

export interface VectorExplorerProps {
  store: EmbeddingStore;
  catalog: Catalog | null;
  value: string;
  onChange: (s: string) => void;
  compare: string;
  onCompare: (s: string) => void;
  presets: { label: string; text: string }[];
}

export function VectorExplorer({ store, catalog, value, onChange, compare, onCompare, presets }: VectorExplorerProps) {
  const [hoverDim, setHoverDim] = useState<number | null>(null);
  const e1 = store.has(value) ? store.get(value) : null;
  const e2 = store.has(compare) ? store.get(compare) : null;
  const tokens = wordpiecesOf(value) ?? [];

  const stats = useMemo(() => {
    if (!e1) return null;
    let iMax = 0;
    let pos = 0;
    for (let i = 0; i < e1.length; i++) {
      if (Math.abs(e1[i]) > Math.abs(e1[iMax])) iMax = i;
      if (e1[i] > 0) pos++;
    }
    return { n: norm(e1), iMax, vMax: e1[iMax], pos };
  }, [e1]);

  const prod = useMemo(() => {
    if (!e1 || !e2) return null;
    const p = new Float32Array(e1.length);
    for (let i = 0; i < p.length; i++) p[i] = e1[i] * e2[i];
    return p;
  }, [e1, e2]);
  const cos = e1 && e2 ? cosine(e1, e2) : NaN;
  const sumProd = prod ? prod.reduce((a, b) => a + b, 0) : NaN;
  const domain = useMemo(() => {
    let m = 0;
    for (const v of [e1, e2]) if (v) for (let i = 0; i < v.length; i++) m = Math.max(m, Math.abs(v[i]));
    return m || 1;
  }, [e1, e2]);

  const isPhrase = (s: string) => !!catalog?.byText.has(s);

  return (
    <div>
      <div className="mb-3 grid gap-x-5 gap-y-3 md:grid-cols-2">
        <StringPicker label="String to inspect" value={value} onChange={onChange} strings={store.strings} catalog={catalog} />
        <StringPicker label="Compare with" value={compare} onChange={onCompare} strings={store.strings} catalog={catalog} />
      </div>
      <div className="mb-5 flex flex-wrap items-center gap-2">
        <span className="text-[15px] text-ink-2">Try:</span>
        {presets.map((p) => (
          <button
            key={p.label}
            type="button"
            onClick={() => onChange(p.text)}
            aria-pressed={p.text === value}
            className={cx(
              'inline-flex h-8 items-center gap-1.5 rounded-full border px-3 text-[14.5px] transition-colors',
              p.text === value ? 'border-accent bg-accent-soft text-ink' : 'border-border bg-surface text-ink-2 hover:text-ink',
            )}
          >
            <span className="text-muted">{p.label}</span>
            <span className="text-ink">{p.text.length > 30 ? `${p.text.slice(0, 29)}…` : p.text}</span>
          </button>
        ))}
      </div>

      <EncoderStages text={value} tokens={tokens} isPhrase={isPhrase(value)} />

      <div className="mt-6">
        {e1 ? (
          <>
            <div className="mb-1.5 flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
              <div className="text-[16px] font-semibold text-ink">
                Stored vector <span className="font-normal">E(</span>
                <Phrase text={value} plain={!isPhrase(value)} className="font-normal" />
                <span className="font-normal">)</span>
              </div>
              <span className="text-[14.5px] text-ink-2">384 numbers, 48 per row · blue &lt; 0 &lt; red</span>
            </div>
            <VectorStrip
              values={e1}
              columns={48}
              cellHeight={16}
              domain={domain}
              highlightIndex={hoverDim}
              onHover={(i) => setHoverDim(i)}
              shape={vec(`E(“${value}”), the stored vector of this string`, axis.text(e1.length))}
            />
            {stats && (
              <div className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-4">
                <Readout label="Length ‖e‖" hint="unit norm (Normalize)">
                  {stats.n.toFixed(4)}
                </Readout>
                <Readout label="Largest |component|" hint={`dimension ${stats.iMax}`}>
                  {f3(stats.vMax)}
                </Readout>
                <Readout label="Positive dims" hint="out of 384">
                  {stats.pos}
                </Readout>
                <Readout label="WordPieces" hint={tokens.length <= ENCODER.maxSeqLength ? `cap ${ENCODER.maxSeqLength}: not truncated` : `truncated at ${ENCODER.maxSeqLength}`}>
                  {tokens.length}
                </Readout>
              </div>
            )}
          </>
        ) : (
          <p className="text-[15px] text-ink-2">No stored vector for this string.</p>
        )}
      </div>

      {e1 && e2 && prod && (
        <div className="mt-7 border-t border-grid pt-5">
          <div className="mb-1.5 text-[16px] font-semibold text-ink">
            Compared with <span className="font-normal">E(</span>
            <Phrase text={compare} plain={!isPhrase(compare)} className="font-normal" />
            <span className="font-normal">)</span>
          </div>
          <VectorStrip
            values={e2}
            columns={48}
            cellHeight={16}
            domain={domain}
            highlightIndex={hoverDim}
            onHover={(i) => setHoverDim(i)}
            shape={vec(`E(“${compare}”), the stored vector of the second string`, axis.text(e2.length))}
          />
          <div className="mt-4 mb-1.5 flex flex-wrap items-baseline justify-between gap-x-4">
            <div className="text-[16px] font-semibold text-ink">Element-wise product e₁ᵢ · e₂ᵢ</div>
            <span className="text-[14.5px] text-ink-2">red where both agree in sign, blue where they disagree</span>
          </div>
          <VectorStrip
            values={prod}
            columns={48}
            cellHeight={16}
            highlightIndex={hoverDim}
            onHover={(i) => setHoverDim(i)}
            showNorm={false}
            shape={vec(
              'e₁ᵢ · e₂ᵢ, the element-wise product of the two vectors above',
              `one per dimension of the two MiniLM vectors (${prod.length}); the products sum to their cosine, because both vectors have length 1`,
            )}
          />
          <div className="mt-3 flex flex-wrap items-center gap-x-6 gap-y-2 rounded-lg bg-surface-2 px-4 py-3 text-[16px]">
            <span className="text-ink-2">
              Σ of the 384 products = <strong className="num text-ink">{f3(sumProd)}</strong>
            </span>
            <span className="text-ink-2">
              cosine = e₁·e₂ / (‖e₁‖‖e₂‖) = <strong className="num text-ink">{f3(cos)}</strong>
            </span>
            <span className="text-[15px] text-muted">Both lengths are 1, so the two numbers agree.</span>
          </div>
        </div>
      )}
    </div>
  );
}

/** The four stages of SentenceTransformer.encode for one string, with real WordPiece tokens. */
function EncoderStages({ text, tokens, isPhrase }: { text: string; tokens: string[]; isPhrase: boolean }) {
  const reduce = useReducedMotion();
  const n = tokens.length;
  return (
    <ol className="!m-0 grid !list-none gap-3 !p-0 lg:grid-cols-[minmax(0,0.9fr)_minmax(0,1.5fr)_minmax(0,1fr)_minmax(0,1fr)]" aria-label="Encoder stages">
      <Stage n={1} title="Text">
        <div className="text-[16px] leading-snug">
          <Phrase text={text} plain={!isPhrase} />
        </div>
      </Stage>
      <Stage n={2} title={`WordPiece tokens (${n})`} note={`lower-cased, vocabulary of ${int(ENCODER.vocab)}`}>
        <Wordpieces tokens={tokens} animateKey={text} />
      </Stage>
      <Stage n={3} title={`${ENCODER.layers} BERT layers`} note={`one ${ENCODER.hidden}-d vector per token`}>
        <div className="flex h-[58px] items-end gap-[3px]" aria-hidden="true">
          {tokens.map((t, i) => (
            <motion.span
              key={`${text}-${i}`}
              initial={reduce ? false : { scaleY: 0.2, opacity: 0 }}
              animate={{ scaleY: 1, opacity: 1 }}
              transition={{ duration: 0.3, delay: reduce ? 0 : 0.15 + i * 0.04 }}
              style={{ originY: 1 }}
              className={cx('h-full w-[7px] rounded-[2px]', t === '[CLS]' || t === '[SEP]' ? 'bg-axis' : 'bg-seq-400')}
            />
          ))}
        </div>
        <div className="mt-1 text-[14px] text-ink-2">
          {n} × {ENCODER.hidden}
        </div>
      </Stage>
      <Stage n={4} title="Mean, then ÷ length" note="Pooling(mean) → Normalize">
        <div className="flex h-[58px] items-end gap-2" aria-hidden="true">
          <motion.span
            key={`${text}-mean`}
            initial={reduce ? false : { scaleX: 3, opacity: 0 }}
            animate={{ scaleX: 1, opacity: 1 }}
            transition={{ duration: 0.45, delay: reduce ? 0 : 0.3 + n * 0.04 }}
            className="h-full w-[10px] rounded-[2px] bg-seq-600"
          />
          <span className="pb-1 text-[15px] text-ink-2">1 × {ENCODER.hidden}, length 1</span>
        </div>
        <div className="mt-1 text-[14px] text-ink-2">averages all {n}, special tokens included</div>
      </Stage>
    </ol>
  );
}

function Stage({ n, title, note, children }: { n: number; title: string; note?: string; children: React.ReactNode }) {
  return (
    <li className="!m-0 min-w-0 rounded-lg border border-border bg-surface px-3 py-2.5 !pl-3">
      <div className="mb-2 flex items-baseline gap-2">
        <span className="num inline-flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-surface-3 text-[13.5px] text-ink">{n}</span>
        <span className="text-[15.5px] font-semibold text-ink">{title}</span>
      </div>
      {children}
      {note && <div className="mt-1.5 text-[14px] leading-snug text-muted">{note}</div>}
    </li>
  );
}
