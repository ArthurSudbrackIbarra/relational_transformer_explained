/**
 * "Same softmax, three kernels": on the live forward pass, recompute one attention sublayer's weights the
 * dense way (every score, −∞ where masked, softmax over all S keys, as python/rt_model.py does) and compare
 * them with what the browser engine computed (a softmax over each query's allowed keys only, closer in
 * spirit to FlexAttention, which skips fully masked blocks). Also shows what each implementation does with
 * a query that has no allowed key: a naive dense softmax gives NaN; FlexAttention, the port and the
 * engine all give 0 (spec §6.1).
 */
import { useMemo, useState } from 'react';
import { ChevronLeft, ChevronRight } from 'lucide-react';
import type { ReactNode } from 'react';
import { Cite, IconButton, LayerLabel, Segmented, StatusBadge, cx } from '@/components';
import { divergingCss, maxAbs, sequentialCss } from '@/components/viz/scales';
import { TooltipBox } from '@/components/viz/chartParts';
import { useContextWindow, useForward, usePreprocessed } from '@/state/hooks';
import { useResolvedTheme } from '@/styles/theme';
import { ATTN_ORDER, type AttnLayer, type PreprocessedDB, type Token } from '@/engine/types';
import { int, sci } from './data';

function tokenLabel(t: Token, pdb: PreprocessedDB): string {
  if (t.isPadding) return 'padding';
  const table = pdb.text[t.tableNameIdx] ?? '?';
  const phrase = pdb.text[t.colNameIdx] ?? '';
  const col = phrase.split(' of ')[0];
  return `${table}[${t.rowPos}].${col}`;
}

interface LayerCheck {
  maxDiff: number;
  /** real query rows × heads compared */
  rows: number;
  /** real query rows with no allowed key */
  empty: number[];
  allowedPairs: number;
  realPairs: number;
}

/** Dense masked softmax of one row (float64), or NaN everywhere when no key is allowed (0/0). */
function denseRow(scores: Float32Array, mask: Uint8Array, off: number, mOff: number, S: number, out: Float64Array): void {
  let mx = -Infinity;
  for (let k = 0; k < S; k++) {
    const s = mask[mOff + k] ? scores[off + k] : -Infinity;
    out[k] = s;
    if (s > mx) mx = s;
  }
  let Z = 0;
  for (let k = 0; k < S; k++) {
    const e = Math.exp(out[k] - mx); // exp(−∞ − (−∞)) = exp(NaN) = NaN when the whole row is masked
    out[k] = e;
    Z += e;
  }
  for (let k = 0; k < S; k++) out[k] /= Z;
}

function ControlGroup({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="min-w-0">
      <div aria-hidden="true" className="mb-1 text-[15px] font-medium text-ink-2">
        {label}
      </div>
      {children}
    </div>
  );
}

export function AttentionKernels() {
  const fr = useForward();
  const ctx = useContextWindow();
  const pdb = usePreprocessed();
  const mode = useResolvedTheme();
  const [layer, setLayer] = useState<AttnLayer>('feat');
  const [block, setBlock] = useState(0);
  const [head, setHead] = useState(0);
  const [query, setQuery] = useState<number | null>(null);
  const [hoverKey, setHoverKey] = useState<number | null>(null);

  const S = ctx.tokens.length;
  const nBlocks = fr?.blocks.length ?? 0;
  const H = fr ? fr.blocks[0].attn.col.probs.length / (S * S) : 0;
  const q = query ?? ctx.targetPos;

  const check = useMemo<Record<AttnLayer, LayerCheck> | null>(() => {
    if (!fr) return null;
    const out = {} as Record<AttnLayer, LayerCheck>;
    const row = new Float64Array(S);
    const real = ctx.tokens.filter((t) => !t.isPadding).map((t) => t.pos);
    for (const l of ATTN_ORDER) {
      const a = fr.blocks[block].attn[l];
      const mask = fr.masks[l];
      let maxDiff = 0;
      let rows = 0;
      const empty: number[] = [];
      let allowed = 0;
      for (const qi of real) {
        let any = false;
        for (let k = 0; k < S; k++) if (mask[qi * S + k]) (any = true), allowed++;
        if (!any) {
          empty.push(qi);
          continue;
        }
        for (let h = 0; h < H; h++) {
          const off = h * S * S + qi * S;
          denseRow(a.scores, mask, off, qi * S, S, row);
          for (let k = 0; k < S; k++) {
            const d = Math.abs(row[k] - a.probs[off + k]);
            if (d > maxDiff) maxDiff = d;
          }
          rows++;
        }
      }
      out[l] = { maxDiff, rows, empty, allowedPairs: allowed, realPairs: real.length * real.length };
    }
    return out;
  }, [fr, ctx, block, H, S]);

  if (!fr || !check) return <p className="text-[16px] text-ink-2">Waiting for the live forward pass…</p>;

  const a = fr.blocks[block].attn[layer];
  const mask = fr.masks[layer];
  const off = head * S * S + q * S;
  const dense = new Float64Array(S);
  denseRow(a.scores, mask, off, q * S, S, dense);
  const scoresRow = a.scores.subarray(off, off + S);
  const probsRow = a.probs.subarray(off, off + S);
  const allowed = Array.from({ length: S }, (_, k) => mask[q * S + k] === 1);
  const nAllowed = allowed.filter(Boolean).length;
  const sMax = maxAbs(Array.from(scoresRow).filter((_, k) => !ctx.tokens[k].isPadding));
  const pMax = Math.max(1e-9, ...Array.from(probsRow));
  const rowDiff = nAllowed ? Math.max(...Array.from(dense).map((d, k) => Math.abs(d - probsRow[k]))) : NaN;
  const lc = check[layer];
  const emptyQ = lc.empty;
  const isEmpty = nAllowed === 0;

  const strip = (kind: 'score' | 'mask' | 'dense' | 'engine') => (
    <div className="flex h-[22px] w-full overflow-hidden rounded-[4px] border border-border" onPointerLeave={() => setHoverKey(null)}>
      {Array.from({ length: S }, (_, k) => {
        const pad = ctx.tokens[k].isPadding;
        let bg: string;
        if (pad) bg = 'var(--surface-3)';
        else if (kind === 'score') bg = divergingCss(scoresRow[k], sMax, mode);
        else if (kind === 'mask') bg = allowed[k] ? 'var(--accent)' : 'var(--surface-2)';
        else {
          const v = kind === 'dense' ? dense[k] : probsRow[k];
          // exactly 0 (a masked key, or a row with no allowed key) stays neutral; NaN is red
          bg = Number.isNaN(v) ? 'var(--critical)' : v === 0 ? 'var(--surface-2)' : sequentialCss(v / pMax, mode);
        }
        return (
          <span
            key={k}
            onPointerEnter={() => setHoverKey(k)}
            className={cx('h-full min-w-0 flex-1', hoverKey === k && 'outline-2 outline-ink outline-solid')}
            style={{ background: bg }}
          />
        );
      })}
    </div>
  );

  const hk = hoverKey;
  const firstEmpty = emptyQ[0];
  // step through real (non-padding) query positions
  let prevQ: number | null = null;
  let nextQ: number | null = null;
  for (let p = q - 1; p >= 0; p--) if (!ctx.tokens[p].isPadding) (prevQ = p), (p = -1);
  for (let p = q + 1; p < S; p++) if (!ctx.tokens[p].isPadding) (nextQ = p), (p = S);

  return (
    <div className="min-w-0">
      <div className="flex flex-wrap items-end gap-x-5 gap-y-3">
        <ControlGroup label="Layer">
          <Segmented
            label="Layer"
            value={layer}
            onChange={(v) => setLayer(v)}
            options={ATTN_ORDER.map((l) => ({ value: l, label: <LayerLabel layer={l} /> }))}
          />
        </ControlGroup>
        <ControlGroup label="Block">
          <Segmented label="Block" value={block} onChange={(v) => setBlock(v)} options={Array.from({ length: nBlocks }, (_, b) => ({ value: b, label: String(b + 1) }))} />
        </ControlGroup>
        <ControlGroup label="Head">
          <Segmented label="Head" value={head} onChange={(v) => setHead(v)} options={Array.from({ length: H }, (_, h) => ({ value: h, label: String(h + 1) }))} />
        </ControlGroup>
      </div>

      <div className="mt-4 flex flex-wrap items-center gap-2 text-[15.5px] text-ink-2">
        <span>Query:</span>
        <button
          type="button"
          onClick={() => setQuery(null)}
          aria-pressed={q === ctx.targetPos}
          className={cx(
            'rounded-full border px-2.5 py-0.5 text-[15px] font-medium',
            q === ctx.targetPos ? 'border-accent bg-accent-soft text-accent-text' : 'border-border text-ink-2 hover:text-ink',
          )}
        >
          the masked target
        </button>
        <button
          type="button"
          disabled={firstEmpty === undefined}
          aria-pressed={isEmpty}
          onClick={() => firstEmpty !== undefined && setQuery(firstEmpty)}
          className={cx(
            'rounded-full border px-2.5 py-0.5 text-[15px] font-medium disabled:opacity-45',
            isEmpty ? 'border-accent bg-accent-soft text-accent-text' : 'border-border text-ink-2 hover:text-ink',
          )}
          title={firstEmpty === undefined ? 'Every real query has at least one allowed key in this layer' : undefined}
        >
          a query with no allowed key
        </button>
        <span className="inline-flex items-center gap-1">
          <IconButton label="Previous query" size="sm" className="border border-border" disabled={prevQ === null} onClick={() => prevQ !== null && setQuery(prevQ)}>
            <ChevronLeft size={16} />
          </IconButton>
          <IconButton label="Next query" size="sm" className="border border-border" disabled={nextQ === null} onClick={() => nextQ !== null && setQuery(nextQ)}>
            <ChevronRight size={16} />
          </IconButton>
        </span>
        <span className="font-mono text-[15px] break-all text-ink">
          #{q} {tokenLabel(ctx.tokens[q], pdb)}
        </span>
      </div>

      <div className="relative mt-3 grid gap-2.5">
        {(
          [
            ['score', 'raw scores q·k/√d_head, all S keys'],
            ['mask', `mask row: ${nAllowed} of ${S} keys allowed`],
            ['dense', 'dense: −∞ where masked, softmax over all S'],
            ['engine', 'browser engine: softmax over allowed keys only'],
          ] as const
        ).map(([kind, label]) => (
          <div key={kind} className="grid min-w-0 gap-1 sm:grid-cols-[minmax(0,300px)_minmax(0,1fr)] sm:items-center sm:gap-3">
            <div className="text-[15px] text-ink-2">{label}</div>
            {strip(kind)}
          </div>
        ))}
        {hk !== null && (
          <TooltipBox style={{ right: 0, top: -8, transform: 'translateY(-100%)' }}>
            <div className="font-mono text-[15px] font-semibold text-ink">
              key #{hk} {tokenLabel(ctx.tokens[hk], pdb)}
            </div>
            <div className="mt-1 flex justify-between gap-4 text-ink-2">
              <span>score</span>
              <span className="num text-ink">{ctx.tokens[hk].isPadding ? '–' : scoresRow[hk].toFixed(4)}</span>
            </div>
            <div className="flex justify-between gap-4 text-ink-2">
              <span>allowed</span>
              <span className="text-ink">{allowed[hk] ? 'yes' : 'no'}</span>
            </div>
            <div className="flex justify-between gap-4 text-ink-2">
              <span>dense weight</span>
              <span className="num text-ink">{Number.isNaN(dense[hk]) ? 'NaN' : dense[hk].toFixed(6)}</span>
            </div>
            <div className="flex justify-between gap-4 text-ink-2">
              <span>engine weight</span>
              <span className="num text-ink">{probsRow[hk].toFixed(6)}</span>
            </div>
          </TooltipBox>
        )}
      </div>

      <div className="mt-4 grid gap-3 md:grid-cols-2">
        <div className="rounded-lg border border-border bg-surface-2 px-3.5 py-3">
          <div className="text-[16px] font-semibold text-ink">This query, head {head + 1}</div>
          {isEmpty ? (
            <p className="mt-1 mb-0 text-[15.5px] leading-snug text-ink-2">
              No key is allowed, so every score is −∞. A naive softmax divides 0 by 0 and returns{' '}
              <strong className="text-ink">NaN</strong> (the red strip). The engine skips the row and outputs exactly 0, like FlexAttention’s
              compiled kernel and the port <Cite kind="spec" ref="§6.1" />.
            </p>
          ) : (
            <p className="mt-1 mb-0 text-[15.5px] leading-snug text-ink-2">
              Masked keys get weight exp(−∞) = 0 in the dense version, so both rows hold the same {nAllowed} non-zero weights. Largest difference:{' '}
              <strong className="num text-ink">{sci(rowDiff, 1)}</strong>.
            </p>
          )}
        </div>
        <div className="rounded-lg border border-border bg-surface-2 px-3.5 py-3">
          <div className="text-[16px] font-semibold text-ink">
            Whole sublayer: <LayerLabel layer={layer} />, block {block + 1}
          </div>
          <p className="mt-1 mb-0 text-[15.5px] leading-snug text-ink-2">
            {int(lc.rows)} query rows × heads compared, largest |dense − engine| = <strong className="num text-ink">{sci(lc.maxDiff, 1)}</strong>.{' '}
            {lc.empty.length > 0 ? (
              <>
                {lc.empty.length} real {lc.empty.length === 1 ? 'query has' : 'queries have'} no allowed key: NaN in a naive softmax, 0 in all three
                implementations.
              </>
            ) : (
              <>Every real query has at least one allowed key.</>
            )}{' '}
            {int(lc.allowedPairs)} of {int(lc.realPairs)} real pairs are allowed ({((100 * lc.allowedPairs) / Math.max(1, lc.realPairs)).toFixed(1)}%).
          </p>
        </div>
      </div>
      {lc.maxDiff > 1e-6 && (
        <div className="mt-3">
          <StatusBadge status="warning">Unexpected difference above 10⁻⁶</StatusBadge>
        </div>
      )}
    </div>
  );
}
