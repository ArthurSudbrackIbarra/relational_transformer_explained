/**
 * Section 1 of the attention chapter: choosing what to inspect.
 *  - SublayerMap: the 4 blocks × (col, feat, nbr, full, SwiGLU) grid of the mini model, each attention
 *    cell showing the size of the update it adds to the query's residual stream (live, from the capture).
 *  - QueryPicker: presets derived from the live context + every token as a clickable chip.
 *  - SelectionBar: the compact block / layer / head / query controls reused above each figure.
 * All selections write the store's inspect fields (shared with the masks chapter).
 */
import { ArrowRight } from 'lucide-react';
import { useMemo } from 'react';
import { Link } from 'react-router-dom';
import { LayerLabel, LAYER_INFO, Segmented, Select, TokenStrip, cx } from '@/components';
import { ATTN_ORDER, type AttnLayer } from '@/engine/types';
import { article, int, norm2, rowName, rowOf, singular, valueOf } from './utils';
import type { AttnLive } from './useAttnLive';

// ─────────────────────────────────────────────────────────────────────────────
// Compact selection bar
// ─────────────────────────────────────────────────────────────────────────────

export function SelectionBar({ live, showQuery = true, showLayer = true, showHead = true }: { live: AttnLive; showQuery?: boolean; showLayer?: boolean; showHead?: boolean }) {
  const { nBlocks, H, block, layer, head, setBlock, setLayer, setHead } = live;
  return (
    <>
      <Segmented
        label="Block"
        value={block}
        onChange={setBlock}
        options={Array.from({ length: nBlocks }, (_, i) => ({ value: i, label: `block ${i + 1}` }))}
      />
      {showLayer && (
        <Segmented
          label="Attention layer"
          value={layer}
          onChange={setLayer}
          options={ATTN_ORDER.map((l) => ({ value: l, label: <LayerLabel layer={l} size={13} />, title: `${LAYER_INFO[l].name} attention` }))}
        />
      )}
      {showHead && (
        <Segmented label="Head" value={head} onChange={setHead} options={Array.from({ length: H }, (_, i) => ({ value: i, label: `head ${i + 1}` }))} />
      )}
      {showQuery && <QuerySelect live={live} />}
    </>
  );
}

export function QuerySelect({ live, className }: { live: AttnLive; className?: string }) {
  const { ctx, labels, query, target, setQuery } = live;
  return (
    <Select
      label="Query token"
      hideLabel
      value={query}
      onChange={(e) => setQuery(Number(e.target.value))}
      className={cx('w-full min-w-[180px] sm:w-[340px]', className)}
    >
      {ctx.tokens
        .filter((t) => !t.isPadding)
        .map((t) => (
          <option key={t.pos} value={t.pos}>
            #{t.pos} {labels[t.pos]}
            {t.pos === target ? ' (masked target)' : ` = ${truncateVal(valueOf(t))}`}
          </option>
        ))}
    </Select>
  );
}

const truncateVal = (s: string) => (s.length > 18 ? `${s.slice(0, 17)}…` : s);

// ─────────────────────────────────────────────────────────────────────────────
// Sublayer map
// ─────────────────────────────────────────────────────────────────────────────

export function SublayerMap({ live }: { live: AttnLive }) {
  const { fr, nBlocks, block, layer, query, d, setBlock, setLayer, disabledLayers } = live;
  const norms = useMemo(() => {
    if (!fr) return null;
    return fr.blocks.map((b) => ATTN_ORDER.map((l) => norm2(rowOf(b.attn[l].out, query, d))));
  }, [fr, query, d]);
  const max = norms ? Math.max(1e-9, ...norms.flat()) : 1;
  const blocks = Array.from({ length: nBlocks }, (_, b) => b);

  return (
    <div className="min-w-0">
      {/* Rows follow the order inside a block (model.py:83-85); columns are the blocks. */}
      <table className="w-full table-fixed border-separate border-spacing-[4px] text-[15px] sm:border-spacing-[6px]" aria-label="The attention sublayers of the mini model">
        <colgroup>
          <col className="w-[100px] sm:w-[150px]" />
          {blocks.map((b) => (
            <col key={b} />
          ))}
        </colgroup>
        <thead>
          <tr>
            <th scope="col" className="pb-1 text-left text-[14.5px] font-medium text-muted">
              <span className="sr-only">Sublayer</span>
              <span aria-hidden="true" className="sm:hidden">
                block →
              </span>
            </th>
            {blocks.map((b) => (
              <th key={b} scope="col" className="pb-1 pl-1 text-left text-[14.5px] font-medium whitespace-nowrap text-ink-2">
                <span className="hidden sm:inline">block </span>
                <span className="sr-only sm:hidden">block </span>
                {b + 1}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {ATTN_ORDER.map((l, li) => {
            const off = disabledLayers.includes(l);
            return (
              <tr key={l}>
                <th scope="row" className="pr-1 text-left align-middle font-normal whitespace-nowrap">
                  <span className="num mr-1.5 hidden text-[14px] text-muted sm:inline">{li + 1}</span>
                  <LayerLabel layer={l} size={14} />
                </th>
                {blocks.map((b) => {
                  const v = norms?.[b][li] ?? null;
                  const on = b === block && l === layer;
                  return (
                    <td key={b} className="p-0 align-middle">
                      <button
                        type="button"
                        aria-pressed={on}
                        aria-label={`Block ${b + 1}, ${LAYER_INFO[l].name} attention${off ? ': switched off' : v !== null ? `: update length ${v.toFixed(3)}` : ''}`}
                        onClick={() => {
                          setBlock(b);
                          setLayer(l);
                        }}
                        className={cx(
                          'flex w-full flex-col gap-1.5 rounded-lg border px-2 py-1.5 text-left transition-colors sm:px-3 sm:py-2',
                          on ? 'border-accent bg-accent-soft' : 'border-border bg-surface hover:bg-surface-2',
                        )}
                      >
                        <span className={cx('num text-[15px] leading-none', on ? 'font-semibold text-ink' : 'text-ink-2')}>
                          {off ? 'off' : v === null ? '…' : v === 0 ? '0' : v.toFixed(2)}
                        </span>
                        <span className="block h-[5px] w-full overflow-hidden rounded-full bg-surface-3">
                          <span
                            className="block h-full rounded-full transition-[width] duration-500"
                            style={{ width: `${v ? Math.max(2, (v / max) * 100) : 0}%`, background: on ? 'var(--accent)' : 'var(--seq-400)' }}
                          />
                        </span>
                      </button>
                    </td>
                  );
                })}
              </tr>
            );
          })}
          <tr>
            <th scope="row" className="pr-1 text-left align-middle text-[15px] font-normal whitespace-nowrap text-muted">
              <span className="num mr-1.5 hidden text-[14px] sm:inline">5</span>
              SwiGLU
            </th>
            <td colSpan={nBlocks} className="rounded-lg border border-dashed border-border px-3 py-1.5 align-middle">
              <Link to="/block" className="inline-flex items-center gap-1.5 text-[15px] text-muted hover:text-accent-text" data-plain>
                feed-forward, chapter 11 <ArrowRight aria-hidden="true" size={14} />
              </Link>
            </td>
          </tr>
        </tbody>
      </table>
      <p className="!mb-0 mt-2 text-[14.5px] leading-snug text-muted">
        Each number is ‖Δx‖, the length of the update that sublayer adds to the query’s residual stream. Read a column top to bottom to follow
        one block in execution order.
      </p>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Query presets
// ─────────────────────────────────────────────────────────────────────────────

export interface QueryPreset {
  id: string;
  label: string;
  pos: number;
}

/** Presets derived from the live context (never hard-coded positions). */
export function queryPresets(live: AttnLive): QueryPreset[] {
  const { ctx, target, inCtx, task } = live;
  const out: QueryPreset[] = [];
  const toks = ctx.tokens;
  if (ctx.targetPos < 0) return out;
  out.push({ id: 'target', label: 'the masked target', pos: target });
  const seed = toks[target];
  const sibling = toks.find((t) => !t.isPadding && t.nodeIdx === seed.nodeIdx && t.pos !== target);
  if (sibling) out.push({ id: 'sibling', label: `the seed row’s ${sibling.colName}`, pos: sibling.pos });
  if (inCtx.self.length) out.push({ id: 'self', label: `a past label of the same ${task.entityTable ? singular(task.entityTable) : 'entity'}`, pos: inCtx.self[0].pos });
  const entity = seed.f2pNbrIdxs.find((p) => p >= 0);
  if (entity !== undefined) {
    const ent = toks.find((t) => !t.isPadding && t.nodeIdx === entity);
    if (ent) out.push({ id: 'entity', label: `${article(singular(ent.tableName))} ${singular(ent.tableName)} cell`, pos: ent.pos });
    const child = toks.find((t) => !t.isPadding && t.tableType === 'Db' && t.f2pNbrIdxs.includes(entity) && t.nodeIdx !== seed.nodeIdx);
    if (child) {
      out.push({ id: 'child', label: `${article(singular(child.tableName))} ${singular(child.tableName)} cell`, pos: child.pos });
      const other = child.f2pNbrIdxs.find((p) => p >= 0 && p !== entity && toks.some((t) => !t.isPadding && t.nodeIdx === p));
      if (other !== undefined) {
        const o = toks.find((t) => !t.isPadding && t.nodeIdx === other)!;
        out.push({ id: 'other-parent', label: `${article(singular(o.tableName))} ${singular(o.tableName)} cell`, pos: o.pos });
      }
    }
  }
  return out;
}

export function PresetButtons({ live, presets }: { live: AttnLive; presets: QueryPreset[] }) {
  const { query, setQuery } = live;
  return (
    <div className="flex flex-wrap gap-1.5" role="group" aria-label="Query presets">
      {presets.map((p) => (
        <button
          key={p.id}
          type="button"
          aria-pressed={query === p.pos}
          onClick={() => setQuery(p.pos)}
          className={cx(
            'h-7 rounded-full border px-2.5 text-[15px] font-medium transition-colors',
            query === p.pos ? 'border-accent bg-accent-soft text-accent-text' : 'border-border bg-surface text-ink-2 hover:bg-surface-2 hover:text-ink',
          )}
        >
          {p.label}
        </button>
      ))}
    </div>
  );
}

export function QueryPicker({ live }: { live: AttnLive }) {
  const { ctx, slots, query, setQuery } = live;
  const presets = useMemo(() => queryPresets(live), [live]);
  return (
    <div className="grid gap-3">
      <PresetButtons live={live} presets={presets} />
      <TokenStrip
        tokens={ctx.tokens}
        tableSlots={slots}
        variant="cells"
        selected={query}
        onSelect={(p) => setQuery(p)}
        label="Context tokens: click one to make it the query"
      />
      <QueryCaption live={live} />
    </div>
  );
}

export function QueryCaption({ live }: { live: AttnLive }) {
  const { ctx, query, target, labels } = live;
  const t = ctx.tokens[query];
  if (!t) return null;
  return (
    <p className="!mb-0 text-[15.5px] leading-snug text-ink-2">
      <span className="eyebrow mr-1.5 !text-[13.5px]">Query</span>
      <span className="num text-ink">
        #{query} {labels[query]}
      </span>{' '}
      · {t.colName} of <span className="font-mono">{rowName(t)}</span> ·{' '}
      {query === target ? <strong className="font-semibold text-accent-text">masked target “?”</strong> : <span className="num text-ink">{valueOf(t)}</span>}{' '}
      · node {int(t.nodeIdx)}
    </p>
  );
}
