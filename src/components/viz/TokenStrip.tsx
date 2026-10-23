/**
 * TokenStrip: the context window as a sequence of token chips, colored by
 * table (left bar), sem type as a glyph, the masked target as a dashed accent
 * chip with "?", padding grayed (BUILD_PLAN §5).
 *
 * Takes engine `Token`s (types only). Order = sampler emission order (spec
 * §4.12); order has no effect on the model (no positional encodings, §6.5).
 */
import { useMemo, useRef, useState, type KeyboardEvent } from 'react';
import { SemType, type Token } from '@/engine/types';
import { catVar } from '@/styles/palette';
import { Portal, useFloatingStyle } from '../ui/floating';
import { cx } from '../ui/cx';
import { formatRaw, formatSigned, truncate } from './format';
import { SemBadge } from './SemBadge';
import { toSet } from './useSize';

export interface TokenStripProps {
  tokens: readonly Token[];
  /** tableName → categorical slot (task tables share slot 5, BUILD_PLAN §3.5). */
  tableSlots: Record<string, number>;
  /** 'chips' = column + value labels; 'cells' = compact squares (good for L ≥ 128). */
  variant?: 'chips' | 'cells';
  /** Positions to ring in accent. */
  highlight?: ReadonlySet<number> | readonly number[];
  /** Dim every token not in `highlight` (when highlight is non-empty). */
  dimOthers?: boolean;
  /** Positions to dim explicitly. */
  muted?: ReadonlySet<number> | readonly number[];
  selected?: number | null;
  onHover?: (pos: number | null) => void;
  onSelect?: (pos: number) => void;
  /** 'collapsed' (default) shows one "+N padding" chip; true shows every padding token; false hides them. */
  showPadding?: boolean | 'collapsed';
  /** Accessible name. */
  label?: string;
  className?: string;
  /** Show the z-scored stored value in the hover card (default true). */
  showStored?: boolean;
}

/** Display string for a token's value (raw, never what the model reads). */
export function tokenValueText(t: Token): string {
  if (t.isPadding) return 'pad';
  if (t.isTarget) return '?';
  return formatRaw(t.raw, t.sem === SemType.DateTime ? 'datetime' : undefined);
}

/** The stored model-side scalar for a token (z-score etc.), for tooltips. */
export function tokenStoredValue(t: Token): string {
  switch (t.sem) {
    case SemType.Number:
      return `number_values = ${formatSigned(t.numberValue)}`;
    case SemType.DateTime:
      return `datetime_values = ${formatSigned(t.datetimeValue)}`;
    case SemType.Boolean:
      return `boolean_values = ${formatSigned(t.booleanValue)}`;
    default:
      return `text_values = E(text[${t.textIdx}])`;
  }
}

export function TokenStrip({
  tokens,
  tableSlots,
  variant = 'chips',
  highlight,
  dimOthers = false,
  muted,
  selected = null,
  onHover,
  onSelect,
  showPadding = 'collapsed',
  label = 'Context window tokens',
  className,
  showStored = true,
}: TokenStripProps) {
  const hl = toSet(highlight);
  const mu = toSet(muted);
  const [hoverPos, setHoverPos] = useState<number | null>(null);
  const [focusPos, setFocusPos] = useState<number | null>(null);
  const anchor = useRef<HTMLElement | null>(null);
  const card = useRef<HTMLDivElement | null>(null);
  const style = useFloatingStyle(hoverPos !== null, anchor, card, 'top', 'center');
  const refs = useRef(new Map<number, HTMLButtonElement>());

  const { shown, padCount } = useMemo(() => {
    const real = tokens.filter((t) => !t.isPadding);
    const pads = tokens.length - real.length;
    if (showPadding === true) return { shown: tokens, padCount: 0 };
    return { shown: real, padCount: showPadding === 'collapsed' ? pads : 0 };
  }, [tokens, showPadding]);

  const interactive = !!onSelect;
  const rovingPos = focusPos ?? selected ?? shown[0]?.pos ?? 0;

  const move = (from: number, delta: number) => {
    const idx = shown.findIndex((t) => t.pos === from);
    const next = shown[Math.max(0, Math.min(shown.length - 1, idx + delta))];
    if (!next) return;
    setFocusPos(next.pos);
    refs.current.get(next.pos)?.focus();
  };

  const onKey = (e: KeyboardEvent, pos: number) => {
    if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {
      e.preventDefault();
      move(pos, 1);
    } else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
      e.preventDefault();
      move(pos, -1);
    } else if (e.key === 'Home') {
      e.preventDefault();
      move(pos, -Infinity);
    } else if (e.key === 'End') {
      e.preventDefault();
      move(pos, Infinity);
    }
  };

  const enter = (pos: number, el: HTMLElement) => {
    anchor.current = el;
    setHoverPos(pos);
    onHover?.(pos);
  };
  const leave = () => {
    setHoverPos(null);
    onHover?.(null);
  };

  const hovered = hoverPos !== null ? tokens.find((t) => t.pos === hoverPos) : undefined;

  return (
    <div className={cx('min-w-0', className)}>
      <div
        role={interactive ? 'listbox' : 'list'}
        aria-label={label}
        className={cx('flex flex-wrap', variant === 'cells' ? 'gap-[3px]' : 'gap-1.5')}
      >
        {shown.map((t) => {
          const slot = t.isPadding ? 0 : tableSlots[t.tableName] ?? 0;
          const isHl = hl.has(t.pos);
          const isDim = mu.has(t.pos) || (dimOthers && hl.size > 0 && !isHl);
          const isSel = selected === t.pos;
          const common = {
            ref: (el: HTMLButtonElement | null) => {
              if (el) refs.current.set(t.pos, el);
              else refs.current.delete(t.pos);
            },
            role: interactive ? 'option' : 'listitem',
            'aria-selected': interactive ? isSel : undefined,
            'aria-label': `token ${t.pos}: ${t.isPadding ? 'padding' : `${t.colName} of ${t.tableName}, ${t.isTarget ? 'masked target' : tokenValueText(t)}`}`,
            tabIndex: interactive ? (t.pos === rovingPos ? 0 : -1) : undefined,
            onPointerEnter: (e: React.PointerEvent<HTMLElement>) => enter(t.pos, e.currentTarget),
            onPointerLeave: leave,
            onFocus: (e: React.FocusEvent<HTMLElement>) => {
              setFocusPos(t.pos);
              // Keyboard focus previews the token; the focus a tap or click leaves behind must not
              // keep the tooltip open after the pointer has gone (touch fires focus after pointerleave).
              if (e.currentTarget.matches(':focus-visible')) enter(t.pos, e.currentTarget);
            },
            onBlur: leave,
            onClick: interactive ? () => onSelect?.(t.pos) : undefined,
            onKeyDown: interactive ? (e: KeyboardEvent) => onKey(e, t.pos) : undefined,
          };
          const Tag = (interactive ? 'button' : 'span') as 'button';
          if (variant === 'cells') {
            return (
              <Tag
                key={t.pos}
                type={interactive ? 'button' : undefined}
                {...common}
                className={cx(
                  'relative inline-flex h-[20px] w-[20px] items-center justify-center rounded-[4px] text-[13.5px] font-bold transition-opacity',
                  t.isPadding && 'border border-dashed border-axis',
                  t.isTarget && 'target-outline bg-accent-soft text-accent-text',
                  isDim && 'opacity-25',
                  (isHl || isSel) && 'ring-2 ring-accent ring-offset-1 ring-offset-surface',
                )}
                style={!t.isPadding && !t.isTarget ? { background: catVar(slot) } : undefined}
              >
                {t.isTarget ? '?' : null}
              </Tag>
            );
          }
          return (
            <Tag
              key={t.pos}
              type={interactive ? 'button' : undefined}
              {...common}
              className={cx(
                'relative inline-flex h-[50px] max-w-[176px] min-w-[58px] flex-col justify-center gap-[4px] overflow-hidden rounded-[7px] border bg-surface py-1 pr-2.5 pl-[10px] text-left transition-[opacity,box-shadow]',
                t.isPadding ? 'border-dashed border-axis bg-transparent text-muted' : 'border-border',
                t.isTarget && 'target-outline border-transparent bg-accent-soft',
                isDim && 'opacity-30',
                isSel && 'ring-2 ring-accent',
                isHl && !isSel && 'ring-2 ring-accent/60',
                interactive && 'hover:bg-surface-2',
              )}
            >
              {!t.isPadding && (
                <span aria-hidden="true" className="absolute top-0 bottom-0 left-0 w-[4px]" style={{ background: catVar(slot) }} />
              )}
              <span className="flex min-w-0 items-center gap-1">
                {!t.isPadding && <SemBadge sem={t.sem} size="xs" bare className="text-muted" />}
                <span className="truncate text-[13.5px] leading-none text-ink-2">
                  {t.isPadding ? `#${t.pos}` : t.colName}
                </span>
              </span>
              <span
                className={cx(
                  'truncate text-[15.5px] leading-[1.1]',
                  t.isTarget ? 'font-bold text-accent-text' : t.isPadding ? 'text-muted' : 'text-ink',
                )}
              >
                {t.isTarget ? '?' : truncate(tokenValueText(t), 17)}
              </span>
            </Tag>
          );
        })}
        {padCount > 0 && (
          <span
            role="listitem"
            className={cx(
              'inline-flex items-center rounded-[7px] border border-dashed border-axis px-2 text-muted',
              variant === 'cells' ? 'h-[20px] text-[13.5px]' : 'h-[50px] text-[14.5px]',
            )}
          >
            + {padCount} padding
          </span>
        )}
      </div>

      {hovered && (
        <Portal>
          <div
            ref={card}
            role="tooltip"
            style={{ ...style, zIndex: 70 }}
            className="pointer-events-none max-w-[320px] rounded-[9px] border border-border bg-surface px-3 py-2 text-[15.5px] leading-snug shadow-[var(--shadow-pop)]"
          >
            {hovered.isPadding ? (
              <div className="text-ink-2">
                <span className="num text-ink">#{hovered.pos}</span> padding (is_padding = true, every field at its default)
              </div>
            ) : (
              <>
                <div className="num font-semibold text-ink">
                  {hovered.isTarget ? '? masked target' : tokenValueText(hovered)}
                </div>
                <div className="mt-0.5 text-ink-2">
                  <span>{hovered.colName}</span> of{' '}
                  <span>{hovered.tableName}</span>
                  {hovered.tableType && hovered.tableType !== 'Db' ? ` (${hovered.tableType})` : ''} · row {hovered.rowPos}
                </div>
                <div className="num mt-1 text-[14.5px] text-muted">
                  pos {hovered.pos} · node {hovered.nodeIdx}
                  {hovered.depth >= 0 ? ` · depth ${hovered.depth}` : ''}
                </div>
                {showStored && (
                  <div className="num mt-0.5 text-[14.5px] text-muted">
                    <span className="font-mono">{tokenStoredValue(hovered)}</span>
                    {hovered.isTarget ? ' (stored, gated out by the mask)' : ''}
                  </div>
                )}
              </>
            )}
          </div>
        </Portal>
      )}
    </div>
  );
}
