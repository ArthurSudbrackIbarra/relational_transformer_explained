/**
 * Minimal floating-element positioning + popover primitives (no dependency).
 *
 *  - `useFloatingStyle` keeps a `position: fixed` element next to its anchor,
 *    flipping above/below and clamping to the viewport.
 *  - `HoverCard`: opens on mouse hover / keyboard focus / tap; used by Cite and Term.
 *  - `Popover`: click-to-open dialog that closes on outside click and Esc; used by
 *    the top bar's simulation controls.
 */
import {
  useCallback,
  useEffect,
  useId,
  useLayoutEffect,
  useRef,
  useState,
  type CSSProperties,
  type ReactNode,
  type RefObject,
} from 'react';
import { createPortal } from 'react-dom';
import { cx } from './cx';

export type Placement = 'bottom' | 'top';
export type Align = 'center' | 'start' | 'end';

const GAP = 8;
const MARGIN = 8;

/** Compute a fixed-position style for `floating` relative to `anchor`. */
export function useFloatingStyle(
  open: boolean,
  anchor: RefObject<HTMLElement | null>,
  floating: RefObject<HTMLElement | null>,
  placement: Placement = 'bottom',
  align: Align = 'center',
): CSSProperties {
  const [style, setStyle] = useState<CSSProperties>({ position: 'fixed', top: -9999, left: -9999, visibility: 'hidden' });

  const update = useCallback(() => {
    const a = anchor.current;
    const f = floating.current;
    if (!a || !f) return;
    const r = a.getBoundingClientRect();
    const w = f.offsetWidth;
    const h = f.offsetHeight;
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    let side: Placement = placement;
    if (side === 'bottom' && r.bottom + GAP + h > vh - MARGIN && r.top - GAP - h > MARGIN) side = 'top';
    else if (side === 'top' && r.top - GAP - h < MARGIN && r.bottom + GAP + h < vh - MARGIN) side = 'bottom';
    const top = side === 'bottom' ? r.bottom + GAP : r.top - GAP - h;
    let left = align === 'start' ? r.left : align === 'end' ? r.right - w : r.left + r.width / 2 - w / 2;
    left = Math.max(MARGIN, Math.min(left, vw - w - MARGIN));
    setStyle({ position: 'fixed', top: Math.max(MARGIN, top), left, visibility: 'visible' });
  }, [anchor, floating, placement, align]);

  useLayoutEffect(() => {
    if (!open) return;
    update();
    const onMove = () => update();
    window.addEventListener('resize', onMove);
    window.addEventListener('scroll', onMove, true);
    const ro = typeof ResizeObserver !== 'undefined' ? new ResizeObserver(onMove) : null;
    if (floating.current) ro?.observe(floating.current);
    return () => {
      window.removeEventListener('resize', onMove);
      window.removeEventListener('scroll', onMove, true);
      ro?.disconnect();
    };
  }, [open, update, floating]);

  return style;
}

/** Portal into document.body (SSR-safe). */
export function Portal({ children }: { children: ReactNode }) {
  if (typeof document === 'undefined') return null;
  return createPortal(children, document.body);
}

// ─────────────────────────────────────────────────────────────────────────────
// HoverCard
// ─────────────────────────────────────────────────────────────────────────────

export interface HoverCardTriggerProps {
  ref: (el: HTMLElement | null) => void;
  'aria-expanded': boolean;
  'aria-controls': string;
  'aria-describedby'?: string;
  onPointerEnter: (e: React.PointerEvent) => void;
  onPointerLeave: (e: React.PointerEvent) => void;
  onPointerDown: (e: React.PointerEvent) => void;
  onFocus: () => void;
  onBlur: (e: React.FocusEvent) => void;
  onClick: (e: React.MouseEvent) => void;
  onKeyDown: (e: React.KeyboardEvent) => void;
}

export interface HoverCardProps {
  /** Render the trigger; spread the given props on a focusable element. */
  trigger: (props: HoverCardTriggerProps, open: boolean) => ReactNode;
  /** Card content. */
  children: ReactNode;
  placement?: Placement;
  align?: Align;
  /** Card width in px (max; shrinks on small screens). */
  width?: number;
  className?: string;
  /** Accessible label of the card region. */
  label?: string;
  /**
   * The card contains links/buttons: it becomes a non-modal dialog, and Tab from
   * the trigger moves focus into it (Esc or tabbing out returns to the trigger).
   */
  interactive?: boolean;
}

const FOCUSABLE = 'a[href], button:not([disabled]), input, select, textarea, [tabindex]:not([tabindex="-1"])';

/**
 * Hover/focus card. Mouse: opens after a short delay, stays open while the
 * pointer is over trigger or card. Keyboard: opens on focus, Esc closes.
 * Touch: tap toggles.
 */
export function HoverCard({
  trigger,
  children,
  placement = 'top',
  align = 'center',
  width = 320,
  className,
  label,
  interactive = false,
}: HoverCardProps) {
  const [open, setOpen] = useState(false);
  const anchorRef = useRef<HTMLElement | null>(null);
  const cardRef = useRef<HTMLDivElement | null>(null);
  const timer = useRef<number | null>(null);
  const lastPointer = useRef<string>('mouse');
  const id = useId();
  const style = useFloatingStyle(open, anchorRef, cardRef, placement, align);

  const clear = () => {
    if (timer.current !== null) window.clearTimeout(timer.current);
    timer.current = null;
  };
  const openSoon = (ms = 120) => {
    clear();
    timer.current = window.setTimeout(() => setOpen(true), ms);
  };
  const closeSoon = (ms = 160) => {
    clear();
    timer.current = window.setTimeout(() => setOpen(false), ms);
  };
  useEffect(() => clear, []);

  // Close on outside tap when opened by touch/click.
  useEffect(() => {
    if (!open) return;
    const onDown = (e: PointerEvent) => {
      const t = e.target as Node;
      if (anchorRef.current?.contains(t) || cardRef.current?.contains(t)) return;
      setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(false);
    };
    document.addEventListener('pointerdown', onDown);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('pointerdown', onDown);
      document.removeEventListener('keydown', onKey);
    };
  }, [open]);

  const props: HoverCardTriggerProps = {
    ref: (el) => {
      anchorRef.current = el;
    },
    'aria-expanded': open,
    'aria-controls': id,
    'aria-describedby': open ? id : undefined,
    onPointerEnter: (e) => {
      if (e.pointerType === 'mouse') openSoon();
    },
    onPointerLeave: (e) => {
      if (e.pointerType === 'mouse') closeSoon();
    },
    onPointerDown: (e) => {
      lastPointer.current = e.pointerType;
    },
    onFocus: () => openSoon(0),
    onBlur: (e) => {
      if (cardRef.current?.contains(e.relatedTarget as Node)) return;
      closeSoon(80);
    },
    onClick: (e) => {
      e.preventDefault();
      clear();
      // mouse: a click pins the already-hovered card open; touch / keyboard: toggle
      setOpen((o) => (lastPointer.current === 'mouse' ? true : !o));
      lastPointer.current = 'keyboard';
    },
    onKeyDown: (e) => {
      if (e.key === 'Escape') setOpen(false);
      if (e.key === 'Tab' && !e.shiftKey && open && interactive) {
        const first = cardRef.current?.querySelector<HTMLElement>(FOCUSABLE);
        if (first) {
          e.preventDefault();
          clear();
          first.focus();
        }
      }
    },
  };

  return (
    <>
      {trigger(props, open)}
      {open && (
        <Portal>
          <div
            ref={cardRef}
            id={id}
            role={interactive ? 'dialog' : 'tooltip'}
            aria-modal={interactive ? false : undefined}
            aria-label={label}
            onKeyDown={(e) => {
              if (!interactive) return;
              const els = Array.from(cardRef.current?.querySelectorAll<HTMLElement>(FOCUSABLE) ?? []);
              const last = els[els.length - 1];
              if (e.key === 'Escape' || (e.key === 'Tab' && !e.shiftKey && document.activeElement === last)) {
                e.preventDefault();
                setOpen(false);
                anchorRef.current?.focus();
              }
            }}
            onBlur={(e) => {
              if (!interactive) return;
              const next = e.relatedTarget as Node | null;
              if (next && (cardRef.current?.contains(next) || anchorRef.current?.contains(next))) return;
              closeSoon(80);
            }}
            style={{ ...style, width: `min(${width}px, calc(100vw - 16px))`, zIndex: 70 }}
            className={cx(
              'rounded-[10px] border border-border bg-surface p-3 text-[16px] leading-snug text-ink shadow-[var(--shadow-pop)]',
              className,
            )}
            onPointerEnter={(e) => {
              if (e.pointerType === 'mouse') clear();
            }}
            onPointerLeave={(e) => {
              if (e.pointerType === 'mouse') closeSoon();
            }}
          >
            {children}
          </div>
        </Portal>
      )}
    </>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Popover (click)
// ─────────────────────────────────────────────────────────────────────────────

export interface PopoverTriggerProps {
  ref: (el: HTMLElement | null) => void;
  'aria-expanded': boolean;
  'aria-controls': string;
  'aria-haspopup': 'dialog';
  onClick: () => void;
}

export interface PopoverProps {
  trigger: (props: PopoverTriggerProps, open: boolean) => ReactNode;
  children: ReactNode | ((close: () => void) => ReactNode);
  placement?: Placement;
  align?: Align;
  width?: number;
  /** Accessible name of the dialog. */
  label: string;
  className?: string;
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
}

/** Click-to-open non-modal dialog. Esc and outside clicks close it; focus returns to the trigger. */
export function Popover({
  trigger,
  children,
  placement = 'bottom',
  align = 'end',
  width = 360,
  label,
  className,
  open: openProp,
  onOpenChange,
}: PopoverProps) {
  const [openState, setOpenState] = useState(false);
  const open = openProp ?? openState;
  const setOpen = useCallback(
    (v: boolean) => {
      if (openProp === undefined) setOpenState(v);
      onOpenChange?.(v);
    },
    [openProp, onOpenChange],
  );
  const anchorRef = useRef<HTMLElement | null>(null);
  const panelRef = useRef<HTMLDivElement | null>(null);
  const id = useId();
  const style = useFloatingStyle(open, anchorRef, panelRef, placement, align);

  useEffect(() => {
    if (!open) return;
    const onDown = (e: PointerEvent) => {
      const t = e.target as Node;
      if (anchorRef.current?.contains(t) || panelRef.current?.contains(t)) return;
      setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setOpen(false);
        anchorRef.current?.focus();
      }
    };
    document.addEventListener('pointerdown', onDown);
    document.addEventListener('keydown', onKey);
    // move focus into the dialog
    const t = window.setTimeout(() => {
      const first = panelRef.current?.querySelector<HTMLElement>(
        'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])',
      );
      first?.focus();
    }, 0);
    return () => {
      document.removeEventListener('pointerdown', onDown);
      document.removeEventListener('keydown', onKey);
      window.clearTimeout(t);
    };
  }, [open, setOpen]);

  const close = useCallback(() => setOpen(false), [setOpen]);

  return (
    <>
      {trigger(
        {
          ref: (el) => {
            anchorRef.current = el;
          },
          'aria-expanded': open,
          'aria-controls': id,
          'aria-haspopup': 'dialog',
          onClick: () => setOpen(!open),
        },
        open,
      )}
      {open && (
        <Portal>
          <div
            ref={panelRef}
            id={id}
            role="dialog"
            aria-modal="false"
            aria-label={label}
            style={{ ...style, width: `min(${width}px, calc(100vw - 16px))`, zIndex: 80 }}
            className={cx(
              'max-h-[calc(100vh-80px)] overflow-y-auto rounded-xl border border-border bg-surface text-ink shadow-[var(--shadow-pop)] scroll-quiet',
              className,
            )}
          >
            {typeof children === 'function' ? children(close) : children}
          </div>
        </Portal>
      )}
    </>
  );
}
