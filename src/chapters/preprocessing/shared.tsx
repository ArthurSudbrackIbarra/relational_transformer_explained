/**
 * Chapter-local hook and small presentational pieces shared by the figures of
 * chapter 4. All data comes from the global simulation hooks (same DB, task and
 * seed as every other chapter).
 */
import { KeyRound, Link2 } from 'lucide-react';
import { useMemo, useSyncExternalStore, type ReactNode } from 'react';
import { SemBadge, TableSwatch, cx } from '@/components';
import { semOfDType } from '@/components/viz/format';
import type { GraphNode, PreprocessedDB, RawDatabase, TaskDef } from '@/engine/types';
import { useDatabase, usePreprocessed, useSeedNode, useTableSlots, useTask } from '@/state/hooks';
import { colRole, fileLabel, tableFiles, type TableFile } from './data';

export interface ChapterData {
  db: RawDatabase;
  pdb: PreprocessedDB;
  task: TaskDef;
  slots: Record<string, number>;
  files: TableFile[];
  seedNode: number;
  seed: GraphNode | undefined;
  /** Forecast: the seed task row's entity (its single FK parent). Autocomplete: null. */
  entityNode: number | null;
  fileByKey: Map<string, TableFile>;
  fileOfNode: (n: number) => TableFile | undefined;
  slotOf: (tableName: string) => number;
}

export function useChapterData(): ChapterData {
  const db = useDatabase();
  const pdb = usePreprocessed();
  const task = useTask();
  const slots = useTableSlots();
  const seedNode = useSeedNode();
  return useMemo(() => {
    const files = tableFiles(db, pdb);
    const fileByKey = new Map(files.map((f) => [f.key, f]));
    const seed = seedNode >= 0 ? pdb.nodes[seedNode] : undefined;
    const entityNode = task.kind === 'forecast' && seed && seed.f2pNbrIdxs.length ? seed.f2pNbrIdxs[0] : null;
    const fileOfNode = (n: number) => files.find((f) => n >= f.offset && n < f.offset + f.numNodes);
    const slotOf = (t: string) => slots[t] ?? 0;
    return { db, pdb, task, slots, files, seedNode, seed, entityNode, fileByKey, fileOfNode, slotOf };
  }, [db, pdb, task, slots, seedNode]);
}

/** True while `query` matches (e.g. '(max-width: 639px)'); false where matchMedia is unavailable. */
export function useMedia(query: string): boolean {
  return useSyncExternalStore(
    (cb) => {
      try {
        const m = window.matchMedia(query);
        m.addEventListener('change', cb);
        return () => m.removeEventListener('change', cb);
      } catch {
        return () => {};
      }
    },
    () => {
      try {
        return window.matchMedia(query).matches;
      } catch {
        return false;
      }
    },
    () => false,
  );
}

/** Phone widths (below Tailwind's sm breakpoint): long formulas break into lines there. */
export const usePhone = () => useMedia('(max-width: 639px)');

/** Table identity: color swatch (+ shape) and the file label in mono ink. */
export function FileTag({ file, slot, className, showType = true }: { file: TableFile; slot: number; className?: string; showType?: boolean }) {
  return (
    <TableSwatch
      slot={slot}
      shape
      size={9}
      className={className}
      label={
        showType ? (
          fileLabel(file)
        ) : (
          file.tableName
        )
      }
    />
  );
}

/** Glyph for a column: key icon, link icon or its sem-type badge. */
export function ColGlyph({ file, col, size = 'xs' }: { file: TableFile; col: string; size?: 'xs' | 'sm' }) {
  const role = colRole(file.table, col);
  const px = size === 'xs' ? 13 : 15;
  if (role === 'pk') return <KeyRound aria-label="primary key" size={px} className="shrink-0 text-ink-2" />;
  if (role === 'fk') return <Link2 aria-label="foreign key" size={px} className="shrink-0 text-ink-2" />;
  const c = file.table.schema.columns.find((x) => x.name === col);
  return <SemBadge sem={semOfDType(c?.dtype ?? 'float')} size={size} />;
}

/** A small labeled readout used under figures: "μ = 37.01". */
export function Readout({ label, children, className, hint }: { label: ReactNode; children: ReactNode; className?: string; hint?: ReactNode }) {
  return (
    <div className={cx('min-w-0 rounded-lg bg-surface-2 px-3 py-2', className)}>
      <div className="text-[14px] font-medium text-ink-2">{label}</div>
      <div className="num mt-0.5 text-[17.5px] leading-tight font-semibold text-ink">{children}</div>
      {hint && <div className="mt-0.5 text-[14px] leading-snug text-muted">{hint}</div>}
    </div>
  );
}

/** A quick-pick chip button. */
export function PickChip({ active, onClick, children, title }: { active?: boolean; onClick: () => void; children: ReactNode; title?: string }) {
  return (
    <button
      type="button"
      onClick={onClick}
      title={title}
      aria-pressed={active}
      className={cx(
        'inline-flex h-7 items-center gap-1.5 rounded-full border px-2.5 text-[15px] font-medium transition-colors',
        active ? 'border-accent bg-accent-soft text-accent-text' : 'border-border-strong bg-surface text-ink-2 hover:bg-surface-2 hover:text-ink',
      )}
    >
      {children}
    </button>
  );
}

/**
 * A subscript that never shrinks below the 14px caption floor (a plain <sub> is 0.75em,
 * i.e. 10.5px inside a 14px label). Used for μ_T, σ_T, μ_c, σ_c.
 */
export function Sub({ children }: { children: ReactNode }) {
  return <span className="relative top-[0.3em] text-[length:max(14px,0.8em)] leading-none">{children}</span>;
}

/** Class for a chapter wrapper: inline code never drops below 14px (the base rule floors it at 13.5px). */
export const CODE_FLOOR = '[&_:not(pre)>code]:text-[length:max(14px,0.86em)]!';

/** Inline code-ish token in prose. */
export function C({ children }: { children: ReactNode }) {
  return <code>{children}</code>;
}

/** Signed number with a real minus sign. */
export function signed(v: number, digits = 3): string {
  if (!Number.isFinite(v)) return Number.isNaN(v) ? 'NaN' : v > 0 ? '+∞' : '−∞';
  const s = Math.abs(v).toFixed(digits);
  return (v < 0 ? '−' : '+') + s;
}

/** Fixed digits with a real minus sign. */
export function fixed(v: number, digits = 3): string {
  if (!Number.isFinite(v)) return Number.isNaN(v) ? 'NaN' : v > 0 ? '+∞' : '−∞';
  return v.toFixed(digits).replace('-', '−');
}

/** Number for a TeX string (ASCII minus, trimmed trailing zeros when `trim`). */
export function texNum(v: number, digits = 3, opts: { sign?: boolean; trim?: boolean } = {}): string {
  if (!Number.isFinite(v)) return Number.isNaN(v) ? '\\text{NaN}' : v > 0 ? '+\\infty' : '-\\infty';
  let s = v.toFixed(digits);
  if (opts.trim) s = s.replace(/\.?0+$/, '');
  if (opts.sign && v >= 0) s = `+${s}`;
  return s;
}

/** Escape a plain string for use inside TeX \text{…} (truncated to `max` characters). */
export function texText(s: string, max = 40): string {
  const t = s.length > max ? s.slice(0, max - 1) + '…' : s;
  return t.replace(/[\\^~]/g, '').replace(/[_{}$&#%]/g, (m) => `\\${m}`);
}
