/**
 * Display formatting shared by every view. Display only: nothing here is
 * read by the model.
 */
import type { DType, RawValue } from '@/engine/types';
import { SemType } from '@/engine/types';

/** The glyph used for a missing value everywhere (BUILD_PLAN §5 TableView). */
export const NULL_GLYPH = '∅';

/** dtype → sem type, exactly as pre.rs:543-611 (spec §3.2): no semantic inference. */
export function semOfDType(dtype: DType): SemType {
  switch (dtype) {
    case 'bool':
      return SemType.Boolean;
    case 'str':
      return SemType.Text;
    case 'datetime':
      return SemType.DateTime;
    default:
      return SemType.Number; // int, float
  }
}

const pad2 = (n: number) => String(n).padStart(2, '0');

/** Epoch ms → "YYYY-MM-DD" (UTC), plus " HH:MM" when not midnight. */
export function formatDate(ms: number, withTime: 'auto' | 'always' | 'never' = 'auto'): string {
  if (!Number.isFinite(ms)) return NULL_GLYPH;
  const d = new Date(ms);
  const date = `${d.getUTCFullYear()}-${pad2(d.getUTCMonth() + 1)}-${pad2(d.getUTCDate())}`;
  const hasTime = d.getUTCHours() !== 0 || d.getUTCMinutes() !== 0 || d.getUTCSeconds() !== 0;
  if (withTime === 'always' || (withTime === 'auto' && hasTime)) return `${date} ${pad2(d.getUTCHours())}:${pad2(d.getUTCMinutes())}`;
  return date;
}

/** Compact number: integers as-is, otherwise ≤ `digits` decimals with trailing zeros trimmed. */
export function formatNumber(v: number, digits = 3): string {
  if (!Number.isFinite(v)) return Number.isNaN(v) ? 'NaN' : v > 0 ? '+∞' : '−∞';
  if (Number.isInteger(v) && Math.abs(v) < 1e15) return v.toLocaleString('en-US');
  const a = Math.abs(v);
  if (a !== 0 && (a < 1e-3 || a >= 1e6)) return v.toExponential(2).replace('-', '−');
  const s = v.toFixed(digits).replace(/\.?0+$/, '');
  return s.replace('-', '−');
}

/** Signed fixed-precision value (z-scores, vector entries): "+0.866", "−1.200". */
export function formatSigned(v: number, digits = 3): string {
  if (!Number.isFinite(v)) return formatNumber(v);
  const s = Math.abs(v).toFixed(digits);
  return (v < 0 ? '−' : '+') + s;
}

/** A raw cell value for display. Datetimes are epoch ms (types.ts RawValue). */
export function formatRaw(value: RawValue | undefined, dtype?: DType): string {
  if (value === null || value === undefined) return NULL_GLYPH;
  if (typeof value === 'boolean') return value ? 'true' : 'false';
  if (typeof value === 'number') {
    if (Number.isNaN(value)) return NULL_GLYPH;
    if (dtype === 'datetime') return formatDate(value);
    return formatNumber(value);
  }
  return value;
}

/** Truncate a string to `n` characters with an ellipsis. */
export function truncate(s: string, n: number): string {
  return s.length <= n ? s : s.slice(0, Math.max(1, n - 1)) + '…';
}

/** Probability as a percentage string. */
export function formatPct(p: number, digits = 1): string {
  return Number.isFinite(p) ? `${(p * 100).toFixed(digits)}%` : '—';
}
