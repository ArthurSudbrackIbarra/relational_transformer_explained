/**
 * Section 3 figure: physical dtype → semantic type (pre.rs:543-611,
 * common.rs:38-44; spec §3.2), with the live DB's feature columns sorted into
 * the four types.
 */
import { SemBadge, cx } from '@/components';
import { SemType, type DType } from '@/engine/types';
import { catVar } from '@/styles/palette';
import { colRole, type TableFile } from './data';
import type { ChapterData } from './shared';

interface Row {
  code: SemType;
  name: string;
  dtypes: string;
  ours: DType[];
  stored: string;
  arm: string;
}

const ROWS: Row[] = [
  { code: SemType.Number, name: 'Number', dtypes: 'UInt32, Int32, Int64, Float32, Float64 (cast to f64)', ours: ['int', 'float'], stored: 'number_values = (v − μ) / σ', arm: 'pre.rs:565-583' },
  { code: SemType.Text, name: 'Text', dtypes: 'String', ours: ['str'], stored: 'text_values = index of the string', arm: 'pre.rs:595-605' },
  { code: SemType.DateTime, name: 'DateTime', dtypes: 'Datetime (nanoseconds)', ours: ['datetime'], stored: 'datetime_values = (t − μ_T) / σ_T', arm: 'pre.rs:584-594' },
  { code: SemType.Boolean, name: 'Boolean', dtypes: 'Boolean', ours: ['bool'], stored: 'boolean_values = (b − μ) / σ', arm: 'pre.rs:553-564' },
];

interface ColRef {
  file: TableFile;
  col: string;
}

export function SemTypeFigure({ data }: { data: ChapterData }) {
  const { files, slotOf } = data;
  // one entry per logical column (task splits share a table name and column phrase, spec §2.2)
  const seen = new Set<string>();
  const byType = new Map<SemType, ColRef[]>();
  for (const f of files) {
    for (const c of f.table.schema.columns) {
      if (colRole(f.table, c.name) !== 'feature') continue;
      const id = `${f.tableName}.${c.name}`;
      if (seen.has(id)) continue;
      seen.add(id);
      const row = ROWS.find((r) => r.ours.includes(c.dtype));
      if (!row) continue;
      if (!byType.has(row.code)) byType.set(row.code, []);
      byType.get(row.code)!.push({ file: f, col: c.name });
    }
  }

  return (
    <div className="min-w-0">
      <div className="hidden grid-cols-[150px_minmax(0,1fr)_minmax(0,1.1fr)_minmax(0,1.4fr)] gap-3 border-b border-grid pb-2 text-[14.5px] font-semibold text-ink-2 md:grid">
        <div>sem type (code)</div>
        <div>physical dtype</div>
        <div>what the cell stores</div>
        <div>columns of {data.db.title.split(' (')[0]}</div>
      </div>
      <ul className="!m-0 !list-none !p-0">
        {ROWS.map((r) => (
          <li key={r.code} className="!m-0 grid gap-x-3 gap-y-1.5 border-b border-grid/70 py-3 md:grid-cols-[150px_minmax(0,1fr)_minmax(0,1.1fr)_minmax(0,1.4fr)] md:items-start">
            <div className="flex items-center gap-2">
              <SemBadge sem={r.code} size="md" />
              <div>
                <div className="text-[16.5px] font-semibold text-ink">{r.name}</div>
                <div className="num text-[14.5px] text-ink-2">code {r.code}</div>
              </div>
            </div>
            <div className="text-[15.5px] text-ink">
              <span className="text-[14px] text-muted md:hidden">dtype </span>
              <span className="font-mono text-[15px]">{r.dtypes}</span>
              <div className="font-mono text-[14px] text-muted">{r.arm}</div>
            </div>
            <div className="font-mono text-[15px] text-ink">{r.stored}</div>
            <div className="flex flex-wrap gap-1.5">
              {(byType.get(r.code) ?? []).map(({ file, col }) => (
                <span
                  key={`${file.tableName}.${col}`}
                  className={cx('inline-flex items-center gap-1.5 rounded-md border border-border bg-surface-2 px-1.5 py-0.5 font-mono text-[14.5px] text-ink')}
                  title={file.isTask ? `${col} of the task table ${file.tableName} (all splits)` : `${col} of ${file.tableName}`}
                >
                  <span aria-hidden="true" className="inline-block h-2 w-2 shrink-0 rounded-[2px]" style={{ background: catVar(slotOf(file.tableName)) }} />
                  <span>
                    <span className="text-ink-2">{file.tableName}.</span>
                    {col}
                  </span>
                </span>
              ))}
              {(byType.get(r.code) ?? []).length === 0 && <span className="text-[15px] text-muted">none in this database</span>}
            </div>
          </li>
        ))}
        <li className="!m-0 grid gap-x-3 gap-y-1 py-3 text-ink-2 md:grid-cols-[150px_minmax(0,1fr)_minmax(0,1.1fr)_minmax(0,1.4fr)]">
          <div className="text-[15.5px] font-semibold">any other dtype</div>
          <div className="font-mono text-[15px]">
            e.g. Int16, UInt64, Date, List
            <div className="font-mono text-[14px] text-muted">pre.rs:606-611</div>
          </div>
          <div className="font-mono text-[15px]">panic!()</div>
          <div className="text-[15px]">pre.rs stops on the first non-null value of such a column</div>
        </li>
      </ul>
    </div>
  );
}
