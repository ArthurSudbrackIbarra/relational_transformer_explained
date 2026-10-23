/**
 * Section 4d figure: task labels of every split are normalized with the
 * task's TRAIN statistics (pre.rs:383-396, spec §3.3). For each forecast task,
 * the split's own label distribution is shown next to the μ and σ actually
 * used, and the value the engine stored for the same label in each split.
 */
import { ArrowDownLeft } from 'lucide-react';
import { formatNumber } from '@/components/viz/format';
import type { Split } from '@/engine/types';
import { catVar } from '@/styles/palette';
import { isMissing, isoDay, type TableFile } from './data';
import { fixed, signed, type ChapterData } from './shared';

const nf = (n: number) => n.toLocaleString('en-US');
const SPLITS: Split[] = ['Train', 'Val', 'Test'];

/** The value the engine stored for the first row of `f` whose label equals `label` (or null). */
function storedFor(data: ChapterData, f: TableFile, col: string, label: number | boolean): number | null {
  const i = f.table.rows.findIndex((r) => r[col] === label);
  if (i < 0) return null;
  const cell = data.pdb.nodes[f.offset + i]?.cells.find((c) => c.colName === col);
  if (!cell) return null;
  return cell.sem === 3 ? cell.boolean : cell.number;
}

export function TrainStatsFigure({ data }: { data: ChapterData }) {
  const tasks = data.db.tasks.filter((t) => t.kind === 'forecast' && t.tables);
  if (tasks.length === 0) return <p className="text-[16.5px] text-ink-2">This database has no forecasting task tables.</p>;
  return (
    <div className="grid gap-5">
      {tasks.map((task) => {
        const col = task.targetCol;
        const isBool = task.targetType === 'binary';
        const files = SPLITS.map((s) => data.fileByKey.get(`${task.name}:${s}`)).filter((f): f is TableFile => !!f);
        const train = files.find((f) => f.tableType === 'Train');
        const trainStats = train ? data.pdb.colStats[train.key]?.[col] : undefined;
        // the label value to follow across splits: true, or the most common Train value
        let follow: number | boolean = true;
        if (!isBool && train) {
          const counts = new Map<number, number>();
          for (const r of train.table.rows) if (!isMissing(r[col])) counts.set(r[col] as number, (counts.get(r[col] as number) ?? 0) + 1);
          follow = [...counts.entries()].sort((a, b) => b[1] - a[1])[0]?.[0] ?? 0;
        }
        return (
          <div key={task.id} className="min-w-0">
            <div className="mb-2 flex flex-wrap items-baseline gap-x-2 text-[16px]">
              <span className="inline-flex items-center gap-1.5 font-semibold text-ink">
                <span aria-hidden="true" className="inline-block h-2.5 w-2.5 rounded-[3px]" style={{ background: catVar(data.slotOf(task.name)) }} />
                {task.name}.{col}
              </span>
              <span className="text-ink-2">
                {isBool ? 'boolean' : 'number'} label · following the label <span className="num text-ink">{String(follow)}</span> across splits
              </span>
            </div>
            <table className="w-full border-collapse text-[15px]">
              <thead>
                <tr className="border-b border-grid text-left text-ink-2">
                  <th scope="col" className="py-1.5 pr-2 font-semibold">
                    split
                  </th>
                  <th scope="col" className="hidden py-1.5 pr-2 text-right font-semibold sm:table-cell">
                    rows
                  </th>
                  <th scope="col" className="py-1.5 pr-2 text-right font-semibold">
                    {isBool ? 'true in split' : 'mean in split'}
                  </th>
                  <th scope="col" className="py-1.5 pr-2 text-right font-semibold">
                    μ used
                  </th>
                  <th scope="col" className="hidden py-1.5 pr-2 text-right font-semibold sm:table-cell">
                    σ used
                  </th>
                  <th scope="col" className="py-1.5 text-right font-semibold">
                    label {String(follow)} → stored
                  </th>
                </tr>
              </thead>
              <tbody>
                {files.map((f) => {
                  const vals = f.table.rows.map((r) => r[col]).filter((v) => !isMissing(v));
                  const own = vals.length ? vals.reduce((s: number, v) => s + (isBool ? (v ? 1 : 0) : (v as number)), 0) / vals.length : NaN;
                  const used = data.pdb.colStats[f.key]?.[col];
                  const stored = storedFor(data, f, col, follow);
                  const inherited = f.tableType !== 'Train';
                  const ts = f.table.schema.timeCol ? (f.table.rows[0]?.[f.table.schema.timeCol] as number | undefined) : undefined;
                  return (
                    <tr key={f.key} className="border-b border-grid/70">
                      <td className="py-1.5 pr-2">
                        <span className="text-ink">{f.tableType}</span>
                        {f.tableType !== 'Train' && ts !== undefined && <span className="block text-[14px] whitespace-nowrap text-muted sm:ml-1.5 sm:inline">{isoDay(ts)}</span>}
                        <span className="num block text-[14px] whitespace-nowrap text-muted sm:hidden">{nf(f.numNodes)} rows</span>
                      </td>
                      <td className="num hidden py-1.5 pr-2 text-right sm:table-cell">{nf(f.numNodes)}</td>
                      <td className="num py-1.5 pr-2 text-right">{isBool ? `${(own * 100).toFixed(1)}%` : fixed(own, 3)}</td>
                      <td className="num py-1.5 pr-2 text-right whitespace-nowrap">
                        {inherited && <ArrowDownLeft aria-label="copied from Train" size={14} className="mr-1 inline text-accent-text" />}
                        {used ? fixed(used.mean, 4) : '—'}
                      </td>
                      <td className="num hidden py-1.5 pr-2 text-right whitespace-nowrap sm:table-cell">
                        {inherited && <ArrowDownLeft aria-label="copied from Train" size={14} className="mr-1 inline text-accent-text" />}
                        {used ? fixed(used.std, 4) : '—'}
                      </td>
                      <td className="num py-1.5 text-right font-semibold text-ink">{stored === null ? '—' : signed(stored, 4)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            {trainStats && (
              <div className="mt-1.5 text-[14.5px] text-muted">
                Train stats: μ = {fixed(trainStats.mean, 4)}, σ = {fixed(trainStats.std, 4)} over {nf(trainStats.count)} labels (sample std). The arrows mark values copied from Train.
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

/** Rows of time-stamped DB tables dated after the Val timestamp, which still feed μ and σ (spec §3.4). */
export function LeakCounts({ data }: { data: ChapterData }) {
  const task = data.task.kind === 'forecast' ? data.task : data.db.tasks.find((t) => t.kind === 'forecast');
  const cut = task?.valTimestamp;
  if (cut === undefined) return null;
  const rows = data.files
    .filter((f) => f.tableType === 'Db' && f.table.schema.timeCol)
    .map((f) => {
      const tc = f.table.schema.timeCol!;
      const after = f.table.rows.filter((r) => !isMissing(r[tc]) && (r[tc] as number) > cut).length;
      const stats = Object.keys(data.pdb.colStats[f.key] ?? {});
      return { f, after, stats };
    })
    .filter((r) => r.stats.length > 0);
  if (rows.length === 0) return null;
  return (
    <ul className="!my-2 !pl-5">
      {rows.map(({ f, after, stats }) => (
        <li key={f.key}>
          <span className="num">{nf(after)}</span> of {nf(f.numNodes)} <code>{f.tableName}</code> rows are dated after the Val timestamp ({isoDay(cut)}), and they still feed μ and σ of{' '}
          {stats.map((s, i) => (
            <span key={s}>
              {i > 0 && (i === stats.length - 1 ? ' and ' : ', ')}
              <code>{s}</code>
            </span>
          ))}
          .
        </li>
      ))}
    </ul>
  );
}
