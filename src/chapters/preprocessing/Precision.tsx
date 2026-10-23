/**
 * Section 6 figure: float32 values stored by pre.rs vs the bfloat16 values a
 * batch carries (fly.rs:482-490, spec §3.7). The rows are the live context
 * window's own tokens of the seed row (and, for a forecast task, its entity
 * row), so the table lists exactly the cells fly.rs emits for them: a task's
 * leakage columns are absent for the seed row and for rows with the seed's
 * timestamp (fly.rs:455-460, spec §4.9).
 */
import { Cite, SemBadge } from '@/components';
import { SemType, type Token } from '@/engine/types';
import { useContextWindow } from '@/state/hooks';
import { catVar } from '@/styles/palette';
import { toBf16 } from './data';
import { signed, type ChapterData } from './shared';

const stored = (t: Token) => (t.sem === SemType.Number ? t.numberValue : t.sem === SemType.DateTime ? t.datetimeValue : t.booleanValue);

export function PrecisionTable({ data }: { data: ChapterData }) {
  const ctx = useContextWindow();
  const { task, seedNode, entityNode, pdb } = data;
  const nodes = [seedNode, entityNode].filter((n): n is number => n !== null && n >= 0);
  const rows = ctx.tokens
    .filter((t) => !t.isPadding && nodes.includes(t.nodeIdx) && t.sem !== SemType.Text)
    .sort((a, b) => nodes.indexOf(a.nodeIdx) - nodes.indexOf(b.nodeIdx) || a.pos - b.pos)
    .map((t) => {
      const v = stored(t);
      return { t, f: data.fileOfNode(t.nodeIdx), v, b: toBf16(v) };
    });
  // leakage cells of the seed row that the sampler left out (fly.rs:455-460)
  const seed = seedNode >= 0 ? pdb.nodes[seedNode] : undefined;
  const dropped = seed ? seed.cells.filter((c) => task.columnsToDrop.includes(c.colName)).map((c) => c.colName) : [];
  if (rows.length === 0) return null;
  return (
    <div className="min-w-0">
      <div className="overflow-x-auto">
        <table className="w-full border-collapse text-[15px]">
          <thead>
            <tr className="border-b border-grid text-left text-ink-2">
              <th scope="col" className="py-1.5 pr-2 font-semibold">
                token
              </th>
              <th scope="col" className="py-1.5 pr-2 text-right align-bottom font-semibold">
                <span className="hidden sm:inline">stored by pre.rs (f32)</span>
                <span className="sm:hidden">
                  f32 (pre.rs)
                  <br />→ bf16 (batch)
                </span>
              </th>
              <th scope="col" className="hidden py-1.5 pr-2 text-right font-semibold sm:table-cell">
                in the batch (bf16)
              </th>
              <th scope="col" className="hidden py-1.5 text-right font-semibold sm:table-cell">
                |difference|
              </th>
            </tr>
          </thead>
          <tbody>
            {rows.map(({ t, f, v, b }) => (
              <tr key={t.pos} className="border-b border-grid/70">
                <td className="py-1.5 pr-2">
                  <div className="flex items-center gap-1.5">
                    <span aria-hidden="true" className="inline-block h-2.5 w-2.5 shrink-0 rounded-[3px]" style={{ background: catVar(data.slotOf(t.tableName)) }} />
                    <span className="text-[14.5px] text-ink-2">
                      {f?.isTask ? (
                        <>
                          {t.tableName}:<wbr />
                          <span className="whitespace-nowrap">
                            {f.tableType}[{t.rowPos}]
                          </span>
                        </>
                      ) : (
                        <span className="whitespace-nowrap">
                          {t.tableName}[{t.rowPos}]
                        </span>
                      )}
                    </span>
                    <span className="num text-[14px] whitespace-nowrap text-muted">pos {t.pos}</span>
                  </div>
                  <div className="mt-0.5 flex flex-wrap items-center gap-1.5">
                    <SemBadge sem={t.sem} size="sm" />
                    <span className="whitespace-nowrap text-ink">{t.colName}</span>
                    {t.isTarget && <span className="rounded bg-accent-soft px-1.5 text-[14px] font-medium text-accent-text">masked target</span>}
                  </div>
                </td>
                <td className="num py-1.5 pr-2 text-right align-top whitespace-nowrap">
                  {signed(v, 7)}
                  <div className="mt-0.5 font-semibold text-ink sm:hidden">→ {signed(b, 7)}</div>
                </td>
                <td className="num hidden py-1.5 pr-2 text-right align-top font-semibold whitespace-nowrap text-ink sm:table-cell">{signed(b, 7)}</td>
                <td className="num hidden py-1.5 text-right align-top text-ink-2 sm:table-cell">{Math.abs(b - v).toExponential(2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="!mt-2.5 !mb-0 text-[15px] leading-snug text-ink-2">
        These are the non-text tokens of the live context window (seed #{ctx.params.seedNode}), read from the sampler’s output, so the table lists exactly the cells a batch
        carries for {entityNode !== null ? 'these two rows' : 'this row'}. A masked target keeps its stored value in the batch; the model replaces only its value term.
        {dropped.length > 0 && (
          <>
            {' '}
            Not listed: the seed’s{' '}
            {dropped.map((c, i) => (
              <span key={c}>
                {i > 0 && (i === dropped.length - 1 ? ' and ' : ', ')}
                <code>{c}</code>
              </span>
            ))}
            , which the task <code>{task.kind === 'autocomplete' ? `${task.table}.${task.targetCol}` : task.name}</code> drops as leakage columns for the seed row and for
            rows with its timestamp <Cite kind="code" ref="rustler/src/fly.rs:457-460" />{' '}
            <Cite kind="spec" ref="§4.9" />.
          </>
        )}
      </p>
    </div>
  );
}
