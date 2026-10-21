/**
 * Section 6 figure: float32 values stored by pre.rs vs the bfloat16 values a
 * batch carries (fly.rs:482-490, spec §3.7), for the live seed row and its
 * entity row.
 */
import { SemBadge } from '@/components';
import { SemType } from '@/engine/types';
import { catVar } from '@/styles/palette';
import { toBf16 } from './data';
import { signed, type ChapterData } from './shared';

export function PrecisionTable({ data }: { data: ChapterData }) {
  const nodes = [data.seedNode, data.entityNode].filter((n): n is number => n !== null && n >= 0);
  const rows = nodes.flatMap((n) => {
    const node = data.pdb.nodes[n];
    const f = data.fileOfNode(n);
    if (!node || !f) return [];
    return node.cells
      .filter((c) => c.sem !== SemType.Text)
      .map((c) => {
        const v = c.sem === SemType.Number ? c.number : c.sem === SemType.DateTime ? c.datetime : c.boolean;
        const b = toBf16(v);
        return { id: `${n}/${c.colName}`, f, n, c, v, b };
      });
  });
  if (rows.length === 0) return null;
  return (
    <div className="overflow-x-auto">
      <table className="w-full border-collapse text-[15px]">
        <thead>
          <tr className="border-b border-grid text-left text-ink-2">
            <th scope="col" className="py-1.5 pr-2 font-semibold">
              cell
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
          {rows.map(({ id, f, n, c, v, b }) => (
            <tr key={id} className="border-b border-grid/70">
              <td className="py-1.5 pr-2">
                <div className="flex items-center gap-1.5">
                  <span aria-hidden="true" className="inline-block h-2.5 w-2.5 shrink-0 rounded-[3px]" style={{ background: catVar(data.slotOf(f.tableName)) }} />
                  <span className="font-mono text-[14.5px] text-ink-2">
                    {f.isTask ? (
                      <>
                        {f.tableName}:<wbr />
                        <span className="whitespace-nowrap">
                          {f.tableType}[{n - f.offset}]
                        </span>
                      </>
                    ) : (
                      <span className="whitespace-nowrap">
                        {f.tableName}[{n - f.offset}]
                      </span>
                    )}
                  </span>
                </div>
                <div className="mt-0.5 flex items-center gap-1.5">
                  <SemBadge sem={c.sem} size="xs" />
                  <span className="font-mono whitespace-nowrap text-ink">{c.colName}</span>
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
  );
}
