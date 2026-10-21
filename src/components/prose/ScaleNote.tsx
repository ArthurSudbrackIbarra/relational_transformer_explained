/**
 * ScaleNote (BUILD_PLAN §0.4): wherever numbers appear, say what is scaled
 * down. Every OPERATION is identical to the paper-era code; only sizes and the
 * training data change, plus float32 instead of bf16.
 *
 *   <ScaleNote />                                  full comparison table
 *   <ScaleNote paper="L = 1024" ours="L = 128" />  one row, with an optional explanation as children
 *   <ScaleNote inline />                           compact chip with the table in a hover card
 */
import { Scale } from 'lucide-react';
import type { ReactNode } from 'react';
import { HoverCard } from '../ui/floating';
import { cx } from '../ui/cx';

/** Paper vs mini settings (BUILD_PLAN §2 MINI_CONFIG; spec §7.3, §9.2). */
export const SCALE_ROWS: { what: string; paper: string; ours: string }[] = [
  { what: 'Hidden size d', paper: '256', ours: '64' },
  { what: 'Attention heads (d_head)', paper: '8 (32)', ours: '4 (16)' },
  { what: 'Blocks', paper: '12', ours: '4' },
  { what: 'Feed-forward width', paper: '1024', ours: '256' },
  { what: 'Context length L', paper: '1024', ours: '128' },
  { what: 'BFS width w', paper: '256', ours: '16' },
  { what: 'Parameters', paper: '22,336,131', ours: '535,491' },
  { what: 'Text encoder (frozen)', paper: 'all-MiniLM-L12-v2, 384-d', ours: 'same' },
  { what: 'Precision', paper: 'bfloat16, RMSNorm ε = 2⁻⁷', ours: 'float32, ε = 2⁻⁷' },
  { what: 'Data', paper: 'RelBench, 7 databases', ours: '4 synthetic databases' },
];

export interface ScaleNoteProps {
  /** Single-row mode: the paper's setting. */
  paper?: ReactNode;
  /** Single-row mode: this site's setting. */
  ours?: ReactNode;
  /** Optional explanation. */
  children?: ReactNode;
  /** Compact inline chip. */
  inline?: boolean;
  className?: string;
}

function ScaleTable() {
  return (
    <table className="w-full border-collapse text-[15px]">
      <thead>
        <tr className="border-b border-grid text-left text-ink-2">
          <th scope="col" className="py-1 pr-2 font-medium" />
          <th scope="col" className="py-1 pr-2 font-semibold">
            Paper
          </th>
          <th scope="col" className="py-1 font-semibold">
            This site
          </th>
        </tr>
      </thead>
      <tbody>
        {SCALE_ROWS.map((r) => (
          <tr key={r.what} className="border-b border-grid/60 last:border-0">
            <th scope="row" className="py-1 pr-2 text-left font-normal text-ink-2">
              {r.what}
            </th>
            <td className="num py-1 pr-2 text-ink">{r.paper}</td>
            <td className="num py-1 text-ink">{r.ours}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function ScaleNote({ paper, ours, children, inline = false, className }: ScaleNoteProps) {
  if (inline) {
    return (
      <HoverCard
        width={380}
        label="Paper vs this site"
        trigger={(p) => (
          <span
            {...p}
            ref={p.ref}
            role="button"
            tabIndex={0}
            className={cx(
              'inline-flex cursor-help items-center gap-1 rounded-full border border-border bg-surface-2 px-2 py-px align-middle text-[15px] font-medium text-ink-2 hover:text-ink',
              className,
            )}
          >
            <Scale aria-hidden="true" size={12} />
            scaled down
          </span>
        )}
      >
        <div className="mb-2 text-[15px] text-ink-2">Every operation matches the paper-era code; only these sizes change.</div>
        <ScaleTable />
      </HoverCard>
    );
  }

  const single = paper !== undefined || ours !== undefined;
  return (
    <aside className={cx('my-5 rounded-xl border border-dashed border-border-strong px-4 py-3 text-[17px]', className)} aria-label="Scale note">
      <div className="flex items-start gap-2.5">
        <Scale aria-hidden="true" size={16} className="mt-[2px] shrink-0 text-muted" />
        <div className="min-w-0 flex-1">
          <div className="text-[16.5px] font-semibold text-ink">Scaled down, not simplified</div>
          {single ? (
            <div className="mt-1 flex flex-wrap gap-x-5 gap-y-1 text-ink-2">
              {paper !== undefined && (
                <span>
                  Paper: <span className="num text-ink">{paper}</span>
                </span>
              )}
              {ours !== undefined && (
                <span>
                  This site: <span className="num text-ink">{ours}</span>
                </span>
              )}
            </div>
          ) : (
            <div className="mt-2 max-w-[520px]">
              <ScaleTable />
            </div>
          )}
          <div className="mt-1.5 text-[16px] leading-snug text-ink-2">
            {children ?? 'Every operation is the paper-era code’s; only the sizes, the data and the floating-point precision differ.'}
          </div>
        </div>
      </div>
    </aside>
  );
}
