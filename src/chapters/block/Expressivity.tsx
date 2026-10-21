/**
 * Section 7: expressivity notes from the paper's App. E (p.19-20), quoted, plus the paper's own ablation
 * numbers: Table 5 (p.10, mean R² of 8 regression tasks) and Table 15 (p.24, attention order). Every
 * number is copied from the PDF and checked against spec §11.4-11.5.
 */
import type { ReactNode } from 'react';
import { Cite, LayerLabel } from '@/components';
import type { AttnLayer } from '@/engine/types';

const APP_E: { layer: AttnLayer; quote: string; page: string; ours: ReactNode }[] = [
  {
    layer: 'col',
    quote:
      'Removing column attention does not reduce expressivity, since global attention can emulate it. In particular, some global heads can learn to restrict attention to tokens from the same column by exploiting table and column name embeddings in their query–key construction.',
    page: 'p.19 App. E',
    ours: (
      <>
        Not needed for expressivity, yet it is the layer with “the highest impact on zero-shot transfer” in the paper’s ablation{' '}
        <Cite kind="paper" ref="p.9 §5.2" /> (Table 5 below).
      </>
    ),
  },
  {
    layer: 'feat',
    quote: 'Feature attention is the only mechanism that explicitly groups cells into rows.',
    page: 'p.19 App. E',
    ours: (
      <>
        The paper adds that neighbor attention gives only partial row information (cells of one row see the same neighbors) and cannot
        uniquely tell rows apart, especially in tables without incoming foreign keys <Cite kind="paper" ref="p.19 App. E" />.
      </>
    ),
  },
  {
    layer: 'nbr',
    quote:
      'Similar to column attention, neighbor attention is not strictly required for expressivity, as feature and global attention together can simulate its effect. Feature attention exposes F→P links, which global attention can then leverage to infer P→F relationships.',
    page: 'p.20 App. E',
    ours: (
      <>
        In the masks, neighbor is exactly the transpose of the parent part of feature <Cite kind="spec" ref="§6.2" />: every neighbor edge
        in the tracer above has a feature edge running the other way.
      </>
    ),
  },
  {
    layer: 'full',
    quote:
      'Without full attention, information can only propagate one hop per layer, limiting expressivity to local message passing. By contrast, full attention enables long-range interactions in a fixed number of layers, independent of database or graph diameter.',
    page: 'p.20 App. E',
    ours: 'Switch full attention off in the tracer to see the one-hop-per-layer limit on the live context.',
  },
];

export function AppECards() {
  return (
    <div className="wide my-7 grid gap-3 md:grid-cols-2">
      {APP_E.map((c) => (
        <div key={c.layer} className="card min-w-0 px-4 py-3.5">
          <div className="text-[17px]">
            <LayerLabel layer={c.layer} size={16} /> <span className="text-ink-2">attention</span>
          </div>
          <blockquote className="!my-2 border-l-2 border-axis pl-3 text-[16.5px] leading-snug text-ink">
            “{c.quote}” <Cite kind="paper" ref={c.page} />
          </blockquote>
          <p className="!mb-0 text-[15.5px] leading-snug text-ink-2">{c.ours}</p>
        </div>
      ))}
    </div>
  );
}

/** Paper Table 5 (p.10), "Mean R² →" row. */
const TABLE5 = {
  zero: { none: 22.8, col: 11.9, feat: 18.8, nbr: 22.0, full: 23.7 },
  ft: { none: 33.2, col: 31.2, feat: 32.2, nbr: 31.4, full: 32.7 },
};

export function Table5() {
  const cols = ['none', 'col', 'feat', 'nbr', 'full'] as const;
  const head: Record<(typeof cols)[number], string> = { none: 'none', col: 'column', feat: 'feature', nbr: 'neighbor', full: 'full' };
  return (
    <div className="my-6">
      <div className="mb-2 text-[16px] font-semibold text-ink">
        Mean R² (%) over 8 RelBench regression tasks, with one attention layer removed <Cite kind="paper" ref="p.10 Table 5" />
      </div>
      <div className="card scroll-quiet overflow-x-auto">
        <table className="w-full min-w-[480px] border-collapse text-[15px]">
          <caption className="sr-only">Paper Table 5, mean R² row</caption>
          <thead>
            <tr className="border-b border-grid text-ink-2">
              <th scope="col" className="px-3 py-1.5 text-left font-semibold">
                removed →
              </th>
              {cols.map((c) => (
                <th key={c} scope="col" className="px-3 py-1.5 text-right font-semibold">
                  {head[c]}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {(['zero', 'ft'] as const).map((r) => (
              <tr key={r} className="border-b border-grid/70 last:border-0">
                <th scope="row" className="px-3 py-1.5 text-left font-normal text-ink">
                  {r === 'zero' ? 'zero-shot' : 'fine-tuned'}
                </th>
                {cols.map((c) => {
                  const v = TABLE5[r][c];
                  const min = Math.min(...cols.map((k) => TABLE5[r][k]));
                  return (
                    <td key={c} className={`num px-3 py-1.5 text-right ${v === min ? 'font-semibold text-ink' : 'text-ink-2'}`}>
                      {v.toFixed(1)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="!mb-0 mt-2 text-[15px] leading-snug text-ink-2">
        Bold: the largest drop in each row. The paper keeps the parameter count constant by adding blocks when a layer is removed{' '}
        <Cite kind="paper" ref="p.10 Table 5" />; the classification counterpart (Table 13) shows only minor differences{' '}
        <Cite kind="paper" ref="p.23 Table 13" />.
      </p>
    </div>
  );
}

/** Paper Table 15 (p.24), mean rows. */
const TABLE15 = {
  orders: ['CFN', 'CNF', 'FCN', 'FNC', 'NCF', 'NFC', 'Parallel'],
  auroc: [70.0, 70.6, 70.0, 69.8, 70.2, 69.5, 69.3],
  r2: [23.0, 22.5, 23.2, 22.5, 22.1, 23.0, 23.6],
};

export function Table15() {
  return (
    <div className="my-6">
      <div className="mb-2 text-[16px] font-semibold text-ink">
        Zero-shot results for different orders of the three relational layers <Cite kind="paper" ref="p.24 Table 15" />
      </div>
      <div className="card scroll-quiet overflow-x-auto">
        <table className="w-full min-w-[560px] border-collapse text-[15px]">
          <caption className="sr-only">Paper Table 15, mean rows</caption>
          <thead>
            <tr className="border-b border-grid text-ink-2">
              <th scope="col" className="px-3 py-1.5 text-left font-semibold">
                order →
              </th>
              {TABLE15.orders.map((o) => (
                <th key={o} scope="col" className="num px-2.5 py-1.5 text-right font-semibold">
                  {o}
                  {o === 'CFN' && <span className="block font-sans text-[13.5px] font-normal text-muted">released</span>}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            <tr className="border-b border-grid/70">
              <th scope="row" className="px-3 py-1.5 text-left font-normal text-ink">
                mean AUROC (10 tasks)
              </th>
              {TABLE15.auroc.map((v, i) => (
                <td key={i} className="num px-2.5 py-1.5 text-right text-ink-2">
                  {v.toFixed(1)}
                </td>
              ))}
            </tr>
            <tr>
              <th scope="row" className="px-3 py-1.5 text-left font-normal text-ink">
                mean R² (8 tasks)
              </th>
              {TABLE15.r2.map((v, i) => (
                <td key={i} className="num px-2.5 py-1.5 text-right text-ink-2">
                  {v.toFixed(1)}
                </td>
              ))}
            </tr>
          </tbody>
        </table>
      </div>
      <p className="!mb-0 mt-2 text-[15px] leading-snug text-ink-2">
        C, F, N = column, feature, neighbor; the letters give the order inside each block. “Parallel” applies the three layers in parallel and
        sums their outputs <Cite kind="paper" ref="p.24 Table 15" />.
      </p>
    </div>
  );
}
