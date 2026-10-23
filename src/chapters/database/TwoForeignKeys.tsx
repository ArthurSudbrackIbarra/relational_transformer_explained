/**
 * The paper's limitation example (p.10 §7): a product table with buyer_id and
 * seller_id, both foreign keys to user. After preprocessing no token carries a
 * foreign-key column name (key columns make no cells); what remains is the
 * parent list (pre.rs:479) and the edges (pre.rs:515-529). The list keeps
 * FK-column order, but the attention masks only test membership
 * (model.py:171-178), so buyer and seller look alike (spec §1.2, §4.13).
 * Hypothetical schema, drawn in neutral colors on purpose.
 */
import { ArrowRight, Link2, KeyRound } from 'lucide-react';
import { Cite } from '@/components';

export function TwoForeignKeys() {
  return (
    <div className="grid items-center gap-3 sm:grid-cols-[minmax(0,1fr)_auto_minmax(0,1fr)]">
      <div className="rounded-xl border border-dashed border-border-strong px-3 py-2.5">
        <div className="mb-1.5 flex items-baseline justify-between gap-2">
          <span className="font-mono text-[15.5px] font-semibold text-ink">product</span>
          <span className="text-[14px] text-ink-2">hypothetical, paper p.10</span>
        </div>
        <table className="w-full text-[15px]">
          <tbody>
            <tr>
              <td className="py-0.5 pr-2 text-ink-2">
                <span className="inline-flex items-center gap-1.5 font-mono">
                  <KeyRound aria-label="primary key" size={12} /> product_id
                </span>
              </td>
              <td className="num text-ink-2">3</td>
            </tr>
            <tr>
              <td className="py-0.5 pr-2 text-ink-2">
                <span className="inline-flex items-center gap-1.5 font-mono">
                  <Link2 aria-label="foreign key" size={12} /> buyer_id → user
                </span>
              </td>
              <td className="num text-ink">7</td>
            </tr>
            <tr>
              <td className="py-0.5 pr-2 text-ink-2">
                <span className="inline-flex items-center gap-1.5 font-mono">
                  <Link2 aria-label="foreign key" size={12} /> seller_id → user
                </span>
              </td>
              <td className="num text-ink">12</td>
            </tr>
          </tbody>
        </table>
      </div>
      <ArrowRight aria-hidden="true" size={18} className="mx-auto rotate-90 text-muted sm:rotate-0" />
      <div className="rounded-xl bg-surface-2 px-3 py-2.5">
        <div className="mb-1.5 font-mono text-[14.5px] text-ink-2">what RT keeps: f2p_nbr_idxs</div>
        <div className="flex flex-wrap items-center gap-1.5 font-mono text-[15.5px]">
          <span className="text-muted">[</span>
          <span className="rounded-md border border-border bg-surface px-2 py-0.5 text-ink">user[7]</span>
          <span className="text-muted">,</span>
          <span className="rounded-md border border-border bg-surface px-2 py-0.5 text-ink">user[12]</span>
          <span className="text-muted">]</span>
        </div>
        <div className="mt-1.5 text-[15px] leading-snug text-ink-2">
          The list holds the two parents’ node indices, offset(user) + 7 and offset(user) + 12 (shown here as user[7] and user[12]), in
          foreign-key-column order, and no column name. The attention masks only ask whether a row is in the list{' '}
          <Cite kind="code" ref="rt/model.py:171-178" />, so the model cannot tell the buyer from the seller.
        </div>
      </div>
    </div>
  );
}
