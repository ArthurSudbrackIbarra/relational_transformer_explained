/**
 * This site's miniature on ShopDB, next to the paper's headline numbers: zero-shot (leave-one-DB-out,
 * checkpoint chosen on ShopDB validation, test reported), the EntityMean baseline and the untrained init.
 * Numbers come from public/data/training/results.json at runtime.
 */
import { Link } from 'react-router-dom';
import { Cite, Figure, SCALE_ROWS } from '@/components';
import { TABLE1, TABLE2 } from './data';
import { f1, TitleWithBadge } from './kit';
import { useMiniResults } from './useMiniResults';

const pct = (v: number | undefined) => (v === undefined || !Number.isFinite(v) ? '—' : f1(v * 100));

export function MiniZeroShot() {
  const { data, error } = useMiniResults();
  const params = SCALE_ROWS.find((r) => r.what === 'Parameters');
  const tasks = [
    {
      id: 'shop/customer-churn',
      metric: 'AUROC',
      paperRt: TABLE1.mean[TABLE1.cols.indexOf('rtNo')] as number,
      paperEm: TABLE1.mean[TABLE1.cols.indexOf('entityMean')] as number,
    },
    {
      id: 'shop/product-sales',
      metric: 'R²',
      paperRt: TABLE2.mean[TABLE2.cols.indexOf('rtNo')] as number,
      paperEm: TABLE2.mean[TABLE2.cols.indexOf('entityMean')] as number,
    },
  ];
  return (
    <Figure
      wide
      title={<TitleWithBadge title="For scale: our miniature on ShopDB" kind="site" badge="results.json" />}
      caption={
        <>
          This site’s {params?.ours ?? ''}-parameter model, pretrained leave-one-database-out on three synthetic databases and evaluated zero-shot on
          ShopDB, with the checkpoint chosen on ShopDB’s validation split as in the paper <Cite kind="spec" ref="§9.1" />. One task each, test sets of
          a few dozen seeds: these numbers say the pipeline works, not how RT compares with anything. Chapter <Link to="/training">13</Link> shows how
          they were obtained.
        </>
      }
    >
      {error && <p className="text-[16px] text-ink-2">Could not load results.json ({error}).</p>}
      {!data && !error && <p className="text-[16px] text-ink-2">Loading this site’s results…</p>}
      {data && (
        <div className="grid gap-3 md:grid-cols-2">
          {tasks.map((t) => {
            const z = data.zeroShot?.[t.id];
            const n = data.labelPersistence?.tasks?.[t.id]?.splits?.Test?.rows;
            const rows = [
              { label: 'RT (mini), zero-shot', v: z?.pretrainBest?.test, strong: true },
              { label: 'EntityMean', v: z?.entityMean?.test ?? undefined },
              { label: 'RT (mini), untrained', v: z?.untrainedInit?.test },
            ];
            return (
              <div key={t.id} className="rounded-xl border border-border bg-surface px-4 py-3.5">
                <div className="font-mono text-[15.5px] text-ink">{t.id}</div>
                <div className="text-[15px] text-ink-2">
                  test {t.metric} (%){n ? `, ${n} seeds` : ''}
                </div>
                <table className="mt-2 w-full border-collapse text-[16px]">
                  <tbody>
                    {rows.map((r) => (
                      <tr key={r.label} className="border-b border-grid/70 last:border-0">
                        <td className={r.strong ? 'py-1 font-semibold text-ink' : 'py-1 text-ink-2'}>{r.label}</td>
                        <td className={`num py-1 text-right ${r.strong ? 'font-semibold text-ink' : 'text-ink'}`}>{pct(r.v)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <div className="mt-2 text-[15px] text-muted">
                  Paper, RelBench mean: RT (No) {f1(t.paperRt)}, EntityMean {f1(t.paperEm)} <Cite kind="paper" ref="p.8 Tables 1-2" />
                </div>
              </div>
            );
          })}
        </div>
      )}
    </Figure>
  );
}
