/**
 * Provenance of the published numbers (spec §11 intro, §13 #1): a timeline of the arXiv versions and
 * the code commits that changed the sampler and the attention order, with each table pinned to the
 * version it last changed in, plus three arithmetic checks computed from the printed tables.
 */
import { Check, X } from 'lucide-react';
import { useState } from 'react';
import { Cite, cx } from '@/components';
import {
  CLF_ORDER,
  FIG3_POINTS,
  TABLE1,
  TABLE10,
  TABLE11,
  TABLE14,
  TABLE15,
  TABLE2,
  TABLE5_13,
  taskLabel,
  TABLE1_V1_MEANS,
  V1_TABLE1_RTNO,
  type TaskKey,
} from './data';
import { f1, mean } from './kit';

interface Milestone {
  date: string;
  kind: 'paper' | 'code';
  title: string;
  detail: string;
  tables?: { id: string; note: string }[];
}

const MILESTONES: Milestone[] = [
  {
    date: '2025-10-06',
    kind: 'code',
    title: 'eeccc2b · ICLR-submission code',
    detail: 'Attention order feat → nbr → col → full; the sampler excludes task rows with the seed’s own timestamp.',
  },
  {
    date: '2025-10-07',
    kind: 'paper',
    title: 'arXiv v1',
    detail: 'First results. These tables are unchanged since; each chip gives its number in v1.',
    tables: [
      { id: 'Table 4', note: 'v1 Table 3: context ablations (means)' },
      { id: 'Table 5', note: 'v1 Table 4: attention ablations, R²' },
      { id: 'Table 10', note: 'v1 Table 7: supervised fine-tuning' },
      { id: 'Table 11', note: 'v1 Table 8: context ablations per task' },
      { id: 'Table 12', note: 'v1 Table 9: in-context label counts' },
      { id: 'Table 13', note: 'v1 Table 10: attention ablations, AUROC' },
      { id: 'Table 14', note: 'v1 Table 11: multi-cell masking' },
    ],
  },
  {
    date: '2025-10-08',
    kind: 'code',
    title: '7f891d1 · “fix in attention order and sampling”',
    detail: 'The order becomes col → feat → nbr → full.',
  },
  {
    date: '2025-10-09',
    kind: 'code',
    title: '66c4247 · sampler change',
    detail: 'Table 12’s label counts were sampled before this commit, when same-timestamp task rows were still excluded.',
  },
  {
    date: '2025-10-16',
    kind: 'code',
    title: '801a32b · “fixed bfs width sampling”',
    detail: 'The sampler code is byte-identical to the released one from here on.',
  },
  {
    date: '2025-10-22',
    kind: 'paper',
    title: 'arXiv v2',
    detail: `Table 1 re-run: RT (No) / RT (Yes) means ${TABLE1_V1_MEANS.rtNo} / ${TABLE1_V1_MEANS.rtYes} → ${f1(TABLE1.mean[5] as number)} / ${f1(TABLE1.mean[8] as number)}.`,
    tables: [{ id: 'Table 1', note: 're-run after the fixes' }],
  },
  {
    date: '2026-02-04',
    kind: 'code',
    title: '2a899c8 · paper-era code',
    detail: 'The code this whole site mirrors (matches the camera-ready paper).',
  },
  {
    date: '2026-03-02',
    kind: 'paper',
    title: 'arXiv v3 (camera-ready, ICLR 2026)',
    detail: 'Table 2 re-run; Tables 3 and 15 are new.',
    tables: [
      { id: 'Table 2', note: 're-run: RT (No) mean 22.8 → 22.7' },
      { id: 'Table 3', note: 'new: RT as a feature extractor' },
      { id: 'Table 15', note: 'new: attention order' },
    ],
  },
];

export function ProvenanceTimeline() {
  const [hover, setHover] = useState<string | null>(null);
  return (
    <div className="wide my-8">
      <ol className="!m-0 grid !list-none gap-0 !p-0">
        {MILESTONES.map((m, i) => (
          <li key={m.date + m.title} className="!m-0 grid grid-cols-[92px_22px_minmax(0,1fr)] gap-x-3 !p-0 sm:grid-cols-[120px_22px_minmax(0,1fr)]">
            <div className="num pt-0.5 text-right text-[15px] text-ink-2">{m.date}</div>
            <div className="relative flex justify-center" aria-hidden="true">
              <div className={cx('absolute top-0 bottom-0 w-[2px] bg-grid', i === 0 && 'top-3', i === MILESTONES.length - 1 && 'bottom-auto h-3')} />
              <div
                className={cx(
                  'relative mt-1.5 h-3.5 w-3.5 rounded-full border-2',
                  m.kind === 'paper' ? 'border-accent bg-accent' : 'border-ink-2 bg-surface',
                )}
              />
            </div>
            <div className="min-w-0 pb-5">
              <div className="flex flex-wrap items-baseline gap-x-2">
                <span className={cx('text-[16.5px] font-semibold', m.kind === 'paper' ? 'text-ink' : 'text-[15.5px] text-ink')}>
                  {m.kind === 'code' ? (
                    <>
                      <span className="font-mono">{m.title.split(' · ')[0]}</span>
                      {m.title.slice(m.title.split(' · ')[0].length)}
                    </>
                  ) : (
                    m.title
                  )}
                </span>
                <span className="text-[14px] text-muted">{m.kind === 'paper' ? 'paper version' : 'code commit'}</span>
              </div>
              <div className="mt-0.5 text-[15.5px] leading-snug text-ink-2">{m.detail}</div>
              {m.tables && (
                <div className="mt-2 flex flex-wrap gap-1.5">
                  {m.tables.map((t) => (
                    <span
                      key={t.id}
                      tabIndex={0}
                      onPointerEnter={() => setHover(t.id)}
                      onPointerLeave={() => setHover(null)}
                      onFocus={() => setHover(t.id)}
                      onBlur={() => setHover(null)}
                      className={cx(
                        'inline-flex items-center gap-1.5 rounded-md border px-2.5 py-1 text-[15px] transition-colors',
                        hover === t.id ? 'border-accent bg-accent-soft text-accent-text' : 'border-border bg-surface text-ink',
                      )}
                    >
                      <strong className="font-semibold">{t.id}</strong>
                      <span className={cx('text-ink-2', hover === t.id ? 'inline' : 'hidden md:inline')}>{t.note}</span>
                    </span>
                  ))}
                </div>
              )}
            </div>
          </li>
        ))}
      </ol>
      <p className="mx-auto max-w-[var(--prose-max)] text-[15.5px] leading-snug text-ink-2">
        Dates and table histories from the spec’s comparison of the three arXiv PDFs and the repository history <Cite kind="spec" ref="§13 #1" />{' '}
        <Cite kind="spec" ref="§11" />. That the v3 tables were produced with exactly 801a32b is plausible but not provable.
      </p>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Checks computed from the printed tables
// ─────────────────────────────────────────────────────────────────────────────

const ok = (b: boolean) =>
  b ? (
    <Check aria-label="equal" size={17} className="inline text-[var(--good-text)]" />
  ) : (
    <X aria-label="different" size={17} className="inline text-[var(--critical)]" />
  );

export function ProvenanceChecks() {
  const none = TABLE11.clf!.rows.map((r) => r.zs[0]);
  const t1 = CLF_ORDER.map((k) => TABLE1.rows.find((r) => r.task === k)!.v[5] as number);
  const sameV1 = V1_TABLE1_RTNO.every((v, i) => v === none[i]);

  // Fig. 3: the 5 / 4 tasks of Table 3 (assumption: the paper does not list Fig. 3's tasks)
  const c5: TaskKey[] = ['amazon/item-churn', 'amazon/user-churn', 'hm/user-churn', 'stack/user-badge', 'stack/user-engagement'];
  const r4: TaskKey[] = ['amazon/item-ltv', 'amazon/user-ltv', 'hm/item-sales', 'stack/post-votes'];
  const pick = (rows: { task: TaskKey; v: unknown[] }[], keys: TaskKey[], j: number) =>
    mean(keys.map((k) => rows.find((r) => r.task === k)!.v[j] as number));
  const pickZs = (rows: { task: TaskKey; zs: number[] }[], keys: TaskKey[]) => mean(keys.map((k) => rows.find((r) => r.task === k)!.zs[0]));
  const fig3 = [
    { what: 'RT, classification', printed: FIG3_POINTS.clf.rtPretrained, early: pickZs(TABLE11.clf!.rows, c5), late: pick(TABLE1.rows, c5, 5) },
    { what: 'RT, regression', printed: FIG3_POINTS.reg.rtPretrained, early: pickZs(TABLE5_13.reg!.rows, r4), late: pick(TABLE2.rows, r4, 5) },
    { what: 'Griffin, classification', printed: FIG3_POINTS.clf.griffinPretrained, early: null, late: pick(TABLE1.rows, c5, 4) },
    { what: 'Griffin, regression', printed: FIG3_POINTS.reg.griffinPretrained, early: null, late: pick(TABLE2.rows, r4, 4) },
  ];

  // Table 14 NP / 0.0 vs Table 10 RT untrained / pretrained
  const t14 = (['clf', 'reg'] as const).flatMap((m) =>
    TABLE14[m]!.rows.map((r) => {
      const t10 = TABLE10[m].rows.find((x) => x.task === r.task)!;
      return {
        np: r.ft![3] === t10.v[TABLE10[m].cols.indexOf('rtNo')],
        p0: r.ft![0] === t10.v[TABLE10[m].cols.indexOf('rtYes')],
      };
    }),
  );
  const npSame = t14.filter((x) => x.np).length;
  const p0Same = t14.filter((x) => x.p0).length;

  // Table 15 CFN vs Table 1 RT (No)
  const cfn = TABLE15.clf!.rows.map((r) => r.zs[0]);
  const cfnDiff = cfn.filter((v, i) => v !== t1[i]).length;

  return (
    <div className="wide my-8 grid gap-4 lg:grid-cols-2">
      <section className="card min-w-0 px-4 py-4" aria-labelledby="chk-1">
        <h3 id="chk-1" className="!m-0 text-[17px] font-semibold text-ink">
          1 · The ablations’ “none” column is arXiv v1’s Table 1
        </h3>
        <p className="!mt-1 !mb-2 text-[15.5px] leading-snug text-ink-2">
          v1’s RT (No) column, as quoted by the spec <Cite kind="spec" ref="§13 #1" />, next to the v3 Table 11 “none” column and the re-run v3 Table
          1. {sameV1 ? 'The first two agree on all ten tasks.' : 'They do not all agree.'}
        </p>
        <div className="scroll-quiet overflow-x-auto">
          <table className="w-full min-w-[430px] border-collapse text-[15px]">
            <thead>
              <tr className="border-b border-axis text-ink-2">
                <th className="px-2 py-1 text-left text-[14px] font-semibold">Task</th>
                <th className="px-2 py-1 text-right text-[14px] font-semibold whitespace-nowrap">v1 Table 1</th>
                <th className="px-2 py-1 text-right text-[14px] font-semibold whitespace-nowrap">Table 11 none</th>
                <th className="px-1 py-1 text-[14px] font-semibold" />
                <th className="px-2 py-1 text-right text-[14px] font-semibold whitespace-nowrap">v3 Table 1</th>
              </tr>
            </thead>
            <tbody>
              {CLF_ORDER.map((k, i) => (
                <tr key={k} className="border-b border-grid/70">
                  <td className="px-2 py-0.5 text-[14.5px] whitespace-nowrap">{taskLabel(k)}</td>
                  <td className="num px-2 py-0.5 text-right">{f1(V1_TABLE1_RTNO[i])}</td>
                  <td className="num px-2 py-0.5 text-right">{f1(none[i])}</td>
                  <td className="px-1 py-0.5 text-center">{ok(V1_TABLE1_RTNO[i] === none[i])}</td>
                  <td className={cx('num px-2 py-0.5 text-right', t1[i] !== none[i] ? 'font-semibold text-ink' : 'text-muted')}>{f1(t1[i])}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="card min-w-0 px-4 py-4" aria-labelledby="chk-2">
        <h3 id="chk-2" className="!m-0 text-[17px] font-semibold text-ink">
          2 · Fig. 3’s zero-shot points match the early numbers
        </h3>
        <p className="!mt-1 !mb-2 text-[15.5px] leading-snug text-ink-2">
          Our arithmetic, assuming Fig. 3 averages the same five / four tasks as Table 3 (the paper does not list them). The printed RT points equal
          the means of the early columns (Tables 11 and 5 “none”), not of the re-run Tables 1 and 2. For Griffin the paper has one set of numbers, and
          it matches.
        </p>
        <div className="scroll-quiet overflow-x-auto">
          <table className="w-full min-w-[420px] border-collapse text-[15px]">
            <thead>
              <tr className="border-b border-axis text-ink-2">
                <th className="px-2 py-1 text-left text-[14px] font-semibold">Point</th>
                <th className="px-2 py-1 text-right text-[14px] font-semibold whitespace-nowrap">printed</th>
                <th className="px-2 py-1 text-right text-[14px] font-semibold whitespace-nowrap">early tables</th>
                <th className="px-2 py-1 text-right text-[14px] font-semibold whitespace-nowrap">Tables 1 / 2</th>
              </tr>
            </thead>
            <tbody>
              {fig3.map((r) => (
                <tr key={r.what} className="border-b border-grid/70">
                  <td className="px-2 py-1 text-ink">{r.what}</td>
                  <td className="num px-2 py-1 text-right font-semibold">{f1(r.printed)}</td>
                  <td className="num px-2 py-1 text-right">
                    {r.early === null ? (
                      <span className="text-muted">—</span>
                    ) : (
                      <>
                        {r.early.toFixed(2)} {ok(Number(r.early.toFixed(1)) === r.printed)}
                      </>
                    )}
                  </td>
                  <td className="num px-2 py-1 text-right">
                    {r.late.toFixed(2)} {ok(Number(r.late.toFixed(1)) === r.printed)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="card min-w-0 px-4 py-4" aria-labelledby="chk-3">
        <h3 id="chk-3" className="!m-0 text-[17px] font-semibold text-ink">
          3 · Table 14 and Table 10 share their fine-tuning runs
        </h3>
        <p className="!mt-1 !mb-0 text-[15.5px] leading-snug text-ink-2">
          Table 14’s “no pretraining” column equals Table 10’s untrained RT on {npSame} of {t14.length} tasks, and its P(mask) = 0.0 fine-tuned column
          equals Table 10’s pretrained RT on {p0Same} of {t14.length}. Both tables date from v1 <Cite kind="spec" ref="§13 #1" />.
        </p>
      </section>

      <section className="card min-w-0 px-4 py-4" aria-labelledby="chk-4">
        <h3 id="chk-4" className="!m-0 text-[17px] font-semibold text-ink">
          4 · Table 15’s CFN is its own run
        </h3>
        <p className="!mt-1 !mb-0 text-[15.5px] leading-snug text-ink-2">
          CFN is the released order, yet its column differs from Table 1’s RT (No) on {cfnDiff} of 10 tasks (means {f1(TABLE15.clf!.mean.zs[0])} vs{' '}
          {f1(TABLE1.mean[5] as number)}). Compare orders within Table 15, not across tables <Cite kind="spec" ref="§10 #25" />.
        </p>
      </section>
    </div>
  );
}
