/**
 * Section figures for the zero-shot results: Table 1 (AUROC, 10 binary tasks) and Table 2 + Table 16
 * (R², 8 regression tasks). A means chart grouped by setting, and a per-task explorer with presets.
 */
import { useMemo, useState } from 'react';
import { Cite, cx, Figure, Segmented } from '@/components';
import { CLF_ORDER, REG_ORDER, STATED_RATIOS, TABLE1, TABLE10, TABLE2, taskLabel, type Cell, type PaperTable } from './data';
import { DotPlot, type DotRow } from './DotPlot';
import { cellText, f1, isNum, MethodGlyph, MethodLegend, PaperTableView, TitleWithBadge, ChartTableFrame } from './kit';
import { MeanChart, type MeanGroup } from './MeanChart';
import { METHODS } from './methods';

type Metric = 'clf' | 'reg';

const GROUP_OF: Record<string, string> = {
  gemma4b: 'Maybe',
  gemma12b: 'Maybe',
  gemma27b: 'Maybe',
  entityMean: 'No',
  griffinNo: 'No',
  rtNo: 'No',
  relllmYes: 'Yes',
  griffinYes: 'Yes',
  rtYes: 'Yes',
};

const GROUP_SUB: Record<string, string> = {
  Maybe: 'LLMs, pretraining data unknown',
  No: 'target database unseen',
  Yes: 'continued pretraining on the target database',
};

function tableFor(metric: Metric): PaperTable<string> {
  return (metric === 'clf' ? TABLE1 : TABLE2) as unknown as PaperTable<string>;
}

/** Column index of a method in Table 1 / Table 2. */
function colIndex(t: PaperTable<string>, id: string): number {
  return t.cols.indexOf(id);
}

// ─────────────────────────────────────────────────────────────────────────────
// Means
// ─────────────────────────────────────────────────────────────────────────────

export function ZeroShotMeans({ metric }: { metric: Metric }) {
  const t = tableFor(metric);
  const groups: MeanGroup[] = (['Maybe', 'No', 'Yes'] as const)
    .map((g) => ({
      label: g === 'Maybe' ? 'Maybe' : g === 'No' ? 'No' : 'Yes',
      sub: GROUP_SUB[g],
      items: t.cols
        .filter((c) => GROUP_OF[c] === g)
        .map((c) => ({
          method: c,
          value: t.mean[colIndex(t, c)],
          note:
            metric === 'reg' && c.startsWith('gemma')
              ? 'Table 16 (p.26). “LLM baselines are poor”: the paper reports R² below −9 on most tasks.'
              : undefined,
        })),
    }))
    .filter((g) => g.items.length > 0);
  if (metric === 'reg') {
    // Table 16 prints "−" for RelLLM: no regression numbers.
    const yes = groups.find((g) => g.label === 'Yes');
    yes?.items.unshift({ method: 'relllmYes', value: null, note: 'Table 16 prints “−” for RelLLM: no regression result.' });
  }
  return (
    <Figure
      wide
      title={
        <TitleWithBadge
          title={metric === 'clf' ? 'Mean zero-shot AUROC over 10 binary tasks' : 'Mean zero-shot R² over 8 regression tasks'}
          kind="paper"
          badge={metric === 'clf' ? 'Table 1 · p.8' : 'Tables 2 and 16 · p.8, p.26'}
        />
      }
      caption={
        metric === 'clf' ? (
          <>
            Each stem starts at the random baseline (AUROC 50) and ends at the method’s mean, as printed. Circles: the target database was never seen
            (“No”). Squares: continued pretraining on the target database’s other tasks (“Yes”). The target task itself is unseen in every column{' '}
            <Cite kind="paper" ref="p.8 Table 1" />. Hover a row for what the method is.
          </>
        ) : (
          <>
            Stems start at the global-mean baseline (R² 0). The Gemma3 rows sit off the left edge: Table 16 prints “&lt; −9” for their means{' '}
            <Cite kind="paper" ref="p.26 Table 16" />. RelLLM has no regression result <Cite kind="paper" ref="p.26 Table 16" />.
          </>
        )
      }
    >
      <MeanChart
        groups={groups}
        domain={metric === 'clf' ? [50, 75] : [-10, 30]}
        baseline={metric === 'clf' ? 50 : 0}
        baselineLabel={metric === 'clf' ? 'random = 50' : 'global mean = 0'}
        axisLabel={metric === 'clf' ? 'mean test AUROC (%)' : 'mean test R² (%)'}
        emphasis={['rtNo', 'rtYes']}
        ariaLabel={metric === 'clf' ? 'Mean zero-shot AUROC per method, paper Table 1' : 'Mean zero-shot R² per method, paper Tables 2 and 16'}
      />
    </Figure>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-task explorer
// ─────────────────────────────────────────────────────────────────────────────

interface Preset {
  id: string;
  label: string;
  methods: string[];
  focus: string;
}

const PRESETS: Record<Metric, Preset[]> = {
  clf: [
    { id: 'no', label: 'Unseen database', methods: ['entityMean', 'griffinNo', 'rtNo'], focus: 'rtNo' },
    { id: 'llm', label: 'RT vs LLMs', methods: ['gemma4b', 'gemma12b', 'gemma27b', 'rtNo'], focus: 'rtNo' },
    { id: 'yes', label: 'Continued pretraining', methods: ['relllmYes', 'griffinYes', 'rtYes', 'rtNo'], focus: 'rtYes' },
    { id: 'all', label: 'All nine', methods: [...TABLE1.cols], focus: 'rtNo' },
  ],
  reg: [
    { id: 'no', label: 'Unseen database', methods: ['entityMean', 'griffinNo', 'rtNo'], focus: 'rtNo' },
    { id: 'llm', label: 'RT vs LLMs', methods: ['gemma4b', 'gemma12b', 'gemma27b', 'rtNo'], focus: 'rtNo' },
    { id: 'yes', label: 'Continued pretraining', methods: ['griffinYes', 'rtYes', 'rtNo'], focus: 'rtYes' },
    { id: 'all', label: 'All eight', methods: [...TABLE2.cols], focus: 'rtNo' },
  ],
};

export function ZeroShotTasks({ metric }: { metric: Metric }) {
  const t = tableFor(metric);
  const [presetId, setPresetId] = useState('no');
  const preset = PRESETS[metric].find((p) => p.id === presetId) ?? PRESETS[metric][0];
  const [sort, setSort] = useState<'paper' | 'focus'>('paper');

  const rows: DotRow[] = useMemo(() => {
    const body = t.rows.map((r) => ({
      id: r.task,
      label: taskLabel(r.task),
      values: Object.fromEntries(t.cols.map((c, j) => [c, r.v[j]])) as Record<string, Cell>,
    }));
    if (sort === 'focus') {
      const j = colIndex(t, preset.focus);
      body.sort((a, b) => {
        const va = t.rows.find((r) => r.task === a.id)!.v[j];
        const vb = t.rows.find((r) => r.task === b.id)!.v[j];
        return (isNum(vb) ? vb : -1e9) - (isNum(va) ? va : -1e9);
      });
    }
    return [
      ...body,
      {
        id: 'mean',
        label: 'Mean (as printed)',
        mean: true,
        values: Object.fromEntries(t.cols.map((c, j) => [c, t.mean[j]])) as Record<string, Cell>,
      },
    ];
  }, [t, sort, preset.focus]);

  const order = metric === 'clf' ? CLF_ORDER : REG_ORDER;
  const table = (
    <PaperTableView
      headCols={['Dataset', 'Task']}
      columns={t.cols.map((c) => ({ id: c, label: METHODS[c]?.label ?? c, group: GROUP_OF[c], method: c }))}
      rows={[
        ...order.map((k) => {
          const r = t.rows.find((x) => x.task === k)!;
          return { id: k, head: [taskLabel(k).split(' · ')[0], taskLabel(k).split(' · ')[1]], cells: r.v };
        }),
        { id: 'mean', head: ['Mean', ''], cells: t.mean, mean: true },
      ]}
      markMax
      caption={
        <>{metric === 'clf' ? TABLE1.caption : TABLE2.caption} The best entry of each row is underlined (computed here; the paper uses bold).</>
      }
    />
  );

  return (
    <Figure
      wide
      title={
        <TitleWithBadge
          title={metric === 'clf' ? 'Task by task: zero-shot AUROC' : 'Task by task: zero-shot R²'}
          kind="paper"
          badge={metric === 'clf' ? 'Table 1' : 'Tables 2 and 16'}
        />
      }
      controls={
        <>
          <Segmented
            label="Compare"
            value={presetId}
            onChange={setPresetId}
            options={PRESETS[metric].map((p) => ({ value: p.id, label: p.label }))}
          />
          <Segmented
            label="Row order"
            value={sort}
            onChange={setSort}
            options={[
              { value: 'paper', label: 'Paper order' },
              { value: 'focus', label: `By ${METHODS[preset.focus]?.label}` },
            ]}
          />
        </>
      }
      caption={
        metric === 'clf' ? (
          <>
            Each row is one RelBench task; the grey bar spans the compared methods. The right column prints {METHODS[preset.focus]?.label}. Off-scale
            values would show as a chevron at the left edge. Switch to the table for the paper’s full Table 1.
          </>
        ) : (
          <>
            Chevrons at the left edge are values below the axis: the Gemma3 models score “&lt; −9” on most tasks (Table 16). The vertical line is the
            global-mean baseline, R² = 0.
          </>
        )
      }
    >
      <ChartTableFrame title={<MethodLegend ids={preset.methods} />} table={table}>
        <DotPlot
          rows={rows}
          methods={preset.methods}
          domain={metric === 'clf' ? [40, 95] : [-10, 60]}
          refLine={metric === 'clf' ? { value: 50, label: 'random = 50' } : { value: 0, label: 'mean = 0' }}
          axisLabel={metric === 'clf' ? 'test AUROC (%)' : 'test R² (%)'}
          focus={preset.focus}
          ariaLabel={
            metric === 'clf' ? 'Zero-shot AUROC per task and method, paper Table 1' : 'Zero-shot R² per task and method, paper Tables 2 and 16'
          }
        />
      </ChartTableFrame>
    </Figure>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// "93% of supervised", precisely
// ─────────────────────────────────────────────────────────────────────────────

export function SupervisedRatio() {
  const sup = TABLE10.clf.mean[TABLE10.clf.cols.indexOf('rtYes')] as number; // 77.2
  const items = [
    {
      id: 'rtYes',
      zs: TABLE1.mean[TABLE1.cols.indexOf('rtYes')] as number,
      stated: STATED_RATIOS.rtYes,
      what: 'RT with continued pretraining on the target database (target task unseen)',
    },
    { id: 'rtNo', zs: TABLE1.mean[TABLE1.cols.indexOf('rtNo')] as number, stated: STATED_RATIOS.rtNo, what: 'RT on a database it never saw' },
    {
      id: 'gemma27b',
      zs: TABLE1.mean[TABLE1.cols.indexOf('gemma27b')] as number,
      stated: STATED_RATIOS.gemma27b,
      what: 'Gemma3-27B with the same context as text',
    },
  ];
  return (
    <Figure
      wide
      title={<TitleWithBadge title="“93% of fully supervised AUROC”, precisely" kind="paper" badge="p.1-2, Tables 1 and 10" />}
      caption={
        <>
          The denominator is the supervised mean AUROC of fine-tuned, pretrained RT, {f1(sup)} <Cite kind="paper" ref="p.21 Table 10" />. Each ratio
          is computed here from the printed means and matches the percentage the paper states <Cite kind="paper" ref="p.2 §1" />. The abstract’s “93%”
          is the continued-pretraining number; on unseen databases it is 90.3% <Cite kind="spec" ref="§10 #26" />.
        </>
      }
    >
      <div className="grid gap-3 md:grid-cols-3">
        {items.map((it) => {
          const ratio = (it.zs / sup) * 100;
          return (
            <div
              key={it.id}
              className={cx('rounded-xl border px-4 py-3.5', it.id === 'rtYes' ? 'border-accent/50 bg-accent-soft' : 'border-border bg-surface')}
            >
              <div className="flex items-center gap-2 text-[15.5px] font-semibold text-ink">
                <MethodGlyph id={it.id} />
                {METHODS[it.id].label}
              </div>
              <div className="mt-1 text-[15px] leading-snug text-ink-2">{it.what}</div>
              <div className="mt-3 flex items-baseline gap-2">
                <span className="text-[34px] leading-none font-semibold tracking-tight text-ink">{ratio.toFixed(1)}%</span>
                <span className="text-[15px] text-muted">paper: {it.stated}%</span>
              </div>
              <div className="num mt-2 text-[15px] text-ink-2">
                {f1(it.zs)} / {f1(sup)} = {(it.zs / sup).toFixed(3)}
              </div>
              <div className="mt-2 h-2.5 overflow-hidden rounded-full bg-surface-3" aria-hidden="true">
                <div className="h-full rounded-full" style={{ width: `${Math.min(100, ratio)}%`, background: `var(--cat-${METHODS[it.id].slot})` }} />
              </div>
            </div>
          );
        })}
      </div>
      <p className="mt-3 text-[15px] leading-snug text-ink-2">
        Abstract: “averaging 93% of fully supervised AUROC … as opposed to 84% for a 27B LLM” <Cite kind="paper" ref="p.1 Abstract" />. The paper adds
        that Gemma3-27B takes 10<sup className="text-[13px]">5</sup>× more inference FLOPs <Cite kind="paper" ref="p.2-3 §1" />.
      </p>
    </Figure>
  );
}

/** Count of tasks where RT (No) beats EntityMean, from Table 1 (computed). */
export function rtVsEntityMean(metric: Metric): { wins: number; total: number; losses: string[] } {
  const t = tableFor(metric);
  const jr = colIndex(t, 'rtNo');
  const je = colIndex(t, 'entityMean');
  let wins = 0;
  const losses: string[] = [];
  for (const r of t.rows) {
    const a = r.v[jr];
    const b = r.v[je];
    if (isNum(a) && isNum(b)) {
      if (a > b) wins++;
      else if (a < b) losses.push(`${taskLabel(r.task)}, ${cellText(b)} vs ${cellText(a)}`);
    }
  }
  return { wins, total: t.rows.length, losses };
}
