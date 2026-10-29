/**
 * Chapter 6 · `sampling`: Building the context window (Algorithm 1).
 *
 * Ground truth: research/RT_TECHNICAL_SPEC.md §4 (all), §2.5, §10 #5-#11, §11.3, and the paper-era
 * sampler rustler/src/fly.rs:357-511. Every figure replays the engine's mirror of fly.rs
 * (src/engine/sampler.ts) through its recorded event trace; nothing is re-decided in the UI.
 */
import { FlaskConical } from 'lucide-react';
import { useMemo, useState, type ReactNode } from 'react';
import {
  Button,
  Callout,
  ChapterHeader,
  ChapterNav,
  ChapterPage,
  Cite,
  CodeRef,
  Figure,
  Quiz,
  ScaleNote,
  Section,
  Term,
  type QuizQuestion,
} from '@/components';
import { useSim } from '@/state/store';
import { AlgorithmPanel } from './AlgorithmPanel';
import { BfsFigure, SamplerControls } from './BfsFigure';
import { FLY_EMIT, FLY_EXPAND, FLY_SELECT } from './code';
import { ExperimentStats, SweepCharts } from './Experiment';
import { ExploreProvider, useExplore } from './explore';
import {
  LabelPaths,
  LabelTimeline,
  LabelTokens,
  LabelViewToggle,
  SplitVisibility,
  Table12,
  useEligibleSelf,
  useLabels,
  useLabelView,
  useOtherEntityPreset,
  useSameHorizonPreset,
} from './Labels';
import { PlayerProvider } from './player';
import { RuleCards } from './RuleCards';
import { plural, rowName, tsDate } from './trace';
import { useHopGrowth, WhySampleFigure } from './WhySample';

// ─────────────────────────────────────────────────────────────────────────────
// Live facts used in the prose
// ─────────────────────────────────────────────────────────────────────────────

function useFacts() {
  const { db, pdb, ctx, task } = useExplore();
  const labels = useLabels();
  const seed = pdb.nodes[ctx.params.seedNode];
  const entity = task.kind === 'forecast' && seed.f2pNbrIdxs.length ? seed.f2pNbrIdxs[0] : null;
  return useMemo(
    () => ({
      seedLong: rowName(db, pdb, seed.nodeIdx).long,
      seedDate: tsDate(seed.ts),
      entityShort: entity !== null ? rowName(db, pdb, entity).short : null,
      entityLong: entity !== null ? rowName(db, pdb, entity).long : null,
      self: labels.self.length,
      other: labels.other.length,
      same: labels.other.filter((l) => l.ts !== null && l.ts === seed.ts).length,
      otherBySplit: {
        Train: labels.other.filter((l) => l.tableType === 'Train').length,
        Val: labels.other.filter((l) => l.tableType === 'Val').length,
        Test: labels.other.filter((l) => l.tableType === 'Test').length,
      },
    }),
    [db, pdb, seed, entity, labels],
  );
}

function RunningExampleNote() {
  const { db, task } = useExplore();
  const setTask = useSim((s) => s.setTask);
  if (task.id === 'shop/customer-churn') return null;
  return (
    <Callout variant="note" title="You are viewing another task">
      <p>
        The figures follow your simulation settings: {db.name} · {task.id.split('/')[1]} (
        {task.kind}). The prose is written for the running example, ShopDB’s customer-churn, so a few sentences may not apply.
      </p>
      <Button size="sm" variant="secondary" onClick={() => setTask('shop/customer-churn')}>
        Switch the course to the running example
      </Button>
    </Callout>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Sections
// ─────────────────────────────────────────────────────────────────────────────

function WhySection() {
  const { db, ctx } = useExplore();
  const f = useFacts();
  const g = useHopGrowth(5);
  const L = ctx.params.L;
  const firstOver = g.cells.findIndex((c) => c > L);
  return (
    <Section id="why" kicker="Step 1" title="Why sample?">
      <p>
        RT reads a database as a set of <Term id="cell-token">cell tokens</Term>, and a whole database does not fit in one sequence. {db.title} is tiny by
        RelBench standards, yet the tables this task’s sampler may enter hold {g.totalCells.toLocaleString('en-US')} feature cells. So for every prediction RT builds a short
        sequence around one <Term id="seed-row">seed row</Term>: the <Term id="context-window">context window</Term> of L cells{' '}
        <Cite kind="paper" ref="p.4 §3.1" />. Here the seed row is {f.seedLong}, and its masked cell is the label to predict.
      </p>
      <p>
        Which L cells? The authors follow “the intuition that most relevant information lies within a few hops of the seed row when following F→P and P→F links,
        and that lower hops are more informative than higher hops” <Cite kind="paper" ref="p.15 App. A" />. A plain search over those hops quickly collects
        more than L cells. The chart counts the cells within k hops of the seed
        {firstOver > 0 ? `, and already at ${firstOver} hop${firstOver > 1 ? 's' : ''} there are more than L = ${L}` : ''}.
      </p>
      <WhySampleFigure />
      <Callout variant="insight" title="Why parents and children are treated differently">
        <p>
          A row has at most one parent per foreign-key column, but it can have any number of children. The paper argues that “each parent row typically
          contains important features (e.g., a transaction row links to user and product)”, while “informative signals from P→F links often arise via
          aggregation, with diminishing returns from including many children”. So the sampler follows every parent right away and keeps only a random sample
          of the children, at most w per row <Cite kind="paper" ref="p.16 App. A" />.
        </p>
      </Callout>
    </Section>
  );
}

function AlgorithmSection() {
  return (
    <Section id="algorithm" kicker="Step 2" title="The algorithm, one decision at a time">
      <p>
        The paper describes the sampler as a breadth-first search with three changes: all parent rows (F→P) are always included, child rows (P→F) are subsampled
        with a BFS width bound w, and rows with timestamps later than the seed row are excluded. Once a row is selected, its non-missing feature cells are added
        to the context <Cite kind="paper" ref="p.4 §3.1" />.
      </p>
      <p>
        The released implementation, <span className="font-mono">rustler/src/fly.rs</span>, keeps the frontier in two structures: an <strong>F→P stack</strong>{' '}
        for parent rows and a list of <strong>P→F depth buckets</strong> for child rows <Cite kind="code" ref="rustler/src/fly.rs:366-368" />. Every frontier
        entry carries a <Term id="depth-label">depth label</Term>, one more than the depth of the row that pushed it <Cite kind="spec" ref="§4.5" />. A row is
        checked against the visited set only when it comes off the frontier, so the same row can wait in the frontier more than once. Every copy after the
        first is skipped <Cite kind="code" ref="rustler/src/fly.rs:399-402" />.
      </p>
      <p>
        The panel below puts the paper’s pseudo-code next to a condensed version of the code’s loop. When you run the replay further down, the lines of the
        current step light up in both, and the replay also quotes them under its player.
      </p>
      <AlgorithmPanel />
      <Callout
        variant="paper-vs-code"
        title="“Closest rows” means the lowest depth label"
        paper={
          <>
            Among child rows, take <span className="font-mono">argmin HopDistance(r, s)</span>, the rows closest to the seed, and pick one at random{' '}
            <Cite kind="paper" ref="p.15 Alg. 1" />.
          </>
        }
        code={
          <>
            Draw an entry uniformly from the lowest depth bucket that is not empty <Cite kind="code" ref="rustler/src/fly.rs:381-396" />. The depth label counts
            the hops of the path through which the entry was found (+1 per hop, F→P or P→F), so it can be larger than the row’s shortest distance to the seed{' '}
            <Cite kind="spec" ref="§4.5" />.
          </>
        }
      />
      <p>
        The replay below opens on the finished context. Press play, or use the arrow keys, to rebuild it one decision at a time; “Each row” steps through it
        faster. You can change the seed, L, w and the RNG seed here without affecting the rest of the course.
      </p>
      <BfsFigure />
    </Section>
  );
}

function RulesSection() {
  const { task } = useExplore();
  return (
    <Section id="rules" kicker="Step 3" title="Four rules worth remembering">
      <p>
        Without the bookkeeping, the released sampler comes down to four rules, including the <Term id="temporal-filter">temporal filter</Term> and the{' '}
        <Term id="bfs-width">width bound w</Term>. Each card shows its rule at work in the context above, and “Show me” moves the replay to that moment.
      </p>
      <p>
        Each rule differs a little from the paper’s description, and each card says how. The released model was trained with the code, so the course follows
        the code <Cite kind="spec" ref="§10" />.
      </p>
      <RuleCards />
      {task.kind === 'autocomplete' ? (
        <Callout variant="note" title="Autocomplete adds one more filter">
          <p>
            For this task, the <Term id="leakage-columns">leakage columns</Term> are skipped for the seed row and for every row whose timestamp equals the seed’s{' '}
            <Cite kind="code" ref="rustler/src/fly.rs:457-461" />. Because the seed is a DB row, no task row enters the context at all{' '}
            <Cite kind="spec" ref="§2.4" />.
          </p>
        </Callout>
      ) : (
        <p>
          Autocomplete tasks add one more filter: their leakage columns are skipped for the seed row and for rows with the seed’s timestamp{' '}
          <Cite kind="code" ref="rustler/src/fly.rs:457-461" />. Forecasting tasks such as {task.name} drop no columns <Cite kind="spec" ref="§4.1" />.
        </p>
      )}
    </Section>
  );
}

function LabelsSection() {
  const ex = useExplore();
  const { task, ctx, split } = ex;
  const f = useFacts();
  const eligible = useEligibleSelf();
  const preset = useSameHorizonPreset();
  const otherPreset = useOtherEntityPreset();
  const [altOn, setAltOn] = useState(false);
  const alt = altOn && otherPreset && otherPreset !== 'pending' ? otherPreset : null;
  const view = useLabelView(alt);
  const toggle = otherPreset && otherPreset !== 'pending' ? <LabelViewToggle preset={otherPreset} value={!!alt} onChange={setAltOn} /> : null;
  const L = ctx.params.L;
  return (
    <Section id="labels" kicker="Step 4" title="In-context labels">
      {task.kind === 'forecast' ? (
        <>
          <p>
            The task table is attached to the database as an ordinary table <Cite kind="paper" ref="p.4 §3.1" />, so the search can walk into it. The path is
            short: the seed row pushes its entity onto the stack (depth 1), and the entity’s children go to bucket 2. Those children are the entity’s DB rows,
            time-filtered and cut to w, and the entity’s rows in the task table (any split) whose timestamp is not later than the seed’s. The task rows are
            never cut to w{' '}
            <Cite kind="spec" ref="§2.5" />
            <Cite kind="code" ref="rustler/src/fly.rs:406-452" />. Their label cells stay visible, which makes them <Term id="in-context-label">in-context labels</Term>.
          </p>
          <p>
            The paper does not count this as few-shot prompting: “While task rows provide ‘in-context labels’, our setting is not few-shot as explicit
            subgraph-label pairs are not required” <Cite kind="paper" ref="p.2" />. The labels reach the model as ordinary cells of the context.
          </p>
          <p>
            In the context of {f.seedLong}, {f.entityShort} contributes{' '}
            <strong>
              {f.self} <Term id="self-label">{f.self === 1 ? 'self label' : 'self labels'}</Term>
            </strong>
            , its own past {task.targetCol} values
            {eligible.length > f.self ? ` (${eligible.length} were eligible; the rest did not fit before L = ${L})` : ''}, and other entities contribute{' '}
            <strong>{f.other}</strong>.
          </p>
        </>
      ) : (
        <p>
          This is an autocomplete task: the seed is a row of {task.table}, so no task row enters the context{' '}
          <Cite kind="code" ref="rustler/src/fly.rs:417" />. Its in-context labels are the visible {task.targetCol} cells of other
          rows of the same table that the search reached <Cite kind="spec" ref="§2.4" />. This context holds {plural(f.other, 'such cell')}.
        </p>
      )}
      <Figure wide title="Labels the model can read, on a time axis" caption={<>Top: label rows by time; everything right of the seed line is later than the seed and is never sampled <Cite kind="code" ref="rustler/src/fly.rs:421-424" />. Bottom: the same label cells among all L tokens.</>}>
        {toggle}
        <LabelTimeline view={view} />
        <div className="mt-5 border-t border-grid pt-4">
          <LabelTokens view={view} />
        </div>
      </Figure>

      {task.kind === 'forecast' ? (
        <p>
          Other entities’ labels arrive by longer paths, for example entity → one of its transactions → the product it points to (F→P) → another transaction of
          that product → its customer (F→P) → that customer’s task rows <Cite kind="spec" ref="§2.5" />. Each hop adds one to the depth label, and the lowest
          bucket is always drawn first, so these rows wait until the shallower buckets are empty <Cite kind="code" ref="rustler/src/fly.rs:381-391" />.
          {otherPreset && otherPreset !== 'pending' ? (
            <>
              {' '}
              With w = {ctx.params.w} this context holds none, so the figures offer a second context of the same seed: with w = {otherPreset.w}, every bucket is
              smaller, and the search reaches other entities within L = {otherPreset.L} cells.
            </>
          ) : null}
        </p>
      ) : (
        <p>
          Here the other rows of {task.table} arrive through shared parents: the seed’s parents are visited first through the
          stack, and their children, which include other {task.table} rows, go to the next bucket <Cite kind="spec" ref="§4.4" />
          <Cite kind="code" ref="rustler/src/fly.rs:406-452" />.
        </p>
      )}
      <Figure title="How each label entered the context" caption="For every label row of the context, the path through which the search visited it, drawn with foreign-key arrows (child → parent).">
        {toggle}
        <LabelPaths view={view} />
      </Figure>

      {task.kind === 'forecast' && (
        <>
          <p>
            Which task rows can appear depends on the seed’s split. All three splits share one table name, so the task-table test lets every split through, and
            the time filter keeps every row with a timestamp no later than the seed’s <Cite kind="spec" ref="§2.5" />:
          </p>
          <ul>
            <li>a Train seed sees Train rows up to its own timestamp;</li>
            <li>a Val seed sees all Train rows, plus other entities’ Val rows at the same timestamp;</li>
            <li>a Test seed sees Train and Val rows, plus other entities’ Test rows at the same timestamp.</li>
          </ul>
          <Figure title={`Visibility of ${task.name} rows by seed split`} caption={`Your seed is a ${split} row. A seed’s own row is the masked target; the rows of other entities at the same timestamp are the “same horizon” rows.`}>
            <SplitVisibility />
          </Figure>
          <Callout variant="note" title="Same-horizon labels are visible">
            <p>
              The code skips a child only when <span className="font-mono">child_ts &gt; seed_ts</span>, so other entities’ task rows with exactly the seed’s
              timestamp pass, and their labels can be in the context <Cite kind="code" ref="rustler/src/fly.rs:422" /> <Cite kind="spec" ref="§2.5" />. Such a row
              belongs to another entity at the <Term id="same-horizon">same horizon</Term>: its label describes that entity’s own future window, the same window
              the seed asks about. In the current context: <strong>{plural(f.same, 'same-horizon label')}</strong>.{' '}
              {preset === 'pending'
                ? 'Searching for settings that reach one…'
                : preset
                  ? `Of the settings L = 128, 256, 512 with w from 16 down to 1 (this RNG seed), the first that reaches one is L = ${preset.L}, w = ${preset.w} (${plural(preset.count, 'same-horizon label')}). A narrower search reaches other entities with fewer cells.`
                  : 'None of the settings tried (L = 128, 256 or 512 with w from 16 down to 1, this RNG seed) reaches one for this seed.'}
            </p>
            {preset && preset !== 'pending' && (
              <div className="mt-2 flex flex-wrap gap-2">
                <Button size="sm" variant="secondary" icon={<FlaskConical size={13} />} onClick={() => ex.set({ w: preset.w, L: preset.L })}>
                  Try L = {preset.L}, w = {preset.w}
                </Button>
                {ex.isLocal && (
                  <Button size="sm" variant="ghost" onClick={ex.reset}>
                    Back to course settings
                  </Button>
                )}
              </div>
            )}
          </Callout>
        </>
      )}

      <h3>How many labels does the paper see?</h3>
      <p>
        For scale, the paper counts the labels in the 1024-cell contexts of its tasks. Two rel-trial tasks, study-outcome and study-adverse, get no labels at
        all, and zero-shot AUROC on study-outcome is 51.8, close to random <Cite kind="paper" ref="p.8 Table 1" />.
      </p>
      <Figure wide>
        <Table12 />
        <p className="mt-3 text-[15px] leading-snug text-ink-2">
          Source: paper Table 12 <Cite kind="paper" ref="p.23 Table 12" />.
        </p>
      </Figure>
      <p>
        Self labels are the main driver of zero-shot transfer. Removing them drops the paper’s mean zero-shot AUROC from 70.1 to 53.8 and R² from 22.8 to −5.5.
        Removing other-entity labels changes little (70.6 AUROC, 22.9 R²) <Cite kind="paper" ref="p.9 Table 4" />. In the authors’ words, “zero-shot transfer is
        driven primarily by the model’s ability to leverage past labels of the target entity” <Cite kind="paper" ref="p.9 §5.1" />.
      </p>
    </Section>
  );
}

function ExperimentSection() {
  return (
    <Section id="experiment" kicker="Step 5" title="Experiment: change L and w">
      <p>
        Move L and w and watch what the context is made of. These sliders share their settings with the replay above, and they stay local to this chapter until
        you apply them to the course.
      </p>
      <ScaleNote paper="L = 1024, w = 256" ours="defaults L = 128, w = 16">
        The sliders go past our defaults on purpose. The rules are the same at any size.
      </ScaleNote>
      <Figure wide title="The context, counted">
        <div className="mb-5 max-w-[640px]">
          <SamplerControls compact />
        </div>
        <ExperimentStats />
      </Figure>
      <SweepCharts />
      <p>
        Two mechanisms shape these counts. First, a bucket is drawn uniformly, so the self labels in bucket 2 compete with the entity’s DB children for the
        early positions. A larger w puts more DB children in that bucket, and a small L can fill up before every self label is drawn{' '}
        <Cite kind="code" ref="rustler/src/fly.rs:381-396" />. Second, a smaller w keeps every bucket smaller, so the search gets to the deeper buckets, where
        other entities’ rows live, after fewer cells <Cite kind="spec" ref="§4.5" />.
      </p>
      <p>
        What does a shorter context cost? The paper varied L at inference for a model pretrained at L = 1024 and found that “RT’s performance degrades
        gracefully, showing a drop of only 4% absolute AUROC and 7% absolute R² with 25% lower time and 20% lower memory” <Cite kind="paper" ref="p.8 §4.5" />{' '}
        (Fig. 4, from L = 1024 down to 256). The playground chapter runs the mini model at different L.
      </p>
    </Section>
  );
}

const COMPARISON: { aspect: string; paper: ReactNode; code: ReactNode; ref: string }[] = [
  { aspect: 'Choice among F→P-added rows', paper: <span className="font-mono">RandomSelect(R)</span>, code: 'LIFO stack pop, deterministic', ref: 'rustler/src/fly.rs:378-379' },
  { aspect: 'Choice among P→F rows', paper: 'min HopDistance, uniform', code: 'lowest depth label, uniform', ref: 'rustler/src/fly.rs:381-396' },
  { aspect: 'Temporal filter', paper: <span className="font-mono">T(q) ≤ T(s)</span>, code: 'same, only when both timestamps exist', ref: 'rustler/src/fly.rs:422' },
  { aspect: 'Width bound', paper: <span className="font-mono">RandomSample(N, w)</span>, code: 'DB children only; task rows never subsampled', ref: 'rustler/src/fly.rs:426-452' },
  { aspect: 'Task tables', paper: '“only one task table is active at a time”', code: 'only the seed’s task table, any split', ref: 'rustler/src/fly.rs:417' },
  { aspect: 'Stop', paper: <span className="font-mono">while |C| &lt; L</span>, code: 'hard stop at exactly L tokens; last row may be partial', ref: 'rustler/src/fly.rs:502-509' },
  { aspect: 'Cells of a row', paper: 'non-missing feature cells', code: 'same, minus leakage columns (autocomplete) for the seed and same-timestamp rows', ref: 'rustler/src/fly.rs:457-461' },
  { aspect: 'Masked cells', paper: 'the target cell', code: 'exactly one: the seed’s target cell', ref: 'rustler/src/fly.rs:492-496' },
];

function CodeSection() {
  return (
    <Section id="alg-vs-code" kicker="Reference" title="Algorithm 1 vs the released code">
      <p>
        The table compares Algorithm 1 with the released code, one aspect per row <Cite kind="spec" ref="§4.15" />. The paper column draws on Algorithm 1{' '}
        <Cite kind="paper" ref="p.15 Alg. 1" />, Appendix A <Cite kind="paper" ref="p.16 App. A" /> and Section 3.1 <Cite kind="paper" ref="p.4 §3.1" />.
      </p>
      {/* narrow screens: one card per aspect, so the Code and Where columns are never scrolled away */}
      <dl className="wide my-6 grid gap-3 sm:hidden">
        {COMPARISON.map((r) => (
          <div key={r.aspect} className="rounded-xl border border-border bg-surface px-3.5 py-3">
            <dt className="text-[16.5px] font-semibold text-ink">{r.aspect}</dt>
            <dd className="mt-1.5 grid gap-1 text-[16px] leading-snug">
              <div className="text-ink-2">
                <span className="eyebrow mr-1.5 !text-[13px]">Paper</span>
                {r.paper}
              </div>
              <div className="text-ink">
                <span className="eyebrow mr-1.5 !text-[13px]">Code</span>
                {r.code} <Cite kind="code" ref={r.ref} />
              </div>
            </dd>
          </div>
        ))}
      </dl>
      <div className="wide my-6 hidden overflow-x-auto sm:block">
        <table className="w-full min-w-[600px] border-collapse text-[16.5px]">
          <thead>
            <tr className="border-b border-grid text-left text-[15px] text-ink-2">
              <th scope="col" className="py-2 pr-4 font-medium">Aspect</th>
              <th scope="col" className="py-2 pr-4 font-medium">Paper</th>
              <th scope="col" className="py-2 pr-4 font-medium">Code (this site)</th>
              <th scope="col" className="py-2 font-medium">Where</th>
            </tr>
          </thead>
          <tbody>
            {COMPARISON.map((r) => (
              <tr key={r.aspect} className="border-b border-grid/70 align-top">
                <th scope="row" className="py-2 pr-4 text-left font-medium text-ink">
                  {r.aspect}
                </th>
                <td className="py-2 pr-4 text-ink-2">{r.paper}</td>
                <td className="py-2 pr-4 text-ink">{r.code}</td>
                <td className="py-2">
                  <Cite kind="code" ref={r.ref} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p>The three excerpts below are the whole sampling loop, verbatim, with the lines that implement the rules highlighted.</p>
      <CodeRef
        title="Setup and row selection: the stack wins, else a uniform draw from the lowest bucket"
        file={FLY_SELECT.file}
        lines={[FLY_SELECT.start, FLY_SELECT.end]}
        code={FLY_SELECT.code}
        highlightLines={[366, 368, 378, 379, 387, 388, 390, 391, 392, 393, 394, 399, 400, 401]}
        note="f2p_ftr is the F→P stack; p2f_ftr[depth] are the P→F buckets. The visited check happens after the pop."
      />
      <CodeRef
        title="Expansion: parents, the task-table test, the time filter and the width bound"
        file={FLY_EXPAND.file}
        lines={[FLY_EXPAND.start, FLY_EXPAND.end]}
        code={FLY_EXPAND.code}
        highlightLines={[406, 407, 417, 422, 426, 427, 431, 436, 439, 440]}
        note="Parents are pushed with no test at all; DB children are collected, then cut to max_bfs_width (w); task rows go straight into the bucket."
      />
      <CodeRef
        title="Emission: leakage drop, the single masked cell, and the hard stop at L"
        file={FLY_EMIT.file}
        lines={[FLY_EMIT.start, FLY_EMIT.end]}
        code={FLY_EMIT.code}
        highlightLines={[457, 458, 492, 493, 496, 503, 504, 507, 508]}
        note="is_target is true only for the seed row’s target column, and masks = is_target: exactly one masked token per context."
      />
    </Section>
  );
}

const QUIZ: QuizQuestion[] = [
  {
    prompt: 'Which rows are never tested by the temporal filter?',
    choices: [
      {
        text: 'Parent rows reached through F→P links',
        correct: true,
        why: 'fly.rs:406-408 pushes every parent without looking at its timestamp. The test at fly.rs:421-424 runs only on P→F children.',
      },
      {
        text: 'Rows of the seed’s own task table',
        why: 'Task rows are children of the entity, so they are tested, and only rows with a timestamp no later than the seed’s pass. They bypass only the width bound.',
      },
      { text: 'The entity’s DB children at depth 2', why: 'They are P→F children, so they are compared with the seed’s timestamp.' },
      {
        text: 'Rows whose timestamp equals the seed’s',
        why: 'They are tested, and they pass because the filter keeps child_ts ≤ seed_ts.',
      },
    ],
    explanation: (
      <>
        “All parent rows (F→P) are always included” (paper p.4). In the code the filter sits inside the P→F loop only, and children without a timestamp pass it
        too (spec §4.6).
      </>
    ),
  },
  {
    prompt: 'A customer has 40 orders before the seed’s timestamp and no reviews. With w = 16, how many of those orders can enter the context through that customer?',
    choices: [
      { text: 'All 40: the target entity’s rows are never subsampled', why: 'That holds for task rows only. Orders are DB children, so the width bound applies.' },
      { text: 'Exactly 16', why: '16 are pushed, but some may already be visited, and L may run out before all of them are drawn.' },
      {
        text: 'At most 16, possibly fewer',
        correct: true,
        why: 'A uniform random 16 are pushed to the next bucket, but some of them may already be visited, and L may run out before all 16 are drawn.',
      },
      { text: 'None: orders are only reached through products', why: 'Orders are children of the customer, so the customer’s expansion pushes them.' },
    ],
    explanation: (
      <>
        The bound keeps a uniform random w of the DB children that passed the time filter (fly.rs:439-452). If the customer also had reviews, orders and reviews
        would share the same 16 places. Its task rows bypass the bound.
      </>
    ),
  },
  {
    prompt: 'The seed is a Val row at the validation timestamp. Which other-entity labels can the released sampler put in its context?',
    choices: [
      { text: 'Train rows only', why: 'Other entities’ Val rows at the same timestamp pass too, because equal timestamps are kept.' },
      {
        text: 'Train rows, plus other entities’ Val rows at the same timestamp',
        correct: true,
        why: 'All splits share one table name (fly.rs:417) and the filter keeps child_ts ≤ seed_ts (fly.rs:422).',
      },
      { text: 'Train, Val and Test rows', why: 'Test rows are later than the validation timestamp, so the time filter skips them.' },
      { text: 'None, because other entities’ labels would leak', why: 'The released sampler does show them: rows with the seed’s own timestamp pass the time filter.' },
    ],
    explanation: (
      <>
        Other entities’ Val rows at the seed’s timestamp are the “same-horizon” rows: each label describes another entity’s outcome over the same future window
        the seed asks about (spec §2.5).
      </>
    ),
  },
];

// ─────────────────────────────────────────────────────────────────────────────
// Page
// ─────────────────────────────────────────────────────────────────────────────

export default function SamplingChapter() {
  return (
    <ChapterPage slug="sampling">
      <ExploreProvider>
        <PlayerProvider>
          <ChapterHeader
            slug="sampling"
            lede="RT never reads a whole database. For each prediction it gathers L cells around the row to predict, with a bounded breadth-first search. This chapter runs the released sampler on the live seed, one decision at a time."
            learn={[
              'Why RT samples a context window, and what Algorithm 1 does',
              'The four rules of the released sampler: parents first, the temporal filter, the width bound w, and the hard stop at L',
              'How past labels of the same entity, and of other entities, end up in the context',
              'Where the paper’s pseudo-code and the released code differ',
            ]}
          >
            <ScaleNote paper="L = 1024, w = 256 (w from the released scripts)" ours="L = 128, w = 16">
              Every rule in this chapter is the released sampler’s, and only these two sizes are smaller. Our random-number generator differs from the Rust
              one but draws from the same distributions.
            </ScaleNote>
          </ChapterHeader>
          <div className="page-grid">
            <RunningExampleNote />
          </div>
          <WhySection />
          <AlgorithmSection />
          <RulesSection />
          <LabelsSection />
          <ExperimentSection />
          <CodeSection />
          <div className="page-grid">
            <Quiz questions={QUIZ} />
          </div>
          <ChapterNav slug="sampling" />
        </PlayerProvider>
      </ExploreProvider>
    </ChapterPage>
  );
}
