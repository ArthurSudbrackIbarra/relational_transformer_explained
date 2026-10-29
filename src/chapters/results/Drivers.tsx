/**
 * The ranked drivers of zero-shot transfer (paper p.10 §7), each with the ablation evidence behind it
 * (Tables 4, 5, 13; App. C) on one shared change scale, the RT component it relates to and the chapter
 * where this course builds it. Driver 1 also shows the live context's self labels.
 */
import { motion, useInView, useReducedMotion } from 'motion/react';
import { ArrowRight } from 'lucide-react';
import { useRef } from 'react';
import { Link } from 'react-router-dom';
import { Cite, LayerLabel } from '@/components';
import { getChapter } from '@/chapters/registry';
import { useInContextLabels, useTask } from '@/state/hooks';
import { MAX_JACCARD, TABLE4, TABLE5_13 } from './data';
import { d1, f1 } from './kit';

const T4 = Object.fromEntries(TABLE4.rows.map((r) => [r.id, r])) as Record<(typeof TABLE4.rows)[number]['id'], (typeof TABLE4.rows)[number]>;
const L5 = (layer: string, m: 'clf' | 'reg') => {
  const j = TABLE5_13.cols.indexOf(layer);
  return { none: TABLE5_13[m]!.mean.zs[0], v: TABLE5_13[m]!.mean.zs[j] };
};

interface Evidence {
  label: string;
  from: number;
  to: number;
  cite: string;
}

interface Driver {
  rank: number;
  quote: string;
  plain: React.ReactNode;
  evidence: Evidence[];
  chapters: string[];
}

const DOMAIN: [number, number] = [-30, 5];

function DeltaBar({ e, index }: { e: Evidence; index: number }) {
  const ref = useRef<HTMLDivElement | null>(null);
  const inView = useInView(ref, { once: true, amount: 0.6 });
  const reduce = useReducedMotion();
  const d = e.to - e.from;
  const span = DOMAIN[1] - DOMAIN[0];
  const zero = ((0 - DOMAIN[0]) / span) * 100;
  const clamped = Math.max(DOMAIN[0], Math.min(DOMAIN[1], d));
  const w = (Math.abs(clamped) / span) * 100;
  const left = d < 0 ? zero - w : zero;
  return (
    <div ref={ref} className="grid gap-1 sm:grid-cols-[minmax(0,13rem)_minmax(0,1fr)_auto] sm:items-center sm:gap-3">
      <div className="flex items-baseline justify-between gap-3 sm:contents">
        <div className="text-[15px] leading-snug text-ink-2 sm:order-1">
          {e.label} <Cite kind="paper" ref={e.cite} />
        </div>
        <span className="num shrink-0 text-right text-[15px] whitespace-nowrap text-ink sm:order-3 sm:min-w-[176px]">
          {f1(e.from)} → {f1(e.to)} <strong>({d1(d)})</strong>
        </span>
      </div>
      <div className="relative h-5 min-w-0 rounded bg-surface-2 sm:order-2" aria-hidden="true">
        <div className="absolute top-0 bottom-0 w-px bg-ink-2" style={{ left: `${zero}%` }} />
        <motion.div
          className="absolute top-1 bottom-1"
          style={{ background: d < 0 ? 'var(--div-pos)' : 'var(--div-neg)', borderRadius: d < 0 ? '4px 0 0 4px' : '0 4px 4px 0' }}
          initial={false}
          animate={inView || reduce ? { left: `${left}%`, width: `${w}%` } : { left: `${zero}%`, width: '0%' }}
          transition={reduce ? { duration: 0 } : { duration: 0.6, delay: 0.1 * index, ease: [0.2, 0.7, 0.2, 1] }}
        />
      </div>
    </div>
  );
}

export function Drivers() {
  const labels = useInContextLabels();
  const task = useTask();
  const drivers: Driver[] = [
    {
      rank: 1,
      quote: 'time-series forecasting from past task rows for the target entity',
      plain: (
        <>
          The target cell reads the entity’s own past labels through column attention: they share the label column.{' '}
          {task.kind === 'forecast' ? (
            <>
              In the live context of <code>{task.id}</code>, the masked cell can see <strong>{labels.self.length}</strong> self{' '}
              {labels.self.length === 1 ? 'label' : 'labels'} and {labels.other.length} other-entity {labels.other.length === 1 ? 'label' : 'labels'}.
            </>
          ) : (
            <>The live task is an autocomplete task, which has no entity: its visible target cells all count as “other”.</>
          )}
        </>
      ),
      evidence: [
        { label: 'self labels removed, zero-shot AUROC', from: T4.none.zsClf, to: T4.selfLabels.zsClf, cite: 'p.9 Table 4' },
        { label: 'self labels removed, zero-shot R²', from: T4.none.zsReg, to: T4.selfLabels.zsReg, cite: 'p.9 Table 4' },
      ],
      chapters: ['sampling', 'masks'],
    },
    {
      rank: 2,
      quote: 'column attention with the help of Relational Attention to generalize to new column distributions',
      plain: (
        <>
          <LayerLabel layer="col" /> attention lets a cell read the other values of its column, “to model value distributions within a column”{' '}
          <Cite kind="paper" ref="p.2 §1" />. Removing it cost more zero-shot R² than removing any other layer.
        </>
      ),
      evidence: [
        { label: 'column attention removed, zero-shot R²', from: L5('col', 'reg').none, to: L5('col', 'reg').v, cite: 'p.10 Table 5' },
        { label: 'column attention removed, zero-shot AUROC', from: L5('col', 'clf').none, to: L5('col', 'clf').v, cite: 'p.23 Table 13' },
      ],
      chapters: ['masks', 'attention'],
    },
    {
      rank: 3,
      quote:
        'feature attention, both in the same table and in parent tables, with the help of Relational Attention to generalize to new entity types',
      plain: (
        <>
          <LayerLabel layer="feat" /> attention mixes a row’s own cells with its parents’ cells. It is the only mechanism that groups cells into
          rows <Cite kind="paper" ref="p.19 App. E" />.
        </>
      ),
      evidence: [
        { label: 'feature attention removed, zero-shot R²', from: L5('feat', 'reg').none, to: L5('feat', 'reg').v, cite: 'p.10 Table 5' },
        { label: 'feature attention removed, zero-shot AUROC', from: L5('feat', 'clf').none, to: L5('feat', 'clf').v, cite: 'p.23 Table 13' },
      ],
      chapters: ['masks', 'block'],
    },
    {
      rank: 4,
      quote: 'schema semantics from table and column names',
      plain: (
        <>
          Every token carries the MiniLM embedding of “&lt;column&gt; of &lt;table&gt;”, and “wrong names lead to consistent degradation”{' '}
          <Cite kind="paper" ref="p.9 §5.1" />. Exact name overlap between RelBench databases is at most {(MAX_JACCARD * 100).toFixed(1)}% (Jaccard),
          so what transfers is similar meaning rather than shared strings <Cite kind="paper" ref="p.17 App. C.2" />.
        </>
      ),
      evidence: [
        { label: 'names shuffled, zero-shot AUROC', from: T4.none.zsClf, to: T4.colNames.zsClf, cite: 'p.9 Table 4' },
        { label: 'names shuffled, zero-shot R²', from: T4.none.zsReg, to: T4.colNames.zsReg, cite: 'p.9 Table 4' },
      ],
      chapters: ['semantics', 'embedding'],
    },
    {
      rank: 5,
      quote: 'in-context learning from task rows for other entities',
      plain: <>Labels of other entities also reach the target. On average, removing them did not lower either mean, which fits its last place.</>,
      evidence: [
        { label: 'other labels removed, zero-shot AUROC', from: T4.none.zsClf, to: T4.otherLabels.zsClf, cite: 'p.9 Table 4' },
        { label: 'other labels removed, zero-shot R²', from: T4.none.zsReg, to: T4.otherLabels.zsReg, cite: 'p.9 Table 4' },
      ],
      chapters: ['sampling', 'tasks'],
    },
  ];

  return (
    <div className="wide my-8">
      <ol className="!m-0 grid !list-none gap-3 !p-0">
        {drivers.map((d, i) => (
          <li key={d.rank} className="card !m-0 grid gap-4 px-4 py-4 sm:grid-cols-[56px_minmax(0,1fr)] sm:px-5">
            <div className="flex items-center gap-3 sm:block">
              <div
                className="flex h-12 w-12 items-center justify-center rounded-full text-[24px] font-semibold"
                style={{ background: i === 0 ? 'var(--accent)' : 'var(--surface-3)', color: i === 0 ? 'var(--accent-ink)' : 'var(--ink)' }}
                aria-label={`Rank ${d.rank}`}
              >
                {d.rank}
              </div>
            </div>
            <div className="min-w-0">
              <p className="!m-0 text-[18px] leading-snug text-ink">
                “{d.quote}” <Cite kind="paper" ref="p.10 §7" />
              </p>
              <p className="!mt-1.5 !mb-0 text-[16.5px] leading-snug text-ink-2">{d.plain}</p>
              <div className="mt-3 grid gap-2">
                {d.evidence.map((e, k) => (
                  <DeltaBar key={e.label} e={e} index={k} />
                ))}
              </div>
              <div className="mt-3 flex flex-wrap gap-2">
                {d.chapters.map((slug) => {
                  const ch = getChapter(slug);
                  if (!ch) return null;
                  return (
                    <Link
                      key={slug}
                      to={`/${slug}`}
                      data-plain
                      className="inline-flex items-center gap-1.5 rounded-full border border-border bg-surface-2 px-3 py-1 text-[15px] text-ink-2 transition-colors hover:border-accent hover:text-accent-text"
                    >
                      Ch. {ch.number}: {ch.short} <ArrowRight aria-hidden="true" size={14} />
                    </Link>
                  );
                })}
              </div>
            </div>
          </li>
        ))}
      </ol>
      <p className="mx-auto mt-3 max-w-[var(--prose-max)] text-[16px] leading-snug text-ink-2">
        Bars share one scale, −30 to +5 points; red is a drop. Each bar is a mean over RelBench tasks. The ranking is the authors’ reading of their
        ablations and does not follow the size of these bars: for classification, several ablations move the mean by less than a point.
      </p>
    </div>
  );
}
