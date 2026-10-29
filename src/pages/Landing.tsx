/**
 * Landing page: hero (title, pitch, animated motif, Start CTA), RT's three
 * ideas, what runs here (scale), and the course map of all 17 chapters.
 */
import { ArrowRight, Award, Check, Columns3, Table2, Type } from 'lucide-react';
import { useEffect } from 'react';
import { Link } from 'react-router-dom';
import { CHAPTERS, PARTS, prefetchChapter } from '@/chapters/registry';
import { Cite } from '@/components/prose/Cite';
import { CODE, PAPER, SITE_TITLE } from '@/components/prose/sources';
import { SCALE_ROWS } from '@/components/prose/ScaleNote';
import { ContentColumn } from '@/components/shell/Layout';
import { cx } from '@/components/ui/cx';
import { useSim } from '@/state/store';
import { HeroMotif } from './HeroMotif';

const IDEAS = [
  {
    icon: Type,
    title: 'Cells as tokens',
    body: (
      <>
        Each non-missing feature cell becomes one token. The token combines the cell’s value with the names of its column and
        table, embedded as a phrase such as “age of customers”, so a new schema needs no hand-made features.
        <Cite kind="paper" ref="p.4 §3.1" />
      </>
    ),
  },
  {
    icon: Table2,
    title: 'Task-table prompting',
    body: (
      <>
        The prediction task is attached to the database as one more table. Each row pairs an entity and a time with a label, and a
        prediction fills in the label of one row. Earlier rows of the same task reach the context through the same links as any
        other row, and they bring their labels with them.
        <Cite kind="paper" ref="p.4 §2.2" /> <Cite kind="paper" ref="p.4 §3.1" />
      </>
    ),
  },
  {
    icon: Columns3,
    title: 'Relational attention',
    body: (
      <>
        The model sees the database structure only through four attention masks: column, feature, neighbor and full. They are
        built from the table, column and row of each token and from the links between rows. RT has no positional encodings.
        <Cite kind="paper" ref="p.5 §3.2" />
      </>
    ),
  },
];

export function Landing() {
  const completed = useSim((s) => s.completed);
  const started = Object.keys(completed).length > 0;
  const nextOpen = CHAPTERS.find((c) => !completed[c.slug]) ?? CHAPTERS[0];

  useEffect(() => {
    document.title = SITE_TITLE;
  }, []);

  return (
    <div className="pb-24">
      {/* ── Hero ─────────────────────────────────────────────────────────── */}
      <section className="border-b border-border">
        <ContentColumn className="grid items-center gap-10 py-14 sm:py-20 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.2fr)] lg:gap-14">
          <div className="min-w-0">
            <div className="eyebrow">An interactive, code-faithful walkthrough</div>
            <h1 className="mt-4 text-[40px] leading-[1.05] font-semibold tracking-[-0.03em] text-balance text-ink sm:text-[54px]">
              {SITE_TITLE}
            </h1>
            <p className="mt-5 max-w-[34em] text-[20.5px] leading-[1.6] text-pretty text-ink-2 sm:text-[21.5px]">
              Follow one prediction through every operation of the Relational Transformer, from the rows of a database to a
              zero-shot answer. The figures run live in your browser, on a scaled-down model that does exactly what the paper’s
              code does.
            </p>
            <div className="mt-8 flex flex-wrap items-center gap-3">
              <Link
                to={started ? `/${nextOpen.slug}` : '/overview'}
                data-plain
                onPointerEnter={() => prefetchChapter('overview')}
                className="inline-flex h-11 items-center gap-2 rounded-xl bg-accent px-5 text-[18px] font-semibold text-accent-ink transition hover:brightness-110"
              >
                {started ? `Continue: ${nextOpen.number}. ${nextOpen.short}` : 'Start with the big picture'}
                <ArrowRight aria-hidden="true" size={17} />
              </Link>
              <a
                href="#course-map"
                data-plain
                onClick={(e) => {
                  e.preventDefault();
                  document.getElementById('course-map')?.scrollIntoView({ behavior: 'smooth' });
                }}
                className="inline-flex h-11 items-center rounded-xl border border-border-strong bg-surface px-4 text-[17.5px] font-medium text-ink hover:bg-surface-2"
              >
                See the 17 chapters
              </a>
            </div>
            <p className="mt-6 text-[15.5px] leading-relaxed text-muted">
              Based on {PAPER.authors}, <em>{PAPER.title}</em>, {PAPER.venue}. Every claim cites the paper or the official code at
              commit <span className="font-mono">{CODE.commit}</span>.
            </p>
          </div>
          <div className="card min-w-0 px-3 pt-4 pb-3 sm:px-5 sm:pt-5">
            <HeroMotif />
          </div>
        </ContentColumn>
      </section>

      {/* ── Three ideas ──────────────────────────────────────────────────── */}
      <ContentColumn className="pt-16 sm:pt-20">
        <div className="max-w-[var(--prose-max)]">
          <div className="eyebrow">Three ideas</div>
          <h2 className="mt-2 text-[28px] leading-tight font-semibold tracking-[-0.018em] text-balance text-ink">What makes RT a relational foundation model</h2>
          <p className="mt-3 text-[19px] leading-[1.65] text-ink-2">
            Relational databases vary in almost everything a model might depend on: how many tables they have, which columns and
            types those tables hold, and how the tables link. RT uses one architecture for all of them. After pretraining on other
            databases, it predicts on a database it has never seen without updating its weights.
            <Cite kind="paper" ref="p.4 §2.3" />
          </p>
        </div>
        <div className="mt-8 grid gap-4 md:grid-cols-3">
          {IDEAS.map((idea, i) => {
            const Icon = idea.icon;
            return (
              <div key={idea.title} className="card px-5 py-5">
                <div className="flex items-center gap-3">
                  <span className="inline-flex h-9 w-9 items-center justify-center rounded-lg bg-accent-soft text-accent-text">
                    <Icon aria-hidden="true" size={18} />
                  </span>
                  <span className="num text-[14.5px] text-muted">0{i + 1}</span>
                </div>
                <h3 className="mt-4 text-[19px] font-semibold tracking-[-0.01em] text-ink">{idea.title}</h3>
                <p className="mt-2 text-[17.5px] leading-[1.6] text-ink-2">{idea.body}</p>
              </div>
            );
          })}
        </div>

        {/* ── What runs here ─────────────────────────────────────────────── */}
        <div className="mt-16 grid gap-8 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)] lg:items-start">
          <div className="max-w-[var(--prose-max)]">
            <div className="eyebrow">What runs here</div>
            <h2 className="mt-2 text-[28px] leading-tight font-semibold tracking-[-0.018em] text-balance text-ink">A real model, at small scale</h2>
            <p className="mt-3 text-[19px] leading-[1.65] text-ink-2">
              The site runs a real miniature RT. Its preprocessing, context sampler, masks, blocks, decoders and losses are the same
              as in the paper’s official code. It was pretrained on three synthetic databases and is tested zero-shot on a fourth that
              it never saw. Only the sizes, the data and the numeric precision differ, and the chapters say so wherever that changes a
              number.
            </p>
            <p className="mt-3 text-[19px] leading-[1.65] text-ink-2">
              The running example comes from ShopDB, that fourth database, which holds the records of a small online shop. Its
              customer-churn task asks: will a customer place no order in the next 30 days?
            </p>
          </div>
          <div className="card overflow-hidden">
            <table className="w-full border-collapse text-[16.5px]">
              <thead>
                <tr className="border-b border-grid bg-surface-2 text-left text-[15px] text-ink-2">
                  <th scope="col" className="px-4 py-2.5 font-medium" />
                  <th scope="col" className="px-3 py-2.5 font-semibold">
                    Paper
                  </th>
                  <th scope="col" className="px-4 py-2.5 font-semibold">
                    This site
                  </th>
                </tr>
              </thead>
              <tbody>
                {SCALE_ROWS.map((r) => (
                  <tr key={r.what} className="border-b border-grid/70 last:border-0">
                    <th scope="row" className="px-4 py-2 text-left font-normal text-ink-2">
                      {r.what}
                    </th>
                    <td className="num px-3 py-2 text-[15.5px] text-ink">{r.paper}</td>
                    <td className="num px-4 py-2 text-[15.5px] text-ink">{r.ours}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* ── Course map ─────────────────────────────────────────────────── */}
        <div id="course-map" className="mt-20 scroll-mt-[calc(var(--topbar-h)+24px)]">
          <div className="max-w-[var(--prose-max)]">
            <div className="eyebrow">Course map</div>
            <h2 className="mt-2 text-[28px] leading-tight font-semibold tracking-[-0.018em] text-balance text-ink">Seventeen chapters, one prediction</h2>
            <p className="mt-3 text-[19px] leading-[1.65] text-ink-2">
              Most chapters take one stage of the pipeline and show it on the same live context. Read them in order, or jump to the
              one you need.
            </p>
          </div>
          <div className="mt-8 grid gap-10">
            {PARTS.map((part) => (
              <div key={part.id} className="grid gap-4 lg:grid-cols-[220px_minmax(0,1fr)]">
                <div>
                  <h3 className="text-[17.5px] font-semibold text-ink">{part.title}</h3>
                  <p className="mt-1 text-[16px] leading-snug text-ink-2">{part.blurb}</p>
                </div>
                <ol className="grid gap-3 sm:grid-cols-2">
                  {CHAPTERS.filter((c) => c.part === part.id).map((c) => (
                    <li key={c.slug}>
                      <Link
                        to={`/${c.slug}`}
                        data-plain
                        onPointerEnter={() => prefetchChapter(c.slug)}
                        className={cx(
                          'group flex h-full gap-3.5 rounded-xl border border-border bg-surface px-4 py-3.5 transition-colors hover:border-accent hover:bg-surface',
                        )}
                      >
                        <span className="num w-6 shrink-0 pt-[2px] text-[17.5px] font-semibold text-muted group-hover:text-accent-text">
                          {String(c.number).padStart(2, '0')}
                        </span>
                        <span className="min-w-0 flex-1">
                          <span className="flex items-center gap-2">
                            <span className="text-[18px] font-semibold text-ink">{c.title}</span>
                            {completed[c.slug] && <Check aria-label="completed" size={14} strokeWidth={2.4} className="shrink-0 text-muted" />}
                          </span>
                          <span className="mt-1 block text-[16px] leading-snug text-ink-2">{c.summary}</span>
                        </span>
                      </Link>
                    </li>
                  ))}
                </ol>
              </div>
            ))}
          </div>
          <p className="mt-10 flex max-w-[var(--prose-max)] items-start gap-2.5 text-[18px] leading-[1.6] text-ink-2">
            <Award aria-hidden="true" size={20} className="mt-[0.2em] shrink-0 text-accent-text" />
            <span>
              Finish all seventeen chapters and pass a short final quiz to earn a{' '}
              <Link to="/certificate">certificate of completion</Link> with an ID that anyone can verify.
            </span>
          </p>
        </div>

      </ContentColumn>
    </div>
  );
}
