/**
 * Course map: every chapter from the registry, grouped by part, each tagged with the pipeline stage
 * of the "RT in one picture" figure that it opens up.
 */
import { Boxes, Database, FlaskConical, Gauge, MapPin, Network, Table2, Waypoints, type LucideIcon } from 'lucide-react';
import { Link } from 'react-router-dom';
import { CHAPTERS, PARTS, prefetchChapter } from '@/chapters/registry';
import { cx } from '@/components/ui/cx';
import { useSim } from '@/state/store';

const STAGE: Record<string, { label: string; icon: LucideIcon }> = {
  overview: { label: 'all stages', icon: MapPin },
  database: { label: 'Database', icon: Database },
  tasks: { label: 'Task row', icon: Table2 },
  preprocessing: { label: 'Tokens', icon: Boxes },
  semantics: { label: 'Tokens', icon: Boxes },
  sampling: { label: 'Context', icon: Waypoints },
  tokens: { label: 'Tokens', icon: Boxes },
  embedding: { label: 'Tokens', icon: Boxes },
  masks: { label: 'Attention', icon: Network },
  attention: { label: 'Attention', icon: Network },
  block: { label: 'Attention', icon: Network },
  decoding: { label: 'Prediction', icon: Gauge },
};

export function CourseMap({ current }: { current: string }) {
  const visited = useSim((s) => s.visited);
  return (
    <div className="wide mt-6 grid gap-8">
      {PARTS.map((part) => (
        <div key={part.id} className="grid gap-3 lg:grid-cols-[200px_minmax(0,1fr)]">
          <div>
            <div className="text-[17.5px] font-semibold text-ink">{part.title}</div>
            <div className="mt-1 text-[15.5px] leading-snug text-ink-2">{part.blurb}</div>
          </div>
          <ol className="grid gap-2.5 !p-0 sm:grid-cols-2" style={{ listStyle: 'none' }}>
            {CHAPTERS.filter((c) => c.part === part.id).map((c) => {
              const st = STAGE[c.slug] ?? {
                label: 'beyond the pipeline',
                icon: FlaskConical,
              };
              const Icon = st.icon;
              const here = c.slug === current;
              return (
                <li key={c.slug} className="!m-0 !p-0">
                  <Link
                    to={`/${c.slug}`}
                    data-plain
                    onPointerEnter={() => prefetchChapter(c.slug)}
                    aria-current={here ? 'page' : undefined}
                    className={cx(
                      'group flex h-full gap-3 rounded-xl border px-4 py-3 transition-colors',
                      here ? 'border-accent bg-accent-soft' : 'border-border bg-surface hover:border-accent',
                    )}
                  >
                    <span className="num w-6 shrink-0 pt-[1px] text-[17px] font-semibold text-muted group-hover:text-accent-text">
                      {String(c.number).padStart(2, '0')}
                    </span>
                    <span className="min-w-0 flex-1">
                      <span className="block text-[17px] leading-snug font-semibold text-ink">{c.title}</span>
                      <span className="mt-0.5 block text-[15px] leading-snug text-ink-2">{c.summary}</span>
                      <span className="mt-1.5 flex flex-wrap items-center gap-x-3 gap-y-1 text-[14px] text-ink-2">
                        <span className="inline-flex items-center gap-1.5">
                          <Icon aria-hidden="true" size={14} />
                          {st.label}
                        </span>
                        {here ? (
                          <span className="font-medium text-accent-text">you are here</span>
                        ) : visited[c.slug] ? (
                          <span className="text-muted">visited</span>
                        ) : null}
                      </span>
                    </span>
                  </Link>
                </li>
              );
            })}
          </ol>
        </div>
      ))}
    </div>
  );
}
