/**
 * Placeholder page for a chapter that has not been written yet. Shows the
 * registry title, summary and planned outline (BUILD_PLAN §6), plus the live
 * simulation knobs read from the store, so navigation and state can be tested
 * end to end. Chapter authors replace `src/chapters/<slug>/index.tsx`.
 */
import { useShallow } from 'zustand/react/shallow';
import { getChapter } from '@/chapters/registry';
import { selectSimParams, useSim } from '@/state/store';
import { Callout } from '../prose/Callout';
import { ScaleNote } from '../prose/ScaleNote';
import { ChapterHeader, ChapterNav, ChapterPage, Section } from './Chapter';

export function ChapterPlaceholder({ slug }: { slug: string }) {
  const meta = getChapter(slug);
  const sim = useSim(useShallow(selectSimParams));
  if (!meta) return null;
  return (
    <ChapterPage slug={slug}>
      <ChapterHeader slug={slug} lede={meta.summary} learn={meta.outline} />
      <Section id="status" title="This chapter is being written">
        <Callout variant="note" title="Coming together">
          <p>
            The interactive figures for this chapter are under construction. The outline above is the plan; every figure will
            read the live simulation below, so what you set here carries through the whole course.
          </p>
        </Callout>
        <dl className="card grid grid-cols-2 gap-x-6 gap-y-2 px-5 py-4 text-[16.5px] sm:grid-cols-4">
          {(
            [
              ['Database', sim.dbName],
              ['Task', sim.taskId.split('/')[1] ?? sim.taskId],
              ['Split', sim.split],
              ['Seed', sim.seedNode === null ? 'auto' : `#${sim.seedNode}`],
              ['L', String(sim.L)],
              ['w', String(sim.w)],
              ['RNG seed', String(sim.rngSeed)],
              ['Checkpoint', sim.ckpt],
            ] as const
          ).map(([k, v]) => (
            <div key={k} className="min-w-0">
              <dt className="text-[14.5px] text-muted">{k}</dt>
              <dd className="num truncate text-ink">{v}</dd>
            </div>
          ))}
        </dl>
        <ScaleNote />
      </Section>
      <ChapterNav slug={slug} />
    </ChapterPage>
  );
}
