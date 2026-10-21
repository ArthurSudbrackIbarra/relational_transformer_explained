/**
 * The independent sampler cross-check (python/sampler_dump.ts + python/sampler_crosscheck.py), as reported in
 * verification.json, and the "run it yourself" command blocks (package.json scripts and the
 * "Reproduce everything" block of python/README.md, both read from the repository).
 */
import { CircleCheck } from 'lucide-react';
import { StatusBadge } from '@/components';
import { DEFAULT_L, DEFAULT_W, PAPER_L, PAPER_W } from '@/engine/model';
import pkg from '../../../package.json';
import { int, useRawSource, useVerification } from './data';
import { SiteCode } from './SiteCode';

export function SamplerCrossCheck() {
  const { data: v, error } = useVerification();
  if (error) return <StatusBadge status="warning">verification.json could not be loaded: {error}</StatusBadge>;
  if (!v) return <p className="text-[16px] text-ink-2">Loading the verification report…</p>;
  const sc = v.details.samplerCrosscheck;
  const configs = [...sc.how.matchAll(/\((\d+),\s*(\d+)\)/g)].map((m) => ({ L: Number(m[1]), w: Number(m[2]) }));
  const note = (L: number, w: number) =>
    L === DEFAULT_L && w === DEFAULT_W
      ? 'this site’s default'
      : L === PAPER_L && w === PAPER_W
        ? 'the paper’s setting'
        : w === 1
          ? 'width 1: almost every row subsampled'
          : w <= 4
            ? 'narrow width: frequent subsampling'
            : L < DEFAULT_L
              ? 'short context: early hard stop'
              : 'long context';
  return (
    <div className="min-w-0">
      <div className="flex flex-wrap items-center gap-2">
        <StatusBadge status={sc.mismatches === 0 ? 'good' : 'critical'}>
          {int(sc.contexts)} contexts rebuilt, {int(sc.mismatches)} mismatches
        </StatusBadge>
      </div>
      <div className="mt-4 grid grid-cols-1 gap-3 min-[480px]:grid-cols-2 lg:grid-cols-4">
        {configs.map((c) => (
          <div key={`${c.L}-${c.w}`} className="rounded-lg border border-border bg-surface px-3.5 py-3">
            <div className="flex items-center gap-1.5 text-[14px] font-medium text-ink-2">
              <CircleCheck aria-hidden="true" size={15} style={{ color: 'var(--good)' }} /> matched
            </div>
            <div className="num mt-1 text-[22px] leading-tight font-semibold whitespace-nowrap text-ink">
              L {c.L}, w {c.w}
            </div>
            <div className="mt-1 text-[15px] leading-snug text-ink-2">{note(c.L, c.w)}</div>
          </div>
        ))}
      </div>
      <p className="mt-4 mb-0 text-[15.5px] leading-snug text-ink-2">
        <span className="font-semibold text-ink">How:</span> {sc.how}
      </p>
    </div>
  );
}

export function RunCommands() {
  const scripts = (pkg as { scripts: Record<string, string> }).scripts;
  const lines = [
    'npm install',
    ...(['dev', 'test', 'build'] as const)
      .filter((k) => scripts[k])
      .map((k) => `${k === 'test' ? 'npm test' : `npm run ${k}`}`.padEnd(17) + `# ${scripts[k]}`),
  ];
  const { data: readme } = useRawSource('pythonReadme');
  let block: { code: string; start: number } | null = null;
  if (readme) {
    const ls = readme.split('\n');
    const h = ls.findIndex((l) => /^## Reproduce everything/.test(l));
    const s = h >= 0 ? ls.findIndex((l, i) => i > h && /^```bash/.test(l)) : -1;
    const e = s >= 0 ? ls.findIndex((l, i) => i > s && /^```\s*$/.test(l)) : -1;
    if (s >= 0 && e > s) block = { code: ls.slice(s + 1, e).join('\n'), start: s + 2 };
  }
  return (
    <>
      <SiteCode
        static
        title="Run the site"
        file="package.json → scripts"
        source="this site’s commands"
        language="bash"
        code={lines.join('\n')}
        note={<>Needs Node and npm (the site was built with Node 24). The dev server prints a local URL; open it and pick a chapter.</>}
      />
      <SiteCode
        title="Rebuild every file in public/data: embeddings, training, verification"
        file="python/README.md"
        start={block?.start}
        source="this site’s commands"
        language="bash"
        code={readme ? (block?.code ?? '# "Reproduce everything" block not found in python/README.md') : null}
        note={
          <>
            The steps of the offline half, in order, read from the README. They need the Python environment it describes (PyTorch and
            sentence-transformers on CPU); the README also lists how long each step took.
          </>
        }
      />
    </>
  );
}
