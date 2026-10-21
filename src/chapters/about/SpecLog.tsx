/**
 * The technical specification (research/RT_TECHNICAL_SPEC.md) as a verified artifact: live counts read
 * from the spec's own text, and its §14 adversarial verification log as a filterable list.
 */
import { useMemo, useState } from 'react';
import type { ReactNode } from 'react';
import { Segmented, Stat, StatusBadge, cx } from '@/components';
import { useRawSource } from './data';

/** Render `code` spans and **bold** of a markdown table cell. */
export function mdInline(s: string): ReactNode[] {
  const out: ReactNode[] = [];
  const re = /`([^`]+)`|\*\*([^*]+)\*\*/g;
  let last = 0;
  let k = 0;
  for (const m of s.matchAll(re)) {
    const i = m.index ?? 0;
    if (i > last) out.push(s.slice(last, i));
    if (m[1] !== undefined)
      out.push(
        <code key={k++} className="!text-[14px] !whitespace-normal">
          {m[1]}
        </code>,
      );
    else out.push(<strong key={k++}>{m[2]}</strong>);
    last = i + m[0].length;
  }
  if (last < s.length) out.push(s.slice(last));
  return out;
}

/** The text of one "## N." section (up to the next "## "). */
function section(spec: string, n: number): string {
  const lines = spec.split('\n');
  const s = lines.findIndex((l) => new RegExp(`^## ${n}\\. `).test(l));
  if (s < 0) return '';
  let e = lines.length;
  for (let i = s + 1; i < lines.length; i++) if (/^## /.test(lines[i])) (e = i), (i = lines.length);
  return lines.slice(s, e).join('\n');
}

export interface SpecFacts {
  sections: number;
  subsections: number;
  codeCites: number;
  paperCites: number;
  verifiedLocally: number;
  discrepancies: number;
  openQuestions: number;
  resolved: number;
  log: { n: number; lens: string; section: string; verdict: string; reason: string }[];
}

export function specFacts(spec: string): SpecFacts {
  const s10 = section(spec, 10);
  const s13 = section(spec, 13);
  const s14 = section(spec, 14);
  const open = s13.split('\n').filter((l) => /^\d+\. /.test(l));
  const log = s14
    .split('\n')
    .map((l) => /^\| (\d+) \| ([^|]+) \| ([^|]+) \| ([^|]+) \| (.+) \|\s*$/.exec(l))
    .filter((m): m is RegExpExecArray => !!m)
    .map((m) => ({ n: Number(m[1]), lens: m[2].trim(), section: m[3].trim(), verdict: m[4].trim(), reason: m[5].trim() }));
  return {
    sections: spec.split('\n').filter((l) => /^## \d+\. /.test(l)).length,
    subsections: spec.split('\n').filter((l) => /^### /.test(l)).length,
    codeCites: (spec.match(/\b[\w-]+\.(?:py|rs|toml|lock):\d+/g) ?? []).length,
    paperCites: (spec.match(/\bp\.\d+/g) ?? []).length,
    verifiedLocally: (spec.match(/\[verified locally/g) ?? []).length,
    discrepancies: s10.split('\n').filter((l) => /^\| \d+ \|/.test(l)).length,
    openQuestions: open.length,
    resolved: open.filter((l) => l.includes('~~')).length,
    log,
  };
}

export function useSpecFacts(): { facts: SpecFacts | null; error: string | null } {
  const { data, error } = useRawSource('spec');
  const facts = useMemo(() => (data ? specFacts(data) : null), [data]);
  return { facts, error };
}

export function SpecStats() {
  const { facts, error } = useSpecFacts();
  if (error) return <StatusBadge status="warning">The spec could not be loaded: {error}</StatusBadge>;
  if (!facts) return <p className="text-[16px] text-ink-2">Reading the spec…</p>;
  return (
    <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
      <Stat label="Sections" value={facts.sections} caption={`${facts.subsections} subsections`} />
      <Stat label="Citations of code lines" value={facts.codeCites.toLocaleString('en-US')} caption="file.py:line or file.rs:line" />
      <Stat label="Citations of paper pages" value={facts.paperCites.toLocaleString('en-US')} caption="“p.N” references" />
      <Stat label="Paper-vs-code rows" value={facts.discrepancies} caption={`§10; plus ${facts.openQuestions} open questions in §13`} />
    </div>
  );
}

const VERDICT_TONE = (v: string) => (v.includes('major') ? 'major' : v.includes('amended') ? 'amended' : v.startsWith('ACCEPTED') ? 'accepted' : 'other');

export function VerificationLog() {
  const { facts, error } = useSpecFacts();
  const [lens, setLens] = useState<string>('all');
  const lenses = useMemo(() => (facts ? ['all', ...new Set(facts.log.map((r) => r.lens))] : ['all']), [facts]);
  if (error) return <StatusBadge status="warning">The spec could not be loaded: {error}</StatusBadge>;
  if (!facts) return <p className="text-[16px] text-ink-2">Reading the verification log…</p>;
  const rows = facts.log.filter((r) => lens === 'all' || r.lens === lens);
  const accepted = facts.log.filter((r) => r.verdict.startsWith('ACCEPTED')).length;
  const rejected = facts.log.filter((r) => r.verdict.startsWith('REJECTED')).length;
  const amended = facts.log.filter((r) => /amended/.test(r.verdict)).length;
  return (
    <div className="min-w-0">
      <div className="flex flex-wrap items-end justify-between gap-x-4 gap-y-3">
        <Segmented
          label="Verifier lens"
          value={lens}
          onChange={setLens}
          options={lenses.map((l) => ({ value: l, label: l === 'all' ? `all (${facts.log.length})` : `${l} (${facts.log.filter((r) => r.lens === l).length})` }))}
        />
        <div className="flex flex-wrap gap-2">
          <StatusBadge status="good">{accepted} accepted</StatusBadge>
          <StatusBadge status={rejected ? 'warning' : 'info'}>{rejected} rejected</StatusBadge>
          <StatusBadge status="info">{amended} with an amendment</StatusBadge>
        </div>
      </div>
      <ol className="scroll-quiet mt-4 grid max-h-[560px] gap-0 overflow-y-auto !p-0 pr-1" style={{ listStyle: 'none' }}>
        {rows.map((r) => {
          const tone = VERDICT_TONE(r.verdict);
          return (
            <li key={r.n} className="!m-0 grid gap-x-3 border-b border-grid py-2.5 !pl-0 sm:grid-cols-[44px_minmax(0,1fr)]">
              <span className="num text-[15px] text-muted">#{r.n}</span>
              <div className="min-w-0">
                <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
                  <span className="rounded-full border border-border bg-surface-2 px-2 text-[13.5px] font-medium text-ink-2">{r.lens}</span>
                  <span className="text-[15px] text-ink-2">§{r.section}</span>
                  <span
                    className={cx(
                      'rounded-full border px-2 text-[13.5px] font-semibold',
                      tone === 'major' ? 'border-accent text-accent-text' : 'border-border text-ink',
                    )}
                  >
                    {r.verdict}
                  </span>
                </div>
                <p className="mt-1 mb-0 text-[15.5px] leading-snug text-ink-2">{mdInline(r.reason)}</p>
              </div>
            </li>
          );
        })}
      </ol>
    </div>
  );
}
