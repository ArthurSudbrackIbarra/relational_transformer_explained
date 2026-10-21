/**
 * Paper Algorithm 1 (p.15, transcribed) next to the released loop (fly.rs:357-511, condensed as in
 * spec §4.3). The lines of the current replay step are highlighted in both columns.
 */
import { memo } from 'react';
import { Cite, cx } from '@/components';
import { ALG1, CODE_LINES } from './narrate';
import { useNarration } from './useNarration';

export function renderAlgText(text: string) {
  // light typesetting: keywords bold, subscripts for N_F→P / N_P→F / argmin_{…}
  const parts = text.split(/(\bwhile\b|\bdo\b|\bif\b|\bthen\b|\belse\b|\bcontinue\b|\breturn\b|N_F→P|N_P→F|argmin_\{r∈F\}|RandomSelect|RandomSample|HopDistance)/g);
  return parts.map((p, i) => {
    if (['while', 'do', 'if', 'then', 'else', 'continue', 'return'].includes(p))
      return (
        <strong key={i} className="font-semibold text-ink">
          {p}
        </strong>
      );
    if (p === 'N_F→P' || p === 'N_P→F')
      return (
        <span key={i}>
          𝒩<sub className="text-[13px]">{p.slice(2)}</sub>
        </span>
      );
    if (p === 'argmin_{r∈F}')
      return (
        <span key={i}>
          argmin<sub className="text-[13px]">r∈F</sub>
        </span>
      );
    if (p === 'RandomSelect' || p === 'RandomSample' || p === 'HopDistance')
      return (
        <span key={i} className="[font-variant:small-caps]">
          {p}
        </span>
      );
    return <span key={i}>{p}</span>;
  });
}

const AlgLine = memo(function AlgLine({ i, active }: { i: number; active: boolean }) {
  const l = ALG1[i];
  return (
    <li aria-current={active ? 'step' : undefined} className={cx('flex gap-3 rounded-[5px] pr-2 transition-colors', active ? 'bg-accent-soft text-ink' : 'text-ink-2')}>
      <span className="w-6 shrink-0 text-right text-[13.5px] text-muted select-none">{i + 1}</span>
      <span style={{ paddingLeft: `${l.indent * 1.4}em` }} className={cx('min-w-0', l.text.startsWith('/*') && 'text-muted italic')}>
        {renderAlgText(l.text)}
        {l.comment && <span className="ml-3 inline-block text-[14px] text-muted">// {l.comment}</span>}
      </span>
    </li>
  );
});

const CodeLine = memo(function CodeLine({ i, active }: { i: number; active: boolean }) {
  const l = CODE_LINES[i];
  return (
    <li aria-current={active ? 'step' : undefined} className={cx('flex gap-3 rounded-[5px] pr-2 transition-colors', active ? 'bg-accent-soft text-ink' : 'text-ink-2')}>
      <span className="w-[54px] shrink-0 text-right text-[13.5px] text-muted select-none">{l.lines}</span>
      <span style={{ paddingLeft: `${l.indent * 1.4}em` }} className="min-w-0">
        {l.text}
      </span>
    </li>
  );
});

const PaperHeader = memo(function PaperHeader() {
  return (
    <>
      <figcaption className="mb-2 flex flex-wrap items-baseline justify-between gap-2">
        <span className="text-[16.5px] font-semibold text-ink">
          Paper: Algorithm 1 <Cite kind="paper" ref="p.15 Alg. 1" />
        </span>
        <span className="text-[14px] text-muted">as printed</span>
      </figcaption>
      <div className="mb-2 rounded-md bg-surface-2 px-2.5 py-1.5 text-[14.5px] leading-snug text-ink-2">
        <strong className="font-semibold text-ink">Input:</strong> seed row s, context length L, width bound w, and, for each row r: non-missing feature cells
        C(r), P→F neighbors 𝒩<sub className="text-[13px]">P→F</sub>(r), F→P neighbors 𝒩<sub className="text-[13px]">F→P</sub>(r) and timestamp 𝒯(r). <strong className="font-semibold text-ink">Output:</strong> the set
        of database cells C in the context window.
      </div>
    </>
  );
});

const CodeHeader = memo(function CodeHeader() {
  return (
    <figcaption className="mb-2 flex flex-wrap items-baseline justify-between gap-2">
      <span className="text-[16.5px] font-semibold text-ink">
        Code: the loop in fly.rs <Cite kind="code" ref="rustler/src/fly.rs:357-511" />
      </span>
      <span className="text-[14px] text-muted">condensed, spec §4.3</span>
    </figcaption>
  );
});

const CodeFooter = memo(function CodeFooter() {
  return (
    <p className="mt-2 text-[14.5px] leading-snug text-ink-2">
      Line numbers are fly.rs lines. The verbatim Rust is in the three excerpts at the end of the chapter. One harmless reordering: the code expands the
      frontier before it emits the row’s cells; this only changes when random numbers are drawn <Cite kind="spec" ref="§4.8" />.
    </p>
  );
});

export function AlgorithmPanel() {
  const n = useNarration();
  const alg = new Set(n.alg);
  const code = new Set(n.code);
  return (
    <div className="wide my-8 grid min-w-0 gap-4 lg:grid-cols-2">
      <figure className="card min-w-0 px-4 py-4">
        <PaperHeader />
        <ol className="m-0 list-none p-0 font-mono text-[15px] leading-[1.75]" aria-label="Algorithm 1">
          {ALG1.map((l, i) => (
            <AlgLine key={l.id} i={i} active={alg.has(l.id)} />
          ))}
        </ol>
      </figure>
      <figure className="card min-w-0 px-4 py-4">
        <CodeHeader />
        <ol className="m-0 list-none p-0 font-mono text-[15px] leading-[1.75]" aria-label="fly.rs loop, condensed">
          {CODE_LINES.map((l, i) => (
            <CodeLine key={l.id} i={i} active={code.has(l.id)} />
          ))}
        </ol>
        <CodeFooter />
      </figure>
    </div>
  );
}
