/**
 * The final quiz: every question on one page, choices shuffled for each attempt, graded on submit.
 * A pass hands the answers to the page (which stores the result and shows the review); a fail shows
 * the score and the chapters behind the missed questions, but not the right answers, and offers a
 * new attempt with the choices shuffled again.
 */
import { CircleCheck, CircleX, RotateCcw } from 'lucide-react';
import { useId, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { getChapter } from '@/chapters/registry';
import { Button } from '@/components/ui/controls';
import { cx } from '@/components/ui/cx';
import { FINAL_QUIZ, gradeQuiz, passScore, type FinalQuestion } from './quiz';

interface Attempt {
  /** For each question, the order in which its choices are shown (indices into `choices`). */
  order: number[][];
  /** Picked choice per question (an index into `choices`), or null. */
  picked: (number | null)[];
}

function newAttempt(): Attempt {
  return { order: FINAL_QUIZ.map((q) => shuffle(q.choices.map((_, i) => i))), picked: FINAL_QUIZ.map(() => null) };
}

function shuffle(a: number[]): number[] {
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

export function FinalQuiz({ onPass }: { onPass: (score: number, total: number, picked: number[]) => void }) {
  const [attempt, setAttempt] = useState(newAttempt);
  const [failed, setFailed] = useState<number | null>(null);
  const top = useRef<HTMLElement | null>(null);
  const total = FINAL_QUIZ.length;
  const need = passScore(total);
  const answered = attempt.picked.filter((p) => p !== null).length;

  const submit = () => {
    const score = gradeQuiz(attempt.picked);
    if (score >= need) onPass(score, total, attempt.picked as number[]);
    else {
      setFailed(score);
      window.setTimeout(() => top.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 30);
    }
  };
  const retry = () => {
    setAttempt(newAttempt());
    setFailed(null);
    window.setTimeout(() => top.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 30);
  };
  const missed = failed === null ? [] : FINAL_QUIZ.filter((q, i) => attempt.picked[i] !== q.answer);

  return (
    <section aria-label="Final quiz" className="scroll-mt-[calc(var(--topbar-h)+24px)]" ref={top}>
      <div aria-live="polite">
        {failed !== null && (
          <div className="mb-8 rounded-2xl border border-[var(--critical)] bg-surface px-5 py-4">
            <p className="!m-0 flex items-center gap-2 text-[19px] font-semibold text-ink">
              <CircleX aria-hidden="true" size={20} style={{ color: 'var(--critical)' }} />
              {failed} of {total} right: {need} are needed to pass.
            </p>
            <p className="!mt-2 !mb-0 text-[17px] leading-relaxed text-ink-2">
              The questions marked below were not answered right. These chapters cover them:
            </p>
            <ul className="!mt-2 !mb-0 grid gap-1 !pl-5 text-[17px]">
              {missed.map((q) => (
                <li key={q.id}>
                  Question {FINAL_QUIZ.indexOf(q) + 1}: <ChapterLinks q={q} />.
                </li>
              ))}
            </ul>
            <Button variant="primary" className="mt-4" icon={<RotateCcw aria-hidden="true" size={16} />} onClick={retry}>
              Try again
            </Button>
          </div>
        )}
      </div>
      <ol className="!m-0 flex flex-col gap-8 !p-0" style={{ listStyle: 'none' }}>
        {FINAL_QUIZ.map((q, i) => (
          <li key={q.id} className="!m-0 !p-0">
            <Question
              q={q}
              n={i + 1}
              order={attempt.order[i]}
              picked={attempt.picked[i]}
              locked={failed !== null}
              wrong={failed !== null && attempt.picked[i] !== q.answer}
              onPick={(c) => setAttempt((a) => ({ ...a, picked: a.picked.map((p, j) => (j === i ? c : p)) }))}
            />
          </li>
        ))}
      </ol>
      {failed === null && (
        <div className="mt-10 flex flex-wrap items-center gap-x-4 gap-y-2 border-t border-grid pt-6">
          <Button variant="primary" onClick={submit} disabled={answered < total}>
            Submit answers
          </Button>
          <span className="text-[16px] text-ink-2 tabular-nums">
            {answered < total ? `${answered} of ${total} answered` : `All ${total} answered. ${need} right answers pass.`}
          </span>
        </div>
      )}
    </section>
  );
}

function Question({
  q,
  n,
  order,
  picked,
  locked,
  wrong,
  onPick,
}: {
  q: FinalQuestion;
  n: number;
  order: number[];
  picked: number | null;
  locked: boolean;
  wrong: boolean;
  onPick: (choice: number) => void;
}) {
  const name = useId();
  return (
    <fieldset className="min-w-0" disabled={locked}>
      <legend className="mb-2.5 text-[18px] leading-snug font-medium text-ink">
        <span className="mr-1.5 text-muted tabular-nums">{n}.</span>
        {q.prompt}
      </legend>
      <div className="flex flex-col gap-1.5">
        {order.map((c) => (
          <label
            key={c}
            className={cx(
              'flex items-start gap-2.5 rounded-lg border px-3 py-2 text-[17px] leading-snug transition-colors has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-accent',
              locked ? 'cursor-default' : 'cursor-pointer hover:bg-surface-2',
              picked === c ? (wrong ? 'border-[var(--critical)] bg-surface' : 'border-accent bg-accent-soft') : 'border-border bg-surface',
            )}
          >
            <input
              type="radio"
              name={name}
              checked={picked === c}
              onChange={() => onPick(c)}
              className="mt-[3px] h-4 w-4 shrink-0 accent-[var(--accent)]"
            />
            <span className="min-w-0 flex-1 text-ink">{q.choices[c]}</span>
          </label>
        ))}
      </div>
      {wrong && (
        <p className="!mt-2 !mb-0 flex items-start gap-1.5 text-[16px] text-ink-2">
          <CircleX aria-hidden="true" size={16} className="mt-[3px] shrink-0" style={{ color: 'var(--critical)' }} />
          <span>
            Not right. Review <ChapterLinks q={q} />.
          </span>
        </p>
      )}
    </fieldset>
  );
}

/** The passing attempt, question by question, with the right answer and why. */
export function QuizReview({ picked }: { picked: readonly number[] }) {
  return (
    <ol className="!m-0 flex flex-col gap-6 !p-0" style={{ listStyle: 'none' }}>
      {FINAL_QUIZ.map((q, i) => {
        const right = picked[i] === q.answer;
        return (
          <li key={q.id} className="!m-0 !p-0">
            <p className="!m-0 text-[18px] leading-snug font-medium text-ink">
              <span className="mr-1.5 text-muted tabular-nums">{i + 1}.</span>
              {q.prompt}
            </p>
            <p className="!mt-2 !mb-0 flex items-start gap-2 text-[17px] leading-snug text-ink">
              {right ? (
                <CircleCheck aria-label="right" size={17} className="mt-[3px] shrink-0" style={{ color: 'var(--good)' }} />
              ) : (
                <CircleX aria-label="not right" size={17} className="mt-[3px] shrink-0" style={{ color: 'var(--critical)' }} />
              )}
              <span>
                {q.choices[q.answer]}
                {!right && <span className="block text-[16px] text-ink-2">Your answer: {q.choices[picked[i]]}</span>}
              </span>
            </p>
            <p className="!mt-1.5 !mb-0 text-[16.5px] leading-relaxed text-ink-2">
              {q.why} See <ChapterLinks q={q} />.
            </p>
          </li>
        );
      })}
    </ol>
  );
}

function ChapterLinks({ q }: { q: FinalQuestion }) {
  return (
    <>
      {q.chapters.map((slug, i) => {
        const c = getChapter(slug)!;
        return (
          <span key={slug}>
            {i > 0 && ' and '}
            <Link to={`/${slug}`}>
              chapter {c.number}, {c.short}
            </Link>
          </span>
        );
      })}
    </>
  );
}
