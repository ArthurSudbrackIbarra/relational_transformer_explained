/**
 * Quiz: 1–3 multiple-choice checks per chapter (BUILD_PLAN §6); the
 * explanation appears once the reader answers. Native radio inputs inside a
 * fieldset, so it is fully keyboard and screen-reader accessible.
 */
import { CircleCheck, CircleX, RotateCcw } from 'lucide-react';
import { useId, useState, type ReactNode } from 'react';
import { cx } from '../ui/cx';

export interface QuizChoice {
  text: ReactNode;
  correct?: boolean;
  /** Why this choice is right or wrong (shown when picked). */
  why?: ReactNode;
}

export interface QuizQuestion {
  prompt: ReactNode;
  choices: QuizChoice[];
  /** General explanation shown after any answer. */
  explanation?: ReactNode;
}

export interface QuizProps {
  questions: QuizQuestion[];
  title?: ReactNode;
  className?: string;
}

function Question({ q, n }: { q: QuizQuestion; n: number }) {
  const [picked, setPicked] = useState<number | null>(null);
  const name = useId();
  const answered = picked !== null;
  const right = answered && !!q.choices[picked!].correct;
  return (
    <fieldset className="min-w-0">
      <legend className="mb-2.5 text-[18px] leading-snug font-medium text-ink">
        <span className="mr-1.5 text-muted tabular-nums">{n}.</span>
        {q.prompt}
      </legend>
      <div className="flex flex-col gap-1.5">
        {q.choices.map((c, i) => {
          const isPicked = picked === i;
          const showCorrect = answered && c.correct;
          return (
            <label
              key={i}
              className={cx(
                'flex cursor-pointer items-start gap-2.5 rounded-lg border px-3 py-2 text-[17px] leading-snug transition-colors has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-accent',
                isPicked && c.correct && 'border-[var(--good)] bg-surface',
                isPicked && !c.correct && 'border-[var(--critical)] bg-surface',
                !isPicked && showCorrect && 'border-[var(--good)] border-dashed',
                !isPicked && !showCorrect && 'border-border bg-surface hover:bg-surface-2',
              )}
            >
              <input
                type="radio"
                name={name}
                checked={isPicked}
                onChange={() => setPicked(i)}
                className="mt-[3px] h-4 w-4 shrink-0 accent-[var(--accent)]"
              />
              <span className="min-w-0 flex-1 text-ink">
                {c.text}
                {isPicked && c.why && <span className="mt-1 block text-[16px] text-ink-2">{c.why}</span>}
              </span>
              {isPicked && (
                <span className="inline-flex shrink-0 items-center gap-1 text-[16px] font-medium text-ink">
                  {c.correct ? (
                    <>
                      <CircleCheck aria-hidden="true" size={16} style={{ color: 'var(--good)' }} /> Correct
                    </>
                  ) : (
                    <>
                      <CircleX aria-hidden="true" size={16} style={{ color: 'var(--critical)' }} /> Not quite
                    </>
                  )}
                </span>
              )}
            </label>
          );
        })}
      </div>
      <div aria-live="polite">
        {answered && (
          <div className="mt-3 rounded-lg bg-surface-2 px-3.5 py-3 text-[17px] leading-relaxed text-ink">
            {q.explanation}
            {!right && (
              <button
                type="button"
                onClick={() => setPicked(null)}
                className="mt-1.5 flex items-center gap-1.5 text-[16px] font-medium text-accent-text hover:underline"
              >
                <RotateCcw aria-hidden="true" size={15} /> Try again
              </button>
            )}
          </div>
        )}
      </div>
    </fieldset>
  );
}

export function Quiz({ questions, title = 'Check your understanding', className }: QuizProps) {
  return (
    <section className={cx('my-10 rounded-2xl border border-border bg-surface px-5 py-5 sm:px-6', className)} aria-label="Quiz">
      <div className="eyebrow mb-1">Quiz</div>
      <h3 className="mb-4 text-[20px] font-semibold tracking-tight text-ink">{title}</h3>
      <div className="flex flex-col gap-7">
        {questions.map((q, i) => (
          <Question key={i} q={q} n={i + 1} />
        ))}
      </div>
    </section>
  );
}
