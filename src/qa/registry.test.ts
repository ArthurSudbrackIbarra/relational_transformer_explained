/** Q&A entries are well-formed: unique ids and numbers, known chapters, dates, non-empty text. */
import { describe, expect, it } from 'vitest';
import { getChapter } from '@/chapters/registry';
import { QA_ENTRIES } from './registry';

describe('Q&A entries', () => {
  it('has at least one entry', () => {
    expect(QA_ENTRIES.length).toBeGreaterThan(0);
  });

  it('uses unique ids and numbers 1..n', () => {
    expect(new Set(QA_ENTRIES.map((q) => q.id)).size).toBe(QA_ENTRIES.length);
    expect(QA_ENTRIES.map((q) => q.number)).toEqual(QA_ENTRIES.map((_, i) => i + 1));
  });

  it.each(QA_ENTRIES.map((q) => [q.id, q] as const))('%s is well-formed', (_, q) => {
    expect(q.id).toMatch(/^[a-z0-9]+(?:-[a-z0-9]+)*$/);
    expect(q.asked).toMatch(/^\d{4}-\d{2}-\d{2}$/);
    expect(q.title.trim().length).toBeGreaterThan(0);
    const questions = typeof q.question === 'string' ? [q.question] : q.question;
    expect(questions.length).toBeGreaterThan(0);
    for (const text of questions) expect(text.trim().length).toBeGreaterThan(0);
    expect(q.short.trim().length).toBeGreaterThan(0);
    expect(q.checked.length).toBeGreaterThan(0);
    for (const slug of q.chapters) expect(getChapter(slug), `unknown chapter "${slug}"`).toBeDefined();
  });
});
