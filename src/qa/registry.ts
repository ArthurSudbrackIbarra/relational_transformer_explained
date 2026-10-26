/**
 * Q&A registry: every ./entries/NNN-<id>.tsx file, ordered by NNN. Adding a
 * file is all it takes; there is no list to update.
 */
import type { ComponentType } from 'react';
import type { QAEntry, QAMeta } from './types';

const modules = import.meta.glob<{ meta: QAMeta; default: ComponentType }>('./entries/*.tsx', { eager: true });

export const QA_ENTRIES: QAEntry[] = Object.entries(modules)
  .map(([path, mod]) => {
    const m = /\/(\d+)-[^/]+\.tsx$/.exec(path);
    if (!m) throw new Error(`Q&A entry ${path}: file name must be NNN-<id>.tsx`);
    return { ...mod.meta, number: Number(m[1]), Answer: mod.default };
  })
  .sort((a, b) => a.number - b.number);

export function getQA(id: string | undefined): QAEntry | undefined {
  return QA_ENTRIES.find((q) => q.id === id);
}

/** The entry's questions as a list (one item unless the prompt asked several related ones). */
export function questionsOf(entry: QAMeta): string[] {
  return typeof entry.question === 'string' ? [entry.question] : entry.question;
}
