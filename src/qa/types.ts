/**
 * Questions & answers (route /#/qa). Each entry is one file in ./entries,
 * `NNN-<id>.tsx`, that exports `meta` and default-exports the answer body.
 * The project skill .claude/skills/rlquestion/SKILL.md writes new entries.
 */
import type { ComponentType } from 'react';

export interface QAMeta {
  /** URL slug: /#/qa/<id>. Lowercase words joined by dashes. */
  id: string;
  /** Short headline for the page title and the list, phrased as a question. */
  title: string;
  /** The question as the reader asked it (lightly cleaned: typos, not meaning). */
  question: string;
  /** Plain-text short answer, 1-3 sentences; the page shows it above the full answer. */
  short: string;
  /** ISO date the question was asked (YYYY-MM-DD). */
  asked: string;
  /** Chapter slugs (src/chapters/registry.ts) that cover the topic. */
  chapters: string[];
  /** What the answer was checked against, e.g. 'rustler/src/fly.rs @2a899c8'. */
  checked: string[];
}

export interface QAEntry extends QAMeta {
  /** 1-based, from the file name's NNN prefix. */
  number: number;
  Answer: ComponentType;
}
