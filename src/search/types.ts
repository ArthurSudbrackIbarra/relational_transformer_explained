/**
 * Site search index (public/search-index.json, written by scripts/search-index.mjs). One entry
 * per page; each page holds its sections (the header first, id ''), and each section the text
 * blocks a reader sees, in page order.
 */

export interface SearchSection {
  /** The Section's anchor id (`?section=`); '' for the page header (title, lede, question). */
  id: string;
  title: string;
  /** Paragraphs, list items, captions… as rendered, whitespace collapsed. */
  blocks: string[];
}

export interface SearchPage {
  /** Route, e.g. '/tasks' or '/qa/where-task-tables-come-from'. */
  path: string;
  kind: 'chapter' | 'qa';
  /** 'Chapter 3' or 'Question 1'. */
  label: string;
  title: string;
  sections: SearchSection[];
}

export interface SearchIndex {
  version: 1;
  pages: SearchPage[];
}

/** Router state a search result passes to the page it opens (read by landing.ts). */
export interface FindState {
  terms: string[];
  phrase: string | null;
  /** Index into the section's blocks of the best match; -1 when only the title matched. */
  block: number;
  /** That block's indexed text: layouts differ by width (tables become cards), so the page looks it up by text first. */
  text: string | null;
}
