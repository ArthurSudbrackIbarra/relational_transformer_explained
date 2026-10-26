/**
 * Hook for scripts/search-index.mjs: the index builder opens every page of a built site in a
 * headless browser and calls window.__rtSearch to list the pages and read their text through
 * extract.ts (the same reader the post-navigation highlight uses).
 */
import { CHAPTERS } from '@/chapters/registry';
import { blockText, extractSections, pageRoot } from './extract';
import type { SearchPage, SearchSection } from './types';

declare global {
  interface Window {
    __rtSearch?: {
      pages: () => Promise<Omit<SearchPage, 'sections'>[]>;
      extract: () => SearchSection[] | null;
    };
  }
}

window.__rtSearch = {
  async pages() {
    const { QA_ENTRIES } = await import('@/qa/registry');
    return [
      ...CHAPTERS.map((c) => ({ path: `/${c.slug}`, kind: 'chapter' as const, label: `Chapter ${c.number}`, title: c.title })),
      ...QA_ENTRIES.map((e) => ({ path: `/qa/${e.id}`, kind: 'qa' as const, label: `Question ${e.number}`, title: e.title })),
    ];
  },
  extract() {
    const root = pageRoot();
    if (!root) return null;
    return extractSections(root).map((s) => ({
      id: s.id,
      title: s.heading ? blockText(s.heading) : '',
      blocks: s.blocks.map(blockText),
    }));
  },
};
