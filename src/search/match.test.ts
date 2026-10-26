/** Site search matching: word-start terms, accents and case, ranking, snippets. */
import { describe, expect, it } from 'vitest';
import { foldWithMap, markText, parseQuery, prepare, search, snippet } from './match';
import type { SearchIndex } from './types';

const INDEX: SearchIndex = {
  version: 1,
  pages: [
    {
      path: '/attention',
      kind: 'chapter',
      label: 'Chapter 10',
      title: 'Relational attention',
      sections: [
        { id: '', title: 'Relational attention', blocks: ['Four attention layers per block.'] },
        { id: 'column', title: 'Column attention', blocks: ['Cells of the same column attend to each other.', 'A self-attention variant.'] },
        { id: 'feature', title: 'Feature attention', blocks: ['Cells of one row see their foreign-key parents.'] },
      ],
    },
    {
      path: '/tasks',
      kind: 'chapter',
      label: 'Chapter 3',
      title: 'Predictive tasks and task tables',
      sections: [
        { id: '', title: 'Predictive tasks and task tables', blocks: ['Every prediction is a masked cell.'] },
        { id: 'build', title: 'Building a task table', blocks: ['The task table is built from timestamps, before the model sees it. Café rows too.'] },
      ],
    },
  ],
};
const pages = prepare(INDEX);

describe('parseQuery', () => {
  it('keeps terms of two or more letters and the phrase', () => {
    expect(parseQuery('  Task  Table ')).toEqual({ terms: ['task', 'table'], phrase: 'task table' });
    expect(parseQuery('attention')).toEqual({ terms: ['attention'], phrase: null });
  });
  it('rejects queries with nothing searchable', () => {
    expect(parseQuery('a')).toBeNull();
    expect(parseQuery('  ')).toBeNull();
  });
});

describe('search', () => {
  it('matches at word starts, so a prefix finds the word and a hyphenated compound', () => {
    const groups = search(pages, parseQuery('attent')!);
    expect(groups[0].page.path).toBe('/attention');
    const ids = groups[0].hits.map((h) => h.section.id);
    expect(ids).toContain('column');
    // "ention" is not a word start
    expect(search(pages, parseQuery('ention')!)).toEqual([]);
  });

  it('requires every term in the section and ranks title matches first', () => {
    const groups = search(pages, parseQuery('column cells')!);
    expect(groups).toHaveLength(1);
    expect(groups[0].hits.map((h) => h.section.id)).toEqual(['column']);
    const g = search(pages, parseQuery('attention')!)[0];
    expect(g.hits[0].section.id).toBe('column'); // title match + body match beats a title-only match below it
  });

  it('points at the block holding the phrase', () => {
    const [g] = search(pages, parseQuery('task table')!);
    const hit = g.hits.find((h) => h.section.id === 'build')!;
    expect(hit.block).toBe(0);
    expect(g.header?.block).toBe(-1); // header: title only
  });

  it('ignores case and accents', () => {
    const [g] = search(pages, parseQuery('CAFE')!);
    expect(g.hits[0].section.id).toBe('build');
  });
});

describe('snippets and marks', () => {
  it('marks every term occurrence in the raw text', () => {
    const segs = markText('Café and cafe', parseQuery('cafe')!);
    expect(segs.filter((s) => s.hit).map((s) => s.text)).toEqual(['Café', 'cafe']);
  });

  it('maps folded offsets back to the raw text', () => {
    const { text, map } = foldWithMap('Ça’s');
    expect(text).toBe("ca's");
    expect(map).toEqual([0, 1, 2, 3, 4]);
  });

  it('cuts a window around the first match with ellipses', () => {
    const long = `${'lorem ipsum '.repeat(20)}the masked cell is predicted ${'dolor sit '.repeat(30)}`;
    const segs = snippet(long, parseQuery('masked')!);
    expect(segs[0].text).toBe('…');
    expect(segs[segs.length - 1].text).toBe('…');
    expect(segs.some((s) => s.hit && s.text === 'masked')).toBe(true);
  });
});
