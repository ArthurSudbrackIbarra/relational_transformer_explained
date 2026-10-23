---
name: rlquestion
description: Answer a question about the Relational Transformer and publish it on the site's Questions & answers page (/#/qa). Use when the user runs /rlquestion <question> or asks to add a question to the site's Q&A. The answer is researched and verified against the paper-era official code (snap-stanford/relational-transformer @2a899c8) and the paper, written as a new src/qa/entries/NNN-<id>.tsx file in the site's chapter style, checked, committed and pushed.
argument-hint: <your question about the Relational Transformer>
---

# /rlquestion: add a verified question and answer to the site

The question is: **$ARGUMENTS**

If that is empty, ask the user for the question and stop.

Your job: answer this question **truthfully**, verified against the real RT code and paper. Write the answer as a new
Q&A entry in this repository, in the same style as the chapters. Check that it builds and renders, then commit and push it.
Finally, give the user the short answer in chat.

## 1. Ground truth: what counts as "true"

- **Code**: the paper-era official code, `snap-stanford/relational-transformer` at commit **`2a899c8`**. That is the
  site's ground truth (`src/components/prose/sources.ts` `CODE`). Current `main` is the later "RT-J" line. Do not describe it
  as RT. Use it only if the question asks about later versions, and then label it as such.
- **Paper**: arXiv 2510.06377v3 (ICLR 2026). Cite printed page numbers; for this PDF, PDF page = printed page.
- **RelBench** (task tables, splits, metrics): `snap-stanford/relbench` at tag **`v1.1.0`**.
- **The spec**: `research/RT_TECHNICAL_SPEC.md`. It is a cited summary of all of the above, so read the relevant sections first.
  Still open the code and paper it cites and confirm each claim you use. The spec is a map, not the territory.
- **This site's engine** (`src/engine`): a faithful TS port, at a smaller scale (d=64, 4 heads, 4 blocks, L=128, w=16). Use it
  for live figures and for "on this site" statements. Do not use it as evidence of what the official code does.

Get local copies in the scratchpad directory (not in the repo):

```bash
git clone -q https://github.com/snap-stanford/relational-transformer.git <scratch>/rt && git -C <scratch>/rt checkout -q 2a899c8
git clone -q --depth 1 --branch v1.1.0 https://github.com/snap-stanford/relbench.git <scratch>/relbench   # only if needed
curl -sL -o <scratch>/rt.pdf https://arxiv.org/pdf/2510.06377v3 && pdftotext -layout <scratch>/rt.pdf <scratch>/rt.txt
```

(The repo root may also hold a local `Relational_Transformer.pdf`. It is gitignored and identical to v3.)

## 2. Research and verify

1. Restate the question to yourself. Find what the user actually wants to know, including any misconception built into it.
   Answer that directly. If the question assumes something false, say so plainly and kindly, and explain what is true.
2. Find the relevant spec sections (`grep -n "^#" research/RT_TECHNICAL_SPEC.md`), then open the cited code lines and paper
   pages and read them yourself.
3. For every claim you plan to write, have one of: code lines at 2a899c8, a paper page/table, a RelBench line at v1.1.0,
   or a spec section that you re-checked. If you cannot verify a claim, leave it out or label it as unknown.
4. If you infer something the code implies but nobody documents, write that it is an inference
   (e.g. "This is what the code implies, not a documented workflow"). Where the paper and the code differ, say so and show both
   (`<Callout variant="paper-vs-code" paper=… code=… verdict=…>`). The site follows the code.
5. When a number matters, check it: re-read the table in the paper, or run a small script against the checkout or the engine
   (`npm test` style, or a scratch script). Never estimate a number and present it as the paper's.

## 3. Write the entry

Create `src/qa/entries/NNN-<id>.tsx`:

- `NNN` is the next number: `ls src/qa/entries`, take the highest and add 1, zero-padded to 3 digits.
- `<id>` is a short kebab-case slug of the topic. It must equal `meta.id`.

The registry (`src/qa/registry.ts`) picks the file up automatically. The page (`src/qa/QAPages.tsx`) renders the title,
the question, the short answer, the related chapters, your `Answer` component and the "Checked against" list.
Read `src/qa/entries/001-where-task-tables-come-from.tsx` as the reference example before writing.

```tsx
/**
 * Q&A NNN: <one line on what it answers>.
 * Checked against relational-transformer @2a899c8 (and RelBench v1.1.0 / the paper, as used).
 */
import { Link } from 'react-router-dom';
import { Callout, Cite, CodeRef, Figure, Section, Term } from '@/components';
import type { QAMeta } from '../types';

export const meta: QAMeta = {
  id: '<id>',
  title: '<short headline, phrased as a question>',
  question: '<the question as asked; fix typos only, never the meaning>',
  short: '<1-3 plain sentences that answer it directly>',
  asked: 'YYYY-MM-DD', // today
  chapters: ['<slug>', '<slug>'], // from src/chapters/registry.ts
  checked: ['relational-transformer @2a899c8: <files>', 'Paper (arXiv 2510.06377v3): <pages>', 'research/RT_TECHNICAL_SPEC.md <sections>'],
};

// GENERATED verbatim by scripts/snippet.mjs. Do not edit.
const SOME_SNIPPET = { file: '…', start: 1, end: 2, code: '…' };

export default function Answer() {
  return (
    <>
      <Section id="<anchor>" title="<sentence-case heading>">
        <p>… <Cite kind="code" ref="rustler/src/fly.rs:415-424" /> …</p>
        <CodeRef title="<what the excerpt shows>" file={SOME_SNIPPET.file} lines={[SOME_SNIPPET.start, SOME_SNIPPET.end]} code={SOME_SNIPPET.code} />
      </Section>
    </>
  );
}
```

Rules for the content:

- **Structure**: 2-6 `<Section>`s. Each one answers one sub-question, and its heading says the point in a sentence.
  The first section addresses the core of the question. `short` must stand alone, since the list page shows it.
- **Citations**: every factual sentence ends with a `<Cite>`. Use `kind="code"` with `file:start-end` at 2a899c8,
  `kind="paper"` with `"p.N §X"` or `"p.N Table K"`, `kind="spec"` with `"§N.M"`, and `kind="url"` for anything else.
- **Code excerpts**: never type code by hand. Generate each one with
  `node scripts/snippet.mjs <checkout> <file> <start> <end> NAME` and paste the output verbatim near the top of the file.
  Excerpts that already exist can be imported, e.g. `src/chapters/tasks/codeSnippets.ts`, `src/chapters/sampling/code.ts`
  and `src/chapters/results/snippets.ts`. For code outside the RT repo (e.g. RelBench), pass
  `source={{ repo: 'snap-stanford/relbench', ref: 'v1.1.0', url: '<permalink with #Lstart-Lend>' }}` to `CodeRef`.
- **Figures**: reuse a chapter's figure when it shows the point (import it from `src/chapters/<slug>/…`, as entry 001 does
  with `ChurnBuilder`). Live figures must read the engine or `src/state/hooks.ts`, never invented numbers.
  Only build a new figure if it clearly helps. Follow the dataviz rules in `src/styles/index.css`: tokens, no colour-only
  encodings, labels on marks.
- **Terms and links**: use `<Term id=…>` for glossary words (ids in `src/components/glossary.ts`). Link chapters with
  `<Link to="/<slug>">chapter N</Link>`, or `?section=<id>` for a section.
- **Voice**: same as the chapters. Plain, precise, short sentences, no hype, no filler. Explain the "why" where it helps.
  Use mostly prose; lists only for real enumerations.
- **Readability (the user's standing rule)**: never set small text. Floors: prose 18px, UI 15px, captions 14px, chips 13px,
  SVG labels 12px. When something does not fit, make room instead of shrinking.

## 4. Check it

Run these with `export PATH=$HOME/.nvm/versions/node/v24.15.0/bin:$PATH` if `npm` is missing.

```bash
npm run typecheck && npm test && npm run build
npx vite preview --port 5311 --strictPort        # run in the background
node scripts/shoot.mjs --port 5311 --routes qa,qa/<id> --out <scratch>/qa-shots --themes light,dark --widths 1440,390
```

`shoot.mjs` must print `ok` for every shot: no console errors, no horizontal overflow. Look at the screenshots
(Read the PNGs, cropping long ones) and fix anything broken or cramped.

Then do a final citation pass. For each `<Cite kind="code">`, open those exact lines in the checkout and confirm they show what
your sentence says. For each paper cite, confirm the page. Fix or remove anything that does not hold.

## 5. Publish

```bash
git add src/qa/entries/NNN-<id>.tsx   # plus any new shared code you added
git commit -m "Q&A: <title>" -m "Co-Authored-By: <the attribution line from the system reminder>"
git push origin main
```

The user has authorized commits and pushes to this repository's `origin/main`. The GitHub Pages workflow deploys on push.

## 6. Report

Reply in chat with:

- the short answer;
- the page path `/#/qa/<id>`;
- any caveat (an inference you flagged, or a point the code and paper leave open).

Keep it brief: the full answer lives on the site.
