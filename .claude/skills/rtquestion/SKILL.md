---
name: rtquestion
description: Answer one or more questions about the Relational Transformer and publish them on the site's Questions & answers page (/#/qa). Use when the user runs /rtquestion <questions> or asks to add questions to the site's Q&A. A prompt with several questions is split into separate questions, grouped into one entry when they are closely related and into separate entries otherwise. Each answer is researched and verified against the paper-era official code (snap-stanford/relational-transformer @2a899c8) and the paper, written as a new src/qa/entries/NNN-<id>.tsx file in the site's chapter style, checked, committed and pushed.
argument-hint: <one or more questions about the Relational Transformer>
---

# /rtquestion: add verified questions and answers to the site

The prompt is: **$ARGUMENTS**

If that is empty, ask the user for the question and stop.

Your job: answer every question in the prompt **truthfully**, verified against the real RT code and paper. Write the answers as
Q&A entries in this repository, in the same style as the chapters. Check that they build and render, refresh the search index,
then commit and push. Finally, give the user the short answers in chat.

## 0. Split the prompt into questions, then group them

The prompt may hold one question or several, written as one paragraph, a list, or a chain of sentences ("When are X created?
And how does Y know Z?"). First list every distinct question:

- One question = one thing the user wants to know. Split a sentence that asks two things into two questions. Keep a sentence
  that only gives context (an assumption, an example) attached to the question it explains.
- Keep the user's wording. Fix typos only; never add, merge or reinterpret what is asked.
- Before answering, read the existing entries (`ls src/qa/entries`; their `meta` holds the questions). If an existing entry
  already answers a question fully, do not write a duplicate: report its page instead. If the new question is a follow-up,
  answer it in a new entry and link the earlier one.

Then decide how many entries to write. Decide yourself; do not ask the user.

- **Same entry** when the questions are about the same mechanism and one answer naturally covers them: they share most of
  their sources, and answering one without the others would leave the reader with half the picture. Example: "When are task
  tables created?", "How are their rows filled before inference?" and "How does that work zero-shot?" are one entry.
- **Separate entries** when the questions are about different things (different mechanisms, usually different chapters), or
  when each stands on its own and someone looking for one would not expect the others. Example: "How are task tables built?"
  and "Why does RT use RMSNorm?" are two entries.
- Rule of thumb: if one headline can name the shared topic honestly, use one entry; if the headline would need "and" to join
  unrelated subjects, split. Never merge to save work, and never split one question across entries.

Research and write each entry on its own (steps 1-3). Then run the checks once over all of them (step 4), publish them in one
commit (step 5) and report (step 6).

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

- `NNN` is the next number: `ls src/qa/entries`, take the highest and add 1, zero-padded to 3 digits. With several entries,
  number them consecutively in the order the questions were asked.
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
  // an entry that groups related questions lists them, in the order asked:
  // question: ['<first question>', '<second question>'],
  short: '<1-3 plain sentences that answer it directly (up to 4 for a group)>',
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
- **Grouped questions**: the page lists them numbered under "The questions". Give each question at least one Section of its
  own, in the same order, whose heading answers that question; shared background can sit in a Section before them. The
  `title` names the shared topic, and `short` answers every question in the group.
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
- **Spelling**: American, like the paper and the code (normalize, color, neighbor, optimizer, labeled).
- **Readability (the user's standing rule)**: never set small text. Floors: prose 18px, UI 15px, captions 14px, chips 13px,
  SVG labels 12px. When something does not fit, make room instead of shrinking.

## 4. Check it

Run these with `export PATH=$HOME/.nvm/versions/node/v24.15.0/bin:$PATH` if `npm` is missing.

```bash
npm run typecheck && npm test && npm run build
npm run search-index                             # re-reads every page from dist/ into public/search-index.json (~1 min)
npx vite preview --port 5311 --strictPort        # run in the background
node scripts/shoot.mjs --port 5311 --routes qa,qa/<id>,qa/<id2> --out <scratch>/qa-shots --themes light,dark --widths 1440,390
```

`search-index` must print `ok` for every page, including each new `/qa/<id>`. The site search reads that file, so a new
answer is not searchable until it is regenerated. `shoot.mjs` must print `ok` for every shot: no console errors, no horizontal
overflow. Look at the screenshots (Read the PNGs, cropping long ones) and fix anything broken or cramped. If you change an
entry after this, rebuild and run `npm run search-index` again.

Then do a final citation pass. For each `<Cite kind="code">`, open those exact lines in the checkout and confirm they show what
your sentence says. For each paper cite, confirm the page. Fix or remove anything that does not hold.

## 5. Publish

```bash
git add src/qa/entries/NNN-<id>.tsx public/search-index.json   # every new entry, plus any new shared code you added
git commit -m "Q&A: <title>" -m "Co-Authored-By: <the attribution line from the system reminder>"
git push origin main
```

One commit for the whole prompt. With several entries, the subject is `Q&A: <n> new questions` and the body lists the titles.

The user has authorized commits and pushes to this repository's `origin/main`. The GitHub Pages workflow deploys on push.

## 6. Report

Reply in chat with:

- how you split and grouped the prompt, in a line or two, when it held more than one question;
- for each entry: the short answer and the page path `/#/qa/<id>`;
- any question an existing entry already answered, with that entry's path;
- any caveat (an inference you flagged, or a point the code and paper leave open).

Keep it brief: the full answer lives on the site.
