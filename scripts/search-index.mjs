#!/usr/bin/env node
/**
 * Builds the site search index. Serves a built site (dist/), opens every chapter and Q&A answer
 * in headless Chromium, and stores the text src/search/extract.ts reads from each rendered page,
 * so the index holds what readers see (live numbers at the default settings included).
 *
 * Usage:
 *   npm run build && npm run search-index                    # writes public/search-index.json
 *   node scripts/search-index.mjs --out dist/search-index.json   # CI: index the build it deploys
 *   node scripts/search-index.mjs --pages /tasks,/sampling --out <scratch>/idx.json   # a quick look at some pages
 *
 * Run it after changing chapter or Q&A text, and commit public/search-index.json.
 */
import { chromium } from 'playwright';
import { preview } from 'vite';
import fs from 'node:fs';
import path from 'node:path';

const args = Object.fromEntries(
  process.argv.slice(2).reduce((acc, a, i, arr) => {
    if (a.startsWith('--')) acc.push([a.slice(2), arr[i + 1] && !arr[i + 1].startsWith('--') ? arr[i + 1] : 'true']);
    return acc;
  }, []),
);
const dist = String(args.dist ?? 'dist');
const outs = String(args.out ?? 'public/search-index.json').split(',');
if (!fs.existsSync(path.join(dist, 'index.html'))) {
  console.error(`${dist}/index.html not found: run npm run build first`);
  process.exit(1);
}

const server = await preview({ build: { outDir: dist }, preview: { port: 5391, open: false }, logLevel: 'error' });
const base = server.resolvedUrls.local[0];

let browser;
try {
  browser = await chromium.launch();
} catch {
  browser = await chromium.launch({ executablePath: '/usr/bin/google-chrome' });
}

/** A fresh context per page, so no page sees settings another one stored. */
async function open(route) {
  const ctx = await browser.newContext({ viewport: { width: 1280, height: 900 }, colorScheme: 'light' });
  const page = await ctx.newPage();
  const errors = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await page.goto(`${base}#${route}`, { waitUntil: 'networkidle' });
  await page.waitForFunction(() => !!window.__rtSearch, null, { timeout: 15000 });
  return { ctx, page, errors };
}

const home = await open('/');
const only = args.pages ? String(args.pages).split(',') : null;
const pages = (await home.page.evaluate(() => window.__rtSearch.pages())).filter((p) => !only || only.includes(p.path));
await home.ctx.close();

const out = [];
let failed = false;
for (const meta of pages) {
  const { ctx, page, errors } = await open(meta.path);
  // lazy chapters and live figures settle after the first paint: read until two reads agree
  let sections = null;
  let prev = '';
  for (let i = 0; i < 20; i++) {
    await page.waitForTimeout(700);
    sections = await page.evaluate(() => window.__rtSearch.extract());
    const sig = JSON.stringify(sections);
    if (sections?.length > 1 && sig === prev) break;
    prev = sig;
  }
  await ctx.close();
  if (!sections?.length) {
    console.error(`FAIL ${meta.path}: no text found`);
    failed = true;
    continue;
  }
  const blocks = sections.reduce((n, s) => n + s.blocks.length, 0);
  console.log(`${errors.length ? 'WARN' : 'ok  '} ${meta.path.padEnd(42)} sections ${String(sections.length).padStart(3)}  blocks ${String(blocks).padStart(4)}${errors.length ? `  error: ${errors[0].slice(0, 120)}` : ''}`);
  out.push({ ...meta, sections });
}
await browser.close();
await server.close();
if (failed) process.exit(1);

const json = JSON.stringify({ version: 1, pages: out });
for (const o of outs) {
  fs.mkdirSync(path.dirname(o), { recursive: true });
  fs.writeFileSync(o, json);
  console.log(`wrote ${o} (${Math.round(json.length / 1024)} KB, ${out.length} pages)`);
}
