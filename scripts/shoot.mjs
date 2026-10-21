#!/usr/bin/env node
/**
 * Visual QA helper: screenshots routes of a running dev server and reports console errors.
 *
 * Usage:
 *   node scripts/shoot.mjs --port 5301 --routes overview,database --out /path/to/dir \
 *        [--themes light,dark] [--widths 1440,390] [--full] [--wait 1500] [--click "selector"]
 *
 * Start the server yourself first, e.g. `npx vite --port 5301 --strictPort &`.
 * Exit code 1 if any page logged a console error / page error / failed request.
 */
import { chromium } from 'playwright';
import fs from 'node:fs';
import path from 'node:path';

const args = Object.fromEntries(
  process.argv.slice(2).reduce((acc, a, i, arr) => {
    if (a.startsWith('--')) acc.push([a.slice(2), arr[i + 1] && !arr[i + 1].startsWith('--') ? arr[i + 1] : 'true']);
    return acc;
  }, []),
);
const port = Number(args.port ?? 5173);
const routes = String(args.routes ?? '').split(',').filter(Boolean);
const out = args.out ?? './shots';
const themes = String(args.themes ?? 'light,dark').split(',');
const widths = String(args.widths ?? '1440,390').split(',').map(Number);
const full = args.full !== undefined ? args.full === 'true' : true;
const wait = Number(args.wait ?? 1500);
fs.mkdirSync(out, { recursive: true });

let browser;
try {
  browser = await chromium.launch();
} catch {
  browser = await chromium.launch({ executablePath: '/usr/bin/google-chrome' });
}

let problems = 0;
for (const theme of themes) {
  for (const width of widths) {
    const ctx = await browser.newContext({
      viewport: { width, height: width < 600 ? 844 : 900 },
      colorScheme: theme === 'dark' ? 'dark' : 'light',
      deviceScaleFactor: 1,
    });
    // The app follows prefers-color-scheme when no theme is persisted (store key 'rt-sbs:sim', theme=null),
    // and Playwright emulates it via `colorScheme` above.
    for (const route of routes) {
      const page = await ctx.newPage();
      const errs = [];
      page.on('console', (m) => m.type() === 'error' && errs.push(`console: ${m.text()}`));
      page.on('pageerror', (e) => errs.push(`pageerror: ${e.message}`));
      page.on('requestfailed', (r) => errs.push(`requestfailed: ${r.url()}`));
      const url = `http://localhost:${port}/#/${route === 'home' ? '' : route}`;
      await page.goto(url, { waitUntil: 'networkidle' }).catch((e) => errs.push(`goto: ${e.message}`));
      await page.waitForTimeout(wait);
      if (args.click) await page.click(args.click).catch(() => {});
      const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
      if (overflow > 1) errs.push(`horizontal overflow: ${overflow}px`);
      const file = path.join(out, `${route.replace(/\//g, '_')}-${theme}-${width}.png`);
      await page.screenshot({ path: file, fullPage: full });
      console.log(`${errs.length ? 'FAIL' : 'ok  '} ${file}${errs.length ? '\n   ' + errs.join('\n   ') : ''}`);
      problems += errs.length;
      await page.close();
    }
    await ctx.close();
  }
}
await browser.close();
process.exit(problems ? 1 : 0);
