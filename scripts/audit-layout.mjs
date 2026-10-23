#!/usr/bin/env node
/**
 * Layout audit: loads routes of a running server and reports, per route × width × mode,
 *   - console errors, page errors, failed requests;
 *   - horizontal page overflow;
 *   - visible text rendered below the readability floors (HTML < 13px, SVG text < 12px);
 *   - elements that stick out past the viewport (outside horizontal scroll containers);
 *   - overlapping SVG text labels inside one SVG;
 *   - clipped text (scrollWidth > clientWidth with overflow hidden and no ellipsis).
 *
 * Usage:
 *   node scripts/audit-layout.mjs --port 5311 [--routes a,b] [--widths 1440,390] [--modes light,dark,reading] [--out report.json]
 *
 * Prints a compact summary; --out also writes the full JSON (selectors + text snippets).
 */
import { chromium } from 'playwright';
import fs from 'node:fs';

const args = Object.fromEntries(
  process.argv.slice(2).reduce((acc, a, i, arr) => {
    if (a.startsWith('--')) acc.push([a.slice(2), arr[i + 1] && !arr[i + 1].startsWith('--') ? arr[i + 1] : 'true']);
    return acc;
  }, []),
);
const ALL = 'home,overview,database,tasks,preprocessing,semantics,sampling,tokens,embedding,masks,attention,block,decoding,training,playground,results,about,qa,qa/where-task-tables-come-from';
const port = Number(args.port ?? 5173);
const routes = String(args.routes ?? ALL).split(',').filter(Boolean);
const widths = String(args.widths ?? '1440,390').split(',').map(Number);
const modes = String(args.modes ?? 'light,dark').split(',');

let browser;
try {
  browser = await chromium.launch();
} catch {
  browser = await chromium.launch({ executablePath: '/usr/bin/google-chrome' });
}

const report = [];
for (const mode of modes) {
  for (const width of widths) {
    const ctx = await browser.newContext({
      viewport: { width, height: width < 600 ? 844 : 900 },
      colorScheme: mode === 'dark' ? 'dark' : 'light',
    });
    if (mode === 'reading')
      await ctx.addInitScript(() => {
        try {
          localStorage.setItem('rt-sbs:sim', JSON.stringify({ state: { reading: true }, version: 1 }));
        } catch {
          /* storage blocked */
        }
      });
    for (const route of routes) {
      const page = await ctx.newPage();
      const errors = [];
      page.on('console', (m) => m.type() === 'error' && errors.push(`console: ${m.text().slice(0, 200)}`));
      page.on('pageerror', (e) => errors.push(`pageerror: ${e.message.slice(0, 200)}`));
      page.on('requestfailed', (r) => errors.push(`requestfailed: ${r.url()}`));
      await page.goto(`http://localhost:${port}/#/${route === 'home' ? '' : route}`, { waitUntil: 'networkidle' }).catch((e) => errors.push(`goto: ${e.message}`));
      await page.waitForTimeout(2000);
      const found = await page.evaluate(() => {
        const vw = window.innerWidth;
        const sel = (el) => {
          const parts = [];
          for (let e = el; e && e.nodeType === 1 && parts.length < 4; e = e.parentElement) {
            let s = e.tagName.toLowerCase();
            if (e.id) s += `#${e.id}`;
            const c = (typeof e.className === 'string' ? e.className : e.getAttribute('class') || '').split(/\s+/).filter(Boolean).slice(0, 2);
            if (c.length) s += `.${c.join('.')}`;
            parts.unshift(s);
          }
          return parts.join(' > ');
        };
        const section = (el) => el.closest('section[id]')?.id ?? el.closest('figure')?.querySelector('div')?.textContent?.slice(0, 40) ?? '';
        const visible = (el) => {
          const r = el.getBoundingClientRect();
          if (r.width === 0 || r.height === 0) return false;
          const st = getComputedStyle(el);
          return st.visibility !== 'hidden' && st.display !== 'none' && Number(st.opacity) > 0.05 && !el.closest('[hidden],[aria-hidden="true"] .sr-only,.sr-only');
        };
        const inScroller = (el) => {
          for (let e = el.parentElement; e; e = e.parentElement) {
            const ox = getComputedStyle(e).overflowX;
            if ((ox === 'auto' || ox === 'scroll' || ox === 'hidden' || ox === 'clip') && e.scrollWidth > e.clientWidth + 1) return true;
          }
          return false;
        };
        const out = { overflow: document.documentElement.scrollWidth - vw, small: [], offscreen: [], overlaps: [], clipped: [] };
        // small text (HTML)
        const seen = new Set();
        const walker = document.createTreeWalker(document.querySelector('main') ?? document.body, NodeFilter.SHOW_TEXT);
        for (let n = walker.nextNode(); n; n = walker.nextNode()) {
          const t = n.textContent.replace(/[\u200b\s]/g, '');
          if (!t) continue;
          const el = n.parentElement;
          // KaTeX: the MathML copy is for screen readers only; vlist struts carry no glyphs
          if (!el || seen.has(el) || el.closest('.katex-mathml') || !visible(el)) continue;
          seen.add(el);
          const fs = parseFloat(getComputedStyle(el).fontSize);
          const isSvg = el instanceof SVGElement;
          // SVG text scales with the viewBox: measure the rendered glyph height instead
          let px = fs;
          if (isSvg && el.getScreenCTM) {
            const m = el.getScreenCTM();
            if (m) px = fs * Math.hypot(m.a, m.b);
          }
          if (px < (isSvg ? 11.5 : 12.5)) out.small.push({ px: Math.round(px * 10) / 10, svg: isSvg, text: t.slice(0, 40), where: section(el), sel: sel(el) });
        }
        // off-screen elements
        for (const el of document.querySelectorAll('main *')) {
          if (!visible(el) || el.closest('.katex')) continue;
          const r = el.getBoundingClientRect();
          if ((r.right > vw + 1 || r.left < -1) && !inScroller(el)) {
            out.offscreen.push({ right: Math.round(r.right), left: Math.round(r.left), where: section(el), sel: sel(el) });
            if (out.offscreen.length > 15) break;
          }
        }
        // overlapping SVG labels within one svg
        for (const svg of document.querySelectorAll('main svg')) {
          const texts = [...svg.querySelectorAll('text')].filter((t) => visible(t) && t.textContent.trim());
          if (texts.length < 2 || texts.length > 400) continue;
          const boxes = texts.map((t) => ({ t, r: t.getBoundingClientRect() }));
          for (let i = 0; i < boxes.length; i++)
            for (let j = i + 1; j < boxes.length; j++) {
              const a = boxes[i].r;
              const b = boxes[j].r;
              const ox = Math.min(a.right, b.right) - Math.max(a.left, b.left);
              const oy = Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top);
              // identical labels stacked on purpose (cross-fades, outlines under fills) are not collisions
              if (boxes[i].t.textContent === boxes[j].t.textContent) continue;
              if (ox > 3 && oy > 3 && ox * oy > 0.25 * Math.min(a.width * a.height, b.width * b.height))
                if (out.overlaps.length < 20)
                  out.overlaps.push({ a: boxes[i].t.textContent.trim().slice(0, 30), b: boxes[j].t.textContent.trim().slice(0, 30), where: section(svg) });
            }
        }
        // clipped text (overflow hidden, no ellipsis, text wider than the box)
        for (const el of document.querySelectorAll('main *')) {
          if (!visible(el) || el.children.length > 0 || !el.textContent.trim()) continue;
          const st = getComputedStyle(el);
          if ((st.overflowX === 'hidden' || st.overflowX === 'clip') && st.textOverflow !== 'ellipsis' && el.scrollWidth > el.clientWidth + 2)
            if (out.clipped.length < 15) out.clipped.push({ text: el.textContent.trim().slice(0, 40), where: section(el), sel: sel(el) });
        }
        return out;
      });
      report.push({ route, width, mode, errors, ...found });
      const n = (k) => found[k].length;
      const flag = errors.length || found.overflow > 1 || n('small') || n('offscreen') || n('overlaps') || n('clipped');
      console.log(
        `${flag ? 'WARN' : 'ok  '} ${route.padEnd(32)} ${String(width).padStart(4)} ${mode.padEnd(7)} errors ${errors.length} overflow ${found.overflow} small ${n('small')} offscreen ${n('offscreen')} overlaps ${n('overlaps')} clipped ${n('clipped')}`,
      );
      await page.close();
    }
    await ctx.close();
  }
}
await browser.close();
if (args.out) fs.writeFileSync(args.out, JSON.stringify(report, null, 1));
