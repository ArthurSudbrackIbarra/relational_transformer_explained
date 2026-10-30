/**
 * Writes the site icon and the link-preview images into public/, from the Logo mark:
 *   favicon.svg            the mark (index.html links it)
 *   apple-touch-icon.png   180×180, the mark on the page surface (iOS home screen, some link previews)
 *   og-image.png           1200×630, the card that Discord, WhatsApp, Slack, X and others show under a link
 *
 * Usage: npx tsx scripts/social-images.ts
 *
 * The card loads Inter from Google Fonts, so run it online; offline it falls back to the system sans.
 * Rerun it after changing the mark (LOGO_TARGET) or the card text below, and keep the og:image
 * dimensions in index.html in step with WIDTH × HEIGHT.
 */
import { chromium, type Browser } from 'playwright';
import fs from 'node:fs';
import path from 'node:path';
import { logoMarkSvg } from '../src/components/shell/logoMark';

const OUT = path.resolve(import.meta.dirname, '../public');
const WIDTH = 1200;
const HEIGHT = 630;

// Light-theme tokens (src/styles/index.css).
const C = {
  surface: '#fcfcfb',
  ink: '#0b0b0b',
  ink2: '#52514e',
  muted: '#898781',
  accent: '#6250d6',
  accentText: '#5242c4',
  blue: '#2a78d6',
  aqua: '#1baf7a',
  magenta: '#e87ba4',
};

const mark = logoMarkSvg();
const sized = (px: number) => mark.replace('<svg ', `<svg width="${px}" height="${px}" `);

// ---- Illustration: two database tables and a task table whose "?" cell attends to their cells ----

type Table = { x: number; y: number; name: string; color: string; cols: [string, number][]; rows: string[][] };
const HEAD = 44;
const COLROW = 32;
const ROW = 42;

const customers: Table = {
  x: 640, y: 48, name: 'customers', color: C.blue,
  cols: [['name', 96], ['country', 88], ['age', 60]],
  rows: [['Kenji', 'JP', '51'], ['Lena', 'DE', '27'], ['Ana', 'BR', '34']],
};
const orders: Table = {
  x: 944, y: 124, name: 'orders', color: C.aqua,
  cols: [['product', 96], ['qty', 48], ['channel', 80]],
  rows: [['Atlas', '1', 'app'], ['Mouse', '2', 'web']],
};
const churn: Table = {
  x: 640, y: 404, name: 'customer-churn', color: C.magenta,
  cols: [['customer', 92], ['date', 124], ['churn', 72]],
  rows: [['Lena', '2024-05-01', 'false'], ['Ana', '2024-06-01', '?']],
};

const width = (t: Table) => t.cols.reduce((s, [, w]) => s + w, 0);
const bottom = (t: Table) => t.y + HEAD + COLROW + t.rows.length * ROW;
/** Cell (row r, column c) of a table: its left edge, width and center. */
function cell(t: Table, r: number, c: number) {
  const x0 = t.x + t.cols.slice(0, c).reduce((s, [, w]) => s + w, 0);
  return { x0, w: t.cols[c][1], x: x0 + t.cols[c][1] / 2, y: t.y + HEAD + COLROW + r * ROW + ROW / 2 };
}

// Attention weight of the "?" cell on other cells, "table:row:col" → weight. The cells it attends to get
// an accent tint; the strongest ones (in each table's last row) also get an arc.
const attention: Record<string, number> = {
  'customers:2:0': 1, 'customers:2:2': 0.7, 'customers:2:1': 0.45, 'customers:1:1': 0.2,
  'orders:1:0': 0.85, 'orders:1:2': 0.5, 'orders:0:0': 0.3,
  'customer-churn:0:2': 0.6,
};

function table(t: Table): string {
  const w = width(t);
  const h = bottom(t) - t.y;
  const parts = [
    `<rect x="${t.x}" y="${t.y}" width="${w}" height="${h}" rx="14" fill="#ffffff" stroke="rgba(11,11,11,0.12)"/>`,
    `<path d="M${t.x} ${t.y + HEAD}V${t.y + 14}a14 14 0 0 1 14 -14H${t.x + w - 14}a14 14 0 0 1 14 14V${t.y + HEAD}Z" fill="${t.color}"/>`,
    `<text x="${t.x + 16}" y="${t.y + 29}" class="tname">${t.name}</text>`,
  ];
  t.cols.forEach(([name], c) => {
    const { x0 } = cell(t, 0, c);
    parts.push(`<text x="${x0 + 14}" y="${t.y + HEAD + 21}" class="col">${name}</text>`);
  });
  t.rows.forEach((row, r) => {
    const y = t.y + HEAD + COLROW + r * ROW;
    parts.push(`<line x1="${t.x}" x2="${t.x + w}" y1="${y}" y2="${y}" stroke="rgba(11,11,11,0.08)"/>`);
    row.forEach((v, c) => {
      const p = cell(t, r, c);
      const a = attention[`${t.name}:${r}:${c}`];
      if (v === '?') {
        parts.push(
          `<rect x="${p.x0 + 8}" y="${y + 6}" width="${p.w - 16}" height="${ROW - 12}" rx="8" fill="rgba(98,80,214,0.09)" stroke="${C.accent}" stroke-width="2" stroke-dasharray="6 4"/>`,
          `<text x="${p.x}" y="${y + 29}" class="q" text-anchor="middle">?</text>`,
        );
        return;
      }
      if (a) parts.push(`<rect x="${p.x0 + 6}" y="${y + 6}" width="${p.w - 12}" height="${ROW - 12}" rx="7" fill="${C.accent}" opacity="${(0.06 + 0.2 * a).toFixed(2)}"/>`);
      parts.push(`<text x="${p.x0 + 14}" y="${y + 27}" class="val">${v}</text>`);
    });
  });
  return parts.join('');
}

// Arcs leave the "?" cell on its right and enter each table from below. The arcs to orders swing wide to the
// right; the arcs to customers rise in the gap beside customer-churn and turn left, nested so none cross.
const q = cell(churn, 1, 2);
function arcs(): string {
  const sx = q.x0 + q.w - 8;
  const sy = q.y;
  const stroke = (a: number) =>
    `fill="none" stroke="${C.accent}" stroke-width="${(1.5 + 2.5 * a).toFixed(2)}" stroke-linecap="round" opacity="${(0.3 + 0.6 * a).toFixed(2)}"`;
  const last = (t: Table) =>
    t.cols.map((_, c) => ({ p: cell(t, t.rows.length - 1, c), a: attention[`${t.name}:${t.rows.length - 1}:${c}`] })).filter((l) => l.a);
  const toCustomers = last(customers).map(({ p, a }, k) => {
    const wx = width(churn) + churn.x + 16 + 12 * k;
    const turn = 380 - 20 * k;
    const ty = bottom(customers);
    return `<path d="M${sx} ${sy}C${wx - 8} ${sy} ${wx} ${sy - 8} ${wx} ${sy - 36}L${wx} ${turn + 30}C${wx} ${turn - 30} ${p.x} ${ty + 70} ${p.x} ${ty + 4}" ${stroke(a)}/>`;
  });
  const toOrders = last(orders).map(({ p, a }) => {
    const ty = bottom(orders);
    return `<path d="M${sx} ${sy}C${sx + 150} ${sy} ${p.x} ${ty + 120} ${p.x} ${ty + 4}" ${stroke(a)}/>`;
  });
  return [...toCustomers, ...toOrders].join('');
}

const illustration = `
<svg class="art" width="${WIDTH}" height="${HEIGHT}" viewBox="0 0 ${WIDTH} ${HEIGHT}">
  ${arcs()}${table(customers)}${table(orders)}${table(churn)}
</svg>`;

const card = `<!doctype html>
<html><head><meta charset="utf-8">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=block">
<style>
  * { margin: 0; box-sizing: border-box; }
  html, body { width: ${WIDTH}px; height: ${HEIGHT}px; overflow: hidden; }
  body {
    position: relative;
    background: ${C.surface};
    font-family: Inter, system-ui, sans-serif;
    color: ${C.ink};
  }
  .dots {
    position: absolute; inset: 0;
    background-image: radial-gradient(rgba(11,11,11,0.09) 1.2px, transparent 1.4px);
    background-size: 24px 24px;
    -webkit-mask-image: linear-gradient(90deg, transparent 38%, #000 62%);
  }
  .art { position: absolute; inset: 0; }
  .art text { font-family: Inter, system-ui, sans-serif; }
  .tname { font-size: 19px; font-weight: 600; fill: #fff; }
  .col { font-size: 15px; font-weight: 500; fill: ${C.muted}; }
  .val { font-size: 18px; fill: ${C.ink}; }
  .q { font-size: 21px; font-weight: 700; fill: ${C.accent}; }
  .text { position: absolute; left: 72px; top: 64px; width: 540px; }
  .brand { display: flex; align-items: center; gap: 16px; font-size: 21px; font-weight: 600; color: ${C.accentText}; }
  h1 { margin-top: 36px; font-size: 72px; line-height: 1.02; font-weight: 800; letter-spacing: -0.025em; }
  h1 span { color: ${C.accent}; }
  .sub { margin-top: 24px; font-size: 26px; line-height: 1.4; color: ${C.ink2}; }
  .foot { position: absolute; left: 72px; bottom: 48px; font-size: 20px; font-weight: 500; color: ${C.muted}; }
</style></head>
<body>
  <div class="dots"></div>
  ${illustration}
  <div class="text">
    <div class="brand">${sized(52)}An interactive course</div>
    <h1>Relational Transformer <span>explained</span></h1>
    <p class="sub">How a foundation model for relational databases works, with a real model running in your browser.</p>
  </div>
  <div class="foot">17 chapters · based on the ICLR 2026 paper</div>
</body></html>`;

const touchIcon = `<!doctype html>
<html><head><style>
  * { margin: 0; }
  body { width: 180px; height: 180px; display: grid; place-items: center; background: ${C.surface}; }
</style></head><body>${sized(132)}</body></html>`;

async function render(browser: Browser, html: string, w: number, h: number, file: string) {
  const page = await browser.newPage({ viewport: { width: w, height: h }, deviceScaleFactor: 1 });
  await page.setContent(html, { waitUntil: 'networkidle' });
  await page.evaluate('document.fonts.ready');
  await page.screenshot({ path: path.join(OUT, file), clip: { x: 0, y: 0, width: w, height: h } });
  await page.close();
  console.log(`${file}  ${(fs.statSync(path.join(OUT, file)).size / 1024).toFixed(0)} KB`);
}

fs.writeFileSync(path.join(OUT, 'favicon.svg'), mark + '\n');
console.log('favicon.svg');

let browser: Browser;
try {
  browser = await chromium.launch();
} catch {
  browser = await chromium.launch({ executablePath: '/usr/bin/google-chrome' });
}
await render(browser, touchIcon, 180, 180, 'apple-touch-icon.png');
await render(browser, card, WIDTH, HEIGHT, 'og-image.png');
await browser.close();
