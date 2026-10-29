/**
 * Draws the certificate on a canvas: the one picture behind the on-screen preview, the PNG and the
 * PDF, so all three always match. It is laid out on a 1600 × 1131 page (A4 landscape) and scaled to
 * the requested width once the fonts have loaded. Its colors are fixed: paper is white in every theme.
 */
import { encode } from 'uqr';
import { SITE_TITLE, SITE_URL } from '@/components/prose/sources';
import { jpegPdf } from './pdf';
import { formatCertDate, type CertFacts } from './seal';

export interface CertificateView extends CertFacts {
  /** Formatted ID ("ABCDE-FGHJK-MNPQR-STVWX"). */
  id: string;
  /** Verification link, encoded in the QR code. */
  url: string;
}

export const PAGE = { width: 1600, height: 1131 };
/** Who issues the certificate (the site's author, as in the footer). */
export const ISSUER = 'Arthur Sudbrack Ibarra';

const SERIF = '"Libron", "Iowan Old Style", Charter, Georgia, serif';
const SANS = 'system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif';
const INK = {
  paper: '#fffdf8',
  ink: '#1c1a17',
  ink2: '#4b4842',
  muted: '#726f67',
  rule: '#d3cdc0',
  accent: '#6250d6',
  frame: '#cfc8f3',
  blue: '#2a78d6',
  aqua: '#1baf7a',
  magenta: '#e87ba4',
};

interface TextStyle {
  size: number;
  family?: string;
  weight?: number;
  italic?: boolean;
  color?: string;
  align?: CanvasTextAlign;
  /** Letter spacing, in em. */
  tracking?: number;
}

/** Renders the certificate at `width` pixels (2400 ≈ 200 dpi on A4). */
export async function renderCertificate(view: CertificateView, width = 2400): Promise<HTMLCanvasElement> {
  await loadFonts();
  const scale = width / PAGE.width;
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = Math.round(PAGE.height * scale);
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('canvas 2D is not available');
  ctx.scale(scale, scale);
  drawCertificate(ctx, view, scale);
  return canvas;
}

export function canvasBlob(canvas: HTMLCanvasElement, type: 'image/png' | 'image/jpeg', quality?: number): Promise<Blob> {
  return new Promise((resolve, reject) =>
    canvas.toBlob((b) => (b ? resolve(b) : reject(new Error(`could not encode ${type}`))), type, quality),
  );
}

/** The certificate as a one-page A4 PDF. */
export async function certificatePdf(canvas: HTMLCanvasElement, view: CertificateView): Promise<Blob> {
  const jpeg = new Uint8Array(await (await canvasBlob(canvas, 'image/jpeg', 0.92)).arrayBuffer());
  const pdf = jpegPdf(jpeg, canvas, {
    title: `Certificate of completion: ${view.name}`,
    author: ISSUER,
    subject: `${SITE_TITLE} · certificate ${view.id} · verify at ${view.url}`,
  });
  return new Blob([pdf as BlobPart], { type: 'application/pdf' });
}

/** Saves a blob under `filename` through a temporary link. */
export function saveBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 30_000);
}

/** "certificate-rt-explained-ada-lovelace". */
export function certificateFileName(name: string): string {
  const slug = name
    .normalize('NFKD')
    .replace(/[̀-ͯ]/g, '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 48);
  return `certificate-rt-explained${slug ? `-${slug}` : ''}`;
}

async function loadFonts(): Promise<void> {
  if (typeof document === 'undefined' || !document.fonts) return;
  // Libron is declared by the site's CSS; a failed load falls back to the next serif
  await Promise.all(['400 40px Libron', 'italic 400 40px Libron', '700 40px Libron'].map((f) => document.fonts.load(f).catch(() => [])));
}

function drawCertificate(ctx: CanvasRenderingContext2D, v: CertificateView, scale: number): void {
  const { width: W, height: H } = PAGE;
  const cx = W / 2;

  ctx.fillStyle = INK.paper;
  ctx.fillRect(0, 0, W, H);
  ctx.strokeStyle = INK.accent;
  ctx.lineWidth = 4;
  ctx.strokeRect(34, 34, W - 68, H - 68);
  ctx.strokeStyle = INK.frame;
  ctx.lineWidth = 1.5;
  ctx.strokeRect(48, 48, W - 96, H - 96);

  drawLogo(ctx, cx - 36, 104, 72);
  text(ctx, SITE_TITLE.toUpperCase(), cx, 232, { size: 20, family: SANS, weight: 600, color: INK.accent, tracking: 0.2 });
  text(ctx, 'Certificate of Completion', cx, 326, { size: 72, weight: 700 });
  text(ctx, 'This certifies that', cx, 404, { size: 30, italic: true, color: INK.ink2 });

  const nameSize = fitSize(ctx, v.name, 80, 30, 1240, `700 {size}px ${SERIF}`);
  text(ctx, v.name, cx, 506, { size: nameSize, weight: 700 });
  const half = Math.max(470, ctx.measureText(v.name).width / 2 + 24); // a long name widens its rule
  rule(ctx, cx - half, cx + half, 540);

  const body = { size: 28, color: INK.ink2 };
  text(ctx, `has completed all ${v.chapters} chapters of ${SITE_TITLE}, an interactive,`, cx, 606, body);
  text(ctx, 'code-faithful course on the Relational Transformer (Ranjan et al., ICLR 2026),', cx, 646, body);
  text(ctx, `and passed its final quiz with ${v.score} of ${v.total} answers correct.`, cx, 686, body);

  // date and issuer on signature lines, the QR code on the right
  const label = { size: 15, family: SANS, weight: 600, color: INK.muted, tracking: 0.14 };
  text(ctx, formatCertDate(v.date), 380, 866, { size: 30 });
  rule(ctx, 230, 530, 890);
  text(ctx, 'DATE OF COMPLETION', 380, 920, label);
  text(ctx, ISSUER, 800, 866, { size: 34, italic: true });
  rule(ctx, 610, 990, 890);
  text(ctx, 'COURSE AUTHOR', 800, 920, label);
  drawQr(ctx, v.url, 1180, 744, 180, scale);
  text(ctx, 'SCAN TO VERIFY', 1270, 952, label);

  text(ctx, `Certificate ID  ${v.id}`, cx, 1006, { size: 21, family: SANS, weight: 600, color: INK.ink, tracking: 0.04 });
  text(ctx, `Verify at ${SITE_URL.replace(/^https:\/\//, '')}#/certificate/verify with the name and this ID.`, cx, 1038, {
    size: 16,
    family: SANS,
    color: INK.ink2,
  });
  text(
    ctx,
    'Issued by an independent educational site, not affiliated with the paper’s authors or their institutions. Not an academic credential.',
    cx,
    1064,
    { size: 15, family: SANS, color: INK.muted },
  );
}

function text(ctx: CanvasRenderingContext2D, s: string, x: number, y: number, st: TextStyle): void {
  const { size, family = SERIF, weight = 400, italic = false, color = INK.ink, align = 'center', tracking = 0 } = st;
  ctx.font = `${italic ? 'italic ' : ''}${weight} ${size}px ${family}`;
  ctx.fillStyle = color;
  ctx.textAlign = align;
  ctx.textBaseline = 'alphabetic';
  const spacing = tracking * size;
  const tracked = spacing > 0 && 'letterSpacing' in ctx;
  if (tracked) ctx.letterSpacing = `${spacing}px`;
  // letter spacing also follows the last glyph: shift centered text back by half of it
  ctx.fillText(s, tracked && align === 'center' ? x + spacing / 2 : x, y);
  if (tracked) ctx.letterSpacing = '0px';
}

/** Largest size in [min, max] at which `s` fits in `width`; `font` has a {size} placeholder. */
function fitSize(ctx: CanvasRenderingContext2D, s: string, max: number, min: number, width: number, font: string): number {
  ctx.font = font.replace('{size}', String(max));
  const w = ctx.measureText(s).width;
  return w <= width ? max : Math.max(min, Math.floor((max * width) / w));
}

function rule(ctx: CanvasRenderingContext2D, x1: number, x2: number, y: number): void {
  ctx.strokeStyle = INK.rule;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(x1, y);
  ctx.lineTo(x2, y);
  ctx.stroke();
}

/** The site's mark (components/shell/Logo.tsx): three cell tokens and the dashed "?" target. */
function drawLogo(ctx: CanvasRenderingContext2D, x: number, y: number, size: number): void {
  ctx.save();
  ctx.translate(x, y);
  ctx.scale(size / 24, size / 24);
  const square = (sx: number, sy: number, s: number, r: number) => {
    ctx.beginPath();
    ctx.moveTo(sx + r, sy);
    ctx.arcTo(sx + s, sy, sx + s, sy + s, r);
    ctx.arcTo(sx + s, sy + s, sx, sy + s, r);
    ctx.arcTo(sx, sy + s, sx, sy, r);
    ctx.arcTo(sx, sy, sx + s, sy, r);
    ctx.closePath();
  };
  for (const [sx, sy, color] of [
    [1.5, 1.5, INK.blue],
    [13, 1.5, INK.aqua],
    [1.5, 13, INK.magenta],
  ] as const) {
    square(sx, sy, 9.5, 2.6);
    ctx.fillStyle = color;
    ctx.fill();
  }
  square(13.6, 13.6, 8.3, 2.2);
  ctx.setLineDash([2, 1.6]);
  ctx.strokeStyle = INK.accent;
  ctx.lineWidth = 1.3;
  ctx.stroke();
  ctx.setLineDash([]);
  // smaller than the header mark's 11.5px "?", which overflows its box at this size in some system fonts
  ctx.font = `800 8.6px ${SANS}`;
  ctx.fillStyle = INK.accent;
  ctx.textAlign = 'center';
  ctx.fillText('?', 17.75, 20.9);
  ctx.restore();
}

/** QR code of `data` in a `size` square, modules snapped to whole device pixels so they stay crisp. */
function drawQr(ctx: CanvasRenderingContext2D, data: string, x: number, y: number, size: number, scale: number): void {
  const qr = encode(data, { ecc: 'M', border: 0 });
  const module = Math.floor((size * scale) / qr.size) / scale;
  const side = module * qr.size;
  const ox = Math.round((x + (size - side) / 2) * scale) / scale;
  const oy = Math.round((y + (size - side) / 2) * scale) / scale;
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(ox - module * 2, oy - module * 2, side + module * 4, side + module * 4);
  ctx.beginPath();
  qr.data.forEach((row, r) =>
    row.forEach((dark, c) => {
      if (dark) ctx.rect(ox + c * module, oy + r * module, module, module);
    }),
  );
  ctx.fillStyle = '#000000';
  ctx.fill();
}
