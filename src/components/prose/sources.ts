/**
 * Canonical source references (BUILD_PLAN §0: every factual claim carries a
 * citation to the spec, the paper or the paper-era code).
 */

/** The site's name: top bar, landing hero, document titles. */
export const SITE_TITLE = 'Relational Transformer Explained';
/** This site's own repository. */
export const SITE_REPO = 'https://github.com/ArthurSudbrackIbarra/relational_transformer_explained';
/** Where the site is published (certificate verification links point here). */
export const SITE_URL = 'https://arthursudbrackibarra.github.io/relational_transformer_explained/';

export const PAPER = {
  title: 'Relational Transformer: Toward Zero-Shot Foundation Models for Relational Data',
  authors: 'Ranjan et al.',
  venue: 'ICLR 2026',
  arxiv: '2510.06377v3',
  url: 'https://arxiv.org/abs/2510.06377',
  pdf: 'https://arxiv.org/pdf/2510.06377v3',
} as const;

export const CODE = {
  repo: 'snap-stanford/relational-transformer',
  /** Paper-era commit: the ground truth of this site (spec header). */
  commit: '2a899c8',
  commitFull: '2a899c8b49ac755b96b0c9c9a1b9c7f6bcdc245d',
  url: 'https://github.com/snap-stanford/relational-transformer',
} as const;

export const SPEC_PATH = 'research/RT_TECHNICAL_SPEC.md';

/** Parse "rt/model.py:20-47" → { file, start, end }. */
export function parseCodeRef(ref: string): { file: string; start?: number; end?: number } {
  const m = /^(.*?):(\d+)(?:\s*[-–]\s*(\d+))?(?:,.*)?$/.exec(ref.trim());
  if (!m) return { file: ref.trim() };
  return { file: m[1], start: Number(m[2]), end: m[3] ? Number(m[3]) : undefined };
}

/** GitHub permalink for a file (and optional line range) at the paper-era commit. */
export function codeUrl(file: string, start?: number, end?: number): string {
  const path = file.replace(/^PE\//, '');
  const lines = start ? `#L${start}${end && end !== start ? `-L${end}` : ''}` : '';
  return `${CODE.url}/blob/${CODE.commitFull}/${path}${lines}`;
}

/** Paper PDF link, jumping to the page in "p.5 §3.2" when present. */
export function paperUrl(ref?: string): string {
  const m = ref ? /p\.?\s*(\d+)/.exec(ref) : null;
  return m ? `${PAPER.pdf}#page=${m[1]}` : PAPER.url;
}
