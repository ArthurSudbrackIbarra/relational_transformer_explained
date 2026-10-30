/**
 * The static icon and link-preview image that scripts/social-images.ts writes into public/: they must match
 * the current mark and the og:image size declared in index.html.
 */
import fs from 'node:fs';
import { describe, expect, test } from 'vitest';
import { logoMarkSvg } from '@/components/shell/logoMark';

const file = (name: string) => new URL(`../${name}`, import.meta.url);

describe('site icon and preview image', () => {
  test('public/favicon.svg is the current mark (rerun scripts/social-images.ts after changing it)', () => {
    expect(fs.readFileSync(file('public/favicon.svg'), 'utf8').trim()).toBe(logoMarkSvg());
  });

  test('og-image.png has the size index.html declares, and stays small enough for WhatsApp (< 300 KB)', () => {
    const png = fs.readFileSync(file('public/og-image.png'));
    const html = fs.readFileSync(file('index.html'), 'utf8');
    const meta = (p: string) => Number(html.match(new RegExp(`property="${p}" content="(\\d+)"`))?.[1]);
    expect([png.readUInt32BE(16), png.readUInt32BE(20)]).toEqual([meta('og:image:width'), meta('og:image:height')]);
    expect(png.length).toBeLessThan(300 * 1024);
  });
});
