/**
 * The certificate: its ID scheme (pinned, so certificates already issued keep verifying), the final
 * quiz's integrity and pass mark, and the hand-written PDF.
 */
import { describe, expect, test } from 'vitest';
import { getChapter } from '@/chapters/registry';
import { jpegPdf } from './pdf';
import { FINAL_QUIZ, gradeQuiz, passScore } from './quiz';
import { certificateId, cleanName, nameProblem, parseId, verifyCertificate, verifyUrl } from './seal';

const ADA = { name: 'Ada Lovelace', date: '2026-10-29', score: 14, total: 15, chapters: 17 };

describe('certificate IDs', () => {
  test('the scheme is pinned: changing it would invalidate every issued certificate', async () => {
    expect(await certificateId(ADA)).toBe('68WA9-TYCZQ-454GD-K1Y3E');
    expect(await certificateId({ ...ADA, name: 'Alan Turing' })).toBe('J8ZD4-KS8C6-BEW2T-9TS88');
    expect(await certificateId({ name: 'José Müller-Øster', date: '2027-02-28', score: 12, total: 15, chapters: 17 })).toBe(
      'KNY1F-H6VY1-EX95Z-C0Z74',
    );
  });

  test('a name and an ID give back the facts', async () => {
    const id = await certificateId(ADA);
    expect(await verifyCertificate('Ada Lovelace', id, '2026-10-29')).toEqual({ ok: true, facts: ADA, id });
  });

  test('case, spacing, dashes and look-alike letters do not matter', async () => {
    const id = await certificateId(ADA);
    const typed = id.toLowerCase().replace(/-/g, ' ').replace(/0/g, 'o').replace(/1/g, 'l');
    const v = await verifyCertificate('  ada   LOVELACE ', typed, '2026-10-29');
    expect(v.ok && v.id).toBe(id);
  });

  test('another name, or an edited ID, does not verify', async () => {
    const id = await certificateId(ADA);
    expect(await verifyCertificate('Ada Byron', id, '2026-10-29')).toEqual({ ok: false, reason: 'mismatch' });
    for (let i = 0; i < id.length; i++) {
      if (id[i] === '-') continue;
      const edited = id.slice(0, i) + (id[i] === 'Z' ? 'Y' : 'Z') + id.slice(i + 1);
      expect((await verifyCertificate('Ada Lovelace', edited, '2026-10-29')).ok).toBe(false);
    }
  });

  test('malformed input is told apart from a mismatch', async () => {
    expect(await verifyCertificate('Ada Lovelace', 'ABC', '2026-10-29')).toEqual({ ok: false, reason: 'id-format' });
    expect(await verifyCertificate('Ada Lovelace', 'UUUUU-UUUUU-UUUUU-UUUUU', '2026-10-29')).toEqual({ ok: false, reason: 'id-format' });
    expect(await verifyCertificate('   ', '68WA9-TYCZQ-454GD-K1Y3E')).toEqual({ ok: false, reason: 'name' });
    expect(parseId('68WA9TYCZQ454GDK1Y3E')).not.toBeNull();
  });

  test('a certificate dated after today does not verify', async () => {
    const id = await certificateId(ADA);
    expect((await verifyCertificate('Ada Lovelace', id, '2026-10-28')).ok).toBe(true); // one day of slack
    expect((await verifyCertificate('Ada Lovelace', id, '2026-10-27')).ok).toBe(false);
  });

  test('no ID for facts a certificate cannot have', async () => {
    await expect(certificateId({ ...ADA, score: 11 })).rejects.toThrow(); // below the pass mark
    await expect(certificateId({ ...ADA, date: '2025-12-31' })).rejects.toThrow();
    await expect(certificateId({ ...ADA, date: '2026-02-30' })).rejects.toThrow();
    await expect(certificateId({ ...ADA, name: '1234' })).rejects.toThrow();
  });

  test('names', () => {
    expect(cleanName('  Ada \n  Lovelace ')).toBe('Ada Lovelace');
    expect(nameProblem('')).not.toBeNull();
    expect(nameProblem('x'.repeat(71))).not.toBeNull();
    expect(nameProblem('Grace Hopper')).toBeNull();
  });

  test('verification link', () => {
    expect(verifyUrl('Ada Lovelace', '68WA9-TYCZQ-454GD-K1Y3E')).toBe(
      'https://arthursudbrackibarra.github.io/relational_transformer_explained/#/certificate/verify?name=Ada+Lovelace&id=68WA9-TYCZQ-454GD-K1Y3E',
    );
  });
});

describe('final quiz', () => {
  test('every question has four distinct choices, one answer and chapters that exist', () => {
    const ids = new Set(FINAL_QUIZ.map((q) => q.id));
    expect(ids.size).toBe(FINAL_QUIZ.length);
    for (const q of FINAL_QUIZ) {
      expect(new Set(q.choices).size).toBe(4);
      expect(q.answer).toBeGreaterThanOrEqual(0);
      expect(q.answer).toBeLessThan(4);
      expect(q.chapters.length).toBeGreaterThan(0);
      for (const slug of q.chapters) expect(getChapter(slug), `${q.id}: ${slug}`).toBeDefined();
    }
  });

  test('pass mark: 12 of 15', () => {
    expect(FINAL_QUIZ.length).toBe(15);
    expect(passScore(15)).toBe(12);
    expect(passScore(10)).toBe(8);
    expect(gradeQuiz(FINAL_QUIZ.map((q) => q.answer))).toBe(15);
    expect(gradeQuiz(FINAL_QUIZ.map(() => null))).toBe(0);
  });
});

describe('PDF', () => {
  test('the cross-reference table points at every object', () => {
    const jpeg = new Uint8Array([0xff, 0xd8, 0xff, 0xe0, 1, 2, 3, 0xff, 0xd9]);
    const bytes = jpegPdf(jpeg, { width: 2, height: 1 }, { title: 'Certificate · Ada', author: 'A', subject: 'S' });
    const s = Array.from(bytes, (b) => String.fromCharCode(b)).join('');
    expect(s.startsWith('%PDF-1.4\n')).toBe(true);
    expect(s.endsWith('%%EOF\n')).toBe(true);
    const xref = Number(/startxref\n(\d+)\n/.exec(s)![1]);
    expect(s.slice(xref, xref + 4)).toBe('xref');
    const entries = s.slice(xref).split('\n').slice(3, 9);
    entries.forEach((e, i) => {
      expect(e.length).toBe(19); // 20 bytes with the newline
      const at = Number(e.slice(0, 10));
      expect(s.slice(at, at + `${i + 1} 0 obj`.length)).toBe(`${i + 1} 0 obj`);
    });
    // the image stream holds the JPEG bytes unchanged, with the right /Length
    const start = s.indexOf('stream\n') + 'stream\n'.length;
    expect(Array.from(bytes.slice(start, start + jpeg.length))).toEqual(Array.from(jpeg));
    expect(s).toContain(`/Length ${jpeg.length} `);
  });
});
