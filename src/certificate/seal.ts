/**
 * Certificate IDs, computed in the browser: the site has no server.
 *
 * An ID is 20 characters of Crockford base32 (100 bits). The first 32 bits hold the facts of the
 * certificate (the day the final quiz was passed, the score, the quiz size and the number of chapters),
 * XOR-masked with a hash of the name so that IDs do not look alike; the other 68 bits are a check, the
 * first bits of SHA-256 over the name and the facts. From a name and an ID anyone can recover the facts
 * and confirm them, and changing any printed detail breaks the check. The scheme is in the site's own
 * code, though, so someone who reads it can compute a matching ID for any name. The verify page says
 * so: this is an honor-system certificate, not a credential signed by a server.
 *
 * Never change the scheme: every certificate already issued would stop verifying (seal.test.ts pins it).
 */
import { SITE_URL } from '@/components/prose/sources';
import { passScore } from './quiz';

export interface CertFacts {
  /** The name as printed (cleanName). */
  name: string;
  /** Day the final quiz was passed, YYYY-MM-DD, on the reader's calendar. */
  date: string;
  /** Right answers. */
  score: number;
  /** Questions in the quiz. */
  total: number;
  /** Chapters in the course when the certificate was issued. */
  chapters: number;
}

export type Verdict =
  | { ok: true; facts: CertFacts; id: string }
  | { ok: false; reason: 'name' | 'id-format' | 'mismatch' };

/** Longest name the certificate prints. */
export const NAME_MAX = 70;

const EPOCH = Date.UTC(2026, 0, 1);
const DAY_MS = 86_400_000;
const ALPHABET = '0123456789ABCDEFGHJKMNPQRSTVWXYZ';
const ID_CHARS = 20;
const FACT_BITS = 32n;
const CHECK_BITS = 68n;

/** The name as printed: Unicode NFC, runs of whitespace as one space, trimmed. */
export function cleanName(raw: string): string {
  return raw.normalize('NFC').replace(/\s+/g, ' ').trim();
}

/** Why a name cannot be printed, or null. */
export function nameProblem(raw: string): string | null {
  const name = cleanName(raw);
  if (!name) return 'Enter your name.';
  if ([...name].length > NAME_MAX) return `Use at most ${NAME_MAX} characters.`;
  if (!/\p{L}/u.test(name)) return 'A name needs at least one letter.';
  return null;
}

/** Today on the reader's calendar, YYYY-MM-DD. */
export function todayIso(now = new Date()): string {
  const p = (n: number) => String(n).padStart(2, '0');
  return `${now.getFullYear()}-${p(now.getMonth() + 1)}-${p(now.getDate())}`;
}

/** "29 October 2026". */
export function formatCertDate(iso: string): string {
  return new Intl.DateTimeFormat('en-GB', { day: 'numeric', month: 'long', year: 'numeric', timeZone: 'UTC' }).format(
    new Date(`${iso}T00:00:00Z`),
  );
}

/** "ABCDE-FGHJK-MNPQR-STVWX". */
export function formatId(raw: string): string {
  return raw.match(/.{1,5}/g)!.join('-');
}

/** An ID as typed (any case, with or without dashes; O, I and L read as 0, 1 and 1), or null. */
export function parseId(input: string): bigint | null {
  const s = input.toUpperCase().replace(/[\s-]/g, '').replace(/O/g, '0').replace(/[IL]/g, '1');
  if (s.length !== ID_CHARS) return null;
  let v = 0n;
  for (const ch of s) {
    const i = ALPHABET.indexOf(ch);
    if (i < 0) return null;
    v = (v << 5n) | BigInt(i);
  }
  return v;
}

/** A typed ID in its printed form, or the input trimmed when it is not a well-formed ID. */
export function normalizeId(input: string): string {
  const bits = parseId(input);
  return bits === null ? input.trim() : formatId(base32(bits));
}

/** The ID of a certificate. Throws on facts that no certificate can have. */
export async function certificateId(f: CertFacts): Promise<string> {
  const day = dayNumber(f.date);
  if (day === null) throw new Error(`certificate date out of range: ${f.date}`);
  if (nameProblem(f.name)) throw new Error('certificate name is not printable');
  if (!plausible(f.score, f.total, f.chapters)) throw new Error('certificate score or size out of range');
  const key = nameKey(f.name);
  const facts = (BigInt(day) << 18n) | (BigInt(f.score) << 12n) | (BigInt(f.total) << 6n) | BigInt(f.chapters);
  const bits = ((facts ^ (await maskOf(key))) << CHECK_BITS) | (await checkOf(key, facts));
  return formatId(base32(bits));
}

/**
 * Checks a name and an ID. Case and extra spaces in the name do not matter. `today` bounds the date,
 * with a day of slack for time zones.
 */
export async function verifyCertificate(name: string, id: string, today = todayIso()): Promise<Verdict> {
  if (nameProblem(name)) return { ok: false, reason: 'name' };
  const bits = parseId(id);
  if (bits === null) return { ok: false, reason: 'id-format' };
  const key = nameKey(name);
  const facts = (bits >> CHECK_BITS) ^ (await maskOf(key));
  if ((bits & ((1n << CHECK_BITS) - 1n)) !== (await checkOf(key, facts))) return { ok: false, reason: 'mismatch' };
  const day = Number(facts >> 18n);
  const score = Number((facts >> 12n) & 63n);
  const total = Number((facts >> 6n) & 63n);
  const chapters = Number(facts & 63n);
  // a matching check over facts no certificate can have was crafted by hand: refuse it too
  const latest = dayNumber(today);
  if (!plausible(score, total, chapters) || (latest !== null && day > latest + 1)) return { ok: false, reason: 'mismatch' };
  return { ok: true, facts: { name: cleanName(name), date: isoDay(day), score, total, chapters }, id: formatId(base32(bits)) };
}

/** The public verification link printed on the certificate (and in its QR code). */
export function verifyUrl(name: string, id: string, base = SITE_URL): string {
  return `${base}#${verifyPath(name, id)}`;
}

/** The in-app route of a certificate's verification page. */
export function verifyPath(name: string, id: string): string {
  return `/certificate/verify?${new URLSearchParams({ name: cleanName(name), id }).toString()}`;
}

function plausible(score: number, total: number, chapters: number): boolean {
  const int6 = (n: number) => Number.isInteger(n) && n >= 0 && n < 64;
  return int6(score) && int6(total) && int6(chapters) && total >= 1 && chapters >= 1 && score <= total && score >= passScore(total);
}

/** Case does not matter when verifying. */
function nameKey(name: string): string {
  return cleanName(name).toLowerCase();
}

/** Days since 2026-01-01 for a valid YYYY-MM-DD in the 14-bit range, else null. */
function dayNumber(iso: string): number | null {
  const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(iso);
  if (!m) return null;
  const t = Date.UTC(Number(m[1]), Number(m[2]) - 1, Number(m[3]));
  if (new Date(t).toISOString().slice(0, 10) !== iso) return null; // 2026-02-30 and the like
  const day = (t - EPOCH) / DAY_MS;
  return day >= 0 && day < 2 ** 14 ? day : null;
}

function isoDay(day: number): string {
  return new Date(EPOCH + day * DAY_MS).toISOString().slice(0, 10);
}

function base32(v: bigint): string {
  let s = '';
  for (let i = ID_CHARS - 1; i >= 0; i--) s += ALPHABET[Number((v >> BigInt(5 * i)) & 31n)];
  return s;
}

async function maskOf(key: string): Promise<bigint> {
  return leadingBits(await sha256(`rt-explained certificate v1 mask\n${key}`), FACT_BITS);
}

async function checkOf(key: string, facts: bigint): Promise<bigint> {
  return leadingBits(await sha256(`rt-explained certificate v1 check\n${key}\n${facts}`), CHECK_BITS);
}

async function sha256(text: string): Promise<Uint8Array> {
  return new Uint8Array(await crypto.subtle.digest('SHA-256', new TextEncoder().encode(text)));
}

function leadingBits(bytes: Uint8Array, n: bigint): bigint {
  let v = 0n;
  for (const b of bytes) v = (v << 8n) | BigInt(b);
  return v >> (BigInt(bytes.length * 8) - n);
}
