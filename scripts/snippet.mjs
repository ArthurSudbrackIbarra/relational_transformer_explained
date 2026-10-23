#!/usr/bin/env node
/**
 * Print a verbatim code excerpt as a TypeScript constant, for CodeRef in chapters and Q&A answers.
 *
 *   node scripts/snippet.mjs <repo-checkout> <file> <start> <end> <NAME>
 *   node scripts/snippet.mjs ~/rt-2a899c8 rustler/src/fly.rs 287 296 FLY_SEEDS
 *
 * Lines are 1-indexed and inclusive. Never edit the printed `code` by hand: CodeRef shows it as the
 * official code, with a permalink to the same lines.
 */
import { readFileSync } from 'node:fs';
import { join } from 'node:path';

const [repo, file, a, b, name] = process.argv.slice(2);
if (!repo || !file || !a || !b || !name) {
  console.error('usage: node scripts/snippet.mjs <repo-checkout> <file> <start> <end> <NAME>');
  process.exit(1);
}
const start = Number(a);
const end = Number(b);
const lines = readFileSync(join(repo, file), 'utf8').split('\n');
if (!(start >= 1 && end >= start && end <= lines.length)) {
  console.error(`bad range ${start}-${end} (file has ${lines.length} lines)`);
  process.exit(1);
}
const code = lines.slice(start - 1, end).join('\n');
console.log(`const ${name} = {\n  file: ${JSON.stringify(file)},\n  start: ${start},\n  end: ${end},\n  code: ${JSON.stringify(code)},\n};`);
