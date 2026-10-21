/**
 * Export every string RT's text encoder must embed, across all four dummy DBs
 * (BUILD_PLAN §4 step 1; spec §3.6).
 *
 * pre.rs interns ONE string table per DB (pre.rs:406-437, 596-597) holding:
 *   - every table name, DB tables AND task tables (pre.rs:424-427);
 *   - the phrase "<col> of <table>" for EVERY column, PK/FK columns and task-table columns
 *     included (they get an index even though PK/FK never produce cells) (pre.rs:432-437);
 *   - every distinct string cell value, embedded raw with no column context (pre.rs:596-597).
 * Identical strings share one embedding, so we export the union as one sorted unique list
 * (the order python/embed.py preserves) plus a per-DB breakdown and the column-phrase metadata
 * used for the 2-D PCA map.
 *
 * Usage: npx tsx scripts/export-strings.ts   →   python/data/strings.json
 */
import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { DB_NAMES, columnPhrase, getDatabase } from '../src/engine/db/index.ts';
import type { RawTable } from '../src/engine/types.ts';

interface PhraseMeta {
  text: string;
  db: string;
  table: string;
  col: string;
  dtype: string;
  /** 'Db' for DB tables, 'Task' for task tables (all splits share the phrase, spec §2.2). */
  tableKind: 'Db' | 'Task';
  role: 'pk' | 'fk' | 'time' | 'feature' | 'target';
}

const sortUnique = (xs: Iterable<string>) => [...new Set(xs)].sort();

const all = new Set<string>();
const byDb: Record<string, { tableNames: string[]; columnPhrases: string[]; values: string[]; count: number }> = {};
const phrases: PhraseMeta[] = [];

for (const name of DB_NAMES) {
  const db = getDatabase(name);
  const tableNames = new Set<string>();
  const colPhrases = new Set<string>();
  const values = new Set<string>();

  const visit = (table: RawTable, kind: 'Db' | 'Task', targetCol?: string) => {
    const s = table.schema;
    tableNames.add(s.name);
    for (const c of s.columns) {
      const text = columnPhrase(c.name, s.name);
      if (!colPhrases.has(text)) {
        colPhrases.add(text);
        const role =
          c.name === s.pkey ? 'pk' : c.name in s.fkeys ? 'fk' : c.name === s.timeCol ? 'time' : c.name === targetCol ? 'target' : 'feature';
        phrases.push({ text, db: name, table: s.name, col: c.name, dtype: c.dtype, tableKind: kind, role });
      }
      if (c.dtype !== 'str') continue;
      for (const r of table.rows) {
        const v = r[c.name];
        if (typeof v === 'string') values.add(v);
      }
    }
  };

  for (const t of db.tables) visit(t, 'Db');
  for (const task of db.tasks) {
    if (task.kind !== 'forecast' || !task.tables) continue;
    // the three splits share one table name and one phrase per column; visit all for values
    for (const split of ['Train', 'Val', 'Test'] as const) visit(task.tables[split], 'Task', task.targetCol);
  }

  const union = new Set([...tableNames, ...colPhrases, ...values]);
  for (const s of union) all.add(s);
  byDb[name] = {
    tableNames: sortUnique(tableNames),
    columnPhrases: sortUnique(colPhrases),
    values: sortUnique(values),
    count: union.size,
  };
}

const strings = sortUnique(all);
for (const s of strings) {
  if (s.length === 0) throw new Error('empty string in text table');
}

const out = {
  generatedBy: 'scripts/export-strings.ts',
  note: 'Sorted unique union over all DBs of table names, "<col> of <table>" phrases and string cell values (spec §3.6).',
  count: strings.length,
  strings,
  byDb,
  columnPhrases: phrases,
};

const path = fileURLToPath(new URL('../python/data/strings.json', import.meta.url));
mkdirSync(dirname(path), { recursive: true });
writeFileSync(path, JSON.stringify(out, null, 1) + '\n');

console.log(`wrote ${path}`);
console.log(`  ${strings.length} unique strings, ${phrases.length} column phrases`);
for (const [name, d] of Object.entries(byDb)) {
  console.log(`  ${name}: ${d.tableNames.length} table names, ${d.columnPhrases.length} column phrases, ${d.values.length} values (${d.count} unique)`);
}
