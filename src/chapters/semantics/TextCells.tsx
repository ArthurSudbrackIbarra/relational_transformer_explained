/**
 * Figure "Two vectors per text token": for a text cell of the live context, the sampler copies two
 * rows of the embedding table into the batch: the column phrase (col_name_values) and the raw
 * value string (text_values), fly.rs:472-486. The value is embedded with no column context
 * (pre.rs:596-597, embed.py:65-68, spec §3.6).
 */
import { useMemo, useState } from 'react';
import { SemBadge, TableSwatch, VectorStrip, cx } from '@/components';
import type { EmbeddingStore } from '@/engine/embeddings';
import type { Token } from '@/engine/types';
import { getPreprocessed } from '@/state/hooks';
import { Phrase, Quoted, Wordpieces } from './bits';
import { DB_SHORT, cosine, f3, nearest, wordpiecesOf, type Catalog } from './data';
import { rowRef, type SemanticsLive } from './live';

export function TextCells({ live, store }: { live: SemanticsLive; store: EmbeddingStore }) {
  const { textTokens, entityNode } = live;
  const defaultPos = useMemo(() => {
    const ofEntity = textTokens.filter((t) => entityNode && t.nodeIdx === entityNode.nodeIdx);
    const pick = ofEntity.find((t) => t.colName === 'country') ?? ofEntity[ofEntity.length - 1] ?? textTokens[0];
    return pick?.pos ?? -1;
  }, [textTokens, entityNode]);
  const [chosen, setChosen] = useState<number | null>(null);
  const pos = chosen !== null && textTokens.some((t) => t.pos === chosen) ? chosen : defaultPos;
  const tok = textTokens.find((t) => t.pos === pos);

  // chips: one per distinct (phrase, value) in sequence order, at most 14
  const chips = useMemo(() => {
    const seen = new Set<string>();
    const out: Token[] = [];
    for (const t of textTokens) {
      const k = `${t.colName}|${t.tableName}|${String(t.raw)}`;
      if (seen.has(k)) continue;
      seen.add(k);
      out.push(t);
    }
    // keep the default first, then a spread of columns
    out.sort((a, b) => (a.pos === defaultPos ? -1 : b.pos === defaultPos ? 1 : a.pos - b.pos));
    return out.slice(0, 10);
  }, [textTokens, defaultPos]);

  const phrase = tok ? `${tok.colName} of ${tok.tableName}` : '';
  const value = tok && typeof tok.raw === 'string' ? tok.raw : '';
  const ePhrase = phrase && store.has(phrase) ? store.get(phrase) : null;
  const eValue = value && store.has(value) ? store.get(value) : null;
  const neighbors = useMemo(() => (value ? nearest(store, value, live.pdb.text, 5) : []), [store, value, live.pdb]);
  const textIdx = tok ? live.pdb.textToIdx.get(value) : undefined;
  const colIdx = tok ? live.pdb.columnIndex[phrase] : undefined;
  const domain = useMemo(() => {
    let m = 0;
    for (const v of [ePhrase, eValue]) if (v) for (let i = 0; i < v.length; i++) m = Math.max(m, Math.abs(v[i]));
    return m || 1;
  }, [ePhrase, eValue]);

  if (!textTokens.length) return <p className="text-[15px] text-ink-2">The live context has no text cells.</p>;

  return (
    <div>
      <div className="mb-4">
        <div className="mb-2 text-[15px] font-medium text-ink-2">Text cells of the live context</div>
        <div className="flex flex-wrap gap-2" role="radiogroup" aria-label="Text cell">
          {chips.map((t) => {
            const on = t.pos === pos;
            return (
              <button
                key={t.pos}
                type="button"
                role="radio"
                aria-checked={on}
                onClick={() => setChosen(t.pos)}
                className={cx(
                  'inline-flex max-w-full min-w-0 items-center gap-1.5 rounded-lg border px-2.5 py-1 text-left text-[14.5px] transition-colors',
                  on ? 'border-accent bg-accent-soft' : 'border-border bg-surface hover:bg-surface-2',
                )}
              >
                <TableSwatch slot={live.slots[t.tableName] ?? 0} />
                <span className="text-ink-2">{t.colName}</span>
                <span className="min-w-0 truncate text-ink" title={String(t.raw)}>
                  {String(t.raw)}
                </span>
              </button>
            );
          })}
        </div>
      </div>

      {tok && (
        <>
          <div className="mb-4 flex flex-wrap items-center gap-x-3 gap-y-1 rounded-lg bg-surface-2 px-3 py-2 text-[15.5px]">
            <span className="text-ink-2">token</span>
            <span className="num text-ink">#{tok.pos}</span>
            <TableSwatch slot={live.slots[tok.tableName] ?? 0} label={rowRef(tok)} />
            <SemBadge sem={tok.sem} size="xs" showLabel />
            <span className="text-ink-2">
              cell <span className="text-ink">{tok.colName}</span> = <Quoted text={value} />
            </span>
          </div>

          <div className="grid gap-4 lg:grid-cols-2">
            <Lane
              field="col_name_values"
              index={colIdx}
              what={<Phrase text={phrase} />}
              tokens={wordpiecesOf(phrase) ?? []}
              vec={ePhrase}
              domain={domain}
              note="Which column of which table: shared by every cell of this column."
            />
            <Lane
              field="text_values"
              index={textIdx}
              what={<Quoted text={value} />}
              tokens={wordpiecesOf(value) ?? []}
              vec={eValue}
              domain={domain}
              note="The raw value string, embedded on its own."
            />
          </div>

          <div className="mt-4 grid gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
            <div className="rounded-lg border border-border px-4 py-3">
              <div className="mb-1 text-[15.5px] font-semibold text-ink">Never embedded</div>
              <p className="!mb-1 text-[15px] text-ink-2 line-through decoration-[var(--muted)]">
                {tok.colName} of {tok.tableName}: {value}
              </p>
              <p className="!mb-0 text-[15px] leading-snug text-ink-2">
                No string joins a value to its column. {store.has(`${tok.colName} of ${tok.tableName}: ${value}`) ? '' : 'It is not in the string table, so it has no vector.'}{' '}
                The two vectors meet only inside the model, after each has been projected and normalized. Cosine between them:{' '}
                <span className="num text-ink">{ePhrase && eValue ? f3(cosine(ePhrase, eValue)) : '—'}</span>.
              </p>
            </div>
            <div className="rounded-lg border border-border px-4 py-3">
              <div className="mb-1.5 text-[15.5px] font-semibold text-ink">
                Closest strings to <Quoted text={value} /> in {DB_SHORT[live.dbName]}’s string table
              </div>
              <ol className="!m-0 grid !list-none gap-1 !p-0">
                {neighbors.map((n) => (
                  <li key={n.text} className="!m-0 flex min-w-0 items-baseline justify-between gap-3 !p-0 text-[15px]">
                    <span className="min-w-0 truncate text-ink">{n.text}</span>
                    <span className="num shrink-0 text-ink-2">{f3(n.cos)}</span>
                  </li>
                ))}
              </ol>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

function Lane({
  field,
  index,
  what,
  tokens,
  vec,
  domain,
  note,
}: {
  field: string;
  index: number | undefined;
  what: React.ReactNode;
  tokens: string[];
  vec: Float32Array | null;
  domain: number;
  note: string;
}) {
  return (
    <div className="min-w-0 rounded-lg border border-border bg-surface px-4 py-3">
      <div className="mb-1 flex flex-wrap items-baseline justify-between gap-x-3">
        <code className="!text-[14.5px]">{field}</code>
        <span className="font-mono text-[14px] text-muted">text[{index ?? '?'}]</span>
      </div>
      <div className="mb-2 text-[16px] leading-snug">
        <span className="text-ink-2">E(</span>
        {what}
        <span className="text-ink-2">)</span>
      </div>
      <Wordpieces tokens={tokens} className="mb-3" animateKey={tokens.join('|')} />
      {vec ? <VectorStrip values={vec} columns={48} cellHeight={11} domain={domain} /> : <p className="text-[15px] text-ink-2">no stored vector</p>}
      <p className="mt-2 !mb-0 text-[14.5px] leading-snug text-ink-2">{note}</p>
    </div>
  );
}

/** Strings that occur as values in two or more columns of one database (one index, one vector). */
export function SharedValues({ catalog, store }: { catalog: Catalog; store: EmbeddingStore }) {
  const items = useMemo(() => {
    const out: { db: keyof Catalog['dbs']; value: string; phrases: string[]; idx: number | undefined }[] = [];
    for (const [name, db] of Object.entries(catalog.dbs) as [keyof Catalog['dbs'], Catalog['dbs'][keyof Catalog['dbs']]][]) {
      const where = new Map<string, Set<string>>();
      for (const t of db.tables)
        for (const c of t.schema.columns) {
          if (c.dtype !== 'str') continue;
          for (const r of t.rows) {
            const v = r[c.name];
            if (typeof v !== 'string') continue;
            if (!where.has(v)) where.set(v, new Set());
            where.get(v)!.add(`${c.name} of ${t.schema.name}`);
          }
        }
      for (const [v, s] of where) if (s.size > 1) out.push({ db: name, value: v, phrases: [...s], idx: undefined });
    }
    // the string-table index of the example shown (pre.rs interns it once per database)
    if (out.length) out[0].idx = getPreprocessed(catalog.dbs[out[0].db]).textToIdx.get(out[0].value);
    return out;
  }, [catalog]);

  if (!items.length) return null;
  const first = items[0];
  const [pa, pb] = first.phrases;
  const cosPhr = store.has(pa) && store.has(pb) ? cosine(store.get(pa), store.get(pb)) : NaN;
  return (
    <div className="rounded-lg border border-border bg-surface px-4 py-3">
      <div className="mb-1.5 text-[15.5px] font-semibold text-ink">One string, one index, in every column</div>
      <p className="!mb-2 text-[15.5px] leading-snug text-ink-2">
        In {DB_SHORT[first.db]}, <Quoted text={first.value} /> is a value of both <Phrase text={pa} /> and <Phrase text={pb} />. pre.rs interns it
        once, as <span className="font-mono text-ink">text[{first.idx ?? '?'}]</span>, so both cells carry the very same value vector. Only their
        column vectors differ (cosine <span className="num text-ink">{f3(cosPhr)}</span>).
      </p>
      <p className="!mb-0 text-[14.5px] text-muted">
        {items.length} such strings across the four databases:{' '}
        {items
          .slice(0, 6)
          .map((i) => `“${i.value}”`)
          .join(', ')}
        {items.length > 6 ? ', …' : ''}
      </p>
    </div>
  );
}
