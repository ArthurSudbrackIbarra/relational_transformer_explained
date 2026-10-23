/**
 * The chapter's code-level blocks (CodeRefs and "In the code" boxes). They do not depend on the
 * focus row, so they are memoized: walking the database never re-highlights the code listings.
 */
import { memo } from 'react';
import { Cite, CodeRef, TableSwatch } from '@/components';
import { InTheCode } from './InTheCode';
import type { timestampCoverage } from './model';
import { CELLS, EDGES, KEYS, META, TEMPORAL, TS } from './snippets';

export const MetaCode = memo(function MetaCode() {
  return (
    <InTheCode title="how a table declares its keys and time column">
      <p>
        Each table file carries three metadata entries: <code>pkey_col</code>, <code>fkey_col_to_pkey_table</code> and <code>time_col</code>.
        They tell preprocessing which column is the primary key, which columns are foreign keys and to which table, and which column is the
        time column <Cite kind="spec" ref="§3.1" />.
      </p>
      <CodeRef title="Reading the three metadata entries" file={META.file} lines={META.lines} code={META.code} defaultOpen />
    </InTheCode>
  );
});

export const KeysCode = memo(function KeysCode() {
  return (
    <>
      <CodeRef
        title="Keys in preprocessing: the primary key is skipped, each foreign-key value becomes a parent pointer"
        file={KEYS.file}
        lines={KEYS.lines}
        code={KEYS.code}
        highlightLines={[439, 440, 441, 442, 462, 463, 464, 477, 479]}
        note={
          <>
            Lines 439-442: the primary-key column is skipped, so it makes no token and no edge. For a foreign key, a null value is skipped
            (462-464), the parent is <code>ptable_offset + val</code> (477), and its index is pushed onto the row’s parent list{' '}
            <code>f2p_nbr_idxs</code> (479). The <code>continue</code> at the end of the branch (line 532, below this excerpt) means no cell is ever made from a key{' '}
            <Cite kind="spec" ref="§1.2" />.
          </>
        }
      />
      <InTheCode title="one foreign-key value, two edges">
        <p>
          The F→P edge is stored on the child; the P→F edge is appended to the parent’s adjacency list and carries the <em>child’s</em> table and
          timestamp, which the sampler reads later <Cite kind="spec" ref="§1.2" />
          <Cite kind="spec" ref="§1.4" />.
        </p>
        <CodeRef title="F→P on the child, P→F on the parent" file={EDGES.file} lines={EDGES.lines} code={EDGES.code} defaultOpen />
      </InTheCode>
    </>
  );
});

export const CellsCode = memo(function CellsCode() {
  return (
    <InTheCode title="how preprocessing splits a row">
      <p>
        The primary-key column is skipped outright <Cite kind="code" ref="rustler/src/pre.rs:439-442" />. A foreign-key column produces no token
        either; each non-null value becomes an entry in the row’s parent list plus a pair of edges{' '}
        <Cite kind="code" ref="rustler/src/pre.rs:444-533" />. Every remaining column is a feature column. Each of its cells that is neither
        null nor a NaN float becomes a cell record <Cite kind="code" ref="rustler/src/pre.rs:552" />
        <Cite kind="code" ref="rustler/src/pre.rs:566-568" />, whose type follows its storage type{' '}
        <Cite kind="code" ref="rustler/src/pre.rs:543-611" />. Chapter 4 follows those records into numbers.
      </p>
      <CodeRef
        title="Feature columns: one cell record per value that is not missing"
        file={CELLS.file}
        lines={CELLS.lines}
        code={CELLS.code}
        highlightLines={[552]}
        defaultOpen
        note={
          <>
            <code>UInt32</code>, <code>Int32</code>, <code>Int64</code> and <code>Float32</code> values are cast to <code>Float64</code> and
                become numbers. A null matches the empty arm on line 552 and
            leaves no trace. The boolean arm shown here is one of four (number, datetime and text follow in lines 565-605){' '}
            <Cite kind="spec" ref="§3.2" />
            <Cite kind="spec" ref="§3.5" />.
          </>
        }
      />
    </InTheCode>
  );
});

export interface TimestampCodeProps {
  coverage: ReturnType<typeof timestampCoverage>;
  slots: Record<string, number>;
  timeNoFk: { title: string; table: string; timeCol: string; rows: number; withTs: number; local: boolean } | null;
  dbShort: string;
  noFkTables: string[];
}

export const TimestampCode = memo(function TimestampCode({ coverage, slots, timeNoFk, dbShort, noFkTables }: TimestampCodeProps) {
  return (
    <InTheCode title="which rows carry a timestamp">
      <p>
        The code takes a row’s timestamp from its time column, in whole seconds, but it sets it only inside the loop over foreign keys{' '}
        <Cite kind="code" ref="rustler/src/pre.rs:481-493" />. A row therefore gets a timestamp only if its table has a time column, its time value
        is not null, and it has at least one non-null foreign key <Cite kind="spec" ref="§1.4" />. Counted in the live preprocessed graph:
      </p>
      <div className="scroll-quiet mb-3 overflow-x-auto">
        <table className="w-full border-collapse text-[15.5px]">
          <thead>
            <tr className="border-b border-grid text-left align-bottom text-ink-2">
              <th scope="col" className="py-1.5 pr-3 font-semibold">
                table<span className="sm:hidden"> and time column</span>
              </th>
              <th scope="col" className="hidden py-1.5 pr-3 font-semibold sm:table-cell">
                time column
              </th>
              <th scope="col" className="py-1.5 pr-3 text-right font-semibold">
                <abbr title="foreign keys" className="no-underline sm:hidden">
                  FKs
                </abbr>
                <span className="hidden sm:inline">foreign keys</span>
              </th>
              <th scope="col" className="py-1.5 text-right font-semibold">
                <span className="sm:hidden">with timestamp</span>
                <span className="hidden sm:inline">rows with a timestamp</span>
              </th>
            </tr>
          </thead>
          <tbody>
            {coverage.map((c) => (
              <tr key={c.table} className="border-b border-grid/70">
                <td className="py-1.5 pr-3">
                  <TableSwatch slot={slots[c.table] ?? 0} label={c.table} />
                  <div className="text-[14px] whitespace-nowrap text-ink-2 sm:hidden">
                    {c.timeCol ? <span>{c.timeCol}</span> : 'no time column'}
                  </div>
                </td>
                <td className="hidden py-1.5 pr-3 text-[15px] text-ink-2 sm:table-cell">{c.timeCol ?? '—'}</td>
                <td className="num py-1.5 pr-3 text-right text-ink">{c.nFk}</td>
                <td className="num py-1.5 text-right whitespace-nowrap text-ink">
                  {c.withTs.toLocaleString('en-US')} / {c.rows.toLocaleString('en-US')}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {timeNoFk && (
        <p>
          A time column alone is therefore not enough.{' '}
          {timeNoFk.local ? `${dbShort} itself shows the edge case: its` : `${timeNoFk.title}, one of the other databases on this site, shows the edge case: its`}{' '}
          <code>{timeNoFk.table}</code> table declares <code>{timeNoFk.timeCol}</code> as its time column but has no foreign key, so{' '}
          {timeNoFk.withTs === 0 ? 'none' : timeNoFk.withTs} of its {timeNoFk.rows} rows get{timeNoFk.withTs <= 1 ? 's' : ''} a timestamp in the
          preprocessed graph <Cite kind="spec" ref="§1.4" />.
        </p>
      )}
      <CodeRef
        title="A row’s timestamp is set inside the foreign-key loop"
        file={TS.file}
        lines={TS.lines}
        code={TS.code}
        defaultOpen
        note={
          <>
            Nanoseconds divided by 10⁹: whole seconds. Because this runs only for non-null foreign-key values, a table without foreign keys
            {noFkTables.length > 0 ? ` (${noFkTables.join(', ')})` : ''} never gets a timestamp, time column or not <Cite kind="spec" ref="§1.4" />.
          </>
        }
      />
      <CodeRef
        title="Preview of chapter 6: parents unfiltered, children checked against the seed’s time"
        file={TEMPORAL.file}
        lines={TEMPORAL.lines}
        code={TEMPORAL.code}
        highlightLines={[406, 407, 408, 421, 422, 423, 424]}
        note={
          <>
            Lines 406-408 push every parent with no time test. Lines 421-424 drop a child only when both timestamps exist and the child’s is later
            than the seed’s <Cite kind="spec" ref="§4.6" />.
          </>
        }
      />
    </InTheCode>
  );
});
