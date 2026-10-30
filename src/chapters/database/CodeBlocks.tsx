/**
 * The chapter's code-level blocks (CodeRefs and "In the code" boxes). They do not depend on the
 * focus row, so they are memoized: walking the database never re-highlights the code listings.
 */
import { memo } from 'react';
import { Cite, CodeRef } from '@/components';
import { InTheCode } from './InTheCode';
import { CELLS, EDGES, KEYS, META, TEMPORAL } from './snippets';

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
            <code>f2p_nbr_idxs</code> (479). The branch ends with a <code>continue</code> (line 532, below this excerpt), so a key never makes a cell{' '}
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
        The primary-key column is skipped outright <Cite kind="code" ref="rustler/src/pre.rs:439-442" />. A foreign-key column makes no token
        either: each non-null value becomes an entry in the row’s parent list plus a pair of edges{' '}
        <Cite kind="code" ref="rustler/src/pre.rs:444-533" />. Every remaining column is a feature column, and each of its non-missing cells
        becomes a cell record <Cite kind="code" ref="rustler/src/pre.rs:552" />, typed by how the column is stored{' '}
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
            Integer and 32-bit float values are cast to <code>Float64</code> and become numbers. A null matches the empty arm on line 552 and
            leaves no trace. The boolean arm shown here is one of four (number, datetime and text follow in lines 565-605){' '}
            <Cite kind="spec" ref="§3.2" />
            <Cite kind="spec" ref="§3.5" />.
          </>
        }
      />
    </InTheCode>
  );
});

export const TimeFilterCode = memo(function TimeFilterCode() {
  return (
    <InTheCode title="how the sampler applies the time rule">
      <CodeRef
        title="Preview of chapter 6: parents unfiltered, children checked against the seed’s time"
        file={TEMPORAL.file}
        lines={TEMPORAL.lines}
        code={TEMPORAL.code}
        highlightLines={[406, 407, 408, 421, 422, 423, 424]}
        defaultOpen
        note={
          <>
            Lines 406-408 push every parent with no time test. Lines 421-424 drop a child whose timestamp is later than the seed’s{' '}
            <Cite kind="spec" ref="§4.6" />.
          </>
        }
      />
    </InTheCode>
  );
});
