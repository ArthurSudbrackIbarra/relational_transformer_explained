/**
 * Browse the live tables (TableView), one tab per table. Clicking a row makes it
 * the chapter's focus row (local state; the global simulation seed is untouched).
 * Rows directly linked to the focus row (its parents and children, spec §1.3)
 * are highlighted, and can be filtered.
 *
 * While the browser is far off screen it keeps showing the tab and focus it had
 * when it left the viewport, so walking the database in the figures further down
 * never re-renders (and re-measures) a whole table nobody is looking at.
 */
import { ChevronLeft, ChevronRight, Crosshair } from 'lucide-react';
import { memo, useMemo, useRef, useState } from 'react';
import { Button, IconButton, NumberField, Segmented, Switch, TableSwatch, TableView, formatRaw } from '@/components';
import type { RawDatabase } from '@/engine/types';
import { useNearViewport } from './hooks';
import { nullCounts, rowCaption, rowLabel, sameRow, type LinkIndex, type RowRef } from './model';

export interface TableBrowserProps {
  db: RawDatabase;
  idx: LinkIndex;
  slots: Record<string, number>;
  tab: string;
  onTab: (t: string) => void;
  focus: RowRef;
  onFocus: (r: RowRef) => void;
  example: RowRef;
}

export const TableBrowser = memo(function TableBrowser(props: TableBrowserProps) {
  const [box, near] = useNearViewport<HTMLDivElement>('300px 0px');
  const frozen = useRef({ tab: props.tab, focus: props.focus });
  if (near) frozen.current = { tab: props.tab, focus: props.focus };
  return (
    <div ref={box} className="min-w-0">
      <BrowserBody {...props} tab={frozen.current.tab} focus={frozen.current.focus} />
    </div>
  );
});

const BrowserBody = memo(function BrowserBody({ db, idx, slots, tab, onTab, focus, onFocus, example }: TableBrowserProps) {
  const table = idx.table(tab) ?? db.tables[0];
  const name = table.schema.name;
  const [linkedOnly, setLinkedOnly] = useState(false);
  const [hover, setHover] = useState<number | null>(null);

  // rows of this table directly linked to the focus row (parents or children)
  const linked = useMemo(() => {
    const out = new Set<number>();
    for (const g of idx.childrenOf(focus)) if (g.childTable === name) g.rows.forEach((r) => out.add(r));
    for (const p of idx.parentsOf(focus)) if (p.parentTable === name && p.parentRow !== null) out.add(p.parentRow);
    return out;
  }, [idx, focus, name]);

  const focusHere = focus.table === name;
  const highlight = useMemo(() => {
    const s = new Set(linked);
    if (focusHere) s.add(focus.row);
    return s;
  }, [linked, focusHere, focus.row]);

  const nulls = useMemo(() => nullCounts(table), [table]);
  const nullText = Object.entries(nulls)
    .filter(([, n]) => n > 0)
    .map(([c, n]) => `${c}: ${n} missing`)
    .join(' · ');

  const preview = hover ?? (focusHere ? focus.row : null);
  const previewRef = preview !== null ? { table: name, row: preview } : null;

  const n = table.rows.length;
  const go = (r: number) => onFocus({ table: name, row: Math.max(0, Math.min(n - 1, Math.round(r))) });
  // tables that do hold rows linked to the focus row (for the hint when this tab has none)
  const linkedTables = useMemo(() => {
    const out: string[] = [];
    for (const p of idx.parentsOf(focus)) if (p.parentRow !== null && !out.includes(p.parentTable)) out.push(p.parentTable);
    for (const g of idx.childrenOf(focus)) if (g.rows.length > 0 && !out.includes(g.childTable)) out.push(g.childTable);
    return out.filter((t) => t !== name);
  }, [idx, focus, name]);

  return (
    <div className="min-w-0">
      <div className="mb-3 flex flex-wrap items-end gap-x-4 gap-y-3">
        <Segmented
          label="Table"
          value={name}
          onChange={(t) => {
            onTab(t);
            setLinkedOnly(false);
          }}
          options={db.tables.map((t) => ({
            value: t.schema.name,
            label: <TableSwatch slot={slots[t.schema.name] ?? 0} label={t.schema.name} size={9} />,
          }))}
        />
        <div className="flex items-end gap-1">
          {/* The field shows the focus row only when it is in this table; otherwise it is a "go to row" box. */}
          <NumberField
            label={focusHere ? 'Focus row' : 'Go to row'}
            value={focusHere ? focus.row : Number.NaN}
            placeholder={`0–${n - 1}`}
            min={0}
            max={n - 1}
            onChange={go}
            className="w-[104px]"
          />
          <IconButton label="Previous row" size="sm" disabled={!focusHere || focus.row <= 0} onClick={() => go(focus.row - 1)}>
            <ChevronLeft size={16} />
          </IconButton>
          <IconButton label="Next row" size="sm" disabled={!focusHere || focus.row >= n - 1} onClick={() => go(focus.row + 1)}>
            <ChevronRight size={16} />
          </IconButton>
        </div>
        {linked.size > 0 ? (
          <Switch
            checked={linkedOnly}
            onChange={setLinkedOnly}
            label={
              <>
                Only rows linked to <span className="font-mono">{rowLabel(focus)}</span>
              </>
            }
          />
        ) : (
          <p className="!m-0 flex flex-wrap items-center gap-x-1.5 gap-y-1 self-center text-[15px] leading-snug text-ink-2">
            <span>
              No {name} row is linked to <span className="font-mono text-ink">{rowLabel(focus)}</span>
              {linkedTables.length > 0 ? '; its links are in' : '.'}
            </span>
            {linkedTables.map((t) => (
              <Button key={t} size="sm" variant="ghost" onClick={() => onTab(t)}>
                <TableSwatch slot={slots[t] ?? 0} label={t} size={9} />
              </Button>
            ))}
          </p>
        )}
        {!sameRow(focus, example) && (
          <Button size="sm" variant="ghost" icon={<Crosshair size={14} />} onClick={() => onFocus(example)}>
            Back to {rowLabel(example)}
          </Button>
        )}
      </div>

      <TableView
        schema={table.schema}
        rows={table.rows}
        slot={slots[name] ?? 0}
        highlightRows={highlight}
        rowFilter={linkedOnly && linked.size > 0 ? [...linked].sort((a, b) => a - b) : undefined}
        onRowClick={(r) => onFocus({ table: name, row: r })}
        onRowHover={setHover}
        scrollToRow={focusHere ? focus.row : linked.size ? Math.min(...linked) : null}
        maxHeight={330}
        footer={nullText ? `∅ = missing value (${nullText}). A missing cell produces no token.` : 'No missing values in this table.'}
      />

      <div className="mt-2.5 min-h-[44px] rounded-lg bg-surface-2 px-3 py-2 text-[15.5px] leading-snug text-ink-2" aria-live="polite">
        {previewRef ? (
          <LinkSummary db={db} idx={idx} at={previewRef} />
        ) : linked.size > 0 ? (
          <>
            <span className="font-semibold text-ink">{linked.size}</span> {name} row{linked.size === 1 ? '' : 's'} linked to{' '}
            <span className="font-mono text-ink">{rowLabel(focus)}</span> are highlighted. Hover a row to see its links; click it to follow it below.
          </>
        ) : (
          <>
            {linkedTables.length === 0 && focus.table !== name ? (
              <>
                {name} and {focus.table} only meet through another table.{' '}
              </>
            ) : null}
            Hover a row to see its links; click it to make it the focus row.
          </>
        )}
      </div>
    </div>
  );
});

/** One-line description of a row's links: "orders[3] → customers[30] (…) · products[17] (…); referenced by …". */
export function LinkSummary({ db, idx, at }: { db: RawDatabase; idx: LinkIndex; at: RowRef }) {
  const parents = idx.parentsOf(at);
  const groups = idx.childrenOf(at);
  const cap = rowCaption(db, at);
  return (
    <span>
      <span className="font-mono font-semibold text-ink">{rowLabel(at)}</span>
      {cap && <span className="text-ink-2"> ({cap})</span>}
      {parents.length > 0 && (
        <>
          {' '}
          points to{' '}
          {parents.map((p, i) => (
            <span key={p.fkCol}>
              {i > 0 && (i === parents.length - 1 ? ' and ' : ', ')}
              {p.parentRow === null ? (
                <span className="text-muted">no {p.parentTable} row ({p.fkCol} = {formatRaw(null)})</span>
              ) : (
                <span className="font-mono text-ink">{rowLabel({ table: p.parentTable, row: p.parentRow })}</span>
              )}
            </span>
          ))}
        </>
      )}
      {parents.length > 0 && groups.length > 0 ? ';' : ''}
      {groups.length > 0 && (
        <>
          {' '}
          is referenced by{' '}
          {groups.map((g, i) => (
            <span key={g.childTable + g.fkCol}>
              {i > 0 && (i === groups.length - 1 ? ' and ' : ', ')}
              <span className="num text-ink">{g.rows.length}</span> {g.childTable}
              {groups.filter((x) => x.childTable === g.childTable).length > 1 ? ` (via ${g.fkCol})` : ''}
            </span>
          ))}
        </>
      )}
      {parents.length === 0 && groups.length === 0 && ' has no links.'}
      {parents.length > 0 || groups.length > 0 ? '.' : ''}
    </span>
  );
}
