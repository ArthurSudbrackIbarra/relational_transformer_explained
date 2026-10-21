/**
 * The chapter's schema figure. Wide screens: one SchemaDiagram with F→P arrows.
 * Narrow screens (phones): the same cards stacked one per row, parents first, so
 * no table hides in a horizontal scroller; each foreign-key row still names the
 * table it points to ("→ customers"). Nothing is selected on load, so no table
 * starts dimmed.
 */
import { memo, useMemo } from 'react';
import { SchemaDiagram, type SchemaTableSpec } from '@/components';
import { useElementSize } from '@/components/viz/useSize';

/** Gap between the two card columns, where the F→P arrows run (wider than the default, so they stay apart). */
const GUTTER = 112;
/** Below this container width the two-column diagram would need a horizontal scroller. */
const STACK_BELOW = 2 * 244 + GUTTER + 40;

export const SchemaOverview = memo(function SchemaOverview({ tables, onOpen }: { tables: SchemaTableSpec[]; onOpen: (table: string) => void }) {
  const [ref, size] = useElementSize<HTMLDivElement>();
  const narrow = size.width > 0 && size.width < STACK_BELOW;
  // parents (no foreign key) first, then the tables that point to them, as in the wide layout
  const stacked = useMemo(
    () =>
      [...tables].sort(
        (a, b) => Number(Object.keys(a.schema.fkeys).length > 0) - Number(Object.keys(b.schema.fkeys).length > 0),
      ),
    [tables],
  );
  return (
    <div ref={ref} className="min-w-0">
      {narrow ? (
        <>
          <div className="grid gap-3">
            {stacked.map((t) => (
              <SchemaDiagram key={t.schema.name} tables={[t]} onTableClick={onOpen} className="[&>p]:hidden" />
            ))}
          </div>
          <p className="!mt-2.5 !mb-0 px-1 text-[15px] leading-snug text-ink-2">
            On a narrow screen the tables are stacked. Each foreign key names the table it points to (→), the F→P direction.
          </p>
        </>
      ) : (
        <SchemaDiagram tables={tables} onTableClick={onOpen} gutter={GUTTER} tintArrows />
      )}
    </div>
  );
});
