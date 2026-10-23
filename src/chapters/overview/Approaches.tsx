/**
 * "Four ways to hand a relational database to a model" (paper p.1-2 §1, p.16 App. B, p.25 App. I.2),
 * each illustrated on ShopDB around the running example:
 *  - hand-engineered features for a tabular model;
 *  - one big joined table;
 *  - text for an LLM (JSON in the style of the paper's LLM-baseline prompt, built from the same
 *    sampled context RT reads);
 *  - a graph of rows for a schema-specific GNN.
 * The illustrations only describe the data; none of them is RT behavior.
 */
import { useMemo, useState, type ReactNode } from 'react';
import { Cite, EntityGraph, Segmented, TableSwatch, formatNumber, type GraphEdgeSpec, type GraphNodeSpec } from '@/components';
import { cx } from '@/components/ui/cx';
import { getDatabase, tableColorSlot } from '@/engine/db/registry';
import { getPreprocessed, simulate } from '@/engine/pipeline';
import type { PreprocessedDB, RawDatabase, RawTable, RawValue, SampledContext } from '@/engine/types';
import type { OverviewLive } from './live';
import { int, rawRow, rowLabel, rowTitle, singular } from './live';
import { discoveries } from './pipelineLayout';

type Tab = 'features' | 'join' | 'text' | 'graph';
const DAY = 86_400_000;

interface ShopExample {
  db: RawDatabase;
  pdb: PreprocessedDB;
  ctx: SampledContext;
  seedNode: number;
  entityRow: number; // customers row position
  tMs: number; // seed timestamp
  valRows: RawTable | undefined;
  slots: Record<string, number>;
}

/** The running example if the simulation is on ShopDB customer-churn, else ShopDB's default seed. */
function useShopExample(live: OverviewLive): ShopExample | null {
  return useMemo(() => {
    let db = live.db;
    let pdb = live.pdb;
    let ctx = live.ctx;
    if (!live.isRunningExample) {
      const sim = simulate({ dbName: 'shop', taskId: 'shop/customer-churn' });
      db = sim.db;
      pdb = sim.pdb;
      ctx = sim.ctx;
    } else {
      db = getDatabase('shop');
      pdb = getPreprocessed(db);
    }
    const seed = pdb.nodes[ctx.params.seedNode];
    if (!seed || seed.ts === null || !seed.f2pNbrIdxs.length) return null;
    const ent = pdb.nodes[seed.f2pNbrIdxs[0]];
    const task = db.tasks.find((t) => t.id === 'shop/customer-churn');
    return {
      db,
      pdb,
      ctx,
      seedNode: seed.nodeIdx,
      entityRow: ent.rowPos,
      tMs: seed.ts * 1000,
      valRows: task?.tables?.Val,
      slots: Object.fromEntries([...db.tables.map((t) => t.schema.name), 'customer-churn'].map((n) => [n, tableColorSlot('shop', n)])),
    };
  }, [live.isRunningExample, live.db, live.pdb, live.ctx]);
}

const table = (db: RawDatabase, name: string) => db.tables.find((t) => t.schema.name === name);

// ─────────────────────────────────────────────────────────────────────────────
// 1 · Hand-engineered features
// ─────────────────────────────────────────────────────────────────────────────

function FeaturesView({ ex }: { ex: ShopExample }) {
  const rows = useMemo(() => {
    const orders = table(ex.db, 'orders')!.rows;
    const reviews = table(ex.db, 'reviews')!.rows;
    const customers = table(ex.db, 'customers')!.rows;
    const val = ex.valRows?.rows ?? [];
    const ids = [
      ex.entityRow,
      ...val
        .map((r) => r.customer_id as number)
        .filter((c) => c !== ex.entityRow)
        .slice(0, 4),
    ];
    return ids.map((c) => {
      const mine = orders.filter((o) => o.customer_id === c && (o.order_time as number) <= ex.tMs);
      const last = mine.reduce((m, o) => Math.max(m, o.order_time as number), -Infinity);
      const revs = reviews.filter((r) => r.customer_id === c && (r.review_time as number) <= ex.tMs && typeof r.rating === 'number');
      const label = val.find((r) => r.customer_id === c)?.churn;
      return {
        c,
        name: String(customers[c]?.name ?? `customer ${c}`),
        o30: mine.filter((o) => (o.order_time as number) > ex.tMs - 30 * DAY).length,
        o90: mine.filter((o) => (o.order_time as number) > ex.tMs - 90 * DAY).length,
        since: Number.isFinite(last) ? Math.floor((ex.tMs - last) / DAY) : null,
        rating: revs.length ? revs.reduce((s, r) => s + (r.rating as number), 0) / revs.length : null,
        label: label as RawValue,
      };
    });
  }, [ex]);
  const date = new Date(ex.tMs).toISOString().slice(0, 10);
  return (
    <div>
      <div className="mb-2 text-[15px] text-ink-2">One row per customer on {date}, with features a person chose:</div>
      <div className="overflow-x-auto rounded-lg border border-border">
        <table className="w-full min-w-[520px] border-collapse text-[14.5px]">
          <thead>
            <tr className="border-b border-grid bg-surface-2 text-left text-ink-2">
              <th className="px-3 py-2 font-medium">customer</th>
              <th className="px-3 py-2 text-right font-medium">orders, 30 days</th>
              <th className="px-3 py-2 text-right font-medium">orders, 90 days</th>
              <th className="px-3 py-2 text-right font-medium">days since last order</th>
              <th className="px-3 py-2 text-right font-medium">mean rating given</th>
              <th className="px-3 py-2 text-right font-medium">churn (label)</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.c} className={cx('border-b border-grid/70 last:border-0', r.c === ex.entityRow && 'bg-accent-soft')}>
                <td className="px-3 py-1.5 whitespace-nowrap text-ink">{r.name}</td>
                <td className="num px-3 py-1.5 text-right">{r.o30}</td>
                <td className="num px-3 py-1.5 text-right">{r.o90}</td>
                <td className="num px-3 py-1.5 text-right">{r.since ?? '—'}</td>
                <td className="num px-3 py-1.5 text-right">{r.rating === null ? '∅' : formatNumber(r.rating, 1)}</td>
                <td className="num px-3 py-1.5 text-right">{String(r.label)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// 2 · One joined table
// ─────────────────────────────────────────────────────────────────────────────

function JoinView({ ex }: { ex: ShopExample }) {
  const info = useMemo(() => {
    const orders = table(ex.db, 'orders')!;
    const customers = table(ex.db, 'customers')!;
    const products = table(ex.db, 'products')!;
    const reviews = table(ex.db, 'reviews')!;
    const nCols = customers.schema.columns.length + orders.schema.columns.length + products.schema.columns.length - 2;
    const perCustOrders = new Map<number, number>();
    const perCustReviews = new Map<number, number>();
    for (const o of orders.rows) perCustOrders.set(o.customer_id as number, (perCustOrders.get(o.customer_id as number) ?? 0) + 1);
    for (const r of reviews.rows) perCustReviews.set(r.customer_id as number, (perCustReviews.get(r.customer_id as number) ?? 0) + 1);
    let withReviews = 0;
    for (const [c, n] of perCustOrders) withReviews += n * Math.max(1, perCustReviews.get(c) ?? 0);
    const preview = orders.rows
      .filter((o) => o.customer_id === ex.entityRow)
      .slice(0, 4)
      .map((o) => {
        const c = customers.rows[o.customer_id as number];
        const p = products.rows[o.product_id as number];
        return {
          id: o.order_id as number,
          name: c.name,
          country: c.country,
          time: o.order_time as number,
          qty: o.quantity,
          product: p.name,
          category: p.category,
        };
      });
    return {
      nOrders: orders.rows.length,
      nCols,
      withReviews,
      nReviewCols: reviews.schema.columns.length - 2,
      preview,
    };
  }, [ex]);
  return (
    <div>
      <div className="mb-2 text-[15px] text-ink-2">
        orders ⋈ customers ⋈ products: one row per order, {int(info.nOrders)} rows × {info.nCols} columns. The customer’s columns repeat on every one
        of their orders:
      </div>
      <div className="overflow-x-auto rounded-lg border border-border">
        <table className="w-full min-w-[560px] border-collapse text-[14.5px]">
          <thead>
            <tr className="border-b border-grid bg-surface-2 text-left text-ink-2">
              {['name', 'country', 'order_time', 'quantity', 'product', 'category'].map((h, i) => (
                <th key={h} className="px-3 py-2 font-medium">
                  <span className="inline-flex items-center gap-1.5">
                    <TableSwatch slot={ex.slots[i < 2 ? 'customers' : i < 4 ? 'orders' : 'products'] ?? 0} />
                    <span>{h}</span>
                  </span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {info.preview.map((r) => (
              <tr key={r.id} className="border-b border-grid/70 last:border-0">
                <td className="bg-accent-soft px-3 py-1.5 whitespace-nowrap">{String(r.name)}</td>
                <td className="bg-accent-soft px-3 py-1.5">{String(r.country)}</td>
                <td className="num px-3 py-1.5">{new Date(r.time).toISOString().slice(0, 10)}</td>
                <td className="num px-3 py-1.5">{String(r.qty)}</td>
                <td className="px-3 py-1.5 whitespace-nowrap">{String(r.product)}</td>
                <td className="px-3 py-1.5">{String(r.category)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="!mb-0 mt-3 text-[15.5px] leading-snug text-ink-2">
        Add the reviews as well (another {info.nReviewCols} columns) and every order of a customer must be paired with every review of that customer:{' '}
        <strong className="text-ink">{int(info.withReviews)} rows</strong> from a database of {int(info.nOrders)} orders. Tables that both point to
        the same parent multiply.
      </p>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// 3 · Text for an LLM (paper App. I.2 style)
// ─────────────────────────────────────────────────────────────────────────────

function jsonValue(v: RawValue, dtype: string): unknown {
  if (v === null || (typeof v === 'number' && Number.isNaN(v))) return null;
  if (dtype === 'datetime' && typeof v === 'number') return new Date(v).toISOString().slice(0, 19);
  return v;
}

/** Serialize the sampled rows like the paper's LLM baseline (p.25): parents merged, children appended as lists. */
function serializeContext(ex: ShopExample): { text: string; rows: number } {
  const disc = discoveries(ex.ctx);
  const records = new Map<number, Record<string, unknown>>();
  const root: Record<string, unknown> = {};
  const recordOf = (node: number) => {
    const n = ex.pdb.nodes[node];
    const schema =
      n.tableType === 'Db' ? table(ex.db, n.tableName)?.schema : ex.db.tasks.find((t) => t.name === n.tableName)?.tables?.[n.tableType]?.schema;
    const raw = rawRow(ex.db, ex.pdb, node) ?? {};
    const rec: Record<string, unknown> = {};
    for (const c of schema?.columns ?? []) rec[c.name] = jsonValue(raw[c.name] ?? null, c.dtype);
    return rec;
  };
  for (const d of disc) {
    const rec = recordOf(d.node);
    if (d.from === null) {
      Object.assign(root, rec);
      records.set(d.node, root);
    } else if (d.via === 'parent') {
      const into = records.get(d.from) ?? root;
      for (const [k, v] of Object.entries(rec)) if (!(k in into)) into[k] = v;
      records.set(d.node, into);
    } else {
      const into = records.get(d.from) ?? root;
      const key = `add_${ex.pdb.nodes[d.node].tableName}`;
      const list = (into[key] as unknown[] | undefined) ?? [];
      list.push(rec);
      into[key] = list;
      records.set(d.node, rec);
    }
  }
  // the label to predict is not given to the LLM
  const seed = ex.pdb.nodes[ex.seedNode];
  if (seed) delete root.churn;
  return { text: JSON.stringify(root, null, 2), rows: disc.length };
}

function TextView({ ex }: { ex: ShopExample }) {
  const { text, rows } = useMemo(() => serializeContext(ex), [ex]);
  const lines = text.split('\n');
  return (
    <div>
      <div className="mb-2 text-[15px] text-ink-2">
        The same {int(rows)} sampled rows as JSON, parents merged into their child and children appended as lists, the way the paper serializes the
        context for its LLM baseline <Cite kind="paper" ref="p.25 App. I.2" />:
      </div>
      <pre className="max-h-[300px] overflow-auto rounded-lg border border-border bg-[var(--code-bg)] px-3.5 py-3 font-mono text-[14px] leading-[1.5] text-[var(--code-ink)]">
        {lines.slice(0, 60).join('\n')}
        {lines.length > 60 && `\n  … ${int(lines.length - 60)} more lines`}
      </pre>
      <p className="!mb-0 mt-3 text-[15.5px] leading-snug text-ink-2">
        <strong className="text-ink">{int(text.length)} characters</strong> of text for what RT reads as {int(ex.ctx.numReal)} cell tokens. Our
        reconstruction follows the paper’s description of the format; the authors’ serializer is not part of the paper-era code.
      </p>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// 4 · A graph of rows
// ─────────────────────────────────────────────────────────────────────────────

function GraphView({ ex }: { ex: ShopExample }) {
  const g = useMemo(() => {
    const nodes: GraphNodeSpec[] = [];
    const edges: GraphEdgeSpec[] = [];
    const inCtx = new Set(ex.ctx.visited.map((v) => v.node));
    for (const v of ex.ctx.visited) {
      const n = ex.pdb.nodes[v.node];
      const name = n.tableType === 'Db' ? `${singular(n.tableName)[0]}${n.rowPos}` : `${n.tableType[0]}${n.rowPos}`;
      nodes.push({
        id: v.node,
        label: name,
        table: n.tableName,
        slot: ex.slots[n.tableName] ?? 0,
        title: rowLabel(ex.db, ex.pdb, v.node).long,
      });
      for (const p of n.f2pNbrIdxs) if (inCtx.has(p)) edges.push({ source: v.node, target: p });
    }
    return { nodes, edges };
  }, [ex]);
  const ent = ex.pdb.nodes[ex.seedNode]?.f2pNbrIdxs[0];
  return (
    <div>
      <div className="mb-2 text-[15px] text-ink-2">
        The same rows as a graph: one node per row, one edge per foreign-key link <Cite kind="paper" ref="p.16 App. B" />. Hover a node.
      </div>
      <EntityGraph
        nodes={g.nodes}
        edges={g.edges}
        highlight={ent !== undefined ? [ex.seedNode, ent] : [ex.seedNode]}
        selected={ex.seedNode}
        height={300}
        seed={7}
        title="The sampled rows as an entity graph"
      />
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// The tabbed figure
// ─────────────────────────────────────────────────────────────────────────────

const TABS: { value: Tab; label: string }[] = [
  { value: 'features', label: 'Engineered features' },
  { value: 'join', label: 'One joined table' },
  { value: 'text', label: 'Text for an LLM' },
  { value: 'graph', label: 'A graph network' },
];

const SAYS: Record<Tab, { who: string; says: ReactNode; cost: ReactNode }> = {
  features: {
    who: 'Tabular models',
    says: (
      <>
        “Traditionally, tasks on relational databases have been solved using tabular models, which depend on manual, error-prone, and costly feature
        engineering.” <Cite kind="paper" ref="p.1 §1" />
      </>
    ),
    cost: <>Every column of the table is a human decision: which table, which time window, which statistic. A new database needs new decisions.</>,
  },
  join: {
    who: 'Flattening for tabular foundation models',
    says: (
      <>
        Tabular foundation models “generalize to unseen datasets but cannot capture the rich relational structure”. Flattening the database into a
        single joined table suffers “from scalability issues and distributional mismatch”. <Cite kind="paper" ref="p.2 §1" />
      </>
    ),
    cost: <>The joined table repeats parent rows, and joining two child tables of the same parent multiplies rows.</>,
  },
  text: {
    who: 'Large language models',
    says: (
      <>
        Serializing the database “into text formats such as XML, JSON, or CSV for processing with LLMs” has the same scalability and
        distribution-mismatch problems <Cite kind="paper" ref="p.2 §1" />. With the same sampled context as text, Gemma3-27B reaches a mean zero-shot
        AUROC of 64.6 against RT’s 69.7, “despite taking 10⁵× more inference FLOPs” <Cite kind="paper" ref="p.8 Table 1" />
        <Cite kind="paper" ref="p.3 §1" />.
      </>
    ),
    cost: <>The rows become many more characters of text than they have cells, and every number becomes a string of digits.</>,
  },
  graph: {
    who: 'Relational deep learning (GNNs, graph transformers)',
    says: (
      <>
        RDL “represents a database as a relational entity graph”: each table a node type, each row a node, each primary–foreign key link an edge. “A
        key limitation, however, is that these architectures are schema-specific, which prevents pretraining and fine-tuning on diverse database
        structures.” <Cite kind="paper" ref="p.16 App. B" />
      </>
    ),
    cost: <>The structure is kept, but the model is built for one schema, so it cannot be pretrained on other databases and reused.</>,
  },
};

export function Approaches({ live }: { live: OverviewLive }) {
  const ex = useShopExample(live);
  const [tab, setTab] = useState<Tab>('features');
  if (!ex) return null;
  const who = rowTitle(ex.db, ex.pdb, ex.pdb.nodes[ex.seedNode].f2pNbrIdxs[0]);
  const s = SAYS[tab];
  return (
    <div>
      <Segmented label="Approach" value={tab} onChange={setTab} options={TABS} className="mb-4 max-w-full overflow-x-auto" />
      <div className="grid gap-6 lg:grid-cols-[minmax(0,1.55fr)_minmax(0,1fr)]">
        <div className="min-w-0">
          {tab === 'features' && <FeaturesView ex={ex} />}
          {tab === 'join' && <JoinView ex={ex} />}
          {tab === 'text' && <TextView ex={ex} />}
          {tab === 'graph' && <GraphView ex={ex} />}
        </div>
        <div className="min-w-0 rounded-xl bg-surface-2 px-4 py-3.5">
          <div className="eyebrow mb-1.5">What the paper says</div>
          <div className="text-[16.5px] font-semibold text-ink">{s.who}</div>
          <p className="!mb-0 mt-1.5 text-[16.5px] leading-[1.55] text-ink">{s.says}</p>
          <div className="eyebrow mt-4 mb-1.5">On ShopDB</div>
          <p className="!m-0 text-[16.5px] leading-[1.55] text-ink-2">{s.cost}</p>
          {who && tab === 'features' && (
            <p className="!mb-0 mt-2 text-[15px] leading-snug text-muted">The highlighted row is {who}, the running example.</p>
          )}
        </div>
      </div>
    </div>
  );
}
