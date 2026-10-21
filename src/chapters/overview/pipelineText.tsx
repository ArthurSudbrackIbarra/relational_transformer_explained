/**
 * The narration under the "RT in one picture" figure: one short, live paragraph per step, with
 * citations, plus the chapter chips that open each stage up.
 */
import { ArrowRight } from 'lucide-react';
import type { ReactNode } from 'react';
import { Link } from 'react-router-dom';
import { getChapter, prefetchChapter } from '@/chapters/registry';
import { Cite, Segmented, Term, formatNumber, formatPct } from '@/components';
import { cx } from '@/components/ui/cx';
import type { AttnLayer } from '@/engine/types';
import type { OverviewLive } from './live';
import { int, listJoin, plural, singular } from './live';
import type { Discovery } from './pipelineLayout';
import type { StepId } from './steps';

export function ChapterChips({ slugs, className }: { slugs: string[]; className?: string }) {
  return (
    <div className={cx('flex flex-wrap items-center gap-2', className)}>
      <span className="text-[15px] text-muted">{slugs.length > 1 ? 'Chapters' : 'Chapter'}:</span>
      {slugs.map((slug) => {
        const c = getChapter(slug);
        if (!c) return null;
        return (
          <Link
            key={slug}
            to={`/${slug}`}
            data-plain
            onPointerEnter={() => prefetchChapter(slug)}
            className="group inline-flex items-center gap-1.5 rounded-full border border-border bg-surface px-3 py-1 text-[15px] text-ink transition-colors hover:border-accent hover:text-accent-text"
          >
            <span className="num text-muted group-hover:text-accent-text">{c.number}</span>
            {c.short}
            <ArrowRight aria-hidden="true" size={14} className="transition-transform group-hover:translate-x-0.5" />
          </Link>
        );
      })}
    </div>
  );
}

/** "45 rows: 4 customers, 17 orders, … and 13 customer-churn rows (12 Train, 1 Val)". */
function visitedBreakdown(live: OverviewLive): string {
  const parts: string[] = [];
  const task: Record<string, number> = {};
  for (const [k, n] of live.rowsByTable) {
    if (k.includes(':')) {
      const [, split] = k.split(':');
      task[split] = n;
    } else parts.push(`${int(n)} ${n === 1 ? singular(k) : k}`);
  }
  const splits = (['Train', 'Val', 'Test'] as const).filter((s) => task[s]);
  if (splits.length) {
    const tot = splits.reduce((a, s) => a + task[s], 0);
    parts.push(`${int(tot)} ${live.task.name} ${tot === 1 ? 'row' : 'rows'} (${splits.map((s) => `${task[s]} ${s}`).join(', ')})`);
  }
  return listJoin(parts);
}

export interface StepTextProps {
  step: StepId;
  live: OverviewLive;
  disc: Discovery[];
  allowed: Record<AttnLayer, Set<number>>;
  attnBlock: number;
  nBlocks: number;
  setAttnBlock: (b: number) => void;
  hasWeights: boolean;
  topKeys: { k: number; w: number }[];
}

export function StepText(p: StepTextProps): ReactNode {
  const { live } = p;
  const { db, pdb, task, ctx } = live;
  const who = live.firstName ?? live.entityShort ?? 'the entity';
  const dbShort = db.title.split(' ')[0];

  switch (p.step) {
    case 'db': {
      let rows = 0;
      let cells = 0;
      let links = 0;
      for (const n of pdb.nodes)
        if (n.tableType === 'Db') {
          rows++;
          cells += n.cells.length;
          links += n.f2pNbrIdxs.length;
        }
      return (
        <p className="!m-0">
          <strong>The database.</strong> {dbShort} has {db.tables.length} tables and {int(rows)} rows; each dot above is one row. Rows are linked by
          foreign keys ({int(links)} links in all), for example an order points to its customer and to its product{' '}
          <Cite kind="paper" ref="p.3 §2.1" />. Together the rows hold {int(cells)} feature cells, far more than one context window of {ctx.params.L}{' '}
          tokens can take. Hover a dot to see its row.
        </p>
      );
    }
    case 'task': {
      if (task.kind !== 'forecast') {
        return (
          <p className="!m-0">
            <strong>The seed row.</strong> In this autocomplete task the seed is a database row, {live.seedLabel?.long}, and its{' '}
            <code>{task.targetCol}</code> cell is the one to predict <Cite kind="spec" ref="§2.4" />.
          </p>
        );
      }
      return (
        <p className="!m-0">
          <strong>The task row.</strong>{' '}
          {live.isRunningExample ? (
            <>
              The question “will {live.entityName} place no order in the {task.timedeltaDays ?? 30} days after {live.seedDate}?” is one row of the
              task table <code>{task.name}</code>
            </>
          ) : (
            <>
              The question “{task.description}” for {live.entityName} at {live.seedDate} is one row of the task table <code>{task.name}</code>
            </>
          )}
          : a timestamp, a foreign key to {live.entityShort}, and <code>{task.targetCol} = ?</code>. The task table is attached to the database like
          any other table <Cite kind="paper" ref="p.4 §3.1" />. This row is the <Term id="seed-row">seed row</Term>, and its {task.targetCol} cell is
          the one the model must fill in.
        </p>
      );
    }
    case 'context':
      return (
        <p className="!m-0">
          <strong>The context.</strong> From the seed row, a bounded-width breadth-first search follows key links: parent rows first, then child rows,
          at most w = {ctx.params.w} database rows per visited row (rows of the seed’s own task table are never subsampled), skipping child rows dated
          after the seed <Cite kind="paper" ref="p.4 §3.1" />
          <Cite kind="spec" ref="§4.7" />. Here it visits {plural(p.disc.length, 'row')}: {visitedBreakdown(live)}, and stops when the window holds L
          = {ctx.params.L} cells. Each line joins a row to the row that led to it: solid when the search went up a foreign key to a parent (F→P),
          dashed when it went down to a child (P→F).
        </p>
      );
    case 'tokens': {
      const target = live.target;
      const phrase = target ? pdb.text[target.colNameIdx] : `${task.targetCol} of ${task.name}`;
      const pad = ctx.tokens.length - ctx.numReal;
      const cut = ctx.events.some((e) => e.kind === 'truncated');
      return (
        <p className="!m-0">
          <strong>Cells become tokens.</strong> Every non-missing feature cell of those rows becomes one <Term id="cell-token">token</Term>, seed row
          first: {int(ctx.numReal)} tokens
          {pad > 0 ? ` and ${pad} padding slots` : ''}
          {cut ? (
            <>
              ; the window filled up in the middle of the last row, which is cut short <Cite kind="spec" ref="§4.11" />
            </>
          ) : null}
          . A token is a value plus the name of its column and table <Cite kind="paper" ref="p.4 §3.1" />; key columns only become links. The cell to
          predict stays in the sequence as the masked target, “{phrase}”: the model knows its column and type, but its value is swapped for a learned
          mask vector <Cite kind="spec" ref="§5.3" />.
        </p>
      );
    }
    case 'col':
    case 'feat':
    case 'nbr':
    case 'full': {
      const l = p.step;
      const n = p.allowed[l].size;
      const top = p.topKeys[0];
      const topTok = top ? ctx.tokens[top.k] : undefined;
      let body: ReactNode;
      if (l === 'col') {
        body = (
          <>
            <strong>Column attention.</strong> The masked cell may attend only to cells of the same column of the same table{' '}
            <Cite kind="code" ref="rt/model.py:181-183" />: {plural(n, 'token')}, itself and{' '}
            {task.kind === 'forecast' ? (
              <>
                {plural(live.labels.self.length, `earlier ${task.targetCol} label`)} of {who}
                {live.labels.other.length > 0 ? ` and ${plural(live.labels.other.length, 'label')} of other entities` : ''}
              </>
            ) : (
              <>{plural(live.labels.other.length, `${task.targetCol} value`)} of other rows</>
            )}
            . These are the <Term id="in-context-label">in-context labels</Term>.
          </>
        );
      } else if (l === 'feat') {
        const own = [...p.allowed.feat].filter((k) => ctx.tokens[k].nodeIdx === ctx.tokens[ctx.targetPos].nodeIdx).length;
        const par = n - own;
        body = (
          <>
            <strong>Feature attention.</strong> Its own row and the rows its row points to <Cite kind="paper" ref="p.5 §3.2" />: {plural(own, 'cell')}{' '}
            of the seed row and {plural(par, 'cell')} of {par > 0 && live.entityShort ? live.entityShort : 'its parents'}, {plural(n, 'token')} in
            all.
          </>
        );
      } else if (l === 'nbr') {
        const S = ctx.tokens.length;
        const entTok = live.entityNode !== null ? ctx.tokens.findIndex((t) => !t.isPadding && t.nodeIdx === live.entityNode) : -1;
        let entKeys = 0;
        const entTables = new Set<string>();
        if (entTok >= 0)
          for (let k = 0; k < S; k++)
            if (live.masks.nbr[entTok * S + k]) {
              entKeys++;
              entTables.add(ctx.tokens[k].tableName);
            }
        body = (
          <>
            <strong>Neighbor attention.</strong> Cells of the rows that point to its row <Cite kind="paper" ref="p.5 §3.2" />.{' '}
            {n === 0 ? (
              <>
                No row points to a task row, so this mask row is empty and the layer adds exactly zero to the masked cell{' '}
                <Cite kind="spec" ref="§6.1" />
                <Cite kind="spec" ref="§6.3" />.
                {entTok >= 0 && entKeys > 0 && (
                  <>
                    {' '}
                    The layer still works elsewhere: each cell of {live.entityShort} reads {plural(entKeys, 'token')} here, from the{' '}
                    {listJoin([...entTables])} rows that point to it.
                  </>
                )}
              </>
            ) : (
              <>{plural(n, 'token')} of its child rows.</>
            )}
          </>
        );
      } else {
        body = (
          <>
            <strong>Full attention.</strong> Every non-padding token, {plural(n, 'token')}, as in a standard transformer{' '}
            <Cite kind="paper" ref="p.5 §3.2" />.
          </>
        );
      }
      return (
        <div>
          <p className="!m-0">
            {body} Each of the {p.nBlocks} blocks runs column, feature, neighbor and full attention in this order, then an MLP{' '}
            <Cite kind="code" ref="rt/model.py:82-86" />.
          </p>
          {n > 0 && (
            <div className="mt-2.5 flex flex-wrap items-center gap-x-3 gap-y-2 text-[15.5px] text-ink-2">
              <Segmented
                label="Weights of block"
                size="sm"
                value={p.attnBlock}
                onChange={p.setAttnBlock}
                options={Array.from({ length: p.nBlocks }, (_, b) => ({
                  value: b,
                  label: `Block ${b + 1}`,
                }))}
              />
              <span>
                {p.hasWeights ? (
                  <>
                    Shading and line width: attention weight at the masked cell, mean over heads.
                    {topTok && top && (
                      <>
                        {' '}
                        Strongest key:{' '}
                        <span className="font-mono text-ink">
                          {topTok.tableName}.{topTok.colName}
                        </span>{' '}
                        ({formatPct(top.w, 1)}).
                      </>
                    )}
                  </>
                ) : (
                  'Weights appear once the model has loaded.'
                )}
              </span>
            </div>
          )}
        </div>
      );
    }
    case 'predict': {
      const pred = live.pred;
      if (!pred) return <p className="!m-0">Loading the weights and the text embeddings…</p>;
      return (
        <p className="!m-0">
          <strong>The prediction.</strong> After {p.nBlocks} blocks and a final RMSNorm, every decoder runs at every position, and only the masked
          cell’s output is read <Cite kind="code" ref="rt/model.py:229-239" />
          <Cite kind="spec" ref="§8.3" />.{' '}
          {pred.taskKind === 'binary' ? (
            <>
              The boolean decoder gives a logit of {formatNumber(pred.output, 3)}, and σ turns it into P(
              {task.targetCol}) = {formatPct(pred.probability ?? NaN)}. The true label, stored in the token but gated out of the input, is{' '}
              <code>{String(pred.label.raw)}</code>.
            </>
          ) : (
            <>
              The number decoder gives a z-score of {formatNumber(pred.output, 3)}, which the Train statistics turn back into{' '}
              {formatNumber(pred.value ?? NaN, 2)}. The true value, gated out of the input, is {formatNumber(Number(pred.label.raw), 2)}.
            </>
          )}{' '}
          {live.assets.weightsKind === 'trained' && db.name === 'shop'
            ? 'No weight was ever trained on ShopDB: this is a zero-shot prediction.'
            : live.assets.weightsKind === 'init'
              ? 'These weights are untrained, so the number means nothing yet.'
              : ''}
        </p>
      );
    }
  }
}
