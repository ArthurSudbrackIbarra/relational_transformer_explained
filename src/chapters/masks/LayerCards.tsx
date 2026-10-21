/**
 * Section 4: what each layer lets flow. One card per mask with the paper's wording, the live numbers,
 * the App. E expressivity remark, and an animated diagram on a small REAL excerpt of the live context
 * (the live masks restricted to those tokens): lines run from every allowed key into the query.
 */
import { RotateCcw } from 'lucide-react';
import { useInView } from 'motion/react';
import { useMemo, useRef, useState, type ReactNode } from 'react';
import { Cite, LayerLabel, TeX, cx } from '@/components';
import type { AttnLayer } from '@/engine/types';
import { useInContextLabels, useTask } from '@/state/hooks';
import { ContextTables } from './ContextTables';
import { columnExcerpt, countRow, excerptPositions, groupContext, int, maskRow, neighborExcerpt, rowName, singular } from './maskUtils';
import type { LiveMasks } from './useLiveMasks';

const PAPER_MASK: Record<AttnLayer, string> = {
  col: String.raw`\mathbf{1}\{\mathrm{Col}(k)=\mathrm{Col}(q)\}`,
  feat: String.raw`\mathbf{1}\{\mathrm{Row}(k)=\mathrm{Row}(q)\ \lor\ \mathrm{Row}(k)\in\mathrm{OutLinks}(q)\}`,
  nbr: String.raw`\mathbf{1}\{\mathrm{Row}(q)\in\mathrm{OutLinks}(k)\}`,
  full: String.raw`1`,
};

export function LayerCards({ live }: { live: LiveMasks }) {
  const { ctx, masks, S, slots, target, holeKind } = live;
  const task = useTask();
  const labels = useInContextLabels();

  const seed = ctx.tokens[target];
  const entityNode = seed?.f2pNbrIdxs.find((p) => p >= 0);

  // One real excerpt per card, chosen so the card's point is visible: the column card shows the rows
  // of the target's column, the neighbor card one row of every child table, feature and full a mix.
  const excerpts = useMemo(() => {
    const general = excerptPositions(ctx.tokens, ctx.targetPos, 30);
    const byLayer: Record<AttnLayer, number[]> = {
      col: columnExcerpt(ctx.tokens, ctx.targetPos, 14),
      feat: general,
      nbr: entityNode !== undefined ? neighborExcerpt(ctx.tokens, entityNode, 3) : general,
      full: general,
    };
    const out = {} as Record<AttnLayer, { size: number; grouping: ReturnType<typeof groupContext> }>;
    for (const l of Object.keys(byLayer) as AttnLayer[]) {
      const toks = byLayer[l].map((p) => ctx.tokens[p]);
      out[l] = { size: toks.length, grouping: groupContext(toks, slots, holeKind) };
    }
    return out;
  }, [ctx, entityNode, slots, holeKind]);
  const entityPos = entityNode !== undefined ? ctx.tokens.find((t) => !t.isPadding && t.nodeIdx === entityNode)?.pos : undefined;
  const entityTok = entityPos !== undefined ? ctx.tokens[entityPos] : undefined;
  const childTok = useMemo(
    () =>
      entityNode === undefined
        ? undefined
        : ctx.tokens.find((t) => !t.isPadding && t.tableType === 'Db' && t.f2pNbrIdxs.includes(entityNode) && t.tableNameIdx !== seed.tableNameIdx),
    [ctx, entityNode, seed],
  );

  const facts = useMemo(() => {
    const rowsAllowed = (q: number, l: AttnLayer) => {
      const r = maskRow(masks[l], S, q);
      const byTable = new Map<string, { cells: number; rows: Set<number> }>();
      for (let k = 0; k < S; k++) {
        if (!r[k]) continue;
        const t = ctx.tokens[k];
        let e = byTable.get(t.tableName);
        if (!e) byTable.set(t.tableName, (e = { cells: 0, rows: new Set() }));
        e.cells++;
        e.rows.add(t.nodeIdx);
      }
      return byTable;
    };
    let emptyNbr = 0;
    for (let qq = 0; qq < S; qq++) if (!ctx.tokens[qq].isPadding && countRow(masks.nbr, S, qq) === 0) emptyNbr++;
    const ownCells = ctx.tokens.filter((t) => !t.isPadding && t.nodeIdx === seed?.nodeIdx).length;
    const parentCells = ctx.tokens.filter((t) => !t.isPadding && seed && t.nodeIdx !== seed.nodeIdx && seed.f2pNbrIdxs.includes(t.nodeIdx)).length;
    return {
      emptyNbr,
      ownCells,
      parentCells,
      entityNbr: entityPos !== undefined ? rowsAllowed(entityPos, 'nbr') : new Map(),
      childFeat: childTok ? countRow(masks.feat, S, childTok.pos) : 0,
      childParents: childTok ? new Set(childTok.f2pNbrIdxs.filter((p) => p >= 0 && ctx.tokens.some((t) => !t.isPadding && t.nodeIdx === p))).size : 0,
      targetCol: countRow(masks.col, S, target),
      targetNbr: countRow(masks.nbr, S, target),
      parentNames: [...new Set(seed?.f2pNbrIdxs.filter((p) => p >= 0) ?? [])]
        .map((p) => ctx.tokens.find((t) => !t.isPadding && t.nodeIdx === p))
        .filter((t): t is NonNullable<typeof t> => !!t)
        .map((t) => rowName(t)),
    };
  }, [masks, S, ctx, seed, entityPos, childTok, target]);

  const splitCounts = useMemo(() => {
    const m = new Map<string, number>();
    for (const l of [...labels.self, ...labels.other]) m.set(l.tableType ?? 'Db', (m.get(l.tableType ?? 'Db') ?? 0) + 1);
    return [...m.entries()].map(([k, v]) => `${v} ${k}`).join(', ');
  }, [labels]);

  const forecast = task.kind === 'forecast';
  const tgtCol = seed?.colName ?? task.targetCol;
  const entName = entityTok ? rowName(entityTok) : 'its entity';

  const queryOf: Record<AttnLayer, number> = { col: target, feat: target, nbr: entityPos ?? target, full: target };

  const cards: { layer: AttnLayer; quote: ReactNode; cite: string; live: ReactNode; appE: ReactNode; appECite: string; queryNote: string }[] = [
    {
      layer: 'col',
      quote: '“Column attention helps model the distribution of values in each column.”',
      cite: 'p.5 §3.2',
      queryNote: 'query: the masked target',
      live: forecast ? (
        <>
          The masked <code>{tgtCol}</code> cell may read the {facts.targetCol - 1} other cells of its column, and these are exactly the visible
          labels of the task in this context: {labels.self.length} from {entName}’s own past rows and {labels.other.length} from other entities
          {splitCounts ? ` (by split: ${splitCounts})` : ''}. Task rows of every split share one table id, so they form a single column group{' '}
          <Cite kind="spec" ref="§6.2" />. The target never sees its own label: its input carries the mask vector instead{' '}
          <Cite kind="spec" ref="§5.3" />.
        </>
      ) : (
        <>
          The masked <code>{tgtCol}</code> cell may read the {facts.targetCol - 1} other cells of its column: the visible{' '}
          <code>{tgtCol}</code> values of the other <code>{seed?.tableName}</code> rows in this context. Only the seed row’s cell is masked{' '}
          <Cite kind="spec" ref="§4.10" />; its input carries the mask vector instead of the value <Cite kind="spec" ref="§5.3" />.
        </>
      ),
      appE: '“Removing column attention does not reduce expressivity, since global attention can emulate it.”',
      appECite: 'p.19 App. E',
    },
    {
      layer: 'feat',
      quote: '“Feature attention is equivalent to row-wise attention after joining each table with its parent tables, and enables feature mixing for entities.”',
      cite: 'p.5 §3.2',
      queryNote: 'query: the masked target',
      live: (
        <>
          The target reads the {facts.ownCells} cells of its own {forecast ? 'task row' : 'row'} and the {facts.parentCells} cells of its parent{' '}
          {facts.parentNames.length === 1 ? 'row' : 'rows'} {facts.parentNames.join(' and ')}:{' '}
          {forecast && entityTok ? `the task row joined with its ${singular(entityTok.tableName)}.` : 'the row joined with its parents.'}
          {childTok && (
            <>
              {' '}
              {/^[aeiou]/i.test(childTok.tableName) ? 'An' : 'A'} <code>{childTok.tableName}</code> cell reads its own row and its{' '}
              {facts.childParents} parent {facts.childParents === 1 ? 'row' : 'rows'} ({facts.childFeat} keys).
            </>
          )}{' '}
          Only the child reads the parent here: {entName} does not see its children in this layer.
        </>
      ),
      appE: '“Feature attention is the only mechanism that explicitly groups cells into rows.”',
      appECite: 'p.19 App. E',
    },
    {
      layer: 'nbr',
      quote: '“Neighbor attention captures information from incoming links to an entity, enabling the model to aggregate signals from its child rows. This module acts analogously to message-passing in GNNs.”',
      cite: 'p.5 §3.2',
      queryNote: entityTok ? `query: a cell of ${entName}` : 'query: the masked target',
      live: (
        <>
          {entityTok ? (
            <>
              A cell of {entName} reads every row in the context that points to it:{' '}
              {[...facts.entityNbr.entries()].map(([t, e], i) => (
                <span key={t}>
                  {i > 0 && ', '}
                  {e.rows.size} <code>{t}</code> {e.rows.size === 1 ? 'row' : 'rows'}
                </span>
              ))}
              {' '}(the diagram shows up to three of each).{forecast ? ' Its task rows are among them, past labels included.' : ''}{' '}
            </>
          ) : null}
          {facts.targetNbr === 0 ? (
            <>
              No row in the context points to the target’s row, so its neighbor mask row is empty and the output there is exactly zero{' '}
              <Cite kind="spec" ref="§6.1" />.
            </>
          ) : (
            <>The target’s row has {facts.targetNbr} child cells in the context.</>
          )}{' '}
          Here {int(facts.emptyNbr)} of {int(ctx.numReal)} tokens have an empty neighbor row.
        </>
      ),
      appE: '“neighbor attention is not strictly required for expressivity, as feature and global attention together can simulate its effect.”',
      appECite: 'p.20 App. E',
    },
    {
      layer: 'full',
      quote: '“Full attention confers the expressive power of standard Transformers, complementing the relationally constrained layers above.”',
      cite: 'p.5 §3.2',
      queryNote: 'query: the masked target',
      live: (
        <>
          Every non-padding token reads all {int(ctx.numReal)} non-padding tokens, whatever their table, row or column. Padding tokens read
          nothing and are never read <Cite kind="code" ref="rt/model.py:190" />.
        </>
      ),
      appE: '“Without full attention, information can only propagate one hop per layer, limiting expressivity to local message passing.”',
      appECite: 'p.20 App. E',
    },
  ];

  return (
    <div className="wide my-8 grid gap-5 md:grid-cols-2">
      {cards.map((c) => (
        <LayerCard
          key={c.layer}
          layer={c.layer}
          quote={c.quote}
          cite={c.cite}
          liveText={c.live}
          appE={c.appE}
          appECite={c.appECite}
          queryNote={c.queryNote}
          diagram={(flow, replay) => (
            <ContextTables
              tokens={ctx.tokens}
              grouping={excerpts[c.layer].grouping}
              allowed={maskRow(masks[c.layer], S, queryOf[c.layer])}
              query={queryOf[c.layer]}
              cell={20}
              gapX={28}
              maxRowsPerStack={c.layer === 'col' ? 14 : 12}
              flow={flow}
              replay={replay}
              showPadding={false}
              label={`${c.layer} mask on an excerpt of the live context: lines run from each allowed key into the query.`}
            />
          )}
          excerptSize={excerpts[c.layer].size}
          total={ctx.numReal}
        />
      ))}
    </div>
  );
}

function LayerCard({
  layer,
  quote,
  cite,
  liveText,
  appE,
  appECite,
  queryNote,
  diagram,
  excerptSize,
  total,
}: {
  layer: AttnLayer;
  quote: ReactNode;
  cite: string;
  liveText: ReactNode;
  appE: ReactNode;
  appECite: string;
  queryNote: string;
  diagram: (flow: boolean, replay: number) => ReactNode;
  excerptSize: number;
  total: number;
}) {
  const ref = useRef<HTMLDivElement | null>(null);
  const inView = useInView(ref, { once: true, amount: 0.35 });
  const [replay, setReplay] = useState(0);
  return (
    <article ref={ref} className={cx('card flex min-w-0 flex-col px-4 py-4 sm:px-5')} aria-label={`${layer} attention`}>
      <div className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1">
        <h3 className="text-[18.5px] font-semibold text-ink">
          <LayerLabel layer={layer} size={16} /> <span className="font-normal text-muted">attention</span>
        </h3>
        <span className="text-[15.5px] text-ink-2">
          <TeX tex={`M^{\\text{${LAYER_TEX[layer]}}}[q,k] = ${PAPER_MASK[layer]}`} />
        </span>
      </div>

      <div className="mt-3 rounded-lg border border-border bg-surface-2 px-3 pt-2.5 pb-3">
        <div className="mb-1.5 flex items-start justify-between gap-2 text-[14px] text-muted">
          <span>
            {queryNote} · excerpt: {excerptSize} of {total} live tokens
          </span>
          <button
            type="button"
            onClick={() => setReplay((r) => r + 1)}
            className="inline-flex items-center gap-1 rounded-md px-1.5 py-0.5 text-[14px] font-medium text-ink-2 hover:bg-surface hover:text-ink"
            aria-label={`Replay the ${layer} flow animation`}
          >
            <RotateCcw aria-hidden="true" size={12} /> replay
          </button>
        </div>
        {diagram(inView, replay)}
      </div>

      <blockquote className="mt-3.5 border-l-2 border-axis pl-3 text-[17px] leading-snug text-ink-2 italic">
        {quote} <Cite kind="paper" ref={cite} />
      </blockquote>
      <p className="mt-3 text-[17px] leading-relaxed text-ink">
        <span className="eyebrow mr-1.5 !text-[13px]">In this context</span>
        {liveText}
      </p>
      <p className="mt-auto pt-3 text-[15.5px] leading-snug text-ink-2">
        <span className="font-medium text-ink">Expressivity: </span>
        {appE} <Cite kind="paper" ref={appECite} />
      </p>
    </article>
  );
}

const LAYER_TEX: Record<AttnLayer, string> = { col: 'column', feat: 'feature', nbr: 'neighbor', full: 'full' };

