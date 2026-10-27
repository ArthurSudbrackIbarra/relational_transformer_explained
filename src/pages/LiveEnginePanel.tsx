/**
 * "Live engine" panel of the component gallery (/#/dev/components): the shared components fed with
 * REAL engine output through src/state/hooks.ts → src/engine/pipeline.ts, proving the end-to-end
 * wiring: database schema → sampled context (tokens) → the four masks → forward pass → prediction.
 *
 * It follows the global simulation settings (top-bar popover), so changing the DB, task, seed, L, w
 * or checkpoint there updates every view here.
 */
import { useMemo, useState } from 'react';
import {
  Cite,
  Figure,
  LayerLabel,
  MaskMatrix,
  ScaleNote,
  SchemaDiagram,
  Segmented,
  Stat,
  StatusBadge,
  TokenStrip,
  TokenTable,
  VectorStrip,
  formatNumber,
  formatPct,
  formatRaw,
  type SchemaTableSpec,
} from '@/components';
import { axis, vec } from '@/chapters/shapes';
import { ATTN_ORDER, type AttnLayer } from '@/engine/types';
import {
  useAssets,
  useContextWindow,
  useDatabase,
  useForward,
  useInContextLabels,
  useMasks,
  usePrediction,
  usePreprocessed,
  useSeedNode,
  useTableSlots,
  useTask,
} from '@/state/hooks';

export function LiveEnginePanel() {
  const db = useDatabase();
  const pdb = usePreprocessed();
  const task = useTask();
  const seedNode = useSeedNode();
  const ctx = useContextWindow();
  const masks = useMasks();
  const slots = useTableSlots();
  const labels = useInContextLabels();
  const assets = useAssets();
  const fr = useForward();
  const pred = usePrediction();

  const [layer, setLayer] = useState<AttnLayer>('col');
  const [selected, setSelected] = useState<number | null>(null);
  const [hover, setHover] = useState<number | null>(null);

  const S = ctx.tokens.length;
  const query = hover ?? selected ?? (ctx.targetPos >= 0 ? ctx.targetPos : 0);
  const allowed = useMemo(() => {
    const out: number[] = [];
    const m = masks[layer];
    for (let k = 0; k < S; k++) if (m[query * S + k]) out.push(k);
    return out;
  }, [masks, layer, query, S]);

  const tokenLabels = useMemo(() => ctx.tokens.map((t) => (t.isPadding ? 'pad' : `${t.tableName}.${t.colName}`)), [ctx]);
  const tokenSlots = useMemo(() => ctx.tokens.map((t) => (t.isPadding ? 0 : slots[t.tableName] ?? 0)), [ctx, slots]);

  const schema: SchemaTableSpec[] = useMemo(() => {
    const out: SchemaTableSpec[] = db.tables.map((t) => ({ schema: t.schema, slot: slots[t.schema.name] ?? 0, rowCount: t.rows.length, kind: 'db' }));
    for (const t of db.tasks) {
      if (t.kind !== 'forecast' || !t.tables) continue;
      const n = t.tables.Train.rows.length + t.tables.Val.rows.length + t.tables.Test.rows.length;
      out.push({ schema: t.tables.Train.schema, slot: slots[t.name] ?? 5, rowCount: n, kind: 'task', subtitle: 'Train · Val · Test' });
    }
    return out;
  }, [db, slots]);

  const seed = pdb.nodes[seedNode];
  // forecast only: the task row's single FK is the entity (autocomplete seeds are DB rows, spec §2.4)
  const entity = task.kind === 'forecast' && seed && seed.f2pNbrIdxs.length ? pdb.nodes[seed.f2pNbrIdxs[0]] : undefined;
  const target = ctx.targetPos >= 0 ? ctx.tokens[ctx.targetPos] : undefined;
  const targetVec = useMemo(() => (fr && fr.targetPos >= 0 ? fr.xFinal.subarray(fr.targetPos * fr.d, (fr.targetPos + 1) * fr.d) : null), [fr]);

  return (
    <>
      <p>
        Everything below is computed in your browser by the real engine: <code>getDatabase</code> → <code>preprocess</code> →{' '}
        <code>sampleContext</code> → <code>buildMasks</code> → <code>forward</code>. It follows the simulation settings in the top bar
        (database <strong>{db.name}</strong>, task <strong>{task.id.split('/')[1]}</strong>, L = {ctx.params.L}, w = {ctx.params.w}).{' '}
        <ScaleNote inline />
      </p>

      <div className="wide grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Stat
          label="Seed row"
          value={<>#{seedNode}</>}
          caption={
            <>
              {seed ? `${seed.tableKey}[${seed.rowPos}]` : '—'}
              {entity ? ` → ${entity.tableName}[${entity.rowPos}]` : ''}
              {seed?.ts != null ? ` · ${new Date(seed.ts * 1000).toISOString().slice(0, 10)}` : ''}
            </>
          }
        />
        <Stat
          label="Context window"
          value={ctx.numReal}
          unit={`/ ${S} tokens`}
          caption={`${ctx.visited.length} rows visited; target at position ${ctx.targetPos}.`}
        />
        <Stat
          label="In-context labels"
          value={labels.self.length}
          unit={`self · ${labels.other.length} other`}
          caption={
            task.kind === 'forecast'
              ? `Visible ${task.targetCol} cells of the seed's own ${task.entityTable ?? 'entity'} vs other entities.`
              : 'Autocomplete: visible target-column cells of other rows.'
          }
        />
        <Stat
          label={task.targetType === 'binary' ? `P(${task.targetCol})` : `Predicted ${task.targetCol}`}
          value={
            pred
              ? pred.taskKind === 'binary'
                ? formatPct(pred.probability ?? NaN)
                : formatNumber(pred.value ?? NaN)
              : assets.status === 'error'
                ? 'error'
                : '…'
          }
          emphasis
          caption={
            pred ? (
              <>
                {pred.taskKind === 'binary' ? `logit ${formatNumber(pred.output)}` : `z ${formatNumber(pred.output)}`} · true label{' '}
                {formatRaw(pred.label.raw)}
              </>
            ) : (
              'Loading embeddings and weights…'
            )
          }
        />
      </div>

      <div className="wide mt-3 flex flex-wrap items-center gap-2">
        {assets.status === 'loading' && <StatusBadge status="info">Loading embeddings and weights…</StatusBadge>}
        {assets.status === 'error' && <StatusBadge status="critical">Assets failed: {assets.error}</StatusBadge>}
        {assets.status === 'ready' && assets.weightsKind === 'trained' && <StatusBadge status="good">Trained checkpoint: {assets.ckpt}</StatusBadge>}
        {assets.status === 'ready' && assets.weightsKind === 'init' && (
          <StatusBadge status="warning">Untrained weights{assets.fallback ? ` (${assets.ckpt} not exported yet)` : ''}</StatusBadge>
        )}
        {assets.status === 'ready' && assets.store && <StatusBadge status="info">{assets.store.strings.length} MiniLM embeddings, {assets.store.dim}-d</StatusBadge>}
        {fr && <StatusBadge status="info">loss {formatNumber(fr.loss)} over 1 masked cell</StatusBadge>}
      </div>
      {assets.note && <p className="wide mt-2 text-[15.5px] text-ink-2">{assets.note}</p>}

      <Figure wide title={`${db.title}: schema`} caption="Live schema from the generator; task tables share slot 5. Arrows point from child (FK) to parent (PK).">
        <SchemaDiagram tables={schema} highlightTables={entity ? [entity.tableName] : []} />
      </Figure>

      <Figure
        wide
        title="Sampled context: tokens"
        caption={
          <>
            Select or hover a token; the keys it may attend to under the chosen mask are ringed. Default query: the masked target{' '}
            {target ? (
              <span>
                {target.tableName}.{target.colName}
              </span>
            ) : null}{' '}
            <Cite kind="spec" ref="§6.2" />.
          </>
        }
        controls={
          <Segmented
            label="Mask"
            value={layer}
            onChange={setLayer}
            options={ATTN_ORDER.map((l) => ({ value: l, label: <LayerLabel layer={l} size={13} /> }))}
          />
        }
      >
        <TokenStrip
          tokens={ctx.tokens}
          tableSlots={slots}
          highlight={allowed}
          dimOthers
          selected={selected ?? ctx.targetPos}
          onSelect={setSelected}
          onHover={setHover}
          label="Live context tokens"
        />
        <div className="mt-5">
          <TokenTable tokens={ctx.tokens} text={pdb.text} tableSlots={slots} selected={selected} highlight={allowed} onSelect={setSelected} maxHeight={320} />
        </div>
      </Figure>

      <Figure wide title="The four relational attention masks of this context" caption="Rows are queries, columns keys; padding is cropped.">
        <div className="grid gap-6 md:grid-cols-2">
          {ATTN_ORDER.map((l) => (
            <MaskMatrix
              key={l}
              layer={l}
              mask={masks[l]}
              S={S}
              crop={ctx.numReal}
              labels={tokenLabels}
              slots={tokenSlots}
              highlightRows={[query]}
              maxCellSize={6}
              showLegend={l === 'col'}
              shape={{
                rows: axis.queries({ shown: ctx.numReal, L: ctx.params.L, real: ctx.numReal }),
                cols: axis.keys({ shown: ctx.numReal, L: ctx.params.L, real: ctx.numReal }),
              }}
            />
          ))}
        </div>
      </Figure>

      <Figure wide title="Forward pass at the target" caption="RMSNorm_out of the residual stream at the masked cell, the input of every decoder head (model.py:229-239).">
        {targetVec ? (
          <VectorStrip
            values={targetVec}
            label={`x_final[${fr!.targetPos}] (${fr!.d}-d)`}
            shape={vec(`x_final at the masked target (position ${fr!.targetPos}), the vector every decoder reads`, axis.model(fr!.d))}
          />
        ) : (
          <p className="text-[16.5px] text-ink-2">{assets.status === 'error' ? `Forward unavailable: ${assets.error}` : 'Running the forward pass…'}</p>
        )}
      </Figure>
    </>
  );
}
