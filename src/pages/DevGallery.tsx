/**
 * Component gallery (/#/dev/components, not linked from the navigation):
 * every shared component with fake data, in both themes. Used for visual QA
 * and as living documentation for chapter authors. The first section, "Live
 * engine" (LiveEnginePanel.tsx), feeds the same components real engine output
 * through the state hooks to prove the end-to-end wiring.
 */
import { useMemo, useState } from 'react';
import {
  BarChart,
  Button,
  Callout,
  ChapterHeader,
  ChapterPage,
  Cite,
  CodeRef,
  EntityGraph,
  ErrorBoundary,
  Figure,
  Heatmap,
  LayerLabel,
  LineChart,
  MaskMatrix,
  MathBlock,
  NumberField,
  Quiz,
  ScaleNote,
  Section,
  Segmented,
  Select,
  SemBadge,
  Slider,
  Stat,
  StatusBadge,
  StepPlayer,
  Switch,
  SchemaDiagram,
  TableSwatch,
  TableView,
  Term,
  TeX,
  TokenStrip,
  TokenTable,
  useStepPlayer,
  VectorStrip,
  type NodeId,
  type RingStyle,
} from '@/components';
import { SemType, type AttnLayer } from '@/engine/types';
import { CATEGORICAL_NAMES } from '@/styles/palette';
import {
  ATTN_LAYERS,
  FAKE_CUSTOMER_Z,
  FAKE_CUSTOMERS,
  FAKE_ORDERS,
  fakeAttention,
  fakeCurves,
  fakeMask,
  fakeVector,
  maskRowAscii,
  SHOP_SCHEMA,
  SPEC_COLUMN_ROW_1,
  TOY_GRAPH,
  TOY_L,
  TOY_LABELS,
  TOY_SLOTS,
  TOY_TOKENS,
  toyMasks,
} from './devFixtures';
import { FLY_P2F_415_452, MODEL_ATTN_20_47, MODEL_MASKS_164_191 } from './devSnippets';
import { LiveEnginePanel } from './LiveEnginePanel';

export function DevGallery() {
  const masks = useMemo(() => toyMasks(TOY_TOKENS), []);
  const [hoverPos, setHoverPos] = useState<number | null>(null);
  const [selPos, setSelPos] = useState<number | null>(1);
  const [layer, setLayer] = useState<AttnLayer>('col');
  const [seg, setSeg] = useState<'raw' | 'z'>('raw');
  const [sw, setSw] = useState(true);
  const [sl, setSl] = useState(128);
  const [nf, setNf] = useState(0);
  const [hoverNode, setHoverNode] = useState<NodeId | null>(null);

  const realCount = TOY_TOKENS.filter((t) => !t.isPadding).length;
  const q = hoverPos ?? selPos ?? 1;
  const allowed = useMemo(() => {
    const out: number[] = [];
    for (let k = 0; k < TOY_L; k++) if (masks[layer][q * TOY_L + k]) out.push(k);
    return out;
  }, [masks, layer, q]);
  const colRowOk = maskRowAscii(masks.col, TOY_L, 1) === SPEC_COLUMN_ROW_1;

  const player = useStepPlayer(realCount, { intervalMs: 600 });
  const revealed = useMemo(() => TOY_TOKENS.filter((t) => t.pos <= player.index).map((t) => t.pos), [player.index]);

  const bigMask = useMemo(() => fakeMask(128), []);
  const bigAttn = useMemo(() => fakeAttention(128, 4, 3, bigMask), [bigMask]);
  const smallAttn = useMemo(() => fakeAttention(TOY_L, 1, 9, masks.full), [masks]);
  const v64 = useMemo(() => fakeVector(64, 1), []);
  const v384 = useMemo(() => fakeVector(384, 2, 0.05), []);
  const curves = useMemo(() => fakeCurves(), []);

  const visited = new Set<NodeId>([16, 0, 14, 4, 9, 12, 7, 2, 15]);
  const rings = new Map<NodeId, RingStyle>([[16, 'target'], [3, 'dashed'], [17, 'dashed']]);
  const badges = new Map<NodeId, string>([
    [16, 'd0'],
    [0, 'd1'],
    [14, 'd2'],
    [4, 'd2'],
    [12, 'd2'],
    [9, 'd3'],
    [7, 'd4'],
    [2, 'd5'],
    [15, 'd6'],
  ]);

  return (
    <ChapterPage>
      <ChapterHeader
        number={0}
        title="Component gallery"
        lede="Every shared component with fake data. Not linked from the navigation. Toggle the theme in the top bar to check both modes."
        learn={['Shell & prose helpers', 'Data views for tables, graphs, tokens, vectors and matrices', 'Charts, stats and step-through controls']}
      />

      <Section id="live" title="Live engine" kicker="real data · end-to-end wiring">
        <ErrorBoundary>
          <LiveEnginePanel />
        </ErrorBoundary>
      </Section>

      <Section id="prose" title="Prose helpers" kicker="Callout · Cite · Term · Math · ScaleNote">
        <p>
          Inline citations come in four kinds: spec <Cite kind="spec" ref="§4.6" title="Temporal filter" />, paper{' '}
          <Cite kind="paper" ref="p.5 §3.2" />, code <Cite kind="code" ref="rustler/src/fly.rs:421-424" title="temporal constraint" /> and web{' '}
          <Cite kind="url" ref="https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2" title="all-MiniLM-L12-v2 model card" />. Glossary terms
          such as the <Term id="seed-row">seed row</Term>, an <Term id="in-context-label">in-context label</Term> or{' '}
          <Term id="neighbor-attention" /> show a definition on hover or focus. Inline math: <TeX tex="z = (v - \mu)/\sigma" />.
        </p>
        <MathBlock
          tex={String.raw`x_i = \mathrm{RMSNorm}_{\text{col}}\!\left(W_{\text{col}}\,c_i + b_{\text{col}}\right) + \begin{cases}\mathrm{RMSNorm}_t\!\left(W_t\,v_i + b_t\right) & \text{unmasked}\\ m_t & \text{masked}\end{cases}`}
          label={<Cite kind="code" ref="rt/model.py:208-224" />}
          caption="The input embedding of token i (spec §5.2)."
        />
        <Callout variant="note">
          <p>A neutral aside. Notes hold context that is useful but not essential.</p>
        </Callout>
        <Callout
          variant="paper-vs-code"
          title="Pre-norm, not post-norm"
          paper={
            <>
              Algorithm 2 writes <TeX tex="X \leftarrow X + \mathrm{Norm}(\mathrm{MHA}(X;M))" /> <Cite kind="paper" ref="p.26 Alg. 2" />.
            </>
          }
          code={
            <>
              <code>x = x + attn(norm(x))</code>, pre-norm <Cite kind="code" ref="rt/model.py:83-85" />.
            </>
          }
        />
        <Callout variant="warning" title="Constant boolean columns">
          <p>
            A boolean column with σ = 0 gives NaN values that poison the whole batch loss <Cite kind="spec" ref="§3.3" />.
          </p>
        </Callout>
        <Callout variant="insight">
          <p>Column attention lets the masked label read every visible label of its task in one hop.</p>
        </Callout>
        <ScaleNote />
        <ScaleNote paper="L = 1024, w = 256" ours="L = 128, w = 16">
          Contexts are shorter, so fewer rows fit; the sampler is the same code path.
        </ScaleNote>
        <p>
          Inline variant: <ScaleNote inline />.
        </p>
      </Section>

      <Section id="coderef" title="CodeRef" kicker="verbatim official code">
        <CodeRef title="The four relational attention masks" file="rt/model.py" lines="164-191" code={MODEL_MASKS_164_191} defaultOpen highlightLines={[165, 186, 187, 188, 189, 190]} />
        <CodeRef title="MaskedAttention" file="rt/model.py" lines="20-47" code={MODEL_ATTN_20_47} note="Bias-free projections; heads split as b s (h d) → b h s d." />
        <CodeRef title="P→F children: task-table test, temporal filter, width bound" file="rustler/src/fly.rs" lines="415-452" code={FLY_P2F_415_452} wide />
      </Section>

      <Section id="badges" title="Badges, swatches, layers, status">
        <div className="card flex flex-wrap items-center gap-4 px-4 py-3">
          {[SemType.Number, SemType.Text, SemType.DateTime, SemType.Boolean].map((s) => (
            <SemBadge key={s} sem={s} showLabel />
          ))}
          <SemBadge sem="number" size="md" />
          <SemBadge sem="text" size="xs" />
        </div>
        <div className="card mt-3 flex flex-wrap items-center gap-x-5 gap-y-2 px-4 py-3 text-[16.5px]">
          {[1, 2, 3, 4, 5, 6, 7, 8].map((s) => (
            <TableSwatch key={s} slot={s} label={`${s} ${CATEGORICAL_NAMES[s]}`} mono={false} />
          ))}
        </div>
        <div className="card mt-3 flex flex-wrap items-center gap-x-5 gap-y-2 px-4 py-3 text-[16.5px]">
          {Object.entries(TOY_SLOTS).map(([t, s]) => (
            <TableSwatch key={t} slot={s} label={t} shape size={11} />
          ))}
        </div>
        <div className="card mt-3 flex flex-wrap items-center gap-4 px-4 py-3">
          {ATTN_LAYERS.map((l) => (
            <LayerLabel key={l} layer={l} />
          ))}
        </div>
        <div className="mt-3 flex flex-wrap items-center gap-2">
          <StatusBadge status="good">Parity: max |Δ| 3.1e-6</StatusBadge>
          <StatusBadge status="warning">bf16 vs float32</StatusBadge>
          <StatusBadge status="critical">Mismatch</StatusBadge>
          <StatusBadge status="info">Not verifiable</StatusBadge>
        </div>
      </Section>

      <Section id="controls" title="Controls">
        <Figure
          controls={
            <>
              <Segmented label="Display" value={seg} onChange={setSeg} options={[{ value: 'raw', label: 'raw' }, { value: 'z', label: 'z-scored' }]} />
              <Select label="Layer" value={layer} onChange={(e) => setLayer(e.target.value as AttnLayer)}>
                {ATTN_LAYERS.map((l) => (
                  <option key={l} value={l}>
                    {l}
                  </option>
                ))}
              </Select>
              <Slider label="L" value={sl} min={8} max={512} step={8} onChange={setSl} className="w-48" />
              <NumberField label="RNG seed" value={nf} onChange={setNf} className="w-28" />
              <Switch checked={sw} onChange={setSw} label="Show padding" />
              <Button variant="primary">Primary</Button>
              <Button>Secondary</Button>
              <Button variant="ghost">Ghost</Button>
            </>
          }
        >
          <p className="text-[16.5px] text-ink-2">
            Figure controls sit in one row above what they scope. Current: {seg}, {layer}, L={sl}, seed={nf}, padding {sw ? 'on' : 'off'}.
          </p>
        </Figure>
      </Section>

      <Section id="schema" title="SchemaDiagram">
        <Figure wide caption="ShopDB-like schema (fake). Hover a table to highlight its links.">
          <SchemaDiagram
            tables={SHOP_SCHEMA.map((s) => ({
              schema: s,
              slot: TOY_SLOTS[s.name] ?? 0,
              rowCount: s.name === 'customers' ? 80 : s.name === 'products' ? 30 : s.name === 'orders' ? 1200 : s.name === 'reviews' ? 300 : undefined,
              kind: s.name === 'customer-churn' ? 'task' : 'db',
              subtitle: s.name === 'customer-churn' ? 'Train · Val · Test' : undefined,
            }))}
            highlightColumns={['customers.age']}
          />
        </Figure>
      </Section>

      <Section id="table" title="TableView">
        <Figure wide caption="Virtualised rows; ∅ = missing (no token); the dashed cell is a masked target; the raw/z toggle appears when z-values are given.">
          <div className="grid gap-4 lg:grid-cols-2">
            <TableView
              schema={SHOP_SCHEMA[0]}
              rows={FAKE_CUSTOMERS}
              slot={1}
              zValue={FAKE_CUSTOMER_Z}
              highlightRows={[3]}
              highlightCells={['5:age']}
              targetCell={{ row: 1, col: 'is_premium' }}
              footer="Fake customers (seeded)."
            />
            <TableView schema={SHOP_SCHEMA[2]} rows={FAKE_ORDERS} slot={3} mutedRows={[0, 1, 2]} maxHeight={300} />
          </div>
        </Figure>
      </Section>

      <Section id="graph" title="EntityGraph">
        <Figure wide caption="The spec §6.3 toy DB. Rings: dashed = frontier, dashed with ? = seed; badges are BFS depth labels. Visited rows highlighted.">
          <EntityGraph
            nodes={TOY_GRAPH.nodes}
            edges={TOY_GRAPH.edges}
            highlight={visited}
            rings={rings}
            badges={badges}
            selected={hoverNode}
            onNodeHover={setHoverNode}
            height={360}
          />
        </Figure>
      </Section>

      <Section id="tokens" title="TokenStrip and TokenTable">
        <p>
          The spec §6.3 toy context (seed = churn Val row #16, L = 24, w = 1). Hover or select a token; the {layer} mask lights up the keys it
          may attend to. Column-mask row 1 matches the spec: <strong>{colRowOk ? 'yes' : 'NO'}</strong>.
        </p>
        <Figure
          wide
          controls={
            <Segmented
              label="Mask"
              value={layer}
              onChange={setLayer}
              options={ATTN_LAYERS.map((l) => ({ value: l, label: <LayerLabel layer={l} size={13} /> }))}
            />
          }
        >
          <TokenStrip
            tokens={TOY_TOKENS}
            tableSlots={TOY_SLOTS}
            highlight={allowed}
            dimOthers
            selected={selPos}
            onSelect={setSelPos}
            onHover={setHoverPos}
          />
          <div className="mt-5">
            <TokenStrip tokens={TOY_TOKENS} tableSlots={TOY_SLOTS} variant="cells" showPadding highlight={allowed} dimOthers label="Compact token cells" />
          </div>
        </Figure>
        <Figure wide>
          <TokenTable tokens={TOY_TOKENS} tableSlots={TOY_SLOTS} selected={selPos} highlight={allowed} onSelect={setSelPos} maxHeight={320} />
        </Figure>
      </Section>

      <Section id="masks" title="MaskMatrix and Heatmap">
        <Figure wide caption="The four masks of the toy context (padding cropped). Identity strips show the table of each token.">
          <div className="grid gap-6 md:grid-cols-2">
            {ATTN_LAYERS.map((l) => (
              <MaskMatrix
                key={l}
                layer={l}
                mask={masks[l]}
                S={TOY_L}
                crop={realCount}
                labels={TOY_LABELS}
                slots={TOY_TOKENS.map((t) => TOY_SLOTS[t.tableName] ?? 0)}
                highlightRows={[q]}
                maxCellSize={13}
                showLegend={l === 'col'}
              />
            ))}
          </div>
        </Figure>
        <Figure wide title="Attention weights, 128 × 128, four heads (fake)">
          <Heatmap
            data={bigAttn}
            rows={128}
            cols={128}
            stacks={4}
            stackLabels={['head 0', 'head 1', 'head 2', 'head 3']}
            blocked={bigMask.map((v) => (v ? 0 : 1))}
            rowName="query"
            colName="key"
            title="Fake attention weights"
            highlightRows={[40]}
          />
        </Figure>
        <Figure wide title="Labelled heatmap (fake weights on the toy context)">
          <Heatmap
            data={smallAttn}
            rows={TOY_L}
            cols={TOY_L}
            crop={{ rows: realCount, cols: realCount }}
            rowLabels={TOY_LABELS}
            colLabels={TOY_LABELS}
            rowSlots={TOY_TOKENS.map((t) => TOY_SLOTS[t.tableName] ?? 0)}
            colSlots={TOY_TOKENS.map((t) => TOY_SLOTS[t.tableName] ?? 0)}
            blocked={masks.full.map((v) => (v ? 0 : 1))}
            selected={{ row: 1, col: 6 }}
            title="Labelled fake attention"
            rowName="query"
            colName="key"
          />
        </Figure>
      </Section>

      <Section id="vectors" title="VectorStrip">
        <Figure wide>
          <VectorStrip values={v64} label="x (64-d, fake)" ranges={[0, 1, 2, 3].map((h) => ({ start: h * 16, end: h * 16 + 16 }))} />
          <div className="mt-5">
            <VectorStrip values={v384} label="E(“age of customers”) (384-d, fake)" columns={64} cellHeight={10} />
          </div>
        </Figure>
      </Section>

      <Section id="player" title="StepPlayer">
        <Figure wide>
          <StepPlayer
            {...player}
            unit="token"
            describe={(i) => {
              const t = TOY_TOKENS[i];
              return (
                <>
                  Emit token <span className="num">{i}</span>: <span className="font-mono">{t.colName}</span> of{' '}
                  <span className="font-mono">{t.tableName}</span>
                  {t.isTarget ? ' (the masked target)' : ''}.
                </>
              );
            }}
          />
          <div className="mt-4">
            <TokenStrip tokens={TOY_TOKENS.filter((t) => !t.isPadding)} tableSlots={TOY_SLOTS} highlight={[player.index]} muted={TOY_TOKENS.filter((t) => t.pos > player.index).map((t) => t.pos)} showPadding={false} />
            <span className="sr-only">{revealed.length} tokens revealed</span>
          </div>
        </Figure>
      </Section>

      <Section id="charts" title="Charts and stats">
        <div className="wide grid gap-4 sm:grid-cols-3">
          <Stat label="Zero-shot AUROC, RT (paper)" value="69.7" unit="%" delta={3.0} deltaLabel="vs EntityMean" caption="Mean of 10 binary tasks (Table 1, p.8)." emphasis />
          <Stat label="Zero-shot R², RT (paper)" value="22.7" unit="%" delta={3.4} deltaLabel="vs EntityMean" caption="Mean of 8 regression tasks (Table 2, p.8)." />
          <Stat label="Parameters (this site)" value="535,491" caption="Paper: 22,336,131." />
        </div>
        <Figure wide>
          <BarChart
            title="Context ablations: zero-shot AUROC (%)"
            subtitle={
              <>
                Paper Table 4 <Cite kind="paper" ref="p.9 Table 4" />
              </>
            }
            categories={['none', 'col/table names shuffled', 'self labels removed', 'other-entity labels removed']}
            series={[{ id: 'auroc', label: 'zero-shot AUROC', values: [70.1, 69.5, 53.8, 70.6] }]}
            emphasis={['self labels removed']}
            refLines={[{ value: 50, label: 'random = 50' }]}
            domain={[0, 80]}
            valueLabel="AUROC (%)"
          />
        </Figure>
        <Figure wide>
          <BarChart
            title="Attention-layer ablations: R² (%)"
            subtitle={
              <>
                Paper Table 5 <Cite kind="paper" ref="p.10 Table 5" />
              </>
            }
            categories={['none', 'column', 'feature', 'neighbor', 'full']}
            series={[
              { id: 'zs', label: 'zero-shot', values: [22.8, 11.9, 18.8, 22.0, 23.7] },
              { id: 'ft', label: 'fine-tuned', values: [33.2, 31.2, 32.2, 31.4, 32.7] },
            ]}
            valueLabel="R² (%)"
          />
        </Figure>
        <Figure wide>
          <BarChart
            orientation="vertical"
            title="Vertical bars with a negative value (paper Table 4, zero-shot R²)"
            categories={['none', 'names shuffled', 'self removed', 'other removed']}
            series={[{ id: 'r2', label: 'R²', values: [22.8, 20.5, -5.5, 22.9] }]}
            valueLabel="R² (%)"
          />
        </Figure>
        <Figure wide>
          <LineChart
            title="Fine-tuning curves (illustrative fake data)"
            series={curves}
            xScale="log"
            xLabel="fine-tuning step"
            yLabel="val AUROC"
            formatX={(x) => (x >= 1000 ? `${x / 1000}k` : String(x))}
            formatY={(y) => y.toFixed(2)}
            markers={[{ x: 256, label: 'checkpoint' }]}
          />
        </Figure>
        <Figure wide>
          <LineChart
            title="Single series with area (fake)"
            series={[{ id: 'lr', label: 'learning rate', points: Array.from({ length: 50 }, (_, i) => ({ x: i * 1000, y: i < 10 ? 4e-5 + (1e-3 - 4e-5) * (i / 10) : 1e-3 * (1 - (i - 10) / 40) })) }]}
            area
            xLabel="step"
            yLabel="lr"
            formatX={(x) => `${x / 1000}k`}
            formatY={(y) => (y === 0 ? "0" : y.toExponential(1))}
          />
        </Figure>
      </Section>

      <Section id="quiz" title="Quiz">
        <Quiz
          questions={[
            {
              prompt: 'In column attention, which cells can the masked churn label attend to?',
              choices: [
                { text: 'Only cells of its own row.', why: 'That is (part of) feature attention.' },
                { text: 'Every visible churn label in the context, from any split, plus itself.', correct: true },
                { text: 'Every token in the context.', why: 'That is full attention.' },
              ],
              explanation: (
                <>
                  The column mask requires the same column-name id and the same table id; Train/Val/Test splits share both, so all label
                  cells of the task form one group <Cite kind="spec" ref="§6.2" />.
                </>
              ),
            },
          ]}
        />
      </Section>
    </ChapterPage>
  );
}
