/**
 * Step 1 figures: where every symbol of the embedding formula comes from (batch field or parameter,
 * with its shape in our mini model and in the paper), and which of the eight value gates of
 * model.py:216-224 fires for each token of the live context.
 */
import { memo, useMemo, useState, type ReactNode } from 'react';
import { Cite, Figure, SemBadge, Stat, TeX, TokenStrip, cx } from '@/components';
import { ENC_KEYS, MINI_CONFIG, PAPER_CONFIG, SEM_KEYS, paramCount, parameterShapes } from '@/engine/model';
import { SemType, type RTConfig } from '@/engine/types';
import type { EmbeddingStable } from './live';
import { int, rowRef, SEM_LABEL } from './live';
import { useCrossPick } from './nav';

// ─────────────────────────────────────────────────────────────────────────────
// Symbol table
// ─────────────────────────────────────────────────────────────────────────────

export const SymbolTable = memo(function SymbolTable({ config }: { config: RTConfig }) {
  const d = config.dModel;
  const D = PAPER_CONFIG.dModel;
  const dt = config.dText;
  const rows: { sym: string | string[]; what: ReactNode; code: ReactNode; ours: ReactNode; paper: ReactNode }[] = [
    {
      sym: 'c_i',
      what: <>MiniLM embedding of the column phrase “&lt;col&gt; of &lt;table&gt;”</>,
      code: <code>col_name_values</code>,
      ours: `${dt}`,
      paper: `${dt}`,
    },
    {
      sym: String.raw`W_{\mathrm{col}},\,b_{\mathrm{col}}`,
      what: 'One linear layer, shared by every column of every table',
      code: <code>enc_dict["col_name"]</code>,
      ours: `${d}×${dt}, ${d}`,
      paper: `${D}×${dt}, ${D}`,
    },
    {
      sym: 'v_i^{(t)}',
      what: 'The slot of type t: one z-score (number, datetime, boolean) or the MiniLM embedding of the string (text). Every slot is filled; the gate picks one',
      code: (
        <>
          <code>number_values</code>, <code>text_values</code>, …
        </>
      ),
      ours: `1 or ${dt}`,
      paper: `1 or ${dt}`,
    },
    {
      sym: String.raw`W_t,\,b_t`,
      what: 'One linear layer per semantic type',
      code: <code>enc_dict[t]</code>,
      ours: `${d}×1 or ${d}×${dt}`,
      paper: `${D}×1 or ${D}×${dt}`,
    },
    {
      sym: [String.raw`\mathrm{RMSNorm}_{\mathrm{col}},`, String.raw`\mathrm{RMSNorm}_t`],
      what: 'Five RMSNorms, each with its own learned gain γ',
      code: <code>norm_dict</code>,
      ours: `γ: ${d} each`,
      paper: `γ: ${D} each`,
    },
    {
      sym: 'm_t',
      what: 'The mask vector of type t, a trainable parameter',
      code: <code>mask_embs[t]</code>,
      ours: `${d}`,
      paper: `${D}`,
    },
    {
      sym: String.raw`t_i,\ \mathrm{mask}_i,\ \mathrm{pad}_i`,
      what: 'Semantic type code, “is masked”, “is padding”',
      code: (
        <>
          <code>sem_types</code>, <code>masks</code>, <code>is_padding</code>
        </>
      ),
      ours: 'scalars',
      paper: 'scalars',
    },
  ];
  return (
    <div className="wide my-7">
      <div className="card overflow-hidden">
        <div className="hidden grid-cols-[minmax(150px,0.9fr)_minmax(0,2fr)_minmax(0,1.3fr)_minmax(0,1fr)_minmax(0,1fr)] gap-4 border-b border-grid bg-surface-2 px-4 py-2 text-[15px] font-semibold text-ink-2 md:grid">
          <span>Symbol</span>
          <span>What it is</span>
          <span>
            In the code <Cite kind="code" ref="rt/model.py:208-224" /> <Cite kind="spec" ref="§5.1" />
          </span>
          <span>Shape here</span>
          <span>Shape in the paper</span>
        </div>
        {rows.map((r) => (
          <div
            key={String(r.sym)}
            className="grid grid-cols-[auto_minmax(0,1fr)] gap-x-4 gap-y-1 border-b border-grid/70 px-4 py-2.5 last:border-0 md:grid-cols-[minmax(150px,0.9fr)_minmax(0,2fr)_minmax(0,1.3fr)_minmax(0,1fr)_minmax(0,1fr)] md:items-center"
          >
            <span className="flex flex-col text-[17.5px] text-ink">
              {(Array.isArray(r.sym) ? r.sym : [r.sym]).map((t) => (
                <TeX key={t} tex={t} />
              ))}
            </span>
            <span className="text-[16px] leading-snug text-ink-2">{r.what}</span>
            <span className="col-start-2 text-[16px] break-words text-ink md:col-start-auto [&_code]:text-[15px]">{r.code}</span>
            <span className="num col-start-2 text-[15px] text-ink md:col-start-auto">
              <span className="text-muted md:hidden">here </span>
              {r.ours}
            </span>
            <span className="num col-start-2 text-[15px] text-ink-2 md:col-start-auto">
              <span className="text-muted md:hidden">paper </span>
              {r.paper}
            </span>
          </div>
        ))}
      </div>
      <p className="mt-2 text-[15px] leading-snug text-ink-2 [&_code]:text-[14.5px]">
        Batch fields are written by the sampler <Cite kind="code" ref="rustler/src/fly.rs:462-500" />; the parameters are created in{' '}
        <Cite kind="code" ref="rt/model.py:115-146" />. Shapes follow PyTorch’s <code>nn.Linear</code> convention, weight = [out × in]. The 384 MiniLM dimensions are the same in both models;
        only d changes ({d} here, {D} in the paper <Cite kind="paper" ref="p.6 §4.1" />).
      </p>
    </div>
  );
});

// ─────────────────────────────────────────────────────────────────────────────
// Parameter count of the embedding layer
// ─────────────────────────────────────────────────────────────────────────────

/** Parameters of enc_dict, norm_dict and mask_embs (the input embedding) for a config. */
function embeddingParams(cfg: RTConfig) {
  let enc = 0;
  let norm = 0;
  let mask = 0;
  for (const [name, shape] of parameterShapes(cfg)) {
    const n = shape.reduce((a, b) => a * b, 1);
    if (name.startsWith('enc_dict.')) enc += n;
    else if (name.startsWith('norm_dict.')) norm += n;
    else if (name.startsWith('mask_embs.')) mask += n;
  }
  return { enc, norm, mask, total: enc + norm + mask };
}

export const EmbeddingParamStats = memo(function EmbeddingParamStats() {
  const ours = useMemo(() => embeddingParams(MINI_CONFIG), []);
  const paper = useMemo(() => embeddingParams(PAPER_CONFIG), []);
  return (
    <div className="my-6 grid gap-3 sm:grid-cols-3">
      <Stat
        label="Embedding parameters, this site"
        value={int(ours.total)}
        caption={`${int(ours.enc)} in the ${ENC_KEYS.length} encoders, ${int(ours.norm)} in the norms, ${int(ours.mask)} in the mask vectors`}
      />
      <Stat
        label="Embedding parameters, paper"
        value={int(paper.total)}
        caption={
          <>
            {int(paper.enc)} + {int(paper.norm)} + {int(paper.mask)}, out of {int(paramCount(PAPER_CONFIG))} in total <Cite kind="spec" ref="§7.3" />
          </>
        }
      />
      <Stat
        label="MiniLM text encoder"
        value="frozen"
        caption={
          <>
            About 33.4M weights, not part of RT: its vectors are precomputed once <Cite kind="spec" ref="§7.3" />
          </>
        }
      />
    </div>
  );
});

// ─────────────────────────────────────────────────────────────────────────────
// Gate grid
// ─────────────────────────────────────────────────────────────────────────────

type GateKey = `${'value' | 'mask'}-${SemType}` | 'padding';

export const GateGrid = memo(function GateGrid({ live }: { live: EmbeddingStable }) {
  const { gates, ctx, slots, target } = live;
  const { pick, chip, inspected } = useCrossPick(live);
  const tgt = ctx.tokens[target];
  const defaultKey: GateKey = tgt ? (`mask-${tgt.sem}` as GateKey) : 'value-0';
  const [picked, setPicked] = useState<GateKey>(defaultKey);
  const types = [SemType.Number, SemType.Text, SemType.DateTime, SemType.Boolean];
  const positions = (k: GateKey): number[] => {
    if (k === 'padding') return gates.padding;
    const [kind, s] = k.split('-');
    return (kind === 'mask' ? gates.mask : gates.value)[Number(s) as SemType];
  };
  const shown = positions(picked);
  const real = ctx.numReal;

  const cell = (k: GateKey, label: string, sub: ReactNode) => {
    const n = positions(k).length;
    const on = picked === k;
    return (
      <button
        type="button"
        onClick={() => setPicked(k)}
        aria-pressed={on}
        className={cx(
          'flex min-h-[54px] min-w-0 flex-col items-start justify-center rounded-lg border px-3 py-1.5 text-left transition-colors',
          on ? 'border-accent bg-accent-soft' : 'border-border bg-surface hover:border-border-strong hover:bg-surface-2',
          n === 0 && !on && 'opacity-70',
        )}
        title={label}
      >
        <span className={cx('num text-[19px] leading-tight font-semibold', n === 0 ? 'text-muted' : 'text-ink')}>{int(n)}</span>
        <span className="font-mono text-[14px] leading-snug text-ink-2">{sub}</span>
      </button>
    );
  };

  const describe = (k: GateKey) => {
    if (k === 'padding') {
      return gates.padding.length === 0 ? (
        <>The live context fills all {int(ctx.tokens.length)} positions, so no gate here is used: there is no padding.</>
      ) : (
        <>
          {int(gates.padding.length)} padding positions: every gate is 0, the column gate too, so their x₀ is the zero vector.
        </>
      );
    }
    const [kind, s] = k.split('-');
    const sem = Number(s) as SemType;
    const n = positions(k).length;
    const key = SEM_KEYS[sem];
    if (kind === 'mask')
      return n === 0 ? (
        <>
          No token of type {SEM_LABEL[sem]} is masked in this context, so <span className="font-mono">m_{key}</span> is not used here.
        </>
      ) : (
        <>
          {n === 1 ? `One token (${tgt ? rowRef(tgt) : 'masked'}) takes` : `${n} tokens take`} the mask vector <span className="font-mono">m_{key}</span> instead of a value
          term. Only the seed row’s target cell is ever masked <Cite kind="spec" ref="§4.10" />.
        </>
      );
    return n === 0 ? (
      <>No visible {SEM_LABEL[sem]} cell in this context.</>
    ) : (
      <>
        {int(n)} visible {SEM_LABEL[sem]} {n === 1 ? 'cell goes' : 'cells go'} through enc_dict["{key}"] and norm_dict["{key}"]: all of them share
        the same weights, whatever their column.
      </>
    );
  };

  return (
    <Figure
      wide
      title="Which gate fires, for every token of the live context"
      caption={
        <>
          Each count is the number of live tokens whose gate is 1 (model.py:213, 219, 223). Click a count to light those tokens up in the strip;
          click a token to follow it through Step 2. Every one of the {int(real)} real tokens gets the column term and exactly one value-side
          term.
        </>
      }
    >
      <div>
        <div className="grid max-w-[900px] grid-cols-2 gap-2 sm:grid-cols-[130px_minmax(0,1fr)_minmax(0,1fr)]">
          <div className="hidden sm:block" />
          <div className="px-1 text-[15px] leading-snug font-semibold text-ink-2">
            value encoder <span className="font-normal text-muted">RMSNorm(W·v + b), not masked</span>
          </div>
          <div className="px-1 text-[15px] leading-snug font-semibold text-ink-2">
            mask vector <span className="font-normal text-muted">m, masked</span>
          </div>
          {types.map((s) => (
            <GateRow key={s} sem={s}>
              {cell(
                `value-${s}` as GateKey,
                `${SEM_LABEL[s]} encoder`,
                <>
                  enc_dict[
                  <wbr />"{SEM_KEYS[s]}"]
                </>,
              )}
              {cell(
                `mask-${s}` as GateKey,
                `m_${SEM_KEYS[s]}`,
                <>
                  mask_embs[
                  <wbr />"{SEM_KEYS[s]}"]
                </>,
              )}
            </GateRow>
          ))}
          <div className="col-span-2 flex items-center gap-2 px-1 pt-1 text-[15.5px] text-ink-2 sm:col-span-1 sm:pt-0">padding</div>
          <div className="col-span-2 sm:max-w-[50%]">{cell('padding', 'padding', <>every gate 0 → x₀ = 0</>)}</div>
        </div>
      </div>
      <p className="mt-3 mb-2 text-[16px] leading-snug text-ink-2" aria-live="polite">
        {describe(picked)}
      </p>
      <TokenStrip
        tokens={ctx.tokens}
        tableSlots={slots}
        variant="cells"
        highlight={shown}
        dimOthers
        selected={inspected}
        onSelect={pick}
        label="Context tokens; the tokens of the chosen gate are highlighted"
      />
      <p className="!mt-2 !mb-0 text-[15px] leading-snug text-ink-2">
        A square’s color is its table, not its type.{' '}
        {shown.length === 0
          ? 'No token uses the selected gate, so none is highlighted.'
          : shown.length === 1
            ? 'The square at full strength is the one token of the selected gate; the faded ones use another gate.'
            : `The squares at full strength are the ${int(shown.length)} tokens of the selected gate; the faded ones use another gate.`}
      </p>
      {chip}
    </Figure>
  );
});

function GateRow({ sem, children }: { sem: SemType; children: ReactNode }) {
  return (
    <>
      <div className="col-span-2 flex items-center gap-2 px-1 pt-1 text-[15.5px] text-ink sm:col-span-1 sm:pt-0">
        <SemBadge sem={sem} size="sm" showLabel />
      </div>
      {children}
    </>
  );
}
