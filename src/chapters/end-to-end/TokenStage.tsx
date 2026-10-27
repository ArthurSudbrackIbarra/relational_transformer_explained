/**
 * Stage 3: the batch dictionary (rustler/src/fly.rs:27-44, 462-500; rt/data.py:116-154; spec §5.1),
 * one row per token, and the two roads its fields take: the ids build the masks (stage 5), the values
 * build the input embedding (stage 4). Three fields are never read by model.py.
 */
import { ArrowRight } from 'lucide-react';
import { Cite, Figure, TokenTable, cx } from '@/components';
import { int } from './format';
import { useE2E } from './live';

const ROADS: { title: string; to: string; fields: string[]; note: string }[] = [
  {
    title: 'Ids',
    to: 'the four masks (stage 5)',
    fields: ['node_idxs', 'f2p_nbr_idxs', 'col_name_idxs', 'table_name_idxs', 'is_padding'],
    note: 'Compared between pairs of tokens; never embedded.',
  },
  {
    title: 'Values',
    to: 'the input embedding (stage 4)',
    fields: ['col_name_values', 'sem_types', 'number_values', 'datetime_values', 'boolean_values', 'text_values', 'masks', 'is_padding'],
    note: 'col_name_values and text_values are 384-wide MiniLM rows, written from col_name_idxs and the text index.',
  },
  {
    title: 'Not read by model.py',
    to: 'the evaluation read-out, or nothing',
    fields: ['is_targets', 'is_task_nodes', 'class_value_idxs'],
    note: 'is_targets only picks the prediction out after the forward pass.',
  },
];

export function TokenStage() {
  const live = useE2E();
  const { ctx, pdb, slots, focus, setFocus, numReal } = live;
  const S = ctx.tokens.length;
  return (
    <Figure
      wide
      title={`The batch dictionary: ${int(S)} tokens, one row each`}
      caption={
        <>
          The official field names, in batch order. Click a row to follow that token on the rest of the page. Values are the stored model inputs
          (z-scores), not the raw cell values; the cell column is for reading only.
        </>
      }
    >
      <div className="mb-4 grid grid-cols-1 gap-2.5 md:grid-cols-3">
        {ROADS.map((r, i) => (
          <div key={r.title} className={cx('rounded-xl border px-3.5 py-3', i === 2 ? 'border-dashed border-border-strong' : 'border-border bg-surface')}>
            <div className="flex flex-wrap items-center gap-1.5 text-[16px] font-semibold text-ink">
              {r.title}
              <ArrowRight aria-hidden="true" size={15} className="text-muted" />
              <span className="font-normal text-ink-2">{r.to}</span>
            </div>
            <div className="mt-2 flex flex-wrap gap-1">
              {r.fields.map((f) => (
                <code key={f} className="!text-[13.5px]">
                  {f}
                </code>
              ))}
            </div>
            <p className="!mt-2 !mb-0 text-[14.5px] leading-snug text-muted">{r.note}</p>
          </div>
        ))}
      </div>
      <TokenTable tokens={ctx.tokens} text={pdb.text} tableSlots={slots} selected={focus} onSelect={setFocus} maxHeight={380} />
      <p className="!mt-3 !mb-0 text-[15px] leading-snug text-ink-2">
        {numReal < S
          ? `${int(numReal)} real tokens and ${int(S - numReal)} padding positions.`
          : `All ${int(S)} positions hold real cells: the sampler filled the context up to L.`}{' '}
        Each field is a flat array the sampler writes position by position <Cite kind="code" ref="rustler/src/fly.rs:462-500" />, reshaped to
        [batch, L] or [batch, L, width] <Cite kind="code" ref="rt/data.py:133-152" />.
      </p>
    </Figure>
  );
}
