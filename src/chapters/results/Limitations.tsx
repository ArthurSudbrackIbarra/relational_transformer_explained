/**
 * Limitations (paper p.10 §7) and an interactive illustration of the two-foreign-key limitation:
 * swapping buyer_id and seller_id changes nothing the masks can see.
 */
import { motion, useReducedMotion } from 'motion/react';
import { ArrowLeftRight, Check, Cpu, KeyRound, Link2, Network } from 'lucide-react';
import { useState } from 'react';
import { Link } from 'react-router-dom';
import { Button, Cite, Figure } from '@/components';

const LIMITS = [
  {
    icon: Network,
    title: 'No recommendation or link prediction',
    quote: '“Limitations of RT include inability to handle recommendation or link prediction tasks.”',
    how: (
      <>
        RT casts every task as predicting one masked cell <Cite kind="paper" ref="p.3 §2.2" />. Recommendation and link prediction ask which rows
        should be linked; the paper names supporting them as future work.
      </>
    ),
    link: { slug: 'tasks', label: 'Ch. 3: tasks as masked cells' },
  },
  {
    icon: KeyRound,
    title: 'Key column names are ignored',
    quote: '“it does not incorporate the names of primary-key and foreign-key columns, which often carry useful semantics.”',
    how: (
      <>
        PK and FK columns never become tokens; they only define rows and links, so a name like <code>customer_id</code> is never embedded{' '}
        <Cite kind="spec" ref="§1.2" />.
      </>
    ),
    link: { slug: 'database', label: 'Ch. 2: feature vs key columns' },
  },
  {
    icon: Link2,
    title: 'Two foreign keys to one table look alike',
    quote: '“RT also does not disambiguate between different foreign key columns.”',
    how: 'A row’s parents are stored as a bare list of row ids. Nothing records which key column each id came from. Try it below.',
    link: { slug: 'masks', label: 'Ch. 9: the feature and neighbor masks' },
  },
  {
    icon: Cpu,
    title: 'Simple cell encoders',
    quote:
      '“more advanced cell encoders can be explored, including graph positional encodings, to enhance performance for supervised fine-tuning settings in large-data regimes.”',
    how: (
      <>
        Today each value enters through one linear layer and an RMSNorm per type <Cite kind="code" ref="rt/model.py:115-140" />, and there are no
        positional encodings <Cite kind="spec" ref="§6.5" />.
      </>
    ),
    link: { slug: 'embedding', label: 'Ch. 8: the token embedding' },
  },
];

export function LimitationCards() {
  return (
    <div className="wide my-8 grid gap-3 md:grid-cols-2">
      {LIMITS.map((l) => {
        const Icon = l.icon;
        return (
          <article key={l.title} className="card min-w-0 px-4 py-4">
            <div className="flex items-center gap-2.5">
              <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-surface-2 text-ink-2">
                <Icon aria-hidden="true" size={18} />
              </span>
              <h3 className="!m-0 text-[17.5px] font-semibold text-ink">{l.title}</h3>
            </div>
            <p className="!mt-2 !mb-0 text-[16.5px] leading-snug text-ink">
              {l.quote} <Cite kind="paper" ref="p.10 §7" />
            </p>
            <p className="!mt-1.5 !mb-0 text-[16px] leading-snug text-ink-2">{l.how}</p>
            <Link to={`/${l.link.slug}`} data-plain className="mt-2 inline-block text-[15px] font-medium text-accent-text hover:underline">
              {l.link.label} →
            </Link>
          </article>
        );
      })}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// buyer_id / seller_id
// ─────────────────────────────────────────────────────────────────────────────

export function TwoForeignKeys() {
  const [swapped, setSwapped] = useState(false);
  const reduce = useReducedMotion();
  const buyer = swapped ? 9 : 3;
  const seller = swapped ? 3 : 9;
  const list = [buyer, seller, -1, -1, -1];
  const tr = reduce ? { duration: 0 } : { type: 'spring' as const, stiffness: 260, damping: 24 };
  const cell = 'flex items-center justify-between gap-3 border-b border-grid/70 px-3 py-1.5 last:border-0';
  return (
    <Figure
      title="The paper’s example, made concrete: who bought, who sold?"
      controls={
        <Button size="sm" icon={<ArrowLeftRight aria-hidden="true" size={15} />} onClick={() => setSwapped((s) => !s)} aria-pressed={swapped}>
          Swap buyer and seller
        </Button>
      }
      caption={
        <>
          An illustration of the example on p.10 (row ids are made up). The preprocessor turns each foreign-key value into a parent row id and pushes
          it onto <code>f2p_nbr_idxs</code>; the column name is dropped <Cite kind="code" ref="rustler/src/pre.rs:472-479" />. The feature and
          neighbor masks then only ask whether a row is in that list <Cite kind="code" ref="rt/model.py:170-178" />. Swapping the two values reorders
          the list and changes nothing else the masks see.
        </>
      }
    >
      <div className="grid items-start gap-4 md:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <div className="min-w-0">
          <div className="eyebrow mb-1.5">The database row</div>
          <div className="overflow-hidden rounded-lg border border-border">
            <div className="bg-surface-2 px-3 py-1.5 text-[15px] font-semibold text-ink">product</div>
            <div className={cell}>
              <span className="text-[15px] text-ink-2">product_id (PK)</span>
              <span className="num text-[15px] text-ink">7</span>
            </div>
            <div className={cell}>
              <span className="text-[15px] text-ink-2">buyer_id → user</span>
              <motion.span
                key={`b${buyer}`}
                initial={reduce ? false : { y: swapped ? 14 : -14, opacity: 0 }}
                animate={{ y: 0, opacity: 1 }}
                transition={tr}
                className="num text-[15px] font-semibold text-ink"
              >
                user #{buyer}
              </motion.span>
            </div>
            <div className={cell}>
              <span className="text-[15px] text-ink-2">seller_id → user</span>
              <motion.span
                key={`s${seller}`}
                initial={reduce ? false : { y: swapped ? -14 : 14, opacity: 0 }}
                animate={{ y: 0, opacity: 1 }}
                transition={tr}
                className="num text-[15px] font-semibold text-ink"
              >
                user #{seller}
              </motion.span>
            </div>
            <div className={cell}>
              <span className="text-[15px] text-ink-2">price</span>
              <span className="num text-[15px] text-ink">12.5</span>
            </div>
          </div>
          <p className="!mt-2 !mb-0 text-[15.5px] text-ink-2">
            Only <code>price</code> becomes a token. The three key columns define the structure and never become cells.
          </p>
        </div>
        <div className="min-w-0">
          <div className="eyebrow mb-1.5">What RT receives</div>
          <div className="rounded-lg bg-surface-2 px-3 py-2.5">
            <div className="text-[15px] text-ink-2">
              <code>f2p_nbr_idxs</code> of the price token
            </div>
            <div className="mt-1 flex gap-1.5">
              {list.map((v, i) => (
                <motion.span
                  key={i < 2 ? `p${v}` : `pad${i}`}
                  layout={!reduce}
                  transition={tr}
                  className={`num inline-flex h-9 min-w-[44px] items-center justify-center rounded-md border px-2 text-[15px] ${v < 0 ? 'border-border text-muted' : 'border-accent/50 bg-surface text-ink'}`}
                >
                  {v < 0 ? '−1' : v}
                </motion.span>
              ))}
            </div>
            <div className="mt-3 grid gap-1.5 text-[15.5px] text-ink">
              <div className="flex items-start gap-2">
                <Check aria-hidden="true" size={17} className="mt-0.5 shrink-0 text-[var(--good-text)]" />
                <span>feature mask: user #3 ∈ parents, user #9 ∈ parents</span>
              </div>
              <div className="flex items-start gap-2">
                <Check aria-hidden="true" size={17} className="mt-0.5 shrink-0 text-[var(--good-text)]" />
                <span>neighbor mask: both users see this product as a child</span>
              </div>
              <div className="flex items-start gap-2">
                <Check aria-hidden="true" size={17} className="mt-0.5 shrink-0 text-[var(--good-text)]" />
                <span>no token says “buyer” or “seller”</span>
              </div>
            </div>
          </div>
          <p className="!mt-2 !mb-0 text-[15.5px] text-ink-2" aria-live="polite">
            {swapped ? 'Swapped: the list order changed, the parent set {3, 9} did not.' : 'Press the button to swap the two roles.'}
          </p>
        </div>
      </div>
    </Figure>
  );
}
