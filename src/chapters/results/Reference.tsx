/**
 * The course glossary (every entry of src/components/glossary.ts, searchable) and the references.
 */
import { ExternalLink, Search } from 'lucide-react';
import { useMemo, useState } from 'react';
import { Cite, cx, GLOSSARY, type GlossaryEntry } from '@/components';
import { CODE, PAPER } from '@/components/prose/sources';

export function GlossaryList() {
  const [q, setQ] = useState('');
  const [flash, setFlash] = useState<string | null>(null);
  const entries = useMemo(
    () =>
      (Object.entries(GLOSSARY) as [string, GlossaryEntry][])
        .map(([id, e]) => ({ id, ...e }))
        .sort((a, b) => a.term.localeCompare(b.term, 'en', { sensitivity: 'base' })),
    [],
  );
  const needle = q.trim().toLowerCase();
  const shown = needle ? entries.filter((e) => `${e.term} ${e.def}`.toLowerCase().includes(needle)) : entries;
  const byId = new Map(entries.map((e) => [e.id, e]));
  const jump = (id: string) => {
    setQ('');
    window.setTimeout(() => {
      document.getElementById(`gl-${id}`)?.scrollIntoView({ behavior: 'smooth', block: 'center' });
      setFlash(id);
      window.setTimeout(() => setFlash(null), 1400);
    }, 30);
  };
  return (
    <div className="wide my-6">
      <label className="relative flex max-w-[420px] items-center">
        <span className="sr-only">Search the glossary</span>
        <Search aria-hidden="true" size={16} className="pointer-events-none absolute left-3 text-muted" />
        <input
          type="search"
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder={`Search ${entries.length} terms`}
          className="h-10 w-full rounded-lg border border-border-strong bg-surface pr-3 pl-9 text-[16px] text-ink placeholder:text-muted focus-visible:outline-2 focus-visible:outline-accent"
        />
      </label>
      <dl className="mt-4 grid gap-x-6 gap-y-0 md:grid-cols-2">
        {shown.map((e) => (
          <div
            key={e.id}
            id={`gl-${e.id}`}
            className={cx('scroll-mt-24 border-b border-grid py-3 transition-colors', flash === e.id && 'rounded-lg bg-accent-soft')}
          >
            <dt className="text-[17px] font-semibold text-ink">{e.term}</dt>
            <dd className="!m-0 mt-0.5 text-[16px] leading-snug text-ink-2">
              {e.def}{' '}
              {e.cite.map((c, i) => (
                <Cite key={i} kind={c.kind} ref={c.ref} />
              ))}
              {e.see && e.see.length > 0 && (
                <span className="mt-1 block text-[15px]">
                  See also:{' '}
                  {e.see
                    .filter((s) => byId.has(s))
                    .map((s, i) => (
                      <span key={s}>
                        {i > 0 && ', '}
                        <button type="button" onClick={() => jump(s)} className="font-medium text-accent-text hover:underline">
                          {byId.get(s)!.term}
                        </button>
                      </span>
                    ))}
                </span>
              )}
            </dd>
          </div>
        ))}
      </dl>
      {shown.length === 0 && <p className="mt-3 text-[16px] text-ink-2">No term matches “{q}”.</p>}
    </div>
  );
}

interface RefItem {
  key: string;
  authors: string;
  title: string;
  venue: string;
  url?: string;
  why: string;
}

const REFS: RefItem[] = [
  {
    key: 'RT',
    authors: 'R. Ranjan, V. Hudovernik, M. Znidar, C. Kanatsoulis, R. Upendra, M. Mohammadi, J. Meyer, T. Palczewski, C. Guestrin, J. Leskovec',
    title: PAPER.title,
    venue: `${PAPER.venue}. arXiv ${PAPER.arxiv} (2 Mar 2026)`,
    url: PAPER.url,
    why: 'The paper this course explains. Every “p.N” citation points to its printed pages.',
  },
  {
    key: 'Code',
    authors: 'snap-stanford (GitHub organization)',
    title: `${CODE.repo}, commit ${CODE.commit}`,
    venue: 'GitHub, paper-era snapshot (2026-02-04)',
    url: `${CODE.url}/tree/${CODE.commitFull}`,
    why: 'The official implementation. Every “file:line” citation and every code excerpt comes from this commit.',
  },
  {
    key: 'RelBench',
    authors: 'J. Robinson, R. Ranjan, W. Hu, K. Huang, J. Han, A. Dobles, M. Fey, J. E. Lenssen, Y. Yuan, Z. Zhang, et al.',
    title: 'RelBench: A Benchmark for Deep Learning on Relational Databases',
    venue: 'Advances in Neural Information Processing Systems 37, 2024',
    url: 'https://relbench.stanford.edu',
    why: 'The seven databases and the forecasting tasks behind every table in this chapter (paper ref. [34]).',
  },
  {
    key: 'RelBench v2',
    authors: 'J. Gu, R. Ranjan, C. Kanatsoulis, et al.',
    title: 'RelBench v2: A large-scale benchmark and repository for relational data',
    venue: 'arXiv 2602.12606, 2026',
    url: 'https://arxiv.org/abs/2602.12606',
    why: 'Cited with RelBench as the source of datasets and tasks (paper ref. [18]).',
  },
  {
    key: 'FlexAttention',
    authors: 'J. Dong, B. Feng, D. Guessous, Y. Liang, H. He',
    title: 'Flex Attention: A Programming Model for Generating Optimized Attention Kernels',
    venue: 'arXiv 2412.05496, 2024',
    url: 'https://arxiv.org/abs/2412.05496',
    why: 'How the code applies the four attention masks as sparse block masks (paper ref. [8]).',
  },
  {
    key: 'LLaMA',
    authors: 'H. Touvron, T. Lavril, G. Izacard, et al.',
    title: 'LLaMA: Open and Efficient Foundation Language Models',
    venue: 'arXiv 2302.13971, 2023',
    url: 'https://arxiv.org/abs/2302.13971',
    why: 'The source of RT’s gated SiLU (SwiGLU) feed-forward layer (paper ref. [38]).',
  },
  {
    key: 'all-MiniLM-L12-v2',
    authors: 'sentence-transformers',
    title: 'all-MiniLM-L12-v2 model card',
    venue: 'Hugging Face',
    url: 'https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2',
    why: 'The frozen 384-dimensional text encoder that embeds column phrases, table names and text values.',
  },
  {
    key: 'Sentence-BERT',
    authors: 'N. Reimers, I. Gurevych',
    title: 'Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks',
    venue: 'EMNLP-IJCNLP 2019',
    why: 'The SentenceTransformers library behind the text encoder (paper ref. [33]).',
  },
  {
    key: 'Griffin',
    authors: 'Y. Wang, X. Wang, Q. Gan, M. Wang, Q. Yang, D. Wipf, M. Zhang',
    title: 'Griffin: Towards a Graph-Centric Relational Database Foundation Model',
    venue: 'ICML 2025',
    why: 'The main pretrained baseline in Tables 1, 2 and 10 (paper ref. [42]).',
  },
  {
    key: 'Gemma 3',
    authors: 'Gemma Team, Google DeepMind',
    title: 'Gemma 3 Technical Report',
    venue: 'arXiv 2503.19786, 2025',
    url: 'https://arxiv.org/abs/2503.19786',
    why: 'The LLM baselines of Tables 1 and 16 (paper ref. [37]).',
  },
  {
    key: 'RelLLM',
    authors: 'F. Wu, V. P. Dwivedi, J. Leskovec',
    title: 'Large Language Models are Good Relational Learners',
    venue: 'arXiv 2506.05725, 2025',
    url: 'https://arxiv.org/abs/2506.05725',
    why: 'The RelLLM baseline of Tables 1 and 10 (paper ref. [43]).',
  },
  {
    key: 'RelGT',
    authors: 'V. P. Dwivedi, S. Jaladi, Y. Shen, F. López, C. I. Kanatsoulis, R. Puri, M. Fey, J. Leskovec',
    title: 'Relational Graph Transformer',
    venue: 'arXiv 2505.10960, 2025',
    url: 'https://arxiv.org/abs/2505.10960',
    why: 'A schema-specific baseline in Table 10 (paper ref. [11]).',
  },
];

export function References() {
  return (
    <ol className="wide my-6 !list-none grid gap-2.5 !p-0">
      {REFS.map((r, i) => (
        <li key={r.key} className="!m-0 grid grid-cols-[34px_minmax(0,1fr)] gap-2 border-b border-grid pb-2.5 !pl-0">
          <span className="num pt-0.5 text-[15px] text-muted">[{i + 1}]</span>
          <div className="min-w-0">
            <div className="text-[16px] leading-snug text-ink">
              {r.authors}
              {r.authors.endsWith('.') ? ' ' : '. '}
              <span className="font-semibold">{r.title}</span>. {r.venue}.
              {r.url && (
                <a
                  href={r.url}
                  target="_blank"
                  rel="noreferrer noopener"
                  data-plain
                  className="ml-1.5 inline-flex items-center gap-1 align-baseline text-[15px] font-medium text-accent-text hover:underline"
                >
                  link <ExternalLink aria-hidden="true" size={13} />
                </a>
              )}
            </div>
            <div className="mt-0.5 text-[15.5px] leading-snug text-ink-2">{r.why}</div>
          </div>
        </li>
      ))}
      <li className="!m-0 grid grid-cols-[34px_minmax(0,1fr)] gap-2 !pl-0">
        <span className="num pt-0.5 text-[15px] text-muted">[{REFS.length + 1}]</span>
        <div className="text-[16px] leading-snug text-ink">
          <span className="font-semibold">Relational Transformer: technical specification for a faithful re-implementation</span>, this project’s
          ground-truth document (<code>research/RT_TECHNICAL_SPEC.md</code>). Every “§” citation points to it <Cite kind="spec" ref="§0" />.
        </div>
      </li>
    </ol>
  );
}
