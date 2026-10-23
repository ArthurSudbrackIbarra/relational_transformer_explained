/**
 * Paper vs this site's training hyperparameters (spec §9.2). Paper values come from the paper-era
 * scripts and main.py (cited per row); this site's values are read from results.json / curves.json /
 * dataset.json, so they always describe the committed run.
 */
import { Check } from 'lucide-react';
import type { ReactNode } from 'react';
import { Cite } from '@/components';
import { big, fx, int, mainRun, minutes, sci, type Curves, type Dataset, type Results } from './data';

interface Row {
  what: string;
  paper: ReactNode;
  ours: ReactNode;
  same?: boolean;
}

export function HyperTable({ curves, results, dataset }: { curves: Curves | null; results: Results | null; dataset: Dataset | null }) {
  const run = mainRun(curves);
  const ft = curves?.runs['finetune-churn-pretrained'];
  const cfg = results?.config;
  const hp = run?.hp;
  const rows: Row[] = [
    {
      what: 'Initialization',
      paper: (
        <>
          PyTorch defaults after <code>seed_everything(0)</code> <Cite kind="code" ref="rt/main.py:98" />
        </>
      ),
      ours: <>PyTorch defaults after torch.manual_seed(0)</>,
      same: true,
    },
    {
      what: 'Pretraining tasks',
      paper: (
        <>
          every task of the other 6 databases: 29–33 <Cite kind="code" ref="scripts/example_pretrain.py:17" /> <Cite kind="spec" ref="§9.1" />
        </>
      ),
      ours: <>every task of the other 3 databases: {dataset?.pretrain?.numTasks ?? '–'}</>,
    },
    {
      what: 'Optimizer',
      paper: (
        <>
          AdamW, β = (0.9, 0.999), ε = 10⁻⁸ <Cite kind="code" ref="rt/main.py:188-195" />
        </>
      ),
      ours: <>AdamW, β = (0.9, 0.999), ε = 10⁻⁸</>,
      same: true,
    },
    {
      what: 'Weight decay',
      paper: (
        <>
          0.1 on every parameter, one group (norms, biases, mask vectors included) <Cite kind="code" ref="rt/main.py:188-191" />
        </>
      ),
      ours: <>{hp?.wd ?? '–'}, one group</>,
      same: hp?.wd === 0.1,
    },
    {
      what: 'Learning rate',
      paper: (
        <>
          OneCycleLR, peak 10⁻³, 20% linear warmup, linear decay <Cite kind="code" ref="rt/main.py:197-204" />
        </>
      ),
      ours: (
        <>
          {run?.lrSchedule ?? '–'}, peak {sci(hp?.lr, 1)}, 20% linear warmup, linear decay
        </>
      ),
      same: hp?.lr === 1e-3,
    },
    {
      what: 'Gradient clipping',
      paper: (
        <>
          global L2 norm 1.0 <Cite kind="code" ref="scripts/example_pretrain.py:26" />
        </>
      ),
      ours: <>global L2 norm {fx(hp?.max_grad_norm, 1)}</>,
      same: hp?.max_grad_norm === 1,
    },
    {
      what: 'Steps',
      paper: (
        <>
          50,001 <Cite kind="code" ref="scripts/example_pretrain.py:27" />
        </>
      ),
      ours: <>{int(run?.maxSteps)}</>,
    },
    {
      what: 'Batch',
      paper: (
        <>
          32 per GPU × 8 GPUs = 256 <Cite kind="code" ref="scripts/example_pretrain.py:19" /> <Cite kind="spec" ref="§4.14" />
        </>
      ),
      ours: <>{cfg?.batch ?? dataset?.batchSize ?? '–'}, one process</>,
    },
    {
      what: 'Context L, BFS width w',
      paper: (
        <>
          1024, 256 <Cite kind="code" ref="scripts/example_pretrain.py:31" /> <Cite kind="code" ref="scripts/example_pretrain.py:21" />
        </>
      ),
      ours: (
        <>
          {cfg?.L ?? '–'}, {cfg?.w ?? '–'}
        </>
      ),
    },
    {
      what: 'Model',
      paper: (
        <>
          12 blocks, d = 256, 8 heads, d_ff = 1024 <Cite kind="code" ref="scripts/example_pretrain.py:32-35" />
        </>
      ),
      ours: (
        <>
          {cfg?.num_blocks ?? '–'} blocks, d = {cfg?.d_model ?? '–'}, {cfg?.num_heads ?? '–'} heads, d_ff = {cfg?.d_ff ?? '–'}
        </>
      ),
    },
    {
      what: 'Precision',
      paper: (
        <>
          the whole model in bfloat16, optimizer state too; no fp32 master copy <Cite kind="code" ref="rt/main.py:186-187" />
        </>
      ),
      ours: <>{cfg?.dtype ?? 'float32'}</>,
    },
    {
      what: 'Evaluation',
      paper: (
        <>
          step 0, every 1,000 steps and powers of two; at most 40 batches per GPU <Cite kind="code" ref="scripts/example_pretrain.py:9-11" />
        </>
      ),
      ours: <>step 0, every {int(hp?.eval_freq)} steps and powers of two; every seed</>,
    },
    {
      what: 'Tokens seen',
      paper: (
        <>
          ≈ {big(dataset?.paper?.tokensSeen)} (50,001 × 256 × 1024) <Cite kind="spec" ref="§9.1" />
        </>
      ),
      ours: <>{big(dataset?.pretrain?.tokensSeen)}</>,
    },
    {
      what: 'Hardware, time',
      paper: (
        <>
          8×A100, ≈ 2 h <Cite kind="paper" ref="p.6 §4.1" />
        </>
      ),
      ours: (
        <>
          {cfg?.device ?? 'cpu'}, {cfg?.threads ?? '–'} threads, {minutes(run?.wallSeconds)}
        </>
      ),
    },
    {
      what: 'Fine-tuning',
      paper: (
        <>
          32,769 steps, constant 10⁻⁴, wd 0, fresh AdamW <Cite kind="code" ref="scripts/example_finetune.py:24-28" />
        </>
      ),
      ours: (
        <>
          {int(ft?.maxSteps)} steps, constant {sci(ft?.hp?.lr, 0)}, wd {ft?.hp?.wd ?? '–'}, fresh AdamW
        </>
      ),
    },
  ];
  const same = (
    <span className="ml-2 inline-flex translate-y-[-1px] items-center gap-1 rounded-full border border-border px-2 py-px text-[13px] whitespace-nowrap text-ink-2">
      <Check aria-hidden="true" size={13} strokeWidth={2.4} />
      same as paper
    </span>
  );
  return (
    <div className="wide my-7 min-w-0">
      {/* wide screens: a three-column table */}
      <div className="card relative hidden overflow-x-auto md:block">
        <table className="w-full border-collapse text-[15.5px]">
          <thead>
            <tr className="border-b border-grid bg-surface-2 text-left text-ink-2">
              <th scope="col" className="px-4 py-2 font-semibold">
                Setting
              </th>
              <th scope="col" className="px-4 py-2 font-semibold">
                Paper (scripts and main.py)
              </th>
              <th scope="col" className="px-4 py-2 font-semibold">
                This site (results.json)
              </th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.what} className="border-b border-grid/70 align-top last:border-0">
                <th scope="row" className="px-4 py-2 text-left font-medium whitespace-nowrap text-ink">
                  {r.what}
                </th>
                <td className="px-4 py-2 leading-snug text-ink">{r.paper}</td>
                <td className="px-4 py-2 leading-snug text-ink">
                  {r.ours}
                  {r.same && same}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {/* narrow screens: one small card per setting */}
      <ul className="!m-0 grid gap-2 !p-0 md:hidden" style={{ listStyle: 'none' }} aria-label="Training settings, paper vs this site">
        {rows.map((r) => (
          <li key={r.what} className="card !m-0 px-3.5 py-2.5 text-[15.5px] leading-snug">
            <div className="font-semibold text-ink">{r.what}</div>
            <div className="mt-1 grid grid-cols-[auto_minmax(0,1fr)] gap-x-3 gap-y-1">
              <span className="text-[14px] text-ink-2">Paper</span>
              <span className="text-ink">{r.paper}</span>
              <span className="text-[14px] text-ink-2">This site</span>
              <span className="text-ink">
                {r.ours}
                {r.same && same}
              </span>
            </div>
          </li>
        ))}
      </ul>
      <p className="mt-2 text-[15px] text-ink-2">
        Rows marked “same as paper” run the identical setting. Every other row differs only in size, data, precision or hardware, never in what an
        operation computes.
      </p>
    </div>
  );
}
