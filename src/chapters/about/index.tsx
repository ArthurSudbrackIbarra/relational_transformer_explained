/**
 * Chapter 17 "about": How this simulation was built and verified (BUILD_PLAN §4, §6; docs/CHAPTER_BRIEFS.md §17).
 *
 * Nothing in this chapter is typed in by hand when a file or the engine can say it:
 *  - verification numbers, checkpoints, test counts: public/data/verification.json (python/write_verification.py);
 *  - training facts: public/data/training/{dataset,results}.json;
 *  - the in-browser parity run re-measures everything from tests/fixtures/parity-trained*;
 *  - test titles, the spec's verification log, README commands and this site's code excerpts are read raw from
 *    the repository (Vite ?raw), so they are verbatim;
 *  - official code excerpts (CodeRef) are verbatim copies from the paper-era snapshot (./snippets.ts).
 * Spec: research/RT_TECHNICAL_SPEC.md §3.6, §3.7, §4.2, §6.1, §7.2, §7.3, §9, §10 (#16, #20), §13, §14.
 */
import { Link } from 'react-router-dom';
import {
  Callout,
  ChapterHeader,
  ChapterNav,
  ChapterPage,
  Cite,
  CodeRef,
  Figure,
  Quiz,
  ScaleNote,
  Section,
  Stat,
  Term,
} from '@/components';
import { AttentionKernels } from './AttentionKernels';
import { BrowserParity } from './BrowserParity';
import { RunCommands, SamplerCrossCheck } from './CrossCheck';
import { findBlock, int, maxCheckpointDiff, sci, useRawSource, useVerification } from './data';
import { DifferenceCards, RandomnessAudit } from './Differences';
import { CheckpointParity, ThreeImplementations, TsFixtureTable } from './Parity';
import { PipelineMap } from './Pipeline';
import { SiteCode } from './SiteCode';
import { DependencyTable, SourceCards } from './Sources';
import { SpecStats, VerificationLog } from './SpecLog';
import {
  FLY_RNG_371_375,
  MAIN_BF16_186_187,
  MODEL_ATTN_FWD_30_47,
  MODEL_BLOCK_NORMS_71_73,
  MODEL_FLEX_IMPORT_14_17,
  MODEL_MAKE_BLOCK_MASK_89_101,
  README_BIBTEX_155_160,
} from './snippets';
import { TestBrowser } from './TestBrowser';

function Glance() {
  const { data: v, error } = useVerification();
  if (error) return null;
  return (
    <div className="mt-6 grid grid-cols-1 gap-3 min-[480px]:grid-cols-2">
      <Stat label="TypeScript vs PyTorch" value={v ? sci(v.ts_vs_torch_max_abs_diff) : '…'} caption={v ? `largest |Δ| over every activation, ${Object.keys(v.details.tsVsTorch).length} fixtures` : ' '} emphasis />
      <Stat label="Official model.py vs port" value={v ? sci(v.official_vs_port_max_abs_diff) : '…'} caption={v ? `largest |Δ|, with random weights; ${v.checkpoints.length} trained checkpoints × ${v.n_contexts} contexts: ≤ ${sci(maxCheckpointDiff(v))}` : ' '} />
      <Stat label="Sampler cross-check" value={v ? `${v.sampler_crosscheck.mismatches} diffs` : '…'} caption={v ? `${int(v.sampler_crosscheck.contexts)} contexts rebuilt independently` : ' '} />
      <Stat label="Tests" value={v ? `${int(v.details.testsTotal - v.details.testsFailed)}/${int(v.details.testsTotal)}` : '…'} caption={v ? `passing, last full run ${v.date}` : ' '} />
    </div>
  );
}

function PortAttentionCode() {
  const { data } = useRawSource('rtModelPy');
  const b = data ? findBlock(data, /^def dense_masked_attention/, /^\s+return torch\.matmul\(probs, v\)/) : null;
  return (
    <SiteCode
      title="The port: all S × S scores, −∞ where masked, then 0 for rows with no allowed key"
      file="python/rt_model.py"
      start={b?.start}
      code={data ? (b?.code ?? '# dense_masked_attention not found') : null}
      language="python"
    />
  );
}

function EngineAttentionCode() {
  const { data } = useRawSource('opsTs');
  const b = data ? findBlock(data, /^\s+for \(let qi = 0; qi < S; qi\+\+\) \{/, /row\[p - a\] \*= invZ;/) : null;
  return (
    <SiteCode
      title="The browser engine: a softmax over each query’s allowed keys (the CSR list keys[start[q]..start[q+1]])"
      file="src/engine/model/ops.ts"
      start={b?.start}
      code={data ? (b?.code ?? '// loop not found') : null}
      language="tsx"
    />
  );
}

export default function AboutChapter() {
  return (
    <ChapterPage slug="about">
      <ChapterHeader
        slug="about"
        lede={
          <>
            Every figure in this course runs a real, miniature Relational Transformer in your browser. This last chapter opens the workshop: how the
            pieces were built, how each one was checked against the paper-era code, and what differs from the paper.
          </>
        }
        learn={[
          'Which parts were built once offline, which run in your browser, and why both share one sampler',
          'How three implementations of the model were checked against each other, and how to rerun that check here',
          'What the sampler cross-check, the test suite and the specification’s verification log cover',
          'What is scaled down or different, and which differences change the numbers',
          'Where every source comes from, under which license, and how to rebuild the site',
        ]}
      >
        <Glance />
      </ChapterHeader>

      {/* ───────────────────────────────────────────────────────────── 1 */}
      <Section id="pipeline" title="Two halves: built once, run on every change" kicker="The pipeline">
        <p>
          The site has two halves. The <strong>offline half</strong> ran once, on a CPU. It generated four small databases, embedded every string
          with the frozen text encoder, trained the model and checked it. Its results are plain files in{' '}
          <span className="font-mono text-[16px]">public/data/</span>.
        </p>
        <p>
          The <strong>online half</strong> runs in your browser whenever you change a setting. It regenerates the database, preprocesses it, samples
          a <Term id="context-window">context window</Term> and runs the forward pass with the shipped weights and embeddings. No server computes
          anything.
        </p>
        <p>
          One decision holds the two halves together: the same TypeScript preprocessing and sampler produce the training contexts offline and the
          live context here. The model was trained on exactly the kind of token sequence the chapters display, and there is no second sampler that
          could drift from the first. Click a stage, or press play, to see what it does and its numbers right now.
        </p>
        <Figure
          wide
          title="The pipeline, stage by stage"
          caption={
            <>
              Cards marked “shared code” run the same TypeScript in both halves. The dashed band holds the committed files that carry the offline
              results to your browser. Every number in the panel is live: read from those files or computed on this page.
            </>
          }
        >
          <PipelineMap />
        </Figure>
      </Section>

      {/* ───────────────────────────────────────────────────────────── 2 */}
      <Section id="parity" title="Three implementations of one function" kicker="Verification">
        <p>
          A browser cannot run the official model: it is PyTorch code that needs FlexAttention, a compiled attention kernel{' '}
          <Cite kind="code" ref="rt/model.py:14-17" />. So the model exists here three times, and the checks connect them in a chain.
        </p>
        <ul>
          <li>
            The <strong>official class</strong> <span className="font-mono text-[16px]">RelationalTransformer</span> from{' '}
            <span className="font-mono text-[16px]">rt/model.py</span> at the paper-era commit, run unmodified apart from one pinned setting (RMSNorm ε,
            below). It only ever serves as the reference.
          </li>
          <li>
            A <strong>PyTorch port</strong> with the same module and parameter names, so a checkpoint loads into either class. It swaps FlexAttention
            for dense masked attention and trains the model.
          </li>
          <li>
            The <strong>browser engine</strong>, plain TypeScript, which runs every forward pass on this site.
          </li>
        </ul>
        <Figure wide title="What was compared with what" caption="Bounds from public/data/verification.json, written by the verification step of the pipeline.">
          <ThreeImplementations />
        </Figure>
        <p>
          Why are the differences not exactly zero? Every value is a float32, which keeps about seven significant digits. The implementations add the
          same products in different orders (blocked matrix kernels in PyTorch, plain loops in TypeScript), and floating-point addition is not
          associative, so the last bit or two can differ. A difference near 10⁻⁶ on activations of size one to ten is a few float32 steps. A real
          bug, such as a wrong mask entry or a missing norm, moves values by 10⁻² or more.
        </p>

        <h3>Run the parity test in your browser</h3>
        <p>
          <span className="font-mono text-[16px]">python/parity.py</span> exports real contexts together with every intermediate activation of the
          PyTorch model. The test suite then rebuilds each context with the TypeScript engine and compares the two forward passes stage by stage. The
          button below runs the same checks on your machine, with the weights your browser already loaded.
        </p>
        <Figure
          wide
          title="The trained-weights parity test, live"
          caption={
            <>
              Each dot is the largest difference over all values of one stage: the input embedding x0, the output of each of the 20 sublayers
              (four attention layers and the feed-forward network in four blocks, numbered from 0 as in the fixtures), the final norm, the decoders,
              the loss and the prediction. The gray tick marks one float32 step at that stage’s largest value. Dots near their tick are rounding.
            </>
          }
        >
          <BrowserParity />
        </Figure>

        <h3>Every checkpoint against the official class</h3>
        <p>
          The chain only holds if the port equals the official class. <span className="font-mono text-[16px]">parity.py</span> loads each exported
          checkpoint into both, runs them on the same batch of real contexts from all four databases, and compares every decoder output and the
          loss. The official class runs its compiled FlexAttention on CPU.
        </p>
        <Figure wide>
          <CheckpointParity />
        </Figure>
        <Figure wide title="TypeScript vs PyTorch on every fixture, from the last full run">
          <TsFixtureTable />
        </Figure>
        <Callout variant="insight" title="The one setting that had to be pinned: RMSNorm ε">
          <p>
            The official block creates its norms without an ε <Cite kind="code" ref="rt/model.py:71-73" />. PyTorch then uses the machine epsilon of
            the input’s dtype. The paper trained in bfloat16 <Cite kind="code" ref="rt/main.py:186-187" />, where that epsilon is 2⁻⁷ = 0.0078125 under
            the pinned torch 2.6.0; in float32 it would be about 1.2 × 10⁻⁷ <Cite kind="spec" ref="§7.2" />. Running the official class in float32
            unchanged would therefore not reproduce the paper environment. <span className="font-mono text-[16px]">parity.py</span> pins it to 2⁻⁷,
            and the port and the browser engine use 2⁻⁷ as well.
          </p>
          <CodeRef title="No eps argument: the default depends on the dtype" file="rt/model.py" lines="71-73" code={MODEL_BLOCK_NORMS_71_73} />
          <CodeRef title="The whole model is cast to bfloat16 before training" file="rt/main.py" lines="186-187" code={MAIN_BF16_186_187} />
        </Callout>

        <h3>The sampler, rebuilt independently</h3>
        <p>
          The forward-pass checks start from contexts built by the TypeScript sampler, so the sampler needs its own check. A separate Python
          program re-implements <span className="font-mono text-[16px]">pre.rs</span>, <span className="font-mono text-[16px]">fly.rs</span> and the
          masks directly from the Rust and Python sources <Cite kind="spec" ref="§4.3" />, rebuilds thousands of contexts from the raw databases, and
          compares every token, the order in which rows were visited, and the masks. It uses an exact copy of the site’s random-number generator, so
          both programs make the same random draws.
        </p>
        <Figure wide title="Independent re-implementation of preprocessing, sampler and masks">
          <SamplerCrossCheck />
        </Figure>

        <h3>The test suite</h3>
        <p>
          The unit tests pin down the behavior the spec describes: preprocessing statistics on the spec’s toy database, the sampler’s invariants
          (one target, the seed first, parents before children, the temporal filter, the width bound, the hard stop at L), the masks of the spec’s
          worked example, the forward pass against PyTorch, and properties of the generated databases. Every title below is read from the test files
          themselves.
        </p>
        <Figure wide title="Every test, as written">
          <TestBrowser />
        </Figure>

        <h3>The specification and its adversarial check</h3>
        <p>
          The paper and the paper-era code were first distilled into a technical specification, the “spec” cited throughout this course, and the
          engine was written against it. Every non-trivial fact in it carries a citation: a file and line of the official code, a page of the paper, or a
          small script that checked it. The counts below are read from the spec’s text.
        </p>
        <Figure wide>
          <SpecStats />
        </Figure>
        <p>
          Then reviewers with three lenses (model, data, training) tried to break it. Each proposed correction was re-checked against the code, the
          repository’s git history, the arXiv versions of the paper and the PyTorch 2.6.0 source, and the outcome was logged in §14{' '}
          <Cite kind="spec" ref="§14" />. The questions that could not be settled are listed in §13 instead of being guessed{' '}
          <Cite kind="spec" ref="§13" />.
        </p>
        <Figure
          wide
          title="The spec’s verification log (§14)"
          caption="Read from research/RT_TECHNICAL_SPEC.md. “Major” marks a correction that changed a number or a conclusion of the spec."
        >
          <VerificationLog />
        </Figure>
        <p>
          The small scripts behind the “verified locally” marks are kept in <span className="font-mono text-[16px]">research/specchecks/</span>: for
          example <span className="font-mono text-[16px]">onecycle.py</span> prints the exact learning-rate schedule <Cite kind="spec" ref="§9.2" />,{' '}
          <span className="font-mono text-[16px]">check1.py</span> tests RMSNorm’s default ε, FlexAttention on fully masked rows and bfloat16 rounding,
          and <span className="font-mono text-[16px]">toy_rt_w1.mjs</span> produced the worked example of the masks chapter{' '}
          <Cite kind="spec" ref="§6.3" />.
        </p>
      </Section>

      {/* ───────────────────────────────────────────────────────────── 3 */}
      <Section id="differences" title="What is scaled down, and what is different" kicker="Honest differences">
        <p>
          Scale is the only thing this site changes on purpose. Every operation, from the preprocessing formulas to the decoders, follows the
          paper-era code. A handful of other differences come with running in a browser on a laptop. For each one, the question that matters is
          whether it changes the mathematics or only the rounding.
        </p>
        <ScaleNote />
        <DifferenceCards />

        <h3>One function, three kernels</h3>
        <p>
          Attention is the place where the three implementations look most different. All three compute the masked softmax of the paper,{' '}
          softmax(Mask(QKᵀ; M)/√d_head)·V with −∞ where the mask is 0 <Cite kind="paper" ref="p.5 §3.2" />. FlexAttention receives the boolean mask as a{' '}
          <em>block mask</em>, skips key blocks that hold no allowed pair, and treats the remaining masked scores as −∞{' '}
          <Cite kind="code" ref="rt/model.py:89-101" />. The port computes all S × S scores and fills the masked ones with −∞. The browser engine keeps,
          for each query, the list of its allowed keys and computes the softmax and the weighted sum over that list only.
        </p>
        <CodeRef
          title="Official: every layer calls the compiled flex_attention with its block mask"
          file="rt/model.py"
          lines="30-47"
          code={MODEL_ATTN_FWD_30_47}
          highlightLines={[43]}
          note="The SDPA branch (lines 39-41) is never taken: every layer receives a mask (spec §6.1)."
        />
        <CodeRef title="Official: a dense boolean mask becomes a block mask, shared by all heads (H=None)" file="rt/model.py" lines="89-101" code={MODEL_MAKE_BLOCK_MASK_89_101} />
        <CodeRef title="Official: flex_attention is compiled once at import" file="rt/model.py" lines="14-17" code={MODEL_FLEX_IMPORT_14_17} />
        <PortAttentionCode />
        <EngineAttentionCode />
        <p>
          One case needs care: a query with no allowed key. It happens in the neighbor layer for every cell whose row has no child rows in the
          context. A naive dense softmax divides 0 by 0 there and returns NaN. FlexAttention’s compiled kernel replaces the zero sum by 1 and outputs
          0, the port zeroes such rows explicitly, and the engine skips them <Cite kind="spec" ref="§6.1" />. The figure recomputes one sublayer of
          the live forward pass the dense way and compares it with the engine’s weights.
        </p>
        <Figure
          wide
          title="Dense softmax vs the engine’s, on the live context"
          caption={
            <>
              Each strip has one cell per key position of the live context (gray = padding). Raw scores use the diverging scale (blue negative, red
              positive). Weights use the blue sequential scale relative to the row’s largest weight; a weight of exactly 0 stays blank and red marks
              NaN. Hover a cell to read its values. The arrows step through the query positions. This uses the global context, so it follows your
              simulation settings.
            </>
          }
        >
          <AttentionKernels />
        </Figure>

        <h3>What the random numbers decide</h3>
        <p>
          The official sampler seeds Rust’s standard generator with epoch + seed row + seed <Cite kind="code" ref="rustler/src/fly.rs:371-375" />.
          This site uses a different generator with the same seed formula, drawing at the same two points of the algorithm and with the same ranges.
          The spec notes that bit-exact parity with Rust is not needed: any seeded generator gives the same distribution over contexts{' '}
          <Cite kind="spec" ref="§4.2" /> <Cite kind="spec" ref="§13" />. What does change is which rows a given draw selects. Here is where the
          draws happened in the live context.
        </p>
        <CodeRef title="The per-context generator of the official sampler" file="rustler/src/fly.rs" lines="371-375" code={FLY_RNG_371_375} />
        <Figure wide title="Where the sampler used random numbers in the live context">
          <RandomnessAudit />
        </Figure>
        <Callout
          variant="paper-vs-code"
          title="Precision"
          paper={
            <>
              One pretraining run “takes around 2 hours … on 8×A100 GPUs at BFloat16 precision” <Cite kind="paper" ref="p.6 §4.1" />.
            </>
          }
          code={
            <>
              Pure bfloat16: the model is cast to bf16, with no autocast and no float32 master weights, so the optimizer state is bf16 too; RMSNorm’s
              ε is therefore 2⁻⁷ <Cite kind="code" ref="rt/main.py:186-187" /> <Cite kind="spec" ref="§9.2" />. Batch values are rounded to bf16 when
              the batch is built <Cite kind="spec" ref="§3.7" />.
            </>
          }
          verdict={
            <>
              This site runs float32 everywhere and pins ε to 2⁻⁷. The preprocessing chapter shows the bf16 rounding of real cell values (
              <Link to="/preprocessing">chapter 4</Link>).
            </>
          }
        />
      </Section>

      {/* ───────────────────────────────────────────────────────────── 4 */}
      <Section id="sources" title="Sources and licenses" kicker="Credits">
        <p>
          The site depends on three outside sources: the paper, the official code and the text encoder. For each one, the license below is only what
          could be checked directly: the paper’s PDF metadata, the official repository’s git history, and the encoder’s model card.
        </p>
        <SourceCards />
        <Callout
          variant="paper-vs-code"
          title="Which MiniLM?"
          paper={
            <>
              “we use the MiniLMv2 [41] model from SentenceTransformers [33], which produces 384 dimensional embeddings”{' '}
              <Cite kind="paper" ref="p.6 §4.1" />.
            </>
          }
          code={
            <>
              Loads <span className="font-mono text-[15px]">sentence-transformers/all-MiniLM-L12-v2</span>{' '}
              <Cite kind="code" ref="rt/embed.py:14-15" /> <Cite kind="code" ref="rt/embed.py:38" />. The “v2” is the sentence-transformers release
              name, not MiniLMv2 <Cite kind="spec" ref="§10" />.
            </>
          }
          verdict={<>This site uses the model the code loads, all-MiniLM-L12-v2, with mean pooling and L2 normalization.</>}
        />
        <p>
          This course is an independent teaching re-implementation. It is not an official release of the paper’s authors, and its small model is not
          one of their checkpoints. If you use RT in your own work, cite the paper; the official repository gives this entry:
        </p>
        <CodeRef title="Citation entry from the official README" file="README.md" lines="155-160" code={README_BIBTEX_155_160} language="plain" defaultOpen />
        <Figure wide title="Software this site is built with" caption="Versions are read from package.json; licenses are the ones each package declares.">
          <DependencyTable />
        </Figure>
      </Section>

      {/* ───────────────────────────────────────────────────────────── 5 */}
      <Section id="run" title="Run it, and rebuild it" kicker="Do it yourself">
        <p>
          Running the site needs only Node. Rebuilding the offline half needs the Python environment described in{' '}
          <span className="font-mono text-[16px]">python/README.md</span>. After any change to a generator, the training or the engine, run the whole
          verification step again, so that <span className="font-mono text-[16px]">verification.json</span> keeps describing the files that are
          actually shipped. That is why the checkpoint table above lists a SHA-256 for each weight file.
        </p>
        <RunCommands />
        <p>
          A good way in: change a line of <span className="font-mono text-[16px]">src/engine/masks.ts</span>, run the tests and watch the mask and
          parity tests fail. Then revert it.
        </p>
      </Section>

      <Quiz
        questions={[
          {
            prompt: 'The browser engine and PyTorch differ by about 10⁻⁶ at every stage. What does that tell you?',
            choices: [
              { text: 'There is a small bug in the TypeScript code.', why: 'A bug in a mask or a norm moves values by 10⁻² or more, not by a few float32 steps.' },
              { text: 'The two add the same float32 products in a different order, so the last bits differ.', correct: true },
              { text: 'The browser runs in bfloat16.', why: 'Every implementation here runs in float32; bfloat16 would give differences near 10⁻².' },
              { text: 'The two use different masks.', why: 'The masks are compared entry by entry and are identical.' },
            ],
            explanation: 'Floating-point addition is not associative. With about seven significant digits, a different summation order changes the last bit or two.',
          },
          {
            prompt: 'Why is a context built here not bit-identical to the one the official Rust sampler would build for the same seed row?',
            choices: [
              { text: 'The site uses dense attention instead of FlexAttention.', why: 'Attention runs after sampling; it cannot change which rows are in the context.' },
              { text: 'The site’s random-number generator differs from Rust’s, so the random draws pick different rows.', correct: true },
              { text: 'The site runs in float32.', why: 'Precision does not affect which rows the BFS visits.' },
              { text: 'The site changes the sampling algorithm to make it faster.', why: 'The algorithm is the same rule for rule; the cross-check confirms it.' },
            ],
            explanation:
              'Same seed formula, same draw points and ranges, different generator (spec §4.2). The distribution over contexts is the same; individual draws are not.',
          },
          {
            prompt: 'Which difference from the paper changes the numbers the model learns, not only their rounding?',
            choices: [
              {
                text: 'float32 instead of bfloat16, with the same RMSNorm ε',
                why: 'More precision changes how each result is rounded, not what is computed. Same weights in, the same outputs to within rounding.',
              },
              { text: 'Dense masked attention instead of FlexAttention', why: 'Same function; the parity checks agree to about 10⁻⁶.' },
              { text: 'Four synthetic databases and a much shorter training run', correct: true },
              { text: 'A browser engine that loops over allowed keys only', why: 'Masked keys have weight exactly 0 either way.' },
            ],
            explanation: 'Different data and far fewer tokens give different weights and metrics, which is why this site never plots its numbers on the paper’s axes.',
          },
        ]}
      />

      <ChapterNav slug="about" />
    </ChapterPage>
  );
}
