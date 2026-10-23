/**
 * Chapter 5 "semantics": Text and schema semantics (BUILD_PLAN §6, docs/CHAPTER_BRIEFS.md §5).
 *
 * How names and strings enter RT through a frozen text encoder, and why that helps transfer across
 * schemas. Every vector shown is a stored all-MiniLM-L12-v2 embedding (public/data/embeddings);
 * every similarity is computed in the browser; the live context comes from the global store.
 * Spec: research/RT_TECHNICAL_SPEC.md §3.6, §10 #16, §11.2, §11.7.
 */
import { useEffect, useMemo, useState } from 'react';
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
  TeX,
  Term,
} from '@/components';
import type { DbName } from '@/engine/db';
import { useAssets } from '@/state/hooks';
import { BreakId, CodeTabs, DbTag, Loading, LoadError, Nw, Phrase, Quoted, useMedia } from './bits';
import { DB_SHORT, ENCODER, ROLE_LABEL, TABLE11_CLF, TABLE11_REG, TABLE4, f1, f2, f3, int, pcaVarianceKept, signed1, useCatalog, useEmbeddingStore, usePca } from './data';
import { LearnedProjection, FrozenPipeline } from './FrozenPipeline';
import { compareTarget, groupMeans, keyMeans, mapStats, pairCos, topKeyPair, topTokenPair } from './insights';
import { rowRef, useSemanticsLive } from './live';
import { AblationDumbbells, OverlapTables, Table4Grid } from './NamesAblation';
import { PhraseList, StringTable } from './PhraseList';
import { PhraseMap } from './PhraseMap';
import { SimilarityMatrix } from './SimilarityMatrix';
import {
  DATA_COLUMN_INDEX_31_43,
  EMBED_MAIN_65_72,
  EMBED_TEXT_EMBEDDER_12_31,
  FLY_COPY_469_486,
  FLY_GET_TEXT_EMB_528_532,
  MODEL_EMBED_208_224,
  MODEL_ENC_DICT_115_123,
  PRE_INTERN_424_437,
  PRE_TEXT_595_605,
} from './snippets';
import { SharedValues, TextCells } from './TextCells';
import { VectorExplorer } from './VectorExplorer';

const HF_MODULES = 'https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2/raw/main/modules.json';

export default function SemanticsChapter() {
  const live = useSemanticsLive();
  const catalog = useCatalog();
  const storeState = useEmbeddingStore();
  const pca = usePca();
  const assets = useAssets();
  const store = storeState.status === 'ready' ? storeState.value : null;

  // The chapter's focus phrase (shared by steps 1-4 and 7); starts at the live task's target phrase.
  const [focus, setFocus] = useState<string>(live.targetPhrase);
  useEffect(() => setFocus(live.targetPhrase), [live.targetPhrase]);
  const [compare, setCompare] = useState<string | null>(null);
  const [view, setView] = useState<DbName>(live.dbName);
  useEffect(() => setView(live.dbName), [live.dbName]);
  const [showKeys, setShowKeys] = useState(true);

  const cmp = useMemo(() => (store && catalog ? compareTarget(store, catalog, live.dbName, live.targetPhrase) : null), [store, catalog, live.dbName, live.targetPhrase]);
  const keyPair = useMemo(() => (store && catalog ? topKeyPair(store, catalog, live.dbName) : null), [store, catalog, live.dbName]);
  const tokenPair = useMemo(() => (store && catalog ? topTokenPair(store, catalog, live.dbName) : null), [store, catalog, live.dbName]);
  const gm = useMemo(() => (store && catalog ? groupMeans(store, catalog) : null), [store, catalog]);
  const pcaPoints = pca.status === 'ready' ? pca.value : null;
  const varKept = useMemo(() => (pcaPoints && store ? pcaVarianceKept(pcaPoints, store) : null), [pcaPoints, store]);
  const mstats = useMemo(() => (pcaPoints && catalog ? mapStats(pcaPoints, catalog) : null), [pcaPoints, catalog]);

  // default comparison (until the reader picks one): for a phrase, the closest phrase of another
  // database that reaches a token; for any other string, the closest embedded string
  const [compareWith, compareReason] = useMemo((): [string, string | null] => {
    if (compare && store?.has(compare)) return [compare, null];
    if (cmp && focus === live.targetPhrase && cmp.nearest[0]) return [cmp.nearest[0].phrase.text, 'auto: the closest phrase of another database'];
    if (store && catalog && store.has(focus)) {
      const fp = catalog.byText.get(focus);
      const pool = fp ? catalog.phrases.filter((p) => p.db !== fp.db && p.onToken).map((p) => p.text) : store.strings;
      let best = live.targetPhrase;
      let bc = -2;
      for (const t of pool) {
        if (t === focus) continue;
        const c = cosineOf(store, focus, t);
        if (c > bc) (bc = c), (best = t);
      }
      return [best, fp ? 'auto: the closest phrase of another database' : 'auto: the closest of all embedded strings'];
    }
    return [live.targetPhrase, 'auto: the live target phrase'];
  }, [compare, store, catalog, cmp, focus, live.targetPhrase]);

  const entityText = live.entityNode?.cells.filter((c) => c.sem === 1) ?? [];
  const sampleValue = entityText.find((c) => c.colName === 'country')?.raw ?? entityText[0]?.raw;
  const presets = useMemo(() => {
    const p: { label: string; text: string }[] = [{ label: 'target', text: live.targetPhrase }];
    const feat = catalog?.byDb[live.dbName].find((x) => x.role === 'feature' && x.dtype !== 'str');
    if (feat) p.push({ label: 'feature', text: feat.text });
    if (typeof sampleValue === 'string') p.push({ label: 'value', text: sampleValue });
    if (live.entityNode) p.push({ label: 'table name', text: live.entityNode.tableName });
    return p.filter((x) => !store || store.has(x.text));
  }, [live.targetPhrase, live.dbName, live.entityNode, catalog, sampleValue, store]);

  // ShopDB is the held-out target; when another database is live, the "other three" include ShopDB.
  const others = live.dbName === 'shop' ? 'pretraining' : 'other';
  const fkExample = catalog?.byDb[live.dbName].find((p) => p.role === 'fk')?.text ?? 'customer_id of orders';
  const targetPieces = ENCODER.wordpieces[live.targetPhrase] ?? [];
  const splitWords = targetPieces.filter((t) => t.startsWith('##'));
  const text0 = live.pdb.text[0];
  const clfDrops = [...TABLE11_CLF].map((r) => ({ name: `${r.db} ${r.task}`, d: r.zs[1] - r.zs[0] })).sort((a, b) => a.d - b.d);
  const regDrops = [...TABLE11_REG].map((r) => ({ name: `${r.db} ${r.task}`, d: r.zs[1] - r.zs[0] })).sort((a, b) => a.d - b.d);
  const clfHurt = clfDrops.filter((r) => r.d < 0).length;
  const regHurt = regDrops.filter((r) => r.d < 0).length;
  const d = assets.weights?.config.dModel ?? 64;
  const narrow = useMedia('(max-width: 639px)');
  const km = useMemo(() => (store && catalog ? keyMeans(store, catalog) : null), [store, catalog]);
  const ageCos = store ? pairCos(store, 'age of customers', 'age of students') : null;
  const suggestions = useMemo(() => {
    const seen = new Set<string>();
    const out: string[] = [];
    for (const t of [...presets.map((p) => p.text), ...(catalog?.byDb[live.dbName].map((p) => p.text) ?? [])]) {
      if (seen.has(t) || (store && !store.has(t))) continue;
      seen.add(t);
      out.push(t);
    }
    return out;
  }, [presets, catalog, live.dbName, store]);

  return (
    <ChapterPage slug="semantics">
      <ChapterHeader
        slug="semantics"
        lede={
          <>
            RT reads the names in a database. Every cell token carries a vector for the phrase “column of table”, made by a frozen sentence
            encoder, and every text value is embedded the same way. This chapter opens those vectors and asks how much they help a model move to
            a database it has never seen.
          </>
        }
        learn={[
          <>Which strings RT embeds, and the exact phrase “&lt;col&gt; of &lt;table&gt;”</>,
          'What a real 384-dimensional all-MiniLM-L12-v2 vector looks like, and why its length is 1',
          'How close the column phrases of ShopDB are to those of the pretraining databases',
          'What it means that a text value is embedded without its column, and that the encoder is frozen',
          'What the paper’s ablations say about the value of names',
        ]}
      >
        <p className="mt-6 text-[17px] leading-relaxed text-ink-2">
          <span className="eyebrow mr-2">Live context</span>
          seed <span className="text-ink">{live.target ? rowRef(live.target) : '–'}</span>
          {live.entityNode && (
            <>
              {' '}
              → <span className="text-ink">{rowRef(live.entityNode)}</span>
              {live.entityName ? ` (${live.entityName})` : ''}
            </>
          )}
          {' · '}target phrase <Phrase text={live.targetPhrase} /> {' · '}
          {int(live.numReal)} tokens use {live.tokensPerPhrase.size} distinct phrases <ScaleNote inline />
        </p>
      </ChapterHeader>

      {/* ───────────────────────────────────────────────────────────── 1 */}
      <Section id="phrase" title="Every cell carries the name of its column" kicker="Step 1">
        <p>
          RT describes a cell as a triple <TeX tex="(v, c, t)" />: its value, its column name and its table name{' '}
          <Nw>
            <Cite kind="paper" ref="p.4 §3.1" />.
          </Nw>{' '}
          The value is normalized (numbers, dates and booleans; chapter 4) or embedded (text; Step 5) and becomes the token’s value term. The
          other two are joined into a short English phrase, “&lt;column_name&gt; of &lt;table_name&gt;”, and embedded by a frozen text encoder.
          The paper’s examples are “price of product” and “age of user”, and it calls this{' '}
          <Nw>
            <em>schema semantics</em> <Cite kind="paper" ref="p.5 §3.1" />.
          </Nw>{' '}
          In the code, the phrase is one format{' '}
          <Nw>
            string <Cite kind="code" ref="rustler/src/pre.rs:432" />.
          </Nw>
        </p>
        <p>
          Every <Term id="cell-token">cell token</Term> carries the vector of its <Term id="column-phrase">column phrase</Term>, whatever its
          type: a number, a date, a boolean, a string, and the <Term id="target-cell">masked target</Term>{' '}
          <Nw>
            too <Cite kind="spec" ref="§3.6" /> <Cite kind="code" ref="rustler/src/fly.rs:472-477" />.
          </Nw>{' '}
          In the live context the target is the cell <Phrase text={live.targetPhrase} />, so that phrase’s vector is part of the very token the
          model must fill in.
        </p>
        <Figure
          wide
          title="The column phrases of a database"
          caption={
            <>
              Every phrase that preprocessing interns, grouped by table. Struck-through phrases belong to key columns: they get a string index but
              never a cell. Bars count the tokens of the live context that carry each phrase. Switching the database here does not change the
              live simulation.
            </>
          }
        >
          <PhraseList
            live={live}
            catalog={catalog}
            view={view}
            setView={setView}
            showKeys={showKeys}
            setShowKeys={setShowKeys}
            focus={focus}
            onFocus={setFocus}
          />
        </Figure>
        <p>
          Why a phrase rather than a column id? One of the paper’s three design principles is to “enable zero-shot generalization across
          heterogeneous{' '}
          <Nw>
            schemas” <Cite kind="paper" ref="p.4 §3" />.
          </Nw>{' '}
          Our explanation of how a phrase serves it: an integer id means nothing outside its own database, while a phrase can be compared
          across databases. “age of customers” in ShopDB and “age of students” in SchoolDB are different columns, but they share words
          {ageCos !== null && gm ? (
            <>
              , and the encoder places them {ageCos > 2 * gm.crossDb ? 'close together' : 'only a little closer than average'}: cosine{' '}
              {f2(ageCos)}, against {f2(gm.crossDb)} for an average pair of phrases from different databases
            </>
          ) : null}
          . Step 3 measures such closeness for every pair.
        </p>
        <Callout variant="note" title="What the paper says about names">
          <p>
            For its cell encoders the paper took inspiration from PORTAL, a tabular model “which has a similar handling of column{' '}
            <Nw>
              names” <Cite kind="paper" ref="p.16 App. B" />.
            </Nw>{' '}
            Its Figure 1 sums up the token embedding as “trainable datatype-specific encoding of cell values and frozen language model (LM)
            embeddings of table/column{' '}
            <Nw>
              names” <Cite kind="paper" ref="p.2 Fig. 1" />.
            </Nw>
          </p>
          <p>
            Names could also tell columns apart inside a single context. Arguing that full attention could emulate column attention, the paper
            notes, as a possibility in its discussion of expressivity, that some global heads “can learn to restrict attention to tokens from the
            same column by exploiting table and column name embeddings in their query–key{' '}
            <Nw>
              construction” <Cite kind="paper" ref="p.19 App. E" />.
            </Nw>
          </p>
        </Callout>
        <p>
          The <Term id="primary-key">primary-key</Term> and <Term id="foreign-key">foreign-key</Term> columns get a phrase as well, because
          pre.rs interns the phrase before it checks the column’s{' '}
          <Nw>
            role <Cite kind="code" ref="rustler/src/pre.rs:432-442" />.
          </Nw>{' '}
          But a primary key produces no cell, and a foreign key only becomes a link to the parent{' '}
          <Nw>
            row <Cite kind="spec" ref="§3.6" />.
          </Nw>{' '}
          So <Phrase text={fkExample} /> has a vector that no token ever uses. The paper lists this among RT’s limitations: it “does not
          incorporate the names of primary-key and foreign-key columns, which often carry useful{' '}
          <Nw>
            semantics” <Cite kind="paper" ref="p.10 §7" />.
          </Nw>
        </p>
        <p>
          The table name reaches a token only through the phrase. Each token also carries <code>table_name_idx</code>, but only as an integer:
          the column mask compares it <Cite kind="code" ref="rt/model.py:181-183" />, and the sampler uses it to recognize task{' '}
          <Nw>
            tables <Cite kind="code" ref="rustler/src/fly.rs:417" />.
          </Nw>{' '}
          No vector is ever looked up through <code>table_name_idx</code>; a table name’s vector reaches a token only as the gated-out{' '}
          <code>text[0]</code> filler of Step{' '}
          <Nw>
            5 <Cite kind="spec" ref="§3.6" />.
          </Nw>
        </p>

        <h3>One string table per database</h3>
        <p>
          Preprocessing collects every string of a database into one list in order of first appearance: table names, column phrases and each
          distinct string value <Cite kind="code" ref="rustler/src/pre.rs:406-437" /> <Cite kind="code" ref="rustler/src/pre.rs:596-597" />. It
          writes the list out as{' '}
          <Nw>
            <code>text.json</code> <Cite kind="code" ref="rustler/src/pre.rs:624-630" />.
          </Nw>{' '}
          Identical strings share one index, and therefore one{' '}
          <Nw>
            vector <Cite kind="spec" ref="§3.6" />.
          </Nw>
        </p>
        <Figure
          title={`The string table of ${DB_SHORT[view]}`}
          caption={
            <>
              The first entries of the string table, in index order. Interning walks table by table and column by column, so names, phrases and
              values interleave. The order and the index numbers are this site’s: pre.rs visits the tables in hash-map{' '}
              <Nw>
                order <Cite kind="code" ref="rustler/src/pre.rs:75" /> <Cite kind="code" ref="rustler/src/pre.rs:415" />,
              </Nw>{' '}
              so its <code>text.json</code> lists the same strings in another order. Indices are only ever compared for equality{' '}
              <Cite kind="code" ref="rt/model.py:181-183" /> or used to fetch a string’s vector{' '}
              <Cite kind="code" ref="rustler/src/fly.rs:472-486" />, so this changes nothing the model sees.
            </>
          }
        >
          <StringTable catalog={catalog} view={view} />
        </Figure>
        <CodeRef
          title="Interning the table name and the phrase “<col> of <table>”"
          file="rustler/src/pre.rs"
          lines="424-437"
          code={PRE_INTERN_424_437}
          highlightLines={[432, 433, 434]}
          note="text_to_idx is the single string table of the database; an existing string keeps its index. column_name_to_idx becomes column_index.json."
        />
        <p>
          The phrase is also how a task names its target. To find the column to mask, the training code looks up “&lt;col&gt; of &lt;table&gt;”
          in <code>column_index.json</code> <Cite kind="code" ref="rt/data.py:31-43" />.
        </p>
        <CodeRef title="get_column_index: a target column is found through its phrase" file="rt/data.py" lines="31-43" code={DATA_COLUMN_INDEX_31_43} highlightLines={[36]} />
      </Section>

      {/* ───────────────────────────────────────────────────────────── 2 */}
      <Section id="vector" title="A real 384-dimensional vector" kicker="Step 2">
        <p>
          The <Term id="text-encoder">text encoder</Term> is{' '}
          <code className="!whitespace-normal">
            <BreakId text="sentence-transformers/all-MiniLM-L12-v2" />
          </code>{' '}
          <Cite kind="code" ref="rt/embed.py:14-20" /> <Cite kind="code" ref="rt/embed.py:38" />: a {ENCODER.layers}-layer BERT with hidden size{' '}
          {ENCODER.hidden} and a lower-casing WordPiece tokenizer <Cite kind="spec" ref="§3.6" />. Its pipeline has three modules, Transformer →
          mean pooling → Normalize <Cite kind="url" ref={HF_MODULES} title="all-MiniLM-L12-v2 modules.json" />. In words: split the string into
          wordpieces, run all {ENCODER.layers} layers to get one {ENCODER.hidden}-dimensional vector per wordpiece, average them, and divide the
          average by its length.
        </p>
        <Callout
          variant="paper-vs-code"
          title="Which MiniLM?"
          paper={
            <>
              “the MiniLMv2 [41] model from SentenceTransformers [33], which produces 384 dimensional embeddings” <Cite kind="paper" ref="p.6 §4.1" />
            </>
          }
          code={
            <>
              <code className="!whitespace-normal">
                SentenceTransformer(
                <br />
                {'  '}"<BreakId text="sentence-transformers/all-MiniLM-L12-v2" />")
              </code>{' '}
              <Cite kind="code" ref="rt/embed.py:14-20" />{' '}
              <Cite kind="code" ref="rt/embed.py:38" />
            </>
          }
          verdict="This site uses the model the code loads. The “-v2” in its name is the sentence-transformers release, not MiniLMv2 (spec §10 #16)."
        />
        <Figure
          wide
          title="Inside the encoder, for one string"
          caption={
            <>
              Any of the {store ? int(store.strings.length) : '…'} strings embedded offline for the four databases. The wordpieces come from the real
              tokenizer; the vector is the stored one. Hover a cell to read a dimension. The comparison rows show that the cosine of two unit
              vectors is the sum of their {ENCODER.hidden} element-wise products.
            </>
          }
        >
          {store ? (
            <VectorExplorer
              store={store}
              catalog={catalog}
              value={focus}
              onChange={setFocus}
              compare={compareWith}
              onCompare={setCompare}
              presets={presets}
            />
          ) : storeState.status === 'error' ? (
            <LoadError error={storeState.error} />
          ) : (
            <Loading height={420} />
          )}
        </Figure>
        <p>
          Two details in the figure deserve a look. First, words missing from the {int(ENCODER.vocab)}-entry vocabulary are split into smaller
          pieces, and every piece after the first is marked “##”
          {splitWords.length > 0 ? (
            <>
              ; the target phrase becomes {targetPieces.length} wordpieces, <span>{targetPieces.join(' ')}</span>
            </>
          ) : null}
          . The encoder still returns one vector for the whole string. Second, the mean runs over every wordpiece, [CLS] and [SEP] included: we
          checked that this reproduces all stored vectors, and that leaving the two special tokens out does not.
        </p>
        <p>
          Because of the final Normalize step, every vector has length{' '}
          <Nw>
            1 <Cite kind="spec" ref="§3.6" />.
          </Nw>{' '}
          All the information is in the direction, and the cosine similarity of two strings is simply their dot product. That is the measure
          Step 3 uses.
        </p>
        <ScaleNote paper="d_text = 384, vectors stored as bf16" ours="d_text = 384, vectors stored as float32">
          The encoder and its 384-dimensional output are identical at both scales. Only the stored precision differs: the paper’s code writes
          the table in{' '}
          <Nw>
            bf16 <Cite kind="code" ref="rt/embed.py:71" />,
          </Nw>{' '}
          which rounds every component, so its stored vectors have length 1 only approximately. This site keeps float32, where the length is 1
          up to float rounding.
        </ScaleNote>
      </Section>

      {/* ───────────────────────────────────────────────────────────── 3 */}
      <Section id="similarity" title="Similarity across databases" kicker="Step 3">
        <p>
          One way names could help a pretrained model on a new database (our hypothesis; the paper does not test it): phrases of the new
          database would land near phrases with related meaning in the pretraining databases. Here every {DB_SHORT[live.dbName]} phrase that can
          reach a token (rows) is compared with those of the other three databases (columns). Each cell is a cosine computed in your browser
          from the stored vectors.
        </p>
        <Figure
          wide
          title={`Cosine similarity: ${DB_SHORT[live.dbName]} phrases vs the ${others} databases`}
          caption="Color runs from 0 to the legend’s maximum: the largest cosine shown, rounded up to 0.1. The ring marks each row’s best match. Rows are grouped by table, columns by database and then by table (named above each group, with the table’s color). On a narrow screen the cells scroll sideways under fixed row labels."
        >
          {store && catalog ? (
            <SimilarityMatrix store={store} catalog={catalog} home={live.dbName} focus={focus} onFocus={setFocus} />
          ) : storeState.status === 'error' ? (
            <LoadError error={storeState.error} />
          ) : (
            <Loading height={520} />
          )}
        </Figure>
        {cmp && (
          <>
            <p>
              Take the label of the live task, <Phrase text={cmp.target} />. One might guess that its closest relatives are the labels of the{' '}
              {others} databases’ tasks.{' '}
              {cmp.nearest[0].phrase.role === 'label' ? (
                <>
                  Here the guess holds: the nearest {others} phrase is the task label <Phrase text={cmp.nearest[0].phrase.text} /> (cosine{' '}
                  {f3(cmp.nearest[0].cos)}).
                </>
              ) : (
                <>
                  The vectors say otherwise: the nearest {others} phrase is <Phrase text={cmp.nearest[0].phrase.text} /> (cosine{' '}
                  {f3(cmp.nearest[0].cos)}), a {ROLE_LABEL[cmp.nearest[0].phrase.role]} rather than a label
                  {cmp.nearest[1] && (
                    <>
                      ; next comes <Phrase text={cmp.nearest[1].phrase.text} /> ({f3(cmp.nearest[1].cos)})
                    </>
                  )}
                  .
                </>
              )}{' '}
              The average over all {cmp.pool} {others} phrases is {f3(cmp.meanAll)}. Here is where each of their task labels ranks:
            </p>
            <LabelRanks cmp={cmp} narrow={narrow} />
          </>
        )}
        <p>
          The encoder knows language, not databases. It places a phrase by its words and their everyday meaning, and the table name inside every
          phrase weighs on it as well.
          {gm && (
            <>
              {' '}
              Over all four databases, two phrases of the same table average a cosine of {f3(gm.sameTable)}, two phrases from different tables of
              one database {f3(gm.sameDb)}, and two phrases from different databases {f3(gm.crossDb)}.
            </>
          )}{' '}
          Whether a closeness like this is useful is not decided here: RT only receives these vectors as input, and pretraining decides what to
          make of them.
        </p>
        {keyPair && tokenPair && keyPair.cos > tokenPair.cos && (
          <Callout variant="note" title="The closest cross-database pair uses phrases that never reach a token">
            <p>
              Switch on “Include key columns”. The most similar cross-database pair is then <Phrase text={keyPair.a} /> and{' '}
              <Phrase text={keyPair.b} /> (cosine {f3(keyPair.cos)}), higher than the best pair of phrases that reach a token,{' '}
              <Phrase text={tokenPair.a} /> and <Phrase text={tokenPair.b} /> ({f3(tokenPair.cos)}). Key columns never produce a cell, so this
              resemblance is never used.
            </p>
            <p>
              RT never compares phrases of two databases directly: every context comes from one database. Any benefit from similar vectors comes
              from applying the same learned projection to both.
            </p>
          </Callout>
        )}
      </Section>

      {/* ───────────────────────────────────────────────────────────── 4 */}
      <Section id="map" title="A map of every column phrase" kicker="Step 4">
        <p>
          To see all {pcaPoints ? pcaPoints.length : '…'} column phrases of the four databases at once, we project their vectors onto the two
          directions along which they vary most (a principal component analysis, computed offline from the same stored vectors). Two axes cannot
          hold 384 dimensions:{' '}
          {varKept !== null ? (
            <>
              they keep <strong>{(100 * varKept).toFixed(0)}%</strong> of the variance
            </>
          ) : (
            'they keep only part of the variance'
          )}
          , so nearby points are not always near neighbors. Pin a phrase to draw its true nearest neighbors, measured in all 384 dimensions.
        </p>
        <Figure
          wide
          title="Column phrases of the four databases, 2-D PCA"
          caption="Color and shape mark the database. Hollow markers are key columns (interned, never on a token). Dashed lines join the pinned phrase to its five nearest phrases in 384-d. Click a database chip to hide it."
        >
          {pcaPoints && store && catalog ? (
            <PhraseMap points={pcaPoints} store={store} catalog={catalog} focus={focus} onFocus={setFocus} />
          ) : pca.status === 'error' ? (
            <LoadError error={pca.error} />
          ) : storeState.status === 'error' ? (
            <LoadError error={storeState.error} />
          ) : (
            <Loading what="the PCA map" height={480} />
          )}
        </Figure>
        {mstats && (
          <p>
            In this projection, the key-column phrases sit at an average PC 1 of {f2s(mstats.keyPc1)}, the other phrases at{' '}
            {f2s(mstats.otherPc1)}.
            {km && (
              <>
                {' '}
                The full 384-d vectors {km.keyKey > km.keyOther && km.keyKey > km.otherOther ? 'agree that the “…_id of …” phrases form a group' : 'do not confirm a separate group'}
                : two key-column phrases average a cosine of {f3(km.keyKey)}, a key-column phrase and another phrase {f3(km.keyOther)}, two
                other phrases {f3(km.otherOther)}.
              </>
            )}{' '}
            Along PC 2 the databases spread out, from {DB_SHORT[mstats.pc2ByDb[0].db]} (average {f2s(mstats.pc2ByDb[0].mean)}) to{' '}
            {DB_SHORT[mstats.pc2ByDb[mstats.pc2ByDb.length - 1].db]} ({f2s(mstats.pc2ByDb[mstats.pc2ByDb.length - 1].mean)}).
            {gm && (
              <>
                {' '}
                In 384-d, two phrases from different tables of one database average {f3(gm.sameDb)}, two phrases from different databases{' '}
                {f3(gm.crossDb)}{gm.sameDb > gm.crossDb ? '. Our reading: the phrases of a database share its table names and domain words' : ''}.
              </>
            )}{' '}
            RT never computes this map; it is only a window onto the vectors it receives.
          </p>
        )}
      </Section>

      {/* ───────────────────────────────────────────────────────────── 5 */}
      <Section id="text-cells" title="Text cells: the raw value, nothing else" kicker="Step 5">
        <p>
          String values go through the same encoder, applied to the raw string with nothing added:{' '}
          <Nw>
            <TeX tex="\mathbf{r} = \mathcal{E}^{\text{text}}(v)" /> <Cite kind="paper" ref="p.5 §3.1" />.
          </Nw>{' '}
          Preprocessing interns the value string{' '}
          <Nw>
            itself <Cite kind="code" ref="rustler/src/pre.rs:596-597" />,
          </Nw>{' '}
          the encoder embeds that{' '}
          <Nw>
            list <Cite kind="code" ref="rt/embed.py:65-68" />,
          </Nw>{' '}
          and the sampler copies the row into the token’s{' '}
          <Nw>
            <code>text_values</code> <Cite kind="code" ref="rustler/src/fly.rs:484-486" />.
          </Nw>{' '}
          The column never becomes part of the{' '}
          <Nw>
            string <Cite kind="spec" ref="§3.6" />.
          </Nw>
          {typeof sampleValue === 'string' && live.entityNode && (
            <>
              {' '}
              In the live context, the {live.entityNode.tableName.replace(/s$/, '')}’s value <Quoted text={sampleValue} /> is embedded as exactly
              that string.
            </>
          )}
        </p>
        <Figure
          wide
          title="Two vectors per text token"
          caption="Pick a text cell of the live context. The token holds two separate rows of the string table: its column phrase and its raw value."
        >
          {store ? <TextCells live={live} store={store} /> : storeState.status === 'error' ? <LoadError error={storeState.error} /> : <Loading height={360} />}
        </Figure>
        <p>
          The column context of a text cell therefore comes only from the separate column{' '}
          <Nw>
            term <Cite kind="spec" ref="§3.6" />.
          </Nw>{' '}
          Inside the model, each vector passes through its own linear layer and RMSNorm, and the two results are{' '}
          <Nw>
            added <Cite kind="code" ref="rt/model.py:209-220" />;
          </Nw>{' '}
          chapter 8 follows that sum.
        </p>
        {store && catalog && <SharedValues catalog={catalog} store={store} />}
        <p className="mt-6">
          What about cells that are not text? Preprocessing stores text index 0 for{' '}
          <Nw>
            them <Cite kind="code" ref="rustler/src/pre.rs:559" />,
          </Nw>{' '}
          so the sampler copies the vector of <code>text[0]</code> into their <code>text_values</code> as well. <code>text[0]</code> is always a
          table name, the first one pre.rs interns: in our preprocessing of {DB_SHORT[live.dbName]} it is <Quoted text={text0} />, while pre.rs’s
          hash order may put another table first. The model multiplies that term by zero, because the token’s <Term id="sem-type">sem type</Term>{' '}
          is not{' '}
          <Nw>
            text <Cite kind="code" ref="rt/model.py:216-220" /> <Cite kind="spec" ref="§3.6" />.
          </Nw>
        </p>
        <p>
          Long strings would be cut: the encoder truncates at {ENCODER.maxSeqLength}{' '}
          <Nw>
            wordpieces <Cite kind="spec" ref="§3.6" />.
          </Nw>{' '}
          The longest
          string in our four databases has {ENCODER.maxWordpieces} wordpieces, [CLS] and [SEP] included, so nothing is cut here.
        </p>
        <CodeTabs
          label="Official code for text cells"
          tabs={[
            {
              tab: 'pre.rs: a string cell',
              title: 'A string cell stores the index of its raw value',
              file: 'rustler/src/pre.rs',
              lines: '595-605',
              code: PRE_TEXT_595_605,
              highlightLines: [597, 600],
              note: 'val is the cell’s string as it is: no column name, no table name. col_name_idx (line 603) is stored separately.',
            },
            {
              tab: 'fly.rs: the two copies',
              title: 'The sampler copies the phrase vector and the value vector into the batch',
              file: 'rustler/src/fly.rs',
              lines: '469-486',
              code: FLY_COPY_469_486,
              highlightLines: [472, 473, 474, 475, 476, 477, 484, 485, 486],
              note: 'Lines 484-486 run for every cell; a non-text cell copies row 0.',
            },
          ]}
        />
      </Section>

      {/* ───────────────────────────────────────────────────────────── 6 */}
      <Section id="why" title="Why names matter" kicker="Step 6">
        <p>
          The paper tests the value of names directly. It shuffles the column and table names in the context and measures performance with the
          same pretrained{' '}
          <Nw>
            checkpoints <Cite kind="paper" ref="p.9 §5.1" />.
          </Nw>{' '}
          “For <Term id="zero-shot">zero-shot</Term> evaluation, the ablations are applied directly to the sampled context used as input. For
          fine-tuning, models are trained to convergence with the same modified{' '}
          <Nw>
            contexts” <Cite kind="paper" ref="p.22 App. G" />.
          </Nw>{' '}
          The paper does not describe the shuffling procedure in more detail. In the same table it removes two kinds of labels from the
          context, which gives a yardstick for the names.
        </p>
        <Figure wide title="Mean over the RelBench tasks for each context ablation (paper Table 4, p.9)">
          <Table4Grid narrow={narrow} />
        </Figure>
        <p>
          In the paper’s words, “the model indeed leverages table/column names for zero-shot transfer, as wrong names lead to consistent
          degradation. After full fine-tuning, context ablations have little effect” (the exception it names is self labels on regression){' '}
          <Cite kind="paper" ref="p.9 §5.1" />. Appendix G adds that shuffling names “also harms transfer, highlighting the importance of semantic
          signals from schema{' '}
          <Nw>
            metadata” <Cite kind="paper" ref="p.22 App. G" />.
          </Nw>{' '}
          On average, zero-shot AUROC falls by {f1(TABLE4.none.zsClf - TABLE4.colNames.zsClf)} points and R² by{' '}
          {f1(TABLE4.none.zsReg - TABLE4.colNames.zsReg)} points. Removing the target entity’s past labels costs{' '}
          {f1(TABLE4.none.zsClf - TABLE4.selfLabels.zsClf)} and {f1(TABLE4.none.zsReg - TABLE4.selfLabels.zsReg)}: in the paper’s words,
          “zero-shot transfer is driven primarily by the model’s ability to leverage past labels of the target{' '}
          <Nw>
            entity” <Cite kind="paper" ref="p.9 §5.1" />.
          </Nw>{' '}
          Names are a consistent but smaller signal. One way to read the zero-shot versus fine-tuned difference (ours, not the paper’s): a
          fine-tuned model has seen many labeled examples of the task and can learn what each column means from its values, while a zero-shot
          model must rely more on what the names say.
        </p>
        <Figure wide>
          <AblationDumbbells />
        </Figure>
        <p>
          The mean hides spread. In zero-shot, shuffling lowers the score on {clfHurt} of the 10 binary tasks and {regHurt} of the 8 regression
          tasks. The largest drops are {clfDrops[0].name} ({signed1(clfDrops[0].d)} AUROC) and {regDrops[0].name} ({signed1(regDrops[0].d)} R²),
          while {clfDrops[clfDrops.length - 1].name}{' '}
          <Nw>
            gains {signed1(clfDrops[clfDrops.length - 1].d)} <Cite kind="paper" ref="p.22 Table 11" />.
          </Nw>
        </p>
        <p>
          The paper’s conclusion ranks what drives zero-shot{' '}
          <Nw>
            transfer <Cite kind="paper" ref="p.10 §7" /> <Cite kind="spec" ref="§11.7" />.
          </Nw>{' '}
          Names come fourth of five, after the target entity’s past labels, which matches the gap between their two rows in Table 4:
        </p>
        <ol>
          <li>forecasting from the target entity’s past task rows;</li>
          <li>column attention;</li>
          <li>feature attention, in the same and in parent tables;</li>
          <li>
            <strong>schema semantics from table and column names</strong>;
          </li>
          <li>in-context learning from other entities’ task rows.</li>
        </ol>
        <Callout variant="note" title="Where the ablation numbers come from">
          <p>
            Tables 4 and 11 come from an earlier run than the headline results (arXiv v1, before later sampler fixes and before the released
            attention order), so their “none” column (70.1 AUROC) differs slightly from Table 1{' '}
            <Nw>
              (69.7) <Cite kind="spec" ref="§10 #25" /> <Cite kind="spec" ref="§13 #1" />.
            </Nw>
          </p>
          <p>
            The paper names this ablation in two ways: Table 11’s caption says “we remove column{' '}
            <Nw>
              names” <Cite kind="paper" ref="p.22 Table 11" />,
            </Nw>{' '}
            while the text describes “shuffling column and table{' '}
            <Nw>
              names” <Cite kind="paper" ref="p.9 §5.1" /> <Cite kind="paper" ref="p.22 App. G" />.
            </Nw>{' '}
            This chapter follows the text.
          </p>
        </Callout>

        <h3>Not lexical copying</h3>
        <p>
          Could the model simply match identical column names across databases? The paper checks with the{' '}
          Jaccard overlap of the column names of two databases: the number of shared names divided by the number of
          distinct names in the two databases together, <TeX tex="|A \cap B| \,/\, |A \cup B|" />. Between RelBench databases it is tiny, at
          most 0.043 (rel-amazon vs rel-hm), and fuzzy string matching gives only 0.47 to{' '}
          <Nw>
            0.59 <Cite kind="paper" ref="p.17-18 App. C.2" />.
          </Nw>{' '}
          Its conclusion: “RT’s zero-shot generalization cannot be attributed to lexical similarity. Instead, the model must learn to recognize
          and exploit structural relational{' '}
          <Nw>
            patterns” <Cite kind="paper" ref="p.18 App. C.2" />.
          </Nw>
        </p>
        <p>
          The paper does not say how names help, then. Two readings fit its results (ours, not the paper’s). Names may carry meaning across
          databases through the encoder, as Step 3 measures. Or they may mainly tell columns apart within one context, the role App. E
          describes. The shuffle ablation removes both, so Table 4 cannot separate them.
        </p>
        <Figure
          wide
          title="Column-name overlap (Jaccard) between databases"
          caption={
            <>
              Jaccard overlap = shared column names / distinct column names of the two databases. Left: the paper’s numbers. Right: the same
              statistic on this site’s four toy databases, computed live. The paper does not say which columns it counted; we count every column
              of the database tables (keys included, task tables excluded). Hover a cell for the counts and the shared names.
            </>
          }
        >
          {catalog ? <OverlapTables catalog={catalog} /> : <Loading what="the four schemas" height={240} />}
        </Figure>
      </Section>

      {/* ───────────────────────────────────────────────────────────── 7 */}
      <Section id="frozen" title="The encoder is frozen" kicker="Step 7">
        <p>
          RT never runs its text encoder while it trains or predicts. <code>embed.py</code> embeds the whole string table once and writes the
          vectors to a{' '}
          <Nw>
            file <Cite kind="code" ref="rt/embed.py:65-72" />.
          </Nw>{' '}
          The sampler memory-maps that file and copies rows by{' '}
          <Nw>
            index <Cite kind="code" ref="rustler/src/fly.rs:263-265" /> <Cite kind="code" ref="rustler/src/fly.rs:528-532" />.
          </Nw>{' '}
          So no gradient can reach the encoder. The first trainable parameters on the text path are two linear maps from 384 to d,{' '}
          <code>enc_dict["col_name"]</code> and <code>enc_dict["text"]</code>, each followed by its own <Term id="rmsnorm">RMSNorm</Term>{' '}
          <Nw>
            <Cite kind="code" ref="rt/model.py:115-140" />.
          </Nw>
        </p>
        <Figure wide title="Where the text vectors come from, and where learning starts">
          <FrozenPipeline nStrings={live.pdb.text.length} dbLabel={DB_SHORT[live.dbName]} d={d} />
        </Figure>
        <p>
          The paper describes the encoder as frozen without discussing the choice: “a frozen text{' '}
          <Nw>
            encoder” <Cite kind="paper" ref="p.5 §3.1" />,
          </Nw>{' '}
          “frozen language model (LM) embeddings of table/column{' '}
          <Nw>
            names” <Cite kind="paper" ref="p.2 Fig. 1" />.
          </Nw>{' '}
          Three consequences follow from
          the design. Training never pays for the encoder: each step reads precomputed rows. A new database
          needs one pass of the encoder over its strings, and no change to RT. And a string has the same vector in every database, so the
          similarities of Step 3 are fixed facts that pretraining can learn to use.
        </p>
        <Figure wide title="The learned part, live">
          {store ? (
            <LearnedProjection phrase={store.has(focus) && catalog?.byText.has(focus) ? focus : live.targetPhrase} store={store} assets={assets} catalog={catalog} />
          ) : (
            <Loading height={200} />
          )}
        </Figure>
        <Callout
          variant="paper-vs-code"
          title="The column term in the paper and in the code"
          paper={
            <>
              <TeX tex="\mathbf{x} = \mathbf{W}_d\,\mathbf{r} + \mathbf{W}\,\mathcal{E}^{\text{schema}}(c, t)" />, with W shared{' '}
              <Cite kind="paper" ref="p.5 §3.1" />
            </>
          }
          code={
            <>
              <TeX tex="\operatorname{RMSNorm}_{\text{col}}(\mathbf{W}_{\text{col}}\,\mathbf{e} + \mathbf{b}_{\text{col}})" />: a bias and an RMSNorm
              after the projection <Cite kind="code" ref="rt/model.py:209-213" />
            </>
          }
          verdict="This site computes what the code computes (spec §10 #2). Chapter 8 takes the whole input embedding apart."
        />
        <ScaleNote paper={`W_col of shape [256, 384]`} ours={`W_col of shape [${d}, 384]`}>
          The frozen 384-dimensional vectors are the same at both scales; only the width d that they are projected to differs.
        </ScaleNote>
        <CodeTabs
          label="Official code for the frozen encoder and the learned projections"
          tabs={[
            {
              tab: 'embed.py: TextEmbedder',
              title: 'TextEmbedder: a SentenceTransformer and its encode()',
              file: 'rt/embed.py',
              lines: '12-31',
              code: EMBED_TEXT_EMBEDDER_12_31,
              highlightLines: [14, 15, 25],
              note: 'Snapshot 2a899c8. The dtype line, the device argument of encode() and the multi-GPU handling were added after the paper’s runs, in 7d564f7; at paper time the encoder ran in fp32 on cuda:0 (spec §3.6).',
            },
            {
              tab: 'embed.py: main',
              title: 'Every string embedded once and stored as a raw bf16 array',
              file: 'rt/embed.py',
              lines: '65-72',
              code: EMBED_MAIN_65_72,
              highlightLines: [70, 71, 72],
            },
            {
              tab: 'fly.rs: row lookup',
              title: 'get_text_emb: a row of the memory-mapped table',
              file: 'rustler/src/fly.rs',
              lines: '528-532',
              code: FLY_GET_TEXT_EMB_528_532,
              highlightLines: [531],
            },
            {
              tab: 'model.py: projections',
              title: 'The learned projections from 384 to d_model',
              file: 'rt/model.py',
              lines: '115-123',
              code: MODEL_ENC_DICT_115_123,
              highlightLines: [118, 120],
            },
            {
              tab: 'model.py: input embedding',
              title: 'The column term and the gated value terms of the input embedding',
              file: 'rt/model.py',
              lines: '208-224',
              code: MODEL_EMBED_208_224,
              highlightLines: [209, 210, 211, 212, 213],
              note: "text_values enters through enc_dict['text'] at line 218, gated by sem_types == 1 (text).",
            },
          ]}
        />
      </Section>

      <div className="page-grid">
        <Quiz
          title="Check your understanding"
          questions={[
            {
              prompt: 'Does RT see the name of a foreign-key column, such as customer_id of orders?',
              choices: [
                {
                  text: 'Yes: it is a cell of the order row like any other.',
                  why: 'A foreign-key column produces no cell. Preprocessing turns it into a link to the parent row.',
                },
                {
                  text: 'No: its phrase is interned and embedded, but no token ever carries it.',
                  correct: true,
                  why: 'pre.rs interns the phrase of every column before checking its role, then skips key columns. The paper lists this as a limitation (p.10 §7).',
                },
                {
                  text: 'Only through the parent table’s name.',
                  why: 'Table names reach a token only inside the phrase of a column that produces a cell.',
                },
              ],
              explanation: 'Key columns shape the context (which rows are linked) and the masks, but their names never enter the model.',
            },
            {
              prompt: (
                <>
                  A customer’s country is the string <span>"DE"</span>. Which strings are embedded for this cell’s token?
                </>
              ),
              choices: [
                { text: '“country of customers: DE”, one combined string', why: 'No string ever joins a value to its column.' },
                { text: 'Only “DE”: text cells have no column term', why: 'Every token carries its column phrase, whatever its type.' },
                {
                  text: '“country of customers” (column term) and “DE” (value term), separately',
                  correct: true,
                  why: 'The value is embedded raw; the column context comes only from the column phrase (spec §3.6).',
                },
              ],
              explanation: 'Two rows of the same string table, projected by two different learned layers and added inside the model.',
            },
            {
              prompt: 'During pretraining, which parameters on the path from a column phrase to the token’s input are updated?',
              choices: [
                { text: 'W_col, its bias and the RMSNorm gain', correct: true, why: 'enc_dict["col_name"] and norm_dict["col_name"] are the first trainable modules (model.py:115-140).' },
                { text: 'All of MiniLM’s layers, plus W_col', why: 'MiniLM never runs during training; its vectors are read from a file.' },
                { text: 'None: the phrase vector is added to the token unchanged', why: 'The 384-d vector is first projected to d and normalized by learned layers.' },
              ],
              explanation: 'The encoder is frozen by construction: its output is plain data stored before training starts.',
            },
          ]}
        />
      </div>

      <ChapterNav slug="semantics" />
    </ChapterPage>
  );
}

// ─────────────────────────────────────────────────────────────────────────────

function cosineOf(store: { get(t: string): Float32Array }, a: string, b: string): number {
  const x = store.get(a);
  const y = store.get(b);
  let s = 0;
  let nx = 0;
  let ny = 0;
  for (let i = 0; i < x.length; i++) {
    s += x[i] * y[i];
    nx += x[i] * x[i];
    ny += y[i] * y[i];
  }
  return s / Math.sqrt(nx * ny);
}

const f2s = (v: number) => (v < 0 ? `−${Math.abs(v).toFixed(2)}` : `+${v.toFixed(2)}`);

/** Where each pretraining task label ranks among all pretraining phrases, for the live target phrase. */
function LabelRanks({ cmp, narrow }: { cmp: NonNullable<ReturnType<typeof compareTarget>>; narrow: boolean }) {
  return (
    <div className="my-5">
      <table className="w-full border-collapse text-[15.5px]">
        <caption className="sr-only">Cosine of the live target phrase with each task label of the other databases</caption>
        <thead>
          <tr className="border-b border-grid text-left text-ink-2">
            {!narrow && (
              <th scope="col" className="py-1.5 pr-3 font-semibold">
                Database
              </th>
            )}
            <th scope="col" className="py-1.5 pr-3 font-semibold">
              Task label phrase
            </th>
            <th scope="col" className="py-1.5 pr-3 text-right font-semibold">
              cosine
            </th>
            <th scope="col" className="py-1.5 text-right font-semibold">
              {narrow ? 'rank' : `rank of ${cmp.pool}`}
            </th>
          </tr>
        </thead>
        <tbody>
          {cmp.labels.map((l) => (
            <tr key={l.phrase.text} className="border-b border-grid/70">
              {!narrow && (
                <td className="py-1.5 pr-3 whitespace-nowrap">
                  <DbTag db={l.phrase.db} neutral />
                </td>
              )}
              <td className="py-1.5 pr-3">
                {narrow ? (
                  <span className="inline-flex items-baseline gap-1.5">
                    <DbTag db={l.phrase.db} showName={false} neutral className="translate-y-[1px]" />
                    <Phrase text={l.phrase.text} />
                  </span>
                ) : (
                  <Phrase text={l.phrase.text} />
                )}
              </td>
              <td className="num py-1.5 pr-3 text-right text-ink">{f3(l.cos)}</td>
              <td className="num py-1.5 text-right text-ink-2">{l.rank}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {narrow && <p className="mt-1.5 !mb-0 text-[14.5px] text-muted">Rank among all {cmp.pool} phrases of the other databases that reach a token.</p>}
    </div>
  );
}
