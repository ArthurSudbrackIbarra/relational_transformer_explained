/**
 * Q&A 1: where task tables come from, and why that is compatible with zero-shot prediction.
 * Checked against relational-transformer @2a899c8, RelBench v1.1.0 and the paper (arXiv v3).
 */
import { useState } from 'react';
import { Link } from 'react-router-dom';
import { ChurnBuilder } from '@/chapters/tasks/ChurnBuilder';
import { FLY_MASK, FLY_SEEDS, FLY_TASK_TABLE_SKIP, PRE_TABLE_TYPE } from '@/chapters/tasks/codeSnippets';
import { useChapterExample } from '@/chapters/tasks/useExample';
import { Callout, Cite, CodeRef, Figure, Section, Term } from '@/components';
import type { QAMeta } from '../types';

export const meta: QAMeta = {
  id: 'where-task-tables-come-from',
  title: 'Who builds the task table, and how can it hold labels in a zero-shot setting?',
  question:
    'When are the task tables created? If their cells are used in the inference step, these tables need to pre-exist to feed the transformer, right? How will they pre-exist and already have data in them if this is a new table that is being created now, and how does the algorithm know how to build these tables and populate them without knowing the context, being used in a completely zero-shot new dataset?',
  short:
    'RT never builds a task table. It is input data, prepared before RT runs by whoever poses the prediction task: in the paper, RelBench’s task code fills every row with a label computed from what already happened in the database. The one unknown label, the seed’s, is masked. “Zero-shot” means no weight updates on the new database; you still have to say what to predict, and that task’s history reaches the model through the context window.',
  asked: '2026-10-23',
  chapters: ['tasks', 'sampling', 'results'],
  checked: [
    'relational-transformer @2a899c8: scripts/download_relbench.py, rustler/src/pre.rs, rustler/src/fly.rs, rt/model.py',
    'RelBench v1.1.0: relbench/base/task_base.py, relbench/tasks/f1.py',
    'Paper (arXiv 2510.06377v3): p.2, p.4 §2.3 and §3.1, p.8 Table 1, p.9 Table 4, p.23 Table 12',
    'research/RT_TECHNICAL_SPEC.md §2, §3.1, §3.5, §3.6, §9.4, §11.2-11.3',
  ],
};

const RELBENCH = (file: string, a: number, b: number) => ({
  repo: 'snap-stanford/relbench',
  ref: 'v1.1.0',
  url: `https://github.com/snap-stanford/relbench/blob/v1.1.0/${file}#L${a}-L${b}`,
});

// GENERATED verbatim by scripts/snippet.mjs from relational-transformer @2a899c8 and relbench @v1.1.0. Do not edit.
const DOWNLOAD_RELBENCH = {
  file: "scripts/download_relbench.py",
  start: 1,
  end: 11,
  code: "from relbench.datasets import get_dataset, get_dataset_names\nfrom relbench.tasks import get_task, get_task_names\n\nif __name__ == \"__main__\":\n    for dataset_name in get_dataset_names():\n        print(f\"Downloading dataset: {dataset_name}\")\n        get_dataset(dataset_name, download=True)\n\n        for task_name in get_task_names(dataset_name):\n            print(f\"Downloading task: {task_name} from dataset: {dataset_name}\")\n            get_task(dataset_name, task_name, download=True)",
};
const PRE_READ_TABLES = {
  file: "rustler/src/pre.rs",
  start: 79,
  end: 92,
  code: "    for (is_db_table, pq_path) in itertools::chain(\n        iter::repeat(true).zip(\n            glob(&format!(\"{}/db/*.parquet\", dataset_path))\n                .unwrap()\n                .map(|p| p.unwrap())\n                .sorted(),\n        ),\n        iter::repeat(false).zip(\n            glob(&format!(\"{}/tasks/*/*.parquet\", dataset_path))\n                .unwrap()\n                .map(|p| p.unwrap())\n                .sorted(),\n        ),\n    ) {",
};
const PRE_TRAIN_STATS = {
  file: "rustler/src/pre.rs",
  start: 383,
  end: 396,
  code: "    let mut col_stats_map = HashMap::new();\n    for ((table_name, table_type), table) in &table_map {\n        if table_type == &TableType::Train {\n            col_stats_map.insert(table_name.clone(), table.col_stats.clone());\n        }\n    }\n    for ((table_name, table_type), table) in &mut table_map {\n        match table_type {\n            TableType::Val | TableType::Test => {\n                table.col_stats = col_stats_map.get(table_name).unwrap().clone();\n            }\n            _ => {}\n        }\n    }",
};
const MODEL_MASK_EMB = {
  file: "rt/model.py",
  start: 216,
  end: 224,
  code: "        for i, t in enumerate([\"number\", \"text\", \"datetime\", \"boolean\"]):\n            x = x + (\n                self.norm_dict[t](self.enc_dict[t](batch[t + \"_values\"]))\n                * ((batch[\"sem_types\"] == i) & ~batch[\"masks\"] & ~is_padding)[..., None]\n            )\n            x = x + (\n                self.mask_embs[t]\n                * ((batch[\"sem_types\"] == i) & batch[\"masks\"] & ~is_padding)[..., None]\n            )",
};
const RELBENCH_DNF = {
  file: "relbench/tasks/f1.py",
  start: 65,
  end: 118,
  code: "class DriverDNFTask(EntityTask):\n    r\"\"\"Predict the if each driver will DNF (not finish) a race in the next 1 month.\"\"\"\n\n    task_type = TaskType.BINARY_CLASSIFICATION\n    entity_col = \"driverId\"\n    entity_table = \"drivers\"\n    time_col = \"date\"\n    target_col = \"did_not_finish\"\n    timedelta = pd.Timedelta(days=30)\n    metrics = [average_precision, accuracy, f1, roc_auc]\n    num_eval_timestamps = 40\n\n    def make_table(self, db: Database, timestamps: \"pd.Series[pd.Timestamp]\") -> Table:\n        timestamp_df = pd.DataFrame({\"timestamp\": timestamps})\n\n        results = db.table_dict[\"results\"].df\n        drivers = db.table_dict[\"drivers\"].df\n        races = db.table_dict[\"races\"].df\n\n        df = duckdb.sql(\n            f\"\"\"\n                SELECT\n                    t.timestamp as date,\n                    dri.driverId as driverId,\n                    MAX(CASE WHEN re.statusId != 1 THEN 1 ELSE 0 END) AS did_not_finish\n                FROM\n                    timestamp_df t\n                LEFT JOIN\n                    results re\n                ON\n                    re.date <= t.timestamp + INTERVAL '{self.timedelta}'\n                    and re.date  > t.timestamp\n                LEFT JOIN\n                    drivers dri\n                ON\n                    re.driverId = dri.driverId\n                WHERE\n                    dri.driverId IN (\n                        SELECT DISTINCT driverId\n                        FROM results\n                        WHERE date > t.timestamp - INTERVAL '1 year'\n                    )\n                GROUP BY t.timestamp, dri.driverId\n\n            ;\n            \"\"\"\n        ).df()\n\n        return Table(\n            df=df,\n            fkey_col_to_pkey_table={self.entity_col: self.entity_table},\n            pkey_col=None,\n            time_col=self.time_col,\n        )",
};
const RELBENCH_GET_TABLE = {
  file: "relbench/base/task_base.py",
  start: 102,
  end: 153,
  code: "    def _get_table(self, split: str) -> Table:\n        r\"\"\"Helper function to get a table for a split.\"\"\"\n\n        db = self.dataset.get_db(upto_test_timestamp=split != \"test\")\n\n        if split == \"train\":\n            start = self.dataset.val_timestamp - self.timedelta\n            end = db.min_timestamp\n            freq = -self.timedelta\n\n        elif split == \"val\":\n            if self.dataset.val_timestamp + self.timedelta > db.max_timestamp:\n                raise RuntimeError(\n                    \"val timestamp + timedelta is larger than max timestamp! \"\n                    \"This would cause val labels to be generated with \"\n                    \"insufficient aggregation time.\"\n                )\n\n            start = self.dataset.val_timestamp\n            end = min(\n                self.dataset.val_timestamp\n                + self.timedelta * (self.num_eval_timestamps - 1),\n                self.dataset.test_timestamp - self.timedelta,\n            )\n            freq = self.timedelta\n\n        elif split == \"test\":\n            if self.dataset.test_timestamp + self.timedelta > db.max_timestamp:\n                raise RuntimeError(\n                    \"test timestamp + timedelta is larger than max timestamp! \"\n                    \"This would cause test labels to be generated with \"\n                    \"insufficient aggregation time.\"\n                )\n\n            start = self.dataset.test_timestamp\n            end = min(\n                self.dataset.test_timestamp\n                + self.timedelta * (self.num_eval_timestamps - 1),\n                db.max_timestamp - self.timedelta,\n            )\n            freq = self.timedelta\n\n        timestamps = pd.date_range(start=start, end=end, freq=freq)\n\n        if split == \"train\" and len(timestamps) < 3:\n            raise RuntimeError(\n                f\"The number of training time frames is too few. \"\n                f\"({len(timestamps)} given)\"\n            )\n\n        table = self.make_table(db, timestamps)\n        table = self.filter_dangling_entities(table)",
};

function Code({ s, title, note, highlight }: { s: { file: string; start: number; end: number; code: string }; title: string; note?: React.ReactNode; highlight?: number[] }) {
  return <CodeRef title={title} file={s.file} lines={[s.start, s.end]} code={s.code} note={note} highlightLines={highlight} />;
}

function RelbenchCode({ s, title, note, highlight }: { s: { file: string; start: number; end: number; code: string }; title: string; note?: React.ReactNode; highlight?: number[] }) {
  return (
    <CodeRef
      title={title}
      file={s.file}
      lines={[s.start, s.end]}
      code={s.code}
      note={note}
      highlightLines={highlight}
      source={RELBENCH(s.file, s.start, s.end)}
    />
  );
}

function LiveChurnTable() {
  const ex = useChapterExample();
  const [customer, setCustomer] = useState(ex.customer);
  return (
    <Figure
      wide
      title="ShopDB’s customer-churn task table, filled from the orders that already happened"
      caption={
        <>
          The same figure as in <Link to="/tasks?section=build-churn">chapter 3</Link>. Every label is a count over the orders table in
          the window after the row’s timestamp: nothing is predicted while the table is built.
        </>
      }
    >
      <ChurnBuilder db={ex.shop} task={ex.churn} customer={customer} onCustomerChange={setCustomer} />
    </Figure>
  );
}

export default function Answer() {
  return (
    <>
      <Section id="input-not-output" title="RT reads task tables; it never makes them">
        <p>
          A <Term id="task-table">task table</Term> is part of the input, like the database’s own tables. It exists before
          preprocessing starts. The paper describes it as a table that “stores the forecasting labels, together with foreign keys
          linking to the relevant entities (e.g., user IDs or item IDs) and a timestamp specifying the prediction horizon”{' '}
          <Cite kind="paper" ref="p.4 §2.2" />, and RT attaches it to the database: “For each task, we attach a dedicated task table
          to the database. Only one task table is active at a time … from the model’s perspective task tables are treated as ordinary
          relational tables” <Cite kind="paper" ref="p.4 §3.1" />.
        </p>
        <p>In the official code the order of events is:</p>
        <ol>
          <li>
            <strong>Download.</strong> <code>scripts/download_relbench.py</code> asks RelBench for every dataset and every task. A
            task arrives as three ready-made files, <code>tasks/&lt;task&gt;/train.parquet</code>, <code>val.parquet</code> and{' '}
            <code>test.parquet</code>, labels included <Cite kind="code" ref="scripts/download_relbench.py:1-11" />{' '}
            <Cite kind="spec" ref="§3.1" />.
          </li>
          <li>
            <strong>Preprocess.</strong> <code>pre.rs</code> reads <code>db/*.parquet</code> and <code>tasks/*/*.parquet</code> in one
            loop and turns both into rows and cells the same way. A task file becomes a table named after the task folder, of type
            Train, Val or Test <Cite kind="code" ref="rustler/src/pre.rs:79-92" /> <Cite kind="code" ref="rustler/src/pre.rs:152-174" />.
          </li>
          <li>
            <strong>Sample and predict.</strong> The sampler takes seed rows from that table and builds a context around each one{' '}
            <Cite kind="spec" ref="§2.3" />.
          </li>
        </ol>
        <p>
          No step computes a label. The repository reads the labels from these files; its only hard-coded edits to task tables cast
          some label columns to booleans (nonzero becomes true) and drop a stray index column{' '}
          <Cite kind="code" ref="rustler/src/pre.rs:208-289" /> <Cite kind="spec" ref="§3.1" />.
        </p>
        <Code s={DOWNLOAD_RELBENCH} title="The task tables are downloaded, not computed" />
        <Code s={PRE_READ_TABLES} title="Preprocessing reads database tables and task tables in the same loop" />
        <Code s={PRE_TABLE_TYPE} title="A task file’s folder names the table; its file name gives the split" />
      </Section>

      <Section id="who-fills-the-rows" title="Who fills the rows: the task definition, run over the database’s past">
        <p>
          RelBench does. A RelBench task is a small piece of code: an entity table, a time column, a target, a window length Δ, and a{' '}
          <code>make_table(db, timestamps)</code> method. For each timestamp t, it lists the active entities and computes each one’s
          label from what happened in the window (t, t + Δ] <Cite kind="spec" ref="§2.2" />. For <code>rel-f1/driver-dnf</code> it is a
          SQL query: a driver gets <code>did_not_finish = 1</code> when one of their race results in the next 30 days is not a finish.
        </p>
        <RelbenchCode s={RELBENCH_DNF} title="RelBench’s driver-dnf task: the labels are a SQL query over the results table" highlight={[89, 95, 96]} />
        <p>
          The timestamps come from two cutoffs fixed by the dataset. Training rows step back from the validation cutoff in steps of
          Δ until the start of the data; validation and test rows sit at their cutoffs <Cite kind="spec" ref="§2.2" />. RelBench
          refuses to build a split whose window would run past the end of the recorded data (“insufficient aggregation time”).
        </p>
        <RelbenchCode
          s={RELBENCH_GET_TABLE}
          title="Where the timestamps come from, and the call that fills the table"
          highlight={[107, 108, 109, 110, 144, 152]}
        />
        <p>
          So a task table is a summary of history. Each label describes something that has already happened and is recorded in the
          database. That is why a table “created now” can be full of labels: it is created from the past. Even the test labels are on
          disk, and RT reads them, because the evaluation compares predictions with them <Cite kind="spec" ref="§2.2" />.
        </p>
        <LiveChurnTable />
      </Section>

      <Section id="the-unknown-label" title="The one label that is unknown is masked">
        <p>
          The label the model must predict is the <Term id="seed-row">seed row</Term>’s own. The sampler marks exactly that cell as the
          target and the mask <Cite kind="code" ref="rustler/src/fly.rs:492-496" />. The model then replaces its value with a learned{' '}
          <Term id="mask-vector">mask vector</Term>: the value encoder is multiplied by <code>~masks</code>, the mask embedding by{' '}
          <code>masks</code> <Cite kind="code" ref="rt/model.py:216-224" />. The true value never enters the forward pass; only the loss
          and the metrics read it.
        </p>
        <Code s={FLY_MASK} title="Only the seed’s target cell is masked" />
        <Code s={MODEL_MASK_EMB} title="A masked cell gets the mask embedding instead of its value" highlight={[219, 222, 223]} />
        <Callout variant="warning" title="The released code is built for evaluation">
          <p>
            A row can be a seed only if it has a cell in the target column <Cite kind="code" ref="rustler/src/fly.rs:287-296" />, and a
            missing value produces no cell at all <Cite kind="spec" ref="§3.5" />. So every seed has a known label, which the evaluation
            needs. The repository has no path for a row whose outcome is still in the future. Doing that with this code would mean
            adding the new row with a placeholder label; because that cell is masked, the model would not see the placeholder. This is
            what the code implies, not a workflow the authors document.
          </p>
        </Callout>
        <Code s={FLY_SEEDS} title="Seeds are rows that have a target cell" />
      </Section>

      <Section id="zero-shot" title="Then what is “zero-shot” about it?">
        <p>
          The paper defines zero-shot relational learning as “predicting new targets on a new RDB with a new schema, without weight
          updates. Information about the new RDB is available only at inference time, solely through the context window”{' '}
          <Cite kind="paper" ref="p.4 §2.3" />. The definition is about the <em>weights</em>: none of them were trained on the new
          database or, in the paper’s experiments, on any table of it. It does not say the task needs no definition. Someone still
          decides what to predict (which entity, which outcome, which horizon), and the task table is how that decision is written
          down <Cite kind="spec" ref="§9.4" />.
        </p>
        <p>The model is never told what the task means. Everything it gets is cells in the context:</p>
        <ul>
          <li>
            the target cell carries its column phrase, such as <code>“did_not_finish of driver-dnf”</code>, embedded as text like every
            other column name <Cite kind="spec" ref="§3.6" />;
          </li>
          <li>
            its type picks the output: a boolean target is read as a logit, a number as a z-score <Cite kind="spec" ref="§2.6" />;
          </li>
          <li>
            the context brings in earlier rows of the same task table, with their labels visible. From the seed, the sampler goes up to
            the entity row, then down to that entity’s other task rows with timestamps at or before the seed’s{' '}
            <Cite kind="code" ref="rustler/src/fly.rs:415-424" /> <Cite kind="spec" ref="§2.5" />. These are the{' '}
            <Term id="in-context-label">in-context labels</Term>.
          </li>
        </ul>
        <Code s={FLY_TASK_TABLE_SKIP} title="Task rows enter the context only for the active task, and only from the seed’s past" />
        <p>
          The paper is explicit that this is not few-shot prompting: “While task rows provide ‘in-context labels’, our setting is not
          few-shot as explicit subgraph-label pairs are not required” <Cite kind="paper" ref="p.2" />. The labels are just more cells
          that the attention can read.
        </p>
      </Section>

      <Section id="no-history" title="And if the task has no history at all?">
        <p>
          Then there are no past labels to put in the context, and the paper’s numbers say to expect little. Removing the target
          entity’s own past labels (its self labels) drops mean zero-shot AUROC from 70.1 to 53.8 <Cite kind="paper" ref="p.9 Table 4" />. On{' '}
          <code>rel-trial/study-outcome</code>, contexts contain no labels at all (0 ± 0 per context){' '}
          <Cite kind="paper" ref="p.23 Table 12" />, and RT’s zero-shot AUROC is 51.8, next to the 50.0 of random guessing{' '}
          <Cite kind="paper" ref="p.8 Table 1" />. <Link to="/results">Chapter 15</Link> shows the same collapse on this site’s model.
        </p>
        <p>
          The code also needs some history before it can run. Validation and test splits copy their normalization statistics from
          the task’s training split, so a task without a training split stops preprocessing with a panic{' '}
          <Cite kind="code" ref="rustler/src/pre.rs:383-396" /> <Cite kind="spec" ref="§3.1" />.
        </p>
        <Code s={PRE_TRAIN_STATS} title="Val and Test reuse the Train split’s statistics" />
      </Section>

      <Section id="autocomplete" title="Autocomplete tasks need no task table">
        <p>
          The other task family avoids the question entirely. An <Term id="autocomplete-task">autocomplete task</Term> predicts a column
          that already exists, such as a review’s rating. The seed is the database row itself, its cell in that column is masked, and
          the in-context labels are the same column’s values in other rows that the sampler reaches. No task-table row ever enters
          such a context <Cite kind="spec" ref="§2.4" /> <Cite kind="code" ref="rustler/src/fly.rs:416-419" />. See{' '}
          <Link to="/tasks">chapter 3</Link> for both families on ShopDB.
        </p>
      </Section>
    </>
  );
}
