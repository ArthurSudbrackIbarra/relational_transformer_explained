/**
 * Engine-backed parts of the simulation popover (loaded lazily so the shell
 * renders even before the engine bundle is fetched): DB / task / seed pickers
 * and the resolved seed label.
 */
import { Shuffle } from 'lucide-react';
import { useMemo } from 'react';
import { getDatabase } from '@/engine/db/registry';
import { useDatabase, useDatabaseList, usePreprocessed, useSeedNode, useSeeds, useTask } from '@/state/hooks';
import { useSim, type DbName } from '@/state/store';
import type { Split } from '@/engine/types';
import { IconButton, Segmented, Select } from '../ui/controls';

/** Describe a seed node for the picker: table[row] plus the entity / time when available. */
function useSeedLabels(seeds: number[]): Map<number, string> {
  const pdb = usePreprocessed();
  return useMemo(() => {
    const m = new Map<number, string>();
    for (const s of seeds) {
      const n = pdb.nodes[s];
      if (!n) continue;
      const ts = n.ts !== null ? new Date(n.ts * 1000).toISOString().slice(0, 10) : '';
      const parent = n.f2pNbrIdxs[0];
      const p = parent !== undefined ? pdb.nodes[parent] : undefined;
      const ent = p ? ` · ${p.tableName}[${p.rowPos}]` : '';
      const tbl = n.tableType !== 'Db' ? `${n.tableName}:${n.tableType}` : n.tableName;
      m.set(s, `#${s} ${tbl}[${n.rowPos}]${ent}${ts ? ` · ${ts}` : ''}`);
    }
    return m;
  }, [seeds, pdb]);
}

/** DB, task and seed selectors. */
export function EngineFields() {
  const dbs = useDatabaseList();
  const db = useDatabase();
  const task = useTask();
  const seeds = useSeeds();
  const seed = useSeedNode();
  const stored = useSim((s) => s.seedNode);
  const setDb = useSim((s) => s.setDb);
  const setTask = useSim((s) => s.setTask);
  const setSeedNode = useSim((s) => s.setSeedNode);
  const split = useSim((s) => s.split);
  const setSplit = useSim((s) => s.setSplit);
  const labels = useSeedLabels(seeds);
  const MAX = 400;
  const shown = seeds.length > MAX ? [...seeds.slice(0, MAX), ...(seeds.slice(MAX).includes(seed) ? [seed] : [])] : seeds;

  return (
    <div className="grid gap-3">
      <Select
        label="Database"
        value={db.name}
        onChange={(e) => {
          const name = e.target.value as DbName;
          // switching DB starts from its first task
          const first = getDatabase(name).tasks[0];
          if (first) setDb(name, first.id);
        }}
        hint={dbs.find((d) => d.name === db.name)?.role === 'target' ? 'Target DB: held out from pretraining.' : 'Pretraining DB: seen during pretraining.'}
      >
        {dbs.map((d) => (
          <option key={d.name} value={d.name}>
            {d.title}
            {d.role === 'target' ? ' (target)' : ''}
          </option>
        ))}
      </Select>
      <Select label="Task" value={task.id} onChange={(e) => setTask(e.target.value)}>
        {db.tasks.map((t) => (
          <option key={t.id} value={t.id}>
            {t.id.split('/')[1]} · {t.kind}, {t.targetType}
          </option>
        ))}
      </Select>
      <div className="flex flex-col gap-1">
        <span className="text-[15px] font-medium text-ink-2">Seed split</span>
        <Segmented<Split>
          label="Seed split"
          value={split}
          onChange={setSplit}
          disabled={task.kind === 'autocomplete'}
          options={[
            { value: 'Train', label: 'Train' },
            { value: 'Val', label: 'Val' },
            { value: 'Test', label: 'Test' },
          ]}
        />
        {task.kind === 'autocomplete' && (
          <p className="text-[14.5px] leading-snug text-muted">Autocomplete seeds are all rows of the table; the split is ignored (spec §2.4).</p>
        )}
      </div>
      <div className="flex items-end gap-1.5">
        <Select
          className="flex-1"
          label={`Seed row (${seeds.length.toLocaleString('en-US')} seeds)`}
          value={stored === null ? 'auto' : String(seed)}
          onChange={(e) => setSeedNode(e.target.value === 'auto' ? null : Number(e.target.value))}
        >
          <option value="auto">auto · {labels.get(seed) ?? `#${seed}`}</option>
          {shown.map((s) => (
            <option key={s} value={s}>
              {labels.get(s) ?? `#${s}`}
            </option>
          ))}
        </Select>
        <IconButton
          label="Random seed row"
          size="sm"
          className="mb-0 border border-border-strong"
          onClick={() => {
            if (seeds.length) setSeedNode(seeds[Math.floor(Math.random() * seeds.length)]);
          }}
        >
          <Shuffle size={14} />
        </IconButton>
      </div>
    </div>
  );
}

/** "#123" for the resolved seed (used in the top-bar chip). */
export function SeedLabel() {
  const seed = useSeedNode();
  return <>#{seed}</>;
}
