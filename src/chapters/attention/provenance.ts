/**
 * Where the loaded weights come from, in words, for the attention chapter's caveats and badges.
 * Read from the live checkpoint name and its meta (public/data/weights/<ckpt>.json) and from the
 * training record public/data/training/dataset.json (`pretrain.dbs`, `pretrain.heldOut`), so the
 * sentence follows whatever checkpoint the reader selects. Nothing here is hard-coded per checkpoint.
 */
import { useEffect, useState } from 'react';
import { listDatabases } from '@/engine/db/registry';
import { loadPublicJson } from '@/engine/model/io';
import type { AttnLive } from './useAttnLive';

interface DatasetJson {
  pretrain?: { dbs?: string[]; heldOut?: string };
}

let datasetPromise: Promise<DatasetJson | null> | null = null;
function loadDataset(): Promise<DatasetJson | null> {
  if (!datasetPromise) {
    datasetPromise = loadPublicJson<DatasetJson>('data/training/dataset.json').catch(() => {
      datasetPromise = null;
      return null;
    });
  }
  return datasetPromise;
}

export type ProvenanceKind = 'init' | 'zero-shot' | 'seen' | 'finetuned';

export interface Provenance {
  kind: ProvenanceKind;
  /** Titles of the pretraining databases, e.g. ["SchoolDB", "RacingDB", "ForumDB"]. */
  pretrainTitles: string[];
  /** Title of the current database. */
  dbTitle: string;
  /** For fine-tuned checkpoints: the task it was fine-tuned on (e.g. "customer-churn"), if known. */
  finetuneTask: string | null;
}

const listWords = (xs: string[]) => (xs.length <= 1 ? xs.join('') : `${xs.slice(0, -1).join(', ')} and ${xs[xs.length - 1]}`);

export function useProvenance(live: AttnLive): Provenance | null {
  const [ds, setDs] = useState<DatasetJson | null>(null);
  useEffect(() => {
    let alive = true;
    loadDataset().then((d) => alive && setDs(d));
    return () => {
      alive = false;
    };
  }, []);
  const { assets, db } = live;
  if (!ds?.pretrain?.dbs) return null;
  const titles = new Map<string, string>(listDatabases().map((d) => [d.name, d.title]));
  const pretrainTitles = ds.pretrain.dbs.map((n) => titles.get(n) ?? n);
  const dbTitle = titles.get(db.name) ?? db.name;
  const seen = ds.pretrain.dbs.includes(db.name);
  if (assets.weightsKind === 'init' || assets.ckpt === 'init') return { kind: 'init', pretrainTitles, dbTitle, finetuneTask: null };
  if (assets.ckpt.startsWith('finetune-')) {
    const sel = String(assets.weights?.meta?.selection ?? '');
    const m = /[a-z]+\/([\w.-]+)/.exec(sel);
    return { kind: 'finetuned', pretrainTitles, dbTitle, finetuneTask: m ? m[1] : null };
  }
  return { kind: seen ? 'seen' : 'zero-shot', pretrainTitles, dbTitle, finetuneTask: null };
}

/** One sentence: what the weights were trained on, and whether the maps are zero-shot. */
export function provenanceSentence(p: Provenance): string {
  const dbs = `${p.pretrainTitles.length === 3 ? 'three' : p.pretrainTitles.length} synthetic databases (${listWords(p.pretrainTitles)})`;
  switch (p.kind) {
    case 'init':
      return 'These weights are untrained (a random initialisation), so the patterns carry no learned meaning.';
    case 'zero-shot':
      return `It was pretrained on ${dbs}; ${p.dbTitle} was held out, so every map here is a zero-shot attention pattern on a database the model never saw.`;
    case 'seen':
      return `It was pretrained on ${dbs}, including ${p.dbTitle}, so these patterns are not zero-shot.`;
    case 'finetuned':
      return `It was pretrained on ${dbs} and then fine-tuned on ${p.dbTitle}’s ${p.finetuneTask ?? 'task'} Train split, so these are no longer zero-shot patterns.`;
  }
}
