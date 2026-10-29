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
  /** How the checkpoint was chosen, from its meta (e.g. "best ShopDB customer-churn VAL AUROC"), if recorded. */
  selection: string | null;
}

const listWords = (xs: string[]) => (xs.length <= 1 ? xs.join('') : `${xs.slice(0, -1).join(', ')} and ${xs[xs.length - 1]}`);
const NUMBER_WORDS = ['no', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'];
const countWord = (n: number) => NUMBER_WORDS[n] ?? String(n);

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
  const sel = assets.weights?.meta?.selection;
  const selection = typeof sel === 'string' && sel ? sel : null;
  if (assets.weightsKind === 'init' || assets.ckpt === 'init') return { kind: 'init', pretrainTitles, dbTitle, finetuneTask: null, selection };
  if (assets.ckpt.startsWith('finetune-')) {
    const m = /[a-z]+\/([\w.-]+)/.exec(selection ?? '');
    return { kind: 'finetuned', pretrainTitles, dbTitle, finetuneTask: m ? m[1] : null, selection };
  }
  return { kind: seen ? 'seen' : 'zero-shot', pretrainTitles, dbTitle, finetuneTask: null, selection };
}

const dbList = (p: Provenance, small = false) => `${countWord(p.pretrainTitles.length)} ${small ? 'small ' : ''}synthetic databases (${listWords(p.pretrainTitles)})`;

/**
 * One sentence: what the weights were trained on, and whether the maps are zero-shot. "Zero-shot"
 * means no gradient update on the target DB (spec §9.1).
 */
export function provenanceSentence(p: Provenance): string {
  const dbs = dbList(p);
  switch (p.kind) {
    case 'init':
      return 'These weights are untrained (a random initialization), so the patterns carry no learned meaning.';
    case 'zero-shot':
      return `The mini model was pretrained on ${dbs}. ${p.dbTitle} was held out, so the model never trained on it, and every map here is a zero-shot attention pattern.`;
    case 'seen':
      return `The mini model was pretrained on ${dbs}, including ${p.dbTitle}, so these patterns are not zero-shot.`;
    case 'finetuned':
      return `The mini model was pretrained on ${dbs} and then fine-tuned on ${p.dbTitle}’s ${p.finetuneTask ?? 'task'} Train split, so these are no longer zero-shot patterns.`;
  }
}

/** A clause for "a N-parameter replica …": what the replica was trained on, in a few words. */
export function provenanceClause(p: Provenance): string {
  const dbs = dbList(p, true);
  switch (p.kind) {
    case 'init':
      return 'left at its random initialization (no training at all)';
    case 'zero-shot':
      return `pretrained on ${dbs}, with ${p.dbTitle} held out, so these are zero-shot patterns`;
    case 'seen':
      return `pretrained on ${dbs}, ${p.dbTitle} among them, so these patterns are not zero-shot`;
    case 'finetuned':
      return `pretrained on ${dbs} and then fine-tuned on ${p.dbTitle}’s ${p.finetuneTask ?? 'task'} Train split`;
  }
}
