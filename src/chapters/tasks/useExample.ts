/**
 * The running example of chapter 3, resolved from the LIVE simulation state.
 *
 * The chapter teaches ShopDB's four tasks, so its figures always draw ShopDB (the held-out target
 * DB, BUILD_PLAN §3.1). Wherever the live simulation is on one of those tasks, the figures start
 * from the live seed (same customer / product / review as every other chapter); otherwise they
 * start from the engine's deterministic defaults (showcase customer, default seeds).
 */
import { useMemo } from 'react';
import { getDatabase, pickShowcaseEntity } from '@/engine/db';
import { getPreprocessed, getSeeds, pickDefaultSeed } from '@/engine/pipeline';
import type { PreprocessedDB, RawDatabase, TaskDef } from '@/engine/types';
import { useSeedNode, useTask } from '@/state/hooks';

export interface ChapterExample {
  shop: RawDatabase;
  shopPdb: PreprocessedDB;
  churn: TaskDef;
  sales: TaskDef;
  rating: TaskDef;
  verified: TaskDef;
  /** The live task (any DB). */
  liveTask: TaskDef;
  liveSeed: number;
  /** True when the live simulation runs on ShopDB. */
  liveIsShop: boolean;
  /** Customer row (= PK) for the churn figures. */
  customer: number;
  /** Product row for the sales figure. */
  product: number;
  /** Review row for the autocomplete figure. */
  review: number;
  /** Where each default came from, for captions. */
  customerFromLive: boolean;
  productFromLive: boolean;
  reviewFromLive: boolean;
}

function mustTask(db: RawDatabase, id: string): TaskDef {
  const t = db.tasks.find((x) => x.id === id);
  if (!t) throw new Error(`tasks chapter: task ${id} missing from ShopDB`);
  return t;
}

/** Entity row of a forecast seed (its single FK parent, spec §2.1), or null. */
function entityRowOf(pdb: PreprocessedDB, seed: number): number | null {
  const n = pdb.nodes[seed];
  if (!n || n.f2pNbrIdxs.length === 0) return null;
  return pdb.nodes[n.f2pNbrIdxs[0]]?.rowPos ?? null;
}

export function useChapterExample(): ChapterExample {
  const liveTask = useTask();
  const liveSeed = useSeedNode();
  return useMemo(() => {
    const shop = getDatabase('shop');
    const shopPdb = getPreprocessed(shop);
    const churn = mustTask(shop, 'shop/customer-churn');
    const sales = mustTask(shop, 'shop/product-sales');
    const rating = mustTask(shop, 'shop/reviews.rating');
    const verified = mustTask(shop, 'shop/reviews.verified');
    const liveIsShop = liveTask.dbName === 'shop';

    let customer = pickShowcaseEntity(shop, churn.id).entityRow;
    let customerFromLive = false;
    if (liveTask.id === churn.id && liveSeed >= 0) {
      const e = entityRowOf(shopPdb, liveSeed);
      if (e !== null) {
        customer = e;
        customerFromLive = true;
      }
    }

    let product = entityRowOf(shopPdb, pickDefaultSeed(shopPdb, shop, sales, 'Val', getSeeds(shopPdb, sales, 'Val'))) ?? 0;
    let productFromLive = false;
    if (liveTask.id === sales.id && liveSeed >= 0) {
      const e = entityRowOf(shopPdb, liveSeed);
      if (e !== null) {
        product = e;
        productFromLive = true;
      }
    }

    let review = shopPdb.nodes[pickDefaultSeed(shopPdb, shop, rating, 'Val', getSeeds(shopPdb, rating, 'Val'))]?.rowPos ?? 0;
    let reviewFromLive = false;
    if ((liveTask.id === rating.id || liveTask.id === verified.id) && liveSeed >= 0) {
      const n = shopPdb.nodes[liveSeed];
      if (n && n.tableName === 'reviews') {
        review = n.rowPos;
        reviewFromLive = true;
      }
    }

    return {
      shop,
      shopPdb,
      churn,
      sales,
      rating,
      verified,
      liveTask,
      liveSeed,
      liveIsShop,
      customer,
      product,
      review,
      customerFromLive,
      productFromLive,
      reviewFromLive,
    };
  }, [liveTask, liveSeed]);
}
