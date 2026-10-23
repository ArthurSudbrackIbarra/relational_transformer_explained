import type { RawTable } from '../../types';

/** What a generator returns: the DB tables only. Task tables are built by tasks.ts. */
export interface GeneratedDb {
  name: string;
  /** DB tables in schema order (also the color-slot order, BUILD_PLAN §3.5). */
  tables: RawTable[];
}
