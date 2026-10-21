/**
 * Public API of the dummy-database layer (BUILD_PLAN §2, §3).
 *
 *   getDatabase('shop')            → RawDatabase (deterministic, cached)
 *   listDatabases()                → [{ name, title, role, description }]
 *   tableColorSlot('shop', 'orders') → 3   (task tables → 5)
 *   pickShowcaseEntity(db)         → running-example entity of shop/customer-churn
 */
export {
  DB_NAMES,
  TASK_TABLE_SLOT,
  buildDatabase,
  getDatabase,
  getTask,
  isDbName,
  listDatabases,
  pickShowcaseEntity,
  tableColorSlot,
  type DbInfo,
  type DbName,
  type DbRole,
  type ShowcaseEntity,
} from './registry';
export {
  RACING_VAL_TIMESTAMP,
  TASK_SPECS,
  buildTasks,
  describeTimeline,
  forecastTimestamps,
  type AutocompleteSpec,
  type EventSource,
  type ForecastSpec,
  type LabelAgg,
} from './tasks';
export { DATA_END, DATA_START, DAY_MS, TEST_TIMESTAMP, VAL_TIMESTAMP } from './util';

/** The column phrase RT embeds for a column: exactly "<col> of <table>" (pre.rs:432-437, spec §3.6). */
export function columnPhrase(colName: string, tableName: string): string {
  return `${colName} of ${tableName}`;
}
