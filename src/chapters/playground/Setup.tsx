/**
 * Left panel: task, split and seed row. These write the GLOBAL store, so every other chapter
 * follows the seed picked here (BUILD_PLAN §7).
 */
import {
  ChevronDown,
  ChevronUp,
  Search,
  Shuffle,
  Sparkles,
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { IconButton, Segmented, cx, formatDate } from "@/components";
import type { RawValue, Split } from "@/engine/types";
import { useSim } from "@/state/store";
import { RadioList } from "./bits";
import { entityName, labelText, nodeTitle, usePlayLive } from "./context";

const SHOP_TASKS: { id: string; name: string; kind: string }[] = [
  {
    id: "shop/customer-churn",
    name: "Customer churn",
    kind: "forecast · binary",
  },
  {
    id: "shop/product-sales",
    name: "Product sales",
    kind: "forecast · regression",
  },
  {
    id: "shop/reviews.rating",
    name: "Review rating",
    kind: "autocomplete · regression",
  },
  {
    id: "shop/reviews.verified",
    name: "Review verified",
    kind: "autocomplete · binary",
  },
];

interface SeedItem {
  node: number;
  title: string;
  sub: string;
  truth: RawValue;
  search: string;
}

export function SetupPanel() {
  const { db, pdb, task, seeds, seedNode } = usePlayLive();
  const split = useSim((s) => s.split);
  const stored = useSim((s) => s.seedNode);
  const setTask = useSim((s) => s.setTask);
  const setSplit = useSim((s) => s.setSplit);
  const setSeedNode = useSim((s) => s.setSeedNode);
  const [q, setQ] = useState("");
  const listRef = useRef<HTMLUListElement | null>(null);

  // a forecast split holds one timestamp (Val, Test) or several (Train): show the date only when it varies
  const manyDates = useMemo(
    () => new Set(seeds.map((s) => pdb.nodes[s]?.ts)).size > 1,
    [seeds, pdb],
  );

  const items = useMemo<SeedItem[]>(() => {
    return seeds.map((s) => {
      const n = pdb.nodes[s];
      const cell = n.cells.find((c) => c.colName === task.targetCol);
      let title: string;
      let sub: string;
      if (task.kind === "autocomplete") {
        // a review: name it by its product and customer
        const row =
          db.tables.find((t) => t.schema.name === n.tableName)?.rows[
            n.rowPos
          ] ?? {};
        const prod =
          typeof row.product_id === "number"
            ? entityName(db, "products", row.product_id)
            : "";
        const cust =
          typeof row.customer_id === "number"
            ? entityName(db, "customers", row.customer_id)
            : "";
        title = prod || `review ${n.rowPos}`;
        sub = `review ${n.rowPos}${cust ? ` · by ${cust}` : ""}`;
      } else {
        const t = nodeTitle(db, pdb, s);
        title = t.title;
        const noun = t.sub.split(" · ")[0];
        sub =
          manyDates && n.ts !== null
            ? `${noun} · ${formatDate(n.ts * 1000)}`
            : noun;
      }
      return {
        node: s,
        title,
        sub,
        truth: cell?.raw ?? null,
        search: `${title} ${sub} #${s}`.toLowerCase(),
      };
    });
  }, [seeds, pdb, db, task, manyDates]);

  const shown = useMemo(() => {
    const k = q.trim().toLowerCase();
    return k ? items.filter((it) => it.search.includes(k)) : items;
  }, [items, q]);

  // keep the selected seed in view (scroll the list only, never the page)
  useEffect(() => {
    const list = listRef.current;
    const el = list?.querySelector<HTMLElement>('[data-current="true"]');
    if (el && list) {
      const box = list.getBoundingClientRect();
      const r = el.getBoundingClientRect();
      if (r.top < box.top || r.bottom > box.bottom)
        list.scrollTop += r.top - box.top - box.height / 2 + r.height / 2;
    }
  }, [seedNode, task.id, split]);

  const isShop = db.name === "shop";
  const noun =
    task.kind === "autocomplete"
      ? "review"
      : task.entityTable === "products"
        ? "product"
        : "customer";
  const idx = seeds.indexOf(seedNode);
  const step = (d: number) => {
    if (!seeds.length) return;
    const i = idx < 0 ? 0 : (idx + d + seeds.length) % seeds.length;
    setSeedNode(seeds[i]);
  };
  const tasks = isShop
    ? SHOP_TASKS
    : db.tasks.map((t) => ({
        id: t.id,
        name: t.name,
        kind: `${t.kind} · ${t.targetType}`,
      }));

  return (
    <div className="@container flex min-w-0 flex-col gap-4">
      <div className="eyebrow">Pick a seed</div>
      <div className="grid min-w-0 gap-4 @2xl:grid-cols-[minmax(0,1fr)_minmax(0,1.25fr)] @2xl:gap-x-8">
        <div className="flex min-w-0 flex-col gap-4">
          {!isShop && (
            <div className="rounded-lg border border-dashed border-border-strong px-3 py-2.5 text-[15px] leading-snug text-ink-2">
              The course is set to {db.title}, a pretraining database, so its
              predictions are not zero-shot.{" "}
              <button
                type="button"
                className="font-medium text-accent-text underline underline-offset-2"
                onClick={() => setTask("shop/customer-churn")}
              >
                Switch to ShopDB
              </button>
            </div>
          )}

          <fieldset className="m-0 min-w-0 border-0 p-0">
            <legend
              className="mb-1.5 p-0 text-[15px] font-medium text-ink-2"
              aria-hidden="true"
            >
              Task
            </legend>
            <RadioList
              label="Task"
              value={task.id}
              onChange={setTask}
              items={tasks.map((t) => ({
                value: t.id,
                label: t.name,
                sub: t.kind,
              }))}
            />
            <p className="!mb-0 mt-2 text-[14.5px] leading-snug text-ink-2">
              {task.description}
            </p>
          </fieldset>

          {task.kind === "forecast" ? (
            <div className="flex flex-col gap-1.5">
              <span className="text-[15px] font-medium text-ink-2">
                Split of the seed row
              </span>
              <Segmented<Split>
                label="Split of the seed row"
                value={split}
                onChange={setSplit}
                options={[
                  { value: "Train", label: "Train" },
                  { value: "Val", label: "Val" },
                  { value: "Test", label: "Test" },
                ]}
              />
            </div>
          ) : (
            <p className="!m-0 text-[14.5px] leading-snug text-muted">
              Autocomplete seeds are all rows of the table, so there is no split
              (spec §2.4).
            </p>
          )}
        </div>
        <div className="flex min-w-0 flex-col gap-1.5">
          <div className="flex items-center justify-between gap-2">
            <span className="text-[15px] font-medium text-ink-2">
              Seed row{" "}
              <span className="num text-muted">
                ({seeds.length.toLocaleString("en-US")})
              </span>
            </span>
            <div className="flex items-center gap-1">
              <IconButton
                label={`Random ${noun}`}
                size="sm"
                title={`Random ${noun}`}
                onClick={() =>
                  seeds.length &&
                  setSeedNode(seeds[Math.floor(Math.random() * seeds.length)])
                }
              >
                <Shuffle size={15} />
              </IconButton>
              <IconButton
                label="The course’s default seed"
                title="The course’s default seed"
                size="sm"
                disabled={stored === null}
                onClick={() => setSeedNode(null)}
              >
                <Sparkles size={15} />
              </IconButton>
            </div>
          </div>
          <label className="relative block">
            <span className="sr-only">Search {noun}s</span>
            <Search
              aria-hidden="true"
              size={15}
              className="pointer-events-none absolute top-1/2 left-2.5 -translate-y-1/2 text-muted"
            />
            <input
              type="search"
              value={q}
              onChange={(e) => setQ(e.target.value)}
              placeholder={`Search ${noun}s`}
              className="h-9 w-full rounded-lg border border-border-strong bg-surface pr-2.5 pl-8 text-[15.5px] text-ink placeholder:text-muted"
            />
          </label>
          <ul
            ref={listRef}
            aria-label={`Seed rows of ${task.name}`}
            className="scroll-quiet !m-0 max-h-[300px] !list-none overflow-y-auto rounded-lg border border-border !p-0"
          >
            {shown.length === 0 && (
              <li className="!m-0 px-3 py-3 text-[15px] text-muted">
                No match.
              </li>
            )}
            {shown.map((it) => {
              const current = it.node === seedNode;
              return (
                <li
                  key={it.node}
                  className="!m-0 border-b border-grid/70 !p-0 last:border-0"
                >
                  <button
                    type="button"
                    data-current={current}
                    aria-current={current ? "true" : undefined}
                    onClick={() => setSeedNode(it.node)}
                    className={cx(
                      "flex w-full items-center gap-2 px-2.5 py-1.5 text-left transition-colors",
                      current ? "bg-accent-soft" : "hover:bg-surface-2",
                    )}
                  >
                    <span className="min-w-0 flex-1">
                      <span
                        className={cx(
                          "block truncate text-[15px] leading-snug",
                          current ? "font-semibold text-ink" : "text-ink",
                        )}
                        title={it.title}
                      >
                        {it.title}
                      </span>
                      <span
                        className="block truncate text-[14px] leading-snug text-muted"
                        title={it.sub}
                      >
                        {it.sub}
                      </span>
                    </span>
                    <span
                      className="num inline-flex h-6 min-w-[28px] shrink-0 items-center justify-center rounded-md bg-surface-2 px-1.5 text-[14px] text-ink-2"
                      title={`Hidden label (the model never sees it): ${task.targetCol} = ${labelText(it.truth)}`}
                    >
                      <span className="sr-only">hidden {task.targetCol} </span>
                      {labelText(it.truth)}
                    </span>
                  </button>
                </li>
              );
            })}
          </ul>
          <div className="flex items-start justify-between gap-2">
            <p className="!m-0 text-[14px] leading-snug text-muted">
              The chip is the hidden {task.targetCol}, shown to help you find
              interesting cases. The model never sees it.
            </p>
            <div className="flex shrink-0 items-center gap-1">
              <IconButton
                label="Previous seed"
                title="Previous seed"
                size="sm"
                onClick={() => step(-1)}
              >
                <ChevronUp size={16} />
              </IconButton>
              <IconButton
                label="Next seed"
                title="Next seed"
                size="sm"
                onClick={() => step(1)}
              >
                <ChevronDown size={16} />
              </IconButton>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
