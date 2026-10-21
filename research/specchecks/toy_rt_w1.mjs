// Independent re-implementation of paper-era RT preprocessing (pre.rs), sampler (fly.rs)
// and model masks/forward (model.py) on a toy DB, to produce worked examples for the spec.
// Not derived by executing upstream code.

// ---------------- PRNG ----------------
function mulberry32(a) { return function () { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
class Rng { constructor(s) { this.r = mulberry32(s); this.log = []; }
  int(n) { const v = Math.floor(this.r() * n); this.log.push(`int(${n})=${v}`); return v; }
  sample(n, k) { const a = [...Array(n).keys()]; for (let i = 0; i < k; i++) { const j = i + Math.floor(this.r() * (n - i)); [a[i], a[j]] = [a[j], a[i]]; } this.log.push(`sample(${n},${k})`); return a.slice(0, k); }
}

// ---------------- toy DB ----------------
const D = s => BigInt(Date.parse(s + "T00:00:00Z")) * 1000000n; // ns
const tables = [
  // glob order: db/*.parquet sorted, then tasks/*/*.parquet sorted (test, train, val)
  { name: "customers", type: "Db", pkey: "customer_id", fkeys: {}, tcol: null,
    cols: ["customer_id", "country", "age", "is_premium"], dtypes: { customer_id: "Int64", country: "String", age: "Float64", is_premium: "Boolean" },
    rows: [[0, "BR", 34, true], [1, "US", 51, false], [2, "BR", null, false]] },
  { name: "orders", type: "Db", pkey: "order_id", fkeys: { customer_id: "customers", product_id: "products" }, tcol: "order_time",
    cols: ["order_id", "customer_id", "product_id", "order_time", "quantity"], dtypes: { order_id: "Int64", customer_id: "Int64", product_id: "Int64", order_time: "Datetime", quantity: "Int64" },
    rows: [[0, 0, 0, D("2024-01-10"), 1], [1, 0, 1, D("2024-02-05"), 2], [2, 1, 0, D("2024-02-20"), 1], [3, 0, 0, D("2024-03-15"), 3], [4, 2, 1, D("2024-01-25"), 1]] },
  { name: "products", type: "Db", pkey: "product_id", fkeys: {}, tcol: null,
    cols: ["product_id", "category", "price"], dtypes: { product_id: "Int64", category: "String", price: "Float64" },
    rows: [[0, "books", 12.0], [1, "toys", 30.0]] },
  { name: "customer-churn", type: "Test", pkey: null, fkeys: { customer_id: "customers" }, tcol: "timestamp",
    cols: ["timestamp", "customer_id", "churn"], dtypes: { timestamp: "Datetime", customer_id: "Int64", churn: "Boolean" },
    rows: [[D("2024-04-01"), 0, false], [D("2024-04-01"), 2, true]] },
  { name: "customer-churn", type: "Train", pkey: null, fkeys: { customer_id: "customers" }, tcol: "timestamp",
    cols: ["timestamp", "customer_id", "churn"], dtypes: { timestamp: "Datetime", customer_id: "Int64", churn: "Boolean" },
    rows: [[D("2024-01-01"), 0, false], [D("2024-01-01"), 1, true], [D("2024-02-01"), 0, false], [D("2024-02-01"), 2, true]] },
  { name: "customer-churn", type: "Val", pkey: null, fkeys: { customer_id: "customers" }, tcol: "timestamp",
    cols: ["timestamp", "customer_id", "churn"], dtypes: { timestamp: "Datetime", customer_id: "Int64", churn: "Boolean" },
    rows: [[D("2024-03-01"), 0, true], [D("2024-03-01"), 1, false]] },
];

// ---------------- pre.rs ----------------
let off = 0; for (const t of tables) { t.offset = off; off += t.rows.length; }
const N = off;
const mean = a => a.length ? a.reduce((s, x) => s + x, 0) / a.length : null;
const stdS = a => { if (a.length <= 1) return null; const m = mean(a); return Math.sqrt(a.reduce((s, x) => s + (x - m) ** 2, 0) / (a.length - 1)); };
let dtCnt = 0, dtSum = 0, dtSq = 0;
for (const t of tables) t.stats = t.cols.map((c, j) => {
  const v = t.rows.map(r => r[j]).filter(x => x !== null);
  const dt = t.dtypes[c];
  if (dt === "Boolean") { const f = v.map(b => b ? 1 : 0); return { mean: mean(f) ?? 0, std: stdS(f) ?? 0 }; }
  if (["Int64", "Int32", "UInt32", "Float32", "Float64"].includes(dt)) { const f = v.map(Number).filter(x => !Number.isNaN(x)); let s = stdS(f) ?? 1; if (s === 0) s = 1; return { mean: mean(f) ?? 0, std: s }; }
  if (dt === "Datetime") { for (const x of v) { const f = Number(x); dtCnt++; dtSum += f; dtSq += f * f; } }
  return { mean: 0, std: 0 };
});
const dtMean = dtSum / dtCnt, dtStd = Math.sqrt(dtSq / dtCnt - dtMean * dtMean);
for (const t of tables) if (t.type === "Val" || t.type === "Test") t.stats = tables.find(x => x.name === t.name && x.type === "Train").stats;

const text = new Map(); const textList = [];
const tid = s => { if (!text.has(s)) { text.set(s, textList.length); textList.push(s); } return text.get(s); };
const nodes = Array.from({ length: N }, () => ({ is_task_node: false, node_idx: 0, table_name_idx: 0, timestamp: null, f2p: [], f2pEdges: [], cells: [] }));
const p2f = Array.from({ length: N }, () => []);
const toSec = ns => ns === null ? null : Number(ns / 1000000000n);
for (const t of tables) {
  const tni = tid(t.name);
  t.cols.forEach((c, j) => {
    const cni = tid(`${c} of ${t.name}`);
    if (c === t.pkey) return;
    if (c in t.fkeys) {
      const p = tables.find(x => x.name === t.fkeys[c] && x.type === "Db");
      t.rows.forEach((r, ri) => { const v = r[j]; if (v === null) return;
        const n = nodes[t.offset + ri]; n.is_task_node = t.type !== "Db"; n.node_idx = t.offset + ri; n.table_name_idx = tni;
        const pidx = p.offset + Number(v); n.f2p.push(pidx);
        const ts = t.tcol ? toSec(r[t.cols.indexOf(t.tcol)]) : null; n.timestamp = ts;
        n.f2pEdges.push({ node: pidx, tableNameIdx: tid(p.name), tableType: "Db" });
        p2f[pidx].push({ node: n.node_idx, tableNameIdx: tni, tableType: t.type, ts });
      });
      return;
    }
    const st = t.stats[j];
    t.rows.forEach((r, ri) => {
      const n = nodes[t.offset + ri]; n.is_task_node = t.type !== "Db"; n.node_idx = t.offset + ri; n.table_name_idx = tni;
      const v = r[j]; if (v === null) return;
      const dt = t.dtypes[c];
      const cell = { col: cni, colName: `${c} of ${t.name}`, sem: 0, num: 0, txt: 0, dt: 0, bool: 0, raw: v };
      if (dt === "Boolean") { cell.sem = 3; cell.bool = ((v ? 1 : 0) - st.mean) / st.std; }
      else if (dt === "Datetime") { cell.sem = 2; cell.dt = (Number(v) - dtMean) / dtStd; cell.raw = new Date(Number(v / 1000000n)).toISOString().slice(0, 10); }
      else if (dt === "String") { cell.sem = 1; cell.txt = tid(v); }
      else { const x = Number(v); if (Number.isNaN(x)) return; cell.sem = 0; cell.num = (x - st.mean) / st.std; }
      n.cells.push(cell);
    });
  });
}
const tableOf = i => tables.find(t => i >= t.offset && i < t.offset + t.rows.length);
const label = i => { const t = tableOf(i); return `${t.name}${t.type === "Db" ? "" : ":" + t.type}[${i - t.offset}]#${i}`; };

console.log("== offsets"); for (const t of tables) console.log(`${t.name}:${t.type} offset=${t.offset} n=${t.rows.length}`);
console.log("== stats"); for (const t of tables) console.log(t.name, t.type, JSON.stringify(t.stats.map((s, j) => [t.cols[j], +s.mean.toFixed(4), +s.std.toFixed(4)])));
console.log("dt mean", new Date(dtMean / 1e6).toISOString(), "dt std days", (dtStd / 1e9 / 86400).toFixed(3), "cnt", dtCnt);
console.log("== nodes");
nodes.forEach((n, i) => console.log(label(i), "ts=", n.timestamp === null ? null : new Date(n.timestamp * 1000).toISOString().slice(0, 10), "f2p=", JSON.stringify(n.f2p), "cells=", n.cells.map(c => `${c.colName}=${c.raw}->z ${[c.num, 0, c.dt, c.bool][c.sem] === 0 && c.sem === 1 ? "txt#" + c.txt : [c.num, c.txt, c.dt, c.bool][c.sem].toFixed(4)}`).join(" | "), "p2f=", JSON.stringify(p2f[i].map(e => e.node))));

// ---------------- fly.rs seq ----------------
function seq(seed, L, w, targetCol, dropCols, rng) {
  const S = nodes[seed]; const visited = new Array(N).fill(false);
  const stack = [[0, seed]]; const buckets = []; const out = []; const trace = [];
  const push = (d, n) => { while (buckets.length <= d) buckets.push([]); buckets[d].push(n); };
  while (true) {
    let depth, n, how;
    if (stack.length) { [depth, n] = stack.pop(); how = "stack"; }
    else { const d = buckets.findIndex(b => b.length > 0); if (d === -1) { trace.push("frontier empty -> stop"); break; }
      const b = buckets[d]; const r = rng.int(b.length); [b[r], b[b.length - 1]] = [b[b.length - 1], b[r]]; n = b.pop(); depth = d; how = `bucket${d}`; }
    if (visited[n]) { trace.push(`pop ${label(n)} via ${how}: already visited, skip`); continue; }
    visited[n] = true; const node = nodes[n];
    for (const e of node.f2pEdges) stack.push([depth + 1, e.node]);
    const db = []; const added = [];
    for (const e of p2f[n]) {
      if (e.tableType !== "Db" && e.tableNameIdx !== S.table_name_idx) continue;
      if (e.ts !== null && S.timestamp !== null && e.ts > S.timestamp) { added.push(`x${label(e.node)}(future)`); continue; }
      if (e.tableType === "Db") { db.push(e.node); continue; }
      push(depth + 1, e.node); added.push(`task:${label(e.node)}`);
    }
    const idx = db.length > w ? rng.sample(db.length, w) : [...db.keys()];
    for (const i of idx) { push(depth + 1, db[i]); added.push(`db:${label(db[i])}`); }
    const before = out.length;
    for (const c of node.cells) {
      if (dropCols.includes(c.col) && (node.node_idx === seed || node.timestamp === S.timestamp)) continue;
      const isT = node.node_idx === seed && c.col === targetCol;
      out.push({ node: n, f2p: [...node.f2p, -1, -1, -1, -1, -1].slice(0, 5), tni: node.table_name_idx, col: c.col, colName: c.colName, sem: c.sem, num: c.num, txt: c.txt, dt: c.dt, bool: c.bool, raw: c.raw, mask: isT, pad: false });
      if (out.length >= L) break;
    }
    trace.push(`visit ${label(n)} depth=${depth} via ${how}; push parents ${JSON.stringify(node.f2pEdges.map(e => label(e.node)))} at d${depth + 1}; children->bucket${depth + 1}: ${added.join(", ")}; emitted ${out.length - before} tokens`);
    if (out.length >= L) { trace.push("reached L -> stop"); break; }
  }
  while (out.length < L) out.push({ node: -1, f2p: [-1, -1, -1, -1, -1], tni: 0, col: 0, colName: "<pad>", sem: 0, num: 0, txt: 0, dt: 0, bool: 0, raw: "", mask: false, pad: true });
  return { out, trace };
}

const targetCol = text.get("churn of customer-churn");
const seed = tables.find(t => t.type === "Val").offset + 0; // val row 0: c0 @ 2024-03-01
const rng = new Rng(12345);
const L = 24, w = 1;
const { out, trace } = seq(seed, L, w, targetCol, [], rng);
console.log("== trace (seed", label(seed), ")"); trace.forEach(s => console.log(" ", s)); console.log(" rng:", rng.log.join(" "));
console.log("== tokens");
out.forEach((t, i) => console.log(String(i).padStart(2), t.pad ? "PAD" : `${label(t.node)} ${t.colName} sem=${t.sem} raw=${t.raw} f2p=${JSON.stringify(t.f2p.filter(x => x >= 0))}${t.mask ? " [MASK]" : ""}`));

// ---------------- model.py masks ----------------
const S_ = out.length;
const M = { col: [], feat: [], nbr: [], full: [] };
for (let q = 0; q < S_; q++) { for (const k of Object.keys(M)) M[k].push([]);
  for (let kv = 0; kv < S_; kv++) {
    const pad = !out[q].pad && !out[kv].pad;
    const same = out[q].node === out[kv].node;
    const kvInF2p = out[q].f2p.includes(out[kv].node);
    const qInF2p = out[kv].f2p.includes(out[q].node);
    const sameCol = out[q].col === out[kv].col && out[q].tni === out[kv].tni;
    M.feat[q].push((same || kvInF2p) && pad); M.nbr[q].push(qInF2p && pad); M.col[q].push(sameCol && pad); M.full[q].push(pad);
  } }
for (const k of ["col", "feat", "nbr", "full"]) {
  console.log(`== mask ${k} (rows=query, cols=key)`);
  console.log("    " + [...Array(S_).keys()].map(i => String(i % 10)).join(""));
  M[k].forEach((r, q) => console.log(String(q).padStart(3) + " " + r.map(b => b ? "#" : ".").join("")));
}
// counts per query for nbr
console.log("queries with empty nbr row:", M.nbr.map((r, q) => r.some(Boolean) ? null : q).filter(x => x !== null).join(","));
console.log("text list:", JSON.stringify(textList));
export { out, M };
