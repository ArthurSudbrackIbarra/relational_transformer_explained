"""Independent port of paper-era pre.rs (stats/nodes/edges) + fly.rs Sampler::seq + model.py masks,
written from the Rust/Python sources (NOT from the TS or the spec pseudo-code), compared against the
TS outputs dumped by python/sampler_dump.ts. The PRNG is a bit-exact port of src/engine/rng.ts
(xoshiro128**), so RNG-consuming decisions line up draw for draw.

Usage:
  npx tsx python/sampler_dump.ts python/data/sampler-dump.json
  python/.venv/bin/python -I python/sampler_crosscheck.py python/data/sampler-dump.json [--json summary.json]
"""
import json, math, sys
import numpy as np

M = 0xFFFFFFFF
def imul(a, b): return (a * b) & M
def rotl(x, k): return ((x << k) | (x >> (32 - k))) & M

class Rng:  # bit-exact port of src/engine/rng.ts
    def __init__(self, seed):
        x = (int(seed) & M) ^ 0x9E3779B9
        st = []
        for _ in range(4):
            x = (x + 0x9E3779B9) & M
            z = x
            z = imul(z ^ (z >> 16), 0x85EBCA6B)
            z = imul(z ^ (z >> 13), 0xC2B2AE35)
            st.append((z ^ (z >> 16)) & M)
        self.s = st
        if not any(st): self.s[0] = 1
    def u32(self):
        s0, s1, s2, s3 = self.s
        res = imul(rotl(imul(s1, 5), 7), 9)
        t = (s1 << 9) & M
        s2 ^= s0; s3 ^= s1; s1 ^= s2; s0 ^= s3; s2 ^= t; s3 = rotl(s3, 11)
        self.s = [s0 & M, s1 & M, s2 & M, s3 & M]
        return res
    def int(self, n):
        if n == 1: return 0
        lim = 2**32 - (2**32 % n)
        while True:
            r = self.u32()
            if r < lim: return r % n
    def sample(self, n, k):
        idx = list(range(n))
        for i in range(k):
            j = i + self.int(n - i)
            idx[i], idx[j] = idx[j], idx[i]
        return idx[:k]

def i32(v): return ((v + 2**31) % 2**32) - 2**31

def path_key(s): return s.encode('utf-8')

def preprocess(raw):
    # pre.rs:79-92 glob order: db/*.parquet sorted, then tasks/*/*.parquet sorted (component-wise)
    db = sorted(raw['tables'], key=lambda t: path_key(t['name'] + '.parquet'))
    tasks = []
    for t in raw['tasks']:
        if t['tables'] is None: continue
        for split, fn in (('Test', 'test'), ('Train', 'train'), ('Val', 'val')):
            tasks.append((t['name'], split, fn, t['tables'][split]))
    tasks.sort(key=lambda x: (path_key(x[0]), path_key(x[2] + '.parquet')))
    entries = [(t['name'], 'Db', t) for t in db] + [(n, s, t) for n, s, _, t in tasks]
    off, tbl = 0, {}
    for name, tt, t in entries:
        tbl[(name, tt)] = dict(name=name, tt=tt, t=t, off=off, n=len(t['rows']), stats=[])
        off += len(t['rows'])
    N = off
    # pre.rs:310-376
    dt_cnt, dt_sum, dt_sumsq = 0, 0.0, 0.0
    for e in tbl.values():
        t = e['t']
        for j, c in enumerate(t['columns']):
            col = [r[j] for r in t['rows'] if r[j] is not None]
            if c['dtype'] == 'bool':
                x = np.array([1.0 if v else 0.0 for v in col])
                mean = float(x.mean()) if len(x) else 0.0
                std = float(x.std(ddof=1)) if len(x) > 1 else 0.0      # unwrap_or(0.0), no guard
                e['stats'].append((mean, std))
            elif c['dtype'] in ('int', 'float'):
                x = np.array([float(v) for v in col if not (isinstance(v, float) and math.isnan(v))])
                mean = float(x.mean()) if len(x) else 0.0
                std = float(x.std(ddof=1)) if len(x) > 1 else 1.0      # unwrap_or(1.0)
                if std == 0.0: std = 1.0
                e['stats'].append((mean, std))
            elif c['dtype'] == 'datetime':
                ns = [float(int(v) * 1_000_000) for v in col]           # i64 ns cast to f64
                dt_cnt += len(ns); dt_sum += sum(ns); dt_sumsq += sum(f * f for f in ns)
                e['stats'].append((0.0, 0.0))
            else:
                e['stats'].append((0.0, 0.0))
    dt_mean = dt_sum / dt_cnt
    dt_std = math.sqrt(dt_sumsq / dt_cnt - dt_mean * dt_mean)
    for (name, tt), e in tbl.items():                                 # pre.rs:383-396
        if tt in ('Val', 'Test'): e['stats'] = list(tbl[(name, 'Train')]['stats'])
    # pre.rs:399-615 (table iteration in glob order, see the TS deviation note)
    text, t2i, colidx = [], {}, {}
    def intern(s):
        if s not in t2i: t2i[s] = len(text); text.append(s); return t2i[s], True
        return t2i[s], False
    nodes = [dict(node_idx=0, tni=0, task=False, ts=None, f2p=[], fts=[], cells=[]) for _ in range(N)]
    p2f = [[] for _ in range(N)]
    for name, tt, _t in entries:
        e = tbl[(name, tt)]; t = e['t']
        tni, _ = intern(name)
        tcol = t['timeCol']
        tj = [c['name'] for c in t['columns']].index(tcol) if tcol else None
        for j, c in enumerate(t['columns']):
            mean, std = e['stats'][j]
            phrase = f"{c['name']} of {name}"
            ci, new = intern(phrase)
            if new: colidx[phrase] = ci
            if c['name'] == (t['pkey'] or ''): continue
            if c['name'] in t['fkeys']:
                pt = t['fkeys'][c['name']]; pe = tbl[(pt, 'Db')]
                for r, row in enumerate(t['rows']):
                    v = row[j]
                    if v is None: continue
                    ni = e['off'] + r; nd = nodes[ni]
                    nd['task'] = tt != 'Db'; nd['node_idx'] = ni; nd['tni'] = tni
                    pni = pe['off'] + int(v)
                    nd['f2p'].append(pni)
                    ts = None
                    if tj is not None and row[tj] is not None:
                        ns = int(row[tj]) * 1_000_000
                        ts = i32(int(ns / abs(ns) * (abs(ns) // 1_000_000_000)) if ns else 0)  # trunc toward 0
                    nd['ts'] = ts
                    ptni, _ = intern(pt)
                    ptcol = pe['t']['timeCol']; pts = None
                    if ptcol:
                        pj = [cc['name'] for cc in pe['t']['columns']].index(ptcol)
                        pv = pe['t']['rows'][int(v)][pj]
                        if pv is not None:
                            ns = int(pv) * 1_000_000
                            pts = i32(int(ns / abs(ns) * (abs(ns) // 1_000_000_000)) if ns else 0)
                    nd['fts'].append(pts)
                    p2f[pni].append((ni, tni, tt, ts))
                continue
            for r, row in enumerate(t['rows']):
                ni = e['off'] + r; nd = nodes[ni]
                nd['task'] = tt != 'Db'; nd['node_idx'] = ni; nd['tni'] = tni
                v = row[j]
                if v is None: continue
                d = c['dtype']
                with np.errstate(divide='ignore', invalid='ignore'):
                    if d == 'bool':
                        z = np.float64((1.0 if v else 0.0) - mean) / np.float64(std)
                        nd['cells'].append((ci, 3, 0.0, 0, 0.0, float(np.float32(z))))
                    elif d in ('int', 'float'):
                        if isinstance(v, float) and math.isnan(v): continue
                        z = (float(v) - mean) / std
                        assert math.isfinite(z), 'pre.rs:570 panic'
                        nd['cells'].append((ci, 0, float(np.float32(z)), 0, 0.0, 0.0))
                    elif d == 'datetime':
                        z = (float(int(v) * 1_000_000) - dt_mean) / dt_std
                        nd['cells'].append((ci, 2, 0.0, 0, float(np.float32(z)), 0.0))
                    elif d == 'str':
                        ti, _ = intern(v)
                        nd['cells'].append((ci, 1, 0.0, ti, 0.0, 0.0))
                    else:
                        raise RuntimeError('pre.rs:606 panic')
    info = {f"{n}:{tt}": dict(nodeIdxOffset=e['off'], numNodes=e['n']) for (n, tt), e in tbl.items()}
    return dict(nodes=nodes, p2f=p2f, text=text, colidx=colidx, info=info, dt=(dt_mean / 1e6, dt_std / 1e6, dt_cnt), tbl=tbl)

def seq(P, seed, L, w, target, drop, rngseed):   # fly.rs:357-511
    nodes, p2f = P['nodes'], P['p2f']
    visited = [False] * len(nodes)
    f2p_ftr = [(0, seed)]
    sn = nodes[seed]
    p2f_ftr = []
    rng = Rng(rngseed)
    out, order = [], []
    while True:
        if f2p_ftr:
            depth, n = f2p_ftr.pop()
        else:
            choices = [i for i, b in enumerate(p2f_ftr) if b]
            if not choices: break
            depth = choices[0]
            r = rng.int(len(p2f_ftr[depth])); l = len(p2f_ftr[depth])
            p2f_ftr[depth][r], p2f_ftr[depth][l - 1] = p2f_ftr[depth][l - 1], p2f_ftr[depth][r]
            n = p2f_ftr[depth].pop()
        if visited[n]: continue
        visited[n] = True
        order.append((n, depth))
        nd = nodes[n]
        for pn in nd['f2p']: f2p_ftr.append((depth + 1, pn))
        dbk = []
        for (cn, ctni, ctt, cts) in p2f[n]:
            if ctni != sn['tni'] and ctt != 'Db': continue
            if cts is not None and sn['ts'] is not None and cts > sn['ts']: continue
            if ctt == 'Db': dbk.append(cn); continue
            while len(p2f_ftr) <= depth + 1: p2f_ftr.append([])
            p2f_ftr[depth + 1].append(cn)
        idxs = rng.sample(len(dbk), w) if len(dbk) > w else list(range(len(dbk)))
        for i in idxs:
            while len(p2f_ftr) <= depth + 1: p2f_ftr.append([])
            p2f_ftr[depth + 1].append(dbk[i])
        for (ci, sem, num, ti, dtv, bv) in nd['cells']:
            if (nd['node_idx'] == seed and ci in drop) or (nd['ts'] == sn['ts'] and ci in drop): continue
            assert len(nd['f2p']) <= 5
            f2p = (nd['f2p'] + [-1] * 5)[:5]
            tgt = seed == nd['node_idx'] and ci == target
            out.append([nd['node_idx'], ci, int(tgt), 0, f2p, nd['tni'], sem, ti, ti if sem == 1 else -1,
                        int(nd['task'] or ci == target), num, dtv, bv])
            if len(out) >= L: break
        if len(out) >= L: break
    while len(out) < L: out.append([-1, 0, 0, 1, [-1] * 5, 0, 0, 0, -1, 0, 0.0, 0.0, 0.0])   # fly.rs:65-85
    return out, order

def masks(tok):   # model.py:164-191
    node = np.array([t[0] for t in tok]); f2p = np.array([t[4] for t in tok]); col = np.array([t[1] for t in tok])
    tab = np.array([t[5] for t in tok]); pad = np.array([bool(t[3]) for t in tok])
    P = (~pad[:, None]) & (~pad[None, :])
    same = node[:, None] == node[None, :]
    kv_in = (node[None, :, None] == f2p[:, None, :]).any(-1)
    q_in = (node[:, None, None] == f2p[None, :, :]).any(-1)
    sct = (col[:, None] == col[None, :]) & (tab[:, None] == tab[None, :])
    return {'feat': (same | kv_in) & P, 'nbr': q_in & P, 'col': sct & P, 'full': P}

def digest(m):
    out = []
    for q in range(m.shape[0]):
        c, h = 0, 0
        for k in np.nonzero(m[q])[0]:
            c += 1; h = (h * 31 + int(k)) % 1000000007
        out.append(c * 1000000007 + h)
    return out

def close(a, b, rel=2e-7):
    if isinstance(a, float) or isinstance(b, float):
        if math.isnan(a) and math.isnan(b): return True
        return a == b or abs(a - b) <= rel * max(1.0, abs(a), abs(b))
    return a == b

def main():
    data = json.load(open(sys.argv[1]))
    total_bad, total_ctx, total_listed = 0, 0, 0
    for name, d in data.items():
        bad = []
        P = preprocess(d['raw']); T = d['pre']
        if P['text'] != T['text']: bad.append('text dictionary differs')
        if P['colidx'] != T['columnIndex']: bad.append('column index differs')
        if P['info'] != T['tableInfo']: bad.append('table_info differs')
        m, s, c = P['dt']
        if c != T['datetimeStats']['count'] or not close(m, T['datetimeStats']['mean'], 1e-12) or not close(s, T['datetimeStats']['std'], 1e-9):
            bad.append(f"datetime stats {P['dt']} vs {T['datetimeStats']}")
        for (tn, tt), e in P['tbl'].items():
            ts_ = T['colStats'][f'{tn}:{tt}']
            for j, cdef in enumerate(e['t']['columns']):
                if cdef['name'] in ts_:
                    mu, sd = e['stats'][j]
                    if not (close(mu, ts_[cdef['name']]['mean'], 1e-12) and close(sd, ts_[cdef['name']]['std'], 1e-12)):
                        bad.append(f"stats {tn}:{tt}.{cdef['name']} {(mu, sd)} vs {ts_[cdef['name']]}")
        nd_bad = 0
        for i, (pn, tn) in enumerate(zip(P['nodes'], T['nodes'])):
            ok = pn['ts'] == tn['ts'] and pn['f2p'] == tn['f2p'] and pn['fts'] == tn['fts'] and pn['task'] == tn['task'] and len(pn['cells']) == len(tn['cells'])
            if ok and pn['cells']: ok = pn['tni'] == tn['tni']
            if ok:
                for a, b in zip(pn['cells'], tn['cells']):
                    if not all(close(x, y) for x, y in zip(a, b)): ok = False
            if not ok:
                nd_bad += 1
                if nd_bad <= 3: bad.append(f'node {i}: py={pn} ts={tn}')
        if nd_bad: bad.append(f'{nd_bad} nodes differ')
        p2f_t = [[tuple(e) for e in es] for es in T['p2f']]
        if P['p2f'] != p2f_t: bad.append('p2f adjacency differs')
        nctx = 0
        for cx in d['ctxs']:
            tok, order = seq(P, cx['seed'], cx['L'], cx['w'], cx['target'], set(cx['drop']), cx['rngSeed'])
            vis = [list(v) for v in order]
            if vis != cx['visited']: bad.append(f"visit order differs {cx['task']} {cx['split']} seed {cx['seed']} L{cx['L']} w{cx['w']}"); continue
            for p, (a, b) in enumerate(zip(tok, cx['tok'])):
                if not all(close(x, y) for x, y in zip(a, b)):
                    bad.append(f"token {p} differs {cx['task']} seed {cx['seed']}: py={a} ts={b}"); break
            mk = masks(tok)
            for l in ('col', 'feat', 'nbr', 'full'):
                if digest(mk[l]) != cx['masks'][l]: bad.append(f"mask {l} differs {cx['task']} seed {cx['seed']}")
            if sum(t[2] for t in tok) != 1: bad.append(f"{cx['task']} seed {cx['seed']}: {sum(t[2] for t in tok)} masked")
            nctx += 1
        print(f'{name}: {len(P["nodes"])} nodes, {nctx}/{len(d["ctxs"])} contexts compared, {len(bad)} problems')
        for b in bad[:15]: print('   ', b)
        total_bad += len(bad)
        total_ctx += nctx
        total_listed += len(d['ctxs'])
    print('TOTAL PROBLEMS', total_bad)
    if '--json' in sys.argv:
        out = sys.argv[sys.argv.index('--json') + 1]
        json.dump({"contexts": total_ctx, "listed": total_listed, "mismatches": total_bad}, open(out, 'w'))
    return total_bad


if __name__ == '__main__':
    sys.exit(1 if main() else 0)
