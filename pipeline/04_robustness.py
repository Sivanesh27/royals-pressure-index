import os
import importlib, pandas as pd, numpy as np, pickle, time
m = importlib.import_module("03_ratings")
b, names = m.b, m.names
OUT = m.OUT
CFG = {"bat": (100, 10), "bowl": (72, 10)}
def top(role, w=m.W, **kw):
    t, _ = m.summarise(role, *CFG[role], **kw); t["score"] = m.score(t, w); t["name"] = names.reindex(t.index); return t
base = {r: top(r) for r in ("bat", "bowl")}
top10 = {r: list(base[r].sort_values("score", ascending=False).index[:10]) for r in base}
res = {"base_top10": top10}

# ---- 1. alternative definitions of pressure / data scope
hp0 = b.hp.copy(); adjb, adjw = b.adj_bat.copy(), b.adj_bowl.copy()
lev_pct = b.LI.rank(pct=True) * 100
chase = (b.innings == 2)
rule = ((chase & (b.balls_left <= 60) & (b.rrr >= 8) & b.wp.between(.15, .85)) |
        ((~chase) & (b.balls_left <= 30) & (b.wkts_before >= 5)))
variants = {
    "V1 top 15% pressure": (b.PI >= 85),
    "V2 top 35% pressure": (b.PI >= 65),
    "V3 pure leverage (no stage term)": (lev_pct >= 75),
    "V4 rule-based (chase RRR/wp, collapse)": rule,
    "V5 IPL deliveries only": hp0 & (b.league == "IPL"),
    "V6 T20I deliveries only": hp0 & (b.league == "T20I"),
}
var_out = {}
for name, mask in variants.items():
    b["hp"] = mask
    out = {}
    for role in ("bat", "bowl"):
        cfg = CFG[role] if "only" not in name else (60 if role == "bat" else 42, 6)
        t, _ = m.summarise(role, *cfg); t["score"] = m.score(t)
        t["name"] = names.reindex(t.index)
        order = list(t.sort_values("score", ascending=False).index)
        ov = len(set(order[:10]) & set(top10[role]))
        common = base[role].index.intersection(t.index)
        rho = base[role].loc[common, "score"].rank().corr(t.loc[common, "score"].rank())
        out[role] = dict(overlap=ov, spearman=float(rho), top10=order[:10], n=len(t))
    var_out[name] = out
    print(name, {r: (out[r]["overlap"], round(out[r]["spearman"], 2), out[r]["n"]) for r in out}, flush=True)
b["hp"] = hp0
res["variants"] = var_out
# no opponent adjustment
b["adj_bat"] = b.v_bat; b["adj_bowl"] = b.v_bowl
out = {}
for role in ("bat", "bowl"):
    t, _ = m.summarise(role, *CFG[role]); t["score"] = m.score(t)
    order = list(t.sort_values("score", ascending=False).index); common = base[role].index.intersection(t.index)
    out[role] = dict(overlap=len(set(order[:10]) & set(top10[role])), spearman=float(base[role].loc[common, "score"].rank().corr(t.loc[common, "score"].rank())), top10=order[:10], n=len(t))
res["variants"]["V7 no opponent adjustment"] = out
print("V7", {r: (out[r]["overlap"], round(out[r]["spearman"], 2)) for r in out})
b["adj_bat"], b["adj_bowl"] = adjb, adjw

# ---- 2. random weights (Dirichlet around the chosen weights)
rng = np.random.RandomState(11)
keys = list(m.W); w0 = np.array([m.W[k] for k in keys])
wt = {}
for role in ("bat", "bowl"):
    t = base[role]; cnt = pd.Series(0.0, index=t.index)
    for _ in range(1000):
        w = rng.dirichlet(w0 * 25); sc = m.score(t, dict(zip(keys, w)))
        cnt[sc.nlargest(10).index] += 1
    wt[role] = (cnt / 1000)
res["weight_top10_prob"] = wt

# ---- 3. bootstrap over matches
mids = b.match_id.unique(); bt = {r: pd.Series(0.0, index=base[r].index) for r in base}
rank_acc = {r: {p: [] for p in base[r].index} for r in base}
B = 120; t0 = time.time()
for i in range(B):
    draw = pd.Series(rng.choice(mids, len(mids))).value_counts()
    for role in ("bat", "bowl"):
        t, _ = m.summarise(role, 1, 1, boot_w=draw)
        t = t.loc[t.index.intersection(base[role].index)]
        t = t[t.n_h >= 40]
        sc = m.score(t)
        rk = sc.rank(ascending=False)
        for p in sc.nlargest(10).index: bt[role][p] += 1
        for p, r_ in rk.items(): rank_acc[role][p].append(r_)
res["boot_top10_prob"] = {r: bt[r] / B for r in bt}
res["boot_rank"] = {r: {p: (np.percentile(v, 10), np.median(v), np.percentile(v, 90)) for p, v in rank_acc[r].items() if len(v) > 20} for r in rank_acc}
print("bootstrap done", round(time.time() - t0), "s")

# ---- 4. split-half reliability (alternate matches by date order)
md = b.groupby("match_id").match_date.first().sort_values(); half = {mid: i % 2 for i, mid in enumerate(md.index)}
b["half"] = b.match_id.map(half)
sh = {}
for role in ("bat", "bowl"):
    parts = []
    for h in (0, 1):
        keep = b.half == h
        saved = b.hp.copy(); b["hp"] = saved & keep
        # low-ball thresholds in halves
        t, _ = m.summarise(role, 40, 4); t["score"] = m.score(t); parts.append(t); b["hp"] = saved
    c = parts[0].index.intersection(parts[1].index)
    sh[role] = dict(n=len(c), P=float(np.corrcoef(parts[0].loc[c, "P"], parts[1].loc[c, "P"])[0, 1]),
                    C=float(np.corrcoef(parts[0].loc[c, "C"], parts[1].loc[c, "C"])[0, 1]),
                    score=float(np.corrcoef(parts[0].loc[c, "score"], parts[1].loc[c, "score"])[0, 1]))
res["split_half"] = sh; print(sh)
pickle.dump(res, open(OUT + "robust.pkl", "wb"))
for role in base:
    t = base[role].sort_values("score", ascending=False).head(16)
    print(role); print(pd.DataFrame({"name": t.name, "score": t.score.round(2), "w_top10": res["weight_top10_prob"][role].reindex(t.index).round(2),
                                     "boot_top10": res["boot_top10_prob"][role].reindex(t.index).round(2)}))
