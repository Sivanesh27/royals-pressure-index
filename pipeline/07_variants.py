"""Step 7: stress tests of the final top 10 (restricted to the same eligible players)."""
import os
import importlib, pandas as pd, numpy as np, pickle
c5 = importlib.import_module("05_consensus"); m = c5.m; b, names, OUT = m.b, m.names, m.OUT
cs = pickle.load(open(OUT + "consensus.pkl", "rb"))
final = {r: list(cs[r].sort_values("consensus", ascending=False).index[:10]) for r in ("bat", "bowl")}
out = {"final": final, "lens": {}, "var": {}}
for r in final:
    el = cs["elig"][r]
    for k, t in cs["lens"][r].items():
        s = t.score.reindex(el).dropna().sort_values(ascending=False)
        out["lens"].setdefault(k, {})[r] = dict(overlap=len(set(s.index[:10]) & set(final[r])), top10=list(s.index[:10]), rho=float(s.rank().corr(cs[r].consensus.reindex(s.index).rank())))
def run(label, mask=None, noadj=False, split=None):
    sv = (b.hp.copy(), b.adj_bat.copy(), b.adj_bowl.copy()); b["hp"] = (b.PI >= 75) if mask is None else mask
    if noadj: b["adj_bat"], b["adj_bowl"] = b.v_bat, b.v_bowl
    for r in final:
        mb, mi = c5.LEN[r]; t, _ = m.summarise(r, mb, mi); t["score"] = m.score(t)
        s = t.score.reindex(cs["elig"][r]).dropna().sort_values(ascending=False)
        out["var"].setdefault(label, {})[r] = dict(overlap=len(set(s.index[:10]) & set(final[r])), top10=list(s.index[:10]), n=len(s), rho=float(s.rank().corr(cs[r].consensus.reindex(s.index).rank())))
    b["hp"], b["adj_bat"], b["adj_bowl"] = sv
run("IPL pressure balls only", (b.PI >= 75) & (b.league == "IPL"))
run("T20I pressure balls only", (b.PI >= 75) & (b.league == "T20I"))
run("2022-24 only", (b.PI >= 75) & (b.match_date < "2025-01-01"))
run("2025-26 only", (b.PI >= 75) & (b.match_date >= "2025-01-01"))
run("No opponent adjustment", None, True)
pickle.dump(out, open(OUT + "variants.pkl", "wb"))
for k, v in {**out["lens"], **out["var"]}.items(): print(k, {r: (v[r]["overlap"], round(v[r]["rho"], 2)) for r in v})
