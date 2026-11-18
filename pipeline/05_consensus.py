"""Step 5: final rating = consensus across five definitions of pressure (+ bootstrap uncertainty)."""
import os
import importlib, pandas as pd, numpy as np, pickle, time
m = importlib.import_module("03_ratings"); b, names, OUT = m.b, m.names, m.OUT
lev_pct = b.LI.rank(pct=True) * 100; chase = b.innings == 2
rule = ((chase & (b.balls_left <= 60) & (b.rrr >= 8) & b.wp.between(.15, .85)) | ((~chase) & (b.balls_left <= 30) & (b.wkts_before >= 5)))
LENS = {"PI>=75": (b.PI >= 75, 2.0), "PI>=85": (b.PI >= 85, 1.0), "PI>=65": (b.PI >= 65, 1.0),
        "Leverage>=75": (lev_pct >= 75, 1.0), "Rule-based": (rule, 1.0)}
ELIG = {"bat": (150, 15), "bowl": (150, 15)}      # final eligibility (base lens)
LEN = {"bat": (35, 4), "bowl": (28, 4)}          # lenient floor inside each lens
def lens_tables(boot_w=None):
    out = {}
    for role in ("bat", "bowl"):
        out[role] = {}
        for k, (mask, w) in LENS.items():
            b["hp"] = mask
            t, _ = m.summarise(role, *LEN[role], boot_w=boot_w); t["score"] = m.score(t); out[role][k] = t
    b["hp"] = b.PI >= 75
    return out
def consensus(tabs, role, elig):
    num = pd.Series(0.0, index=elig); den = pd.Series(0.0, index=elig); cnt = pd.Series(0, index=elig)
    for k, (_, w) in LENS.items():
        s = tabs[role][k].score.reindex(elig); ok = s.notna()
        num[ok] += w * s[ok]; den[ok] += w; cnt[ok] += 1
    c = (num / den.replace(0, np.nan)); c[cnt < 4] = np.nan
    return c, cnt
if __name__ == "__main__":
    base = lens_tables(); res = {"lens": base}
    elig = {}
    for role in ("bat", "bowl"):
        e = m.summarise(role, *ELIG[role])[0].index; elig[role] = e
        c, cnt = consensus(base, role, e); res[role] = pd.DataFrame({"name": names.reindex(e), "consensus": c, "lenses": cnt})
        print(role, res[role].sort_values("consensus", ascending=False).head(14).round(2).to_string())
    # bootstrap the whole consensus
    rng = np.random.RandomState(5); mids = b.match_id.unique(); B = 60; acc = {r: {} for r in ("bat", "bowl")}; t0 = time.time()
    for i in range(B):
        draw = pd.Series(rng.choice(mids, len(mids))).value_counts()
        tabs = lens_tables(draw)
        for role in acc:
            c, _ = consensus(tabs, role, elig[role]); rk = c.rank(ascending=False)
            for p, r_ in rk.dropna().items(): acc[role].setdefault(p, []).append(r_)
    res["boot"] = acc; res["elig"] = elig
    pickle.dump(res, open(OUT + "consensus.pkl", "wb")); print("boot s", round(time.time() - t0))
    for role in acc:
        d = res[role].sort_values("consensus", ascending=False).head(14)
        print(role, [(n, round(float(np.mean(np.array(acc[role].get(p, [99])) <= 10)), 2)) for p, n in d.name.items()])
