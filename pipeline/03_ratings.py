"""Step 3: Pressure Index, opponent-adjusted ball values, player ratings, robustness checks."""
import os
import pandas as pd, numpy as np, json, scipy.sparse as sp
from sklearn.linear_model import Ridge
OUT = os.environ.get("OUT_DIR", "pipeline/out/")
b = pd.read_pickle(OUT + "model.pkl").sort_values(["match_id", "innings", "innings_ball_num"]).reset_index(drop=True)
HP = 75                                    # high-pressure = top quartile of Pressure Index
b["stage"] = (b.legal_before / b.max_balls).clip(0, 1)
b["PIraw"] = b.LI * (0.5 + b.stage)
b["PI"] = b.PIraw.rank(pct=True) * 100     # computed on all deliveries (illegal ones carry their state's value)
b["hp"] = b.PI >= HP
# win-probability change of each ball (batting side), last ball of an innings uses the final outcome
g = b.groupby(["match_id", "innings"])
nxt = g.wp.shift(-1)
last_val = np.where(b.innings == 2, b.bat_won, np.nan)
b["dwp"] = (nxt.fillna(pd.Series(last_val, index=b.index)) - b.wp)
b["dwp_bat"] = b.dwp.where(~b.is_wide)
b["dwp_bowl"] = -b.dwp
# ---------- opponent adjustment: ridge on ordinary (non-pressure) deliveries only
ids = pd.Index(sorted(set(b.striker_id) | set(b.bowler_id)))
ix = {p: i for i, p in enumerate(ids)}; N = len(ids)
o = b[(~b.hp) & b.v_bat.notna()]
rows = np.arange(len(o))
X = sp.csr_matrix((np.r_[np.ones(len(o)), -np.ones(len(o))],
                   (np.r_[rows, rows], np.r_[o.striker_id.map(ix), N + o.bowler_id.map(ix)])), shape=(len(o), 2 * N))
rg = Ridge(alpha=250, fit_intercept=True, solver="sparse_cg").fit(X, o.v_bat.values)
bat_eff = pd.Series(rg.coef_[:N], index=ids); bowl_eff = pd.Series(rg.coef_[N:], index=ids)
b["adj_bat"] = b.v_bat + b.bowler_id.map(bowl_eff).values
b["adj_bowl"] = b.v_bowl + b.striker_id.map(bat_eff).values
b.loc[b.is_wide, "adj_bat"] = np.nan

names = pd.concat([b[["striker_id", "striker_name"]].set_axis(["id", "name"], axis=1),
                   b[["bowler_id", "bowler_name"]].set_axis(["id", "name"], axis=1)]).drop_duplicates("id").set_index("id").name
b.to_pickle(OUT + "final_balls.pkl")
pd.DataFrame({"bat_eff": bat_eff, "bowl_eff": bowl_eff, "name": names}).to_pickle(OUT + "effects.pkl")

# ---------- player tables
def build(role):
    pid, val, wpa, bcount = (("striker_id", "adj_bat", "dwp_bat", None) if role == "bat" else ("bowler_id", "adj_bowl", "dwp_bowl", None))
    d = b[b[val].notna()].copy()
    d["innkey"] = d.match_id.astype(str) + "_" + d.innings.astype(str)
    return d, pid, val, wpa

def eb(sum_, n, prior, k):          # empirical-Bayes shrink of a mean toward the prior
    return (sum_ + k * prior) / (n + k)

def summarise(role, min_balls, min_inns, w=None, boot_w=None):
    d, pid, val, wpa = build(role)
    hp, lo = d[d.hp], d[~d.hp]
    if boot_w is not None:
        hp = hp.assign(_w=hp.match_id.map(boot_w).fillna(0)); lo = lo.assign(_w=lo.match_id.map(boot_w).fillna(0))
    else:
        hp = hp.assign(_w=1.0); lo = lo.assign(_w=1.0)
    def agg(x, tag):
        x = x.assign(_v=x[val] * x._w, _p=x[wpa].fillna(0) * x._w)
        r = x.groupby(pid).agg(n=("_w", "sum"), sv=("_v", "sum"), sp=("_p", "sum"))
        return r.add_suffix("_" + tag)
    H, L = agg(hp, "h"), agg(lo, "l")
    t = H.join(L, how="left").fillna(0)
    # innings-level consistency on pressure balls
    hi = hp.assign(_v=hp[val] * hp._w).groupby([pid, "innkey"]).agg(n=("_w", "sum"), v=("_v", "sum")).reset_index()
    hi = hi[hi.n >= 4]
    hi["pos"] = (hi.v > 0).astype(float)
    ci = hi.groupby(pid).agg(inns=("pos", "size"), hits=("pos", "sum"))
    t = t.join(ci, how="left").fillna({"inns": 0, "hits": 0})
    t = t[(t.n_h >= min_balls) & (t.inns >= min_inns)].copy()
    # shrinkage constants by method of moments
    mu_h = hp[val].mean(); mu_l = lo[val].mean(); mu_p = hp[wpa].mean()
    def kconst(x, col):
        s2 = x[col].var(); m = x.groupby(pid)[col].agg(["mean", "size"])
        m = m[m["size"] >= 30]; tau2 = max(m["mean"].var() - (s2 / m["size"]).mean(), 1e-6)
        return float(np.clip(s2 / tau2, 20, 4000))
    kh, kl, kp = kconst(hp, val), kconst(lo, val), kconst(hp.assign(_x=hp[wpa].fillna(0)), "_x")
    t["P"] = eb(t.sv_h, t.n_h, mu_h, kh)                      # pressure value / ball (shrunk)
    t["B"] = eb(t.sv_l, t.n_l, mu_l, kl)                      # usual (ordinary) value / ball (shrunk)
    t["I"] = eb(t.sp_h, t.n_h, mu_p, kp)                      # win-probability added / pressure ball
    t["C"] = (t.hits + 6 * (hi.pos.mean())) / (t.inns + 6)    # share of pressure innings with positive value
    sh, sl = hp[val].var(), lo[val].var()
    draw = t.sv_h / t.n_h - np.where(t.n_l > 0, t.sv_l / t.n_l.clip(lower=1), mu_l)
    se2 = sh / t.n_h + sl / t.n_l.clip(lower=1)
    tau2 = max(draw.var() - se2.mean(), 1e-6)
    t["D"] = (tau2 / (tau2 + se2)) * (draw - draw.mean())          # shrunk "pressure minus usual" gap
    t["D_raw"] = draw
    return t, dict(kh=kh, kl=kl, kp=kp, mu_h=mu_h, mu_l=mu_l)

W = dict(P=.45, I=.20, C=.20, D=.15)
def score(t, w=W):
    z = lambda s: (s - s.mean()) / s.std()
    return sum(w[k] * z(t[k]) for k in w)

if __name__ == "__main__":
    pd.set_option("display.width", 230)
    res = {}
    for role, mb, mi in (("bat", 100, 10), ("bowl", 72, 10)):
        t, consts = summarise(role, mb, mi); t["score"] = score(t); t["name"] = names.reindex(t.index)
        res[role] = (t, consts)
        print(role, len(t), consts)
        print(t.sort_values("score", ascending=False).head(15)[["name", "n_h", "inns", "P", "B", "D", "I", "C", "score"]].round(3))
    pd.to_pickle(res, OUT + "res_base.pkl")
