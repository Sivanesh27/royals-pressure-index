"""Step 2: win-probability model -> leverage (Pressure Index); run-expectancy model -> ball values."""
import os
import pandas as pd, numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier as HGC, HistGradientBoostingRegressor as HGR
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

b = pd.read_pickle(os.environ.get("OUT_DIR", "pipeline/out/") + "state.pkl")
b["t20i"] = (b.league == "T20I").astype(int)
b["wk_next_if_wkt"] = np.where(b.is_wicket, 1, 0)

# ---------------- run expectancy (resource table) from first innings
def feat_re(balls_left, wk, t20i):
    return np.c_[balls_left, wk, t20i]
inn1 = b[b.innings == 1].copy()
tot1 = inn1.groupby("match_id").total_runs.sum().rename("tot")
inn1 = inn1.merge(tot1, on="match_id")
inn1["rest"] = inn1.tot - inn1.runs_before
inn1 = inn1[inn1.max_balls == 120]
re_model = HGR(max_depth=4, learning_rate=0.06, max_iter=250, min_samples_leaf=300,
               monotonic_cst=[1, -1, 0], random_state=1)
re_model.fit(feat_re(inn1.balls_left, inn1.wkts_before, inn1.t20i), inn1.rest)
def RE(bl, wk, t20i):
    bl = np.asarray(bl, float); wk = np.asarray(wk, float)
    out = re_model.predict(feat_re(bl, np.minimum(wk, 10), t20i))
    return np.where((bl <= 0) | (wk >= 10), 0.0, out)

# ---------------- win probability models (innings-specific), grouped CV
def f1(d):  # first innings
    return np.c_[d.legal_before, d.wkts_before, d.runs_before, d.runs_before / d.legal_before.clip(lower=1) * 6, d.t20i, d.max_balls]
def f2(d):  # chase
    need = d.runs_needed; bl = d.balls_left
    return np.c_[bl, d.wkts_before, need, need / bl.clip(lower=1) * 6, d.runs_before / d.legal_before.clip(lower=1) * 6, d.t20i, d.target]
mk = lambda cst: HGC(max_depth=4, learning_rate=0.06, max_iter=250, min_samples_leaf=250, l2_regularization=1.0,
                     monotonic_cst=cst, random_state=1)
CST = {1: None, 2: [1, -1, -1, -1, 0, 0, 0]}
FE = {1: f1, 2: f2}

# next-ball outcome distribution by innings x phase x wicket bucket (legal balls)
leg = b[b.legal].copy()
leg["wb"] = pd.cut(leg.wkts_before, [-1, 2, 5, 10], labels=False)
leg["ph"] = pd.cut(leg.over_idx, [-1, 5, 14, 19], labels=False)
def outcome(d):
    r = d.runs.clip(upper=6).where(~d.runs.isin([5]), 4)
    return np.where(d.is_wicket, 7, r)
leg["oc"] = outcome(leg)
OUT = [(0,0),(1,0),(2,0),(3,0),(4,0),(6,0),(0,1)]   # (runs, wicket)
codes = [0,1,2,3,4,6,7]
dist = (leg.groupby(["innings","ph","wb"]).oc.value_counts(normalize=True).unstack(fill_value=0)
        .reindex(columns=codes, fill_value=0))

b["wp"] = np.nan; b["lev"] = np.nan
b["wb"] = pd.cut(b.wkts_before, [-1, 2, 5, 10], labels=False)
b["ph"] = pd.cut(b.over_idx, [-1, 5, 14, 19], labels=False)
b["fold_pred_ok"] = False
rng_idx = {}
def terminal_chase(need, bl, wk):
    # value to chasing side once the chase is over
    return np.where(need <= 0, 1.0, np.where((bl <= 0) | (wk >= 10), np.where(need == 1, 0.5, 0.0), np.nan))

for inn in (1, 2):
    d = b[(b.innings == inn)].copy()
    lab = d[d.bat_won.notna()]
    gkf = GroupKFold(n_splits=5)
    ids = d.match_id.values
    wp = np.full(len(d), np.nan); lev = np.full(len(d), np.nan)
    # need chase model to value end of first innings -> train inn2 first when inn==1
    pass
# train chase models first (needed for first-innings terminal values), keep fold models
folds = {}
mats = b[b.bat_won.notna()].match_id.unique()
rs = np.random.RandomState(7); rs.shuffle(mats)
fold_of = {m: i % 5 for i, m in enumerate(mats)}
b["fold"] = b.match_id.map(fold_of).fillna(-1).astype(int)
models = {}
for k in range(5):
    for inn in (2,):
        tr = b[(b.innings == inn) & b.bat_won.notna() & (b.fold != k) & (b.fold >= 0)]
        mdl = mk(CST[inn]); mdl.fit(FE[inn](tr), tr.bat_won); models[(inn, k)] = mdl

# spread of the final score around run-expectancy, by balls left / wickets
inn1["res2"] = (inn1.rest - re_model.predict(feat_re(inn1.balls_left, inn1.wkts_before, inn1.t20i))) ** 2
sd_model = HGR(max_depth=3, learning_rate=0.06, max_iter=150, min_samples_leaf=400, monotonic_cst=[1, 0, 0], random_state=1)
sd_model.fit(feat_re(inn1.balls_left, inn1.wkts_before, inn1.t20i), inn1.res2)
GH_x, GH_w = np.polynomial.hermite_e.hermegauss(11); GH_w = GH_w / GH_w.sum()
grid = np.arange(20, 321, 5.0)
start_curve = {}
def curve(k, t20i):
    if (k, t20i) not in start_curve:
        g = pd.DataFrame({"balls_left": 120, "wkts_before": 0, "runs_needed": grid, "runs_before": 0, "legal_before": 0,
                          "t20i": t20i, "target": grid})
        start_curve[(k, t20i)] = models[(2, k)].predict_proba(f2(g))[:, 1]
    return start_curve[(k, t20i)]
def wp1(runs, wk, bl, t20i, k):
    runs = np.asarray(runs, float); wk = np.asarray(wk, float); bl = np.asarray(bl, float); t20i = np.asarray(t20i)
    done = (bl <= 0) | (wk >= 10)
    X = feat_re(np.where(done, 0, bl), np.minimum(wk, 10), t20i)
    mu = runs + np.where(done, 0, re_model.predict(X))
    sd = np.where(done, 0.0, np.sqrt(np.clip(sd_model.predict(X), 1, None)))
    out = np.zeros(len(runs))
    for ti in (0, 1):
        s_ = t20i == ti
        if not s_.any(): continue
        T = mu[s_, None] + sd[s_, None] * GH_x[None, :] + 1       # chase target
        out[s_] = 1 - (np.interp(T, grid, curve(k, ti)) * GH_w[None, :]).sum(1)
    return out

def p(inn, k, d):
    if inn == 1:
        return wp1(d.runs_before.values, d.wkts_before.values, d.balls_left.values, d.t20i.values, k)
    return models[(2, k)].predict_proba(f2(d))[:, 1]

def next_state(d, runs, wk, inn):
    n = d.copy()
    n["runs_before"] = d.runs_before + runs
    n["wkts_before"] = d.wkts_before + wk
    n["legal_before"] = d.legal_before + 1
    n["balls_left"] = (d.balls_left - 1).clip(lower=0)
    if inn == 2:
        n["runs_needed"] = d.runs_needed - runs
    return n

def wp_after(d, inn, k, runs, wk):
    n = next_state(d, runs, wk, inn)
    if inn == 2:
        raw = p(2, k, n)
        term = terminal_chase(n.runs_needed.values, n.balls_left.values, n.wkts_before.values)
        return np.where(np.isnan(term), raw, term)
    return p(1, k, n)

for inn in (1, 2):
    for k in range(5):
        sel = (b.innings == inn) & (b.fold == k) & b.legal
        d = b[sel]
        w0 = p(inn, k, d)
        pr = dist.loc[[(inn, int(a), int(c_)) for a, c_ in zip(d.ph, d.wb)]].values   # n x 7
        lv = np.zeros(len(d))
        for j, (r, wk) in enumerate(OUT):
            lv += pr[:, j] * np.abs(wp_after(d, inn, k, r, wk) - w0)
        b.loc[sel, "wp"] = w0; b.loc[sel, "lev"] = lv
# illegal deliveries inherit state of the delivery that follows (same state)
b = b.sort_values(["match_id", "innings", "innings_ball_num"])
for col in ("wp", "lev"):
    b[col] = b.groupby(["match_id", "innings"])[col].transform(lambda s: s.bfill().ffill())

# ---- diagnostics
v = b[b.bat_won.notna() & b.legal]
for inn in (1, 2):
    x = v[v.innings == inn]
    print("inn", inn, "AUC", round(roc_auc_score(x.bat_won, x.wp), 3), "Brier", round(brier_score_loss(x.bat_won, x.wp), 3),
          "base", round(brier_score_loss(x.bat_won, np.full(len(x), .5)), 3))
# calibration
v = v.assign(bin=pd.cut(v.wp, np.linspace(0, 1, 11)))
print(v.groupby("bin", observed=True).agg(pred=("wp", "mean"), obs=("bat_won", "mean"), n=("wp", "size")).round(3))
b["LI"] = b.lev / b.loc[b.legal, "lev"].mean()
print(b.groupby("innings").LI.describe().round(2))
print(b.groupby(["innings", "phase"]).LI.mean().round(2))

# ---------------- ball values (run expectancy change)
t = b.t20i.values
re0 = RE(b.balls_left, b.wkts_before, t)
bl1 = np.where(b.legal, b.balls_left - 1, b.balls_left)
wk_bat = np.where(b.is_wicket & (b.dismissed_player_id == b.striker_id), b.wkts_before + 1, b.wkts_before)
wk_team = b.wkts_before + b.is_wicket.astype(int)
wk_bowl = b.wkts_before + b.bowler_wkt.astype(int)
b["RE0"] = re0
batruns = np.where(b.extras_type.isin(["byes", "legbyes"]), 0, b.runs)
b["v_bat"] = batruns + RE(bl1, wk_bat, t) - re0
b.loc[b.is_wide, "v_bat"] = np.nan                       # not a ball faced
b["v_bowl"] = -(b.bowler_runs + RE(bl1, wk_bowl, t) - re0)
b["v_team"] = b.total_runs + RE(bl1, wk_team, t) - re0
print("mean ball values (per ball):", b.v_bat.mean().round(4), b.v_bowl.mean().round(4), b.v_team.mean().round(4))
b.drop(columns=["wk_next_if_wkt", "fold_pred_ok"], errors="ignore").to_pickle(os.environ.get("OUT_DIR", "pipeline/out/") + "model.pkl")
