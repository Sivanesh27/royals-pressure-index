"""Step 6: export everything the dashboard, API and report need."""
import os
import importlib, pandas as pd, numpy as np, pickle, json
m = importlib.import_module("03_ratings"); b, names, OUT = m.b, m.names, m.OUT
cs = pickle.load(open(OUT + "consensus.pkl", "rb")); rb = pickle.load(open(OUT + "robust.pkl", "rb"))
D = os.environ.get("DATA_DIR", "raw_data/Data Pack/")
mp = pd.read_csv(D + "competition_team_player_mapping.csv"); mt = pd.read_csv(D + "matches.csv"); tm = pd.read_csv(D + "teams.csv").drop_duplicates("team_id").set_index("team_id").team_name
comp = pd.read_csv(D + "competitions.csv")
mp = mp.merge(comp[["comp_id", "match_format", "season"]], on="comp_id")
ipl_team = mp[mp.comp_name.str.startswith("IPL")].sort_values("comp_id").groupby("player_id").team_name.last()
nat_team = mp[mp.match_format == "T20I"].sort_values("comp_id").groupby("player_id").team_name.last()
bat_hand = b.groupby("striker_id").batting_type.agg(lambda s: s.mode().iat[0]); bowl_type = b.groupby("bowler_id").bowling_type.agg(lambda s: s.mode().iat[0])
b["hp"] = b.PI >= 75
r2 = lambda x, n=3: None if pd.isna(x) else round(float(x), n)

def desc(role, pid):
    if role == "bat":
        d = b[b.striker_id == pid]; d = d[~d.is_wide]
        h, o = d[d.hp], d[~d.hp]
        f = lambda x: dict(balls=int(len(x)), runs=int(x.runs.sum()), outs=int(((x.is_wicket) & (x.dismissed_player_id == pid)).sum()),
                           sr=r2(x.runs.sum() / max(len(x), 1) * 100, 1), bdry=r2((x.is_four | x.is_six).mean() * 100, 1), dot=r2(((x.runs == 0)).mean() * 100, 1))
    else:
        d = b[b.bowler_id == pid]; d = d[d.legal]
        h, o = d[d.hp], d[~d.hp]
        f = lambda x: dict(balls=int(len(x)), runs=int(x.bowler_runs.sum()), wkts=int(x.bowler_wkt.sum()),
                           econ=r2(x.bowler_runs.sum() / max(len(x), 1) * 6, 2), dot=r2((x.total_runs == 0).mean() * 100, 1),
                           bdry=r2((x.is_four | x.is_six).mean() * 100, 1))
    return f(h), f(o), r2(h.league.eq("IPL").mean() * 100, 0)

def player_rows(role):
    pidc = "striker_id" if role == "bat" else "bowler_id"
    c = cs[role].copy(); c = c[c.consensus.notna()].sort_values("consensus", ascending=False); c["rank"] = np.arange(1, len(c) + 1)
    base = cs["lens"][role]["PI>=75"]
    rows = []
    for pid, r in c.iterrows():
        bt = np.array(cs["boot"][role].get(pid, [999])); h, o, ipl = desc(role, pid)
        t = base.loc[pid] if pid in base.index else None
        rows.append(dict(id=int(pid), name=r["name"], rank=int(r["rank"]), score=r2(r.consensus, 2), team=ipl_team.get(pid), nat=nat_team.get(pid),
                         style=(bat_hand.get(pid) if role == "bat" else bowl_type.get(pid)),
                         p=h, u=o, ipl_pct=ipl, inns=int(t.inns) if t is not None else None,
                         P=r2(t.P, 4) if t is not None else None, B=r2(t.B, 4) if t is not None else None, D=r2(t.D, 4) if t is not None else None,
                         I=r2(t.I * 100, 4) if t is not None else None, C=r2(t.C, 3) if t is not None else None,
                         top10=r2(np.mean(bt <= 10), 2), r10=r2(np.percentile(bt, 10), 0), rmed=r2(np.median(bt), 0), r90=r2(np.percentile(bt, 90), 0),
                         lens={k: r2(cs["lens"][role][k].score.get(pid), 2) for k in cs["lens"][role]}))
    return rows
players = {r: player_rows(r) for r in ("bat", "bowl")}

# what-if tables for several pressure cut-offs
whatif = {"bat": {}, "bowl": {}}
for cut in (60, 65, 70, 75, 80, 85, 90):
    b["hp"] = b.PI >= cut
    for role, (mb, mi) in (("bat", (35, 4)), ("bowl", (28, 4))):
        t, _ = m.summarise(role, mb, mi)
        whatif[role][cut] = {int(p): [r2(x.P, 4), r2(x.B, 4), r2(x.D, 4), r2(x.I * 100, 4), r2(x.C, 3), int(x.n_h), int(x.inns)] for p, x in t.iterrows()}
b["hp"] = b.PI >= 75

# per-innings pressure log for ranked players
logs = {}
for role, pidc, val in (("bat", "striker_id", "adj_bat"), ("bowl", "bowler_id", "adj_bowl")):
    ids = [p["id"] for p in players[role]]
    d = b[b.hp & b[pidc].isin(ids) & b[val].notna()].copy()
    d["opp"] = np.where(role == "bat", d.bowling_team_id, d.batting_team_id) if False else (d.bowling_team_id if role == "bat" else d.batting_team_id)
    if role == "bat":
        d["out"] = (d.is_wicket & (d.dismissed_player_id == d.striker_id)).astype(int); d["rr"] = d.runs; d["bl"] = (~d.is_wide).astype(int)
    else:
        d["out"] = d.bowler_wkt.astype(int); d["rr"] = d.bowler_runs; d["bl"] = d.legal.astype(int)
    g = d.groupby([pidc, "match_id", "innings"]).agg(date=("match_date", "first"), opp=("opp", "first"), comp=("comp_name", "first"), pi=("PI", "mean"),
                                                      balls=("bl", "sum"), runs=("rr", "sum"), out=("out", "sum"), v=(val, "sum")).reset_index()
    for pid, x in g.groupby(pidc):
        logs[f"{role}{int(pid)}"] = [[r.date[:10] if isinstance(r.date, str) else str(r.date)[:10], tm.get(r.opp, "?"), r.comp, round(r.pi, 0), int(r.balls), int(r.runs), int(r.out), round(r.v, 2), int(r.innings)]
                                     for r in x.sort_values("date").itertuples()]

# ---- context evidence
L = b[b.legal].copy(); L["hpb"] = L.PI >= 75
cmp = []
def row(label, f): cmp.append(dict(metric=label, high=r2(f(L[L.hpb]), 2), ordinary=r2(f(L[~L.hpb]), 2)))
row("Balls remaining", lambda x: x.balls_left.mean()); row("Wickets already lost", lambda x: x.wkts_before.mean())
row("Share of deliveries in the chase (%)", lambda x: (x.innings == 2).mean() * 100)
row("Share in the last five overs (%)", lambda x: (x.balls_left <= 30).mean() * 100)
row("Chase: required rate", lambda x: x[x.innings == 2].rrr.mean()); row("Chase: required minus current rate", lambda x: (x[x.innings == 2].rrr - x[x.innings == 2].crr).mean())
row("Win probability distance from 50% (pts)", lambda x: (x.wp - .5).abs().mean() * 100)
row("Runs per ball", lambda x: x.total_runs.mean()); row("Boundary rate (%)", lambda x: (x.is_four | x.is_six).mean() * 100)
row("Wicket rate (%)", lambda x: x.is_wicket.mean() * 100); row("Dot-ball rate (%)", lambda x: (x.total_runs == 0).mean() * 100)
row("Knockout-match share (%)", lambda x: x.knock_out.mean() * 100)
heat = {}
for inn in (1, 2):
    x = L[L.innings == inn]; h = x.groupby(["over_idx", x.wkts_before.clip(upper=7)]).PI.mean().unstack()
    heat[inn] = [[r2(h.at[o, w], 0) if (o in h.index and w in h.columns and not pd.isna(h.at[o, w])) else None for w in range(8)] for o in range(20)]
mix = L.groupby("phase").agg(n=("PI", "size"), mean=("PI", "mean"), hp=("hpb", "mean")); mix = {k: dict(n=int(v.n), mean=r2(v["mean"], 1), hp=r2(v.hp * 100, 1)) for k, v in mix.iterrows()}
byover = L.groupby(["innings", "over_idx"]).PI.mean().unstack(0).round(1); byover = {"first": byover[1].tolist(), "chase": byover[2].tolist()}
v = L[L.bat_won.notna()]; cal = v.assign(bin=pd.cut(v.wp, np.linspace(0, 1, 11))).groupby("bin", observed=True).agg(p=("wp", "mean"), o=("bat_won", "mean"), n=("wp", "size"))
from sklearn.metrics import roc_auc_score, brier_score_loss
wpm = {f"inn{i}": dict(auc=r2(roc_auc_score(v[v.innings == i].bat_won, v[v.innings == i].wp), 3), brier=r2(brier_score_loss(v[v.innings == i].bat_won, v[v.innings == i].wp), 3)) for i in (1, 2)}
context = dict(compare=cmp, heat=heat, mix=mix, byover=byover, calib=[[r2(r.p), r2(r.o), int(r.n)] for r in cal.itertuples()], wpmodel=wpm)

# robustness packet
variants = {k: {r: dict(overlap=int(v[r]["overlap"]), rho=r2(v[r]["spearman"], 2), n=int(v[r]["n"])) for r in v} for k, v in rb["variants"].items()}
final = {r: [p["id"] for p in players[r][:10]] for r in players}
final_names = {r: [p["name"] for p in players[r][:10]] for r in players}
meta = dict(matches=int(b.match_id.nunique()), deliveries=int(len(b)), ipl_matches=int(b[b.league == "IPL"].match_id.nunique()), t20i_matches=int(b[b.league == "T20I"].match_id.nunique()),
            hp_cut=75, weights=m.W, lens={k: w for k, (_, w) in __import__("05_consensus").LENS.items()} if False else None,
            hp_balls=int(L.hpb.sum()), date_from=str(b.match_date.min())[:10], date_to=str(b.match_date.max())[:10],
            players_bat=len(players["bat"]), players_bowl=len(players["bowl"]))
data = dict(meta=meta, final=final, final_names=final_names, players=players, whatif=whatif, logs=logs, context=context,
            robust=dict(variants=variants, split_half=rb["split_half"], weight_top10={r: {int(k): r2(v, 2) for k, v in rb["weight_top10_prob"][r].items() if v > 0} for r in rb["weight_top10_prob"]}))
json.dump(data, open(OUT + "data.json", "w"), separators=(",", ":"))
import os; print("data.json KB:", os.path.getsize(OUT + "data.json") // 1024)
print(final_names)
print(pd.DataFrame(cmp)); print(mix); print(wpm); print(variants)
