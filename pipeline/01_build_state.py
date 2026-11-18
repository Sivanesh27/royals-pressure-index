"""Step 1: clean ball-by-ball data and attach match state to every delivery."""
import os
import pandas as pd, numpy as np, sys
D = os.environ.get("DATA_DIR", "raw_data/Data Pack/")
b = pd.read_csv(D+"ball_by_ball.csv")
m = pd.read_csv(D+"matches.csv")
c = pd.read_csv(D+"competitions.csv")

m = m.merge(c, on="comp_id", how="left")
# scope: IPL + full T20I only (Emerging/Rising Stars A-team tournaments dropped)
keep = m[(m.comp_name.str.startswith("IPL")) | (m.match_format == "T20I")]
b = b[b.match_id.isin(keep.match_id)].copy()
# regular innings only; super overs (innings 3/4) are a different game
b = b[b.innings.isin([1, 2])].copy()

# ---- delivery accounting
b["extra_runs"] = b.extras1_runs + b.extras_2_runs
b["total_runs"] = b.runs + b.extra_runs
b["is_wide"] = b.extras_type.eq("wide")
b["is_noball"] = b.extras_type.isin(["Overstep", "other"])
b["legal"] = ~(b.is_wide | b.is_noball) & ~b.extras_type.eq("Penalty")
b = b[~b.extras_type.eq("Penalty")].copy()
# runs charged to bowler: everything except byes / leg byes / penalty
b["bowler_runs"] = np.where(b.extras_type.isin(["byes", "legbyes"]), 0, b.total_runs)
b["bat_ball_faced"] = ~b.is_wide                      # wides are not balls faced
b["bowler_wkt"] = b.is_wicket & ~b.dismissal_type.isin(["RunOut", "Retire Out", "Obstracting the field", "Obstructing the field"])

b = b.sort_values(["match_id", "innings", "innings_ball_num"]).reset_index(drop=True)
g = b.groupby(["match_id", "innings"], sort=False)
b["legal_cum"] = g.legal.cumsum()
b["legal_before"] = b.legal_cum - b.legal.astype(int)
b["runs_before"] = g.total_runs.cumsum() - b.total_runs
b["wkts_before"] = g.is_wicket.cumsum() - b.is_wicket.astype(int)

# ---- innings length (revised overs for rain-hit games)
mm = m.set_index("match_id")
def max_balls(row):
    mi = mm.loc[row.match_id]
    ov = mi.first_inn_revised_overs if row.innings == 1 else mi.second_inn_revised_overs
    return 120 if pd.isna(ov) else int(round(ov * 6))
keys = b[["match_id", "innings"]].drop_duplicates()
keys["max_balls"] = keys.apply(max_balls, axis=1)
b = b.merge(keys, on=["match_id", "innings"])
b["balls_left"] = (b.max_balls - b.legal_before).clip(lower=0)

# ---- chase target: revised target where given, else first-innings total + 1
first_tot = b[b.innings == 1].groupby("match_id").total_runs.sum().rename("first_total")
b = b.merge(first_tot, on="match_id", how="left")
rt = mm.revised_target.rename("rev_target")
b = b.merge(rt, left_on="match_id", right_index=True, how="left")
b["target"] = np.where(b.innings == 2, b.rev_target.fillna(b.first_total + 1), np.nan)
b["runs_needed"] = b.target - b.runs_before
b["rrr"] = np.where(b.innings == 2, b.runs_needed / (b.balls_left.clip(lower=1) / 6), np.nan)
b["crr"] = b.runs_before / (b.legal_before.clip(lower=1) / 6)
b["over_idx"] = (b.legal_before // 6).clip(upper=19)          # 0..19
b["phase"] = pd.cut(b.over_idx, [-1, 5, 14, 19], labels=["Powerplay", "Middle", "Death"]).astype(str)

# ---- who won (batting side), via match result
win = mm.win_team_id.rename("win_team")
b = b.merge(win, left_on="match_id", right_index=True, how="left")
b["bat_won"] = (b.batting_team_id == b.win_team).astype(float)
b.loc[b.win_team.isna() | (b.win_team == 0), "bat_won"] = np.nan   # no result / tie
# attach competition info
b = b.merge(m[["match_id", "comp_name", "match_format", "season", "match_date", "stage", "knock_out", "venue_country"]], on="match_id")
b["league"] = np.where(b.match_format == "T20I", "T20I", "IPL")
print(len(b), "deliveries |", b.match_id.nunique(), "matches |", b.bat_won.notna().mean().round(3), "with result")
print(b.groupby("league").match_id.nunique())
# sanity: chase innings first ball target
print(b[(b.innings==2)&(b.legal_before==0)][["target","first_total"]].describe().round(1))
b.to_pickle(os.environ.get("OUT_DIR", "pipeline/out/") + "state.pkl")
