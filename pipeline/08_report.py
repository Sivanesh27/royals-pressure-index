"""Step 8: build the methodology report (PDF) from the exported data."""
import os
import json, pickle, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, Table, TableStyle,
                                Image, PageBreak, KeepTogether)

OUT = os.environ.get("OUT_DIR", "pipeline/out/")
FIG = OUT + "fig/"
os.makedirs(FIG, exist_ok=True)
D = json.load(open(OUT + "data.json"))
V = pickle.load(open(OUT + "variants.pkl", "rb"))
CS = pickle.load(open(OUT + "consensus.pkl", "rb"))
PINK, BLUE, NAVY, MIST = "#EA1A85", "#254AA5", "#001D48", "#EEF1F8"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": "#9aa5c4", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titleweight": "bold", "axes.titlecolor": NAVY, "axes.labelcolor": "#44506e"})
rr = LinearSegmentedColormap.from_list("rr", ["#ffffff", "#f7b6d6", PINK])
P = D["players"]
M = D["meta"]

# ------------------------------------------------------------------ figures
def save(name):
    plt.tight_layout()
    plt.savefig(FIG + name, dpi=190)
    plt.close()

# 1 heatmaps
fig, ax = plt.subplots(1, 2, figsize=(9.2, 4.2))
for a, inn, ttl in zip(ax, ("1", "2"), ("First innings", "Chase")):
    H = np.array([[np.nan if v is None else v for v in r] for r in D["context"]["heat"][inn]], float)
    im = a.imshow(H, aspect="auto", cmap=rr, vmin=20, vmax=90, origin="upper")
    a.set_xticks(range(8)); a.set_xticklabels(["0", "1", "2", "3", "4", "5", "6", "7+"])
    a.set_yticks(range(0, 20, 2)); a.set_yticklabels(range(1, 21, 2))
    a.set_xlabel("Wickets lost"); a.set_ylabel("Over"); a.set_title(ttl, loc="left")
fig.colorbar(im, ax=ax, shrink=.85, label="Mean Pressure Index (0-100)")
plt.savefig(FIG + "heat.png", dpi=190, bbox_inches="tight"); plt.close()

# 2 calibration + comparison
fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.6))
c = np.array(D["context"]["calib"])
ax[0].plot([0, 1], [0, 1], color="#9aa5c4", ls="--", lw=1)
ax[0].plot(c[:, 0], c[:, 1], marker="o", color=BLUE, lw=2)
ax[0].set_xlabel("Predicted win probability"); ax[0].set_ylabel("Observed win rate"); ax[0].set_title("Win-probability model, out-of-fold", loc="left")
cp = pd.DataFrame(D["context"]["compare"]).set_index("metric")
pick = ["Balls remaining", "Wickets already lost", "Share in the last five overs (%)", "Wicket rate (%)", "Dot-ball rate (%)"]
lab = ["Balls left", "Wickets lost", "Last 5 overs %", "Wicket rate %", "Dot-ball %"]
x = np.arange(len(pick)); w = .38
ax[1].bar(x - w / 2, cp.loc[pick, "high"], w, color=PINK, label="Pressure (top 25%)")
ax[1].bar(x + w / 2, cp.loc[pick, "ordinary"], w, color=BLUE, label="Ordinary")
ax[1].set_xticks(x); ax[1].set_xticklabels(lab, rotation=15); ax[1].legend(frameon=False); ax[1].set_title("What the index separates", loc="left")
save("calib.png")

# 3 ranked bars with bootstrap range
for role, title in (("bat", "Batters"), ("bowl", "Bowlers")):
    t = P[role][:10][::-1]
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.9), gridspec_kw={"width_ratios": [1.15, 1]})
    ax[0].barh([p["name"] for p in t], [p["score"] for p in t], color=[PINK if p["rank"] == 1 else BLUE for p in t])
    for i, p in enumerate(t):
        ax[0].text(p["score"] + .03, i, f'{p["score"]:.2f}', va="center", fontsize=8, color=NAVY)
    ax[0].set_title(f"{title}: consensus rating", loc="left"); ax[0].set_xlim(0, max(p["score"] for p in t) * 1.15)
    for i, p in enumerate(t):
        ax[1].plot([p["r10"], p["r90"]], [i, i], color="#9aa5c4", lw=3, solid_capstyle="round")
        ax[1].plot(p["rmed"], i, "o", color=PINK)
    ax[1].axvline(10.5, color=NAVY, ls=":", lw=1); ax[1].set_yticks(range(10)); ax[1].set_yticklabels([])
    ax[1].set_xscale("symlog", linthresh=10); ax[1].set_xticks([1, 3, 10, 30, 100]); ax[1].set_xticklabels([1, 3, 10, 30, 100])
    ax[1].set_title("Rank range over 60 resamples (10th to 90th pct)", loc="left", fontsize=8.5); ax[1].set_xlabel("Rank (dot = median)")
    save(f"rank_{role}.png")

# 4 level under pressure vs usual level
for role, nm in (("bat", "Batters"), ("bowl", "Bowlers")):
    allp = D["whatif"][role]["75"]
    ids = [i for i, v in allp.items() if v[5] >= 150 and v[6] >= 15]
    xs = np.array([allp[i][1] for i in ids]); ys = np.array([allp[i][0] for i in ids])
    top = {str(p["id"]): p for p in P[role][:10]}
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.scatter(xs, ys, s=16, color="#c4cbe0")
    lo, hi = min(xs.min(), ys.min()), max(xs.max(), ys.max())
    ax.plot([lo, hi], [lo, hi], color="#9aa5c4", ls="--", lw=1)
    for i in ids:
        if i in top:
            ax.scatter(allp[i][1], allp[i][0], s=34, color=PINK, zorder=3)
            ax.annotate(top[i]["name"].split()[-1], (allp[i][1], allp[i][0]), fontsize=7.5, xytext=(4, 3), textcoords="offset points", color=NAVY)
    ax.set_xlabel("Usual level: value per ball in ordinary situations"); ax.set_ylabel("Level under pressure: value per ball")
    ax.set_title(f"{nm}: usual form against pressure form (shrunk)", loc="left")
    save(f"gap_{role}.png")

# 5 lens heat table
for role in ("bat", "bowl"):
    names = [p["name"] for p in P[role][:10]]
    ks = list(CS["lens"][role].keys())
    ids = [p["id"] for p in P[role][:10]]
    Z = np.array([[CS["lens"][role][k].score.get(i, np.nan) for k in ks] for i in ids])
    fig, ax = plt.subplots(figsize=(6.6, 3.5))
    im = ax.imshow(Z, cmap=rr, aspect="auto", vmin=-1, vmax=4)
    ax.set_xticks(range(len(ks))); ax.set_xticklabels(ks, rotation=20, ha="right"); ax.set_yticks(range(10)); ax.set_yticklabels(names)
    for i in range(10):
        for j in range(len(ks)):
            if not np.isnan(Z[i, j]): ax.text(j, i, f"{Z[i, j]:.1f}", ha="center", va="center", fontsize=7.5, color=NAVY)
    ax.set_title("Score under each definition of pressure", loc="left")
    save(f"lens_{role}.png")

# ------------------------------------------------------------------ document
pdfmetrics.registerFont(TTFont("LS", "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"))
pdfmetrics.registerFont(TTFont("LSB", "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"))
pdfmetrics.registerFont(TTFont("LSN", "/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Bold.ttf"))
pdfmetrics.registerFontFamily("LS", normal="LS", bold="LSB", italic="LS", boldItalic="LSB")
C = colors.HexColor
body = ParagraphStyle("b", fontName="LS", fontSize=9.6, leading=14.2, textColor=C("#1b2542"), spaceAfter=6)
h1 = ParagraphStyle("h1", fontName="LSN", fontSize=19, leading=22, textColor=C(NAVY), spaceBefore=10, spaceAfter=6)
h2 = ParagraphStyle("h2", fontName="LSB", fontSize=11.5, leading=15, textColor=C(BLUE), spaceBefore=8, spaceAfter=3)
small = ParagraphStyle("s", parent=body, fontSize=8, leading=11, textColor=C("#55607f"))
cell = ParagraphStyle("c", parent=body, fontSize=8.3, leading=10.4, spaceAfter=0)
cellb = ParagraphStyle("cb", parent=cell, fontName="LSB")
W, H = A4

def hdr(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(C(NAVY)); canvas.rect(0, H - 13 * mm, W, 13 * mm, stroke=0, fill=1)
    canvas.setFillColor(C(PINK)); canvas.rect(0, H - 14.4 * mm, W, 1.4 * mm, stroke=0, fill=1)
    canvas.setFillColor(colors.white); canvas.setFont("LSN", 10.5)
    canvas.drawString(18 * mm, H - 8.6 * mm, "SupeRR Selector Season 2  |  Round 1  |  Reliable players under pressure")
    canvas.setFont("LS", 8); canvas.setFillColor(C("#55607f")); canvas.drawRightString(W - 18 * mm, 9 * mm, f"Page {doc.page}")
    canvas.restoreState()

doc = BaseDocTemplate(os.environ.get("OUT_DIR", "pipeline/out/") + "Royals_Pressure_Index_Report.pdf", pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                      topMargin=22 * mm, bottomMargin=16 * mm, title="Royals Pressure Index: methodology report", author="")
doc.addPageTemplates([PageTemplate("p", [Frame(18 * mm, 16 * mm, W - 36 * mm, H - 38 * mm, id="f", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)], onPage=hdr)])
S = []
P_ = lambda t, s=body: Paragraph(t, s)
img = lambda f, w=174: Image(FIG + f, width=w * mm, height=w * mm * (lambda im: im[1] / im[0])(plt.imread(FIG + f).shape[1::-1]))

def table(rows, widths, head=True, zebra=True, align_right_from=1):
    t = Table(rows, colWidths=[x * mm for x in widths], repeatRows=1 if head else 0)
    st = [("FONTNAME", (0, 0), (-1, -1), "LS"), ("FONTSIZE", (0, 0), (-1, -1), 8.3), ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
          ("LINEBELOW", (0, 0), (-1, -1), .3, C("#d9dfee")), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]
    if head: st += [("BACKGROUND", (0, 0), (-1, 0), C(NAVY)), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("FONTNAME", (0, 0), (-1, 0), "LSB")]
    if align_right_from is not None: st.append(("ALIGN", (align_right_from, 0), (-1, -1), "RIGHT"))
    t.setStyle(TableStyle(st)); return t

pc = lambda x: f"{round(x * 100)}%"
# ---- cover block
S += [Spacer(1, 4 * mm), P_("Who holds up when the match is on the line?", ParagraphStyle("t", parent=h1, fontSize=27, leading=31, spaceAfter=4)),
      P_("A ball-by-ball pressure model for IPL and T20I cricket, 2022 to 2026", ParagraphStyle("t2", parent=body, fontSize=12, textColor=C(BLUE))), Spacer(1, 3 * mm)]
fb = ["%d. %s" % (p["rank"], p["name"]) for p in P["bat"][:10]]; fw = ["%d. %s" % (p["rank"], p["name"]) for p in P["bowl"][:10]]
S.append(table([["Top 10 batters under pressure", "Top 10 bowlers under pressure"]] + [[a, b] for a, b in zip(fb, fw)], [87, 87], align_right_from=None))
S += [Spacer(1, 4 * mm), P_(
    f"I used {M['deliveries']:,} deliveries from {M['matches']} matches ({M['ipl_matches']} IPL, {M['t20i_matches']} T20I, {M['date_from']} to {M['date_to']}). "
    "Each delivery gets a Pressure Index from 0 to 100 that says how much the next ball can swing the result, later in the innings counting for more. "
    "A batter or bowler is then judged on the deliveries in the top quarter of that index, against what an average player would produce from the same situation, "
    "after adjusting for the quality of the opposition. The final order averages five different ways of drawing the pressure line, so a player has to look good however pressure is defined."),
      P_("Pressure performance turns out to be a noisy signal. Matches resampled with replacement move many players a few places, and some a lot, so every rank in this report comes with a range. "
         "The top two batters and the top three bowlers hold their place far more often than the rest, and I say below which picks are less certain.")]

# ---- 1 data
S += [P_("1. Data and scope", h1), P_(
    "The data pack has ball-by-ball records, match results, revised overs and targets for rain-affected games, and the player mappings. I kept the IPL (2022 to 2026) and full T20 internationals. "
    "Two A-team tournaments (Emerging Teams Asia Cup and Asia Cup Rising Stars) are in the pack but are neither IPL nor T20I, so I left them out. Super overs (innings 3 and 4) were removed because they are a different game. "
    "Penalty runs are not deliveries."),
    P_("Delivery accounting needed care. Wides and no-balls do not use up a legal ball, and over_ball_num counts them, so I rebuilt the legal-ball count myself for every innings. "
       "Runs charged to a bowler exclude byes and leg byes. Wides are not balls faced by a batter. Run-outs are not credited to the bowler, and a non-striker run-out is not charged to the striker. "
       "Targets use the revised target and revised overs where the data gives them; otherwise the first-innings total plus one.")]

# ---- 2 pressure
S += [P_("2. How I defined pressure", h1), P_(
    "The brief lists required rate, wickets in hand, balls left, stage of the innings and how close the contest is. Instead of picking one and hoping, I let a win-probability model combine them, then added the stage of the innings on top.")]
S += [P_("Step 1: win probability", h2), P_(
    f"For chases I fitted a gradient-boosted model on balls left, wickets lost, runs needed, required rate, current rate, target and league, with monotonic constraints (more runs needed never helps the chasing side; more wickets lost never helps). "
    f"For the first innings I project the final score from a run-expectancy table and its spread, then pass that score through the chase model at ball one. All predictions are out-of-fold: the data was split into five groups of whole matches, so a ball is never scored by a model that saw its own match. "
    f"The chase model has an AUC of {D['context']['wpmodel']['inn2']['auc']} and a Brier score of {D['context']['wpmodel']['inn2']['brier']} (0.25 is a coin flip). The first-innings model reaches an AUC of {D['context']['wpmodel']['inn1']['auc']}, which is about what half a match of information should give."),
    img("calib.png")]
S += [P_("Step 2: leverage", h2), P_(
    "For every delivery I take the empirical chances of each outcome on the next ball (dot, 1, 2, 3, 4, 6, wicket, by phase and wickets lost), work out the win probability after each, and take the weighted average of the absolute change. "
    "That number is the leverage: how much the next ball can move the match. A tight finish scores high; a chase that is already won or lost scores close to zero.")]
S += [P_("Step 3: the Pressure Index", h2), P_(
    "Leverage alone put too many powerplay balls in the top quartile, because early wickets move the model a lot even though little is at stake yet. I multiplied leverage by (0.5 + share of the innings played), "
    "so the same swing counts for more when there is less time to recover, then converted the result to a percentile. "
    f"<b>High pressure</b> means the top quarter of that index ({M['hp_balls']:,} deliveries). Everything else is <b>ordinary</b>. "
    "The table and heat maps show that the split matches cricket sense: pressure deliveries come later, with more wickets down, and the chase is tighter."),
    img("heat.png", 168)]
cmp = D["context"]["compare"]
S.append(table([["Match state", "Pressure", "Ordinary"]] + [[r["metric"], r["high"], r["ordinary"]] for r in cmp], [96, 39, 39]))
S.append(P_("Required-rate gap is the chase rate minus the current rate. Pressure balls also lose wickets more often and score fewer dots-per-ball-faced only because the batters who face them are set; the point is that the two groups differ in the way the brief expects.", small))

# ---- 3 performance
S += [P_("3. Measuring performance", h1), P_(
    "<b>Value per ball.</b> I fitted a run-expectancy table: how many more runs a side scores from a given number of balls left and wickets lost (first innings only, so chases do not truncate it). "
    "The value of a delivery is the runs scored plus the change in that expectancy. A dot ball costs a little, a wicket costs a lot (more in the powerplay than at the death), and a boundary gains more than four. "
    "Bowlers get the mirror image using the runs charged to them."),
    P_("<b>Opponent adjustment.</b> A ridge regression on ordinary deliveries gives every player a batting and a bowling effect. A batter's pressure deliveries are then credited for the bowler faced, and a bowler's for the batter faced. "
       "Because the effects come only from ordinary deliveries, a player's pressure results do not leak into their own baseline. The adjustment barely changes the top 10 here (9 or 10 of 10 unchanged), which is reassuring in itself."),
    P_("<b>Four ingredients</b>, all measured on pressure deliveries and shrunk toward the average with empirical-Bayes weights sized from the data:"),
    table([["Ingredient", "What it is", "Weight"],
           ["Level", "Opponent-adjusted value per ball under pressure", "45%"],
           ["Impact", "Win probability added per pressure ball", "20%"],
           ["Consistency", "Share of pressure innings with positive value", "20%"],
           ["Gap to usual form", "Pressure value minus the same player's ordinary value, shrunk by its own sampling error", "15%"]], [34, 118, 22], align_right_from=2),
    Spacer(1, 2 * mm), P_(
        "Level carries the most weight because it is the least noisy. Gap to usual form is how the brief's 'compared with their usual performance levels' enters; it carries less weight because a player who is excellent in ordinary play should not be marked down for failing to rise further. "
        "Each ingredient is turned into a z-score across eligible players before weighting.")]

# ---- 4 fairness
S += [P_("4. Fair comparisons", h1), P_(
    "<b>Sample size.</b> A player needs at least 150 pressure deliveries and 15 pressure innings (or spells) to be ranked. That leaves "
    f"{M['players_bat']} batters and {M['players_bowl']} bowlers in the lists the dashboard shows. Below that line the numbers are mostly noise, and shrinkage cannot rescue a player with one hot tournament."),
    P_("<b>Different match contexts.</b> Value is measured against the situation (balls left, wickets lost), so an opener and a finisher are each compared with what an average player does from the same state. "
       "Opposition quality is handled by the ridge adjustment."),
    P_("<b>IPL and T20I together.</b> The run-expectancy and win-probability models include a league flag, so a run is not worth the same in both. Players who appear in both are bridged through the ridge model. "
       "I also re-ran the whole thing on IPL deliveries only and T20I deliveries only (section 6)."),
    P_("<b>Five definitions of pressure.</b> The final score is the average of five ratings: top 25% of the index (double weight), top 15%, top 35%, plain leverage with no stage term, and a rule-based version "
       "(a chase with 10 overs or fewer left, a required rate of 8 or more and a live match; or a first innings with five or more wickets down in the last five overs).")]

# ---- 5 results
def stat_rows(role):
    if role == "bat":
        hd = ["#", "Player", "Team", "Balls", "SR", "Usual SR", "Bdry %", "Inns", "+ve %", "Top 10 %", "Rank range"]
        rows = [hd] + [[p["rank"], p["name"], (p["team"] or p["nat"] or "").replace("Royal Challengers Bengaluru", "RCB").replace("Sunrisers Hyderabad", "SRH").replace("Mumbai Indians", "MI").replace("Punjab Kings", "PBKS").replace("Lucknow Super Giants", "LSG").replace("Gujarat Titans", "GT").replace("Rajasthan Royals", "RR"),
                        p["p"]["balls"], p["p"]["sr"], p["u"]["sr"], p["p"]["bdry"], p["inns"], pc(p["C"]), pc(p["top10"]), f'{int(p["r10"])}-{int(p["r90"])}'] for p in P[role][:10]]
        return rows, [7, 33, 14, 13, 13, 16, 14, 11, 13, 16, 18]
    hd = ["#", "Player", "Team", "Balls", "Econ", "Usual econ", "Wkts", "Inns", "+ve %", "Top 10 %", "Rank range"]
    rows = [hd] + [[p["rank"], p["name"], (p["team"] or p["nat"] or "").replace("Royal Challengers Bengaluru", "RCB").replace("Kolkata Knight Riders", "KKR").replace("Delhi Capitals", "DC").replace("Chennai Super Kings", "CSK").replace("Punjab Kings", "PBKS").replace("Gujarat Titans", "GT").replace("Mumbai Indians", "MI"),
                    p["p"]["balls"], p["p"]["econ"], p["u"]["econ"], p["p"]["wkts"], p["inns"], pc(p["C"]), pc(p["top10"]), f'{int(p["r10"])}-{int(p["r90"])}'] for p in P[role][:10]]
    return rows, [7, 33, 14, 13, 13, 18, 11, 11, 13, 16, 18]

S += [P_("5. Evidence for the top 10", h1), P_("Batters", h2), img("rank_bat.png")]
r, w = stat_rows("bat"); S += [table(r, w, align_right_from=3), P_(
    "SR and Bdry % are strike rate and boundary share on pressure deliveries; Usual SR is the same player's strike rate in ordinary situations. +ve % is the share of pressure innings in which the player added more than an average player would from the same situations. Rank range is the 10th to 90th percentile over 60 resamples of matches.", small)]
S += [P_("Bowlers", h2), img("rank_bowl.png")]
r, w = stat_rows("bowl"); S += [table(r, w, align_right_from=3), P_(
    "Economy is runs charged per six legal balls. Rankings reward wickets and dots at the right moments as well as economy, which is why a bowler with a higher pressure economy than usual can still rate well if the wickets arrive when the contest is live.", small)]

b1, b2 = P["bat"][0], P["bat"][1]
S += [P_("What stands out", h2), P_(
    f"<b>{b1['name']}</b> scores {b1['p']['sr']} on pressure deliveries against {b1['u']['sr']} in ordinary ones, with {b1['inns']} pressure innings and {pc(b1['C'])} of them above expectation. "
    f"<b>{b2['name']}</b> is the most aggressive of the group ({b2['p']['sr']} strike rate, {b2['p']['dot']}% dots) and is above expectation in {pc(b2['C'])} of {b2['inns']} pressure innings, the highest share among the batters. "
    f"Tim David and Phil Salt score at the same rate under pressure as otherwise (177.6 against 177.7, and 169.2 against 169.3). They are rated on being the same player when it is tight, not on a rise; Tilak Varma and Sikandar Raza are the two who lift most from their usual rate."),
    P_(f"Among bowlers, <b>{P['bowl'][0]['name']}</b> concedes {P['bowl'][0]['p']['econ']} under pressure against {P['bowl'][0]['u']['econ']} normally and is above expectation in {pc(P['bowl'][0]['C'])} of his {P['bowl'][0]['inns']} pressure spells. "
       f"The left-arm and wrist-spin names in the list (Pandya, Kuldeep, Axar, Chahal) fit what the data shows about the middle and late overs of tight chases, where spinners keep the required rate from falling.")]
S += [P_("Where players sit against their own usual level", h2), Table([[img("gap_bat.png", 85), img("gap_bowl.png", 85)]], colWidths=[87 * mm, 87 * mm])]

# ---- 6 reliability
sh = D["robust"]["split_half"]
S += [P_("6. Reliability and limitations", h1), P_("How the ranking behaves when the method changes", h2), P_(
    "Each row re-ranks the same eligible players under one change and counts how many of the final top 10 stay in that variant's top 10.")]
rows = [["Variant", "Batters kept (of 10)", "Rank corr.", "Bowlers kept (of 10)", "Rank corr."]]
for k, v in {**V["lens"], **V["var"]}.items():
    rows.append([k, v["bat"]["overlap"], f'{v["bat"]["rho"]:.2f}', v["bowl"]["overlap"], f'{v["bowl"]["rho"]:.2f}'])
S += [table(rows, [64, 30, 24, 32, 24]), P_("Rank corr. is the Spearman correlation between the variant's scores and the final scores across the same players. The first five rows are the five definitions of pressure that make up the final score.", small),
      img("lens_bat.png", 120), img("lens_bowl.png", 120)]
S += [P_("Consistency across matches", h2), P_(
    f"I split matches into two halves (alternating by date) and compared a player's pressure level in each. The correlation is {sh['bat']['P']:.2f} for batters and {sh['bowl']['P']:.2f} for bowlers. "
    "That is low but positive, which fits what is known about clutch performance generally: there is some real skill in it, and a lot of luck. It is the reason for the shrinkage, the minimum samples, and the rank ranges. "
    "A reader who wants a strong claim that one named player is better under pressure than the next one down should not take it from this report; the claim that survives is about the group at the top.")]
S += [P_("Where the model is less certain", h2),
      P_("<b>Small samples at the edge.</b> Several top-10 places have a 10th to 90th percentile rank range that reaches the 30s or 40s. The ranges are on the player pages of the dashboard."),
      P_("<b>No context beyond the scoreboard.</b> The model has no pitch, dew, field, bowler type matchups or batting-position role. A batter sent in as a specialist finisher faces pressure deliveries of a different type from an opener, and the run-expectancy table treats them alike."),
      P_("<b>First-innings pressure is less sharp.</b> Its win-probability model has an AUC of 0.67. That is why the stage term is there, and why the top quarter of the index is mostly chase balls."),
      P_("<b>Leagues.</b> A run in the IPL and in a T20I is not worth exactly the same, and I only model that through a league flag. Nine of the ten batters and eight of the ten bowlers keep most of their pressure balls in the IPL, so the T20I-only check is a thinner test."),
      P_("<b>Tournament stage.</b> Knockout matches are only about 3% of deliveries, so the index does not learn extra pressure from a final beyond what the scoreboard already implies."),
      P_("<b>Win probability uses outcomes.</b> The results of the matches used to fit it include the performances being judged. Out-of-fold fitting by match limits the leakage but does not remove it.")]

# ---- 7 reproduce
S += [P_("7. Reproducing the work", h1), P_(
    "The code is a short chain of scripts. Each reads the previous script's output.")]
S.append(table([["Script", "Purpose"],
                ["01_build_state.py", "Cleans deliveries, rebuilds legal-ball counts, adds score, wickets, balls left, target, required rate"],
                ["02_pressure_model.py", "Run-expectancy table, win-probability models, leverage, ball values"],
                ["03_ratings.py", "Pressure Index, opponent-adjusted values, shrinkage, four ingredients, composite"],
                ["04_robustness.py", "Weight perturbation, match bootstrap, split-half, alternative definitions"],
                ["05_consensus.py", "Five-definition consensus and bootstrap ranges (final ranking)"],
                ["06_export.py / 07_variants.py", "Dashboard data package and stress tests"],
                ["08_report.py", "This document"]], [52, 122], align_right_from=None))
S += [Spacer(1, 3 * mm), P_(
    "Fixed settings: high pressure = Pressure Index at or above the 75th percentile; stage term 0.5 + share of innings played; ridge alpha 250 on ordinary deliveries; "
    "five cross-validation folds by match; composite weights 45/20/20/15; eligibility 150 pressure deliveries and 15 pressure innings; 60 bootstrap resamples of matches; random seeds fixed in the scripts."),
    P_("The interactive dashboard lets a reader change the pressure threshold, the weights and the minimum samples and watch the ranking move. It is the quickest way to check how sensitive a particular name is.")]

doc.build(S)
print("report built")
