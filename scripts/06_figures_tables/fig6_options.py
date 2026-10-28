"""
FIGURE 6 - FOUR DESIGN OPTIONS (A-D) FOR "Implied specific yield versus GEC norms"
==================================================================================
Same data and calculation as Fig6_Sy_match_vs_GEC_norms.png in gec_comparison.py; nothing new is estimated:
  Sy_match (%) = 100 x GEC_R_mm / (1000 x GROSS_C)   per station (district GEC-2024 monsoon rainfall recharge, mm,
                 divided by the station's MRC-WTF rise per unit Sy). A consistency diagnostic: the Sy at which the
                 observed DWLR rise equals the district GEC recharge. Not a measured Sy, not an independent validation.
  Uncertainty  = bootstrap 95 % CI of the median (2000 resamples, seed 20261013), computed for groups with n >= 3.
                 The CIs are reproduced EXACTLY as in gec_comparison.py by repeating its random-number sequence,
                 and checked against the district/class CIs stored in T5_GEC_consistency.csv.
  GEC norms    = as in gec_comparison.py (GEC-1997, 2009 reprint, Sec. 5.9.1): granite/gneiss 1-4 % (full range,
                 low- and high-clay rows), alluvium 4-20 % (full range, sandy/silty/clayey rows).
                 SHOW_CLAYEY = True also marks the clayey-alluvium row (4-8 %, the norm used in the Methods for R).
  Log y/x axis : kept from the original (Sy_match spans more than one order of magnitude and is a ratio).
Designs: A forest plot (district x geology rows, grouped by geology) | B station dot plot by district |
         C two panels: station variability by geology (a) + group medians vs norms (b) |
         D compact district interval plot, both geology classes per district row
Missing district-geology combinations are written as "no stations", never drawn as zero.
Inputs : paper3_final\\T5b_station_Sy_match.csv, T1_station_recharge.csv, T5_GEC_consistency.csv (gec_comparison.py)
Output : paper3_final\\fig6_options\\Fig6_options_sheet.png + Fig6A..D.png (300 dpi). 
Run    :  %run fig6_options.py        Requires: pandas numpy matplotlib
"""
import os
import re
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ======================================================================================== CONFIG
BASE = os.environ.get("TEST_BASE", r"C:\Users\IITI")
FINAL = os.path.join(BASE, "paper3_final")
T5B, T1F, T5F = (os.path.join(FINAL, f) for f in ("T5b_station_Sy_match.csv", "T1_station_recharge.csv",
                                                  "T5_GEC_consistency.csv"))
OUT = os.path.join(FINAL, "fig6_options")
N_BOOT, SEED = 2000, 20261013                       # as in gec_comparison.py
HARD_RANGE, ALLUV_RANGE, CLAYEY = (1, 4), (4, 20), (4, 8)
SHOW_CLAYEY = True
LOG_LIM = (0.1, 25)                                  # y limits of the original figure (log)
DISTS = ["HAMIRPUR", "JALAUN", "JHANSI", "LALITPUR", "MAHOBA"]   # order of the GEC table in gec_comparison.py
ORDER = ("HARD_ROCK", "ALLUVIUM")

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                     "font.size": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "legend.fontsize": 6.5, "legend.frameon": False, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": False, "savefig.dpi": 300, "savefig.bbox": "tight"})
INK, MUTED = "#1f1f1e", "#6b6a64"
CL = {"HARD_ROCK": dict(color="#a8435a", marker="^", label="Hard rock"),
      "ALLUVIUM": dict(color="#b8860b", marker="o", label="Alluvium")}
NORM = {"HARD_ROCK": HARD_RANGE, "ALLUVIUM": ALLUV_RANGE}

# ======================================================================================== DATA
for f in (T5B, T1F, T5F):
    if not os.path.exists(f):
        sys.exit(f"[STOP] not found: {f}  (run gec_comparison.py first)")
os.makedirs(OUT, exist_ok=True)
T5b = pd.read_csv(T5B)
T1 = pd.read_csv(T1F)
T1["DIST"] = T1.DIST.astype(str).str.upper().str.strip()
T1 = T1[T1.GROSS_C > 0].copy()                       # same filter and row order as gec_comparison.py
gec_mm = T5b.drop_duplicates("DIST").set_index("DIST").GEC_R_mm
T1["GEC_R_mm"] = T1.DIST.map(gec_mm)
T1["SY"] = 100 * T1.GEC_R_mm / (1000 * T1.GROSS_C)
chk_station = np.nanmax(np.abs(T1.set_index("SID").SY - T5b.set_index("SID").Sy_match_pct.reindex(T1.SID).values))

rng = np.random.default_rng(SEED)


def ci(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 3:
        return (np.median(x) if len(x) else np.nan), np.nan, np.nan, len(x)
    b = np.median(rng.choice(x, (N_BOOT, len(x))), axis=1)
    return np.median(x), *np.percentile(b, [2.5, 97.5]), len(x)


# replay the random-number sequence of gec_comparison.py: T5 districts (Sy, GROSS_C), T5 classes, Fig6 groups
DCI, CCI, GCI = {}, {}, {}
for d, q in T1.groupby("DIST"):
    DCI[d] = ci(q.SY); ci(q.GROSS_C)
for c, q in T1.groupby("CLASS"):
    CCI[c] = ci(q.SY)
for d in [d for d in DISTS if d in set(T1.DIST)]:
    for c, q in T1[T1.DIST == d].groupby("CLASS"):
        GCI[(d, c)] = ci(q.SY)
        rng.uniform(-0.05, 0.05, len(q))              # jitter draw of the original figure
# check against the CIs stored in T5
T5 = pd.read_csv(T5F)
dev = []
for _, r in T5.iterrows():
    g = str(r.group)
    v = CCI.get(g.replace("class ", "")) if g.startswith("class ") else DCI.get(g)
    m_ = re.findall(r"-?\d+\.?\d*", str(r.Sy_match_CI))
    if v is None or len(m_) != 2 or not np.isfinite(v[1]):
        continue
    dev.append(max(abs(v[0] - r.Sy_match_pct), abs(v[1] - float(m_[0])), abs(v[2] - float(m_[1]))))
chk_ci = max(dev) if dev else np.nan
G = pd.DataFrame([{"DIST": d, "CLASS": c, "med": v[0], "lo": v[1], "hi": v[2], "n": v[3]} for (d, c), v in GCI.items()])
DL = [d for d in DISTS if d in set(T1.DIST)]
S = T1[T1.CLASS.isin(ORDER)]


# ======================================================================================== HELPERS
def style(ax, axis="y"):
    (ax.set_yscale if axis == "y" else ax.set_xscale)("log")
    (ax.set_ylim if axis == "y" else ax.set_xlim)(*LOG_LIM)
    t = [0.1, 0.2, 0.5, 1, 2, 4, 8, 20]
    (ax.set_yticks if axis == "y" else ax.set_xticks)(t)
    (ax.set_yticklabels if axis == "y" else ax.set_xticklabels)([f"{v:g}" for v in t])
    ax.minorticks_off()
    ax.tick_params(length=2.5, color=MUTED)
    for s_ in ("left", "bottom"):
        ax.spines[s_].set_color(MUTED); ax.spines[s_].set_linewidth(0.6)
    lab = "Sy reconciling DWLR rise with GEC-2024 recharge (%)\n[consistency diagnostic, not measured Sy]"
    (ax.set_ylabel if axis == "y" else ax.set_xlabel)(lab)


def norm_band(ax, c, axis="y", span=(0, 1)):
    lo, hi = NORM[c]
    f = ax.axhspan if axis == "y" else ax.axvspan
    kw = dict(xmin=span[0], xmax=span[1]) if axis == "y" else dict(ymin=span[0], ymax=span[1])
    f(lo, hi, color=CL[c]["color"], alpha=0.10, lw=0, zorder=0, **kw)
    if c == "ALLUVIUM" and SHOW_CLAYEY:
        f(*CLAYEY, facecolor="none", edgecolor=CL[c]["color"], hatch="////", lw=0, alpha=0.45, zorder=0, **kw)


def interval(ax, pos, r, axis="y", ms=5):
    st = CL[r.CLASS]
    if np.isfinite(r.lo):
        (ax.plot([pos, pos], [r.lo, r.hi]) if axis == "y" else ax.plot([r.lo, r.hi], [pos, pos]))
        ax.lines[-1].set(color=st["color"], lw=1.6, solid_capstyle="butt", zorder=3)
    x, y = (pos, r.med) if axis == "y" else (r.med, pos)
    ax.plot(x, y, marker=st["marker"], ms=ms, mfc=st["color"] if np.isfinite(r.lo) else "white",
            mec=st["color"], mew=1.0, ls="", zorder=4)


LEG = [Line2D([], [], ls="", marker=CL[c]["marker"], ms=5, mfc=CL[c]["color"], mec=CL[c]["color"],
              label=f"{CL[c]['label']}: median, bootstrap 95 % CI") for c in ORDER] + \
      [Line2D([], [], ls="", marker="o", ms=5, mfc="white", mec=MUTED, label="median only (n < 3, no CI)"),
       Patch(fc=CL["HARD_ROCK"]["color"], alpha=0.18, lw=0, label=f"GEC norm, granite/gneiss {HARD_RANGE[0]}–{HARD_RANGE[1]} %"),
       Patch(fc=CL["ALLUVIUM"]["color"], alpha=0.18, lw=0, label=f"GEC norm, alluvium {ALLUV_RANGE[0]}–{ALLUV_RANGE[1]} %")]
if SHOW_CLAYEY:
    LEG.append(Patch(fc="none", ec=CL["ALLUVIUM"]["color"], hatch="////", lw=0,
                     label=f"of which clayey alluvium {CLAYEY[0]}–{CLAYEY[1]} %"))


def panel_A(ax):
    """forest plot: rows = district within geology block; each block has only its own norm band"""
    ypos, labels, y = [], [], 0
    blocks = []
    for c in ORDER:
        y0 = y
        for d in DL:
            r = G[(G.DIST == d) & (G.CLASS == c)]
            if len(r):
                interval(ax, y, r.iloc[0], axis="x")
                labels.append(f"{d.title()} (n = {int(r.iloc[0].n)})")
            else:
                ax.text(LOG_LIM[0] * 1.15, y, "no stations", va="center", fontsize=6, color=MUTED, style="italic")
                labels.append(f"{d.title()}")
            ypos.append(y); y += 1
        blocks.append((c, y0, y - 1)); y += 0.8
    ax.set_ylim(y - 0.4, -0.8)
    for c, a, b in blocks:
        lo, hi = NORM[c]
        ax.fill_between([lo, hi], a - 0.45, b + 0.45, color=CL[c]["color"], alpha=0.10, lw=0, zorder=0)
        if c == "ALLUVIUM" and SHOW_CLAYEY:
            ax.fill_between(list(CLAYEY), a - 0.45, b + 0.45, facecolor="none", edgecolor=CL[c]["color"],
                            hatch="////", lw=0, alpha=0.45, zorder=0)
        ax.text(LOG_LIM[1], a - 0.75, CL[c]["label"], ha="right", va="bottom", fontsize=7, fontweight="bold",
                color=CL[c]["color"])
    ax.set_yticks(ypos); ax.set_yticklabels(labels); ax.tick_params(axis="y", length=0)
    style(ax, "x")


def panel_B(ax):
    """station dot plot by district (closest to the original)"""
    norm_band(ax, "HARD_ROCK"); norm_band(ax, "ALLUVIUM")
    rng_j = np.random.default_rng(1)
    for i, d in enumerate(DL):
        for j, c in enumerate(ORDER):
            x = i + (j - 0.5) * 0.3
            q = S[(S.DIST == d) & (S.CLASS == c)]
            r = G[(G.DIST == d) & (G.CLASS == c)]
            if not len(q):
                ax.text(x, LOG_LIM[0] * 1.25, "–", ha="center", fontsize=8, color=MUTED)
                continue
            ax.scatter(x + rng_j.uniform(-0.07, 0.07, len(q)), q.SY, s=7, marker=CL[c]["marker"], color=CL[c]["color"],
                       alpha=0.45, lw=0, zorder=2)
            interval(ax, x + 0.11, r.iloc[0])
    ax.set_xticks(range(len(DL))); ax.set_xticklabels([d.title() for d in DL]); ax.tick_params(axis="x", length=0)
    ax.text(0.01, 0.01, "– = no stations of that geology", transform=ax.transAxes, fontsize=6, color=MUTED)
    style(ax)


def panel_C(fig, spec):
    gs = spec.subgridspec(1, 2, width_ratios=[1, 2.2], wspace=0.08)
    a1 = fig.add_subplot(gs[0]); a2 = fig.add_subplot(gs[1], sharey=a1)
    rng_j = np.random.default_rng(2)
    for j, c in enumerate(ORDER):
        q = S[S.CLASS == c]
        norm_band(a1, c, span=(j / 2 + 0.02, (j + 1) / 2 - 0.02))
        a1.scatter(j + rng_j.uniform(-0.15, 0.15, len(q)), q.SY, s=8, marker=CL[c]["marker"], color=CL[c]["color"],
                   alpha=0.55, lw=0, zorder=2)
        m, lo, hi, n = CCI.get(c, (np.nan,) * 4)
        a1.plot([j - 0.25, j + 0.25], [m, m], color=INK, lw=1.4, zorder=3)
        a1.text(j, LOG_LIM[1] * 0.92, f"n = {len(q)}", ha="center", va="top", fontsize=6.5, color=INK)
    a1.set_xticks([0, 1]); a1.set_xticklabels([CL[c]["label"] for c in ORDER]); a1.set_xlim(-0.6, 1.6)
    a1.tick_params(axis="x", length=0)
    style(a1)
    a1.text(0.02, 1.01, "(a) stations; bar = class median", transform=a1.transAxes, fontsize=6.8, va="bottom")
    norm_band(a2, "HARD_ROCK"); norm_band(a2, "ALLUVIUM")
    for i, d in enumerate(DL):
        for j, c in enumerate(ORDER):
            r = G[(G.DIST == d) & (G.CLASS == c)]
            x = i + (j - 0.5) * 0.28
            if len(r):
                interval(a2, x, r.iloc[0])
            else:
                a2.text(x, LOG_LIM[0] * 1.25, "–", ha="center", fontsize=8, color=MUTED)
    a2.set_xticks(range(len(DL))); a2.set_xticklabels([d.title() for d in DL]); a2.tick_params(axis="x", length=0)
    plt.setp(a2.get_yticklabels(), visible=False); a2.set_ylabel("")
    a2.tick_params(axis="y", length=0); a2.spines["left"].set_visible(False)
    a2.spines["bottom"].set_color(MUTED)
    a2.text(0.02, 1.01, "(b) district medians, bootstrap 95 % CI; – = no stations", transform=a2.transAxes,
            fontsize=6.8, va="bottom")
    return a1, a2


def panel_D(ax):
    """compact interval plot: one row per district, both geology classes on that row"""
    norm_band(ax, "HARD_ROCK", axis="x"); norm_band(ax, "ALLUVIUM", axis="x")
    for i, d in enumerate(DL):
        for j, c in enumerate(ORDER):
            y = i + (j - 0.5) * 0.3
            r = G[(G.DIST == d) & (G.CLASS == c)]
            if len(r):
                interval(ax, y, r.iloc[0], axis="x")
                ax.text(LOG_LIM[1] * 0.95, y, f"n = {int(r.iloc[0].n)}", ha="right", va="center", fontsize=6,
                        color=CL[c]["color"])
            else:
                ax.text(LOG_LIM[0] * 1.15, y, f"{CL[c]['label'].lower()}: no stations", va="center", fontsize=5.8,
                        color=MUTED, style="italic")
    ax.set_yticks(range(len(DL))); ax.set_yticklabels([d.title() for d in DL]); ax.set_ylim(len(DL) - 0.4, -0.6)
    ax.tick_params(axis="y", length=0)
    style(ax, "x")


# ======================================================================================== SHEET
fig = plt.figure(figsize=(15, 11))
gs = fig.add_gridspec(2, 2, wspace=0.42, hspace=0.42)
aA = fig.add_subplot(gs[0, 0]); panel_A(aA)
aA.set_title("A  Forest plot (district × geology, grouped by geology)", loc="left", fontsize=10, fontweight="bold")
aB = fig.add_subplot(gs[0, 1]); panel_B(aB)
aB.set_title("B  Station dot plot by district", loc="left", fontsize=10, fontweight="bold")
c1, _ = panel_C(fig, gs[1, 0])
c1.set_title("C  Station variability (a) + group medians vs norms (b)", loc="left", fontsize=10, fontweight="bold", pad=14)
aD = fig.add_subplot(gs[1, 1]); panel_D(aD)
aD.set_title("D  Compact district interval plot", loc="left", fontsize=10, fontweight="bold")
fig.legend(handles=LEG, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0), fontsize=8)
fig.subplots_adjust(bottom=0.12)
fig.savefig(os.path.join(OUT, "Fig6_options_sheet.png")); plt.close(fig)

for key, size in (("A", (4.2, 4.4)), ("B", (5.2, 3.6)), ("C", (6.4, 3.6)), ("D", (4.4, 3.6))):
    f = plt.figure(figsize=size)
    if key == "C":
        panel_C(f, f.add_gridspec(1, 1)[0, 0])
    else:
        a = f.add_subplot(111); {"A": panel_A, "B": panel_B, "D": panel_D}[key](a)
    f.legend(handles=LEG, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.0))
    f.savefig(os.path.join(OUT, f"Fig6{key}.png")); plt.close(f)

# ======================================================================================== VERDICT
pd.set_option("display.width", 200)
print("=" * 90)
print("VERDICT  fig6_options")
print("=" * 90)
print(f"stations {len(T1)} | used in geology groups {len(S)} {S.CLASS.value_counts().to_dict()} | no class {int(T1.CLASS.isna().sum())}")
print(f"check 1, station Sy_match vs T5b: max |diff| {chk_station:.1e} % -> {'same' if chk_station < 1e-3 else 'DIFFERENT - check inputs'}")
print(f"check 2, district/class median + CI vs T5 (replayed bootstrap): max |diff| {chk_ci:.3f} % "
      f"-> {'same (rounding)' if chk_ci <= 0.006 else 'DIFFERENT - check inputs'}")
print("district x geology groups (as in the original Fig6):")
print(G.assign(CI=G.apply(lambda r: f"[{r.lo:.2f}, {r.hi:.2f}]" if np.isfinite(r.lo) else "n<3", axis=1))
      [["DIST", "CLASS", "n", "med", "CI"]].round(2).to_string(index=False))
miss = [(d, c) for d in DL for c in ORDER if not ((G.DIST == d) & (G.CLASS == c)).any()]
print(f"missing district-geology combinations (shown as 'no stations'): {miss}")
print(f"Files: {OUT} -> Fig6_options_sheet.png, Fig6A.png ... Fig6D.png")
