"""
FIGURE 3 (FINAL v2, design A): recharge implied by the observed monsoon rise for a given specific yield
======================================================================================================
Same data and calculation as Fig2_recharge_vs_Sy.png in final_results.py; nothing new is estimated.
  per station  R (mm)        = 1000 x Sy x Delta H            (Delta H = GROSS_C, m per unit Sy)
               R (% of rain) = 100 x R / P_W                  (P_W = RAIN, June-October rainfall, mm)
  per class    median line + interquartile range (25th-75th percentile) ACROSS STATIONS
               (station-level variability, NOT a confidence interval)
Sy scenarios (Methods text, GEC 1997): hard rock 3 % (1.5-4 %), clayey alluvium 6 % (4-8 %).
  Drawn as thin range bars just above the x-axis; central value = diamond. No vertical reference lines.
Switches: SY_SET "manuscript" (default) or "original" (old figure: 1-4 %, 6-16 %)
          BAND   "off" (default; the old 10-30 % band has no documented source) or "keep"
Checks printed in the VERDICT: station counts per class, R formula vs. the stored R_pct_* columns of T1
(written by final_results.py) where those columns exist.
Input : T1_station_recharge.csv (run_Q123\\DWLR_QC\\final\\ first, then paper3_final\\)
Output: paper3_final\\fig3_final\\Fig3_recharge_vs_Sy.png (600 dpi) + .pdf + Fig3_values.csv + Fig3_caption_draft.txt
Run   :  %run fig3_final.py        Requires: pandas numpy matplotlib.   
"""
import os
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
T1_CANDS = [os.path.join(BASE, "run_Q123", "DWLR_QC", "final", "T1_station_recharge.csv"),
            os.path.join(BASE, "paper3_final", "T1_station_recharge.csv")]
OUT = os.path.join(BASE, "paper3_final", "fig3_final")
SY_SET = "manuscript"
BAND = "off"
SY_SETS = {"manuscript": {"HARD_ROCK": (0.03, 0.015, 0.04), "ALLUVIUM": (0.06, 0.04, 0.08)},
           "original": {"HARD_ROCK": (None, 0.01, 0.04), "ALLUVIUM": (None, 0.06, 0.16)}}
OLD_BAND = (10, 30)
X_MAX = 16                              # x limit of the original figure (kept)
Y_MAX = "auto"                          # "auto" = highest upper IQR bound in 0..X_MAX, rounded up to 10; or a number
FIG_W, FIG_H = 3.74, 3.9                # inches: WRR single column (95 mm)
IQR_ALPHA = 0.10

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                     "font.size": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "legend.fontsize": 6.5, "legend.frameon": False, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": False, "pdf.fonttype": 42, "savefig.bbox": "tight"})
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5e0"
CL = {"HARD_ROCK": dict(color="#a8435a", ls="-", label="Hard rock"),
      "ALLUVIUM": dict(color="#b8860b", ls=(0, (5, 2)), label="Alluvium")}
ORDER = ("HARD_ROCK", "ALLUVIUM")

# ======================================================================================== DATA
T1P = next((p for p in T1_CANDS if os.path.exists(p)), None)
if T1P is None:
    sys.exit(f"[STOP] T1_station_recharge.csv not found in {T1_CANDS}")
if SY_SET not in SY_SETS or BAND not in ("off", "keep"):
    sys.exit("[STOP] SY_SET must be 'manuscript' or 'original'; BAND must be 'off' or 'keep'")
os.makedirs(OUT, exist_ok=True)
T0 = pd.read_csv(T1P)
for c in ("CLASS", "GROSS_C", "RAIN"):
    if c not in T0.columns:
        sys.exit(f"[STOP] column {c} missing in {T1P}")
T1 = T0[T0.CLASS.isin(CL) & np.isfinite(T0.GROSS_C) & np.isfinite(T0.RAIN) & (T0.GROSS_C > 0)].copy()
dropped = T0.loc[~T0.index.isin(T1.index)]
SYS = SY_SETS[SY_SET]
sy = np.linspace(0, X_MAX / 100, 321)


def pct_rain(q, s):
    """R = 1000 Sy dH (mm); % of June-October rain = 100 R / P_W. s may be an array of Sy (fraction)."""
    R_mm = 1000 * np.asarray(s)[..., None] * q.GROSS_C.values
    return 100 * R_mm / q.RAIN.values


# ======================================================================================== FIGURE
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
if BAND == "keep":
    ax.axhspan(*OLD_BAND, color=GRID, lw=0, zorder=0)
rows, ymax_data = [], 0.0
for c in ORDER:
    q = T1[T1.CLASS == c]
    if not len(q):
        continue
    cur = pct_rain(q, sy)
    med, lo, hi = np.median(cur, 1), np.percentile(cur, 25, 1), np.percentile(cur, 75, 1)
    st = CL[c]
    ax.fill_between(100 * sy, lo, hi, color=st["color"], alpha=IQR_ALPHA, lw=0, zorder=1)
    ax.plot(100 * sy, lo, color=st["color"], lw=0.5, alpha=0.6, zorder=2)          # thin IQR edges
    ax.plot(100 * sy, hi, color=st["color"], lw=0.5, alpha=0.6, zorder=2)
    ax.plot(100 * sy, med, color=st["color"], ls=st["ls"], lw=1.6, zorder=3)
    ymax_data = max(ymax_data, float(np.nanmax(hi)))
    for s_ in [v for v in SYS[c] if v is not None]:
        r = pct_rain(q, s_)
        rows.append({"class": c, "Sy_pct": round(100 * s_, 2), "n": len(q), "median_pct_rain": np.median(r),
                     "q25": np.percentile(r, 25), "q75": np.percentile(r, 75)})
Y_TOP = float(np.ceil(ymax_data / 10) * 10) if Y_MAX == "auto" else float(Y_MAX)
ax.set_xlim(0, X_MAX); ax.set_ylim(0, Y_TOP)
ax.set_xlabel("Specific yield, Sy (%)")
ax.set_ylabel("Implied recharge (% of June–October rainfall)")
ax.tick_params(length=2.5, color=MUTED)
for s_ in ("left", "bottom"):
    ax.spines[s_].set_color(MUTED); ax.spines[s_].set_linewidth(0.6)

# Sy scenarios: thin range bars just above the x-axis (data y), central value = diamond
BAR_Y = {"HARD_ROCK": 0.030 * Y_TOP, "ALLUVIUM": 0.065 * Y_TOP}
for c in ORDER:
    cen, lo, hi = SYS[c]
    y_ = BAR_Y[c]
    ax.plot([100 * lo, 100 * min(hi, X_MAX / 100)], [y_, y_], color=CL[c]["color"], lw=2.4, solid_capstyle="butt",
            zorder=4)
    if cen is not None:
        ax.plot(100 * cen, y_, marker="D", ms=4, mfc="white", mec=CL[c]["color"], mew=1.1, zorder=5)
    ax.text(100 * min(hi, X_MAX / 100) + 0.3, y_, f"{100 * lo:g}–{100 * hi:g} %", va="center", fontsize=6,
            color=CL[c]["color"], zorder=5)

hand = [Line2D([], [], color=CL[c]["color"], ls=CL[c]["ls"], lw=1.6,
               label=f"{CL[c]['label']}, station median (n = {int((T1.CLASS == c).sum())})")
        for c in ORDER if (T1.CLASS == c).any()]
hand.append(Patch(facecolor=MUTED, alpha=0.25, edgecolor=MUTED, lw=0.4,
                  label="Station-level IQR (25th–75th percentile; not a confidence interval)"))
hand.append(Line2D([], [], color=MUTED, lw=2.4, label="Sy range by geology (GEC, 1997)" if SY_SET == "manuscript"
                   else "Sy range by geology (old figure)"))
if SY_SET == "manuscript":
    hand.append(Line2D([], [], ls="", marker="D", ms=4, mfc="white", mec=MUTED, mew=1.1, label="Central Sy"))
if BAND == "keep":
    hand.append(Patch(facecolor=GRID, lw=0, label=f"{OLD_BAND[0]}–{OLD_BAND[1]} % of rainfall [add reference]"))
ax.legend(handles=hand, loc="upper left", bbox_to_anchor=(-0.02, -0.12), ncol=1, handlelength=2.2)

fig.savefig(os.path.join(OUT, "Fig3_recharge_vs_Sy.png"), dpi=600)
fig.savefig(os.path.join(OUT, "Fig3_recharge_vs_Sy.pdf"))
plt.close(fig)
V = pd.DataFrame(rows).round(2)
V.to_csv(os.path.join(OUT, "Fig3_values.csv"), index=False)

# ======================================================================================== CHECKS
checks = []
for col, s_ in (("R_pct_GEC_low", 0.01), ("R_pct_GEC_high", 0.03), ("R_pct_NAQUIM_shallow", 0.04)):
    if col in T1.columns:
        d = np.nanmax(np.abs(pct_rain(T1, s_) - T1[col].values))
        checks.append(f"{col} (Sy {100 * s_:g} %): max |diff| {d:.2e} -> {'same' if d < 1e-6 else 'DIFFERENT - check inputs'}")
# independent IQR check: every station curve is a line through 0, so percentile(Sy x k x ratio) = Sy x k x percentile(ratio)
iqr_chk = 0.0
for _, r in pd.DataFrame(rows).iterrows():
    q = T1[T1.CLASS == r["class"]]
    k = 100 * 1000 * (r.Sy_pct / 100) / 1  # % of rain per unit ratio
    alt = [k * (q.GROSS_C / q.RAIN).quantile(p_) for p_ in (0.25, 0.5, 0.75)]
    iqr_chk = max(iqr_chk, *(abs(a - b) for a, b in zip(alt, (r.q25, r.median_pct_rain, r.q75))))
nH, nA = (int((T1.CLASS == c).sum()) for c in ORDER)
cap = (f"Figure 3. Monsoon recharge implied by the observed June–October water-table rise as a function of specific "
       f"yield. Recharge is inferred from an assumed Sy, not measured independently: for each station, "
       f"R = 1000 Sy ΔH (mm) is expressed as a percentage of June–October rainfall. Lines "
       f"show the median and shading the interquartile range across stations (station-level variability, not a "
       f"confidence interval) for hard rock (n = {nH}) and alluvium (n = {nA}). Bars near the x-axis mark the Sy "
       + ("scenarios from GEC (1997): weathered granite and gneiss 3 % (range 1.5–4 %, from the recommended value for "
          "high-clay to the maximum for low-clay weathered rock) and clayey alluvium 6 % (4–8 %); diamonds mark the "
          "central values." if SY_SET == "manuscript" else "ranges [SOURCE].")
       + (f" The upper interquartile bound of alluvium exceeds the y-axis limit above Sy = [X] %." if Y_MAX != "auto" else ""))
open(os.path.join(OUT, "Fig3_caption_draft.txt"), "w", encoding="utf-8").write(cap + "\n")

print("=" * 90)
print("VERDICT  fig3_final v2")
print("=" * 90)
print(f"T1: {T1P}")
print(f"stations in file {len(T0)} | plotted {len(T1)}: hard rock {nH}, alluvium {nA} | not plotted {len(dropped)} "
      f"(CLASS {dropped.CLASS.astype(str).tolist()[:5]})")
print(f"SY_SET {SY_SET} {SYS} | BAND {BAND} | y-axis top {Y_TOP:g} (highest upper IQR bound in plot {ymax_data:.1f})")
print(f"IQR check (curve percentiles vs Sy x percentile of GROSS_C/RAIN): max |diff| {iqr_chk:.2e} -> "
      f"{'same' if iqr_chk < 1e-6 else 'DIFFERENT - check inputs'}")
print("formula check vs. stored columns from final_results.py:", *(checks or ["no R_pct_* columns in T1 (skipped)"]),
      sep="\n  ")
print("recharge % of June-October rainfall at the Sy scenario values (median [IQR] across stations):")
print(V.to_string(index=False))
print(f"Files: {OUT} -> Fig3_recharge_vs_Sy.png (600 dpi), .pdf, Fig3_values.csv, Fig3_caption_draft.txt")
