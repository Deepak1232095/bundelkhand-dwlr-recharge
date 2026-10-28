"""
PAPER 3 FINAL RESULTS: tables + figures from EXISTING outputs (no new analysis)
================================================================================
Reads:  DWLR_QC/dwlr_sign_smoothness.csv                      (lat, lon)
        DWLR_QC/wtf_mrc/recharge_station_years.csv            (GROSS_C, PEAKLOW, BIANN, rain per station-year)
        DWLR_QC/wtf_mrc/recharge_episodes.csv                 (episodes)
        DWLR_QC/wtf_mrc/diagnostics/diag_station.csv          (CLASS, LB, SY100, rain-free share)
        virtual_confined/confined_wtf_results.csv              (optional, flopy bias experiment)
        DWLR_QC/fingerprints/real_fingerprints.csv            (optional)
Writes: DWLR_QC/final/  T1..T5 csv + Fig1..Fig5 png (300 dpi)

Sy is NOT estimated. Hard rock: recharge on a Sy range from published sources (GEC-2015 norm 1-3 %, NAQUIM
shallow aquifer 4 %). Alluvium: the reverse question - which Sy makes recharge a plausible share of rain
(10/20/30 %) - because the GEC alluvium norm (6-16 %, UNVERIFIED) gives recharge > 45 % of rain.
Run in Jupyter:  %run final_results.py      Requires: pandas numpy scipy matplotlib
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ======================================================================================== CONFIG
QC = r"DWLR_QC"
SIGN = os.path.join(QC, "dwlr_sign_smoothness.csv")
YRS = os.path.join(QC, "wtf_mrc", "recharge_station_years.csv")
EPS = os.path.join(QC, "wtf_mrc", "recharge_episodes.csv")
DIAG = os.path.join(QC, "wtf_mrc", "diagnostics", "diag_station.csv")
FLOPY = os.environ.get("TEST_FLOPY", os.path.join("virtual_confined", "confined_wtf_results.csv"))
FPR = os.path.join(QC, "fingerprints", "real_fingerprints.csv")
OUT = os.path.join(QC, "final")
SY_HR = {"GEC low (1%)": 0.01, "GEC high (3%)": 0.03, "NAQUIM shallow (4%)": 0.04}
SY_AL_GEC = (0.06, 0.16)                 # UNVERIFIED placeholder - to be confirmed from GEC-2015
FRAC = [0.10, 0.20, 0.30]                # plausible recharge / Jun-Oct rain for the alluvium Sy constraint
N_BOOT, SEED = 2000, 20261013
COL = {"ALLUVIUM": "#2a78d6", "HARD_ROCK": "#eb6834"}     # categorical slots 1, 2
MRK = {"ALLUVIUM": "o", "HARD_ROCK": "^"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5e0"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(SEED)
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
                     "savefig.dpi": 300, "savefig.bbox": "tight"})

for f in (SIGN, YRS, EPS, DIAG):
    if not os.path.exists(f):
        sys.exit(f"[STOP] not found: {f}  (run wtf_mrc_recharge.py and wtf_diagnostics.py first)")
sg = pd.read_csv(SIGN).drop_duplicates("SID").set_index("SID")
Y = pd.read_csv(YRS)
E = pd.read_csv(EPS, parse_dates=["start", "end"])
D = pd.read_csv(DIAG).drop_duplicates("SID").set_index("SID")


def ci(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 3:
        return np.median(x) if len(x) else np.nan, np.nan, np.nan, len(x)
    b = np.median(rng.choice(x, (N_BOOT, len(x))), axis=1)
    return np.median(x), *np.percentile(b, [2.5, 97.5]), len(x)


def fmt(x, f="{:.2f}"):
    m, lo, hi, n = ci(x)
    return f"{f.format(m)} [{f.format(lo)}, {f.format(hi)}]" if np.isfinite(lo) else f"{f.format(m)} [n<3]"


# ======================================================================================== T1 station table
P = Y[(Y.MRC == "PRIMARY") & Y.FULL.astype(bool)]
st = P.groupby("SID").agg(PEAKLOW=("PEAKLOW_m", "median"), BIANN=("BIANN_m", "median"))
T1 = D[[c for c in ("DIST", "CLASS", "n_years", "RAIN", "GROSS_C", "LB", "LB_RATIO", "SY100", "RF_SHARE_AN") if c in D.columns]].join(st)
T1 = T1.join(sg[["lat", "lon"]])
T1["BIANN_over_GC"] = T1.BIANN / T1.GROSS_C
for nm, sy in SY_HR.items():
    k = nm.split(" (")[0].replace(" ", "_")
    T1[f"R_mm_{k}"] = 1000 * sy * T1.GROSS_C
    T1[f"R_pct_{k}"] = 100 * T1[f"R_mm_{k}"] / T1.RAIN
for f in FRAC:
    T1[f"SYeff_{int(100 * f)}pct"] = f * T1.RAIN / 1000 / T1.GROSS_C
T1["R_pct_GEClow_alluv"] = 100 * 1000 * SY_AL_GEC[0] * T1.GROSS_C / T1.RAIN
T1.round(4).to_csv(os.path.join(OUT, "T1_station_recharge.csv"))

# ======================================================================================== T2 class / district
rows = []
for grp_name, grp in [("ALL", T1)] + [(c, q) for c, q in T1.groupby("CLASS")] + [(d, q) for d, q in T1.groupby("DIST")]:
    r = {"group": grp_name, "n": len(grp), "rain_mm": fmt(grp.RAIN, "{:.0f}"), "GROSS_C_m": fmt(grp.GROSS_C),
         "LB_over_GC": fmt(grp.LB_RATIO), "BIANN_over_GC": fmt(grp.BIANN_over_GC)}
    for nm, sy in SY_HR.items():
        k = nm.split(" (")[0].replace(" ", "_")
        r[f"R_mm_{k}"] = fmt(grp[f"R_mm_{k}"], "{:.0f}"); r[f"R_pct_{k}"] = fmt(grp[f"R_pct_{k}"], "{:.1f}")
    for f in FRAC:
        r[f"SYeff_{int(100 * f)}pct"] = fmt(grp[f"SYeff_{int(100 * f)}pct"], "{:.3f}")
    rows.append(r)
T2 = pd.DataFrame(rows)
T2.to_csv(os.path.join(OUT, "T2_class_district_summary.csv"), index=False)

# ======================================================================================== T3 events
EF = E.copy()
EF["RBIN"] = pd.cut(EF.RAIN_mm, [-0.1, 10, 25, 50, 100, 1e9], labels=["<10", "10-25", "25-50", "50-100", ">100"])
per = EF.groupby(["SID", "YEAR"]).RECH_m.apply(lambda v: np.sort(v.values)[::-1])
top_share = per.apply(lambda v: v[:3].sum() / v.sum() if v.sum() > 0 else np.nan)
T3 = EF.groupby("RBIN", observed=True).RECH_m.agg(n="size", median_m="median",
                                                   q25=lambda v: v.quantile(0.25), q75=lambda v: v.quantile(0.75))
T3["share_of_total"] = EF.groupby("RBIN", observed=True).RECH_m.sum() / EF.RECH_m.sum()
T3.round(3).to_csv(os.path.join(OUT, "T3_episodes_by_rain.csv"))
q = EF.dropna(subset=["RAIN_mm"])
rho_ev = stats.spearmanr(q.RAIN_mm, q.RECH_m)[0]

# ======================================================================================== T4 flopy bias (optional)
FL = pd.read_csv(FLOPY) if os.path.exists(FLOPY) else None
if FL is not None:
    T4 = FL[FL.well.str.endswith("between")].pivot_table(index=["Kv_clay", "pump"], columns="well", values="OVEREST")
    T4.round(2).to_csv(os.path.join(OUT, "T4_flopy_bias.csv"))
FP = pd.read_csv(FPR).set_index("SID") if os.path.exists(FPR) else None

# ======================================================================================== FIGURES
def save(fig, name):
    fig.savefig(os.path.join(OUT, name)); plt.close(fig)


# Fig1 map of GROSS_C
fig, ax = plt.subplots(figsize=(5.2, 4.2))
cm = matplotlib.colors.LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
vmin, vmax = np.nanpercentile(T1.GROSS_C, [5, 95])
for c, q in T1.groupby("CLASS"):
    sc = ax.scatter(q.lon, q.lat, c=q.GROSS_C, cmap=cm, vmin=vmin, vmax=vmax, marker=MRK.get(c, "o"), s=42,
                    edgecolors="white", linewidths=0.8, label=c.replace("_", " ").title(), zorder=3)
cb = fig.colorbar(sc, ax=ax, shrink=0.8); cb.set_label("GROSS_C, monsoon rise per unit Sy (m)"); cb.outline.set_visible(False)
leg = ax.legend(loc="lower right", title="Geology")
for h in leg.legend_handles:
    h.set_facecolor(MUTED)
ax.set_xlabel("Longitude (°E)"); ax.set_ylabel("Latitude (°N)"); ax.set_aspect("equal")
ax.set_title("Monsoon water-table response at DWLR stations", loc="left", color=INK)
save(fig, "Fig1_map_GROSS_C.png")

# Fig2 recharge % of rain vs Sy, by class
sy = np.linspace(0.005, 0.16, 120)
fig, ax = plt.subplots(figsize=(5.6, 3.8))
ax.axhspan(10, 30, color=GRID, alpha=0.7, lw=0, zorder=0)
ax.text(15.6, 20, "plausible\n10-30 % of rain", ha="right", va="center", color=MUTED, fontsize=8)
for c, q in T1.groupby("CLASS"):
    ratio = (q.GROSS_C / q.RAIN).dropna().values
    curves = 100 * 1000 * sy[:, None] * ratio[None, :]
    med, lo, hi = np.median(curves, 1), np.percentile(curves, 25, 1), np.percentile(curves, 75, 1)
    ax.plot(100 * sy, lo, color=COL.get(c, MUTED), lw=1, ls=(0, (4, 3)))
    ax.plot(100 * sy, hi, color=COL.get(c, MUTED), lw=1, ls=(0, (4, 3)))
    ax.plot(100 * sy, med, color=COL.get(c, MUTED), lw=2, label=f"{c.replace('_', ' ').title()} (n={len(ratio)}): median, dashed = IQR")
ax.axvspan(1, 4, ymin=0, ymax=0.04, color=COL["HARD_ROCK"], alpha=0.9, lw=0)
ax.axvspan(100 * SY_AL_GEC[0], 100 * SY_AL_GEC[1], ymin=0.04, ymax=0.08, color=COL["ALLUVIUM"], alpha=0.9, lw=0)
ax.text(4.3, 0.005, "hard-rock Sy (GEC/NAQUIM)", color=INK, fontsize=7, ha="left", va="bottom", transform=ax.get_xaxis_transform())
ax.text(100 * SY_AL_GEC[0], 0.095, "alluvium Sy norm (GEC, unverified)", color=INK, fontsize=7, transform=ax.get_xaxis_transform())
ax.set_xlim(0, 16); ax.set_ylim(0, 130)
ax.set_xlabel("Specific yield Sy (%)"); ax.set_ylabel("Implied recharge (% of Jun–Oct rain)")
ax.set_title("Recharge implied by the observed rise for a given Sy", loc="left", color=INK)
ax.legend(loc="upper left")
save(fig, "Fig2_recharge_vs_Sy.png")

# Fig3 episodes: by rain bin + concentration
fig, (a1, a2) = plt.subplots(1, 2, figsize=(8, 3.4), gridspec_kw={"width_ratios": [1.2, 1]})
bins = [b for b in ["<10", "10-25", "25-50", "50-100", ">100"] if b in set(EF.RBIN.dropna().astype(str))]
data = [EF.RECH_m[EF.RBIN.astype(str) == b].values for b in bins]
bp = a1.boxplot(data, widths=0.5, showfliers=False, patch_artist=True,
                medianprops=dict(color=INK, lw=1.5), whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED))
a1.set_xticks(range(1, len(bins) + 1)); a1.set_xticklabels(bins)
for p_ in bp["boxes"]:
    p_.set(facecolor="#86b6ef", edgecolor="#1c5cab", lw=1)
for i, b in enumerate(bins):
    a1.text(i + 1, 0.98, f"n={len(data[i])}", transform=a1.get_xaxis_transform(), ha="center", va="top", color=MUTED, fontsize=7)
a1.set_xlabel("Rain during episode (mm)"); a1.set_ylabel("Episode recharge per unit Sy (m)")
a1.set_title("Recharge per episode", loc="left", color=INK)
mx = per.apply(len).max()
cum = np.array([np.r_[np.cumsum(v) / v.sum(), np.ones(mx - len(v))] for v in per if v.sum() > 0])
k = np.arange(1, mx + 1)
a2.fill_between(k, np.percentile(cum, 25, 0), np.percentile(cum, 75, 0), color="#86b6ef", alpha=0.4, lw=0)
a2.plot(k, np.median(cum, 0), color="#1c5cab", lw=2, marker="o", ms=4)
a2.axvline(3, color=MUTED, lw=0.8, ls="--")
a2.text(3.2, 0.1, f"3 largest = {100 * np.nanmedian(top_share):.0f} % (median)", color=INK, fontsize=8)
a2.set_xlim(0.5, min(mx, 15) + 0.5); a2.set_ylim(0, 1.02)
a2.set_xlabel("Episodes ranked by size"); a2.set_ylabel("Cumulative share of season recharge")
a2.set_title("Concentration in few events", loc="left", color=INK)
fig.tight_layout()
save(fig, "Fig3_episodes.png")

# Fig4 biannual vs MRC
fig, ax = plt.subplots(figsize=(4.4, 4))
for c, q in T1.groupby("CLASS"):
    ax.scatter(q.GROSS_C, q.BIANN, color=COL.get(c, MUTED), marker=MRK.get(c, "o"), s=30, edgecolors="white",
               linewidths=0.6, label=c.replace("_", " ").title(), zorder=3)
lim = np.nanmax([T1.GROSS_C.max(), T1.BIANN.max()]) * 1.05
ax.plot([0, lim], [0, lim], color=MUTED, lw=1, ls="--"); ax.text(lim * 0.97, lim * 0.97, "1:1", color=MUTED, ha="right", va="top")
ax.set_xlim(0, lim); ax.set_ylim(min(0, np.nanmin(T1.BIANN) * 1.1), lim)
ax.set_xlabel("MRC-WTF season recharge per unit Sy (m)"); ax.set_ylabel("May 20 – Nov 10 rise (m)")
ax.set_title(f"May–Nov rise = {100 * np.nanmedian(T1.BIANN_over_GC):.0f} % of MRC-WTF rise (median of stations, n={int(T1.BIANN_over_GC.notna().sum())})", loc="left", color=INK, fontsize=8)
ax.legend(loc="upper left")
save(fig, "Fig4_biannual_vs_MRC.png")

# Fig5 flopy bias (optional)
if FL is not None:
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    seq = {0.001: "#0d366b", 0.01: "#3987e5", 0.1: "#9ec5f4"}
    for kv, q in FL[FL.well == "L3_between"].groupby("Kv_clay"):
        ax.plot(q.pump, q.OVEREST, color=seq.get(kv, MUTED), lw=2, marker="o", ms=5, label=f"confined well, clay Kv {kv:g} m/d")
    w = FL[FL.well == "L1_between"].groupby("pump").OVEREST.median()
    ax.plot(w.index, w.values, color=MUTED, lw=2, ls="--", marker="s", ms=5, label="water-table well (median)")
    ax.axhline(1, color=INK, lw=0.8)
    ax.set_yscale("log"); ax.set_xlabel("Pumping from confined layer (× annual recharge)")
    ax.set_ylabel("MRC-WTF recharge / true recharge")
    ax.set_title("Synthetic test: pumped confined well inflates WTF", loc="left", color=INK)
    ax.legend(loc="upper left", fontsize=7)
    save(fig, "Fig5_flopy_bias.png")

# ======================================================================================== VERDICT
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
print("=" * 100)
print("VERDICT  final_results")
print("=" * 100)
print(f"stations {len(T1)} | by class {T1.CLASS.value_counts().to_dict()} | episodes {len(E)}")
show = ["group", "n", "rain_mm", "GROSS_C_m"] + [c for c in T2.columns if c.startswith("R_pct_")] + ["LB_over_GC", "BIANN_over_GC"]
print("\nT2 (median [95% CI]) — recharge % of Jun-Oct rain on the hard-rock Sy range:")
print(T2[show].to_string(index=False))
print("\nT2 — Sy that makes recharge 10 / 20 / 30 % of rain:")
print(T2[["group", "n"] + [c for c in T2.columns if c.startswith("SYeff_")]].to_string(index=False))
al = T1[T1.CLASS == "ALLUVIUM"]
if len(al):
    print(f"\nalluvium at GEC low Sy {SY_AL_GEC[0]:.2f}: recharge = {fmt(al.R_pct_GEClow_alluv, '{:.0f}')} % of rain "
          f"| stations > 100 %: {int((al.R_pct_GEClow_alluv > 100).sum())} of {len(al)}")
print(f"\nT3 episodes: median share of top-3 episodes {np.nanmedian(top_share):.2f} | rain-recharge Spearman {rho_ev:+.2f}")
print(T3.round(3).to_string())
if FL is not None:
    print("\nT4 flopy OVEREST (between-well position):")
    print(T4.round(2).to_string())
if FP is not None:
    print(f"\nfingerprints (real): F1n {fmt(FP.F1n)} | F2 {fmt(FP.F2)}")
print(f"\nFiles: {sorted(os.listdir(OUT))}")
