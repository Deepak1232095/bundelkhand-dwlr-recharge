"""
PAPER 3: FINAL PACKAGE + GEC CONSISTENCY CHECK  ->  new folder  C:\\Users\\IITI\\paper3_final\\
==================================================================================================
1. Copies the validated (Q123) tables and figures from run_Q123\\DWLR_QC\\final into paper3_final\\.
2. T5: district consistency check with the official assessment (NOT an independent validation: GEC itself
   uses Sy / rainfall-infiltration norms):
     GEC_R_mm   = monsoon rainfall recharge (ham) x 10 / recharge-worthy area (km2)      [UP 2024 report]
     Sy_match   = GEC_R_mm / (1000 x GROSS_C)  -> the Sy at which our observed DWLR rise equals GEC recharge
                  (station-level, median + bootstrap 95% CI per district and per geology class)
     compared with the GEC Sy norms (GEC-1997, 2009 reprint, Sec. 5.9.1).
3. Fig6: Sy_match per district / class against the norm bands.
4. README_sources.txt: where every number comes from.
Run in Jupyter:  %run gec_comparison.py      Requires: pandas numpy matplotlib
"""
import os
import sys
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ======================================================================================== CONFIG
BASE = os.environ.get("TEST_BASE", r"C:\Users\IITI")
SRC = os.path.join(BASE, "run_Q123", "DWLR_QC", "final")
OUT = os.path.join(BASE, "paper3_final")
N_BOOT, SEED = 2000, 20261013
rng = np.random.default_rng(SEED)

# Dynamic Ground Water Resources of Uttar Pradesh, 2024 (CGWB & UP GWD, 2025): district table of recharge (ham),
# recharge-worthy area table (km2), monsoon rainfall table (mm, normal and actual 2023)
GEC = pd.DataFrame({
    "DIST": ["HAMIRPUR", "JALAUN", "JHANSI", "LALITPUR", "MAHOBA"],
    "R_monsoon_rain_ham": [32695.52, 64813.61, 32707.65, 16130.37, 9414.19],
    "area_km2": [3815.4, 4565.83, 4619.37, 3819.42, 2293.41],
    "P_monsoon_normal_mm": [660.6, 613.4, 692.7, 843.3, 733.0],
    "P_monsoon_2023_mm": [722.44, 563.74, 679.17, 869.21, 686.0]}).set_index("DIST")
GEC["GEC_R_mm"] = GEC.R_monsoon_rain_ham * 10 / GEC.area_km2
GEC["GEC_R_pct_normal"] = 100 * GEC.GEC_R_mm / GEC.P_monsoon_normal_mm

# GEC-1997 (2009 reprint) Sec. 5.9.1 specific-yield norms (%): recommended, min, max
SY_NORMS = {"Sandy alluvium": (16, 12, 20), "Silty alluvium": (10, 8, 12), "Clayey alluvium": (6, 4, 8),
            "Weathered granite/gneiss, low clay": (3, 2, 4), "Weathered granite/gneiss, high clay": (1.5, 1, 2)}
ALLUV_RANGE, HARD_RANGE = (4, 20), (1, 4)        # full min-max over the relevant rows

# ======================================================================================== run
if not os.path.isdir(SRC):
    sys.exit(f"[STOP] {SRC} not found -> run run_all_audited.py first")
os.makedirs(OUT, exist_ok=True)
copied = []
for f in sorted(os.listdir(SRC)):
    if f.endswith((".png", ".csv")):
        shutil.copyfile(os.path.join(SRC, f), os.path.join(OUT, f)); copied.append(f)
T1 = pd.read_csv(os.path.join(SRC, "T1_station_recharge.csv"))
T1["DIST"] = T1.DIST.astype(str).str.upper().str.strip()
T1 = T1[T1.GROSS_C > 0].copy()
T1["GEC_R_mm"] = T1.DIST.map(GEC.GEC_R_mm)
T1["Sy_match_pct"] = 100 * T1.GEC_R_mm / (1000 * T1.GROSS_C)


def ci(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 3:
        return (np.median(x) if len(x) else np.nan), np.nan, np.nan, len(x)
    b = np.median(rng.choice(x, (N_BOOT, len(x))), axis=1)
    return np.median(x), *np.percentile(b, [2.5, 97.5]), len(x)


rows = []
for d, q in T1.groupby("DIST"):
    m, lo, hi, n = ci(q.Sy_match_pct)
    g, glo, ghi, _ = ci(q.GROSS_C)
    rows.append({"group": d, "n_stations": n, "classes": q.CLASS.value_counts().to_dict(),
                 "GROSS_C_m": round(g, 2), "GROSS_C_CI": f"[{glo:.2f}, {ghi:.2f}]",
                 "GEC_R_mm": round(GEC.loc[d, "GEC_R_mm"], 1) if d in GEC.index else np.nan,
                 "GEC_R_pct_normal_rain": round(GEC.loc[d, "GEC_R_pct_normal"], 1) if d in GEC.index else np.nan,
                 "Sy_match_pct": round(m, 2), "Sy_match_CI": f"[{lo:.2f}, {hi:.2f}]"})
for c, q in T1.groupby("CLASS"):
    m, lo, hi, n = ci(q.Sy_match_pct)
    rng_ = ALLUV_RANGE if c == "ALLUVIUM" else HARD_RANGE
    rows.append({"group": f"class {c}", "n_stations": n, "Sy_match_pct": round(m, 2), "Sy_match_CI": f"[{lo:.2f}, {hi:.2f}]",
                 "GEC_norm_range_pct": f"{rng_[0]}-{rng_[1]}",
                 "share_below_norm_min": round(float((q.Sy_match_pct < rng_[0]).mean()), 2)})
T5 = pd.DataFrame(rows)
T5.to_csv(os.path.join(OUT, "T5_GEC_consistency.csv"), index=False)
T1[["SID", "DIST", "CLASS", "GROSS_C", "GEC_R_mm", "Sy_match_pct"]].round(3).to_csv(
    os.path.join(OUT, "T5b_station_Sy_match.csv"), index=False)

# ======================================================================================== Fig6
COL = {"ALLUVIUM": "#2a78d6", "HARD_ROCK": "#eb6834"}
fig, ax = plt.subplots(figsize=(6, 3.6))
ax.axhspan(*ALLUV_RANGE, color="#2a78d6", alpha=0.10, lw=0)
ax.axhspan(*HARD_RANGE, color="#eb6834", alpha=0.15, lw=0)
ax.text(5.6, 15, "GEC Sy norm, alluvium (4-20 %)", color="#1c5cab", fontsize=7, ha="right")
ax.text(5.6, 2.6, "GEC Sy norm, granite/gneiss (1-4 %)", color="#a3401b", fontsize=7, ha="right")
dists = [d for d in GEC.index if d in set(T1.DIST)]
for i, d in enumerate(dists):
    for j, (c, q) in enumerate(T1[T1.DIST == d].groupby("CLASS")):
        x = i + (j - 0.5) * 0.25
        m, lo, hi, n = ci(q.Sy_match_pct)
        ax.scatter(np.full(len(q), x) + rng.uniform(-0.05, 0.05, len(q)), q.Sy_match_pct, s=8, color=COL.get(c, "#6b6a64"), alpha=0.35, lw=0)
        if np.isfinite(lo):
            ax.plot([x, x], [lo, hi], color=COL.get(c, "#6b6a64"), lw=2)
        ax.scatter([x], [m], s=40, color=COL.get(c, "#6b6a64"), edgecolors="white", lw=0.8, zorder=3,
                   label=c.replace("_", " ").title() if i == 0 or c not in [h.get_label().upper().replace(" ", "_") for h in ax.collections] else None)
ax.set_xticks(range(len(dists))); ax.set_xticklabels([d.title() for d in dists])
ax.set_yscale("log"); ax.set_ylim(0.1, 25)
ax.set_ylabel("Sy that reconciles DWLR rise\nwith GEC-2024 recharge (%)")
h, l = ax.get_legend_handles_labels(); keep = dict(zip(l, h))
ax.legend(keep.values(), keep.keys(), loc="lower left", fontsize=7, frameon=False)
ax.set_title("Implied specific yield vs GEC norms (median, 95% CI; dots = stations)", loc="left", fontsize=9)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.savefig(os.path.join(OUT, "Fig6_Sy_match_vs_GEC_norms.png"), dpi=300, bbox_inches="tight"); plt.close(fig)

# ======================================================================================== README
with open(os.path.join(OUT, "README_sources.txt"), "w", encoding="utf-8") as fh:
    fh.write("paper3_final - validated outputs (Q123 audited DWLR data)\n")
    fh.write("Copied from run_Q123\\DWLR_QC\\final: " + ", ".join(copied) + "\n\n")
    fh.write("T5 / Fig6 sources:\n")
    fh.write("  GEC recharge, area, rainfall: Dynamic Ground Water Resources of Uttar Pradesh, 2024 (CGWB NR & UP GWD, 2025)\n")
    fh.write("  Sy norms: Ground Water Resource Estimation Committee report (GEC-1997, 2009 reprint), Sec. 5.9.1\n")
    fh.write("  GROSS_C: MRC-WTF on audited DWLR (Q123), station median over FULL Jun-Oct seasons 2023-2024\n")
    fh.write("NOTE: GEC recharge is norm-based (WLF/RIF with Sy and RIF norms), so T5 is a consistency check, not validation.\n")
    for k, v in SY_NORMS.items():
        fh.write(f"  {k:<40} recommended {v[0]} %, range {v[1]}-{v[2]} %\n")

pd.set_option("display.width", 200); pd.set_option("display.max_columns", 20)
print("=" * 100)
print("VERDICT  gec_comparison")
print("=" * 100)
print("GEC 2024 district monsoon rainfall recharge:")
print(GEC[["GEC_R_mm", "GEC_R_pct_normal"]].round(1).to_string())
print("\nT5:")
print(T5.to_string(index=False))
print(f"\nFolder: {OUT}  ({len(copied)} files copied + T5, T5b, Fig6, README_sources.txt)")
