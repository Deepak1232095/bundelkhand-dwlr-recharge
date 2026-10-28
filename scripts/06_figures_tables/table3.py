"""
TABLE 3: district GEC-2024 monsoon rainfall recharge vs. this study (consistency diagnostic, not validation)
==========================================================================================================
GEC values (ha m, km2, mm) are the same constants as in gec_comparison.py, taken from Dynamic Ground Water Resources
of Uttar Pradesh, 2024 (CGWB & GWD-UP). Columns:
  GEC recharge (ha m), assessment area (km2), GEC recharge (mm) = 10 x ha m / km2, % of normal monsoon rainfall
  Matching Sy, daily rise (%)   = median over the district's stations of 100 x GEC_mm / (1000 x GROSS_C)  [95% CI]
  Matching Sy, biannual rise (%)= the same with BIANN (20 May - 10 Nov rise); stations with BIANN <= 0 excluded (counted)
  This study, recharge (mm)     = median of 1000 x Sy(class) x GROSS_C (same definition as Table 2; range for the Sy range)
The old Table 3 used the monthly-sum recharge (353 / 541 / ...) in the last column, which is the map definition and does
not match Table 2. The script prints the matching-Sy medians next to T5_GEC_consistency.csv as a check.
Input : T1_station_recharge.csv (run_Q123\\DWLR_QC\\final\\ first, then paper3_final\\), optional paper3_final\\T5_GEC_consistency.csv
Output: paper3_final\\Table3_GEC_districts.csv.     Run:  %run table3.py   and 
"""
import os
import sys
import numpy as np
import pandas as pd

BASE = os.environ.get("TEST_BASE", r"C:\Users\IITI")
FINAL = os.path.join(BASE, "paper3_final")
T1P = next((p for p in (os.path.join(BASE, "run_Q123", "DWLR_QC", "final", "T1_station_recharge.csv"),
                        os.path.join(FINAL, "T1_station_recharge.csv")) if os.path.exists(p)), None)
T5P = os.path.join(FINAL, "T5_GEC_consistency.csv")
GEC = pd.DataFrame({                              # as in gec_comparison.py (UP 2024 report)
    "DIST": ["HAMIRPUR", "JALAUN", "JHANSI", "LALITPUR", "MAHOBA"],
    "R_ham": [32695.52, 64813.61, 32707.65, 16130.37, 9414.19],
    "area_km2": [3815.4, 4565.83, 4619.37, 3819.42, 2293.41],
    "P_normal_mm": [660.6, 613.4, 692.7, 843.3, 733.0]}).set_index("DIST")
GEC["R_mm"] = GEC.R_ham * 10 / GEC.area_km2
SY = {"HARD_ROCK": (0.03, 0.015, 0.04), "ALLUVIUM": (0.06, 0.04, 0.08)}
rng = np.random.default_rng(20261013)
if T1P is None:
    sys.exit("[STOP] T1_station_recharge.csv not found")
T = pd.read_csv(T1P)
T["DIST"] = T.DIST.astype(str).str.upper().str.strip()
T = T[np.isfinite(T.GROSS_C) & (T.GROSS_C > 0) & T.DIST.isin(GEC.index)].copy()
T["GEC_mm"] = T.DIST.map(GEC.R_mm)
T["SYM_daily"] = 100 * T.GEC_mm / (1000 * T.GROSS_C)
T["SYM_biann"] = np.where(T.BIANN > 0, 100 * T.GEC_mm / (1000 * T.BIANN.where(T.BIANN > 0)), np.nan)
for k, i in (("R", 0), ("R_lo", 1), ("R_hi", 2)):
    T[k] = T.CLASS.map(lambda c: SY[c][i] if c in SY else np.nan) * 1000 * T.GROSS_C


def med_ci(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 3:
        return f"{np.median(x):.2f}" if len(x) else "–"
    b = np.median(rng.choice(x, (2000, len(x))), axis=1)
    lo, hi = np.percentile(b, [2.5, 97.5])
    return f"{np.median(x):.2f} [{lo:.2f}–{hi:.2f}]"


rows = []
for d, g in GEC.iterrows():
    q = T[T.DIST == d]
    qs = q[q.CLASS.isin(SY)]
    rows.append({"District": d.title(), "GEC rainfall recharge (ha m)": f"{g.R_ham:,.0f}",
                 "Assessment area (km²)": f"{g.area_km2:,.0f}", "GEC recharge (mm)": f"{g.R_mm:.0f}",
                 "GEC, % of normal monsoon rain": f"{100 * g.R_mm / g.P_normal_mm:.0f}",
                 "n": len(q), "Matching Sy, daily rise (%)": med_ci(q.SYM_daily),
                 "Matching Sy, biannual rise (%)": med_ci(q.SYM_biann),
                 "n with BIANN <= 0": int((q.BIANN <= 0).sum()),
                 "This study, recharge (mm)": f"{np.nanmedian(qs.R):.0f} ({np.nanmedian(qs.R_lo):.0f}–{np.nanmedian(qs.R_hi):.0f})"})
O = pd.DataFrame(rows)
O.to_csv(os.path.join(FINAL, "Table3_GEC_districts.csv"), index=False, encoding="utf-8-sig")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
print("=" * 90); print("TABLE 3"); print("=" * 90)
print(f"T1: {T1P}\n"); print(O.to_string(index=False))
if os.path.exists(T5P):
    t5 = pd.read_csv(T5P).set_index("group")
    chk = [(d, round(T[T.DIST == d].SYM_daily.median(), 2), t5.Sy_match_pct.get(d, np.nan)) for d in GEC.index]
    print("\ncheck, matching Sy (daily) median: this script vs T5_GEC_consistency.csv:")
    for d, a, b in chk:
        print(f"  {d:<9} {a:5.2f} | {b}  {'same' if abs(a - float(b)) < 0.006 else 'DIFFERENT - check inputs'}")
print("\nCaption note: GEC values from CGWB & GWD-UP (2024); matching Sy is a consistency diagnostic, not a measured Sy.")
