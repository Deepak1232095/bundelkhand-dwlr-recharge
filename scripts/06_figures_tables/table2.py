"""
TABLE 2 (rebuilt): monsoon rise and recharge by geology (2a) and by district (2b), ONE definition throughout
==========================================================================================================
Definition (Methods, Eq. 8): station recharge R = 1000 x Sy x Delta H (seasonal MRC-WTF rise, GROSS_C), with the
GEC (1997) Sy of the station's class: hard rock 3 % (1.5-4 %), alluvium 6 % (4-8 %). % of rain = 100 R / P_W.
Group values = median over stations; [ ] = bootstrap 95 % CI of the median (2000 resamples, seed 20261013).
The old Table 2 mixed this seasonal definition (Delta H column) with sums of monthly recharge (mm column, as used
for the maps), so Delta H x Sy did not give the recharge shown. The monthly-sum values are printed at the end for
comparison only (they belong to Figs. 8-9).
Input : T1_station_recharge.csv (run_Q123\\DWLR_QC\\final\\ first, then paper3_final\\), optional T6_monthly_station_year.csv
Output: paper3_final\\Table2a_geology.csv, Table2b_district.csv.   Run:  %run table2.py   and 
"""
import os
import sys
import numpy as np
import pandas as pd

BASE = os.environ.get("TEST_BASE", r"C:\Users\IITI")
FINAL = os.path.join(BASE, "paper3_final")
T1P = next((p for p in (os.path.join(BASE, "run_Q123", "DWLR_QC", "final", "T1_station_recharge.csv"),
                        os.path.join(FINAL, "T1_station_recharge.csv")) if os.path.exists(p)), None)
T6P = os.path.join(FINAL, "T6_monthly_station_year.csv")
SY = {"HARD_ROCK": (0.03, 0.015, 0.04), "ALLUVIUM": (0.06, 0.04, 0.08)}
N_BOOT = 2000
rng = np.random.default_rng(20261013)
if T1P is None:
    sys.exit("[STOP] T1_station_recharge.csv not found")
T = pd.read_csv(T1P)
T["DIST"] = T.DIST.astype(str).str.upper().str.strip()
T = T[np.isfinite(T.GROSS_C) & (T.GROSS_C > 0)].copy()
for k, i in (("R", 0), ("R_lo", 1), ("R_hi", 2)):
    T[k] = T.CLASS.map(lambda c: SY[c][i] if c in SY else np.nan) * 1000 * T.GROSS_C
for k in ("R", "R_lo", "R_hi"):
    T[k.replace("R", "PCT")] = 100 * T[k] / T.RAIN


def med_ci(x, f="{:.1f}"):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 3:
        return f.format(np.median(x)) if len(x) else "–"
    b = np.median(rng.choice(x, (N_BOOT, len(x))), axis=1)
    lo, hi = np.percentile(b, [2.5, 97.5])
    return f"{f.format(np.median(x))} [{f.format(lo)}–{f.format(hi)}]"


def m(x, f="{:.0f}"):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return f.format(np.median(x)) if len(x) else "–"


rows_a = []
for c, lab in (("ALLUVIUM", "Alluvium"), ("HARD_ROCK", "Hard rock")):
    q = T[T.CLASS == c]
    cen, lo, hi = SY[c]
    rows_a.append({"Group": lab, "n": len(q), "ΔH per unit Sy (m)": med_ci(q.GROSS_C),
                   "Sy (%)": f"{100 * cen:g} ({100 * lo:g}–{100 * hi:g})",
                   "Recharge (mm)": f"{m(q.R)} ({m(q.R_lo)}–{m(q.R_hi)})",
                   "% of Jun–Oct rain": f"{m(q.PCT)} ({m(q.PCT_lo)}–{m(q.PCT_hi)})"})
rows_a.append({"Group": "All stations", "n": len(T), "ΔH per unit Sy (m)": med_ci(T.GROSS_C), "Sy (%)": "by class",
               "Recharge (mm)": m(T.R), "% of Jun–Oct rain": m(T.PCT)})
A = pd.DataFrame(rows_a)
rows_b = []
for d, q in T.groupby("DIST"):
    qa = q[q.CLASS.isin(SY)]
    mix = f"{int((q.CLASS == 'ALLUVIUM').sum())} / {int((q.CLASS == 'HARD_ROCK').sum())}" + \
          (f" / {int((~q.CLASS.isin(SY)).sum())}" if (~q.CLASS.isin(SY)).any() else "")
    rows_b.append({"District": d.title(), "n (alluvium / hard rock[ / unclassified])": mix,
                   "ΔH per unit Sy (m)": med_ci(q.GROSS_C), "n with Sy": len(qa),
                   "Recharge (mm), class Sy": f"{m(qa.R)} ({m(qa.R_lo)}–{m(qa.R_hi)})",
                   "% of Jun–Oct rain": f"{m(qa.PCT)} ({m(qa.PCT_lo)}–{m(qa.PCT_hi)})"})
B = pd.DataFrame(rows_b)
A.to_csv(os.path.join(FINAL, "Table2a_geology.csv"), index=False, encoding="utf-8-sig")
B.to_csv(os.path.join(FINAL, "Table2b_district.csv"), index=False, encoding="utf-8-sig")

pd.set_option("display.width", 220); pd.set_option("display.max_columns", 20)
print("=" * 90); print("TABLE 2 rebuilt"); print("=" * 90)
print(f"T1: {T1P}")
print("\nTable 2a  by geology (seasonal definition; recharge = 1000 x Sy x dH):"); print(A.to_string(index=False))
print("\nTable 2b  by district (station Sy by class; brackets = Sy range):"); print(B.to_string(index=False))
chk = T[T.CLASS.isin(SY)].groupby("CLASS").apply(lambda q: abs(np.median(q.R) - 1000 * SY[q.name][0] * np.median(q.GROSS_C)))
print(f"\ncheck within class: median(R) = Sy x median(dH)? max |diff| {chk.max():.2e} mm")
if os.path.exists(T6P):
    D = pd.read_csv(T6P); D["DIST"] = D.DIST.astype(str).str.upper()
    S = D.groupby(["SID", "DIST", "CLASS", "MONTH"]).R_mm.median().reset_index()
    S = S.groupby(["SID", "DIST", "CLASS"]).agg(R=("R_mm", "sum"), n=("MONTH", "nunique")).reset_index()
    S = S[S.n == 5]
    print("\nfor comparison only - sum of monthly recharge (definition of the maps, Figs. 8-9), median mm:")
    print("  by class:", S.groupby("CLASS").R.median().round(0).to_dict())
    print("  by district:", S.groupby("DIST").R.median().round(0).to_dict())
