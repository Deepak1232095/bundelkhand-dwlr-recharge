"""
TABLE 2 district rows + Table 3 last column (seasonal ΔH; R = 1000 x Sy(class) x GROSS_C; median over stations)
Run:  %run table2_districts.py      
"""
import os, sys
import numpy as np, pandas as pd

BASE = os.environ.get("TEST_BASE", r"C:\Users\IITI")
T1P = next((p for p in (os.path.join(BASE, "run_Q123", "DWLR_QC", "final", "T1_station_recharge.csv"),
                        os.path.join(BASE, "paper3_final", "T1_station_recharge.csv")) if os.path.exists(p)), None)
if T1P is None:
    sys.exit("[STOP] T1_station_recharge.csv not found")
SY = {"ALLUVIUM": 0.06, "HARD_ROCK": 0.03}
T = pd.read_csv(T1P).drop_duplicates("SID")
T["DIST"] = T.DIST.astype(str).str.upper().str.strip()
T = T[np.isfinite(T.GROSS_C) & (T.GROSS_C > 0)].copy()
T["R"] = T.CLASS.map(SY) * 1000 * T.GROSS_C
T["PCT"] = 100 * T.R / T.RAIN

rows = []
for c, lab in (("ALLUVIUM", "Alluvium"), ("HARD_ROCK", "Hard rock")):
    q = T[T.CLASS == c]
    rows.append([lab, len(q), f"{q.GROSS_C.median():.1f}", f"{100 * SY[c]:g}", f"{q.R.median():.0f}", f"{q.PCT.median():.0f}"])
rows.append(["All stations", len(T), f"{T.GROSS_C.median():.1f}", "–", "–", "–"])
for d, q in T.groupby("DIST"):
    qs = q[q.CLASS.isin(SY)]
    nA, nH = int((q.CLASS == "ALLUVIUM").sum()), int((q.CLASS == "HARD_ROCK").sum())
    sy = "6" if nH == 0 else ("3" if nA == 0 else f"6 / 3 ({nA} A, {nH} H)")
    rows.append([d.title(), len(q), f"{q.GROSS_C.median():.1f}", sy, f"{qs.R.median():.0f}", f"{qs.PCT.median():.0f}"])
O = pd.DataFrame(rows, columns=["Group", "n", "ΔH per unit Sy (m)", "Sy (%)", "Recharge (mm)", "% of Jun–Oct rain"])
pd.set_option("display.width", 200)
print("=" * 80); print("VERDICT  table2_districts"); print("=" * 80)
print(f"T1: {T1P}\n"); print(O.to_string(index=False))
print("\nTable 3 last column (This study, mm) = Recharge (mm) of the district rows above")
