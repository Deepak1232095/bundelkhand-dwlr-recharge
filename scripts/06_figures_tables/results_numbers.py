"""
RESULTS NUMBERS: prints the values still missing in the Results text (yellow [A1]...[A7] marks), from existing outputs.
Nothing is re-estimated except A5 synchrony p (re-run of the same random-shift test) and A6 (new within-district ranks).
  A1 audit: station-days removed by Q1/Q2/Q3                      DWLR_QC\\audit\\audit_station.csv
  A2 Q123 MRC: stations used / rejected, median slope b and sigma  run_Q123\\DWLR_QC\\wtf_mrc\\mrc_parameters.csv
  A3 monthly recharge per class (median [95% CI], % of rain)       paper3_final\\T6_monthly_recharge_summary.csv
  A4 district n, median Delta H, median recharge, % of rain        T1_station_recharge.csv + T6_monthly_station_year.csv
  A5 episode rain-recharge Spearman + synchrony test               run_Q123\\DWLR_QC\\wtf_mrc\\diagnostics\\diag_episodes.csv
  A6 recharge vs river distance with ranks within districts        paper3_final\\T10b_station_river_distance.csv
  A7 leave-one-out skill per district                              paper3_final\\T8*.csv
Run:  %run results_numbers.py      (from C:\\Users\\IITI)   
"""
import os
import glob
import numpy as np
import pandas as pd
from scipy import stats

BASE = os.environ.get("TEST_BASE", r"C:\Users\IITI")
FINAL = os.path.join(BASE, "paper3_final")
RUN = os.path.join(BASE, "run_Q123", "DWLR_QC")
F = {"audit": os.path.join(BASE, "DWLR_QC", "audit", "audit_station.csv"),
     "mrc": os.path.join(RUN, "wtf_mrc", "mrc_parameters.csv"),
     "t6s": os.path.join(FINAL, "T6_monthly_recharge_summary.csv"),
     "t6": os.path.join(FINAL, "T6_monthly_station_year.csv"),
     "t1": next((p for p in (os.path.join(RUN, "final", "T1_station_recharge.csv"),
                             os.path.join(FINAL, "T1_station_recharge.csv")) if os.path.exists(p)), ""),
     "eps": os.path.join(RUN, "wtf_mrc", "diagnostics", "diag_episodes.csv"),
     "t10b": os.path.join(FINAL, "T10b_station_river_distance.csv")}
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 20)
print("=" * 90); print("RESULTS NUMBERS"); print("=" * 90)


def have(k):
    ok = bool(F[k]) and os.path.exists(F[k])
    if not ok:
        print(f"  [missing] {F[k] or k}")
    return ok


print("\n[A1] audit (core stations with an MRC):")
if have("audit"):
    A = pd.read_csv(F["audit"])
    tot = int(A.Q1_code.sum() + A.Q2_deep.sum() + A.Q3_step.sum())
    print(f"  Q1 {int(A.Q1_code.sum())} days ({int((A.Q1_code > 0).sum())} stations) | Q2 {int(A.Q2_deep.sum())} days "
          f"({int((A.Q2_deep > 0).sum())}) | Q3 {int(A.Q3_step.sum())} days ({int((A.Q3_step > 0).sum())}) | total {tot} of "
          f"{int(A.n_days.sum())} station-days ({100 * tot / A.n_days.sum():.2f} %) | stations {len(A)}")

print("\n[A2] primary MRC, Q123 run:")
if have("mrc"):
    M = pd.read_csv(F["mrc"]); M = M[M.MRC == "PRIMARY"]
    ok = M[M.note == "ok"]
    print(f"  stations {M.SID.nunique()} | used {len(ok)} | slope >= 0 {int((M.b >= 0).sum())} | too few dry days "
          f"{int((M.note == 'too few dry days').sum())}")
    print(f"  used: median slope b {ok.b.median():.4f} /d | median sigma {ok.sigma.median():.4f} m/d")

print("\n[A3] monthly recharge per class:")
if have("t6s"):
    T = pd.read_csv(F["t6s"])
    print(T[T.group.isin(["HARD_ROCK", "ALLUVIUM"])].to_string(index=False))

print("\n[A4] districts:")
if have("t1") and have("t6"):
    T1 = pd.read_csv(F["t1"]); T1["DIST"] = T1.DIST.astype(str).str.upper()
    D = pd.read_csv(F["t6"]); D["DIST"] = D.DIST.astype(str).str.upper()
    S = D.groupby(["SID", "DIST", "MONTH"]).agg(R=("R_mm", "median"), P=("RAIN_mm", "median")).reset_index()
    S = S.groupby(["SID", "DIST"]).agg(R=("R", "sum"), P=("P", "sum"), n=("MONTH", "nunique")).reset_index()
    S = S[S.n == 5]
    S["PCT"] = 100 * S.R / S.P
    out = pd.DataFrame({"n_T1": T1.groupby("DIST").size(), "dH_median_m": T1.groupby("DIST").GROSS_C.median(),
                        "n_recharge": S.groupby("DIST").size(), "R_median_mm": S.groupby("DIST").R.median(),
                        "pct_rain_median": S.groupby("DIST").PCT.median()}).round(2)
    print(out.to_string())
    print("  (R_median_mm should reproduce 353 / 541 / 143 / 322 / 160 mm of Table 2)")

print("\n[A5] episodes:")
if have("eps"):
    E = pd.read_csv(F["eps"], parse_dates=["start", "end"])
    for lab, q in (("all episodes", E), ("FULL years", E[E.FULLYR.astype(bool)])):
        q = q.dropna(subset=["RAIN_mm", "RECH_m"])
        r, p = stats.spearmanr(q.RAIN_mm, q.RECH_m)
        w = q.groupby("SID").apply(lambda g: stats.spearmanr(g.RAIN_mm, g.RECH_m)[0] if len(g) >= 5 else np.nan)
        print(f"  {lab}: pooled Spearman rho {r:+.2f} (p {p:.1e}, n {len(q)}) | within-station median {w.median():+.2f} "
              f"({int(w.notna().sum())} stations)")
    RF = E[E.RAIN_FREE.astype(bool) & E.FULLYR.astype(bool) & E.ABOVE_NOISE.astype(bool)].copy()

    def share(df):
        h = [bool(((df.DIST == r.DIST) & (df.YEAR == r.YEAR) & (df.SID != r.SID) &
                   ((df.start - r.start).dt.days.abs() <= 2)).any()) for r in df.itertuples()]
        return np.mean(h) if h else np.nan
    if len(RF) > 1:
        obs = share(RF)
        rng = np.random.default_rng(20261013)
        null = []
        for _ in range(200):
            sh = RF.copy()
            off = {s: int(rng.integers(-60, 61)) for s in sh.SID.unique()}
            sh["start"] = sh.start + pd.to_timedelta(sh.SID.map(off), unit="D")
            null.append(share(sh))
        null = np.array(null)
        p = (np.sum(null >= obs) + 1) / (len(null) + 1)
        print(f"  synchrony (rain-free, above noise, FULL years, +-2 d): observed {obs:.2f} | null median "
              f"{np.median(null):.2f} [P95 {np.percentile(null, 95):.2f}] | p = {p:.3f} | episodes {len(RF)}")

print("\n[A6] recharge vs river distance, ranks within districts (districts with >= 3 stations of the class):")
if have("t10b"):
    R = pd.read_csv(F["t10b"])
    for col in [c for c in R.columns if c.startswith("dist_km_ge")]:
        for cl, q in R.groupby("CLASS"):
            q = q.dropna(subset=[col, "R"])
            q = q[q.groupby("DIST").R.transform("size") >= 3].copy()
            if len(q) < 5:
                continue
            q["rd"] = q.groupby("DIST")[col].rank(pct=True)
            q["rr"] = q.groupby("DIST").R.rank(pct=True)
            r, p = stats.spearmanr(q.rd, q.rr)
            print(f"  {col:<15} {cl:<9} rho {r:+.2f} p {p:.3f} n {len(q)} districts {q.DIST.nunique()}")

print("\n[A7] leave-one-out skill files:")
for f in sorted(glob.glob(os.path.join(FINAL, "T8*.csv"))):
    print(f"  {os.path.basename(f)}"); print(pd.read_csv(f).to_string(index=False))
