"""
TEST A ROBUSTNESS  —  is "May/Nov captures only ~40% of the monsoon rise" real?
==============================================================================
Checks (core 102 stations, monsoons 2023 + 2024):
 1. Smoothing window 7 / 15 / 31 days      -> sensor noise can inflate max-min "true rise"
 2. CGWB-like SINGLE-DAY sampling          -> 20 May (+-3 d) and 10 Nov (+-3 d) instead of monthly medians
 3. Placebo in the dry season (Jan-May)    -> same "low-before-peak minus peak" metric where there is
                                              no recharge; tells the noise floor of the method
 4. By district and by year
 5. Does capture depend on depth / peak timing? (Spearman)

  Requires: pandas numpy scipy
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

DWLR = r"EDIT\ME\UPGWD_Bundelkhand_DWLR.csv"
QC = os.path.join(os.path.dirname(DWLR), "DWLR_QC")
SIGN = os.path.join(QC, "dwlr_sign_smoothness.csv")
YEARS = [2023, 2024]

sg = pd.read_csv(SIGN)
CORE = set(sg[sg.sign_verdict.isin(["single sign (fine)", "sign errors -> |level| fixes it"])].SID)

d = pd.read_csv(DWLR)
d["T"] = pd.to_datetime(d["Data Acquisition Time"].astype(str), format="%d-%m-%Y %H:%M", errors="coerce")
d["LVL"] = pd.to_numeric(d["Groundwater Level Telemetry 6 Hourly (meter)"], errors="coerce")
d["SID"] = (d["District"].astype(str).str.strip() + "|" + d["Block"].astype(str).str.strip() + "|" +
            d["Station"].astype(str).str.strip())
d = d[d.SID.isin(CORE)]
d.loc[d["LVL"] <= -900, "LVL"] = np.nan
d = d.dropna(subset=["T", "LVL"]).sort_values(["SID", "T"]).drop_duplicates(["SID", "T"])
a = d["LVL"].abs()
med = a.groupby(d["SID"]).transform(lambda s: s.rolling(7, center=True, min_periods=3).median())
d = d[((a - med).abs() <= 5.0) & (d["LVL"] != 0)].copy()
d["Z"] = d["LVL"].abs()
daily = d.set_index("T").groupby("SID")["Z"].resample("D").median().dropna()
DIST = d.groupby("SID")["District"].first()


def rise(zr, t0, t1, p0, _unused):
    """shallowest level in [t0,t1]; deepest level between p0 and that peak -> (rise, peak date, low)."""
    w = zr[(zr.index >= t0) & (zr.index <= t1)]
    if len(w) < 0.6 * ((pd.Timestamp(t1) - pd.Timestamp(t0)).days + 1):
        return np.nan, pd.NaT, np.nan
    tp = w.idxmin()
    pre = zr[(zr.index >= p0) & (zr.index <= tp)]
    if len(pre) < 20:
        return np.nan, pd.NaT, np.nan
    return pre.max() - w.min(), tp, pre.max()


def day_val(z, date, tol=3):
    w = z[(z.index >= pd.Timestamp(date) - pd.Timedelta(days=tol)) & (z.index <= pd.Timestamp(date) + pd.Timedelta(days=tol))]
    return w.median() if len(w) else np.nan


rows = []
for sid in daily.index.get_level_values(0).unique():
    z = daily.loc[sid]
    for win in (7, 15, 31):
        zr = z.rolling(win, center=True, min_periods=max(3, win // 2)).median()
        for yr in YEARS:
            tr, tp, zlow = rise(zr, f"{yr}-07-01", f"{yr}-10-31", f"{yr}-05-01", f"{yr}-10-31")
            # placebo: dry season, "peak" searched in Mar-May, "low" from Jan
            pl, _, _ = rise(zr, f"{yr}-03-01", f"{yr}-05-31", f"{yr}-01-01", f"{yr}-05-31")
            may_m = z[(z.index.year == yr) & (z.index.month == 5)]
            nov_m = z[(z.index.year == yr) & (z.index.month == 11)]
            bi_month = may_m.median() - nov_m.median() if len(may_m) >= 10 and len(nov_m) >= 10 else np.nan
            bi_day = day_val(z, f"{yr}-05-20") - day_val(z, f"{yr}-11-10")
            rows.append({"SID": sid, "DIST": DIST[sid], "YEAR": yr, "WIN": win, "TRUE_RISE": tr,
                         "PEAK_DOY": tp.dayofyear if pd.notna(tp) else np.nan, "Z_LOW": zlow,
                         "PLACEBO_RISE": pl, "BI_MONTH": bi_month, "BI_DAY": bi_day})
r = pd.DataFrame(rows)
r["CAP_MONTH"] = np.where(r.TRUE_RISE > 0.05, r.BI_MONTH / r.TRUE_RISE, np.nan)
r["CAP_DAY"] = np.where(r.TRUE_RISE > 0.05, r.BI_DAY / r.TRUE_RISE, np.nan)
r.to_csv(os.path.join(QC, "dwlr_testA_robustness.csv"), index=False)


def summ(s, col):
    s = s[col].dropna()
    return f"{s.median():.2f} (IQR {s.quantile(.25):.2f}-{s.quantile(.75):.2f}, n={len(s)})"


print("=" * 72)
print("VERDICT")
print("=" * 72)
print("1+2. by smoothing window:")
for win, g in r.groupby("WIN"):
    print(f"  {win:>2} d | true rise {summ(g, 'TRUE_RISE')} | placebo (dry) {summ(g, 'PLACEBO_RISE')}")
    print(f"       | capture monthly-median {summ(g, 'CAP_MONTH')} | capture single-day {summ(g, 'CAP_DAY')}"
          f" | share<0.5 (day) {100 * (g.CAP_DAY.dropna() < .5).mean():.0f}% | share<=0 {100 * (g.CAP_DAY.dropna() <= 0).mean():.0f}%")
g = r[r.WIN == 15]
excess = (g.TRUE_RISE - g.PLACEBO_RISE).dropna()
print(f"3. true rise minus dry-season placebo (15 d): median {excess.median():.2f} m | "
      f"Wilcoxon p {stats.wilcoxon(excess).pvalue:.2g}")
print("4. 15-d window, single-day capture by district / year:")
print(g.groupby("DIST").agg(n=("CAP_DAY", "count"), true_rise=("TRUE_RISE", "median"),
                            bi_day=("BI_DAY", "median"), capture=("CAP_DAY", "median")).round(2).to_string())
print(g.groupby("YEAR").agg(n=("CAP_DAY", "count"), true_rise=("TRUE_RISE", "median"),
                            capture=("CAP_DAY", "median"), peak_doy=("PEAK_DOY", "median")).round(2).to_string())
for x in ("Z_LOW", "PEAK_DOY", "TRUE_RISE"):
    q = g.dropna(subset=[x, "CAP_DAY"])
    rho, p = stats.spearmanr(q[x], q.CAP_DAY)
    print(f"5. capture vs {x}: spearman {rho:+.2f} (p {p:.3f}, n {len(q)})")
