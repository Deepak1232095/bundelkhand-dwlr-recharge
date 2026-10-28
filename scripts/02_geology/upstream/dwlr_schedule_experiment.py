"""
MONITORING-SCHEDULE EXPERIMENT  (synthetic sampling of verified DWLR hydrographs)
=================================================================================
Reference = daily DWLR depth (core 102 stations, monsoons 2023 + 2024), 15-day smoothed for the
"true" seasonal rise / peak. Each schedule samples the RAW daily series (one reading = median of
+-1 day around the date, like a field visit), then estimates rise and peak date.

Schedules
  S2_current   20-May, 10-Nov                                 (pre/post practice)
  S4_cgwb      15-Jan, 20-May, 20-Aug, 10-Nov                 (4x/yr manual network)
  S2_late_sep  20-May, 26-Sep
  S3           20-May, 26-Sep, 10-Nov
  M12          15th of every month
  F24          1st + 15th of every month
  MON_FOCUS    monthly + 1st/15th in Jul-Oct

Outputs per schedule: capture of true rise, peak-date error, and DECISION effects:
  - Spearman of station ranking (est vs true rise) per year
  - % stations changing tercile class (low/mid/high rise)
Holdout: best single post-monsoon date chosen on one year (or 4 districts) and scored on the other.

  Requires: pandas numpy scipy
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

QC = r"DWLR_QC"
DAILY = os.path.join(QC, "dwlr_daily_depth_abs.csv")
SIGN = os.path.join(QC, "dwlr_sign_smoothness.csv")
ROB = os.path.join(QC, "dwlr_testA_robustness.csv")
YEARS = [2023, 2024]

sg = pd.read_csv(SIGN)
CORE = set(sg[sg.sign_verdict.isin(["single sign (fine)", "sign errors -> |level| fixes it"])].SID)
dd = pd.read_csv(DAILY, parse_dates=["T"])
dd = dd[dd.SID.isin(CORE)]
DIST = {s: s.split("|")[0] for s in CORE}

# noise floor per station (dry-season placebo, 15-d window) -> "clear signal" flag
rb = pd.read_csv(ROB)
rb = rb[rb.WIN == 15]
floor = rb.groupby("SID").PLACEBO_RISE.median()


def sched(yr):
    md = lambda m, d: pd.Timestamp(yr, m, d)
    monthly = [md(m, 15) for m in range(1, 13)]
    fort = [md(m, d) for m in range(1, 13) for d in (1, 15)]
    return {
        "S2_current": [md(5, 20), md(11, 10)],
        "S4_cgwb": [md(1, 15), md(5, 20), md(8, 20), md(11, 10)],
        "S2_late_sep": [md(5, 20), md(9, 26)],
        "S3": [md(5, 20), md(9, 26), md(11, 10)],
        "M12": monthly,
        "F24": fort,
        "MON_FOCUS": sorted(set(monthly + [md(m, d) for m in (7, 8, 9, 10) for d in (1, 15)])),
    }


def sample(z, t):
    w = z[t - pd.Timedelta(days=1):t + pd.Timedelta(days=1)]
    return w.median() if len(w) else np.nan


def estimate(z, dates, yr):
    s = pd.Series({t: sample(z, t) for t in dates}).dropna()
    mon = s[(s.index >= pd.Timestamp(yr, 7, 1)) & (s.index <= pd.Timestamp(yr, 11, 30))]
    if mon.empty:
        return np.nan, pd.NaT
    tp = mon.idxmin()
    pre = s[(s.index >= pd.Timestamp(yr, 4, 1)) & (s.index < tp)]
    if pre.empty:
        return np.nan, pd.NaT
    return pre.max() - mon.min(), tp


rows = []
post_grid = pd.date_range("2000-08-02", "2000-12-15", freq="5D")   # includes 26-Sep and 10-Nov
grid_rows = []
for sid, g in dd.groupby("SID"):
    z = g.set_index("T")["DEPTH_ABS"].sort_index()
    z = z[~z.index.duplicated()].asfreq("D")
    zr = z.interpolate(limit=7).rolling(15, center=True, min_periods=7).median()
    for yr in YEARS:
        mon = zr[f"{yr}-07-01":f"{yr}-10-31"].dropna()
        if len(mon) < 100:
            continue
        tp = mon.idxmin()
        pre = zr[f"{yr}-05-01":tp].dropna()
        if len(pre) < 20:
            continue
        true_rise = pre.max() - mon.min()
        if true_rise <= 0.05:
            continue
        fl = floor.get(sid, rb.PLACEBO_RISE.median())
        clear = true_rise > 2 * max(fl, 0.3)
        for name, dates in sched(yr).items():
            er, et = estimate(z, dates, yr)
            rows.append({"SID": sid, "DIST": DIST[sid], "YEAR": yr, "SCHED": name, "N_OBS": len(dates),
                         "TRUE_RISE": true_rise, "TRUE_PEAK": tp, "EST_RISE": er,
                         "PEAK_ERR_D": (et - tp).days if pd.notna(et) else np.nan, "CLEAR": clear})
        zmay = sample(z, pd.Timestamp(yr, 5, 20))
        for md in post_grid:
            t = pd.Timestamp(yr, md.month, md.day)
            grid_rows.append({"SID": sid, "DIST": DIST[sid], "YEAR": yr, "CLEAR": clear, "DATE": md.strftime("%d-%b"),
                              "CAP": (zmay - sample(z, t)) / true_rise})

R = pd.DataFrame(rows); R["CAPTURE"] = R.EST_RISE / R.TRUE_RISE
G = pd.DataFrame(grid_rows)
R.to_csv(os.path.join(QC, "schedule_experiment.csv"), index=False)


def decision_effects(df):
    out = []
    for yr, q in df.groupby("YEAR"):
        q = q.dropna(subset=["EST_RISE"])
        if len(q) < 10:
            continue
        rho = stats.spearmanr(q.TRUE_RISE, q.EST_RISE)[0]
        tc = pd.qcut(q.TRUE_RISE.rank(method="first"), 3, labels=False)
        ec = pd.qcut(q.EST_RISE.rank(method="first"), 3, labels=False)
        out.append((rho, 100 * (tc != ec).mean()))
    return np.mean([o[0] for o in out]) if out else np.nan, np.mean([o[1] for o in out]) if out else np.nan


def best_date(df):
    m = df.groupby("DATE").CAP.median()
    return m.idxmax(), m.max()


print("=" * 72)
print("VERDICT")
print("=" * 72)
for label, sub in (("ALL", R), ("CLEAR", R[R.CLEAR])):
    print(f"\n[{label}] station-monsoons {sub.groupby(['SID', 'YEAR']).ngroups} | stations {sub.SID.nunique()}")
    print(f"  {'schedule':<12}{'obs/yr':>7}{'capture med':>12}{'IQR':>14}{'|peak err| d':>14}{'rank rho':>10}{'%class chg':>11}")
    order = R.groupby("SCHED").N_OBS.first().sort_values().index
    for s in order:
        q = sub[sub.SCHED == s]
        rho, chg = decision_effects(q)
        print(f"  {s:<12}{q.N_OBS.iloc[0]:>7}{q.CAPTURE.median():>12.2f}"
              f"{q.CAPTURE.quantile(.25):>7.2f}-{q.CAPTURE.quantile(.75):<6.2f}"
              f"{q.PEAK_ERR_D.abs().median():>14.0f}{rho:>10.2f}{chg:>11.0f}")

print("\nHOLDOUT: best single post-monsoon date (with 20-May pre reading), clear-signal cases")
Gc = G[G.CLEAR]
for tr, te in ((2023, 2024), (2024, 2023)):
    d, _ = best_date(Gc[Gc.YEAR == tr])
    te_best, te_cap = best_date(Gc[Gc.YEAR == te])
    cap_on_test = Gc[(Gc.YEAR == te) & (Gc.DATE == d)].CAP.median()
    nov = Gc[(Gc.YEAR == te) & (Gc.DATE == "10-Nov")].CAP.median()
    print(f"  train {tr} -> date {d} | test {te}: capture {cap_on_test:.2f} (test-optimal {te_best} {te_cap:.2f}; 10-Nov {nov:.2f})")
for dname in sorted(Gc.DIST.unique()):
    d, _ = best_date(Gc[Gc.DIST != dname])
    q = Gc[Gc.DIST == dname]
    print(f"  leave-out {dname:<9} -> date {d} | held-out capture {q[q.DATE == d].CAP.median():.2f} "
          f"vs 10-Nov {q[q.DATE == '10-Nov'].CAP.median():.2f} (n {q.SID.nunique()})")
