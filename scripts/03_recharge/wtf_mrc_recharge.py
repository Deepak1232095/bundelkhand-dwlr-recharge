"""
MONSOON RECHARGE FROM DAILY DWLR: WATER-TABLE FLUCTUATION WITH A MASTER RECESSION CURVE (+ EPISODES)
===================================================================================================
Method (decided before running; references in the paper):
  h(t)   water level = -|depth| (m, up = positive), daily, from DWLR_QC/dwlr_daily_depth_abs.csv (core stations)
  s(t)   3-day centred rolling MEAN of h (gaps <= GAP_FILL days linearly filled, never longer).
         (A rolling median was rejected: on dummy data it created exact-zero rates and flattened the MRC.)
  MRC    Heppner & Nimmo (2005) master recession curve: rate r = s(t+1)-s(t) regressed on s(t) over DRY days
         (no rain >= RAIN_THR mm in the previous LAG_DAYS days nor the next day; both signs kept, because
         keeping only r < 0 truncates the noise and flattens the slope), robust Theil-Sen line r = a + b*s.
         Primary MRC: dry days Aug-Nov (post-monsoon drainage, before rabi pumping).
         Sensitivity MRC: dry days Aug-Feb (includes rabi pumping -> steeper decline, more 'recharge').
         Physical check: slope b must be < 0, otherwise the station is skipped (MRC not identifiable).
         Predicted decline f(s) = min(0, a + b*s) (never a predicted rise).
  Daily recharge per unit Sy (fixed-time-step MRC-WTF):  d(t) = max(0, [s(t+1)-s(t)] - f(s(t)))   [m]
  GROSS  = sum of d over the monsoon window 1 Jun - 31 Oct                                         [m]
  NOISE  sigma = 1.4826 x MAD of MRC residuals on the fit days. Summing max(0, noise) adds sigma/sqrt(2*pi)
         per day WITHOUT real recharge, so GROSS_C = GROSS - sigma/sqrt(2*pi) x (window days with d <= EPS),
         EPS = max(2 sigma, 0.5 cm). (A Jan-Apr dry-window placebo was rejected: rabi pumping biases it.)
  Recharge (mm) = 1000 x Sy x GROSS_C, reported on a Sy GRID (1, 2, 5, 10 %) -- no class-specific Sy is assumed.
  Comparisons (same smoothed series):  PEAKLOW = max(depth 1 May..peak) - min(depth Jul-Oct)   (simple rise)
                                       BIANN   = depth(20 May) - depth(10 Nov)                   (GEC-style)
EPISODES (Nimmo et al. 2015 spirit): runs of days with d > EPS,
  merged if <= MERGE_GAP days apart; per episode: recharge (sum d), rain in [start-LAG_DAYS, end], max daily rain.
  Episodes use every available day, including partial monsoons (2022, 2025).
Season totals only for station-years with >= MIN_COVER of window days observed ("FULL"); others flagged.
Inference unit = station (median over its FULL years); bootstrap 95% CI over stations.
Run in Jupyter:  %run wtf_mrc_recharge.py      Requires: pandas numpy scipy xarray netCDF4
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

warnings.filterwarnings("ignore")

# ======================================================================================== CONFIG
QC = r"DWLR_QC"
DAILY = os.path.join(QC, "dwlr_daily_depth_abs.csv")
SIGN = os.path.join(QC, "dwlr_sign_smoothness.csv")
GEOL = os.path.join(QC, "regime_controls", "geology_v3", "station_geology_audit.csv")   # optional
RAIN = r"E:\up_bundel\up_bundel_rainfall_1990_2025.nc"
OUT = os.path.join(QC, "wtf_mrc")
YEARS = [2021, 2022, 2023, 2024, 2025]
GAP_FILL = 3
SMOOTH = 3
RAIN_THR, LAG_DAYS = 2.0, 5
MRC_SETS = {"PRIMARY": [8, 9, 10, 11], "AUG_FEB": [8, 9, 10, 11, 12, 1, 2]}
MIN_RECESSION_DAYS = 30
WIN = ((6, 1), (10, 31))
MIN_COVER = 0.8
MERGE_GAP = 2
SY_GRID = [0.01, 0.02, 0.05, 0.10]
N_BOOT, SEED = 2000, 20261013
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(SEED)

for f in (DAILY, SIGN, RAIN):
    if not os.path.exists(f):
        sys.exit(f"[STOP] not found: {f}  (run from the notebook folder)")

sg = pd.read_csv(SIGN)
core = sg[sg.sign_verdict.isin(["single sign (fine)", "sign errors -> |level| fixes it"])][["SID", "lat", "lon"]]
dd = pd.read_csv(DAILY, parse_dates=["T"])
dd = dd[dd.SID.isin(set(core.SID))]
cls = None
if os.path.exists(GEOL):
    g_ = pd.read_csv(GEOL)
    if {"SID", "CLASS"} <= set(g_.columns):
        cls = g_.set_index("SID").CLASS

# rain
ds = xr.open_dataset(RAIN)
var = [v for v in ds.data_vars if ds[v].ndim == 3][0]
la = next(c for c in ds.coords if "lat" in c.lower())
lo = next(c for c in ds.coords if "lon" in c.lower())
tm = next(c for c in ds.coords if "time" in c.lower())
R = ds[var].transpose(tm, la, lo).values.astype(float)
TT = pd.to_datetime(ds[tm].values).normalize()
LA, LO = np.meshgrid(ds[la].values, ds[lo].values, indexing="ij")
valid = ~np.all(np.isnan(R), axis=0)
ds.close()


def rain_at(lat, lon):
    d = np.hypot(LA - lat, LO - lon)
    d[~valid] = np.inf
    i, j = np.unravel_index(np.argmin(d), d.shape)
    s = pd.Series(R[:, i, j], index=TT)
    s = s[~s.index.duplicated()]
    return s.where(s >= 0)


def window(idx, y, w):
    return (idx >= pd.Timestamp(y, *w[0])) & (idx <= pd.Timestamp(y, *w[1]))


# ======================================================================================== per station
MRC_ROWS, YR_ROWS, EP_ROWS = [], [], []
for _, st in core.iterrows():
    g = dd[dd.SID == st.SID]
    if g.empty:
        continue
    dep = g.groupby("T").DEPTH_ABS.median()
    full = pd.date_range(dep.index.min(), dep.index.max(), freq="D")
    h = -dep.reindex(full)
    h = h.interpolate(limit=GAP_FILL, limit_area="inside")
    s = h.rolling(SMOOTH, center=True, min_periods=SMOOTH).mean()
    s[h.isna()] = np.nan                                        # never invent data in gaps
    rain = rain_at(st.lat, st.lon).reindex(full)
    rate = s.shift(-1) - s                                       # change from t to t+1
    wet_prev = rain.rolling(LAG_DAYS, min_periods=1).max()
    dry = (wet_prev < RAIN_THR) & (rain.shift(-1).fillna(0) < RAIN_THR)
    rec = dry & s.notna() & rate.notna()

    fits = {}
    for nm, months in MRC_SETS.items():
        m = rec & s.index.month.isin(months)
        row = {"SID": st.SID, "MRC": nm, "n_days": int(m.sum())}
        if m.sum() < MIN_RECESSION_DAYS:
            MRC_ROWS.append({**row, "note": "too few dry days"})
            continue
        b, a, b_lo, b_hi = stats.theilslopes(rate[m].values, s[m].values)
        res = rate[m].values - (a + b * s[m].values)
        sig = 1.4826 * np.median(np.abs(res - np.median(res)))
        ok = bool(b < 0)
        MRC_ROWS.append({**row, "a": a, "b": b, "b_lo": b_lo, "b_hi": b_hi, "sigma": sig,
                         "level_range_m": float(s[m].max() - s[m].min()),
                         "spearman_rate_level": stats.spearmanr(s[m].values, rate[m].values)[0],
                         "median_rate_m_per_d": float(rate[m].median()),
                         "note": "ok" if ok else "slope >= 0 (not physical) -> not used"})
        if ok:
            fits[nm] = (a, b, sig)
    if "PRIMARY" not in fits:
        continue

    for nm, (a, b, sig) in fits.items():
        f = np.minimum(0.0, a + b * s)                           # predicted decline (<= 0)
        dpos = (rate - f).clip(lower=0)                          # daily recharge per unit Sy
        eps = max(2 * sig, 0.005)
        bias = sig / np.sqrt(2 * np.pi)                          # E[max(0, noise)] per quiet day
        for y in YEARS:
            wm = window(full, y, WIN)
            if wm.sum() == 0:
                continue
            nwin = int(pd.Timestamp(y, *WIN[1]).dayofyear - pd.Timestamp(y, *WIN[0]).dayofyear + 1)
            cover = float(dpos[wm].notna().sum()) / nwin
            gross = float(dpos[wm].sum(min_count=1)) if dpos[wm].notna().any() else np.nan
            n_quiet = int((dpos[wm] <= eps).sum())                # noise bias only on days without recharge
            gross_c = max(gross - bias * n_quiet, 0) if np.isfinite(gross) else np.nan
            net = float((rate - f)[wm].sum(min_count=1)) if wm.any() else np.nan
            dep_s = -s
            mon = dep_s[window(full, y, ((7, 1), (10, 31)))].dropna()
            peaklow = biann = np.nan
            if len(mon) >= 60:
                tp = mon.idxmin()
                pre = dep_s[(full >= pd.Timestamp(y, 5, 1)) & (full <= tp)].dropna()
                if len(pre) >= 20:
                    peaklow = float(pre.max() - mon.min())
            dm, dn = dep_s.get(pd.Timestamp(y, 5, 20), np.nan), dep_s.get(pd.Timestamp(y, 11, 10), np.nan)
            if np.isfinite(dm) and np.isfinite(dn):
                biann = float(dm - dn)
            P = float(rain[wm].sum(min_count=1))
            YR_ROWS.append({"SID": st.SID, "DIST": st.SID.split("|")[0], "YEAR": y, "MRC": nm, "cover": cover,
                            "FULL": cover >= MIN_COVER, "GROSS_m": gross, "NOISE_CORR_m": gross - gross_c if np.isfinite(gross) else np.nan,
                            "GROSS_C_m": gross_c, "NET_MRC_m": net, "PEAKLOW_m": peaklow, "BIANN_m": biann,
                            "RAIN_JJASO_mm": P, "SIGMA": sig})
            if nm != "PRIMARY":
                continue
            # ---- episodes (primary MRC only)
            x = dpos[wm]
            on = (x > eps).astype(int)
            idx = np.where(on.values == 1)[0]
            if len(idx) == 0:
                continue
            groups, cur = [], [idx[0]]
            for k in idx[1:]:
                if k - cur[-1] <= MERGE_GAP + 1:
                    cur.append(k)
                else:
                    groups.append(cur); cur = [k]
            groups.append(cur)
            for gi in groups:
                t0, t1 = x.index[gi[0]], x.index[gi[-1]]
                rr = rain[(rain.index >= t0 - pd.Timedelta(days=LAG_DAYS)) & (rain.index <= t1)]
                EP_ROWS.append({"SID": st.SID, "DIST": st.SID.split("|")[0], "YEAR": y, "start": t0, "end": t1,
                                "days": (t1 - t0).days + 1, "RECH_m": float(x.loc[t0:t1].sum()),
                                "RAIN_mm": float(rr.sum()), "RAIN_MAX_mm": float(rr.max()) if len(rr) else np.nan})

M = pd.DataFrame(MRC_ROWS)
Y = pd.DataFrame(YR_ROWS)
E = pd.DataFrame(EP_ROWS)
if Y.empty:
    sys.exit("[STOP] no station-year could be processed")
if cls is not None:
    Y["CLASS"] = Y.SID.map(cls)
    E["CLASS"] = E.SID.map(cls) if len(E) else None
M.to_csv(os.path.join(OUT, "mrc_parameters.csv"), index=False)
Y.to_csv(os.path.join(OUT, "recharge_station_years.csv"), index=False)
E.to_csv(os.path.join(OUT, "recharge_episodes.csv"), index=False)


# ======================================================================================== summary
def ci(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return np.nan, np.nan, np.nan, len(x)
    b = np.median(rng.choice(x, (N_BOOT, len(x))), axis=1)
    return np.median(x), *np.percentile(b, [2.5, 97.5]), len(x)


def line(lab, x, fmt="{:.3f}"):
    m, lo_, hi_, n = ci(x)
    return f"  {lab:<52} {fmt.format(m)} [{fmt.format(lo_)}, {fmt.format(hi_)}]  n={n}"


P = Y[(Y.MRC == "PRIMARY") & Y.FULL].copy()
P["RATIO_GROSSC_PEAKLOW"] = P.GROSS_C_m / P.PEAKLOW_m
P["RATIO_BIANN_GROSSC"] = P.BIANN_m / P.GROSS_C_m
P["DH_PER_100MM"] = 100 * P.GROSS_C_m / P.RAIN_JJASO_mm
S = P.groupby("SID").agg(DIST=("DIST", "first"), GROSS_C_m=("GROSS_C_m", "median"), GROSS_m=("GROSS_m", "median"),
                         PEAKLOW_m=("PEAKLOW_m", "median"), BIANN_m=("BIANN_m", "median"),
                         R1=("RATIO_GROSSC_PEAKLOW", "median"), R2=("RATIO_BIANN_GROSSC", "median"),
                         DH100=("DH_PER_100MM", "median"), RAIN=("RAIN_JJASO_mm", "median"),
                         NC=("NOISE_CORR_m", "median"), n_years=("YEAR", "nunique"))
A = Y[(Y.MRC == "AUG_FEB") & Y.FULL][["SID", "YEAR", "GROSS_C_m"]].rename(columns={"GROSS_C_m": "GC_ALL"})
SENS = P.merge(A, on=["SID", "YEAR"], how="inner")

pd.set_option("display.width", 200)
print("=" * 90)
print("VERDICT  wtf_mrc_recharge")
print("=" * 90)
mp = M[M.MRC == "PRIMARY"]
bb = mp.b.dropna()
print(f"core stations {core.SID.nunique()} | with data {mp.SID.nunique()} | PRIMARY MRC used {int((mp.note == 'ok').sum())} | "
      f"too few dry days {int((mp.note == 'too few dry days').sum())} | slope >= 0 (rejected) {int((bb >= 0).sum())}")
print(f"PRIMARY MRC (dry days Aug-Nov): slope b median {bb.median():.4f} /day | dry days median {mp.n_days.median():.0f} | "
      f"level range median {mp.level_range_m.median():.2f} m | noise sigma median {mp.sigma.median():.4f} m/day | "
      f"Spearman(rate, level) median {mp.spearman_rate_level.median():+.2f}")
ma = M[M.MRC == "AUG_FEB"]
print(f"AUG_FEB MRC (sensitivity): slope b median {ma.b.median():.4f} | slope >= 0 {int((ma.b >= 0).sum())} of {int(ma.b.notna().sum())}")
print("\nstation-years by year (PRIMARY MRC): processed / FULL (>= 80% of Jun-Oct observed)")
print(Y[Y.MRC == "PRIMARY"].groupby("YEAR").agg(processed=("SID", "nunique"), full=("FULL", "sum")).to_string())
print(f"\n######## SEASON TOTALS (FULL years only; unit = station, median over its years) ########")
print(line("GROSS recharge per unit Sy, raw (m)", S.GROSS_m))
print(line("noise correction subtracted (m)", S.NC))
print(line("GROSS_C recharge per unit Sy, noise-corrected (m)", S.GROSS_C_m))
print(line("simple seasonal rise PEAKLOW (m)", S.PEAKLOW_m))
print(line("GEC-style May20-Nov10 rise BIANN (m)", S.BIANN_m))
print(line("GROSS_C / PEAKLOW  (MRC adds drainage during monsoon)", S.R1, "{:.2f}"))
print(line("BIANN / GROSS_C  (share of MRC recharge seen by May/Nov)", S.R2, "{:.2f}"))
print(line("GROSS_C per 100 mm Jun-Oct rain (m)", S.DH100))
print(line("Jun-Oct rain (mm)", S.RAIN, "{:.0f}"))
print("\n  recharge (mm) for a Sy grid = 1000 x Sy x GROSS_C  (median over stations; NOT a Sy claim):")
gc = S.GROSS_C_m.median()
pr = (S.GROSS_C_m / S.RAIN).median()
for sy in SY_GRID:
    print(f"    Sy {100 * sy:4.0f}% -> {1000 * sy * gc:6.0f} mm | recharge/rain {100 * 1000 * sy * pr:5.1f}%")
print("\n  by district (median GROSS_C m | GROSS_C/PEAKLOW | BIANN/GROSS_C | m per 100 mm | n stations):")
for d, q in S.groupby("DIST"):
    print(f"    {d:<9} {q.GROSS_C_m.median():6.2f} | {q.R1.median():5.2f} | {q.R2.median():5.2f} | {q.DH100.median():6.3f} | {len(q)}")
if cls is not None:
    S["CLASS"] = S.index.map(cls)
    print("  by geology class [descriptive; class ~ district]:")
    for c, q in S.groupby("CLASS"):
        print(f"    {str(c):<9} {q.GROSS_C_m.median():6.2f} | {q.R1.median():5.2f} | {q.R2.median():5.2f} | {q.DH100.median():6.3f} | {len(q)}")
if len(SENS):
    rs = stats.spearmanr(SENS.GROSS_C_m, SENS.GC_ALL, nan_policy="omit")[0]
    print(f"\n  MRC sensitivity (Aug-Feb MRC incl. rabi pumping / Aug-Nov MRC): median ratio "
          f"{np.nanmedian(SENS.GC_ALL / SENS.GROSS_C_m):.2f} | Spearman {rs:+.2f} | n {len(SENS)}")
print("\n######## EPISODES (all available days, incl. partial 2022/2025) ########")
if len(E):
    print(f"  episodes {len(E)} | stations {E.SID.nunique()} | by year: {E.groupby('YEAR').size().to_dict()}")
    per = E.groupby(["SID", "YEAR"]).agg(n=("RECH_m", "size"), tot=("RECH_m", "sum"),
                                         top3=("RECH_m", lambda v: np.sort(v)[::-1][:3].sum()))
    print(line("episodes per station-year", per.n, "{:.0f}"))
    print(line("share of station-year recharge in its 3 largest episodes", per.top3 / per.tot, "{:.2f}"))
    print(line("episode length (days)", E.days, "{:.0f}"))
    q = E.dropna(subset=["RAIN_mm"])
    r1 = stats.spearmanr(q.RAIN_mm, q.RECH_m)[0]
    r2 = stats.spearmanr(q.RAIN_MAX_mm, q.RECH_m)[0]
    within = q.groupby("SID").apply(lambda g: stats.spearmanr(g.RAIN_mm, g.RECH_m)[0] if len(g) >= 5 else np.nan)
    print(f"  episode recharge vs episode rain: pooled Spearman {r1:+.2f} | vs max daily rain {r2:+.2f} | "
          f"within-station median {within.median():+.2f} (stations with >= 5 episodes: {int(within.notna().sum())})")
    E["RBIN"] = pd.cut(E.RAIN_mm, [0, 10, 25, 50, 100, 1e9], labels=["<10", "10-25", "25-50", "50-100", ">100"])
    print("  median episode recharge (m) by episode rain (mm):",
          E.groupby("RBIN", observed=True).RECH_m.agg(["median", "size"]).round(3).to_dict("index"))
else:
    print("  no episodes detected")
print("\nNOTES: recharge per unit Sy is GROSS (MRC-WTF; Heppner & Nimmo 2005). Concurrent pumping/ET faster than the")
print("MRC is not added back. Noise correction removes the positive bias of summing max(0, noise).")
print(f"Outputs: {OUT}")
