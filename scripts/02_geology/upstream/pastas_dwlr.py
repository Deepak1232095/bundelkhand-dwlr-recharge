"""
PASTAS ON DWLR (DAILY)  ->  VALIDATED MODEL  ->  35-YEAR HINDCAST OF THE "CAPTURE" PROBLEM
==========================================================================================
Why a model now: the DWLR record has only 2 monsoons. A rain-driven model that is calibrated
on one monsoon and VALIDATED on the other can be run with IMD rain 1990-2025 to ask:
   "Across ~30 monsoons (wet, dry, early, late), how much of the seasonal rise does a
    20-May / 10-Nov reading capture, and what is the best post-monsoon date?"
The model does NOT include pumping (no data) -> results = present-day aquifer behaviour
driven by observed rain variability. Not a historical reconstruction.

Rules fixed BEFORE looking at results (write them in the paper):
  GOOD model = rch_A significant (A > 1.96*se)  AND  validation NSE >= 0.5 (main split)
  Model choice per station by calibration AIC (M1 rain-only vs M2 rain - f*ET0), never by validation.

Splits: MAIN  cal = start -> 2024-03-31 (monsoon 2023) | val = 2024-04-01 -> end (monsoon 2024)
        REV   cal = 2024-04-01 -> end                  | val = start -> 2024-03-31


Requires: pastas xarray netCDF4 pandas numpy scipy matplotlib
"""
import os
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import pastas as ps
import matplotlib.pyplot as plt
from scipy import stats

warnings.filterwarnings("ignore")
ps.set_log_level("ERROR")

QC = r"DWLR_QC"                                      # folder made by dwlr_qc.py (same notebook dir)
DAILY = os.path.join(QC, "dwlr_daily_depth_abs.csv")
SIGN = os.path.join(QC, "dwlr_sign_smoothness.csv")
RAIN = r"E:\up_bundel\up_bundel_rainfall_1990_2025.nc"
TMAX = r"E:\up_bundel\up_bundel_tmax_1990_2025.nc"
TMIN = r"E:\up_bundel\up_bundel_tmin_1990_2025.nc"
OUT = os.path.join(QC, "Pastas_DWLR")
SPLIT = pd.Timestamp("2024-04-01")
STRESS_START = pd.Timestamp("1990-01-01")
HIND_Y0, HIND_Y1 = 1995, 2024
NSE_MIN = 0.5
POST_DATES = pd.date_range("2000-09-01", "2000-12-15", freq="5D")
os.makedirs(OUT, exist_ok=True)

# ---------------- data ----------------
sg = pd.read_csv(SIGN)
core = sg[sg.sign_verdict.isin(["single sign (fine)", "sign errors -> |level| fixes it"])].set_index("SID")
dd = pd.read_csv(DAILY, parse_dates=["T"])
dd = dd[dd.SID.isin(core.index)]
print(f"core stations with daily data: {dd.SID.nunique()}")


def load_cube(path):
    ds = xr.open_dataset(path)
    var = [v for v in ds.data_vars if ds[v].ndim == 3][0]
    la = next(c for c in ds.coords if "lat" in c.lower()); lo = next(c for c in ds.coords if "lon" in c.lower())
    tm = next(c for c in ds.coords if "time" in c.lower())
    R = ds[var].transpose(tm, la, lo).values.astype(float); T = pd.to_datetime(ds[tm].values)
    LA, LO = np.meshgrid(ds[la].values, ds[lo].values, indexing="ij")
    ds.close()
    return R, T, LA, LO, ~np.all(np.isnan(R), axis=0)


CUBES = {k: load_cube(p) for k, p in (("rain", RAIN), ("tmax", TMAX), ("tmin", TMIN))}


def point(kind, lat, lon):
    R, T, LA, LO, valid = CUBES[kind]
    d = np.hypot(LA - lat, LO - lon); d[~valid] = np.inf
    i, j = np.unravel_index(np.argmin(d), d.shape)
    s = pd.Series(R[:, i, j], index=T)
    if kind == "rain":
        return s.where(s >= 0).asfreq("D").fillna(0.0)
    s = s.where((s > -20) & (s < 60)).asfreq("D").interpolate(limit=5)
    return s.fillna(s.groupby(s.index.dayofyear).transform("mean"))


def hargreaves(tmax, tmin, lat_deg):
    J = tmax.index.dayofyear.values; phi = np.deg2rad(lat_deg)
    dr = 1 + 0.033 * np.cos(2 * np.pi * J / 365); dec = 0.409 * np.sin(2 * np.pi * J / 365 - 1.39)
    ws = np.arccos(np.clip(-np.tan(phi) * np.tan(dec), -1, 1))
    ra = (24 * 60 / np.pi) * 0.0820 * dr * (ws * np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.sin(ws))
    return (0.0023 * ((tmax + tmin) / 2 + 17.8) * np.sqrt((tmax - tmin).clip(lower=0)) * 0.408 * ra).clip(lower=0)


# ---------------- model helpers ----------------
def build(h, p, e, kind):
    ml = ps.Model(h, name=kind)
    if kind == "M1":
        ml.add_stressmodel(ps.StressModel(p, rfunc=ps.Gamma(), name="rch", settings="prec"))
    else:
        ml.add_stressmodel(ps.RechargeModel(p, e, rfunc=ps.Gamma(), name="rch", recharge=ps.rch.Linear()))
    try:
        ml.add_noisemodel(ps.ArNoiseModel())
    except Exception:
        pass
    return ml


def solve(ml, tmin, tmax):
    warm = max((pd.Timestamp(tmin) - STRESS_START).days - 1, 365)
    try:
        ml.solve(tmin=tmin, tmax=tmax, warmup=warm, report=False)
    except TypeError:
        ml.solve(tmin=tmin, tmax=tmax, report=False)
    return ml


def nse(obs, sim):
    o, s = obs.align(sim, join="inner")
    return 1 - np.sum((o - s) ** 2) / np.sum((o - o.mean()) ** 2) if len(o) > 30 else np.nan


def pval(ml, name):
    p = ml.parameters
    return float(p.loc[name, "optimal"]) if name in p.index else np.nan, \
        float(p.loc[name, "stderr"]) if name in p.index and "stderr" in p.columns else np.nan


def capture_metrics(depth, yr):
    """depth: daily series (m bgl, + down). returns true rise, peak doy, capture at each POST_DATE."""
    mon = depth[f"{yr}-07-01":f"{yr}-10-31"]
    if len(mon) < 100:
        return None
    tp = mon.idxmin(); zp = mon.min()
    pre = depth[f"{yr}-05-01":tp]
    if len(pre) < 20:
        return None
    rise = pre.max() - zp
    z_may = depth[f"{yr}-05-17":f"{yr}-05-23"].median()
    out = {"YEAR": yr, "TRUE_RISE": rise, "PEAK_DOY": tp.dayofyear}
    for md in POST_DATES:
        t = pd.Timestamp(yr, md.month, md.day)
        z = depth[t - pd.Timedelta(days=3):t + pd.Timedelta(days=3)].median()
        out[md.strftime("%d-%b")] = (z_may - z) / rise if rise > 0.05 else np.nan
    return out


# ---------------- loop ----------------
rows, hind, obs_vs_mod = [], [], []
for sid, g in dd.groupby("SID"):
    lat, lon = core.at[sid, "lat"], core.at[sid, "lon"]
    h = -g.set_index("T")["DEPTH_ABS"].sort_index()
    h = h[~h.index.duplicated()]
    if h[:SPLIT - pd.Timedelta(days=1)].size < 200 or h[SPLIT:].size < 200:
        continue
    p = point("rain", lat, lon)
    e = hargreaves(point("tmax", lat, lon), point("tmin", lat, lon), lat).reindex(p.index).ffill()
    r = {"SID": sid, "DIST": sid.split("|")[0], "lat": lat, "lon": lon, "n_days": len(h)}
    try:
        cal = {k: solve(build(h, p, e, k), h.index.min(), SPLIT - pd.Timedelta(days=1)) for k in ("M1", "M2")}
        best = min(cal, key=lambda k: cal[k].stats.aic())
        ml = cal[best]
        A, Ase = pval(ml, "rch_A")
        sim = ml.simulate(tmin=h.index.min(), tmax=h.index.max())
        r.update({"MODEL": best, "EVP_CAL": ml.stats.evp(), "A": A, "A_se": Ase,
                  "SIG": bool(np.isfinite(Ase) and A > 1.96 * Ase),
                  "NSE_VAL": nse(h[SPLIT:], sim[SPLIT:]),
                  "f_ET": pval(ml, "rch_f")[0] if best == "M2" else np.nan})
        try:
            r["t50_d"] = ml.get_response_tmax("rch", cutoff=0.5)
        except Exception:
            r["t50_d"] = np.nan
        # reverse split, same model type
        mr = solve(build(h, p, e, best), SPLIT, h.index.max())
        r["NSE_VAL_REV"] = nse(h[:SPLIT - pd.Timedelta(days=1)], mr.simulate(tmin=h.index.min(), tmax=h.index.max())[:SPLIT])
        r["GOOD"] = r["SIG"] and r["NSE_VAL"] >= NSE_MIN
        # observed vs model capture in the VALIDATION monsoon (2024)
        cm_obs = capture_metrics(-h.asfreq("D").interpolate(limit=7).rolling(15, center=True, min_periods=7).median(), 2024)
        cm_mod = capture_metrics(-sim, 2024)
        if cm_obs and cm_mod:
            obs_vs_mod.append({"SID": sid, "GOOD": r["GOOD"], "obs_10Nov": cm_obs["10-Nov"], "mod_10Nov": cm_mod["10-Nov"],
                               "obs_peak": cm_obs["PEAK_DOY"], "mod_peak": cm_mod["PEAK_DOY"],
                               "obs_rise": cm_obs["TRUE_RISE"], "mod_rise": cm_mod["TRUE_RISE"]})
        # hindcast with model refit on ALL data (only for GOOD stations)
        if r["GOOD"]:
            mf = solve(build(h, p, e, best), h.index.min(), h.index.max())
            hs = -mf.simulate(tmin=f"{HIND_Y0}-01-01", tmax=f"{HIND_Y1}-12-31")
            jj = p[(p.index.month >= 6) & (p.index.month <= 10)]
            for yr in range(HIND_Y0, HIND_Y1 + 1):
                cm = capture_metrics(hs, yr)
                if cm is None:
                    continue
                py = jj[jj.index.year == yr]
                c = py.cumsum() / py.sum() if py.sum() > 0 else py * np.nan
                cm.update({"SID": sid, "DIST": r["DIST"], "RAIN_JJASO": py.sum(),
                           "W90_DOY": c[c >= 0.9].index.min().dayofyear if py.sum() > 0 else np.nan})
                hind.append(cm)
    except Exception as ex:
        r["ERROR"] = str(ex)[:120]
    rows.append(r)
    print(f"  {sid[:45]:<45} {r.get('MODEL', '-')} NSE_val {r.get('NSE_VAL', np.nan):+.2f} GOOD {r.get('GOOD', False)}")

S = pd.DataFrame(rows); H = pd.DataFrame(hind); OV = pd.DataFrame(obs_vs_mod)
S.to_csv(os.path.join(OUT, "station_models.csv"), index=False)
H.to_csv(os.path.join(OUT, "hindcast_capture_by_year.csv"), index=False)
OV.to_csv(os.path.join(OUT, "obs_vs_model_capture_2024.csv"), index=False)

# ---------------- figure: capture curve, observed-period vs 30-yr hindcast ----------------
dcols = [m.strftime("%d-%b") for m in POST_DATES]
if len(H):
    q = H[dcols].quantile([.1, .5, .9])
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(dcols))
    ax.fill_between(x, q.loc[.1], q.loc[.9], alpha=.25, label="hindcast 1995-2024, 10-90%")
    ax.plot(x, q.loc[.5], lw=2, label="hindcast median")
    ax.axvline(dcols.index("10-Nov"), c="k", ls="--", lw=.8)
    ax.set_xticks(x[::3]); ax.set_xticklabels(dcols[::3], rotation=45)
    ax.set_ylabel("capture of seasonal rise"); ax.set_xlabel("post-monsoon reading date")
    ax.legend(fontsize=8); plt.tight_layout(); plt.savefig(os.path.join(OUT, "Fig_capture_hindcast.png"), dpi=300)
    plt.close()

# ---------------- VERDICT ----------------
print("\n" + "=" * 72)
print("VERDICT")
print("=" * 72)
ok = S.dropna(subset=["NSE_VAL"])
print(f"stations modelled {len(ok)} / {len(S)} | errors {int(S.get('ERROR', pd.Series(dtype=str)).notna().sum())}")
print(f"model choice (cal AIC): {ok.MODEL.value_counts().to_dict()} | significant A: {int(ok.SIG.sum())}")
print(f"NSE_val MAIN median {ok.NSE_VAL.median():.2f} (IQR {ok.NSE_VAL.quantile(.25):.2f}-{ok.NSE_VAL.quantile(.75):.2f}) | "
      f"REV median {ok.NSE_VAL_REV.median():.2f} | GOOD (sig & NSE_val>={NSE_MIN}): {int(ok.GOOD.sum())}")
print("GOOD by district:", ok[ok.GOOD].groupby("DIST").size().to_dict())
g = ok[ok.GOOD]
if len(g):
    print(f"GOOD: EVP_cal median {g.EVP_CAL.median():.0f}% | t50 median {g.t50_d.median():.0f} d | "
          f"f_ET (M2 only) median {g.f_ET.median():.2f}")
if len(OV):
    v = OV[OV.GOOD].dropna()
    if len(v) >= 5:
        print(f"VALIDATION of the capture metric (2024, GOOD n={len(v)}): obs 10-Nov capture median {v.obs_10Nov.median():.2f} "
              f"vs model {v.mod_10Nov.median():.2f} | spearman {stats.spearmanr(v.obs_10Nov, v.mod_10Nov)[0]:+.2f} | "
              f"peak DOY obs {v.obs_peak.median():.0f} vs model {v.mod_peak.median():.0f}")
if len(H):
    print(f"\nHINDCAST {HIND_Y0}-{HIND_Y1}: {H.SID.nunique()} stations x {H.YEAR.nunique()} yrs = {len(H)} station-years")
    print(f"  capture at 10-Nov: median {H['10-Nov'].median():.2f} | 10-90% {H['10-Nov'].quantile(.1):.2f}-{H['10-Nov'].quantile(.9):.2f}")
    best = H[dcols].idxmax(axis=1)
    print(f"  best date per station-year: mode {best.mode().iloc[0]} | share best in 11-Sep..06-Oct "
          f"{100 * best.isin(dcols[2:8]).mean():.0f}%")
    print(f"  capture at best date median {H[dcols].max(axis=1).median():.2f}")
    yr = H.groupby("YEAR")[["10-Nov", "RAIN_JJASO", "W90_DOY", "PEAK_DOY"]].median()
    for x in ("RAIN_JJASO", "W90_DOY"):
        rho, pv = stats.spearmanr(yr[x], yr["10-Nov"])
        print(f"  year-level: capture@10-Nov vs {x}: spearman {rho:+.2f} (p {pv:.3f}, n {len(yr)})")
    rho, pv = stats.spearmanr(yr["W90_DOY"], yr["PEAK_DOY"])
    print(f"  year-level: peak DOY vs rain-withdrawal DOY: spearman {rho:+.2f} (p {pv:.3f})")
    print("  worst 3 years (capture@10-Nov):", yr["10-Nov"].nsmallest(3).round(2).to_dict())
print(f"\nOutputs: {OUT}")
