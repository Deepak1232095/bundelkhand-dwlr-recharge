"""
ROBUSTNESS TESTS (Text S6) - one script for the code-based review points M1-M3, M4, M5, M6, M7, M9, D3, P5, DA
==========================================================================================================
Re-implements wtf_mrc_recharge.py exactly (TRACE CHECK first: must reproduce the stored GROSS_C), then varies ONE
setting at a time. Nothing in the existing outputs is changed.

  STEP 0  trace check            base settings reproduce run_Q123 recharge_station_years.csv (|diff| < 1 mm)
  M2      noise correction       per-station raw sum, correction term, quiet days; eps = 1/2/3 sigma; Gaussian vs
                                 EMPIRICAL E[max(0,e)] from the dry-day residuals; no correction
  M3      October                share of the raw rise in October, October rain, share of the October rise in
                                 multi-day (>= 3 d) runs; dH with a Jun-Sep window
  M4      MRC window             Aug-Nov (base) vs Aug-15 Oct vs Aug-Oct vs Aug-Feb; dry days used; Dec-Feb dry-day
                                 rate vs Aug-Nov rate (is rabi pumping visible in the recession?)
  M5      dry-day leakage        dry rule extended to rain(t+2) < 2 mm (because ds(t) contains h(t+2))
  M1      synthetic @ field σ    synthetic_verification generator with level noise set from the FIELD residual σ
                                 (median and 90th percentile), Gaussian / Student-t (df 3) / AR(1) noise;
                                 Gaussian vs empirical correction
  MC      total uncertainty      Monte Carlo: bootstrap of the MRC fit days + correction choice + Sy prior (uniform in
                                 the class range) + station resampling -> 95 % interval of class median recharge
  D3      Sy norms               alluvium 4 / 6 / 10 / 16 %, hard rock 1 / 1.5 / 3 / 4 %: median R, % of rain,
                                 stations with R > Jun-Oct rain
  M6      episodes               duration-controlled (partial Spearman) rain-rise link for total and max-daily rain;
                                 top-3 share of rise vs top-3 share of episode RAIN (same episodes) and vs 3/n
  M7      |z| stations           stations retained after taking |z|: count and dH without them
  M9      excluded stations      excluded vs retained: district, depth, MRC rejection reason
  P5      semivariogram          of log10 dH (pairs by distance class) -> nugget share
  DA      June rain-free rises   share starting within ±3 d of a Q3-corrected day (same station / same district)
          rainfall anomaly       Jun-Oct rain 2023 and 2024 vs 1991-2020 mean at the station cells
Output : <BASE>\\paper3_final\\reviewer\\  TableR*.csv, FigS6_reviewer_tests.png/.pdf
Run    :  %run robustness_tests.py   (from C:\\Users\\IITI; ~5-15 min)   Requires pandas numpy scipy xarray netCDF4 matplotlib

"""
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
T_START = time.time()

# ======================================================================================== CONFIG
BASE = os.environ.get("TEST_BASE", r"C:\Users\IITI")
RUN = os.path.join(BASE, "run_Q123", "DWLR_QC")
FINAL = os.path.join(BASE, "paper3_final")
OUT = os.path.join(FINAL, "reviewer")
DAILY = os.path.join(RUN, "dwlr_daily_depth_abs.csv")                 # audited (Q123) daily depths
RAW = os.path.join(BASE, "DWLR_QC", "dwlr_daily_depth_abs.csv")       # unaudited, for Q3 dates
SIGN = os.path.join(RUN, "dwlr_sign_smoothness.csv")
YRS = os.path.join(RUN, "wtf_mrc", "recharge_station_years.csv")
MRCP = os.path.join(RUN, "wtf_mrc", "mrc_parameters.csv")
EPIS = os.path.join(RUN, "wtf_mrc", "diagnostics", "diag_episodes.csv")
GEOL = os.path.join(RUN, "regime_controls", "geology_v3", "station_geology_audit.csv")
T1P = next((p for p in (os.path.join(RUN, "final", "T1_station_recharge.csv"), os.path.join(FINAL, "T1_station_recharge.csv"))
            if os.path.exists(p)), None)
RAIN = os.environ.get("TEST_RAIN", r"E:\up_bundel\up_bundel_rainfall_1990_2025.nc")
# settings of wtf_mrc_recharge.py
YEARS = [2021, 2022, 2023, 2024, 2025]
GAP_FILL, SMOOTH, RAIN_THR, LAG_DAYS, MIN_REC, MIN_COVER, MERGE_GAP = 3, 3, 2.0, 5, 30, 0.8, 2
WIN = ((6, 1), (10, 31))
SY = {"HARD_ROCK": (0.03, 0.015, 0.04), "ALLUVIUM": (0.06, 0.04, 0.08)}
SY_GRID = {"ALLUVIUM": [0.04, 0.06, 0.10, 0.16], "HARD_ROCK": [0.01, 0.015, 0.03, 0.04]}
N_MC, N_SYN, SEED = int(os.environ.get("TEST_NMC", 200)), int(os.environ.get("TEST_NSYN", 30)), 20261013
LAB = {"ALLUVIUM": "Alluvium", "HARD_ROCK": "Hard rock"}
COL = {"ALLUVIUM": "#2a78d6", "HARD_ROCK": "#eb6834"}
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(SEED)
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                     "font.size": 7.5, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                     "legend.frameon": False, "legend.fontsize": 6.8, "pdf.fonttype": 42, "savefig.dpi": 600,
                     "savefig.bbox": "tight"})
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5e0"
LOG = []


def say(s=""):
    LOG.append(str(s))


for f in (DAILY, SIGN, YRS, RAIN):
    if not os.path.exists(f):
        sys.exit(f"[STOP] not found: {f}")
if T1P is None:
    sys.exit("[STOP] T1_station_recharge.csv not found")

T1 = pd.read_csv(T1P).drop_duplicates("SID").set_index("SID")
T1["DIST"] = T1.DIST.astype(str).str.upper().str.strip()
sg = pd.read_csv(SIGN).drop_duplicates("SID").set_index("SID")
dd = pd.read_csv(DAILY, parse_dates=["T"])
YS = pd.read_csv(YRS)
YS = YS[YS.MRC == "PRIMARY"]

ds = xr.open_dataset(RAIN)
var = [v for v in ds.data_vars if ds[v].ndim == 3][0]
la = next(c for c in ds.coords if "lat" in c.lower()); lo = next(c for c in ds.coords if "lon" in c.lower())
tm = next(c for c in ds.coords if "time" in c.lower())
RR = ds[var].transpose(tm, la, lo).values.astype(float); TT = pd.to_datetime(ds[tm].values).normalize()
LA, LO = np.meshgrid(ds[la].values, ds[lo].values, indexing="ij"); valid = ~np.all(np.isnan(RR), axis=0)
ds.close()


def rain_at(lat, lon):
    d = np.hypot(LA - lat, LO - lon); d[~valid] = np.inf
    i, j = np.unravel_index(np.argmin(d), d.shape)
    s = pd.Series(RR[:, i, j], index=TT)
    s = s[~s.index.duplicated()]
    return s.where(s >= 0)


# ======================================================================================== pipeline (exact copy)
def prep(sid):
    g = dd[dd.SID == sid]
    if g.empty or sid not in sg.index:
        return None
    dep = g.groupby("T").DEPTH_ABS.median()
    full = pd.date_range(dep.index.min(), dep.index.max(), freq="D")
    h = -dep.reindex(full)
    h = h.interpolate(limit=GAP_FILL, limit_area="inside")
    s = h.rolling(SMOOTH, center=True, min_periods=SMOOTH).mean()
    s[h.isna()] = np.nan
    rain = rain_at(sg.loc[sid, "lat"], sg.loc[sid, "lon"]).reindex(full)
    rate = s.shift(-1) - s
    return {"full": full, "s": s, "rate": rate, "rain": rain, "dep": dep}


def dry_mask(p, mode="base"):
    rain = p["rain"]
    wet_prev = rain.rolling(LAG_DAYS, min_periods=1).max()
    dry = (wet_prev < RAIN_THR) & (rain.shift(-1).fillna(0) < RAIN_THR)
    if mode == "t2":
        dry = dry & (rain.shift(-2).fillna(0) < RAIN_THR)
    return dry & p["s"].notna() & p["rate"].notna()


def months_mask(idx, mode):
    m, d = idx.month, idx.day
    return {"aug_nov": np.isin(m, [8, 9, 10, 11]), "aug_oct": np.isin(m, [8, 9, 10]),
            "aug_15oct": np.isin(m, [8, 9]) | ((m == 10) & (d <= 15)),
            "aug_feb": np.isin(m, [8, 9, 10, 11, 12, 1, 2])}[mode]


def fit(p, months="aug_nov", dry="base", pick=None):
    m = dry_mask(p, dry) & months_mask(p["full"], months)
    idx = np.where(m.values)[0]
    if len(idx) < MIN_REC:
        return None, "too few dry days", len(idx)
    if pick is not None:
        idx = idx[pick(len(idx))]
    x, y = p["s"].values[idx], p["rate"].values[idx]
    b, a, _, _ = stats.theilslopes(y, x)
    res = y - (a + b * x)
    sig = 1.4826 * np.median(np.abs(res - np.median(res)))
    if not b < 0:
        return None, "slope >= 0", len(idx)
    return {"a": a, "b": b, "sig": sig, "emp": float(np.mean(np.clip(res, 0, None))), "n": len(idx)}, "ok", len(idx)


def season(p, fp, win=WIN, eps_k=2.0, corr="gauss"):
    s, rate, full = p["s"], p["rate"], p["full"]
    f = np.minimum(0.0, fp["a"] + fp["b"] * s)
    dpos = (rate - f).clip(lower=0)
    sig = fp["sig"]
    eps = max(eps_k * sig, 0.005)
    bias = {"gauss": sig / np.sqrt(2 * np.pi), "emp": fp["emp"], "none": 0.0}[corr]
    out = {}
    for y in YEARS:
        w = (full >= pd.Timestamp(y, *win[0])) & (full <= pd.Timestamp(y, *win[1]))
        if w.sum() == 0:
            continue
        nwin = int(pd.Timestamp(y, *win[1]).dayofyear - pd.Timestamp(y, *win[0]).dayofyear + 1)
        x = dpos[w]
        cover = float(x.notna().sum()) / nwin
        gross = float(x.sum(min_count=1)) if x.notna().any() else np.nan
        nq = int((x <= eps).sum())
        gc = max(gross - bias * nq, 0) if np.isfinite(gross) else np.nan
        out[y] = {"gross": gross, "nq": nq, "corr": gross - gc if np.isfinite(gross) else np.nan, "gc": gc,
                  "full": cover >= MIN_COVER, "dpos": x, "eps": eps}
    return out


def station_dh(sea):
    v = [r["gc"] for r in sea.values() if r["full"] and np.isfinite(r["gc"])]
    return float(np.median(v)) if v else np.nan


SIDS = [s for s in T1.index if T1.loc[s, "CLASS"] in SY or True]
P = {}
for sid in SIDS:
    p = prep(sid)
    if p is not None:
        P[sid] = p
say(f"stations in T1 {len(T1)} | with daily data and coordinates {len(P)}")

# ======================================================================================== STEP 0 trace
BASEFIT, BASESEA, rows = {}, {}, []
for sid, p in P.items():
    fp, note, n = fit(p)
    if fp is None:
        continue
    BASEFIT[sid] = fp
    BASESEA[sid] = season(p, fp)
    for y, r in BASESEA[sid].items():
        rows.append({"SID": sid, "YEAR": y, "GC_new": r["gc"]})
TR = pd.DataFrame(rows).merge(YS[["SID", "YEAR", "GROSS_C_m"]], on=["SID", "YEAR"], how="inner")
TR["diff_mm"] = 1000 * (TR.GC_new - TR.GROSS_C_m)
mx = TR.diff_mm.abs().max()
say("\nSTEP 0 TRACE CHECK (re-implementation vs stored GROSS_C, station-years)")
say(f"  n {len(TR)} | max |diff| {mx:.3f} mm -> " + ("REPRODUCED" if mx < 1 else "NOT REPRODUCED - results below are NOT comparable; send this block"))
DH0 = pd.Series({s: station_dh(v) for s, v in BASESEA.items()})
cmp_t1 = (DH0 - T1.GROSS_C.reindex(DH0.index)).abs().max()
say(f"  station dH vs T1 GROSS_C: max |diff| {1000 * cmp_t1:.2f} mm")
CLS = T1.CLASS.reindex(DH0.index)


def by_class(series):
    return {LAB.get(c, c): round(float(series[CLS == c].median()), 3) for c in ("ALLUVIUM", "HARD_ROCK")}


# ======================================================================================== M2 / M3 / M4 / M5 variants
VARIANTS = [("base (Aug-Nov MRC, eps 2σ, Gaussian)", {}),
            ("eps 1σ", {"eps_k": 1.0}), ("eps 3σ", {"eps_k": 3.0}),
            ("empirical correction E[max(0,e)]", {"corr": "emp"}), ("no noise correction", {"corr": "none"}),
            ("Jun-Sep window (October excluded)", {"win": ((6, 1), (9, 30))}),
            ("MRC Aug-15 Oct", {"months": "aug_15oct"}), ("MRC Aug-Oct", {"months": "aug_oct"}),
            ("MRC Aug-Feb", {"months": "aug_feb"}), ("dry rule to rain(t+2)", {"dry": "t2"})]
VROWS, VDH = [], {}
for name, kw in VARIANTS:
    fkw = {k: kw[k] for k in ("months", "dry") if k in kw}
    skw = {k: kw[k] for k in ("win", "eps_k", "corr") if k in kw}
    dh, rej = {}, {"too few dry days": 0, "slope >= 0": 0}
    for sid, p in P.items():
        if sid not in BASEFIT:
            continue
        fp, note, n = (BASEFIT[sid], "ok", BASEFIT[sid]["n"]) if not fkw else fit(p, **fkw)
        if fp is None:
            rej[note] += 1; continue
        dh[sid] = station_dh(season(p, fp, **skw))
    dh = pd.Series(dh); VDH[name] = dh
    ratio = (dh / DH0.reindex(dh.index)).replace([np.inf, -np.inf], np.nan)
    VROWS.append({"variant": name, "stations": int(dh.notna().sum()), "rejected": f"{rej['too few dry days']} few days / {rej['slope >= 0']} b>=0",
                  "dH alluvium (m)": round(dh[CLS.reindex(dh.index) == "ALLUVIUM"].median(), 2),
                  "dH hard rock (m)": round(dh[CLS.reindex(dh.index) == "HARD_ROCK"].median(), 2),
                  "station ratio to base, median": round(ratio.median(), 3),
                  "IQR": f"{ratio.quantile(.25):.2f}-{ratio.quantile(.75):.2f}"})
TV = pd.DataFrame(VROWS)
TV.to_csv(os.path.join(OUT, "TableR1_sensitivity_variants.csv"), index=False, encoding="utf-8-sig")
say("\nM2-M5  SENSITIVITY OF dH (one setting changed at a time; same stations)")
say("  " + TV.to_string(index=False).replace("\n", "\n  "))

# per-station correction table (M2) and October (M3)
srows = []
for sid, sea in BASESEA.items():
    fy = [r for r in sea.values() if r["full"]]
    if not fy:
        continue
    oct_share, oct_multi, oct_rain = [], [], []
    for y, r in sea.items():
        if not r["full"]:
            continue
        x = r["dpos"]
        tot = x.sum()
        xo = x[x.index.month == 10]
        oct_share.append(xo.sum() / tot if tot > 0 else np.nan)
        on = (xo > r["eps"]).values
        runs, cur = [], 0
        for k, v in enumerate(on):
            cur = cur + 1 if v else 0
            runs.append(cur)
        lens = np.zeros(len(on), int)
        k = len(on) - 1
        while k >= 0:
            if on[k]:
                L = runs[k]; lens[k - L + 1:k + 1] = L; k -= L
            else:
                k -= 1
        multi = xo.values[(lens >= 3)].sum()
        oct_multi.append(multi / xo.sum() if xo.sum() > 0 else np.nan)
        oct_rain.append(float(P[sid]["rain"][x.index][x.index.month == 10].sum()))
    srows.append({"SID": sid, "CLASS": CLS.get(sid), "sigma_m_d": BASEFIT[sid]["sig"],
                  "raw_sum_m": np.median([r["gross"] for r in fy]), "correction_m": np.median([r["corr"] for r in fy]),
                  "quiet_days": np.median([r["nq"] for r in fy]), "dH_m": DH0.get(sid),
                  "emp_over_gauss": BASEFIT[sid]["emp"] / (BASEFIT[sid]["sig"] / np.sqrt(2 * np.pi)),
                  "oct_share_raw": np.nanmedian(oct_share), "oct_multiday_share": np.nanmedian(oct_multi),
                  "oct_rain_mm": np.median(oct_rain)})
SC = pd.DataFrame(srows)
SC.round(4).to_csv(os.path.join(OUT, "TableR2_station_correction_october.csv"), index=False)
say("\nM2  NOISE CORRECTION per station (medians over FULL years)")
for c in ("ALLUVIUM", "HARD_ROCK"):
    q = SC[SC.CLASS == c]
    say(f"  {LAB[c]:<9} σ {q.sigma_m_d.median():.4f} m/d | raw sum {q.raw_sum_m.median():.2f} m | correction {q.correction_m.median():.2f} m "
        f"({100 * (q.correction_m / q.raw_sum_m).median():.0f}% of raw) | quiet days {q.quiet_days.median():.0f} | "
        f"empirical/Gaussian correction per day {q.emp_over_gauss.median():.2f}")
say("\nM3  OCTOBER")
for c in ("ALLUVIUM", "HARD_ROCK"):
    q = SC[SC.CLASS == c]
    say(f"  {LAB[c]:<9} October share of raw rise {100 * q.oct_share_raw.median():.0f}% | October rain {q.oct_rain_mm.median():.0f} mm | "
        f"share of October rise in runs >= 3 d {100 * q.oct_multiday_share.median():.0f}%")

# M4 extra: Dec-Feb dry-day rates vs Aug-Nov
r_an, r_df, n_df = [], [], []
for sid, p in P.items():
    if sid not in BASEFIT:
        continue
    dm = dry_mask(p)
    fp = BASEFIT[sid]
    resid = p["rate"] - (fp["a"] + fp["b"] * p["s"])
    m1 = dm & months_mask(p["full"], "aug_nov"); m2 = dm & np.isin(p["full"].month, [12, 1, 2])
    r_an.append(resid[m1].median()); r_df.append(resid[m2].median() if m2.sum() >= 10 else np.nan); n_df.append(int(m2.sum()))
say(f"\nM4  Dec-Feb dry days per station: median {np.median(n_df):.0f} | MRC residual on Dec-Feb dry days median "
    f"{np.nanmedian(r_df):+.4f} m/d vs Aug-Nov {np.nanmedian(r_an):+.4f} m/d (negative = faster decline than the "
    f"Aug-Nov MRC, i.e. a pumping signature)")

# ======================================================================================== M1 synthetic at field σ
say("\nM1  SYNTHETIC VERIFICATION AT FIELD NOISE")
sig_field = pd.Series({s: f["sig"] for s, f in BASEFIT.items()})
LEV = {"median field σ": float(sig_field.median()), "P90 field σ": float(sig_field.quantile(0.9))}
T = pd.date_range("2021-01-01", "2024-12-31", freq="D"); SYN_YEARS = [2021, 2022, 2023, 2024]
srng = np.random.default_rng(20261018)


def make(recession, pumping, sd_level, kind):
    wet = T.month.isin([7, 8, 9]) | ((T.month == 6) & (T.day >= 15))
    Pp = np.where(wet & (srng.random(len(T)) < 0.4), np.clip(srng.gamma(0.8, 20, len(T)) * srng.lognormal(0, 0.4, len(T)), 0, 250), 0.0)
    c = srng.uniform(0.002, 0.006)
    q = np.zeros(len(T)); q[1:] = c * np.clip(Pp[:-1] - 5, 0, None)
    pump = np.zeros(len(T))
    if pumping in ("rabi", "rabi+kharif"):
        pump[T.month.isin([12, 1, 2, 3])] = -0.02
    if pumping == "rabi+kharif":
        dry5 = pd.Series(Pp).rolling(5, min_periods=1).max().values < 5
        pump[T.month.isin([6, 7, 8, 9, 10]) & dry5] = -0.01
    h = np.zeros(len(T)); h[0] = -11.0
    for t in range(1, len(T)):
        rec = -0.012 * (h[t - 1] + 12.0) if recession == "linear" else -0.004 * max(h[t - 1] + 12.0, 0) ** 1.5
        h[t] = h[t - 1] + rec + q[t] + pump[t]
    if kind == "gaussian":
        e = srng.normal(0, sd_level, len(T))
    elif kind == "student-t (df 3)":
        e = srng.standard_t(3, len(T)) * sd_level / np.sqrt(3.0)
    else:
        z = srng.normal(0, sd_level * np.sqrt(1 - 0.6 ** 2), len(T)); e = np.zeros(len(T))
        for t in range(len(T)):
            e[t] = (0.6 * e[t - 1] if t else 0) + z[t]
    truth = {y: q[(T >= pd.Timestamp(y, 6, 1)) & (T <= pd.Timestamp(y, 10, 31))].sum() for y in SYN_YEARS}
    return pd.Series(h + e, index=T), pd.Series(Pp, index=T), truth


def syn_est(hh, Pp):
    p = {"full": T, "s": hh.rolling(3, center=True, min_periods=3).mean(), "rain": Pp}
    p["rate"] = p["s"].shift(-1) - p["s"]
    fp, note, n = fit(p)
    if fp is None:
        return None
    out = {}
    for corr in ("gauss", "emp"):
        sea = season(p, fp, corr=corr)
        out[corr] = {y: sea[y]["gc"] for y in SYN_YEARS if y in sea and sea[y]["full"]}
    return out


# level SD such that the 3-day-mean increment has SD = field σ:  sd(Δs) = sqrt(2)/3 * sd(level)
syn_rows = []
for lev_name, sd_d in LEV.items():
    sd_level = sd_d * 3 / np.sqrt(2)
    for kind in ("gaussian", "student-t (df 3)", "AR(1) φ=0.6"):
        for rec in ("linear", "nonlinear"):
            for pump in ("none", "rabi", "rabi+kharif"):
                rej = 0
                for _ in range(N_SYN):
                    hh, Pp, truth = make(rec, pump, sd_level, kind)
                    est = syn_est(hh, Pp)
                    if est is None:
                        rej += 1; continue
                    for corr, d_ in est.items():
                        for y, e in d_.items():
                            if truth[y] > 0:
                                syn_rows.append({"noise level": lev_name, "noise type": kind, "recession": rec,
                                                 "pumping": pump, "correction": corr, "ratio": e / truth[y]})
                syn_rows.append({"noise level": lev_name, "noise type": kind, "recession": rec, "pumping": pump,
                                 "correction": "_rejected", "ratio": rej})
SYN = pd.DataFrame(syn_rows)
SR = SYN[SYN.correction != "_rejected"].groupby(["noise level", "noise type", "pumping", "correction"]).ratio.agg(
    ["count", "mean", "median"]).round(3).reset_index()
REJ = SYN[SYN.correction == "_rejected"].groupby(["noise level", "noise type"]).ratio.sum()
SR.to_csv(os.path.join(OUT, "TableR3_synthetic_field_noise.csv"), index=False)
say(f"  field σ: median {LEV['median field σ']:.4f}, P90 {LEV['P90 field σ']:.4f} m/d -> level noise SD "
    f"{100 * LEV['median field σ'] * 3 / np.sqrt(2):.1f} / {100 * LEV['P90 field σ'] * 3 / np.sqrt(2):.1f} cm | {N_SYN} realizations per scenario")
piv = SR.pivot_table(index=["noise level", "noise type", "pumping"], columns="correction", values="mean")
say("  mean estimated/planted ratio (Gaussian vs empirical correction):")
say("  " + piv.round(2).to_string().replace("\n", "\n  "))
say(f"  MRC fits rejected (of {N_SYN * 6} per noise type): " + ", ".join(f"{k[0]}/{k[1]} {int(v)}" for k, v in REJ.items()))

# ======================================================================================== Monte Carlo total uncertainty
say("\nMC  TOTAL UNCERTAINTY (MRC fit-day bootstrap + correction choice + Sy prior + station resampling)")
DRAWS = {}
for sid, p in P.items():
    if sid not in BASEFIT or CLS.get(sid) not in SY:
        continue
    vals = []
    for _ in range(N_MC):
        fp, note, n = fit(p, pick=lambda n: rng.integers(0, n, n))
        if fp is None:
            vals.append(np.nan); continue
        vals.append(station_dh(season(p, fp, corr="gauss" if rng.random() < 0.5 else "emp")))
    DRAWS[sid] = np.array(vals)
MCR = []
for c in ("ALLUVIUM", "HARD_ROCK"):
    sids = [s for s in DRAWS if CLS.get(s) == c]
    if len(sids) < 3:
        continue
    M = np.vstack([DRAWS[s] for s in sids]); rain = T1.RAIN.reindex(sids).values
    cen, lo_, hi_ = SY[c]
    med_R, med_pct, med_dh = [], [], []
    for k in range(N_MC):
        pick = rng.integers(0, len(sids), len(sids))
        dh = M[pick, k]; sy = rng.uniform(lo_, hi_)
        med_dh.append(np.nanmedian(dh)); med_R.append(1000 * sy * np.nanmedian(dh)); med_pct.append(np.nanmedian(100 * 1000 * sy * dh / rain[pick]))
    MCR.append({"class": LAB[c], "stations": len(sids), "dH median (m) 95%": f"{np.nanpercentile(med_dh, 2.5):.2f}-{np.nanpercentile(med_dh, 97.5):.2f}",
                "R median (mm) central": round(1000 * cen * np.nanmedian(DH0.reindex(sids)), 0),
                "R median (mm) 95% total": f"{np.nanpercentile(med_R, 2.5):.0f}-{np.nanpercentile(med_R, 97.5):.0f}",
                "% of rain 95% total": f"{np.nanpercentile(med_pct, 2.5):.0f}-{np.nanpercentile(med_pct, 97.5):.0f}"})
TMC = pd.DataFrame(MCR)
TMC.to_csv(os.path.join(OUT, "TableR4_total_uncertainty.csv"), index=False, encoding="utf-8-sig")
say("  " + TMC.to_string(index=False).replace("\n", "\n  "))

# ======================================================================================== D3 Sy norms
say("\nD3  RECHARGE UNDER ALL Sy NORMS (seasonal dH from T1)")
drows = []
for c, grid in SY_GRID.items():
    q = T1[(T1.CLASS == c) & np.isfinite(T1.GROSS_C)]
    for sy in grid:
        R = 1000 * sy * q.GROSS_C
        drows.append({"class": LAB[c], "Sy (%)": 100 * sy, "R median (mm)": round(R.median(), 0),
                      "% of Jun-Oct rain": round((100 * R / q.RAIN).median(), 0),
                      "stations with R > rain": f"{int((R > q.RAIN).sum())}/{len(q)}"})
TD3 = pd.DataFrame(drows)
TD3.to_csv(os.path.join(OUT, "TableR5_sy_norms.csv"), index=False, encoding="utf-8-sig")
say("  " + TD3.to_string(index=False).replace("\n", "\n  "))

# ======================================================================================== M6 episodes
say("\nM6  EPISODES: intensity vs duration, concentration null")
if os.path.exists(EPIS):
    E = pd.read_csv(EPIS, parse_dates=["start", "end"])
    if "days" not in E.columns:
        E["days"] = (E.end - E.start).dt.days + 1
    E = E[E.SID.isin(T1.index)]
    q = E.dropna(subset=["RAIN_mm"])

    def pspear(x, y, z):
        rx, ry, rz = stats.rankdata(x), stats.rankdata(y), stats.rankdata(z)
        rxy, rxz, ryz = np.corrcoef(rx, ry)[0, 1], np.corrcoef(rx, rz)[0, 1], np.corrcoef(ry, rz)[0, 1]
        return (rxy - rxz * ryz) / np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))

    say(f"  episodes {len(q)} | Spearman rise~total rain {stats.spearmanr(q.RAIN_mm, q.RECH_m)[0]:+.2f} | rise~duration "
        f"{stats.spearmanr(q.days, q.RECH_m)[0]:+.2f} | total rain~duration {stats.spearmanr(q.days, q.RAIN_mm)[0]:+.2f}")
    say(f"  partial Spearman (duration held constant): rise~total rain {pspear(q.RAIN_mm, q.RECH_m, q.days):+.2f}"
        + (f" | rise~max daily rain {pspear(q.RAIN_MAX_mm, q.RECH_m, q.days):+.2f}" if "RAIN_MAX_mm" in q.columns else " | RAIN_MAX_mm not in file"))
    big = q.RAIN_mm > 100
    say(f"  episodes > 100 mm: {100 * big.mean():.0f}% of episodes, median duration {q.days[big].median():.0f} d vs "
        f"{q.days[~big].median():.0f} d for the rest")
    t3 = q.groupby(["SID", "YEAR"]).apply(lambda g: pd.Series({
        "n": len(g), "top3_rise": np.sort(g.RECH_m.values)[::-1][:3].sum() / g.RECH_m.sum(),
        "top3_rain_same_eps": np.sort(g.RAIN_mm.values)[::-1][:3].sum() / max(g.RAIN_mm.sum(), 1e-9),
        "uniform": min(3 / len(g), 1)})).dropna()
    t3 = t3[t3.n >= 4]
    w = stats.wilcoxon(t3.top3_rise, t3.top3_rain_same_eps)
    say(f"  top-3 share (station-years with >= 4 episodes, n={len(t3)}): rise {t3.top3_rise.median():.2f} | "
        f"episode RAIN of the same episodes {t3.top3_rain_same_eps.median():.2f} | equal shares {t3.uniform.median():.2f} | "
        f"Wilcoxon rise vs rain p = {w.pvalue:.3g}")
else:
    say("  SKIPPED - diag_episodes.csv not found")

# ======================================================================================== M7 |z|
say("\nM7  STATIONS RETAINED AFTER TAKING |z|")
if "sign_verdict" in sg.columns:
    zs = set(sg.index[sg.sign_verdict.astype(str).str.contains(r"\|level\|", regex=True)]) & set(DH0.index)
    rest = DH0.drop(index=list(zs), errors="ignore")
    say(f"  |z| stations in the analysis: {len(zs)} {sorted(zs)[:12]}")
    say(f"  dH median all: {by_class(DH0)} | without |z| stations: "
        f"{ {LAB[c]: round(float(rest[CLS.reindex(rest.index) == c].median()), 3) for c in ('ALLUVIUM', 'HARD_ROCK')} }")
else:
    say("  SKIPPED - no sign_verdict column")

# ======================================================================================== M9 excluded stations
say("\nM9  EXCLUDED vs RETAINED STATIONS")
allst = set(sg.index) | set(dd.SID.unique())
ret = set(T1.index)
exc = allst - ret
dep_med = dd.groupby("SID").DEPTH_ABS.median()
reason = {}
if os.path.exists(MRCP):
    mp = pd.read_csv(MRCP); mp = mp[mp.MRC == "PRIMARY"].set_index("SID")
    reason = mp.note.to_dict()
er = pd.Series({s: ("not in daily file" if s not in set(dd.SID) else
                    ("sign/coherence screen" if "sign_verdict" in sg.columns and s in sg.index and
                     sg.loc[s, "sign_verdict"] not in ["single sign (fine)", "sign errors -> |level| fixes it"] else
                     reason.get(s, "no FULL year / other"))) for s in exc})
say(f"  all stations {len(allst)} | retained {len(ret)} | excluded {len(exc)} | reasons: {er.value_counts().to_dict()}")
de, dr = dep_med.reindex(list(exc)).dropna(), dep_med.reindex(list(ret)).dropna()
if len(de) >= 3 and len(dr) >= 3:
    say(f"  median depth (m bgl): excluded {de.median():.1f} vs retained {dr.median():.1f} | Mann-Whitney p = "
        f"{stats.mannwhitneyu(de, dr).pvalue:.3f}")
dist_ = lambda S: pd.Series([str(s).split("|")[0].upper() for s in S]).value_counts().to_dict()
say(f"  district: excluded {dist_(exc)} | retained {dist_(ret)}")
for why in ("slope >= 0 (not physical) -> not used",):
    sel = [s for s in exc if er.get(s) == why]
    if sel:
        say(f"  b >= 0 stations ({len(sel)}): median depth {dep_med.reindex(sel).median():.1f} m | districts {dist_(sel)}")
if os.path.exists(GEOL):
    gg = pd.read_csv(GEOL).drop_duplicates("SID").set_index("SID")
    if "CLASS" in gg.columns:
        say(f"  geology: excluded {gg.CLASS.reindex(list(exc)).value_counts().to_dict()} | retained {T1.CLASS.value_counts().to_dict()}")

# ======================================================================================== P5 semivariogram
say("\nP5  SEMIVARIOGRAM of log10 dH")
v = np.log10(DH0[DH0 > 0]); v = v[v.index.isin(sg.index)]
lat_, lon_ = np.radians(sg.loc[v.index, "lat"].values), np.radians(sg.loc[v.index, "lon"].values)
i_, j_ = np.triu_indices(len(v), 1)
dkm = 6371 * 2 * np.arcsin(np.sqrt(np.sin((lat_[j_] - lat_[i_]) / 2) ** 2 + np.cos(lat_[i_]) * np.cos(lat_[j_]) * np.sin((lon_[j_] - lon_[i_]) / 2) ** 2))
g2 = 0.5 * (v.values[i_] - v.values[j_]) ** 2
same = (CLS.reindex(v.index).values[i_] == CLS.reindex(v.index).values[j_])
edges = [0, 5, 10, 20, 30, 50, 75, 100, 150, 250]
VG = []
for a_, b_ in zip(edges[:-1], edges[1:]):
    m_ = (dkm >= a_) & (dkm < b_)
    VG.append({"distance (km)": f"{a_}-{b_}", "pairs": int(m_.sum()), "gamma": round(g2[m_].mean(), 4) if m_.any() else np.nan,
               "gamma same class": round(g2[m_ & same].mean(), 4) if (m_ & same).any() else np.nan})
TVG = pd.DataFrame(VG)
TVG.to_csv(os.path.join(OUT, "TableR6_semivariogram.csv"), index=False)
sill = float(np.var(v.values))
ok_ = TVG[TVG.pairs >= 10]
nug = (ok_.gamma.iloc[0] / sill) if len(ok_) else np.nan
say(f"  sample variance (sill) {sill:.4f} | nugget share = gamma(first class with >= 10 pairs: "
    f"{ok_['distance (km)'].iloc[0] if len(ok_) else '-'} km) / sill = {nug:.2f} (near 1 = no spatial correlation at station spacing)")
say("  " + TVG.to_string(index=False).replace("\n", "\n  "))

# ======================================================================================== DA June rises vs Q3 dates; rain anomaly
say("\nDA  JUNE RAIN-FREE RISES vs Q3-CORRECTED DAYS")
if os.path.exists(RAW) and os.path.exists(EPIS):
    raw = pd.read_csv(RAW, parse_dates=["T"])
    raw = raw[raw.SID.isin(T1.index)]
    q3 = {}
    for sid, g in raw.groupby("SID"):
        dep = g.groupby("T").DEPTH_ABS.median().sort_index()
        q1 = dep.apply(lambda x: any(abs(x - c) <= 0.5 for c in (99.0, 99.9, 999.0))) | (dep >= 90)
        x = dep.where(~(q1 | (dep > 60)))
        med = x.rolling(15, center=True, min_periods=5).median()
        q3a = (x < 1.0) & (med > 3.0)
        q3b = ((x - med).abs() > 3.0) & ((x.shift(1) - med).abs() <= 1.0) & ((x.shift(-1) - med).abs() <= 1.0) & ~q3a
        q3[sid] = set(dep.index[(q3a | q3b).fillna(False).values])
    E = pd.read_csv(EPIS, parse_dates=["start", "end"])
    for c in ("FULLYR", "RAIN_FREE", "ABOVE_NOISE"):
        if c in E.columns:
            E[c] = E[c].astype(str).str.lower().isin(["true", "1"])
    E = E[E.SID.isin(T1.index)]
    dq = {}
    for sid, ss in q3.items():
        dq.setdefault(str(sid).split("|")[0].upper(), set()).update(ss)

    def near(row, pool):
        return any(abs((row.start - d).days) <= 3 for d in pool)

    J = E[E.RAIN_FREE & E.ABOVE_NOISE & (E.start.dt.month == 6)]
    O = E[~(E.RAIN_FREE & (E.start.dt.month == 6))]
    f_st = lambda D: np.mean([near(r, q3.get(r.SID, set())) for _, r in D.iterrows()]) if len(D) else np.nan
    f_di = lambda D: np.mean([near(r, dq.get(str(r.SID).split("|")[0].upper(), set())) for _, r in D.iterrows()]) if len(D) else np.nan
    say(f"  Q3-corrected days: {sum(len(v) for v in q3.values())} at {sum(1 for v in q3.values() if v)} stations")
    say(f"  June rain-free above-noise episodes {len(J)}: within ±3 d of a Q3 day at the SAME station {100 * f_st(J):.0f}% | "
        f"same DISTRICT {100 * f_di(J):.0f}%  (all other episodes: {100 * f_st(O):.0f}% | {100 * f_di(O):.0f}%)")
else:
    say("  SKIPPED - need DWLR_QC\\dwlr_daily_depth_abs.csv (raw) and diag_episodes.csv")
say("\nDA  RAINFALL 2023 / 2024 vs 1991-2020 (Jun-Oct, station grid cells)")
yrs_av = sorted(set(TT.year))
if min(yrs_av) <= 1991 and max(yrs_av) >= 2024:
    arows = []
    for sid in DH0.index:
        r = rain_at(sg.loc[sid, "lat"], sg.loc[sid, "lon"])
        jj = r[r.index.month.isin([6, 7, 8, 9, 10])]
        tot = jj.groupby(jj.index.year).sum(min_count=100)
        nrm = tot.loc[1991:2020].mean()
        arows.append({"SID": sid, "DIST": T1.loc[sid, "DIST"], "normal": nrm,
                      "2023 %": 100 * (tot.get(2023, np.nan) / nrm - 1), "2024 %": 100 * (tot.get(2024, np.nan) / nrm - 1)})
    AN = pd.DataFrame(arows)
    AN.round(1).to_csv(os.path.join(OUT, "TableR7_rain_anomaly.csv"), index=False)
    ad = AN.groupby("DIST")[["normal", "2023 %", "2024 %"]].median().round(0)
    say("  " + ad.to_string().replace("\n", "\n  "))
else:
    say(f"  SKIPPED - rain file covers {min(yrs_av)}-{max(yrs_av)}")

# ======================================================================================== figure S6
fig, axs = plt.subplots(1, 3, figsize=(19 / 2.54, 19 / 2.54 * 0.46), gridspec_kw={"width_ratios": [1.25, 1, 0.9]})
ax = axs[0]; names = [n for n, _ in VARIANTS[1:]]
for k, c in enumerate(("ALLUVIUM", "HARD_ROCK")):
    for i, n in enumerate(names):
        dh = VDH[n]; idx = dh.index[CLS.reindex(dh.index) == c]
        r = (dh[idx] / DH0.reindex(idx)).replace([np.inf, -np.inf], np.nan).dropna()
        if len(r):
            y = i + (0.18 if k else -0.18)
            ax.plot([r.quantile(.25), r.quantile(.75)], [y, y], color=COL[c], lw=1.2)
            ax.plot(r.median(), y, "o", ms=3.2, color=COL[c], label=LAB[c] if i == 0 else None)
ax.axvline(1, color=MUTED, ls=":", lw=0.8)
ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=6.5); ax.invert_yaxis()
ax.set_xlabel("ΔH relative to base (station median, IQR)"); ax.legend(loc="upper right"); ax.grid(axis="x", color=GRID, lw=0.5)
ax = axs[1]
sub = SR[(SR["noise level"] == "median field σ")]
kinds = ["gaussian", "student-t (df 3)", "AR(1) φ=0.6"]; pumps = ["none", "rabi", "rabi+kharif"]
for j, (corr, mk) in enumerate((("gauss", "o"), ("emp", "s"))):
    for i, kd in enumerate(kinds):
        q = sub[(sub["noise type"] == kd) & (sub.correction == corr)].set_index("pumping").reindex(pumps)
        ax.plot(np.arange(3) + (i - 1) * 0.12, q["mean"], marker=mk, ms=3, lw=0.8,
                color=["#2b5c8a", "#a0522d", "#6b6a64"][i], ls="-" if corr == "gauss" else "--",
                mfc="white" if corr == "emp" else None, label=f"{kd}, {'Gaussian' if corr == 'gauss' else 'empirical'} corr.")
ax.axhline(1, color=MUTED, ls=":", lw=0.8)
ax.set_xticks(range(3)); ax.set_xticklabels(["none", "rabi", "rabi +\nmonsoon"]); ax.set_xlabel("Pumping")
ax.set_ylabel("Estimated / planted rise (field σ)"); ax.legend(fontsize=5.2, loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2); ax.grid(axis="y", color=GRID, lw=0.5)
ax = axs[2]
mid = [(a_ + b_) / 2 for a_, b_ in zip(edges[:-1], edges[1:])]
okv = (TVG.pairs >= 10).values
mid = np.array(mid)[okv]
ax.plot(mid, TVG.gamma[okv], "o-", color=INK, ms=3, lw=0.9, label="all pairs")
ax.plot(mid, TVG["gamma same class"][okv], "s--", color=MUTED, ms=3, lw=0.8, mfc="white", label="same geology class")
ax.axhline(sill, color="#a0522d", ls=":", lw=0.9, label="sample variance")
ax.set_xlabel("Station separation (km)"); ax.set_ylabel("Semivariance of log10 ΔH"); ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22))
ax.grid(color=GRID, lw=0.5)
for a_, t in zip(axs, "abc"):
    a_.text(-0.02, 1.04, t, transform=a_.transAxes, fontweight="bold", fontsize=8.5, ha="right")
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"FigS6_reviewer_tests.{ext}"))
plt.close(fig)

pd.set_option("display.width", 230); pd.set_option("display.max_columns", 20)
print("=" * 100); print("VERDICT  robustness_tests"); print("=" * 100)
print("\n".join(LOG))
print(f"\nrun time {(time.time() - T_START) / 60:.1f} min | outputs: {OUT}")
