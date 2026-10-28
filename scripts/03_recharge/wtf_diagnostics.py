"""
SECTION-5 DIAGNOSTICS FOR THE MRC-WTF RECHARGE (reads the outputs of wtf_mrc_recharge.py; run that first)
=======================================================================================================
Question: how much of GROSS_C (per unit Sy) is rain-driven diffuse recharge, and how much is something else
(focused recharge near tanks/canals, canal releases, pump-off recovery, sensor steps, noise)?

 (0) NOISE NULL   per station, pure noise with the station's MRC sigma is pushed through the SAME pipeline
                  (3-day mean -> daily rate -> d > eps -> episodes merged within MERGE_GAP). Gives the noise-only
                  episode size (P95) and expected noise episodes per Jun-Oct. Episode flagged ABOVE_NOISE if
                  RECH > that station's null P95. (Gaussian noise; heavy-tailed real noise makes this lenient.)
 (a) SY CEILING   Sy_100 = rain_JJASO / GROSS_C = the Sy at which recharge would equal ALL Jun-Oct rain.
                  If Sy_100 < the low end of the class Sy norm, GROSS_C x Sy(norm) > rain -> that well's rise
                  cannot be diffuse rain recharge with a normal Sy (focused/lateral/confined/non-recharge).
                  Same with the rain-linked lower bound (c). SY_NORM below is UNVERIFIED until checked against GEC-2015.
 (b) RAIN-FREE    episodes with < RAIN_FREE_MM rain in [start-5 d, end]: share of recharge, month, district,
     RISES        class, above-noise share, 1-2 day steps, days since last >= 10 mm rain (delayed recharge
                  would sit just after wet spells; canal/pump signals would not), and SYNCHRONY: share of
                  rain-free episodes that start within +-SYNC_DAYS of a rain-free episode at another well of
                  the same district (canal release = synchronous; pump-off recovery = asynchronous), compared
                  with a null where each well's episodes are shifted by a random offset.
 (c) LOWER BOUND  LB = sum of episode recharge with rain >= RAIN_LINK_MM (per unit Sy, FULL years) and LB/GROSS_C.
                  "Rain-linked" is a timing criterion, not proof of causation.
 (d) DISTANCE     distance from each well to the nearest feature in each shapefile in WATER_SHP (tanks,
                  waterbodies, canals). Optional: skipped if WATER_SHP is empty or geopandas is missing.
                  Within-district Spearman is reported because district confounds everything.
Inference unit = station (median over its FULL years); bootstrap 95% CI over stations.
Run in Jupyter:  %run wtf_diagnostics.py      Requires: pandas numpy scipy (+ xarray for rain, geopandas for d)
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# ======================================================================================== CONFIG
QC = r"DWLR_QC"
SIGN = os.path.join(QC, "dwlr_sign_smoothness.csv")
IN = os.path.join(QC, "wtf_mrc")
OUT = os.path.join(IN, "diagnostics")
RAIN = os.environ.get("TEST_RAIN", r"E:\up_bundel\up_bundel_rainfall_1990_2025.nc")   # optional (days-since-rain)
WATER_SHP = os.environ.get("TEST_SHP", "").split(";") if os.environ.get("TEST_SHP") else [
    # r"E:\up_bundel\waterbodies.shp",   # <-- put tank / waterbody / canal shapefiles here
    # r"E:\up_bundel\canals.shp",
]
SY_NORM = {"ALLUVIUM": (0.06, 0.16), "HARD_ROCK": (0.01, 0.03)}   # UNVERIFIED placeholder; replace with GEC-2015
RAIN_FREE_MM, RAIN_LINK_MM = 10.0, 10.0
BIG_RAIN_MM = 10.0            # for "days since last big rain"
SYNC_DAYS = 2
NEAR_KM = 2.0
SMOOTH, MERGE_GAP, WIN_DAYS = 3, 2, 153        # must match wtf_mrc_recharge.py
N_NULL, N_SHIFT, N_BOOT, SEED = 300, 200, 2000, 20261013
SY_GRID = [0.01, 0.02, 0.05, 0.10]
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(SEED)

need = [os.path.join(IN, f) for f in ("mrc_parameters.csv", "recharge_station_years.csv", "recharge_episodes.csv")]
for f in need + [SIGN]:
    if not os.path.exists(f):
        sys.exit(f"[STOP] not found: {f}  (run wtf_mrc_recharge.py first, from the notebook folder)")
M = pd.read_csv(need[0])
Y = pd.read_csv(need[1])
E = pd.read_csv(need[2], parse_dates=["start", "end"])
sg = pd.read_csv(SIGN).drop_duplicates("SID").set_index("SID")
if E.empty:
    sys.exit("[STOP] no episodes in recharge_episodes.csv")

MP = M[(M.MRC == "PRIMARY") & (M.note == "ok")].set_index("SID")
P = Y[(Y.MRC == "PRIMARY") & Y.FULL.astype(bool)].copy()
FULLKEY = set(zip(P.SID, P.YEAR))
E["FULLYR"] = [(s, y) in FULLKEY for s, y in zip(E.SID, E.YEAR)]
E["SIGMA"] = E.SID.map(MP.sigma)
if "CLASS" not in E.columns:
    E["CLASS"] = np.nan
if "CLASS" not in P.columns:
    P["CLASS"] = np.nan


def ci(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return np.nan, np.nan, np.nan, len(x)
    b = np.median(rng.choice(x, (N_BOOT, len(x))), axis=1)
    return np.median(x), *np.percentile(b, [2.5, 97.5]), len(x)


def line(lab, x, fmt="{:.3f}"):
    m, lo_, hi_, n = ci(x)
    return f"  {lab:<58} {fmt.format(m)} [{fmt.format(lo_)}, {fmt.format(hi_)}]  n={n}"


# ======================================================================================== (0) noise null
def episodes_from(d, eps):
    """d: 2-D array (reps, days). Returns list of episode sums (same rule as the main script)."""
    out, n_per = [], []
    for row in d:
        idx = np.where(row > eps)[0]
        n = 0
        if len(idx):
            cur = [idx[0]]
            for k in idx[1:]:
                if k - cur[-1] <= MERGE_GAP + 1:
                    cur.append(k)
                else:
                    out.append(row[cur[0]:cur[-1] + 1][row[cur[0]:cur[-1] + 1] > 0].sum()); n += 1; cur = [k]
            out.append(row[cur[0]:cur[-1] + 1][row[cur[0]:cur[-1] + 1] > 0].sum()); n += 1
        n_per.append(n)
    return np.array(out), np.array(n_per)


NULL = {}
for sid, r in MP.iterrows():
    sig = r.sigma
    if not np.isfinite(sig) or sig <= 0:
        continue
    eps = max(2 * sig, 0.005)
    sx = 3 * sig / np.sqrt(2)                       # white level noise -> 3-day mean -> daily rate has sd = sig
    x = rng.normal(0, sx, (N_NULL, WIN_DAYS + SMOOTH + 1))
    s = np.lib.stride_tricks.sliding_window_view(x, SMOOTH, axis=1).mean(axis=2)
    rate = np.diff(s, axis=1)[:, :WIN_DAYS]
    ep, n_per = episodes_from(np.clip(rate, 0, None), eps)
    NULL[sid] = (np.percentile(ep, 95) if len(ep) else eps, n_per.mean(), ep.sum() / N_NULL)
E["NULL_P95"] = E.SID.map({k: v[0] for k, v in NULL.items()})
E["ABOVE_NOISE"] = E.RECH_m > E.NULL_P95
E["RAIN_FREE"] = E.RAIN_mm < RAIN_FREE_MM
E["RAIN_LINKED"] = E.RAIN_mm >= RAIN_LINK_MM
E["STEP"] = E.days <= 2
E["MONTH"] = E.start.dt.month

# days since last big rain (optional)
E["DAYS_SINCE_BIG"] = np.nan
if os.path.exists(RAIN):
    try:
        import xarray as xr
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
        for sid, g in E.groupby("SID"):
            if sid not in sg.index:
                continue
            dd_ = np.hypot(LA - sg.loc[sid, "lat"], LO - sg.loc[sid, "lon"]); dd_[~valid] = np.inf
            i, j = np.unravel_index(np.argmin(dd_), dd_.shape)
            rs = pd.Series(R[:, i, j], index=TT); rs = rs[~rs.index.duplicated()]
            big = rs.index[rs.values >= BIG_RAIN_MM]
            for k, t0 in g.start.items():
                prev = big[big < t0]
                E.loc[k, "DAYS_SINCE_BIG"] = (t0 - prev[-1]).days if len(prev) else np.nan
    except Exception as ex:
        print(f"[note] rain file not used ({ex})")
else:
    print(f"[note] rain file not found ({RAIN}); days-since-rain skipped")

# synchrony of rain-free episodes within district
def sync_share(df):
    hits = []
    for k, r in df.iterrows():
        oth = df[(df.DIST == r.DIST) & (df.YEAR == r.YEAR) & (df.SID != r.SID)]
        hits.append(bool(len(oth)) and bool((np.abs((oth.start - r.start).dt.days) <= SYNC_DAYS).any()))
    return np.mean(hits) if hits else np.nan, np.array(hits)


RF = E[E.RAIN_FREE & E.FULLYR & E.ABOVE_NOISE].copy()
sync_obs, sync_hits = sync_share(RF) if len(RF) > 1 else (np.nan, np.array([]))
if len(RF):
    RF["SYNC"] = sync_hits
sync_null = []
if len(RF) > 1:
    for _ in range(N_SHIFT):
        sh = RF.copy()
        off = {s: int(rng.integers(-60, 61)) for s in sh.SID.unique()}
        sh["start"] = sh.start + pd.to_timedelta(sh.SID.map(off), unit="D")
        sync_null.append(sync_share(sh)[0])
sync_null = np.array(sync_null)

# ======================================================================================== per station table
EF = E[E.FULLYR]
sy_ = EF.groupby(["SID", "YEAR"]).apply(lambda g: pd.Series({
    "EP_TOT": g.RECH_m.sum(),
    "RF_SUM": g.RECH_m[g.RAIN_FREE].sum(),
    "RF_SUM_AN": g.RECH_m[g.RAIN_FREE & g.ABOVE_NOISE].sum(),
    "LB": g.RECH_m[g.RAIN_LINKED].sum(),
    "LB_AN": g.RECH_m[g.RAIN_LINKED & g.ABOVE_NOISE].sum(),
    "STEP_SUM": g.RECH_m[g.STEP & g.ABOVE_NOISE].sum()})).reset_index()
P = P.merge(sy_, on=["SID", "YEAR"], how="left")
for c in ["EP_TOT", "RF_SUM", "RF_SUM_AN", "LB", "LB_AN", "STEP_SUM"]:
    P[c] = P[c].fillna(0.0)
P["RF_SHARE"] = np.where(P.EP_TOT > 0, P.RF_SUM / P.EP_TOT, np.nan)
P["RF_SHARE_AN"] = np.where(P.EP_TOT > 0, P.RF_SUM_AN / P.EP_TOT, np.nan)
P["LB_RATIO"] = np.where(P.GROSS_C_m > 0, P.LB / P.GROSS_C_m, np.nan)
P["SY100"] = np.where(P.GROSS_C_m > 0, P.RAIN_JJASO_mm / 1000 / P.GROSS_C_m, np.nan)
P["SY100_LB"] = np.where(P.LB > 0, P.RAIN_JJASO_mm / 1000 / P.LB, np.nan)
S = P.groupby("SID").agg(DIST=("DIST", "first"), CLASS=("CLASS", "first"), GROSS_C=("GROSS_C_m", "median"),
                         RAIN=("RAIN_JJASO_mm", "median"), RF_SHARE=("RF_SHARE", "median"),
                         RF_SHARE_AN=("RF_SHARE_AN", "median"), LB=("LB", "median"), LB_AN=("LB_AN", "median"),
                         LB_RATIO=("LB_RATIO", "median"), SY100=("SY100", "median"), SY100_LB=("SY100_LB", "median"),
                         STEP_SUM=("STEP_SUM", "median"), n_years=("YEAR", "nunique"))
S["SIGMA"] = S.index.map(MP.sigma)
S["NULL_EP_PER_WIN"] = S.index.map({k: v[1] for k, v in NULL.items()})
S["NULL_RECH_PER_WIN"] = S.index.map({k: v[2] for k, v in NULL.items()})
S["SY_LOW"] = S.CLASS.map({k: v[0] for k, v in SY_NORM.items()})
S["SY_HIGH"] = S.CLASS.map({k: v[1] for k, v in SY_NORM.items()})
S["IMPOSSIBLE_AT_SYLOW"] = S.SY100 < S.SY_LOW          # GROSS_C x Sy_low > all Jun-Oct rain

# ======================================================================================== (d) distance
DIST_COLS = []
shp_list = [p for p in WATER_SHP if p and os.path.exists(p)]
if WATER_SHP and not shp_list:
    print(f"[note] none of WATER_SHP exist: {WATER_SHP}")
if shp_list:
    try:
        import geopandas as gpd
        pts = gpd.GeoDataFrame(index=S.index, geometry=gpd.points_from_xy(sg.loc[S.index, "lon"], sg.loc[S.index, "lat"]),
                               crs="EPSG:4326").to_crs("EPSG:32644")
        for p in shp_list:
            g = gpd.read_file(p)
            if g.crs is None:
                print(f"[note] {os.path.basename(p)} has no CRS -> assumed EPSG:4326")
                g = g.set_crs("EPSG:4326")
            g = g[g.geometry.notna() & ~g.geometry.is_empty].to_crs("EPSG:32644")
            un = g.geometry.unary_union
            col = "KM_" + os.path.splitext(os.path.basename(p))[0][:20]
            S[col] = [pt.distance(un) / 1000 for pt in pts.geometry]
            DIST_COLS.append(col)
        if len(DIST_COLS) > 1:
            S["KM_ANY"] = S[DIST_COLS].min(axis=1); DIST_COLS.append("KM_ANY")
    except ImportError:
        print("[note] geopandas not installed -> (d) skipped  (pip install geopandas)")

S.to_csv(os.path.join(OUT, "diag_station.csv"))
E.to_csv(os.path.join(OUT, "diag_episodes.csv"), index=False)

# ======================================================================================== VERDICT
pd.set_option("display.width", 200)
print("=" * 100)
print("VERDICT  wtf_diagnostics")
print("=" * 100)
print(f"stations with PRIMARY MRC {len(MP)} | with FULL years {S.shape[0]} | episodes {len(E)} (in FULL years {int(E.FULLYR.sum())})")
print("\n######## (0) NOISE NULL (Gaussian, station sigma, same pipeline) ########")
print(line("noise-only episodes per Jun-Oct window", S.NULL_EP_PER_WIN, "{:.2f}"))
print(line("noise-only 'recharge' summed over episodes per window (m)", S.NULL_RECH_PER_WIN))
print(line("noise-only episode size P95 (m)", E.drop_duplicates("SID").NULL_P95))
print(f"  episodes above station noise P95: {E.ABOVE_NOISE.mean():.2f} of all | rain-free: "
      f"{E[E.RAIN_FREE].ABOVE_NOISE.mean():.2f} | rain-linked: {E[E.RAIN_LINKED].ABOVE_NOISE.mean():.2f}")

print("\n######## (a) SY CEILING  Sy_100 = rain / GROSS_C  (recharge = 100% of Jun-Oct rain) ########")
print("  SY_NORM used (UNVERIFIED):", SY_NORM)
print(line("Sy_100 from GROSS_C (fraction)", S.SY100))
print(line("Sy_100 from rain-linked lower bound LB (fraction)", S.SY100_LB))
for c, q in S.groupby("CLASS"):
    lo_ = SY_NORM.get(c, (np.nan,))[0]
    print(f"  {str(c):<10} n={len(q):3d} | Sy_100 median {q.SY100.median():.3f} | Sy_100(LB) median {q.SY100_LB.median():.3f} | "
          f"wells with Sy_100 < Sy_low({lo_}) : {int((q.SY100 < lo_).sum())} | with Sy_100(LB) < Sy_low: {int((q.SY100_LB < lo_).sum())}")
print("  recharge/rain implied at Sy grid (median over stations):  Sy -> GROSS_C based | LB based")
for sy in SY_GRID:
    print(f"    {100 * sy:4.0f}% -> {100 * (sy * 1000 * S.GROSS_C / S.RAIN).median():6.1f}% | {100 * (sy * 1000 * S.LB / S.RAIN).median():6.1f}%")

print(f"\n######## (b) RAIN-FREE RISES  (episode rain < {RAIN_FREE_MM:g} mm in [start-5 d, end]; FULL years) ########")
ef = E[E.FULLYR]
print(f"  rain-free episodes {int(ef.RAIN_FREE.sum())} of {len(ef)} ({ef.RAIN_FREE.mean():.2f}) | "
      f"above noise {int((ef.RAIN_FREE & ef.ABOVE_NOISE).sum())} | pooled share of episode recharge "
      f"{ef.RECH_m[ef.RAIN_FREE].sum() / ef.RECH_m.sum():.2f} (above noise only {ef.RECH_m[ef.RAIN_FREE & ef.ABOVE_NOISE].sum() / ef.RECH_m.sum():.2f})")
print(line("station share of episode recharge that is rain-free", S.RF_SHARE, "{:.2f}"))
print(line("  same, rain-free AND above noise", S.RF_SHARE_AN, "{:.2f}"))
print(line("median rain-free episode recharge (m)", ef.RECH_m[ef.RAIN_FREE]))
print(line("median rain-linked episode recharge (m)", ef.RECH_m[ef.RAIN_LINKED]))
rfa = ef[ef.RAIN_FREE & ef.ABOVE_NOISE]
if len(rfa):
    mo = rfa.groupby("MONTH").RECH_m.agg(["size", "sum"])
    mo["share_of_RF_rech"] = (mo["sum"] / mo["sum"].sum()).round(2)
    print("  above-noise rain-free episodes by start month:", mo[["size", "share_of_RF_rech"]].to_dict("index"))
    print("  by district: (n wells | station median RF share (above noise) | rain-free episodes)")
    for d, q in S.groupby("DIST"):
        print(f"    {d:<10} {len(q):3d} | {q.RF_SHARE_AN.median():.2f} | {int((rfa.DIST == d).sum())}")
    for c, q in S.groupby("CLASS"):
        print(f"    class {str(c):<10} n={len(q):3d} | RF share (above noise) {q.RF_SHARE_AN.median():.2f}")
    print(f"  1-2 day 'step' episodes among above-noise rain-free: {rfa.STEP.mean():.2f} "
          f"(their recharge share {rfa.RECH_m[rfa.STEP].sum() / rfa.RECH_m.sum():.2f})")
    if rfa.DAYS_SINCE_BIG.notna().any():
        dsb = rfa.DAYS_SINCE_BIG.dropna()
        print(f"  days since last >= {BIG_RAIN_MM:g} mm rain: median {dsb.median():.0f} | <= 10 d {np.mean(dsb <= 10):.2f} | "
              f"11-30 d {np.mean((dsb > 10) & (dsb <= 30)):.2f} | > 30 d {np.mean(dsb > 30):.2f}")
    if np.isfinite(sync_obs) and len(sync_null):
        p_sync = (np.sum(sync_null >= sync_obs) + 1) / (len(sync_null) + 1)
        print(f"  SYNCHRONY within district (+-{SYNC_DAYS} d): observed {sync_obs:.2f} | random-shift null median "
              f"{np.median(sync_null):.2f} [P95 {np.percentile(sync_null, 95):.2f}] | p = {p_sync:.3f}")
        if "SYNC" in rfa.columns or "SYNC" in RF.columns:
            sync_m = RF.groupby("MONTH").SYNC.mean().round(2).to_dict()
            print(f"  synchronous share by month: {sync_m}")
else:
    print("  no above-noise rain-free episodes")

print(f"\n######## (c) RAIN-LINKED LOWER BOUND  (episodes with rain >= {RAIN_LINK_MM:g} mm; per unit Sy) ########")
print(line("LB (m)", S.LB))
print(line("LB, above-noise episodes only (m)", S.LB_AN))
print(line("GROSS_C (m)  [for reference]", S.GROSS_C))
print(line("LB / GROSS_C", S.LB_RATIO, "{:.2f}"))
for c, q in S.groupby("CLASS"):
    print(f"    class {str(c):<10} LB {q.LB.median():.2f} m | LB/GROSS_C {q.LB_RATIO.median():.2f} | n={len(q)}")
for d, q in S.groupby("DIST"):
    print(f"    {d:<10} LB {q.LB.median():.2f} m | LB/GROSS_C {q.LB_RATIO.median():.2f} | n={len(q)}")

print("\n######## (d) DISTANCE TO TANKS / WATERBODIES / CANALS ########")
if not DIST_COLS:
    print("  SKIPPED: set WATER_SHP in CONFIG (tank/waterbody/canal shapefiles)")
for col in DIST_COLS:
    near = S[col] <= NEAR_KM
    print(f"  {col}: median {S[col].median():.1f} km | wells <= {NEAR_KM:g} km: {int(near.sum())} of {len(S)}")
    for y_ in ["GROSS_C", "RF_SHARE_AN", "SY100", "LB_RATIO"]:
        ok = S[[col, y_]].dropna()
        if len(ok) < 8:
            continue
        rho, p = stats.spearmanr(ok[col], ok[y_])
        wd = []
        for d, q in ok.join(S.DIST).groupby("DIST"):
            if len(q) >= 5:
                wd.append(stats.spearmanr(q[col], q[y_])[0])
        a_, b_ = ok[y_][ok[col] <= NEAR_KM], ok[y_][ok[col] > NEAR_KM]
        mw = stats.mannwhitneyu(a_, b_).pvalue if len(a_) >= 3 and len(b_) >= 3 else np.nan
        print(f"    {y_:<12} Spearman {rho:+.2f} (p {p:.3f}) | within-district median {np.nanmedian(wd) if wd else np.nan:+.2f} "
              f"({len(wd)} districts) | near {a_.median():.3f} vs far {b_.median():.3f} (MW p {mw:.3f})")

print("\nNOTES: all recharge is per unit Sy. Rain-free != non-recharge (could be delayed/lateral recharge); see days-since-rain")
print("and synchrony. SY_NORM is a placeholder until verified against GEC-2015.")
print(f"Outputs: {OUT}")
