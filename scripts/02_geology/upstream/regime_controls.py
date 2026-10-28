"""
WHAT CONTROLS THE "BUFFERED" vs "DECOUPLED" HYDROGRAPHS?  —  rivers, terrain, depth (DWLR core stations)
=====================================================================================================
Response metrics per station (from files you already have):
  SHIFT       |peak DOY 2024 - peak DOY 2023|                       (schedule_experiment.csv)
  RAIN_SHIFT  |W50 2024 - W50 2023| of station rain                  (IMD NetCDF)
  EXCESS      SHIFT - RAIN_SHIFT  (peak timing change NOT explained by rain timing)
  NSE_VAL     rain-only Pastas validation skill                      (station_models.csv)
  CAPTURE     20-May/10-Nov capture of the seasonal rise (mean of 2 yrs)
  PEAK_DOY    mean peak day | RISE mean seasonal rise | DEPTH mean May depth (m bgl)

Candidate controls (FIXED LIST, decided before looking -> report all, FDR-corrected):
  dist to river: any stream / >=10 km / >=30 km / >=100 km long rivers ; stream order of nearest river
  (if a field exists) ; DEPTH ; and, only if DEM_PATH is given: elevation, height above nearest
  river (HAND-like), local relief (elev - min elev within 2 km).

Tests: Spearman for every control x metric, Benjamini-Hochberg FDR over the whole grid,
       within-district permutation for the 5 strongest pairs (is it more than "which district"?).


Requires: pandas numpy scipy geopandas shapely xarray netCDF4 (+ rasterio only if DEM_PATH)
"""
import os
import importlib.util
if importlib.util.find_spec("rasterio"):          # rasterio's own PROJ db (avoids proj.db version clash)
    _pd = os.path.join(os.path.dirname(importlib.util.find_spec("rasterio").origin), "proj_data")
    if os.path.isdir(_pd):
        os.environ["PROJ_DATA"] = _pd
        os.environ["PROJ_LIB"] = _pd
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from shapely.geometry import Point, box
from scipy import stats

QC = r"DWLR_QC"
SIGN = os.path.join(QC, "dwlr_sign_smoothness.csv")
SCHED = os.path.join(QC, "schedule_experiment.csv")
MODELS = os.path.join(QC, "Pastas_DWLR", "station_models.csv")
RIVER_SHP = r"E:\river_network_shape\River_Network.shp"
RAIN = r"E:\up_bundel\up_bundel_rainfall_1990_2025.nc"
DEM_PATH = None                       # e.g. r"E:\dem\srtm_30m.tif"  (any CRS) -> enables terrain controls
UTM = "EPSG:32644"
BOX = (77.5, 23.5, 81.0, 27.0)
LEN_CLASSES = [0, 10, 30, 100]
N_PERM = 5000
OUT = os.path.join(QC, "regime_controls")
rng = np.random.default_rng(21)
os.makedirs(OUT, exist_ok=True)

# =====================================================================
# 1. RESPONSE METRICS
# =====================================================================
sg = pd.read_csv(SIGN)
core = sg[sg.sign_verdict.isin(["single sign (fine)", "sign errors -> |level| fixes it"])][["SID", "DIST", "lat", "lon"]]
sc = pd.read_csv(SCHED, parse_dates=["TRUE_PEAK"])
s2 = sc[sc.SCHED == "S2_current"].copy()
s2["DOY"] = s2.TRUE_PEAK.dt.dayofyear
pk = s2.pivot_table(index="SID", columns="YEAR", values="DOY")
met = pd.DataFrame({
    "SHIFT": (pk.get(2024) - pk.get(2023)).abs(),
    "PEAK_DOY": pk.mean(axis=1),
    "RISE": s2.groupby("SID").TRUE_RISE.mean(),
    "CAPTURE": s2.groupby("SID").CAPTURE.mean(),
})
m = pd.read_csv(MODELS)[["SID", "NSE_VAL"]].set_index("SID")
met = met.join(m, how="left")
df = core.merge(met, left_on="SID", right_index=True, how="inner")

# station May depth (from the daily file used everywhere else)
dd = pd.read_csv(os.path.join(QC, "dwlr_daily_depth_abs.csv"), parse_dates=["T"])
may = dd[dd["T"].dt.month == 5].groupby("SID").DEPTH_ABS.median().rename("DEPTH")
df = df.merge(may, left_on="SID", right_index=True, how="left")

# rain timing shift
ds = xr.open_dataset(RAIN)
var = [v for v in ds.data_vars if ds[v].ndim == 3][0]
la = next(c for c in ds.coords if "lat" in c.lower()); lo = next(c for c in ds.coords if "lon" in c.lower())
tm = next(c for c in ds.coords if "time" in c.lower())
R = ds[var].transpose(tm, la, lo).values.astype(float); TT = pd.to_datetime(ds[tm].values)
LA, LO = np.meshgrid(ds[la].values, ds[lo].values, indexing="ij"); valid = ~np.all(np.isnan(R), axis=0)
ds.close()


def w50(lat, lon, yr):
    d = np.hypot(LA - lat, LO - lon); d[~valid] = np.inf
    i, j = np.unravel_index(np.argmin(d), d.shape)
    s = pd.Series(R[:, i, j], index=TT).clip(lower=0).fillna(0)
    s = s[(s.index.year == yr) & (s.index.month >= 6) & (s.index.month <= 10)]
    c = s.cumsum() / s.sum()
    return c[c >= 0.5].index.min().dayofyear


df["RAIN_SHIFT"] = [abs(w50(a, b, 2024) - w50(a, b, 2023)) for a, b in zip(df.lat, df.lon)]
df["EXCESS"] = df.SHIFT - df.RAIN_SHIFT
METRICS = ["SHIFT", "EXCESS", "NSE_VAL", "CAPTURE", "PEAK_DOY", "RISE"]

# =====================================================================
# 2. CONTROLS
# =====================================================================
riv_crs = gpd.read_file(RIVER_SHP, rows=1).crs or "EPSG:4326"
bb = tuple(float(v) for v in gpd.GeoSeries([box(*BOX)], crs="EPSG:4326").to_crs(riv_crs).total_bounds)
riv = gpd.read_file(RIVER_SHP, bbox=bb)
if riv.crs is None:
    riv = riv.set_crs("EPSG:4326")
riv = riv.to_crs(UTM)
riv["LEN_KM"] = riv.length / 1000
ord_col = next((c for c in riv.columns if any(k in c.lower() for k in ("order", "strahler", "strm_ord"))), None)
print(f"rivers in window: {len(riv)} | stream-order field: {ord_col}")

pts = gpd.GeoDataFrame(df, geometry=[Point(xy) for xy in zip(df.lon, df.lat)], crs="EPSG:4326").to_crs(UTM)
CONTROLS = ["DEPTH"]
for L in LEN_CLASSES:
    sub = riv[riv.LEN_KM >= L]
    u = sub.geometry.union_all() if hasattr(sub.geometry, "union_all") else sub.unary_union
    pts[f"DRIV_{L}"] = pts.geometry.distance(u) / 1000
    CONTROLS.append(f"DRIV_{L}")
if ord_col:
    sidx = riv.sindex
    near = [riv.iloc[sidx.nearest(g, return_all=False)[1][0]][ord_col] for g in pts.geometry]
    pts["NEAR_ORDER"] = pd.to_numeric(pd.Series(near, index=pts.index), errors="coerce")
    CONTROLS.append("NEAR_ORDER")

if DEM_PATH:
    import rasterio
    with rasterio.open(DEM_PATH) as src:
        p_dem = pts.to_crs(src.crs)
        L, B, Rr, T = src.bounds

        def dem_at(xy):
            """NaN outside the DEM or on masked/nodata cells (never a silent 0)."""
            out = []
            for (x, y), v in zip(xy, src.sample(xy, masked=True)):
                inside = L <= x <= Rr and B <= y <= T
                bad = (not inside) or np.ma.is_masked(v[0]) or (src.nodata is not None and v[0] == src.nodata)
                out.append(np.nan if bad else float(v[0]))
            return np.array(out)

        pts["ELEV"] = dem_at([(g.x, g.y) for g in p_dem.geometry])
        # height above nearest stream (any length): elevation at the nearest river point
        u0 = riv.geometry.union_all() if hasattr(riv.geometry, "union_all") else riv.unary_union
        from shapely.ops import nearest_points
        npnt = gpd.GeoSeries([nearest_points(g, u0)[1] for g in pts.geometry], crs=UTM).to_crs(src.crs)
        pts["HAND"] = pts["ELEV"] - dem_at([(g.x, g.y) for g in npnt])
        # local relief within 2 km
        relief = []
        for g in p_dem.geometry:
            win = rasterio.windows.from_bounds(g.x - 2000, g.y - 2000, g.x + 2000, g.y + 2000, src.transform) \
                if src.crs.is_projected else rasterio.windows.from_bounds(g.x - 0.018, g.y - 0.018, g.x + 0.018, g.y + 0.018, src.transform)
            a = src.read(1, window=win, masked=True)
            relief.append(np.nan if a.count() == 0 else float(a.min()))
        pts["RELIEF"] = pts["ELEV"] - np.array(relief)
    CONTROLS += ["ELEV", "HAND", "RELIEF"]

D = pd.DataFrame(pts.drop(columns="geometry"))
D.to_csv(os.path.join(OUT, "station_metrics_and_controls.csv"), index=False)

# =====================================================================
# 3. TESTS
# =====================================================================
rows = []
for c in CONTROLS:
    for mtr in METRICS:
        q = D[[c, mtr]].dropna()
        if len(q) < 15:
            continue
        r, p = stats.spearmanr(q[c], q[mtr])
        rows.append({"control": c, "metric": mtr, "n": len(q), "rho": r, "p": p})
T = pd.DataFrame(rows).sort_values("p")
k = np.arange(1, len(T) + 1)
T["p_FDR"] = np.minimum.accumulate((T.p * len(T) / k)[::-1])[::-1].clip(upper=1)


def within_perm(c, mtr):
    q = D[["DIST", c, mtr]].dropna()
    x = q[c] - q.groupby("DIST")[c].transform("mean"); y = q[mtr] - q.groupby("DIST")[mtr].transform("mean")
    obs = stats.spearmanr(x, y)[0]; null = []
    for _ in range(N_PERM):
        xs = q.groupby("DIST")[c].transform(lambda s: rng.permutation(s.values))
        xs = xs - q.groupby("DIST")[c].transform("mean")
        null.append(stats.spearmanr(xs, y)[0])
    return obs, (np.sum(np.abs(null) >= abs(obs)) + 1) / (N_PERM + 1)


top = T.head(5).copy()
wp = [within_perm(r.control, r.metric) for r in top.itertuples()]
top["rho_within_district"] = [a for a, _ in wp]; top["p_within_perm"] = [b for _, b in wp]
T.to_csv(os.path.join(OUT, "controls_spearman_FDR.csv"), index=False)

# =====================================================================
# VERDICT
# =====================================================================
print("=" * 72)
print("VERDICT")
print("=" * 72)
print(f"stations {len(D)} | controls tested {len(CONTROLS)} x metrics {len(METRICS)} = {len(T)} tests | DEM used: {bool(DEM_PATH)}")
print("control medians by district:")
print(D.groupby("DIST")[CONTROLS].median().round(1).to_string())
print("\nall tests passing FDR < 0.05:")
print(T[T.p_FDR < 0.05].round(3).to_string(index=False) if (T.p_FDR < 0.05).any() else "  none")
print("\n5 strongest (raw p) + within-district permutation  [NOTE: chosen by p on same data -> a district-confound check, NOT confirmation]")
print(top[["control", "metric", "n", "rho", "p", "p_FDR", "rho_within_district", "p_within_perm"]].round(3).to_string(index=False))
