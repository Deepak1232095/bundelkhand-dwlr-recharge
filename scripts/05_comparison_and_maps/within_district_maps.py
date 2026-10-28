"""
WITHIN-DISTRICT VARIATION OF JUN-OCT RECHARGE (one panel per district)  ->  C:\\Users\\IITI\\paper3_final\\
===================================================================================================
Inputs : paper3_final\\T6_monthly_station_year.csv, paper3_final\\districts_from_lithology.gpkg (recharge_maps.py),
         run_Q123\\DWLR_QC\\dwlr_sign_smoothness.csv (lat/lon)
Method : station Jun-Oct recharge (mm; Sy hard rock 3 %, alluvium 6 %) = sum of monthly medians (all 5 months).
         Inverse-distance weighting (power 2) on a 1 km grid (UTM 44N), using stations inside the district plus
         stations within EDGE_KM of its boundary. Cells farther than MAX_KM from any station are masked (no data).
         Leave-one-out cross-validation per district: MAE of IDW vs MAE of the "district median" guess;
         skill = 1 - MAE_IDW / MAE_median (> 0 means the map pattern carries real spatial information).
Note   : recharge jumps where geology (and so Sy) changes; IDW smooths across that boundary - read with the
         station dots (marker = geology class).
Output : Fig10_within_district_recharge.png, T8_within_district_LOOCV.csv
Run    :  %run within_district_maps.py      Requires: geopandas numpy pandas matplotlib
"""
import os
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

BASE = os.environ.get("TEST_BASE", r"C:\Users\IITI")
FINAL = os.path.join(BASE, "paper3_final")
T6 = os.path.join(FINAL, "T6_monthly_station_year.csv")
GPKG = os.path.join(FINAL, "districts_from_lithology.gpkg")
SIGN = os.path.join(BASE, "run_Q123", "DWLR_QC", "dwlr_sign_smoothness.csv")
CELL_M, POWER, EDGE_KM, MAX_KM = 1000.0, 2.0, 10.0, 15.0
UTM = "EPSG:32644"
CMAP = plt.get_cmap("YlGnBu")
MRK = {"ALLUVIUM": "o", "HARD_ROCK": "^"}

for f in (T6, GPKG, SIGN):
    if not os.path.exists(f):
        sys.exit(f"[STOP] not found: {f}  (run monthly_recharge.py and recharge_maps.py first)")
D = pd.read_csv(T6)
SM = D.groupby(["SID", "DIST", "CLASS", "MONTH"]).R_mm.median().reset_index()
S = SM.groupby(["SID", "DIST", "CLASS"]).agg(R=("R_mm", "sum"), n=("MONTH", "nunique")).reset_index()
S = S[S.n == 5].set_index("SID")
sg = pd.read_csv(SIGN).drop_duplicates("SID").set_index("SID")
S = S.join(sg[["lat", "lon"]], how="inner")
S["DIST"] = S.DIST.astype(str).str.upper()
P = gpd.GeoDataFrame(S, geometry=gpd.points_from_xy(S.lon, S.lat), crs="EPSG:4326").to_crs(UTM)
P["x"], P["y"] = P.geometry.x, P.geometry.y
G = gpd.read_file(GPKG).to_crs(UTM)
G["DIST"] = G.DIST.astype(str).str.upper()


def idw(xs, ys, vs, xq, yq):
    d = np.hypot(xq[:, None] - xs[None, :], yq[:, None] - ys[None, :])
    d = np.maximum(d, 1.0)
    w = 1.0 / d ** POWER
    return (w * vs[None, :]).sum(1) / w.sum(1), d.min(1)


dists = [d for d in sorted(G.DIST) if (P.DIST == d).sum() >= 3]
vmin, vmax = np.nanpercentile(P.R, [5, 95])
rows = []
n = len(dists)
ncol = 3
nrow = int(np.ceil((n + 1) / ncol))
fig, axs = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 4.0 * nrow))
axs = axs.ravel()
for ax, d in zip(axs, dists):
    poly = G[G.DIST == d].geometry.union_all()
    own = P[P.DIST == d]
    outside = own[own.geometry.distance(poly) > 1000]          # labelled d but > 1 km outside its polygon
    near = P[(P.geometry.distance(poly) <= EDGE_KM * 1000) | (P.DIST == d)]   # own stations always included
    xs, ys, vs = near.x.values, near.y.values, near.R.values
    # grid
    x0, y0, x1, y1 = poly.bounds
    gx = np.arange(x0, x1 + CELL_M, CELL_M); gy = np.arange(y0, y1 + CELL_M, CELL_M)
    XX, YY = np.meshgrid(gx, gy)
    inside = gpd.GeoSeries(gpd.points_from_xy(XX.ravel(), YY.ravel()), crs=UTM).within(poly).values
    Z = np.full(XX.size, np.nan)
    zq, dmin = idw(xs, ys, vs, XX.ravel()[inside], YY.ravel()[inside])
    zq[dmin > MAX_KM * 1000] = np.nan
    Z[inside] = zq
    Z = Z.reshape(XX.shape)
    # LOOCV on own stations (neighbours may include edge stations)
    err_idw, err_med = [], []
    for sid, r in own.iterrows():
        rest = near.drop(index=sid, errors="ignore")
        if len(rest) < 2:
            continue
        p_, _ = idw(rest.x.values, rest.y.values, rest.R.values, np.array([r.x]), np.array([r.y]))
        err_idw.append(abs(p_[0] - r.R)); err_med.append(abs(own.drop(index=sid).R.median() - r.R))
    mae_i, mae_m = (np.mean(err_idw), np.mean(err_med)) if err_idw else (np.nan, np.nan)
    skill = 1 - mae_i / mae_m if mae_m > 0 else np.nan
    rows.append({"district": d, "n_stations": len(own), "median_mm": round(own.R.median(), 0),
                 "min_mm": round(own.R.min(), 0), "max_mm": round(own.R.max(), 0),
                 "IQR_mm": round(own.R.quantile(0.75) - own.R.quantile(0.25), 0),
                 "LOOCV_MAE_IDW": round(mae_i, 0), "LOOCV_MAE_median": round(mae_m, 0), "skill": round(skill, 2),
                 "area_covered_pct": round(100 * np.isfinite(zq).mean(), 0),
                 "own_stations_outside_polygon": "; ".join(f"{i} ({own.geometry[i].distance(poly) / 1000:.0f} km)" for i in outside.index)})
    # draw
    ax.set_facecolor("white")
    gpd.GeoSeries([poly], crs=UTM).plot(ax=ax, facecolor="#e9e9e6", edgecolor="none", hatch="////", zorder=0)
    ax.pcolormesh(XX, YY, Z, cmap=CMAP, vmin=vmin, vmax=vmax, shading="auto", zorder=1)
    gpd.GeoSeries([poly], crs=UTM).boundary.plot(ax=ax, color="#3d3d3a", linewidth=0.8, zorder=2)
    for c, q in own.groupby("CLASS"):
        ax.scatter(q.x, q.y, c=q.R, cmap=CMAP, vmin=vmin, vmax=vmax, marker=MRK.get(c, "o"), s=34,
                   edgecolors="#1f1f1e", linewidths=0.7, zorder=3)
    ax.set_title(f"{d.title()}  (n={len(own)}; median {own.R.median():.0f} mm, range {own.R.min():.0f}-{own.R.max():.0f};"
                 f" LOOCV skill {skill:+.2f})", fontsize=7.5, loc="left")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(""); ax.set_ylabel("")
    for s_ in ax.spines.values():
        s_.set_visible(False)
for ax in axs[len(dists):]:
    ax.axis("off")
cax = axs[len(dists)]
sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(vmin, vmax))
cb = fig.colorbar(sm, ax=cax, location="left", shrink=0.7)
cb.set_label("Jun-Oct recharge (mm)"); cb.outline.set_visible(False)
cax.text(0.15, 0.25, f"o alluvium (Sy 6 %)\n^ hard rock (Sy 3 %)\nhatched = > {MAX_KM:g} km from a station\nIDW (p = 2), 1 km grid",
         transform=cax.transAxes, fontsize=8, va="top")
fig.suptitle("Within-district variation of monsoon recharge, 2023-2024 (stations + IDW surface)", x=0.02, ha="left", fontsize=10)
fig.savefig(os.path.join(FINAL, "Fig10_within_district_recharge.png"), dpi=300, bbox_inches="tight"); plt.close(fig)

T8 = pd.DataFrame(rows)
T8.to_csv(os.path.join(FINAL, "T8_within_district_LOOCV.csv"), index=False)
print("=" * 100)
print("VERDICT  within_district_maps")
print("=" * 100)
print(T8.to_string(index=False))
print("\nskill > 0.2: the map pattern inside the district is supported; skill <= 0: show stations only, not the surface.")
print(f"Files in {FINAL}: Fig10_within_district_recharge.png, T8_within_district_LOOCV.csv")
