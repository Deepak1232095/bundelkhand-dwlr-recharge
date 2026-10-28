"""
WITHIN-DISTRICT VARIATION OF JUN-OCT RECHARGE, v2 (layout fix)  ->  C:\\Users\\IITI\\paper3_final\\
=================================================================================================
Same data and calculation as within_district_maps.py (station Jun-Oct recharge = sum of monthly medians, Sy hard rock
3 %, alluvium 6 %; IDW power 2 on a 1 km UTM 44N grid; own stations + stations within EDGE_KM; cells > MAX_KM from a
station masked; LOOCV skill = 1 - MAE_IDW / MAE_median). Only the figure is redesigned:
  - five panels of identical size AND identical map scale (same km span), so districts can be compared directly
  - panel title = district (n) only; median, range and LOOCV skill go to the caption / Table S2 (T8_v2)
  - one shared colour bar + a real symbol legend in the sixth cell, one scale bar per panel, north arrow
  - BOUNDARY = "gadm" (default): official GADM district polygons (same as Figure 2). This removes the straight
    sheet-edge lines and gaps of the lithology-derived polygons. Because the polygon defines the interpolation mask and
    the edge stations, LOOCV skill can differ slightly from the old T8 -> the VERDICT prints old vs new.
    BOUNDARY = "lithology" reproduces the old polygons (and the old numbers) exactly.
Output : Fig10_within_district_recharge_v2.png (600 dpi) + .pdf, T8_within_district_LOOCV_v2.csv
Run    :  %run within_district_maps_v2.py      Requires: geopandas numpy pandas matplotlib.   
"""
import os
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle, Polygon as MPoly

BASE = os.environ.get("TEST_BASE", r"C:\Users\IITI")
FINAL = os.path.join(BASE, "paper3_final")
T6 = os.path.join(FINAL, "T6_monthly_station_year.csv")
GPKG = os.path.join(FINAL, "districts_from_lithology.gpkg")
DIST_SHP = os.environ.get("TEST_DIST", r"E:\up_bundel\up_bundel_boxes\up_5dist.shp")
DIST_COL = "NAME_2"
BOUNDARY = "gadm"                                    # "gadm" or "lithology"
SIGN = os.path.join(BASE, "run_Q123", "DWLR_QC", "dwlr_sign_smoothness.csv")
T8_OLD = os.path.join(FINAL, "T8_within_district_LOOCV.csv")
CELL_M, POWER, EDGE_KM, MAX_KM = 1000.0, 2.0, 10.0, 15.0   # as in within_district_maps.py
UTM = "EPSG:32644"
BBOX = (77.0, 23.5, 81.0, 27.0)
SCALE_KM = 25

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                     "font.size": 7.5, "legend.fontsize": 7, "legend.frameon": False, "axes.grid": False,
                     "pdf.fonttype": 42, "savefig.bbox": "tight"})
INK, MUTED, EDGE = "#1f1f1e", "#6b6a64", "#3d3d3a"
CMAP = plt.get_cmap("YlGnBu")
MRK = {"ALLUVIUM": "o", "HARD_ROCK": "^"}

for f in (T6, SIGN) + ((DIST_SHP,) if BOUNDARY == "gadm" else (GPKG,)):
    if not os.path.exists(f):
        sys.exit(f"[STOP] not found: {f}")
D = pd.read_csv(T6)
SM = D.groupby(["SID", "DIST", "CLASS", "MONTH"]).R_mm.median().reset_index()
S = SM.groupby(["SID", "DIST", "CLASS"]).agg(R=("R_mm", "sum"), n=("MONTH", "nunique")).reset_index()
S = S[S.n == 5].set_index("SID")
sg = pd.read_csv(SIGN).drop_duplicates("SID").set_index("SID")
S = S.join(sg[["lat", "lon"]], how="inner")
S["DIST"] = S.DIST.astype(str).str.upper()
P = gpd.GeoDataFrame(S, geometry=gpd.points_from_xy(S.lon, S.lat), crs="EPSG:4326").to_crs(UTM)
P["x"], P["y"] = P.geometry.x, P.geometry.y
if BOUNDARY == "gadm":
    G = gpd.read_file(DIST_SHP)
    G = (G.set_crs("EPSG:4326") if G.crs is None else G).to_crs("EPSG:4326")
    G = G[G.geometry.representative_point().within(box(*BBOX))].copy()
    G["DIST"] = G[DIST_COL].astype(str).str.upper().str.strip()
    G["geometry"] = G.geometry.buffer(0)
    G = G.dissolve(by="DIST", as_index=False).to_crs(UTM)
else:
    G = gpd.read_file(GPKG).to_crs(UTM)
    G["DIST"] = G.DIST.astype(str).str.upper()


def idw(xs, ys, vs, xq, yq):
    d = np.hypot(xq[:, None] - xs[None, :], yq[:, None] - ys[None, :])
    d = np.maximum(d, 1.0)
    w = 1.0 / d ** POWER
    return (w * vs[None, :]).sum(1) / w.sum(1), d.min(1)


dists = [d for d in sorted(G.DIST) if (P.DIST == d).sum() >= 3]
vmin, vmax = np.nanpercentile(P.R, [5, 95])
rows, panels = [], []
for d in dists:
    poly = G[G.DIST == d].geometry.union_all()
    own = P[P.DIST == d]
    outside = own[own.geometry.distance(poly) > 1000]
    near = P[(P.geometry.distance(poly) <= EDGE_KM * 1000) | (P.DIST == d)]
    xs, ys, vs = near.x.values, near.y.values, near.R.values
    x0, y0, x1, y1 = poly.bounds
    gx = np.arange(x0, x1 + CELL_M, CELL_M); gy = np.arange(y0, y1 + CELL_M, CELL_M)
    XX, YY = np.meshgrid(gx, gy)
    inside = gpd.GeoSeries(gpd.points_from_xy(XX.ravel(), YY.ravel()), crs=UTM).within(poly).values
    Z = np.full(XX.size, np.nan)
    zq, dmin = idw(xs, ys, vs, XX.ravel()[inside], YY.ravel()[inside])
    zq[dmin > MAX_KM * 1000] = np.nan
    Z[inside] = zq
    Z = Z.reshape(XX.shape)
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
                 "own_stations_outside_polygon": "; ".join(f"{i} ({own.geometry[i].distance(poly) / 1000:.0f} km)"
                                                           for i in outside.index)})
    panels.append((d, poly, own, XX, YY, Z, skill))

# ---------------------------------------------------------------- figure: equal-size panels, equal map scale
span = max(max(p[1].bounds[2] - p[1].bounds[0], p[1].bounds[3] - p[1].bounds[1]) for p in panels) * 1.06
fig = plt.figure(figsize=(7.4, 5.6))
gs = fig.add_gridspec(2, 3, wspace=0.08, hspace=0.16)
cells = [gs[0, 0], gs[0, 1], gs[0, 2], gs[1, 0], gs[1, 1]]
for (d, poly, own, XX, YY, Z, skill), cell in zip(panels, cells):
    ax = fig.add_subplot(cell)
    gpd.GeoSeries([poly], crs=UTM).plot(ax=ax, facecolor="#ececea", edgecolor="none", hatch="////", lw=0, zorder=0)
    ax.pcolormesh(XX, YY, Z, cmap=CMAP, vmin=vmin, vmax=vmax, shading="auto", zorder=1, rasterized=True)
    gpd.GeoSeries([poly], crs=UTM).boundary.plot(ax=ax, color=EDGE, linewidth=0.7, zorder=2)
    for c, q in own.groupby("CLASS"):
        ax.scatter(q.x, q.y, c=q.R, cmap=CMAP, vmin=vmin, vmax=vmax, marker=MRK.get(c, "o"), s=26,
                   edgecolors=INK, linewidths=0.6, zorder=3)
    cx, cy = (poly.bounds[0] + poly.bounds[2]) / 2, (poly.bounds[1] + poly.bounds[3]) / 2
    ax.set_xlim(cx - span / 2, cx + span / 2); ax.set_ylim(cy - span / 2, cy + span / 2)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(""); ax.set_ylabel("")
    for s_ in ax.spines.values():
        s_.set_visible(True); s_.set_color(MUTED); s_.set_linewidth(0.5)
    ax.set_title(f"{d.title()} (n = {len(own)})", loc="left", fontsize=8, fontweight="bold", pad=4)
    X0, X1 = ax.get_xlim(); Y0, Y1 = ax.get_ylim()
    bx, by, hh = X0 + 0.05 * (X1 - X0), Y0 + 0.05 * (Y1 - Y0), 0.012 * (Y1 - Y0)
    ax.add_patch(Rectangle((bx, by), SCALE_KM * 500, hh, fc=INK, ec=INK, lw=0.4, zorder=5))
    ax.add_patch(Rectangle((bx + SCALE_KM * 500, by), SCALE_KM * 500, hh, fc="white", ec=INK, lw=0.4, zorder=5))
    ax.text(bx, by + 2 * hh, "0", fontsize=6, ha="center", va="bottom")
    ax.text(bx + SCALE_KM * 1000, by + 2 * hh, f"{SCALE_KM} km", fontsize=6, ha="center", va="bottom")
    nx, ny, w, h = X0 + 0.92 * (X1 - X0), Y0 + 0.86 * (Y1 - Y0), 0.025 * (X1 - X0), 0.07 * (Y1 - Y0)
    ax.add_patch(MPoly([[nx - w, ny - h / 2], [nx, ny + h / 2], [nx + w, ny - h / 2], [nx, ny - h / 4]], closed=True,
                       fc=INK, ec=INK, lw=0.4, zorder=5))
    ax.text(nx, ny + h / 2 + 0.01 * (Y1 - Y0), "N", ha="center", va="bottom", fontsize=6.5, fontweight="bold")

lg = fig.add_subplot(gs[1, 2]); lg.axis("off")
cax = lg.inset_axes((0.08, 0.80, 0.84, 0.06))
cb = fig.colorbar(plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(vmin, vmax)), cax=cax, orientation="horizontal",
                  extend="both")
cb.set_label("June–October recharge (mm)", fontsize=7); cb.ax.tick_params(labelsize=6.5); cb.outline.set_linewidth(0.4)
hand = [Line2D([], [], ls="", marker="o", ms=5, mfc="#9ec5d8", mec=INK, mew=0.6, label="Alluvium station (Sy 6%)"),
        Line2D([], [], ls="", marker="^", ms=5, mfc="#9ec5d8", mec=INK, mew=0.6, label="Hard-rock station (Sy 3%)"),
        Patch(fc="#ececea", ec=MUTED, hatch="////", lw=0.3, label=f"> {MAX_KM:g} km from a station"),
        Patch(fc="white", ec="white", label=f"IDW (p = {POWER:g}), {CELL_M / 1000:g} km grid; same map scale")]
lg.legend(handles=hand, loc="upper left", bbox_to_anchor=(0.04, 0.62), handletextpad=0.5)
fig.savefig(os.path.join(FINAL, "Fig10_within_district_recharge_v2.png"), dpi=600)
fig.savefig(os.path.join(FINAL, "Fig10_within_district_recharge_v2.pdf"))
plt.close(fig)

T8 = pd.DataFrame(rows)
T8.to_csv(os.path.join(FINAL, "T8_within_district_LOOCV_v2.csv"), index=False)
print("=" * 90)
print("VERDICT  within_district_maps_v2")
print("=" * 90)
print(f"boundaries: {BOUNDARY} | colour limits (5-95 pct of stations) {vmin:.0f}-{vmax:.0f} mm | common map span {span / 1000:.0f} km")
print(T8.to_string(index=False))
if os.path.exists(T8_OLD):
    o = pd.read_csv(T8_OLD).set_index("district")
    cmp = T8.set_index("district")[["n_stations", "median_mm", "skill", "area_covered_pct"]].join(
        o[["n_stations", "median_mm", "skill", "area_covered_pct"]], rsuffix="_old")
    print("\nnew vs old (T8_within_district_LOOCV.csv):")
    print(cmp.to_string())
    print("  median/range must be identical (same stations); skill and area may change only if BOUNDARY = gadm")
print(f"\nFiles in {FINAL}: Fig10_within_district_recharge_v2.png, .pdf, T8_within_district_LOOCV_v2.csv")
