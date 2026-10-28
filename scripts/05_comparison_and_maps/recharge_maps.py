"""
SPATIAL MAPS OF RECHARGE PER DISTRICT (monthly Jun-Oct + season)  ->  C:\\Users\\IITI\\paper3_final\\
=================================================================================================
Inputs : paper3_final\\T6_monthly_station_year.csv  (from monthly_recharge.py)
         run_Q123\\DWLR_QC\\dwlr_sign_smoothness.csv  (station lat/lon)
         district boundaries:
           DIST_SHP if set (any district polygon shapefile), otherwise built by dissolving each
           GSI lithology file E:\\up_bundel\\Lithology\\**\\LITHOLOGY_DIST_*.shp (one file = one district).
         Each polygon gets the district name of the majority of DWLR stations inside it (falls back to the file name).
Maps   : district fill = median station recharge (mm) for that month / season; dots = stations (same colour scale).
Output : Fig8_recharge_maps_monthly.png, Fig9_recharge_map_season.png, T7_district_recharge_map_values.csv,
         districts_from_lithology.gpkg (the dissolved boundaries, to reuse in ArcMap).
Run    :  %run recharge_maps.py      Requires: geopandas pandas numpy matplotlib
"""
import os
import re
import sys
import glob
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shapely.geometry import box
import matplotlib.patheffects as pe

BASE = os.environ.get("TEST_BASE", r"C:\Users\IITI")
FINAL = os.path.join(BASE, "paper3_final")
T6 = os.path.join(FINAL, "T6_monthly_station_year.csv")
SIGN = os.path.join(BASE, "run_Q123", "DWLR_QC", "dwlr_sign_smoothness.csv")
DIST_SHP = os.environ.get("TEST_DIST_SHP", "")              # optional ready-made district shapefile
LITHO_DIR = os.environ.get("TEST_LITHO", r"E:\up_bundel\Lithology")
BBOX = (77.0, 23.5, 81.0, 27.0)                              # drops the Himachal Hamirpur polygons
MONTHS = {6: "Jun", 7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct"}
CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "blues", ["#f2f6fc", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

for f in (T6, SIGN):
    if not os.path.exists(f):
        sys.exit(f"[STOP] not found: {f}  (run monthly_recharge.py first)")
D = pd.read_csv(T6)
sg = pd.read_csv(SIGN).drop_duplicates("SID").set_index("SID")
D = D[D.SID.isin(sg.index)].copy()
D["DIST"] = D.DIST.astype(str).str.upper().str.strip()

# ---------------- station values: median over years per month, season = sum of months (all 5 present)
SM = D.groupby(["SID", "DIST", "CLASS", "MONTH"]).R_mm.median().reset_index()
SEAS = SM.groupby(["SID", "DIST", "CLASS"]).agg(R_mm=("R_mm", "sum"), n=("MONTH", "nunique")).reset_index()
SEAS = SEAS[SEAS.n == len(MONTHS)]
pts = gpd.GeoDataFrame(sg.loc[SM.SID.unique(), ["lat", "lon"]],
                       geometry=gpd.points_from_xy(sg.loc[SM.SID.unique(), "lon"], sg.loc[SM.SID.unique(), "lat"]),
                       crs="EPSG:4326")
pts["DIST"] = D.drop_duplicates("SID").set_index("SID").DIST.reindex(pts.index)

# ---------------- district polygons
clip = gpd.GeoDataFrame(geometry=[box(*BBOX)], crs="EPSG:4326")
if DIST_SHP and os.path.exists(DIST_SHP):
    g = gpd.read_file(DIST_SHP)
    g = (g.set_crs("EPSG:4326") if g.crs is None else g).to_crs("EPSG:4326")
    g = gpd.overlay(g, clip, how="intersection")
    g["SRC"] = g.index.astype(str)
    src = f"user shapefile {os.path.basename(DIST_SHP)}"
else:
    files = sorted(set(glob.glob(os.path.join(LITHO_DIR, "**", "LITHOLOGY_DIST_*.shp"), recursive=True)))
    files = [f for f in files if f"{os.sep}DWLR_QC{os.sep}" not in f] or files     # skip the duplicate copies
    if not files:
        sys.exit(f"[STOP] no LITHOLOGY_DIST_*.shp under {LITHO_DIR} and no DIST_SHP set")
    parts, seen = [], set()
    for f in files:
        tag = re.sub(r"^LITHOLOGY_DIST_", "", os.path.splitext(os.path.basename(f))[0], flags=re.I).upper()
        if tag in seen:
            continue
        seen.add(tag)
        x = gpd.read_file(f)
        x = (x.set_crs("EPSG:4326") if x.crs is None else x).to_crs("EPSG:4326")
        x = x[x.geometry.notna() & ~x.geometry.is_empty]
        x["geometry"] = x.geometry.buffer(0)
        geom = x.geometry.unary_union.intersection(box(*BBOX))
        if not geom.is_empty:
            parts.append({"SRC": tag, "geometry": geom})
    g = gpd.GeoDataFrame(parts, crs="EPSG:4326")
    src = f"dissolved GSI lithology files ({len(g)})"
# name each polygon by the majority district of the stations inside it
j = gpd.sjoin(pts[["DIST", "geometry"]], g[["SRC", "geometry"]], how="inner", predicate="intersects")
maj = j.groupby("SRC").DIST.agg(lambda v: v.value_counts().index[0])
g["DIST"] = g.SRC.map(maj)
g["DIST"] = g.DIST.fillna(g.SRC.astype(str).str.upper())
g = g.dissolve(by="DIST", as_index=False)
g.to_file(os.path.join(FINAL, "districts_from_lithology.gpkg"), driver="GPKG")

# ---------------- district values
VAL = {}
for m, nm in MONTHS.items():
    VAL[nm] = SM[SM.MONTH == m].groupby("DIST").R_mm.median()
VAL["Jun-Oct"] = SEAS.groupby("DIST").R_mm.median()
T7 = pd.DataFrame(VAL).round(1)
T7["n_stations"] = SEAS.groupby("DIST").SID.nunique()
T7.to_csv(os.path.join(FINAL, "T7_district_recharge_map_values.csv"))


def draw(ax, key, vals, stvals, vmax, title):
    gg = g.copy()
    gg["v"] = gg.DIST.map(vals)
    gg.plot(ax=ax, column="v", cmap=CMAP, vmin=0, vmax=vmax, edgecolor="#6b6a64", linewidth=0.6,
            missing_kwds={"color": "#eeeeee", "edgecolor": "#6b6a64", "hatch": "///"})
    p = pts.join(stvals.rename("s"), how="inner")
    ax.scatter(p.lon, p.lat, c=p.s, cmap=CMAP, vmin=0, vmax=vmax, s=14, edgecolors="#1f1f1e", linewidths=0.4, zorder=3)
    for _, r in gg.iterrows():
        if np.isfinite(r.v):
            c = r.geometry.representative_point()
            ax.annotate(f"{r.DIST.title()}\n{r.v:.0f}", (c.x, c.y), ha="center", va="center", fontsize=7,
                        fontweight="bold", color="#1f1f1e", zorder=4,
                        path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])
    ax.set_title(title, fontsize=8, loc="left"); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(""); ax.set_ylabel("")
    for s_ in ax.spines.values():
        s_.set_visible(False)


# Fig8: 5 monthly panels, one common colour scale
vmax_m = np.nanpercentile(SM.R_mm, 95)
fig, axs = plt.subplots(2, 3, figsize=(11, 7.5))
axs = axs.ravel(); axs[-1].axis("off")
for ax, (m, nm) in zip(axs, MONTHS.items()):
    st = SM[SM.MONTH == m].set_index("SID").R_mm
    draw(ax, nm, VAL[nm], st, vmax_m, f"{nm}")
sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(0, vmax_m))
cb = fig.colorbar(sm, ax=axs[-1], location="left", shrink=0.8); cb.set_label("Recharge (mm/month)"); cb.outline.set_visible(False)
fig.suptitle("Monthly monsoon recharge by district (median of stations, 2023-2024; Sy: hard rock 3 %, alluvium 6 %)",
             x=0.02, ha="left", fontsize=9)
fig.savefig(os.path.join(FINAL, "Fig8_recharge_maps_monthly.png"), dpi=300, bbox_inches="tight"); plt.close(fig)

# Fig9: season total
vmax_s = np.nanpercentile(SEAS.R_mm, 95)
fig, ax = plt.subplots(figsize=(5.2, 4.6))
draw(ax, "Jun-Oct", VAL["Jun-Oct"], SEAS.set_index("SID").R_mm, vmax_s, "Jun-Oct recharge (mm), district median; dots = stations")
sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(0, vmax_s))
cb = fig.colorbar(sm, ax=ax, shrink=0.8); cb.set_label("Recharge (mm, Jun-Oct)"); cb.outline.set_visible(False)
fig.savefig(os.path.join(FINAL, "Fig9_recharge_map_season.png"), dpi=300, bbox_inches="tight"); plt.close(fig)

print("=" * 100)
print("VERDICT  recharge_maps")
print("=" * 100)
print(f"boundaries: {src} | polygons after naming: {len(g)} -> {sorted(g.DIST.tolist())}")
print(f"stations mapped: {len(pts)} | stations outside every polygon: {len(pts) - j.index.nunique()}")
print("\nT7 district medians (mm):")
print(T7.to_string())
print(f"\nFiles in {FINAL}: Fig8_recharge_maps_monthly.png, Fig9_recharge_map_season.png, T7_district_recharge_map_values.csv,"
      f" districts_from_lithology.gpkg")
