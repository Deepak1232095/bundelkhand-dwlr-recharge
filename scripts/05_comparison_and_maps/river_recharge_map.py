"""
RECHARGE MAP WITH RIVERS + RECHARGE vs DISTANCE TO RIVER  ->  C:\\Users\\IITI\\paper3_final\\
=========================================================================================
Inputs : paper3_final\\T6_monthly_station_year.csv, districts_from_lithology.gpkg (recharge_maps.py),
         run_Q123\\DWLR_QC\\dwlr_sign_smoothness.csv (lat/lon),
         optional: T9_combined_validation.csv + T9b_cgwb_wells_recharge_equiv.csv (combined_maps.py) ->
                   CGWB wells are added ONLY in districts where T9 says "yes",
         river network shapefile RIVER_SHP (all-India; read only inside the study window).
Map    : one regional IDW surface (power 2, 1 km grid, UTM 44N) of Jun-Oct recharge (mm), clipped to the 5 districts,
         cells > 15 km from a point masked; rivers on top (thicker = longer), major rivers labelled.
Test   : distance of every DWLR station to the nearest river (>= 30 km long, and >= 100 km long);
         recharge per distance bin and Spearman r, separately per geology class (alluvium sits along the
         Yamuna/Betwa, so a pooled correlation would only show geology).
CAUTION: a well close to a large river can rise with river stage (bank storage); MRC-WTF counts that rise as
         recharge. High values within ~2 km of a big river need that caveat.
Output : Fig12_recharge_rivers.png, Fig12b_recharge_vs_river_distance.png, T10_recharge_vs_river_distance.csv,
         T10b_station_river_distance.csv
Run    :  %run river_recharge_map.py      Requires: geopandas numpy pandas scipy matplotlib
"""
import os
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D

BASE = os.environ.get("TEST_BASE", r"C:\Users\IITI")
FINAL = os.path.join(BASE, "paper3_final")
T6 = os.path.join(FINAL, "T6_monthly_station_year.csv")
GPKG = os.path.join(FINAL, "districts_from_lithology.gpkg")
T9 = os.path.join(FINAL, "T9_combined_validation.csv")
T9B = os.path.join(FINAL, "T9b_cgwb_wells_recharge_equiv.csv")
if os.path.exists(os.path.join(FINAL, "T9_3r_validation.csv")) and os.environ.get("TEST_CGWB_SET", "2r") == "3r":
    T9 = os.path.join(FINAL, "T9_3r_validation.csv")             # 3-reading CGWB file (combined_maps_3reading.py)
    T9B = os.path.join(FINAL, "T9b_3r_cgwb_wells.csv")
SIGN = os.path.join(BASE, "run_Q123", "DWLR_QC", "dwlr_sign_smoothness.csv")
RIVER_SHP = os.environ.get("TEST_RIVER", r"E:\river_network_shape\River_Network.shp")
UTM = "EPSG:32644"
CELL_M, POWER, MAX_KM = 1000.0, 2.0, 15.0
RIV_MIN_KM = (10, 30, 100)                 # drawn / "major" for the test / "large" for the test
BINS = [0, 2, 5, 10, 20, 1e9]
BIN_LAB = ["0-2", "2-5", "5-10", "10-20", ">20"]
MAJOR_NAMES = ("Betwa", "Dhasan", "Ken", "Yamuna", "Pahuj", "Jamni", "Birma", "Sajnam", "Shahzad", "Urmil",
               "Chandrawal", "Lakheri", "Bairma", "Shahzad", "Kuwari", "Sindh", "Jamuna")
CMAP = plt.get_cmap("RdYlBu")              # red = low, yellow = middle, blue = high
MRK = {"ALLUVIUM": "o", "HARD_ROCK": "^"}

for f in (T6, GPKG, SIGN, RIVER_SHP):
    if not os.path.exists(f):
        sys.exit(f"[STOP] not found: {f}\n  (river file elsewhere? set os.environ['TEST_RIVER'] = r'...\\River_Network.shp')")

# ---------------- DWLR stations: Jun-Oct recharge (sum of monthly medians, all 5 months)
D = pd.read_csv(T6)
SM = D.groupby(["SID", "DIST", "CLASS", "MONTH"]).R_mm.median().reset_index()
S = SM.groupby(["SID", "DIST", "CLASS"]).agg(R=("R_mm", "sum"), n=("MONTH", "nunique")).reset_index()
S = S[S.n == 5].set_index("SID")
sg = pd.read_csv(SIGN).drop_duplicates("SID").set_index("SID")
S = S.join(sg[["lat", "lon"]], how="inner")
S["DIST"] = S.DIST.astype(str).str.upper()
Pd = gpd.GeoDataFrame(S, geometry=gpd.points_from_xy(S.lon, S.lat), crs="EPSG:4326").to_crs(UTM)
Pd["SRC"] = "DWLR"

# ---------------- CGWB wells only where combined_maps validated them
Pc = gpd.GeoDataFrame(columns=["R", "DIST", "CLASS", "SRC", "geometry"], geometry="geometry", crs=UTM)
cg_note = "CGWB not used (run combined_maps.py first to add them)"
if os.path.exists(T9) and os.path.exists(T9B):
    t9 = pd.read_csv(T9)
    ok = set(t9[t9.use_combined_map.astype(str).str.lower().str.startswith("yes")].district.str.upper())
    c = pd.read_csv(T9B)
    c = c[c.DIST.astype(str).str.upper().isin(ok) & c.R.notna()]
    if len(c):
        Pc = gpd.GeoDataFrame(c[["R", "DIST", "CLASS"]].assign(SRC="CGWB"),
                              geometry=gpd.points_from_xy(c.LON, c.LAT), crs="EPSG:4326").to_crs(UTM)
    cg_note = f"[{os.path.basename(T9)}] CGWB wells added in {sorted(ok) or 'no district'} ({len(Pc)} wells; T9 skill > 0 only)"
P = pd.concat([Pd[["R", "DIST", "CLASS", "SRC", "geometry"]], Pc[["R", "DIST", "CLASS", "SRC", "geometry"]]])
P = gpd.GeoDataFrame(P, geometry="geometry", crs=UTM)
P["x"], P["y"] = P.geometry.x, P.geometry.y

G = gpd.read_file(GPKG).to_crs(UTM)
G["DIST"] = G.DIST.astype(str).str.upper()
REG = G.geometry.union_all()

# ---------------- rivers (read only the window)
head = gpd.read_file(RIVER_SHP, rows=1)
crs_r = head.crs if head.crs is not None else "EPSG:4326"
win = gpd.GeoSeries([REG.buffer(30000)], crs=UTM).to_crs(crs_r).total_bounds
riv = gpd.read_file(RIVER_SHP, bbox=tuple(win))
riv = (riv.set_crs("EPSG:4326") if riv.crs is None else riv).to_crs(UTM)
riv = riv[riv.geometry.notna() & ~riv.geometry.is_empty]
riv = riv[riv.geom_type.isin(["LineString", "MultiLineString"])].copy()
# full length of each feature (before clipping), used for "major river" classes
riv["L_km"] = riv.length / 1000
lcol = next((c for c in riv.columns if c.lower() in ("length_km", "len_km", "length", "shape_leng")), None)
if lcol is not None and pd.api.types.is_numeric_dtype(riv[lcol]):
    v = riv[lcol].astype(float)
    riv["L_km"] = np.where(v.median() > 1000, v / 1000, v) if lcol.lower() in ("length", "shape_leng") else v
ncol = next((c for c in riv.columns if any(k in c.lower() for k in ("rivname", "riv_name", "river_name", "name", "rname"))), None)
riv["NAME"] = riv[ncol].astype(str).str.strip().str.title() if ncol else ""
# a named river split into many short segments: length of the river = sum of its segments
if ncol:
    tot = riv.groupby("NAME").L_km.transform("sum")
    riv["L_km"] = np.where(riv.NAME.isin(["", "Nan", "None", "0"]), riv.L_km, np.maximum(riv.L_km, tot))
clip = REG.buffer(5000)
rc = riv[riv.intersects(clip)].copy()
rc["geometry"] = rc.geometry.intersection(clip)
rc = rc[~rc.geometry.is_empty]
if rc.empty:
    sys.exit("[STOP] no river lines inside the study area - check the river file / CRS")


# ---------------- regional IDW surface
def idw(xs, ys, vs, xq, yq):
    d = np.maximum(np.hypot(xq[:, None] - xs[None, :], yq[:, None] - ys[None, :]), 1.0)
    w = 1.0 / d ** POWER
    return (w * vs[None, :]).sum(1) / w.sum(1), d.min(1)


x0, y0, x1, y1 = REG.bounds
gx = np.arange(x0, x1 + CELL_M, CELL_M); gy = np.arange(y0, y1 + CELL_M, CELL_M)
XX, YY = np.meshgrid(gx, gy)
inside = gpd.GeoSeries(gpd.points_from_xy(XX.ravel(), YY.ravel()), crs=UTM).within(REG).values
Z = np.full(XX.size, np.nan)
xq, yq = XX.ravel()[inside], YY.ravel()[inside]
zq = np.empty(len(xq)); dq = np.empty(len(xq))
for i in range(0, len(xq), 20000):                       # chunks keep memory small
    zq[i:i + 20000], dq[i:i + 20000] = idw(P.x.values, P.y.values, P.R.values, xq[i:i + 20000], yq[i:i + 20000])
zq[dq > MAX_KM * 1000] = np.nan
Z[inside] = zq
Z = Z.reshape(XX.shape)
vmin, vmax = np.nanpercentile(P.R, [5, 95])

# ---------------- Fig12: map
fig, ax = plt.subplots(figsize=(8.6, 8.0))
ax.set_facecolor("white")
G.plot(ax=ax, facecolor="#ececea", edgecolor="none", hatch="////", zorder=0)
ax.pcolormesh(XX, YY, Z, cmap=CMAP, vmin=vmin, vmax=vmax, shading="auto", zorder=1)
G.boundary.plot(ax=ax, color="#3d3d3a", linewidth=1.0, zorder=3)
RIVC = "#0b2e59"
for lo_, hi_, lw in ((RIV_MIN_KM[0], RIV_MIN_KM[1], 0.6), (RIV_MIN_KM[1], RIV_MIN_KM[2], 1.1), (RIV_MIN_KM[2], 1e9, 2.0)):
    q = rc[(rc.L_km >= lo_) & (rc.L_km < hi_)]
    if len(q):
        q.plot(ax=ax, color=RIVC, linewidth=lw, zorder=4,
               path_effects=[pe.Stroke(linewidth=lw + 1.2, foreground="white"), pe.Normal()])
for c_, q in P.groupby(["SRC", "CLASS"]):
    src, cl = c_
    ax.scatter(q.x, q.y, c=q.R, cmap=CMAP, vmin=vmin, vmax=vmax, marker="s" if src == "CGWB" else MRK.get(cl, "o"),
               s=12 if src == "CGWB" else 40, edgecolors="#1f1f1e", linewidths=0.6, zorder=5)
# river labels: one per named major river, at the midpoint of its longest clipped piece
if ncol:
    big = rc[rc.NAME.str.contains("|".join(MAJOR_NAMES), case=False, na=False) | (rc.L_km >= RIV_MIN_KM[2])]
    big = big[~big.NAME.isin(["", "Nan", "None", "0"])]
    for nm, q in big.groupby("NAME"):
        g_ = q.geometry.iloc[np.argmax(q.length.values)]
        pt = g_.interpolate(0.5, normalized=True)
        ax.annotate(nm, (pt.x, pt.y), fontsize=7.5, style="italic", color=RIVC, ha="center", va="bottom", zorder=6,
                    path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])
for _, r in G.iterrows():
    c = r.geometry.representative_point()
    ax.annotate(r.DIST.title(), (c.x, c.y), fontsize=9, fontweight="bold", color="#1f1f1e", ha="center", zorder=6,
                path_effects=[pe.withStroke(linewidth=3, foreground="white")])
ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(""); ax.set_ylabel("")
for s_ in ax.spines.values():
    s_.set_visible(False)
sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(vmin, vmax))
cb = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.01); cb.set_label("Jun-Oct recharge (mm)"); cb.outline.set_visible(False)
hand = [Line2D([], [], marker="o", ls="", mfc="#bbbbbb", mec="k", label="DWLR, alluvium (Sy 6 %)"),
        Line2D([], [], marker="^", ls="", mfc="#bbbbbb", mec="k", label="DWLR, hard rock (Sy 3 %)"),
        Line2D([], [], color=RIVC, lw=2.0, label=f"river >= {RIV_MIN_KM[2]} km"),
        Line2D([], [], color=RIVC, lw=0.6, label=f"stream {RIV_MIN_KM[0]}-{RIV_MIN_KM[1]} km")]
if len(Pc):
    hand.insert(2, Line2D([], [], marker="s", ls="", ms=4, mfc="#bbbbbb", mec="k", label="CGWB (biannual, scaled)"))
ax.legend(handles=hand, loc="upper left", bbox_to_anchor=(0, -0.01), ncol=3, fontsize=7.5, frameon=False)
ax.set_title(f"Monsoon (Jun-Oct) recharge 2023-2024 with the river network\n"
             f"IDW (p = 2, 1 km); hatched = > {MAX_KM:g} km from a well", loc="left", fontsize=9.5)
fig.savefig(os.path.join(FINAL, "Fig12_recharge_rivers.png"), dpi=300, bbox_inches="tight"); plt.close(fig)

# ---------------- test: recharge vs distance to river (DWLR stations only = measured values)
for L in RIV_MIN_KM[1:]:
    sub = riv[riv.L_km >= L]
    Pd[f"dist_km_ge{L}"] = Pd.geometry.distance(sub.geometry.union_all()) / 1000 if len(sub) else np.nan
rows = []
for L in RIV_MIN_KM[1:]:
    col = f"dist_km_ge{L}"
    for cl, q in list(Pd.groupby("CLASS")) + [("ALL", Pd)]:
        q = q[np.isfinite(q[col])]
        if len(q) < 5:
            continue
        r, p = stats.spearmanr(q[col], q.R)
        b = pd.cut(q[col], BINS, labels=BIN_LAB, right=False)
        med = q.groupby(b, observed=False).R.agg(["median", "size"])
        row = {"rivers": f">= {L} km", "class": cl, "n": len(q), "median_dist_km": round(q[col].median(), 1),
               "spearman_r": round(r, 2), "p": round(p, 3)}
        for lab in BIN_LAB:
            row[f"R_{lab}km"] = f"{med.loc[lab, 'median']:.0f} (n{int(med.loc[lab, 'size'])})" if med.loc[lab, "size"] else "-"
        rows.append(row)
T10 = pd.DataFrame(rows)
T10.to_csv(os.path.join(FINAL, "T10_recharge_vs_river_distance.csv"), index=False)
Pd.drop(columns="geometry").to_csv(os.path.join(FINAL, "T10b_station_river_distance.csv"))

col = f"dist_km_ge{RIV_MIN_KM[1]}"
fig, ax = plt.subplots(figsize=(5.6, 3.8))
colc = {"ALLUVIUM": "#2a78d6", "HARD_ROCK": "#eb6834"}
for cl, q in Pd.groupby("CLASS"):
    ax.scatter(q[col].clip(lower=0.1), q.R, s=30, marker=MRK.get(cl, "o"), color=colc.get(cl, "grey"),
               edgecolors="#1f1f1e", linewidths=0.5, label=cl.replace("_", " ").title())
ax.set_xscale("log"); ax.set_xlabel(f"Distance to nearest river >= {RIV_MIN_KM[1]} km long (km)")
ax.set_ylabel("Jun-Oct recharge (mm)"); ax.axvline(2, color="#888", lw=0.8, ls="--")
ax.text(2.1, ax.get_ylim()[1] * 0.95, "2 km", fontsize=7, color="#555", va="top")
ax.legend(frameon=False, fontsize=8)
for s_ in ("top", "right"):
    ax.spines[s_].set_visible(False)
ax.set_title("DWLR station recharge vs distance to river", loc="left", fontsize=9)
fig.savefig(os.path.join(FINAL, "Fig12b_recharge_vs_river_distance.png"), dpi=300, bbox_inches="tight"); plt.close(fig)

pd.set_option("display.width", 220)
print("=" * 100)
print("VERDICT  river_recharge_map")
print("=" * 100)
print(f"river file: {os.path.basename(RIVER_SHP)} | name field: {ncol} | length field: {lcol} | "
      f"features in area: {len(rc)} (>= {RIV_MIN_KM[2]} km: {(rc.L_km >= RIV_MIN_KM[2]).sum()})")
if ncol:
    print("named rivers >= 100 km in area:", sorted(rc[rc.L_km >= RIV_MIN_KM[2]].NAME.unique().tolist())[:25])
print(cg_note)
print(f"DWLR stations: {len(Pd)} | map coverage: {100 * np.isfinite(zq).mean():.0f} % of the 5-district area")
print("\nRecharge (mm, median (n)) by distance to river, per geology class:")
print(T10.to_string(index=False))
print("\nRead: compare WITHIN a class. Higher recharge only in the 0-2 km bin of large rivers -> possible river-stage")
print("(bank storage) rise, not rain recharge -> flag those stations. No trend -> rivers do not organise recharge here.")
print(f"Files in {FINAL}: Fig12_recharge_rivers.png, Fig12b_recharge_vs_river_distance.png, T10_*.csv")
