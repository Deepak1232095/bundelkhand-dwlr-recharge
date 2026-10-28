"""
FIGURE 2 (FINAL, design D): DWLR station map + district station counts by geology
================================================================================
(a) Map: official district boundaries (GADM file), rivers, DWLR stations used (shape + colour = geology),
    core stations not used (hollow), lat/lon ticks, scale bar, north arrow, India locator inset.
(b) Bars: stations per district (used, by geology; core but not used, hatched) with used/core labels.
Nothing is estimated here. Every point, polygon, line and count comes from your files.
Inputs:
  DIST_SHP   E:\\up_bundel\\up_bundel_boxes\\up_5dist.shp  (NAME_2 = district name)
  T1         paper3_final\\T1_station_recharge.csv  (or run_Q123\\DWLR_QC\\final\\)  SID, DIST, CLASS, lat, lon
  SIGN       run_Q123\\DWLR_QC\\dwlr_sign_smoothness.csv  (core stations)
  RIVER_SHP  E:\\river_network_shape\\River_Network.shp
  INDIA_SHP  optional India outline. If empty, the script searches C:\\Users\\IITI, E:\\ and Downloads (3 levels).
             Not found -> figure is made WITHOUT the inset (the VERDICT says so). Use the Survey of India
             boundary if the journal or your institute requires the official depiction of India.
Output (paper3_final\\fig2_final\\): Fig2_station_map.png (600 dpi), Fig2_station_map.pdf (vector),
        Fig2_counts.csv, Fig2_caption_draft.txt
Run:  %run fig2_final.py        Requires: geopandas matplotlib pandas numpy.   
"""
import os
import sys
import glob
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle, Polygon as MPoly
from matplotlib.ticker import FuncFormatter, MultipleLocator

warnings.filterwarnings("ignore")

# ======================================================================================== CONFIG
BASE = os.environ.get("TEST_BASE", r"C:\Users\IITI")
FINAL = os.path.join(BASE, "paper3_final")
OUT = os.path.join(FINAL, "fig2_final")
DIST_SHP = os.environ.get("TEST_DIST", r"E:\up_bundel\up_bundel_boxes\up_5dist.shp")
DIST_COL = "NAME_2"
T1_CANDS = [os.path.join(FINAL, "T1_station_recharge.csv"),
            os.path.join(BASE, "run_Q123", "DWLR_QC", "final", "T1_station_recharge.csv")]
SIGN = os.path.join(BASE, "run_Q123", "DWLR_QC", "dwlr_sign_smoothness.csv")
RIVER_SHP = os.environ.get("TEST_RIVER", r"E:\river_network_shape\River_Network.shp")
RIVER_NAME_COL = ""                                   # leave "" to auto-detect; set e.g. "RIVNAME" if labels are missing
SHOW_INDIA_INSET = False                              # True = add the India locator inset
INDIA_SHP = os.environ.get("TEST_INDIA", "")
INDIA_SEARCH = [BASE, "E:\\", os.path.join(os.path.expanduser("~"), "Downloads")]
CORE_OK = ["single sign (fine)", "sign errors -> |level| fixes it"]
BBOX = (77.0, 23.5, 81.0, 27.0)                       # keeps UP Bundelkhand only (drops Himachal Hamirpur if present)
RIV_MAJOR_KM, RIV_MINOR_KM = 100, 30
RIV_LABEL = ("Betwa", "Yamuna", "Ken", "Dhasan", "Pahuj", "Jamni", "Sajnam", "Birma", "Chandrawal", "Sindh")
SCALE_KM = 50
RIVER_BAND_DEG = 0.04                                 # rivers drawn up to ~4 km outside the districts
FIG_W, FIG_H = 7.2, 4.6                               # inches (WRR two-column width ~ 7.5 in)

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                     "font.size": 8, "axes.titlesize": 8, "legend.fontsize": 7, "legend.frameon": False,
                     "axes.grid": False, "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42,
                     "savefig.bbox": "tight"})
INK, MUTED, LAND, EDGE, RIV, GRID = "#1f1f1e", "#6b6a64", "#f3f2ee", "#3d3d3a", "#3f6fa8", "#e3e2dd"
GEO = {"ALLUVIUM": dict(marker="o", color="#d9a425", label="Alluvium"),
       "HARD_ROCK": dict(marker="^", color="#a8435a", label="Hard rock"),
       "NA": dict(marker="D", color="#9a9a94", label="Unclassified")}
HALO = [pe.withStroke(linewidth=2.2, foreground="white")]

# ======================================================================================== DATA
T1P = next((p for p in T1_CANDS if os.path.exists(p)), None)
for f, lab in ((DIST_SHP, DIST_SHP), (T1P, "T1_station_recharge.csv"), (SIGN, SIGN), (RIVER_SHP, RIVER_SHP)):
    if not f or not os.path.exists(f):
        sys.exit(f"[STOP] not found: {lab}")
os.makedirs(OUT, exist_ok=True)

G = gpd.read_file(DIST_SHP)
G = (G.set_crs("EPSG:4326") if G.crs is None else G).to_crs("EPSG:4326")
if DIST_COL not in G.columns:
    sys.exit(f"[STOP] column {DIST_COL} not in {list(G.columns)}")
G = G[G.geometry.representative_point().within(box(*BBOX))].copy()
G["DIST"] = G[DIST_COL].astype(str).str.upper().str.strip()
G["geometry"] = G.geometry.buffer(0)
G = G.dissolve(by="DIST", as_index=False).sort_values("DIST").reset_index(drop=True)
REG = G.geometry.union_all()
LAT0 = (REG.bounds[1] + REG.bounds[3]) / 2
ASPECT = 1 / np.cos(np.deg2rad(LAT0))                 # equal distances in x and y on a lat/lon plot
KM_PER_DEG_X = 111.32 * np.cos(np.deg2rad(LAT0))

T1 = pd.read_csv(T1P)
if "SID" not in T1.columns:
    T1 = T1.rename(columns={T1.columns[0]: "SID"})
sg = pd.read_csv(SIGN).drop_duplicates("SID")
if not {"lat", "lon"} <= set(T1.columns):
    T1 = T1.merge(sg[["SID", "lat", "lon"]], on="SID", how="left")
T1 = T1.dropna(subset=["lat", "lon"]).copy()
T1["DIST"] = T1.DIST.astype(str).str.upper().str.strip()
T1["CLASS"] = T1.CLASS.where(T1.CLASS.isin(["ALLUVIUM", "HARD_ROCK"]), "NA")
core = sg[sg.sign_verdict.isin(CORE_OK)].copy()
if "DIST" not in core.columns:
    core["DIST"] = core.SID.astype(str).str.split("|").str[0]
core["DIST"] = core.DIST.astype(str).str.upper().str.strip()
NU = core[~core.SID.isin(T1.SID)].dropna(subset=["lat", "lon"]).copy()

S = gpd.GeoDataFrame(T1, geometry=gpd.points_from_xy(T1.lon, T1.lat), crs="EPSG:4326")
N = gpd.GeoDataFrame(NU, geometry=gpd.points_from_xy(NU.lon, NU.lat), crs="EPSG:4326")
# QC: district attribute of each station vs the polygon it falls in
J = gpd.sjoin(pd.concat([S.assign(SET="used"), N.assign(SET="not used")])[["SID", "DIST", "SET", "geometry"]],
              G[["DIST", "geometry"]].rename(columns={"DIST": "POLY"}), how="left", predicate="within")
J = J.drop_duplicates("SID")
mism = J[J.POLY.notna() & (J.POLY != J.DIST)]
outside = J[J.POLY.isna()]

# rivers
head = gpd.read_file(RIVER_SHP, rows=1)
crs_r = head.crs if head.crs is not None else "EPSG:4326"
win = gpd.GeoSeries([REG.buffer(0.3)], crs="EPSG:4326").to_crs(crs_r).total_bounds
riv = gpd.read_file(RIVER_SHP, bbox=tuple(win))
riv = (riv.set_crs("EPSG:4326") if riv.crs is None else riv)
riv = riv[riv.geometry.notna() & ~riv.geometry.is_empty & riv.geom_type.isin(["LineString", "MultiLineString"])].copy()
riv["L_km"] = riv.to_crs("EPSG:32644").length / 1000
ncol = RIVER_NAME_COL or next((c for c in riv.columns if any(k in c.lower() for k in
                                                           ("rivname", "riv_name", "river_name", "name", "rname"))), None)
riv["NAME"] = riv[ncol].astype(str).str.strip().str.title() if ncol else ""
if ncol:
    named = ~riv.NAME.isin(["", "Nan", "None", "0"])
    riv.loc[named, "L_km"] = np.maximum(riv.loc[named, "L_km"], riv[named].groupby("NAME").L_km.transform("sum"))
riv = riv.to_crs("EPSG:4326")
RCLIP = REG.buffer(RIVER_BAND_DEG)                     # boundary rivers (Yamuna) drawn in full width
rc = riv[riv.intersects(RCLIP)].copy()
rc["geometry"] = rc.geometry.intersection(RCLIP)
rc = rc[~rc.geometry.is_empty & (rc.L_km >= RIV_MINOR_KM)]

# India outline (optional)
india, india_src = None, "switched off (SHOW_INDIA_INSET = False)" if not SHOW_INDIA_INSET else "not found -> no inset"
cands = [INDIA_SHP] if (SHOW_INDIA_INSET and INDIA_SHP) else []
if SHOW_INDIA_INSET and not INDIA_SHP:
    for root in INDIA_SEARCH:
        for pat in ("*india*.shp", "*India*.shp", "*INDIA*.shp", "*IND_0*.shp"):
            for d in ("", "*", os.path.join("*", "*"), os.path.join("*", "*", "*")):
                cands += glob.glob(os.path.join(root, d, pat))
for c in dict.fromkeys(cands):
    try:
        x = gpd.read_file(c)
        x = (x.set_crs("EPSG:4326") if x.crs is None else x).to_crs("EPSG:4326")
        b = x.total_bounds
        if b[0] < 70 and b[2] > 95 and b[1] < 10 and b[3] > 33 and x.geom_type.str.contains("Polygon").all():
            india, india_src = x, c
            break
    except Exception:
        continue

# ======================================================================================== FIGURE
fig = plt.figure(figsize=(FIG_W, FIG_H))
gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1], wspace=0.32)
ax = fig.add_subplot(gs[0, 0])

# --- (a) map
G.plot(ax=ax, facecolor=LAND, edgecolor="none", zorder=0)
if len(rc):
    rc[rc.L_km < RIV_MAJOR_KM].plot(ax=ax, color=RIV, linewidth=0.4, alpha=0.75, zorder=1)
    rc[rc.L_km >= RIV_MAJOR_KM].plot(ax=ax, color=RIV, linewidth=1.2, zorder=1)
G.boundary.plot(ax=ax, color=EDGE, linewidth=0.7, zorder=2)
if len(N):
    ax.scatter(N.lon, N.lat, s=11, marker="o", facecolors="none", edgecolors=MUTED, linewidths=0.6, zorder=4)
for c, q in S.groupby("CLASS"):
    ax.scatter(q.lon, q.lat, s=24, marker=GEO[c]["marker"], color=GEO[c]["color"], edgecolors=INK,
               linewidths=0.45, zorder=5)
DLAB = []
for _, r in G.iterrows():
    p = r.geometry.representative_point(); DLAB.append(p)
    ax.annotate(r.DIST.title(), (p.x, p.y), fontsize=7, fontweight="bold", color=INK, ha="center", va="center",
                zorder=6, path_effects=HALO)
labelled = []
if ncol and len(rc):
    big = rc[rc.NAME.str.contains("|".join(RIV_LABEL), case=False, na=False) & (rc.L_km >= RIV_MAJOR_KM)]
    for nm, q in big.groupby("NAME"):
        g_ = max(q.geometry, key=lambda g: g.length)
        g_ = max(g_.geoms, key=lambda g: g.length) if g_.geom_type == "MultiLineString" else g_
        pts = [g_.interpolate(f, normalized=True) for f in (0.5, 0.35, 0.65, 0.2, 0.8)]
        dl = [min(np.hypot(p.x - c.x, p.y - c.y) for c in DLAB) for p in pts]
        pt = pts[int(np.argmax(np.array(dl) > 0.12))] if max(dl) > 0.12 else pts[int(np.argmax(dl))]
        ax.annotate(nm, (pt.x, pt.y), fontsize=6.3, style="italic", color=RIV, ha="center", va="bottom", zorder=6,
                    path_effects=HALO)
        labelled.append(nm)
x0, y0, x1, y1 = REG.bounds
px, py = 0.05 * (x1 - x0), 0.05 * (y1 - y0)
ax.set_xlim(x0 - px, x1 + px); ax.set_ylim(y0 - 2.2 * py, y1 + 1.8 * py); ax.set_aspect(ASPECT)
ax.xaxis.set_major_locator(MultipleLocator(0.5)); ax.yaxis.set_major_locator(MultipleLocator(0.5))
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}°E"))
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}°N"))
ax.tick_params(length=2.5, color=MUTED, labelcolor=INK)
plt.setp(ax.get_yticklabels(), rotation=90, va="center")
ax.set_xlabel(""); ax.set_ylabel("")
for s_ in ax.spines.values():
    s_.set_visible(True); s_.set_color(MUTED); s_.set_linewidth(0.6)

# scale bar (bottom left) and north arrow (top right), in data coordinates
X0, X1 = ax.get_xlim(); Y0, Y1 = ax.get_ylim()
dx = SCALE_KM / KM_PER_DEG_X
bx_, by_ = X1 - 0.06 * (X1 - X0) - dx, Y0 + 0.035 * (Y1 - Y0)
hh = 0.012 * (Y1 - Y0)
ax.add_patch(Rectangle((bx_, by_), dx / 2, hh, fc=INK, ec=INK, lw=0.5, zorder=7))
ax.add_patch(Rectangle((bx_ + dx / 2, by_), dx / 2, hh, fc="white", ec=INK, lw=0.5, zorder=7))
for xx, t in ((bx_, "0"), (bx_ + dx, f"{SCALE_KM} km")):
    ax.text(xx, by_ + 1.8 * hh, t, fontsize=6.3, ha="center", va="bottom", color=INK, zorder=7, path_effects=HALO)
cx, cy = X0 + 0.93 * (X1 - X0), Y0 + 0.88 * (Y1 - Y0)
w, h = 0.018 * (X1 - X0), 0.06 * (Y1 - Y0)
ax.add_patch(MPoly([[cx - w, cy - h / 2], [cx, cy + h / 2], [cx + w, cy - h / 2], [cx, cy - h / 4]], closed=True,
                   fc=INK, ec=INK, lw=0.5, zorder=7))
ax.text(cx, cy + h / 2 + 0.008 * (Y1 - Y0), "N", ha="center", va="bottom", fontsize=7.5, fontweight="bold", zorder=7)

if SHOW_INDIA_INSET and india is not None:              # locator inset (switched off by default)
    ia = ax.inset_axes((0.70, 0.02, 0.29, 0.31))
    india.plot(ax=ia, facecolor="#e2e1dc", edgecolor=MUTED, linewidth=0.25)
    G.plot(ax=ia, facecolor="#a8435a", edgecolor="none")
    ia.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc="none", ec=INK, lw=0.6))
    ia.set_xlim(67.5, 97.8); ia.set_ylim(6.5, 37.3); ia.set_aspect(1 / np.cos(np.deg2rad(22)))
    ia.set_xticks([]); ia.set_yticks([]); ia.set_xlabel(""); ia.set_ylabel("")
    ia.set_facecolor("white")
    for s_ in ia.spines.values():
        s_.set_color(MUTED); s_.set_linewidth(0.5)
    ia.text(0.04, 0.95, "India", transform=ia.transAxes, fontsize=6, va="top", color=MUTED)

hand = [Line2D([], [], ls="", marker=GEO[c]["marker"], ms=5, mfc=GEO[c]["color"], mec=INK, mew=0.45,
               label=f"{GEO[c]['label']} (n = {int((S.CLASS == c).sum())})")
        for c in ("ALLUVIUM", "HARD_ROCK", "NA") if (S.CLASS == c).any()]
hand += [Line2D([], [], ls="", marker="o", ms=4, mfc="none", mec=MUTED, mew=0.6,
                label=f"Core station, not used (n = {len(N)})"),
         Line2D([], [], color=RIV, lw=1.2, label=f"River ≥ {RIV_MAJOR_KM} km"),
         Line2D([], [], color=RIV, lw=0.4, label=f"River {RIV_MINOR_KM}–{RIV_MAJOR_KM} km")]
ax.legend(handles=hand, loc="upper left", bbox_to_anchor=(0, -0.07), ncol=2, handletextpad=0.3, columnspacing=1.0)
ax.text(-0.13, 1.02, "(a)", transform=ax.transAxes, fontsize=9, fontweight="bold", ha="right", va="bottom")

# --- (b) bars
bx = fig.add_subplot(gs[0, 1])
dists = sorted(G.DIST)
cnt = pd.DataFrame({"alluvium": [int(((S.DIST == d) & (S.CLASS == "ALLUVIUM")).sum()) for d in dists],
                    "hard_rock": [int(((S.DIST == d) & (S.CLASS == "HARD_ROCK")).sum()) for d in dists],
                    "unclassified": [int(((S.DIST == d) & (S.CLASS == "NA")).sum()) for d in dists],
                    "core_not_used": [int((N.DIST == d).sum()) for d in dists]}, index=[d.title() for d in dists])
cnt["used"] = cnt.alluvium + cnt.hard_rock + cnt.unclassified
cnt["core"] = cnt.used + cnt.core_not_used
yy = np.arange(len(cnt))[::-1]
left = np.zeros(len(cnt))
for col, key in (("alluvium", "ALLUVIUM"), ("hard_rock", "HARD_ROCK"), ("unclassified", "NA")):
    if cnt[col].sum():
        bx.barh(yy, cnt[col], left=left, color=GEO[key]["color"], edgecolor=INK, linewidth=0.4, height=0.6, zorder=3)
        left += cnt[col].values
bx.barh(yy, cnt.core_not_used, left=left, color="white", edgecolor=MUTED, hatch="////", linewidth=0.4, height=0.6,
        zorder=3)
for y_, u, t in zip(yy, cnt.used, cnt.core):
    bx.text(t + 0.5, y_, f"{u}/{t}", va="center", fontsize=6.5, color=INK)
bx.set_yticks(yy); bx.set_yticklabels(cnt.index)
bx.set_xlim(0, cnt.core.max() * 1.22 + 1)
bx.set_xlabel("DWLR stations (used / core)")
bx.set_axisbelow(True); bx.grid(axis="x", color=GRID, linewidth=0.6, zorder=0); bx.grid(axis="y", visible=False)
for s_ in ("top", "right"):
    bx.spines[s_].set_visible(False)
for s_ in ("left", "bottom"):
    bx.spines[s_].set_color(MUTED); bx.spines[s_].set_linewidth(0.6)
bx.tick_params(axis="y", length=0); bx.tick_params(axis="x", length=2.5, color=MUTED)
lh = [Patch(fc=GEO["ALLUVIUM"]["color"], ec=INK, lw=0.4, label="Alluvium"),
      Patch(fc=GEO["HARD_ROCK"]["color"], ec=INK, lw=0.4, label="Hard rock")]
if cnt.unclassified.sum():
    lh.append(Patch(fc=GEO["NA"]["color"], ec=INK, lw=0.4, label="Unclassified"))
lh.append(Patch(fc="white", ec=MUTED, hatch="////", lw=0.4, label="Core, not used"))
bx.legend(handles=lh, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, handlelength=1.2, columnspacing=0.8)
fig.canvas.draw()                                       # bar panel height = map height
pm, pb = ax.get_position(), bx.get_position()
bx.set_position([pb.x0, pm.y0 + 0.06 * pm.height, pb.width, 0.80 * pm.height])
bx.text(-0.30, 1.0 + 0.25 * (pm.height / (0.80 * pm.height) - 1) + 0.13, "(b)", transform=bx.transAxes, fontsize=9,
        fontweight="bold", ha="left", va="bottom")

fig.savefig(os.path.join(OUT, "Fig2_station_map.png"), dpi=600)
fig.savefig(os.path.join(OUT, "Fig2_station_map.pdf"))
plt.close(fig)
cnt.to_csv(os.path.join(OUT, "Fig2_counts.csv"))

# caption draft with the real numbers
nA, nH, nNA = (int((S.CLASS == c).sum()) for c in ("ALLUVIUM", "HARD_ROCK", "NA"))
cap = (f"Figure 2. DWLR stations in the five Bundelkhand districts of Uttar Pradesh. (a) Stations used in the recharge "
       f"analysis (n = {len(S)}; {nA} alluvium, {nH} hard rock" + (f", {nNA} unclassified" if nNA else "") +
       f"; geology from GSI 1:50,000 lithology) and core stations not used (n = {len(N)}), with rivers "
       f"≥ {RIV_MINOR_KM} km long. (b) Stations per district by geology; hatched = core stations not used "
       f"[REASON: e.g. fewer than 30 dry days, recession slope ≥ 0, or < 80% June–October coverage]. "
       f"District boundaries: GADM; rivers: [SOURCE]." + (" Inset: location in India." if SHOW_INDIA_INSET and india is not None else ""))
open(os.path.join(OUT, "Fig2_caption_draft.txt"), "w", encoding="utf-8").write(cap + "\n")

# ======================================================================================== VERDICT
pd.set_option("display.width", 200)
print("=" * 90)
print("VERDICT  fig2_final")
print("=" * 90)
print(f"districts: {list(cnt.index)} | stations used {len(S)} {S.CLASS.value_counts().to_dict()} | core not used {len(N)}")
print(f"stations outside every district polygon: {len(outside)} {outside.SID.tolist()[:10]}")
print(f"stations whose DIST attribute differs from the polygon they fall in: {len(mism)}")
if len(mism):
    print(mism[["SID", "SET", "DIST", "POLY"]].head(15).to_string(index=False))
print(f"rivers drawn {len(rc)} (>= {RIV_MAJOR_KM} km: {(rc.L_km >= RIV_MAJOR_KM).sum()}) | name column: {ncol} | "
      f"labelled: {labelled or 'none'}")
if not labelled:
    print(f"  river columns: {[c for c in riv.columns if c != 'geometry']}")
print(f"India inset: {india_src}")
print("\ncounts (bars):"); print(cnt.to_string())
print(f"\nFiles: {OUT} -> Fig2_station_map.png (600 dpi), .pdf, Fig2_counts.csv, Fig2_caption_draft.txt")
