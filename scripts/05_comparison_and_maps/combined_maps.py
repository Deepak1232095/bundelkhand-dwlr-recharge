"""
WITHIN-DISTRICT RECHARGE MAPS: DWLR + CGWB (biannual) WELLS, WITH A VALIDATION TEST  ->  C:\\Users\\IITI\\paper3_final\\
====================================================================================================================
Why    : 70 DWLR stations leave large blank areas. CGWB has ~250 wells but only May / Nov readings.
How    : CGWB well -> biannual rise dH = depth(May) - depth(Nov), median of YEARS (same monsoons as DWLR).
         DWLR shows that the May->Nov rise is only a fraction k of the daily MRC-WTF rise (k = BIANN/GROSS_C,
         median per district from the DWLR stations; class median if a district has < 3 stations).
         CGWB recharge-equivalent (mm) = 1000 x Sy x max(dH, 0) / k     (Sy: hard rock 3 %, alluvium 6 %)
         CGWB geology class = class of the nearest DWLR station within CLASS_KM, else the district majority.
TEST   : before trusting the combined map, a CGWB-only IDW surface predicts every DWLR station (same district):
         skill = 1 - MAE(CGWB surface) / MAE(district median). skill > 0 -> the CGWB wells add real spatial
         information; skill <= 0 -> the combined map should NOT be used (the scaling does not hold there).
Maps   : IDW (p = 2), 1 km grid, cells > MAX_KM from any well masked; colour red (low) - yellow - blue (high).
Output : Fig10b_within_district_DWLR.png (DWLR only), Fig11_within_district_DWLR_CGWB.png (combined),
         T9_combined_validation.csv, T9b_cgwb_wells_recharge_equiv.csv
Run    :  %run combined_maps.py      (after monthly_recharge.py and recharge_maps.py)
"""
import os
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.environ.get("TEST_BASE", r"C:\Users\IITI")
FINAL = os.path.join(BASE, "paper3_final")
T6 = os.path.join(FINAL, "T6_monthly_station_year.csv")
T1 = os.path.join(FINAL, "T1_station_recharge.csv")
GPKG = os.path.join(FINAL, "districts_from_lithology.gpkg")
SIGN = os.path.join(BASE, "run_Q123", "DWLR_QC", "dwlr_sign_smoothness.csv")
CGWB = os.environ.get("TEST_CGWB", r"G:/Paper coleection/Paper 10/Cleaned_Data/UP_GW_wide_final.csv")
YEARS = [2023, 2024]
SY = {"HARD_ROCK": 0.03, "ALLUVIUM": 0.06}
CELL_M, POWER, EDGE_KM, MAX_KM, CLASS_KM = 1000.0, 2.0, 10.0, 15.0, 10.0
UTM = "EPSG:32644"
CMAP = plt.get_cmap("RdYlBu")          # red = low, yellow = middle, blue = high
for f in (T6, T1, GPKG, SIGN, CGWB):
    if not os.path.exists(f):
        sys.exit(f"[STOP] not found: {f}")

# ---------------- DWLR stations: Jun-Oct recharge (mm) and k = BIANN/GROSS_C
D = pd.read_csv(T6)
SM = D.groupby(["SID", "DIST", "CLASS", "MONTH"]).R_mm.median().reset_index()
S = SM.groupby(["SID", "DIST", "CLASS"]).agg(R=("R_mm", "sum"), n=("MONTH", "nunique")).reset_index()
S = S[S.n == 5].set_index("SID")
sg = pd.read_csv(SIGN).drop_duplicates("SID").set_index("SID")
S = S.join(sg[["lat", "lon"]], how="inner")
S["DIST"] = S.DIST.astype(str).str.upper()
t1 = pd.read_csv(T1).set_index("SID")
t1["DIST"] = t1.DIST.astype(str).str.upper()
t1["k"] = t1.BIANN / t1.GROSS_C
k_ok = t1[(t1.k > 0) & (t1.k <= 1.5)]
K_DIST = k_ok.groupby("DIST").k.agg(["median", "size"])
K_CLASS = k_ok.groupby("CLASS").k.median()
K_ALL = k_ok.k.median()

# ---------------- CGWB wells
w = pd.read_csv(CGWB)
w["DIST"] = w["DISTRICT"].astype(str).str.strip().str.upper()
w["LAT"] = pd.to_numeric(w["LATITUDE"], errors="coerce"); w["LON"] = pd.to_numeric(w["LONGITUDE"], errors="coerce")
w["WELL_ID"] = (w.DIST + "|" + w["BLOCK"].astype(str).str.strip() + "|" + w["HYDROGRAPH STATION NAME"].astype(str).str.strip()
                + "|" + w.LAT.round(4).astype(str) + "," + w.LON.round(4).astype(str))
w = w.dropna(subset=["LAT", "LON"]).drop_duplicates("WELL_ID")
w = w[w.DIST.isin(set(S.DIST))]
dH = []
for y in YEARS:
    k_ = f"{y % 100:02d}"
    if f"PRM_{k_}" in w.columns and f"PTM_{k_}" in w.columns:
        dH.append(pd.to_numeric(w[f"PRM_{k_}"], errors="coerce") - pd.to_numeric(w[f"PTM_{k_}"], errors="coerce"))
if not dH:
    sys.exit(f"[STOP] no PRM_/PTM_ columns for {YEARS} in the CGWB file")
w["dH"] = pd.concat(dH, axis=1).median(axis=1)
w["n_years"] = pd.concat(dH, axis=1).notna().sum(axis=1)
w = w[w.dH.notna()].copy()

# geology class for CGWB wells: nearest DWLR within CLASS_KM, else district majority
Pd = gpd.GeoDataFrame(S, geometry=gpd.points_from_xy(S.lon, S.lat), crs="EPSG:4326").to_crs(UTM)
Pc = gpd.GeoDataFrame(w.set_index("WELL_ID"), geometry=gpd.points_from_xy(w.LON, w.LAT), crs="EPSG:4326").to_crs(UTM)
nn = gpd.sjoin_nearest(Pc[["geometry"]], Pd[["CLASS", "geometry"]], how="left", distance_col="dist_m")
nn = nn[~nn.index.duplicated()]
maj = S.groupby("DIST").CLASS.agg(lambda v: v.value_counts().index[0])
Pc["CLASS"] = np.where(nn.dist_m.reindex(Pc.index) <= CLASS_KM * 1000, nn.CLASS.reindex(Pc.index), Pc.DIST.map(maj))
Pc["k"] = Pc.DIST.map(lambda d: K_DIST.loc[d, "median"] if d in K_DIST.index and K_DIST.loc[d, "size"] >= 3 else np.nan)
Pc["k"] = Pc.k.fillna(Pc.CLASS.map(K_CLASS)).fillna(K_ALL)
Pc["R"] = 1000 * Pc.CLASS.map(SY) * Pc.dH.clip(lower=0) / Pc.k
Pc["negative_dH"] = Pc.dH < 0
Pc.drop(columns="geometry").to_csv(os.path.join(FINAL, "T9b_cgwb_wells_recharge_equiv.csv"))
Pd["SRC"], Pc["SRC"] = "DWLR", "CGWB"
for P in (Pd, Pc):
    P["x"], P["y"] = P.geometry.x, P.geometry.y
G = gpd.read_file(GPKG).to_crs(UTM)
G["DIST"] = G.DIST.astype(str).str.upper()


def idw(xs, ys, vs, xq, yq):
    d = np.maximum(np.hypot(xq[:, None] - xs[None, :], yq[:, None] - ys[None, :]), 1.0)
    w_ = 1.0 / d ** POWER
    return (w_ * vs[None, :]).sum(1) / w_.sum(1), d.min(1)


dists = [d for d in sorted(G.DIST) if (Pd.DIST == d).sum() >= 3]
allR = np.r_[Pd.R.values, Pc.R.values]
vmin, vmax = np.nanpercentile(allR, [5, 95])
rows = []


def panel(ax, d, pts, title_extra):
    poly = G[G.DIST == d].geometry.union_all()
    use = pts[(pts.geometry.distance(poly) <= EDGE_KM * 1000) | (pts.DIST == d)]
    x0, y0, x1, y1 = poly.bounds
    gx = np.arange(x0, x1 + CELL_M, CELL_M); gy = np.arange(y0, y1 + CELL_M, CELL_M)
    XX, YY = np.meshgrid(gx, gy)
    inside = gpd.GeoSeries(gpd.points_from_xy(XX.ravel(), YY.ravel()), crs=UTM).within(poly).values
    Z = np.full(XX.size, np.nan)
    zq, dmin = idw(use.x.values, use.y.values, use.R.values, XX.ravel()[inside], YY.ravel()[inside])
    zq[dmin > MAX_KM * 1000] = np.nan
    Z[inside] = zq
    gpd.GeoSeries([poly], crs=UTM).plot(ax=ax, facecolor="#ececea", edgecolor="none", hatch="////", zorder=0)
    ax.pcolormesh(XX, YY, Z.reshape(XX.shape), cmap=CMAP, vmin=vmin, vmax=vmax, shading="auto", zorder=1)
    gpd.GeoSeries([poly], crs=UTM).boundary.plot(ax=ax, color="#3d3d3a", linewidth=0.8, zorder=2)
    own = use[use.DIST == d]
    for src, mk, sz in (("CGWB", "s", 16), ("DWLR", "o", 36)):
        q = own[own.SRC == src]
        if len(q):
            ax.scatter(q.x, q.y, c=q.R, cmap=CMAP, vmin=vmin, vmax=vmax, marker=mk, s=sz, edgecolors="#1f1f1e",
                       linewidths=0.6, zorder=3 if src == "CGWB" else 4)
    ax.set_title(f"{d.title()}  {title_extra}  covered {100 * np.isfinite(zq).mean():.0f} %", fontsize=7.5, loc="left")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(""); ax.set_ylabel("")
    for s_ in ax.spines.values():
        s_.set_visible(False)
    return np.isfinite(zq).mean()


# ---------------- validation: CGWB-only surface -> DWLR stations
for d in dists:
    own_d = Pd[Pd.DIST == d]
    poly = G[G.DIST == d].geometry.union_all()
    cg = Pc[(Pc.geometry.distance(poly) <= EDGE_KM * 1000) | (Pc.DIST == d)]
    e_c, e_m = [], []
    if len(cg) >= 3:
        pred, _ = idw(cg.x.values, cg.y.values, cg.R.values, own_d.x.values, own_d.y.values)
        e_c = np.abs(pred - own_d.R.values)
        e_m = [abs(own_d.drop(index=i).R.median() - r) for i, r in own_d.R.items()]
    mc, mm = (np.mean(e_c), np.mean(e_m)) if len(e_c) else (np.nan, np.nan)
    rows.append({"district": d, "n_DWLR": len(own_d), "n_CGWB": int((Pc.DIST == d).sum()),
                 "k_used": round(float(cg.k.median()) if len(cg) else np.nan, 2),
                 "DWLR_median_mm": round(own_d.R.median(), 0), "CGWB_equiv_median_mm": round(cg[cg.DIST == d].R.median(), 0) if len(cg) else np.nan,
                 "MAE_CGWB_surface": round(mc, 0), "MAE_district_median": round(mm, 0),
                 "skill_CGWB_predicts_DWLR": round(1 - mc / mm, 2) if mm > 0 else np.nan,
                 "CGWB_negative_dH_share": round(float(Pc[Pc.DIST == d].negative_dH.mean()), 2) if (Pc.DIST == d).any() else np.nan})
T9 = pd.DataFrame(rows)
T9["use_combined_map"] = np.where(T9.skill_CGWB_predicts_DWLR > 0, "yes", "NO - show DWLR only")
T9.to_csv(os.path.join(FINAL, "T9_combined_validation.csv"), index=False)

# ---------------- figures
for name, pts, extra in (("Fig10b_within_district_DWLR.png", Pd, "DWLR only"),
                         ("Fig11_within_district_DWLR_CGWB.png", pd.concat([Pd, Pc]), "DWLR + CGWB")):
    pts = gpd.GeoDataFrame(pts, geometry="geometry", crs=UTM)
    n = len(dists); ncol = 3; nrow = int(np.ceil((n + 1) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(4.3 * ncol, 4.1 * nrow)); axs = axs.ravel()
    for ax, d in zip(axs, dists):
        sk = T9.set_index("district").skill_CGWB_predicts_DWLR.get(d, np.nan)
        lab = f"(skill CGWB->DWLR {sk:+.2f})" if "CGWB" in extra else ""
        panel(ax, d, pts, lab)
    for ax in axs[n:]:
        ax.axis("off")
    cax = axs[n]
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(vmin, vmax))
    cb = fig.colorbar(sm, ax=cax, location="left", shrink=0.7); cb.set_label("Jun-Oct recharge (mm)"); cb.outline.set_visible(False)
    cax.text(0.15, 0.3, ("o DWLR station (MRC-WTF)\n" + ("s CGWB well (biannual rise / k)\n" if "CGWB" in extra else "")
                         + f"hatched = > {MAX_KM:g} km from a well\nIDW (p = 2), 1 km grid\nSy: hard rock 3 %, alluvium 6 %"),
             transform=cax.transAxes, fontsize=8, va="top")
    fig.suptitle(f"Within-district variation of Jun-Oct recharge, 2023-2024 ({extra})", x=0.02, ha="left", fontsize=10)
    fig.savefig(os.path.join(FINAL, name), dpi=300, bbox_inches="tight"); plt.close(fig)

pd.set_option("display.width", 220)
print("=" * 100)
print("VERDICT  combined_maps")
print("=" * 100)
print(f"CGWB wells used {len(Pc)} (years {YEARS}) | k = BIANN/GROSS_C by district: {K_DIST['median'].round(2).to_dict()} | all {K_ALL:.2f}")
print(T9.to_string(index=False))
print("\nRead: skill > 0 -> CGWB wells improve the map in that district; skill <= 0 -> use the DWLR-only map there.")
print(f"Files: Fig10b_within_district_DWLR.png, Fig11_within_district_DWLR_CGWB.png, T9_combined_validation.csv, T9b_cgwb_wells_recharge_equiv.csv in {FINAL}")
