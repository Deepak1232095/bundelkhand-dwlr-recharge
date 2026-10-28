#!/usr/bin/env python
"""
regime_geology.py  --  v3 (audited rewrite)
==========================================
RESEARCH QUESTION (unchanged):
  Does geology explain buffered vs decoupled groundwater hydrographs AFTER accounting for district differences?

INPUTS (unchanged)
  * station_metrics_and_controls.csv written by regime_controls.py
  * GSI 1:50k lithology shapefiles (LITHOLOGY_DIST_*.shp), field 'group_name' (+ 'lithologic' for auditing)

METRICS (taken AS-IS from regime_controls.py -- this script never recomputes or redefines them)
  SHIFT     |peak DOY 2024 - peak DOY 2023|          (S2_current rows of schedule_experiment.csv)
  EXCESS    SHIFT - RAIN_SHIFT, RAIN_SHIFT = |W50_2024 - W50_2023| of station IMD rain
  NSE_VAL   rain-only Pastas validation NSE          (station_models.csv)
  CAPTURE   mean 20-May/10-Nov capture of the seasonal rise (2 yrs)
  PEAK_DOY  mean peak day-of-year ; RISE mean seasonal rise (m)
  Known upstream limitation (documented, NOT changed): SHIFT/EXCESS rest on 2 monsoons only.
  Validation: if RAIN_SHIFT is present, EXCESS == SHIFT - RAIN_SHIFT is checked.

GEOLOGY ASSIGNMENT
  * predicate 'intersects' (a point on a polygon edge is not lost, unlike 'within')
  * every station gets an audit row: n polygons matched, groups/classes/files matched, resolution rule
      SINGLE / MULTI_SAME_GROUP                -> accepted
      MULTI_CONFLICT_OWN_DISTRICT_FILE         -> conflicting overlap resolved by the station's own district map
      MULTI_CONFLICT_SAME_CLASS                -> groups differ but class identical -> class accepted
      AMBIGUOUS_EXCLUDED                       -> conflicting classes, no rule resolves -> excluded
      NEAREST_WITHIN_TOL                       -> in a gap, nearest polygon within NEAREST_TOL_M -> accepted, flagged
      UNMATCHED_EXCLUDED
  * stations closer than BOUNDARY_UNCERT_M to a polygon of another class are flagged (sensitivity run without them)
  * group_name -> class by an EXPLICIT table (no substring guessing); unknown groups -> UNMAPPED (excluded, warned).
    Missing group_name is NOT filled from 'lithologic' (v1/v2 did this silently).

CLASS SCHEMES
  BROAD  (primary): HARD_ROCK = Bundelkhand Granitoid Complex + Older Supracrustal + Older Metamorphics
  NARROW (v1/v2)  : HARD_ROCK = Bundelkhand Granitoid Complex + Older Supracrustal
  Justification per unit is written to geology_class_mapping.csv; units flagged VERIFY must be checked
  against the 'lithologic' composition printed by this script / the GSI legend.

INFERENCE DESIGN  (test families fixed before looking; BH-FDR inside each confirmatory family only)
  F1 (confirmatory) ALLUVIUM vs HARD_ROCK WITHIN district.
     Only districts with >= MIN_CELL stations of BOTH classes enter (others carry no within-district information).
     Estimand  r_strat = sum_d (2U_d - nA_d nH_d) / sum_d nA_d nH_d
               = P(alluvium > hard rock) - P(alluvium < hard rock) for a random within-district pair.
     Test      the SAME statistic recomputed after permuting class labels within each district
               (district group sizes preserved exactly).  Van Elteren z-test reported as a check.
     CI        stratified bootstrap (resample within district x class cells).
     Family    the metrics (6), BH-FDR; NaN p-values are excluded from the family, never set to 1.
  F2 (secondary) distance to the alluvium/hard-rock CONTACT, WITHIN class and WITHIN district.
     Statistic district-adjusted rank correlation = Pearson of within-district centred percentile ranks;
               null = x ranks permuted within district (denominator invariant -> obs and null identical in form).
     Family    2 classes x 6 metrics, BH-FDR.
  Descriptive only (NO inferential claim, no FDR): pooled Mann-Whitney (confounded with district),
     pooled Spearman, per-district Mann-Whitney (exploratory), NEWER vs OLDER alluvium.
  Sensitivity (no FDR): cluster-level F1 (spatial pseudo-replication), F1 without uncertain assignments,
     F1 under the alternative class scheme, F2 under 6 contact definitions.

SPATIAL DEPENDENCE
  Station-level permutation assumes stations are exchangeable within district. Neighbouring wells share local
  conditions, so this can be anti-conservative. The script (i) measures it: Moran's I of within-district
  rank residuals, permutation test within district; (ii) repeats F1 at the level of spatial clusters
  (stations of the same district & class linked within CLUSTER_KM, cluster median). If a station-level
  result does not survive at cluster level, treat it as pseudo-replication.

CONTACT  (SDIST_KM sign convention: + inside ALLUVIUM, - inside HARD_ROCK, NaN for OTHER/unresolved)
  contact = boundary(HARD_ROCK union) INTERSECT buffer(ALLUVIUM union, tol) -> only shared edges count.
  Diagnostics: invalid geometries repaired (make_valid, polygon parts kept, area change reported), overlap
  area of hard rock and alluvium (conflicting maps), share of contact lying on map-file edges (possible
  edge-matching artefact between district sheets). Variants: BASE (tol 50 m), TOL_5, TOL_200, NO_FILE_EDGE,
  MIN_AREA (drop polygons < 0.05 km2), ALT_SCHEME.

Run:  Jupyter -> set GEO_DIR in CFG below, then run the cell / %run regime_geology.py
      terminal -> python regime_geology.py --geo-dir "E:\\...\\geology"   (all options have defaults)

Requires: numpy pandas scipy geopandas shapely
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import platform
import sys
import zlib
from datetime import datetime

import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
from shapely.ops import unary_union
from scipy import stats
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

# =====================================================================================================
# CONFIG  (defaults; every entry can be overridden on the command line)
# =====================================================================================================
CFG = dict(
    GEO_DIR=r"E:\up_bundel\Lithology",
    GEO_GLOB="*.shp",
    IN_CSV=r"C:\Users\IITI\DWLR_QC\regime_controls\station_metrics_and_controls.csv",
    OUT=r"C:\Users\IITI\DWLR_QC\regime_controls\geology_v3",
    STUDY_BBOX=(77.0, 23.5, 81.5, 27.0),   # lon/lat; polygons whose interior point lies outside are dropped
    UTM="EPSG:32644",            # UTM 44N; study area 77.75-80.75E -> scale error < 0.2 % at the western edge
    SCHEME="BROAD",              # primary hard-rock definition: BROAD or NARROW
    SEED=20261013,
    N_PERM=9999,
    N_BOOT=2000,
    N_PERM_MORAN=999,
    MIN_CELL=3,                  # F1: min stations of EACH class in a district for it to enter
    MIN_CELL_CLUSTER=2,          # F1 cluster-level: min clusters of each class per district
    MIN_STRAT_N=5,               # F2: min stations of the class in a district for it to enter
    NEAREST_TOL_M=100.0,
    BOUNDARY_UNCERT_M=100.0,
    CONTACT_TOL_M=50.0,
    CONTACT_TOL_SENS=(5.0, 200.0),
    EDGE_TOL_M=250.0,
    MIN_AREA_KM2_SENS=0.05,
    CLUSTER_KM=5.0,
    MORAN_KM=10.0,
    ALPHA=0.05,
)
METRICS = ["SHIFT", "EXCESS", "NSE_VAL", "CAPTURE", "PEAK_DOY", "RISE"]
REQ_COLS = ["SID", "DIST", "lat", "lon"]
CLASSES = ("ALLUVIUM", "HARD_ROCK")

# GSI group_name -> (class NARROW, class BROAD, alluvium age, geological basis, VERIFY?)
# Basis statements are standard Bundelkhand stratigraphy; units marked VERIFY must be checked against the
# GSI legend / the 'lithologic' composition this script prints before the paper uses them.
GSI_MAP = {
    "BUNDELKHAND GRANITOID COMPLEX": ("HARD_ROCK", "HARD_ROCK", None,
        "Archaean granitoids/gneisses of the Bundelkhand craton; weathered-fractured crystalline aquifer", False),
    "OLDER SUPRACRUSTAL": ("HARD_ROCK", "HARD_ROCK", None,
        "Archaean metavolcano-sedimentary enclaves (amphibolite, BIF, quartzite, schist) within the granitoid complex", False),
    "OLDER METAMORPHICS": ("OTHER", "HARD_ROCK", None,
        "Pre-granitoid gneisses/schists of the cratonic basement -> crystalline, same aquifer type as the granitoids", True),
    "NEWER ALLUVIUM": ("ALLUVIUM", "ALLUVIUM", "NEWER", "Holocene unconsolidated fluvial deposits", False),
    "OLDER ALLUVIUM": ("ALLUVIUM", "ALLUVIUM", "OLDER", "Pleistocene unconsolidated fluvial deposits", False),
    "CHAMBAL NEWER ALLUVIUM": ("ALLUVIUM", "ALLUVIUM", "NEWER", "Holocene alluvium, Chambal-Yamuna system", False),
    "CHAMBAL OLDER ALLUVIUM": ("ALLUVIUM", "ALLUVIUM", "OLDER", "Pleistocene alluvium, Chambal-Yamuna system", False),
    "BIJAWAR": ("OTHER", "OTHER", None,
        "Palaeoproterozoic Bijawar metasediments; consolidated but not the granitoid basement", True),
    "BIJAWAR (UNDIFF)": ("OTHER", "OTHER", None,
        "Palaeoproterozoic Bijawar metasediments (undifferentiated)", True),
    "SEMRI": ("OTHER", "OTHER", None, "Vindhyan Supergroup, Semri Group sedimentary rocks", False),
    "KAIMUR": ("OTHER", "OTHER", None, "Vindhyan Supergroup, Kaimur Group sandstone", False),
    "REWA": ("OTHER", "OTHER", None, "Vindhyan Supergroup, Rewa Group sandstone/shale", False),
    "MALWA": ("OTHER", "OTHER", None, "Malwa Group (Deccan Trap basalt flows)", False),
    "LIMESTONE": ("OTHER", "OTHER", None, "Carbonate unit without stratigraphic group in the attribute table", True),
    "LATERITE": ("OTHER", "OTHER", None, "Lateritic duricrust (weathering profile)", False),
    "GRAVEL/PEBBLE OF LATERITE WITH LITHOMARGIC CLAY": ("OTHER", "OTHER", None,
        "Lateritic gravel and lithomarge (weathering profile)", False),
    "SIWALIK": ("OTHER", "OTHER", None, "Neogene Siwalik-type sediments; unexpected in Bundelkhand", True),
}


# =====================================================================================================
# helpers
# =====================================================================================================
def parse_args():
    ap = argparse.ArgumentParser(description="Geology vs hydrograph regime (audited v3)")
    for k, v in CFG.items():
        if isinstance(v, tuple):
            ap.add_argument("--" + k.lower().replace("_", "-"), type=float, nargs="+", default=list(v))
        else:
            ap.add_argument("--" + k.lower().replace("_", "-"), type=type(v), default=v)
    a, _unknown = ap.parse_known_args()      # Jupyter passes "-f kernel.json" -> ignored
    a = vars(a)
    cfg = {k: a[k.lower()] for k in CFG}
    cfg["CONTACT_TOL_SENS"] = tuple(cfg["CONTACT_TOL_SENS"])
    if cfg["SCHEME"] not in ("BROAD", "NARROW"):
        sys.exit("[STOP] --scheme must be BROAD or NARROW")
    return cfg


def gen(cfg, name):
    """independent, reproducible RNG per test (results do not depend on test order)."""
    return np.random.default_rng([cfg["SEED"], zlib.crc32(name.encode())])


def U(gs):
    gs = [g for g in gs if g is not None and not g.is_empty]
    return unary_union(gs) if gs else None


def poly_only(g):
    if g is None or g.is_empty:
        return None
    if g.geom_type in ("Polygon", "MultiPolygon"):
        return g
    if g.geom_type == "GeometryCollection":
        ps = [p for p in g.geoms if p.geom_type in ("Polygon", "MultiPolygon")]
        return unary_union(ps) if ps else None
    return None


def lines_only(g):
    if g is None or g.is_empty:
        return None
    if g.geom_type in ("LineString", "MultiLineString"):
        return g
    if hasattr(g, "geoms"):
        ls = [p for p in g.geoms if p.geom_type in ("LineString", "MultiLineString")]
        return unary_union(ls) if ls else None
    return None


def bh(p):
    """Benjamini-Hochberg; NaN p-values stay NaN and do NOT count towards the family size."""
    p = np.asarray(p, float)
    q = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    m = ok.sum()
    if m == 0:
        return q
    po = p[ok]
    o = np.argsort(po)
    adj = po[o] * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty(m)
    out[o] = np.clip(adj, 0, 1)
    q[ok] = out
    return q


def p_two(null, obs):
    if not np.isfinite(obs):
        return np.nan
    return (np.sum(np.abs(null) >= abs(obs) - 1e-12) + 1) / (len(null) + 1)


def centred_pct_rank(y, groups):
    s = pd.Series(np.asarray(y, float))
    g = pd.Series(np.asarray(groups))
    r = s.groupby(g).rank(method="average", pct=True)
    return (r - r.groupby(g).transform("mean")).to_numpy()


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# =====================================================================================================
# F1: stratified ALLUVIUM vs HARD_ROCK
# =====================================================================================================
def f1_strata(df, m, cls_col, min_cell):
    q = df[df[cls_col].isin(CLASSES)][["DIST", cls_col, m]].dropna()
    strata, info = [], []
    for d, g in q.groupby("DIST"):
        a = (g[cls_col] == "ALLUVIUM").to_numpy()
        nA, nH = int(a.sum()), int((~a).sum())
        used = nA >= min_cell and nH >= min_cell
        info.append({"DIST": d, "nA": nA, "nH": nH, "used": used})
        if used:
            strata.append((d, g[m].to_numpy(float), a))
    return strata, pd.DataFrame(info)


def r_strat(strata):
    num = den = 0.0
    for _, y, a in strata:
        r = stats.rankdata(y)
        nA, nH = a.sum(), (~a).sum()
        Uv = r[a].sum() - nA * (nA + 1) / 2
        num += 2 * Uv - nA * nH
        den += nA * nH
    return num / den if den else np.nan


def perm_r_strat(strata, rng, nperm):
    num = np.zeros(nperm)
    den = 0.0
    for _, y, a in strata:
        r = stats.rankdata(y)
        n, nA = len(y), int(a.sum())
        nH = n - nA
        idx = np.argsort(rng.random((nperm, n)), axis=1)[:, :nA]     # exactly nA alluvium labels per draw
        Uv = r[idx].sum(axis=1) - nA * (nA + 1) / 2
        num += 2 * Uv - nA * nH
        den += nA * nH
    return num / den


def boot_r_strat(strata, rng, nboot):
    out = np.empty(nboot)
    for b in range(nboot):
        bs = []
        for d, y, a in strata:
            ya, yh = y[a], y[~a]
            yb = np.r_[rng.choice(ya, len(ya)), rng.choice(yh, len(yh))]
            bs.append((d, yb, np.r_[np.ones(len(ya), bool), np.zeros(len(yh), bool)]))
        out[b] = r_strat(bs)
    return np.nanpercentile(out, [2.5, 97.5])


def van_elteren(strata):
    T = V = 0.0
    for _, y, a in strata:
        n, nA = len(y), int(a.sum())
        nH = n - nA
        r = stats.rankdata(y)
        _, t = np.unique(y, return_counts=True)
        tie = float((t ** 3 - t).sum())
        var = nA * nH / 12 * ((n + 1) - tie / (n * (n - 1)))
        w = 1 / (n + 1)
        T += w * (r[a].sum() - nA * (n + 1) / 2)
        V += w * w * var
    if V <= 0:
        return np.nan, np.nan
    z = T / np.sqrt(V)
    return z, 2 * stats.norm.sf(abs(z))


def run_f1(df, m, cls_col, min_cell, cfg, tag):
    strata, info = f1_strata(df, m, cls_col, min_cell)
    row = {"metric": m, "analysis": tag, "districts_used": ",".join(s[0] for s in strata),
           "n_alluv": int(sum(s[2].sum() for s in strata)), "n_hard": int(sum((~s[2]).sum() for s in strata))}
    if not strata:
        row.update(r_strat=np.nan, ci_lo=np.nan, ci_hi=np.nan, p_perm=np.nan, z_vanElteren=np.nan,
                   p_vanElteren=np.nan, note="no district has both classes at the minimum size")
        return row, info
    obs = r_strat(strata)
    null = perm_r_strat(strata, gen(cfg, f"F1perm|{tag}|{m}"), cfg["N_PERM"])
    lo, hi = boot_r_strat(strata, gen(cfg, f"F1boot|{tag}|{m}"), cfg["N_BOOT"])
    z, pz = van_elteren(strata)
    row.update(r_strat=obs, ci_lo=lo, ci_hi=hi, p_perm=p_two(null, obs), z_vanElteren=z, p_vanElteren=pz,
               note="" if min(row["n_alluv"], row["n_hard"]) >= 10 else "small n: CI approximate")
    return row, info


# =====================================================================================================
# F2: within-class, within-district rank correlation
# =====================================================================================================
def f2_test(df, xcol, m, cls, cfg, tag, do_boot=True):
    q = df[df.CLASS == cls][["DIST", xcol, m]].dropna()
    q = q[q.groupby("DIST")[m].transform("size") >= cfg["MIN_STRAT_N"]]
    row = {"class": cls, "metric": m, "control": xcol, "analysis": tag, "n": len(q),
           "districts_used": ",".join(sorted(q.DIST.unique()))}
    if len(q) < 10:
        row.update(rho_adj=np.nan, ci_lo=np.nan, ci_hi=np.nan, p_perm=np.nan, note="insufficient n")
        return row
    xc, yc = centred_pct_rank(q[xcol], q.DIST), centred_pct_rank(q[m], q.DIST)
    den = np.sqrt((xc ** 2).sum() * (yc ** 2).sum())
    if den == 0:
        row.update(rho_adj=np.nan, ci_lo=np.nan, ci_hi=np.nan, p_perm=np.nan, note="no within-district variation")
        return row
    obs = (xc * yc).sum() / den
    rng = gen(cfg, f"F2perm|{tag}|{cls}|{xcol}|{m}")
    N = cfg["N_PERM"]
    XP = np.empty((N, len(xc)))
    for d in q.DIST.unique():
        ii = np.where(q.DIST.to_numpy() == d)[0]
        XP[:, ii] = xc[ii][np.argsort(rng.random((N, len(ii))), axis=1)]
    null = XP @ yc / den
    lo = hi = np.nan
    if do_boot:
        rb = gen(cfg, f"F2boot|{tag}|{cls}|{xcol}|{m}")
        bs = np.empty(cfg["N_BOOT"])
        groups = [np.where(q.DIST.to_numpy() == d)[0] for d in q.DIST.unique()]
        X, Y, Dd = q[xcol].to_numpy(float), q[m].to_numpy(float), q.DIST.to_numpy()
        for b in range(cfg["N_BOOT"]):
            ii = np.concatenate([rb.choice(g, len(g)) for g in groups])
            xb, yb = centred_pct_rank(X[ii], Dd[ii]), centred_pct_rank(Y[ii], Dd[ii])
            dd = np.sqrt((xb ** 2).sum() * (yb ** 2).sum())
            bs[b] = (xb * yb).sum() / dd if dd > 0 else np.nan
        lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    row.update(rho_adj=obs, ci_lo=lo, ci_hi=hi, p_perm=p_two(null, obs), note="")
    return row


# =====================================================================================================
# spatial diagnostics
# =====================================================================================================
def moran_within(df, m, cfg):
    q = df[["DIST", "X", "Y", m]].dropna()
    n = len(q)
    if n < 10:
        return {"metric": m, "n": n, "moran_I": np.nan, "E_I": np.nan, "p_perm": np.nan, "n_pairs": 0}
    z = centred_pct_rank(q[m], q.DIST)
    xy = q[["X", "Y"]].to_numpy()
    d = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])
    W = ((d > 0) & (d <= cfg["MORAN_KM"] * 1000)).astype(float)
    S0 = W.sum()
    if S0 == 0 or (z @ z) == 0:
        return {"metric": m, "n": n, "moran_I": np.nan, "E_I": np.nan, "p_perm": np.nan, "n_pairs": 0}
    I = n / S0 * (z @ W @ z) / (z @ z)
    rng = gen(cfg, f"moran|{m}")
    groups = [np.where(q.DIST.to_numpy() == d_)[0] for d_ in q.DIST.unique()]
    null = np.empty(cfg["N_PERM_MORAN"])
    for k in range(len(null)):
        zp = z.copy()
        for g in groups:
            zp[g] = z[g][rng.permutation(len(g))]
        null[k] = n / S0 * (zp @ W @ zp) / (zp @ zp)
    return {"metric": m, "n": n, "moran_I": I, "E_I": -1 / (n - 1),
            "p_perm": (np.sum(null >= I - 1e-12) + 1) / (len(null) + 1), "n_pairs": int(S0 / 2)}


def spatial_clusters(df, km):
    xy = df[["X", "Y"]].to_numpy()
    d = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])
    key = (df.DIST.astype(str) + "|" + df.CLASS.fillna("NA_" + df.SID.astype(str))).to_numpy()
    A = (d <= km * 1000) & (key[:, None] == key[None, :])
    _, lab = connected_components(csr_matrix(A), directed=False)
    return lab


# =====================================================================================================
# MAIN
# =====================================================================================================
def main():
    cfg = parse_args()
    os.makedirs(cfg["OUT"], exist_ok=True)
    LOG = []

    def log(s):
        LOG.append(s)
        print(s)

    # ---------------------------------------------------------------- stations: validation
    if not os.path.exists(cfg["IN_CSV"]):
        sys.exit(f"[STOP] input not found: {cfg['IN_CSV']} (run regime_controls.py first, from the notebook folder)")
    D = pd.read_csv(cfg["IN_CSV"])
    miss = [c for c in REQ_COLS if c not in D.columns]
    if miss:
        sys.exit(f"[STOP] input lacks required columns {miss}")
    mets = [m for m in METRICS if m in D.columns]
    if len(mets) < len(METRICS):
        log(f"[WARN] metrics missing from input, skipped: {sorted(set(METRICS) - set(mets))}")
    if not mets:
        sys.exit("[STOP] none of the metrics are present")
    for m in mets + ["lat", "lon"] + (["RAIN_SHIFT"] if "RAIN_SHIFT" in D.columns else []):
        D[m] = pd.to_numeric(D[m], errors="coerce").replace([np.inf, -np.inf], np.nan)
    D["SID"] = D["SID"].astype(str)
    D["DIST"] = D["DIST"].astype(str).str.strip().str.upper()
    if D.SID.duplicated().any():
        sys.exit(f"[STOP] duplicated SIDs in input: {D.SID[D.SID.duplicated()].tolist()[:10]} -> fix upstream")
    bad_xy = ~(D.lat.between(20, 30) & D.lon.between(74, 85))
    if bad_xy.any():
        log(f"[WARN] {int(bad_xy.sum())} stations with missing/out-of-region coordinates excluded: "
            f"{D.SID[bad_xy].tolist()}")
    D = D[~bad_xy].reset_index(drop=True)
    excess_chk = np.nan
    if {"SHIFT", "EXCESS", "RAIN_SHIFT"} <= set(D.columns):
        excess_chk = float(np.nanmax(np.abs(D.EXCESS - (D.SHIFT - D.RAIN_SHIFT)))) if D.EXCESS.notna().any() else np.nan

    # ---------------------------------------------------------------- geology: load, repair, classify
    files = sorted(glob.glob(os.path.join(cfg["GEO_DIR"], "**", cfg["GEO_GLOB"]), recursive=True))
    if not files:
        sys.exit(f"[STOP] no shapefiles under GEO_DIR={cfg['GEO_DIR']}")
    # identical files (e.g. a second copy in a sub-folder) are read only once
    seen, uniq = {}, []
    for f in files:
        h = hashlib.md5()
        for ext in (".shp", ".dbf"):
            pth = os.path.splitext(f)[0] + ext
            if os.path.exists(pth):
                h.update(open(pth, "rb").read())
        k = h.hexdigest()
        if k in seen:
            log(f"[WARN] duplicate shapefile ignored: {f}  (identical to {seen[k]})")
        else:
            seen[k] = f
            uniq.append(f)
    files = uniq
    parts, fdiag = [], []
    for f in files:
        g = gpd.read_file(f)
        src = os.path.splitext(os.path.relpath(f, cfg["GEO_DIR"]))[0]
        crs_in = str(g.crs)
        if g.crs is None:
            log(f"[WARN] {src}: no CRS -> assumed EPSG:4326")
            g = g.set_crs("EPSG:4326")
        if "group_name" not in g.columns:
            sys.exit(f"[STOP] {src} has no 'group_name' column")
        n0 = len(g)
        grp = g["group_name"].astype("string").str.strip().str.upper()
        n_missing_grp = int((grp.isna() | (grp == "")).sum())
        grp = grp.mask(grp == "").fillna("UNKNOWN_GROUP")
        lit = (g["lithologic"].astype("string").str.strip().str.upper().fillna("")
               if "lithologic" in g.columns else pd.Series([""] * n0, index=g.index))
        gg = gpd.GeoDataFrame({"GROUP": grp.astype(str).to_numpy(), "LITHO": lit.astype(str).to_numpy(), "SRC": src},
                              geometry=g.geometry.to_numpy(), crs=g.crs)
        keep = gg.geometry.notna()
        keep &= ~gg.geometry.is_empty.fillna(True)
        n_null = int((~keep).sum())
        gg = gg[keep]
        rp = gg.to_crs("EPSG:4326").geometry.representative_point()
        bx = cfg["STUDY_BBOX"]
        inb = (rp.x >= bx[0]) & (rp.x <= bx[2]) & (rp.y >= bx[1]) & (rp.y <= bx[3])
        n_out = int((~inb).sum())
        if n_out:
            log(f"[WARN] {src}: {n_out} polygons outside STUDY_BBOX dropped "
                f"(groups: {sorted(gg.GROUP[~inb].unique().tolist())})")
        gg = gg[inb].to_crs(cfg["UTM"])
        bad = ~gg.is_valid
        n_invalid = int(bad.sum())
        a0 = gg.area.sum()
        if n_invalid:
            rep = gg.geometry[bad].make_valid() if hasattr(gg.geometry, "make_valid") else gg.geometry[bad].buffer(0)
            new = gg.geometry.copy()
            new.loc[bad] = gpd.GeoSeries([poly_only(x) for x in rep], index=rep.index, crs=gg.crs)
            gg = gg.set_geometry(new)
        gg = gg[gg.geometry.notna() & ~gg.geometry.is_empty]
        a1 = gg.area.sum()
        lonlat = gg.to_crs("EPSG:4326").total_bounds
        fdiag.append({"file": src, "crs_in": crs_in, "features": n0, "outside_bbox_dropped": n_out, "null_or_empty_dropped": n_null,
                      "invalid_repaired": n_invalid, "missing_group_name": n_missing_grp,
                      "area_km2": a1 / 1e6, "repair_area_change_pct": 100 * (a1 - a0) / a0 if a0 else np.nan,
                      "lon_min": lonlat[0], "lon_max": lonlat[2], "lat_min": lonlat[1], "lat_max": lonlat[3]})
        parts.append(gg)
    geo = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=cfg["UTM"])
    geo["PID"] = np.arange(len(geo))
    prim, alt = (1, 0) if cfg["SCHEME"] == "BROAD" else (0, 1)
    geo["CLASS"] = geo.GROUP.map(lambda s: GSI_MAP[s][prim] if s in GSI_MAP else "UNMAPPED")
    geo["CLASS_ALT"] = geo.GROUP.map(lambda s: GSI_MAP[s][alt] if s in GSI_MAP else "UNMAPPED")
    geo["AGE"] = geo.GROUP.map(lambda s: GSI_MAP[s][2] if s in GSI_MAP else None)
    FD = pd.DataFrame(fdiag)

    # mapping audit table, with lithologic composition by area
    geo["_A"] = geo.area / 1e6
    mp = []
    for s, g in geo.groupby("GROUP"):
        lit = g.groupby("LITHO")._A.sum().sort_values(ascending=False)
        tot = lit.sum()
        top = "; ".join(f"{k or '(blank)'} {100 * v / tot:.0f}%" for k, v in lit.head(4).items())
        e = GSI_MAP.get(s)
        mp.append({"GROUP": s, "n_polygons": len(g), "area_km2": round(g._A.sum(), 2),
                   "CLASS_primary": g.CLASS.iloc[0], "CLASS_alt": g.CLASS_ALT.iloc[0],
                   "AGE": e[2] if e else None, "basis": e[3] if e else "NOT IN MAPPING TABLE -> excluded",
                   "VERIFY": (e[4] if e else True), "top_lithologic_by_area": top})
    MP = pd.DataFrame(mp)
    unmapped = MP[MP.CLASS_primary == "UNMAPPED"].GROUP.tolist()

    # ---------------------------------------------------------------- spatial join + auditable resolution
    pts = gpd.GeoDataFrame(D[["SID", "DIST", "lat", "lon"]].copy(),
                           geometry=gpd.points_from_xy(D.lon, D.lat), crs="EPSG:4326").to_crs(cfg["UTM"])
    pts["X"], pts["Y"] = pts.geometry.x, pts.geometry.y
    cols = ["PID", "GROUP", "CLASS", "CLASS_ALT", "AGE", "SRC", "geometry"]
    J = gpd.sjoin(pts[["SID", "DIST", "geometry"]], geo[cols], how="left", predicate="intersects")
    J = J[J.PID.notna()]

    def resolve(m, dist):
        groups, classes, alts = set(m.GROUP), set(m.CLASS), set(m.CLASS_ALT)
        ages = set(a for a in m.AGE if isinstance(a, str))
        if len(groups) == 1:
            r = m.iloc[0]
            return ("SINGLE" if len(m) == 1 else "MULTI_SAME_GROUP", r.GROUP, r.CLASS, r.CLASS_ALT, r.AGE)
        own = m[m.SRC.str.upper().str.contains(dist, regex=False)]
        if len(own) and own.GROUP.nunique() == 1:
            r = own.iloc[0]
            return ("MULTI_CONFLICT_OWN_DISTRICT_FILE", r.GROUP, r.CLASS, r.CLASS_ALT, r.AGE)
        if len(classes) == 1 and len(alts) == 1:
            return ("MULTI_CONFLICT_SAME_CLASS", "|".join(sorted(groups)), classes.pop(), alts.pop(),
                    ages.pop() if len(ages) == 1 else ("MIXED" if ages else None))
        return ("AMBIGUOUS_EXCLUDED", "|".join(sorted(groups)), None, None, None)

    rows = []
    for _, p in pts.iterrows():
        m = J[J.SID == p.SID]
        rec = {"SID": p.SID, "DIST": p.DIST, "n_match": len(m),
               "matched_groups": "|".join(sorted(set(m.GROUP))), "matched_classes": "|".join(sorted(set(m.CLASS))),
               "matched_files": "|".join(sorted(set(m.SRC))), "nearest_m": 0.0 if len(m) else np.nan}
        if len(m):
            res = resolve(m, p.DIST)
        else:
            nn = gpd.sjoin_nearest(pts[pts.SID == p.SID][["SID", "geometry"]], geo[cols],
                                   max_distance=cfg["NEAREST_TOL_M"], distance_col="dist_m")
            nn = nn[nn.PID.notna()] if "PID" in nn else nn.iloc[0:0]
            if len(nn):
                r0 = resolve(nn, p.DIST)
                res = ("NEAREST_WITHIN_TOL" if not r0[0].startswith("AMBIG") else "AMBIGUOUS_EXCLUDED",) + r0[1:]
                rec["nearest_m"] = float(nn.dist_m.min())
                rec["matched_groups"] = "|".join(sorted(set(nn.GROUP)))
                rec["matched_files"] = "|".join(sorted(set(nn.SRC)))
            else:
                res = ("UNMATCHED_EXCLUDED", None, None, None, None)
        rec.update(resolution=res[0], GROUP=res[1], CLASS=res[2], CLASS_ALT=res[3], AGE=res[4])
        rows.append(rec)
    AU = pd.DataFrame(rows)
    AU.loc[AU.CLASS.isin(["UNMAPPED"]), ["CLASS", "resolution"]] = [None, "UNMAPPED_GROUP_EXCLUDED"]
    AU.loc[AU.CLASS_ALT.isin(["UNMAPPED"]), "CLASS_ALT"] = None

    # distance to the nearest polygon of ANOTHER class (assignment uncertainty near boundaries)
    cls_union = {c: U(geo[geo.CLASS != c].geometry) for c in ("ALLUVIUM", "HARD_ROCK", "OTHER")}
    P = pts.set_index("SID")
    AU["d_other_class_m"] = [P.geometry[s].distance(cls_union[c]) if isinstance(c, str) and cls_union.get(c) is not None
                             else np.nan for s, c in zip(AU.SID, AU.CLASS)]
    AU["NEAR_BOUNDARY"] = AU.d_other_class_m < cfg["BOUNDARY_UNCERT_M"]
    AU["UNAMBIGUOUS"] = AU.resolution.isin(["SINGLE", "MULTI_SAME_GROUP"]) & ~AU.NEAR_BOUNDARY

    # ---------------------------------------------------------------- contact construction + diagnostics
    file_edges = U([U(geo[geo.SRC == s].geometry).boundary for s in geo.SRC.unique()])
    unions = {}

    def get_unions(col, min_area):
        k = (col, min_area)
        if k not in unions:
            g = geo if min_area <= 0 else geo[geo._A >= min_area]
            unions[k] = (U(g[g[col] == "HARD_ROCK"].geometry), U(g[g[col] == "ALLUVIUM"].geometry))
        return unions[k]

    def contact(col="CLASS", tol=cfg["CONTACT_TOL_M"], drop_edges=False, min_area=0.0):
        hard, allu = get_unions(col, min_area)
        if hard is None or allu is None:
            return None
        c = lines_only(hard.boundary.intersection(allu.buffer(max(tol, 0.5))))
        if c is not None and drop_edges and file_edges is not None:
            c = lines_only(c.difference(file_edges.buffer(cfg["EDGE_TOL_M"])))
        return c

    VAR = {"BASE": dict(), "NO_FILE_EDGE": dict(drop_edges=True),
           "MIN_AREA": dict(min_area=cfg["MIN_AREA_KM2_SENS"]), "ALT_SCHEME": dict(col="CLASS_ALT")}
    for t in cfg["CONTACT_TOL_SENS"]:
        VAR[f"TOL_{t:g}"] = dict(tol=t)
    CON = {k: contact(**v) for k, v in VAR.items()}
    hard_b, allu_b = get_unions("CLASS", 0.0)
    overlap_km2 = hard_b.intersection(allu_b).area / 1e6 if hard_b is not None and allu_b is not None else np.nan
    base = CON["BASE"]
    edge_share = (base.intersection(file_edges.buffer(cfg["EDGE_TOL_M"])).length / base.length
                  if base is not None and base.length > 0 and file_edges is not None else np.nan)
    src_u = {s: U(geo[geo.SRC == s].geometry) for s in geo.SRC.unique()}
    ss = list(src_u)
    file_overlap_km2 = sum(src_u[a].intersection(src_u[b]).area for i, a in enumerate(ss) for b in ss[i + 1:]) / 1e6
    CD = pd.DataFrame([{"variant": k, "contact_km": (c.length / 1000 if c is not None else 0.0)} for k, c in CON.items()])

    for k, c in CON.items():
        AU[f"CONTACT_KM_{k}"] = [P.geometry[s].distance(c) / 1000 if c is not None else np.nan for s in AU.SID]
    sign = np.where(AU.CLASS == "ALLUVIUM", 1.0, np.where(AU.CLASS == "HARD_ROCK", -1.0, np.nan))
    AU["SDIST_KM"] = sign * AU.CONTACT_KM_BASE
    for k in CON:
        AU.loc[~AU.CLASS.isin(CLASSES), f"CONTACT_KM_{k}"] = np.nan
    CD["spearman_vs_BASE"] = [stats.spearmanr(AU.CONTACT_KM_BASE, AU[f"CONTACT_KM_{k}"], nan_policy="omit")[0]
                              if AU[f"CONTACT_KM_{k}"].notna().sum() > 3 else np.nan for k in CON]

    # ---------------------------------------------------------------- analysis table
    G = D.merge(AU.drop(columns=["DIST"]), on="SID", how="left")
    G = G.merge(pts[["SID", "X", "Y"]], on="SID", how="left")
    G["CLUSTER"] = spatial_clusters(G, cfg["CLUSTER_KM"])
    AU = AU.merge(G[["SID", "CLUSTER"]], on="SID", how="left")
    AU.to_csv(os.path.join(cfg["OUT"], "station_geology_audit.csv"), index=False)
    MP.to_csv(os.path.join(cfg["OUT"], "geology_class_mapping.csv"), index=False)
    FD.to_csv(os.path.join(cfg["OUT"], "geology_file_diagnostics.csv"), index=False)

    xt = pd.crosstab(G.DIST, G.CLASS.fillna("(excluded)"))
    for c in CLASSES:
        if c not in xt:
            xt[c] = 0
    single_class = [d for d in xt.index if (xt.loc[d, "ALLUVIUM"] == 0) != (xt.loc[d, "HARD_ROCK"] == 0)]
    mixed = [d for d in xt.index if xt.loc[d, "ALLUVIUM"] >= cfg["MIN_CELL"] and xt.loc[d, "HARD_ROCK"] >= cfg["MIN_CELL"]]

    # ---------------------------------------------------------------- F1 (confirmatory) + sensitivities
    F1, F1S, F1info = [], [], {}
    for m in mets:
        r, info = run_f1(G, m, "CLASS", cfg["MIN_CELL"], cfg, "PRIMARY")
        F1.append(r)
        F1info[m] = info
        r2, _ = run_f1(G[G.UNAMBIGUOUS.fillna(False)], m, "CLASS", cfg["MIN_CELL"], cfg, "UNAMBIGUOUS_ONLY")
        F1S.append(r2)
        GC = (G[G.CLASS.isin(CLASSES)].dropna(subset=[m]).groupby("CLUSTER")
              .agg(DIST=("DIST", "first"), CLASS=("CLASS", "first"), **{m: (m, "median")}).reset_index())
        r3, _ = run_f1(GC, m, "CLASS", cfg["MIN_CELL_CLUSTER"], cfg, "CLUSTER_LEVEL")
        r3["n_alluv"], r3["n_hard"] = f"{r3['n_alluv']} clusters", f"{r3['n_hard']} clusters"
        F1S.append(r3)
    n_alt_change = int((G.CLASS.fillna("") != G.CLASS_ALT.fillna("")).sum())
    if n_alt_change:
        G2 = G.copy()
        G2["CLASS"] = G2.CLASS_ALT
        for m in mets:
            F1S.append(run_f1(G2, m, "CLASS", cfg["MIN_CELL"], cfg, "ALT_SCHEME")[0])
    F1 = pd.DataFrame(F1)
    F1["q_FDR"] = bh(F1.p_perm)
    F1S = pd.DataFrame(F1S)

    # ---------------------------------------------------------------- F2 (secondary) + contact sensitivity
    F2 = pd.DataFrame([f2_test(G, "CONTACT_KM_BASE", m, c, cfg, "PRIMARY") for c in CLASSES for m in mets])
    F2["q_FDR"] = bh(F2.p_perm)
    F2S = pd.DataFrame([f2_test(G, f"CONTACT_KM_{k}", m, c, cfg, k, do_boot=False)
                        for k in CON if k != "BASE" for c in CLASSES for m in mets])

    # ---------------------------------------------------------------- descriptive / exploratory
    POOL, PDX = [], []
    for m in mets:
        a = G.loc[G.CLASS == "ALLUVIUM", m].dropna()
        h = G.loc[G.CLASS == "HARD_ROCK", m].dropna()
        if len(a) >= 2 and len(h) >= 2:
            Uv, p = stats.mannwhitneyu(a, h, alternative="two-sided")
            POOL.append({"metric": m, "n_alluv": len(a), "n_hard": len(h), "med_alluv": a.median(),
                         "med_hard": h.median(), "r_rb": 2 * Uv / (len(a) * len(h)) - 1, "p_raw": p,
                         "status": "CONFOUNDED WITH DISTRICT - descriptive only"})
        for d in mixed:
            g = G[(G.DIST == d) & G.CLASS.isin(CLASSES)].dropna(subset=[m])
            a, h = g.loc[g.CLASS == "ALLUVIUM", m], g.loc[g.CLASS == "HARD_ROCK", m]
            if len(a) >= 2 and len(h) >= 2:
                Uv, p = stats.mannwhitneyu(a, h, alternative="two-sided")
                st = [(d, g[m].to_numpy(float), (g.CLASS == "ALLUVIUM").to_numpy())]
                lo, hi = boot_r_strat(st, gen(cfg, f"pd|{d}|{m}"), cfg["N_BOOT"])
                PDX.append({"DIST": d, "metric": m, "n_alluv": len(a), "n_hard": len(h), "med_alluv": a.median(),
                            "med_hard": h.median(), "r_rb": 2 * Uv / (len(a) * len(h)) - 1, "ci_lo": lo, "ci_hi": hi,
                            "p_raw": p})
    POOL, PDX = pd.DataFrame(POOL), pd.DataFrame(PDX)
    SP = pd.DataFrame([moran_within(G, m, cfg) for m in mets])
    nn_km = []
    xy = G[["X", "Y"]].to_numpy()
    if len(xy) > 1:
        dd = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])
        np.fill_diagonal(dd, np.inf)
        nn_km = dd.min(axis=1) / 1000
    ages = G[G.CLASS == "ALLUVIUM"].AGE.fillna("UNKNOWN").value_counts()

    for name, df_ in [("F1_stratified", F1), ("F1_sensitivity", F1S), ("F2_contact", F2),
                      ("F2_contact_sensitivity", F2S), ("pooled_descriptive", POOL),
                      ("per_district_exploratory", PDX), ("spatial_diagnostics", SP), ("contact_diagnostics", CD)]:
        df_.to_csv(os.path.join(cfg["OUT"], f"geology_{name}.csv"), index=False)
    with open(os.path.join(cfg["OUT"], "run_log.json"), "w") as f:
        json.dump({"time": datetime.now().isoformat(timespec="seconds"), "config": cfg,
                   "input_csv_md5": md5(cfg["IN_CSV"]), "shapefiles": files,
                   "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                                "geopandas": gpd.__version__, "shapely": shapely.__version__,
                                "scipy": __import__("scipy").__version__},
                   "warnings": LOG}, f, indent=2, default=str)

    # =================================================================================================
    # VERDICT
    # =================================================================================================
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_colwidth", 70)
    R = lambda x: x.round(3).to_string(index=False) if len(x) else "  (none)"
    print("\n" + "=" * 90)
    print("VERDICT  regime_geology v3")
    print("=" * 90)
    print(f"scheme {cfg['SCHEME']} | seed {cfg['SEED']} | N_PERM {cfg['N_PERM']} | N_BOOT {cfg['N_BOOT']} | "
          f"stations {len(G)} | metrics {mets}")
    print(f"EXCESS == SHIFT - RAIN_SHIFT check (max abs diff): {excess_chk}")
    print("\n[1] GEOLOGY FILES (CRS, repairs):")
    print(R(FD))
    print(f"    overlap between district files: {file_overlap_km2:.2f} km2 | "
          f"hard-rock/alluvium overlapping area (conflicting polygons): {overlap_km2:.3f} km2")
    print("\n[2] CLASS MAPPING (VERIFY=True -> check against GSI legend before publishing):")
    print(MP[["GROUP", "n_polygons", "area_km2", "CLASS_primary", "CLASS_alt", "VERIFY", "top_lithologic_by_area"]]
          .to_string(index=False))
    if unmapped:
        print(f"    [WARN] groups NOT in mapping table (excluded): {unmapped}")
    print(f"    stations whose class differs between BROAD and NARROW schemes: {n_alt_change}")
    print("\n[3] STATION ASSIGNMENT AUDIT:")
    print(AU.resolution.value_counts().to_string())
    print(f"    stations with >1 polygon match: {int((AU.n_match > 1).sum())} | "
          f"near another class (<{cfg['BOUNDARY_UNCERT_M']:.0f} m): {int(AU.NEAR_BOUNDARY.sum())}")
    flag = AU[~AU.resolution.isin(["SINGLE", "MULTI_SAME_GROUP"]) | AU.NEAR_BOUNDARY]
    if len(flag):
        print(flag[["SID", "DIST", "n_match", "matched_groups", "resolution", "CLASS", "nearest_m", "d_other_class_m"]]
              .round(1).to_string(index=False))
    print("\n[4] DISTRICT x CLASS (confounding):")
    print(pd.crosstab(G.DIST, G.CLASS.fillna("(excluded)"), margins=True).to_string())
    print(f"    single-class districts (no within-district geology information): {single_class}")
    print(f"    mixed districts entering F1 (>= {cfg['MIN_CELL']} of each class): {mixed}")
    print("\n[5] CONTACT DIAGNOSTICS:  SDIST_KM = + inside ALLUVIUM, - inside HARD_ROCK")
    print(R(CD))
    print(f"    share of BASE contact within {cfg['EDGE_TOL_M']:.0f} m of a map-file edge: "
          f"{100 * edge_share:.1f}%  (high -> contact partly an edge-matching artefact)")
    print("\n[6] SPATIAL DEPENDENCE:")
    if len(nn_km):
        print(f"    nearest-neighbour distance: median {np.median(nn_km):.1f} km | "
              f"stations with a neighbour < {cfg['CLUSTER_KM']:.0f} km: {100 * np.mean(nn_km < cfg['CLUSTER_KM']):.0f}% | "
              f"clusters {G.CLUSTER.nunique()} for {len(G)} stations")
    print("    Moran's I of within-district rank residuals (binary weights <= "
          f"{cfg['MORAN_KM']:.0f} km; one-sided perm. within district):")
    print(R(SP))
    print("\n[7] F1 CONFIRMATORY: ALLUVIUM vs HARD_ROCK within district  "
          "[r_strat>0 -> alluvium higher; BH-FDR over metrics]")
    print(R(F1))
    print("\n    F1 SENSITIVITY (no FDR):")
    print(R(F1S[["metric", "analysis", "n_alluv", "n_hard", "r_strat", "ci_lo", "ci_hi", "p_perm", "note"]]))
    if not n_alt_change:
        print("    ALT_SCHEME: no station changes class -> F1 identical under NARROW and BROAD")
    print("\n[8] F2 SECONDARY: distance to contact, within class & district  [BH-FDR over 2 classes x metrics]")
    print(R(F2[["class", "metric", "n", "districts_used", "rho_adj", "ci_lo", "ci_hi", "p_perm", "q_FDR", "note"]]))
    if len(F2S):
        sens = F2S.pivot_table(index=["class", "metric"], columns="analysis", values="rho_adj")
        sens.insert(0, "BASE", F2.set_index(["class", "metric"]).rho_adj)
        print("    F2 rho_adj under alternative contact definitions (sensitivity, no FDR):")
        print(sens.round(3).to_string())
    print("\n[9] DESCRIPTIVE ONLY")
    print("    pooled (confounded with district -- NOT evidence of a geology effect):")
    print(R(POOL))
    print("    per mixed district (exploratory, no FDR):")
    print(R(PDX))
    print("\n[10] NEWER vs OLDER ALLUVIUM:")
    print("    station counts:", ages.to_dict())
    if ages.get("NEWER", 0) < cfg["MIN_CELL"] or ages.get("OLDER", 0) < cfg["MIN_CELL"]:
        print(f"    -> comparison CANNOT be performed (needs >= {cfg['MIN_CELL']} stations of each age). No result reported.")
    else:
        print("    -> enough stations; NOT tested here (not prespecified). Add as a new prespecified family if needed.")

    # ---------------------------------------------------------------- interpretation template
    print("\n" + "-" * 90)
    print("INTERPRETATION (auto-filled, conservative)")
    print("-" * 90)
    if not mixed:
        print("Geology cannot be separated from district in this network: no district contains both classes at the "
              "minimum size. No within-district geology effect can be estimated.")
    else:
        nA = int(F1.n_alluv.max()) if len(F1) else 0
        nH = int(F1.n_hard.max()) if len(F1) else 0
        print(f"Scope: geology varies within only {len(mixed)} district(s) ({', '.join(mixed)}; up to {nA} alluvium vs "
              f"{nH} hard-rock stations). Districts {single_class} are single-class, so any basin-wide contrast is "
              f"inseparable from district. Results below apply ONLY to the mixed districts.")
        mor_sig = SP.loc[SP.p_perm < cfg["ALPHA"], "metric"].tolist() if len(SP) else []
        CL = F1S[F1S.analysis == "CLUSTER_LEVEL"].set_index("metric")
        for _, r in F1.iterrows():
            if not np.isfinite(r.r_strat):
                status = "NOT ESTIMABLE"
            else:
                ci0 = r.ci_lo <= 0 <= r.ci_hi
                cl = CL.r_strat.get(r.metric, np.nan)
                same = np.isfinite(cl) and np.sign(cl) == np.sign(r.r_strat)
                if r.q_FDR < cfg["ALPHA"] and not ci0 and same:
                    status = "SUPPORTED within mixed districts (survives FDR, CI excludes 0, same sign at cluster level)"
                elif r.q_FDR < cfg["ALPHA"]:
                    status = "NOT ROBUST (passes FDR but CI includes 0 or cluster-level sign differs/unavailable)"
                elif r.p_perm < cfg["ALPHA"]:
                    status = "SUGGESTIVE ONLY (raw p < 0.05, fails FDR) -- do not claim an effect"
                else:
                    status = "NO EVIDENCE"
                if r.metric in mor_sig and status.startswith("SUPPORTED"):
                    status += "; residual spatial autocorrelation present -> rely on cluster-level result"
            print(f"  {r.metric:<9} r_strat {r.r_strat:+.2f} [{r.ci_lo:+.2f}, {r.ci_hi:+.2f}] "
                  f"p {r.p_perm:.3f} q {r.q_FDR:.3f}  -> {status}")
    print("Always state: (i) geology is collinear with district outside the mixed districts; (ii) station-level "
          "permutations assume within-district exchangeability (see Moran's I and cluster-level results); "
          "(iii) SHIFT/EXCESS rest on two monsoons; (iv) the class mapping of units flagged VERIFY is an assumption.")
    print(f"\nOutputs in {cfg['OUT']}: station_geology_audit.csv, geology_class_mapping.csv, geology_F1_*.csv, "
          f"geology_F2_*.csv, geology_spatial_diagnostics.csv, geology_contact_diagnostics.csv, run_log.json")


if __name__ == "__main__":
    main()
