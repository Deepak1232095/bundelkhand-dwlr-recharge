"""
RECHARGE AUDIT (read-only; nothing in the existing outputs is changed)
=====================================================================
Rules are physical and fixed BEFORE looking at results. Days are corrected, stations are only FLAGGED.

STEP A  TRACE CHECK: GROSS_C is recomputed from dwlr_daily_depth_abs.csv with the stored MRC parameters
        (a, b, sigma) and the exact rules of wtf_mrc_recharge.py. Must reproduce the stored values (|diff| < 1 mm)
        before anything else is trusted.
STEP B  DAY-LEVEL QC (physical rules):
        Q1 CODE      daily depth within 0.5 m of 99 / 99.9 / 999, or >= 90 m   (99.0 is a code in the raw file)
        Q2 DEPTH     depth > MAX_DEPTH m (NAQUIM static water levels in these districts are <= ~26 m)
        Q3a ZERO     depth < 1 m while the 15-day median is > 3 m (telemetry zero code; seen Jun-Jul 2023)
        Q3b ISOLATED one-day excursion > 3 m from the 15-day median with both neighbours within 1 m of it
        Q4 DUPLICATE dates in the daily file with values differing > 0.5 m (reported only)
STEP C  GROSS_C recomputed with Q1-Q3 days removed (same MRC a, b, sigma -> isolates the effect of the data fix).
        Station FLAG if > 20 % of its Jun-Oct days were corrected or the correction changes GROSS_C by > 50 %.
STEP D  Summary statistics: original vs corrected vs corrected-without-flagged stations (median, 95 % bootstrap CI).
        Also: median-of-ratios vs ratio-of-medians for recharge % rain and BIANN / GROSS_C (definition check).
STEP E  Hydrographs of the 12 most extreme / flagged stations (diagnostic only).
Run in Jupyter:  %run recharge_audit.py      Requires: pandas numpy scipy matplotlib
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ======================================================================================== CONFIG
QC = r"DWLR_QC"
DAILY = os.path.join(QC, "dwlr_daily_depth_abs.csv")
MRC = os.path.join(QC, "wtf_mrc", "mrc_parameters.csv")
YRS = os.path.join(QC, "wtf_mrc", "recharge_station_years.csv")
DIAG = os.path.join(QC, "wtf_mrc", "diagnostics", "diag_station.csv")
OUT = os.path.join(QC, "audit")
CODES, CODE_TOL, CODE_MIN = [99.0, 99.9, 999.0], 0.5, 90.0
MAX_DEPTH = 60.0
ZERO_M, ISO_M = 1.0, 3.0
FLAG_SHARE, FLAG_CHANGE = 0.20, 0.50
WIN, MIN_COVER, GAP_FILL, SMOOTH = ((6, 1), (10, 31)), 0.8, 3, 3
SY = {"GEC_low": 0.01, "GEC_high": 0.03, "NAQUIM_shallow": 0.04}
N_BOOT, SEED = 2000, 20261013
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(SEED)
for f in (DAILY, MRC, YRS, DIAG):
    if not os.path.exists(f):
        sys.exit(f"[STOP] not found: {f}")
dd = pd.read_csv(DAILY, parse_dates=["T"])
M = pd.read_csv(MRC)
M = M[(M.MRC == "PRIMARY") & (M.note == "ok")].set_index("SID")
Y = pd.read_csv(YRS)
Y = Y[Y.MRC == "PRIMARY"]
D = pd.read_csv(DIAG).drop_duplicates("SID").set_index("SID")


def gross_c(dep, a, b, sig):
    """exact rules of wtf_mrc_recharge.py; dep = daily depth Series (DatetimeIndex). Returns {year: (gc, cover)}"""
    full = pd.date_range(dep.index.min(), dep.index.max(), freq="D")
    h = -dep.reindex(full)
    h = h.interpolate(limit=GAP_FILL, limit_area="inside")
    s = h.rolling(SMOOTH, center=True, min_periods=SMOOTH).mean()
    s[h.isna()] = np.nan
    rate = s.shift(-1) - s
    f = np.minimum(0.0, a + b * s)
    dpos = (rate - f).clip(lower=0)
    eps, bias = max(2 * sig, 0.005), sig / np.sqrt(2 * np.pi)
    out = {}
    for y in sorted(set(full.year)):
        w = (full >= pd.Timestamp(y, *WIN[0])) & (full <= pd.Timestamp(y, *WIN[1]))
        if w.sum() == 0:
            continue
        nwin = int(pd.Timestamp(y, *WIN[1]).dayofyear - pd.Timestamp(y, *WIN[0]).dayofyear + 1)
        x = dpos[w]
        cover = float(x.notna().sum()) / nwin
        g = float(x.sum(min_count=1)) if x.notna().any() else np.nan
        out[y] = (max(g - bias * int((x <= eps).sum()), 0) if np.isfinite(g) else np.nan, cover)
    return out


def flags(dep):
    """returns boolean Series (True = remove) and per-rule counts"""
    q1 = dep.apply(lambda v: any(abs(v - c) <= CODE_TOL for c in CODES)) | (dep >= CODE_MIN)
    q2 = (dep > MAX_DEPTH) & ~q1
    # Q3 (replaces the old step-and-return rule, which also removed real days between two nearby spikes):
    #   Q3a ZERO     depth < ZERO_M while the station's centred 15-day median (other days) is > 3 m  -> telemetry zero code
    #   Q3b ISOLATED a single day > ISO_M away from the 15-day median while BOTH neighbours are within 1 m of it
    #                (daily median of a day mixing code and real 6-hourly readings); a real rise lasts > 1 day
    x = dep.where(~(q1 | q2))
    med = x.rolling(15, center=True, min_periods=5).median()
    q3a = (x < ZERO_M) & (med > 3.0)
    dev = (x - med).abs()
    near_prev = (x.shift(1) - med).abs() <= 1.0
    near_next = (x.shift(-1) - med).abs() <= 1.0
    q3b = (dev > ISO_M) & near_prev & near_next & ~q3a
    q3 = (q3a | q3b).fillna(False)
    bad = q1 | q2 | q3
    return bad, int(q1.sum()), int(q2.sum()), int(q3.sum())


def ci(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 3:
        return f"{np.median(x) if len(x) else np.nan:.3f} [n={len(x)}]"
    b = np.median(rng.choice(x, (N_BOOT, len(x))), axis=1)
    lo, hi = np.percentile(b, [2.5, 97.5])
    return f"{np.median(x):.3f} [{lo:.3f}, {hi:.3f}] n={len(x)}"


# ======================================================================================== per station
ROWS, YROWS = [], []
for sid, g in dd[dd.SID.isin(M.index)].groupby("SID"):
    a, b, sig = M.loc[sid, ["a", "b", "sigma"]].astype(float)
    nd = g.T.duplicated(keep=False)
    dup_diff = int((g[nd].groupby("T").DEPTH_ABS.agg(lambda v: v.max() - v.min()) > 0.5).sum()) if nd.any() else 0
    dep = g.groupby("T").DEPTH_ABS.median().sort_index()
    bad, n1, n2, n3 = flags(dep)
    orig = gross_c(dep, a, b, sig)
    clean = gross_c(dep[~bad], a, b, sig)
    jj = dep.index.month.isin([6, 7, 8, 9, 10])
    for y, (gc, cov) in orig.items():
        gc2, cov2 = clean.get(y, (np.nan, 0))
        YROWS.append({"SID": sid, "YEAR": y, "GC_recalc": gc, "cover": cov, "GC_clean": gc2, "cover_clean": cov2})
    ROWS.append({"SID": sid, "n_days": len(dep), "Q1_code": n1, "Q2_deep": n2, "Q3_step": n3, "dup_dates_diff": dup_diff,
                 "share_JJASO_removed": float(bad[jj].mean()) if jj.any() else np.nan,
                 "depth_min": dep.min(), "depth_max": dep.max(), "depth_max_clean": dep[~bad].max()})
A = pd.DataFrame(ROWS).set_index("SID")
YA = pd.DataFrame(YROWS)
chk = Y.merge(YA, on=["SID", "YEAR"], how="inner")
chk["diff_mm"] = 1000 * (chk.GROSS_C_m - chk.GC_recalc)
FULLK = chk[chk.FULL.astype(bool)]
st_o = FULLK.groupby("SID").GROSS_C_m.median()
st_r = FULLK.groupby("SID").GC_recalc.median()
cl = YA[YA.cover_clean >= MIN_COVER].groupby("SID").GC_clean.median()
A = A.join(D[[c for c in ("DIST", "CLASS", "RAIN", "GROSS_C", "SY100") if c in D.columns]])
A["GC_stored"] = st_o; A["GC_recalc"] = st_r; A["GC_clean"] = cl
A = A.join(Y[Y.FULL.astype(bool)].groupby("SID").agg(BIANN=("BIANN_m", "median"), PEAKLOW=("PEAKLOW_m", "median")))
A["change_frac"] = (A.GC_clean - A.GC_recalc) / A.GC_recalc
A["FLAG"] = (A.share_JJASO_removed > FLAG_SHARE) | (A.change_frac.abs() > FLAG_CHANGE) | A.GC_clean.isna()
A["FLAG_reason"] = np.select([A.GC_clean.isna(), A.share_JJASO_removed > FLAG_SHARE, A.change_frac.abs() > FLAG_CHANGE],
                             ["no FULL year left after day fixes", f"> {int(100 * FLAG_SHARE)}% Jun-Oct days corrected",
                              f"GROSS_C changed > {int(100 * FLAG_CHANGE)}%"], "")
A.round(4).to_csv(os.path.join(OUT, "audit_station.csv"))
# corrected daily files (days removed only; no station removed) for a full re-run of the pipeline:
#   _Q12  = unambiguous codes / impossible depths only;  _Q123 = also step-and-return blocks (Q3, less certain)
def q12(dep):
    q1 = dep.apply(lambda v: any(abs(v - c) <= CODE_TOL for c in CODES)) | (dep >= CODE_MIN)
    return q1 | (dep > MAX_DEPTH)
K12, K123 = [], []
for sid, g in dd.groupby("SID"):
    dep = g.groupby("T").DEPTH_ABS.median().sort_index()
    bad, *_ = flags(dep)
    b12 = q12(dep)
    K12.append(pd.DataFrame({"SID": sid, "T": dep.index[~b12.values], "DEPTH_ABS": dep.values[~b12.values]}))
    K123.append(pd.DataFrame({"SID": sid, "T": dep.index[~bad.values], "DEPTH_ABS": dep.values[~bad.values]}))
pd.concat(K12).to_csv(os.path.join(OUT, "dwlr_daily_depth_abs_Q12.csv"), index=False)
pd.concat(K123).to_csv(os.path.join(OUT, "dwlr_daily_depth_abs_Q123.csv"), index=False)
YA.round(4).to_csv(os.path.join(OUT, "audit_station_years.csv"), index=False)

# ======================================================================================== VERDICT
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
print("=" * 100)
print("VERDICT  recharge_audit")
print("=" * 100)
print("STEP A  trace check (recomputed vs stored GROSS_C, station-years):")
print(f"  n {len(chk)} | |diff| max {chk.diff_mm.abs().max():.3f} mm | median {chk.diff_mm.abs().median():.4f} mm "
      f"-> {'REPRODUCED' if chk.diff_mm.abs().max() < 1 else 'NOT REPRODUCED - stop and send this block'}")
print("\nSTEP B  day-level QC (days removed):")
print(f"  Q1 code-like (99/99.9/999, >= 90 m): {int(A.Q1_code.sum())} days at {int((A.Q1_code > 0).sum())} stations")
print(f"  Q2 depth > {MAX_DEPTH:g} m: {int(A.Q2_deep.sum())} days at {int((A.Q2_deep > 0).sum())} stations")
print(f"  Q3 zero-code / isolated one-day excursions: {int(A.Q3_step.sum())} days at {int((A.Q3_step > 0).sum())} stations")
print(f"  Q4 duplicate dates with values differing > 0.5 m: {int(A.dup_dates_diff.sum())} at {int((A.dup_dates_diff > 0).sum())} stations")
print("\nSTEP C  extremes (top 12 by stored GROSS_C) and negative BIANN:")
cols = ["DIST", "CLASS", "GC_stored", "GC_clean", "change_frac", "Q1_code", "Q2_deep", "Q3_step", "depth_max",
        "depth_max_clean", "PEAKLOW", "BIANN", "FLAG_reason"]
print(A.sort_values("GC_stored", ascending=False)[cols].head(12).round(2).to_string())
neg = A[A.BIANN < 0]
print(f"\n  stations with negative May20-Nov10 rise: {len(neg)}")
if len(neg):
    print(neg[["DIST", "CLASS", "BIANN", "PEAKLOW", "GC_clean", "Q1_code", "Q3_step"]].round(2).to_string())
print(f"\n  stations FLAGGED: {int(A.FLAG.sum())} of {len(A)} | reasons: {A[A.FLAG].FLAG_reason.value_counts().to_dict()}")
print(f"  stations remaining (not flagged, >= 1 FULL year): {int((~A.FLAG & A.GC_clean.notna()).sum())}")

print("\nSTEP D  medians [95% CI] — ORIGINAL | DAY-CORRECTED | CORRECTED, FLAGGED STATIONS EXCLUDED")
K = A[~A.FLAG]
for lab, f in [("GROSS_C per unit Sy (m)", lambda q, c: q[c]),
               ("Sy_100 = rain/GROSS_C", lambda q, c: q.RAIN / 1000 / q[c])]:
    print(f"  {lab}")
    print(f"    original  {ci(f(A, 'GC_recalc'))}\n    corrected {ci(f(A, 'GC_clean'))}\n    excl.flag {ci(f(K, 'GC_clean'))}")
for nm, sy in SY.items():
    print(f"  recharge % of Jun-Oct rain at Sy {sy:.2f} ({nm}), hard rock only:")
    for lab, q, c in (("original", A, "GC_recalc"), ("corrected", A, "GC_clean"), ("excl.flag", K, "GC_clean")):
        h = q[q.CLASS == "HARD_ROCK"]
        print(f"    {lab:<9} median-of-ratios {ci(100 * 1000 * sy * h[c] / h.RAIN)} | ratio-of-medians "
              f"{100 * 1000 * sy * h[c].median() / h.RAIN.median():.1f}")
print("  BIANN / GROSS_C (two definitions, corrected, excl. flagged):")
print(f"    median of station ratios {ci(K.BIANN / K.GC_clean)} | ratio of medians {K.BIANN.median() / K.GC_clean.median():.3f}")
for c, q in K.groupby("CLASS"):
    print(f"  {c:<9} GROSS_C {ci(q.GC_clean)} | Sy_100 {ci(q.RAIN / 1000 / q.GC_clean)}")

# ======================================================================================== STEP E plots
show = list(A.sort_values("GC_stored", ascending=False).head(6).index) + list(A[A.FLAG].index[:6])
show = list(dict.fromkeys(show))[:12]
if show:
    fig, axs = plt.subplots(len(show), 1, figsize=(10, 1.9 * len(show)))
    axs = np.atleast_1d(axs)
    for ax, sid in zip(axs, show):
        dep = dd[dd.SID == sid].groupby("T").DEPTH_ABS.median().sort_index()
        bad, *_ = flags(dep)
        ax.plot(dep.index, dep.values, lw=0.8, color="#6b6a64", label="daily depth (QC file)")
        ax.plot(dep.index[bad], dep.values[bad], "o", ms=2.5, color="#eb6834", label="removed by Q1-Q3")
        for y in sorted(set(dep.index.year)):
            ax.axvspan(pd.Timestamp(y, 6, 1), pd.Timestamp(y, 10, 31), color="#e6e5e0", lw=0, zorder=0)
        ax.invert_yaxis(); ax.set_ylabel("depth (m bgl)", fontsize=7)
        r = A.loc[sid]
        ax.set_title(f"{sid} | {r.CLASS} | GROSS_C stored {r.GC_stored:.1f} -> corrected {r.GC_clean:.1f} m | {r.FLAG_reason}",
                     fontsize=7, loc="left")
        ax.tick_params(labelsize=7)
    axs[0].legend(fontsize=7, loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "audit_hydrographs.png"), dpi=200); plt.close(fig)
print(f"\nOutputs: {OUT}  (audit_station.csv, audit_station_years.csv, audit_hydrographs.png)")
