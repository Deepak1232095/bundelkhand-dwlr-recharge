"""
SUPPORTING INFORMATION: Figures S1-S5 and Tables S1-S4 (Paper 3, WRR)
=====================================================================
Reads outputs you already have; recomputes nothing in the main analysis. Each block runs on its own:
a missing input skips only that figure/table and is reported in the VERDICT.

  Fig S1 / Text S1  seasonal dH (T1 GROSS_C) vs sum of monthly rises (T6 RISE_m)
  Table S1          day-level audit rules Q1, Q2, Q3a, Q3b: days removed and stations affected (recomputed from the
                    raw daily file with the same rules as recharge_audit.py) + flagged-station sensitivity (Table S1b)
  Fig S2 / Table S2 synthetic verification (TableS2_synthetic_verification.csv, TableS2b_design_variants.csv from
                    synthetic_verification.py)
  Fig S3            rain-free rises: monthly share + within-district synchrony vs random-shift null (diag_episodes.csv)
  Fig S4 / Table S3 MODFLOW 6 cross-section (confined_wtf_results.csv from flopy_confined_wtf.py); Table S3 = model setup
  Fig S5 / Table S4 CGWB-assisted map, only for districts with skill > 0 (T9_combined_validation.csv, T9b_..., T6)

Output : <BASE>\\paper3_final\\SI\\  FigS1..S5 (.png 600 dpi + .pdf), TableS1..S4 (.csv)
Run    :  %run si_figures_tables.py      Requires: pandas numpy scipy matplotlib (+ geopandas for Fig S5)

"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

# ======================================================================================== CONFIG
BASE = os.environ.get("TEST_BASE", r"C:\Users\IITI")
FINAL = os.path.join(BASE, "paper3_final")
OUT = os.path.join(FINAL, "SI")
RUN = os.path.join(BASE, "run_Q123", "DWLR_QC")


def first(*paths):
    return next((p for p in paths if os.path.exists(p)), None)


T1P = first(os.path.join(RUN, "final", "T1_station_recharge.csv"), os.path.join(FINAL, "T1_station_recharge.csv"))
T6P = first(os.path.join(FINAL, "T6_monthly_station_year.csv"))
DAILY = first(os.path.join(BASE, "DWLR_QC", "dwlr_daily_depth_abs.csv"))           # raw (unaudited) daily depths
AUDIT = first(os.path.join(BASE, "DWLR_QC", "audit", "audit_station.csv"))
SYN = first(*[os.path.join(d, "TableS2_synthetic_verification.csv") for d in (BASE, FINAL, os.getcwd())])
SYNV = first(*[os.path.join(d, "TableS2b_design_variants.csv") for d in (BASE, FINAL, os.getcwd())])
EPIS = first(os.path.join(RUN, "wtf_mrc", "diagnostics", "diag_episodes.csv"))
MF = first(os.path.join(BASE, "virtual_confined", "confined_wtf_results.csv"))
T9 = first(os.path.join(FINAL, "T9_combined_validation.csv"))
T9B = first(os.path.join(FINAL, "T9b_cgwb_wells_recharge_equiv.csv"))
SIGN = first(os.path.join(RUN, "dwlr_sign_smoothness.csv"))
DIST_SHP = os.environ.get("TEST_DIST", r"E:\up_bundel\up_bundel_boxes\up_5dist.shp")
DIST_COL = "NAME_2"

SY = {"HARD_ROCK": 0.03, "ALLUVIUM": 0.06}
CODES, CODE_TOL, CODE_MIN, MAX_DEPTH = [99.0, 99.9, 999.0], 0.5, 90.0, 60.0     # as recharge_audit.py
SYNC_DAYS, N_SHIFT, SEED = 2, 200, 20261013                                     # as wtf_diagnostics.py
CELL_M, POWER, EDGE_KM, MAX_KM, UTM = 1000.0, 2.0, 10.0, 15.0, "EPSG:32644"    # as combined_maps.py
W1, W2 = 9.5 / 2.54, 19.0 / 2.54                                                # AGU one / two column (in)
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(SEED)

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                     "font.size": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "legend.fontsize": 7, "legend.frameon": False, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.linewidth": 0.6, "pdf.fonttype": 42,
                     "savefig.bbox": "tight", "savefig.dpi": 600})
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5e0"
COL = {"ALLUVIUM": "#2a78d6", "HARD_ROCK": "#eb6834"}
MRK = {"ALLUVIUM": "o", "HARD_ROCK": "^"}
LAB = {"ALLUVIUM": "Alluvium", "HARD_ROCK": "Hard rock"}
LOG = []


def say(s=""):
    LOG.append(s)


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"))
    plt.close(fig)


def tag(ax, t):
    ax.text(-0.02, 1.04, t, transform=ax.transAxes, fontsize=8.5, fontweight="bold", va="bottom", ha="right")


T1 = None
if T1P:
    T1 = pd.read_csv(T1P)
    T1["DIST"] = T1.DIST.astype(str).str.upper().str.strip()
    T1 = T1.drop_duplicates("SID").set_index("SID")

# ======================================================================================== Fig S1
say("FIG S1  seasonal dH vs sum of monthly rises")
if T1 is None or T6P is None:
    say("  SKIPPED - need T1_station_recharge.csv and T6_monthly_station_year.csv")
else:
    D6 = pd.read_csv(T6P)
    m = D6.groupby(["SID", "MONTH"]).RISE_m.median().reset_index()
    ms = m.groupby("SID").agg(MSUM=("RISE_m", "sum"), n=("MONTH", "nunique"))
    S1 = T1[["CLASS", "GROSS_C"]].join(ms, how="inner")
    S1 = S1[(S1.n == 5) & np.isfinite(S1.GROSS_C) & (S1.GROSS_C > 0)]
    fig, ax = plt.subplots(figsize=(W1, W1))
    lim = 1.05 * max(S1.GROSS_C.max(), S1.MSUM.max())
    ax.plot([0, lim], [0, lim], color=MUTED, lw=0.8, ls="--", zorder=1)
    for c in ("ALLUVIUM", "HARD_ROCK"):
        q = S1[S1.CLASS == c]
        ax.scatter(q.GROSS_C, q.MSUM, s=14, marker=MRK[c], color=COL[c], edgecolor="white", lw=0.3,
                   label=f"{LAB[c]} (n = {len(q)})", zorder=3)
    oth = S1[~S1.CLASS.isin(COL)]
    if len(oth):
        ax.scatter(oth.GROSS_C, oth.MSUM, s=12, marker="s", color=MUTED, label=f"Unclassified (n = {len(oth)})")
    r = S1.MSUM / S1.GROSS_C
    rho = stats.spearmanr(S1.GROSS_C, S1.MSUM)[0]
    ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_aspect("equal")
    ax.set_xlabel("Seasonal rise per unit Sy, ΔH (m)"); ax.set_ylabel("Sum of monthly rises per unit Sy (m)")
    ax.text(0.04, 0.96, f"median ratio {r.median():.2f}\nSpearman ρ = {rho:.2f}", transform=ax.transAxes,
            va="top", fontsize=7, color=INK)
    ax.legend(loc="lower right")
    save(fig, "FigS1_seasonal_vs_monthly")
    say(f"  stations {len(S1)} | ratio monthly-sum/seasonal median {r.median():.3f} (IQR {r.quantile(.25):.3f}-{r.quantile(.75):.3f})"
        f" | Spearman {rho:.2f}")
    for c, q in S1.groupby("CLASS"):
        say(f"    {c:<10} n={len(q):2d} ratio median {(q.MSUM / q.GROSS_C).median():.3f}")

# ======================================================================================== Table S1
say("\nTABLE S1  audit rules (recomputed from the raw daily file, stations of T1)")
if DAILY is None:
    say("  SKIPPED - need DWLR_QC\\dwlr_daily_depth_abs.csv")
else:
    dd = pd.read_csv(DAILY, parse_dates=["T"])
    if T1 is not None:
        dd = dd[dd.SID.isin(T1.index)]
    cnt = {k: [0, set()] for k in ("Q1", "Q2", "Q3a", "Q3b")}
    ndays_all = 0
    for sid, g in dd.groupby("SID"):
        dep = g.groupby("T").DEPTH_ABS.median().sort_index()
        ndays_all += len(dep)
        q1 = dep.apply(lambda v: any(abs(v - c) <= CODE_TOL for c in CODES)) | (dep >= CODE_MIN)
        q2 = (dep > MAX_DEPTH) & ~q1
        x = dep.where(~(q1 | q2))
        med = x.rolling(15, center=True, min_periods=5).median()
        q3a = ((x < 1.0) & (med > 3.0)).fillna(False)
        dev = (x - med).abs()
        q3b = ((dev > 3.0) & ((x.shift(1) - med).abs() <= 1.0) & ((x.shift(-1) - med).abs() <= 1.0) & ~q3a).fillna(False)
        for k, v in (("Q1", q1), ("Q2", q2), ("Q3a", q3a), ("Q3b", q3b)):
            if v.sum():
                cnt[k][0] += int(v.sum()); cnt[k][1].add(sid)
    crit = {"Q1": "Within 0.5 m of 99, 99.9 or 999 m, or ≥ 90 m", "Q2": "Depth > 60 m",
            "Q3a": "Depth < 1 m while centred 15-day median > 3 m",
            "Q3b": "Single-day excursion > 3 m from 15-day median, both neighbours within 1 m of it"}
    TS1 = pd.DataFrame([{"Rule": k, "Criterion": crit[k], "Days removed": v[0],
                         "% of station-days": round(100 * v[0] / max(ndays_all, 1), 2),
                         "Stations affected": len(v[1])} for k, v in cnt.items()])
    TS1.to_csv(os.path.join(OUT, "TableS1_audit_rules.csv"), index=False, encoding="utf-8-sig")
    say(f"  stations {dd.SID.nunique()} | station-days {ndays_all}")
    say("  " + TS1.drop(columns="Criterion").to_string(index=False).replace("\n", "\n  "))
    if AUDIT and T1 is not None:
        A = pd.read_csv(AUDIT).drop_duplicates("SID").set_index("SID")
        fl = A.FLAG.astype(str).str.lower().isin(["true", "1"]) if "FLAG" in A.columns else pd.Series(False, index=A.index)
        flagged = set(A.index[fl]) & set(T1.index)
        rows = []
        for c in ("ALLUVIUM", "HARD_ROCK"):
            q = T1[T1.CLASS == c]
            k_ = q[~q.index.isin(flagged)]
            rows.append({"Class": LAB[c], "n all": len(q), "ΔH median all (m)": round(q.GROSS_C.median(), 2),
                         "n without flagged": len(k_), "ΔH median without flagged (m)": round(k_.GROSS_C.median(), 2)})
        TS1b = pd.DataFrame(rows)
        TS1b.to_csv(os.path.join(OUT, "TableS1b_flagged_sensitivity.csv"), index=False, encoding="utf-8-sig")
        reasons = A.loc[list(flagged), "FLAG_reason"].value_counts().to_dict() if flagged and "FLAG_reason" in A.columns else {}
        say(f"  flagged stations in T1: {len(flagged)} {reasons}")
        say("  " + TS1b.to_string(index=False).replace("\n", "\n  "))
    else:
        say("  Table S1b skipped - need DWLR_QC\\audit\\audit_station.csv")

# ======================================================================================== Fig S2 / Table S2
say("\nFIG S2 / TABLE S2  synthetic verification")
if SYN is None:
    say("  SKIPPED - run synthetic_verification.py first (writes TableS2_synthetic_verification.csv)")
else:
    S2 = pd.read_csv(SYN)
    PUMP = [("none", "No pumping"), ("rabi", "Rabi pumping"), ("rabi+kharif", "Rabi + monsoon pumping")]
    fig, axs = plt.subplots(1, 3, figsize=(W2, W2 * 0.33), sharey=True)
    for ax, (p, pl), t in zip(axs, PUMP, "abc"):
        for mrc, ls in (("Aug-Nov", "-"), ("Aug-Feb", "--")):
            for rec, c in (("linear", "#2b5c8a"), ("nonlinear", "#a0522d")):
                q = S2[(S2.MRC == mrc) & (S2.recession == rec) & (S2.pumping == p)].sort_values("noise_cm")
                if q.empty:
                    continue
                ax.errorbar(q.noise_cm + (0.06 if rec == "nonlinear" else -0.06), q["mean"], yerr=q.sd, color=c,
                            ls=ls, marker="o" if mrc == "Aug-Nov" else "s", ms=3, lw=0.9, capsize=1.5, elinewidth=0.6,
                            mfc=c if mrc == "Aug-Nov" else "white")
        ax.axhline(1, color=MUTED, lw=0.7, ls=":")
        ax.set_title(pl, fontsize=7.5, loc="left"); ax.set_xlabel("Level noise SD (cm)"); ax.set_xticks([0, 1, 2, 3])
        ax.grid(axis="y", color=GRID, lw=0.5); tag(ax, t)
    axs[0].set_ylabel("Estimated / planted rise")
    h = [Line2D([], [], color="#2b5c8a", marker="o", ms=3, label="Linear recession"),
         Line2D([], [], color="#a0522d", marker="o", ms=3, label="Nonlinear recession"),
         Line2D([], [], color=INK, ls="-", label="Aug–Nov MRC (final)"),
         Line2D([], [], color=INK, ls="--", marker="s", mfc="white", ms=3, label="Aug–Feb MRC")]
    fig.legend(handles=h, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.12))
    save(fig, "FigS2_synthetic_verification")
    rows = []
    for mrc in ("Aug-Nov", "Aug-Feb"):
        r_ = {"MRC window": mrc}
        for p, pl in PUMP:
            q = S2[(S2.MRC == mrc) & (S2.pumping == p)]
            r_[pl] = f"{q['mean'].min():.2f}–{q['mean'].max():.2f}" if len(q) else "–"
        r_["MRC rejected (of all realizations)"] = int(S2[S2.MRC == mrc].rejected_MRC.sum())
        rows.append(r_)
    TS2 = pd.DataFrame(rows)
    TS2.to_csv(os.path.join(OUT, "TableS2_synthetic_summary.csv"), index=False, encoding="utf-8-sig")
    say("  mean ratio range over recession types and noise levels:")
    say("  " + TS2.to_string(index=False).replace("\n", "\n  "))
    if SYNV:
        V = pd.read_csv(SYNV)
        VS = V.groupby("variant").agg(mean_min=("mean", "min"), mean_max=("mean", "max"),
                                      station_years=("count", "sum")).round(2).reset_index()
        VS.to_csv(os.path.join(OUT, "TableS2b_design_variants_summary.csv"), index=False, encoding="utf-8-sig")
        say("  design variants (Aug-Nov MRC, rabi pumping; station_years = valid estimates, lower = more MRC failures):")
        say("  " + VS.to_string(index=False).replace("\n", "\n  "))

# ======================================================================================== Fig S3
say("\nFIG S3  rain-free rises")
if EPIS is None:
    say("  SKIPPED - need run_Q123\\DWLR_QC\\wtf_mrc\\diagnostics\\diag_episodes.csv (run wtf_diagnostics.py)")
else:
    E = pd.read_csv(EPIS, parse_dates=["start", "end"])
    for c in ("FULLYR", "RAIN_FREE", "ABOVE_NOISE"):
        E[c] = E[c].astype(str).str.lower().isin(["true", "1"])
    E["MONTH"] = E.start.dt.month
    EF = E[E.FULLYR]
    RF = EF[EF.RAIN_FREE & EF.ABOVE_NOISE].copy()
    share_rf = EF.RECH_m[EF.RAIN_FREE].sum() / EF.RECH_m.sum()
    mon = RF.groupby("MONTH").RECH_m.sum().reindex([6, 7, 8, 9, 10], fill_value=0)
    mon = mon / mon.sum() if mon.sum() > 0 else mon

    def sync_share(df):
        hits = []
        for _, r in df.iterrows():
            oth = df[(df.DIST == r.DIST) & (df.YEAR == r.YEAR) & (df.SID != r.SID)]
            hits.append(bool(len(oth)) and bool((np.abs((oth.start - r.start).dt.days) <= SYNC_DAYS).any()))
        return np.mean(hits) if hits else np.nan

    obs = sync_share(RF) if len(RF) > 1 else np.nan
    null = []
    if len(RF) > 1:
        for _ in range(N_SHIFT):
            sh = RF.copy()
            off = {s: int(rng.integers(-60, 61)) for s in sh.SID.unique()}
            sh["start"] = sh.start + pd.to_timedelta(sh.SID.map(off), unit="D")
            null.append(sync_share(sh))
    null = np.array(null)
    p = (np.sum(null >= obs) + 1) / (len(null) + 1) if len(null) else np.nan
    fig, axs = plt.subplots(1, 2, figsize=(W2, W2 * 0.33))
    axs[0].bar(["Jun", "Jul", "Aug", "Sep", "Oct"], 100 * mon.values, color="#2b5c8a", width=0.6)
    axs[0].set_ylabel("Share of rain-free rise (%)"); axs[0].grid(axis="y", color=GRID, lw=0.5); tag(axs[0], "a")
    if len(null):
        axs[1].hist(100 * null, bins=20, color="#cfcfcf", edgecolor="white", lw=0.4, label="Random time shifts")
        axs[1].axvline(100 * obs, color="#a0522d", lw=1.4, label=f"Observed ({100 * obs:.0f}%)")
        axs[1].text(0.03, 0.72, f"p = {p:.3f}", transform=axs[1].transAxes, ha="left", va="top", fontsize=7)
        axs[1].legend(loc="upper left")
    axs[1].set_xlabel("Rain-free episodes starting within ±2 d of another\nstation in the same district (%)")
    axs[1].set_ylabel("Count"); tag(axs[1], "b")
    save(fig, "FigS3_rain_free_rises")
    say(f"  rain-free share of episode rise (all rain-free, FULL years) {share_rf:.3f} | above-noise rain-free episodes {len(RF)}")
    say(f"  monthly share of above-noise rain-free rise: {dict(zip(['Jun','Jul','Aug','Sep','Oct'], [round(float(v), 2) for v in mon.values]))}")
    if len(null):
        say(f"  synchrony observed {obs:.2f} | null median {np.median(null):.2f} [P95 {np.percentile(null, 95):.2f}] | p = {p:.3f}"
            f"  (re-drawn null; may differ slightly from wtf_diagnostics.py)")

# ======================================================================================== Fig S4 / Table S3
say("\nFIG S4 / TABLE S3  MODFLOW 6 cross-section")
TS3 = pd.DataFrame([
    ("Grid", "1 row × 50 columns, Δx = 40 m (2 km); 3 layers"),
    ("Layer 1 (phreatic sand)", "25 m; K = 10 m/d (Kv = 1 m/d); Sy = 0.10"),
    ("Layer 2 (clay)", "10 m; Kh = 0.01 m/d; Kv = 0.001, 0.01 or 0.1 m/d"),
    ("Layer 3 (confined sand)", "20 m; K = 40 m/d (Kv = 4 m/d); Ss = 2e-5 1/m"),
    ("Boundaries", "River at last column (stage 140 m, conductance 500 m2/d); no-flow divide at column 1"),
    ("Recharge", "15% of Jun–Oct rain above 5 mm/d (IMD, Jalaun, 2015–2024)"),
    ("Pumping (layer 3)", "7 wells; 0, 0.5 or 1 × annual recharge volume; 70% rabi, 30% monsoon dry days"),
    ("Observation points", "Pumping cell, 80 m and 280 m from wells; layers 1 and 3"),
    ("Analysis", "MRC-WTF rules of the field data; 1 cm Gaussian noise")], columns=["Item", "Value"])
TS3.to_csv(os.path.join(OUT, "TableS3_modflow_setup.csv"), index=False, encoding="utf-8-sig")
say("  Table S3 written (setup constants of flopy_confined_wtf.py)")
if MF is None:
    say("  Fig S4 SKIPPED - need virtual_confined\\confined_wtf_results.csv (run flopy_confined_wtf.py)")
else:
    R4 = pd.read_csv(MF)
    R4 = R4[R4.note == "ok"].copy()
    R4["layer"] = R4.well.str.split("_").str[0]
    R4["pos"] = R4.well.str.split("_", n=1).str[1]
    R4["true_rise"] = R4.GROSS_C / R4.OVEREST
    true_rise = float(R4.true_rise.median())
    POS = [("pumpcell", "pumping cell"), ("between", "80 m"), ("far", "280 m")]
    kv = sorted(R4.Kv_clay.unique()); pumps = sorted(R4.pump.unique())
    fig, axs = plt.subplots(1, 2, figsize=(W2, W2 * 0.36))
    ax = axs[0]; w = 0.8 / (2 * len(kv)); x0 = np.arange(len(pumps))
    shades = ["#9ecae1", "#4292c6", "#08519c"]
    for i, k in enumerate(kv):
        for j, (lay, hatch) in enumerate((("L1", ""), ("L3", "////"))):
            q = R4[(R4.Kv_clay == k) & (R4.layer == lay) & (R4.pos == "between")].set_index("pump").reindex(pumps)
            ax.bar(x0 + (2 * i + j - len(kv) + 0.5) * w, q.GROSS_C, w, color=shades[i % 3], hatch=hatch,
                   edgecolor="white", lw=0.4)
    ax.axhline(true_rise, color="#a0522d", ls="--", lw=0.9)
    ax.text(-0.45, true_rise * 1.02, "true rise", color="#a0522d", fontsize=6.5, va="bottom", ha="left")
    ax.set_xticks(x0); ax.set_xticklabels([f"{p:g}" for p in pumps])
    ax.set_xlabel("Pumping (× annual recharge volume)"); ax.set_ylabel("MRC-WTF rise per unit Sy (m)")
    ax.grid(axis="y", color=GRID, lw=0.5); tag(ax, "a")
    hh = [plt.Rectangle((0, 0), 1, 1, color=shades[i % 3], label=f"clay Kv {k:g} m/d") for i, k in enumerate(kv)]
    hh += [plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=INK, lw=0.5, label="water table (layer 1)"),
           plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=INK, hatch="////", lw=0.5, label="confined (layer 3)")]
    ax.legend(handles=hh, loc="upper left", fontsize=6.5, bbox_to_anchor=(0.0, 1.0))
    ax = axs[1]
    for (pos, pl), mk in zip(POS, ("o", "s", "^")):
        q = R4[(R4.layer == "L3") & (R4.pos == pos)].groupby("pump").OVEREST.median().reindex(pumps)
        ax.plot(pumps, q.values, marker=mk, ms=3.5, lw=0.9, color=INK, label=f"confined, {pl}")
        q1 = R4[(R4.layer == "L1") & (R4.pos == pos)].groupby("pump").OVEREST.median().reindex(pumps)
        ax.plot(pumps, q1.values, marker=mk, ms=3.5, lw=0.9, color=MUTED, ls="--", mfc="white",
                label=f"water table, {pl}")
    ax.axhline(1, color=MUTED, lw=0.7, ls=":")
    ax.set_xlabel("Pumping (× annual recharge volume)"); ax.set_ylabel("Estimated / true recharge (median of Kv)")
    ax.grid(axis="y", color=GRID, lw=0.5); ax.legend(fontsize=6.3, ncol=2, loc="upper left"); tag(ax, "b")
    save(fig, "FigS4_modflow_cross_section")
    pv = R4[R4.pos == "between"].pivot_table(index=["Kv_clay", "pump"], columns="layer", values="GROSS_C")
    if {"L1", "L3"} <= set(pv.columns):
        pv["ratio_L3_L1"] = pv.L3 / pv.L1
    say(f"  true per-unit-Sy rise {true_rise:.2f} m | rise at 80 m from wells by run:")
    say("  " + pv.round(2).to_string().replace("\n", "\n  "))
    ov = R4[R4.layer == "L3"].groupby("pump").OVEREST.agg(["min", "median", "max"]).round(2)
    say("  confined well: estimated / true recharge by pumping (over Kv and positions):")
    say("  " + ov.to_string().replace("\n", "\n  "))

# ======================================================================================== Table S4 / Fig S5
say("\nTABLE S4 / FIG S5  CGWB-assisted reconstruction")
if T9 is None:
    say("  SKIPPED - need paper3_final\\T9_combined_validation.csv (run combined_maps.py)")
else:
    t9 = pd.read_csv(T9)
    TS4 = pd.DataFrame({"District": t9.district.str.title(), "CGWB wells (n)": t9.n_CGWB, "DWLR stations (n)": t9.n_DWLR,
                        "k used": t9.k_used, "MAE CGWB surface (mm)": t9.MAE_CGWB_surface,
                        "MAE district median (mm)": t9.MAE_district_median, "Skill": t9.skill_CGWB_predicts_DWLR,
                        "CGWB points used": np.where(t9.skill_CGWB_predicts_DWLR > 0, "Yes", "No")})
    TS4.to_csv(os.path.join(OUT, "TableS4_cgwb_skill.csv"), index=False, encoding="utf-8-sig")
    say("  " + TS4.to_string(index=False).replace("\n", "\n  "))
    use = list(t9.district[t9.skill_CGWB_predicts_DWLR > 0].str.upper())
    need = [T9B, T6P, SIGN] + ([DIST_SHP] if os.path.exists(DIST_SHP) else [None])
    if not use:
        say("  Fig S5 SKIPPED - no district with skill > 0")
    elif any(v is None for v in need):
        say("  Fig S5 SKIPPED - need T9b_cgwb_wells_recharge_equiv.csv, T6, dwlr_sign_smoothness.csv and the district shapefile")
    else:
        try:
            import geopandas as gpd
            D6 = pd.read_csv(T6P)
            SM = D6.groupby(["SID", "DIST", "CLASS", "MONTH"]).R_mm.median().reset_index()
            Sd = SM.groupby(["SID", "DIST", "CLASS"]).agg(R=("R_mm", "sum"), n=("MONTH", "nunique")).reset_index()
            Sd = Sd[Sd.n == 5].set_index("SID")
            sg = pd.read_csv(SIGN).drop_duplicates("SID").set_index("SID")
            Sd = Sd.join(sg[["lat", "lon"]], how="inner"); Sd["DIST"] = Sd.DIST.astype(str).str.upper()
            Pd = gpd.GeoDataFrame(Sd, geometry=gpd.points_from_xy(Sd.lon, Sd.lat), crs="EPSG:4326").to_crs(UTM)
            cb = pd.read_csv(T9B); cb["DIST"] = cb.DIST.astype(str).str.upper(); cb = cb[cb.R.notna()]
            Pc = gpd.GeoDataFrame(cb, geometry=gpd.points_from_xy(cb.LON, cb.LAT), crs="EPSG:4326").to_crs(UTM)
            G = gpd.read_file(DIST_SHP).to_crs(UTM); G["DIST"] = G[DIST_COL].astype(str).str.upper()
            allv = np.r_[Pd.R.values, Pc.R.values]
            vmin, vmax = np.percentile(allv, [5, 95])
            cmap = plt.get_cmap("RdYlBu")

            def surface(pts, poly):
                xmin, ymin, xmax, ymax = poly.bounds
                xs = np.arange(xmin, xmax + CELL_M, CELL_M); ys = np.arange(ymin, ymax + CELL_M, CELL_M)
                X, Yg = np.meshgrid(xs, ys)
                px, py, pv = pts.geometry.x.values, pts.geometry.y.values, pts.R.values
                dist = np.maximum(np.hypot(X[..., None] - px, Yg[..., None] - py), 1.0)
                wgt = dist ** -POWER
                Z = (wgt * pv).sum(-1) / wgt.sum(-1)
                Z[dist.min(-1) > MAX_KM * 1000] = np.nan
                inside = gpd.GeoSeries(gpd.points_from_xy(X.ravel(), Yg.ravel()), crs=UTM).within(poly).values.reshape(X.shape)
                Z[~inside] = np.nan
                return xs, ys, Z

            fig, axs = plt.subplots(len(use), 2, figsize=(W2, W2 * 0.45 * len(use)), squeeze=False)
            for i, d in enumerate(use):
                poly = G[G.DIST == d].geometry.union_all()
                near = lambda P: P[(P.DIST == d) | (P.geometry.distance(poly) <= EDGE_KM * 1000)]
                pd_, pc_ = near(Pd), near(Pc)
                for j, (pts, title) in enumerate(((pd_, "DWLR only"), (pd.concat([pd_, pc_]), "DWLR + CGWB"))):
                    ax = axs[i, j]
                    xs, ys, Z = surface(gpd.GeoDataFrame(pts, geometry="geometry", crs=UTM), poly)
                    ax.pcolormesh(xs, ys, Z, cmap=cmap, vmin=vmin, vmax=vmax, shading="nearest")
                    G[G.DIST == d].boundary.plot(ax=ax, color=INK, lw=0.6)
                    for c, mk in MRK.items():
                        q = pd_[(pd_.CLASS == c) & (pd_.DIST == d)]
                        ax.scatter(q.geometry.x, q.geometry.y, c=q.R, cmap=cmap, vmin=vmin, vmax=vmax, marker=mk,
                                   s=22, edgecolor=INK, lw=0.5, zorder=4)
                    if j == 1:
                        q = pc_[pc_.DIST == d]
                        ax.scatter(q.geometry.x, q.geometry.y, c=q.R, cmap=cmap, vmin=vmin, vmax=vmax, marker="s",
                                   s=9, edgecolor=MUTED, lw=0.4, zorder=3)
                    x0_, y0_, x1_, y1_ = poly.bounds
                    ax.plot([x1_ - 25000, x1_], [y0_ + 2000] * 2, color=INK, lw=1.5)
                    ax.text(x1_ - 12500, y0_ + 3500, "25 km", ha="center", fontsize=6.5)
                    ax.annotate("N", xy=(x0_ + 3000, y1_ - 1500), xytext=(x0_ + 3000, y1_ - 9000), ha="center",
                                fontsize=7, arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.8))
                    ax.set_title(f"{d.title()}: {title}", fontsize=7.5, loc="left")
                    ax.set_aspect("equal"); ax.set_axis_off(); tag(ax, "abcdefgh"[2 * i + j])
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin, vmax))
            cbar = fig.colorbar(sm, ax=axs, orientation="horizontal", fraction=0.04, pad=0.04, shrink=0.6)
            cbar.set_label("Jun–Oct recharge, sum of monthly values (mm)"); cbar.outline.set_visible(False)
            h = [Line2D([], [], marker="o", ls="", mfc="white", mec=INK, label="DWLR, alluvium"),
                 Line2D([], [], marker="^", ls="", mfc="white", mec=INK, label="DWLR, hard rock"),
                 Line2D([], [], marker="s", ls="", ms=4, mfc="white", mec=MUTED, label="CGWB recharge-equivalent")]
            fig.legend(handles=h, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.0))
            save(fig, "FigS5_cgwb_assisted_map")
            say(f"  Fig S5 drawn for {use} | colour limits {vmin:.0f}-{vmax:.0f} mm (5th-95th pct of DWLR + CGWB)")
        except ImportError:
            say("  Fig S5 SKIPPED - geopandas not installed")

# ======================================================================================== VERDICT
print("=" * 90); print("VERDICT  si_figures_tables"); print("=" * 90)
print("\n".join(LOG))
print(f"\nOutputs: {OUT}")
