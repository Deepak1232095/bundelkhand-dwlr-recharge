"""
CONFINED vs UNCONFINED FROM EARTH TIDES IN 6-HOURLY DWLR
=========================================================
Physics: a confined/semi-confined head responds to earth-tide strain (M2 12.42 h, O1 25.82 h); a water table
practically does not. 6-hourly samples (00/06/12/18): Nyquist = 2 cpd, so M2 (1.9323 cpd) and O1 (0.9295 cpd)
are resolvable by least squares over >= 15-30 days; S2 (2.000) and K1/S1 (~1.0) coincide with the sampling
grid and are fitted only as nuisance terms (they also absorb barometric/thermal/logger daily cycles).

Per station:
  1. raw level, sentinels removed (|L| >= 90, <= -900, 99.0); |level| used (sign issues are known); only exact
     00/06/12/18 h readings kept; duplicates averaged.
  2. RESOLUTION check: share of values that are exact multiples of 0.1 m. True mm data -> ~10%.
     Station used only if that share <= MAX_DM_SHARE (dm-rounded data cannot show cm-scale tides).
  3. 30-day windows (>= MIN_FILL of 120 samples): least squares with cubic trend + S1 (cos, sin) + S2 (cos only,
     sin is unidentifiable at 4/day) + N2 + O1 + M2. Complex amplitudes are VECTOR-averaged over windows
     (a real tide keeps its phase in absolute time; noise averages out).
  4. NOISE NULL: the same fit with one extra sinusoid at each of 20 non-tidal frequencies near O1 and M2;
     SNR = A_M2 / median(noise amplitude); coherence R = |mean| / mean(|.|) of the M2 window vectors.
  5. CLASS:  TIDAL      M2 SNR >= SNR_TIDAL and A_M2 >= MIN_AMP_MM and R >= MIN_R
             NO_TIDE    A_M2 < MIN_AMP_MM and noise floor <= MIN_AMP_MM/2 (the test had power to see a >= 1 mm tide)
             UNCLEAR    everything else;  LOW_RES / TOO_SHORT if not testable.
Then joins with geology class and the MRC-WTF diagnostics (GROSS_C, Sy_100) if those files exist.
Limitation: no barometric data -> barometric efficiency not estimated (ERA5 pressure could be added later).
Run in Jupyter:  %run dwlr_earth_tide.py      Requires: pandas numpy scipy
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# ======================================================================================== CONFIG
RAW = os.environ.get("TEST_RAW", r"C:\Users\IITI\UPGWD_Bundelkhand_DWLR.csv")
QC = r"DWLR_QC"
SIGN = os.path.join(QC, "dwlr_sign_smoothness.csv")                                      # optional (core set)
GEOL = os.path.join(QC, "regime_controls", "geology_v3", "station_geology_audit.csv")  # optional
DIAG = os.path.join(QC, "wtf_mrc", "diagnostics", "diag_station.csv")                  # optional
OUT = os.path.join(QC, "earth_tide")
COLS = {"Station": "STN", "District": "DIST", "Latitude": "lat", "Longitude": "lon",
        "Data Acquisition Time": "T", "Groundwater Level Telemetry 6 Hourly (meter)": "L"}
F = {"O1": 0.9295357, "M2": 1.9322736, "N2": 1.8959820}        # cycles per day
NOISE_F = list(np.r_[np.linspace(0.78, 0.88, 6), np.linspace(1.05, 1.15, 4), np.linspace(1.62, 1.86, 10)])
WIN_DAYS, MIN_FILL = 30, 0.6
MAX_DM_SHARE = 0.30
SNR_TIDAL, MIN_AMP_MM, MIN_R = 3.0, 1.0, 0.3
MIN_WINDOWS = 3
os.makedirs(OUT, exist_ok=True)

if not os.path.exists(RAW):
    sys.exit(f"[STOP] not found: {RAW}")
r = pd.read_csv(RAW, usecols=list(COLS))
r = r.rename(columns=COLS)
r["T"] = pd.to_datetime(r["T"], dayfirst=True, errors="coerce")
r = r.dropna(subset=["T", "L"])
r = r[(r.L > -900) & (r.L.abs() < 90) & (r.L != 99.0)]
r = r[(r["T"].dt.minute == 0) & (r["T"].dt.hour.isin([0, 6, 12, 18]))]
r["L"] = r.L.abs()
r["SID"] = r.DIST.astype(str).str.strip() + "|" + r.STN.astype(str).str.strip()
print(f"rows kept {len(r)} | stations {r.SID.nunique()}")


def design(t, extra=None):
    """t in days (absolute). Columns: cubic trend, S1 cos/sin, S2 cos, N2, O1, M2 (+ extra sinusoid)."""
    tc = (t - t.mean()) / max(np.ptp(t), 1)
    X = [np.ones_like(t), tc, tc ** 2, tc ** 3,
         np.cos(2 * np.pi * t), np.sin(2 * np.pi * t), np.cos(4 * np.pi * t)]
    names = []
    for k in ("N2", "O1", "M2"):
        X += [np.cos(2 * np.pi * F[k] * t), np.sin(2 * np.pi * F[k] * t)]; names.append(k)
    if extra is not None:
        X += [np.cos(2 * np.pi * extra * t), np.sin(2 * np.pi * extra * t)]
    return np.column_stack(X), names


def amp(beta, j):            # complex amplitude from cos/sin coefficients at column j, j+1
    return beta[j] - 1j * beta[j + 1]


T0 = pd.Timestamp("2020-01-01")
ROWS = []
for sid, g in r.groupby("SID"):
    s = g.groupby("T").L.mean().sort_index()
    row = {"SID": sid, "DIST": g.DIST.iloc[0], "STN": g.STN.iloc[0], "lat": g.lat.iloc[0], "lon": g.lon.iloc[0],
           "n_obs": len(s)}
    v = s.values
    row["dm_share"] = float(np.mean(np.abs(v * 10 - np.round(v * 10)) < 1e-6))
    if row["dm_share"] > MAX_DM_SHARE:
        ROWS.append({**row, "CLASS_TIDE": "LOW_RES"}); continue
    t_all = (s.index - T0).total_seconds().values / 86400.0
    W = {k: [] for k in F}
    NZ = []
    start = t_all.min()
    while start + WIN_DAYS <= t_all.max() + 1e-9:
        m = (t_all >= start) & (t_all < start + WIN_DAYS)
        start += WIN_DAYS
        if m.sum() < MIN_FILL * WIN_DAYS * 4:
            continue
        t, y = t_all[m], v[m]
        X, names = design(t)
        b = np.linalg.lstsq(X, y, rcond=None)[0]
        res = y - X @ b
        if np.std(res) > 0.5:                      # window dominated by jumps -> skip
            continue
        for i, k in enumerate(names):
            W[k].append(amp(b, 7 + 2 * i))
        nz = []
        for fn in NOISE_F:
            Xn, _ = design(t, fn)
            bn = np.linalg.lstsq(Xn, y, rcond=None)[0]
            nz.append(amp(bn, Xn.shape[1] - 2))
        NZ.append(nz)
    nw = len(W["M2"])
    row["n_windows"] = nw
    if nw < MIN_WINDOWS:
        ROWS.append({**row, "CLASS_TIDE": "TOO_SHORT"}); continue
    NZ = np.array(NZ)                                     # windows x noise freqs
    noise_amp = np.abs(NZ.mean(axis=0)) * 1000            # vector-averaged, mm
    floor = float(np.median(noise_amp))
    for k in F:
        z = np.array(W[k])
        row[f"A_{k}_mm"] = float(np.abs(z.mean()) * 1000)
        row[f"R_{k}"] = float(np.abs(z.mean()) / np.mean(np.abs(z))) if np.mean(np.abs(z)) > 0 else np.nan
        row[f"SNR_{k}"] = row[f"A_{k}_mm"] / floor if floor > 0 else np.nan
        row[f"PH_{k}_deg"] = float(np.degrees(np.angle(z.mean())))
    row["noise_floor_mm"] = floor
    row["p_M2"] = float((np.sum(noise_amp >= row["A_M2_mm"]) + 1) / (len(noise_amp) + 1))
    if row["SNR_M2"] >= SNR_TIDAL and row["A_M2_mm"] >= MIN_AMP_MM and row["R_M2"] >= MIN_R:
        c = "TIDAL"
    elif row["A_M2_mm"] < MIN_AMP_MM and floor <= MIN_AMP_MM / 2:
        c = "NO_TIDE"
    else:
        c = "UNCLEAR"
    ROWS.append({**row, "CLASS_TIDE": c})

D = pd.DataFrame(ROWS)
# ---- joins (optional)
core = None
if os.path.exists(SIGN):
    sg = pd.read_csv(SIGN)
    core = set(sg[sg.sign_verdict.isin(["single sign (fine)", "sign errors -> |level| fixes it"])].SID)
    D["CORE"] = D.SID.isin(core)
if os.path.exists(GEOL):
    g_ = pd.read_csv(GEOL)
    if {"SID", "CLASS"} <= set(g_.columns):
        D["GEOL"] = D.SID.map(g_.drop_duplicates("SID").set_index("SID").CLASS)
if os.path.exists(DIAG):
    dg = pd.read_csv(DIAG).drop_duplicates("SID").set_index("SID")
    for c in ("GROSS_C", "SY100", "LB_RATIO", "RF_SHARE_AN"):
        if c in dg.columns:
            D[c] = D.SID.map(dg[c])
D.to_csv(os.path.join(OUT, "earth_tide_stations.csv"), index=False)

# ======================================================================================== VERDICT
pd.set_option("display.width", 200)
print("=" * 100)
print("VERDICT  dwlr_earth_tide")
print("=" * 100)
print(f"stations {len(D)} | class counts: {D.CLASS_TIDE.value_counts().to_dict()}")
if core is not None:
    print(f"matched to QC core set: {int(D.CORE.sum())} of {len(core)}  (if low, the SID format differs -> check inputs)")
    print(f"core stations class counts: {D[D.CORE].CLASS_TIDE.value_counts().to_dict()}")
T_ = D[D.CLASS_TIDE.isin(["TIDAL", "NO_TIDE", "UNCLEAR"])]
if len(T_):
    print(f"\ntestable stations {len(T_)} | dm_share median {T_.dm_share.median():.2f} | windows median {T_.n_windows.median():.0f}")
    print(f"  noise floor median {T_.noise_floor_mm.median():.2f} mm | A_M2 median {T_.A_M2_mm.median():.2f} mm | "
          f"A_O1 median {T_.A_O1_mm.median():.2f} mm")
    tt = T_[T_.CLASS_TIDE == "TIDAL"]
    if len(tt):
        print(f"  TIDAL wells: A_M2 median {tt.A_M2_mm.median():.2f} mm [range {tt.A_M2_mm.min():.1f}-{tt.A_M2_mm.max():.1f}] | "
              f"SNR median {tt.SNR_M2.median():.1f} | R median {tt.R_M2.median():.2f} | O1/M2 median {(tt.A_O1_mm / tt.A_M2_mm).median():.2f}"
              f" | M2 phase spread (circ. std, deg) {np.degrees(stats.circstd(np.radians(tt.PH_M2_deg))):.0f}")
    print("  by district:")
    print(T_.groupby("DIST").CLASS_TIDE.value_counts().unstack(fill_value=0).to_string())
    if "GEOL" in T_.columns and T_.GEOL.notna().any():
        print("  by geology class:")
        ct = pd.crosstab(T_.GEOL, T_.CLASS_TIDE)
        print(ct.to_string())
        if {"ALLUVIUM", "HARD_ROCK"} <= set(ct.index) and "TIDAL" in ct.columns:
            a = ct.loc[["ALLUVIUM", "HARD_ROCK"]]
            tab = np.c_[a["TIDAL"], a.drop(columns="TIDAL").sum(axis=1)]
            print(f"  Fisher (TIDAL vs rest, alluvium vs hard rock): odds {stats.fisher_exact(tab)[0]:.2f}, p {stats.fisher_exact(tab)[1]:.3f}")
    for c in ("GROSS_C", "SY100", "LB_RATIO"):
        if c in T_.columns and T_[c].notna().sum() >= 6:
            a, b = T_[T_.CLASS_TIDE == "TIDAL"][c].dropna(), T_[T_.CLASS_TIDE == "NO_TIDE"][c].dropna()
            p = stats.mannwhitneyu(a, b).pvalue if len(a) >= 3 and len(b) >= 3 else np.nan
            print(f"  {c:<9} TIDAL median {a.median():.3f} (n {len(a)}) vs NO_TIDE {b.median():.3f} (n {len(b)}) | MW p {p:.3f}")
        if c in T_.columns and T_[[c, "A_M2_mm"]].dropna().shape[0] >= 8:
            q = T_[[c, "A_M2_mm"]].dropna()
            print(f"  {c:<9} vs A_M2 Spearman {stats.spearmanr(q[c], q.A_M2_mm)[0]:+.2f} (n {len(q)})")
print("\nNOTES: TIDAL = head responds to earth-tide strain -> confined/semi-confined response at that well (WTF not")
print("valid there). NO_TIDE = no tide where one could have been seen -> consistent with a water table.")
print("LOW_RES = >30% of values rounded to 0.1 m -> not testable. No barometric correction was applied.")
print(f"Outputs: {OUT}")
