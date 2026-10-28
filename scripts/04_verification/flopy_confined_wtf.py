"""
VIRTUAL EXPERIMENT (MODFLOW 6 / flopy): CAN A CONFINED, PUMPED PIEZOMETER FAKE "RECHARGE" IN MRC-WTF?
=====================================================================================================
NOT a model of Jalaun. A synthetic alluvial cross-section driven by REAL IMD rain, with layer parameters
taken from the Jalaun/Hamirpur NAQUIM reports (Aq-I phreatic, clay, Aq-II S ~ 1e-4..1e-3).

Hypothesis H (written before running):
  The large alluvial GROSS_C (obs: Sy_100 ~ 0.13, LB/GROSS_C ~ 0.85, GROSS_C ~ 7.4 m per unit Sy) is mainly
  RECOVERY of pumping drawdown in the confined Aq-II. Kharif pumping stops when rain comes, so the recovery is
  timed with rain and looks "rain-linked".
Predictions if H is true:
  P1  GROSS_C at the Aq-II well >> GROSS_C at the water-table well, increasing with pumping intensity.
  P2  LB/GROSS_C at the Aq-II well stays high (> 0.7) although no rain enters Aq-II directly.
  P3  GROSS_C(Aq-II) x Sy(Aq-I) overestimates the TRUE recharge by a factor that grows with pumping.
Falsified if: Aq-II GROSS_C <= water-table GROSS_C in all pumped runs.

Model: 1 row x 50 cols (dx 40 m, 2 km), 3 layers: L1 phreatic sand (Sy 0.10), L2 clay (Kv varied),
L3 confined sand (Ss 2e-5 -> S ~ 4e-4 over 20 m). Stream (RIV) at last column, no-flow divide at col 0.
Recharge into L1 = C x max(P - 5 mm, 0), C set so Jun-Oct recharge = 15% of Jun-Oct rain.
Pumping from L3 (cols 10..40): annual volume = INTENSITY x annual recharge volume; 70% on rabi days
(15 Nov - 28 Feb), 30% on kharif dry days (Jun-Oct, < 5 mm rain in the previous 5 days).
MRC-WTF applied to daily simulated heads (+ 1 cm noise) with the SAME rules as wtf_mrc_recharge.py.
Run in Jupyter:  %run flopy_confined_wtf.py      Requires: flopy xarray netCDF4 pandas numpy scipy ; mf6.exe
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import flopy
from scipy import stats

warnings.filterwarnings("ignore")

# ======================================================================================== CONFIG
MF6 = os.environ.get("TEST_MF6", r"EDIT\ME\symple_flopy-main\bin\mf6.exe")   # same mf6.exe as flopy_virtual_experiment.py
RAIN = os.environ.get("TEST_RAIN", r"E:\up_bundel\up_bundel_rainfall_1990_2025.nc")
WS = os.environ.get("TEST_WS", r"virtual_confined")
RAIN_POINT = (25.99, 79.45)                      # Jalaun (Orai) area
T0, T1 = os.environ.get("TEST_T0", "2015-01-01"), os.environ.get("TEST_T1", "2024-12-31")
NCOL, DX = 50, 40.0
TOP, B1, B2, B3 = 150.0, 125.0, 115.0, 95.0      # L1 25 m | L2 clay 10 m | L3 20 m
STAGE = 140.0
K1, K3, KH2 = 10.0, 40.0, 0.01
SY1, SS = 0.10, 2e-5
KV2_SET = [0.001, 0.01, 0.1]                     # clay vertical K (m/d)
INTENSITY = [0.0, 0.5, 1.0]                      # annual pumping / annual recharge
REC_FRAC = 0.15
OBS = {"pumpcell": 25, "between": 27, "far": 3}   # col 25 IS a pumping cell; 27 = 80 m from wells; 3 = 280 m
NOISE, SEED = 0.01, 20261013
WELL_COLS = list(range(10, 41, 5))
# MRC-WTF settings (identical to wtf_mrc_recharge.py)
SMOOTH, RAIN_THR, LAG_DAYS, MRC_MONTHS, MIN_REC, MERGE_GAP, LINK_MM = 3, 2.0, 5, [8, 9, 10, 11], 30, 2, 10.0
os.makedirs(WS, exist_ok=True)
rng = np.random.default_rng(SEED)
if not os.path.isfile(MF6):
    sys.exit(f"[STOP] mf6.exe not found at {MF6!r} -> set MF6 (same path as in flopy_virtual_experiment.py)")

# ---------------- rain at point
ds = xr.open_dataset(RAIN)
var = [v for v in ds.data_vars if ds[v].ndim == 3][0]
la = next(c for c in ds.coords if "lat" in c.lower()); lo = next(c for c in ds.coords if "lon" in c.lower())
tm = next(c for c in ds.coords if "time" in c.lower())
R = ds[var].transpose(tm, la, lo).values.astype(float); TT = pd.to_datetime(ds[tm].values).normalize()
LA, LO = np.meshgrid(ds[la].values, ds[lo].values, indexing="ij"); valid = ~np.all(np.isnan(R), axis=0)
ds.close()
d_ = np.hypot(LA - RAIN_POINT[0], LO - RAIN_POINT[1]); d_[~valid] = np.inf
i_, j_ = np.unravel_index(np.argmin(d_), d_.shape)
P = pd.Series(R[:, i_, j_], index=TT)
P = P[~P.index.duplicated()].where(lambda s: s >= 0).reindex(pd.date_range(T0, T1, freq="D"))
n_miss = int(P.isna().sum())
print(f"RAIN AUDIT: missing/negative days {n_miss} of {len(P)}"
      + (f" | first {P[P.isna()].index[0].date()} last {P[P.isna()].index[-1].date()} | Jun-Oct missing {int(P[P.index.month.isin([6,7,8,9,10])].isna().sum())}" if n_miss else ""))
if n_miss > 0.01 * len(P):
    sys.exit("[STOP] > 1% rain days missing -> check RAIN_POINT / file before trusting results")
P = P.fillna(0.0)
days = P.index
jjaso = days.month.isin([6, 7, 8, 9, 10])
pe = np.clip(P.values - 5, 0, None)
C = REC_FRAC * P.values[jjaso].sum() / max(pe[jjaso].sum(), 1e-9)
rch = C * pe / 1000.0                                          # m/day
ann_rch_vol = rch.sum() / (len(days) / 365.25) * NCOL * DX * DX  # m3/yr over the domain
rabi = ((days.month == 11) & (days.day >= 15)) | days.month.isin([12, 1, 2])
dry5 = P.rolling(5, min_periods=1).max().shift(1).fillna(0).values < 5.0
kharif_dry = jjaso & dry5
print(f"rain point {RAIN_POINT} | days {len(days)} | Jun-Oct rain/yr {P[jjaso].sum() / (len(days) / 365.25):.0f} mm | "
      f"C {C:.3f} | true recharge/yr {1000 * rch.sum() / (len(days) / 365.25):.0f} mm")


def pumping(intensity):
    q = np.zeros(len(days))
    if intensity <= 0:
        return q
    vol_yr = intensity * ann_rch_vol
    nyr = len(days) / 365.25
    for mask, share in ((rabi, 0.7), (kharif_dry, 0.3)):
        n = mask.sum()
        if n:
            q[mask] = -share * vol_yr * nyr / n / len(WELL_COLS)
    return q


def run(name, kv2, q):
    ws = os.path.join(WS, name)
    sim = flopy.mf6.MFSimulation(sim_name="sim", sim_ws=ws, exe_name=MF6, continue_=True)
    nper = len(days) + 1
    flopy.mf6.ModflowTdis(sim, time_units="DAYS", nper=nper, perioddata=[(1.0, 1, 1.0)] * nper)
    flopy.mf6.ModflowIms(sim, complexity="COMPLEX", outer_maximum=200, inner_maximum=300, linear_acceleration="BICGSTAB")
    gwf = flopy.mf6.ModflowGwf(sim, modelname="gwf", newtonoptions="NEWTON UNDER_RELAXATION")
    flopy.mf6.ModflowGwfdis(gwf, nlay=3, nrow=1, ncol=NCOL, delr=DX, delc=DX, top=TOP, botm=[B1, B2, B3])
    flopy.mf6.ModflowGwfic(gwf, strt=STAGE + 2)
    flopy.mf6.ModflowGwfnpf(gwf, icelltype=[1, 0, 0], k=[K1, KH2, K3], k33=[K1 / 10, kv2, K3 / 10])
    flopy.mf6.ModflowGwfsto(gwf, iconvert=[1, 0, 0], ss=[1e-5, 1e-4, SS], sy=[SY1, 0.02, 0.05],
                            steady_state={0: True}, transient={1: True})
    t = np.arange(nper + 1, dtype=float)
    r_ts = np.r_[rch.mean(), rch, rch[-1]]
    q_ts = np.r_[0.0, q, q[-1]]
    rp = flopy.mf6.ModflowGwfrch(gwf, stress_period_data={0: [((0, 0, c), "rech") for c in range(NCOL)]})
    rp.ts.initialize(filename="rch.ts", timeseries=list(zip(t, r_ts)), time_series_namerecord="rech",
                     interpolation_methodrecord="stepwise")
    flopy.mf6.ModflowGwfriv(gwf, stress_period_data={0: [((0, 0, NCOL - 1), STAGE, 500.0, STAGE - 3.0)]})
    if np.any(q != 0):
        wp = flopy.mf6.ModflowGwfwel(gwf, stress_period_data={0: [((2, 0, c), "q") for c in WELL_COLS]})
        wp.ts.initialize(filename="wel.ts", timeseries=list(zip(t, q_ts)), time_series_namerecord="q",
                         interpolation_methodrecord="stepwise")
    flopy.mf6.ModflowGwfoc(gwf, head_filerecord="gwf.hds", saverecord=[("HEAD", "ALL")])
    sim.write_simulation(silent=True)
    ok, _ = sim.run_simulation(silent=True)
    hf = os.path.join(ws, "gwf.hds")
    if not ok or not os.path.exists(hf):
        print(f"  !!! {name} failed"); return None
    hd = flopy.utils.HeadFile(hf)
    out = {}
    for pos, col in OBS.items():
        for lay, tag in ((0, "L1"), (2, "L3")):
            ts = hd.get_ts((lay, 0, col))
            # SP0 (steady, totim 1) dropped; totim k+2 = end of days[k]
            assert len(ts) == nper and np.allclose(ts[1:, 0], np.arange(2, nper + 1)), "time alignment failed"
            out[f"{tag}_{pos}"] = pd.Series(ts[1:, 1], index=days)
    return out


def mrc_wtf(h):
    """same rules as wtf_mrc_recharge.py; returns dict of season metrics (median over years)"""
    h = h + rng.normal(0, NOISE, len(h))
    s = h.rolling(SMOOTH, center=True, min_periods=SMOOTH).mean()
    rate = s.shift(-1) - s
    dry = (P.rolling(LAG_DAYS, min_periods=1).max() < RAIN_THR) & (P.shift(-1).fillna(0) < RAIN_THR)
    m = dry & s.notna() & rate.notna() & s.index.month.isin(MRC_MONTHS)
    if m.sum() < MIN_REC:
        return {"note": "few dry days"}
    b, a, _, _ = stats.theilslopes(rate[m].values, s[m].values)
    if b >= 0:
        return {"note": "slope >= 0"}
    res = rate[m].values - (a + b * s[m].values)
    sig = 1.4826 * np.median(np.abs(res - np.median(res)))
    f = np.minimum(0.0, a + b * s)
    d = (rate - f).clip(lower=0)
    eps = max(2 * sig, 0.005); bias = sig / np.sqrt(2 * np.pi)
    G, LBR, PL = [], [], []
    for y in sorted(set(days.year)):
        w = (days >= pd.Timestamp(y, 6, 1)) & (days <= pd.Timestamp(y, 10, 31))
        x = d[w]
        if x.notna().sum() < 120:
            continue
        gc = max(float(x.sum()) - bias * int((x <= eps).sum()), 0)
        idx = np.where((x > eps).values)[0]
        lb = 0.0
        if len(idx):
            grp, cur = [], [idx[0]]
            for k in idx[1:]:
                if k - cur[-1] <= MERGE_GAP + 1: cur.append(k)
                else: grp.append(cur); cur = [k]
            grp.append(cur)
            for gi in grp:
                t0, t1 = x.index[gi[0]], x.index[gi[-1]]
                if P[t0 - pd.Timedelta(days=LAG_DAYS):t1].sum() >= LINK_MM:
                    lb += float(x.loc[t0:t1].sum())
        dep = -s
        mon = dep[(days >= pd.Timestamp(y, 7, 1)) & (days <= pd.Timestamp(y, 10, 31))]
        pre = dep[(days >= pd.Timestamp(y, 5, 1)) & (days <= mon.idxmin())]
        G.append(gc); LBR.append(lb / gc if gc > 0 else np.nan); PL.append(float(pre.max() - mon.min()))
    G = np.array(G)
    return {"note": "ok", "b": b, "GROSS_C": np.median(G), "LB_RATIO": np.nanmedian(LBR), "PEAKLOW": np.median(PL),
            "G_min": G.min(), "G_max": G.max(), "n_yr": len(G), "G_years": G}


rain_jjaso = P[jjaso].groupby(P[jjaso].index.year).sum().median()
true_rch_yr = pd.Series(rch, index=days)[jjaso].groupby(days[jjaso].year).sum()
true_rch_m = true_rch_yr.median()
ROWS = []
for kv2 in KV2_SET:
    for inten in INTENSITY:
        name = f"kv{kv2:g}_p{inten:g}"
        out = run(name, kv2, pumping(inten))
        if out is None:
            continue
        for lay, h in out.items():
            r_ = mrc_wtf(h)
            row = {"run": name, "Kv_clay": kv2, "pump": inten, "well": lay, **r_,
                   "amp_m": float(h.groupby(h.index.year).agg(lambda v: v.max() - v.min()).median())}
            if r_.get("note") == "ok":
                row["SY100"] = rain_jjaso / 1000 / r_["GROSS_C"] if r_["GROSS_C"] > 0 else np.nan
                row["OVEREST"] = r_["GROSS_C"] * SY1 / true_rch_m
                ov = r_["G_years"] * SY1 / true_rch_yr.reindex(range(days.year.min(), days.year.max() + 1)).values[:len(r_["G_years"])]
                row["OV_min"], row["OV_max"] = float(np.nanmin(ov)), float(np.nanmax(ov))
                row["BIAS_mm"] = 1000 * (r_["GROSS_C"] * SY1 - true_rch_m)
            row.pop("G_years", None)
            ROWS.append(row)
        print(f"  done {name}")
D = pd.DataFrame(ROWS)
D.to_csv(os.path.join(WS, "confined_wtf_results.csv"), index=False)

pd.set_option("display.width", 200)
print("=" * 100)
print("VERDICT  flopy_confined_wtf")
print("=" * 100)
print(f"Jun-Oct rain median {rain_jjaso:.0f} mm | TRUE Jun-Oct recharge {1000 * true_rch_m:.0f} mm (= {100 * REC_FRAC:.0f}%) | "
      f"true per-unit-Sy rise at Sy1={SY1}: {true_rch_m / SY1:.2f} m")
print("OBSERVED alluvium (for comparison): GROSS_C ~7.4 m | Sy_100 ~0.13 | LB/GROSS_C ~0.85")
cols = ["Kv_clay", "pump", "well", "note", "GROSS_C", "amp_m", "SY100", "LB_RATIO", "OVEREST", "OV_min", "OV_max", "BIAS_mm"]
print(D[[c for c in cols if c in D.columns]].round(3).to_string(index=False))
ok = D[D.note == "ok"].pivot_table(index=["Kv_clay", "pump"], columns="well", values="GROSS_C")
for pos in OBS:
    if f"L3_{pos}" in ok.columns and f"L1_{pos}" in ok.columns:
        ok[f"ratio_{pos}"] = ok[f"L3_{pos}"] / ok[f"L1_{pos}"]
print("\nGROSS_C (m per unit Sy) by well; ratio = confined / water-table at the same column:")
print(ok.round(2).to_string())
pumped = ok[ok.index.get_level_values("pump") > 0]
for pos in OBS:
    c = f"ratio_{pos}"
    if c in pumped.columns:
        print(f"H test P1 at '{pos}' (col {OBS[pos]}): L3 > L1 in {int((pumped[c] > 1).sum())} of {int(pumped[c].notna().sum())} pumped runs")
print("NOTES: OVEREST = GROSS_C x Sy(Aq-I) / true Jun-Oct recharge (1 = unbiased); OV_min/max = range over years;")
print("BIAS_mm = GROSS_C x Sy(Aq-I) - true recharge. Synthetic section, not a calibrated model.")
