"""
SYNTHETIC VERIFICATION OF MRC-WTF (Text S2) - reproducible, self-contained (no input files)
===========================================================================================
Generates daily hydrographs with a PLANTED recharge-driven rise and applies EXACTLY the rules of
wtf_mrc_recharge.py (3-day mean, dry days Aug-Nov with rain < 2 mm over [t-4, t+1], Theil-Sen of rate vs level
with both signs, f = min(0, a + b s), r = max(0, ds - f), sigma = 1.4826 MAD, eps = max(2 sigma, 0.005),
noise correction sigma/sqrt(2 pi) on quiet days, Jun 1 - Oct 31, coverage >= 80 %).

Generating model (per unit Sy, daily step, 2021-01-01 .. 2024-12-31):
  rain      P(t): Jun 15 - Sep 30 wet days with prob 0.4, depth Gamma(0.8, 20 mm) x LogNormal(0, 0.4), <= 250 mm
  recharge  q(t) = c * max(P(t-1) - 5 mm, 0),  c ~ U(0.002, 0.006) m per mm   (1-day lag)
  recession LINEAR     dh = b (h - h_eq),  b = -0.012 d-1, h_eq = -12 m
            NONLINEAR  dh = -beta (h - h_eq)^1.5 for h > h_eq, beta = 0.004 (m^-0.5 d-1)
  pumping   none | rabi (Dec-Mar, -0.02 m d-1) | rabi + kharif (also -0.01 m d-1 on monsoon dry days)
  noise     Gaussian on the observed level, SD 0, 1, 2, 3 cm
TRUTH = sum of q(t) over Jun 1 - Oct 31.  Metric = estimated dH / TRUTH per station-year (mean +- SD, median).
Run:  python synthetic_verification.py     (N_REAL realizations x scenarios; ~1-3 min)
"""
import numpy as np
import pandas as pd
from scipy import stats

N_REAL, SEED = 50, 20261018
T = pd.date_range("2021-01-01", "2024-12-31", freq="D")
YEARS = [2021, 2022, 2023, 2024]
RAIN_THR, LAG, MIN_DRY, MIN_COVER = 2.0, 5, 30, 0.8
rng = np.random.default_rng(SEED)


def make(recession, pumping, noise):
    wet = T.month.isin([7, 8, 9]) | ((T.month == 6) & (T.day >= 15))
    P = np.where(wet & (rng.random(len(T)) < 0.4), np.clip(rng.gamma(0.8, 20, len(T)) * rng.lognormal(0, 0.4, len(T)), 0, 250), 0.0)
    c = rng.uniform(0.002, 0.006)
    q = np.zeros(len(T)); q[1:] = c * np.clip(P[:-1] - 5, 0, None)
    pump = np.zeros(len(T))
    if pumping in ("rabi", "rabi+kharif"):
        pump[T.month.isin([12, 1, 2, 3])] = -0.02
    if pumping == "rabi+kharif":
        dry5 = pd.Series(P).rolling(5, min_periods=1).max().values < 5
        pump[T.month.isin([6, 7, 8, 9, 10]) & dry5] = -0.01
    h = np.zeros(len(T)); h[0] = -11.0; heq = -12.0
    for t in range(1, len(T)):
        if recession == "linear":
            rec = -0.012 * (h[t - 1] - heq)
        else:
            rec = -0.004 * max(h[t - 1] - heq, 0) ** 1.5
        h[t] = h[t - 1] + rec + q[t] + pump[t]
    obs = h + rng.normal(0, noise, len(T)) if noise > 0 else h
    truth = {y: q[(T >= pd.Timestamp(y, 6, 1)) & (T <= pd.Timestamp(y, 10, 31))].sum() for y in YEARS}
    return pd.Series(obs, index=T), pd.Series(P, index=T), truth


def mrc_wtf(h, rain, months=(8, 9, 10, 11), smooth="mean", fit="both"):
    roll = h.rolling(3, center=True, min_periods=3)
    s = roll.mean() if smooth == "mean" else roll.median()
    rate = s.shift(-1) - s
    wet_prev = rain.rolling(LAG, min_periods=1).max()
    dry = (wet_prev < RAIN_THR) & (rain.shift(-1).fillna(0) < RAIN_THR) & s.notna() & rate.notna()
    m = dry & s.index.month.isin(months)
    if fit == "neg":
        m = m & (rate < 0)
    if m.sum() < MIN_DRY:
        return None
    b, a, _, _ = stats.theilslopes(rate[m].values, s[m].values)
    if b >= 0:
        return None
    res = rate[m].values - (a + b * s[m].values)
    sig = 1.4826 * np.median(np.abs(res - np.median(res)))
    f = np.minimum(0.0, a + b * s)
    d = (rate - f).clip(lower=0)
    eps, bias = max(2 * sig, 0.005), sig / np.sqrt(2 * np.pi)
    out = {}
    for y in YEARS:
        w = (d.index >= pd.Timestamp(y, 6, 1)) & (d.index <= pd.Timestamp(y, 10, 31))
        x = d[w]
        if x.notna().sum() / 153 < MIN_COVER:
            continue
        out[y] = max(float(x.sum()) - bias * int((x <= eps).sum()), 0.0)
    return out


SCEN = [(r, p, n) for r in ("linear", "nonlinear") for p in ("none", "rabi", "rabi+kharif") for n in (0.0, 0.01, 0.02, 0.03)]
rows = []
for rec, pump, noise in SCEN:
    for i in range(N_REAL):
        h, P, truth = make(rec, pump, noise)
        for nm, months in (("Aug-Nov", (8, 9, 10, 11)), ("Aug-Feb", (8, 9, 10, 11, 12, 1, 2))):
            est = mrc_wtf(h, P, months)
            if est is None:
                rows.append({"recession": rec, "pumping": pump, "noise_cm": 100 * noise, "MRC": nm, "real": i, "ratio": np.nan, "rejected": 1})
                continue
            for y, e in est.items():
                rows.append({"recession": rec, "pumping": pump, "noise_cm": 100 * noise, "MRC": nm, "real": i,
                             "ratio": e / truth[y] if truth[y] > 0 else np.nan, "rejected": 0})
D = pd.DataFrame(rows)
S = (D.groupby(["MRC", "recession", "pumping", "noise_cm"])
       .agg(n=("ratio", "count"), mean=("ratio", "mean"), sd=("ratio", "std"), median=("ratio", "median"),
            rejected_MRC=("rejected", "sum")).round(3).reset_index())
S.to_csv("TableS2_synthetic_verification.csv", index=False)
pd.set_option("display.width", 200)
print("=" * 90)
print(f"VERDICT  synthetic_verification  ({N_REAL} realizations per scenario, seed {SEED})")
print("=" * 90)
print(S.to_string(index=False))

# ---------------- design-choice variants (Aug-Nov MRC): running median, negative-increment-only fit
V = []
for rec in ("linear", "nonlinear"):
    for noise in (0.01, 0.02, 0.03):
        for i in range(N_REAL):
            h, P, truth = make(rec, "rabi", noise)
            for nm, kw in (("final (mean, both signs)", {}), ("running median", {"smooth": "median"}),
                           ("negative increments only", {"fit": "neg"})):
                est = mrc_wtf(h, P, **kw)
                if est:
                    V += [{"variant": nm, "recession": rec, "noise_cm": 100 * noise, "ratio": e / truth[y]} for y, e in est.items() if truth[y] > 0]
V = pd.DataFrame(V).groupby(["variant", "recession", "noise_cm"]).ratio.agg(["count", "mean", "std", "median"]).round(3)
V.to_csv("TableS2b_design_variants.csv")
print("\nDESIGN VARIANTS (Aug-Nov MRC, rabi pumping):")
print(V.to_string())
