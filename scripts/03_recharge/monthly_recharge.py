"""
MONTHLY RECHARGE (mm) FROM THE AUDITED DWLR DATA  ->  C:\\Users\\IITI\\paper3_final\\
====================================================================================
Same MRC-WTF as wtf_mrc_recharge.py (stored PRIMARY MRC a, b, sigma; 3-day mean; d = max(0, ds - f(s))),
summed per calendar month Jun-Oct, noise-corrected per month (sigma/sqrt(2 pi) x quiet days).
Recharge (mm) = 1000 x Sy x monthly rise, with GEC-1997 (2009 reprint) Sec 5.9.1 Sy norms:
    HARD_ROCK  weathered granite/gneiss: central 3 %, range 1.5-4 %
    ALLUVIUM   clayey alluvium:          central 6 %, range 4-8 %
Months with < 80 % of days observed are not used. Station value = median over its years; district/class =
median over stations (bootstrap 95 % CI). Monthly rain from the same IMD grid cell.
Run from C:\\Users\\IITI (uses run_Q123 outputs):   %run monthly_recharge.py
"""
import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.environ.get("TEST_BASE", r"C:\Users\IITI")
RUN = os.path.join(BASE, "run_Q123", "DWLR_QC")
DAILY = os.path.join(RUN, "dwlr_daily_depth_abs.csv")
MRC = os.path.join(RUN, "wtf_mrc", "mrc_parameters.csv")
SIGN = os.path.join(RUN, "dwlr_sign_smoothness.csv")
GEOL = os.path.join(RUN, "regime_controls", "geology_v3", "station_geology_audit.csv")
RAIN = os.environ.get("TEST_RAIN", r"E:\up_bundel\up_bundel_rainfall_1990_2025.nc")
OUT = os.path.join(BASE, "paper3_final")
SY = {"HARD_ROCK": (0.03, 0.015, 0.04), "ALLUVIUM": (0.06, 0.04, 0.08)}     # central, low, high
MONTHS = [6, 7, 8, 9, 10]
MNAME = {6: "Jun", 7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct"}
MIN_COVER, GAP_FILL, SMOOTH, N_BOOT = 0.8, 3, 3, 2000
rng = np.random.default_rng(20261013)
for f in (DAILY, MRC, SIGN, RAIN):
    if not os.path.exists(f):
        sys.exit(f"[STOP] not found: {f}")
os.makedirs(OUT, exist_ok=True)

dd = pd.read_csv(DAILY, parse_dates=["T"])
M = pd.read_csv(MRC)
M = M[(M.MRC == "PRIMARY") & (M.note == "ok")].set_index("SID")
sg = pd.read_csv(SIGN).drop_duplicates("SID").set_index("SID")
cls = pd.read_csv(GEOL).drop_duplicates("SID").set_index("SID").CLASS if os.path.exists(GEOL) else pd.Series(dtype=str)

ds = xr.open_dataset(RAIN)
var = [v for v in ds.data_vars if ds[v].ndim == 3][0]
la = next(c for c in ds.coords if "lat" in c.lower()); lo = next(c for c in ds.coords if "lon" in c.lower())
tm = next(c for c in ds.coords if "time" in c.lower())
R = ds[var].transpose(tm, la, lo).values.astype(float); TT = pd.to_datetime(ds[tm].values).normalize()
LA, LO = np.meshgrid(ds[la].values, ds[lo].values, indexing="ij"); valid = ~np.all(np.isnan(R), axis=0)
ds.close()


def rain_at(lat, lon):
    d = np.hypot(LA - lat, LO - lon); d[~valid] = np.inf
    i, j = np.unravel_index(np.argmin(d), d.shape)
    s = pd.Series(R[:, i, j], index=TT)
    return s[~s.index.duplicated()].where(lambda v: v >= 0)


ROWS = []
for sid, g in dd[dd.SID.isin(M.index)].groupby("SID"):
    a, b, sig = M.loc[sid, ["a", "b", "sigma"]].astype(float)
    dep = g.groupby("T").DEPTH_ABS.median().sort_index()
    full = pd.date_range(dep.index.min(), dep.index.max(), freq="D")
    h = -dep.reindex(full).interpolate(limit=GAP_FILL, limit_area="inside")
    s = h.rolling(SMOOTH, center=True, min_periods=SMOOTH).mean()
    s[h.isna()] = np.nan
    d = ((s.shift(-1) - s) - np.minimum(0.0, a + b * s)).clip(lower=0)
    eps, bias = max(2 * sig, 0.005), sig / np.sqrt(2 * np.pi)
    rain = rain_at(sg.loc[sid, "lat"], sg.loc[sid, "lon"]).reindex(full) if sid in sg.index else pd.Series(np.nan, index=full)
    for (y, m), x in d.groupby([full.year, full.month]):
        if m not in MONTHS:
            continue
        ndays = pd.Timestamp(y, m, 1).days_in_month
        if x.notna().sum() < MIN_COVER * ndays:
            continue
        rise = max(float(x.sum()) - bias * int((x <= eps).sum()), 0.0)
        ROWS.append({"SID": sid, "DIST": str(sid).split("|")[0].upper(), "CLASS": cls.get(sid, np.nan), "YEAR": y, "MONTH": m,
                     "RISE_m": rise, "RAIN_mm": float(rain[(full.year == y) & (full.month == m)].sum(min_count=1))})
D = pd.DataFrame(ROWS)
if D.empty:
    sys.exit("[STOP] no station-months with enough data")
D = D[D.CLASS.isin(SY.keys())].copy()
for k, i in (("R_mm", 0), ("R_low_mm", 1), ("R_high_mm", 2)):
    D[k] = 1000 * D.CLASS.map(lambda c: SY[c][i]) * D.RISE_m
D.round(3).to_csv(os.path.join(OUT, "T6_monthly_station_year.csv"), index=False)

# station-month medians over years, then group medians over stations
S = D.groupby(["SID", "DIST", "CLASS", "MONTH"]).agg(R=("R_mm", "median"), RL=("R_low_mm", "median"),
                                                     RH=("R_high_mm", "median"), P=("RAIN_mm", "median")).reset_index()


def ci(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 3:
        return (np.median(x) if len(x) else np.nan), np.nan, np.nan
    bb = np.median(rng.choice(x, (N_BOOT, len(x))), axis=1)
    return np.median(x), *np.percentile(bb, [2.5, 97.5])


TAB = []
for keys, q in list(S.groupby(["CLASS", "MONTH"])) + list(S.groupby(["DIST", "MONTH"])):
    grp, m = keys
    r, rlo, rhi = ci(q.R)
    TAB.append({"group": grp, "month": MNAME[m], "n_stations": q.SID.nunique(), "recharge_mm": round(r, 1),
                "CI95": f"[{rlo:.1f}, {rhi:.1f}]", "Sy_range_mm": f"{q.RL.median():.1f}-{q.RH.median():.1f}",
                "rain_mm": round(q.P.median(), 0), "pct_of_rain": round(100 * r / q.P.median(), 1) if q.P.median() > 0 else np.nan})
T = pd.DataFrame(TAB)
T.to_csv(os.path.join(OUT, "T6_monthly_recharge_summary.csv"), index=False)

season = S.groupby(["SID", "DIST", "CLASS"]).agg(R=("R", "sum"), RL=("RL", "sum"), RH=("RH", "sum"), P=("P", "sum"),
                                                n_m=("MONTH", "nunique")).reset_index()
season = season[season.n_m == len(MONTHS)]

# figure: monthly recharge by class (median + 95% CI), rain on the same panel as % labels
fig, ax = plt.subplots(figsize=(6, 3.4))
col = {"ALLUVIUM": "#2a78d6", "HARD_ROCK": "#eb6834"}
w = 0.38
for j, c in enumerate(["HARD_ROCK", "ALLUVIUM"]):
    t = T[T.group == c].set_index("month").reindex([MNAME[m] for m in MONTHS])
    x = np.arange(len(MONTHS)) + (j - 0.5) * w
    lo_ = t.CI95.str.extract(r"\[([-\d.]+),")[0].astype(float); hi_ = t.CI95.str.extract(r",\s*([-\d.]+)\]")[0].astype(float)
    ax.bar(x, t.recharge_mm, width=w * 0.92, color=col[c], label=f"{c.replace('_', ' ').title()} (Sy {100 * SY[c][0]:g} %)")
    ax.errorbar(x, t.recharge_mm, yerr=[t.recharge_mm - lo_, hi_ - t.recharge_mm], fmt="none", ecolor="#1f1f1e", lw=1, capsize=2)
ax.set_xticks(range(len(MONTHS))); ax.set_xticklabels([MNAME[m] for m in MONTHS])
ax.set_ylabel("Recharge (mm/month)"); ax.legend(frameon=False, fontsize=8)
ax.set_title("Monthly monsoon recharge, median of stations (95% CI), 2023-2024", loc="left", fontsize=9)
for s_ in ("top", "right"):
    ax.spines[s_].set_visible(False)
fig.savefig(os.path.join(OUT, "Fig7_monthly_recharge.png"), dpi=300, bbox_inches="tight"); plt.close(fig)

pd.set_option("display.width", 200)
print("=" * 100)
print("VERDICT  monthly_recharge")
print("=" * 100)
print(f"station-months used {len(D)} | stations {D.SID.nunique()} | Sy (GEC-1997): {SY}")
print("\nMonthly recharge, median over stations [95% CI], mm:")
print(T[T.group.isin(SY.keys())].to_string(index=False))
print("\nBy district:")
print(T[~T.group.isin(SY.keys())].to_string(index=False))
print("\nJun-Oct total (stations with all 5 months), median mm [Sy range]:")
for c, q in season.groupby("CLASS"):
    print(f"  {c:<9} {q.R.median():.0f} mm [{q.RL.median():.0f}-{q.RH.median():.0f}] | rain {q.P.median():.0f} mm | "
          f"{100 * q.R.median() / q.P.median():.0f} % of rain | n {len(q)}")
for d_, q in season.groupby("DIST"):
    print(f"  {d_:<9} {q.R.median():.0f} mm [{q.RL.median():.0f}-{q.RH.median():.0f}] | n {len(q)}")
print(f"\nFiles in {OUT}: T6_monthly_station_year.csv, T6_monthly_recharge_summary.csv, Fig7_monthly_recharge.png")
