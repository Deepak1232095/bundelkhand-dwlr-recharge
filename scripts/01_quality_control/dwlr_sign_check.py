"""
DWLR SIGN + SANITY CHECK  —  decides whether |level| = depth below ground is safe
=================================================================================
1. Smoothness test per station: if flipping everything to |level| makes the series much
   SMOOTHER, the +/- mix is a recording error and |level| is correct.
2. Why does 2025 monsoon have 0 usable stations? (month-by-month counts)
3. Independent truth: nearest CGWB well (<= 5 km) — compare DWLR |level| in May/Nov 2023-24
   with CGWB PRM/PTM of the same season. Also checks the odd Hamirpur median depth (26.8 m).

  Requires: pandas numpy
"""
import os
import numpy as np
import pandas as pd

DWLR = r"EDIT\ME\UPGWD_Bundelkhand_DWLR.csv"
CGWB = "G:/Paper coleection/Paper 10/Cleaned_Data/UP_GW_wide_final.csv"
OUT = os.path.join(os.path.dirname(DWLR), "DWLR_QC")
MAX_KM = 5.0
os.makedirs(OUT, exist_ok=True)

# ---------------- load DWLR ----------------
d = pd.read_csv(DWLR)
d["T"] = pd.to_datetime(d["Data Acquisition Time"].astype(str), format="%d-%m-%Y %H:%M", errors="coerce")
d["LVL"] = pd.to_numeric(d["Groundwater Level Telemetry 6 Hourly (meter)"], errors="coerce")
d.loc[d["LVL"] <= -900, "LVL"] = np.nan
d["SID"] = (d["District"].astype(str).str.strip() + "|" + d["Block"].astype(str).str.strip() + "|" +
            d["Station"].astype(str).str.strip())
d = d.dropna(subset=["T", "LVL"]).sort_values(["SID", "T"]).drop_duplicates(["SID", "T"])
# drop spikes on |level| (same rule as dwlr_qc.py) and exact zeros (dead sensor)
a = d["LVL"].abs()
med = a.groupby(d["SID"]).transform(lambda s: s.rolling(7, center=True, min_periods=3).median())
d = d[((a - med).abs() <= 5.0) & (d["LVL"] != 0)].copy()
d["ABS"] = d["LVL"].abs()

# ---------------- 1. smoothness test ----------------
rows = []
for sid, g in d.groupby("SID"):
    if len(g) < 50:
        continue
    tv_raw = g["LVL"].diff().abs().sum()
    tv_abs = g["ABS"].diff().abs().sum()
    pos = (g["LVL"] > 0).mean()
    rows.append({"SID": sid, "DIST": g["District"].iloc[0], "n": len(g),
                 "pct_pos": 100 * pos, "tv_ratio_abs_over_raw": tv_abs / tv_raw if tv_raw > 0 else np.nan,
                 "med_abs": g["ABS"].median(), "lat": g["Latitude"].iloc[0], "lon": g["Longitude"].iloc[0]})
sm = pd.DataFrame(rows)


def verdict(r):
    if r.pct_pos < 2 or r.pct_pos > 98:
        return "single sign (fine)"
    if r.tv_ratio_abs_over_raw < 0.5:
        return "sign errors -> |level| fixes it"
    if r.tv_ratio_abs_over_raw < 0.9:
        return "partly fixed by |level|"
    return "NOT fixed by |level| (offset/real change)"


sm["sign_verdict"] = sm.apply(verdict, axis=1)
sm.to_csv(os.path.join(OUT, "dwlr_sign_smoothness.csv"), index=False)

# ---------------- 2. 2025 coverage ----------------
d["YM"] = d["T"].dt.to_period("M")
cov = d[d["T"].dt.year >= 2024].groupby("YM").agg(stations=("SID", "nunique"), readings=("LVL", "size"))

# ---------------- 3. CGWB cross-check ----------------
w = pd.read_csv(CGWB)
w["LAT"] = pd.to_numeric(w["LATITUDE"], errors="coerce"); w["LON"] = pd.to_numeric(w["LONGITUDE"], errors="coerce")
w = w.dropna(subset=["LAT", "LON"]).reset_index(drop=True)


def hav(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    x = np.sin((lat2 - lat1) * p / 2) ** 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2
    return 12742 * np.arcsin(np.sqrt(x))


pairs = []
for _, s in sm.iterrows():
    dist = hav(s.lat, s.lon, w["LAT"].values, w["LON"].values)
    j = int(np.nanargmin(dist))
    if dist[j] > MAX_KM:
        continue
    g = d[d.SID == s.SID]
    for yr in (2023, 2024):
        for season, mon, col in (("May", 5, "PRM"), ("Nov", 11, "PTM")):
            dw = g[(g["T"].dt.year == yr) & (g["T"].dt.month == mon)]
            cg = pd.to_numeric(w.at[j, f"{col}_{yr % 100:02d}"], errors="coerce") if f"{col}_{yr % 100:02d}" in w else np.nan
            if len(dw) >= 20 and np.isfinite(cg):
                pairs.append({"SID": s.SID, "DIST": s.DIST, "km": dist[j], "year": yr, "season": season,
                              "dwlr_abs": dw["ABS"].median(), "dwlr_raw": dw["LVL"].median(), "cgwb": cg,
                              "sign_verdict": s.sign_verdict})
pr = pd.DataFrame(pairs)
pr.to_csv(os.path.join(OUT, "dwlr_vs_cgwb_pairs.csv"), index=False)

# ---------------- VERDICT ----------------
print("=" * 72)
print("VERDICT")
print("=" * 72)
print("1. sign verdicts:", sm.sign_verdict.value_counts().to_dict())
print("   by district:")
print(pd.crosstab(sm.DIST, sm.sign_verdict).to_string())
print("\n2. coverage 2024-2025 (stations reporting / readings):")
print(cov.to_string())
print(f"\n3. CGWB pairs within {MAX_KM} km: {pr.SID.nunique() if len(pr) else 0} stations, {len(pr)} season-pairs")
if len(pr):
    pr["diff"] = pr["dwlr_abs"] - pr["cgwb"]
    print(f"   |level| vs CGWB: median diff {pr['diff'].median():+.2f} m | MAD {(pr['diff'] - pr['diff'].median()).abs().median():.2f} m "
          f"| spearman r {pr[['dwlr_abs', 'cgwb']].corr(method='spearman').iloc[0, 1]:.2f}")
    print(f"   raw level vs CGWB depth spearman r {pr[['dwlr_raw', 'cgwb']].corr(method='spearman').iloc[0, 1]:+.2f} "
          f"(near -1 = raw is -depth)")
    print("   by district (median DWLR |level| vs median CGWB depth, m):")
    print(pr.groupby("DIST").agg(n=("diff", "size"), dwlr=("dwlr_abs", "median"), cgwb=("cgwb", "median"),
                                 med_diff=("diff", "median")).round(2).to_string())
    print("   by sign verdict:")
    print(pr.groupby("sign_verdict").agg(n=("diff", "size"), med_diff=("diff", "median"),
                                         abs_err=("diff", lambda x: x.abs().median())).round(2).to_string())
print(f"\nOutputs: {OUT}")
