"""
DWLR QUALITY CHECK  —  UPGWD Bundelkhand (read + flag only, nothing is deleted)
==============================================================================
Why: the audit table shows problems the first audit did not catch
  - -999 sentinel values counted as real data
  - sign flips inside one station (e.g. median -23.7, max +25.5)
  - flat-line sensors (all 0.0 but '100% coverage')
  - spikes (e.g. -79.7 m in a ~-11 m well)
This script flags each problem per station, builds a cleaned DAILY series,
and plots a few stations so YOU can see what the sign problem really is.


Requires: pandas numpy matplotlib
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =====================================================================
# CONFIG
# =====================================================================
FILE = r"EDIT\ME\UPGWD_Bundelkhand_DWLR.csv"
OUT = os.path.join(os.path.dirname(FILE), "DWLR_QC")
SENTINEL = -900          # anything <= this is a missing-value code
FLAT_RUN = 12            # >= 12 identical consecutive readings (~3 days at 6 h) = stuck sensor
SPIKE_M = 5.0            # |value - 7-reading rolling median| > 5 m = spike
MIN_DAYS = 365           # station needs >= 1 year of clean daily data to be "usable"
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(FILE)
print("COLUMNS:", list(df.columns))


def pick(cands):
    for c in df.columns:
        if c.strip().lower() in cands:
            return c
    return None


# exact UPGWD column names
C_DIST, C_STN, C_BLK = "District", "Station", "Block"
C_TIME = "Data Acquisition Time"
C_LVL = "Groundwater Level Telemetry 6 Hourly (meter)"
C_LAT, C_LON, C_RL = "Latitude", "Longitude", "RL_MSL"
for c in (C_DIST, C_STN, C_BLK, C_TIME, C_LVL, C_LAT, C_LON, C_RL):
    if c not in df.columns:
        raise SystemExit(f"! column missing: {c!r}")
print("sample raw time values:", df[C_TIME].dropna().astype(str).head(3).tolist())
print("RL_MSL sample:", df[C_RL].dropna().head(3).tolist())

df = df.rename(columns={C_DIST: "DIST", C_STN: "STN", C_TIME: "T", C_LVL: "LVL"})
raw_t = df["T"].astype(str)
df["T"] = pd.to_datetime(raw_t, format="%d-%m-%Y %H:%M", errors="coerce")   # e.g. 20-01-2024 16:00
if df["T"].isna().mean() > 0.05:                                            # mixed formats -> fallback
    df["T"] = df["T"].fillna(pd.to_datetime(raw_t, errors="coerce", dayfirst=True))
print(f"unparsed timestamps: {100 * df['T'].isna().mean():.2f}%")
df["LVL"] = pd.to_numeric(df["LVL"], errors="coerce")
df["SID"] = (df["DIST"].astype(str).str.strip() + "|" + df[C_BLK].astype(str).str.strip() + "|" +
             df["STN"].astype(str).str.strip())
df = df.dropna(subset=["T"]).sort_values(["SID", "T"]).drop_duplicates(["SID", "T"])

# =====================================================================
# FLAGS
# =====================================================================
df["F_SENT"] = df["LVL"] <= SENTINEL
df.loc[df["F_SENT"], "LVL"] = np.nan


def run_len(s):
    g = (s != s.shift()).cumsum()
    return s.groupby(g).transform("size")


df["F_FLAT"] = df.groupby("SID")["LVL"].transform(run_len).ge(FLAT_RUN) & df["LVL"].notna()
med = df.groupby("SID")["LVL"].transform(lambda s: s.abs().rolling(7, center=True, min_periods=3).median())
df["F_SPIKE"] = (df["LVL"].abs() - med).abs() > SPIKE_M
# sign flip: sign changes but |value| barely changes  -> same depth written with opposite sign
prev = df.groupby("SID")["LVL"].shift()
df["F_SIGNFLIP"] = (np.sign(df["LVL"]) != np.sign(prev)) & ((df["LVL"].abs() - prev.abs()).abs() < 1.0) \
                   & df["LVL"].notna() & prev.notna() & (df["LVL"] != 0)

rows = []
for sid, g in df.groupby("SID"):
    v = g["LVL"].dropna()
    rows.append({
        "SID": sid, "DIST": g["DIST"].iloc[0], "n": len(g),
        "first": g["T"].min().date(), "last": g["T"].max().date(),
        "span_yr": (g["T"].max() - g["T"].min()).days / 365.25,
        "med_step_h": g["T"].diff().dt.total_seconds().median() / 3600,
        "pct_sentinel": 100 * g["F_SENT"].mean(),
        "pct_flat": 100 * g["F_FLAT"].mean(),
        "pct_spike": 100 * g["F_SPIKE"].mean(),
        "n_signflip": int(g["F_SIGNFLIP"].sum()),
        "pct_positive": 100 * (v > 0).mean() if len(v) else np.nan,
        "pct_negative": 100 * (v < 0).mean() if len(v) else np.nan,
        "med_abs_level": v.abs().median() if len(v) else np.nan,
        "lat": pd.to_numeric(g[C_LAT], errors="coerce").median(),
        "lon": pd.to_numeric(g[C_LON], errors="coerce").median(),
        "rl_msl": pd.to_numeric(g[C_RL], errors="coerce").median(),
        "n_coord_versions": g[[C_LAT, C_LON]].drop_duplicates().shape[0],
    })
st = pd.DataFrame(rows)

# sign pattern per station
def sign_class(r):
    if r.pct_positive >= 98: return "all +"
    if r.pct_negative >= 98: return "all -"
    if r.n_signflip >= 3: return "flip (same depth, opposite sign)"
    return "mixed (real crossing or offset)"
st["sign_class"] = st.apply(sign_class, axis=1)

# =====================================================================
# CLEAN DAILY SERIES  (depth candidate = |level|; NOT final until you check the plots)
# =====================================================================
ok = df[~(df.F_FLAT | df.F_SPIKE) & df.LVL.notna()].copy()
ok["DEPTH_ABS"] = ok["LVL"].abs()
daily = (ok.set_index("T").groupby("SID")["DEPTH_ABS"].resample("D").median()
         .reset_index().dropna())
dn = daily.groupby("SID").size().rename("clean_days")
st = st.merge(dn, left_on="SID", right_index=True, how="left").fillna({"clean_days": 0})
st["usable"] = (st.clean_days >= MIN_DAYS) & (st.pct_flat < 50)

st.to_csv(os.path.join(OUT, "dwlr_station_flags.csv"), index=False)
daily.to_csv(os.path.join(OUT, "dwlr_daily_depth_abs.csv"), index=False)

# =====================================================================
# PLOTS: 2 stations from each sign class, raw vs |level|
# =====================================================================
show = st.groupby("sign_class").head(2)["SID"].tolist()[:8]
fig, axs = plt.subplots(len(show), 1, figsize=(11, 2.2 * len(show)), sharex=False)
axs = np.atleast_1d(axs)
for ax, sid in zip(axs, show):
    g = df[df.SID == sid]
    ax.plot(g["T"], g["LVL"], lw=0.5, c="0.6", label="raw")
    d = daily[daily.SID == sid]
    ax.plot(d["T"], -d["DEPTH_ABS"], lw=1, c="C3", label="-|level| daily")
    ax.set_title(f"{sid}  [{st.loc[st.SID == sid, 'sign_class'].iloc[0]}]", fontsize=8)
    ax.axhline(0, c="k", lw=0.5)
axs[0].legend(fontsize=7)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "Fig_DWLR_sign_check.png"), dpi=200); plt.close()

# =====================================================================
# VERDICT
# =====================================================================
print("\n" + "=" * 72)
print("VERDICT")
print("=" * 72)
print(f"coords missing for {int(st.lat.isna().sum())} stations | stations with >1 coord pair: "
      f"{int((st.n_coord_versions > 1).sum())} | RL_MSL median {st.rl_msl.median():.1f} m")
print(f"stations {len(st)} | "
      f"median step {st.med_step_h.median():.1f} h | date range {df['T'].min().date()} -> {df['T'].max().date()}")
print(f"sentinel rows {int(df.F_SENT.sum())} | flat rows {int(df.F_FLAT.sum())} | spike rows {int(df.F_SPIKE.sum())} "
      f"| sign-flip events {int(df.F_SIGNFLIP.sum())}")
print("sign classes:", st.sign_class.value_counts().to_dict())
print("stations mostly flat-lined (>50%):", int((st.pct_flat > 50).sum()))
u = st[st.usable]
print(f"USABLE (>= {MIN_DAYS} clean days, not flat): {len(u)}")
print(u.groupby("DIST").agg(n=("SID", "size"), med_span_yr=("span_yr", "median"),
                            med_depth_m=("med_abs_level", "median")).round(2).to_string())
print("monsoons covered by usable stations (Jun-Oct with >=100 clean days):")
daily["Y"] = daily["T"].dt.year
mons = daily[daily["T"].dt.month.between(6, 10)].groupby(["SID", "Y"]).size()
print((mons >= 100).groupby(level="Y").sum().to_string())
print(f"\nOutputs: {OUT}  (LOOK at Fig_DWLR_sign_check.png before trusting |level|)")
