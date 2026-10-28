"""
ONE SCRIPT: re-run the whole recharge pipeline on the AUDITED daily data (Q12 and Q123) + comparison table
=======================================================================================================
Does, for each version in VERSIONS:
  1. makes  <BASE>\run_<version>\DWLR_QC\  and copies into it:
       audit\dwlr_daily_depth_abs_<version>.csv  ->  dwlr_daily_depth_abs.csv
       dwlr_sign_smoothness.csv, regime_controls\geology_v3\station_geology_audit.csv
  2. runs wtf_mrc_recharge.py -> wtf_diagnostics.py -> final_results.py inside that folder
     (originals in <BASE>\DWLR_QC are NOT touched)
  3. prints one comparison table: ORIGINAL vs Q12 vs Q123 (medians + 95% bootstrap CI)
Needs recharge_audit.py to have been run first (it writes the two audited files).
Run in Jupyter from C:\\Users\\IITI:   %run run_all_audited.py
"""
import os
import sys
import shutil
import runpy
import numpy as np
import pandas as pd

# ======================================================================================== CONFIG
BASE = os.environ.get("TEST_BASE", r"C:\Users\IITI")                 # folder with the scripts and DWLR_QC
VERSIONS = ["Q123", "Q12"]                                            # Q123 = main, Q12 = sensitivity
SCRIPTS = ["wtf_mrc_recharge.py", "wtf_diagnostics.py", "final_results.py"]
FLOPY = os.path.join(BASE, "virtual_confined", "confined_wtf_results.csv")
SY = {"GEC_low_1%": 0.01, "GEC_high_3%": 0.03, "NAQUIM_4%": 0.04}
N_BOOT, SEED = 2000, 20261013
rng = np.random.default_rng(SEED)

QC0 = os.path.join(BASE, "DWLR_QC")
COPY = {os.path.join(QC0, "dwlr_sign_smoothness.csv"): "dwlr_sign_smoothness.csv",
        os.path.join(QC0, "regime_controls", "geology_v3", "station_geology_audit.csv"):
            os.path.join("regime_controls", "geology_v3", "station_geology_audit.csv")}
# locate each script: BASE, current folder, then BASE\** and Downloads (3 levels deep); newest copy wins
import glob
SEARCH = [BASE, os.getcwd(), os.path.join(os.path.expanduser("~"), "Downloads")]
PATH = {}
for s in SCRIPTS:
    hits = []
    for root in SEARCH:
        hits += [os.path.join(root, s)] + glob.glob(os.path.join(root, "*", s)) + glob.glob(os.path.join(root, "*", "*", s)) \
                + glob.glob(os.path.join(root, "*", "*", "*", s))
    hits = [h for h in dict.fromkeys(os.path.abspath(h) for h in hits)
            if os.path.isfile(h) and f"{os.sep}run_Q" not in h]
    if not hits:
        sys.exit(f"[STOP] {s} not found in {SEARCH} (or 3 sub-levels). Put it in {BASE} and re-run.")
    PATH[s] = max(hits, key=os.path.getmtime)
    print(f"using {s}: {PATH[s]}" + (f"   ({len(hits)} copies found, newest used)" if len(hits) > 1 else ""))
for v in VERSIONS:
    f = os.path.join(QC0, "audit", f"dwlr_daily_depth_abs_{v}.csv")
    if not os.path.exists(f):
        sys.exit(f"[STOP] {f} not found -> run recharge_audit.py first")
if os.path.exists(FLOPY):
    os.environ["TEST_FLOPY"] = FLOPY

home = os.getcwd()
done = {}
for v in VERSIONS:
    run = os.path.join(BASE, f"run_{v}")
    qc = os.path.join(run, "DWLR_QC")
    os.makedirs(os.path.join(qc, "regime_controls", "geology_v3"), exist_ok=True)
    shutil.copyfile(os.path.join(QC0, "audit", f"dwlr_daily_depth_abs_{v}.csv"), os.path.join(qc, "dwlr_daily_depth_abs.csv"))
    for src, dst in COPY.items():
        if os.path.exists(src):
            shutil.copyfile(src, os.path.join(qc, dst))
    print("\n" + "#" * 100 + f"\n#  VERSION {v}   (folder {run})\n" + "#" * 100)
    os.chdir(run)
    ok = True
    try:
        for s in SCRIPTS:
            print(f"\n----- running {s} -----")
            try:
                runpy.run_path(PATH[s], run_name="__main__")
            except SystemExit as e:
                if e.code not in (None, 0):
                    print(f"[{v}] {s} stopped: {e.code}"); ok = False; break
    finally:
        os.chdir(home)
    done[v] = ok


# ======================================================================================== comparison
def ci(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 3:
        return f"{np.median(x) if len(x) else np.nan:.3f} [n={len(x)}]"
    b = np.median(rng.choice(x, (N_BOOT, len(x))), axis=1)
    lo, hi = np.percentile(b, [2.5, 97.5])
    return f"{np.median(x):.3f} [{lo:.3f}, {hi:.3f}] n={len(x)}"


def load(folder):
    d = os.path.join(folder, "DWLR_QC", "wtf_mrc", "diagnostics", "diag_station.csv")
    y = os.path.join(folder, "DWLR_QC", "wtf_mrc", "recharge_station_years.csv")
    if not (os.path.exists(d) and os.path.exists(y)):
        return None
    D = pd.read_csv(d).drop_duplicates("SID").set_index("SID")
    Y = pd.read_csv(y)
    Y = Y[(Y.MRC == "PRIMARY") & Y.FULL.astype(bool)]
    Y["R"] = Y.BIANN_m / Y.GROSS_C_m
    D["BIANN_GC"] = Y.groupby("SID").R.median()
    return D


sets = {"ORIGINAL": load(BASE)}
for v in VERSIONS:
    sets[v] = load(os.path.join(BASE, f"run_{v}"))
print("\n" + "=" * 100)
print("VERDICT  run_all_audited")
print("=" * 100)
print("runs finished:", done)
for lab, D in sets.items():
    if D is None:
        print(f"\n{lab}: outputs not found"); continue
    print(f"\n{lab}: stations {len(D)} | {D.CLASS.value_counts().to_dict() if 'CLASS' in D else ''}")
    print(f"  GROSS_C (m/unit Sy)        {ci(D.GROSS_C)}")
    print(f"  Sy_100                     {ci(D.SY100)}")
    print(f"  LB / GROSS_C               {ci(D.LB_RATIO)}")
    print(f"  rain-free share (>noise)   {ci(D.RF_SHARE_AN)}")
    print(f"  BIANN / GROSS_C            {ci(D.BIANN_GC)}")
    if "CLASS" in D:
        for c, q in D.groupby("CLASS"):
            r = " | ".join(f"{k} {ci(100 * 1000 * sy * q.GROSS_C / q.RAIN)}" for k, sy in SY.items())
            print(f"  {c:<9} GROSS_C {ci(q.GROSS_C)} | Sy_100 {ci(q.SY100)}")
            print(f"            recharge % rain: {r}")
print(f"\nFigures/tables: {os.path.join(BASE, 'run_Q123', 'DWLR_QC', 'final')}  (main)  and run_Q12 (sensitivity)")
