"""
SUPPORTING INFORMATION: Figure S7 (audit hydrograph) and Table S12 (stations changed by the audit) + station counts
=================================================================================================================
Reads existing outputs only (nothing in the analysis is changed):
  DWLR_QC\\dwlr_daily_depth_abs.csv            unaudited daily depths   (dwlr_qc.py)
  run_Q123\\DWLR_QC\\dwlr_daily_depth_abs.csv   audited daily depths     (recharge_audit.py -> run_all_audited.py)
  run_Q123\\DWLR_QC\\dwlr_sign_smoothness.csv   station universe         (dwlr_sign_check.py)
  DWLR_QC\\audit\\audit_station.csv             per-station audit result (recharge_audit.py)
  T1_station_recharge.csv                       station rise and class   (final_results.py)
Output: paper3_final\\SI\\FigS7_audit_hydrograph.png/.pdf, TableS12_audit_stations.csv
Run   :  %run si_audit_figS7_tableS12.py      Requires pandas numpy matplotlib
"""
import os
import re
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.environ.get("TEST_BASE", r"C:\Users\IITI")
FINAL = os.path.join(BASE, "paper3_final")
RUN = os.path.join(BASE, "run_Q123", "DWLR_QC")
OUT_SI = os.path.join(FINAL, "SI")


def first(*paths):
    return next((p for p in paths if p and os.path.exists(p)), None)


T1P = first(os.path.join(RUN, "final", "T1_station_recharge.csv"), os.path.join(FINAL, "T1_station_recharge.csv"))
RAW = first(os.path.join(BASE, "DWLR_QC", "dwlr_daily_depth_abs.csv"))
AUD = first(os.path.join(RUN, "dwlr_daily_depth_abs.csv"))
SIGN = first(os.path.join(RUN, "dwlr_sign_smoothness.csv"))
AUDIT = first(os.path.join(BASE, "DWLR_QC", "audit", "audit_station.csv"), os.path.join(RUN, "audit", "audit_station.csv"))
FIG_S6 = first(os.path.join(FINAL, "reviewer", "FigS6_reviewer_tests.png"))
SY = {"HARD_ROCK": 0.03, "ALLUVIUM": 0.06}
N_RECEIVED = 276                    # stations received from GWD-UP (Section 2.2)
CHANGE_MIN = 0.10                   # Table S12: |change of ΔH| > 10 %
DISTS = ["HAMIRPUR", "JALAUN", "JHANSI", "LALITPUR", "MAHOBA"]
os.makedirs(OUT_SI, exist_ok=True)
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                     "font.size": 7.5, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                     "legend.frameon": False, "legend.fontsize": 7, "pdf.fonttype": 42, "savefig.dpi": 600,
                     "savefig.bbox": "tight"})
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5e0"
LOG = []


def say(s=""):
    LOG.append(str(s))



if T1P is None:
    sys.exit("[STOP] T1_station_recharge.csv not found")
T = pd.read_csv(T1P).drop_duplicates("SID")
TOK = {}
# ======================================================================================== 3  station count
say("\n3  STATION COUNT")
ids_sign = set(pd.read_csv(SIGN, usecols=["SID"]).SID) if SIGN else set()
ids_aud = set(pd.read_csv(AUD, usecols=["SID"]).SID) if AUD else set()
ids_raw = set(pd.read_csv(RAW, usecols=["SID"]).SID) if RAW else set()
nst = len(ids_sign | ids_aud)
TOK["NSTAT"] = str(nst)
TOK["NOTE_NSTAT"] = "" if nst == N_RECEIVED else f" Daily records were usable at {nst} of them."
say(f"  sign file {len(ids_sign)} | audited daily {len(ids_aud)} | union {nst} | raw daily {len(ids_raw)} | received {N_RECEIVED}")
say(f"  -> «NSTAT» = {nst}; note in Section 2.2: '{TOK['NOTE_NSTAT'].strip() or '(none, counts agree)'}'")
if ids_raw and len(ids_raw) != nst:
    say(f"  stations in raw but not in sign/audited files: {sorted(ids_raw - (ids_sign | ids_aud))[:5]}")

# ======================================================================================== 4  Fig S7
say("\n4  FIG S7  largest code value")
FIG_S7 = None
if RAW and AUD:
    raw = pd.read_csv(RAW, parse_dates=["T"])
    aud = pd.read_csv(AUD, parse_dates=["T"])
    pool = raw[raw.SID.isin(set(T.SID))] if raw.SID.isin(set(T.SID)).any() else raw
    sid = pool.loc[pool.DEPTH_ABS.idxmax(), "SID"]
    r = raw[raw.SID == sid].groupby("T").DEPTH_ABS.median().sort_index()
    a = aud[aud.SID == sid].groupby("T").DEPTH_ABS.median().sort_index()
    removed = r[~r.index.isin(a.index)]
    fig, axs = plt.subplots(2, 1, figsize=(19 / 2.54, 19 / 2.54 * 0.55), sharex=True)
    for ax, s_, t in ((axs[0], r, "a"), (axs[1], a, "b")):
        for y in sorted(set(s_.index.year)):
            ax.axvspan(pd.Timestamp(y, 6, 1), pd.Timestamp(y, 10, 31), color=GRID, lw=0, zorder=0)
        ax.plot(s_.index, s_.values, lw=0.7, color=INK, zorder=2)
        ax.invert_yaxis(); ax.set_ylabel("Depth below ground (m)")
        ax.text(-0.01, 1.03, t, transform=ax.transAxes, fontweight="bold", fontsize=8.5, ha="right")
    axs[0].plot(removed.index, removed.values, "o", ms=3, color="#eb6834", zorder=3, label=f"removed by Q1–Q3 ({len(removed)} days)")
    axs[0].set_yscale("log"); axs[0].invert_yaxis() if not axs[0].yaxis_inverted() else None
    axs[0].legend(loc="lower left"); axs[0].set_title("Unaudited daily depth (log scale)", loc="left", fontsize=7.5)
    axs[1].set_title("Audited daily depth used in the analysis", loc="left", fontsize=7.5)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_SI, f"FigS7_audit_hydrograph.{ext}"))
    plt.close(fig)
    FIG_S7 = os.path.join(OUT_SI, "FigS7_audit_hydrograph.png")
    say(f"  station {sid} | max raw depth {r.max():.1f} m on {r.idxmax().date()} | audited median depth {a.median():.1f} m"
        f" | days removed {len(removed)}")
    say(f"  manuscript says: 298 m code value, true depth about 8.3 m at a Jalaun station -> "
        f"{'CONSISTENT' if str(sid).upper().startswith('JALAUN') and r.max() > 250 else 'CHECK the sentence in Section 4.1'}")
else:
    say("  SKIPPED - need DWLR_QC\\dwlr_daily_depth_abs.csv and run_Q123\\DWLR_QC\\dwlr_daily_depth_abs.csv")

# ======================================================================================== 5  Table S12
say("\n5  TABLE S12  stations changed by the audit")
S12 = None
if AUDIT:
    A = pd.read_csv(AUDIT).drop_duplicates("SID")
    A = A[A.SID.isin(set(T.SID))].copy()
    A["chg"] = (A.GC_clean - A.GC_recalc) / A.GC_recalc
    sel = A[A.chg.abs() > CHANGE_MIN].sort_values("chg")
    cls = T.set_index("SID").CLASS
    S12 = pd.DataFrame({
        "Station": [str(s).split("|")[-1].strip() for s in sel.SID],
        "District": [str(s).split("|")[0].title() for s in sel.SID],
        "Class": [{"ALLUVIUM": "Alluvium", "HARD_ROCK": "Hard rock"}.get(cls.get(s), "–") for s in sel.SID],
        "Q1 days": sel.Q1_code.astype(int).values, "Q2 days": sel.Q2_deep.astype(int).values,
        "Q3 days": sel.Q3_step.astype(int).values,
        "ΔH before audit (m)": sel.GC_recalc.round(2).values, "ΔH after audit (m)": sel.GC_clean.round(2).values,
        "Change (%)": (100 * sel.chg).round(0).astype(int).values})
    S12.to_csv(os.path.join(OUT_SI, "TableS12_audit_stations.csv"), index=False, encoding="utf-8-sig")
    say(f"  stations in T1 with audit info {len(A)} | |change| > {100 * CHANGE_MIN:.0f}%: {len(S12)}")
    say("  " + S12.to_string(index=False).replace("\n", "\n  "))
else:
    say("  SKIPPED - need DWLR_QC\\audit\\audit_station.csv (written by recharge_audit.py)")


print("=" * 90); print("VERDICT  si_audit_figS7_tableS12"); print("=" * 90)
print("\n".join(LOG))
