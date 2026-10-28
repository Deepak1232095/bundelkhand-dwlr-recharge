# Groundwater recharge from daily water levels in Bundelkhand (Uttar Pradesh, India)

Code for the paper:

> Mishra, D., & Goyal, M. K. (in review). *Resolving Groundwater Recharge From Daily Water Levels: Episodic Dynamics
> and Specific-Yield Uncertainty in a Semi-Arid Region.* Submitted to *Water Resources Research*.

The scripts estimate the June–October water-table rise per unit specific yield (ΔH) from daily telemetric
groundwater levels (DWLR) using a master-recession-curve water-table-fluctuation method (MRC-WTF). They then convert
ΔH to recharge over the GEC (1997) specific-yield norms, characterise recharge episodes, compare the result with
fixed-date (May/November) rises and the GEC-2024 district assessment, and map station values. The repository also
contains the synthetic verification, the MODFLOW 6 sensitivity experiment and the robustness tests reported in the
Supporting Information.

Each script prints a `VERDICT` block that summarises its results. Random seeds are fixed (20261013 unless stated).

## Data (not included)

The data are not redistributed here. Each source and where the scripts expect it:

| Data | Source | Path used in the scripts |
|---|---|---|
| DWLR 6-hourly groundwater levels, 276 stations | Ground Water Department, Uttar Pradesh (on request) | `UPGWD_Bundelkhand_DWLR.csv` (`FILE` / `DWLR` / `RAW` in the scripts) |
| Daily gridded rainfall, 0.25° (Pai et al., 2014) | India Meteorological Department | `E:\up_bundel\up_bundel_rainfall_1990_2025.nc` (`RAIN`, or env `TEST_RAIN`) |
| Daily Tmax / Tmin, 1° (only for `pastas_dwlr.py`) | India Meteorological Department | `E:\up_bundel\up_bundel_tmax_…nc`, `…tmin…nc` |
| CGWB pre- and post-monsoon water levels | Central Ground Water Board / India-WRIS | `UP_GW_wide_final.csv` (`CGWB`, or env `TEST_CGWB`) |
| Lithology 1:50,000 | Geological Survey of India (Bhukosh) | `E:\up_bundel\Lithology\**\LITHOLOGY_DIST_*.shp` |
| District boundaries | GADM | `E:\up_bundel\up_bundel_boxes\up_5dist.shp` (field `NAME_2`) |
| River network | national river network shapefile | `E:\river_network_shape\River_Network.shp` (env `TEST_RIVER`) |
| GEC-2024 district recharge and area | CGWB & GWD-UP (2024), *Dynamic Ground Water Resources of Uttar Pradesh* | typed as constants in `gec_comparison.py` and `table3.py` |

**Before running:** edit the paths at the top of each script (section `CONFIG`), or set the environment variables
named there. Most scripts take the working folder from `TEST_BASE` (default `C:\Users\IITI`). Scripts that use the
relative folder `DWLR_QC\` must be run from that working folder, for example
`%run scripts/03_recharge/wtf_mrc_recharge.py` from the folder that contains `DWLR_QC\`.

## Run order

| Step | Script | What it does | Main outputs |
|---|---|---|---|
| 1 | `01_quality_control/dwlr_qc.py` | Reads 6-hourly DWLR, removes sentinels, flat lines and spikes, writes daily depths | `DWLR_QC/dwlr_daily_depth_abs.csv` |
| 2 | `01_quality_control/dwlr_sign_check.py` | Sign/coherence screen against nearby CGWB wells (102 core stations) | `DWLR_QC/dwlr_sign_smoothness.csv` |
| 3 | `02_geology/regime_geology.py` | Assigns each station to alluvium or hard rock from GSI lithology | `DWLR_QC/regime_controls/geology_v3/station_geology_audit.csv` |
| 4 | `01_quality_control/recharge_audit.py` | Day-level audit rules Q1–Q3 (codes, impossible depths, telemetry zeros, one-day spikes) | `DWLR_QC/audit/…Q12.csv`, `…Q123.csv`, `audit_station.csv` |
| 5 | `01_quality_control/run_all_audited.py` | Re-runs steps 6–8 on the audited data (Q123 = main, Q12 = sensitivity) | `run_Q123/DWLR_QC/…`, `run_Q12/DWLR_QC/…` |
| 6 | `03_recharge/wtf_mrc_recharge.py` | MRC-WTF: master recession curve, daily rise, noise correction, ΔH, episodes | `wtf_mrc/mrc_parameters.csv`, `recharge_station_years.csv`, `recharge_episodes.csv` |
| 7 | `03_recharge/wtf_diagnostics.py` | Noise-only null, rain-free episodes, synchrony test, Sy₁₀₀ | `wtf_mrc/diagnostics/diag_station.csv`, `diag_episodes.csv` |
| 8 | `03_recharge/final_results.py` | Station table T1, class/district summaries, episode and biannual figures | `final/T1_station_recharge.csv`, figures |
| 9 | `03_recharge/monthly_recharge.py` | Monthly rise and recharge, June–October | `paper3_final/T6_monthly_*.csv`, monthly figure |
| 10 | `05_comparison_and_maps/gec_comparison.py` | Sy that reconciles ΔH with GEC-2024 district recharge | `paper3_final/T5_GEC_consistency.csv`, `T5b_station_Sy_match.csv` |
| 11 | `05_comparison_and_maps/recharge_maps.py` | District polygons and district medians | `paper3_final/districts_from_lithology.gpkg`, `T7_*.csv` |
| 12 | `05_comparison_and_maps/within_district_maps.py` | IDW surfaces and leave-one-out skill per district | `paper3_final/T8_within_district_LOOCV.csv` |
| 13 | `05_comparison_and_maps/combined_maps.py` | CGWB recharge-equivalent points and their skill test | `paper3_final/T9_combined_validation.csv`, `T9b_*.csv` |
| 14 | `05_comparison_and_maps/river_recharge_map.py` | Regional map with rivers; recharge vs distance to river | `paper3_final/T10_*.csv`, figures |
| 15 | `04_verification/synthetic_verification.py` | Synthetic hydrographs with planted recharge (self-contained) | `TableS2_synthetic_verification.csv`, `TableS2b_design_variants.csv` |
| 16 | `04_verification/flopy_confined_wtf.py` | MODFLOW 6 cross-section: apparent rise at a pumped semi-confined piezometer (needs `mf6.exe`) | `virtual_confined/confined_wtf_results.csv` |
| 17 | `04_verification/dwlr_earth_tide.py` | Earth-tide test for confined response in 6-hourly records | `DWLR_QC/earth_tide/earth_tide_stations.csv` |
| 18 | `04_verification/robustness_tests.py` | Processing variants, noise correction, field-noise synthetic test, Monte Carlo total uncertainty, Sy norms, episode tests, station selection, semivariogram, rainfall anomaly | `paper3_final/reviewer/TableR1–R7.csv`, `FigS6_reviewer_tests.png` |
| 19 | `06_figures_tables/*`, `07_supporting_information/*` | Final figures and tables (see below) | `paper3_final/…` |

Step 3 needs `station_metrics_and_controls.csv`, which is written by `02_geology/upstream/regime_controls.py`.
That script in turn uses `dwlr_testA_robust.py` → `dwlr_schedule_experiment.py` and `pastas_dwlr.py`. These upstream
scripts come from a companion analysis of hydrograph behaviour; for this paper only the geology class written by
`regime_geology.py` is used.

## Figures and tables in the paper

| Item | Script |
|---|---|
| Fig. 1 Study area | prepared in ArcMap (not scripted) |
| Fig. 2 Workflow | `06_figures_tables/workflow_figure.js` (pptxgenjs, editable PowerPoint) |
| Fig. 3 DWLR stations | `06_figures_tables/fig2_final.py` |
| Fig. 4 Recharge vs specific yield | `06_figures_tables/fig3_final.py` |
| Fig. 5 Monthly recharge | `03_recharge/monthly_recharge.py` |
| Fig. 6 Recharge episodes | `03_recharge/final_results.py` (`Fig3_episodes.png`) |
| Fig. 7 Biannual vs daily-resolved rise | `03_recharge/final_results.py` (`Fig4_biannual_vs_MRC.png`) |
| Fig. 8 Sy reconciling ΔH with GEC-2024 | `05_comparison_and_maps/gec_comparison.py`, final design in `06_figures_tables/fig6_options.py` |
| Fig. 9 Regional recharge map with rivers | `05_comparison_and_maps/river_recharge_map.py` (`Fig12_recharge_rivers.png`) |
| Fig. 10 Within-district variation | `06_figures_tables/within_district_maps_v2.py` (or `fig10b_within_district_DWLR_v2.py`) |
| Fig. 11 Recharge vs distance to river | `05_comparison_and_maps/river_recharge_map.py` (`Fig12b_…png`) |
| Table 2 | `06_figures_tables/table2.py`, `table2_districts.py` |
| Table 3 | `06_figures_tables/table3.py` |
| Figs. S1–S5, Tables S1–S4 | `07_supporting_information/si_figures_tables.py` |
| Fig. S6, Tables S5–S11 | `04_verification/robustness_tests.py` |
| Fig. S7, Table S12 | `07_supporting_information/si_audit_figS7_tableS12.py` |
| Numbers quoted in the Results | `06_figures_tables/results_numbers.py` |

File names inside the scripts (for example `Fig2_station_map.png`, `Fig3_recharge_vs_Sy.png`) follow the order of an
earlier draft; the table above gives the figure number used in the paper.

## Requirements

Python ≥ 3.10 with the packages in `requirements.txt`. `flopy_confined_wtf.py` also needs the MODFLOW 6 executable
(`mf6`). `workflow_figure.js` needs Node.js and `pptxgenjs`.

## Licence

MIT (see `LICENSE`). If you use the code, please cite the paper.
