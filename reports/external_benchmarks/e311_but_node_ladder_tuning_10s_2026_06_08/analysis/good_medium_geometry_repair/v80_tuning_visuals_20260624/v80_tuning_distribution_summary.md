# v80 Parameter Tuning and Distribution Snapshot

## Takeaways
- Best PTB-only test: `v80 lr1e4 20e` acc `0.9534`, G/M/B `0.881/0.939/0.987`.
- Best BUT-only test: `v80 default` acc `0.9273`, G/M/B `0.934/0.902/0.984`.
- New PTB `lr7.5e-5/30e` and `lr1e-4/30e` did not improve held-out test over `lr1e-4/20e`.
- New BUT `lr1.5e-4`/`lowaux` and two additional seeds did not beat the original v80 seed0.
- Therefore small LR/epoch/loss-weight tuning is not the main remaining lever; distribution/label boundary alignment is.

## Figures
- Raw train-main PCA: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\v80_tuning_visuals_20260624\v80_train_main_shared_pca.png`
- Clipped train-main PCA: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\v80_tuning_visuals_20260624\v80_train_main_shared_pca_clipped.png`
- Feature gap heatmap: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\v80_tuning_visuals_20260624\v80_train_main_key_feature_gap_heatmap_clipped.png`
- Training curves: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\v80_tuning_visuals_20260624\v80_training_curve_tuning.png`
- Self-test summary: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\v80_tuning_visuals_20260624\v80_selftest_tuning_summary.png`

## CSVs
- Exact feature gaps: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\v80_tuning_visuals_20260624\v80_train_main_key_feature_gap.csv`
- Training curve source: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\v80_tuning_visuals_20260624\v80_training_curves_source.csv`
- Self-test summary source: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\v80_tuning_visuals_20260624\v80_selftest_tuning_summary.csv`