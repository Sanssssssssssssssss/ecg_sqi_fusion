# P20 Original-Test Failure Deep Dive

Candidate: `predtop20_sqiquery_subject111_impulsebad_dual_p20` raw mode.

## Error Counts

- `correct`: 6970
- `medium_to_good`: 667
- `good_to_medium`: 512
- `bad_outlier_missed`: 288
- `nonbad_to_bad`: 28
- `bad_core_missed`: 12

## Largest Feature Gaps

### good_to_medium
- `pc1` KS=0.826, median good_to_medium=-2.354, vs correct_good=-5.47, delta=3.115
- `flatline_ratio` KS=0.824, median good_to_medium=0.1705, vs correct_good=0.4051, delta=-0.2346
- `pca_margin` KS=0.799, median good_to_medium=-0.702, vs correct_good=1.849, delta=-2.551
- `boundary_confidence` KS=0.684, median good_to_medium=0.3058, vs correct_good=0.6076, delta=-0.3018
- `knn_label_purity` KS=0.630, median good_to_medium=0.6, vs correct_good=1, delta=-0.4
- `sample_entropy_proxy` KS=0.544, median good_to_medium=0.4078, vs correct_good=0.2218, delta=0.1861
- `non_qrs_diff_p95` KS=0.543, median good_to_medium=0.03757, vs correct_good=0.01898, delta=0.01859
- `fatal_or_score` KS=0.522, median good_to_medium=1, vs correct_good=0.8575, delta=0.1425

### medium_to_good
- `pca_margin` KS=0.810, median medium_to_good=-0.1611, vs correct_medium=1.832, delta=-1.993
- `boundary_confidence` KS=0.805, median medium_to_good=0.06867, vs correct_medium=0.588, delta=-0.5193
- `pc1` KS=0.803, median medium_to_good=-4.933, vs correct_medium=-0.6712, delta=-4.261
- `knn_label_purity` KS=0.754, median medium_to_good=0.03333, vs correct_medium=0.8667, delta=-0.8333
- `flatline_ratio` KS=0.753, median medium_to_good=0.4123, vs correct_medium=0.1081, delta=0.3042
- `non_qrs_diff_p95` KS=0.700, median medium_to_good=0.01523, vs correct_medium=0.08822, delta=-0.07299
- `pc3` KS=0.689, median medium_to_good=-1.633, vs correct_medium=1.976, delta=-3.608
- `band_15_30` KS=0.655, median medium_to_good=0.01909, vs correct_medium=0.188, delta=-0.1689

### bad_outlier_missed
- `pc1` KS=1.000, median bad_outlier_missed=-4.227, vs bad_core_correct=9.043, delta=-13.27
- `pc2` KS=1.000, median bad_outlier_missed=11.51, vs bad_core_correct=-0.9928, delta=12.51
- `pca_margin` KS=1.000, median bad_outlier_missed=-6.197, vs bad_core_correct=5.39, delta=-11.59
- `boundary_confidence` KS=1.000, median bad_outlier_missed=0.02308, vs bad_core_correct=0.3907, delta=-0.3676
- `knn_label_purity` KS=1.000, median bad_outlier_missed=0, vs bad_core_correct=0.9667, delta=-0.9667
- `flatline_ratio` KS=1.000, median bad_outlier_missed=0.4135, vs bad_core_correct=0.008807, delta=0.4047
- `sqi_basSQI` KS=1.000, median bad_outlier_missed=0.624, vs bad_core_correct=0.9803, delta=-0.3564
- `qrs_slope_median` KS=1.000, median bad_outlier_missed=0.3482, vs bad_core_correct=2.819, delta=-2.471

## Files

- PCA: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\p20_original_failure_deep_dive\p20_original_failure_pca.png`
- Boxplots: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\p20_original_failure_deep_dive\p20_original_failure_feature_boxplots.png`
- Feature gaps: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\p20_original_failure_deep_dive\p20_original_failure_feature_gaps.csv`
- Counts: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\p20_original_failure_deep_dive\p20_original_failure_counts.csv`