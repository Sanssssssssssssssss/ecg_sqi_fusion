# N7200 Current Best Error Gap

Uses only N7200 node diagnostics and Clean/SemiClean target rows; original BUT remains report-only.

Best diagnostic mode: `medium_guarded_pmed0005` acc=0.936432, good=0.931389, medium=0.922083, bad=0.970617.

Best threshold sweep acc=0.936432, good=0.931389, medium=0.922083, bad=0.970617, flips=1928.

## Top Groups
- bad original_region=`near_bad_boundary`: n=120, miss_rate=1.000
- bad ambiguous_type=`bad_outlier`: n=120, miss_rate=1.000
- medium ambiguous_type=`good_medium_low_purity`: n=139, miss_rate=0.266
- medium original_region=`outlier_low_confidence`: n=720, miss_rate=0.106
- medium clean_tier=`ambiguous_boundary`: n=5230, miss_rate=0.101
- medium original_region=`good_medium_overlap`: n=4501, miss_rate=0.100
- medium ambiguous_type=`good_medium_boundary`: n=4501, miss_rate=0.100
- good original_region=`good_medium_overlap`: n=5040, miss_rate=0.089
- good ambiguous_type=`good_medium_boundary`: n=5040, miss_rate=0.089
- good clean_tier=`ambiguous_boundary`: n=5760, miss_rate=0.085
- medium nearest_other_class=`good`: n=7187, miss_rate=0.078
- good nearest_other_class=`medium`: n=7200, miss_rate=0.069

## Top Feature Gaps
- good `amplitude_entropy`: KS=0.727, median_gap=1.36 IQR
- good `low_amp_ratio`: KS=0.709, median_gap=-1.21 IQR
- good `sqi_sSQI`: KS=0.605, median_gap=-1.17 IQR
- good `sqi_kSQI`: KS=0.568, median_gap=-0.71 IQR
- good `baseline_step`: KS=0.525, median_gap=1.21 IQR
- good `non_qrs_rms_ratio`: KS=0.440, median_gap=0.47 IQR
- good `qrs_prom_p90`: KS=0.439, median_gap=-0.46 IQR
- good `mean_abs`: KS=0.435, median_gap=0.73 IQR
- good `qrs_prom_median`: KS=0.380, median_gap=-0.08 IQR
- good `rms`: KS=0.349, median_gap=-0.82 IQR
- good `lf_ratio`: KS=0.316, median_gap=0.18 IQR
- good `band_0p3_1`: KS=0.316, median_gap=0.18 IQR
- good `hjorth_activity`: KS=0.310, median_gap=-0.66 IQR
- good `std`: KS=0.310, median_gap=-0.69 IQR

## Figures
![confusion PCA](E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\n7200_current_best_error_gap\current_best_confusion_pca.png)
![medium PCA](E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\n7200_current_best_error_gap\current_best_medium_pca_errors.png)
![good PCA](E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\n7200_current_best_error_gap\current_best_good_pca_errors.png)