# N7200 Medium Error Analysis

Selection note: original BUT and filtered-original metrics are report-only. This analysis uses N7200 node diagnostics on Clean/SemiClean node rows.

## Top Medium Error Dimensions
- `sqi_sSQI`: mean KS=0.643, median gap=0.73 IQR, missed>correct_q95=0.537, missed<correct_q05=0.001
- `sqi_kSQI`: mean KS=0.638, median gap=0.91 IQR, missed>correct_q95=0.436, missed<correct_q05=0.018
- `rms`: mean KS=0.595, median gap=1.19 IQR, missed>correct_q95=0.452, missed<correct_q05=0.001
- `amplitude_entropy`: mean KS=0.581, median gap=-0.83 IQR, missed>correct_q95=0.008, missed<correct_q05=0.412
- `hjorth_activity`: mean KS=0.575, median gap=1.32 IQR, missed>correct_q95=0.432, missed<correct_q05=0.002
- `std`: mean KS=0.575, median gap=1.14 IQR, missed>correct_q95=0.432, missed<correct_q05=0.002
- `ptp_p99_p01`: mean KS=0.573, median gap=1.03 IQR, missed>correct_q95=0.438, missed<correct_q05=0.028
- `qrs_prom_p90`: mean KS=0.552, median gap=0.56 IQR, missed>correct_q95=0.261, missed<correct_q05=0.017
- `low_amp_ratio`: mean KS=0.487, median gap=1.06 IQR, missed>correct_q95=0.366, missed<correct_q05=0.013
- `mean_abs`: mean KS=0.366, median gap=0.58 IQR, missed>correct_q95=0.119, missed<correct_q05=0.005
- `non_qrs_rms_ratio`: mean KS=0.363, median gap=-0.51 IQR, missed>correct_q95=0.019, missed<correct_q05=0.197
- `baseline_step`: mean KS=0.343, median gap=-0.44 IQR, missed>correct_q95=0.016, missed<correct_q05=0.176
- `template_corr`: mean KS=0.334, median gap=-0.10 IQR, missed>correct_q95=0.001, missed<correct_q05=0.007
- `band_0p3_1`: mean KS=0.328, median gap=-0.29 IQR, missed>correct_q95=0.014, missed<correct_q05=0.255

## Highest Miss Concentrations
- `bad_preserve_6703` `raw` ambiguous_type=`good_medium_low_purity`: n=139, miss_rate=0.791
- `bad_preserve_6703` `guard_pmed001` ambiguous_type=`good_medium_low_purity`: n=139, miss_rate=0.770
- `bad_preserve_6703` `guard_pmed002` ambiguous_type=`good_medium_low_purity`: n=139, miss_rate=0.770
- `medium_boost_118_190` `raw` ambiguous_type=`good_medium_low_purity`: n=139, miss_rate=0.770
- `medium_boost_118_190` `guard_pmed002` ambiguous_type=`good_medium_low_purity`: n=139, miss_rate=0.741
- `bad_preserve_6703` `guard_pmed0005` ambiguous_type=`good_medium_low_purity`: n=139, miss_rate=0.712
- `medium_boost_118_190` `guard_pmed001` ambiguous_type=`good_medium_low_purity`: n=139, miss_rate=0.698
- `best_acc_bad_guard_c4e3` `raw` ambiguous_type=`good_medium_low_purity`: n=139, miss_rate=0.691
- `medium_forward_3cad` `raw` ambiguous_type=`good_medium_low_purity`: n=139, miss_rate=0.683
- `dense_overlap_s1600` `raw` ambiguous_type=`good_medium_low_purity`: n=139, miss_rate=0.604
- `good_nudge_135_160` `raw` ambiguous_type=`good_medium_low_purity`: n=139, miss_rate=0.604
- `dense_overlap_s2400` `raw` ambiguous_type=`good_medium_low_purity`: n=139, miss_rate=0.568
- `dense_overlap_s2400` `guard_pmed001` ambiguous_type=`good_medium_low_purity`: n=139, miss_rate=0.540
- `dense_overlap_s2400` `guard_pmed002` ambiguous_type=`good_medium_low_purity`: n=139, miss_rate=0.540
- `best_acc_bad_guard_c4e3` `guard_pmed002` ambiguous_type=`good_medium_low_purity`: n=139, miss_rate=0.532
- `dense_overlap_s2400` `guard_pmed0005` ambiguous_type=`good_medium_low_purity`: n=139, miss_rate=0.511
- `best_acc_bad_guard_c4e3` `guard_pmed001` ambiguous_type=`good_medium_low_purity`: n=139, miss_rate=0.504
- `good_nudge_135_160` `guard_pmed002` ambiguous_type=`good_medium_low_purity`: n=139, miss_rate=0.489

## Artifacts
- Feature consensus: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\n7200_medium_error_analysis\medium_error_feature_consensus.csv`
- Group summary: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\n7200_medium_error_analysis\medium_error_group_summary.csv`
- PCA errors: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\n7200_medium_error_analysis\best_acc_bad_guard_c4e3_raw_medium_pca_errors.png`
- Guard PCA errors: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\n7200_medium_error_analysis\best_acc_bad_guard_c4e3_guard_pmed001_medium_pca_errors.png`
- Dense s1600 PCA errors: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\n7200_medium_error_analysis\dense_overlap_s1600_guard_pmed001_medium_pca_errors.png`
- GM boost PCA errors: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\n7200_medium_error_analysis\gm_boost_115_170_guard_pmed001_medium_pca_errors.png`
- Medium boost PCA errors: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\n7200_medium_error_analysis\medium_boost_115_182_guard_pmed001_medium_pca_errors.png`
- Current best PCA errors: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\n7200_medium_error_analysis\med_mid_118_176_guard_pmed0005_medium_pca_errors.png`