# Data v1 Fit Report

> Archive note: this is a frozen fit-report snapshot kept for audit context.
> Current runnable outputs are under `outputs/transformer/` and Chapter 4
> evidence outputs are under `outputs/transformer/supplemental/`.
> Use `python -m src.transformer_pipeline.run_all --run --train E31` for the
> current mainline.

- policy: `v116_gapfill_dual_goodorig_nm40_ms10_smc_s20260876`
- protocol path: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\clean_but_protocols\v116_gapfill_dual_goodorig_nm40_ms10_smc_s20260876`
- split path: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_sqi_conformer\rh_splits\v116_gapfill_dual_goodorig_nm40__k1_s20260876\fold0`
- selector runtime: CPU numpy/sklearn; `--device auto` is recorded but not used by the current data selector.

## Protocol Counts

v116_candidate_type  original_but  but_native_morph  ptb_morph  clean_style
class_name                                                                 
good                        10530                 0          0            0
medium                       6449              1616       2424           41
bad                          1656              3514       5271           89

## Protocol Percent By Class

v116_candidate_type  original_but  but_native_morph  ptb_morph  clean_style
class_name                                                                 
good                       100.00              0.00       0.00         0.00
medium                      61.24             15.35      23.02         0.39
bad                         15.73             33.37      50.06         0.85

## Generated Gap Composition

v116_candidate_type  but_native_morph  ptb_morph  clean_style  but_native_morph_pct  ptb_morph_pct  clean_style_pct
class_name                                                                                                         
medium                           1616       2424           41                  39.6           59.4              1.0
bad                              3514       5271           89                  39.6           59.4              1.0

## Split Counts

class_name  good  medium   bad
split                         
train       8310    8310  8310
val         1053     618   178
test        1053     632   164

## Train Composition

v116_candidate_type  original_but  but_native_morph  ptb_morph  clean_style
class_name                                                                 
good                         8310                 0          0            0
medium                       5199              1061       2019           31
bad                          1314              2393       4535           68

## Train Percent By Class

v116_candidate_type  original_but  but_native_morph  ptb_morph  clean_style
class_name                                                                 
good                       100.00              0.00       0.00         0.00
medium                      62.56             12.77      24.30         0.37
bad                         15.81             28.80      54.57         0.82

## Pool Capacity And Selector Result

                  class_name gap_fill_component  selected_n  pool_n   rbf_mmd  sym_domain_auc  pca_density_overlap  pool_to_selected
60        medium_clean_style        clean_style          41     300  0.259555        1.000000             0.137337              7.32
145  medium_but_native_morph   but_native_morph        1616   51592  0.006918        0.994700             0.810238             31.93
346         medium_ptb_morph          ptb_morph        2424    6000  0.164117        1.000000             0.716124              2.48
407          bad_clean_style        clean_style          89     300  0.482773        1.000000             0.004380              3.37
492     bad_but_native_morph   but_native_morph        3514   13248  0.044235        0.993951             0.900477              3.77
693            bad_ptb_morph          ptb_morph        5271    7000  0.134500        1.000000             0.643726              1.33

## Distribution Metrics

          scope  but_n  synthetic_n   rbf_mmd  sliced_wasserstein  quantile_loss  sym_domain_auc  pca_density_overlap
0    all_labels  18635        31590  0.059592            0.346813       0.335463        0.704572             0.775698
1    class_good  10530        10530  0.001377            0.055613       0.000000        0.509960             1.000000
2  class_medium   6449        10530  0.008691            0.117459       0.036022        0.649138             0.924499
3     class_bad   1656        10530  0.045898            0.229435       0.199833        0.933622             0.798753

## Raw Feature Alias Consistency

                      check  max_abs_delta
0            raw_rms_vs_rms       0.000003
1            raw_ptp_vs_ptp       0.000020
2  raw_diff_vs_non_qrs_diff       0.000045

## Top Feature Drift Versus Original

  class_name    candidate_type  rows                                                                                          top_drift
0     medium  but_native_morph  1616      sqi_bSQI=0.543; raw_rms=0.028; raw_ptp_p99_p01=0.026; raw_diff_abs_p95=0.010; band_5_15=0.006
1     medium         ptb_morph  2424  band_5_15=2.142; pca_margin=0.211; sqi_bSQI=0.156; knn_label_purity=0.132; raw_diff_abs_p95=0.083
2     medium       clean_style    41  band_5_15=2.142; knn_label_purity=0.628; sqi_bSQI=0.474; band_15_30=0.468; raw_diff_abs_p95=0.353
3        bad  but_native_morph  3514            sqi_bSQI=4.476; raw_rms=0.067; band_15_30=0.020; hf_ratio=0.020; raw_diff_abs_p95=0.011
4        bad         ptb_morph  5271                 band_5_15=5.359; pca_margin=4.611; sqi_bSQI=2.135; band_15_30=0.842; raw_rms=0.557
5        bad       clean_style    89     band_15_30=15.861; sqi_bSQI=12.473; raw_diff_abs_p95=12.276; pca_margin=6.889; band_5_15=5.359

## Gates

- protocol rows: `31590`
- val/test generated rows: `0`
- final selected count is hard-gated by selector; insufficient pools raise instead of silently underfilling.

## Figures

- `E:\GPTProject2\ecg\docs\data_v1_figures\distribution_domain_auc.png`
- `E:\GPTProject2\ecg\docs\data_v1_figures\distribution_pca_overlap.png`
- `E:\GPTProject2\ecg\docs\data_v1_figures\distribution_rbf_mmd.png`
- `E:\GPTProject2\ecg\docs\data_v1_figures\distribution_sliced_wasserstein.png`
- `E:\GPTProject2\ecg\docs\data_v1_figures\generated_gap_component_share.png`
- `E:\GPTProject2\ecg\docs\data_v1_figures\original_but_class_counts.png`
- `E:\GPTProject2\ecg\docs\data_v1_figures\split_class_counts.png`
- `E:\GPTProject2\ecg\docs\data_v1_figures\train_candidate_type_composition.png`
