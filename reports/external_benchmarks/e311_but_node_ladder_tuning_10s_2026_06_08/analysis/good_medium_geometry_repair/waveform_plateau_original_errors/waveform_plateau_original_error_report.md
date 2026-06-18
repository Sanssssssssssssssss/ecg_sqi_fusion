# Waveform Original Plateau Error Analysis

This is report-only analysis. Original BUT is not used for training, threshold selection, or model selection.

## Model Buckets

| Model | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->g | b->m |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| featuretx_top20_report_only | 0.908458 | 0.932228 | 1.000 | 0.825 | 0.998 | 0 | 766 | 1 | 0 |
| aug_convtx_balanced_focal | 0.811018 | 0.730436 | 0.961 | 0.726 | 0.401 | 143 | 1190 | 62 | 184 |
| qrscritical_teacherselect | 0.792969 | 0.555412 | 0.874 | 0.798 | 0.022 | 456 | 886 | 200 | 202 |
| qrscritical_multiscale | 0.779167 | 0.664393 | 0.900 | 0.727 | 0.268 | 360 | 1196 | 256 | 45 |
| statconformer_badguard | 0.763596 | 0.521753 | 0.913 | 0.711 | 0.000 | 315 | 1277 | 261 | 150 |

## Shared Plateau Rows

Rows with high `wrong_count_across_models` are the likely plateau set: they are repeatedly missed by very different waveform-only architectures.

- `medium` / `outlier_low_confidence` / wrong_by `5` models: `507` rows
- `bad` / `outlier_low_confidence` / wrong_by `5` models: `1` rows
- `medium` / `outlier_low_confidence` / wrong_by `4` models: `392` rows
- `bad` / `outlier_low_confidence` / wrong_by `4` models: `242` rows
- `good` / `outlier_low_confidence` / wrong_by `4` models: `131` rows
- `good` / `good_medium_overlap` / wrong_by `4` models: `4` rows
- `medium` / `good_medium_overlap` / wrong_by `4` models: `1` rows
- `medium` / `outlier_low_confidence` / wrong_by `3` models: `174` rows
- `good` / `outlier_low_confidence` / wrong_by `3` models: `141` rows
- `bad` / `outlier_low_confidence` / wrong_by `3` models: `47` rows
- `good` / `good_medium_overlap` / wrong_by `3` models: `24` rows
- `bad` / `near_bad_boundary` / wrong_by `3` models: `10` rows
- `medium` / `good_medium_overlap` / wrong_by `3` models: `3` rows
- `medium` / `outlier_low_confidence` / wrong_by `2` models: `204` rows
- `bad` / `near_bad_boundary` / wrong_by `2` models: `104` rows
- `good` / `good_medium_overlap` / wrong_by `2` models: `41` rows

## Top Feature Gaps

### qrscritical_multiscale: bad_to_good
- `pca_margin` KS `0.991`; error median `-6.16` is lower than correct median `5.369`
- `pc2` KS `0.973`; error median `11.89` is higher than correct median `-0.9683`
- `non_qrs_diff_p95` KS `0.973`; error median `0.03751` is lower than correct median `0.4169`
- `knn_label_purity` KS `0.973`; error median `0` is lower than correct median `0.9667`
- `zero_crossing_rate` KS `0.973`; error median `0.03042` is lower than correct median `0.4916`
- `wavelet_e4` KS `0.973`; error median `0.0009305` is lower than correct median `0.1796`
- `wavelet_e3` KS `0.973`; error median `0.005203` is lower than correct median `0.17`
- `pc1` KS `0.973`; error median `-4.534` is lower than correct median `9.037`

### qrscritical_teacherselect: bad_to_good
- `pca_margin` KS `0.965`; error median `-6.205` is lower than correct median `5.145`
- `region_confidence` KS `0.925`; error median `0.005339` is lower than correct median `0.3036`
- `boundary_confidence` KS `0.925`; error median `0.02136` is lower than correct median `0.3892`
- `knn_label_purity` KS `0.920`; error median `0` is lower than correct median `0.9667`
- `sqi_kSQI` KS `0.905`; error median `8.189` is higher than correct median `2.932`
- `pc4` KS `0.864`; error median `1.163` is higher than correct median `-1.501`
- `sqi_pSQI` KS `0.849`; error median `0.7678` is higher than correct median `0.3316`
- `diff_zero_crossing_rate` KS `0.829`; error median `0.2696` is lower than correct median `0.6723`

### aug_convtx_balanced_focal: good_to_medium
- `pc1` KS `0.932`; error median `-1.306` is higher than correct median `-5.362`
- `pca_margin` KS `0.909`; error median `-1.174` is lower than correct median `1.665`
- `pc4` KS `0.887`; error median `-2.736` is lower than correct median `3.349`
- `boundary_confidence` KS `0.885`; error median `0.1007` is lower than correct median `0.5848`
- `flatline_ratio` KS `0.871`; error median `0.1265` is lower than correct median `0.3915`
- `region_confidence` KS `0.867`; error median `0.02518` is lower than correct median `0.1556`
- `sample_entropy_proxy` KS `0.860`; error median `0.5987` is higher than correct median `0.2309`
- `knn_label_purity` KS `0.848`; error median `0.1` is lower than correct median `1`

### statconformer_badguard: good_to_medium
- `pca_margin` KS `0.923`; error median `-1.197` is lower than correct median `1.749`
- `pc1` KS `0.901`; error median `-1.613` is higher than correct median `-5.414`
- `flatline_ratio` KS `0.893`; error median `0.1425` is lower than correct median `0.3987`
- `boundary_confidence` KS `0.834`; error median `0.1432` is lower than correct median `0.596`
- `region_confidence` KS `0.793`; error median `0.03581` is lower than correct median `0.1573`
- `pc4` KS `0.782`; error median `-0.6808` is lower than correct median `3.473`
- `knn_label_purity` KS `0.781`; error median `0.2333` is lower than correct median `1`
- `sample_entropy_proxy` KS `0.754`; error median `0.4871` is higher than correct median `0.2257`

### qrscritical_teacherselect: good_to_medium
- `flatline_ratio` KS `0.911`; error median `0.1637` is lower than correct median `0.4043`
- `pc1` KS `0.911`; error median `-2.076` is higher than correct median `-5.456`
- `pca_margin` KS `0.809`; error median `-0.8614` is lower than correct median `1.81`
- `non_qrs_diff_p95` KS `0.708`; error median `0.04268` is higher than correct median `0.01889`
- `sqi_fSQI` KS `0.703`; error median `0.004804` is lower than correct median `0.01281`
- `boundary_confidence` KS `0.648`; error median `0.2513` is lower than correct median `0.603`
- `knn_label_purity` KS `0.639`; error median `0.4667` is lower than correct median `1`
- `fatal_or_score` KS `0.637`; error median `1` is higher than correct median `0.8555`

### qrscritical_multiscale: good_to_medium
- `pca_margin` KS `0.899`; error median `-1.105` is lower than correct median `1.774`
- `flatline_ratio` KS `0.899`; error median `0.1493` is lower than correct median `0.3995`
- `pc1` KS `0.895`; error median `-1.728` is higher than correct median `-5.425`
- `boundary_confidence` KS `0.785`; error median `0.1718` is lower than correct median `0.5989`
- `knn_label_purity` KS `0.739`; error median `0.3` is lower than correct median `1`
- `pc4` KS `0.720`; error median `-0.4685` is lower than correct median `3.497`
- `region_confidence` KS `0.716`; error median `0.04295` is lower than correct median `0.1571`
- `sample_entropy_proxy` KS `0.683`; error median `0.4782` is higher than correct median `0.2247`

### qrscritical_teacherselect: medium_to_good
- `pc1` KS `0.880`; error median `-4.876` is lower than correct median `-0.5741`
- `boundary_confidence` KS `0.866`; error median `0.07541` is lower than correct median `0.6039`
- `region_confidence` KS `0.866`; error median `0.01885` is lower than correct median `0.5176`
- `pca_margin` KS `0.864`; error median `-0.1162` is lower than correct median `1.901`
- `flatline_ratio` KS `0.839`; error median `0.4171` is higher than correct median `0.1009`
- `higuchi_fd_proxy` KS `0.828`; error median `1.236` is lower than correct median `1.534`
- `zero_crossing_rate` KS `0.807`; error median `0.01841` is lower than correct median `0.09608`
- `non_qrs_diff_p95` KS `0.805`; error median `0.01474` is lower than correct median `0.09285`

### featuretx_top20_report_only: medium_to_good
- `pca_margin` KS `0.873`; error median `-0.002582` is lower than correct median `1.859`
- `boundary_confidence` KS `0.841`; error median `0.09201` is lower than correct median `0.5939`
- `region_confidence` KS `0.841`; error median `0.023` is lower than correct median `0.5048`
- `pc1` KS `0.790`; error median `-4.502` is lower than correct median `-0.631`
- `knn_label_purity` KS `0.785`; error median `0.06667` is lower than correct median `0.8667`
- `flatline_ratio` KS `0.710`; error median `0.3927` is higher than correct median `0.1049`
- `non_qrs_diff_p95` KS `0.695`; error median `0.01597` is lower than correct median `0.09025`
- `pc3` KS `0.675`; error median `-1.427` is lower than correct median `1.97`

### statconformer_badguard: medium_to_good
- `pc1` KS `0.851`; error median `-4.368` is lower than correct median `-0.4074`
- `pca_margin` KS `0.840`; error median `-0.01514` is lower than correct median `2.035`
- `boundary_confidence` KS `0.837`; error median `0.0988` is lower than correct median `0.6314`
- `region_confidence` KS `0.836`; error median `0.0247` is lower than correct median `0.5506`
- `flatline_ratio` KS `0.797`; error median `0.3579` is higher than correct median `0.09007`
- `knn_label_purity` KS `0.779`; error median `0.1` is lower than correct median `0.9`
- `higuchi_fd_proxy` KS `0.759`; error median `1.263` is lower than correct median `1.543`
- `diff_zero_crossing_rate` KS `0.742`; error median `0.3125` is lower than correct median `0.5008`

### qrscritical_multiscale: medium_to_good
- `pc1` KS `0.838`; error median `-4.419` is lower than correct median `-0.4345`
- `flatline_ratio` KS `0.808`; error median `0.3619` is higher than correct median `0.09127`
- `pca_margin` KS `0.805`; error median `-0.03044` is lower than correct median `2.007`
- `region_confidence` KS `0.805`; error median `0.02377` is lower than correct median `0.5449`
- `boundary_confidence` KS `0.804`; error median `0.09509` is lower than correct median `0.6257`
- `higuchi_fd_proxy` KS `0.778`; error median `1.258` is lower than correct median `1.54`
- `diff_zero_crossing_rate` KS `0.767`; error median `0.3093` is lower than correct median `0.5008`
- `zero_crossing_rate` KS `0.757`; error median `0.02082` is lower than correct median `0.1033`

### qrscritical_multiscale: bad_to_medium
- `pca_margin` KS `0.785`; error median `-5.982` is lower than correct median `5.369`
- `pc2` KS `0.780`; error median `7.578` is higher than correct median `-0.9683`
- `boundary_confidence` KS `0.771`; error median `0.03794` is lower than correct median `0.3907`
- `region_confidence` KS `0.771`; error median `0.009485` is lower than correct median `0.3048`
- `sqi_basSQI` KS `0.751`; error median `0.7294` is lower than correct median `0.9802`
- `baseline_step` KS `0.741`; error median `1.035` is higher than correct median `0.2424`
- `wavelet_e4` KS `0.741`; error median `0.005271` is lower than correct median `0.1796`
- `pc1` KS `0.741`; error median `-0.8092` is lower than correct median `9.037`

### aug_convtx_balanced_focal: bad_to_good
- `non_qrs_diff_p95` KS `0.782`; error median `0.02103` is lower than correct median `0.4039`
- `zero_crossing_rate` KS `0.758`; error median `0.02122` is lower than correct median `0.4828`
- `boundary_confidence` KS `0.743`; error median `0.01864` is lower than correct median `0.3887`
- `region_confidence` KS `0.743`; error median `0.004659` is lower than correct median `0.3032`
- `pc1` KS `0.741`; error median `-5.49` is lower than correct median `8.803`
- `flatline_ratio` KS `0.735`; error median `0.5236` is higher than correct median `0.01041`
- `sample_entropy_proxy` KS `0.735`; error median `0.3217` is lower than correct median `0.8861`
- `sqi_fSQI` KS `0.733`; error median `0.02362` is higher than correct median `0.0008006`

### aug_convtx_balanced_focal: bad_to_medium
- `boundary_confidence` KS `0.770`; error median `0.02372` is lower than correct median `0.3887`
- `region_confidence` KS `0.770`; error median `0.005931` is lower than correct median `0.3032`
- `knn_label_purity` KS `0.765`; error median `0` is lower than correct median `0.9667`
- `pca_margin` KS `0.747`; error median `-6.194` is lower than correct median `5.145`
- `sample_entropy_proxy` KS `0.735`; error median `0.4343` is lower than correct median `0.8861`
- `wavelet_e2` KS `0.733`; error median `0.0283` is lower than correct median `0.1928`
- `baseline_step` KS `0.728`; error median `1.386` is higher than correct median `0.2838`
- `zero_crossing_rate` KS `0.727`; error median `0.04003` is lower than correct median `0.4828`

### aug_convtx_balanced_focal: medium_to_good
- `region_confidence` KS `0.763`; error median `0.02531` is lower than correct median `0.5448`
- `boundary_confidence` KS `0.760`; error median `0.1012` is lower than correct median `0.6258`
- `pc1` KS `0.751`; error median `-4.36` is lower than correct median `-0.4321`
- `pca_margin` KS `0.735`; error median `-0.02061` is lower than correct median `2.003`
- `knn_label_purity` KS `0.733`; error median `0.1` is lower than correct median `0.9`
- `higuchi_fd_proxy` KS `0.692`; error median `1.262` is lower than correct median `1.54`
- `zero_crossing_rate` KS `0.681`; error median `0.02122` is lower than correct median `0.1025`
- `non_qrs_diff_p95` KS `0.677`; error median `0.01776` is lower than correct median `0.09917`

### qrscritical_teacherselect: bad_to_medium
- `ptp_p99_p01` KS `0.545`; error median `0.7633` is higher than correct median `0.7266`
- `sqi_kSQI` KS `0.520`; error median `3.116` is higher than correct median `2.932`
- `pca_margin` KS `0.436`; error median `4.887` is lower than correct median `5.145`
- `qrs_prom_p90` KS `0.431`; error median `2.998` is higher than correct median `2.831`
- `boundary_confidence` KS `0.431`; error median `0.3865` is lower than correct median `0.3892`
- `region_confidence` KS `0.431`; error median `0.3015` is lower than correct median `0.3036`
- `knn_label_purity` KS `0.421`; error median `0.9667` is lower than correct median `0.9667`
- `non_qrs_rms_ratio` KS `0.394`; error median `0.8966` is lower than correct median `0.9344`

## Files

- Metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_plateau_original_errors\plateau_model_bucket_metrics.csv`
- Error region counts: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_plateau_original_errors\error_region_counts.csv`
- Feature KS gaps: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_plateau_original_errors\error_feature_ks_gaps.csv`
- Shared rows: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_plateau_original_errors\shared_original_test_error_rows.csv`