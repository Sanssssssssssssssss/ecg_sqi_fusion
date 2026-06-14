# Reduced UFormer + SQI/Geometry Feature Candidates

Frozen UFormer waveform feature extractor plus a reduced tabular branch. Model selection uses train/val only; held-out BUT test is report-only.

## Held-Out BUT Test

| Candidate | n features | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| uformer_top20 | 20 | 0.886398 | 0.832309 | 0.928022 | 0.883190 | 0.552311 | 262 | 517 | 73 |
| uformer_top20_badcal | 20 | 0.887932 | 0.841379 | 0.928022 | 0.883190 | 0.583942 | 262 | 517 | 71 |
| uformer_top22 | 22 | 0.872243 | 0.833410 | 0.982692 | 0.806823 | 0.598540 | 63 | 852 | 111 |
| uformer_top22_badcal | 22 | 0.878849 | 0.866034 | 0.982692 | 0.803434 | 0.771290 | 63 | 840 | 71 |
| uformer_top14 | 14 | 0.883685 | 0.833770 | 0.853297 | 0.937867 | 0.569343 | 534 | 275 | 69 |
| uformer_top14_badcal | 14 | 0.889584 | 0.867747 | 0.852747 | 0.936060 | 0.715328 | 534 | 275 | 29 |
| 47-feature current best, threshold p_bad>=0.13 | 47 | 0.963548 | 0.930683 | 0.956319 | 0.972887 | 0.927007 | 102 | 76 | 23 |

## Original Buckets

| candidate | bucket | n | acc | macro_f1 | good_recall | medium_recall | bad_recall | good_precision | medium_precision | bad_precision | good_to_medium | medium_to_good | bad_to_medium | confusion_3x3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| uformer_top20 | original_test_all_10s+ | 8477 | 0.886398 | 0.832309 | 0.928022 | 0.883190 | 0.552311 | 0.843235 | 0.921065 | 1.000000 | 262 | 517 | 73 | [[3378, 262, 0], [517, 3909, 0], [111, 73, 227]] |
| uformer_top20 | original_all_10s+ | 32956 | 0.953848 | 0.957587 | 0.972657 | 0.918423 | 0.964428 | 0.944236 | 0.947486 | 0.999804 | 465 | 867 | 76 | [[16577, 465, 1], [867, 9761, 0], [112, 76, 5097]] |
| uformer_top20 | original_test_without_bad_outlier_stress | 8185 | 0.904826 | 0.935334 | 0.928022 | 0.883190 | 1.000000 | 0.867266 | 0.937185 | 1.000000 | 262 | 517 | 0 | [[3378, 262, 0], [517, 3909, 0], [0, 0, 119]] |
| uformer_top20 | bad_core_nearboundary | 119 | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 | [[0, 0, 0], [0, 0, 0], [0, 0, 119]] |
| uformer_top20 | bad_outlier_stress | 292 | 0.369863 | 0.180000 | 0.000000 | 0.000000 | 0.369863 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 73 | [[0, 0, 0], [0, 0, 0], [111, 73, 108]] |
| uformer_top20_badcal | original_test_all_10s+ | 8477 | 0.887932 | 0.841379 | 0.928022 | 0.883190 | 0.583942 | 0.845557 | 0.921499 | 1.000000 | 262 | 517 | 71 | [[3378, 262, 0], [517, 3909, 0], [100, 71, 240]] |
| uformer_top20_badcal | original_all_10s+ | 32956 | 0.954272 | 0.958185 | 0.972657 | 0.918423 | 0.967077 | 0.944881 | 0.947670 | 0.999804 | 465 | 867 | 74 | [[16577, 465, 1], [867, 9761, 0], [100, 74, 5111]] |
| uformer_top20_badcal | original_test_without_bad_outlier_stress | 8185 | 0.904826 | 0.935334 | 0.928022 | 0.883190 | 1.000000 | 0.867266 | 0.937185 | 1.000000 | 262 | 517 | 0 | [[3378, 262, 0], [517, 3909, 0], [0, 0, 119]] |
| uformer_top20_badcal | bad_core_nearboundary | 119 | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 | [[0, 0, 0], [0, 0, 0], [0, 0, 119]] |
| uformer_top20_badcal | bad_outlier_stress | 292 | 0.414384 | 0.195319 | 0.000000 | 0.000000 | 0.414384 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 71 | [[0, 0, 0], [0, 0, 0], [100, 71, 121]] |
| uformer_top22 | original_test_all_10s+ | 8477 | 0.872243 | 0.833410 | 0.982692 | 0.806823 | 0.598540 | 0.797903 | 0.953538 | 0.987952 | 63 | 852 | 111 | [[3577, 63, 0], [852, 3571, 3], [54, 111, 246]] |
| uformer_top22 | original_all_10s+ | 32956 | 0.953726 | 0.956889 | 0.992607 | 0.884927 | 0.966698 | 0.929557 | 0.975117 | 0.999413 | 126 | 1220 | 114 | [[16917, 126, 0], [1220, 9405, 3], [62, 114, 5109]] |
| uformer_top22 | original_test_without_bad_outlier_stress | 8185 | 0.887844 | 0.920086 | 0.982692 | 0.806823 | 1.000000 | 0.807632 | 0.982664 | 0.975410 | 63 | 852 | 0 | [[3577, 63, 0], [852, 3571, 3], [0, 0, 119]] |
| uformer_top22 | bad_core_nearboundary | 119 | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 | [[0, 0, 0], [0, 0, 0], [0, 0, 119]] |
| uformer_top22 | bad_outlier_stress | 292 | 0.434932 | 0.202068 | 0.000000 | 0.000000 | 0.434932 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 111 | [[0, 0, 0], [0, 0, 0], [54, 111, 127]] |
| uformer_top22_badcal | original_test_all_10s+ | 8477 | 0.878849 | 0.866034 | 0.982692 | 0.803434 | 0.771290 | 0.805631 | 0.963686 | 0.913545 | 63 | 840 | 71 | [[3577, 63, 0], [840, 3556, 30], [23, 71, 317]] |
| uformer_top22_badcal | original_all_10s+ | 32956 | 0.955607 | 0.959337 | 0.992607 | 0.883515 | 0.981268 | 0.931963 | 0.979349 | 0.994248 | 126 | 1208 | 72 | [[16917, 126, 0], [1208, 9390, 30], [27, 72, 5186]] |
| uformer_top22_badcal | original_test_without_bad_outlier_stress | 8185 | 0.886011 | 0.886670 | 0.982692 | 0.803434 | 1.000000 | 0.809826 | 0.982592 | 0.798658 | 63 | 840 | 0 | [[3577, 63, 0], [840, 3556, 30], [0, 0, 119]] |
| uformer_top22_badcal | bad_core_nearboundary | 119 | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 | [[0, 0, 0], [0, 0, 0], [0, 0, 119]] |
| uformer_top22_badcal | bad_outlier_stress | 292 | 0.678082 | 0.269388 | 0.000000 | 0.000000 | 0.678082 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 71 | [[0, 0, 0], [0, 0, 0], [23, 71, 198]] |
| uformer_top14 | original_test_all_10s+ | 8477 | 0.883685 | 0.833770 | 0.853297 | 0.937867 | 0.569343 | 0.890226 | 0.873159 | 1.000000 | 534 | 275 | 69 | [[3106, 534, 0], [275, 4151, 0], [108, 69, 234]] |
| uformer_top14 | original_all_10s+ | 32956 | 0.960280 | 0.963470 | 0.955466 | 0.965751 | 0.964806 | 0.971773 | 0.924685 | 1.000000 | 759 | 364 | 77 | [[16284, 759, 0], [364, 10264, 0], [109, 77, 5099]] |
| uformer_top14 | original_test_without_bad_outlier_stress | 8185 | 0.901038 | 0.930554 | 0.853297 | 0.937867 | 0.991597 | 0.918663 | 0.885830 | 1.000000 | 534 | 275 | 1 | [[3106, 534, 0], [275, 4151, 0], [0, 1, 118]] |
| uformer_top14 | bad_core_nearboundary | 119 | 0.991597 | 0.331927 | 0.000000 | 0.000000 | 0.991597 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 1 | [[0, 0, 0], [0, 0, 0], [0, 1, 118]] |
| uformer_top14 | bad_outlier_stress | 292 | 0.397260 | 0.189542 | 0.000000 | 0.000000 | 0.397260 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 68 | [[0, 0, 0], [0, 0, 0], [108, 68, 116]] |
| uformer_top14_badcal | original_test_all_10s+ | 8477 | 0.889584 | 0.867747 | 0.852747 | 0.936060 | 0.715328 | 0.895299 | 0.880365 | 0.967105 | 534 | 275 | 29 | [[3104, 534, 2], [275, 4143, 8], [88, 29, 294]] |
| uformer_top14_badcal | original_all_10s+ | 32956 | 0.961949 | 0.965953 | 0.955348 | 0.964904 | 0.977294 | 0.972989 | 0.928390 | 0.997875 | 759 | 364 | 32 | [[16282, 759, 2], [364, 10255, 9], [88, 32, 5165]] |
| uformer_top14_badcal | original_test_without_bad_outlier_stress | 8185 | 0.899939 | 0.918128 | 0.852747 | 0.936060 | 1.000000 | 0.918615 | 0.885824 | 0.922481 | 534 | 275 | 0 | [[3104, 534, 2], [275, 4143, 8], [0, 0, 119]] |
| uformer_top14_badcal | bad_core_nearboundary | 119 | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 | [[0, 0, 0], [0, 0, 0], [0, 0, 119]] |
| uformer_top14_badcal | bad_outlier_stress | 292 | 0.599315 | 0.249822 | 0.000000 | 0.000000 | 0.599315 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 29 | [[0, 0, 0], [0, 0, 0], [88, 29, 175]] |

## Candidate Feature Sets

- `uformer_top20` (20): pca_margin, sample_entropy_proxy, flatline_ratio, pc1, higuchi_fd_proxy, non_qrs_diff_p95, diff_zero_crossing_rate, zero_crossing_rate, qrs_prom_p90, sqi_fSQI, amplitude_entropy, sqi_kSQI, sqi_sSQI, low_amp_ratio, non_qrs_rms_ratio, baseline_step, ptp_p99_p01, band_30_45, qrs_visibility, band_15_30
- `uformer_top22` (22): pca_margin, sample_entropy_proxy, flatline_ratio, pc1, higuchi_fd_proxy, non_qrs_diff_p95, diff_zero_crossing_rate, zero_crossing_rate, qrs_prom_p90, sqi_fSQI, amplitude_entropy, sqi_kSQI, sqi_sSQI, low_amp_ratio, non_qrs_rms_ratio, baseline_step, ptp_p99_p01, band_30_45, qrs_visibility, band_15_30, hjorth_mobility, wavelet_e4
- `uformer_top14` (14): pc1, qrs_visibility, baseline_step, flatline_ratio, non_qrs_diff_p95, pca_margin, qrs_band_ratio, template_corr, amplitude_entropy, sqi_bSQI, region_confidence, detector_agreement, mean_abs, sqi_basSQI

## Outputs

- Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\current_reduced_uformer_metrics.csv`
- Checkpoints: `outputs/.../runs/reduced_uformer_nodecal/N17043_gm_probe/<candidate>/ckpt_best.pt`
