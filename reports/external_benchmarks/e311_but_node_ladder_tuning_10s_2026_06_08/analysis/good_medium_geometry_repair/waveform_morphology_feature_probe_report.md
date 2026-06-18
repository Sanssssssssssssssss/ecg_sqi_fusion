# Waveform Morphology Feature Probe

This report adds raw-waveform morphology features to the original-test error analysis. Original BUT remains report-only.

## Focus Group Medians
| focus_group | n | wf_lf_std_ratio | wf_density_abs_gt3 | wf_diff_ratio_99_95 | wf_spike_rate | wf_spike_interval_cv | wf_hf_diff_std | pc1 | pc2 | qrs_prom_p90 | qrs_visibility |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| correct medium | 4097 | 0.6169 | 0.0352 | 3.2039 | 1.2000 | 0.3786 | 0.6332 | -0.9224 | 4.9999 | 5.5199 | 0.0640 |
| correct good | 3080 | 0.4775 | 0.0568 | 14.3358 | 1.2000 | 0.5298 | 1.0921 | -5.4751 | 5.1005 | 7.9475 | 0.2361 |
| 111001 good->medium | 246 | 0.7838 | 0.0176 | 7.2959 | 0.9000 | 0.3399 | 0.2899 | -3.6826 | 10.9955 | 4.0606 | 0.0644 |
| 111001 medium->good | 213 | 0.6670 | 0.0232 | 8.0057 | 1.0000 | 0.3588 | 0.3891 | -3.4649 | 8.4275 | 5.0746 | 0.0790 |
| 111001 bad->medium | 140 | 0.7571 | 0.0264 | 2.3225 | 0.7000 | 0.7931 | 0.2437 | -3.4469 | 12.9947 | 2.7721 | 0.0252 |
| correct bad | 119 | 0.1279 | 0.0024 | 1.2883 | 1.3000 | 0.6518 | 0.8286 | 9.0054 | -0.9604 | 2.9843 | 0.1041 |
| 125001 good->medium | 75 | 0.8394 | 0.0096 | 3.0518 | 0.9000 | 0.3576 | 0.2481 | -2.2697 | 8.4315 | 3.1799 | 0.1038 |

## Top Feature Gaps
| comparison | feature | left_median | right_median | robust_effect | ks | left_n | right_n |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 111001 bad->medium vs correct bad | non_qrs_diff_p95 | 0.0403 | 0.4172 | -9.2714 | 1.0000 | 140 | 119 |
| 111001 bad->medium vs correct bad | pc1 | -3.4469 | 9.0054 | -7.5434 | 1.0000 | 140 | 119 |
| 111001 bad->medium vs correct bad | wf_diff_p95 | 0.4742 | 2.6947 | -6.7448 | 1.0000 | 140 | 119 |
| 111001 bad->medium vs correct bad | wf_lf_std_ratio | 0.7571 | 0.1279 | 5.5624 | 1.0000 | 140 | 119 |
| 111001 bad->medium vs correct bad | pc2 | 12.9947 | -0.9604 | 3.7641 | 1.0000 | 140 | 119 |
| 111001 bad->medium vs correct bad | wf_diff_ratio_99_95 | 2.3225 | 1.2883 | 2.2716 | 1.0000 | 140 | 119 |
| 111001 bad->medium vs correct bad | baseline_step | 1.5210 | 0.2475 | 3.3003 | 0.9857 | 140 | 119 |
| 111001 bad->medium vs correct bad | wf_hf_diff_std | 0.2437 | 0.8286 | -3.8191 | 0.9357 | 140 | 119 |
| 111001 good->medium vs correct good | wf_diff_ratio_99_95 | 7.2959 | 14.3358 | -1.7927 | 0.6383 | 246 | 3080 |
| 111001 good->medium vs correct good | qrs_visibility | 0.0644 | 0.2361 | -1.4188 | 0.6067 | 246 | 3080 |
| 111001 good->medium vs correct good | qrs_prom_p90 | 4.0606 | 7.9475 | -1.4538 | 0.6043 | 246 | 3080 |
| 111001 good->medium vs correct good | wf_diff_p99 | 1.8136 | 7.4540 | -1.2371 | 0.5664 | 246 | 3080 |
| 111001 good->medium vs correct good | wf_hf_diff_std | 0.2899 | 1.0921 | -1.1526 | 0.5559 | 246 | 3080 |
| 111001 good->medium vs correct good | amplitude_entropy | 0.8284 | 0.6522 | 1.2002 | 0.5334 | 246 | 3080 |
| 111001 good->medium vs correct good | wf_spike_amp_median | 4.2026 | 10.4539 | -0.9719 | 0.5319 | 246 | 3080 |
| 111001 good->medium vs correct good | wf_abs_p99 | 3.7245 | 11.2420 | -1.1781 | 0.5229 | 246 | 3080 |
| 111001 medium->good vs 111001 good->medium | qrs_prom_p90 | 5.0746 | 4.0606 | 0.2918 | 0.3555 | 213 | 246 |
| 111001 medium->good vs 111001 good->medium | non_qrs_diff_p95 | 0.0300 | 0.0192 | 0.4013 | 0.2928 | 213 | 246 |
| 111001 medium->good vs 111001 good->medium | wf_lf_std_ratio | 0.6670 | 0.7838 | -0.3926 | 0.2882 | 213 | 246 |
| 111001 medium->good vs 111001 good->medium | wf_abs_ratio_99_95 | 2.1038 | 1.6244 | 0.2573 | 0.2807 | 213 | 246 |
| 111001 medium->good vs 111001 good->medium | wf_spike_amp_median | 5.4139 | 4.2026 | 0.1968 | 0.2682 | 213 | 246 |
| 111001 medium->good vs 111001 good->medium | qrs_visibility | 0.0790 | 0.0644 | 0.2455 | 0.2562 | 213 | 246 |
| 111001 medium->good vs 111001 good->medium | wf_diff_p95 | 0.3809 | 0.2560 | 0.3407 | 0.2535 | 213 | 246 |
| 111001 medium->good vs 111001 good->medium | wf_diff_p99 | 2.4398 | 1.8136 | 0.1602 | 0.2517 | 213 | 246 |
| 111001 medium->good vs correct good | pc1 | -3.4649 | -5.4751 | 1.5070 | 0.6921 | 213 | 3080 |
| 111001 medium->good vs correct good | wf_diff_ratio_99_95 | 8.0057 | 14.3358 | -1.7808 | 0.6661 | 213 | 3080 |
| 111001 medium->good vs correct good | qrs_visibility | 0.0790 | 0.2361 | -1.3484 | 0.5776 | 213 | 3080 |
| 111001 medium->good vs correct good | qrs_prom_p90 | 5.0746 | 7.9475 | -1.0265 | 0.4598 | 213 | 3080 |
| 111001 medium->good vs correct good | pc3 | 0.4412 | -1.4306 | 0.8336 | 0.4332 | 213 | 3080 |
| 111001 medium->good vs correct good | wf_spike_interval_cv | 0.3588 | 0.5298 | -0.9450 | 0.4260 | 213 | 3080 |
| 111001 medium->good vs correct good | wf_diff_p99 | 2.4398 | 7.4540 | -0.8943 | 0.4195 | 213 | 3080 |
| 111001 medium->good vs correct good | wf_hf_diff_std | 0.3891 | 1.0921 | -0.8222 | 0.4095 | 213 | 3080 |
| 111001 medium->good vs correct medium | pc1 | -3.4649 | -0.9224 | -1.2581 | 0.6320 | 213 | 4097 |
| 111001 medium->good vs correct medium | wf_diff_ratio_99_95 | 8.0057 | 3.2039 | 1.3397 | 0.6050 | 213 | 4097 |
| 111001 medium->good vs correct medium | non_qrs_diff_p95 | 0.0300 | 0.0773 | -0.7885 | 0.4837 | 213 | 4097 |
| 111001 medium->good vs correct medium | wf_diff_p95 | 0.3809 | 0.9010 | -0.4514 | 0.3501 | 213 | 4097 |
| 111001 medium->good vs correct medium | wf_spike_count | 10.0000 | 12.0000 | -0.2500 | 0.2852 | 213 | 4097 |
| 111001 medium->good vs correct medium | wf_spike_rate | 1.0000 | 1.2000 | -0.2500 | 0.2852 | 213 | 4097 |
| 111001 medium->good vs correct medium | qrs_visibility | 0.0790 | 0.0640 | 0.1411 | 0.2845 | 213 | 4097 |
| 111001 medium->good vs correct medium | qrs_prom_p90 | 5.0746 | 5.5199 | -0.1231 | 0.2837 | 213 | 4097 |
| 125001 good->medium vs correct good | wf_diff_ratio_99_95 | 3.0518 | 14.3358 | -6.3077 | 0.9981 | 75 | 3080 |
| 125001 good->medium vs correct good | pc1 | -2.2697 | -5.4751 | 3.0081 | 0.9285 | 75 | 3080 |
| 125001 good->medium vs correct good | qrs_prom_p90 | 3.1799 | 7.9475 | -3.1083 | 0.8228 | 75 | 3080 |
| 125001 good->medium vs correct good | wf_spike_amp_median | 3.2260 | 10.4539 | -1.5806 | 0.7711 | 75 | 3080 |
| 125001 good->medium vs correct good | amplitude_entropy | 0.8546 | 0.6522 | 1.8344 | 0.7427 | 75 | 3080 |
| 125001 good->medium vs correct good | wf_abs_ratio_99_95 | 1.4577 | 3.0109 | -2.3075 | 0.7358 | 75 | 3080 |
| 125001 good->medium vs correct good | wf_diff_p99 | 1.3098 | 7.4540 | -1.7356 | 0.7035 | 75 | 3080 |
| 125001 good->medium vs correct good | wf_lf_std_ratio | 0.8394 | 0.4775 | 2.0386 | 0.6981 | 75 | 3080 |

## Visual
![Waveform morphology boxes](E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_morphology_feature_boxes.png)

## Interpretation
- These features are intentionally simple waveform summaries, not a complex ECG parser.
- The goal is to find one or two stable morphology axes before adding more generator or classifier complexity.
