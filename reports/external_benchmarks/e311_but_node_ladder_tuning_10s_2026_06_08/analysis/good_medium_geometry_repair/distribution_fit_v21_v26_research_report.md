# Distribution Fit V21-V26 Research Report

- Generated: 2026-06-21 03:08:54
- Code: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\build_ptb_v21_gm_featurematched.py`
- Scope: PTB synthetic distribution fitting to clean BUT waveform-computable SQI/geometry distributions; waveform-only Event-Factorized SQI Conformer cross tests.
- Formal no-leakage target uses BUT train feature distribution only. `v26_alltarget_diag` uses all BUT feature distribution and is diagnostic only.

## Main Finding

V23 is the best distribution strategy so far. The important change is not another waveform perturbation; it is quantile-anchor selection, which covers BUT q10/q25/q50/q75/q90 feature shells instead of collapsing every PTB candidate toward the class median. This improved PTB->BUT cross_all from V22 E1 0.7533 to V23 E1 0.8269, with medium recall from 0.5218 to 0.6450 while keeping bad recall about 1.0.

## Distribution Gap Summary

| version | class_name | mean_abs_gap | median_abs_gap | max_abs_gap |
| --- | --- | --- | --- | --- |
| v21 | good | 0.5494 | 0.4932 | 1.0741 |
| v21 | medium | 0.1170 | 0.0864 | 0.3280 |
| v22 | good | 0.4922 | 0.4563 | 0.9446 |
| v22 | medium | 0.1185 | 0.0833 | 0.2797 |
| v23 | good | 0.4687 | 0.3333 | 0.9590 |
| v23 | medium | 0.1294 | 0.0905 | 0.3993 |
| v24 | good | 0.4679 | 0.3331 | 0.9705 |
| v24 | medium | 0.1299 | 0.0900 | 0.4061 |
| v25 | good | 0.4679 | 0.3331 | 0.9705 |
| v25 | medium | 0.1252 | 0.0697 | 0.3138 |
| v26_alltarget_diag | good | 0.4692 | 0.3333 | 0.9612 |
| v26_alltarget_diag | medium | 0.0918 | 0.0370 | 0.3065 |

## PTB Synthetic -> BUT Report-Only Cross Results

| version | candidate | bucket | acc | macro_f1 | good | medium | bad | record_macro_acc | bad_record_recall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v23 | E1_query_only | cross_all | 0.8269 | 0.8366 | 0.8655 | 0.6450 | 0.9998 | 0.8521 | 0.6667 |
| v26_alltarget_diag | E1_query_only | cross_all | 0.7869 | 0.7605 | 0.9314 | 0.3891 | 0.9998 | 0.8064 | 0.6667 |
| v24 | E1_query_only | cross_all | 0.7832 | 0.7952 | 0.8092 | 0.5955 | 0.9998 | 0.7944 | 0.6667 |
| v25 | E1_query_only | cross_all | 0.7763 | 0.7810 | 0.8123 | 0.5662 | 0.9998 | 0.7791 | 0.6667 |
| v23 | E2_query_highres | cross_all | 0.7695 | 0.7915 | 0.7416 | 0.6693 | 0.9998 | 0.8023 | 0.6667 |
| v22 | E1_query_only | cross_all | 0.7533 | 0.7538 | 0.7929 | 0.5218 | 0.9998 | 0.7222 | 0.6667 |
| v22 | E2_query_highres | cross_all | 0.7225 | 0.7368 | 0.6949 | 0.5914 | 0.9998 | 0.6367 | 0.6667 |
| v21 | E2_query_highres | cross_all | 0.7099 | 0.7128 | 0.7484 | 0.4519 | 1.0000 | 0.7118 | 1.0000 |
| v21 | E1_query_only | cross_all | 0.6826 | 0.7114 | 0.6266 | 0.5765 | 0.9998 | 0.6095 | 0.6667 |

## BUT -> PTB Synthetic Cross Results

| version | candidate | bucket | acc | macro_f1 | good | medium | bad | record_macro_acc | bad_record_recall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v23 | E2_query_highres | cross_all | 0.8789 | 0.8820 | 0.8310 | 0.9970 | 0.8087 | 0.9137 | 0.6969 |
| v26_alltarget_diag | E1_query_only | cross_all | 0.8748 | 0.8779 | 0.8477 | 0.9983 | 0.7783 | 0.9227 | 0.6316 |
| v25 | E1_query_only | cross_all | 0.8701 | 0.8735 | 0.8293 | 0.9973 | 0.7837 | 0.9130 | 0.6387 |
| v22 | E2_query_highres | cross_all | 0.8630 | 0.8656 | 0.8677 | 1.0000 | 0.7213 | 0.9334 | 0.5943 |
| v21 | E2_query_highres | cross_all | 0.8611 | 0.8642 | 0.8510 | 1.0000 | 0.7323 | 0.9251 | 0.6007 |
| v23 | E1_query_only | cross_all | 0.8602 | 0.8637 | 0.8330 | 0.9977 | 0.7500 | 0.9150 | 0.5907 |
| v22 | E1_query_only | cross_all | 0.8537 | 0.8565 | 0.8550 | 1.0000 | 0.7060 | 0.9270 | 0.5321 |
| v21 | E1_query_only | cross_all | 0.8411 | 0.8439 | 0.8443 | 1.0000 | 0.6790 | 0.9217 | 0.5082 |
| v24 | E1_query_only | cross_all | 0.7588 | 0.7543 | 0.8173 | 0.9987 | 0.4603 | 0.9073 | 0.3049 |

## Version Notes

- V21: fixed good/medium median matching after V20 bad distribution fit; improved feature gaps but PTB->BUT medium remained weak.
- V22: strengthened BUT-good high-amplitude/QRS-band/baseline morphology; PTB->BUT cross_all improved to 0.7533.
- V23: quantile-anchor matching; current best PTB->BUT cross_all 0.8269 and best medium recall 0.6450 among no-leakage PTB->BUT runs.
- V24/V25: forced medium hard-negative modes did not help; likely too artificial or not the missing natural medium mechanism.
- V26 all-target diagnostic: using all BUT feature distribution reduced medium recall, so aggregate target matching alone is insufficient and not a formal result.

## BUT Split Shift Audit

- Report: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\but_split_distribution_shift_audit\but_split_distribution_shift_audit.md`
- Key issue: BUT test differs strongly from BUT train, especially bad frequency bands and medium 30-45Hz. This explains why cross_test is much lower than cross_all.

## Key Visualizations

### v23
- Good gap: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v23_gm_featurematched\v23_gm_good_feature_gap.png`
- Medium gap: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v23_gm_featurematched\v23_gm_medium_feature_gap.png`
- Good boxplots: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v23_gm_featurematched\v23_gm_good_feature_boxplots.png`
- Medium boxplots: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v23_gm_featurematched\v23_gm_medium_feature_boxplots.png`

### v25
- Good gap: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v25_gm_featurematched\v25_gm_good_feature_gap.png`
- Medium gap: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v25_gm_featurematched\v25_gm_medium_feature_gap.png`
- Good boxplots: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v25_gm_featurematched\v25_gm_good_feature_boxplots.png`
- Medium boxplots: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v25_gm_featurematched\v25_gm_medium_feature_boxplots.png`

### v26_alltarget_diag
- Good gap: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v26_alltarget_diag_gm_featurematched\v26_alltarget_diag_gm_good_feature_gap.png`
- Medium gap: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v26_alltarget_diag_gm_featurematched\v26_alltarget_diag_gm_medium_feature_gap.png`
- Good boxplots: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v26_alltarget_diag_gm_featurematched\v26_alltarget_diag_gm_good_feature_boxplots.png`
- Medium boxplots: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v26_alltarget_diag_gm_featurematched\v26_alltarget_diag_gm_medium_feature_boxplots.png`

## Literature-Informed Interpretation

- ECG quality papers repeatedly separate baseline wander, electrode/motion/contact artifacts, high-frequency/muscle noise, and QRS detector agreement rather than treating poor quality as one scalar noise level.
- The V20-V23 results agree with that: bad improved only after matching peak-detector failure mechanisms; good/medium improved only after covering distribution shells, not only medians.

## Next Recommended Experiment

Use V23 as the current best PTB synthetic protocol. The next generator should not force artificial hard-negative modes. Instead, build medium subtype targets from real BUT train/val split shift clusters, especially high 30-45Hz medium and QRS-visible/detail-degraded medium, then use quantile-anchor matching inside each subtype. Keep V26 as diagnostic evidence only, not as a model claim.
