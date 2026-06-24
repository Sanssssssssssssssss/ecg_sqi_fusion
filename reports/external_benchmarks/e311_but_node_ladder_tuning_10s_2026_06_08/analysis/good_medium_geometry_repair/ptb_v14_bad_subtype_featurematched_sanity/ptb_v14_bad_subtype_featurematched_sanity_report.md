# BUT Subtype-Auxiliary Waveform Experiment

- Generated: 2026-06-21 00:39:04
- Protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v14_bad_subtype_featurematched\protocol_v14_pc3000_s20260621`
- Inference input: waveform-derived channels only.
- Formal output remains 3-class good/medium/bad.
- Subtype labels are auxiliary supervision and diagnostics only.

## Split Counts

| split | class_name | n |
| --- | --- | --- |
| test | bad | 450 |
| test | good | 452 |
| test | medium | 506 |
| train | bad | 2100 |
| train | good | 2105 |
| train | medium | 2014 |
| val | bad | 450 |
| val | good | 443 |
| val | medium | 480 |

## Subtype Counts

| split | class_name | subtype | n |
| --- | --- | --- | --- |
| test | bad | bad_baseline_wander_lowfreq | 7 |
| test | bad | bad_contact_reset_flatline | 20 |
| test | bad | bad_dense_right_island | 276 |
| test | bad | bad_detector_template_disagree | 100 |
| test | bad | bad_highfreq_detail_noise | 6 |
| test | bad | bad_low_qrs_visibility | 4 |
| test | bad | bad_other_boundary | 37 |
| test | good | good_overlap_or_mild_artifact | 452 |
| test | medium | medium_overlap_or_detail | 506 |
| train | bad | bad_baseline_wander_lowfreq | 31 |
| train | bad | bad_contact_reset_flatline | 96 |
| train | bad | bad_dense_right_island | 1285 |
| train | bad | bad_detector_template_disagree | 467 |
| train | bad | bad_highfreq_detail_noise | 28 |
| train | bad | bad_low_qrs_visibility | 20 |
| train | bad | bad_other_boundary | 173 |
| train | good | good_overlap_or_mild_artifact | 2105 |
| train | medium | medium_overlap_or_detail | 2014 |
| val | bad | bad_baseline_wander_lowfreq | 7 |
| val | bad | bad_contact_reset_flatline | 21 |
| val | bad | bad_dense_right_island | 275 |
| val | bad | bad_detector_template_disagree | 100 |
| val | bad | bad_highfreq_detail_noise | 6 |
| val | bad | bad_low_qrs_visibility | 4 |
| val | bad | bad_other_boundary | 37 |
| val | good | good_overlap_or_mild_artifact | 443 |
| val | medium | medium_overlap_or_detail | 480 |

## Results

| candidate | bucket | runs | acc | acc_std | macro_f1 | macro_f1_std | good_recall | good_recall_std | medium_recall | medium_recall_std | bad_recall | bad_recall_std | subtype_acc | subtype_acc_std | bad_subtype_acc | bad_subtype_acc_std | good_to_medium | good_to_medium_std | medium_to_good | medium_to_good_std | bad_to_nonbad | bad_to_nonbad_std | nonbad_to_bad | nonbad_to_bad_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E1_subtype_aux_keep | test | 1 | 0.99858 | 0 | 0.9986 | 0 | 1 | 0 | 1 | 0 | 0.995556 | 0 | 0.776989 | 0 | 0.302222 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 |
| E1_baseline_keep | test | 1 | 0.993608 | 0 | 0.993707 | 0 | 1 | 0 | 0.98419 | 0 | 0.997778 | 0 | 0.355114 | 0 | 0.0133333 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 8 | 0 |
| E1_baseline_keep | val | 1 | 0.997815 | 0 | 0.997847 | 0 | 1 | 0 | 1 | 0 | 0.993333 | 0 | 0.330663 | 0 | 0.0133333 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 |
| E1_subtype_aux_keep | val | 1 | 0.997815 | 0 | 0.997841 | 0 | 1 | 0 | 0.997917 | 0 | 0.995556 | 0 | 0.77713 | 0 | 0.32 | 0 | 0 | 0 | 1 | 0 | 2 | 0 | 0 | 0 |

## Output Files

- Metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_v14_bad_subtype_featurematched_sanity\but_subtype_aux_metrics.csv`
- Subtype metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_v14_bad_subtype_featurematched_sanity\but_subtype_aux_subtype_metrics.csv`
- Bad mechanism audit: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_v14_bad_subtype_featurematched_sanity\but_bad_subtype_multilabel_audit.csv`
- Checkpoints/logs: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\ptb_v14_bad_subtype_featurematched_sanity`