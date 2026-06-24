# PTB v12 Bad-Subtype Model -> BUT Keep-Outlier Report

- Training/normalization source: PTB v12 train only.
- BUT is report-only; no BUT row used for training/selection.
- This is a transfer sanity check, not a model selection result.

## Metrics

| candidate | but_split | n_rows | acc | macro_f1 | good_recall | medium_recall | bad_recall | bad_to_nonbad | nonbad_to_bad | subtype_acc | bad_subtype_acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E1_baseline_keep | train | 21117 | 0.78316 | 0.772345 | 0.774638 | 0.609616 | 0.993914 | 29 | 1023 | 0.157693 | 0 |
| E1_baseline_keep | val | 991 | 0.859738 | 0.678659 | 0.875441 | 0.45 | 1 | 0 | 36 | 0.0776993 | 0.85 |
| E1_baseline_keep | test | 7302 | 0.658176 | 0.565403 | 0.803247 | 0.542317 | 0.678457 | 100 | 863 | 0.256231 | 0.00321543 |
| E1_subtype_aux_keep | train | 21117 | 0.763271 | 0.696646 | 0.899469 | 0.265407 | 0.993284 | 32 | 1161 | 0.588436 | 0.238615 |
| E1_subtype_aux_keep | val | 991 | 0.843592 | 0.614367 | 0.87309 | 0.216667 | 1 | 0 | 50 | 0.76892 | 0.0625 |
| E1_subtype_aux_keep | test | 7302 | 0.492331 | 0.429199 | 0.758766 | 0.251854 | 0.877814 | 38 | 1471 | 0.450561 | 0.398714 |

## Output Files

- Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_v12_bad_subtype_aux_sanity\ptb_v12_to_but_keepoutlier_metrics.csv`
- Subtype CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_v12_bad_subtype_aux_sanity\ptb_v12_to_but_keepoutlier_subtype_metrics.csv`