# Phase 0 Correctness Report

- Generated: 2026-06-20 01:06:27
- Policy: `margin_ge_5s_drop_outlier`
- Scope: external experiment runner only; no `src/sqi_pipeline` changes.
- Artifact target rule excludes `clean_policy` text. `clean_policy=margin_ge_5s_drop_outlier` is no longer treated as artifact evidence.

## Acceptance Checks

- `artifact_positive_nonbad_count`: `2780`
- `artifact_positive_nonbad_equals_all_nonbad`: `False`
- `clean_policy_used_as_artifact_evidence`: `False`
- `detector_grad_norm`: `1.68694`
- `detector_same_value_in_factor_and_report`: `True`
- `detector_range`: `{'min': 0.8716700077056885, 'max': 0.9380949139595032, 'mean': 0.9115002155303955}`

## Artifact x Class Crosstab

| split | class_name | False | True | All |
| --- | --- | --- | --- | --- |
| test | bad | 0 | 118 | 118 |
| test | good | 776 | 228 | 1004 |
| test | medium | 1499 | 578 | 2077 |
| train | bad | 0 | 3963 | 3963 |
| train | good | 8441 | 1162 | 9603 |
| train | medium | 3531 | 614 | 4145 |
| val | bad | 0 | 1 | 1 |
| val | good | 446 | 175 | 621 |
| val | medium | 20 | 23 | 43 |
| All |  | 14713 | 6862 | 21575 |

## Record Support Preview

| split | record_id | bad | good | medium |
| --- | --- | --- | --- | --- |
| test | 111001 | 0 | 927 | 2053 |
| test | 122001 | 118 | 70 | 24 |
| test | 125001 | 0 | 7 | 0 |
| train | 100001 | 0 | 4457 | 1669 |
| train | 100002 | 0 | 56 | 23 |
| train | 104001 | 0 | 0 | 42 |
| train | 105001 | 3963 | 4554 | 2250 |
| train | 113001 | 0 | 125 | 29 |
| train | 115001 | 0 | 116 | 22 |
| train | 118001 | 0 | 116 | 29 |
| train | 121001 | 0 | 89 | 0 |
| train | 123001 | 0 | 90 | 2 |
| train | 124001 | 0 | 0 | 79 |
| val | 103001 | 0 | 95 | 12 |
| val | 103002 | 0 | 175 | 0 |
| val | 103003 | 0 | 37 | 1 |
| val | 114001 | 1 | 209 | 30 |
| val | 126001 | 0 | 105 | 0 |

## Output Files

- Artifact audit CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_sqi_conformer\phase0_correctness\artifact_targets_audit.csv`
- Gradient audit CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_sqi_conformer\phase0_correctness\gradient_audit_onebatch.csv`
- Summary JSON: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_sqi_conformer\phase0_correctness\phase0_correctness_summary.json`