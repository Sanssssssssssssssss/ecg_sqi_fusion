# Phase 0 Correctness Report

- Generated: 2026-06-25 04:07:27
- Policy: `ptb_v112_gm_buffered_large_hybrid_s20260741`
- Scope: external experiment runner only; no `src/sqi_pipeline` changes.
- Artifact target rule excludes `clean_policy` text. `clean_policy=margin_ge_5s_drop_outlier` is no longer treated as artifact evidence.

## Acceptance Checks

- `artifact_positive_nonbad_count`: `2498`
- `artifact_positive_nonbad_equals_all_nonbad`: `False`
- `clean_policy_used_as_artifact_evidence`: `False`
- `detector_grad_norm`: `2.0261`
- `detector_same_value_in_factor_and_report`: `True`
- `detector_range`: `{'min': 0.8845213651657104, 'max': 0.9471899271011353, 'mean': 0.9229362607002258}`

## Artifact x Class Crosstab

| split | class_name | False | True | All |
| --- | --- | --- | --- | --- |
| test | bad | 0 | 904 | 904 |
| test | good | 186 | 199 | 385 |
| test | medium | 331 | 184 | 515 |
| train | bad | 0 | 4200 | 4200 |
| train | good | 926 | 887 | 1813 |
| train | medium | 1570 | 831 | 2401 |
| val | bad | 0 | 896 | 896 |
| val | good | 209 | 208 | 417 |
| val | medium | 354 | 189 | 543 |
| All |  | 3576 | 8498 | 12074 |

## Record Support Preview

| split | record_id | bad | good | medium |
| --- | --- | --- | --- | --- |
| test | v108_ptb_ptbxl_10006 | 1 | 0 | 0 |
| test | v108_ptb_ptbxl_10018 | 1 | 0 | 0 |
| test | v108_ptb_ptbxl_10070 | 1 | 0 | 0 |
| test | v108_ptb_ptbxl_10101 | 1 | 0 | 0 |
| test | v108_ptb_ptbxl_1011 | 1 | 0 | 0 |
| test | v108_ptb_ptbxl_10137 | 1 | 0 | 0 |
| test | v108_ptb_ptbxl_10166 | 1 | 0 | 0 |
| test | v108_ptb_ptbxl_10176 | 1 | 0 | 0 |
| test | v108_ptb_ptbxl_10179 | 1 | 0 | 0 |
| test | v108_ptb_ptbxl_10191 | 1 | 0 | 0 |
| test | v108_ptb_ptbxl_102 | 1 | 0 | 0 |
| test | v108_ptb_ptbxl_10206 | 1 | 0 | 0 |
| test | v108_ptb_ptbxl_10240 | 1 | 0 | 0 |
| test | v108_ptb_ptbxl_1031 | 1 | 0 | 0 |
| test | v108_ptb_ptbxl_10321 | 1 | 0 | 0 |
| test | v108_ptb_ptbxl_10332 | 1 | 0 | 0 |
| test | v108_ptb_ptbxl_10435 | 1 | 0 | 0 |
| test | v108_ptb_ptbxl_10444 | 1 | 0 | 0 |
| test | v108_ptb_ptbxl_10445 | 1 | 0 | 0 |
| test | v108_ptb_ptbxl_10455 | 1 | 0 | 0 |
| test | v108_ptb_ptbxl_10462 | 1 | 0 | 0 |
| test | v108_ptb_ptbxl_10496 | 1 | 0 | 0 |
| test | v108_ptb_ptbxl_10550 | 1 | 0 | 0 |
| test | v108_ptb_ptbxl_1056 | 1 | 0 | 0 |
| test | v108_ptb_ptbxl_10573 | 1 | 0 | 0 |
| test | v108_ptb_ptbxl_10606 | 1 | 0 | 0 |
| test | v108_ptb_ptbxl_10621 | 1 | 0 | 0 |
| test | v108_ptb_ptbxl_10701 | 1 | 0 | 0 |
| test | v108_ptb_ptbxl_10702 | 1 | 0 | 0 |
| test | v108_ptb_ptbxl_10727 | 1 | 0 | 0 |

## Output Files

- Artifact audit CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_sqi_conformer\phase0_correctness\artifact_targets_audit.csv`
- Gradient audit CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_sqi_conformer\phase0_correctness\gradient_audit_onebatch.csv`
- Summary JSON: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_sqi_conformer\phase0_correctness\phase0_correctness_summary.json`