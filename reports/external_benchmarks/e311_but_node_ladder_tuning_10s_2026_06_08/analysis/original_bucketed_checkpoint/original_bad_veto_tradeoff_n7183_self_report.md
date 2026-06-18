# Original Bad-Veto Tradeoff Analysis

Report-only. Original BUT is used here only to explain domain gaps, not for model selection.

## What This Tests

- Base prediction: `nl_n7183_gm_trim_bad_boundaryblocks_badattackwide_dual_n7_4868958ba680` / `simple_pc1_gm_gate_t254`.
- Bad evidence: raw bad probabilities from `nl_n7183_gm_trim_bad_boundaryblocks_badattackwide_dual_n7_4868958ba680`, `nl_n7183_gm_trim_bad_boundaryblocks_badattackwide_dual_n7_4868958ba680`, `nl_n7183_gm_trim_bad_boundaryblocks_badattackwide_dual_n7_4868958ba680`.
- Search space: a bad-score threshold plus optional one-feature gate (`pc1`, `pc2`, `pc3`, `qrs_visibility`).

## Top Balanced Report-Only Rules

| score_col | score_threshold | gate | gate_threshold | test_all_acc | test_all_good_recall | test_all_medium_recall | test_all_bad_recall | bad_core_bad_recall | bad_outlier_bad_recall | gm_false_bad_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| score_dual | 0.0100 | pc3__le | 2.1507 | 0.7687 | 0.9124 | 0.6853 | 0.3942 | 1.0000 | 0.1473 | 0.0761 |
| score_all_max | 0.0100 | pc3__le | 2.1507 | 0.7687 | 0.9124 | 0.6853 | 0.3942 | 1.0000 | 0.1473 | 0.0761 |
| score_core | 0.0100 | pc3__le | 2.1507 | 0.7687 | 0.9124 | 0.6853 | 0.3942 | 1.0000 | 0.1473 | 0.0761 |
| score_outlier | 0.0100 | pc3__le | 2.1507 | 0.7687 | 0.9124 | 0.6853 | 0.3942 | 1.0000 | 0.1473 | 0.0761 |
| score_dual | 0.0100 | pc3__le | 3.2812 | 0.7654 | 0.9124 | 0.6789 | 0.3942 | 1.0000 | 0.1473 | 0.0797 |
| score_all_max | 0.0100 | pc3__le | 3.2812 | 0.7654 | 0.9124 | 0.6789 | 0.3942 | 1.0000 | 0.1473 | 0.0797 |
| score_core | 0.0100 | pc3__le | 3.2812 | 0.7654 | 0.9124 | 0.6789 | 0.3942 | 1.0000 | 0.1473 | 0.0797 |
| score_outlier | 0.0100 | pc3__le | 3.2812 | 0.7654 | 0.9124 | 0.6789 | 0.3942 | 1.0000 | 0.1473 | 0.0797 |
| score_dual | 0.0100 | none |  | 0.7651 | 0.9124 | 0.6785 | 0.3942 | 1.0000 | 0.1473 | 0.0800 |
| score_dual | 0.0100 | qrs_visibility__le | 0.2801 | 0.7651 | 0.9124 | 0.6785 | 0.3942 | 1.0000 | 0.1473 | 0.0800 |
| score_dual | 0.0100 | qrs_visibility__le | 0.3732 | 0.7651 | 0.9124 | 0.6785 | 0.3942 | 1.0000 | 0.1473 | 0.0800 |
| score_all_max | 0.0100 | none |  | 0.7651 | 0.9124 | 0.6785 | 0.3942 | 1.0000 | 0.1473 | 0.0800 |
| score_all_max | 0.0100 | qrs_visibility__le | 0.2801 | 0.7651 | 0.9124 | 0.6785 | 0.3942 | 1.0000 | 0.1473 | 0.0800 |
| score_all_max | 0.0100 | qrs_visibility__le | 0.3732 | 0.7651 | 0.9124 | 0.6785 | 0.3942 | 1.0000 | 0.1473 | 0.0800 |
| score_core | 0.0100 | none |  | 0.7651 | 0.9124 | 0.6785 | 0.3942 | 1.0000 | 0.1473 | 0.0800 |

## Highest Bad Recall Rules

| score_col | score_threshold | gate | gate_threshold | test_all_acc | test_all_good_recall | test_all_medium_recall | test_all_bad_recall | bad_core_bad_recall | bad_outlier_bad_recall | gm_false_bad_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| score_dual | 0.0100 | pc3__le | 2.1507 | 0.7687 | 0.9124 | 0.6853 | 0.3942 | 1.0000 | 0.1473 | 0.0761 |
| score_all_max | 0.0100 | pc3__le | 2.1507 | 0.7687 | 0.9124 | 0.6853 | 0.3942 | 1.0000 | 0.1473 | 0.0761 |
| score_core | 0.0100 | pc3__le | 2.1507 | 0.7687 | 0.9124 | 0.6853 | 0.3942 | 1.0000 | 0.1473 | 0.0761 |
| score_outlier | 0.0100 | pc3__le | 2.1507 | 0.7687 | 0.9124 | 0.6853 | 0.3942 | 1.0000 | 0.1473 | 0.0761 |
| score_dual | 0.0100 | pc3__le | 3.2812 | 0.7654 | 0.9124 | 0.6789 | 0.3942 | 1.0000 | 0.1473 | 0.0797 |
| score_all_max | 0.0100 | pc3__le | 3.2812 | 0.7654 | 0.9124 | 0.6789 | 0.3942 | 1.0000 | 0.1473 | 0.0797 |
| score_core | 0.0100 | pc3__le | 3.2812 | 0.7654 | 0.9124 | 0.6789 | 0.3942 | 1.0000 | 0.1473 | 0.0797 |
| score_outlier | 0.0100 | pc3__le | 3.2812 | 0.7654 | 0.9124 | 0.6789 | 0.3942 | 1.0000 | 0.1473 | 0.0797 |
| score_dual | 0.0100 | none |  | 0.7651 | 0.9124 | 0.6785 | 0.3942 | 1.0000 | 0.1473 | 0.0800 |
| score_dual | 0.0100 | qrs_visibility__le | 0.2801 | 0.7651 | 0.9124 | 0.6785 | 0.3942 | 1.0000 | 0.1473 | 0.0800 |
| score_dual | 0.0100 | qrs_visibility__le | 0.3732 | 0.7651 | 0.9124 | 0.6785 | 0.3942 | 1.0000 | 0.1473 | 0.0800 |
| score_all_max | 0.0100 | none |  | 0.7651 | 0.9124 | 0.6785 | 0.3942 | 1.0000 | 0.1473 | 0.0800 |
| score_all_max | 0.0100 | qrs_visibility__le | 0.2801 | 0.7651 | 0.9124 | 0.6785 | 0.3942 | 1.0000 | 0.1473 | 0.0800 |
| score_all_max | 0.0100 | qrs_visibility__le | 0.3732 | 0.7651 | 0.9124 | 0.6785 | 0.3942 | 1.0000 | 0.1473 | 0.0800 |
| score_core | 0.0100 | none |  | 0.7651 | 0.9124 | 0.6785 | 0.3942 | 1.0000 | 0.1473 | 0.0800 |

## Accuracy-Preserving Rules With Bad Recall >= 0.30

| score_col | score_threshold | gate | gate_threshold | test_all_acc | test_all_good_recall | test_all_medium_recall | test_all_bad_recall | bad_core_bad_recall | bad_outlier_bad_recall | gm_false_bad_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| score_core | 0.0100 | pc2__le | 1.9185 | 0.7720 | 0.9140 | 0.6948 | 0.3455 | 1.0000 | 0.0788 | 0.0502 |
| score_outlier | 0.0100 | pc2__le | 1.9185 | 0.7720 | 0.9140 | 0.6948 | 0.3455 | 1.0000 | 0.0788 | 0.0502 |
| score_dual | 0.0100 | pc2__le | 1.9185 | 0.7720 | 0.9140 | 0.6948 | 0.3455 | 1.0000 | 0.0788 | 0.0502 |
| score_all_max | 0.0100 | pc2__le | 1.9185 | 0.7720 | 0.9140 | 0.6948 | 0.3455 | 1.0000 | 0.0788 | 0.0502 |
| score_core | 0.0100 | pc3__le | 0.0403 | 0.7719 | 0.9124 | 0.6943 | 0.3625 | 0.9160 | 0.1370 | 0.0691 |
| score_outlier | 0.0100 | pc3__le | 0.0403 | 0.7719 | 0.9124 | 0.6943 | 0.3625 | 0.9160 | 0.1370 | 0.0691 |
| score_dual | 0.0100 | pc3__le | 0.0403 | 0.7719 | 0.9124 | 0.6943 | 0.3625 | 0.9160 | 0.1370 | 0.0691 |
| score_all_max | 0.0100 | pc3__le | 0.0403 | 0.7719 | 0.9124 | 0.6943 | 0.3625 | 0.9160 | 0.1370 | 0.0691 |
| score_dual | 0.0200 | pc2__le | 1.9185 | 0.7719 | 0.9140 | 0.6948 | 0.3431 | 0.9916 | 0.0788 | 0.0502 |
| score_all_max | 0.0200 | pc2__le | 1.9185 | 0.7719 | 0.9140 | 0.6948 | 0.3431 | 0.9916 | 0.0788 | 0.0502 |

## Score Distribution Summary

| score_col | bucket | n | mean | p50 | p75 | p90 | p95 | p99 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| score_core | bad_core | 119 | 0.8364 | 0.9619 | 0.9861 | 0.9951 | 0.9979 | 0.9998 |
| score_core | bad_outlier | 292 | 0.0857 | 0.0002 | 0.0010 | 0.0761 | 0.9828 | 0.9999 |
| score_core | good | 3640 | 0.0052 | 0.0000 | 0.0000 | 0.0001 | 0.0002 | 0.0029 |
| score_core | medium | 4426 | 0.0880 | 0.0000 | 0.0003 | 0.1257 | 0.9960 | 1.0000 |
| score_outlier | bad_core | 119 | 0.8364 | 0.9619 | 0.9861 | 0.9951 | 0.9979 | 0.9998 |
| score_outlier | bad_outlier | 292 | 0.0857 | 0.0002 | 0.0010 | 0.0761 | 0.9828 | 0.9999 |
| score_outlier | good | 3640 | 0.0052 | 0.0000 | 0.0000 | 0.0001 | 0.0002 | 0.0029 |
| score_outlier | medium | 4426 | 0.0880 | 0.0000 | 0.0003 | 0.1257 | 0.9960 | 1.0000 |
| score_dual | bad_core | 119 | 0.8364 | 0.9619 | 0.9861 | 0.9951 | 0.9979 | 0.9998 |
| score_dual | bad_outlier | 292 | 0.0857 | 0.0002 | 0.0010 | 0.0761 | 0.9828 | 0.9999 |
| score_dual | good | 3640 | 0.0052 | 0.0000 | 0.0000 | 0.0001 | 0.0002 | 0.0029 |
| score_dual | medium | 4426 | 0.0880 | 0.0000 | 0.0003 | 0.1257 | 0.9960 | 1.0000 |
| score_all_max | bad_core | 119 | 0.8364 | 0.9619 | 0.9861 | 0.9951 | 0.9979 | 0.9998 |
| score_all_max | bad_outlier | 292 | 0.0857 | 0.0002 | 0.0010 | 0.0761 | 0.9828 | 0.9999 |
| score_all_max | good | 3640 | 0.0052 | 0.0000 | 0.0000 | 0.0001 | 0.0002 | 0.0029 |
| score_all_max | medium | 4426 | 0.0880 | 0.0000 | 0.0003 | 0.1257 | 0.9960 | 1.0000 |

## Interpretation

- Clean/node split says the bad specialist is useful; original says the same score is miscalibrated and sweeps many good/medium rows into bad.
- A simple bad-veto branch is promising, but the original threshold needs either domain calibration or a second simple geometry gate.
- The next training-side experiment should therefore be a decoupled bad-veto/head-style objective, not another broad class-weight sweep.

![score distributions](original_bad_veto_tradeoff_n7183_self_score_distributions.png)

![bad tradeoff](original_bad_veto_tradeoff_n7183_self_bad_tradeoff.png)
