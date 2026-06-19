# LocalSQI Query Conformer stage_b Report

- Created: 2026-06-19 23:31:57
- Policy: `margin_ge_5s_drop_outlier`
- Fold: `4`

## Metrics

| candidate | bucket | acc | macro_f1 | good_recall | medium_recall | bad_recall | gm_balanced_acc | bad_fpr_nonbad | artifact_positive_nonbad_bad_fpr | factor_mae |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0_query_ce_nolocal | clean_val | 0.557994 | 0.367882 | 0.330144 | 1 | 0 | 0.665072 | 0 | 0 | 1.65619 |
| B0_query_ce_nolocal | clean_test | 0.867526 | 0.829594 | 0.827744 | 1 | 1 | 0.913872 | 0 | 0 | 1.57499 |
| B1_query_ce_local | clean_val | 0.971787 | 0.979449 | 0.956938 | 1 | 1 | 0.978469 | 0 | 0 | 1.49022 |
| B1_query_ce_local | clean_test | 0.902696 | 0.861765 | 0.875 | 0.987342 | 1 | 0.931171 | 0 | 0 | 1.41718 |
| B2_query_hier_nolocal | clean_val | 0.874608 | 0.580686 | 0.813397 | 1 | 0 | 0.906699 | 0 | 0 | 1.60903 |
| B2_query_hier_nolocal | clean_test | 0.873388 | 0.834761 | 0.835366 | 1 | 1 | 0.917683 | 0 | 0 | 1.50301 |
| B3_query_hier_local | clean_val | 0.557994 | 0.367882 | 0.330144 | 1 | 0 | 0.665072 | 0 | 0 | 1.7175 |
| B3_query_hier_local | clean_test | 0.872216 | 0.833718 | 0.833841 | 1 | 1 | 0.916921 | 0 | 0 | 1.58649 |

## Factor Recovery Corr

| candidate | qrs_visibility | qrs_band_ratio | template_corr | baseline_step | flatline_ratio | contact_loss_win_ratio | non_qrs_diff_p95 | amplitude_entropy | sqi_basSQI | detector_agreement |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0_query_ce_nolocal | 0.615715 | -0.634709 | 0.829504 | 0.705443 | 0.834109 | 0 | 0.971948 | 0.918461 | 0.444415 | 0.481719 |
| B1_query_ce_local | 0.658794 | -0.642139 | 0.821559 | 0.520308 | 0.827927 | 0 | 0.982081 | 0.845591 | 0.480153 | 0.0387524 |
| B2_query_hier_nolocal | 0.671798 | -0.634069 | 0.82508 | 0.683982 | 0.832951 | 0 | 0.975662 | 0.903388 | 0.515223 | -0.421686 |
| B3_query_hier_local | 0.574422 | -0.436947 | 0.830139 | 0.668383 | 0.869298 | 0 | 0.967882 | 0.940592 | 0.64157 | -0.341958 |

## Interpretation Contract

- A1 > A0 suggests early high-resolution evidence matters.
- A2 > A0 suggests task query readout matters.
- A3 > A1/A2 suggests both local evidence and task readout are needed.
- B3 > B1/B2 suggests local supervision and coherent hierarchical probability are both useful.