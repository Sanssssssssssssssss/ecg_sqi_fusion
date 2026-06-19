# LocalSQI Query Conformer stage_a Report

- Created: 2026-06-19 23:20:41
- Policy: `margin_ge_5s_drop_outlier`
- Fold: `4`

## Metrics

| candidate | bucket | acc | macro_f1 | good_recall | medium_recall | bad_recall | gm_balanced_acc | bad_fpr_nonbad | artifact_positive_nonbad_bad_fpr | factor_mae |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A0_nohi_noquery | clean_val | 0.996865 | 0.997679 | 0.995215 | 1 | 1 | 0.997608 | 0 | 0 | 1.50076 |
| A0_nohi_noquery | clean_test | 0.958968 | 0.927858 | 0.952744 | 0.949367 | 1 | 0.951055 | 0 | 0 | 1.43323 |
| A1_hi_noquery | clean_val | 0.755486 | 0.50356 | 0.631579 | 1 | 0 | 0.815789 | 0 | 0 | 1.52452 |
| A1_hi_noquery | clean_test | 0.872216 | 0.833718 | 0.833841 | 1 | 1 | 0.916921 | 0 | 0 | 1.53156 |
| A2_nohi_query | clean_val | 0.962382 | 0.640266 | 0.947368 | 1 | 0 | 0.973684 | 0 | 0 | 1.44369 |
| A2_nohi_query | clean_test | 0.878077 | 0.838017 | 0.842988 | 0.987342 | 1 | 0.915165 | 0 | 0 | 1.35717 |
| A3_hi_query | clean_val | 0.965517 | 0.642485 | 0.952153 | 1 | 0 | 0.976077 | 0 | 0 | 1.68482 |
| A3_hi_query | clean_test | 0.881594 | 0.842212 | 0.846037 | 1 | 1 | 0.923018 | 0 | 0 | 1.55751 |

## Factor Recovery Corr

| candidate | qrs_visibility | qrs_band_ratio | template_corr | baseline_step | flatline_ratio | contact_loss_win_ratio | non_qrs_diff_p95 | amplitude_entropy | sqi_basSQI | detector_agreement |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A0_nohi_noquery | 0.680744 | -0.530256 | 0.802007 | 0.716994 | 0.829899 | 0 | 0.985153 | 0.918056 | 0.532283 | -0.561657 |
| A1_hi_noquery | 0.654468 | -0.0934409 | 0.79688 | 0.596187 | 0.858273 | 0 | 0.965321 | 0.968047 | 0.661251 | 0.0553196 |
| A2_nohi_query | 0.664208 | -0.624619 | 0.841105 | 0.675693 | 0.827155 | 0 | 0.980051 | 0.943937 | 0.511373 | 0.418706 |
| A3_hi_query | 0.672642 | -0.535545 | 0.828501 | 0.692693 | 0.850013 | 0 | 0.975408 | 0.919514 | 0.622757 | 0.504038 |

## Interpretation Contract

- A1 > A0 suggests early high-resolution evidence matters.
- A2 > A0 suggests task query readout matters.
- A3 > A1/A2 suggests both local evidence and task readout are needed.
- B3 > B1/B2 suggests local supervision and coherent hierarchical probability are both useful.