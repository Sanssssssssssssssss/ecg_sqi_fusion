# Waveform Geometry Student Continuation - 2026-06-15

## Executive Readout

- Pure waveform Transformer variants still do not reach 0.90 on the held-out BUT test split. The best new pure-waveform result in this continuation remains around 0.812 original_test accuracy, despite synthetic/node accuracy near 0.995.
- Deterministic waveform-stat probes also top out around 0.787 on original_test, so the missing signal is not just Transformer capacity; it is the atlas-level geometry carried by pc1/pca_margin/boundary_confidence/knn_label_purity-style features.
- A feature-assisted Transformer over selected SQI/geometry tokens does cross 0.90: featuretx_top22 reaches original_test_all_10s+ acc 0.910935 and original_all_10s+ acc 0.967108. This uses the current BUT/node train/val split and is not PTB-only waveform inference.

## Best Rows By Bucket

| source_report | candidate | bucket | acc | macro_f1 | good_recall | medium_recall | bad_recall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| feature_token_transformer | featuretx_top22 | node_test | 0.910935 | 0.909913 | 0.996429 | 0.846588 | 0.846715 |
| feature_token_transformer | featuretx_top20 | node_test | 0.908458 | 0.932228 | 0.999725 | 0.825124 | 0.997567 |
| feature_token_transformer | featuretx_top14 | node_test | 0.878849 | 0.773303 | 0.990385 | 0.833710 | 0.377129 |
| feature_token_transformer | featuretx_top22 | original_all_10s+ | 0.967108 | 0.970435 | 0.993546 | 0.914283 | 0.988079 |
| feature_token_transformer | featuretx_top20 | original_all_10s+ | 0.965864 | 0.970558 | 0.996127 | 0.900546 | 0.999622 |
| feature_token_transformer | featuretx_top14 | original_all_10s+ | 0.964862 | 0.964877 | 0.992607 | 0.927362 | 0.950804 |
| waveform_stats_probe | logreg_wave_stats | original_all_10s+ | 0.838542 | 0.859687 | 0.828551 | 0.807866 | 0.932450 |
| waveform_student_search | featurefusion_stressbank_teacher | original_all_10s+ | 0.828529 | 0.853457 | 0.766590 | 0.875047 | 0.934721 |
| waveform_student_search | featurefusion_stressbank_teacher_badcal | original_all_10s+ | 0.825495 | 0.846762 | 0.761544 | 0.872318 | 0.937559 |
| waveform_stats_probe | hgb_wave_stats | original_all_10s+ | 0.824888 | 0.845174 | 0.844159 | 0.740779 | 0.931883 |
| waveform_stats_probe | extratrees_wave_stats | original_all_10s+ | 0.821611 | 0.848442 | 0.781553 | 0.830918 | 0.932072 |
| feature_token_transformer | featuretx_top22 | original_test_all_10s+ | 0.910935 | 0.909913 | 0.996429 | 0.846588 | 0.846715 |
| feature_token_transformer | featuretx_top20 | original_test_all_10s+ | 0.908458 | 0.932228 | 0.999725 | 0.825124 | 0.997567 |
| feature_token_transformer | featuretx_top14 | original_test_all_10s+ | 0.878849 | 0.773303 | 0.990385 | 0.833710 | 0.377129 |
| waveform_student_search | featurefusion_multiscale_core | original_test_all_10s+ | 0.812434 | 0.681101 | 0.874176 | 0.812020 | 0.270073 |
| waveform_student_search | featurefusion_multiscale_core_badcal | original_test_all_10s+ | 0.807479 | 0.676055 | 0.868956 | 0.799593 | 0.347932 |
| waveform_student_search | featurefusion_stressbank_teacher | original_test_all_10s+ | 0.802289 | 0.689333 | 0.900000 | 0.767058 | 0.316302 |
| waveform_student_search | featurefusion_stressbank_teacher_badcal | original_test_all_10s+ | 0.800401 | 0.678255 | 0.895604 | 0.764347 | 0.345499 |
| waveform_stats_probe | rf_wave_stats | original_test_all_10s+ | 0.786835 | 0.681606 | 0.897527 | 0.741979 | 0.289538 |
| waveform_stats_probe | logreg_wave_stats | synthetic_test | 0.995900 | 0.994103 | 0.995816 | 0.995942 | 0.995851 |
| waveform_stats_probe | hgb_wave_stats | synthetic_test | 0.995900 | 0.995441 | 0.995816 | 0.996753 | 0.991701 |
| waveform_student_search | featurefusion_statfed_teacher | synthetic_test | 0.995387 | 0.993892 | 0.991632 | 1.000000 | 0.979253 |
| waveform_student_search | featurefusion_statfed_teacher_badcal | synthetic_test | 0.995387 | 0.993337 | 0.991632 | 1.000000 | 0.979253 |
| waveform_student_search | featurefusion_stressbank_teacher | synthetic_test | 0.995387 | 0.993892 | 0.991632 | 1.000000 | 0.979253 |
| waveform_stats_probe | extratrees_wave_stats | synthetic_test | 0.995387 | 0.994604 | 0.993724 | 0.997565 | 0.987552 |
| waveform_student_search | featurefusion_multiscale_core | synthetic_test | 0.993849 | 0.992432 | 0.985356 | 1.000000 | 0.979253 |
| waveform_student_search | featurefusion_multiscale_core_badcal | synthetic_test | 0.993849 | 0.992432 | 0.985356 | 1.000000 | 0.979253 |

## Interpretation

- The prototype-atlas and waveform-atlas heads help synthetic/node fit but do not transfer enough to the held-out BUT test split.
- The feature-token result is the current 90+ method, but it should be presented as feature-assisted BUT split learning, not as a PTB-only waveform student.
- For a pure waveform model, the next serious direction is not more class-weight sweeping; it is either stronger self-supervised pretraining to recover atlas geometry or accepting a compact SQI/geometry feature branch as part of the formal model.

Summary CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_geometry_student_continuation_summary_20260615.csv`
