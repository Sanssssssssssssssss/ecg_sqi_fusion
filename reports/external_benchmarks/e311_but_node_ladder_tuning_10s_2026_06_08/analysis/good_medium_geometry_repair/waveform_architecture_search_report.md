# Waveform-Only Architecture Search

Classifier input is waveform channels only. The 47 SQI/geometry columns supervise an auxiliary head during training but are not passed into the classifier.

## Key Metrics

| Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m | Aux MAE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| resnet_raw1_aux | synthetic_test | 0.994362 | 0.992955 | 0.993724 | 0.995942 | 0.987552 | 3 | 3 | 3 | 0.6466 |
| resnet_raw1_aux_badcal | synthetic_test | 0.994362 | 0.992955 | 0.993724 | 0.995942 | 0.987552 | 3 | 3 | 3 | nan |
| resnet_raw1_aux | original_test_all_10s+ | 0.840510 | 0.716485 | 0.755495 | 0.961591 | 0.289538 | 890 | 170 | 268 | nan |
| resnet_raw1_aux_badcal | original_test_all_10s+ | 0.840510 | 0.716485 | 0.755495 | 0.961591 | 0.289538 | 890 | 170 | 268 | nan |
| resnet_raw1_aux | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 | nan |
| resnet_raw1_aux_badcal | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 | nan |
| resnet_raw1_aux | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 268 | nan |
| resnet_raw1_aux_badcal | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 268 | nan |
| resnet_robust3_badstrong_aux | synthetic_test | 0.996412 | 0.994864 | 1.000000 | 0.998377 | 0.979253 | 0 | 2 | 5 | 0.6476 |
| resnet_robust3_badstrong_aux_badcal | synthetic_test | 0.996412 | 0.994864 | 1.000000 | 0.998377 | 0.979253 | 0 | 2 | 5 | nan |
| resnet_robust3_badstrong_aux | original_test_all_10s+ | 0.558216 | 0.433418 | 0.086538 | 0.971080 | 0.289538 | 3325 | 128 | 273 | nan |
| resnet_robust3_badstrong_aux_badcal | original_test_all_10s+ | 0.558216 | 0.433418 | 0.086538 | 0.971080 | 0.289538 | 3325 | 128 | 273 | nan |
| resnet_robust3_badstrong_aux | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 | nan |
| resnet_robust3_badstrong_aux_badcal | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 | nan |
| resnet_robust3_badstrong_aux | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 273 | nan |
| resnet_robust3_badstrong_aux_badcal | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 273 | nan |
| resnet_robust3_focal_bad | synthetic_test | 0.993849 | 0.992818 | 1.000000 | 0.993506 | 0.983402 | 0 | 8 | 4 | 0.6485 |
| resnet_robust3_focal_bad_badcal | synthetic_test | 0.993849 | 0.992818 | 1.000000 | 0.993506 | 0.983402 | 0 | 8 | 4 | nan |
| resnet_robust3_focal_bad | original_test_all_10s+ | 0.865047 | 0.733907 | 0.893681 | 0.894939 | 0.289538 | 387 | 460 | 211 | nan |
| resnet_robust3_focal_bad_badcal | original_test_all_10s+ | 0.865047 | 0.733907 | 0.893681 | 0.894939 | 0.289538 | 387 | 460 | 211 | nan |
| resnet_robust3_focal_bad | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 | nan |
| resnet_robust3_focal_bad_badcal | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 | nan |
| resnet_robust3_focal_bad | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 211 | nan |
| resnet_robust3_focal_bad_badcal | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 211 | nan |
| resnet_robust3_wide96_aux | synthetic_test | 0.990261 | 0.989841 | 1.000000 | 0.987013 | 0.987552 | 0 | 16 | 3 | 0.6533 |
| resnet_robust3_wide96_aux_badcal | synthetic_test | 0.990261 | 0.989841 | 1.000000 | 0.987013 | 0.987552 | 0 | 16 | 3 | nan |
| resnet_robust3_wide96_aux | original_test_all_10s+ | 0.776218 | 0.668459 | 0.684341 | 0.896972 | 0.289538 | 1149 | 449 | 224 | nan |
| resnet_robust3_wide96_aux_badcal | original_test_all_10s+ | 0.776100 | 0.668144 | 0.684341 | 0.896746 | 0.289538 | 1149 | 449 | 224 | nan |
| resnet_robust3_wide96_aux | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 | nan |
| resnet_robust3_wide96_aux_badcal | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 | nan |
| resnet_robust3_wide96_aux | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 224 | nan |
| resnet_robust3_wide96_aux_badcal | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 224 | nan |

## Reference

- 47-feature tabular current best original test acc: `0.963548`; recalls good/medium/bad `0.956/0.973/0.927`.

## Candidate Configs

- `resnet_raw1_aux`: best_epoch=4, threshold=0.48, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\waveform_architecture_search\N17043_gm_probe\resnet_raw1_aux\ckpt_best.pt`
- `resnet_robust3_badstrong_aux`: best_epoch=8, threshold=0.43, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\waveform_architecture_search\N17043_gm_probe\resnet_robust3_badstrong_aux\ckpt_best.pt`
- `resnet_robust3_focal_bad`: best_epoch=3, threshold=0.48, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\waveform_architecture_search\N17043_gm_probe\resnet_robust3_focal_bad\ckpt_best.pt`
- `resnet_robust3_wide96_aux`: best_epoch=8, threshold=0.39, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\waveform_architecture_search\N17043_gm_probe\resnet_robust3_wide96_aux\ckpt_best.pt`

## Notes

- Original BUT rows are report-only and are never used for training or model selection.
- This search tests whether architecture + waveform transforms can close the gap without tabular inputs.
- Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_architecture_search_metrics.csv`
