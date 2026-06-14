# Waveform-Only UFormer + Auxiliary SQI/Geometry Prediction

The classifier input is ECG waveform only. The 47 SQI/geometry columns supervise an auxiliary regression head during training, but they are not passed to the classifier at inference.

## Held-Out Original BUT Test

| Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m | Aux MAE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| wave_aux_headonly_a025 | synthetic_test | 0.996925 | 0.995347 | 1.000000 | 0.999188 | 0.979253 | 0 | 1 | 5 | 0.6557 |
| wave_aux_headonly_a025_badcal | synthetic_test | 0.996925 | 0.995347 | 1.000000 | 0.999188 | 0.979253 | 0 | 1 | 5 | nan |
| wave_aux_headonly_a025 | original_test_all_10s+ | 0.753215 | 0.515774 | 0.819505 | 0.768640 | 0.000000 | 657 | 991 | 341 | nan |
| wave_aux_headonly_a025_badcal | original_test_all_10s+ | 0.752625 | 0.515542 | 0.819505 | 0.767510 | 0.000000 | 657 | 991 | 341 | nan |
| wave_aux_headonly_a025 | bad_core_nearboundary | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 119 | nan |
| wave_aux_headonly_a025_badcal | bad_core_nearboundary | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 119 | nan |
| wave_aux_headonly_a025 | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 222 | nan |
| wave_aux_headonly_a025_badcal | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 222 | nan |
| wave_aux_headonly_a075 | synthetic_test | 0.997437 | 0.996187 | 1.000000 | 0.999188 | 0.983402 | 0 | 1 | 4 | 0.6702 |
| wave_aux_headonly_a075_badcal | synthetic_test | 0.995387 | 0.992923 | 1.000000 | 0.995130 | 0.987552 | 0 | 1 | 3 | nan |
| wave_aux_headonly_a075 | original_test_all_10s+ | 0.751681 | 0.521818 | 0.773352 | 0.802756 | 0.009732 | 819 | 766 | 348 | nan |
| wave_aux_headonly_a075_badcal | original_test_all_10s+ | 0.745429 | 0.524533 | 0.773352 | 0.789652 | 0.021898 | 819 | 762 | 343 | nan |
| wave_aux_headonly_a075 | bad_core_nearboundary | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 119 | nan |
| wave_aux_headonly_a075_badcal | bad_core_nearboundary | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 119 | nan |
| wave_aux_headonly_a075 | bad_outlier_stress | 0.013699 | 0.009009 | 0.000000 | 0.000000 | 0.013699 | 0 | 0 | 229 | nan |
| wave_aux_headonly_a075_badcal | bad_outlier_stress | 0.030822 | 0.019934 | 0.000000 | 0.000000 | 0.030822 | 0 | 0 | 224 | nan |
| wave_aux_bottleneck_lowlr_a050 | synthetic_test | 0.997437 | 0.996187 | 1.000000 | 0.999188 | 0.983402 | 0 | 1 | 4 | 0.6651 |
| wave_aux_bottleneck_lowlr_a050_badcal | synthetic_test | 0.997437 | 0.996187 | 1.000000 | 0.999188 | 0.983402 | 0 | 1 | 4 | nan |
| wave_aux_bottleneck_lowlr_a050 | original_test_all_10s+ | 0.752625 | 0.519054 | 0.803022 | 0.780615 | 0.004866 | 717 | 905 | 344 | nan |
| wave_aux_bottleneck_lowlr_a050_badcal | original_test_all_10s+ | 0.752389 | 0.518950 | 0.803022 | 0.780163 | 0.004866 | 717 | 905 | 344 | nan |
| wave_aux_bottleneck_lowlr_a050 | bad_core_nearboundary | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 119 | nan |
| wave_aux_bottleneck_lowlr_a050_badcal | bad_core_nearboundary | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 119 | nan |
| wave_aux_bottleneck_lowlr_a050 | bad_outlier_stress | 0.006849 | 0.004535 | 0.000000 | 0.000000 | 0.006849 | 0 | 0 | 225 | nan |
| wave_aux_bottleneck_lowlr_a050_badcal | bad_outlier_stress | 0.006849 | 0.004535 | 0.000000 | 0.000000 | 0.006849 | 0 | 0 | 225 | nan |

## Reference

- 47-feature tabular current best original test acc: `0.963548`; recalls good/medium/bad `0.956/0.973/0.927`.

## Candidate Configs

- `wave_aux_headonly_a025`: frozen_backbone=True, best_epoch=2, bad_threshold_trainval=0.42, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\waveform_aux_sqi_geometry\N17043_gm_probe\wave_aux_headonly_a025\ckpt_best.pt`
- `wave_aux_headonly_a075`: frozen_backbone=True, best_epoch=4, bad_threshold_trainval=0.30, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\waveform_aux_sqi_geometry\N17043_gm_probe\wave_aux_headonly_a075\ckpt_best.pt`
- `wave_aux_bottleneck_lowlr_a050`: frozen_backbone=False, best_epoch=4, bad_threshold_trainval=0.47, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\waveform_aux_sqi_geometry\N17043_gm_probe\wave_aux_bottleneck_lowlr_a050\ckpt_best.pt`

## Notes

- Original BUT rows are report-only and are never used for training, feature normalization, threshold selection, or model selection.
- This is a normal neural checkpoint experiment, not a rule artifact and not a tabular-input model.
- Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_aux_sqi_geometry_metrics.csv`
