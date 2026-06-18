# Waveform-Only Transformer Augmented Original Training

Original train labels are used with ECG-style waveform augmentation; validation selects epoch and bad threshold; original test is held out.

| Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| aug_convtx_badoutlier_style | original_val | 0.977528 | 0.953175 | 0.985552 | 0.895238 | 0.987952 | 14 | 11 | 1 |
| aug_convtx_badoutlier_style | original_test_all_10s+ | 0.758759 | 0.659537 | 0.958791 | 0.637822 | 0.289538 | 150 | 1597 | 186 |
| aug_convtx_badoutlier_style_badcal | original_test_all_10s+ | 0.731863 | 0.620364 | 0.958791 | 0.578174 | 0.377129 | 150 | 1590 | 154 |
| aug_convtx_badoutlier_style | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| aug_convtx_badoutlier_style_badcal | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| aug_convtx_badoutlier_style | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 186 |
| aug_convtx_badoutlier_style_badcal | bad_outlier_stress | 0.123288 | 0.073171 | 0.000000 | 0.000000 | 0.123288 | 0 | 0 | 154 |

## Candidates

- `aug_convtx_badoutlier_style`: best_epoch=4, threshold=0.12, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\waveform_transformer_augmented_original\N17043_gm_probe\aug_convtx_badoutlier_style\ckpt_best.pt`

Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_transformer_augmented_original_metrics.csv`
