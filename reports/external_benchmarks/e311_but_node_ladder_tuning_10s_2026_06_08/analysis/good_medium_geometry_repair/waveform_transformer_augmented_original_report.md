# Waveform-Only Transformer Augmented Original Training

Original train labels are used with ECG-style waveform augmentation; validation selects epoch and bad threshold; original test is held out.

| Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| aug_convtx_balanced_focal_trainval | original_val_in_train | 0.982714 | 0.962248 | 0.987616 | 0.942857 | 0.975904 | 12 | 6 | 2 |
| aug_convtx_balanced_focal_trainval | original_test_all_10s+ | 0.814793 | 0.720276 | 0.963462 | 0.734975 | 0.357664 | 133 | 1155 | 224 |
| aug_convtx_balanced_focal_trainval_badcal | original_test_all_10s+ | 0.808659 | 0.748161 | 0.963187 | 0.688884 | 0.729927 | 128 | 1100 | 91 |
| aug_convtx_balanced_focal_trainval | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| aug_convtx_balanced_focal_trainval_badcal | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| aug_convtx_balanced_focal_trainval | bad_outlier_stress | 0.095890 | 0.058333 | 0.000000 | 0.000000 | 0.095890 | 0 | 0 | 224 |
| aug_convtx_balanced_focal_trainval_badcal | bad_outlier_stress | 0.619863 | 0.255109 | 0.000000 | 0.000000 | 0.619863 | 0 | 0 | 91 |
| aug_convtx_medium_guard | original_val | 0.967156 | 0.939251 | 0.968008 | 0.933333 | 1.000000 | 31 | 7 | 0 |
| aug_convtx_medium_guard | original_test_all_10s+ | 0.819394 | 0.698397 | 0.939835 | 0.768188 | 0.304136 | 218 | 979 | 256 |
| aug_convtx_medium_guard_badcal | original_test_all_10s+ | 0.808187 | 0.690930 | 0.939835 | 0.739268 | 0.384428 | 214 | 976 | 224 |
| aug_convtx_medium_guard | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| aug_convtx_medium_guard_badcal | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| aug_convtx_medium_guard | bad_outlier_stress | 0.020548 | 0.013423 | 0.000000 | 0.000000 | 0.020548 | 0 | 0 | 256 |
| aug_convtx_medium_guard_badcal | bad_outlier_stress | 0.133562 | 0.078550 | 0.000000 | 0.000000 | 0.133562 | 0 | 0 | 224 |

## Candidates

- `aug_convtx_balanced_focal_trainval`: best_epoch=6, threshold=0.05, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\waveform_transformer_augmented_original\N17043_gm_probe\aug_convtx_balanced_focal_trainval\ckpt_best.pt`
- `aug_convtx_medium_guard`: best_epoch=4, threshold=0.20, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\waveform_transformer_augmented_original\N17043_gm_probe\aug_convtx_medium_guard\ckpt_best.pt`

Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_transformer_augmented_original_metrics.csv`
