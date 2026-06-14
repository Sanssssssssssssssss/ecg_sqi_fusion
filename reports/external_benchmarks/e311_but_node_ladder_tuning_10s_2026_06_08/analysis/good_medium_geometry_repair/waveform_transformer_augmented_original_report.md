# Waveform-Only Transformer Augmented Original Training

Original train labels are used with ECG-style waveform augmentation; validation selects epoch and bad threshold; original test is held out.

| Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| aug_convtx_badoutlier_style | original_val | 0.974935 | 0.947134 | 0.982456 | 0.904762 | 0.975904 | 17 | 10 | 2 |
| aug_convtx_badoutlier_style | original_test_all_10s+ | 0.756164 | 0.658473 | 0.968681 | 0.624718 | 0.289538 | 114 | 1659 | 165 |
| aug_convtx_badoutlier_style_badcal | original_test_all_10s+ | 0.739295 | 0.650442 | 0.968681 | 0.577271 | 0.452555 | 114 | 1645 | 106 |
| aug_convtx_badoutlier_style | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| aug_convtx_badoutlier_style_badcal | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| aug_convtx_badoutlier_style | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 165 |
| aug_convtx_badoutlier_style_badcal | bad_outlier_stress | 0.229452 | 0.124420 | 0.000000 | 0.000000 | 0.229452 | 0 | 0 | 106 |

## Candidates

- `aug_convtx_badoutlier_style`: best_epoch=4, threshold=0.11, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\waveform_transformer_augmented_original\N17043_gm_probe\aug_convtx_badoutlier_style\ckpt_best.pt`

Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_transformer_augmented_original_metrics.csv`
