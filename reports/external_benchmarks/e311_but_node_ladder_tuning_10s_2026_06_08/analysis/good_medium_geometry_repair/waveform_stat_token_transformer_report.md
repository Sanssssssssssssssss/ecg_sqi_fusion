# Waveform-Only StatToken Transformer

The model receives waveform channels only.  Global stat tokens are computed inside the model from waveform tensors and fused with ConvSubsample Transformer features.

| Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| stattx_balanced | original_val | 0.985307 | 0.970467 | 0.989680 | 0.933333 | 1.000000 | 10 | 7 | 0 |
| stattx_balanced | original_test_all_10s+ | 0.797806 | 0.724198 | 0.978571 | 0.683235 | 0.430657 | 77 | 1360 | 179 |
| stattx_balanced_badcal | original_test_all_10s+ | 0.773623 | 0.703970 | 0.971703 | 0.603479 | 0.851582 | 70 | 1232 | 32 |
| stattx_balanced | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| stattx_balanced_badcal | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| stattx_balanced | bad_outlier_stress | 0.198630 | 0.110476 | 0.000000 | 0.000000 | 0.198630 | 0 | 0 | 179 |
| stattx_balanced_badcal | bad_outlier_stress | 0.791096 | 0.294455 | 0.000000 | 0.000000 | 0.791096 | 0 | 0 | 32 |
| stattx_medium_guard | original_val | 0.977528 | 0.953909 | 0.983488 | 0.914286 | 0.987952 | 16 | 9 | 1 |
| stattx_medium_guard | original_test_all_10s+ | 0.813141 | 0.733720 | 0.962088 | 0.725938 | 0.433090 | 138 | 1161 | 194 |
| stattx_medium_guard_badcal | original_test_all_10s+ | 0.803468 | 0.732952 | 0.953571 | 0.690917 | 0.686131 | 123 | 1106 | 110 |
| stattx_medium_guard | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| stattx_medium_guard_badcal | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| stattx_medium_guard | bad_outlier_stress | 0.202055 | 0.112061 | 0.000000 | 0.000000 | 0.202055 | 0 | 0 | 194 |
| stattx_medium_guard_badcal | bad_outlier_stress | 0.558219 | 0.238828 | 0.000000 | 0.000000 | 0.558219 | 0 | 0 | 110 |

## Candidates

- `stattx_balanced`: best_epoch=8, threshold=0.04, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\waveform_stat_token_transformer\N17043_gm_probe\stattx_balanced\ckpt_best.pt`
- `stattx_medium_guard`: best_epoch=8, threshold=0.09, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\waveform_stat_token_transformer\N17043_gm_probe\stattx_medium_guard\ckpt_best.pt`

Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_stat_token_transformer_metrics.csv`
