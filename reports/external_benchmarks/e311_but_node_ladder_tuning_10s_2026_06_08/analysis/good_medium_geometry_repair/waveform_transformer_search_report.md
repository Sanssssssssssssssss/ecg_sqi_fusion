# Waveform-Only Transformer Search

Only Transformer-family waveform models are tested here. Inputs are waveform channels only; SQI/geometry columns are auxiliary targets, never inference inputs.

| Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m | Aux MAE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| statconformer_robust3_balanced | synthetic_test | 0.993337 | 0.991957 | 0.989540 | 0.997565 | 0.979253 | 5 | 3 | 5 | 0.6497 |
| statconformer_robust3_balanced_badcal | synthetic_test | 0.993337 | 0.991957 | 0.989540 | 0.997565 | 0.979253 | 5 | 3 | 5 | nan |
| statconformer_robust3_balanced | original_test_all_10s+ | 0.757933 | 0.517815 | 0.863736 | 0.741301 | 0.000000 | 496 | 1144 | 154 | nan |
| statconformer_robust3_balanced_badcal | original_test_all_10s+ | 0.761826 | 0.568594 | 0.863736 | 0.741075 | 0.082725 | 496 | 1132 | 121 | nan |
| statconformer_robust3_balanced | bad_core_nearboundary | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 119 | nan |
| statconformer_robust3_balanced_badcal | bad_core_nearboundary | 0.277311 | 0.144737 | 0.000000 | 0.000000 | 0.277311 | 0 | 0 | 86 | nan |
| statconformer_robust3_balanced | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 35 | nan |
| statconformer_robust3_balanced_badcal | bad_outlier_stress | 0.003425 | 0.002275 | 0.000000 | 0.000000 | 0.003425 | 0 | 0 | 35 | nan |
| statconformer_robust3_badguard | synthetic_test | 0.991799 | 0.990526 | 0.993724 | 0.993506 | 0.979253 | 3 | 8 | 5 | 0.6502 |
| statconformer_robust3_badguard_badcal | synthetic_test | 0.991799 | 0.990526 | 0.993724 | 0.993506 | 0.979253 | 3 | 8 | 5 | nan |
| statconformer_robust3_badguard | original_test_all_10s+ | 0.763596 | 0.521753 | 0.913462 | 0.711252 | 0.000000 | 315 | 1277 | 150 | nan |
| statconformer_robust3_badguard_badcal | original_test_all_10s+ | 0.776690 | 0.662082 | 0.912363 | 0.708992 | 0.304136 | 314 | 1211 | 30 | nan |
| statconformer_robust3_badguard | bad_core_nearboundary | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 119 | nan |
| statconformer_robust3_badguard_badcal | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 | nan |
| statconformer_robust3_badguard | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 31 | nan |
| statconformer_robust3_badguard_badcal | bad_outlier_stress | 0.020548 | 0.013423 | 0.000000 | 0.000000 | 0.020548 | 0 | 0 | 30 | nan |
| statconformer_robust3_coreheavy | synthetic_test | 0.994362 | 0.992929 | 0.993724 | 0.997565 | 0.979253 | 3 | 3 | 5 | 0.6493 |
| statconformer_robust3_coreheavy_badcal | synthetic_test | 0.994362 | 0.992929 | 0.993724 | 0.997565 | 0.979253 | 3 | 3 | 5 | nan |
| statconformer_robust3_coreheavy | original_test_all_10s+ | 0.749322 | 0.512024 | 0.879396 | 0.711930 | 0.000000 | 439 | 1274 | 147 | nan |
| statconformer_robust3_coreheavy_badcal | original_test_all_10s+ | 0.762770 | 0.659166 | 0.879396 | 0.711704 | 0.279805 | 439 | 1267 | 32 | nan |
| statconformer_robust3_coreheavy | bad_core_nearboundary | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 119 | nan |
| statconformer_robust3_coreheavy_badcal | bad_core_nearboundary | 0.966387 | 0.327635 | 0.000000 | 0.000000 | 0.966387 | 0 | 0 | 4 | nan |
| statconformer_robust3_coreheavy | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 28 | nan |
| statconformer_robust3_coreheavy_badcal | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 28 | nan |

## Reference

- 47-feature tabular current best original test acc: `0.963548`; recalls good/medium/bad `0.956/0.973/0.927`.

## Candidate Configs

- `statconformer_robust3_balanced`: best_epoch=5, threshold=0.01, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\waveform_transformer_search\N17043_gm_probe\statconformer_robust3_balanced\ckpt_best.pt`
- `statconformer_robust3_badguard`: best_epoch=10, threshold=0.01, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\waveform_transformer_search\N17043_gm_probe\statconformer_robust3_badguard\ckpt_best.pt`
- `statconformer_robust3_coreheavy`: best_epoch=8, threshold=0.01, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\waveform_transformer_search\N17043_gm_probe\statconformer_robust3_coreheavy\ckpt_best.pt`

Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_transformer_search_metrics.csv`
