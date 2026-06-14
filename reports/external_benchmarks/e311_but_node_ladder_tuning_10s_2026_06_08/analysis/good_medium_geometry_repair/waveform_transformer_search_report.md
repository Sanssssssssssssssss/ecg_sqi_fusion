# Waveform-Only Transformer Search

Only Transformer-family waveform models are tested here. Inputs are waveform channels only; SQI/geometry columns are auxiliary targets, never inference inputs.

| Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m | Aux MAE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| convtx_robust3_auxheavy | synthetic_test | 0.981035 | 0.980072 | 0.937238 | 0.998377 | 0.979253 | 30 | 2 | 5 | 0.6453 |
| convtx_robust3_auxheavy_badcal | synthetic_test | 0.981548 | 0.980909 | 0.937238 | 0.998377 | 0.983402 | 30 | 2 | 4 | nan |
| convtx_robust3_auxheavy | original_test_all_10s+ | 0.805002 | 0.688993 | 0.687637 | 0.949390 | 0.289538 | 1137 | 222 | 227 | nan |
| convtx_robust3_auxheavy_badcal | original_test_all_10s+ | 0.802878 | 0.683416 | 0.687637 | 0.945323 | 0.289538 | 1137 | 222 | 227 | nan |
| convtx_robust3_auxheavy | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 | nan |
| convtx_robust3_auxheavy_badcal | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 | nan |
| convtx_robust3_auxheavy | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 227 | nan |
| convtx_robust3_auxheavy_badcal | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 227 | nan |
| multipatch_robust3_auxheavy | synthetic_test | 0.996925 | 0.996055 | 0.995816 | 0.999188 | 0.987552 | 2 | 1 | 3 | 0.6514 |
| multipatch_robust3_auxheavy_badcal | synthetic_test | 0.985648 | 0.978567 | 0.995816 | 0.980519 | 0.991701 | 2 | 1 | 2 | nan |
| multipatch_robust3_auxheavy | original_test_all_10s+ | 0.796862 | 0.656097 | 0.866484 | 0.793493 | 0.216545 | 486 | 901 | 186 | nan |
| multipatch_robust3_auxheavy_badcal | original_test_all_10s+ | 0.795800 | 0.665216 | 0.866484 | 0.788070 | 0.253041 | 485 | 901 | 171 | nan |
| multipatch_robust3_auxheavy | bad_core_nearboundary | 0.747899 | 0.285256 | 0.000000 | 0.000000 | 0.747899 | 0 | 0 | 30 | nan |
| multipatch_robust3_auxheavy_badcal | bad_core_nearboundary | 0.873950 | 0.310912 | 0.000000 | 0.000000 | 0.873950 | 0 | 0 | 15 | nan |
| multipatch_robust3_auxheavy | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 156 | nan |
| multipatch_robust3_auxheavy_badcal | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 156 | nan |
| conformer_robust3_auxheavy | synthetic_test | 0.990774 | 0.989554 | 0.989540 | 0.993506 | 0.979253 | 5 | 8 | 5 | 0.6493 |
| conformer_robust3_auxheavy_badcal | synthetic_test | 0.991287 | 0.990393 | 0.989540 | 0.993506 | 0.983402 | 5 | 8 | 4 | nan |
| conformer_robust3_auxheavy | original_test_all_10s+ | 0.828241 | 0.706101 | 0.818681 | 0.886127 | 0.289538 | 660 | 494 | 132 | nan |
| conformer_robust3_auxheavy_badcal | original_test_all_10s+ | 0.827415 | 0.703929 | 0.818681 | 0.884546 | 0.289538 | 660 | 494 | 132 | nan |
| conformer_robust3_auxheavy | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 | nan |
| conformer_robust3_auxheavy_badcal | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 | nan |
| conformer_robust3_auxheavy | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 132 | nan |
| conformer_robust3_auxheavy_badcal | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 132 | nan |

## Reference

- 47-feature tabular current best original test acc: `0.963548`; recalls good/medium/bad `0.956/0.973/0.927`.

## Candidate Configs

- `convtx_robust3_auxheavy`: best_epoch=2, threshold=0.21, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\waveform_transformer_search\N17043_gm_probe\convtx_robust3_auxheavy\ckpt_best.pt`
- `multipatch_robust3_auxheavy`: best_epoch=6, threshold=0.10, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\waveform_transformer_search\N17043_gm_probe\multipatch_robust3_auxheavy\ckpt_best.pt`
- `conformer_robust3_auxheavy`: best_epoch=2, threshold=0.34, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\waveform_transformer_search\N17043_gm_probe\conformer_robust3_auxheavy\ckpt_best.pt`

Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_transformer_search_metrics.csv`
