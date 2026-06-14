# Waveform-Only Transformer Original Adaptation

Diagnostic only: original BUT train labels are used for training, val for selection, test held out. Inputs remain waveform only.

| Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| orig_conformer_robust3_aux | original_val | 0.980121 | 0.954938 | 0.992776 | 0.866667 | 0.975904 | 7 | 14 | 2 |
| orig_conformer_robust3_aux | original_test_all_10s+ | 0.743305 | 0.652381 | 0.963736 | 0.603028 | 0.301703 | 132 | 1749 | 134 |
| orig_conformer_robust3_aux_badcal | original_test_all_10s+ | 0.731627 | 0.653103 | 0.961264 | 0.560325 | 0.542579 | 120 | 1664 | 80 |
| orig_conformer_robust3_aux | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| orig_conformer_robust3_aux_badcal | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| orig_conformer_robust3_aux | bad_outlier_stress | 0.017123 | 0.011223 | 0.000000 | 0.000000 | 0.017123 | 0 | 0 | 134 |
| orig_conformer_robust3_aux_badcal | bad_outlier_stress | 0.356164 | 0.175084 | 0.000000 | 0.000000 | 0.356164 | 0 | 0 | 80 |
| orig_multipatch_robust3_aux | original_val | 0.885048 | 0.776224 | 0.901961 | 0.904762 | 0.662651 | 95 | 10 | 28 |
| orig_multipatch_robust3_aux | original_test_all_10s+ | 0.828241 | 0.710972 | 0.989011 | 0.745142 | 0.299270 | 40 | 1118 | 197 |
| orig_multipatch_robust3_aux_badcal | original_test_all_10s+ | 0.793559 | 0.693470 | 0.984066 | 0.645956 | 0.695864 | 36 | 985 | 77 |
| orig_multipatch_robust3_aux | bad_core_nearboundary | 0.957983 | 0.326180 | 0.000000 | 0.000000 | 0.957983 | 0 | 0 | 5 |
| orig_multipatch_robust3_aux_badcal | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| orig_multipatch_robust3_aux | bad_outlier_stress | 0.030822 | 0.019934 | 0.000000 | 0.000000 | 0.030822 | 0 | 0 | 192 |
| orig_multipatch_robust3_aux_badcal | bad_outlier_stress | 0.571918 | 0.242556 | 0.000000 | 0.000000 | 0.571918 | 0 | 0 | 77 |
| orig_convtx_robust3_aux | original_val | 0.979257 | 0.959158 | 0.983488 | 0.923810 | 1.000000 | 16 | 8 | 0 |
| orig_convtx_robust3_aux | original_test_all_10s+ | 0.807715 | 0.756927 | 0.983516 | 0.679169 | 0.635036 | 57 | 1281 | 104 |
| orig_convtx_robust3_aux_badcal | original_test_all_10s+ | 0.785537 | 0.708236 | 0.970055 | 0.631044 | 0.815085 | 42 | 1142 | 55 |
| orig_convtx_robust3_aux | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| orig_convtx_robust3_aux_badcal | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| orig_convtx_robust3_aux | bad_outlier_stress | 0.486301 | 0.218126 | 0.000000 | 0.000000 | 0.486301 | 0 | 0 | 104 |
| orig_convtx_robust3_aux_badcal | bad_outlier_stress | 0.739726 | 0.283465 | 0.000000 | 0.000000 | 0.739726 | 0 | 0 | 55 |

## Candidates

- `orig_conformer_robust3_aux`: best_epoch=4, threshold=0.01, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\waveform_transformer_original_adaptation\N17043_gm_probe\orig_conformer_robust3_aux\ckpt_best.pt`
- `orig_multipatch_robust3_aux`: best_epoch=7, threshold=0.01, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\waveform_transformer_original_adaptation\N17043_gm_probe\orig_multipatch_robust3_aux\ckpt_best.pt`
- `orig_convtx_robust3_aux`: best_epoch=8, threshold=0.10, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\waveform_transformer_original_adaptation\N17043_gm_probe\orig_convtx_robust3_aux\ckpt_best.pt`

Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_transformer_original_adaptation_metrics.csv`
