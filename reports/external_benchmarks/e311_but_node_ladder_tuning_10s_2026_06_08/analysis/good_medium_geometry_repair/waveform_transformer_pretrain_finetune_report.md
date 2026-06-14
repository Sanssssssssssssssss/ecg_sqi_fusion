# Waveform-Only Transformer Pretrain + Finetune

Diagnostic only: initialized from synthetic boundary-block Transformer checkpoints, then fine-tuned on original BUT train labels. Validation selects epoch and bad threshold; original test is held out.

| Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| synthft_conformer_head_then_all | original_val | 0.971478 | 0.931712 | 0.989680 | 0.828571 | 0.939759 | 10 | 18 | 5 |
| synthft_conformer_head_then_all | original_test_all_10s+ | 0.746491 | 0.664209 | 0.956593 | 0.612291 | 0.330900 | 154 | 1707 | 96 |
| synthft_conformer_head_then_all_badcal | original_test_all_10s+ | 0.745547 | 0.662184 | 0.955220 | 0.611161 | 0.335766 | 154 | 1705 | 96 |
| synthft_conformer_head_then_all | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| synthft_conformer_head_then_all_badcal | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| synthft_conformer_head_then_all | bad_outlier_stress | 0.058219 | 0.036677 | 0.000000 | 0.000000 | 0.058219 | 0 | 0 | 96 |
| synthft_conformer_head_then_all_badcal | bad_outlier_stress | 0.065068 | 0.040729 | 0.000000 | 0.000000 | 0.065068 | 0 | 0 | 96 |
| synthft_conformer_badguard | original_val | 0.929127 | 0.868102 | 0.940144 | 0.828571 | 0.927711 | 58 | 18 | 5 |
| synthft_conformer_badguard | original_test_all_10s+ | 0.740238 | 0.646090 | 0.943681 | 0.614550 | 0.291971 | 205 | 1693 | 87 |
| synthft_conformer_badguard_badcal | original_test_all_10s+ | 0.728324 | 0.620578 | 0.943407 | 0.590601 | 0.306569 | 205 | 1691 | 84 |
| synthft_conformer_badguard | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| synthft_conformer_badguard_badcal | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| synthft_conformer_badguard | bad_outlier_stress | 0.003425 | 0.002275 | 0.000000 | 0.000000 | 0.003425 | 0 | 0 | 87 |
| synthft_conformer_badguard_badcal | bad_outlier_stress | 0.023973 | 0.015608 | 0.000000 | 0.000000 | 0.023973 | 0 | 0 | 84 |
| synthft_convtx_balanced | original_val | 0.960242 | 0.920029 | 0.965944 | 0.914286 | 0.951807 | 33 | 9 | 4 |
| synthft_convtx_balanced | original_test_all_10s+ | 0.779403 | 0.665003 | 0.934615 | 0.697244 | 0.289538 | 238 | 1291 | 197 |
| synthft_convtx_balanced_badcal | original_test_all_10s+ | 0.761944 | 0.631727 | 0.934615 | 0.662901 | 0.299270 | 238 | 1291 | 193 |
| synthft_convtx_balanced | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| synthft_convtx_balanced_badcal | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| synthft_convtx_balanced | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 197 |
| synthft_convtx_balanced_badcal | bad_outlier_stress | 0.013699 | 0.009009 | 0.000000 | 0.000000 | 0.013699 | 0 | 0 | 193 |
| synthft_convtx_badguard | original_val | 0.967156 | 0.925327 | 0.982456 | 0.847619 | 0.939759 | 17 | 16 | 5 |
| synthft_convtx_badguard | original_test_all_10s+ | 0.761118 | 0.660735 | 0.939011 | 0.658608 | 0.289538 | 222 | 1501 | 158 |
| synthft_convtx_badguard_badcal | original_test_all_10s+ | 0.743305 | 0.620353 | 0.939011 | 0.624492 | 0.289538 | 222 | 1501 | 158 |
| synthft_convtx_badguard | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| synthft_convtx_badguard_badcal | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| synthft_convtx_badguard | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 158 |
| synthft_convtx_badguard_badcal | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 158 |

## Candidates

- `synthft_conformer_head_then_all`: best_epoch=6, threshold=0.35, init=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\waveform_transformer_search\N17043_gm_probe\conformer_robust3_auxheavy\ckpt_best.pt`, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\waveform_transformer_pretrain_finetune\N17043_gm_probe\synthft_conformer_head_then_all\ckpt_best.pt`
- `synthft_conformer_badguard`: best_epoch=3, threshold=0.08, init=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\waveform_transformer_search\N17043_gm_probe\conformer_robust3_auxheavy\ckpt_best.pt`, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\waveform_transformer_pretrain_finetune\N17043_gm_probe\synthft_conformer_badguard\ckpt_best.pt`
- `synthft_convtx_balanced`: best_epoch=6, threshold=0.23, init=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\waveform_transformer_search\N17043_gm_probe\convtx_robust3_auxheavy\ckpt_best.pt`, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\waveform_transformer_pretrain_finetune\N17043_gm_probe\synthft_convtx_balanced\ckpt_best.pt`
- `synthft_convtx_badguard`: best_epoch=6, threshold=0.10, init=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\waveform_transformer_search\N17043_gm_probe\convtx_robust3_auxheavy\ckpt_best.pt`, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\waveform_transformer_pretrain_finetune\N17043_gm_probe\synthft_convtx_badguard\ckpt_best.pt`

Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_transformer_pretrain_finetune_metrics.csv`
