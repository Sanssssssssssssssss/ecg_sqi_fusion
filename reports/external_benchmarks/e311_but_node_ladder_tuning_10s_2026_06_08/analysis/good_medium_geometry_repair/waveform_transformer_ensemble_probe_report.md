# Waveform-Only Transformer Ensemble Probe

Weights and bad threshold are selected on original validation only. Test is held out. Inputs are waveform-model probabilities only.

| Rank | Models | Weights | Bad thr | Test Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m | Val Acc |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | aug_convtx_balanced | 1.00 | 0.28 | 0.813141 | 0.752877 | 0.960714 | 0.720515 | 0.503650 | 142 | 1185 | 146 | 0.980985 |
| 2 | aug_convtx_balanced | 1.00 | 0.3 | 0.812906 | 0.751510 | 0.960714 | 0.720967 | 0.493917 | 142 | 1188 | 150 | 0.980985 |
| 3 | aug_convtx_balanced | 1.00 | 0.36 | 0.812788 | 0.745554 | 0.960714 | 0.724130 | 0.457421 | 142 | 1189 | 161 | 0.980985 |
| 4 | aug_convtx_balanced | 1.00 | 0.38 | 0.812788 | 0.744308 | 0.960714 | 0.724808 | 0.450122 | 142 | 1189 | 164 | 0.980985 |
| 5 | aug_convtx_balanced | 1.00 | 0.34 | 0.812552 | 0.745773 | 0.960714 | 0.723000 | 0.464720 | 142 | 1188 | 160 | 0.980985 |
| 6 | aug_convtx_balanced | 1.00 | 0.32 | 0.812434 | 0.747792 | 0.960714 | 0.721419 | 0.479319 | 142 | 1188 | 155 | 0.980985 |
| 7 | aug_convtx_balanced | 1.00 | 0.2 | 0.812316 | 0.759592 | 0.960714 | 0.711478 | 0.583942 | 141 | 1179 | 120 | 0.980985 |
| 8 | aug_convtx_balanced | 1.00 | 0.26 | 0.812316 | 0.752648 | 0.960714 | 0.717352 | 0.520681 | 142 | 1184 | 139 | 0.980985 |
| 9 | aug_convtx_balanced | 1.00 | 0.4 | 0.812316 | 0.740563 | 0.960714 | 0.725260 | 0.435523 | 142 | 1190 | 170 | 0.980985 |
| 10 | aug_convtx_balanced | 1.00 | 0.24 | 0.812198 | 0.755194 | 0.960714 | 0.714867 | 0.545012 | 141 | 1183 | 133 | 0.980985 |
| 11 | aug_convtx_balanced | 1.00 | 0.42 | 0.812080 | 0.739235 | 0.960714 | 0.725260 | 0.430657 | 143 | 1190 | 172 | 0.980985 |
| 12 | aug_convtx_balanced | 1.00 | 0.22 | 0.811962 | 0.755713 | 0.960714 | 0.713059 | 0.559611 | 141 | 1180 | 127 | 0.980985 |
| 13 | aug_convtx_balanced | 1.00 | 0.14 | 0.811608 | 0.763211 | 0.960714 | 0.705151 | 0.637470 | 140 | 1177 | 101 | 0.980985 |
| 14 | aug_convtx_balanced | 1.00 | 0.44 | 0.811608 | 0.735529 | 0.960714 | 0.725486 | 0.418491 | 143 | 1190 | 177 | 0.980985 |
| 15 | aug_convtx_balanced | 1.00 | 0.16 | 0.811490 | 0.760383 | 0.960714 | 0.706959 | 0.615572 | 140 | 1177 | 108 | 0.980985 |
| 16 | aug_convtx_balanced | 1.00 | 0.18 | 0.811372 | 0.757665 | 0.960714 | 0.708766 | 0.593674 | 141 | 1177 | 116 | 0.980985 |
| 17 | aug_convtx_balanced | 1.00 | 0.46 | 0.811136 | 0.731753 | 0.960714 | 0.725712 | 0.406326 | 143 | 1190 | 182 | 0.980985 |
| 18 | aug_convtx_balanced | 1.00 | 0.12 | 0.810664 | 0.761591 | 0.960714 | 0.701988 | 0.652068 | 140 | 1174 | 96 | 0.980985 |
| 19 | aug_convtx_balanced | 1.00 | 0.1 | 0.808659 | 0.757531 | 0.960714 | 0.696792 | 0.666667 | 140 | 1172 | 92 | 0.980985 |
| 20 | aug_convtx_balanced | 1.00 | 0.08 | 0.808069 | 0.757880 | 0.960440 | 0.693177 | 0.695864 | 136 | 1170 | 84 | 0.980985 |

## Models

- `aug_convtx_balanced`: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\waveform_transformer_augmented_original\N17043_gm_probe\aug_convtx_balanced_focal\ckpt_best.pt`
- `orig_multipatch`: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\waveform_transformer_original_adaptation\N17043_gm_probe\orig_multipatch_robust3_aux\ckpt_best.pt`
- `orig_convtx`: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\waveform_transformer_original_adaptation\N17043_gm_probe\orig_convtx_robust3_aux\ckpt_best.pt`

Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_transformer_ensemble_probe_metrics.csv`
