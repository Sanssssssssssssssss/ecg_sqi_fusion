# E3.11f MTL Conflict Report

- Completed runs: 15
- Reference focused best acc: `0.964578`
- Bad recall target: `0.990463`

![pareto](E:\GPTProject2\ecg\outputs\experiment\e311_sqi_denoise_classifier_grid\reports\conflict_mtl\pareto_acc_bad_denoise.png)

![strategy](E:\GPTProject2\ecg\outputs\experiment\e311_sqi_denoise_classifier_grid\reports\conflict_mtl\strategy_best_bars.png)

![epoch](E:\GPTProject2\ecg\outputs\experiment\e311_sqi_denoise_classifier_grid\reports\conflict_mtl\epoch_conflict_A_ce_primary_pcgrad_gated_sc0p035_e8_fa64d045.png)

![epoch](E:\GPTProject2\ecg\outputs\experiment\e311_sqi_denoise_classifier_grid\reports\conflict_mtl\epoch_conflict_B_ce_primary_pcgrad_gated_sc0p04_e8_a6d5e42a.png)

![epoch](E:\GPTProject2\ecg\outputs\experiment\e311_sqi_denoise_classifier_grid\reports\conflict_mtl\epoch_conflict_B_bounded_uncertainty_gated_sc0p04_e8_c5d915d4.png)

![epoch](E:\GPTProject2\ecg\outputs\experiment\e311_sqi_denoise_classifier_grid\reports\conflict_mtl\epoch_conflict_B_fixed_gated_sc0p04_e8_adb7ae00.png)

| rank | strategy | anchor | acc | good | medium | bad | denoise | CE-vs-den cos | out |
|---:|---|---|---:|---:|---:|---:|---:|---:|---|
| 1 | ce_primary_pcgrad | A | 0.965486 | 0.953678 | 0.953678 | 0.989101 | 2.896 | -0.227 | `E:\GPTProject2\ecg\outputs\experiment\e311_sqi_denoise_classifier_grid\artifact\conflict_mtl_grid\conflict_A_ce_primary_pcgrad_gated_sc0p035_e8_fa64d045` |
| 2 | ce_primary_pcgrad | B | 0.964578 | 0.950954 | 0.955041 | 0.987738 | 2.913 | -0.261 | `E:\GPTProject2\ecg\outputs\experiment\e311_sqi_denoise_classifier_grid\artifact\conflict_mtl_grid\conflict_B_ce_primary_pcgrad_gated_sc0p04_e8_a6d5e42a` |
| 3 | bounded_uncertainty | B | 0.965032 | 0.956403 | 0.953678 | 0.985014 | 2.880 | -0.190 | `E:\GPTProject2\ecg\outputs\experiment\e311_sqi_denoise_classifier_grid\artifact\conflict_mtl_grid\conflict_B_bounded_uncertainty_gated_sc0p04_e8_c5d915d4` |
| 4 | fixed | B | 0.964578 | 0.955041 | 0.952316 | 0.986376 | 2.893 | -0.236 | `E:\GPTProject2\ecg\outputs\experiment\e311_sqi_denoise_classifier_grid\artifact\conflict_mtl_grid\conflict_B_fixed_gated_sc0p04_e8_adb7ae00` |
| 5 | bounded_gradnorm | B | 0.963215 | 0.949591 | 0.952316 | 0.987738 | 2.919 | -0.310 | `E:\GPTProject2\ecg\outputs\experiment\e311_sqi_denoise_classifier_grid\artifact\conflict_mtl_grid\conflict_B_bounded_gradnorm_gated_sc0p04_e8_b0fa897b` |
| 6 | bounded_gradnorm | A | 0.963215 | 0.950954 | 0.952316 | 0.986376 | 2.909 | -0.306 | `E:\GPTProject2\ecg\outputs\experiment\e311_sqi_denoise_classifier_grid\artifact\conflict_mtl_grid\conflict_A_bounded_gradnorm_gated_sc0p035_e8_12140e75` |
| 7 | ce_primary_pcgrad | C | 0.960945 | 0.937330 | 0.955041 | 0.990463 | 2.943 | -0.391 | `E:\GPTProject2\ecg\outputs\experiment\e311_sqi_denoise_classifier_grid\artifact\conflict_mtl_grid\conflict_C_ce_primary_pcgrad_delta_sc0p0175_e8_94091fdb` |
| 8 | alternating | A | 0.963669 | 0.956403 | 0.948229 | 0.986376 | 2.867 | -0.236 | `E:\GPTProject2\ecg\outputs\experiment\e311_sqi_denoise_classifier_grid\artifact\conflict_mtl_grid\conflict_A_alternating_gated_sc0p035_e8_de0808a0` |
| 9 | bounded_uncertainty | A | 0.961399 | 0.945504 | 0.952316 | 0.986376 | 2.930 | -0.253 | `E:\GPTProject2\ecg\outputs\experiment\e311_sqi_denoise_classifier_grid\artifact\conflict_mtl_grid\conflict_A_bounded_uncertainty_gated_sc0p035_e8_98888c02` |
| 10 | ce_distill_guard | C | 0.962307 | 0.953678 | 0.946866 | 0.986376 | 2.835 | -0.272 | `E:\GPTProject2\ecg\outputs\experiment\e311_sqi_denoise_classifier_grid\artifact\conflict_mtl_grid\conflict_C_ce_distill_guard_delta_sc0p0175_e8_8782b170` |
| 11 | fixed | A | 0.961853 | 0.953678 | 0.948229 | 0.983651 | 2.905 | -0.224 | `E:\GPTProject2\ecg\outputs\experiment\e311_sqi_denoise_classifier_grid\artifact\conflict_mtl_grid\conflict_A_fixed_gated_sc0p035_e8_6e264fb9` |
| 12 | fixed | C | 0.960945 | 0.946866 | 0.949591 | 0.986376 | 2.853 | -0.392 | `E:\GPTProject2\ecg\outputs\experiment\e311_sqi_denoise_classifier_grid\artifact\conflict_mtl_grid\conflict_C_fixed_delta_sc0p0175_e8_c6d29044` |
