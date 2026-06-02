# E3.11f PCGrad+Cap Conflict Report

- Completed runs: 12
- Focused acc reference: `0.964578`
- Bad recall floor: `0.989101`

![pareto](E:\GPTProject2\ecg\outputs\experiment\e311_sqi_denoise_classifier_grid\reports\conflict_cap\pareto_acc_bad_denoise.png)

![gradient](E:\GPTProject2\ecg\outputs\experiment\e311_sqi_denoise_classifier_grid\reports\conflict_cap\gradient_raw_projected_applied.png)

| rank | run | strategy | acc | good | medium | bad | denoise | applied cos | applied sum ratio | out |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A_decay_0p10 | ce_primary_pcgrad_cap_decay | 0.972752 | 0.950954 | 0.974114 | 0.993188 | 2.283 | 0.092 | 0.100 | `E:\GPTProject2\ecg\outputs\experiment\e311_sqi_denoise_classifier_grid\artifact\conflict_cap_grid\cap_A_decay_0p10_A_ce_primary_pcgrad_cap_decay_cap0p1_ab490fa6` |
| 2 | A_allaux_0p2 | ce_primary_pcgrad_cap_allaux | 0.971844 | 0.957766 | 0.961853 | 0.995913 | 2.390 | 0.073 | 0.200 | `E:\GPTProject2\ecg\outputs\experiment\e311_sqi_denoise_classifier_grid\artifact\conflict_cap_grid\cap_A_allaux_0p2_A_ce_primary_pcgrad_cap_allaux_cap0p2_bb9e72e9` |
| 3 | C_delta_bad_0p10 | ce_primary_pcgrad_cap_allaux | 0.970936 | 0.950954 | 0.967302 | 0.994550 | 2.225 | 0.102 | 0.094 | `E:\GPTProject2\ecg\outputs\experiment\e311_sqi_denoise_classifier_grid\artifact\conflict_cap_grid\cap_C_delta_bad_0p10_C_ce_primary_pcgrad_cap_allaux_cap0p1_60b6ebf8` |
| 4 | B_allaux_0p05 | ce_primary_pcgrad_cap_allaux | 0.969573 | 0.949591 | 0.964578 | 0.994550 | 2.362 | 0.101 | 0.050 | `E:\GPTProject2\ecg\outputs\experiment\e311_sqi_denoise_classifier_grid\artifact\conflict_cap_grid\cap_B_allaux_0p05_B_ce_primary_pcgrad_cap_allaux_cap0p05_c89bca1d` |
| 5 | B_allaux_0p2 | ce_primary_pcgrad_cap_allaux | 0.969119 | 0.950954 | 0.961853 | 0.994550 | 2.376 | 0.023 | 0.200 | `E:\GPTProject2\ecg\outputs\experiment\e311_sqi_denoise_classifier_grid\artifact\conflict_cap_grid\cap_B_allaux_0p2_B_ce_primary_pcgrad_cap_allaux_cap0p2_6c5aee86` |
| 6 | A_denonly_0p10 | ce_primary_pcgrad_cap_denonly | 0.968211 | 0.948229 | 0.961853 | 0.994550 | 2.311 | 0.124 | 0.100 | `E:\GPTProject2\ecg\outputs\experiment\e311_sqi_denoise_classifier_grid\artifact\conflict_cap_grid\cap_A_denonly_0p10_A_ce_primary_pcgrad_cap_denonly_cap0p1_54193ba6` |
| 7 | B_denonly_0p10 | ce_primary_pcgrad_cap_denonly | 0.968211 | 0.948229 | 0.961853 | 0.994550 | 2.322 | 0.111 | 0.100 | `E:\GPTProject2\ecg\outputs\experiment\e311_sqi_denoise_classifier_grid\artifact\conflict_cap_grid\cap_B_denonly_0p10_B_ce_primary_pcgrad_cap_denonly_cap0p1_c476c6ca` |
| 8 | A_lightdistill_0p10 | ce_primary_pcgrad_cap_lightdistill | 0.967302 | 0.942779 | 0.964578 | 0.994550 | 2.367 | 0.076 | 0.100 | `E:\GPTProject2\ecg\outputs\experiment\e311_sqi_denoise_classifier_grid\artifact\conflict_cap_grid\cap_A_lightdistill_0p10_A_ce_primary_pcgrad_cap_lightdistill_cap0p1_c9851e8d` |
| 9 | A_allaux_0p05 | ce_primary_pcgrad_cap_allaux | 0.967302 | 0.942779 | 0.964578 | 0.994550 | 2.267 | 0.106 | 0.050 | `E:\GPTProject2\ecg\outputs\experiment\e311_sqi_denoise_classifier_grid\artifact\conflict_cap_grid\cap_A_allaux_0p05_A_ce_primary_pcgrad_cap_allaux_cap0p05_b3d5acf7` |
| 10 | B_allaux_0p1 | ce_primary_pcgrad_cap_allaux | 0.967757 | 0.946866 | 0.961853 | 0.994550 | 2.327 | 0.069 | 0.100 | `E:\GPTProject2\ecg\outputs\experiment\e311_sqi_denoise_classifier_grid\artifact\conflict_cap_grid\cap_B_allaux_0p1_B_ce_primary_pcgrad_cap_allaux_cap0p1_9dd56050` |
| 11 | A_allaux_0p1 | ce_primary_pcgrad_cap_allaux | 0.970027 | 0.968665 | 0.946866 | 0.994550 | 2.313 | 0.128 | 0.100 | `E:\GPTProject2\ecg\outputs\experiment\e311_sqi_denoise_classifier_grid\artifact\conflict_cap_grid\cap_A_allaux_0p1_A_ce_primary_pcgrad_cap_allaux_cap0p1_20c2b4f5` |
| 12 | B_lightdistill_0p10 | ce_primary_pcgrad_cap_lightdistill | 0.969119 | 0.970027 | 0.942779 | 0.994550 | 2.374 | 0.102 | 0.100 | `E:\GPTProject2\ecg\outputs\experiment\e311_sqi_denoise_classifier_grid\artifact\conflict_cap_grid\cap_B_lightdistill_0p10_B_ce_primary_pcgrad_cap_lightdistill_cap0p1_f211b00a` |
