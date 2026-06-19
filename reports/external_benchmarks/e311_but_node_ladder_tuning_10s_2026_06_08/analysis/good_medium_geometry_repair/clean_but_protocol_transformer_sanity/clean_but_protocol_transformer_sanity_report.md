# Clean BUT Protocol Transformer Sanity

Fixed-length 10s only. No variable-length model is used. Inputs remain waveform-only; SQI/geometry columns are auxiliary targets.

Important caveat: after dropping `outlier_low_confidence`, the legacy validation split can contain very few bad rows, so bad-threshold calibration is diagnostic only.

| Policy | Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| margin_ge_2s_drop_outlier | margin_ge_2s_drop_outlier_orig_convtx_robust3_aux | original_val | 0.833333 | 0.781023 | 0.821317 | 1.000000 | 1.000000 |
| margin_ge_2s_drop_outlier | margin_ge_2s_drop_outlier_orig_convtx_robust3_aux | original_test_all_10s+ | 0.938056 | 0.637806 | 0.990394 | 0.964795 | 0.000000 |
| margin_ge_2s_drop_outlier | margin_ge_2s_drop_outlier_orig_convtx_robust3_aux_badcal | original_test_all_10s+ | 0.938056 | 0.637806 | 0.990394 | 0.964795 | 0.000000 |
| margin_ge_2s_drop_outlier | margin_ge_2s_drop_outlier_orig_convtx_robust3_aux | bad_core_nearboundary | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| margin_ge_2s_drop_outlier | margin_ge_2s_drop_outlier_orig_convtx_robust3_aux_badcal | bad_core_nearboundary | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| margin_ge_2s_drop_outlier | margin_ge_2s_drop_outlier_orig_convtx_robust3_aux | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| margin_ge_2s_drop_outlier | margin_ge_2s_drop_outlier_orig_convtx_robust3_aux_badcal | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| margin_ge_5s_drop_outlier | margin_ge_5s_drop_outlier_orig_convtx_robust3_aux | original_val | 0.840602 | 0.784869 | 0.829308 | 1.000000 | 1.000000 |
| margin_ge_5s_drop_outlier | margin_ge_5s_drop_outlier_orig_convtx_robust3_aux | original_test_all_10s+ | 0.928415 | 0.946367 | 0.997012 | 0.891189 | 1.000000 |
| margin_ge_5s_drop_outlier | margin_ge_5s_drop_outlier_orig_convtx_robust3_aux_badcal | original_test_all_10s+ | 0.928415 | 0.946367 | 0.997012 | 0.891189 | 1.000000 |
| margin_ge_5s_drop_outlier | margin_ge_5s_drop_outlier_orig_convtx_robust3_aux | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 |
| margin_ge_5s_drop_outlier | margin_ge_5s_drop_outlier_orig_convtx_robust3_aux_badcal | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 |
| margin_ge_5s_drop_outlier | margin_ge_5s_drop_outlier_orig_convtx_robust3_aux | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| margin_ge_5s_drop_outlier | margin_ge_5s_drop_outlier_orig_convtx_robust3_aux_badcal | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

## Checkpoints

- `margin_ge_2s_drop_outlier` / `margin_ge_2s_drop_outlier_orig_convtx_robust3_aux`: best_epoch=3, threshold=0.01, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\waveform_transformer_original_adaptation\N17043_clean_margin_ge_2s_drop_outlier\margin_ge_2s_drop_outlier_orig_convtx_robust3_aux\ckpt_best.pt`
- `margin_ge_5s_drop_outlier` / `margin_ge_5s_drop_outlier_orig_convtx_robust3_aux`: best_epoch=3, threshold=0.01, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\waveform_transformer_original_adaptation\N17043_clean_margin_ge_5s_drop_outlier\margin_ge_5s_drop_outlier_orig_convtx_robust3_aux\ckpt_best.pt`

## Interpretation Contract

- These runs test whether cleaning the fixed 10s protocol makes the waveform Transformer problem easier.
- They do not replace full BUT stress evaluation.
- Full/outlier stress must stay report-only and separate.
- If clean policies improve sharply, the next model work should use clean-body training plus explicit stress-bucket reporting.