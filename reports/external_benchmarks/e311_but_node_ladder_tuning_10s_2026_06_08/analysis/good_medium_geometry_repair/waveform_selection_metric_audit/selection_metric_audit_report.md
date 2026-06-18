# Waveform Selection Metric Audit

Candidates audited: 32

## Top positive correlations by original target

### original_acc

| metric | spearman | n |
|---|---:|---:|
| val_bad_precision_at_best_valacc | 0.577 | 32 |
| val_bad_precision_max | 0.559 | 32 |
| stress_val_bad_to_medium_max | 0.493 | 28 |
| stress_val_good_recall_max | 0.467 | 28 |
| stress_val_bad_precision_max | 0.447 | 28 |
| stress_val_macro_f1_last | 0.427 | 28 |
| val_bad_precision_last | 0.423 | 32 |
| stress_val_acc_last | 0.407 | 28 |
| stress_val_medium_precision_max | 0.379 | 28 |
| stress_val_bad_precision_last | 0.357 | 28 |
| stress_val_medium_to_good_max | 0.346 | 28 |
| val_macro_f1_last | 0.345 | 32 |

Negative correlations:

| metric | spearman | n |
|---|---:|---:|
| val_teacher_core_mae_at_best_valacc | -0.348 | 32 |
| stress_val_teacher_core_mae_at_best_valacc | -0.333 | 28 |
| stress_val_good_to_medium_last | -0.279 | 28 |
| val_teacher_core_mae_last | -0.278 | 32 |
| val_teacher_mae_at_best_valacc | -0.250 | 32 |
| stress_val_teacher_mae_at_best_valacc | -0.229 | 28 |
| val_teacher_mae_last | -0.205 | 32 |
| stress_val_teacher_core_mae_last | -0.196 | 28 |

### original_good

| metric | spearman | n |
|---|---:|---:|
| val_good_recall_at_best_valacc | 0.560 | 32 |
| val_good_recall_last | 0.490 | 32 |
| stress_val_acc_last | 0.454 | 28 |
| val_medium_precision_at_best_valacc | 0.448 | 32 |
| stress_val_bad_precision_max | 0.447 | 28 |
| stress_val_macro_f1_last | 0.427 | 28 |
| val_medium_precision_last | 0.426 | 32 |
| val_medium_precision_max | 0.411 | 32 |
| val_macro_f1_at_best_valacc | 0.342 | 32 |
| val_macro_f1_max | 0.334 | 32 |
| val_acc_max | 0.309 | 32 |
| val_acc_at_best_valacc | 0.309 | 32 |

Negative correlations:

| metric | spearman | n |
|---|---:|---:|
| val_good_to_medium_at_best_valacc | -0.560 | 32 |
| val_good_to_medium_last | -0.490 | 32 |
| stress_val_good_to_medium_last | -0.293 | 28 |
| stress_val_good_to_medium_at_best_valacc | -0.269 | 28 |
| stress_val_bad_to_medium_last | -0.250 | 28 |
| stress_val_teacher_mae_last | -0.190 | 28 |
| stress_val_teacher_mae_at_best_valacc | -0.184 | 28 |
| val_teacher_mae_last | -0.181 | 32 |

### original_medium

| metric | spearman | n |
|---|---:|---:|
| val_bad_precision_at_best_valacc | 0.548 | 32 |
| stress_val_bad_to_medium_max | 0.527 | 28 |
| val_bad_precision_max | 0.494 | 32 |
| stress_val_good_recall_max | 0.483 | 28 |
| stress_val_bad_precision_max | 0.447 | 28 |
| val_bad_precision_last | 0.412 | 32 |
| stress_val_bad_precision_last | 0.394 | 28 |
| stress_val_medium_precision_max | 0.389 | 28 |
| stress_val_macro_f1_last | 0.311 | 28 |
| stress_val_medium_recall_last | 0.308 | 28 |
| stress_val_medium_to_good_max | 0.305 | 28 |
| stress_val_acc_last | 0.292 | 28 |

Negative correlations:

| metric | spearman | n |
|---|---:|---:|
| stress_val_teacher_core_mae_at_best_valacc | -0.364 | 28 |
| val_teacher_core_mae_at_best_valacc | -0.338 | 32 |
| val_teacher_core_mae_last | -0.296 | 32 |
| stress_val_teacher_core_mae_last | -0.287 | 28 |
| stress_val_teacher_mae_at_best_valacc | -0.279 | 28 |
| val_teacher_mae_at_best_valacc | -0.273 | 32 |
| stress_val_bad_recall_last | -0.250 | 28 |
| stress_val_good_to_medium_last | -0.220 | 28 |

### original_bad

| metric | spearman | n |
|---|---:|---:|
| val_bad_recall_at_best_valacc | 0.470 | 32 |
| val_bad_recall_last | 0.400 | 32 |
| val_good_to_medium_at_best_valacc | 0.337 | 32 |
| val_medium_to_good_last | 0.293 | 32 |
| stress_val_medium_to_good_at_best_valacc | 0.262 | 28 |
| val_bad_recall_max | 0.259 | 32 |
| stress_val_bad_to_medium_last | 0.250 | 28 |
| val_medium_precision_max | 0.217 | 32 |
| val_medium_to_good_at_best_valacc | 0.152 | 32 |
| stress_val_medium_to_good_last | 0.149 | 28 |
| stress_val_bad_to_medium_max | 0.143 | 28 |
| val_medium_precision_last | 0.124 | 32 |

Negative correlations:

| metric | spearman | n |
|---|---:|---:|
| stress_val_bad_precision_at_best_valacc | -0.552 | 28 |
| val_bad_to_medium_at_best_valacc | -0.470 | 32 |
| stress_val_bad_precision_max | -0.447 | 28 |
| stress_val_medium_recall_at_best_valacc | -0.432 | 28 |
| val_bad_to_medium_max | -0.428 | 32 |
| val_medium_recall_last | -0.426 | 32 |
| stress_val_bad_precision_last | -0.422 | 28 |
| val_bad_to_medium_last | -0.400 | 32 |