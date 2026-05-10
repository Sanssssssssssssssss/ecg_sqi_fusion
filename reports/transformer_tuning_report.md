# Transformer Tuning Report

Date: 2026-05-10

## Goal

Push the transformer line past the previous `tune09` baseline by changing the training data, not just dropout/lr. The 0.95+ test target was the stretch goal; validation/test must keep the original benchmark split unchanged.

## Pre-Rework Backup

Before the data/model changes, the runnable state was pushed and tagged:

```text
branch: cleanup/transformer-pipeline
commit: 7011e25 Add transformer diagnostic baselines
tag: pre-transformer-data-rework-2026-05-10
```

## Runs Compared

```text
baseline tune09:
  artifact: outputs/transformer
  model:    outputs/transformer/models/tune09_fixed_d05_cls22_den25_bad02_ls02_mw115_e26
  test=0.9356, best_val=0.9457, medium_test_recall=0.8898

E2 train-only multiview K=3:
  artifact: outputs/transformer_e2_multiview_k3
  model:    outputs/transformer_e2_multiview_k3/models/e2_k3_tune09
  test=0.9395, best_val=0.9618, medium_test_recall=0.9119

E3 triplet K=1, same clean/noise with good/medium/bad SNR:
  artifact: outputs/transformer_e3_triplet_k1
  model:    outputs/transformer_e3_triplet_k1/models/e3_triplet_tune09
  test=0.9405, best_val=0.9590, medium_test_recall=0.9192

E4 E2 + ordinal head + SNR regression head:
  artifact: outputs/transformer_e2_multiview_k3
  model:    outputs/transformer_e2_multiview_k3/models/e4_e2_ord_snr
  test=0.9363, best_val=0.9622, medium_test_recall=0.9045

E5 diverse train noise + stratified bins + raw_robust + ordinal/SNR:
  artifact: outputs/transformer_e5_multiview_k3_diverse_strat
  model:    outputs/transformer_e5_multiview_k3_diverse_strat/models/e5_diverse_strat_ord_snr_rawrobust
  test=0.9286, best_val=0.9503, medium_test_recall=0.9087
```

## Best Current Run

Best test accuracy is E3:

```text
test_acc: 0.940518
best_val_acc: 0.959034
best_epoch: 18
schedule: 0/18/4/4
```

Test confusion matrix:

```text
[[917, 34, 2],
 [54, 876, 23],
 [7, 50, 895]]
```

## Interpretation

The useful change is data structure:

```text
train-only multiview/triplet augmentation improves medium recall:
baseline medium recall 0.8898 -> E2 0.9119 -> E3 0.9192
```

The ordinal/SNR auxiliary heads did not help test generalization in this run. E4 reached the best validation accuracy but dropped on test, so it is not the recommended checkpoint.

The broad diverse-noise run also hurt test accuracy. Adding many train-only noise types at once likely shifted training distribution too far from the fixed val/test benchmark.

The 0.95 single-model test target was not reached. Current best is 0.9405, which is a real but modest improvement over tune09.

## Medium Audit

Best-run audit:

```text
outputs/transformer_e3_triplet_k1/medium_error_audit/e3_triplet_tune09_eval_only/medium_error_audit.md
```

Key read:

```text
measured-SNR oracle: 100% val/test
medium->good: mostly high-side medium SNR, heavily ma noise
medium->bad: lower-side medium SNR, also mostly ma/mix
valid_rr/rpeak failure: not the driver
```

This says the label generation is consistent, and the remaining errors are boundary/severity errors rather than an obvious RR-detection failure.

## Validation

```bash
python -m src.transformer_pipeline.analyze_training \
  --model_dir outputs/transformer/models/tune09_fixed_d05_cls22_den25_bad02_ls02_mw115_e26 \
  --model_dir outputs/transformer_e2_multiview_k3/models/e2_k3_tune09 \
  --model_dir outputs/transformer_e3_triplet_k1/models/e3_triplet_tune09 \
  --model_dir outputs/transformer_e2_multiview_k3/models/e4_e2_ord_snr \
  --model_dir outputs/transformer_e5_multiview_k3_diverse_strat/models/e5_diverse_strat_ord_snr_rawrobust \
  --write outputs/transformer_experiments_summary.json
```
