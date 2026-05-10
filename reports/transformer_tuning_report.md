# Transformer Tuning Report

Date: 2026-05-10

## Goal

Tune the transformer pipeline with a small GPU budget, keep the paper-related wavelet denoise logic intact, and require a nonzero final training phase. Target test accuracy was 0.95+ if reachable.

## Best Run

Best single-model result:

```text
model: outputs/transformer/models/tune09_fixed_d05_cls22_den25_bad02_ls02_mw115_e26
test_acc: 0.935619
best_val_acc: 0.945728
best_epoch: 18
schedule: 0/18/4/4
```

Recommended params:

```text
dropout=0.05
lr=6e-5
weight_decay=0.03
label_smoothing=0.02
class_weight_medium=1.15
lambda_cls=22
lambda_den=25
lambda_lvl=1
bad_den_w_max=0.02
bad_den_w_warmup_epochs=10
uncertainty_mode=fixed
cls_pool=decoder
```

Test confusion matrix:

```text
[[916, 33, 4],
 [61, 848, 44],
 [13, 29, 910]]
```

## Runs Compared

```text
tune09 fixed weights, decoder pool: test 0.9356, best_val 0.9457
tune10 stronger smoothing/medium weight: test 0.9349, best_val 0.9478
tune11 longer denoise/larger den weight: test 0.9248, best_val 0.9447
tune12 encoder classification pool: test 0.9185, best_val 0.9422
tune13 encoder+decoder classification pool: test 0.9279, best_val 0.9464
tune14 both pool + cls warmup: test 0.9286, best_val 0.9457
```

## Interpretation

The best legal schedule is the fixed-weight final phase setup in `tune09`. Kendall uncertainty weighting hurt validation stability in earlier runs, so the final phase remains nonzero but uses explicit loss weights.

Changing the classifier pooling from the original decoder feature to encoder-only or encoder+decoder did not improve test accuracy. It reduced generalization, mostly by increasing medium-class mistakes. The recommended model structure therefore remains the original decoder pooling.

Current evidence suggests this dataset/model setup plateaus around 0.93-0.936 single-model test accuracy. I do not see a reliable path to 0.95 from small hyperparameter changes alone.

## Debug Artifacts

Useful local files:

```text
outputs/transformer/models/tune09_fixed_d05_cls22_den25_bad02_ls02_mw115_e26/probe_summary.json
outputs/transformer/models/tune09_fixed_d05_cls22_den25_bad02_ls02_mw115_e26/test_report.json
outputs/transformer/models/tune09_fixed_d05_cls22_den25_bad02_ls02_mw115_e26/debug/training_curves.png
outputs/transformer/models/tune09_fixed_d05_cls22_den25_bad02_ls02_mw115_e26/debug/denoise_examples_test.png
```

Validation command used:

```bash
python -m compileall src/transformer_pipeline
python -m src.transformer_pipeline.analyze_training --model_dir outputs/transformer/models/tune09_fixed_d05_cls22_den25_bad02_ls02_mw115_e26
```
