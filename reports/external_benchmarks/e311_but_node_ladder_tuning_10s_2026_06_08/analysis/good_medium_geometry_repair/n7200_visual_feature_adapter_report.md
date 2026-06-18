# N7200 Visual Feature Adapter

This is a trained good/medium feature adapter on Clean/SemiClean/node diagnostic rows. It does not use original BUT for selection.

## Best Adapter

- model: `rf_depth4_balanced`
- threshold: `0.5300`
- all-node acc: `0.993346`
- macro-F1: `0.992137`
- good/medium/bad recall: `0.999583` / `1.000000` / `0.970617`
- confusion: `[[7197, 3, 0], [0, 7200, 0], [0, 120, 3964]]`
- decision: `adapter_passed_gates`

## All-Node Comparison

```
                 model  threshold      acc  macro_f1  good_recall  medium_recall  bad_recall  good_to_medium  medium_to_good
    rf_depth4_balanced       0.53 0.993346  0.992137     0.999583       1.000000    0.970617               3               0
 logreg_l2_balanced_c1       0.05 0.991723  0.990747     1.000000       0.995417    0.970617               0              33
  tree_depth3_balanced       0.05 0.991668  0.990703     0.999306       0.995972    0.970617               5              29
       sgd_log_loss_l2       0.05 0.990262  0.989497     1.000000       0.991667    0.970617               0              60
logreg_l2_balanced_c03       0.05 0.989937  0.989220     1.000000       0.990833    0.970617               0              66
```

## Held-Out Test Comparison

```
                 model  threshold      acc  macro_f1  good_recall  medium_recall  bad_recall  good_to_medium  medium_to_good
    rf_depth4_balanced       0.53 0.970435  0.988390          1.0       1.000000         0.0               0               0
 logreg_l2_balanced_c1       0.05 0.969441  0.987279          1.0       0.998402         0.0               0               4
logreg_l2_balanced_c03       0.05 0.968696  0.986446          1.0       0.997203         0.0               0               7
       sgd_log_loss_l2       0.05 0.968447  0.986169          1.0       0.996804         0.0               0               8
  tree_depth3_balanced       0.05 0.967702  0.985337          1.0       0.995605         0.0               0              11
```

## Residual Plot

![adapter residuals](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/n7200_visual_feature_adapter_residuals.png)

## Caveat

This is a trained feature adapter, not a normal neural checkpoint. It is useful as a target for future logit-level or architecture-level integration.
