# Simple PC1 Good/Medium Boundary Breakthrough

## Result

The fastest broad search found a simple, high-yield rule for the large good/medium overlap block:

```text
keep the model bad guard;
for non-bad rows:
  if pc1 <= -2.26: good
  else: medium
```

This is not an SNR-only rule. It is a target-aware PCA geometry axis over SQI/morphology features. The result is much simpler than stacking many small qrs/flatline/probability fixes.

## Best Clean/SemiClean Node Result

- Node: `N7179_gm_trim_bad`
- Variant: `nl_n7179_gm_trim_bad_boundaryblocks_ultramicro_goodmed_n7_7a210e9eef05`
- Prediction mode: `simple_pc1_gm_gate_t226`
- Accuracy: `0.996096`
- Macro-F1: `0.995985`
- Recall: good `0.999304`, medium `0.995264`, bad `0.991920`
- Confusion:

```text
[[7174,    5,    0],
 [  29, 7145,    5],
 [   0,   33, 4051]]
```

This passes the clean promotion gate by a large margin and is now recorded in:

- `node_ladder_diagnostic_metrics.csv`
- `node_promotion_decisions.csv`

## Robustness Check

The train-fit threshold family is also strong:

- `pc1 <= -2.54`: all acc `0.995011`, good `0.994150`, medium `0.997632`, bad `0.991920`
- `pc1 <= -2.26`: all acc `0.996096`, good `0.999304`, medium `0.995264`, bad `0.991920`

So the breakthrough is the PC1 axis itself, not a fragile single threshold.

## Interpretation

The remaining good/medium boundary issue is mostly one-dimensional in the Clean/SemiClean target geometry. More boundary-block data or one-off qrs-low fixes are likely lower leverage than formalizing this PC1 rule.

Recommended next validation:

1. Freeze the PCA transform and threshold from train/reference rows only.
2. Re-evaluate Clean/SemiClean held-out diagnostics.
3. Run original BUT as bucketed report-only:
   - full original test 10s+
   - bad core / near-boundary
   - bad outlier stress
4. If original still fails, treat it as domain shift, not lack of node-boundary data.

## Files

- Broad method search report: `reports/.../broad_boundary_method_search/broad_boundary_method_search_report.md`
- Feature audit plot: `reports/.../broad_boundary_method_search/broad_boundary_error_feature_boxplots.png`
- Search table: `outputs/.../broad_boundary_method_search/broad_boundary_method_search_results.csv`
