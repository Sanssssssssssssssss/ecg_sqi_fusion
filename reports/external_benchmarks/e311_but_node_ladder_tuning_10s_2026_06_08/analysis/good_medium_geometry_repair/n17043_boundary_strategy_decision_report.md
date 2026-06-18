# N17043 Boundary Strategy Decision Report

This report separates three questions that were getting mixed together:

1. Can the main good/medium/bad geometry be represented by simple features?
2. Can the ordinary UFormer checkpoint learn that full geometry directly from the current synthetic blocks?
3. If not, which rows should become explicit ambiguous/stress buckets instead of silently dragging the main score?

Original BUT remains report-only. Clean/SemiClean/node diagnostic remains the model-selection surface.

## Main Findings

- The full N17043 target is 31,755 rows: good 17,043, medium 10,628, bad core/near-boundary 4,084. The separate original bad-stress bucket has 1,201 rows, with 292 in original test.
- The best ordinary N17043 checkpoint still does not learn the full boundary: best ordinary all-node result is around acc 0.831-0.854 depending on simple gate mode. The best ordinary body before stress probing remains `nl_n17043_gm_trim_bad_boundaryblocks_teacherwall_goodsafe_5e30e8c689a6`.
- A shallow feature-tree diagnostic trained only on node train+val rows can solve the node target:
  - `dtree_depth8_leaf10_trainval_node`
  - node_test: acc 0.976542, good/medium/bad recall 0.990659/0.964302/1.000000
  - node_all: acc 0.989702, good/medium/bad recall 0.993018/0.980429/1.000000
- The same feature-tree diagnostic reaches original_all_10s+ acc 0.979063 with good/medium/bad recall 0.993018/0.980429/0.931315.
- The original_test_all_10s+ score is lower, acc 0.942904, because original test concentrates a hard bad-stress record (`111001`): bad recall is only 0.289538 there.
- If the 292 original-test `bad_extreme_stress` rows are reported as a separate stress bucket, the remaining original-test main subset has 8,185 rows and acc 0.951650 under the depth8 feature tree.

## What Failed

The new N17043 111-like bad-stress ordinary-UFormer probes were trained as small stress diagnostics:

- `nl_n17043_gm_trim_bad_boundaryblocks_badstress111_flatlin_8c8ab0dd32ec`
- `nl_n17043_gm_trim_bad_boundaryblocks_badstress111_pc2_vis_f5aa424db111`

Best non-feature diagnostic modes for these reached only about acc 0.8506. They preserved bad near 0.97, but the good/medium boundary remained too damaged:

- flatline-light, `simple_pc1_gm_gate_trainfit`: acc 0.850606, good/medium/bad 0.880127/0.757151/0.970617
- pc2-visible, `simple_pc1_gm_gate_trainfit`: acc 0.850323, good/medium/bad 0.880009/0.757057/0.969148

Interpretation: simply injecting small 111-like bad-stress rows does not make the ordinary checkpoint learn the stress morphology. It mostly trades medium/good stability for bad robustness.

## What Worked

Two things worked clearly:

- A simple feature geometry works: high PC1 is bad core; low-PC1/QRS-prominent is good; low-QRS/flatline veto returns ambiguous cases to medium.
- A main/high-confidence subset works: keeping good/medium with `boundary_confidence >= 0.6` and `pca_margin >= 1.2`, while keeping all bad, retains 19,600 / 31,755 rows and reaches acc 0.950918 on the all-node diagnostic.

This suggests the right reporting structure is:

- Main learnable target: high-confidence good/medium plus bad core/near-boundary.
- Ambiguous boundary bucket: low-confidence good/medium overlap rows.
- Bad stress bucket: flatline/high-PC2/low-QRS bad outliers, especially original test record `111001`.

## Recommended Next Move

Do not keep broad ordinary-UFormer sweeps on the full N17043 target. They are burning time and repeatedly rediscovering the same class-balance failure.

Instead:

1. Formalize the feature-tree/rule artifact as the current transparent geometry solution.
2. Report full original BUT plus bucketed diagnostics:
   - original_test_all_10s+
   - original_test_main_without_bad_extreme_stress
   - bad_core/near-boundary
   - bad_extreme_stress
   - ambiguous good/medium boundary
3. If a pure neural checkpoint is required, train it on the main learnable subset first, not the full ambiguous/stress target.
4. Treat 111-like bad stress as a separate future domain-adaptation problem, not as a small-row augmentation problem.

## Key Files

- `n17043_feature_tree_original_bucketed_report.md`
- `n17043_feature_tree_original_bucketed_metrics.csv`
- `n17043_high_conf_learnable_subset_report.md`
- `n17043_bad_split_and_ambiguity_report.md`
- `original_bad_stress_split_diagnostic_report.md`
- `original_bad_stress_split_pca.png`
