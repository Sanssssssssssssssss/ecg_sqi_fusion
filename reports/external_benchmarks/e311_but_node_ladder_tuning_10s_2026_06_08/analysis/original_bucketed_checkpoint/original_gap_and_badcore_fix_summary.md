# Original Gap And Bad-Core Fix Summary

Selection remains Clean/SemiClean/node diagnostic only. Original BUT is report-only.

## Current Clean/SemiClean Frontier
- Node: `N12000_gm_trim_bad`
- Variant: `nl_n12000_gm_trim_bad_boundaryblocks_n10000shell_balanced_102f5b22e35b`
- Mode: `feature_pc1_qrsprom_tree_n12000_trainval`
- Node diagnostic: acc=0.957023, macro-F1=0.965980, good/medium/bad=0.966917/0.929338/1.000000
- Confusion: `[[11603, 397, 0], [751, 9877, 0], [0, 0, 4084]]`

## Original Report-Only Gap Counts
- N11200 simple baseline / `original_test_all_10s+`: n=8477, acc=0.8041, recall good/medium/bad=0.738/0.931/0.015, missed good/medium/bad=952/304/405
- N11200 simple baseline / `original_all_10s+`: n=32956, acc=0.8377, recall good/medium/bad=0.752/0.939/0.910, missed good/medium/bad=4227/646/475
- N12000 shallow-tree frontier / `original_test_all_10s+`: n=8477, acc=0.8477, recall good/medium/bad=0.805/0.935/0.290, missed good/medium/bad=710/289/292
- N12000 shallow-tree frontier / `original_all_10s+`: n=32956, acc=0.8565, recall good/medium/bad=0.788/0.929/0.931, missed good/medium/bad=3614/751/363

## Bad-Core 0 Diagnosis
- The earlier `bad_core=0` was real for the original test bucket: 119 bad-core/near-boundary windows from record `122001` were all predicted as medium by the N11200 simple PC1/low-QRS mode.
- It was not caused by short-label leakage or deletion: original protocol windows are full 10s windows; short consensus segments were already excluded before protocol generation.
- The failure is a high-PC1 bad island being swallowed by the good/medium shell expansion. A clean-node shallow tree with high-PC1 bad guard fixes it.

## Current Fix
- `original_test_bad_core_near_boundary`: n=119, acc=1.0000, bad_recall=1.000, missed_bad=0, confusion=`[[0, 0, 0], [0, 0, 0], [0, 0, 119]]`
- `original_test_bad_outlier_stress`: n=292, acc=0.0000, bad_recall=0.000, missed_bad=292, confusion=`[[0, 0, 0], [0, 0, 0], [32, 260, 0]]`
- `original_test_drop_bad_outlier_reference`: n=8185, acc=0.8779, bad_recall=1.000, missed_bad=0, confusion=`[[2930, 710, 0], [289, 4137, 0], [0, 0, 119]]`

## Interpretation
- On original test, after fixing bad core, the remaining misses are mostly good->medium plus bad-outlier-stress. N12000 test misses good/medium/bad = 710/289/292.
- On original all 10s+, N12000 misses good/medium/bad = 3614/751/363. So the full-dataset gap is not mainly bad; good-to-medium remains the largest block.
- N12800 currently fails because the expanded good shell becomes too broad; the next useful target is N12400/N12600 with the same shallow-tree family, not blind bad-heavy expansion.

- Source metrics: `nl_n11200_gm_trim_bad_boundaryblocks__f66de47f51__simple_pc1_lowqrs_medium_keep_qbr_d3821ed8_original_bucketed_metrics.csv`, `nl_n12000_gm_trim_bad_boundaryblocks__909e2a8e00__feature_pc1_qrsprom_tree_n12000_trainval_original_bucketed_metrics.csv`
