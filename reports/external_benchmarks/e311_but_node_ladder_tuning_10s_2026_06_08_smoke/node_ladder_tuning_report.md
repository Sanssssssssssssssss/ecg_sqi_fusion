# Original-Aware SemiCleanBUT Node Ladder

This package freezes each SemiClean boundary as a node, fits PTB synthetic data to that node in 64D/PCA space, then trains and promotes only after the node passes. It is a diagnostic/generator workflow, not a replacement for formal BUT original test.

## Node Registry

| node_id | level | role | status | existing_acc | existing_medium_recall | existing_bad_recall | reason |
| --- | --- | --- | --- | --- | --- | --- | --- |
| N3600_anchor | 3600 | frozen_anchor | frozen_anchor | 0.9544 | 0.9147 | 1.0000 | Current feasible 0.95 node; freeze as sanity baseline. |
| N4200_bridge | 4200 | active_bridge | pending | 0.9356 | 0.8957 | 0.9714 | First failing widening node; add good/medium overlap and controlled near-bad shell. |
| N4800_wide | 4800 | wide_after_bridge | pending | 0.9279 | 0.8808 | 0.9744 | Wider target after N4200 is stable; increases medium overlap and bad near-boundary coverage. |
| N5200_stress | 5200 | stress_diagnostic_only | pending | 0.9049 | 0.8719 | 0.9208 | Stress node with outlier shell; do not promote until N4800 succeeds. |

## Best Node 64D Fits

| node_id | rank | variant_id | node_score | node_region_score | good_64d_KS | medium_64d_KS | bad_64d_KS | good_gm_overlap_coverage | medium_gm_overlap_coverage | bad_near_boundary_coverage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| N4200_bridge | 1 | nl_n4200_bridge_scan_001_sc_overlap_qrs_visible_compact_c_47e1dbead201 | 0.5196 | 0.7488 | 0.4025 | 0.3010 | 0.7016 | 0.7878 | 0.5922 | 0.8583 |
| N4200_bridge | 2 | nl_n4200_bridge_scan_002_sc_overlap_qrs_visible_compact_c_21fde1f0156b | 0.5326 | 0.7864 | 0.4288 | 0.3010 | 0.7016 | 0.5700 | 0.5922 | 0.8583 |

## Node Diagnostics

_No node training diagnostics yet._

## Promotion Decisions

_No promotion decisions yet._

## Files

- `node_registry.json` / `node_registry.csv`: immutable node definitions plus current status.
- `nodes/<node_id>/node_boundary_manifest.csv`: selected BUT windows for that node.
- `nodes/<node_id>/node_target_distributions.json`: 64D target distribution and region mix.
- `nodes/<node_id>/node64_distance_leaderboard.csv`: no-training synthetic fit ranking.
- `nodes/<node_id>/figures/best_rule_node_overlay.png`: node target vs best PTB synthetic.
- `node_ladder_training_summary.jsonl` and `node_ladder_diagnostic_metrics.csv`: training and node-filtered diagnostics.

## Current Recommendation

Treat `N3600_anchor` as frozen. Work on `N4200_bridge` until it reaches the promotion gates; only then move to `N4800_wide`. If `N4200` misses, inspect medium good/medium-overlap coverage and bad near-boundary coverage before changing network size.
