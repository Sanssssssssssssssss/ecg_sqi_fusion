# Simple Feature Breakthrough Report

## Takeaway

A shallow, interpretable `pc1 + qrs_prom_p90` rule almost perfectly separates the N7188 Clean/SemiClean node geometry and improves original-test report-only accuracy. The same idea did not transfer when forced into a normal checkpoint via extra synthetic rows, so the next research path should stabilize a simple rule/adaptor rather than keep piling boundary blocks.

## Rule Artifact

- Artifact: `rule_n7188_simple_pc1_qrsprom_featuretree`
- Clean-node rule: `node_trained_tree_d4_leaf20`
- Main split: `pc1 <= -2.2571`; within that low-PC1 band, `qrs_prom_p90 > 5.0252` rescues good, otherwise medium.
- Bad split: `pc1 > 6.2370` marks the clean-node bad island.
- Optional original report-only veto: visible-QRS bad-stress precision veto from the bucketed original script.

## Clean/SemiClean Node Result

- Acc: `0.999751`
- Macro-F1: `0.999814`
- Good/medium/bad recall: `0.999286` / `1.000000` / `1.000000`

## Original BUT Report-Only

- Raw N7188 original-test acc: `0.796626`
- Simple rule + precision veto original-test acc: `0.853840` (`+0.057214` vs raw)
- Good/medium/bad recall: `0.778846` / `0.950520` / `0.476886`
- Bad core recall: `1.000000`; bad outlier stress recall: `0.263699`
- Original-all acc: `0.839331`; original-all bad recall: `0.946831`

## Negative Result To Preserve

- N7190 pc1/qrs-prom synthetic block training did not reproduce the explicit rule on original.
- The light N7190 checkpoint stayed clean-node strong but original-test acc dropped to about `0.748`; balanced/medium-guard variants became class-biased.
- Therefore, extra synthetic rows are not enough; this boundary is better treated as an explicit low-dimensional adaptor or a distillation target.

## Next Research Direction

- Bootstrap the `pc1/qrs_prom_p90` thresholds on node train+val to test stability.
- Run waveform panels for original rows fixed by the simple rule versus still wrong.
- Try a small post-hoc logit adapter or rule-engine artifact first; avoid another broad synthetic block sweep until the adapter failure mode is known.

## Visuals

- PCA/QRS-prom geometry: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\simple_feature_pc1_qrsprom_geometry.png`
- Node confusion: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\simple_feature_node_test_best_confusion.png`
- Original confusion: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\simple_feature_original_test_best_confusion.png`

## Top Tree

```text
|--- pc1 <= 6.2370
|   |--- pc1 <= -2.2571
|   |   |--- qrs_prom_p90 <= 5.0252
|   |   |   |--- class: 1
|   |   |--- qrs_prom_p90 >  5.0252
|   |   |   |--- sqi_sSQI <= 2.0342
|   |   |   |   |--- class: 0
|   |   |   |--- sqi_sSQI >  2.0342
|   |   |   |   |--- class: 0
|   |--- pc1 >  -2.2571
|   |   |--- pc1 <= -1.9965
|   |   |   |--- class: 1
|   |   |--- pc1 >  -1.9965
|   |   |   |--- amplitude_entropy <= 0.5836
|   |   |   |   |--- class: 1
|   |   |   |--- amplitude_entropy >  0.5836
|   |   |   |   |--- class: 1
|--- pc1 >  6.2370
|   |--- sqi_bSQI <= 0.0179
|   |   |--- class: 2
|   |--- sqi_bSQI >  0.0179
|   |   |--- class: 2

```
