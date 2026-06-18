# Simple Feature Rule Sweep

Rules are trained only on the Clean/SemiClean N7188 node train+val manifest. Original BUT is report-only.

## Best Clean-Node Multiclass Rules
| rule | acc | macro_f1 | good_recall | medium_recall | bad_recall |
| --- | --- | --- | --- | --- | --- |
| node_trained_tree_d4_leaf20 | 0.999751 | 0.999814 | 0.999286 | 1.000000 | 1.000000 |
| node_trained_tree_d3_leaf20 | 0.999253 | 0.999443 | 0.997859 | 1.000000 | 1.000000 |
| node_trained_tree_d2_leaf20 | 0.997261 | 0.997960 | 1.000000 | 0.995593 | 1.000000 |
| node_trained_tree_d2_leaf80 | 0.997261 | 0.997960 | 1.000000 | 0.995593 | 1.000000 |
| node_trained_tree_d2_leaf200 | 0.997261 | 0.997960 | 1.000000 | 0.995593 | 1.000000 |
| node_trained_tree_d3_leaf80 | 0.997261 | 0.997960 | 1.000000 | 0.995593 | 1.000000 |
| node_trained_tree_d3_leaf200 | 0.997261 | 0.997960 | 1.000000 | 0.995593 | 1.000000 |
| node_trained_tree_d4_leaf80 | 0.997261 | 0.997960 | 1.000000 | 0.995593 | 1.000000 |
| node_trained_tree_d4_leaf200 | 0.997261 | 0.997960 | 1.000000 | 0.995593 | 1.000000 |
| node_trained_tree_d1_leaf20 | 0.378486 | 0.509626 | 1.000000 | 0.000000 | 1.000000 |

## Best Original-Test Report-Only Rules
| rule | acc | macro_f1 | good_recall | medium_recall | bad_recall |
| --- | --- | --- | --- | --- | --- |
| node_trained_tree_d4_leaf20 | 0.851363 | 0.724639 | 0.781044 | 0.961365 | 0.289538 |
| node_trained_tree_d3_leaf20 | 0.806535 | 0.690313 | 0.668132 | 0.968369 | 0.289538 |
| n7188_raw_checkpoint | 0.796626 | 0.666487 | 0.753297 | 0.876412 | 0.321168 |
| node_trained_tree_d2_leaf20 | 0.764185 | 0.665774 | 0.932143 | 0.670131 | 0.289538 |
| node_trained_tree_d2_leaf80 | 0.764185 | 0.665774 | 0.932143 | 0.670131 | 0.289538 |
| node_trained_tree_d2_leaf200 | 0.764185 | 0.665774 | 0.932143 | 0.670131 | 0.289538 |
| node_trained_tree_d3_leaf80 | 0.764185 | 0.665774 | 0.932143 | 0.670131 | 0.289538 |
| node_trained_tree_d3_leaf200 | 0.764185 | 0.665774 | 0.932143 | 0.670131 | 0.289538 |
| node_trained_tree_d4_leaf80 | 0.764185 | 0.665774 | 0.932143 | 0.670131 | 0.289538 |
| node_trained_tree_d4_leaf200 | 0.764185 | 0.665774 | 0.932143 | 0.670131 | 0.289538 |

## Original Bad-Outlier Recall Stress
| rule | n | bad_recall | bad_precision | acc |
| --- | --- | --- | --- | --- |
| n7188_raw_checkpoint | 292 | 0.044521 | 1.000000 | 0.044521 |
| node_trained_tree_d1_leaf20 | 292 | 0.000000 | 0.000000 | 0.000000 |
| node_trained_tree_d1_leaf80 | 292 | 0.000000 | 0.000000 | 0.000000 |
| node_trained_tree_d1_leaf200 | 292 | 0.000000 | 0.000000 | 0.000000 |
| node_trained_tree_d2_leaf20 | 292 | 0.000000 | 0.000000 | 0.000000 |
| node_trained_tree_d2_leaf80 | 292 | 0.000000 | 0.000000 | 0.000000 |
| node_trained_tree_d2_leaf200 | 292 | 0.000000 | 0.000000 | 0.000000 |
| node_trained_tree_d3_leaf20 | 292 | 0.000000 | 0.000000 | 0.000000 |
| node_trained_tree_d3_leaf80 | 292 | 0.000000 | 0.000000 | 0.000000 |
| node_trained_tree_d3_leaf200 | 292 | 0.000000 | 0.000000 | 0.000000 |

## Best Good/Medium Binary Trees On Node Test
| rule | acc | macro_f1 | negative_recall | positive_recall |
| --- | --- | --- | --- | --- |
| gm_medium_positive_tree_d2_leaf20 | 0.999743 | 0.999721 | 0.999286 | 1.000000 |
| gm_medium_positive_tree_d3_leaf20 | 0.999743 | 0.999721 | 0.999286 | 1.000000 |
| gm_medium_positive_tree_d4_leaf20 | 0.999743 | 0.999721 | 0.999286 | 1.000000 |
| gm_medium_positive_tree_d1_leaf20 | 0.997177 | 0.996941 | 1.000000 | 0.995593 |
| gm_medium_positive_tree_d1_leaf80 | 0.997177 | 0.996941 | 1.000000 | 0.995593 |
| gm_medium_positive_tree_d1_leaf200 | 0.997177 | 0.996941 | 1.000000 | 0.995593 |
| gm_medium_positive_tree_d2_leaf80 | 0.997177 | 0.996941 | 1.000000 | 0.995593 |
| gm_medium_positive_tree_d2_leaf200 | 0.997177 | 0.996941 | 1.000000 | 0.995593 |

## Best Bad Binary Trees On Node Test
| rule | acc | macro_f1 | negative_recall | positive_recall |
| --- | --- | --- | --- | --- |
| bad_positive_tree_d1_leaf20 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| bad_positive_tree_d1_leaf80 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| bad_positive_tree_d1_leaf200 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| bad_positive_tree_d2_leaf20 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| bad_positive_tree_d2_leaf80 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| bad_positive_tree_d2_leaf200 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| bad_positive_tree_d3_leaf20 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| bad_positive_tree_d3_leaf80 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |

## Top Clean-Node Tree
`node_trained_tree_d4_leaf20`

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

## Interpretation
- If shallow trees are strong on node but weak on original, the blocker is domain/label transfer, not missing clean-node capacity.
- If a bad tree recovers original bad outlier only by hurting original good/medium, the outlier band is not separable by a clean-node-only feature rule.
- Use these trees as generator hints, not as promoted classifiers.