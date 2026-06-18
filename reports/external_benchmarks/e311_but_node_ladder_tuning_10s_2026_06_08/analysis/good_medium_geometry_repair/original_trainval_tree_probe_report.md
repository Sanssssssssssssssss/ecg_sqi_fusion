# Original Train/Val Tree Probe

This is a report-only upper-bound probe: shallow trees are trained on original BUT train+val labels and evaluated on original test. It is not used for Clean/SemiClean/node selection or checkpoint promotion.

## Best Original-Test Trees
| rule | acc | macro_f1 | good_recall | medium_recall | bad_recall | good_precision | medium_precision | bad_precision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| original_trainval_tree_d6_leaf200 | 0.785537 | 0.679685 | 0.937363 | 0.706733 | 0.289538 | 0.689711 | 0.918110 | 0.967480 |
| original_trainval_tree_d6_leaf20 | 0.778813 | 0.674309 | 0.921978 | 0.706507 | 0.289538 | 0.687987 | 0.900374 | 0.944444 |
| original_trainval_tree_d6_leaf80 | 0.777634 | 0.674124 | 0.939286 | 0.690014 | 0.289538 | 0.681619 | 0.914919 | 0.967480 |
| original_trainval_tree_d5_leaf20 | 0.770084 | 0.668805 | 0.940934 | 0.674198 | 0.289538 | 0.672888 | 0.914216 | 0.967480 |
| original_trainval_tree_d5_leaf80 | 0.770084 | 0.668805 | 0.940934 | 0.674198 | 0.289538 | 0.672888 | 0.914216 | 0.967480 |
| original_trainval_tree_d5_leaf200 | 0.759113 | 0.661151 | 0.940659 | 0.653412 | 0.289538 | 0.658082 | 0.917804 | 0.967480 |
| original_trainval_tree_d2_leaf20 | 0.744485 | 0.650631 | 0.942857 | 0.623588 | 0.289538 | 0.643540 | 0.913605 | 0.967480 |
| original_trainval_tree_d2_leaf80 | 0.744485 | 0.650631 | 0.942857 | 0.623588 | 0.289538 | 0.643540 | 0.913605 | 0.967480 |
| original_trainval_tree_d2_leaf200 | 0.744485 | 0.650631 | 0.942857 | 0.623588 | 0.289538 | 0.643540 | 0.913605 | 0.967480 |
| original_trainval_tree_d2_leaf500 | 0.744485 | 0.650631 | 0.942857 | 0.623588 | 0.289538 | 0.643540 | 0.913605 | 0.967480 |
| original_trainval_tree_d3_leaf20 | 0.744485 | 0.650631 | 0.942857 | 0.623588 | 0.289538 | 0.643540 | 0.913605 | 0.967480 |
| original_trainval_tree_d3_leaf80 | 0.744485 | 0.650631 | 0.942857 | 0.623588 | 0.289538 | 0.643540 | 0.913605 | 0.967480 |

## Best Tree
`original_trainval_tree_d6_leaf200`

![Best original-test confusion](E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\original_trainval_tree_probe_best_test_confusion.png)

![Best tree](E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\original_trainval_tree_probe_best_tree.png)

## Interpretation
- If this shallow original-trained probe is high, BUT can be separated by simple features and the PTB/SemiClean gap is mainly domain adaptation.
- If it remains low, original labels/boundaries are intrinsically ambiguous or require waveform features beyond the current SQI/PCA feature set.
