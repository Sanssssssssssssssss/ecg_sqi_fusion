# Simple Block Rule Expansion Sweep

This report-only sweep tests three coarse blocks on top of the wide-good rule: visible-good rescue, medium guard, and controlled bad-stress rescue. It is not a promotion path because original BUT labels are used only for analysis.

## Best Original-Test Accuracy Among Clean-Node-Viable Rules
| rule | node_acc | node_good | node_medium | node_bad | acc | macro_f1 | good_recall | medium_recall | bad_recall | use_visible_good | use_medium_guard | use_bad_stress | vis_pc1_min | vis_pc1_max | vis_qv_min | med_qv_max | bad_pc2_min |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| block_expand_02308 | 0.989542 | 0.999286 | 0.983574 | 1.000000 | 0.881208 | 0.790671 | 0.851648 | 0.943064 | 0.476886 | 1.000000 | 0.000000 | 0.000000 | -2.100000 | 0.000000 | 0.100000 | 0.060000 | 10.000000 |
| block_expand_02309 | 0.989542 | 0.999286 | 0.983574 | 1.000000 | 0.881208 | 0.790671 | 0.851648 | 0.943064 | 0.476886 | 1.000000 | 0.000000 | 0.000000 | -2.100000 | 0.000000 | 0.100000 | 0.085000 | 10.000000 |
| block_expand_02324 | 0.995767 | 0.999286 | 0.993590 | 1.000000 | 0.881090 | 0.790503 | 0.843956 | 0.949164 | 0.476886 | 1.000000 | 0.000000 | 0.000000 | -2.100000 | 0.000000 | 0.100000 | 0.060000 | 10.000000 |
| block_expand_02325 | 0.995767 | 0.999286 | 0.993590 | 1.000000 | 0.881090 | 0.790503 | 0.843956 | 0.949164 | 0.476886 | 1.000000 | 0.000000 | 0.000000 | -2.100000 | 0.000000 | 0.100000 | 0.085000 | 10.000000 |
| block_expand_02304 | 0.995269 | 0.999286 | 0.992788 | 1.000000 | 0.880854 | 0.790345 | 0.844505 | 0.948260 | 0.476886 | 1.000000 | 0.000000 | 0.000000 | -2.100000 | 0.000000 | 0.100000 | 0.060000 | 10.000000 |
| block_expand_02305 | 0.995269 | 0.999286 | 0.992788 | 1.000000 | 0.880854 | 0.790345 | 0.844505 | 0.948260 | 0.476886 | 1.000000 | 0.000000 | 0.000000 | -2.100000 | 0.000000 | 0.100000 | 0.085000 | 10.000000 |
| block_expand_02400 | 0.995269 | 0.999286 | 0.992788 | 1.000000 | 0.880854 | 0.790345 | 0.844505 | 0.948260 | 0.476886 | 1.000000 | 0.000000 | 0.000000 | -2.100000 | 0.600000 | 0.100000 | 0.060000 | 10.000000 |
| block_expand_02401 | 0.995269 | 0.999286 | 0.992788 | 1.000000 | 0.880854 | 0.790345 | 0.844505 | 0.948260 | 0.476886 | 1.000000 | 0.000000 | 0.000000 | -2.100000 | 0.600000 | 0.100000 | 0.085000 | 10.000000 |
| block_expand_02404 | 0.988297 | 0.999286 | 0.981571 | 1.000000 | 0.880618 | 0.790260 | 0.851648 | 0.941934 | 0.476886 | 1.000000 | 0.000000 | 0.000000 | -2.100000 | 0.600000 | 0.100000 | 0.060000 | 10.000000 |
| block_expand_02405 | 0.988297 | 0.999286 | 0.981571 | 1.000000 | 0.880618 | 0.790260 | 0.851648 | 0.941934 | 0.476886 | 1.000000 | 0.000000 | 0.000000 | -2.100000 | 0.600000 | 0.100000 | 0.085000 | 10.000000 |
| block_expand_02420 | 0.994522 | 0.999286 | 0.991587 | 1.000000 | 0.880500 | 0.790092 | 0.843956 | 0.948034 | 0.476886 | 1.000000 | 0.000000 | 0.000000 | -2.100000 | 0.600000 | 0.100000 | 0.060000 | 10.000000 |
| block_expand_02421 | 0.994522 | 0.999286 | 0.991587 | 1.000000 | 0.880500 | 0.790092 | 0.843956 | 0.948034 | 0.476886 | 1.000000 | 0.000000 | 0.000000 | -2.100000 | 0.600000 | 0.100000 | 0.085000 | 10.000000 |

## Best Original-Test Macro-F1 Among Clean-Node-Viable Rules
| rule | node_acc | node_good | node_medium | node_bad | acc | macro_f1 | good_recall | medium_recall | bad_recall | use_visible_good | use_medium_guard | use_bad_stress | vis_pc1_min | vis_pc1_max | vis_qv_min | med_qv_max | bad_pc2_min |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| block_expand_02308 | 0.989542 | 0.999286 | 0.983574 | 1.000000 | 0.881208 | 0.790671 | 0.851648 | 0.943064 | 0.476886 | 1.000000 | 0.000000 | 0.000000 | -2.100000 | 0.000000 | 0.100000 | 0.060000 | 10.000000 |
| block_expand_02309 | 0.989542 | 0.999286 | 0.983574 | 1.000000 | 0.881208 | 0.790671 | 0.851648 | 0.943064 | 0.476886 | 1.000000 | 0.000000 | 0.000000 | -2.100000 | 0.000000 | 0.100000 | 0.085000 | 10.000000 |
| block_expand_02324 | 0.995767 | 0.999286 | 0.993590 | 1.000000 | 0.881090 | 0.790503 | 0.843956 | 0.949164 | 0.476886 | 1.000000 | 0.000000 | 0.000000 | -2.100000 | 0.000000 | 0.100000 | 0.060000 | 10.000000 |
| block_expand_02325 | 0.995767 | 0.999286 | 0.993590 | 1.000000 | 0.881090 | 0.790503 | 0.843956 | 0.949164 | 0.476886 | 1.000000 | 0.000000 | 0.000000 | -2.100000 | 0.000000 | 0.100000 | 0.085000 | 10.000000 |
| block_expand_02304 | 0.995269 | 0.999286 | 0.992788 | 1.000000 | 0.880854 | 0.790345 | 0.844505 | 0.948260 | 0.476886 | 1.000000 | 0.000000 | 0.000000 | -2.100000 | 0.000000 | 0.100000 | 0.060000 | 10.000000 |
| block_expand_02305 | 0.995269 | 0.999286 | 0.992788 | 1.000000 | 0.880854 | 0.790345 | 0.844505 | 0.948260 | 0.476886 | 1.000000 | 0.000000 | 0.000000 | -2.100000 | 0.000000 | 0.100000 | 0.085000 | 10.000000 |
| block_expand_02400 | 0.995269 | 0.999286 | 0.992788 | 1.000000 | 0.880854 | 0.790345 | 0.844505 | 0.948260 | 0.476886 | 1.000000 | 0.000000 | 0.000000 | -2.100000 | 0.600000 | 0.100000 | 0.060000 | 10.000000 |
| block_expand_02401 | 0.995269 | 0.999286 | 0.992788 | 1.000000 | 0.880854 | 0.790345 | 0.844505 | 0.948260 | 0.476886 | 1.000000 | 0.000000 | 0.000000 | -2.100000 | 0.600000 | 0.100000 | 0.085000 | 10.000000 |
| block_expand_02404 | 0.988297 | 0.999286 | 0.981571 | 1.000000 | 0.880618 | 0.790260 | 0.851648 | 0.941934 | 0.476886 | 1.000000 | 0.000000 | 0.000000 | -2.100000 | 0.600000 | 0.100000 | 0.060000 | 10.000000 |
| block_expand_02405 | 0.988297 | 0.999286 | 0.981571 | 1.000000 | 0.880618 | 0.790260 | 0.851648 | 0.941934 | 0.476886 | 1.000000 | 0.000000 | 0.000000 | -2.100000 | 0.600000 | 0.100000 | 0.085000 | 10.000000 |
| block_expand_02420 | 0.994522 | 0.999286 | 0.991587 | 1.000000 | 0.880500 | 0.790092 | 0.843956 | 0.948034 | 0.476886 | 1.000000 | 0.000000 | 0.000000 | -2.100000 | 0.600000 | 0.100000 | 0.060000 | 10.000000 |
| block_expand_02421 | 0.994522 | 0.999286 | 0.991587 | 1.000000 | 0.880500 | 0.790092 | 0.843956 | 0.948034 | 0.476886 | 1.000000 | 0.000000 | 0.000000 | -2.100000 | 0.600000 | 0.100000 | 0.085000 | 10.000000 |

## Best Rule Detailed Fixed/Regressed Counts
| state | record_id | class_name | original_region | n |
| --- | --- | --- | --- | --- |
| unchanged | 111001 | medium | outlier_low_confidence | 1967 |
| unchanged | 111001 | good | outlier_low_confidence | 1759 |
| unchanged | 111001 | medium | good_medium_overlap | 1644 |
| unchanged | 111001 | good | good_medium_overlap | 1093 |
| unchanged | 111001 | medium | clean_core | 522 |
| still_wrong | 111001 | good | outlier_low_confidence | 429 |
| still_wrong | 111001 | bad | outlier_low_confidence | 215 |
| still_wrong | 111001 | medium | outlier_low_confidence | 205 |
| fixed | 125001 | good | outlier_low_confidence | 151 |
| unchanged | 122001 | bad | near_bad_boundary | 119 |
| unchanged | 111001 | bad | outlier_low_confidence | 77 |
| still_wrong | 125001 | good | outlier_low_confidence | 68 |
| unchanged | 122001 | good | good_medium_overlap | 51 |
| regressed | 111001 | medium | good_medium_overlap | 41 |
| still_wrong | 111001 | good | good_medium_overlap | 35 |
| unchanged | 122001 | medium | good_medium_overlap | 22 |
| unchanged | 122001 | good | clean_core | 21 |
| unchanged | 111001 | good | clean_core | 17 |
| unchanged | 122001 | medium | clean_core | 8 |
| unchanged | 122001 | medium | outlier_low_confidence | 6 |
| unchanged | 111001 | medium | medium_bad_overlap | 5 |
| still_wrong | 125001 | good | good_medium_overlap | 5 |
| still_wrong | 111001 | medium | good_medium_overlap | 4 |
| fixed | 111001 | good | outlier_low_confidence | 3 |
| fixed | 125001 | good | good_medium_overlap | 3 |
| still_wrong | 122001 | good | good_medium_overlap | 3 |
| regressed | 111001 | medium | outlier_low_confidence | 1 |
| regressed | 111001 | medium | clean_core | 1 |
| fixed | 122001 | good | good_medium_overlap | 1 |
| unchanged | 125001 | good | outlier_low_confidence | 1 |

## Interpretation
- If this sweep beats the wide-good baseline, the improvement should come from one of three interpretable blocks rather than a pile of tiny exceptions.
- A useful candidate must keep Clean/SemiClean node gates while improving original report-only metrics.
- If no block improves the baseline, the remaining error likely needs record/domain adaptation or waveform-level denoising features, not more threshold stacking.
