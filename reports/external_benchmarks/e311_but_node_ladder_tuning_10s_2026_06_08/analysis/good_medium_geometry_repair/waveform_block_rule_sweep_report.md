# Waveform Block Rule Sweep

This report-only sweep starts from the visible-good adapter and adds coarse waveform morphology guards. Original BUT is not used for promotion.

## Best Accuracy
| rule | acc | macro_f1 | good_recall | medium_recall | bad_recall | use_good_rescue | use_medium_guard | use_bad_stress | good_pc1_max | med_pc1_min | bad_pc2_min |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| waveblock_00486 | 0.897487 | 0.802400 | 0.906868 | 0.928830 | 0.476886 | 1.000000 | 0.000000 | 0.000000 | -4.000000 | -3.600000 | 10.000000 |
| waveblock_00487 | 0.897487 | 0.802400 | 0.906868 | 0.928830 | 0.476886 | 1.000000 | 0.000000 | 0.000000 | -4.000000 | -3.600000 | 12.000000 |
| waveblock_00488 | 0.897487 | 0.802400 | 0.906868 | 0.928830 | 0.476886 | 1.000000 | 0.000000 | 0.000000 | -4.000000 | -3.600000 | 14.000000 |
| waveblock_00489 | 0.897487 | 0.802400 | 0.906868 | 0.928830 | 0.476886 | 1.000000 | 0.000000 | 0.000000 | -4.000000 | -3.200000 | 10.000000 |
| waveblock_00490 | 0.897487 | 0.802400 | 0.906868 | 0.928830 | 0.476886 | 1.000000 | 0.000000 | 0.000000 | -4.000000 | -3.200000 | 12.000000 |
| waveblock_00491 | 0.897487 | 0.802400 | 0.906868 | 0.928830 | 0.476886 | 1.000000 | 0.000000 | 0.000000 | -4.000000 | -3.200000 | 14.000000 |
| waveblock_00492 | 0.897487 | 0.802400 | 0.906868 | 0.928830 | 0.476886 | 1.000000 | 0.000000 | 0.000000 | -4.000000 | -2.800000 | 10.000000 |
| waveblock_00493 | 0.897487 | 0.802400 | 0.906868 | 0.928830 | 0.476886 | 1.000000 | 0.000000 | 0.000000 | -4.000000 | -2.800000 | 12.000000 |
| waveblock_00494 | 0.897487 | 0.802400 | 0.906868 | 0.928830 | 0.476886 | 1.000000 | 0.000000 | 0.000000 | -4.000000 | -2.800000 | 14.000000 |
| waveblock_00708 | 0.897487 | 0.802376 | 0.901648 | 0.933122 | 0.476886 | 1.000000 | 1.000000 | 0.000000 | -4.000000 | -2.800000 | 10.000000 |
| waveblock_00709 | 0.897487 | 0.802376 | 0.901648 | 0.933122 | 0.476886 | 1.000000 | 1.000000 | 0.000000 | -4.000000 | -2.800000 | 12.000000 |
| waveblock_00710 | 0.897487 | 0.802376 | 0.901648 | 0.933122 | 0.476886 | 1.000000 | 1.000000 | 0.000000 | -4.000000 | -2.800000 | 14.000000 |

## Best Macro-F1
| rule | acc | macro_f1 | good_recall | medium_recall | bad_recall | use_good_rescue | use_medium_guard | use_bad_stress | good_pc1_max | med_pc1_min | bad_pc2_min |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| waveblock_00486 | 0.897487 | 0.802400 | 0.906868 | 0.928830 | 0.476886 | 1.000000 | 0.000000 | 0.000000 | -4.000000 | -3.600000 | 10.000000 |
| waveblock_00487 | 0.897487 | 0.802400 | 0.906868 | 0.928830 | 0.476886 | 1.000000 | 0.000000 | 0.000000 | -4.000000 | -3.600000 | 12.000000 |
| waveblock_00488 | 0.897487 | 0.802400 | 0.906868 | 0.928830 | 0.476886 | 1.000000 | 0.000000 | 0.000000 | -4.000000 | -3.600000 | 14.000000 |
| waveblock_00489 | 0.897487 | 0.802400 | 0.906868 | 0.928830 | 0.476886 | 1.000000 | 0.000000 | 0.000000 | -4.000000 | -3.200000 | 10.000000 |
| waveblock_00490 | 0.897487 | 0.802400 | 0.906868 | 0.928830 | 0.476886 | 1.000000 | 0.000000 | 0.000000 | -4.000000 | -3.200000 | 12.000000 |
| waveblock_00491 | 0.897487 | 0.802400 | 0.906868 | 0.928830 | 0.476886 | 1.000000 | 0.000000 | 0.000000 | -4.000000 | -3.200000 | 14.000000 |
| waveblock_00492 | 0.897487 | 0.802400 | 0.906868 | 0.928830 | 0.476886 | 1.000000 | 0.000000 | 0.000000 | -4.000000 | -2.800000 | 10.000000 |
| waveblock_00493 | 0.897487 | 0.802400 | 0.906868 | 0.928830 | 0.476886 | 1.000000 | 0.000000 | 0.000000 | -4.000000 | -2.800000 | 12.000000 |
| waveblock_00494 | 0.897487 | 0.802400 | 0.906868 | 0.928830 | 0.476886 | 1.000000 | 0.000000 | 0.000000 | -4.000000 | -2.800000 | 14.000000 |
| waveblock_00708 | 0.897487 | 0.802376 | 0.901648 | 0.933122 | 0.476886 | 1.000000 | 1.000000 | 0.000000 | -4.000000 | -2.800000 | 10.000000 |
| waveblock_00709 | 0.897487 | 0.802376 | 0.901648 | 0.933122 | 0.476886 | 1.000000 | 1.000000 | 0.000000 | -4.000000 | -2.800000 | 12.000000 |
| waveblock_00710 | 0.897487 | 0.802376 | 0.901648 | 0.933122 | 0.476886 | 1.000000 | 1.000000 | 0.000000 | -4.000000 | -2.800000 | 14.000000 |

## Best Rule Fixed/Regressed Detail
| state | record_id | class_name | original_region | n |
| --- | --- | --- | --- | --- |
| unchanged | 111001 | medium | outlier_low_confidence | 1904 |
| unchanged | 111001 | good | outlier_low_confidence | 1762 |
| unchanged | 111001 | medium | good_medium_overlap | 1644 |
| unchanged | 111001 | good | good_medium_overlap | 1093 |
| unchanged | 111001 | medium | clean_core | 522 |
| still_wrong | 111001 | good | outlier_low_confidence | 228 |
| still_wrong | 111001 | bad | outlier_low_confidence | 215 |
| still_wrong | 111001 | medium | outlier_low_confidence | 206 |
| fixed | 111001 | good | outlier_low_confidence | 201 |
| unchanged | 125001 | good | outlier_low_confidence | 152 |
| unchanged | 122001 | bad | near_bad_boundary | 119 |
| unchanged | 111001 | bad | outlier_low_confidence | 77 |
| still_wrong | 125001 | good | outlier_low_confidence | 68 |
| regressed | 111001 | medium | outlier_low_confidence | 63 |
| unchanged | 122001 | good | good_medium_overlap | 52 |
| still_wrong | 111001 | medium | good_medium_overlap | 45 |
| still_wrong | 111001 | good | good_medium_overlap | 35 |
| unchanged | 122001 | medium | good_medium_overlap | 22 |
| unchanged | 122001 | good | clean_core | 21 |
| unchanged | 111001 | good | clean_core | 17 |
| unchanged | 122001 | medium | clean_core | 8 |
| unchanged | 122001 | medium | outlier_low_confidence | 6 |
| still_wrong | 125001 | good | good_medium_overlap | 5 |
| unchanged | 111001 | medium | medium_bad_overlap | 5 |
| still_wrong | 122001 | good | good_medium_overlap | 3 |
| unchanged | 125001 | good | good_medium_overlap | 3 |
| still_wrong | 111001 | medium | clean_core | 1 |

## Interpretation
- The test asks whether simple waveform morphology can recover the remaining large blocks without many tiny rules.
- If the best rule is only a modest gain, we should avoid adding it to the main adapter and instead use it as generator guidance.
