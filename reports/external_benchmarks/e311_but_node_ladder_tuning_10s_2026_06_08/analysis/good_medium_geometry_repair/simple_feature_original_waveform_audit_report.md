# Simple Feature Original Waveform Audit

This audit compares the promoted N7188 raw checkpoint with the transparent `pc1 + qrs_prom_p90` wide-good feature rule plus precision bad veto on the original BUT test windows. Original BUT remains report-only.

## Counts
| fix_state | class_name | n |
| --- | --- | --- |
| fixed_by_feature_rule | bad | 77 |
| fixed_by_feature_rule | good | 346 |
| fixed_by_feature_rule | medium | 443 |
| regressed_by_feature_rule | bad | 13 |
| regressed_by_feature_rule | good | 146 |
| regressed_by_feature_rule | medium | 105 |
| still_wrong | bad | 202 |
| still_wrong | good | 552 |
| still_wrong | medium | 104 |
| unchanged | bad | 119 |
| unchanged | good | 2596 |
| unchanged | medium | 3774 |

## Median Feature Profile
| fix_state | class_name | pc1 | qrs_prom_p90 | qrs_visibility | pc2 | baseline_step |
| --- | --- | --- | --- | --- | --- | --- |
| fixed_by_feature_rule | bad | -4.8394 | 3.8716 | 0.0680 | 10.2233 | 1.2716 |
| fixed_by_feature_rule | good | -4.8921 | 5.9541 | 0.1508 | 9.0166 | 1.1177 |
| fixed_by_feature_rule | medium | -2.1235 | 3.9756 | 0.0515 | 8.1140 | 1.1618 |
| regressed_by_feature_rule | bad | -3.4790 | 2.0311 | 0.0066 | 17.5049 | 1.8074 |
| regressed_by_feature_rule | good | -4.4643 | 4.4705 | 0.0869 | 10.3837 | 1.0137 |
| regressed_by_feature_rule | medium | -3.5955 | 4.9742 | 0.0723 | 9.6451 | 1.0661 |
| still_wrong | bad | -3.5195 | 3.2008 | 0.0321 | 12.1020 | 1.4449 |
| still_wrong | good | -3.2940 | 3.6837 | 0.0933 | 10.8276 | 1.1544 |
| still_wrong | medium | -3.4098 | 7.6985 | 0.0796 | 3.0602 | 0.7203 |
| unchanged | bad | 9.0054 | 2.9843 | 0.1041 | -0.9604 | 0.2475 |
| unchanged | good | -5.6072 | 8.1823 | 0.2621 | 4.4446 | 0.7162 |
| unchanged | medium | -0.7498 | 5.5190 | 0.0657 | 5.0474 | 1.0445 |

## Waveforms
![Original waveform audit](E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\simple_feature_original_waveform_audit.png)

## Interpretation
- Fixed good/medium rows show the rule is capturing a real morphology axis, not just adding noise thresholds.
- Regressed rows are the important next target: they identify where the low-dimensional rule is too blunt.
- Bad outlier rows need a separate stress policy; mixing them into the good/medium rule is harmful.
