# Flatline Bad Augmentation

Synthetic-only augmentation targeted at report-only bad_outlier gap: flatline/dropout bad plus non-bad flat/low-amplitude hard negatives.

## Best Original Test Report-Only

| Candidate | Acc | Macro-F1 | Good R | Medium R | Bad R | Bad outlier R |
|---|---:|---:|---:|---:|---:|---:|
| flatbad_strong_histgb | 0.795447 | 0.684158 | 0.893407 | 0.716674 | 0.776156 | 0.684932 |
| flatbad_balanced_histgb | 0.794149 | 0.668876 | 0.914560 | 0.709896 | 0.635036 | 0.592466 |
| flatbad_mild_histgb | 0.794031 | 0.664949 | 0.925824 | 0.704700 | 0.588808 | 0.510274 |
| flatbad_mild_histgb_bad | 0.792733 | 0.669983 | 0.921978 | 0.699277 | 0.654501 | 0.554795 |
| flatbad_strong_histgb_bad | 0.792025 | 0.674072 | 0.896703 | 0.713737 | 0.708029 | 0.595890 |
| flatbad_balanced_histgb_bad | 0.790610 | 0.669805 | 0.902747 | 0.709670 | 0.669100 | 0.585616 |
| flatbad_mild_rf | 0.759821 | 0.653046 | 0.836264 | 0.694532 | 0.785888 | 0.698630 |
| flatbad_balanced_rf | 0.753333 | 0.651841 | 0.831868 | 0.680750 | 0.839416 | 0.773973 |
| flatbad_strong_rf | 0.751445 | 0.650260 | 0.825549 | 0.682332 | 0.839416 | 0.773973 |
| flatbad_mild_extra | 0.743305 | 0.641439 | 0.757143 | 0.725260 | 0.815085 | 0.739726 |
| flatbad_balanced_extra | 0.725139 | 0.628728 | 0.731593 | 0.708992 | 0.841849 | 0.777397 |
| flatbad_strong_extra | 0.704612 | 0.614103 | 0.691758 | 0.699955 | 0.868613 | 0.815068 |

## Synthetic Val/Test

| Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R |
|---|---|---:|---:|---:|---:|---:|
| flatbad_balanced_extra | synthetic_test | 0.992824 | 0.989259 | 0.985356 | 0.998377 | 0.979253 |
| flatbad_balanced_extra | synthetic_val_clean | 0.979058 | 0.972952 | 0.990172 | 0.971591 | 0.992188 |
| flatbad_balanced_extra | synthetic_val_stress | 0.980635 | 0.978155 | 0.990876 | 0.972516 | 0.993478 |
| flatbad_balanced_histgb | synthetic_test | 0.996925 | 0.996055 | 0.995816 | 0.999188 | 0.987552 |
| flatbad_balanced_histgb | synthetic_val_clean | 0.997673 | 0.997249 | 1.000000 | 0.997159 | 0.996094 |
| flatbad_balanced_histgb | synthetic_val_stress | 0.996704 | 0.996342 | 0.998175 | 0.995772 | 0.997826 |
| flatbad_balanced_histgb_bad | synthetic_test | 0.995900 | 0.995089 | 0.995816 | 0.997565 | 0.987552 |
| flatbad_balanced_histgb_bad | synthetic_val_clean | 0.992437 | 0.990083 | 1.000000 | 0.988636 | 0.996094 |
| flatbad_balanced_histgb_bad | synthetic_val_stress | 0.992995 | 0.992075 | 0.998175 | 0.989429 | 0.997826 |
| flatbad_balanced_rf | synthetic_test | 0.988724 | 0.985365 | 0.974895 | 0.995942 | 0.979253 |
| flatbad_balanced_rf | synthetic_val_clean | 0.982548 | 0.976532 | 0.987715 | 0.978220 | 0.992188 |
| flatbad_balanced_rf | synthetic_val_stress | 0.983931 | 0.981193 | 0.990876 | 0.978154 | 0.993478 |
| flatbad_mild_extra | synthetic_test | 0.993337 | 0.989742 | 0.985356 | 0.999188 | 0.979253 |
| flatbad_mild_extra | synthetic_val_clean | 0.983130 | 0.978235 | 0.990172 | 0.978220 | 0.992188 |
| flatbad_mild_extra | synthetic_val_stress | 0.982702 | 0.978844 | 0.989960 | 0.978074 | 0.989011 |
| flatbad_mild_histgb | synthetic_test | 0.996412 | 0.995572 | 0.995816 | 0.998377 | 0.987552 |
| flatbad_mild_histgb | synthetic_val_clean | 0.996510 | 0.995871 | 1.000000 | 0.996212 | 0.992188 |
| flatbad_mild_histgb | synthetic_val_stress | 0.996260 | 0.995459 | 1.000000 | 0.996085 | 0.991758 |
| flatbad_mild_histgb_bad | synthetic_test | 0.996925 | 0.996405 | 0.995816 | 0.998377 | 0.991701 |
| flatbad_mild_histgb_bad | synthetic_val_clean | 0.991274 | 0.988731 | 1.000000 | 0.986742 | 0.996094 |
| flatbad_mild_histgb_bad | synthetic_val_stress | 0.991585 | 0.989887 | 1.000000 | 0.986688 | 0.997253 |
| flatbad_mild_rf | synthetic_test | 0.992824 | 0.990372 | 0.991632 | 0.995942 | 0.979253 |
| flatbad_mild_rf | synthetic_val_clean | 0.993601 | 0.992561 | 0.997543 | 0.992424 | 0.992188 |
| flatbad_mild_rf | synthetic_val_stress | 0.993455 | 0.991827 | 0.995984 | 0.993735 | 0.989011 |
| flatbad_strong_extra | synthetic_test | 0.993337 | 0.989747 | 0.987448 | 0.998377 | 0.979253 |
| flatbad_strong_extra | synthetic_val_clean | 0.979639 | 0.973346 | 0.990172 | 0.972538 | 0.992188 |
| flatbad_strong_extra | synthetic_val_stress | 0.981065 | 0.979819 | 0.993410 | 0.970625 | 0.996622 |
| flatbad_strong_histgb | synthetic_test | 0.996925 | 0.996403 | 0.993724 | 0.999188 | 0.991701 |
| flatbad_strong_histgb | synthetic_val_clean | 0.997673 | 0.996997 | 0.997543 | 0.998106 | 0.996094 |
| flatbad_strong_histgb | synthetic_val_stress | 0.996785 | 0.996390 | 0.996705 | 0.996250 | 0.998311 |
| flatbad_strong_histgb_bad | synthetic_test | 0.996925 | 0.996055 | 0.995816 | 0.999188 | 0.987552 |
| flatbad_strong_histgb_bad | synthetic_val_clean | 0.998837 | 0.998622 | 0.997543 | 1.000000 | 0.996094 |
| flatbad_strong_histgb_bad | synthetic_val_stress | 0.998571 | 0.998470 | 0.996705 | 0.999375 | 0.998311 |
| flatbad_strong_rf | synthetic_test | 0.990774 | 0.986789 | 0.985356 | 0.995130 | 0.979253 |
| flatbad_strong_rf | synthetic_val_clean | 0.993601 | 0.992561 | 0.997543 | 0.992424 | 0.992188 |
| flatbad_strong_rf | synthetic_val_stress | 0.995713 | 0.995269 | 0.998353 | 0.994375 | 0.996622 |

## Manifest Counts

          config     split                  block  label    n
flatbad_balanced     train   bad_flatline_dropout      2 1700
flatbad_balanced     train nonbad_flatlow_hardneg      0 1211
flatbad_balanced     train nonbad_flatlow_hardneg      1 2989
flatbad_balanced valstress   bad_flatline_dropout      2  204
flatbad_balanced valstress nonbad_flatlow_hardneg      0  141
flatbad_balanced valstress nonbad_flatlow_hardneg      1  363
    flatbad_mild     train   bad_flatline_dropout      2  900
    flatbad_mild     train nonbad_flatlow_hardneg      0  771
    flatbad_mild     train nonbad_flatlow_hardneg      1 1829
    flatbad_mild valstress   bad_flatline_dropout      2  108
    flatbad_mild valstress nonbad_flatlow_hardneg      0   91
    flatbad_mild valstress nonbad_flatlow_hardneg      1  221
  flatbad_strong     train   bad_flatline_dropout      2 2800
  flatbad_strong     train nonbad_flatlow_hardneg      0 1875
  flatbad_strong     train nonbad_flatlow_hardneg      1 4325
  flatbad_strong valstress   bad_flatline_dropout      2  336
  flatbad_strong valstress nonbad_flatlow_hardneg      0  200
  flatbad_strong valstress nonbad_flatlow_hardneg      1  544

## Files

- Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_flatline_bad_aug_metrics.csv`
- Manifest CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_flatline_bad_aug_manifest.csv`
- Summary JSON: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_flatline_bad_aug_summary.json`
