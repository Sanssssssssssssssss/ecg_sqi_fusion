# BUT-Like Low-Detail Augmentation

Synthetic-only domain randomization for low-derivative/low-detail original-like morphology across good, medium, and bad.

## Best Original Test Report-Only

| Candidate | Acc | Macro-F1 | Good R | Medium R | Bad R | Bad outlier R |
|---|---:|---:|---:|---:|---:|---:|
| butlow_balanced_histgb | 0.883567 | 0.797254 | 0.901648 | 0.904880 | 0.493917 | 0.325342 |
| butlow_broad_histgb | 0.880854 | 0.796824 | 0.887912 | 0.912110 | 0.481752 | 0.270548 |
| butlow_medium_histgb_lite | 0.880500 | 0.790646 | 0.892033 | 0.909399 | 0.467153 | 0.260274 |
| butlow_medium_histgb | 0.878849 | 0.779881 | 0.883516 | 0.916403 | 0.433090 | 0.243151 |
| butlow_balanced_histgb_lite | 0.878023 | 0.797372 | 0.900275 | 0.894261 | 0.506083 | 0.304795 |
| butlow_broad_histgb_lite | 0.877433 | 0.773962 | 0.890110 | 0.909851 | 0.416058 | 0.250000 |
| butlow_medium_mlp | 0.817860 | 0.651207 | 0.818407 | 0.869182 | 0.260341 | 0.366438 |
| butlow_balanced_mlp | 0.808305 | 0.650216 | 0.895330 | 0.790330 | 0.231144 | 0.315068 |
| butlow_broad_mlp | 0.807479 | 0.644991 | 0.850549 | 0.819928 | 0.291971 | 0.410959 |
| butlow_broad_extra | 0.791790 | 0.705482 | 0.619505 | 0.962720 | 0.476886 | 0.263699 |
| butlow_balanced_rf | 0.788015 | 0.702943 | 0.610989 | 0.964528 | 0.454988 | 0.232877 |
| butlow_balanced_extra | 0.782116 | 0.694576 | 0.598626 | 0.963624 | 0.452555 | 0.229452 |
| butlow_medium_extra | 0.773387 | 0.688627 | 0.569505 | 0.971532 | 0.445255 | 0.219178 |
| butlow_medium_rf | 0.751327 | 0.671102 | 0.513462 | 0.976502 | 0.433090 | 0.202055 |
| butlow_broad_rf | 0.748614 | 0.672076 | 0.517857 | 0.966335 | 0.447689 | 0.222603 |

## Synthetic Val/Test

| Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R |
|---|---|---:|---:|---:|---:|---:|
| butlow_balanced_extra | synthetic_test | 0.989749 | 0.987961 | 0.970711 | 0.999188 | 0.979253 |
| butlow_balanced_extra | synthetic_val_clean | 0.994764 | 0.994410 | 0.992629 | 0.996212 | 0.992188 |
| butlow_balanced_extra | synthetic_val_stress | 0.975026 | 0.974835 | 0.940582 | 0.989881 | 0.980583 |
| butlow_balanced_histgb | synthetic_test | 0.996925 | 0.996401 | 0.991632 | 1.000000 | 0.991701 |
| butlow_balanced_histgb | synthetic_val_clean | 0.998837 | 0.998382 | 1.000000 | 0.999053 | 0.996094 |
| butlow_balanced_histgb | synthetic_val_stress | 0.992716 | 0.992706 | 0.986094 | 0.995833 | 0.992718 |
| butlow_balanced_histgb_lite | synthetic_test | 0.996412 | 0.995218 | 0.995816 | 0.999188 | 0.983402 |
| butlow_balanced_histgb_lite | synthetic_val_clean | 0.999418 | 0.999190 | 1.000000 | 1.000000 | 0.996094 |
| butlow_balanced_histgb_lite | synthetic_val_stress | 0.993757 | 0.993635 | 0.988622 | 0.997024 | 0.990291 |
| butlow_balanced_mlp | synthetic_test | 0.995900 | 0.995438 | 0.993724 | 0.997565 | 0.991701 |
| butlow_balanced_mlp | synthetic_val_clean | 0.981385 | 0.981996 | 1.000000 | 0.970644 | 0.996094 |
| butlow_balanced_mlp | synthetic_val_stress | 0.983004 | 0.983675 | 0.992415 | 0.976786 | 0.990291 |
| butlow_balanced_rf | synthetic_test | 0.989236 | 0.987468 | 0.968619 | 0.999188 | 0.979253 |
| butlow_balanced_rf | synthetic_val_clean | 0.993601 | 0.993247 | 0.980344 | 0.999053 | 0.992188 |
| butlow_balanced_rf | synthetic_val_stress | 0.980576 | 0.980587 | 0.954488 | 0.991071 | 0.987864 |
| butlow_broad_extra | synthetic_test | 0.988724 | 0.986985 | 0.968619 | 0.998377 | 0.979253 |
| butlow_broad_extra | synthetic_val_clean | 0.992437 | 0.992115 | 0.980344 | 0.997159 | 0.992188 |
| butlow_broad_extra | synthetic_val_stress | 0.966962 | 0.967814 | 0.905213 | 0.993446 | 0.983083 |
| butlow_broad_histgb | synthetic_test | 0.997437 | 0.997231 | 0.991632 | 1.000000 | 0.995851 |
| butlow_broad_histgb | synthetic_val_clean | 0.994764 | 0.992794 | 1.000000 | 0.992424 | 0.996094 |
| butlow_broad_histgb | synthetic_val_stress | 0.991673 | 0.989684 | 0.989573 | 0.993446 | 0.988722 |
| butlow_broad_histgb_lite | synthetic_test | 0.997950 | 0.997719 | 0.995816 | 0.999188 | 0.995851 |
| butlow_broad_histgb_lite | synthetic_val_clean | 0.999418 | 0.999190 | 1.000000 | 1.000000 | 0.996094 |
| butlow_broad_histgb_lite | synthetic_val_stress | 0.993554 | 0.992387 | 0.989573 | 0.996723 | 0.988722 |
| butlow_broad_mlp | synthetic_test | 0.995900 | 0.995227 | 0.991632 | 0.997565 | 0.995851 |
| butlow_broad_mlp | synthetic_val_clean | 0.993601 | 0.991814 | 0.992629 | 0.994318 | 0.992188 |
| butlow_broad_mlp | synthetic_val_stress | 0.992748 | 0.991754 | 0.989573 | 0.994850 | 0.990602 |
| butlow_broad_rf | synthetic_test | 0.989749 | 0.987952 | 0.968619 | 1.000000 | 0.979253 |
| butlow_broad_rf | synthetic_val_clean | 0.991856 | 0.991280 | 0.972973 | 0.999053 | 0.992188 |
| butlow_broad_rf | synthetic_val_stress | 0.981735 | 0.981569 | 0.953555 | 0.994382 | 0.986842 |
| butlow_medium_extra | synthetic_test | 0.989236 | 0.987468 | 0.968619 | 0.999188 | 0.979253 |
| butlow_medium_extra | synthetic_val_clean | 0.991856 | 0.991541 | 0.977887 | 0.997159 | 0.992188 |
| butlow_medium_extra | synthetic_val_stress | 0.976305 | 0.973577 | 0.923505 | 0.995935 | 0.974771 |
| butlow_medium_histgb | synthetic_test | 0.997950 | 0.997372 | 0.995816 | 1.000000 | 0.991701 |
| butlow_medium_histgb | synthetic_val_clean | 0.998255 | 0.997577 | 1.000000 | 0.998106 | 0.996094 |
| butlow_medium_histgb | synthetic_val_stress | 0.993916 | 0.992937 | 0.990264 | 0.995427 | 0.993119 |
| butlow_medium_histgb_lite | synthetic_test | 0.997950 | 0.997372 | 0.995816 | 1.000000 | 0.991701 |
| butlow_medium_histgb_lite | synthetic_val_clean | 0.993601 | 0.991219 | 1.000000 | 0.990530 | 0.996094 |
| butlow_medium_histgb_lite | synthetic_val_stress | 0.993276 | 0.991441 | 0.991655 | 0.993394 | 0.995413 |
| butlow_medium_mlp | synthetic_test | 0.990261 | 0.989972 | 0.993724 | 0.987825 | 0.995851 |
| butlow_medium_mlp | synthetic_val_clean | 0.990692 | 0.988059 | 0.990172 | 0.989583 | 0.996094 |
| butlow_medium_mlp | synthetic_val_stress | 0.991354 | 0.989684 | 0.984701 | 0.992378 | 0.997706 |
| butlow_medium_rf | synthetic_test | 0.987699 | 0.985974 | 0.960251 | 1.000000 | 0.979253 |
| butlow_medium_rf | synthetic_val_clean | 0.991856 | 0.991521 | 0.972973 | 0.999053 | 0.992188 |
| butlow_medium_rf | synthetic_val_stress | 0.981108 | 0.979883 | 0.941586 | 0.993902 | 0.988532 |

## Manifest Counts

         config     split                          block  label    n
butlow_balanced     train           bad_flatline_dropout      2 1300
butlow_balanced     train      good_lowdetail_qrsvisible      0 3200
butlow_balanced     train medium_lowdetail_flat_boundary      1 5200
butlow_balanced valstress           bad_flatline_dropout      2  156
butlow_balanced valstress      good_lowdetail_qrsvisible      0  384
butlow_balanced valstress medium_lowdetail_flat_boundary      1  624
   butlow_broad     train           bad_flatline_dropout      2 2300
   butlow_broad     train      good_lowdetail_qrsvisible      0 5400
   butlow_broad     train medium_lowdetail_flat_boundary      1 9000
   butlow_broad valstress           bad_flatline_dropout      2  276
   butlow_broad valstress      good_lowdetail_qrsvisible      0  648
   butlow_broad valstress medium_lowdetail_flat_boundary      1 1080
  butlow_medium     train           bad_flatline_dropout      2 1500
  butlow_medium     train      good_lowdetail_qrsvisible      0 2600
  butlow_medium     train medium_lowdetail_flat_boundary      1 7600
  butlow_medium valstress           bad_flatline_dropout      2  180
  butlow_medium valstress      good_lowdetail_qrsvisible      0  312
  butlow_medium valstress medium_lowdetail_flat_boundary      1  912

## Files

- Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_butlike_lowdetail_aug_metrics.csv`
- Manifest CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_butlike_lowdetail_aug_manifest.csv`
- Summary JSON: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_butlike_lowdetail_aug_summary.json`
