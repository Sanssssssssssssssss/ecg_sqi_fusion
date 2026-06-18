# Bad Low-Frequency Plateau Augmentation

Synthetic-only augmentation for original bad stress: high low-frequency baseline energy, contact-like steps, long low-detail plateaus, and suppressed high-frequency detail. Original BUT remains report-only.

## Best Original Test Report-Only

| Candidate | Acc | Macro-F1 | Good R | Medium R | Bad R | Bad outlier R |
|---|---:|---:|---:|---:|---:|---:|
| badlf_light_histgb_bw28 | 0.876725 | 0.755460 | 0.882418 | 0.917307 | 0.389294 | 0.191781 |
| badlf_balanced_histgb_bw28 | 0.876489 | 0.772718 | 0.895879 | 0.897198 | 0.481752 | 0.280822 |
| badlf_balanced_histgb_lite | 0.876017 | 0.766139 | 0.897802 | 0.897198 | 0.454988 | 0.246575 |
| badlf_balanced_histgb_bw36 | 0.875074 | 0.761878 | 0.895055 | 0.899458 | 0.435523 | 0.263699 |
| badlf_strong_histgb_lite | 0.873304 | 0.779395 | 0.880769 | 0.904654 | 0.469586 | 0.253425 |
| badlf_light_histgb_lite | 0.872243 | 0.759700 | 0.871703 | 0.915725 | 0.408759 | 0.167808 |
| badlf_light_histgb_bw36 | 0.870355 | 0.743189 | 0.878846 | 0.909851 | 0.369830 | 0.157534 |
| badlf_strong_histgb_bw28 | 0.866816 | 0.765591 | 0.857967 | 0.914370 | 0.433090 | 0.219178 |
| badlf_strong_histgb_bw36 | 0.865400 | 0.770896 | 0.858791 | 0.908495 | 0.459854 | 0.239726 |
| badlf_balanced_mlp | 0.824348 | 0.650783 | 0.865110 | 0.845911 | 0.231144 | 0.325342 |
| badlf_strong_mlp | 0.817742 | 0.640691 | 0.860165 | 0.839132 | 0.211679 | 0.297945 |
| badlf_light_extra | 0.809602 | 0.708343 | 0.678571 | 0.950068 | 0.457421 | 0.236301 |
| badlf_light_mlp | 0.802760 | 0.637591 | 0.887637 | 0.786715 | 0.223844 | 0.315068 |
| badlf_light_rf | 0.794857 | 0.697380 | 0.639286 | 0.955264 | 0.445255 | 0.219178 |
| badlf_balanced_extra | 0.772561 | 0.672947 | 0.593956 | 0.946453 | 0.481752 | 0.270548 |
| badlf_strong_extra | 0.766545 | 0.667873 | 0.579121 | 0.950972 | 0.440389 | 0.212329 |
| badlf_balanced_rf | 0.744249 | 0.653584 | 0.517033 | 0.957298 | 0.462287 | 0.243151 |
| badlf_strong_rf | 0.729385 | 0.643836 | 0.477747 | 0.962268 | 0.450122 | 0.226027 |

## Best Bad Recall Report-Only

| Candidate | Acc | Good R | Medium R | Bad R | Bad outlier R |
|---|---:|---:|---:|---:|---:|
| badlf_balanced_histgb_bw28 | 0.876489 | 0.895879 | 0.897198 | 0.481752 | 0.280822 |
| badlf_balanced_extra | 0.772561 | 0.593956 | 0.946453 | 0.481752 | 0.270548 |
| badlf_strong_histgb_lite | 0.873304 | 0.880769 | 0.904654 | 0.469586 | 0.253425 |
| badlf_balanced_rf | 0.744249 | 0.517033 | 0.957298 | 0.462287 | 0.243151 |
| badlf_strong_histgb_bw36 | 0.865400 | 0.858791 | 0.908495 | 0.459854 | 0.239726 |
| badlf_light_extra | 0.809602 | 0.678571 | 0.950068 | 0.457421 | 0.236301 |
| badlf_balanced_histgb_lite | 0.876017 | 0.897802 | 0.897198 | 0.454988 | 0.246575 |
| badlf_strong_rf | 0.729385 | 0.477747 | 0.962268 | 0.450122 | 0.226027 |
| badlf_light_rf | 0.794857 | 0.639286 | 0.955264 | 0.445255 | 0.219178 |
| badlf_strong_extra | 0.766545 | 0.579121 | 0.950972 | 0.440389 | 0.212329 |
| badlf_balanced_histgb_bw36 | 0.875074 | 0.895055 | 0.899458 | 0.435523 | 0.263699 |
| badlf_strong_histgb_bw28 | 0.866816 | 0.857967 | 0.914370 | 0.433090 | 0.219178 |
| badlf_light_histgb_lite | 0.872243 | 0.871703 | 0.915725 | 0.408759 | 0.167808 |
| badlf_light_histgb_bw28 | 0.876725 | 0.882418 | 0.917307 | 0.389294 | 0.191781 |
| badlf_light_histgb_bw36 | 0.870355 | 0.878846 | 0.909851 | 0.369830 | 0.157534 |
| badlf_balanced_mlp | 0.824348 | 0.865110 | 0.845911 | 0.231144 | 0.325342 |
| badlf_light_mlp | 0.802760 | 0.887637 | 0.786715 | 0.223844 | 0.315068 |
| badlf_strong_mlp | 0.817742 | 0.860165 | 0.839132 | 0.211679 | 0.297945 |

## Synthetic Val/Test

| Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R |
|---|---|---:|---:|---:|---:|---:|
| badlf_balanced_extra | synthetic_test | 0.988724 | 0.986985 | 0.968619 | 0.998377 | 0.979253 |
| badlf_balanced_extra | synthetic_val_clean | 0.992437 | 0.992115 | 0.980344 | 0.997159 | 0.992188 |
| badlf_balanced_extra | synthetic_val_lowfreq_stress | 0.965621 | 0.964796 | 0.917998 | 0.980303 | 0.986595 |
| badlf_balanced_histgb_bw28 | synthetic_test | 0.995900 | 0.995427 | 0.987448 | 1.000000 | 0.991701 |
| badlf_balanced_histgb_bw28 | synthetic_val_clean | 0.999418 | 0.999190 | 1.000000 | 1.000000 | 0.996094 |
| badlf_balanced_histgb_bw28 | synthetic_val_lowfreq_stress | 0.994816 | 0.994706 | 0.991480 | 0.995455 | 0.997319 |
| badlf_balanced_histgb_bw36 | synthetic_test | 0.995900 | 0.995427 | 0.987448 | 1.000000 | 0.991701 |
| badlf_balanced_histgb_bw36 | synthetic_val_clean | 0.999418 | 0.999190 | 1.000000 | 1.000000 | 0.996094 |
| badlf_balanced_histgb_bw36 | synthetic_val_lowfreq_stress | 0.993724 | 0.993699 | 0.987220 | 0.995455 | 0.997319 |
| badlf_balanced_histgb_lite | synthetic_test | 0.996412 | 0.995914 | 0.989540 | 1.000000 | 0.991701 |
| badlf_balanced_histgb_lite | synthetic_val_clean | 0.998837 | 0.998382 | 1.000000 | 0.999053 | 0.996094 |
| badlf_balanced_histgb_lite | synthetic_val_lowfreq_stress | 0.994816 | 0.994750 | 0.990415 | 0.995960 | 0.997319 |
| badlf_balanced_mlp | synthetic_test | 0.995900 | 0.995089 | 0.995816 | 0.997565 | 0.987552 |
| badlf_balanced_mlp | synthetic_val_clean | 0.987202 | 0.987197 | 1.000000 | 0.981061 | 0.992188 |
| badlf_balanced_mlp | synthetic_val_lowfreq_stress | 0.988267 | 0.988451 | 0.986155 | 0.986869 | 0.994638 |
| badlf_balanced_rf | synthetic_test | 0.989236 | 0.987458 | 0.966527 | 1.000000 | 0.979253 |
| badlf_balanced_rf | synthetic_val_clean | 0.991856 | 0.991280 | 0.972973 | 0.999053 | 0.992188 |
| badlf_balanced_rf | synthetic_val_lowfreq_stress | 0.975989 | 0.975621 | 0.943557 | 0.985859 | 0.990617 |
| badlf_light_extra | synthetic_test | 0.989236 | 0.987478 | 0.970711 | 0.998377 | 0.979253 |
| badlf_light_extra | synthetic_val_clean | 0.994764 | 0.994410 | 0.992629 | 0.996212 | 0.992188 |
| badlf_light_extra | synthetic_val_lowfreq_stress | 0.965046 | 0.963476 | 0.919298 | 0.980381 | 0.988189 |
| badlf_light_histgb_bw28 | synthetic_test | 0.996925 | 0.996053 | 0.993724 | 1.000000 | 0.987552 |
| badlf_light_histgb_bw28 | synthetic_val_clean | 0.999418 | 0.999190 | 1.000000 | 1.000000 | 0.996094 |
| badlf_light_histgb_bw28 | synthetic_val_lowfreq_stress | 0.992691 | 0.992193 | 0.984795 | 0.996076 | 0.994094 |
| badlf_light_histgb_bw36 | synthetic_test | 0.997437 | 0.996887 | 0.993724 | 1.000000 | 0.991701 |
| badlf_light_histgb_bw36 | synthetic_val_clean | 0.999418 | 0.999190 | 1.000000 | 1.000000 | 0.996094 |
| badlf_light_histgb_bw36 | synthetic_val_lowfreq_stress | 0.992374 | 0.991642 | 0.985965 | 0.995516 | 0.992126 |
| badlf_light_histgb_lite | synthetic_test | 0.996412 | 0.995570 | 0.993724 | 0.999188 | 0.987552 |
| badlf_light_histgb_lite | synthetic_val_clean | 0.999418 | 0.999190 | 1.000000 | 1.000000 | 0.996094 |
| badlf_light_histgb_lite | synthetic_val_lowfreq_stress | 0.993327 | 0.992898 | 0.985965 | 0.997197 | 0.992126 |
| badlf_light_mlp | synthetic_test | 0.994874 | 0.994479 | 0.995816 | 0.995130 | 0.991701 |
| badlf_light_mlp | synthetic_val_clean | 0.974404 | 0.967525 | 0.997543 | 0.960227 | 0.996094 |
| badlf_light_mlp | synthetic_val_lowfreq_stress | 0.980934 | 0.977203 | 0.995322 | 0.970291 | 0.994094 |
| badlf_light_rf | synthetic_test | 0.989749 | 0.987952 | 0.968619 | 1.000000 | 0.979253 |
| badlf_light_rf | synthetic_val_clean | 0.993601 | 0.993006 | 0.980344 | 0.999053 | 0.992188 |
| badlf_light_rf | synthetic_val_lowfreq_stress | 0.974579 | 0.973320 | 0.943860 | 0.984305 | 0.992126 |
| badlf_strong_extra | synthetic_test | 0.989749 | 0.987961 | 0.970711 | 0.999188 | 0.979253 |
| badlf_strong_extra | synthetic_val_clean | 0.992437 | 0.992115 | 0.980344 | 0.997159 | 0.992188 |
| badlf_strong_extra | synthetic_val_lowfreq_stress | 0.963457 | 0.962773 | 0.918443 | 0.972124 | 0.992095 |
| badlf_strong_histgb_bw28 | synthetic_test | 0.997950 | 0.997372 | 0.995816 | 1.000000 | 0.991701 |
| badlf_strong_histgb_bw28 | synthetic_val_clean | 0.998255 | 0.998057 | 1.000000 | 0.998106 | 0.996094 |
| badlf_strong_histgb_bw28 | synthetic_val_lowfreq_stress | 0.992875 | 0.992781 | 0.988879 | 0.993363 | 0.996047 |
| badlf_strong_histgb_bw36 | synthetic_test | 0.994362 | 0.994308 | 0.981172 | 0.999188 | 0.995851 |
| badlf_strong_histgb_bw36 | synthetic_val_clean | 0.990692 | 0.987282 | 1.000000 | 0.986742 | 0.992188 |
| badlf_strong_histgb_bw36 | synthetic_val_lowfreq_stress | 0.987589 | 0.987373 | 0.987025 | 0.984071 | 0.996047 |
| badlf_strong_histgb_lite | synthetic_test | 0.995900 | 0.995427 | 0.987448 | 1.000000 | 0.991701 |
| badlf_strong_histgb_lite | synthetic_val_clean | 0.986038 | 0.981382 | 0.997543 | 0.979167 | 0.996094 |
| badlf_strong_histgb_lite | synthetic_val_lowfreq_stress | 0.987129 | 0.986887 | 0.987952 | 0.982743 | 0.996047 |
| badlf_strong_mlp | synthetic_test | 0.993849 | 0.993652 | 0.991632 | 0.993506 | 1.000000 |
| badlf_strong_mlp | synthetic_val_clean | 0.996510 | 0.995626 | 0.997543 | 0.996212 | 0.996094 |
| badlf_strong_mlp | synthetic_val_lowfreq_stress | 0.993335 | 0.993087 | 0.989805 | 0.993805 | 0.996047 |
| badlf_strong_rf | synthetic_test | 0.989236 | 0.987458 | 0.966527 | 1.000000 | 0.979253 |
| badlf_strong_rf | synthetic_val_clean | 0.987784 | 0.985692 | 0.972973 | 0.992424 | 0.992188 |
| badlf_strong_rf | synthetic_val_lowfreq_stress | 0.973569 | 0.973108 | 0.949954 | 0.976991 | 0.991107 |

## Manifest Counts

        config     split                          block  label    n
badlf_balanced     train           bad_flatline_dropout      2 1300
badlf_balanced     train       bad_lowfreq_plateau_step      2 2200
badlf_balanced     train      good_lowdetail_qrsvisible      0 3800
badlf_balanced     train medium_lowdetail_flat_boundary      1 6600
badlf_balanced valstress           bad_flatline_dropout      2  182
badlf_balanced valstress       bad_lowfreq_plateau_step      2  308
badlf_balanced valstress      good_lowdetail_qrsvisible      0  532
badlf_balanced valstress medium_lowdetail_flat_boundary      1  924
   badlf_light     train           bad_flatline_dropout      2  900
   badlf_light     train       bad_lowfreq_plateau_step      2  900
   badlf_light     train      good_lowdetail_qrsvisible      0 3200
   badlf_light     train medium_lowdetail_flat_boundary      1 5200
   badlf_light valstress           bad_flatline_dropout      2  126
   badlf_light valstress       bad_lowfreq_plateau_step      2  126
   badlf_light valstress      good_lowdetail_qrsvisible      0  448
   badlf_light valstress medium_lowdetail_flat_boundary      1  728
  badlf_strong     train           bad_flatline_dropout      2 1200
  badlf_strong     train       bad_lowfreq_plateau_step      2 4200
  badlf_strong     train      good_lowdetail_qrsvisible      0 4800
  badlf_strong     train medium_lowdetail_flat_boundary      1 8600
  badlf_strong valstress           bad_flatline_dropout      2  168
  badlf_strong valstress       bad_lowfreq_plateau_step      2  588
  badlf_strong valstress      good_lowdetail_qrsvisible      0  672
  badlf_strong valstress medium_lowdetail_flat_boundary      1 1204

## Files

- Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\bad_lowfreq_plateau_aug_metrics.csv`
- Manifest CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\bad_lowfreq_plateau_aug_manifest.csv`
- Summary JSON: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\bad_lowfreq_plateau_aug_summary.json`
