# Big Waveform Student Checkpoint Evaluation

Checkpoints are waveform-only Transformer-family students. BUT buckets are report-only; no BUT row is used for training or selection.

| Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m | Teacher core MAE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| strict_recoverable_localmorph_patch50 | synthetic_val | 0.991274 | 0.990216 | 0.975430 | 1.000000 | 0.980469 | 10 | 0 | 5 | 0.7042 |
| strict_recoverable_localmorph_patch50 | synthetic_test | 0.992824 | 0.991455 | 0.981172 | 1.000000 | 0.979253 | 9 | 0 | 5 | 0.6398 |
| strict_recoverable_localmorph_patch50_badcal | synthetic_test | 0.992312 | 0.990409 | 0.979079 | 1.000000 | 0.979253 | 9 | 0 | 5 | nan |
| strict_recoverable_localmorph_patch50 | original_test_all_10s+ | 0.830365 | 0.712688 | 0.838462 | 0.872119 | 0.309002 | 588 | 546 | 112 | nan |
| strict_recoverable_localmorph_patch50_badcal | original_test_all_10s+ | 0.829775 | 0.712951 | 0.838462 | 0.870086 | 0.318735 | 586 | 545 | 108 | nan |
| strict_recoverable_localmorph_patch50 | original_all_10s+ | 0.805104 | 0.837688 | 0.689139 | 0.926797 | 0.934342 | 5297 | 749 | 170 | nan |
| strict_recoverable_localmorph_patch50_badcal | original_all_10s+ | 0.804831 | 0.837175 | 0.689139 | 0.925386 | 0.935478 | 5295 | 748 | 165 | nan |
| strict_recoverable_localmorph_patch50 | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 | nan |
| strict_recoverable_localmorph_patch50_badcal | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 | nan |
| strict_recoverable_localmorph_patch50 | bad_outlier_stress | 0.027397 | 0.017778 | 0.000000 | 0.000000 | 0.027397 | 0 | 0 | 112 | nan |
| strict_recoverable_localmorph_patch50_badcal | bad_outlier_stress | 0.041096 | 0.026316 | 0.000000 | 0.000000 | 0.041096 | 0 | 0 | 108 | nan |
| strict_qrspeak_critical_lowaug | synthetic_val | 0.995928 | 0.994536 | 0.997543 | 1.000000 | 0.976562 | 1 | 0 | 6 | 0.6673 |
| strict_qrspeak_critical_lowaug | synthetic_test | 0.995900 | 0.994378 | 0.993724 | 1.000000 | 0.979253 | 3 | 0 | 5 | 0.6312 |
| strict_qrspeak_critical_lowaug_badcal | synthetic_test | 0.996412 | 0.995217 | 0.993724 | 1.000000 | 0.983402 | 3 | 0 | 4 | nan |
| strict_qrspeak_critical_lowaug | original_test_all_10s+ | 0.787897 | 0.649717 | 0.863736 | 0.779033 | 0.211679 | 496 | 975 | 102 | nan |
| strict_qrspeak_critical_lowaug_badcal | original_test_all_10s+ | 0.795447 | 0.717485 | 0.862912 | 0.773385 | 0.435523 | 496 | 935 | 43 | nan |
| strict_qrspeak_critical_lowaug | original_all_10s+ | 0.757647 | 0.800339 | 0.621604 | 0.892360 | 0.925449 | 6449 | 1139 | 170 | nan |
| strict_qrspeak_critical_lowaug_badcal | original_all_10s+ | 0.758860 | 0.800671 | 0.621311 | 0.886526 | 0.945695 | 6447 | 1095 | 97 | nan |
| strict_qrspeak_critical_lowaug | bad_core_nearboundary | 0.705882 | 0.275862 | 0.000000 | 0.000000 | 0.705882 | 0 | 0 | 35 | nan |
| strict_qrspeak_critical_lowaug_badcal | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 | nan |
| strict_qrspeak_critical_lowaug | bad_outlier_stress | 0.010274 | 0.006780 | 0.000000 | 0.000000 | 0.010274 | 0 | 0 | 67 | nan |
| strict_qrspeak_critical_lowaug_badcal | bad_outlier_stress | 0.205479 | 0.113636 | 0.000000 | 0.000000 | 0.205479 | 0 | 0 | 43 | nan |
| strict_qrspeak_bin_widepeak | synthetic_val | 0.994183 | 0.992830 | 0.990172 | 1.000000 | 0.976562 | 4 | 0 | 6 | 0.6473 |
| strict_qrspeak_bin_widepeak | synthetic_test | 0.995387 | 0.993892 | 0.991632 | 1.000000 | 0.979253 | 4 | 0 | 5 | 0.6310 |
| strict_qrspeak_bin_widepeak_badcal | synthetic_test | 0.995387 | 0.993892 | 0.991632 | 1.000000 | 0.979253 | 4 | 0 | 5 | nan |
| strict_qrspeak_bin_widepeak | original_test_all_10s+ | 0.805828 | 0.680565 | 0.848077 | 0.821961 | 0.257908 | 553 | 784 | 92 | nan |
| strict_qrspeak_bin_widepeak_badcal | original_test_all_10s+ | 0.808305 | 0.708016 | 0.848077 | 0.819024 | 0.340633 | 552 | 778 | 66 | nan |
| strict_qrspeak_bin_widepeak | original_all_10s+ | 0.758314 | 0.801421 | 0.609400 | 0.912213 | 0.929044 | 6657 | 929 | 161 | nan |
| strict_qrspeak_bin_widepeak_badcal | original_all_10s+ | 0.758678 | 0.801508 | 0.609341 | 0.909296 | 0.937370 | 6654 | 920 | 125 | nan |
| strict_qrspeak_bin_widepeak | bad_core_nearboundary | 0.873950 | 0.310912 | 0.000000 | 0.000000 | 0.873950 | 0 | 0 | 15 | nan |
| strict_qrspeak_bin_widepeak_badcal | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 | nan |
| strict_qrspeak_bin_widepeak | bad_outlier_stress | 0.006849 | 0.004535 | 0.000000 | 0.000000 | 0.006849 | 0 | 0 | 77 | nan |
| strict_qrspeak_bin_widepeak_badcal | bad_outlier_stress | 0.071918 | 0.044728 | 0.000000 | 0.000000 | 0.071918 | 0 | 0 | 66 | nan |
| strict_hybrid_template_teacherselect | synthetic_val | 0.995346 | 0.994222 | 0.992629 | 1.000000 | 0.980469 | 3 | 0 | 5 | 0.6499 |
| strict_hybrid_template_teacherselect | synthetic_test | 0.994874 | 0.993406 | 0.989540 | 1.000000 | 0.979253 | 5 | 0 | 5 | 0.6313 |
| strict_hybrid_template_teacherselect_badcal | synthetic_test | 0.994874 | 0.993406 | 0.989540 | 1.000000 | 0.979253 | 5 | 0 | 5 | nan |
| strict_hybrid_template_teacherselect | original_test_all_10s+ | 0.808305 | 0.760921 | 0.860714 | 0.789652 | 0.545012 | 507 | 881 | 50 | nan |
| strict_hybrid_template_teacherselect_badcal | original_test_all_10s+ | 0.810664 | 0.766581 | 0.860714 | 0.788748 | 0.603406 | 507 | 849 | 44 | nan |
| strict_hybrid_template_teacherselect | original_all_10s+ | 0.780708 | 0.819564 | 0.657220 | 0.893583 | 0.951939 | 5840 | 1075 | 116 | nan |
| strict_hybrid_template_teacherselect_badcal | original_all_10s+ | 0.781132 | 0.819371 | 0.657103 | 0.892642 | 0.956859 | 5840 | 1043 | 109 | nan |
| strict_hybrid_template_teacherselect | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 | nan |
| strict_hybrid_template_teacherselect_badcal | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 | nan |
| strict_hybrid_template_teacherselect | bad_outlier_stress | 0.359589 | 0.176322 | 0.000000 | 0.000000 | 0.359589 | 0 | 0 | 50 | nan |
| strict_hybrid_template_teacherselect_badcal | bad_outlier_stress | 0.441781 | 0.204276 | 0.000000 | 0.000000 | 0.441781 | 0 | 0 | 44 | nan |
| strict_qrsformula_template_pretrain_lowaug | synthetic_val | 0.994764 | 0.994174 | 1.000000 | 0.994318 | 0.988281 | 0 | 6 | 3 | 0.6619 |
| strict_qrsformula_template_pretrain_lowaug | synthetic_test | 0.990774 | 0.990316 | 1.000000 | 0.987825 | 0.987552 | 0 | 15 | 3 | 0.6358 |
| strict_qrsformula_template_pretrain_lowaug_badcal | synthetic_test | 0.990774 | 0.990316 | 1.000000 | 0.987825 | 0.987552 | 0 | 15 | 3 | nan |
| strict_qrsformula_template_pretrain_lowaug | original_test_all_10s+ | 0.766427 | 0.678162 | 0.913736 | 0.673068 | 0.467153 | 295 | 1257 | 23 | nan |
| strict_qrsformula_template_pretrain_lowaug_badcal | original_test_all_10s+ | 0.764893 | 0.677023 | 0.911538 | 0.669679 | 0.491484 | 295 | 1244 | 22 | nan |
| strict_qrsformula_template_pretrain_lowaug | original_all_10s+ | 0.828954 | 0.849436 | 0.799507 | 0.817934 | 0.946074 | 3382 | 1719 | 88 | nan |
| strict_qrsformula_template_pretrain_lowaug_badcal | original_all_10s+ | 0.828195 | 0.847886 | 0.798803 | 0.815581 | 0.948344 | 3380 | 1704 | 85 | nan |
| strict_qrsformula_template_pretrain_lowaug | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 | nan |
| strict_qrsformula_template_pretrain_lowaug_badcal | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 | nan |
| strict_qrsformula_template_pretrain_lowaug | bad_outlier_stress | 0.250000 | 0.133333 | 0.000000 | 0.000000 | 0.250000 | 0 | 0 | 23 | nan |
| strict_qrsformula_template_pretrain_lowaug_badcal | bad_outlier_stress | 0.284247 | 0.147556 | 0.000000 | 0.000000 | 0.284247 | 0 | 0 | 22 | nan |
| strict_qrsformula_learnablecore_domainrand | synthetic_val | 0.995928 | 0.994536 | 1.000000 | 0.999053 | 0.976562 | 0 | 1 | 6 | 0.6812 |
| strict_qrsformula_learnablecore_domainrand | synthetic_test | 0.996925 | 0.995347 | 0.997908 | 1.000000 | 0.979253 | 1 | 0 | 5 | 0.6310 |
| strict_qrsformula_learnablecore_domainrand_badcal | synthetic_test | 0.996925 | 0.995347 | 0.997908 | 1.000000 | 0.979253 | 1 | 0 | 5 | nan |
| strict_qrsformula_learnablecore_domainrand | original_test_all_10s+ | 0.775982 | 0.619261 | 0.906044 | 0.726164 | 0.160584 | 342 | 1212 | 100 | nan |
| strict_qrsformula_learnablecore_domainrand_badcal | original_test_all_10s+ | 0.781762 | 0.677490 | 0.906044 | 0.724808 | 0.294404 | 342 | 1208 | 46 | nan |
| strict_qrsformula_learnablecore_domainrand | original_all_10s+ | 0.779858 | 0.816580 | 0.681922 | 0.866673 | 0.921097 | 5421 | 1417 | 171 | nan |
| strict_qrsformula_learnablecore_domainrand_badcal | original_all_10s+ | 0.781102 | 0.818086 | 0.681864 | 0.864791 | 0.932829 | 5419 | 1411 | 110 | nan |
| strict_qrsformula_learnablecore_domainrand | bad_core_nearboundary | 0.554622 | 0.237838 | 0.000000 | 0.000000 | 0.554622 | 0 | 0 | 53 | nan |
| strict_qrsformula_learnablecore_domainrand_badcal | bad_core_nearboundary | 0.991597 | 0.331927 | 0.000000 | 0.000000 | 0.991597 | 0 | 0 | 1 | nan |
| strict_qrsformula_learnablecore_domainrand | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 47 | nan |
| strict_qrsformula_learnablecore_domainrand_badcal | bad_outlier_stress | 0.010274 | 0.006780 | 0.000000 | 0.000000 | 0.010274 | 0 | 0 | 45 | nan |

## Checkpoints

- `strict_recoverable_localmorph_patch50` best_epoch=7, synthetic_test_acc=0.992824, core_mae=0.6398, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\waveform_geometry_student\N17043_gm_probe\search\strict_recoverable_localmorph_patch50\ckpt_best.pt`
- `strict_qrspeak_critical_lowaug` best_epoch=5, synthetic_test_acc=0.995900, core_mae=0.6312, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\waveform_geometry_student\N17043_gm_probe\search\strict_qrspeak_critical_lowaug\ckpt_best.pt`
- `strict_qrspeak_bin_widepeak` best_epoch=6, synthetic_test_acc=0.995387, core_mae=0.6310, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\waveform_geometry_student\N17043_gm_probe\search\strict_qrspeak_bin_widepeak\ckpt_best.pt`
- `strict_hybrid_template_teacherselect` best_epoch=3, synthetic_test_acc=0.994874, core_mae=0.6313, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\waveform_geometry_student\N17043_gm_probe\search\strict_hybrid_template_teacherselect\ckpt_best.pt`
- `strict_qrsformula_template_pretrain_lowaug` best_epoch=6, synthetic_test_acc=0.990774, core_mae=0.6358, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\waveform_geometry_student\N17043_gm_probe\search\strict_qrsformula_template_pretrain_lowaug\ckpt_best.pt`
- `strict_qrsformula_learnablecore_domainrand` best_epoch=6, synthetic_test_acc=0.996925, core_mae=0.6310, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\waveform_geometry_student\N17043_gm_probe\search\strict_qrsformula_learnablecore_domainrand\ckpt_best.pt`

## Read

- High synthetic accuracy with stagnant teacher-core MAE means the encoder is likely learning synthetic class shortcuts rather than the transferable geometry map.
- Original test remains the external report-only check.
