# Current Waveform Coverage Gap

Candidate: `featurefirst_top20_qrsbase_artbad_dualcoreout_recall_a050`

This is report-only analysis of original BUT rows. It is not used for training or selection.

## Bucket counts

| bucket | true | pred_badcal | n |
| --- | --- | --- | --- |
| bad_to_good | bad | good | 102 |
| bad_to_medium | bad | medium | 48 |
| correct | bad | bad | 261 |
| correct | good | good | 3238 |
| correct | medium | medium | 3826 |
| good_to_medium | good | medium | 397 |
| medium_to_good | medium | good | 424 |
| nonbad_to_bad | good | bad | 5 |
| nonbad_to_bad | medium | bad | 176 |

## Row support summary

| bucket | true | n | oos01_mean | oos05_mean |
| --- | --- | --- | --- | --- |
| bad_to_good | bad | 102 | 30.8529 | 42.6176 |
| bad_to_medium | bad | 48 | 26.5625 | 40.1667 |
| correct | bad | 261 | 21.3640 | 37.6820 |
| correct | good | 3238 | 4.0766 | 8.7551 |
| correct | medium | 3826 | 1.1158 | 4.9179 |
| good_to_medium | good | 397 | 4.0705 | 12.1990 |
| medium_to_good | medium | 424 | 3.4552 | 11.6132 |
| nonbad_to_bad | good | 5 | 24.0000 | 31.4000 |
| nonbad_to_bad | medium | 176 | 7.0170 | 18.0284 |

## Same-class PTB synthetic nearest-neighbor distance

| bucket | true | n | nn_median | nn_p90 |
| --- | --- | --- | --- | --- |
| bad_to_good | bad | 102 | 686131.9062 | 814724.0000 |
| bad_to_medium | bad | 48 | 657278.8438 | 945449.2500 |
| correct | bad | 261 | 550877.1250 | 975047.9375 |
| correct | good | 3238 | 5.5757 | 9.3905 |
| correct | medium | 3826 | 1.9044 | 3.6652 |
| good_to_medium | good | 397 | 6.8600 | 8.8051 |
| medium_to_good | medium | 424 | 1.9846 | 1441.1898 |
| nonbad_to_bad | good | 5 | 128.3619 | 100000.0391 |
| nonbad_to_bad | medium | 176 | 4.3275 | 2882.3008 |

## Top feature support gaps by error bucket

| bucket | true | feature | n | oos_01_99_rate | oos_05_95_rate | delta_vs_synth_iqr | delta_vs_correct_iqr | below_q01_rate | above_q99_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bad_to_good | bad | fatal_or_score | 102 | 0.6176 | 0.6176 | -104323.4588 | -104323.4588 | 0.6176 | 0.0000 |
| bad_to_good | bad | sqi_bSQI | 102 | 0.0196 | 0.9706 | 500000.0000 | 45454.5455 | 0.0000 | 0.0196 |
| bad_to_good | bad | sqi_basSQI | 102 | 1.0000 | 1.0000 | -2469.8412 | -876.0186 | 1.0000 | 0.0000 |
| bad_to_good | bad | hjorth_complexity | 102 | 1.0000 | 1.0000 | 294.7529 | 136.6710 | 0.0000 | 1.0000 |
| bad_to_good | bad | flatline_ratio | 102 | 1.0000 | 1.0000 | 136.8750 | 55.3750 | 0.0000 | 1.0000 |
| bad_to_good | bad | pc2 | 102 | 1.0000 | 1.0000 | 74.1058 | 26.4324 | 0.0000 | 1.0000 |
| bad_to_good | bad | sqi_pSQI | 102 | 0.9118 | 1.0000 | 89.7446 | 22.4806 | 0.0000 | 0.9118 |
| bad_to_good | bad | higuchi_fd_proxy | 102 | 0.8137 | 1.0000 | -78.9197 | -21.8671 | 0.8137 | 0.0000 |
| bad_to_medium | bad | sqi_bSQI | 48 | 0.0000 | 0.9375 | 447222.2222 | -7323.2323 | 0.0000 | 0.0000 |
| bad_to_medium | bad | sqi_basSQI | 48 | 0.9792 | 1.0000 | -2459.6907 | -865.8681 | 0.9792 | 0.0000 |
| bad_to_medium | bad | hjorth_complexity | 48 | 0.9167 | 1.0000 | 302.9632 | 144.8814 | 0.0000 | 0.9167 |
| bad_to_medium | bad | pc2 | 48 | 0.9167 | 1.0000 | 70.2366 | 22.5633 | 0.0000 | 0.9167 |
| bad_to_medium | bad | baseline_step | 48 | 0.9583 | 1.0000 | 92.8278 | 21.2946 | 0.0000 | 0.9583 |
| bad_to_medium | bad | sqi_pSQI | 48 | 0.7292 | 1.0000 | 80.9832 | 13.7192 | 0.0000 | 0.7292 |
| bad_to_medium | bad | hjorth_mobility | 48 | 0.9167 | 0.9792 | -62.6103 | -10.7759 | 0.9167 | 0.0000 |
| bad_to_medium | bad | qrs_band_ratio | 48 | 1.0000 | 1.0000 | -36.7794 | -7.4465 | 1.0000 | 0.0000 |
| good_to_medium | good | sample_entropy_proxy | 397 | 0.4055 | 0.6650 | 1.9594 | 3.9241 | 0.0504 | 0.3552 |
| good_to_medium | good | qrs_prom_p90 | 397 | 0.1537 | 0.2846 | -4.1442 | -7.5216 | 0.0076 | 0.1461 |
| good_to_medium | good | knn_label_purity | 397 | 0.2796 | 0.5693 | -4.7500 | -4.7500 | 0.2796 | 0.0000 |
| good_to_medium | good | sqi_pSQI | 397 | 0.4836 | 0.5315 | 0.7682 | 1.6698 | 0.0000 | 0.4836 |
| good_to_medium | good | sqi_kSQI | 397 | 0.1285 | 0.4458 | -3.4754 | -5.6277 | 0.0126 | 0.1159 |
| good_to_medium | good | sqi_sSQI | 397 | 0.0831 | 0.5819 | -4.7477 | -5.5359 | 0.0076 | 0.0756 |
| good_to_medium | good | boundary_confidence | 397 | 0.3451 | 0.5894 | -1.1156 | -1.0540 | 0.3451 | 0.0000 |
| good_to_medium | good | region_confidence | 397 | 0.3451 | 0.5894 | -1.1459 | -0.2489 | 0.3451 | 0.0000 |
| medium_to_bad | medium | hjorth_complexity | 176 | 0.2784 | 0.6364 | 11.3590 | 10.7707 | 0.0000 | 0.2784 |
| medium_to_bad | medium | flatline_ratio | 176 | 0.3352 | 0.5909 | 4.6442 | 4.6154 | 0.0000 | 0.3352 |
| medium_to_bad | medium | sqi_fSQI | 176 | 0.2159 | 0.5682 | 4.6000 | 4.6000 | 0.0000 | 0.2159 |
| medium_to_bad | medium | sqi_basSQI | 176 | 0.2102 | 0.5909 | -5.5345 | -4.3088 | 0.2102 | 0.0000 |
| medium_to_bad | medium | pc1 | 176 | 0.2614 | 0.5852 | -2.3156 | -2.4318 | 0.2614 | 0.0000 |
| medium_to_bad | medium | pc2 | 176 | 0.2614 | 0.6420 | 2.7051 | 2.0367 | 0.0000 | 0.2614 |
| medium_to_bad | medium | sqi_bSQI | 176 | 0.0000 | 0.0000 | -7.8889 | -7.6389 | 0.0000 | 0.0000 |
| medium_to_bad | medium | qrs_prom_p90 | 176 | 0.2727 | 0.5966 | -1.4258 | -1.6328 | 0.2727 | 0.0000 |
| medium_to_good | medium | fatal_or_score | 424 | 0.2028 | 0.4481 | -150000.0000 | -150000.0000 | 0.2028 | 0.0000 |
| medium_to_good | medium | higuchi_fd_proxy | 424 | 0.1156 | 0.5755 | -1.4474 | -2.3139 | 0.1156 | 0.0000 |
| medium_to_good | medium | boundary_confidence | 424 | 0.1745 | 0.4623 | -1.6079 | -1.5738 | 0.1745 | 0.0000 |
| medium_to_good | medium | pc4 | 424 | 0.1486 | 0.4741 | 1.0839 | 1.1782 | 0.0000 | 0.1486 |
| medium_to_good | medium | region_confidence | 424 | 0.1745 | 0.4623 | -0.9397 | -0.8841 | 0.1745 | 0.0000 |
| medium_to_good | medium | pc1 | 424 | 0.1156 | 0.3090 | -1.8552 | -1.9714 | 0.1156 | 0.0000 |
| medium_to_good | medium | sqi_pSQI | 424 | 0.0472 | 0.4080 | 1.6453 | 2.3780 | 0.0000 | 0.0472 |
| medium_to_good | medium | band_30_45 | 424 | 0.0896 | 0.3467 | -0.9278 | -2.0138 | 0.0896 | 0.0000 |
| nonbad_to_bad | good | contact_loss_win_ratio | 5 | 1.0000 | 1.0000 | 200000.0000 | 200000.0000 | 0.0000 | 1.0000 |
| nonbad_to_bad | good | hjorth_complexity | 5 | 1.0000 | 1.0000 | 91.6869 | 89.8221 | 0.0000 | 1.0000 |
| nonbad_to_bad | good | sqi_bSQI | 5 | 1.0000 | 1.0000 | -13.7879 | -12.6389 | 1.0000 | 0.0000 |
| nonbad_to_bad | good | qrs_prom_p90 | 5 | 1.0000 | 1.0000 | -9.2368 | -12.6142 | 1.0000 | 0.0000 |
| nonbad_to_bad | good | sqi_basSQI | 5 | 0.6000 | 1.0000 | -18.2098 | -12.8699 | 0.6000 | 0.0000 |
| nonbad_to_bad | good | hjorth_mobility | 5 | 1.0000 | 1.0000 | -7.9823 | -6.9070 | 1.0000 | 0.0000 |
| nonbad_to_bad | good | pc2 | 5 | 1.0000 | 1.0000 | 6.8198 | 4.8103 | 0.0000 | 1.0000 |
| nonbad_to_bad | good | qrs_band_ratio | 5 | 1.0000 | 1.0000 | -5.6034 | -4.2116 | 1.0000 | 0.0000 |
