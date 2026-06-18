# Original Bad-Stress Split Diagnostic

Report-only. This explains why original test bad remains low even for feature trees trained on original train+val.

## Record Counts

split  record_id   n
 test     111001 292
train     105001 770
train     124001  31
train     100001  22
train     113001   4
  val     114001  82

## Top Feature Gaps

            feature  trainval_median  test_median  delta_test_minus_trainval  trainval_p10  trainval_p90  test_p10  test_p90       ks
     flatline_ratio         0.007206     0.415933                   0.408727      0.004003      0.012010  0.237070  0.647398 0.993399
boundary_confidence         0.463926     0.023359                  -0.440568      0.413010      0.562265  0.009528  0.039589 0.982398
                pc1        10.433906    -4.251713                 -14.685619      9.942942     10.817819 -6.674762 -1.248616 0.967972
                pc2        -0.122371    11.613217                  11.735588     -0.378598      0.302657  6.880951 17.690417 0.963199
      baseline_step         0.026724     1.378621                   1.351897      0.017079      0.049345  0.863527  2.145517 0.959899
   non_qrs_diff_p95         0.375416     0.039336                  -0.336080      0.354362      0.404421  0.016471  0.109798 0.937294
       diff_abs_p95         0.392093     0.072075                  -0.320018      0.379171      0.409912  0.032048  0.160570 0.936194
         pca_margin        10.297380    -6.189709                 -16.487089      9.281843     10.710360 -7.757585 -4.547715 0.935094
         band_30_45         0.105119     0.006896                  -0.098223      0.087758      0.124686  0.001452  0.035662 0.933993
     qrs_band_ratio         0.805752     0.140322                  -0.665430      0.712401      0.825482  0.036586  0.321502 0.931793
         band_15_30         0.830086     0.031042                  -0.799044      0.490406      0.846691  0.006661  0.099197 0.928493
     qrs_visibility         0.243382     0.037808                  -0.205574      0.141626      0.258883  0.007385  0.116211 0.876626
  non_qrs_rms_ratio         0.979824     0.624695                  -0.355129      0.918814      1.029115  0.429083  0.840548 0.869174
           sqi_bSQI         0.000000     0.444444                   0.444444      0.000000      0.045455  0.062917  0.714286 0.831627
        ptp_p99_p01         0.686135     0.877077                   0.190942      0.655397      0.714134  0.664176  1.289905 0.746632

## Interpretation

- Bad stress is record/domain shifted across splits; original train+val bad-stress rows do not fully cover original test bad-stress morphology.
- The main learnable good/medium geometry is now strong; original-test gap is concentrated in bad stress, not in ordinary good/medium.

![PCA](E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\original_bad_stress_split_pca.png)
