# Medium-Cluster Real Noise Training Summary

Reference qrs balanced best raw: macro-F1 0.6997, recalls [0.864, 0.436, 0.844].
Reference h_bad_rescue_05 original BUT: macro-F1 0.7454, recalls [0.887, 0.773, 0.793].

| rank | mode | variant | return | bal cal macro | bal raw macro | raw recalls G/M/B | cal recalls G/M/B | orig macro | PTB acc | PTB bad |
| --- | --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: |
| 1 | quick | `blockwise_snr06_triangular__m07_medium_raw_logit_probe__cw1p00_1p75_1p55` | 0 | 0.6458 | 0.5439 | 0.793/0.119/0.925 | 0.696/0.445/0.813 | 0.5296 | 0.9019 | 0.9864 |
| 2 | full | `blockwise_snr06_triangular__m01_medium_detail_soft__cw1p00_1p75_1p55` | 0 | 0.6449 | 0.5697 | 0.869/0.131/0.925 | 0.813/0.282/0.946 | 0.4990 | 0.9301 | 0.9891 |
| 3 | quick | `blockwise_medium_wide_uniform__m03_medium_tst_heavy__cw1p00_1p55_1p70` | 0 | 0.6414 | 0.5458 | 0.849/0.127/0.849 | 0.779/0.523/0.613 | 0.5565 | 0.9619 | 0.9905 |
| 4 | quick | `blockwise_snr06_triangular__m07_medium_raw_logit_probe__cw1p00_1p55_1p70` | 0 | 0.6401 | 0.5519 | 0.771/0.117/0.978 | 0.693/0.392/0.866 | 0.5114 | 0.9083 | 0.9959 |
| 5 | quick | `blockwise_snr06_triangular__m01_medium_detail_soft__cw1p00_1p75_1p55` | 0 | 0.6178 | 0.6377 | 0.873/0.248/0.920 | 0.813/0.219/0.959 | 0.4646 | 0.9110 | 0.9946 |
| 6 | quick | `blockwise_medium_wide_uniform__m03_medium_tst_heavy__cw1p00_1p75_1p55` | 0 | 0.6372 | 0.5969 | 0.842/0.190/0.912 | 0.810/0.328/0.835 | 0.5115 | 0.9578 | 0.9986 |
| 7 | quick | `blockwise_snr06_triangular__m08_medium_boundary_flat__cw1p00_1p55_1p70` | 0 | 0.6311 | 0.6033 | 0.796/0.204/0.954 | 0.749/0.265/0.976 | 0.4947 | 0.9137 | 0.9918 |
| 8 | quick | `blockwise_snr06_triangular__m01_medium_detail_soft__cw1p00_1p90_1p45` | 0 | 0.6051 | 0.6272 | 0.800/0.238/0.973 | 0.737/0.217/0.990 | 0.4430 | 0.9028 | 0.9959 |
| 9 | full | `blockwise_snr06_triangular__m07_medium_raw_logit_probe__cw1p00_1p55_1p70` | 0 | 0.6247 | 0.5466 | 0.866/0.092/0.927 | 0.847/0.307/0.796 | 0.4858 | 0.9342 | 0.9837 |
| 10 | quick | `blockwise_snr06_triangular__m05_medium_good_overlap__cw1p00_1p55_1p70` | 0 | 0.6184 | 0.6131 | 0.793/0.268/0.881 | 0.725/0.273/0.954 | 0.4791 | 0.9087 | 0.9823 |
| 11 | full | `blockwise_medium_wide_uniform__m03_medium_tst_heavy__cw1p00_1p75_1p55` | 0 | 0.6134 | 0.5817 | 0.942/0.131/0.895 | 0.891/0.204/0.900 | 0.4646 | 0.9796 | 0.9973 |
| 12 | quick | `blockwise_snr06_triangular__m04_medium_qrs_clean_extreme__cw1p00_1p75_1p55` | 0 | 0.6120 | 0.5647 | 0.849/0.144/0.893 | 0.805/0.251/0.886 | 0.4771 | 0.9387 | 0.9918 |
| 13 | quick | `blockwise_snr06_triangular__m05_medium_good_overlap__cw1p00_1p75_1p55` | 0 | 0.6064 | 0.6001 | 0.798/0.209/0.929 | 0.735/0.238/0.951 | 0.4705 | 0.9033 | 0.9959 |
| 14 | quick | `blockwise_snr06_triangular__m04_medium_qrs_clean_extreme__cw1p00_1p55_1p70` | 0 | 0.6061 | 0.5493 | 0.832/0.146/0.852 | 0.749/0.367/0.723 | 0.5104 | 0.9223 | 0.9768 |
| 15 | quick | `blockwise_snr06_triangular__m08_medium_boundary_flat__cw1p00_1p75_1p55` | 0 | 0.6047 | 0.5787 | 0.803/0.180/0.908 | 0.766/0.217/0.961 | 0.4532 | 0.9169 | 0.9877 |
| 16 | full | `blockwise_medium_wide_uniform__m03_medium_tst_heavy__cw1p00_1p55_1p70` | 0 | 0.6014 | 0.5253 | 0.959/0.083/0.771 | 0.937/0.229/0.764 | 0.4735 | 0.9787 | 0.9891 |
| 17 | full | `blockwise_snr06_triangular__m07_medium_raw_logit_probe__cw1p00_1p75_1p55` | 0 | 0.5994 | 0.5431 | 0.932/0.122/0.762 | 0.881/0.394/0.552 | 0.5332 | 0.9378 | 0.9741 |
| 18 | quick | `blockwise_snr06_triangular__m07_medium_raw_logit_probe__cw1p00_1p90_1p45` | 0 | 0.5944 | 0.5877 | 0.805/0.251/0.805 | 0.781/0.265/0.825 | 0.4634 | 0.8810 | 0.9128 |
| 19 | quick | `blockwise_medium_wide_uniform__m03_medium_tst_heavy__cw1p00_1p90_1p45` | 0 | 0.5792 | 0.5380 | 0.793/0.114/0.910 | 0.730/0.202/0.927 | 0.4353 | 0.9628 | 0.9986 |
| 20 | quick | `blockwise_snr06_triangular__m08_medium_boundary_flat__cw1p00_1p90_1p45` | 0 | 0.5685 | 0.5712 | 0.720/0.173/0.976 | 0.572/0.238/0.995 | 0.4238 | 0.9015 | 0.9877 |
| 21 | quick | `blockwise_snr06_triangular__m04_medium_qrs_clean_extreme__cw1p00_1p90_1p45` | 0 | 0.5543 | 0.5538 | 0.844/0.127/0.883 | 0.783/0.117/0.964 | 0.4057 | 0.9351 | 0.9932 |
| 22 | quick | `blockwise_snr06_triangular__m01_medium_detail_soft__cw1p00_1p55_1p70` | 0 | 0.5493 | 0.5446 | 0.788/0.129/0.905 | 0.669/0.158/0.961 | 0.4014 | 0.9028 | 0.9973 |
| 23 | quick | `blockwise_snr06_triangular__m05_medium_good_overlap__cw1p00_1p90_1p45` | 0 | 0.5408 | 0.5489 | 0.774/0.112/0.981 | 0.611/0.165/0.995 | 0.3905 | 0.9146 | 0.9973 |
| 24 | full | `blockwise_snr06_triangular__m08_medium_boundary_flat__cw1p00_1p55_1p70` | 0 | 0.5191 | 0.5193 | 0.854/0.046/0.966 | 0.825/0.049/0.983 | 0.3526 | 0.9355 | 0.9850 |
