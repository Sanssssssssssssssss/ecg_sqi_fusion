# Real NSTDB Noise SNR Model Validation

Rule-based reference `h_bad_rescue_05`: acc `0.8229`, balanced `0.8177`, macro-F1 `0.7454`.

| rank | mode | variant | return | BUT acc | BUT bal | BUT macro | recalls G/M/B | PTB acc | PTB bad | distance score |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| 1 | quick | `blockwise_medium_wide_uniform__m03_medium_tst_heavy__cw1p00_1p55_1p70` | 0 | 0.6534 | 0.6497 | 0.5565 | 0.797/0.539/0.613 | 0.9619 | 0.9905 | 0.4372 |
| 2 | full | `blockwise_snr06_triangular__m07_medium_raw_logit_probe__cw1p00_1p75_1p55` | 0 | 0.6233 | 0.6177 | 0.5332 | 0.892/0.409/0.552 | 0.9378 | 0.9741 | 0.4364 |
| 3 | quick | `blockwise_snr06_triangular__m07_medium_raw_logit_probe__cw1p00_1p75_1p55` | 0 | 0.6025 | 0.6731 | 0.5296 | 0.722/0.485/0.813 | 0.9019 | 0.9864 | 0.4364 |
| 4 | quick | `blockwise_medium_wide_uniform__m03_medium_tst_heavy__cw1p00_1p75_1p55` | 0 | 0.5889 | 0.6775 | 0.5115 | 0.832/0.366/0.835 | 0.9578 | 0.9986 | 0.4372 |
| 5 | quick | `blockwise_snr06_triangular__m07_medium_raw_logit_probe__cw1p00_1p55_1p70` | 0 | 0.5746 | 0.6713 | 0.5114 | 0.719/0.429/0.866 | 0.9083 | 0.9959 | 0.4364 |
| 6 | quick | `blockwise_snr06_triangular__m04_medium_qrs_clean_extreme__cw1p00_1p55_1p70` | 0 | 0.5895 | 0.6411 | 0.5104 | 0.782/0.419/0.723 | 0.9223 | 0.9768 | 0.4365 |
| 7 | full | `blockwise_snr06_triangular__m01_medium_detail_soft__cw1p00_1p75_1p55` | 0 | 0.5726 | 0.7015 | 0.4990 | 0.841/0.317/0.946 | 0.9301 | 0.9891 | 0.4369 |
| 8 | quick | `blockwise_snr06_triangular__m08_medium_boundary_flat__cw1p00_1p55_1p70` | 0 | 0.5505 | 0.6930 | 0.4947 | 0.786/0.317/0.976 | 0.9137 | 0.9918 | 0.4342 |
| 9 | full | `blockwise_snr06_triangular__m07_medium_raw_logit_probe__cw1p00_1p55_1p70` | 0 | 0.5658 | 0.6526 | 0.4858 | 0.859/0.304/0.796 | 0.9342 | 0.9837 | 0.4364 |
| 10 | quick | `blockwise_snr06_triangular__m05_medium_good_overlap__cw1p00_1p55_1p70` | 0 | 0.5359 | 0.6745 | 0.4791 | 0.742/0.328/0.954 | 0.9087 | 0.9823 | 0.4365 |
| 11 | quick | `blockwise_snr06_triangular__m04_medium_qrs_clean_extreme__cw1p00_1p75_1p55` | 0 | 0.5522 | 0.6699 | 0.4771 | 0.837/0.287/0.886 | 0.9387 | 0.9918 | 0.4365 |
| 12 | full | `blockwise_medium_wide_uniform__m03_medium_tst_heavy__cw1p00_1p55_1p70` | 0 | 0.5686 | 0.6498 | 0.4735 | 0.942/0.243/0.764 | 0.9787 | 0.9891 | 0.4372 |
| 13 | quick | `blockwise_snr06_triangular__m05_medium_good_overlap__cw1p00_1p75_1p55` | 0 | 0.5232 | 0.6663 | 0.4705 | 0.755/0.293/0.951 | 0.9033 | 0.9959 | 0.4365 |
| 14 | full | `blockwise_medium_wide_uniform__m03_medium_tst_heavy__cw1p00_1p75_1p55` | 0 | 0.5507 | 0.6774 | 0.4646 | 0.906/0.226/0.900 | 0.9796 | 0.9973 | 0.4372 |
| 15 | quick | `blockwise_snr06_triangular__m01_medium_detail_soft__cw1p00_1p75_1p55` | 0 | 0.5352 | 0.6811 | 0.4646 | 0.837/0.247/0.959 | 0.9110 | 0.9946 | 0.4369 |
| 16 | quick | `blockwise_snr06_triangular__m07_medium_raw_logit_probe__cw1p00_1p90_1p45` | 0 | 0.5367 | 0.6397 | 0.4634 | 0.804/0.290/0.825 | 0.8810 | 0.9128 | 0.4364 |
| 17 | quick | `blockwise_snr06_triangular__m08_medium_boundary_flat__cw1p00_1p75_1p55` | 0 | 0.5132 | 0.6652 | 0.4532 | 0.793/0.242/0.961 | 0.9169 | 0.9877 | 0.4342 |
| 18 | quick | `blockwise_snr06_triangular__m01_medium_detail_soft__cw1p00_1p90_1p45` | 0 | 0.4939 | 0.6589 | 0.4430 | 0.745/0.241/0.990 | 0.9028 | 0.9959 | 0.4369 |
| 19 | quick | `blockwise_medium_wide_uniform__m03_medium_tst_heavy__cw1p00_1p90_1p45` | 0 | 0.4889 | 0.6373 | 0.4353 | 0.758/0.227/0.927 | 0.9628 | 0.9986 | 0.4372 |
| 20 | quick | `blockwise_snr06_triangular__m08_medium_boundary_flat__cw1p00_1p90_1p45` | 0 | 0.4471 | 0.6218 | 0.4238 | 0.599/0.272/0.995 | 0.9015 | 0.9877 | 0.4342 |
| 21 | quick | `blockwise_snr06_triangular__m04_medium_qrs_clean_extreme__cw1p00_1p90_1p45` | 0 | 0.4732 | 0.6420 | 0.4057 | 0.820/0.143/0.964 | 0.9351 | 0.9932 | 0.4365 |
| 22 | quick | `blockwise_snr06_triangular__m01_medium_detail_soft__cw1p00_1p55_1p70` | 0 | 0.4371 | 0.6106 | 0.4014 | 0.693/0.178/0.961 | 0.9028 | 0.9973 | 0.4369 |
| 23 | quick | `blockwise_snr06_triangular__m05_medium_good_overlap__cw1p00_1p90_1p45` | 0 | 0.4218 | 0.6079 | 0.3905 | 0.637/0.192/0.995 | 0.9146 | 0.9973 | 0.4365 |
| 24 | full | `blockwise_snr06_triangular__m08_medium_boundary_flat__cw1p00_1p55_1p70` | 0 | 0.4452 | 0.6319 | 0.3526 | 0.852/0.061/0.983 | 0.9355 | 0.9850 | 0.4342 |
