# Waveform Endpoint Router Probe

Report-only probe. Router training uses PTB synthetic endpoint probabilities only; Original BUT is held out.

| Candidate | Bucket | Acc | Macro-F1 | Good | Medium | Bad |
|---|---|---:|---:|---:|---:|---:|
| prob_mean | synthetic_test | 0.995900 | 0.994379 | 0.996 | 0.999 | 0.979 |
| prob_mean | original_test_all_10s+ | 0.817388 | 0.697796 | 0.903 | 0.795 | 0.302 |
| prob_mean | bad_core_nearboundary | 0.991597 | 0.331927 | 0.000 | 0.000 | 0.992 |
| prob_mean | bad_outlier_stress | 0.020548 | 0.013423 | 0.000 | 0.000 | 0.021 |
| router_logreg_balanced | synthetic_test | 0.994362 | 0.991933 | 0.996 | 0.996 | 0.983 |
| router_logreg_balanced | original_test_all_10s+ | 0.794503 | 0.662227 | 0.903 | 0.750 | 0.311 |
| router_logreg_balanced | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000 | 0.000 | 1.000 |
| router_logreg_balanced | bad_outlier_stress | 0.030822 | 0.019934 | 0.000 | 0.000 | 0.031 |
| router_hgb_lite | synthetic_test | 0.996925 | 0.996055 | 0.996 | 0.999 | 0.988 |
| router_hgb_lite | original_test_all_10s+ | 0.813731 | 0.698553 | 0.897 | 0.794 | 0.292 |
| router_hgb_lite | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000 | 0.000 | 1.000 |
| router_hgb_lite | bad_outlier_stress | 0.003425 | 0.002275 | 0.000 | 0.000 | 0.003 |
| router_extratrees_lite | synthetic_test | 0.996412 | 0.994863 | 0.996 | 1.000 | 0.979 |
| router_extratrees_lite | original_test_all_10s+ | 0.827415 | 0.704419 | 0.904 | 0.813 | 0.304 |
| router_extratrees_lite | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000 | 0.000 | 1.000 |
| router_extratrees_lite | bad_outlier_stress | 0.020548 | 0.013423 | 0.000 | 0.000 | 0.021 |
