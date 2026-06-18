# Waveform Student Ensemble Diagnostic

Fixed equal-weight averages of waveform-only checkpoints. This is an architecture diagnostic for branch complementarity; no BUT row fits weights or thresholds.

| Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ens_local_qrspeak_hybrid | synthetic_test | 0.995900 | 0.994378 | 0.993724 | 1.000000 | 0.979253 | 3 | 0 | 5 |
| ens_local_qrspeak_hybrid | original_test_all_10s+ | 0.816562 | 0.705673 | 0.857143 | 0.830547 | 0.306569 | 520 | 742 | 94 |
| ens_local_qrspeak_hybrid | original_all_10s+ | 0.779797 | 0.818359 | 0.648595 | 0.913813 | 0.933396 | 5989 | 905 | 160 |
| ens_local_qrspeak_hybrid | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| ens_local_qrspeak_hybrid | bad_outlier_stress | 0.023973 | 0.015608 | 0.000000 | 0.000000 | 0.023973 | 0 | 0 | 94 |
| ens_local_hybrid_badstress | synthetic_test | 0.996412 | 0.994863 | 0.995816 | 1.000000 | 0.979253 | 2 | 0 | 5 |
| ens_local_hybrid_badstress | original_test_all_10s+ | 0.815029 | 0.736839 | 0.884341 | 0.794171 | 0.425791 | 419 | 876 | 60 |
| ens_local_hybrid_badstress | original_all_10s+ | 0.811476 | 0.842489 | 0.720120 | 0.892548 | 0.943046 | 4765 | 1101 | 123 |
| ens_local_hybrid_badstress | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| ens_local_hybrid_badstress | bad_outlier_stress | 0.191781 | 0.107280 | 0.000000 | 0.000000 | 0.191781 | 0 | 0 | 60 |
| ens_diverse_four_branch | synthetic_test | 0.996412 | 0.994863 | 0.995816 | 1.000000 | 0.979253 | 2 | 0 | 5 |
| ens_diverse_four_branch | original_test_all_10s+ | 0.815501 | 0.723282 | 0.875549 | 0.807953 | 0.364964 | 453 | 833 | 70 |
| ens_diverse_four_branch | original_all_10s+ | 0.795303 | 0.830340 | 0.684621 | 0.901769 | 0.938127 | 5374 | 1023 | 135 |
| ens_diverse_four_branch | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| ens_diverse_four_branch | bad_outlier_stress | 0.106164 | 0.063983 | 0.000000 | 0.000000 | 0.106164 | 0 | 0 | 70 |
| ens_peak_formula_template | synthetic_test | 0.996412 | 0.994863 | 0.995816 | 1.000000 | 0.979253 | 2 | 0 | 5 |
| ens_peak_formula_template | original_test_all_10s+ | 0.797924 | 0.732363 | 0.888187 | 0.757117 | 0.437956 | 407 | 1051 | 46 |
| ens_peak_formula_template | original_all_10s+ | 0.789173 | 0.825568 | 0.685619 | 0.878528 | 0.943425 | 5357 | 1262 | 113 |
| ens_peak_formula_template | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| ens_peak_formula_template | bad_outlier_stress | 0.208904 | 0.115203 | 0.000000 | 0.000000 | 0.208904 | 0 | 0 | 46 |
| ens_qrs_long_shape_badstress | synthetic_test | 0.995900 | 0.994378 | 0.993724 | 1.000000 | 0.979253 | 3 | 0 | 5 |
| ens_qrs_long_shape_badstress | original_test_all_10s+ | 0.802289 | 0.699926 | 0.848352 | 0.809083 | 0.321168 | 551 | 833 | 73 |
| ens_qrs_long_shape_badstress | original_all_10s+ | 0.760651 | 0.803414 | 0.615737 | 0.906850 | 0.933964 | 6548 | 975 | 142 |
| ens_qrs_long_shape_badstress | bad_core_nearboundary | 0.974790 | 0.329078 | 0.000000 | 0.000000 | 0.974790 | 0 | 0 | 3 |
| ens_qrs_long_shape_badstress | bad_outlier_stress | 0.054795 | 0.034632 | 0.000000 | 0.000000 | 0.054795 | 0 | 0 | 70 |
| ens_bestlocal_qrs_shape | synthetic_test | 0.996412 | 0.994863 | 0.995816 | 1.000000 | 0.979253 | 2 | 0 | 5 |
| ens_bestlocal_qrs_shape | original_test_all_10s+ | 0.811962 | 0.707255 | 0.860440 | 0.817442 | 0.323601 | 507 | 795 | 79 |
| ens_bestlocal_qrs_shape | original_all_10s+ | 0.782012 | 0.819887 | 0.655929 | 0.908261 | 0.934721 | 5861 | 956 | 145 |
| ens_bestlocal_qrs_shape | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| ens_bestlocal_qrs_shape | bad_outlier_stress | 0.047945 | 0.030501 | 0.000000 | 0.000000 | 0.047945 | 0 | 0 | 79 |
| ens_template_badstress_shape | synthetic_test | 0.995900 | 0.994378 | 0.993724 | 1.000000 | 0.979253 | 3 | 0 | 5 |
| ens_template_badstress_shape | original_test_all_10s+ | 0.806653 | 0.710642 | 0.855220 | 0.809986 | 0.340633 | 527 | 831 | 90 |
| ens_template_badstress_shape | original_all_10s+ | 0.762077 | 0.804729 | 0.618260 | 0.906473 | 0.935478 | 6506 | 981 | 159 |
| ens_template_badstress_shape | bad_core_nearboundary | 0.974790 | 0.329078 | 0.000000 | 0.000000 | 0.974790 | 0 | 0 | 3 |
| ens_template_badstress_shape | bad_outlier_stress | 0.082192 | 0.050633 | 0.000000 | 0.000000 | 0.082192 | 0 | 0 | 87 |
| ens_diverse_shape_five | synthetic_test | 0.995900 | 0.994378 | 0.993724 | 1.000000 | 0.979253 | 3 | 0 | 5 |
| ens_diverse_shape_five | original_test_all_10s+ | 0.812788 | 0.712852 | 0.853297 | 0.823543 | 0.338200 | 534 | 767 | 85 |
| ens_diverse_shape_five | original_all_10s+ | 0.774244 | 0.814053 | 0.638679 | 0.911554 | 0.935289 | 6157 | 922 | 154 |
| ens_diverse_shape_five | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| ens_diverse_shape_five | bad_outlier_stress | 0.068493 | 0.042735 | 0.000000 | 0.000000 | 0.068493 | 0 | 0 | 85 |

## Interpretation

- If equal-weight ensembles improve BUT buckets, the next model should be a single waveform-only multi-branch Transformer with a learned synthetic-only gate.
- If they do not improve, the bottleneck is not just branch specialization; it is a PTB-to-BUT waveform-domain mismatch.
