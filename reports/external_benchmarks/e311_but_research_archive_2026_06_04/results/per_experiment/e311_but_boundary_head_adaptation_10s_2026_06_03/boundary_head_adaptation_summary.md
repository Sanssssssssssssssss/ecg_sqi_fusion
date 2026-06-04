# BUT 10s Boundary Head-Only Adaptation

Frozen Uformer features, BUT train/val-only head fitting, and validation-only calibration.  This is not zero-shot; it tests whether the representation can support the BUT bad boundary once the final boundary is learned on BUT.

| rank | checkpoint | features | model | calibration | acc | bal | macro-F1 | recalls good/medium/bad |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |
| 1 | full_b10_wearable | full_plus_handcrafted | mlp_128_64 | balanced_primary | 0.8058 | 0.8541 | 0.7402 | 0.965/0.663/0.934 |
| 2 | full_b10_wearable | full_plus_handcrafted | mlp_128_64 | acc_primary | 0.7950 | 0.8479 | 0.7315 | 0.977/0.632/0.934 |
| 3 | full_b10_wearable | full_plus_handcrafted | mlp_128_64 | macro_primary | 0.7950 | 0.8479 | 0.7315 | 0.977/0.632/0.934 |
| 4 | full_b10_wearable | full_plus_handcrafted | mlp_128_64 | bad60_acc | 0.7950 | 0.8479 | 0.7315 | 0.977/0.632/0.934 |
| 5 | full_b10_wearable | full_plus_handcrafted | mlp_128_64 | bad70_acc | 0.7950 | 0.8479 | 0.7315 | 0.977/0.632/0.934 |
| 6 | full_b10_wearable | full_plus_handcrafted | mlp_128_64 | bad80_acc | 0.7950 | 0.8479 | 0.7315 | 0.977/0.632/0.934 |
| 7 | full_b10_wearable | full_plus_handcrafted | mlp_128_64 | bad85_acc | 0.7950 | 0.8479 | 0.7315 | 0.977/0.632/0.934 |
| 8 | full_r08_bad_prior_mild | bottleneck_only | mlp_128_64 | balanced_primary | 0.8446 | 0.8433 | 0.7743 | 0.910/0.792/0.827 |
| 9 | full_r08_bad_prior_mild | bottleneck_only | mlp_128_64 | macro_primary | 0.8387 | 0.8408 | 0.7698 | 0.932/0.763/0.827 |
| 10 | full_r08_bad_prior_mild | full_plus_handcrafted | mlp_128_64 | balanced_primary | 0.7793 | 0.8398 | 0.6884 | 0.935/0.636/0.949 |
| 11 | full_r08_bad_prior_mild | full_plus_summary | logreg_c1 | balanced_primary | 0.8466 | 0.8386 | 0.7930 | 0.935/0.778/0.803 |
| 12 | full_b10_wearable | bottleneck_plus_handcrafted | mlp_128_64 | balanced_primary | 0.8371 | 0.8384 | 0.7727 | 0.909/0.779/0.827 |
| 13 | full_r08_bad_prior_mild | full_plus_summary | logreg_c3 | balanced_primary | 0.8474 | 0.8381 | 0.7954 | 0.930/0.784/0.800 |
| 14 | full_r08_bad_prior_mild | full_tokens | logreg_c1 | balanced_primary | 0.8399 | 0.8380 | 0.7793 | 0.934/0.765/0.815 |
| 15 | quick_b10_wearable | full_tokens | mlp_192_96 | balanced_primary | 0.8102 | 0.8376 | 0.7504 | 0.975/0.669/0.869 |
| 16 | full_r08_bad_prior_mild | full_plus_handcrafted | logreg_c3 | balanced_primary | 0.8404 | 0.8364 | 0.7818 | 0.926/0.772/0.810 |
| 17 | full_r08_bad_prior_mild | full_tokens | logreg_c3 | balanced_primary | 0.8471 | 0.8358 | 0.7969 | 0.931/0.783/0.793 |
| 18 | full_b10_wearable | bottleneck_only | mlp_192_96 | balanced_primary | 0.8019 | 0.8326 | 0.7212 | 0.980/0.650/0.869 |
| 19 | full_b10_wearable | bottleneck_only | mlp_128_64 | balanced_primary | 0.7678 | 0.8324 | 0.6774 | 0.933/0.615/0.949 |
| 20 | full_r08_bad_prior_mild | full_plus_handcrafted | mlp_128_64 | acc_primary | 0.7652 | 0.8316 | 0.6767 | 0.948/0.598/0.949 |
| 21 | full_r08_bad_prior_mild | full_plus_handcrafted | mlp_128_64 | macro_primary | 0.7652 | 0.8316 | 0.6767 | 0.948/0.598/0.949 |
| 22 | full_r08_bad_prior_mild | full_plus_handcrafted | mlp_128_64 | bad60_acc | 0.7652 | 0.8316 | 0.6767 | 0.948/0.598/0.949 |
| 23 | full_r08_bad_prior_mild | full_plus_handcrafted | mlp_128_64 | bad70_acc | 0.7652 | 0.8316 | 0.6767 | 0.948/0.598/0.949 |
| 24 | full_r08_bad_prior_mild | full_plus_handcrafted | mlp_128_64 | bad80_acc | 0.7652 | 0.8316 | 0.6767 | 0.948/0.598/0.949 |
| 25 | full_r08_bad_prior_mild | full_plus_handcrafted | mlp_128_64 | bad85_acc | 0.7652 | 0.8316 | 0.6767 | 0.948/0.598/0.949 |
| 26 | full_b10_wearable | full_plus_handcrafted | mlp_128_64 | bias_acc | 0.8184 | 0.8308 | 0.7863 | 0.982/0.683/0.827 |
| 27 | full_b10_wearable | full_plus_handcrafted | mlp_128_64 | bias_macro | 0.8184 | 0.8308 | 0.7863 | 0.982/0.683/0.827 |
| 28 | full_b10_wearable | full_plus_handcrafted | mlp_128_64 | bias_bad70_acc | 0.8184 | 0.8308 | 0.7863 | 0.982/0.683/0.827 |
| 29 | full_r08_bad_prior_mild | full_plus_handcrafted | logreg_c1 | balanced_primary | 0.8474 | 0.8300 | 0.7920 | 0.930/0.786/0.774 |
| 30 | full_r08_bad_prior_mild | full_plus_handcrafted | mlp_192_96 | balanced_primary | 0.8061 | 0.8268 | 0.7398 | 0.912/0.715/0.854 |
| 31 | quick_b10_wearable | full_plus_summary | mlp_128_64 | balanced_primary | 0.8236 | 0.8239 | 0.7771 | 0.958/0.715/0.798 |
| 32 | full_b10_wearable | full_plus_handcrafted | mlp_128_64 | raw_argmax | 0.8201 | 0.8238 | 0.7909 | 0.982/0.689/0.800 |
| 33 | quick_b10_wearable | full_tokens | mlp_192_96 | raw_argmax | 0.7995 | 0.8235 | 0.7538 | 0.989/0.639/0.842 |
| 34 | quick_b10_wearable | full_tokens | mlp_192_96 | acc_primary | 0.7859 | 0.8231 | 0.7314 | 0.992/0.609/0.869 |
| 35 | quick_b10_wearable | full_tokens | mlp_192_96 | macro_primary | 0.7859 | 0.8231 | 0.7314 | 0.992/0.609/0.869 |
| 36 | quick_b10_wearable | full_tokens | mlp_192_96 | bad60_acc | 0.7859 | 0.8231 | 0.7314 | 0.992/0.609/0.869 |
| 37 | quick_b10_wearable | full_tokens | mlp_192_96 | bad70_acc | 0.7859 | 0.8231 | 0.7314 | 0.992/0.609/0.869 |
| 38 | quick_b10_wearable | full_tokens | mlp_192_96 | bad80_acc | 0.7859 | 0.8231 | 0.7314 | 0.992/0.609/0.869 |
| 39 | quick_b10_wearable | full_tokens | mlp_192_96 | bad85_acc | 0.7859 | 0.8231 | 0.7314 | 0.992/0.609/0.869 |
| 40 | full_b10_wearable | bottleneck_plus_handcrafted | mlp_128_64 | acc_primary | 0.8069 | 0.8230 | 0.7495 | 0.976/0.666/0.827 |

## Decision Notes

- If head-only adaptation reaches high bad recall with balanced/macro-F1 improvement, the frozen representation is useful and the remaining issue is boundary/domain calibration.
- If it cannot recover bad, the representation itself lacks BUT bad cues and generator/denoiser data design must change more deeply.
