# BUT 10s Morphology-Guided Synthetic Grid

Formal protocol: BUT 10s P1, validation-only calibration, test reporting only.

## Anchor

`b10_all_bad_wearable`: acc 0.7735, balanced 0.8045, macro-F1 0.7238, recalls 0.824/0.724/0.866.

## Hypothesis Verifier Top 20
| rank | mode | seed | spec | family | BUT acc | bal | macro | recalls G/M/B | minMB | PTB acc | PTB bad | morph | feature | mismatch | note |
|---:|---|---:|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---|---|
| 1 | hypothesis | 0 | `h_coexistence_04` | h_coexistence | 0.6664 | 0.7064 | 0.6542 | 0.9926/0.3918/0.7348 | 0.3918 | 0.9796 | 1.0000 | 0.3051 | 0.7662 | False | Hypothesis H3: b10/s02/mix05 coexistence interpolation. |
| 2 | hypothesis | 0 | `h_negative_control_03` | h_negative_control | 0.7133 | 0.7611 | 0.6380 | 0.8387/0.5978/0.8467 | 0.5978 | 0.9791 | 1.0000 | 0.3032 | 0.7674 | False | Negative control: flatline/SNR-heavy bad without morphology-rich medium. |
| 3 | hypothesis | 0 | `h_bad_rescue_01` | h_bad_rescue | 0.6266 | 0.5275 | 0.5243 | 0.9964/0.3597/0.2263 | 0.2263 | 0.9768 | 0.9986 | 0.3075 | 0.7648 | False | Hypothesis H2: class-3 BUT bad is QRS-confounded rather than only low-amplitude. |
| 4 | hypothesis | 0 | `h_bad_rescue_05` | h_bad_rescue | 0.8229 | 0.8177 | 0.7454 | 0.8865/0.7734/0.7932 | 0.7734 | 0.9687 | 1.0000 | 0.3089 | 0.7640 | False | Hypothesis H2: class-3 BUT bad is QRS-confounded rather than only low-amplitude. |
| 5 | hypothesis | 0 | `h_coexistence_03` | h_coexistence | 0.7650 | 0.8310 | 0.6978 | 0.9525/0.5940/0.9465 | 0.5940 | 0.9664 | 1.0000 | 0.3073 | 0.7649 | False | Hypothesis H3: b10/s02/mix05 coexistence interpolation. |
| 6 | hypothesis | 0 | `h_coexistence_07` | h_coexistence | 0.7447 | 0.7730 | 0.6976 | 0.8876/0.6211/0.8102 | 0.6211 | 0.9550 | 0.9986 | 0.3063 | 0.7655 | False | Hypothesis H3: b10/s02/mix05 coexistence interpolation. |
| 7 | hypothesis | 0 | `h_negative_control_02` | h_negative_control | 0.7590 | 0.7228 | 0.6942 | 0.9426/0.6225/0.6034 | 0.6034 | 0.9619 | 1.0000 | 0.3137 | 0.7612 | False | Negative control: flatline/SNR-heavy bad without morphology-rich medium. |
| 8 | hypothesis | 0 | `h_medium_rescue_02` | h_medium_rescue | 0.6913 | 0.7258 | 0.6740 | 0.9533/0.4700/0.7543 | 0.4700 | 0.9691 | 0.9986 | 0.3092 | 0.7638 | False | Hypothesis H1: class-2 BUT medium is QRS-visible but locally unreliable. |
| 9 | hypothesis | 0 | `h_coexistence_02` | h_coexistence | 0.7057 | 0.7841 | 0.6721 | 0.9742/0.4657/0.9124 | 0.4657 | 0.9718 | 1.0000 | 0.3055 | 0.7660 | False | Hypothesis H3: b10/s02/mix05 coexistence interpolation. |
| 10 | hypothesis | 0 | `h_coexistence_05` | h_coexistence | 0.7163 | 0.7585 | 0.6514 | 0.9736/0.4964/0.8054 | 0.4964 | 0.9673 | 1.0000 | 0.3062 | 0.7656 | False | Hypothesis H3: b10/s02/mix05 coexistence interpolation. |
| 11 | hypothesis | 0 | `h_medium_rescue_04` | h_medium_rescue | 0.6621 | 0.7289 | 0.6447 | 0.7602/0.5628/0.8637 | 0.5628 | 0.9714 | 1.0000 | 0.3086 | 0.7642 | False | Hypothesis H1: class-2 BUT medium is QRS-visible but locally unreliable. |
| 12 | hypothesis | 0 | `h_negative_control_04` | h_negative_control | 0.7359 | 0.6161 | 0.6239 | 0.9808/0.5757/0.2920 | 0.2920 | 0.9673 | 0.9905 | 0.3138 | 0.7612 | False | Negative control: flatline/SNR-heavy bad without morphology-rich medium. |
| 13 | hypothesis | 0 | `h_coexistence_01` | h_coexistence | 0.6979 | 0.6724 | 0.6230 | 0.9860/0.4740/0.5572 | 0.4740 | 0.9587 | 1.0000 | 0.3076 | 0.7648 | False | Hypothesis H3: b10/s02/mix05 coexistence interpolation. |
| 14 | hypothesis | 0 | `h_medium_rescue_05` | h_medium_rescue | 0.7192 | 0.6324 | 0.6229 | 0.9626/0.5502/0.3844 | 0.3844 | 0.9614 | 0.9959 | 0.3093 | 0.7638 | False | Hypothesis H1: class-2 BUT medium is QRS-visible but locally unreliable. |
| 15 | hypothesis | 0 | `h_medium_rescue_01` | h_medium_rescue | 0.6743 | 0.6549 | 0.6018 | 0.7953/0.5829/0.5864 | 0.5829 | 0.9691 | 0.9986 | 0.3087 | 0.7641 | False | Hypothesis H1: class-2 BUT medium is QRS-visible but locally unreliable. |
| 16 | hypothesis | 0 | `h_negative_control_01` | h_negative_control | 0.6862 | 0.6756 | 0.5964 | 0.7681/0.6236/0.6350 | 0.6236 | 0.9678 | 1.0000 | 0.3063 | 0.7655 | False | Negative control: flatline/SNR-heavy bad without morphology-rich medium. |
| 17 | hypothesis | 0 | `h_coexistence_06` | h_coexistence | 0.6546 | 0.6487 | 0.5895 | 0.9907/0.3861/0.5693 | 0.3861 | 0.9714 | 0.9986 | 0.3096 | 0.7636 | False | Hypothesis H3: b10/s02/mix05 coexistence interpolation. |
| 18 | hypothesis | 0 | `h_bad_rescue_03` | h_bad_rescue | 0.6495 | 0.5548 | 0.5785 | 0.5533/0.7560/0.3552 | 0.3552 | 0.9632 | 1.0000 | 0.3070 | 0.7651 | False | Hypothesis H2: class-3 BUT bad is QRS-confounded rather than only low-amplitude. |
| 19 | hypothesis | 0 | `h_medium_rescue_03` | h_medium_rescue | 0.6756 | 0.5885 | 0.5765 | 0.9396/0.4901/0.3358 | 0.3358 | 0.9705 | 0.9973 | 0.3078 | 0.7646 | False | Hypothesis H1: class-2 BUT medium is QRS-visible but locally unreliable. |
| 20 | hypothesis | 0 | `h_bad_rescue_04` | h_bad_rescue | 0.6511 | 0.6732 | 0.5709 | 0.9948/0.3678/0.6569 | 0.3678 | 0.9668 | 1.0000 | 0.3049 | 0.7663 | False | Hypothesis H2: class-3 BUT bad is QRS-confounded rather than only low-amplitude. |

## Guided Quick Top 30
| rank | mode | seed | spec | family | BUT acc | bal | macro | recalls G/M/B | minMB | PTB acc | PTB bad | morph | feature | mismatch | note |
|---:|---|---:|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---|---|
| 1 | quick | 0 | `g_t01_m1_b2_g2` | guided_morphology_large | 0.7836 | 0.6388 | 0.6686 | 0.8481/0.7763/0.2920 | 0.2920 | 0.9759 | 0.9973 | 0.3078 | 0.7646 | False | Guided expansion from hypothesis templates: medium P/T/ST unreliability, bad pseudo-QRS/contact pressure, and wearable-good overlap. |
| 2 | quick | 0 | `g_t01_m1_b1_g2` | guided_morphology_large | 0.6796 | 0.7272 | 0.6385 | 0.8788/0.5047/0.7981 | 0.5047 | 0.9782 | 1.0000 | 0.3101 | 0.7633 | False | Guided expansion from hypothesis templates: medium P/T/ST unreliability, bad pseudo-QRS/contact pressure, and wearable-good overlap. |
| 3 | quick | 0 | `g_t01_m1_b2_g1` | guided_morphology_large | 0.7536 | 0.6268 | 0.6360 | 0.9574/0.6286/0.2944 | 0.2944 | 0.9782 | 0.9986 | 0.3066 | 0.7654 | False | Guided expansion from hypothesis templates: medium P/T/ST unreliability, bad pseudo-QRS/contact pressure, and wearable-good overlap. |
| 4 | quick | 0 | `g_t01_m1_b3_g1` | guided_morphology_large | 0.7497 | 0.7841 | 0.6660 | 0.9473/0.5802/0.8248 | 0.5802 | 0.9650 | 1.0000 | 0.3074 | 0.7649 | False | Guided expansion from hypothesis templates: medium P/T/ST unreliability, bad pseudo-QRS/contact pressure, and wearable-good overlap. |
| 5 | quick | 0 | `g_t01_m1_b3_g2` | guided_morphology_large | 0.6731 | 0.7550 | 0.6227 | 0.9591/0.4180/0.8881 | 0.4180 | 0.9746 | 1.0000 | 0.3086 | 0.7642 | False | Guided expansion from hypothesis templates: medium P/T/ST unreliability, bad pseudo-QRS/contact pressure, and wearable-good overlap. |
| 6 | quick | 0 | `g_t01_m1_b1_g1` | guided_morphology_large | 0.6226 | 0.6926 | 0.5419 | 0.9588/0.3308/0.7883 | 0.3308 | 0.9668 | 0.9986 | 0.3088 | 0.7640 | False | Guided expansion from hypothesis templates: medium P/T/ST unreliability, bad pseudo-QRS/contact pressure, and wearable-good overlap. |
