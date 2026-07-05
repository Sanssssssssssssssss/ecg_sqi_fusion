# BUT Local Evidence Sensitivity Counterfactuals

- Verdict: **strong for SQI-locked high-frequency/EMG local noise sensitivity; raw segment transplant is supportive but not definitive**.
- SQI-locked degradation accepted rows: `1189/1200`.
- Segment transplant accepted rows: `186/360`.
- High-frequency burst at severity 2.0: Conformer `delta_medium=0.6779`, LM-MLP `delta_GM=0.0396`, Conformer `delta_bad=0.1708`.
- EMG noise floor at severity 2.0: Conformer `delta_medium=0.6704`, LM-MLP `delta_GM=0.0316`, Conformer `delta_bad=0.1635`.
- Reset spike at severity 2.0: accepted `36/40`, Conformer `delta_medium=0.3343`, LM-MLP `delta_GM=0.0301`, Conformer `delta_bad=0.0848`.
- Raw transplant medium-to-good: top-local `0.5652`, random `0.5228`, same-class `0.2423`; reverse good-to-medium top-local `0.0385`.
- Figures: `outputs\transformer\supplemental\chapter4_evidence_frozen_final\figures\fig_M8_but_sqi_locked_local_degradation.png`, `outputs\transformer\supplemental\chapter4_evidence_frozen_final\figures\fig_M9_but_real_pair_segment_transplant.png`.

## SQI-locked local degradation

| perturbation | severity | n | accepted_n | accepted_rate | median_sqi_distance | mean_delta_conformer_medium_logit | mean_delta_conformer_bad_logit | mean_delta_mlp_gm_margin |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| emg_noise_floor | 0.2000 | 40 | 40 | 1.0000 | 0.0028 | 0.0030 | 0.0011 | -0.0012 |
| emg_noise_floor | 0.4000 | 40 | 40 | 1.0000 | 0.0027 | 0.0200 | 0.0042 | 0.0061 |
| emg_noise_floor | 0.6000 | 40 | 40 | 1.0000 | 0.0032 | 0.0536 | 0.0109 | 0.0000 |
| emg_noise_floor | 0.8000 | 40 | 40 | 1.0000 | 0.0037 | 0.1036 | 0.0200 | 0.0145 |
| emg_noise_floor | 1.0000 | 40 | 40 | 1.0000 | 0.0048 | 0.1731 | 0.0343 | 0.0077 |
| emg_noise_floor | 1.2000 | 40 | 40 | 1.0000 | 0.0063 | 0.2594 | 0.0531 | 0.0151 |
| emg_noise_floor | 1.4000 | 40 | 40 | 1.0000 | 0.0072 | 0.3591 | 0.0778 | 0.0097 |
| emg_noise_floor | 1.6000 | 40 | 40 | 1.0000 | 0.0083 | 0.4642 | 0.1059 | 0.0146 |
| emg_noise_floor | 1.8000 | 40 | 40 | 1.0000 | 0.0105 | 0.5693 | 0.1349 | 0.0231 |
| emg_noise_floor | 2.0000 | 40 | 40 | 1.0000 | 0.0127 | 0.6704 | 0.1635 | 0.0316 |
| hf_burst | 0.2000 | 40 | 40 | 1.0000 | 0.0015 | 0.0060 | 0.0013 | 0.0092 |
| hf_burst | 0.4000 | 40 | 40 | 1.0000 | 0.0019 | 0.0309 | 0.0054 | 0.0111 |
| hf_burst | 0.6000 | 40 | 40 | 1.0000 | 0.0033 | 0.0806 | 0.0152 | 0.0116 |
| hf_burst | 0.8000 | 40 | 40 | 1.0000 | 0.0042 | 0.1635 | 0.0355 | 0.0175 |
| hf_burst | 1.0000 | 40 | 40 | 1.0000 | 0.0061 | 0.2627 | 0.0602 | 0.0198 |
| hf_burst | 1.2000 | 40 | 40 | 1.0000 | 0.0082 | 0.3611 | 0.0846 | 0.0152 |
| hf_burst | 1.4000 | 40 | 40 | 1.0000 | 0.0111 | 0.4554 | 0.1097 | 0.0148 |
| hf_burst | 1.6000 | 40 | 40 | 1.0000 | 0.0141 | 0.5404 | 0.1331 | 0.0233 |
| hf_burst | 1.8000 | 40 | 40 | 1.0000 | 0.0176 | 0.6148 | 0.1539 | 0.0360 |
| hf_burst | 2.0000 | 40 | 40 | 1.0000 | 0.0215 | 0.6779 | 0.1708 | 0.0396 |
| reset_spike | 0.2000 | 40 | 40 | 1.0000 | 0.0039 | 0.0021 | 0.0014 | -0.0073 |
| reset_spike | 0.4000 | 40 | 40 | 1.0000 | 0.0082 | 0.0111 | 0.0040 | -0.0047 |
| reset_spike | 0.6000 | 40 | 40 | 1.0000 | 0.0133 | 0.0268 | 0.0074 | -0.0057 |
| reset_spike | 0.8000 | 40 | 40 | 1.0000 | 0.0199 | 0.0551 | 0.0145 | -0.0011 |
| reset_spike | 1.0000 | 40 | 40 | 1.0000 | 0.0258 | 0.0908 | 0.0233 | 0.0030 |
| reset_spike | 1.2000 | 40 | 40 | 1.0000 | 0.0316 | 0.1341 | 0.0338 | 0.0052 |
| reset_spike | 1.4000 | 40 | 40 | 1.0000 | 0.0412 | 0.1862 | 0.0467 | 0.0115 |
| reset_spike | 1.6000 | 40 | 37 | 0.9250 | 0.0574 | 0.2095 | 0.0471 | 0.0136 |
| reset_spike | 1.8000 | 40 | 36 | 0.9000 | 0.0707 | 0.2713 | 0.0638 | 0.0229 |
| reset_spike | 2.0000 | 40 | 36 | 0.9000 | 0.0792 | 0.3343 | 0.0848 | 0.0301 |

## Real-pair segment transplant

| direction | control | recipient_class | n | accepted_n | accepted_rate | median_sqi_distance | mean_delta_conformer_medium_logit | mean_delta_conformer_bad_logit | mean_delta_mlp_gm_margin |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| good_into_good_same_class | same-class | good | 60 | 30 | 0.5000 | 0.3446 | 0.2423 | 0.0222 | -0.1009 |
| good_into_medium_random | random | medium | 60 | 35 | 0.5833 | 0.1707 | -0.0095 | -0.0673 | -0.0272 |
| good_into_medium_top | top-local | medium | 60 | 27 | 0.4500 | 0.3164 | 0.0385 | -0.0576 | -0.0930 |
| medium_into_good_random | random | good | 60 | 36 | 0.6000 | 0.2168 | 0.5228 | 0.1092 | -0.0197 |
| medium_into_good_top | top-local | good | 60 | 27 | 0.4500 | 0.2721 | 0.5652 | 0.0941 | 0.0105 |
| medium_into_medium_same_class | same-class | medium | 60 | 31 | 0.5167 | 0.3802 | 0.0298 | -0.0426 | -0.1379 |
