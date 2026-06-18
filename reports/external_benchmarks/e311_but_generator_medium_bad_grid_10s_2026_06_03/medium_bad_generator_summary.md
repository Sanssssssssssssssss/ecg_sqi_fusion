# BUT 10s Medium/Bad Generator Continuation

This pass continues data generation rather than head-only adaptation.  The formal BUT protocol remains 10s P1, and calibration is validation-only.

Primary reading: a useful synthetic rule should keep bad recall high without swallowing medium.

| rank | spec | cw | return | BUT acc | BUT bal | macro-F1 | recalls good/medium/bad | PTB acc | PTB bad | denoise | note |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |
| 1 | m12_bad_contact_step_medium_cleaner | 1,1.50,1.82 | 0 | 0.6863 | 0.7523 | 0.6262 | 0.846/0.538/0.873 | 0.9873 | 1.0000 | 2.674 | A cleaner medium / contact-step bad contrast. |
| 2 | m07_bad_extreme_medium_protected | 1,1.48,1.78 | 0 | 0.7336 | 0.7708 | 0.7154 | 0.983/0.521/0.808 | 0.9841 | 1.0000 | 2.547 | A stricter bad class while protecting medium from over-damage. |
| 3 | m04_bad_lowamp_guarded | 1,1.55,1.70 | 0 | 0.7098 | 0.7361 | 0.6747 | 0.990/0.477/0.742 | 0.9832 | 1.0000 | 2.740 | Low-amplitude domain shift without making bad only low amplitude. |
| 4 | m03_bad_contact_no_spurious | 1,1.58,1.78 | 0 | 0.6935 | 0.6814 | 0.6670 | 0.999/0.452/0.594 | 0.9882 | 0.9973 | 2.748 | Remove spurious peaks to avoid confusing medium with artificial bad. |
| 5 | m06_medium_motion_bad_contact | 1,1.55,1.80 | 0 | 0.7320 | 0.6315 | 0.6347 | 0.996/0.551/0.348 | 0.9855 | 0.9986 | 2.614 | Wearable motion boundary: medium drift/HF, bad contact loss. |
| 6 | m01_medium_detail_bad_contact_bal | 1,1.58,1.72 | 0 | 0.6008 | 0.6597 | 0.5886 | 0.998/0.264/0.718 | 0.9891 | 1.0000 | 2.721 | Medium detail corruption with softened b10 bad. |
| 7 | m08_medium_qrs_visible_bad_flat | 1,1.60,1.75 | 0 | 0.5812 | 0.5753 | 0.5079 | 0.990/0.254/0.482 | 0.9900 | 0.9986 | 2.681 | Medium QRS stays visible; bad includes flat/contact loss. |
| 8 | m09_bad_burst_qrs_unreliable | 1,1.58,1.78 | 0 | 0.5754 | 0.6582 | 0.5405 | 0.998/0.211/0.766 | 0.9805 | 1.0000 | 2.692 | Noisy unusable bad, but still keyed to unreliable QRS. |
| 9 | m10_b10_medium_prior | 1,1.70,1.65 | 0 | 0.5342 | 0.5782 | 0.4681 | 0.999/0.147/0.589 | 0.9950 | 1.0000 | 2.788 | Class pressure favors medium recovery while retaining b10-style bad. |
| 10 | m11_medium_bad_snr_overlap | 1,1.62,1.68 | 0 | 0.5266 | 0.5896 | 0.5145 | 0.999/0.127/0.642 | 0.9909 | 0.9986 | 2.812 | Let medium have worse SNR without losing QRS reliability. |
| 11 | m05_good_lenient_medium_gap | 1,1.62,1.70 | 0 | 0.5294 | 0.6788 | 0.4806 | 0.996/0.108/0.932 | 0.9923 | 1.0000 | 2.924 | BUT good is not pristine; widen good-medium realism. |
| 12 | m02_medium_stronger_bad_qrs_missing | 1,1.60,1.72 | 0 | 0.5056 | 0.6482 | 0.4889 | 0.999/0.065/0.881 | 0.9868 | 1.0000 | 2.728 | Bad is QRS unreliable; medium is locally unreliable but detectable. |

## Interpretation

- If medium rises while bad stays above 0.85, continue this generator family.
- If bad is high but medium collapses, the rule is still over-bad-biased.
- If PTB acc or bad recall collapses, keep the rule as external-only diagnostic evidence.
