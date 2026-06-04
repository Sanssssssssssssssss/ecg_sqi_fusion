# Generator / Head-Only Minimal Validation Plan

This is deliberately held behind the protocol sweep.  The old v0 grid is superseded because it skipped this audit.

| spec | purpose | value |
| --- | --- | --- |
| g0_current_generator_baseline | Baseline PTB synthetic generator; no BUT-style protocol changes. | Separates protocol effects from generator effects. |
| g1_but_snr_table1 | Good 16-18dB, medium 5-14dB, bad <= -3dB using BUT-like severity bands. | Tests whether logit mismatch is mostly SNR calibration. |
| g2_good_lenient | Good can include mild noise if P/QRS/T remain clear. | Reduces over-rejection of BUT good under real wearable noise. |
| g3_medium_partial_local | QRS detectable but P/T/ST local details unreliable. | Targets BUT class-2 expert definition instead of middle SNR only. |
| g4_bad_qrs_unreliable | Bad has unreliable QRS: missing/spurious peaks, attenuation, broadening. | Directly attacks current bad false negatives. |
| g5_bad_contact_loss | Adds flatline, clipping, contact loss, baseline step and dropout. | Models BUT free-living wearable failures absent in PTB synthetic. |
| g6_cinc_noisy_bad | Adds burst/HF/motion unusable noisy patterns. | Keeps CinC-style too-noisy rejection aligned with BUT bad. |
| g7_mixed_but_style | Mixture of G2-G6 with PTB balanced split preserved. | Candidate for full Uformer training after head-only validation. |
