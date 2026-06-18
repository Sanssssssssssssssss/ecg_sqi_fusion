# N17043 Good Outer Shell Failure Analysis

N17043 target = full good 17043 + full medium 10628 + bad core 4084. Relative to N12800, this adds 4243 good outer-shell rows.

N12800 shallow rule on the 4243 new good rows:
- captured as good: 1471
- missed as medium/bad: 2772

Key feature gap for missed good: pc1 is much less negative, qrs_prom/qrs_visibility are lower, baseline_step and amplitude_entropy are higher. These are good labels, but visually and dimensionally they sit in the medium-like shell.

Artifacts:
- `n17043_outer_good_captured_vs_missed_pca.png`
- `n17043_outer_good_captured_vs_missed_waveforms.png`
- `n17043_missed_good_vs_medium_waveform_pairs.png`
- `n17043_outer_good_missed_feature_gaps.csv`
- `n17043_missed_good_medium_nearest_pairs.csv`

Conclusion: the next generator should not just loosen pc1/qrs_prom. It needs a dedicated good outer block: low/medium QRS visibility, higher baseline_step, higher amplitude_entropy, and less-negative PC1, while medium hard negatives must be preserved.
