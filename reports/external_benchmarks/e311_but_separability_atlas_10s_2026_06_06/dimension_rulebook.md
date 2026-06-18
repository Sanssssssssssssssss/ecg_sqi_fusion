# Dimension Rulebook For Next Synthetic Targets

## Good: AND(all critical dimensions acceptable)

- Strong QRS visibility and detector agreement.
- Low fatal/contact/flat score.
- Stable non-QRS morphology and low detail instability.
- Baseline/HF motion not dominant.

## Medium: QRS usable + details unreliable

- QRS remains detectable, but P/T/ST or local baseline/detail features drift.
- Avoid heavy flat/contact in medium generation; that pushes the sample toward bad.
- Medium should be treated as an independent cluster family, not just midpoint SNR.

## Bad: OR(any fatal dimension fails hard)

- QRS detectability failure, severe spurious peaks, contact/flat/low amplitude, clipping, strong baseline jump/platform, or HF/motion burst can each be sufficient.
- Do not make every bad sample fail on every dimension; use sample-level fatal subtype mixtures.

## Candidate generator targets

Use these top BUT-separating features as distribution targets before training:

| feature | family | separability_score |
| --- | --- | --- |
| sample_entropy_proxy | Detail reliability | 0.9616 |
| higuchi_fd_proxy | Detail reliability | 0.8406 |
| sqi_bSQI | SQI | 0.8028 |
| non_qrs_diff_p95 | Detail reliability | 0.778 |
| template_corr | QRS detectability | 0.7685 |
| diff_abs_median | Detail reliability | 0.7625 |
| zero_crossing_rate | Motion/frequency | 0.7444 |
| band_30_45 | Motion/frequency | 0.7255 |
| hjorth_mobility | Motion/frequency | 0.7143 |
| sqi_pSQI | SQI | 0.7115 |
| wavelet_e4 | Motion/frequency | 0.7047 |
| diff_zero_crossing_rate | Motion/frequency | 0.687 |
| flatline_ratio | Contact/flat/fatal | 0.6835 |
| band_5_15 | Motion/frequency | 0.6828 |
| non_qrs_rms_ratio | Detail reliability | 0.6823 |