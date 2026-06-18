# Record111 Slow-LR Stability Update - 2026-06-18

## Question

We tested whether waveform Transformer generalization is failing because the record111-physio branch trains too aggressively and collapses after very few epochs.

## Experiment

Three slow-LR variants were added to the external-only runner:

- `featurefirst_wavecomp_record111physio_slowlr_a050`: LR `4e-6`, lower synthetic stress selection.
- `featurefirst_wavecomp_record111physio_slowlr_stressguard_a050`: LR `6e-6`, stronger stress selection control.
- `featurefirst_wavecomp_record111physio_slowlr_warm_a050`: LR `3e-6`, longer auxiliary warmup.

All candidates used waveform-only inference. SQI/geometry targets were used only as training auxiliary targets. Original BUT remained report-only.

## Results

| Candidate | Synthetic Test Acc | Original Test Acc | Good R | Medium R | Bad R | Bad Outlier Stress |
|---|---:|---:|---:|---:|---:|---:|
| current best `record111physio_guard` | 0.9928 | 0.8601 | 0.8635 | 0.8771 | 0.6472 | 0.5411 |
| `slowlr_a050` | 0.9882 | 0.8485 | 0.8613 | 0.8592 | 0.6204 | 0.5240 |
| `slowlr_stressguard_a050` | 0.9887 | 0.8302 | 0.9124 | 0.7856 | 0.5839 | 0.4897 |
| `slowlr_warm_a050` | 0.9908 | 0.8428 | 0.7574 | 0.9311 | 0.6472 | 0.5616 |

## Feature Recovery

| Feature | Best guard | Slowlr | Stressguard | Warm |
|---|---:|---:|---:|---:|
| `qrs_visibility` | 0.463 | 0.477 | 0.453 | 0.500 |
| `qrs_band_ratio` | -0.404 | -0.275 | -0.380 | 0.485 |
| `flatline_ratio` | 0.172 | 0.249 | 0.248 | 0.520 |
| `detector_agreement` | -0.126 | -0.077 | -0.114 | -0.052 |
| `baseline_step` | -0.226 | -0.189 | -0.227 | -0.078 |
| `sqi_basSQI` | -0.141 | -0.118 | -0.143 | -0.056 |
| `non_qrs_diff_p95` | 0.922 | 0.931 | 0.926 | 0.923 |
| `sqi_bSQI` | 0.927 | 0.927 | 0.927 | 0.928 |

## Interpretation

Slowing LR does stabilize synthetic/node training: the slow candidates reach balanced synthetic accuracy near `0.99`.

It does not improve BUT transfer. The current 2-epoch `record111physio_guard` remains the best waveform-only checkpoint on `original_test_all_10s+`.

The warm candidate is informative: it improves `qrs_band_ratio`, `flatline_ratio`, and bad-outlier stress, but it trades away too much good recall. This means the model can begin to learn the missing low-frequency/contact axes, but the current token/generator alignment makes that information compete with the good/medium boundary.

## Decision

Do not continue pure slow-LR or longer-training sweeps. The next useful move is structural:

1. improve the baseline/contact/filterbank token bank so `sqi_basSQI`, `baseline_step`, `flatline_ratio`, and `qrs_band_ratio` are exposed without forcing the classifier into medium-heavy behavior;
2. recalibrate record111 synthetic morphology against original bad-outlier summaries before training; or
3. add an explicit auxiliary consistency check that rejects checkpoints where synthetic accuracy rises while `teacher_core_mae` and BUT bucket transfer worsen.

Current best waveform-only checkpoint remains:

`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\waveform_geometry_student\N17043_gm_probe\search\featurefirst_wavecomp_record111physio_guard_a050\ckpt_best.pt`

