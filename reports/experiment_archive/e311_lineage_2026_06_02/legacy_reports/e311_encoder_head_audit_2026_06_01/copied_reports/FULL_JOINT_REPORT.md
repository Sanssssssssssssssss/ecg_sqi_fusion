# Full Joint Denoise-Before-Classifier + SQI Training

## What Changed

The cached SQI-head experiment was replaced with full model training on the complete sweep dataset.

Full training path:

`noisy ECG -> trainable residual_unet denoiser -> denoised ECG -> trainable Transformer classifier -> optional SQI residual delta`

All runs use the full `med6p25_badgap7_badcm0p75` artifact:

- train: `10935 = 3645 x 3`
- val: `2184 = 728 x 3`
- test: `2202 = 734 x 3`
- total: `15321`

The code is experiment-only:

- `full_joint_train.py`
- `run_full_grid.py`
- outputs under `artifact/full_grid`

No source under `src/transformer_pipeline` was modified.

## Controls

Cached/frozen denoise-before-classifier reference before full training:

- Acc: `0.9613987284`
- Recall good/medium/bad: `0.9441416894 / 0.9495912807 / 0.9904632153`

Full no-SQI control with bad-priority checkpointing:

- Run: `pilot_grid_none_sc0p0_cw1c1p15c1p5_lq0p0_ad4ae3a5`
- Acc: `0.9613987284`
- Recall good/medium/bad: `0.9468664850 / 0.9509536785 / 0.9863760218`
- Denoise score: `2.9143`
- MSE ratio: `0.09437`
- SNR gain: `+7.982 dB`

Higher medium/bad weight no-SQI:

- Run: `pilot_grid3_none_sc0p0_cw1c1p35c1p65_lq0p0_mg0p0_5b22a28b`
- Acc: `0.9623069936`
- Recall good/medium/bad: `0.9495912807 / 0.9509536785 / 0.9863760218`
- Denoise score: `2.9036`

## Best SQI Full-Training Candidate

- Run: `pilot_grid2_delta_sc0p03_cw1c1p25c1p5_lq0p05_mg0p005_b1357c47`
- SQI: delta residual, scale `0.03`
- Class weight: `1,1.25,1.5`
- Quality aux: `0.05`
- Medium guard: `0.005`
- Acc: `0.9627611262`
- Recall good/medium/bad: `0.9536784741 / 0.9455040872 / 0.9891008174`
- Corrected / harmed by SQI vs base logits in the same model: `6 / 3`
- Denoise score: `2.9655`
- MSE ratio: `0.09202`
- SNR gain: `+8.142 dB`

This is the best acc so far in full training, and it beats the full no-SQI controls on accuracy. It does not yet satisfy the desired medium/bad balance.

## Best Bad-Recall SQI Candidate

- Run: `pilot_grid_delta_sc0p05_cw1c1p15c1p5_lq0p05_58eccfd6`
- Acc: `0.9613987284`
- Recall good/medium/bad: `0.9495912807 / 0.9441416894 / 0.9904632153`
- Denoise score: `2.9770`
- MSE ratio: `0.09135`
- SNR gain: `+8.179 dB`

This keeps baseline-level acc and restores the target bad recall, but it sacrifices medium recall too much.

## Interpretation

The complete runs show a real tradeoff:

- Full joint training improves denoise quality strongly compared with the frozen baseline.
- No-SQI full training can already improve acc to `0.96231` with higher medium/bad weighting.
- SQI residual can push acc higher (`0.96276`) or recover bad recall (`0.99046`), but not both with medium recall preserved in this first full grid.

Current SQI conclusion:

SQI is useful as a correction signal, but it is not ready for promotion as a clean mainline head. The next useful grid should target the Pareto gap directly: keep the `scale=0.02-0.03` SQI region, add a stronger medium-preservation term, and train on mixed gap5/gap6.5/gap7 so the SQI head learns a more stable quality boundary instead of overfitting the gap7 decision edge.
