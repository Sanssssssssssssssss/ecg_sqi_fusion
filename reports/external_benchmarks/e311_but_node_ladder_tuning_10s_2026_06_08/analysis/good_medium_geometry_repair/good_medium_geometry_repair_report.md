# Good/Medium Geometry Repair

## Held-Out Gate Check

- train/val focus rows: 1446
- held-out test rows: 3783
- Thresholds are estimated on train/val only; original BUT is not used for selection.

## Top Thresholds

- `flatline_ratio >= 0.169736`: rescue pass 0.953, medium-lost pass 0.003, spread 0.950
- `pc1 <= -2.66085`: rescue pass 0.949, medium-lost pass 0.000, spread 0.949
- `qrs_visibility >= 0.502622`: rescue pass 0.949, medium-lost pass 0.013, spread 0.936
- `pc3 <= 1.16717`: rescue pass 0.949, medium-lost pass 0.032, spread 0.917
- `non_qrs_diff_p95 <= 0.076231`: rescue pass 0.949, medium-lost pass 0.097, spread 0.852
- `template_corr >= 0.612288`: rescue pass 0.889, medium-lost pass 0.117, spread 0.773

## Held-Out Simulations

- `pc1_only`: acc 0.9638, macro-F1 0.9613, recall good/medium/bad 0.9861/0.9512/nan, flips 92
- `pc1_flatline`: acc 0.9638, macro-F1 0.9613, recall good/medium/bad 0.9861/0.9512/nan, flips 92
- `pc1_flatline_pc3`: acc 0.9638, macro-F1 0.9613, recall good/medium/bad 0.9861/0.9512/nan, flips 92

## Generator Translation

- Good aux: pc1-low + flatline-high rescue band, with guardrails against fatal/contact-like bad.
- Medium aux: visible-QRS hard negatives outside the rescue band, with local detail/non-QRS derivative preserved.
- Bad remains trim-bad core/right-island/near-boundary guardrail.
