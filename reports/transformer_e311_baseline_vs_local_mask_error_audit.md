# E3.11 Baseline vs Local-Mask Error Audit

Compared runs on `e311f_lite_e310_morph` test split:

- baseline: `e311f_lite_e310_morph_hc03_base_lr625_s1`
- local-mask best: `e311f_lite_e310_morph_hc22_mask010_lr625_s1`

## Summary

| Model | Test Acc | Recall G/M/B | Confusion Matrix |
| --- | ---: | --- | --- |
| baseline | 0.9464 | 0.9292 / 0.9346 / 0.9755 | `[[682,48,4],[35,686,13],[2,16,716]]` |
| local-mask best | 0.9505 | 0.9278 / 0.9496 / 0.9741 | `[[681,47,6],[27,697,10],[2,17,715]]` |

Net change:

- fixed baseline errors: 31
- new regressions: 22
- still wrong in both: 87
- net improvement: +9 samples

## Fixed Errors

| True class | Baseline pred | Count |
| --- | --- | ---: |
| medium | good | 12 |
| good | medium | 11 |
| medium | bad | 5 |
| bad | medium | 3 |

Fixed by noise kind:

| Noise kind | Count |
| --- | ---: |
| em | 12 |
| ma | 12 |
| mix | 7 |

## Regressions

| True class | Local-mask pred | Count |
| --- | --- | ---: |
| good | medium | 10 |
| medium | good | 4 |
| bad | medium | 3 |
| good | bad | 2 |
| medium | bad | 2 |
| bad | good | 1 |

Regressions by noise kind:

| Noise kind | Count |
| --- | ---: |
| mix | 10 |
| ma | 7 |
| em | 5 |

## Interpretation

Low-weight local-mask supervision improves the medium boundary without materially harming good or bad recall. The remaining risk is variance and mix-noise regressions. The next useful sweep is not more broad heads; it is the already queued focused round2: `mask=0.0075/0.0125/0.015`, `lr=6.4`, and small `mask+rank/ordinal/noise` checks.

Prediction CSVs:

- `outputs/transformer_e311_mainline_strict/e311f_lite_e310_morph/error_audit/e311f_lite_e310_morph_hc03_base_lr625_s1_test/predictions_test.csv`
- `outputs/transformer_e311_mainline_strict/e311f_lite_e310_morph/error_audit/e311f_lite_e310_morph_hc22_mask010_lr625_s1_test/predictions_test.csv`
- `outputs/transformer_e311_mainline_strict/e311f_lite_e310_morph/error_audit/e311_baseline_vs_mask_hc22_test_comparison.csv`
