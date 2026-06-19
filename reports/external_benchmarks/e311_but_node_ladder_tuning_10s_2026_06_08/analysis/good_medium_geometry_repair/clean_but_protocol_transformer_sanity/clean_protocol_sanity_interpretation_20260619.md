# Clean Protocol Transformer Sanity Interpretation

## What Was Done

Variable-length modeling was intentionally skipped. This round only cleaned the existing fixed 10s BUT protocol and trained/evaluated ordinary waveform-only Transformer diagnostics.

Materialized clean 10s protocol bundles:

- `margin_ge_2s_drop_outlier`
- `margin_ge_5s_drop_outlier`
- `clean_core_plus_overlap_margin2`
- `clean_core_only_margin2`

Each bundle contains:

- `signals.npz`
- `metadata.csv`
- `original_region_atlas.csv`
- `window_segment_margins.csv`
- `audit.json`

Code:

- `outputs/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/build_clean_but_protocols.py`
- `outputs/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/run_clean_but_protocol_transformer_sanity.py`

## Clean-Policy Internal Test Results

These results evaluate on the corresponding cleaned fixed-10s protocol, not the full original BUT test.

| training/eval policy | acc | macro-F1 | good recall | medium recall | bad recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| `margin_ge_2s_drop_outlier` | 0.938056 | 0.637806 | 0.990394 | 0.964795 | 0.000000 |
| `margin_ge_5s_drop_outlier` | 0.928415 | 0.946367 | 0.997012 | 0.891189 | 1.000000 |

Interpretation:

- Cleaning makes the fixed-10s problem much easier for good/medium.
- `margin_ge_5s_drop_outlier` is the better clean policy because it preserves bad core behavior; `margin_ge_2s_drop_outlier` collapses bad in this quick run.
- The legacy validation split is still weak for calibration because after dropping outlier it has almost no bad rows.

## Full BUT Stress Results

The same clean-trained checkpoints were then evaluated on the full original BUT test.

| checkpoint | full BUT acc | macro-F1 | good recall | medium recall | bad recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| `clean_margin2_drop_outlier_on_full_but` | 0.827415 | 0.565426 | 0.873352 | 0.866471 | 0.000000 |
| `clean_margin5_drop_outlier_on_full_but` | 0.775392 | 0.673331 | 0.938462 | 0.686399 | 0.289538 |

Interpretation:

- Clean training alone does not solve full BUT.
- It confirms a split in the problem:
  - clean learnable body: waveform Transformer can learn it;
  - full/outlier stress: still needs explicit outlier/contact/bad-stress modeling.
- `margin_ge_5s_drop_outlier` learns bad core but does not transfer to record `111001` bad outlier.

## Feature Recovery On Full BUT

The clean-trained checkpoints still recover some waveform-computable features, but the hard features remain unstable:

`clean_margin5_drop_outlier_on_full_but`:

- `sqi_basSQI` corr 0.829
- `qrs_band_ratio` corr 0.752
- `baseline_step` corr 0.716
- `flatline_ratio` corr 0.727
- `qrs_visibility` corr 0.489
- `detector_agreement` corr 0.476
- `contact_loss_win_ratio` corr approximately 0

This supports the reviewer's diagnosis: input/model still do not preserve or use contact/RR/detector reliability well enough.

## Decision

Direct cleaning is useful, but not sufficient.

Next implementation should be:

1. Keep `margin_ge_5s_drop_outlier` as the clean-body training sanity policy.
2. Keep full BUT and bad outlier as stress-only reports.
3. Stop using the current legacy val as serious bad calibration; introduce record/grouped validation before making claims.
4. Implement fixed-length dual-view input, not variable length:
   - physical/global-normalized waveform,
   - robust waveform,
   - derivative recomputed after augmentation,
   - long baseline/trend,
   - local envelope/log-RMS.
5. Implement intrinsic-only aux targets:
   - 7 SQI,
   - RR/QRS count/reliability,
   - detector agreement,
   - baseline/contact,
   - detail/frequency.
6. Implement hierarchical classification:
   - bad vs non-bad,
   - good vs medium conditional on non-bad,
   - medium false-bad penalty.

This path keeps the model interpretable and waveform-first while acknowledging the dataset issue rather than hiding it.
