# E3.11f BUT 10s Next Research Plan

## Decision

Stop blind synthetic-generator expansion for now.  The strict zero-shot anchor is
`b10_all_bad_wearable`; v1/v2/v3 morphology grids did not beat it cleanly.

## Why

- BUT labels are expert-usability labels, not synthetic noise-strength labels.
- BUT medium means QRS is still usable but P/T/ST and local baseline details are
  unreliable.
- BUT bad means QRS reliability itself is compromised by pseudo-peaks, motion,
  contact loss, or severe morphology ambiguity.
- Current generator knobs mostly change artifact recipe intensity.  They can
  trade good/medium/bad recall, but they do not encode the expert boundary well
  enough.

## Track A: Strict Synthetic Zero-Shot

Keep `b10_all_bad_wearable` as the current strict synthetic baseline:

- BUT acc: 0.7735
- balanced acc: 0.8045
- macro-F1: 0.7238
- recalls good/medium/bad: 0.824/0.724/0.866

Report later morphology grids as ablations:

- `mix05`: stronger medium boundary, but macro lower.
- `s02`: medium/bad coexistence clue, but good drops.
- v2/v3: confirm hard tradeoff; no clean improvement.

## Track B: BUT-Supervised Boundary Adaptation

Run this as a separate, clearly labeled supervised adaptation experiment:

- Freeze current Uformer denoiser/features.
- Train only lightweight heads on BUT train.
- Select calibration/thresholds only on BUT val.
- Evaluate BUT test once.
- Compare full-token, bottleneck, summary, handcrafted SQI, and hybrid features.

This answers whether the current representation already contains the right
information and only needs the BUT expert boundary.

## Track C: Generator Redesign

Do not tune more scalar noise parameters.  Redesign labels around explicit
detectability:

- QRS prominence and false-peak density.
- QRS interval reliability and missing/spurious beat probability.
- P/T/ST local trustworthiness.
- baseline discontinuity and contact-loss coverage.
- segment-level usable / partially usable / unusable rules.

Only after those rules are implemented should a new synthetic zero-shot grid be
started.

## Reporting Rule

Never mix the three tracks:

- Strict synthetic zero-shot: no BUT labels used for training.
- BUT-supervised adaptation: BUT train/val used, test held out.
- Generator redesign: new synthetic artifact, must also report PTB mainline
  sanity metrics.
