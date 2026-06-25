# v112 Accuracy Tuning Report

Generated: 2026-06-25

## Setup

- Data protocol: `ptb_v112_gm_buffered_large_hybrid_s20260741`
- Model: `EventFactorizedSQIConformer`
- Inference input: waveform-derived channels only
- Feature/SQI targets: training teacher and diagnostic only
- All runs in this report were trained from scratch.

## Baseline

The previous v112 quick baseline used 3 folds and 6 finetune epochs.

| candidate | mean acc | macro-F1 | good | medium | bad |
| --- | ---: | ---: | ---: | ---: | ---: |
| `E4_query_highres_local_art` | 0.9077 | 0.8867 | 0.8056 | 0.8829 | 0.9655 |
| `E4_query_highres_local_art_unified_subtype` | 0.9140 | 0.8951 | 0.8394 | 0.8676 | 0.9727 |

Unified subtype helped the class boundary, but it also weakened some feature recovery, especially `sqi_basSQI` and `non_qrs_rms_ratio`.

## Tuning Runs

Three 10-epoch phase1 variants were tested:

| candidate | mean acc | macro-F1 | good | medium | bad | best fold acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `E4_query_highres_local_art_unified_lowaux_lr15e4` | 0.9238 | 0.9075 | 0.9172 | 0.8369 | 0.9768 | 0.9289 |
| `E4_query_highres_local_art_unified_factorprotect_lr15e4` | 0.9223 | 0.9058 | 0.9025 | 0.8386 | 0.9796 | 0.9250 |
| `E4_query_highres_local_art_unified_gmweighted_lr15e4` | 0.9201 | 0.9034 | 0.8847 | 0.8629 | 0.9677 | 0.9270 |

Best overall candidate: `E4_query_highres_local_art_unified_lowaux_lr15e4`.

Its best fold confusion was:

```text
[[847,  76,   7],
 [106, 1004, 44],
 [ 23,  30, 1888]]
```

The remaining errors are dominated by good/medium mutual confusion, not bad recall.

## Calibration Check

Val-tuned class-prior/subtype-head calibration was tested as a diagnostic. It improved the best mean acc only from about 0.9238 to about 0.9260, with best fold around 0.9324. Therefore the 0.95 gap is not mainly a threshold/calibration issue.

## Pretraining Check

Two v112 phase3 variants were tested with 2 pretrain epochs plus 10 finetune epochs.

| candidate | mean acc | macro-F1 | good | medium | bad | best fold acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `P2_ecg_mask_unified_lowaux_v112` | 0.9211 | 0.9045 | 0.8734 | 0.8782 | 0.9657 | 0.9289 |
| `P3_factorized_proxy_unified_gmweighted_v112` | 0.9212 | 0.9050 | 0.9094 | 0.8416 | 0.9716 | 0.9245 |

Pretraining did not beat the best phase1 tune.

## Feature Recovery

The strongest tune recovered several teacher targets well:

- `baseline_step`: about 0.95 correlation
- `detector_agreement`: about 0.81-0.82
- `flatline_ratio`: about 0.79
- `non_qrs_diff_p95`: strong in the full CSV

Still weak:

- `qrs_visibility`: about 0.37-0.38
- `amplitude_entropy`: about 0.32-0.36
- `contact_loss_win_ratio`: about 0.19-0.21
- medium-class `sqi_basSQI` / `non_qrs_rms_ratio` recovery remains unstable under unified subtype training.

## Interpretation

The current v112 model can be tuned from about 0.914 to about 0.924 mean acc, and individual folds can reach about 0.929-0.932 after calibration. It does not reach 0.95 through learning-rate, class-weight, subtype-weight, or light pretraining changes.

The clean conclusion is:

1. Bad is already strong on this synthetic diagnostic.
2. The main blocker is good/medium boundary consistency.
3. The model can learn many waveform-computable SQI targets, but qrs visibility, contact/dropout evidence, amplitude entropy, and medium-side basSQI/non-QRS balance are still the weak points.
4. The next improvement should be data-side boundary repair plus targeted local supervision, not more broad weight sweeps.

## Recommended Next Step

Keep `E4_query_highres_local_art_unified_lowaux_lr15e4` as the current tuned model baseline. For a realistic shot at 0.95, create a small v113 boundary repair dataset focused only on the good/medium ambiguous rows:

- reduce label-inconsistent good/medium rows more aggressively;
- add matched pairs for good-like medium and medium-like good;
- oversample the four hard boundary subtypes without increasing bad;
- add local/pair supervision for qrs visibility, contact/reset, amplitude entropy, and non-QRS dominance.

Then rerun only the best tuned candidate, not the whole matrix.
