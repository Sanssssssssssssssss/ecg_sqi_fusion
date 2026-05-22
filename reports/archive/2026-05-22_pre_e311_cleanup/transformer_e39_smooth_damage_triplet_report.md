# E3.9 Smooth-Damage Margin Triplet

E3.9 stops the E3.7/E3.8 max-axis line and returns to the simpler E3.5/E3.6
form: smooth weighted damage, matched SNR without over-tightening, clear gray
zone margins, and the same simple CLS/raw transformer.

Reference results:

| Reference | Test Acc | Note |
| --- | ---: | --- |
| E3.5 CLS | 0.9387 | strongest prior simple CLS result |
| E3.6 margin-tuned R4 | 0.9255 | critical-damage fallback |
| E3.8 C2 warm-start | 0.8859 | max-axis line did not recover |

## Generator Patch

Changed file:

- `src/transformer_pipeline/noise/synthesize_morph_damage_triplet.py`

New label versions:

- `e39a_smooth_morph_margin`
- `e39b_smooth_critical_margin`

Saved label columns:

- `smooth_morph_score`
- `smooth_critical_score`
- `diagnostic_damage_score`
- `core_diagnostic_score`
- `beat_axis`
- `dominant_axis`

E3.9a label axis:

```text
smooth_morph_score =
    0.45*qrs_nprd
  + 0.25*tst_nprd
  + 0.20*(1 - beat_corr)
  + 0.10*max_beat_nprd
```

E3.9b label axis:

```text
smooth_critical_score =
    0.45*qrs_nprd
  + 0.35*tst_nprd
  + 0.15*(1 - beat_corr)
  + 0.05*max_beat_nprd
```

Triplet selection:

- prefer candidates within `0.50 dB` of the matched SNR
- final triplet measured-SNR gap remains `<= 0.75 dB`
- within the preferred SNR pool, select by SNR closeness first and smooth-score target second

I also added an observable waveform audit:

```text
d_gm = nrmse(noisy_good, noisy_medium)
d_mb = nrmse(noisy_medium, noisy_bad)
d_gb = nrmse(noisy_good, noisy_bad)
observable_margin = min(d_gm, d_mb)
```

This checks whether the class boundary is visible in the noisy waveform, not
only in clean-reference damage metrics.

## Data Audit

| Run | Data | Triplets train / val / test | SNR Oracle | Max SNR Mean Gap | pSQI Gap | basSQI Gap | Obs Margin p10 / p50 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| D1 | E3.9a smooth morph, 6-12 dB | 3340 / 656 / 673 | 0.3330 | 0.0520 dB | 0.0028 | 0.0043 | 0.3458 / 0.4272 |
| D2 | E3.9a smooth morph, 5-10 dB | 3309 / 667 / 665 | 0.3339 | 0.0545 dB | 0.0027 | 0.0039 | 0.3495 / 0.4385 |
| D3 | E3.9b smooth critical, 6-12 dB | 3725 / 748 / 750 | 0.3344 | 0.0285 dB | 0.0021 | 0.0044 | 0.2930 / 0.3814 |
| D4 | E3.9b smooth critical, 5-10 dB | 3640 / 715 / 728 | 0.3354 | 0.0213 dB | 0.0021 | 0.0054 | 0.3117 / 0.3956 |

The SNR oracle stays at chance-level for all four datasets, and SQI proxy gaps
remain tiny. D1/D2 have larger observable margins than D3/D4, which matches the
later transformer results.

Medium audit:

| Run | Medium noncritical share | Medium beat-dominant share |
| --- | ---: | ---: |
| D1 | 0.58% | 11.20% |
| D2 | 0.97% | 10.97% |
| D3 | 0.38% | 8.92% |
| D4 | 0.92% | 10.21% |

This fixes the E3.7/E3.8 failure mode: medium is no longer dominated by
noncritical or beat-axis-only cases.

Score medians:

| Run | Score | good p50 | medium p50 | bad p50 |
| --- | --- | ---: | ---: | ---: |
| D1 | smooth morph | 0.0863 | 0.3256 | 0.3221 |
| D2 | smooth morph | 0.0864 | 0.3331 | 0.3310 |
| D3 | smooth critical | 0.0734 | 0.3263 | 0.2471 |
| D4 | smooth critical | 0.0765 | 0.3396 | 0.2595 |

The bad score median can overlap medium because bad still includes explicit QRS
or beat-correlation failure triggers. This is intentional and is why
score-aware soft CE was not used in this round.

## SQI Baseline

| Run | SVM Acc | SVM Macro F1 | SVM Medium Recall | MLP Acc | MLP Macro F1 | MLP Medium Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| D1 | 0.5364 | 0.5365 | 0.5795 | 0.5052 | 0.5017 | 0.5825 |
| D2 | 0.5574 | 0.5580 | 0.5699 | 0.5118 | 0.5086 | 0.5910 |
| D3 | 0.4996 | 0.4982 | 0.5733 | 0.4809 | 0.4788 | 0.5507 |
| D4 | 0.5096 | 0.5082 | 0.5810 | 0.5005 | 0.4984 | 0.5522 |

Readout: SQI baselines remain capped around `0.50-0.56`, so the transformer
gain is not coming from a global SQI shortcut.

## Transformer Runs

All four runs used the simple CLS/raw setup:

- `cls_pool=cls`
- `input_mode=raw`
- no rank loss
- no local mask head
- no noise type head
- no SQI head
- no teacher distillation
- no score-aware soft CE

| Run | Best Val Epoch | Best Val Acc | Test Acc | Balanced Acc | Macro F1 | Recall good / medium / bad |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| D1 | 20 | 0.9512 | 0.9465 | 0.9465 | 0.9466 | 0.9153 / 0.9465 / 0.9777 |
| D2 | 19 | 0.9485 | 0.9378 | 0.9378 | 0.9379 | 0.9323 / 0.9173 / 0.9639 |
| D3 | 13 | 0.9398 | 0.9267 | 0.9267 | 0.9268 | 0.9467 / 0.9120 / 0.9213 |
| D4 | 24 | 0.9441 | 0.9350 | 0.9350 | 0.9353 | 0.9299 / 0.9258 / 0.9492 |

Confusion matrices, row order `good / medium / bad`:

```text
D1
[[616, 53, 4],
 [22, 637, 14],
 [2, 13, 658]]

D2
[[620, 41, 4],
 [36, 610, 19],
 [4, 20, 641]]

D3
[[710, 32, 8],
 [48, 684, 18],
 [13, 46, 691]]

D4
[[677, 48, 3],
 [42, 674, 12],
 [6, 31, 691]]
```

## Gains

| Run | Test Acc | vs E3.5 CLS | vs E3.6 R4 | vs E3.8 C2 |
| --- | ---: | ---: | ---: | ---: |
| D1 | 0.9465 | +0.0078 | +0.0210 | +0.0606 |
| D2 | 0.9378 | -0.0009 | +0.0123 | +0.0519 |
| D3 | 0.9267 | -0.0120 | +0.0012 | +0.0408 |
| D4 | 0.9350 | -0.0037 | +0.0095 | +0.0491 |

The best result is D1: E3.9a smooth-morph margin at 6-12 dB. It beats the
previous E3.5 CLS reference while keeping SQI baseline performance low.

## Decision

Keep E3.9a/D1 as the current mainline candidate. It is the first small patch in
this sequence that improves over E3.5 without adding model complexity.

Do not continue the E3.7/E3.8 max-axis line. It made labels cleaner by some
audits, but the task became less learnable from the raw waveform.

Do not add score-aware soft CE yet. The continuous smooth score is useful for
audit, but bad-class membership also comes from explicit QRS and correlation
failure conditions. A soft target based only on the smooth score would blur the
bad boundary we actually want to preserve.
