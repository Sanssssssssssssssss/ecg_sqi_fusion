# BUT 10s Morphology-Guided Synthetic Grid

Formal protocol: BUT 10s P1, validation-only calibration, test reporting only.

## Anchor

`b10_all_bad_wearable`: acc 0.7735, balanced 0.8045, macro-F1 0.7238, recalls 0.824/0.724/0.866.

## Hypothesis Verifier Top 20
| rank | mode | seed | spec | family | BUT acc | bal | macro | recalls G/M/B | minMB | PTB acc | PTB bad | morph | feature | mismatch | note |
|---:|---|---:|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---|---|
| 1 | hypothesis | 0 | `h_medium_rescue_01` | h_medium_rescue | 0.7385 | 0.6459 | 0.6189 | 0.8346/0.6896/0.4136 | 0.4136 | 0.9342 | 0.9959 | 0.3087 | 0.7641 | False | Hypothesis H1: class-2 BUT medium is QRS-visible but locally unreliable. |

## Guided Quick Top 30
_No guided quick rows yet._

## Full Confirmation Top 20
_No full rows yet._

## Seed Confirmation
_No seed rows yet._

## Interpretation Guard

Feature target score is a tie-breaker only. Promotion requires BUT macro/balanced/min(M,B) and PTB sanity; a feature-only improvement without BUT metric movement is marked as proxy-insufficient.
