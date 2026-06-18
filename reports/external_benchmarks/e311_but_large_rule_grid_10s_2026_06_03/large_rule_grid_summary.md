# E3.11f BUT 10s Large Rule Grid

Formal protocol: BUT 10s P1, validation-only calibration, test reporting only.

## Anchor

`b10_all_bad_wearable`: acc 0.7735, balanced 0.8045, macro-F1 0.7238, recalls 0.824/0.724/0.866.

## Quick Stage Top 20
| rank | mode | seed | spec | family | BUT acc | bal | macro | recalls G/M/B | minMB | PTB acc | PTB bad | denoise | note |
|---:|---|---:|---|---|---:|---:|---:|---|---:|---:|---:|---:|---|
| 1 | quick | 0 | `b10_anchor_family_01` | b10_anchor_family | 0.7532 | 0.6923 | 0.6734 | 0.9495/0.6141/0.5134 | 0.5134 | 0.9768 | 0.9986 | 2.6256 | Conservative perturbation around b10_all_bad_wearable. |
| 2 | quick | 0 | `medium_qrs_visible_family_11` | medium_qrs_visible_family | 0.7044 | 0.7659 | 0.6560 | 0.9918/0.4544/0.8516 | 0.4544 | 0.9796 | 1.0000 | 2.6842 | BUT class-2 style: visible QRS, unreliable P/T/ST or local baseline. |
| 3 | quick | 0 | `b10_anchor_family_03` | b10_anchor_family | 0.7428 | 0.7192 | 0.6415 | 0.9071/0.6179/0.6326 | 0.6179 | 0.9755 | 0.9986 | 2.6412 | Conservative perturbation around b10_all_bad_wearable. |
| 4 | quick | 0 | `b10_anchor_family_11` | b10_anchor_family | 0.6946 | 0.6354 | 0.6149 | 0.9802/0.4831/0.4428 | 0.4428 | 0.9750 | 0.9986 | 2.8252 | Conservative perturbation around b10_all_bad_wearable. |
| 5 | quick | 0 | `medium_qrs_visible_family_02` | medium_qrs_visible_family | 0.6748 | 0.6200 | 0.6120 | 0.9970/0.4324/0.4307 | 0.4307 | 0.9759 | 1.0000 | 2.6785 | BUT class-2 style: visible QRS, unreliable P/T/ST or local baseline. |
| 6 | quick | 0 | `medium_qrs_visible_family_13` | medium_qrs_visible_family | 0.8002 | 0.7397 | 0.7656 | 0.9695/0.6826/0.5669 | 0.5669 | 0.9614 | 1.0000 | 2.5097 | BUT class-2 style: visible QRS, unreliable P/T/ST or local baseline. |
| 7 | quick | 0 | `bad_qrs_unreliable_family_06` | bad_qrs_unreliable_family | 0.7589 | 0.7721 | 0.7318 | 0.9934/0.5662/0.7567 | 0.5662 | 0.9655 | 1.0000 | 2.5323 | BUT class-3 style: QRS itself becomes unreliable through pseudo-peaks/contact. |
| 8 | quick | 0 | `medium_qrs_visible_family_09` | medium_qrs_visible_family | 0.7227 | 0.7737 | 0.7289 | 0.8898/0.5725/0.8589 | 0.5725 | 0.9696 | 1.0000 | 2.6964 | BUT class-2 style: visible QRS, unreliable P/T/ST or local baseline. |
| 9 | quick | 0 | `medium_qrs_visible_family_07` | medium_qrs_visible_family | 0.7647 | 0.8048 | 0.7234 | 0.9121/0.6338/0.8686 | 0.6338 | 0.9673 | 1.0000 | 2.5772 | BUT class-2 style: visible QRS, unreliable P/T/ST or local baseline. |
| 10 | quick | 0 | `medium_qrs_visible_family_05` | medium_qrs_visible_family | 0.7759 | 0.8036 | 0.7124 | 0.9453/0.6310/0.8345 | 0.6310 | 0.9737 | 1.0000 | 2.6293 | BUT class-2 style: visible QRS, unreliable P/T/ST or local baseline. |
| 11 | quick | 0 | `bad_qrs_unreliable_family_12` | bad_qrs_unreliable_family | 0.7768 | 0.7040 | 0.7027 | 0.9679/0.6455/0.4988 | 0.4988 | 0.9709 | 0.9986 | 2.6326 | BUT class-3 style: QRS itself becomes unreliable through pseudo-peaks/contact. |
| 12 | quick | 0 | `bad_qrs_unreliable_family_14` | bad_qrs_unreliable_family | 0.7237 | 0.7595 | 0.6989 | 0.9865/0.5014/0.7908 | 0.5014 | 0.9659 | 1.0000 | 2.6409 | BUT class-3 style: QRS itself becomes unreliable through pseudo-peaks/contact. |
| 13 | quick | 0 | `b10_anchor_family_10` | b10_anchor_family | 0.7110 | 0.7497 | 0.6964 | 0.9821/0.4810/0.7859 | 0.4810 | 0.9746 | 0.9986 | 2.6236 | Conservative perturbation around b10_all_bad_wearable. |
| 14 | quick | 0 | `bad_qrs_unreliable_family_05` | bad_qrs_unreliable_family | 0.7414 | 0.7817 | 0.6946 | 0.9338/0.5743/0.8370 | 0.5743 | 0.9569 | 0.9986 | 2.5286 | BUT class-3 style: QRS itself becomes unreliable through pseudo-peaks/contact. |
| 15 | quick | 0 | `good_not_pristine_family_01` | good_not_pristine_family | 0.7680 | 0.7953 | 0.6931 | 0.8530/0.6911/0.8418 | 0.6911 | 0.9728 | 1.0000 | 2.6607 | Good class gets wearable variation so BUT good is not over-rejected. |
| 16 | quick | 0 | `bad_qrs_unreliable_family_04` | bad_qrs_unreliable_family | 0.7605 | 0.7291 | 0.6877 | 0.9709/0.6010/0.6156 | 0.6010 | 0.9646 | 1.0000 | 2.6185 | BUT class-3 style: QRS itself becomes unreliable through pseudo-peaks/contact. |
| 17 | quick | 0 | `bad_qrs_unreliable_family_03` | bad_qrs_unreliable_family | 0.7834 | 0.6812 | 0.6860 | 0.9577/0.6746/0.4112 | 0.4112 | 0.9714 | 0.9986 | 2.6406 | BUT class-3 style: QRS itself becomes unreliable through pseudo-peaks/contact. |
| 18 | quick | 0 | `bad_qrs_unreliable_family_08` | bad_qrs_unreliable_family | 0.7276 | 0.7673 | 0.6842 | 0.9887/0.5054/0.8078 | 0.5054 | 0.9628 | 1.0000 | 2.5079 | BUT class-3 style: QRS itself becomes unreliable through pseudo-peaks/contact. |
| 19 | quick | 0 | `medium_qrs_visible_family_14` | medium_qrs_visible_family | 0.7218 | 0.7316 | 0.6735 | 0.9953/0.4989/0.7007 | 0.4989 | 0.9578 | 1.0000 | 2.4502 | BUT class-2 style: visible QRS, unreliable P/T/ST or local baseline. |
| 20 | quick | 0 | `medium_qrs_visible_family_06` | medium_qrs_visible_family | 0.7663 | 0.6608 | 0.6732 | 0.9099/0.6832/0.3893 | 0.3893 | 0.9691 | 0.9986 | 2.6118 | BUT class-2 style: visible QRS, unreliable P/T/ST or local baseline. |

## Full Stage Top 20
_No full-stage rows yet._

## Seed Confirmation
_No seed-confirmation rows yet._

## Rule Family Summary

| family | n | best balanced | best macro | best min(M,B) |
|---|---:|---:|---:|---:|
| b10_anchor_family | 12 | 0.7497 | 0.6964 | 0.6179 |
| bad_qrs_unreliable_family | 14 | 0.8149 | 0.7318 | 0.6010 |
| good_not_pristine_family | 2 | 0.7953 | 0.6931 | 0.6911 |
| medium_qrs_visible_family | 14 | 0.8057 | 0.7656 | 0.6338 |

## Interpretation Notes

- Rank uses macro-F1, balanced accuracy, and medium/bad coexistence; raw accuracy alone is not the target.
- Candidates that beat b10 balanced but lower macro-F1 should be treated as partial evidence, not automatic winners.
- Negative controls are included to prove why SNR-only, flatline-only, and low-amplitude-only rules are insufficient.
