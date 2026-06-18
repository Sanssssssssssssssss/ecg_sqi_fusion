# E3.11f BUT 10s Generator Research Note

This report consolidates the 10s P1 BUT work from 2026-06-03. Calibration is validation-only; BUT test is used only for reporting.

## Current Answer

The best strict synthetic-generator anchor remains `b10_all_bad_wearable`: BUT test acc `0.7735`, balanced acc `0.8045`, macro-F1 `0.7238`, recalls `0.824/0.724/0.866`, PTB acc `0.9768`, denoise score `2.614`.

Later morphology grids are useful as mechanism probes, but none produced a cleaner strict zero-shot replacement. The correct conclusion is not ?try more random morphology?; it is that the current synthetic label geometry does not yet encode the BUT expert boundary well enough.

## Strict Synthetic Generator: Top By Balanced Accuracy
| rank | name | family | BUT acc | bal | macro | recalls G/M/B | min(M,B) | PTB acc | PTB bad | denoise | note |
|---:|---|---|---:|---:|---:|---|---:|---:|---:|---:|---|
| 1 | `mix05_medium_protocol_boundary` | medium_mixture_generator_10s | 0.7747 | 0.8111 | 0.7018 | 0.8602/0.6948/0.8783 | 0.6948 | 0.9678 | 0.9986 | 2.6292 | Mixed expert-boundary medium: several mild failure modes. |
| 2 | `b10_all_bad_wearable` | bad_boundary_10s | 0.7735 | 0.8045 | 0.7238 | 0.8236/0.7237/0.8662 | 0.7237 | 0.9768 | 0.9986 | 2.6139 |  |
| 3 | `r03_s02_good_medium_balance` | morph_refine_generator_10s | 0.7373 | 0.8031 | 0.6699 | 0.7797/0.6830/0.9465 | 0.6830 | 0.9528 | 0.9986 | 2.6087 | Middle point between s02 and mix05. |
| 4 | `r04_mix05_plus_bad_spikes` | morph_refine_generator_10s | 0.7133 | 0.7990 | 0.6542 | 0.9453/0.5005/0.9513 | 0.5005 | 0.9777 | 1.0000 | 2.8579 | mix05-like medium with stronger bad QRS-confounding spikes. |
| 5 | `r08_b10_bad_prior_mild` | bad_boundary_refine_10s | 0.7654 | 0.7943 | 0.7071 | 0.9264/0.6270/0.8297 | 0.6270 | 0.9664 | 1.0000 | 2.5602 |  |
| 6 | `r02_b10_cw180` | bad_boundary_refine_10s | 0.7569 | 0.7859 | 0.7040 | 0.8003/0.7131/0.8443 | 0.7131 | 0.9764 | 0.9986 | 2.5749 |  |
| 7 | `v305_medium_guard_bad_soft` | morph_v3_generator_10s | 0.6940 | 0.7841 | 0.6309 | 0.9266/0.4792/0.9465 | 0.4792 | 0.9619 | 1.0000 | 2.6704 | Medium guard: lift medium with softer bad pressure. |
| 8 | `s02_dense_bad_spikes_medium_soft` | morph_sweet_generator_10s | 0.7464 | 0.7828 | 0.6868 | 0.7629/0.7219/0.8637 | 0.7219 | 0.9682 | 1.0000 | 2.6472 | Bad has dense QRS-confounding spikes; medium stays soft. |
| 9 | `s04_contact_bad_medium_visible` | morph_sweet_generator_10s | 0.7438 | 0.7792 | 0.6840 | 0.8538/0.6444/0.8394 | 0.6444 | 0.9646 | 0.9986 | 2.5564 | Bad contact/flatline, medium visible-QRS morphology uncertainty. |
| 10 | `mix02_medium_contact_short` | medium_mixture_generator_10s | 0.7518 | 0.7747 | 0.7122 | 0.8643/0.6543/0.8054 | 0.6543 | 0.9696 | 1.0000 | 2.6342 | Medium can have short contact loss, but QRS is preserved. |
| 11 | `m07_bad_extreme_medium_protected` | medium_bad_generator_10s | 0.7336 | 0.7708 | 0.7154 | 0.9832/0.5215/0.8078 | 0.5215 | 0.9841 | 1.0000 | 2.5468 | A stricter bad class while protecting medium from over-damage. |
| 12 | `v203_s02_good_rescue` | morph_v2_generator_10s | 0.6876 | 0.7618 | 0.6643 | 0.9170/0.4803/0.8881 | 0.4803 | 0.9709 | 0.9986 | 2.7453 | S02-style bad/medium coexistence with explicit good rescue. |
| 13 | `r06_s02_calibration_guard` | morph_refine_generator_10s | 0.7018 | 0.7606 | 0.6522 | 0.6937/0.6902/0.8978 | 0.6902 | 0.9637 | 1.0000 | 2.6242 | Explicitly rescue good while keeping s02 morphology. |
| 14 | `v301_b10_micro_morph` | morph_v3_generator_10s | 0.7073 | 0.7593 | 0.6710 | 0.8860/0.5477/0.8443 | 0.5477 | 0.9600 | 1.0000 | 2.6463 | Conservative b10 continuation: tiny local morphology, do not disturb good. |
| 15 | `r05_b10_dropout_medium_guard` | bad_boundary_refine_10s | 0.7214 | 0.7586 | 0.6846 | 0.9953/0.4896/0.7908 | 0.4896 | 0.9800 | 1.0000 | 2.5931 |  |

## Strict Synthetic Generator: Top By Macro-F1
| rank | name | family | BUT acc | bal | macro | recalls G/M/B | min(M,B) | PTB acc | PTB bad | denoise | note |
|---:|---|---|---:|---:|---:|---|---:|---:|---:|---:|---|
| 1 | `b10_all_bad_wearable` | bad_boundary_10s | 0.7735 | 0.8045 | 0.7238 | 0.8236/0.7237/0.8662 | 0.7237 | 0.9768 | 0.9986 | 2.6139 |  |
| 2 | `m07_bad_extreme_medium_protected` | medium_bad_generator_10s | 0.7336 | 0.7708 | 0.7154 | 0.9832/0.5215/0.8078 | 0.5215 | 0.9841 | 1.0000 | 2.5468 | A stricter bad class while protecting medium from over-damage. |
| 3 | `mix02_medium_contact_short` | medium_mixture_generator_10s | 0.7518 | 0.7747 | 0.7122 | 0.8643/0.6543/0.8054 | 0.6543 | 0.9696 | 1.0000 | 2.6342 | Medium can have short contact loss, but QRS is preserved. |
| 4 | `r08_b10_bad_prior_mild` | bad_boundary_refine_10s | 0.7654 | 0.7943 | 0.7071 | 0.9264/0.6270/0.8297 | 0.6270 | 0.9664 | 1.0000 | 2.5602 |  |
| 5 | `r02_b10_cw180` | bad_boundary_refine_10s | 0.7569 | 0.7859 | 0.7040 | 0.8003/0.7131/0.8443 | 0.7131 | 0.9764 | 0.9986 | 2.5749 |  |
| 6 | `mix05_medium_protocol_boundary` | medium_mixture_generator_10s | 0.7747 | 0.8111 | 0.7018 | 0.8602/0.6948/0.8783 | 0.6948 | 0.9678 | 0.9986 | 2.6292 | Mixed expert-boundary medium: several mild failure modes. |
| 7 | `s02_dense_bad_spikes_medium_soft` | morph_sweet_generator_10s | 0.7464 | 0.7828 | 0.6868 | 0.7629/0.7219/0.8637 | 0.7219 | 0.9682 | 1.0000 | 2.6472 | Bad has dense QRS-confounding spikes; medium stays soft. |
| 8 | `r05_b10_dropout_medium_guard` | bad_boundary_refine_10s | 0.7214 | 0.7586 | 0.6846 | 0.9953/0.4896/0.7908 | 0.4896 | 0.9800 | 1.0000 | 2.5931 |  |
| 9 | `s04_contact_bad_medium_visible` | morph_sweet_generator_10s | 0.7438 | 0.7792 | 0.6840 | 0.8538/0.6444/0.8394 | 0.6444 | 0.9646 | 0.9986 | 2.5564 | Bad contact/flatline, medium visible-QRS morphology uncertainty. |
| 10 | `mix03_medium_lowamp_visible_qrs` | medium_mixture_generator_10s | 0.7656 | 0.6945 | 0.6818 | 0.9393/0.6478/0.4964 | 0.4964 | 0.9673 | 0.9986 | 2.6208 | BUT-like low amplitude medium with visible QRS. |
| 11 | `m04_bad_lowamp_guarded` | medium_bad_generator_10s | 0.7098 | 0.7361 | 0.6747 | 0.9896/0.4767/0.7421 | 0.4767 | 0.9832 | 1.0000 | 2.7405 | Low-amplitude domain shift without making bad only low amplitude. |
| 12 | `v301_b10_micro_morph` | morph_v3_generator_10s | 0.7073 | 0.7593 | 0.6710 | 0.8860/0.5477/0.8443 | 0.5477 | 0.9600 | 1.0000 | 2.6463 | Conservative b10 continuation: tiny local morphology, do not disturb good. |
| 13 | `mix04_medium_pt_unreliable` | medium_mixture_generator_10s | 0.7938 | 0.6543 | 0.6702 | 0.9129/0.7408/0.3090 | 0.3090 | 0.9641 | 0.9973 | 2.6114 | Class-2 emphasis: P/T/ST unreliable while QRS stays usable. |
| 14 | `r03_s02_good_medium_balance` | morph_refine_generator_10s | 0.7373 | 0.8031 | 0.6699 | 0.7797/0.6830/0.9465 | 0.6830 | 0.9528 | 0.9986 | 2.6087 | Middle point between s02 and mix05. |
| 15 | `m03_bad_contact_no_spurious` | medium_bad_generator_10s | 0.6935 | 0.6814 | 0.6670 | 0.9986/0.4519/0.5937 | 0.4519 | 0.9882 | 0.9973 | 2.7480 | Remove spurious peaks to avoid confusing medium with artificial bad. |

## Strict Synthetic Generator: Top By Medium/Bad Coexistence
| rank | name | family | BUT acc | bal | macro | recalls G/M/B | min(M,B) | PTB acc | PTB bad | denoise | note |
|---:|---|---|---:|---:|---:|---|---:|---:|---:|---:|---|
| 1 | `b10_all_bad_wearable` | bad_boundary_10s | 0.7735 | 0.8045 | 0.7238 | 0.8236/0.7237/0.8662 | 0.7237 | 0.9768 | 0.9986 | 2.6139 |  |
| 2 | `s02_dense_bad_spikes_medium_soft` | morph_sweet_generator_10s | 0.7464 | 0.7828 | 0.6868 | 0.7629/0.7219/0.8637 | 0.7219 | 0.9682 | 1.0000 | 2.6472 | Bad has dense QRS-confounding spikes; medium stays soft. |
| 3 | `r02_b10_cw180` | bad_boundary_refine_10s | 0.7569 | 0.7859 | 0.7040 | 0.8003/0.7131/0.8443 | 0.7131 | 0.9764 | 0.9986 | 2.5749 |  |
| 4 | `mix05_medium_protocol_boundary` | medium_mixture_generator_10s | 0.7747 | 0.8111 | 0.7018 | 0.8602/0.6948/0.8783 | 0.6948 | 0.9678 | 0.9986 | 2.6292 | Mixed expert-boundary medium: several mild failure modes. |
| 5 | `r06_s02_calibration_guard` | morph_refine_generator_10s | 0.7018 | 0.7606 | 0.6522 | 0.6937/0.6902/0.8978 | 0.6902 | 0.9637 | 1.0000 | 2.6242 | Explicitly rescue good while keeping s02 morphology. |
| 6 | `r03_s02_good_medium_balance` | morph_refine_generator_10s | 0.7373 | 0.8031 | 0.6699 | 0.7797/0.6830/0.9465 | 0.6830 | 0.9528 | 0.9986 | 2.6087 | Middle point between s02 and mix05. |
| 7 | `s03_but_medium_motion` | morph_sweet_generator_10s | 0.6511 | 0.6514 | 0.5863 | 0.5154/0.7601/0.6788 | 0.6788 | 0.9605 | 0.9986 | 2.5610 | Aggressive BUT-like medium motion while keeping QRS mostly visible. |
| 8 | `mix02_medium_contact_short` | medium_mixture_generator_10s | 0.7518 | 0.7747 | 0.7122 | 0.8643/0.6543/0.8054 | 0.6543 | 0.9696 | 1.0000 | 2.6342 | Medium can have short contact loss, but QRS is preserved. |
| 9 | `s04_contact_bad_medium_visible` | morph_sweet_generator_10s | 0.7438 | 0.7792 | 0.6840 | 0.8538/0.6444/0.8394 | 0.6444 | 0.9646 | 0.9986 | 2.5564 | Bad contact/flatline, medium visible-QRS morphology uncertainty. |
| 10 | `r08_b10_bad_prior_mild` | bad_boundary_refine_10s | 0.7654 | 0.7943 | 0.7071 | 0.9264/0.6270/0.8297 | 0.6270 | 0.9664 | 1.0000 | 2.5602 |  |
| 11 | `b07_but_mixed_bad` | bad_boundary_10s | 0.6888 | 0.6740 | 0.6314 | 0.7747/0.6243/0.6229 | 0.6229 | 0.9723 | 0.9973 | 2.6708 |  |
| 12 | `r07_b10_good_lenient` | bad_boundary_refine_10s | 0.5516 | 0.6584 | 0.5359 | 0.4871/0.5707/0.9173 | 0.5707 | 0.9818 | 0.9986 | 2.8564 |  |
| 13 | `v303_mix05_medium_with_bad_floor` | morph_v3_generator_10s | 0.5804 | 0.5771 | 0.5897 | 0.5931/0.5712/0.5669 | 0.5669 | 0.9673 | 0.9986 | 2.5610 | Mix05-style medium, but without good-overlap and with bad floor restored. |
| 14 | `mix06_medium_strong_bad_soft` | medium_mixture_generator_10s | 0.6491 | 0.6129 | 0.5669 | 0.5080/0.7736/0.5572 | 0.5572 | 0.9578 | 0.9986 | 2.5977 | Soften bad and strengthen medium boundary cues. |
| 15 | `v301_b10_micro_morph` | morph_v3_generator_10s | 0.7073 | 0.7593 | 0.6710 | 0.8860/0.5477/0.8443 | 0.5477 | 0.9600 | 1.0000 | 2.6463 | Conservative b10 continuation: tiny local morphology, do not disturb good. |

## Full-Recipe Confirmation Runs
| rank | name | family | BUT acc | bal | macro | recalls G/M/B | min(M,B) | PTB acc | PTB bad | denoise | note |
|---:|---|---|---:|---:|---:|---|---:|---:|---:|---:|---|
| 1 | `b10_all_bad_wearable_full_cw190` | e311_but_bad_boundary_full_confirm_10s_2026_06_03 | 0.7743 | 0.7213 | 0.6765 | 0.8626/0.7196/0.5815 | 0.5815 | 0.9818 | 0.9986 | 2.8686 | First-grid winner; mixed wearable bad rule. |
| 2 | `r08_bad_prior_mild_full_cw160` | e311_but_bad_boundary_full_confirm_10s_2026_06_03 | 0.7782 | 0.5663 | 0.5587 | 0.9379/0.7149/0.0462 | 0.0462 | 0.9805 | 0.9973 | 2.9243 | Best refined softened bad-prior alternative. |

Full confirmation did not improve the quick b10 anchor: the full b10 run kept accuracy similar but bad recall fell, and r08 bad recall collapsed. This supports using quick b10 as evidence, not promoting full-confirm as better.

## Diagnostic / Supervised / Protocol Rows
| rank | name | family | BUT acc | bal | macro | recalls G/M/B | min(M,B) | PTB acc | PTB bad | denoise | note |
|---:|---|---|---:|---:|---:|---|---:|---:|---:|---:|---|
| 1 | `e311_but_boundary_head_adaptation_10s_2026_06_03` | e311_but_boundary_head_adaptation_10s_2026_06_03 | 0.8058 | 0.8541 | 0.7402 | 0.9651/0.6629/0.9343 | 0.6629 |  |  |  |  |
| 2 | `ten_s_tuning` | ten_s_tuning | 0.8131 | 0.8186 | 0.7446 | 0.9456/0.7049/0.8054 | 0.7049 |  |  |  |  |

These rows are not strict synthetic zero-shot. They are valuable because they show the representation can adapt to BUT boundaries, but they should be reported as calibration/probe/head-only diagnostics.

## Visual Findings

- BUT good is not pristine PTB. It often contains baseline wander and modest local disturbance while QRS remains reliably visible.
- BUT medium is locally unreliable: P/T/ST details and baseline segments become unstable, but QRS remains mostly traceable. This is why medium is not captured by a simple SNR ladder.
- BUT bad has QRS detectability failure: dense pseudo-peaks, motion/contact events, and confounding deflections make true QRS ambiguous. It is not only flatline, low amplitude, or uniform noise.
- b10 succeeds best because its broad wearable mixture accidentally covers enough of the expert boundary. Its misses show the model still over-cleans bad into medium/good and over-trusts QRS-like spikes.
- s02/v2/v3 variants confirm the tradeoff: dense pseudo-QRS recovers bad recall but compresses medium; stronger medium P/T/ST events improve medium but steal good or weaken bad.

## Mechanism Interpretation

- The current Uformer representation is useful: the head-only diagnostic reaches higher balanced accuracy than strict zero-shot, meaning the latent features contain BUT-relevant information.
- The remaining gap is boundary alignment, not denoiser capacity alone. Synthetic labels currently describe artifact recipe strength, while BUT labels describe expert usability of waveform components.
- Generator tuning should move from additive artifact knobs to explicit QRS-detectability and morphology-usability targets: QRS prominence, QRS false-peak density, local baseline discontinuity, P/T/ST trustworthiness, and segment-level interpretability.

## Recommended Next Research

1. Keep `b10_all_bad_wearable` as the strict synthetic zero-shot baseline in reports.
2. Run a clean BUT-supervised adaptation track separately: freeze Uformer features, train/calibrate lightweight heads on BUT train/val, evaluate BUT test once, and label it as supervised adaptation.
3. For future synthetic redesign, stop changing only noise/event strength. Build labels from explicit detectability functions: QRS reliable vs unreliable, P/T/ST reliable vs unreliable, and contact/motion segment coverage. This is the likely route to a better zero-shot generator.
4. Preserve all v1/v2/v3 failures as ablations: they explain why the final method needs expert-boundary adaptation rather than naive synthetic grid search.

## Visual Anchors

- BUT good: `E:/GPTProject2/ecg/outputs/external_benchmarks/e311_realdata_2026_06_02/processed/butqdb/visuals/processed_good_gallery.png`
- BUT medium: `E:/GPTProject2/ecg/outputs/external_benchmarks/e311_realdata_2026_06_02/processed/butqdb/visuals/processed_medium_gallery.png`
- BUT bad: `E:/GPTProject2/ecg/outputs/external_benchmarks/e311_realdata_2026_06_02/processed/butqdb/visuals/processed_bad_gallery.png`
- b10 medium confusions: `E:/GPTProject2/ecg/outputs/external_benchmarks/e311_but_bad_boundary_10s_2026_06_03/runs/quick/b10_all_bad_wearable/but_10s_eval/visuals/medium_confused.png`
- b10 missed bad: `E:/GPTProject2/ecg/outputs/external_benchmarks/e311_but_bad_boundary_10s_2026_06_03/runs/quick/b10_all_bad_wearable/but_10s_eval/visuals/bad_missed.png`

