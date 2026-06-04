# BUT Morphology Analysis 10s

Protocol: BUT 10s P1, expert consensus 1/2/3 mapped to good/medium/bad. Synthetic candidates are PTB-derived only; BUT test is not used for training or threshold selection.

## Key Findings
- Strict b10 anchor remains the reference: acc `0.7735`, balanced `0.8045`, macro-F1 `0.7238`, recalls `0.824/0.724/0.866`.
- Best completed large-grid macro row is `medium_qrs_visible_family_13`: macro-F1 `0.7656`, balanced `0.7397`, recalls `0.970/0.683/0.567`.
- Closest morphology-distance row is `bad_qrs_unreliable_family_06` with overall distance `0.2998` and macro-F1 `0.7318`.
- Overall morphology distance vs macro-F1 Spearman is `-0.280` over `99` rows, so this proxy currently has a `weak` relationship with external performance.

## Top By BUT Macro-F1
| variant | family | acc | balanced | macro | G/M/B recall | morph distance |
|---|---|---:|---:|---:|---|---:|
| `medium_qrs_visible_family_13` | medium_qrs_visible_family | 0.8002 | 0.7397 | 0.7656 | 0.970/0.683/0.567 | 0.3021 |
| `bad_qrs_unreliable_family_06` | bad_qrs_unreliable_family | 0.7589 | 0.7721 | 0.7318 | 0.993/0.566/0.757 | 0.2998 |
| `medium_qrs_visible_family_09` | medium_qrs_visible_family | 0.7227 | 0.7737 | 0.7289 | 0.890/0.573/0.859 | 0.3017 |
| `b10_all_bad_wearable` | bad_boundary_10s | 0.7735 | 0.8045 | 0.7238 | 0.824/0.724/0.866 | 0.3072 |
| `medium_qrs_visible_family_07` | medium_qrs_visible_family | 0.7647 | 0.8048 | 0.7234 | 0.912/0.634/0.869 | 0.3026 |
| `m07_bad_extreme_medium_protected` | medium_bad_generator_10s | 0.7336 | 0.7708 | 0.7154 | 0.983/0.521/0.808 | 0.3155 |
| `medium_qrs_visible_family_05` | medium_qrs_visible_family | 0.7759 | 0.8036 | 0.7124 | 0.945/0.631/0.835 | 0.3016 |
| `mix02_medium_contact_short` | medium_mixture_generator_10s | 0.7518 | 0.7747 | 0.7122 | 0.864/0.654/0.805 | 0.3071 |

## Closest Synthetic Morphology To BUT
| variant | family | morph distance | qrs | ptst | contact | macro | G/M/B recall |
|---|---|---:|---:|---:|---:|---:|---|
| `bad_qrs_unreliable_family_06` | bad_qrs_unreliable_family | 0.2998 | 0.2475 | 0.3428 | 0.2558 | 0.7318 | 0.993/0.566/0.757 |
| `s04_contact_bad_medium_visible` | morph_sweet_generator_10s | 0.3011 | 0.2500 | 0.3453 | 0.2598 | 0.6840 | 0.854/0.644/0.839 |
| `bad_qrs_unreliable_family_10` | bad_qrs_unreliable_family | 0.3012 | 0.2516 | 0.3412 | 0.2624 | 0.5821 | 0.878/0.579/0.255 |
| `medium_qrs_visible_family_14` | medium_qrs_visible_family | 0.3014 | 0.2490 | 0.3459 | 0.2602 | 0.6735 | 0.995/0.499/0.701 |
| `s06_medium_local_morph_bad_hard` | morph_sweet_generator_10s | 0.3014 | 0.2487 | 0.3470 | 0.2604 | 0.5523 | 0.945/0.395/0.353 |
| `medium_qrs_visible_family_05` | medium_qrs_visible_family | 0.3016 | 0.2492 | 0.3466 | 0.2602 | 0.7124 | 0.945/0.631/0.835 |
| `medium_qrs_visible_family_09` | medium_qrs_visible_family | 0.3017 | 0.2493 | 0.3464 | 0.2605 | 0.7289 | 0.890/0.573/0.859 |
| `r02_s02_less_bad_pressure` | morph_refine_generator_10s | 0.3017 | 0.2489 | 0.3442 | 0.2641 | 0.6142 | 0.992/0.376/0.871 |

## Interpretation
- BUT medium is best described as QRS-visible but locally unreliable: P/T/ST and short baseline/contact events matter more than simple SNR.
- BUT bad requires QRS detectability failure or QRS-confounding events. Rules that only strengthen flatline/low-amplitude often hurt medium or create unstable bad recall.
- Rows that improve macro-F1 without improving morphology distance are likely exploiting calibration/head bias rather than truly matching BUT waveform morphology.

## Artifacts
- BUT feature table: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_morphology_analysis_10s_2026_06_03\but_morph_features.csv`
- Synthetic feature table: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_morphology_analysis_10s_2026_06_03\synthetic_morph_features.csv`
- Distance join: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_morphology_analysis_10s_2026_06_03\grid_metric_distance_join.csv`
- BUT class gallery: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_morphology_analysis_10s_2026_06_03\visuals\but_class_profiles.png`
- Distance scatter: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_morphology_analysis_10s_2026_06_03\visuals\distance_vs_metrics.png`
- Nearest/farthest gallery: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_morphology_analysis_10s_2026_06_03\visuals\synthetic_nearest_farthest.png`
