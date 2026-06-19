# Controlled Bad-Outlier Regeneration Report - 2026-06-19

## Why This Run

The representative waveform sheet showed the current PTB/BUT `bad` examples were too close to pure noise/contact stress.  That makes `bad` easy on the clean/right-island body but too narrow for controlled outlier morphology.  This run relaxed the bad-outlier policy without fully admitting extreme stress as an ordinary training target.

## Experiment-Only Code Changes

- `build_ptb_bad_waveform_feature_match.py`
  - Added `clean_margin_ge_5s_keep_outlier_*` bad targets.
  - Added controlled outlier shell selection: keep all bad core/near-boundary, then keep the outlier rows closest to bad core in waveform-computable SQI/morphology space.
  - Added `soft_controlled_bad` mode profile to avoid pure high-ZCR/cleanhf/stress collapse.
  - Added `source-classes good,medium,bad` so controlled bad can start from visible ECG morphology, not only already-noisy PTB bad.
  - Added `block_profile` selector to force explicit bad blocks instead of letting nearest matching collapse back into right-island noise.
- `combine_ptb_boundary_banks.py`
  - Added optional `extra-bad-bank` so old bad core can be preserved while adding a small controlled-bad shell.

No `src/sqi_pipeline` or mainline checkpoint was modified.

## New Bad Data

Main controlled bad bank:

- Signals: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_waveform_feature_match\ptb_bad_waveform_feature_match_keepbad5s_train_controlled__a3fb38c6d8bb_signals.npz`
- Manifest: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_waveform_feature_match\ptb_bad_waveform_feature_match_keepbad5s_train_controlled__a3fb38c6d8bb_manifest.csv`
- Figure: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_waveform_feature_match\ptb_bad_waveform_feature_match_keepbad5s_train_controlled__a3fb38c6d8bb_waveforms.png`

Block composition:

| block | rows |
| --- | ---: |
| bad_core_guard | 770 |
| bad_controlled_qrs_visible | 660 |
| bad_controlled_contact | 550 |
| bad_mild_noise_guard | 220 |

Source classes:

| source class | rows |
| --- | ---: |
| medium | 1001 |
| bad | 770 |
| good | 429 |

This fixed the visual failure: generated bad now includes visible-QRS baseline/contact/reset examples, not only pure noise.

## New Combo Banks

| bank | rows | good | medium | bad | note |
| --- | ---: | ---: | ---: | ---: | --- |
| `ptb_combo_keepbad5s_controlled_block_gm_cvfold2_wavefact_v1` | 5600 | 900 | 2500 | 2200 | full controlled bad replacement |
| `ptb_combo_corebad_plus_controlledbad25_gm_cvfold2_wavefact_v1` | 5400 | 900 | 2500 | 2000 | old core bad + 25% controlled shell |
| `ptb_combo_corebad_plus_controlledbad10_gm_cvfold2_wavefact_v1` | 5070 | 900 | 2500 | 1670 | old core bad + 10% controlled shell |

## Keep-Outlier CV

Built a new `margin_ge_5s_keep_outlier` 5-fold CV so bad outlier is actually present in validation/test:

- Report: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\clean_but_protocols\margin_ge_5s_keep_outlier_cv_seed20260619_report.md`
- Fold2 test counts: good 3006, medium 1844, bad 1033.

## Training Results

All runs are waveform-only Transformer-family `dualview_convtx_hier` with PTB synthetic training and BUT CV report.  Original/BUT rows are not used inside PTB synthetic training; for CV runs, BUT val is only used when the selection mode explicitly says `clean_val_joint`.

| setup | BUT policy | acc | good R | medium R | bad R | interpretation |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| previous v3 core-bad baseline | drop-outlier fold2 | 0.9349 | 0.9358 | 0.8908 | 1.0000 | best clean-body baseline, bad too narrow visually |
| full controlled block, replay 0.35 | keep-outlier fold2 | 0.7794 | 0.7409 | 0.7381 | 0.9652 | bad learned, good/medium collapse |
| full controlled block, replay 0.35 | drop-outlier fold2 | 0.8519 | 0.7643 | 0.9124 | 1.0000 | too much good->medium shift |
| full controlled block, replay 0.18 | drop-outlier fold2 | 0.8821 | 0.8641 | 0.9028 | 0.8996 | less harmful, still below baseline |
| core + controlled25, clean-val select | drop-outlier fold2 | 0.9038 | 0.8971 | 0.8645 | 0.9829 | balanced but still below baseline |
| core + controlled10, replay 0.5 | drop-outlier fold2 | 0.8837 | 0.8209 | 0.9203 | 1.0000 | stress improves, good recall falls |
| core + controlled10, replay 0.5 | keep-outlier fold2 | 0.8287 | 0.8167 | 0.7771 | 0.9555 | bad retained, G/M outlier boundary dominates |

Bias calibration on validation helps only modestly.  Best calibrated controlled result in this batch reaches about 0.916 on drop-outlier test, so this is not merely a threshold shift.

## Conclusion

Relaxing bad outlier was the right diagnostic move:

- The generated bad data is no longer visually pure noise.
- The model can now recognize controlled/stress bad much better; keep-outlier bad recall is around 0.95 and internal stress-val bad recall can reach about 0.90.

But treating controlled bad outlier as ordinary `bad` label damages the good/medium boundary.  The new bottleneck is not bad recall anymore; it is the interaction between a wider bad shell and ambiguous good/medium windows.

Recommended next step:

1. Keep old core bad as the ordinary `bad` body.
2. Keep controlled bad as a small auxiliary/stress factor or curriculum slice, not a large ordinary label block.
3. Add matched non-bad hard negatives with the same baseline/contact artifacts so the Transformer learns "artifact morphology" without turning all borderline good/medium into bad/medium.
4. Continue using keep-outlier CV as a stress protocol, but do not replace the clean-body target with full keep-outlier until good/medium outlier labels are separately cleaned or modeled.
