# PTB-XL Strict Clean Filter Audit

Purpose: verify whether the transformer clean-source pool was polluted by PTB-XL records with non-Lead-I noise/electrode annotations.

## Filter Counts

- total PTB-XL records: `21799`
- records rejected by old Lead-I-only filter: `2755`
- records kept by old Lead-I-only filter: `19044`
- records rejected by strict any-lead filter: `5010`
- records kept by strict any-lead filter: `16789`
- extra records previously kept but rejected by strict filter: `2255`
- manual visual clean-source exclusions: `2`
- strict + manual records kept: `16787`

## Probe Records

| ecg_id | baseline_drift | static_noise | burst_noise | electrodes_problems | old kept | metadata-strict kept | final strict+manual kept | figure |
| ---: | --- | --- | --- | --- | ---: | ---: | ---: | --- |
| 5447 |  |  , I-AVF,   |  |  | False | False | False | `outputs/ptbxl_clean_filter_audit/ptbxl_ecg_5447_12lead.png` |
| 5484 |  |  |  |  | True | True | False | `outputs/ptbxl_clean_filter_audit/ptbxl_ecg_5484_12lead.png` |
| 5428 |  |  |  |  | True | True | False | `outputs/ptbxl_clean_filter_audit/ptbxl_ecg_5428_12lead.png` |

## Interpretation

- `5447` has `static_noise = , I-AVF,`, so it is rejected by both the old Lead-I-only filter and the new strict any-lead filter.
- `5428` and `5484` have no annotation in `baseline_drift/static_noise/burst_noise/electrodes_problems`; strict metadata filtering alone will still keep them.
- `5428` and `5484` are now listed in `config/ptbxl_clean_source_exclude_ecg_ids.txt` so they are removed from the synthetic clean-source pool despite missing PTB-XL noise metadata.
- Therefore strict any-lead filtering fixes the known non-Lead-I annotation leakage, and the manual exclusion list covers visually unacceptable unannotated records found during audit.

## Synthetic Dataset Verification

After regenerating the E3.11 mainline data with strict+manual filtering, the synthetic label CSVs were checked by the actual `ecg_id` column, not by internal segment/source indices:

| dataset | unique clean-source ecg_ids | any PTB-XL noise/electrode metadata hits | ecg_id hits for 5428/5447/5484 |
| --- | ---: | ---: | ---: |
| `e311f_lite_e310_morph` | 5107 | 0 | 0 |
| `e311h_lite_relaxed_morph` | 5581 | 0 | 0 |
| `e311i_wide_relaxed_morph` | 5597 | 0 | 0 |

Note: these numbers may still appear in fields such as `seg_id`, `source_npz_index`, or `counterfactual_group`; those are synthetic/internal identifiers and are not PTB-XL `ecg_id`.

## Active Source Artifact Check

- The legacy source artifact `outputs/transformer_source` still contains `5428` and `5484` because it was built before the manual visual exclusions.
- The active E3.11 source artifact is `outputs/transformer_source_strict_clean`; it contains none of `5428`, `5447`, or `5484`.
- Current E3.11 labels should therefore be interpreted by the `ecg_id` column only. Internal fields such as `seg_id`, `source_npz_index`, and `counterfactual_group` can numerically match PTB-XL IDs by coincidence.

## Remaining Non-Noise Metadata

The strict filter removes PTB-XL records with any annotation in:

- `baseline_drift`
- `static_noise`
- `burst_noise`
- `electrodes_problems`

It does not currently remove rhythm/morphology metadata such as `extra_beats`, `pacemaker`, or unvalidated report flags. In the active strict source pool:

| metadata condition | records in strict source | records used by E3.11f |
| --- | ---: | ---: |
| `extra_beats` non-empty | 1405 | 446 |
| `pacemaker` non-empty | 231 | 71 |
| `validated_by_human = False` | 4023 | 1262 |

These are not lead-noise annotations, but they can explain some visually awkward clean-source examples. If the next visual pass still finds unacceptable clean morphologies, the smallest follow-up data patch should be a stricter source pool that excludes `pacemaker` first, and only then considers excluding `extra_beats`.
