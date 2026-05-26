# PTB-XL Strict Clean Filter Audit

Purpose: verify whether the transformer clean-source pool was polluted by PTB-XL records with non-Lead-I noise/electrode annotations.

## Filter Counts

- total PTB-XL records: `21799`
- records rejected by old Lead-I-only filter: `2755`
- records kept by old Lead-I-only filter: `19044`
- records rejected by strict any-lead filter: `5010`
- records kept by strict any-lead filter: `16789`
- extra records previously kept but rejected by strict filter: `2255`

## Probe Records

| ecg_id | baseline_drift | static_noise | burst_noise | electrodes_problems | old kept | strict kept | figure |
| ---: | --- | --- | --- | --- | ---: | ---: | --- |
| 5447 |  |  , I-AVF,   |  |  | False | False | `outputs/ptbxl_clean_filter_audit/ptbxl_ecg_5447_12lead.png` |
| 5484 |  |  |  |  | True | True | `outputs/ptbxl_clean_filter_audit/ptbxl_ecg_5484_12lead.png` |
| 5428 |  |  |  |  | True | True | `outputs/ptbxl_clean_filter_audit/ptbxl_ecg_5428_12lead.png` |

## Interpretation

- `5447` has `static_noise = , I-AVF,`, so it is rejected by both the old Lead-I-only filter and the new strict any-lead filter.
- `5428` and `5484` have no annotation in `baseline_drift/static_noise/burst_noise/electrodes_problems`; strict metadata filtering alone will still keep them.
- Therefore strict any-lead filtering fixes the known non-Lead-I annotation leakage, but visually bad unannotated records need a second visual/proxy-clean guard if they remain unacceptable.
