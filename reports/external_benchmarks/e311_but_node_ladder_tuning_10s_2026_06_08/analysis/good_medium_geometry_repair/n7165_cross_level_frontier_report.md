# N7165 Cross-Level Frontier Report

## Executive Summary

N7165 block retraining did not promote, even after wide-focus and direct high-flatline gatefocus variants. The important breakthrough is that the previously promoted ordinary checkpoint from N7155 already generalizes across the larger node manifests:

- `nl_n7155_gm_trim_bad_boundaryblocks_breakthrough_softbala_5c50f0ee6d7a`
- best mode on N7165/N7200 manifests: `medium_guarded_pmed001`
- N7200 clean-node diagnostic replay: acc `0.963211`, good/medium/bad recall `0.980972/0.941250/0.970617`

This means the current best ordinary-checkpoint frontier should be the N7155 checkpoint reused across the N7160/N7165/N7175/N7200 manifests, not the later retrained checkpoints. Later N7160/N7165 retraining disturbed the good/medium boundary rather than improving coverage.

## Why N7165 Retraining Failed

The N7160 to N7165 ring added only 10 clean-node windows: 5 good and 5 medium. Nine are `good_medium_overlap`; one is `clean_core`; no bad rows were added.

Despite that tiny ring, retraining variants shifted the full good/medium decision surface:

- Best N7165 retrained checkpoint: acc `0.944988`, good/medium/bad `0.956595/0.918772/0.970617`
- N7155 checkpoint replayed on N7165: acc `0.963397`, good/medium/bad `0.981158/0.941521/0.970617`

The failure is not insufficient data volume. It is training instability around the good/medium overlap shell.

## Experiments Run

Wide-focus N7165 variants enlarged the local good/medium boundary blocks. They looked promising in quick split, but failed on full node diagnostic.

Gatefocus N7165 variants directly targeted the endpoint-disagreement high-flatline shell. Quick metrics were strong, especially `gatefocus_wideflat`, but formal node diagnostic again collapsed into good-heavy or medium-heavy endpoints.

Cross-level checkpoint replay then showed the better answer: preserve the stable N7155 checkpoint and evaluate it on larger manifests instead of retraining.

## Original Bucketed Report-Only

N7155 raw mode is strongest on `original_all_10s+`: acc `0.859540`, macro-F1 `0.872301`, good/medium/bad recall `0.884762/0.794599/0.908798`.

N7155 `medium_guarded_pmed001` is strongest on clean-node replay and reaches `original_all_10s+` acc `0.835022`, with good/medium/bad recall `0.775392/0.893959/0.908798`.

The original test split still has a severe bad-domain issue:

- `original_test_all_10s+`, raw: acc `0.799457`, bad recall `0.000000`
- `original_test_bad_core_near_boundary`, raw: 119 bad windows, all predicted medium
- `original_all_bad_core_near_boundary`, raw: bad recall `0.970617`
- `original_all_bad_outlier_stress`, raw: bad recall `0.698585`

So bad is not globally zero. The original test bad bucket is a distinct domain slice and should be analyzed separately. Original remains report-only and is not used for selection.

## Next Move

Treat N7155 as the ordinary-checkpoint frontier through N7200 for Clean/SemiClean/node diagnostics. The next useful work is not another broad N7165 retrain; it is:

1. preserve N7155 as the best normal checkpoint for N7160/N7165/N7175/N7200 manifests;
2. analyze original-test bad core/stress waveforms as a separate report-only domain gap;
3. if training continues, use very small adapter-style or regularization-preserving runs from N7155, not full boundary rebalancing from N7160.

Key files:

- `cross_level_frontier_checkpoint_metrics.csv`
- `n7160_to_n7165_added_ring_rows.csv`
- `n7160_to_n7165_added_ring_feature_gap.csv`
- `n7165_endpoint_gate_search_top100.csv`
- original bucketed reports under `analysis/original_bucketed_checkpoint`
