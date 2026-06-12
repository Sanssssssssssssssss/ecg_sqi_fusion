# Boundary Blocks N7165 Balance Failure Report

Generated: 2026-06-12 03:56 local.

## Summary

N7165 did not promote after two conservative boundary-block attempts from the N7160 promoted base. Bad remains stable; the failure is entirely a good/medium geometry limit.

Current ordinary checkpoint frontier remains:

| frontier | variant | mode | acc | good | medium | bad |
|---|---|---:|---:|---:|---:|---:|
| N7160 | `nl_n7160_gm_trim_bad_boundaryblocks_breakthrough_mediumgu_35ddcfe3c52a` | `medium_guarded_pmed001` | 0.951315 | 0.962849 | 0.928631 | 0.970862 |

Best N7165 balance retry:

| node | variant | mode | acc | good | medium | bad |
|---|---|---:|---:|---:|---:|---:|
| N7165 | `nl_n7165_gm_trim_bad_boundaryblocks_balance_goodprotect_n_bb93f422c517` | calibrated | 0.944988 | 0.956595 | 0.918772 | 0.970617 |

This misses the promotion gate on accuracy only; class recalls are above the per-class gates. The problem is both good->medium and medium->good errors increasing at the same time relative to N7160.

## Confusion Comparison

N7160 promoted:

```text
[[6894, 266, 0],
 [ 511, 6649, 0],
 [   0, 119, 3965]]
```

N7165 best balance:

```text
[[6854, 311, 0],
 [ 582, 6583, 0],
 [   1, 119, 3964]]
```

Delta from N7160 to N7165:

- good->medium errors increase by 45;
- medium->good errors increase by 71;
- bad remains effectively unchanged.

## What We Learned

The first N7165 attempt split into bad endpoints:

- mediumguard became medium-heavy and lost too many good rows;
- softbalance became good-heavy and lost too many medium rows.

The second N7165 balance attempt landed between those endpoints, but still worse than N7160 on both overlap directions. That means the N7160->N7165 added shell is not just a weighting issue. The added rows likely mix two visually similar good/medium subtypes whose separating dimensions are not represented cleanly by the current blocks.

## Visual Artifacts

Best N7165 balance candidate:

![N7165 balance PCA](E:/GPTProject2/ecg/outputs/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/boundary_blocks_n7165_bb93f422c517_pca.png)

![N7165 balance waveforms](E:/GPTProject2/ecg/outputs/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/boundary_blocks_n7165_bb93f422c517_waveforms.png)

## Next Step

Do not run another blind block-size sweep. The next useful step is a targeted N7160-to-N7165 delta analysis:

- isolate rows added by the N7165 shell that are correctly handled by N7160 versus newly confused by N7165;
- compare raw waveforms, PCA, `qrs_visibility`, `flatline_ratio`, `pc1/pc3`, `non_qrs_diff_p95`, and amplitude/detail features;
- decide whether the N7165 shell should be split into two smaller sub-blocks or held out as an ambiguity/stress shell.

