# Boundary Blocks N7155/N7160 Breakthrough Report

Generated: 2026-06-12 02:35 local.

## Summary

The conservative boundary-block strategy promoted two new ordinary checkpoints beyond the prior N7150 frontier. These are normal neural checkpoints, not transparent rule-mode artifacts.

| node | promoted variant | mode | acc | macro-F1 | good recall | medium recall | bad recall |
|---|---|---:|---:|---:|---:|---:|---:|
| N7150 previous frontier | `nl_n7150_gm_trim_bad_boundaryblocks_micro_bad_probe_n7125_84064417a3c4` | raw | 0.954254 | 0.958757 | 0.958881 | 0.940280 | 0.970617 |
| N7155 new | `nl_n7155_gm_trim_bad_boundaryblocks_breakthrough_softbala_5c50f0ee6d7a` | `medium_guarded_pmed001` | 0.963466 | 0.966588 | 0.981272 | 0.941579 | 0.970617 |
| N7160 new | `nl_n7160_gm_trim_bad_boundaryblocks_breakthrough_mediumgu_35ddcfe3c52a` | `medium_guarded_pmed001` | 0.951315 | 0.956248 | 0.962849 | 0.928631 | 0.970862 |

Promotion gates remain `acc >= 0.95`, good `>= 0.92`, medium `>= 0.90`, bad `>= 0.90`. Original BUT was report-only.

## What Changed

N7155 used the N7150 promoted checkpoint as the base and succeeded with the soft-balance block mix:

- very small good rescue block;
- small visible-QRS medium detail block;
- very-low-QRS medium hard-negative block;
- tiny controlled bad-outlier block;
- no broad bad outlier chase.

N7160 was rebuilt after N7155 promoted, using N7155 as the base rather than the older N7150 base. The promoted N7160 variant is the medium-guard block mix, which retained bad stability and kept good/medium overlap above gate.

## Node Diagnostic Confusions

N7155 best:

```text
[[7021, 134, 0],
 [ 418, 6737, 0],
 [   1, 119, 3964]]
```

N7160 best:

```text
[[6894, 266, 0],
 [ 511, 6649, 0],
 [   0, 119, 3965]]
```

The N7160 step is valid but less comfortable than N7155. It gives back some good/medium overlap margin while still passing all gates.

## Original Bucketed Report-Only

| checkpoint | original all acc | original all good/medium/bad recall | original test acc | original test good/medium/bad recall | original test GM-only acc |
|---|---:|---:|---:|---:|---:|
| N7155 | 0.835022 | 0.775392 / 0.893959 / 0.908798 | 0.781880 | 0.701374 / 0.920696 / 0.000000 | 0.821721 |
| N7160 | 0.827861 | 0.798099 / 0.835152 / 0.909177 | 0.737997 | 0.677473 / 0.856304 / 0.000000 | 0.775601 |

Interpretation: N7160 is the Clean/SemiClean/node frontier, but N7155 transfers better to original-test good/medium. The original-test bad slice still has zero bad recall because that test bad bucket is from a domain-specific slice that the current trim-bad training does not learn. Original all bad core remains strong for both N7155 and N7160 because the all-scope contains the broader core/near-boundary bad population.

## Visual Artifacts

N7155 promoted block overlay:

![N7155 block PCA](E:/GPTProject2/ecg/outputs/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/boundary_blocks_n7155_5c50f0ee6d7a_pca.png)

![N7155 waveform panel](E:/GPTProject2/ecg/outputs/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/boundary_blocks_n7155_5c50f0ee6d7a_waveforms.png)

N7160 promoted block overlay:

![N7160 block PCA](E:/GPTProject2/ecg/outputs/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/boundary_blocks_n7160_35ddcfe3c52a_pca.png)

![N7160 waveform panel](E:/GPTProject2/ecg/outputs/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/boundary_blocks_n7160_35ddcfe3c52a_waveforms.png)

## Next Breakthrough Hypothesis

N7160 passed, but the margin shrank relative to N7155. The next step should not be a broad sweep. Use N7160 as the clean frontier and N7155 as the original-transfer control:

- Try N7165 as a very small bisection from N7160 with a smaller good-rescue block than N7160 and a medium guard no stronger than `pmed001`.
- Keep controlled bad outlier at the same tiny scale or lower; do not add broad bad outliers yet.
- Add an explicit original-test GM-overlap comparison table after each promoted clean node, because N7155 shows that original-transfer can move opposite to clean-node acc.

