# Boundary Blocks Iteration Summary - 2026-06-11

Selection used Clean/SemiClean/node diagnostic only. Original BUT remains report-only.

## Implemented

- Added `boundary_blocks_*` stages to `run_good_medium_geometry_repair.py`.
- Added fixed blocks:
  - `gm_clean_overlap_body`
  - `gm_good_rescue_pc1flat_qrsvisible`
  - `gm_medium_qrslow_hardneg`
  - `gm_visible_qrs_medium_detail`
  - `bad_core_guard`
  - `bad_controlled_outlier`
  - `bad_extreme_stress_holdout` remains report-only by policy.
- Added PCA overlays, clean/noisy waveform panels, block/source manifests, and feature summaries.
- Switched default boundary block configs to level-specific micro blocks:
  - N7110 starts from `nl_n7110_gm_trim_bad_geom_directrule_n7100base_g003_m008__69ab5b71cf7d`.
  - N7125 starts from `nl_n7125_gm_trim_bad_geom_addedring_n7100base_g004_m014_g_e48e7d59927b`.

## Diagnostics

The first wide block pass was too disruptive:

- Best new N7110 block: `boundaryblocks_badstress_guard_only`, calibrated, acc `0.940286`, good/medium/bad `0.888889/0.974121/0.970862`.
- Best new N7125 block: `boundaryblocks_good_rescue_lightbad`, raw, acc `0.916658`, good/medium/bad `0.896702/0.905684/0.970617`.

The level-specific micro block pass improved stability but still did not beat the old N7110 best:

- Best micro N7110 block: `boundaryblocks_micro_bad_probe`, `medium_guarded_pmed001`, acc `0.947389`, good/medium/bad `0.957525/0.923910/0.970617`.
- Current best ordinary N7110 remains `nl_n7110_gm_trim_bad_geom_directrule_n7100base_g003_m008__69ab5b71cf7d`, raw, acc `0.949628`, good/medium/bad `0.960338/0.926864/0.970617`.

Stability reseed without adding new rows also did not pass:

- Best reseed reached acc `0.934058`, good/medium/bad `0.936287/0.910830/0.970617`.

## Decision

Do not widen to N7150/N7200 from these block variants. The evidence says the remaining N7110/N7125 ordinary-checkpoint blocker is not solved by adding more block rows or by seed-only retraining. The qrs-low transparent rule artifact remains the only current method that passes this local boundary, but it is not a standalone checkpoint and should stay separate from ordinary promotion.

Next ordinary-checkpoint work should analyze probability/margin geometry or implement a more structured two-endpoint distillation objective, rather than continuing broad data block, class-weight, or seed sweeps.
