# N7110 QRS-Low Distillation

This experiment tries to convert the transparent qrs-low endpoint rule into one ordinary checkpoint without adding broad synthetic rows.

## Build

- `nl_n7110_gm_trim_bad_distill_qrslow_soft_mild_ganchor_seed20261111`: medium soft rows 4, good anchors 13
- `nl_n7110_gm_trim_bad_distill_qrslow_soft_tight_noanchor_seed20261112`: medium soft rows 4, good anchors 0
- `nl_n7110_gm_trim_bad_distill_qrslow_soft_balanced_hinge_seed20261113`: medium soft rows 4, good anchors 13

## Quick Training

- `nl_n7110_gm_trim_bad_distill_qrslow_soft_mild_ganchor_seed20261111`: returncode 0
- `nl_n7110_gm_trim_bad_distill_qrslow_soft_tight_noanchor_seed20261112`: returncode 0
- `nl_n7110_gm_trim_bad_distill_qrslow_soft_balanced_hinge_seed20261113`: returncode 0

## Diagnostic

- best by acc: `nl_n7110_gm_trim_bad_geom_directrule_n7100base_g003_m008__69ab5b71cf7d` / `raw`
- acc 0.9496284965034965, macro-F1 0.9546322740979828
- recall good/medium/bad 0.960337552742616/0.9268635724331928/0.970617042115573

Original BUT is report-only and was not used for selection.
