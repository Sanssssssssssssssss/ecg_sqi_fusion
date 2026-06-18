# N7110 Margin Distillation

This experiment keeps the best N7110 ordinary checkpoint data and only applies tiny soft-target adjustments on synthetic-only good/medium overlap rows.
Original BUT is report-only and is not used for selection.

## Endpoint Geometry

- endpoint disagreement rows: 873
- by class: {'bad': 1, 'good': 146, 'medium': 726}

## Build

- `nl_n7110_gm_trim_bad_margin_fullatlas_ultratiny_seed20261501`: medium soft rows 75, good anchors 75
- `nl_n7110_gm_trim_bad_margin_fullatlas_mild_seed20261502`: medium soft rows 125, good anchors 125
- `nl_n7110_gm_trim_bad_margin_fullatlas_anchor_guard_seed20261503`: medium soft rows 100, good anchors 150

## Quick Training

- `nl_n7110_gm_trim_bad_margin_fullatlas_ultratiny_seed20261501`: returncode 0
- `nl_n7110_gm_trim_bad_margin_fullatlas_mild_seed20261502`: returncode 0
- `nl_n7110_gm_trim_bad_margin_fullatlas_anchor_guard_seed20261503`: returncode 0

## Diagnostic

- best by acc: `nl_n7110_gm_trim_bad_geom_directrule_n7100base_g003_m008__69ab5b71cf7d` / `raw`
- acc 0.9496284965034965, macro-F1 0.9546322740979828
- recall good/medium/bad 0.960337552742616/0.9268635724331928/0.970617042115573

### Margin Variants

- `nl_n7110_gm_trim_bad_margin_fullatlas_ultratiny_seed20261501` / `calibrated`: acc 0.939576048951049, good/medium/bad 0.9492264416315048/0.9120956399437412/0.970617042115573
- `nl_n7110_gm_trim_bad_margin_fullatlas_ultratiny_seed20261501` / `raw`: acc 0.936680506993007, good/medium/bad 0.9160337552742616/0.9378340365682136/0.970617042115573
- `nl_n7110_gm_trim_bad_margin_fullatlas_ultratiny_seed20261501` / `medium_guarded_pmed002`: acc 0.9293597027972028, good/medium/bad 0.8817158931082981/0.9533052039381154/0.970617042115573
- `nl_n7110_gm_trim_bad_margin_fullatlas_ultratiny_seed20261501` / `medium_guarded_pmed001`: acc 0.9260817307692308, good/medium/bad 0.869198312236287/0.9573839662447258/0.970617042115573
- `nl_n7110_gm_trim_bad_margin_fullatlas_anchor_guard_seed20261503` / `medium_guarded_pmed001`: acc 0.9117679195804196, good/medium/bad 0.9720112517580872/0.8177215189873418/0.970617042115573
- `nl_n7110_gm_trim_bad_margin_fullatlas_anchor_guard_seed20261503` / `medium_guarded_pmed002`: acc 0.9105113636363636, good/medium/bad 0.9722925457102672/0.8142053445850914/0.970617042115573
- `nl_n7110_gm_trim_bad_margin_fullatlas_anchor_guard_seed20261503` / `raw`: acc 0.907506555944056, good/medium/bad 0.9742616033755276/0.8045007032348804/0.970617042115573
- `nl_n7110_gm_trim_bad_margin_fullatlas_anchor_guard_seed20261503` / `calibrated`: acc 0.9055944055944056, good/medium/bad 0.9751054852320676/0.7987341772151899/0.970617042115573
- `nl_n7110_gm_trim_bad_margin_fullatlas_mild_seed20261502` / `medium_guarded_pmed001`: acc 0.8824300699300699, good/medium/bad 0.9547116736990154/0.7593530239099859/0.9708619000979432
- `nl_n7110_gm_trim_bad_margin_fullatlas_mild_seed20261502` / `medium_guarded_pmed002`: acc 0.8816105769230769, good/medium/bad 0.9563994374120957/0.7555555555555555/0.9708619000979432
- `nl_n7110_gm_trim_bad_margin_fullatlas_mild_seed20261502` / `raw`: acc 0.8768575174825175, good/medium/bad 0.9617440225035162/0.7379746835443038/0.9708619000979432
- `nl_n7110_gm_trim_bad_margin_fullatlas_mild_seed20261502` / `calibrated`: acc 0.8735249125874126, good/medium/bad 0.9648382559774964/0.7263009845288326/0.9708619000979432
- `nl_n7110_gm_trim_bad_margin_fullatlas_ultratiny_seed20261501` / `medium_guarded_pmed0005`: acc 0.8086756993006993, good/medium/bad 0.5379746835443038/0.9863572433192688/0.970617042115573
- `nl_n7110_gm_trim_bad_margin_fullatlas_anchor_guard_seed20261503` / `medium_guarded_pmed0005`: acc 0.7795563811188811, good/medium/bad 0.5983122362869199/0.8510548523206751/0.970617042115573
- `nl_n7110_gm_trim_bad_margin_fullatlas_mild_seed20261502` / `medium_guarded_pmed0005`: acc 0.6065887237762237, good/medium/bad 0.1634317862165963/0.8405063291139241/0.9708619000979432
