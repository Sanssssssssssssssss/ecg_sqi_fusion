# UFormer Geometry Internal PC1 Bad Gate Metrics

This is a normal checkpoint with an internal parametric PC1 bad gate saved in the model branch config, not an external post-hoc gate.

| node_id | scope | variant_id | acc | macro_f1 | good_recall | medium_recall | bad_recall | good_to_medium | medium_to_good | bad_to_medium |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| N17043_gm_trim_bad | node_all | nl_n17043_gm_trim_bad_boundaryblocks_teacherwall_goodsafe_5e30e8c689a6_uformergeom_N17043_gm_probe_nodecal_geom_tabular_pc1gate_internal | 0.988884 | 0.990954 | 0.988969 | 0.984475 | 1.000000 | 188 | 163 | 0 |
| N17043_gm_trim_bad | node_test | nl_n17043_gm_trim_bad_boundaryblocks_teacherwall_goodsafe_5e30e8c689a6_uformergeom_N17043_gm_probe_nodecal_geom_tabular_pc1gate_internal | 0.979352 | 0.984567 | 0.984890 | 0.974243 | 1.000000 | 55 | 113 | 0 |
| N17043_gm_probe | node_all | nl_n17043_gm_trim_bad_boundaryblocks_teacherwall_goodsafe_5e30e8c689a6_uformergeom_N17043_gm_probe_nodecal_geom_tabular_pc1gate_internal | 0.981248 | 0.979488 | 0.988969 | 0.984475 | 0.949858 | 188 | 163 | 63 |
| N17043_gm_probe | node_test | nl_n17043_gm_trim_bad_boundaryblocks_teacherwall_goodsafe_5e30e8c689a6_uformergeom_N17043_gm_probe_nodecal_geom_tabular_pc1gate_internal | 0.949039 | 0.817841 | 0.984890 | 0.974243 | 0.360097 | 55 | 113 | 61 |

## Learned gate

```json
{
  "threshold_norm": 1.3972392082214355,
  "threshold_raw": 7.817184378491561,
  "temperature_norm": 0.07118828594684601,
  "temperature_raw": 0.3982796732300642,
  "strength": 6.0139851570129395
}
```
