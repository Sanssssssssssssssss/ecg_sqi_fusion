# UFormer Geometry Nodecal + PC1 Bad Gate on N17043 Trim-Bad

Same checkpoints trained with N17043_gm_probe nodecal; evaluated on N17043_gm_trim_bad node manifest. Gate: `pc1 > 7.82 -> bad`.

| variant_id | pc1_bad_gate_threshold | node_test_acc | node_test_good_recall | node_test_medium_recall | node_test_bad_recall | node_all_acc | node_all_good_recall | node_all_medium_recall | node_all_bad_recall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| nl_n17043_gm_trim_bad_boundaryblocks_teacherwall_goodsafe_5e30e8c689a6_uformergeom_N17043_gm_probe_nodecal_geom_tabular_badgate | 7.820000 | 0.981674 | 0.994780 | 0.970402 | 1.000000 | 0.988191 | 0.988500 | 0.983158 | 1.000000 |
| nl_n17043_gm_trim_bad_boundaryblocks_teacherwall_goodsafe_5e30e8c689a6_uformergeom_N17043_gm_probe_nodecal_geom_bad_guard | 7.820000 | 0.980819 | 0.985440 | 0.976502 | 1.000000 | 0.986049 | 0.988441 | 0.976854 | 1.000000 |
| nl_n17043_gm_trim_bad_boundaryblocks_teacherwall_goodsafe_5e30e8c689a6_uformergeom_N17043_gm_probe_nodecal_geom_balanced | 7.820000 | 0.968112 | 0.979670 | 0.957750 | 1.000000 | 0.985514 | 0.989556 | 0.973466 | 1.000000 |
| nl_n17043_gm_trim_bad_boundaryblocks_teacherwall_goodsafe_5e30e8c689a6_uformergeom_N17043_gm_probe_nodecal_geom_tabular_pc1bad_aug | 7.820000 | 0.952230 | 0.974725 | 0.932445 | 1.000000 | 0.981609 | 0.989673 | 0.961611 | 1.000000 |
| nl_n17043_gm_trim_bad_boundaryblocks_teacherwall_goodsafe_5e30e8c689a6_uformergeom_N17043_gm_probe_nodecal_geom_medium_guard | 7.820000 | 0.916555 | 0.997253 | 0.847944 | 1.000000 | 0.970839 | 0.994132 | 0.922281 | 1.000000 |
