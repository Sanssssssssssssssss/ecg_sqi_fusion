# N7110 Added Boundary Targets

N7110 is a conservative bisection between promoted N7100 and failed N7125/N7150. It keeps the promoted N7100 base and adds only the first 20 rows from the N7150-N7100 boundary ring: 10 good and 10 medium.

Selection remains Clean/SemiClean/node diagnostic only. Original BUT is report-only. Bad is not expanded; trim-bad guardrail remains unchanged.

## Counts

- rows: 20
- class_counts: {'good': 10, 'medium': 10}
- split_counts: {'train': 16, 'test': 4}
- original_region_counts: {'clean_core': 10, 'outlier_low_confidence': 5, 'good_medium_overlap': 5}
- ambiguous_type_counts: {'clean_or_target': 10, 'good_medium_boundary': 5, 'good_medium_low_purity': 3, 'isolated_medium': 2}
