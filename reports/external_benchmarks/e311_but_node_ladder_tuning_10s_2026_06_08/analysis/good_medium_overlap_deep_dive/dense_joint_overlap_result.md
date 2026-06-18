# Dense Joint Overlap Result

Run timestamp: 2026-06-10

## What changed

The compact joint-overlap attempt used an N7000 compact scan body and was too small to be trustworthy. A dense/full retry was added in `run_joint_overlap_bisection.py`:

- Base body: `nl_n6800_gm_trim_bad_scan_029_sc_overlap_narrow_oscillato_873d37fc4791` promoted N6800 full artifact, 15321 rows.
- Added paired overlap tails only:
  - `aux_lightly_contaminated_good`
  - `aux_visible_qrs_medium`
- Bad class was not expanded; bad stayed trim-bad guardrail.
- Original BUT was not used for selection.

## Dense artifacts

| variant | rows | good | medium | bad |
| --- | ---: | ---: | ---: | ---: |
| `nl_n7000_gm_trim_bad_dense_joint_n6800full_lcg025_vqm035__bfc4b452a166` | 15627 | 5234 | 5286 | 5107 |
| `nl_n7000_gm_trim_bad_dense_joint_n6800full_lcg035_vqm045__64d67cd66f63` | 15730 | 5286 | 5337 | 5107 |
| `nl_n7000_gm_trim_bad_dense_joint_n6800full_lcg045_vqm045__d0f7b043135d` | 15781 | 5337 | 5337 | 5107 |

## Diagnostic outcome

N7000 still did not promote. Best overall remains the older medium-strong compact variant:

| variant | mode | acc | macro-F1 | good | medium | bad |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `nl_n7000_gm_trim_bad_scan_014_sc_overlap_compact_pca_core_ddc377ccab88` | calibrated | 0.934638 | 0.943098 | 0.864571 | 0.967000 | 0.999265 |
| `nl_n7000_gm_trim_bad_dense_joint_n6800full_lcg045_vqm045__d0f7b043135d` | medium_guarded_pmed0005 | 0.933919 | 0.940653 | 0.986857 | 0.859429 | 0.970862 |
| `nl_n7000_gm_trim_bad_dense_joint_n6800full_lcg025_vqm035__bfc4b452a166` | medium_guarded_pmed0005 | 0.929717 | 0.937616 | 0.947143 | 0.887429 | 0.972331 |

## Interpretation

The dense joint generator successfully flips the error direction: good recall can be recovered strongly, but medium recall drops. The older compact model has the opposite shape: medium and bad are strong, but good is too low. This confirms that the blocker is not bad coverage; it is the good/medium overlap decision surface.

Continuing to add more paired tail is unlikely to solve the node by itself. The next useful step is to analyze disagreement between:

- medium-strong model: `nl_n7000_gm_trim_bad_scan_014_sc_overlap_compact_pca_core_ddc377ccab88`
- good-strong model: `nl_n7000_gm_trim_bad_dense_joint_n6800full_lcg045_vqm045__d0f7b043135d`

Use the overlap-only feature thresholds already found in `threshold_candidates.csv` to design a small feature/probability gate or a new generator that specifically targets disagreement rows, rather than global class-weight or tail-size sweeps.

## Current decision

Do not proceed to N7200 or original-filtered evaluation from this N7000 branch yet. N7000 remains below the clean gate, and original remains report-only.
