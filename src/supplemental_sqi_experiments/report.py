from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd


def copy_main_figures_to_shared_images(report_root: str | Path, shared_images_dir: str | Path) -> list[str]:
    rep = Path(report_root)
    shared = Path(shared_images_dir)
    shared.mkdir(parents=True, exist_ok=True)
    mapping = {
        "strict_table6/fig_supp_01_strict_table6_subset_selection": "fig_10_strict_table6_subset_selection",
        "model_diagnostics/fig_supp_02_model_stratified_diagnostics": "fig_11_model_stratified_diagnostics",
        "fsqi_mechanism/fig_supp_03_fsqi_mechanism": "fig_12_fsqi_mechanism",
        "final_claims/fig_12_fsqi_mechanism": "fig_12_fsqi_mechanism",
        "final_claims/fig_13_sqi_domain_shift": "fig_13_sqi_domain_shift",
        "final_claims/fig_14_bassqi_domain_shift": "fig_14_bassqi_domain_shift",
        "final_claims/fig_15_sqi_subgroup_separability": "fig_15_sqi_subgroup_separability",
    }
    copied: list[str] = []
    for src_stem, dst_stem in mapping.items():
        for suffix in [".png", ".pdf", ".svg"]:
            src = rep / f"{src_stem}{suffix}"
            if not src.exists():
                continue
            dst = shared / f"{dst_stem}{suffix}"
            shutil.copy2(src, dst)
            copied.append(str(dst))
    return copied


def write_summary(
    *,
    out_root: str | Path,
    report_root: str | Path,
    shared_images_dir: str | Path,
) -> Path:
    out = Path(out_root)
    rep = Path(report_root)
    rep.mkdir(parents=True, exist_ok=True)
    lines: list[str] = ["# SQI supplemental protocol experiments", ""]

    identity_path = out / "strict_table6" / "subset_identity_match.json"
    if identity_path.exists():
        identity = json.loads(identity_path.read_text(encoding="utf-8"))
        lines.extend(
            [
                "## Strict Table 6 subset selection",
                "",
                f"- Paper selected five: `{','.join(identity['paper_selected_five'])}`.",
                f"- Validation-recovered five: `{','.join(identity['validation_recovered_cardinality5'])}`.",
                f"- Same set as paper: `{identity['matches_paper_selected_five_as_set']}`.",
                "- Protocol: all 127 non-empty SQI subsets are selected by validation accuracy before test evaluation.",
                "",
            ]
        )
    selected_path = out / "strict_table6" / "selected_by_cardinality_test.csv"
    if selected_path.exists():
        selected = pd.read_csv(selected_path)
        keep = ["cardinality", "Selected_SQI", "val_Ac", "test_Ac", "test_AUC"]
        lines.append(selected[keep].to_markdown(index=False) if _has_tabulate() else _fallback_md(selected[keep]))
        lines.append("")

    metrics_path = out / "model_diagnostics" / "selected5_source_bootstrap_metrics.csv"
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        lines.extend(["## Selected-five model diagnostics", ""])
        lines.append(metrics.to_markdown(index=False) if _has_tabulate() else _fallback_md(metrics))
        lines.append("")

    fsqi_path = out / "fsqi_mechanism" / "fsqi_logdiff_subgroup_distribution.csv"
    if fsqi_path.exists():
        fsqi = pd.read_csv(fsqi_path)
        lines.extend(["## fSQI mechanism", ""])
        lines.append(fsqi.to_markdown(index=False) if _has_tabulate() else _fallback_md(fsqi))
        lines.append("")

    generalization_path = out / "generalization" / "cross_noise_generalization_svm.csv"
    if generalization_path.exists():
        gen = pd.read_csv(generalization_path)
        lines.extend(["## Cross-noise generalization", ""])
        keep = ["scenario", "test_Ac", "test_Se", "test_Sp", "test_AUC"]
        lines.append(gen[keep].to_markdown(index=False) if _has_tabulate() else _fallback_md(gen[keep]))
        lines.append("")

    final_dir = out / "final_claims"
    rank_path = final_dir / "table6_paper_quintuplet_rank.csv"
    plateau_path = final_dir / "table6_five_sqi_plateau_summary.csv"
    domain_path = final_dir / "domain_shift_metrics.csv"
    cross_path = final_dir / "cross_domain_auc_matrix.csv"
    fsqi_scan_path = final_dir / "fsqi_fixed_rbf_threshold_scan.csv"
    fsqi_model_path = final_dir / "fsqi_linear_vs_rbf.csv"
    bassqi_recall_path = final_dir / "bassqi_subgroup_recall.csv"
    separability_path = final_dir / "sqi_poor_domain_separability.csv"
    subset_recall_path = final_dir / "subset_subgroup_recall.csv"
    rescue_path = final_dir / "pair_to_quintuplet_error_rescue.csv"
    if rank_path.exists() or domain_path.exists() or fsqi_scan_path.exists():
        lines.extend(["## Remaining Evidence for Final Claims", ""])
        lines.extend(
            [
                "- Absolute paper accuracy is not strictly reproducible because the expert-adjudicated paper labels are unavailable.",
                "- SQI fusion remains effective, but the exact five-SQI optimum is assessed as a validation-selected subset rather than assumed from the paper.",
                "- Synthetic poor and original poor are treated as separate mechanisms when interpreting aggregate performance.",
                "- fSQI is reported as a feature-level mechanism check, not only as a scalar distribution.",
                "",
            ]
        )
        lines.extend(
            [
                "### Diagnostic analyses",
                "",
                "- Provenance groups were derived after merging `record84_norm.parquet` with the paper-balanced split table. Records were assigned to `original acceptable`, `original unacceptable`, `synthetic em`, or `synthetic ma` from `y`, `is_augmented`, and `noise_type`; `source_record_id` was retained to link each noisy derivative to its clean source.",
                "- PCA used the 84 normalized SQI features (`12 leads x 7 SQIs`). Standardization and the two-component PCA were fitted on original Set-a records only (`original acceptable` plus `original unacceptable`), then applied unchanged to all paper-aligned records including `paper EM` and `paper MA`.",
                "- The domain classifier used poor records only, with original unacceptable coded as 0 and synthetic em/ma as 1. A standardized logistic regression was evaluated by `StratifiedGroupKFold`, grouped by `source_record_id`, and summarized by AUC.",
                "- Distribution shift was also tested by RBF-MMD on standardized poor-record SQI features. The RBF bandwidth used the median-distance heuristic, and significance used a permutation null with the reported number of permutations.",
                "- Cross-domain transfer used the selected-five SQI RBF-SVM (`C=1`, `gamma=0.14`). Each run trained and validated on original acceptable plus one poor-domain source, selected the operating threshold on validation accuracy, refit on train+validation, and evaluated once on original acceptable plus the target poor domain.",
                "- Subgroup AUC/recall analyses used the same train/validation/test split and validation-selected thresholds. Acceptable records report acceptable specificity; original unacceptable, synthetic em, and synthetic ma report poor recall.",
                "- fSQI threshold sweep recomputed 12 lead-specific fSQI values from 125 Hz waveforms as the fraction of adjacent absolute differences below each flatness threshold, then evaluated fixed-RBF models under the same validation-threshold protocol.",
                "- basSQI paired deltas used `1-basSQI = P_0-1 / P_0-40` and compared each synthetic noisy record with its matched clean `source_record_id`, isolating the augmentation-induced change in low-frequency power fraction.",
                "",
            ]
        )
    if rank_path.exists():
        rank = pd.read_csv(rank_path)
        lines.extend(["### Paper quintuplet rank", ""])
        lines.append(rank.to_markdown(index=False) if _has_tabulate() else _fallback_md(rank))
        lines.append("")
    if plateau_path.exists():
        plateau = pd.read_csv(plateau_path)
        lines.extend(["### Five-SQI validation plateau", ""])
        lines.append(plateau.to_markdown(index=False) if _has_tabulate() else _fallback_md(plateau))
        lines.append("")
    if domain_path.exists():
        domain = pd.read_csv(domain_path)
        lines.extend(["### SQI domain shift", ""])
        lines.append(domain.to_markdown(index=False) if _has_tabulate() else _fallback_md(domain))
        lines.append("")
    if cross_path.exists():
        cross = pd.read_csv(cross_path)
        keep = ["train_poor_domain", "test_poor_domain", "test_Ac", "test_Se", "test_Sp", "test_AUC"]
        lines.extend(["### Cross-domain AUC matrix source table", ""])
        lines.append(cross[keep].to_markdown(index=False) if _has_tabulate() else _fallback_md(cross[keep]))
        lines.append("")
    if fsqi_scan_path.exists():
        scan = pd.read_csv(fsqi_scan_path)
        keep = ["threshold_mv", "val_Ac", "test_Ac", "test_Se", "test_Sp", "test_AUC"]
        lines.extend(["### fSQI fixed-RBF threshold scan", ""])
        lines.append(scan[keep].to_markdown(index=False) if _has_tabulate() else _fallback_md(scan[keep]))
        lines.append("")
    if fsqi_model_path.exists():
        comp = pd.read_csv(fsqi_model_path)
        keep = ["model", "threshold_mv", "test_Ac", "test_Se", "test_Sp", "test_AUC"]
        lines.extend(["### fSQI linear vs RBF", ""])
        lines.append(comp[keep].to_markdown(index=False) if _has_tabulate() else _fallback_md(comp[keep]))
        lines.append("")
    if bassqi_recall_path.exists():
        bass = pd.read_csv(bassqi_recall_path)
        lines.extend(["### basSQI mechanism", ""])
        lines.append("`1-basSQI = P_0-1 / P_0-40`, the low-frequency power fraction.")
        lines.append("")
        lines.append(bass.to_markdown(index=False) if _has_tabulate() else _fallback_md(bass))
        lines.append("")
    if separability_path.exists():
        sep = pd.read_csv(separability_path)
        keep = ["SQI", "poor_domain", "test_Ac", "test_Se", "test_Sp", "test_AUC"]
        lines.extend(["### SQI subgroup separability", ""])
        lines.append(sep[keep].to_markdown(index=False) if _has_tabulate() else _fallback_md(sep[keep]))
        lines.append("")
    if subset_recall_path.exists():
        sub = pd.read_csv(subset_recall_path)
        lines.extend(["### Subset subgroup recall", ""])
        lines.append(sub.to_markdown(index=False) if _has_tabulate() else _fallback_md(sub))
        lines.append("")
    if rescue_path.exists():
        rescue = pd.read_csv(rescue_path)
        lines.extend(["### Pair-to-quintuplet error rescue", ""])
        lines.append(rescue.to_markdown(index=False) if _has_tabulate() else _fallback_md(rescue))
        lines.append("")

    stability_path = out / "stability" / "multi_seed_stability_summary.csv"
    if stability_path.exists():
        stability = pd.read_csv(stability_path)
        lines.extend(["## Multi-seed stability smoke", ""])
        lines.append(stability.to_markdown(index=False) if _has_tabulate() else _fallback_md(stability))
        lines.append("")

    copied = copy_main_figures_to_shared_images(rep, shared_images_dir)
    lines.extend(
        [
            "## Figure index",
            "",
            "- `fig_supp_01_strict_table6_subset_selection`: strict 127-subset validation selection and inclusion heatmap.",
            "- `fig_supp_02_model_stratified_diagnostics`: selected-five ROC, score distributions, and source-bootstrap CI.",
            "- `fig_supp_03_fsqi_mechanism`: log-difference distributions and flat-threshold sensitivity.",
            "- `fig_12_fsqi_mechanism`: updated fSQI lead-level mechanism, fixed-RBF threshold scan, and subgroup recall.",
            "- `fig_13_sqi_domain_shift`: PCA, per-SQI domain AUC, and cross-domain AUC matrix.",
            "- `fig_14_bassqi_domain_shift`: conditional basSQI mechanism figure when basSQI shows strong domain shift.",
            "- `fig_15_sqi_subgroup_separability`: SQI by poor-domain AUC, subset subgroup recall, and pair-to-quintuplet error rescue.",
            "- `model_diagnostics/error_gallery/high_confidence_gallery/*`: high-confidence error/control waveform review pack.",
            "",
            "Shared image copies:",
            "",
            *[f"- `{Path(p).name}`" for p in copied],
            "",
            "## Reproducibility commands",
            "",
            "```powershell",
            ".\\.venv\\Scripts\\python.exe -m src.supplemental_sqi_experiments.run diagnose-existing",
            ".\\.venv\\Scripts\\python.exe -m src.supplemental_sqi_experiments.run final-claims",
            ".\\.venv\\Scripts\\python.exe -m src.supplemental_sqi_experiments.run build-isolated --seed 0",
            ".\\.venv\\Scripts\\python.exe -m src.supplemental_sqi_experiments.run stability",
            ".\\.venv\\Scripts\\python.exe -m src.supplemental_sqi_experiments.run stability --include-mlp",
            "```",
            "",
        ]
    )
    summary = rep / "supplemental_protocol_summary.md"
    summary.write_text("\n".join(lines), encoding="utf-8")
    return summary


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
    except Exception:
        return False
    return True


def _fallback_md(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    rows = [[_fmt(v) for v in r] for r in df.to_numpy()]
    widths = [max([len(cols[i])] + [len(row[i]) for row in rows]) for i in range(len(cols))]
    lines = [
        "| " + " | ".join(cols[i].ljust(widths[i]) for i in range(len(cols))) + " |",
        "| " + " | ".join("-" * widths[i] for i in range(len(cols))) + " |",
    ]
    lines.extend("| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(cols))) + " |" for row in rows)
    return "\n".join(lines)


def _fmt(v: Any) -> str:
    if pd.isna(v):
        return ""
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)
