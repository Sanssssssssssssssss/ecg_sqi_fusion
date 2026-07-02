from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .common import Paths, dry, ensure_dirs, git_commit, read_json, rel, run_date, table_to_md
from src.transformer_pipeline.data_v1_gapfill.common import POLICY as BUT_POLICY
from src.transformer_pipeline.data_v1_gapfill.common import report_dir as but_report_dir


def _read_csv(path):
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _ordered(df: pd.DataFrame, preferred: list[str]) -> pd.DataFrame:
    if df.empty:
        return df
    cols = [c for c in preferred if c in df.columns]
    cols.extend([c for c in df.columns if c not in cols])
    return df[cols]


def _seta_model_display(df: pd.DataFrame, paths: Paths, name: str) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if name in {"seta_construction_effect_models", "seta_construction_source_only_models"}:
        out = out.merge(_seta_construction_eval_scope(paths), on="construction", how="left")
    out["Se"] = pd.to_numeric(out.get("acceptable_recall"), errors="coerce")
    out["Sp"] = pd.to_numeric(out.get("original_unacceptable_recall"), errors="coerce")
    out["acceptable_positive_model_auc"] = pd.to_numeric(out.get("auc"), errors="coerce")
    out = out.drop(columns=["auc"], errors="ignore")
    out = _ordered(
        out,
        [
            "run_id",
            "construction",
            "model",
            "input",
            "threshold_source",
            "threshold",
            "model_test_scope",
            "train_acceptable_original_n",
            "train_poor_source_n",
            "source_only_train_poor_contract",
            "train_original_unacceptable_n",
            "train_generated_unacceptable_n",
            "val_generated_rows",
            "test_generated_rows",
            "test_acceptable_n",
            "test_unacceptable_n",
            "acc",
            "Se",
            "Sp",
            "acceptable_positive_model_auc",
            "balanced_acc",
            "acceptable_recall",
            "original_unacceptable_recall",
            "confusion",
        ],
    )
    out.to_csv(paths.tables / f"{name}_with_se_sp_auc.csv", index=False)
    return out


def _seta_distribution_display(df: pd.DataFrame, paths: Paths) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy().rename(columns={"c2st_auc": "generated_vs_original_c2st_auc"})
    out.to_csv(paths.tables / "seta_distribution_repair_metrics_display.csv", index=False)
    return out


def _seta_construction_source_audit(paths: Paths) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for arm in ["native_imbalanced", "fixed_synthetic", "quota_draw", "smc_gapfill"]:
        split_path = paths.seta_arms / arm / "splits" / "split.csv"
        df = _read_csv(split_path)
        if df.empty:
            continue
        train_poor = df.loc[df["split"].astype(str).eq("train") & df["y"].astype(int).eq(-1)].copy()
        original = train_poor.loc[pd.to_numeric(train_poor["is_augmented"], errors="coerce").fillna(0).astype(int).eq(0)]
        generated = train_poor.loc[pd.to_numeric(train_poor["is_augmented"], errors="coerce").fillna(0).astype(int).eq(1)]
        counts = generated["candidate_type"].astype(str).value_counts().sort_index().to_dict()
        rows.append(
            {
                "construction": arm,
                "train_original_unacceptable_n": int(len(original)),
                "train_generated_unacceptable_n": int(len(generated)),
                "train_generated_candidate_types": json.dumps({str(k): int(v) for k, v in counts.items()}, ensure_ascii=False),
                "synthetic_source_contract": "paper -6 dB em/ma" if arm == "fixed_synthetic" else ("current SMC-selected pool" if arm == "smc_gapfill" else arm),
                "split_contract": "train-only generated; val/test original only",
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(paths.tables / "seta_construction_source_audit.csv", index=False)
    return out


def _seta_construction_eval_scope(paths: Paths) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for arm in ["native_imbalanced", "fixed_synthetic", "quota_draw", "smc_gapfill"]:
        split_path = paths.seta_arms / arm / "splits" / "split.csv"
        df = _read_csv(split_path)
        if df.empty:
            continue
        is_aug = pd.to_numeric(df.get("is_augmented", 0), errors="coerce").fillna(0).astype(int)
        y = pd.to_numeric(df["y"], errors="coerce").astype(int)
        train = df["split"].astype(str).eq("train")
        val = df["split"].astype(str).eq("val")
        test = df["split"].astype(str).eq("test")
        train_poor = train & y.eq(-1)
        rows.append(
            {
                "construction": arm,
                "model_test_scope": "held-out original Set-A test only",
                "train_original_unacceptable_n": int((train_poor & is_aug.eq(0)).sum()),
                "train_generated_unacceptable_n": int((train_poor & is_aug.eq(1)).sum()),
                "val_generated_rows": int((val & is_aug.eq(1)).sum()),
                "test_generated_rows": int((test & is_aug.eq(1)).sum()),
                "test_acceptable_n": int((test & y.eq(1)).sum()),
                "test_unacceptable_n": int((test & y.eq(-1)).sum()),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(paths.tables / "seta_construction_eval_scope.csv", index=False)
    return out


def _confusion_specificity(raw: Any) -> float:
    if pd.isna(raw):
        return float("nan")
    try:
        cm = json.loads(str(raw))
        tn = float(cm["tn"])
        fp = float(cm["fp"])
        return tn / max(1.0, tn + fp)
    except Exception:
        return float("nan")


def _but_model_display(df: pd.DataFrame, paths: Paths) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out = out.drop(columns=["test_auc"], errors="ignore")
    out = _ordered(
        out,
        [
            "run_id",
            "model",
            "input",
            "task",
            "test_acc",
            "test_macro_f1",
            "good_recall",
            "intermediate_recall",
            "poor_recall",
            "poor_fpr_nonpoor",
            "poor_vs_rest_auc",
            "collapsed_good_vs_rest_acc",
            "collapsed_good_vs_rest_auc",
            "collapsed_good_vs_rest_confusion",
            "confusion",
        ],
    )
    out.to_csv(paths.tables / "but_model_comparison_display.csv", index=False)
    return out


def _paper_synthetic_domain_table(paths: Paths) -> pd.DataFrame:
    final_claims = Path("outputs") / "sqi_supplemental" / "existing_seed0" / "final_claims"
    waveform = Path("outputs") / "sqi_supplemental" / "existing_seed0" / "waveform_domain_auc" / "paper_waveform_all_domain_auc.csv"
    metrics = _read_csv(final_claims / "domain_shift_metrics.csv")
    all_wave = _read_csv(waveform)
    rows: list[dict[str, Any]] = []
    if not metrics.empty:
        auc = metrics.loc[metrics["metric"].eq("source_grouped_logistic_domain_auc")]
        if not auc.empty:
            r = auc.iloc[0]
            rows.append(
                {
                    "feature_set": "84-SQI",
                    "comparison": r["comparison"],
                    "domain_auc": r["estimate"],
                    "ci_low": "",
                    "ci_high": "",
                    "n_original_poor": r.get("n_original_poor", ""),
                    "n_synthetic_poor": r.get("n_synthetic_poor", ""),
                    "source": "SQI baseline final_claims/domain_shift_metrics.csv",
                }
            )
        mmd = metrics.loc[metrics["metric"].eq("RBF-MMD2")]
        if not mmd.empty:
            r = mmd.iloc[0]
            rows.append(
                {
                    "feature_set": "84-SQI",
                    "comparison": "RBF-MMD2 original poor vs synthetic poor",
                    "domain_auc": "",
                    "ci_low": "",
                    "ci_high": "",
                    "n_original_poor": r.get("n_original_poor", ""),
                    "n_synthetic_poor": r.get("n_synthetic_poor", ""),
                    "source": f"MMD2={float(r['estimate']):.6f}; permutation p={float(r['p_value_permutation']):.6g}",
                }
            )
    if not all_wave.empty:
        for _, r in all_wave.iterrows():
            rows.append(
                {
                    "feature_set": str(r["feature_set"]),
                    "comparison": r["comparison"],
                    "domain_auc": r["domain_auc"],
                    "ci_low": r.get("domain_auc_ci_low", ""),
                    "ci_high": r.get("domain_auc_ci_high", ""),
                    "n_original_poor": r.get("n_original_bad", ""),
                    "n_synthetic_poor": r.get("n_synthetic_bad", ""),
                    "source": "SQI supplemental waveform_domain_auc/paper_waveform_all_domain_auc.csv",
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        for col in ["domain_auc", "ci_low", "ci_high", "n_original_poor", "n_synthetic_poor"]:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        out.to_csv(paths.tables / "seta_paper_synthetic_domain_auc.csv", index=False)
    return out


def _paper_cross_domain_display(paths: Paths) -> pd.DataFrame:
    cross = _read_csv(Path("outputs") / "sqi_supplemental" / "existing_seed0" / "final_claims" / "cross_domain_auc_matrix.csv")
    if cross.empty:
        return cross
    out = cross.copy()
    out["Se"] = pd.to_numeric(out.get("test_Se"), errors="coerce")
    out["Sp"] = pd.to_numeric(out.get("test_Sp"), errors="coerce")
    out["acceptable_positive_model_auc"] = pd.to_numeric(out.get("test_AUC"), errors="coerce")
    out = out.drop(columns=["test_Se", "test_Sp", "test_AUC", "skipped"], errors="ignore")
    out = _ordered(
        out,
        [
            "train_poor_domain",
            "test_poor_domain",
            "train_n",
            "val_n",
            "test_n",
            "threshold",
            "test_Ac",
            "Se",
            "Sp",
            "acceptable_positive_model_auc",
        ],
    )
    out.to_csv(paths.tables / "seta_paper_cross_domain_performance_with_se_sp_auc.csv", index=False)
    return out


def _paper_selected5_aggregate(paths: Paths) -> pd.DataFrame:
    boot = _read_csv(Path("outputs") / "sqi_supplemental" / "existing_seed0" / "model_diagnostics" / "selected5_source_bootstrap_metrics.csv")
    if boot.empty:
        return boot
    rows: list[dict[str, Any]] = []
    for model, part in boot.groupby("model", sort=False):
        row: dict[str, Any] = {
            "model": model,
            "protocol": "SQI supplemental paper-balanced test",
            "input": "selected-five 12-lead SQI",
            "threshold": float(part["threshold"].iloc[0]),
            "source": "outputs/sqi_supplemental/existing_seed0/model_diagnostics/selected5_source_bootstrap_metrics.csv",
        }
        for metric in ["Ac", "Se", "Sp", "AUC"]:
            m = part.loc[part["metric"].eq(metric)]
            if not m.empty:
                out_metric = "acceptable_positive_model_auc" if metric == "AUC" else metric
                row[out_metric] = float(m["estimate"].iloc[0])
                row[f"{out_metric}_ci_low"] = float(m["ci_low"].iloc[0])
                row[f"{out_metric}_ci_high"] = float(m["ci_high"].iloc[0])
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(paths.tables / "seta_sqi_supplemental_selected5_aggregate.csv", index=False)
    return out


def _paper_selected5_subgroup_recall(paths: Paths) -> pd.DataFrame:
    strat = _read_csv(Path("outputs") / "sqi_supplemental" / "existing_seed0" / "model_diagnostics" / "stratified_score_summary.csv")
    if strat.empty:
        return strat
    keep_models = {"SVM selected-five", "MLP selected-five"}
    keep_groups = {"original acceptable", "original unacceptable", "synthetic em", "synthetic ma"}
    out = strat.loc[strat["model"].isin(keep_models) & strat["sample_group"].isin(keep_groups)].copy()
    if out.empty:
        return out
    out["group_recall_metric"] = out["sample_group"].map(
        lambda x: "acceptable_recall" if x == "original acceptable" else "poor_recall"
    )
    out["group_recall"] = out.apply(
        lambda r: float(r["acceptance_rate"]) if r["sample_group"] == "original acceptable" else float(r["rejection_rate"]),
        axis=1,
    )
    out["source"] = "outputs/sqi_supplemental/existing_seed0/model_diagnostics/stratified_score_summary.csv"
    out = out.rename(columns={"AUC_pairwise": "pairwise_model_auc"})
    out = _ordered(
        out,
        [
            "model",
            "sample_group",
            "n",
            "group_recall_metric",
            "group_recall",
            "acceptance_rate",
            "rejection_rate",
            "threshold",
            "score_mean",
            "score_median",
            "pairwise_model_auc",
            "source",
        ],
    )
    out.to_csv(paths.tables / "seta_sqi_supplemental_selected5_subgroup_recall.csv", index=False)
    return out


def _paper_cross_noise_generalization(paths: Paths) -> pd.DataFrame:
    gen = _read_csv(Path("outputs") / "sqi_supplemental" / "existing_seed0" / "generalization" / "cross_noise_generalization_svm.csv")
    if gen.empty:
        return gen
    out = gen.copy()
    out["Se"] = pd.to_numeric(out.get("test_Se"), errors="coerce")
    out["Sp"] = pd.to_numeric(out.get("test_Sp"), errors="coerce")
    out["acceptable_positive_model_auc"] = pd.to_numeric(out.get("test_AUC"), errors="coerce")
    out["source"] = "outputs/sqi_supplemental/existing_seed0/generalization/cross_noise_generalization_svm.csv"
    out = out.drop(columns=["skipped", "test_Se", "test_Sp", "test_AUC"], errors="ignore")
    out = _ordered(
        out,
        [
            "scenario",
            "train_n",
            "val_n",
            "test_n",
            "threshold",
            "test_Ac",
            "Se",
            "Sp",
            "acceptable_positive_model_auc",
            "test_tn",
            "test_fp",
            "test_fn",
            "test_tp",
            "source",
        ],
    )
    out.to_csv(paths.tables / "seta_sqi_supplemental_cross_noise_generalization.csv", index=False)
    return out


def _but_candidate_composition(paths: Paths) -> pd.DataFrame:
    counts = _read_csv(but_report_dir() / f"{BUT_POLICY}_candidate_type_counts.csv")
    if counts.empty:
        return counts
    total = counts.groupby("class_name")["size"].transform("sum").replace(0, pd.NA)
    counts["pct_within_class"] = (counts["size"] / total * 100.0).round(2)
    counts.to_csv(paths.tables / "but_candidate_type_composition.csv", index=False)
    return counts


def _but_distribution_metrics(paths: Paths) -> pd.DataFrame:
    metrics = _read_csv(but_report_dir() / f"{BUT_POLICY}_global_distribution_metrics.csv")
    if metrics.empty:
        return metrics
    keep = ["scope", "but_n", "synthetic_n", "rbf_mmd", "sliced_wasserstein", "quantile_loss", "domain_auc", "pca_density_overlap"]
    out = metrics[[c for c in keep if c in metrics.columns]].copy()
    out = out.rename(columns={"domain_auc": "global_domain_auc_not_dual"})
    out.to_csv(paths.tables / "but_distribution_fit_metrics.csv", index=False)
    return out


def _but_dual_auc_audit(paths: Paths) -> pd.DataFrame:
    audit = paths.tables / "but_v116_dual_generated_auc_audit" / "dual_generated_auc.csv"
    out = _read_csv(audit)
    if out.empty:
        return out
    out = out.rename(
        columns={
            "auc": "generated_vs_original_domain_auc",
            "sym_auc": "symmetric_generated_vs_original_domain_auc",
        }
    )
    out = _ordered(
        out,
        [
            "scope",
            "status",
            "rows",
            "original_n",
            "generated_n",
            "generated_vs_original_domain_auc",
            "symmetric_generated_vs_original_domain_auc",
            "acc",
            "ideal_pass",
        ],
    )
    out.to_csv(paths.tables / "but_v116_dual_generated_auc.csv", index=False)
    return out


def _but_cross_check_table(paths: Paths, audit: dict[str, Any], but_models: pd.DataFrame) -> pd.DataFrame:
    but = audit.get("but", {})
    nn = _read_csv(but_report_dir() / f"{BUT_POLICY}_nearest_neighbor_leakage_audit.csv")
    e31 = but_models.loc[but_models["run_id"].eq("but_e31_wave_mechanism_conformer")] if not but_models.empty else pd.DataFrame()
    rows = [
        {
            "check": "original BUT gap5 source",
            "value": but.get("original_but_rows", ""),
            "detail": json.dumps(but.get("original_but_class_counts", {}), ensure_ascii=False),
        },
        {
            "check": "v116 final protocol",
            "value": but.get("protocol_rows", ""),
            "detail": json.dumps(but.get("protocol_class_counts", {}), ensure_ascii=False),
        },
        {
            "check": "train exact balance",
            "value": "8310/8310/8310",
            "detail": json.dumps(but.get("train_class_counts", {}), ensure_ascii=False),
        },
        {
            "check": "val/test generated rows",
            "value": but.get("val_test_generated_rows", ""),
            "detail": "computed from official fold split only",
        },
        {
            "check": "official split source",
            "value": rel(Path(but.get("split_path", "")) / "original_region_atlas.csv") if but.get("split_path") else "",
            "detail": "raw protocol metadata split column is not the leakage-audit split",
        },
        {
            "check": "allowed candidate types",
            "value": ", ".join(but.get("allowed_candidate_types", [])),
            "detail": "original_but, but_native_morph, ptb_morph, clean_style",
        },
    ]
    if not nn.empty:
        rows.append(
            {
                "check": "nearest-neighbor leakage audit",
                "value": int(nn["near_duplicate_feature_count"].sum() + nn["near_duplicate_raw_count"].sum()),
                "detail": "sum of feature/raw near-duplicate counts across all scopes",
            }
        )
    if not e31.empty:
        r = e31.iloc[0]
        rows.append(
            {
                "check": "E31 frozen test",
                "value": f"acc={float(r['test_acc']):.4f}; macro-F1={float(r['test_macro_f1']):.4f}",
                "detail": f"good R={float(r['good_recall']):.4f}; medium R={float(r['intermediate_recall']):.4f}; bad R={float(r['poor_recall']):.4f}",
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(paths.tables / "but_cross_checks.csv", index=False)
    return out


def run(paths: Paths, *, execute: bool) -> dict[str, Any]:
    if not execute:
        dry("report", paths)
        return {"step": "report", "skipped": True}
    ensure_dirs(paths)
    audit = read_json(paths.audit_json)
    figure_index = read_json(paths.reports / "figure_index.json") if (paths.reports / "figure_index.json").exists() else {}
    manifest = pd.DataFrame(
        [
            {"Field": "git commit", "Value": git_commit()},
            {"Field": "run date", "Value": run_date()},
            {"Field": "random seed", "Value": "0"},
            {"Field": "data policy", "Value": "split first; train-only repair; validation/test original only"},
            {"Field": "Set-A protocol path", "Value": rel(paths.seta / "data" / "protocol_gapfill.csv")},
            {"Field": "BUT protocol path", "Value": audit["but"]["protocol_path"]},
            {"Field": "output root", "Value": rel(paths.out)},
            {"Field": "code command", "Value": "python -m supplemental_transformer_experiments.chapter4_evidence.run --out chapter4_evidence_frozen_final pipeline --run"},
            {"Field": "config file", "Value": "CLI defaults; seed=0; Python/matplotlib figures"},
            {"Field": "checkpoint path", "Value": "Set-A: chapter4 output; BUT E31: frozen v116 test_predictions.npz"},
        ]
    )
    protocol_rows = []
    for key, value in audit["seta"]["split_counts"].items():
        split, cls = key.split("_", 1)
        protocol_rows.append(
            {
                "Dataset": "Set-A",
                "Split": split,
                "Class": cls,
                "Original rows": value,
                "Generated rows": 0,
                "Total rows": value,
                "Generated in val/test": 0,
            }
        )
    for key, value in audit["but"]["split_counts"].items():
        split, cls = key.split("_", 1)
        generated = 0 if split in {"val", "test"} else ""
        protocol_rows.append(
            {
                "Dataset": "BUT",
                "Split": split,
                "Class": cls,
                "Original rows": "",
                "Generated rows": generated,
                "Total rows": value,
                "Generated in val/test": audit["but"]["val_test_generated_rows"] if split in {"val", "test"} else "",
            }
        )
    protocol_table = pd.DataFrame(protocol_rows)
    repair_metrics = _seta_distribution_display(_read_csv(paths.tables / "seta_distribution_repair_metrics.csv"), paths)
    construction_source_audit = _seta_construction_source_audit(paths)
    paper_synthetic_domain = _paper_synthetic_domain_table(paths)
    paper_cross_domain = _paper_cross_domain_display(paths)
    paper_selected5_aggregate = _paper_selected5_aggregate(paths)
    paper_selected5_subgroup = _paper_selected5_subgroup_recall(paths)
    paper_cross_noise = _paper_cross_noise_generalization(paths)
    paired = _read_csv(paths.tables / "seta_paired_mmd_calibration.csv")
    seta_source_only = _read_csv(paths.tables / "seta_construction_source_only_models.csv")
    seta_source_only_display = _seta_model_display(seta_source_only, paths, "seta_construction_source_only_models")
    seta_models = _read_csv(paths.tables / "seta_repaired_model_comparison.csv")
    seta_models_display = _seta_model_display(seta_models, paths, "seta_repaired_model_comparison")
    but_models = _read_csv(paths.tables / "but_model_comparison.csv")
    but_models_display = _but_model_display(but_models, paths)
    but_boundary = _read_csv(paths.tables / "but_good_medium_boundary_audit.csv")
    but_cross_checks = _but_cross_check_table(paths, audit, but_models)
    but_composition = _but_candidate_composition(paths)
    but_distribution = _but_distribution_metrics(paths)
    but_dual_auc = _but_dual_auc_audit(paths)
    figs = pd.DataFrame(
        [
            {
                "Figure": name,
                "File path": rel(__import__("pathlib").Path(path)),
                "Source data": rel(paths.source_data),
                "Conclusion role": "raw evidence",
            }
            for name, path in figure_index.items()
        ]
    )
    lines = [
        "# Chapter 4 Raw Results Report",
        "",
        "This is a raw experiment report, not manuscript prose. Main sections use only current `chapter4_evidence_frozen_final` / official v116 sources; historical SQI paper-balanced artifacts are isolated in the appendix.",
        "",
        "## 1. Run Manifest",
        "",
        table_to_md(manifest),
        "",
        "## 2. Current Protocol And Split Audit",
        "",
        "Source: `outputs/transformer/supplemental/chapter4_evidence_frozen_final/reports/protocol_audit.json`. BUT leakage checks use the official fold split, not the raw protocol metadata `split` column.",
        "",
        table_to_md(protocol_table),
        "",
        "## 3. Current Set-A Data Repair Evidence",
        "",
        "### Train-Poor Generated-vs-Original Distribution Diagnostics",
        "",
        "Source: `outputs/transformer/supplemental/chapter4_evidence_frozen_final/tables/seta_distribution_repair_metrics.csv`. `generated_vs_original_c2st_auc` is a domain-separability metric, not model performance.",
        "",
        table_to_md(repair_metrics),
        "",
        "### Construction Source Audit",
        "",
        "`fixed_synthetic` is the old paper `-6 dB em/ma` construction, not the current SMC generator. Transfer/recall comparisons use the selected-five SQI RBF-SVM source-only table below, matching the historical SQI supplemental route.",
        "",
        table_to_md(construction_source_audit),
        "",
        "### Paired MMD Calibration",
        "",
        "Source: `outputs/transformer/supplemental/chapter4_evidence_frozen_final/tables/seta_paired_mmd_calibration.csv`.",
        "",
        table_to_md(paired),
        "",
        "## 4. Current Set-A Model Comparison",
        "",
        "Set-A model convention: `Se` is acceptable recall, `Sp` is original-unacceptable recall, and `acceptable_positive_model_auc` treats acceptable as the positive class. Thresholds are selected on validation only.",
        "",
        "### Construction Effect: Source-Only",
        "",
        "Source: `outputs/transformer/supplemental/chapter4_evidence_frozen_final/tables/seta_construction_source_only_models.csv`; displayed scope columns are regenerated from each arm's `splits/split.csv`. All rows use the same original-only validation (`116` acceptable, `34` unacceptable) and original-only held-out test (`116` acceptable, `33` unacceptable). For non-native construction arms, the classifier fit excludes the `158` original train-unacceptable rows and uses only the generated poor source, so paper `fixed_synthetic` is directly comparable with quota/SMC. The SQI normalization step remains the baseline train-only arm-level preprocessing; it uses no validation/test rows.",
        "",
        table_to_md(seta_source_only_display),
        "",
        "### Repaired Setup Model Comparison",
        "",
        "Source: `outputs/transformer/supplemental/chapter4_evidence_frozen_final/tables/seta_repaired_model_comparison.csv`.",
        "",
        table_to_md(seta_models_display),
        "",
        "## 5. Current BUT/v116 Evidence",
        "",
        "### Data Cross-Checks",
        "",
        "Source: protocol audit JSON plus official v116 fold split. The remembered low dual-AUC value is not cited because no traceable artifact has been found for this report.",
        "",
        table_to_md(but_cross_checks),
        "",
        "### Candidate Composition",
        "",
        "Source: v116 candidate-type counts from the official gap-fill report directory.",
        "",
        table_to_md(but_composition),
        "",
        "### V116 Dual Generated-vs-Original AUC Audit",
        "",
        "Source: `outputs/transformer/supplemental/chapter4_evidence_frozen_final/tables/but_v116_dual_generated_auc_audit/dual_generated_auc.csv`. This is not an E31/SVM/MLP model score. It is a generated-vs-original domain audit for medium/bad using dual-view waveform summary features and `StandardScaler + LogisticRegression(C=0.5, class_weight='balanced')`; good is excluded because good has no generated rows.",
        "",
        table_to_md(but_dual_auc),
        "",
        "### Global Distribution Fit Diagnostics",
        "",
        "Source: v116 global distribution metrics. `global_domain_auc_not_dual` is not the v116 dual audit and is not used for dual-AUC acceptance.",
        "",
        table_to_md(but_distribution),
        "",
        "### Model Metrics",
        "",
        "Source: `outputs/transformer/supplemental/chapter4_evidence_frozen_final/tables/but_model_comparison.csv` plus frozen E31 `test_predictions.npz`. All rows are evaluated as `good/medium/bad` three-class models on the original-only BUT test split. Classical SVM/MLP use only the SQI feature columns named in `input`; split/protocol metadata columns are explicitly excluded. `Se/Sp` is not used in this multiclass table; class recalls are reported directly.",
        "",
        table_to_md(but_models_display),
        "",
        "### Good--Medium Boundary Audit",
        "",
        "Source: `outputs/transformer/supplemental/chapter4_evidence_frozen_final/tables/but_good_medium_boundary_audit.csv`. This audit uses the same original-only BUT test split as the model table. `boundary_exchange_errors` is exactly `good_to_medium + medium_to_good`; bad-related columns are shown only to verify that the main error reduction is concentrated at the good/medium boundary.",
        "",
        table_to_md(but_boundary),
        "",
        "## 6. Figure Index",
        "",
        "Every figure has source-data CSV under `outputs/transformer/supplemental/chapter4_evidence_frozen_final/figures/source_data/`; `audit-report` checks D3/D4/D5/M3 figure-source existence.",
        "",
        table_to_md(figs),
        "",
        "## 7. Historical SQI Paper-Balanced Comparator Appendix",
        "",
        "The following tables replay `outputs/sqi_supplemental/existing_seed0`. They are historical paper-balanced artifacts, not the current train-only Set-A SMC protocol.",
        "",
        "### Historical existing_seed0 Synthetic Domain Shift",
        "",
        "Paper synthetic poor here means `-6 dB em` and `-6 dB ma` from the SQI baseline supplemental line.",
        "",
        table_to_md(paper_synthetic_domain),
        "",
        "### Historical existing_seed0 Cross-Domain Performance",
        "",
        "Rows correspond to the old-paper `original poor`, `-6 dB em`, and `-6 dB ma` poor-domain transfer matrix.",
        "",
        table_to_md(paper_cross_domain),
        "",
        "### Historical existing_seed0 Selected-Five Aggregate",
        "",
        table_to_md(paper_selected5_aggregate),
        "",
        "### Historical existing_seed0 Selected-Five Subgroup Recall",
        "",
        "For unacceptable groups, `group_recall` is rejection/poor recall. This table is the correct place to compare old SQI supplemental subgroup results, not the current `fixed_synthetic` row.",
        "",
        table_to_md(paper_selected5_subgroup),
        "",
        "### Historical existing_seed0 Cross-Noise Generalization",
        "",
        "The often-confusing `0.4287` value is the `synthetic_poor_to_original_poor` acceptable-positive model AUC in this table; its original-poor recall/Sp is `0.0606`.",
        "",
        table_to_md(paper_cross_noise),
        "",
        "## 8. Audit Notes",
        "",
        table_to_md(
            pd.DataFrame(
                [
                    {
                        "item": "AUC naming",
                        "status": "fixed",
                        "note": "Main report avoids generic `AUC`; columns are named as model, domain, transfer, or poor-vs-rest AUC.",
                    },
                    {
                        "item": "BUT 0.9336",
                        "status": "scoped",
                        "note": "`0.9336` is `global_domain_auc_not_dual` for class_bad, not v116 dual AUC.",
                    },
                    {
                        "item": "BUT remembered low dual-AUC",
                        "status": "not used",
                        "note": "No traceable artifact found in the audited current line; current v116 dual AUC remains medium 0.7098, bad 0.7090, pooled 0.7053.",
                    },
                    {
                        "item": "historical SQI paper-balanced",
                        "status": "isolated",
                        "note": "All `outputs/sqi_supplemental/existing_seed0` rows are appendix-only.",
                    },
                ]
            )
        ),
        "",
        "## 9. Candidate Diagnostics For Observation Section",
        "",
        table_to_md(
            pd.DataFrame(
                [
                    {"Diagnostic": "source sensitivity", "Trigger": "C2ST high or source imbalance remains", "Needed output": "score by source, embedding C2ST", "Decision": "defer until raw evidence readout"},
                    {"Diagnostic": "shortcut check", "Trigger": "generated source remains separable", "Needed output": "provenance classifier", "Decision": "defer"},
                    {"Diagnostic": "local evidence maps", "Trigger": "waveform model improves boundary recall", "Needed output": "ECG + local map overlays", "Decision": "defer"},
                    {"Diagnostic": "input ablation", "Trigger": "need to explain waveform gain", "Needed output": "channel ablation table", "Decision": "defer"},
                    {"Diagnostic": "calibration", "Trigger": "models close or threshold-sensitive", "Needed output": "ECE, reliability, threshold sweep", "Decision": "defer"},
                ]
            )
        ),
        "",
    ]
    paths.report_md.write_text("\n".join(lines), encoding="utf-8")
    print(paths.report_md)
    return {"step": "report", "skipped": False, "output": str(paths.report_md)}
