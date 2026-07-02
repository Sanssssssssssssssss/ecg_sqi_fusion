from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, roc_auc_score

from .common import Paths, dry, ensure_dirs, read_json, rel, write_json


EPS = 1e-6


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False) if path.exists() else pd.DataFrame()


def _safe_float(value: Any) -> float:
    try:
        if pd.isna(value):
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def _max_error(pairs: list[tuple[float, float]]) -> float:
    errors = [abs(a - b) for a, b in pairs if not (pd.isna(a) or pd.isna(b))]
    return max(errors) if errors else 0.0


def _row(
    rows: list[dict[str, Any]],
    *,
    section: str,
    table_name: str,
    source_path: Path | str,
    source_kind: str = "csv",
    calculation_script: str = "supplemental_transformer_experiments.chapter4_evidence",
    positive_class: str = "",
    split_scope: str = "",
    current_scope: str = "current_frozen_final",
    recomputable: bool = True,
    audit_status: str = "pass",
    max_abs_error: float | str = 0.0,
    notes: str = "",
) -> None:
    source = Path(source_path) if not isinstance(source_path, Path) else source_path
    rows.append(
        {
            "section": section,
            "table_name": table_name,
            "source_path": rel(source),
            "source_kind": source_kind,
            "calculation_script": calculation_script,
            "positive_class": positive_class,
            "split_scope": split_scope,
            "current_scope": current_scope,
            "recomputable": bool(recomputable),
            "audit_status": audit_status,
            "max_abs_error": max_abs_error,
            "notes": notes,
        }
    )


def _parse_confusion(raw: Any) -> dict[str, int] | list[list[int]] | None:
    if raw is None or pd.isna(raw):
        return None
    try:
        return json.loads(str(raw))
    except Exception:
        return None


def _audit_seta_model_table(rows: list[dict[str, Any]], path: Path, *, table_name: str) -> None:
    df = _read_csv(path)
    if df.empty:
        _row(
            rows,
            section="Set-A model comparison",
            table_name=table_name,
            source_path=path,
            audit_status="missing",
            max_abs_error="",
            notes="source table is missing or empty",
        )
        return

    pairs: list[tuple[float, float]] = []
    bad_rows: list[str] = []
    for _, r in df.iterrows():
        cm = _parse_confusion(r.get("confusion"))
        if not isinstance(cm, dict):
            bad_rows.append(str(r.get("run_id", "unknown")))
            continue
        tn, fp, fn, tp = (float(cm[k]) for k in ["tn", "fp", "fn", "tp"])
        n = tn + fp + fn + tp
        acc = (tn + tp) / max(1.0, n)
        se = tp / max(1.0, tp + fn)
        sp = tn / max(1.0, tn + fp)
        bal = (se + sp) / 2.0
        pairs.extend(
            [
                (acc, _safe_float(r.get("acc"))),
                (se, _safe_float(r.get("acceptable_recall", r.get("Se")))),
                (sp, _safe_float(r.get("original_unacceptable_recall", r.get("Sp")))),
                (bal, _safe_float(r.get("balanced_acc"))),
            ]
        )
    err = _max_error(pairs)
    status = "pass" if err < EPS and not bad_rows else "fail"
    _row(
        rows,
        section="Set-A model comparison",
        table_name=table_name,
        source_path=path,
        positive_class="acceptable=1; unacceptable/poor=0",
        split_scope="Set-A seed0 held-out test; validation threshold only",
        audit_status=status,
        max_abs_error=err,
        notes="confusion-derived acc/Se/Sp/balanced_acc recomputed" + (f"; unparsable rows={bad_rows}" if bad_rows else ""),
    )


def _audit_fixed_synthetic_source(rows: list[dict[str, Any]], paths: Paths) -> None:
    split_path = paths.seta_arms / "fixed_synthetic" / "splits" / "split.csv"
    paper_path = Path("outputs") / "sqi_paper_aligned" / "splits" / "split_seta_seed0_paper_balanced.csv"
    split = _read_csv(split_path)
    paper = _read_csv(paper_path)
    if split.empty or paper.empty:
        _row(
            rows,
            section="Set-A data repair",
            table_name="fixed_synthetic paper source audit",
            source_path=split_path,
            positive_class="n/a",
            split_scope="fixed_synthetic arm train generated rows",
            audit_status="missing",
            max_abs_error="",
            notes=f"missing split or paper source; paper={rel(paper_path)}",
        )
        return
    gen = split.loc[pd.to_numeric(split["is_augmented"], errors="coerce").fillna(0).astype(int).eq(1)].copy()
    paper_train_ids = set(
        paper.loc[
            paper["split"].eq("train")
            & pd.to_numeric(paper["is_augmented"], errors="coerce").fillna(0).astype(int).eq(1)
            & paper["noise_type"].isin(["em", "ma"]),
            "record_id",
        ].astype(str)
    )
    candidate_types = sorted(gen["candidate_type"].astype(str).unique().tolist())
    nontrain = int((~gen["split"].astype(str).eq("train")).sum())
    not_paper = sorted(set(gen["orig_record_id"].astype(str)) - paper_train_ids) if "orig_record_id" in gen.columns else ["missing orig_record_id"]
    status = (
        "pass"
        if len(gen) == 383
        and nontrain == 0
        and set(candidate_types).issubset({"paper_em", "paper_ma"})
        and not not_paper
        else "fail"
    )
    counts = gen["candidate_type"].astype(str).value_counts().sort_index().to_dict()
    _row(
        rows,
        section="Set-A data repair",
        table_name="fixed_synthetic paper source audit",
        source_path=split_path,
        positive_class="n/a",
        split_scope="fixed_synthetic arm train generated rows",
        audit_status=status,
        max_abs_error=0.0 if status == "pass" else "",
        notes=f"generated_n={len(gen)}; nontrain_generated={nontrain}; candidate_counts={counts}; paper_source={rel(paper_path)}; nonpaper_ids={not_paper[:5]}",
    )


def _audit_seta_construction_eval_scope(rows: list[dict[str, Any]], paths: Paths) -> None:
    checks: list[tuple[float, float]] = []
    notes: list[str] = []
    for arm in ["native_imbalanced", "fixed_synthetic", "quota_draw", "smc_gapfill"]:
        split_path = paths.seta_arms / arm / "splits" / "split.csv"
        df = _read_csv(split_path)
        if df.empty:
            notes.append(f"{arm}: missing split")
            continue
        is_aug = pd.to_numeric(df.get("is_augmented", 0), errors="coerce").fillna(0).astype(int)
        y = pd.to_numeric(df["y"], errors="coerce").astype(int)
        val = df["split"].astype(str).eq("val")
        test = df["split"].astype(str).eq("test")
        values = {
            "val_generated": int((val & is_aug.eq(1)).sum()),
            "test_generated": int((test & is_aug.eq(1)).sum()),
            "val_accept": int((val & y.eq(1)).sum()),
            "val_unaccept": int((val & y.eq(-1)).sum()),
            "test_accept": int((test & y.eq(1)).sum()),
            "test_unaccept": int((test & y.eq(-1)).sum()),
        }
        expected = {
            "val_generated": 0,
            "test_generated": 0,
            "val_accept": 116,
            "val_unaccept": 34,
            "test_accept": 116,
            "test_unaccept": 33,
        }
        for key, exp in expected.items():
            checks.append((float(values[key]), float(exp)))
        notes.append(f"{arm}: {values}")
    err = _max_error(checks)
    _row(
        rows,
        section="Set-A model comparison",
        table_name="Set-A construction original-only eval scope",
        source_path=paths.seta_arms,
        source_kind="split csvs",
        positive_class="acceptable=1; unacceptable/poor=0",
        split_scope="all construction arms use original-only validation and original-only held-out test",
        audit_status="pass" if err < EPS else "fail",
        max_abs_error=err,
        notes="; ".join(notes),
    )


def _audit_candidate_percentages(rows: list[dict[str, Any]], path: Path) -> None:
    df = _read_csv(path)
    if df.empty:
        _row(rows, section="BUT/v116", table_name="candidate composition", source_path=path, audit_status="missing", max_abs_error="")
        return
    pairs: list[tuple[float, float]] = []
    for cls, part in df.groupby("class_name"):
        total = float(part["size"].sum())
        for _, r in part.iterrows():
            expected = float(r["size"]) / total * 100.0
            pairs.append((round(expected, 2), _safe_float(r.get("pct_within_class"))))
    err = _max_error(pairs)
    _row(
        rows,
        section="BUT/v116",
        table_name="candidate composition",
        source_path=path,
        positive_class="n/a",
        split_scope="full v116 protocol by class",
        audit_status="pass" if err < EPS else "fail",
        max_abs_error=err,
        notes="pct_within_class recomputed from size within class",
    )


def _audit_but_official_split(rows: list[dict[str, Any]], audit: dict[str, Any]) -> None:
    split_dir = Path(audit["but"]["split_path"])
    split_path = split_dir / "original_region_atlas.csv"
    df = _read_csv(split_path)
    if df.empty:
        _row(rows, section="BUT/v116", table_name="official record-heldout split", source_path=split_path, audit_status="missing", max_abs_error="")
        return
    split_counts = df.groupby(["split", "class_name"]).size().to_dict()
    errors: list[tuple[float, float]] = []
    for key, expected in audit["but"]["split_counts"].items():
        split, cls = key.split("_", 1)
        got = int(split_counts.get((split, cls), 0))
        errors.append((float(got), float(expected)))
    val_test = df.loc[df["split"].isin(["val", "test"])].copy()
    generated_rows = int(pd.to_numeric(val_test.get("v116_generated", 0), errors="coerce").fillna(0).sum())
    candidate_types = sorted(str(x) for x in val_test.get("v116_candidate_type", pd.Series(dtype=str)).dropna().unique())
    err = _max_error(errors)
    status = "pass" if err < EPS and generated_rows == 0 and candidate_types == ["original_but"] else "fail"
    _row(
        rows,
        section="BUT/v116",
        table_name="official record-heldout split",
        source_path=split_path,
        source_kind="csv",
        positive_class="n/a",
        split_scope="official fold0 split; train balanced; val/test original only",
        audit_status=status,
        max_abs_error=err,
        notes=f"val/test generated rows={generated_rows}; val/test candidate types={candidate_types}; raw protocol split column is not used for leakage audit",
    )


def _audit_but_binary_or_multiclass(rows: list[dict[str, Any]], path: Path, pred_path: Path) -> None:
    df = _read_csv(path)
    if df.empty:
        _row(rows, section="BUT/v116", table_name="BUT model metrics", source_path=path, audit_status="missing", max_abs_error="")
        return
    pairs: list[tuple[float, float]] = []
    bad_rows: list[str] = []
    for _, r in df.iterrows():
        run_id = str(r.get("run_id", "unknown"))
        cm = _parse_confusion(r.get("confusion"))
        if isinstance(cm, dict):
            tn, fp, fn, tp = (float(cm[k]) for k in ["tn", "fp", "fn", "tp"])
            n = tn + fp + fn + tp
            pairs.append(((tn + tp) / max(1.0, n), _safe_float(r.get("test_acc"))))
            pairs.append((tp / max(1.0, tp + fn), _safe_float(r.get("good_recall", r.get("Se")))))
            sp = tn / max(1.0, tn + fp)
            pairs.append((sp, _safe_float(r.get("Sp")) if "Sp" in df.columns else sp))
        elif isinstance(cm, list):
            arr = np.asarray(cm, dtype=float)
            if arr.ndim != 2 or arr.shape != (3, 3):
                bad_rows.append(run_id)
                continue
            n = arr.sum()
            recalls = np.divide(np.diag(arr), arr.sum(axis=1), out=np.zeros(3), where=arr.sum(axis=1) != 0)
            pairs.extend(
                [
                    (float(np.trace(arr) / max(1.0, n)), _safe_float(r.get("test_acc"))),
                    (float(recalls[0]), _safe_float(r.get("good_recall"))),
                    (float(recalls[1]), _safe_float(r.get("intermediate_recall"))),
                    (float(recalls[2]), _safe_float(r.get("poor_recall"))),
                ]
            )
        else:
            bad_rows.append(run_id)

    if pred_path.exists():
        z = np.load(pred_path, allow_pickle=True)
        y = z["y"].astype(int)
        probs = z["probs"].astype(float)
        pred = probs.argmax(axis=1)
        e31 = df.loc[df["run_id"].eq("but_e31_wave_mechanism_conformer")]
        if not e31.empty:
            r = e31.iloc[0]
            pairs.extend(
                [
                    (float(accuracy_score(y, pred)), _safe_float(r.get("test_acc"))),
                    (float(f1_score(y, pred, average="macro", zero_division=0)), _safe_float(r.get("test_macro_f1"))),
                    (float(recall_score(y, pred, labels=[0, 1, 2], average=None, zero_division=0)[2]), _safe_float(r.get("poor_recall"))),
                    (float(roc_auc_score((y == 2).astype(int), probs[:, 2])), _safe_float(r.get("poor_vs_rest_auc"))),
                ]
            )
    else:
        bad_rows.append("missing E31 test_predictions.npz")

    err = _max_error(pairs)
    _row(
        rows,
        section="BUT/v116",
        table_name="BUT model metrics",
        source_path=path,
        positive_class="SQI baselines: good=1; E31 poor_vs_rest: poor=1",
        split_scope="official v116 fold0 held-out test",
        audit_status="pass" if err < EPS and not bad_rows else "fail",
        max_abs_error=err,
        notes="confusion and E31 npz metrics recomputed" + (f"; issues={bad_rows}" if bad_rows else ""),
    )


def _audit_but_boundary(rows: list[dict[str, Any]], path: Path, model_path: Path) -> None:
    df = _read_csv(path)
    models = _read_csv(model_path)
    if df.empty:
        _row(rows, section="BUT/v116", table_name="good-medium boundary audit", source_path=path, audit_status="missing", max_abs_error="")
        return
    pairs: list[tuple[float, float]] = []
    bad: list[str] = []
    for _, r in df.iterrows():
        model = str(r["model"])
        pairs.append((float(int(r["good_to_medium"]) + int(r["medium_to_good"])), _safe_float(r["boundary_exchange_errors"])))
        if not models.empty:
            source = models.loc[models["model"].eq("E31 wave-mechanism Conformer" if model == "Conformer" else model)]
            if not source.empty:
                cm = np.asarray(_parse_confusion(source.iloc[0]["confusion"]), dtype=int)
                pairs.extend(
                    [
                        (float(cm[0, 1]), _safe_float(r["good_to_medium"])),
                        (float(cm[0, 2]), _safe_float(r["good_to_bad"])),
                        (float(cm[1, 0]), _safe_float(r["medium_to_good"])),
                        (float(cm[2, 0]), _safe_float(r["bad_to_good"])),
                    ]
                )
    err = _max_error(pairs)
    _row(
        rows,
        section="BUT/v116",
        table_name="good-medium boundary audit",
        source_path=path,
        positive_class="n/a",
        split_scope="official v116 original-only held-out test; three-class confusion",
        audit_status="pass" if err < EPS and not bad else "fail",
        max_abs_error=err,
        notes="boundary counts recomputed from three-class confusion" + (f"; issues={bad}" if bad else ""),
    )


def _audit_figures(rows: list[dict[str, Any]], paths: Paths) -> None:
    expected = {
        "fig_D3_seta_ours_vs_paper_em_ma_distribution": "fig_D3_seta_ours_vs_paper_em_ma_pca.csv",
        "fig_D4_but_medium_bad_gapfill_distribution": "fig_D4_but_medium_bad_pca.csv",
        "fig_D5_but_v116_generation_examples": "fig_D5_but_v116_generation_examples_source_data.csv",
        "fig_M3_but_good_medium_boundary_audit": "fig_M3_boundary_counts.csv",
        "fig_M4_but_mlp_error_conformer_correct_examples": "fig_M4_mlp_error_conformer_correct_examples.csv",
    }
    for fig, source_name in expected.items():
        figure_files = [paths.figures / f"{fig}.{ext}" for ext in ["png", "svg", "pdf", "tiff"]]
        source_path = paths.source_data / source_name
        missing = [rel(p) for p in figure_files + [source_path] if not p.exists()]
        _row(
            rows,
            section="figures",
            table_name=fig,
            source_path=source_path,
            source_kind="figure/source-data",
            positive_class="n/a",
            split_scope="chapter4 figure export",
            audit_status="pass" if not missing else "fail",
            max_abs_error=0.0 if not missing else "",
            notes="all png/svg/pdf/tiff plus source data exist" if not missing else "missing: " + "; ".join(missing),
        )


def _audit_0529(rows: list[dict[str, Any]], paths: Paths) -> None:
    report_text = paths.report_md.read_text(encoding="utf-8") if paths.report_md.exists() else ""
    in_current_report = "0.529" in report_text
    _row(
        rows,
        section="artifact search",
        table_name="untraceable 0.529 dual-AUC memory",
        source_path=paths.report_md,
        source_kind="md",
        calculation_script="manual historical artifact search plus current report scan",
        positive_class="generated-vs-original dual audit would use generated=1",
        split_scope="v116 medium/bad only",
        current_scope="historical_unverified",
        recomputable=False,
        audit_status="fail" if in_current_report else "pass",
        max_abs_error="",
        notes="No traceable current artifact for 0.529 is used here; current report must not cite it. If needed, rerun a separate data-repair experiment rather than reuse memory.",
    )


def run(paths: Paths, *, execute: bool) -> dict[str, Any]:
    if not execute:
        dry("audit-report", paths)
        return {"step": "audit-report", "skipped": True}
    ensure_dirs(paths)
    audit = read_json(paths.audit_json)
    rows: list[dict[str, Any]] = []

    _audit_seta_model_table(rows, paths.tables / "seta_construction_source_only_models.csv", table_name="Set-A source-only construction models")
    _audit_seta_model_table(rows, paths.tables / "seta_repaired_model_comparison.csv", table_name="Set-A repaired setup models")
    _audit_fixed_synthetic_source(rows, paths)
    _audit_seta_construction_eval_scope(rows, paths)
    _audit_candidate_percentages(rows, paths.tables / "but_candidate_type_composition.csv")
    _audit_but_official_split(rows, audit)
    _audit_but_binary_or_multiclass(
        rows,
        paths.tables / "but_model_comparison.csv",
        Path("outputs") / "transformer" / "v116_e31" / "runs" / "gm_mechanism_repair_suite" / "E31_wave_mechanism_conformer_fold0_seed0" / "test_predictions.npz",
    )
    _audit_but_boundary(rows, paths.tables / "but_good_medium_boundary_audit.csv", paths.tables / "but_model_comparison.csv")
    _audit_figures(rows, paths)
    _audit_0529(rows, paths)

    manifest = pd.DataFrame(rows)
    csv_path = paths.tables / "report_audit_manifest.csv"
    json_path = paths.reports / "report_audit_manifest.json"
    manifest.to_csv(csv_path, index=False)
    failures = manifest.loc[manifest["audit_status"].eq("fail")].to_dict(orient="records")
    warnings = manifest.loc[manifest["audit_status"].isin(["warn", "missing"])].to_dict(orient="records")
    status = "pass" if not failures and not warnings else ("fail" if failures else "warn")
    payload = {
        "status": status,
        "failures": failures,
        "warnings": warnings,
        "manifest_csv": rel(csv_path),
        "row_count": int(len(manifest)),
    }
    write_json(json_path, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return {"step": "audit-report", "skipped": False, "status": status, "manifest_csv": str(csv_path), "manifest_json": str(json_path)}
