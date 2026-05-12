from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score

try:
    from src.transformer_pipeline.noise.synthesize_local_counterfactual import old_snr_label
    from src.utils.paths import project_root
except ModuleNotFoundError:
    this_file = Path(__file__).resolve()
    root = this_file.parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from src.transformer_pipeline.noise.synthesize_local_counterfactual import old_snr_label
    from src.utils.paths import project_root


CLASS_TO_INT = {"good": 0, "medium": 1, "bad": 2}
INT_TO_CLASS = {v: k for k, v in CLASS_TO_INT.items()}
LABELS = [0, 1, 2]


def run(params: dict[str, Any] | None = None) -> dict[str, Any]:
    params = params or {}
    root = project_root()
    artifact_dir = _path(params.get("artifact_dir"), root / "outputs/transformer_factorial_local")
    model_dir_value = str(params.get("model_dir", "") or "")
    model_dir = _path(model_dir_value, Path("")) if model_dir_value else None
    out_dir = _path(params.get("out_dir"), artifact_dir / "validation")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "factorial_local_evaluation.json"
    out_md = out_dir / "factorial_local_evaluation.md"
    out_preds = out_dir / "transformer_predictions.csv"

    labels_path = artifact_dir / "datasets" / "synth_10s_125hz_labels_with_level.csv"
    if not labels_path.exists():
        labels_path = artifact_dir / "datasets" / "synth_10s_125hz_labels.csv"
    labels = pd.read_csv(labels_path).sort_values("idx").reset_index(drop=True)
    data_summary = summarize_dataset(artifact_dir, labels)

    transformer_summary: dict[str, Any] | None = None
    if model_dir is not None and model_dir.exists():
        preds, transformer_summary = evaluate_transformer_model(
            artifact_dir=artifact_dir,
            model_dir=model_dir,
            batch_size=int(params.get("batch_size", 128)),
        )
        preds.to_csv(out_preds, index=False)

    sqi_summary = load_optional_json(params.get("sqi_summary"))
    sqi_predictions = load_optional_csv(params.get("sqi_predictions"))
    sqi_eval = evaluate_prediction_csv(sqi_predictions) if sqi_predictions is not None else None

    summary = {
        "benchmark": "Factorial Local Quality Benchmark",
        "artifact_dir": _rel(artifact_dir, root),
        "model_dir": _rel(model_dir, root) if model_dir is not None else "",
        "dataset": data_summary,
        "transformer": transformer_summary,
        "sqi_ml_summary": sqi_summary,
        "sqi_ml_prediction_eval": sqi_eval,
        "outputs": {
            "json": _rel(out_json, root),
            "markdown": _rel(out_md, root),
            "transformer_predictions": _rel(out_preds, root) if transformer_summary else "",
        },
    }
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    out_md.write_text(render_markdown(summary), encoding="utf-8")
    print(render_console(summary))
    return {"step": "evaluate_factorial_local", "outputs": [_rel(out_json, root), _rel(out_md, root)]}


def summarize_dataset(artifact_dir: Path, labels: pd.DataFrame) -> dict[str, Any]:
    clean_path = artifact_dir / "datasets" / "synth_10s_125hz_clean.npz"
    noisy_path = artifact_dir / "datasets" / "synth_10s_125hz_noisy.npz"
    X_clean = np.load(clean_path)["X_clean"].astype(np.float32)
    X_noisy = np.load(noisy_path)["X_noisy"].astype(np.float32)
    measured = 10.0 * np.log10(
        (np.mean(X_clean * X_clean, axis=1) + 1e-12)
        / (np.mean((X_noisy - X_clean) ** 2, axis=1) + 1e-12)
    )
    oracle = np.array([old_snr_label(float(v)) for v in measured], dtype=object)
    y = labels["y_class"].astype(str).to_numpy()
    group_sizes = labels.groupby("counterfactual_group").size() if "counterfactual_group" in labels else pd.Series(dtype=int)
    return {
        "rows": int(len(labels)),
        "split_y_class_counts": split_counts(labels, "y_class"),
        "split_label_subtype_counts": split_counts(labels, "label_subtype") if "label_subtype" in labels else {},
        "split_noise_kind_counts": split_counts(labels, "noise_kind"),
        "split_placement_counts": split_counts(labels, "placement") if "placement" in labels else {},
        "split_snr_profile_counts": split_counts(labels, "snr_profile") if "snr_profile" in labels else {},
        "counterfactual_groups": int(labels["counterfactual_group"].nunique()) if "counterfactual_group" in labels else 0,
        "group_size_min": int(group_sizes.min()) if len(group_sizes) else 0,
        "group_size_max": int(group_sizes.max()) if len(group_sizes) else 0,
        "measured_snr_oracle_acc": float(np.mean(oracle == y)),
        "measured_snr_db": {
            "min": float(np.min(measured)),
            "mean": float(np.mean(measured)),
            "max": float(np.max(measured)),
        },
        "label_pairwise_structure": pairwise_structure(labels),
    }


def evaluate_transformer_model(*, artifact_dir: Path, model_dir: Path, batch_size: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    from src.transformer_pipeline import train as m

    flags = infer_model_flags(model_dir)
    params = {
        "artifact_dir": str(artifact_dir),
        "model_dir": str(model_dir),
        "batch_size": batch_size,
        **flags,
    }
    m.configure_from_params(params)
    m.seed_all(m.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets, _ = m.build_split_arrays()
    model, uw = m.build_model(device)
    ckpt_path = m.OUT_BEST if m.OUT_BEST.exists() else m.OUT_LAST
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    if "uw_state" in ckpt:
        uw.load_state_dict(ckpt["uw_state"], strict=True)
    model.eval()

    labels = pd.read_csv(m.IN_LABELS).sort_values("idx").reset_index(drop=True)
    pred_parts: list[pd.DataFrame] = []
    local_targets: list[np.ndarray] = []
    local_scores: list[np.ndarray] = []
    for split_name in ("val", "test"):
        loader = m.DataLoader(
            datasets[split_name],
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )
        rows: list[dict[str, Any]] = []
        offset = 0
        split_labels = labels[labels["split"].astype(str) == split_name].sort_values("idx").reset_index(drop=True)
        for batch in loader:
            x = batch["x_noisy"].to(device=device, dtype=torch.float32)
            with torch.no_grad():
                out = model(x)
            logits = out[2]
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
            pred = np.argmax(probs, axis=1)
            n = len(pred)
            meta = split_labels.iloc[offset : offset + n].copy()
            offset += n
            for i in range(n):
                row = meta.iloc[i].to_dict()
                row.update(
                    {
                        "true_int": CLASS_TO_INT[str(row["y_class"])],
                        "pred_int": int(pred[i]),
                        "pred_class": INT_TO_CLASS[int(pred[i])],
                        "prob_good": float(probs[i, 0]),
                        "prob_medium": float(probs[i, 1]),
                        "prob_bad": float(probs[i, 2]),
                        "severity_score": float(probs[i, 1] + 2.0 * probs[i, 2]),
                    }
                )
                rows.append(row)

            if m.USE_LOCAL_MASK_HEAD:
                extra_i = 3
                if m.USE_ORDINAL_HEAD:
                    extra_i += 1
                if m.USE_SNR_HEAD:
                    extra_i += 1
                local_logits = out[extra_i]
                local_targets.append(batch["local_mask"].detach().cpu().numpy().reshape(-1))
                local_scores.append(torch.sigmoid(local_logits).detach().cpu().numpy().reshape(-1))
        pred_parts.append(pd.DataFrame(rows))

    preds = pd.concat(pred_parts, ignore_index=True)
    local_alignment = None
    if local_targets and local_scores:
        y_true = np.concatenate(local_targets)
        y_score = np.concatenate(local_scores)
        y_bin = (y_true > 0.5).astype(np.uint8)
        pred_bin = (y_score > 0.5).astype(np.uint8)
        denom = int(((y_bin == 1) | (pred_bin == 1)).sum())
        corr = float(np.corrcoef(y_true, y_score)[0, 1]) if np.std(y_score) > 1e-9 and np.std(y_true) > 1e-9 else 0.0
        local_alignment = {
            "auprc": float(average_precision_score(y_bin, y_score)) if int(y_bin.sum()) > 0 else None,
            "iou_at_0_5": float(((y_bin == 1) & (pred_bin == 1)).sum() / max(1, denom)),
            "corr": corr,
        }

    return preds, {
        "checkpoint": str(ckpt_path),
        "flags": flags,
        "split_A_id_factorial_test": classification_from_preds(preds[preds["split"] == "test"]),
        "split_B_counterfactual_pairs": pairwise_prediction_metrics(preds[preds["split"] == "test"]),
        "split_C_heldout_noise_ma_test": classification_from_preds(
            preds[(preds["split"] == "test") & role_mask(preds, "noise_holdout_role", "test")]
        ),
        "split_D_heldout_snr_profile_test": classification_from_preds(
            preds[(preds["split"] == "test") & role_mask(preds, "snr_holdout_role", "test")]
        ),
        "test_by_placement": grouped_metrics(preds[preds["split"] == "test"], "placement"),
        "test_by_label_subtype": grouped_metrics(preds[preds["split"] == "test"], "label_subtype"),
        "test_by_noise_kind": grouped_metrics(preds[preds["split"] == "test"], "noise_kind"),
        "test_by_snr_profile": grouped_metrics(preds[preds["split"] == "test"], "snr_profile"),
        "group_consistency": group_consistency(preds[preds["split"] == "test"]),
        "local_alignment": local_alignment,
    }


def infer_model_flags(model_dir: Path) -> dict[str, Any]:
    flags = {
        "ordinal_head": False,
        "snr_head": False,
        "local_mask_head": False,
        "noise_type_head": False,
        "teacher_distill": False,
        "sqi_head": False,
        "input_mode": "raw",
        "cls_pool": "decoder",
    }
    probe = model_dir / "probe_summary.json"
    if probe.exists():
        hp = json.loads(probe.read_text(encoding="utf-8")).get("hyperparams", {})
        for key in list(flags):
            if key in hp:
                flags[key] = hp[key]

    ckpt_path = model_dir / "ckpt_best_val.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu")
        keys = set(ckpt.get("model_state", {}).keys())
        flags["local_mask_head"] = flags["local_mask_head"] or any("head_local_mask" in k for k in keys)
        flags["noise_type_head"] = flags["noise_type_head"] or any("noise_type_fc" in k for k in keys)
        flags["ordinal_head"] = flags["ordinal_head"] or any("ordinal_fc" in k for k in keys)
        flags["snr_head"] = flags["snr_head"] or any("snr_fc" in k for k in keys)
        flags["sqi_head"] = flags["sqi_head"] or any("sqi_fc" in k for k in keys)
    return flags


def evaluate_prediction_csv(preds: pd.DataFrame) -> dict[str, Any]:
    required = {"split", "true_int", "pred_int"}
    if not required.issubset(preds.columns):
        return {"error": f"prediction csv missing {sorted(required - set(preds.columns))}"}
    test = preds[preds["split"].astype(str) == "test"]
    return {
        "split_A_id_factorial_test": classification_from_preds(test),
        "split_B_counterfactual_pairs": pairwise_prediction_metrics(test) if "severity_score" in test.columns else None,
        "split_C_heldout_noise_ma_test": classification_from_preds(test[role_mask(test, "noise_holdout_role", "test")]),
        "split_D_heldout_snr_profile_test": classification_from_preds(test[role_mask(test, "snr_holdout_role", "test")]),
        "test_by_placement": grouped_metrics(test, "placement"),
        "test_by_label_subtype": grouped_metrics(test, "label_subtype"),
    }


def classification_from_preds(df: pd.DataFrame) -> dict[str, Any]:
    if len(df) == 0:
        return {"n": 0}
    y_true = df["true_int"].to_numpy(dtype=int)
    y_pred = df["pred_int"].to_numpy(dtype=int)
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    recalls = {}
    present_recalls: list[float] = []
    for i, label in enumerate(LABELS):
        raw_denom = int(cm[i].sum())
        value = float(cm[i, i] / max(1, raw_denom))
        recalls[INT_TO_CLASS[label]] = value
        if raw_denom > 0:
            present_recalls.append(value)
    return {
        "n": int(len(df)),
        "acc": float(accuracy_score(y_true, y_pred)),
        "balanced_acc": float(np.mean(present_recalls)) if present_recalls else 0.0,
        "macro_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
        "confusion_matrix_3x3": cm.astype(int).tolist(),
        "per_class_recall": recalls,
    }


def grouped_metrics(df: pd.DataFrame, column: str) -> dict[str, Any]:
    if column not in df.columns:
        return {}
    return {str(k): classification_from_preds(d) for k, d in df.groupby(column)}


def pairwise_structure(labels: pd.DataFrame) -> dict[str, Any]:
    if not {"counterfactual_group", "noise_kind", "snr_profile", "placement", "y_class"}.issubset(labels.columns):
        return {}
    temp = labels.copy()
    temp["true_severity"] = temp["y_class"].astype(str).map(CLASS_TO_INT).astype(float)
    return {
        "qrs_vs_noncritical_label": placement_pair_accuracy(temp, "true_severity", "qrs_overlap", "noncritical"),
        "tst_vs_noncritical_label": placement_pair_accuracy(temp, "true_severity", "tst_overlap", "noncritical"),
        "snr_monotonic_label": snr_monotonic_accuracy(temp, "true_severity"),
    }


def pairwise_prediction_metrics(df: pd.DataFrame) -> dict[str, Any]:
    if "severity_score" not in df.columns:
        return {}
    return {
        "qrs_vs_noncritical_score": placement_pair_accuracy(df, "severity_score", "qrs_overlap", "noncritical"),
        "tst_vs_noncritical_score": placement_pair_accuracy(df, "severity_score", "tst_overlap", "noncritical"),
        "bad_label_vs_good_label_score": label_pair_accuracy(df, "severity_score", "bad", "good"),
    }


def group_consistency(df: pd.DataFrame) -> dict[str, Any]:
    if "severity_score" not in df.columns:
        return {}
    return {
        "snr_monotonic_score": snr_monotonic_accuracy(df, "severity_score"),
        "placement_ordering_score": {
            "qrs_over_noncritical": placement_pair_accuracy(df, "severity_score", "qrs_overlap", "noncritical"),
            "tst_over_noncritical": placement_pair_accuracy(df, "severity_score", "tst_overlap", "noncritical"),
        },
    }


def placement_pair_accuracy(df: pd.DataFrame, score_col: str, worse: str, better: str) -> dict[str, Any]:
    needed = {"counterfactual_group", "noise_kind", "snr_profile", "placement", score_col}
    if not needed.issubset(df.columns):
        return {"n_pairs": 0}
    wins = 0
    ties = 0
    total = 0
    for _, d in df.groupby(["counterfactual_group", "noise_kind", "snr_profile"], sort=False):
        a = d[d["placement"] == worse]
        b = d[d["placement"] == better]
        if len(a) == 0 or len(b) == 0:
            continue
        av = float(a.iloc[0][score_col])
        bv = float(b.iloc[0][score_col])
        total += 1
        if av > bv:
            wins += 1
        elif abs(av - bv) <= 1e-12:
            ties += 1
    return {"n_pairs": int(total), "accuracy": float(wins / max(1, total)), "tie_frac": float(ties / max(1, total))}


def label_pair_accuracy(df: pd.DataFrame, score_col: str, worse_label: str, better_label: str) -> dict[str, Any]:
    needed = {"counterfactual_group", "noise_kind", "placement", "y_class", score_col}
    if not needed.issubset(df.columns):
        return {"n_pairs": 0}
    wins = 0
    total = 0
    for _, d in df.groupby(["counterfactual_group", "noise_kind", "placement"], sort=False):
        a = d[d["y_class"].astype(str) == worse_label]
        b = d[d["y_class"].astype(str) == better_label]
        if len(a) == 0 or len(b) == 0:
            continue
        total += 1
        if float(a[score_col].mean()) > float(b[score_col].mean()):
            wins += 1
    return {"n_pairs": int(total), "accuracy": float(wins / max(1, total))}


def snr_monotonic_accuracy(df: pd.DataFrame, score_col: str) -> dict[str, Any]:
    needed = {"counterfactual_group", "noise_kind", "placement", "snr_db", score_col}
    if not needed.issubset(df.columns):
        return {"n_pairs": 0}
    wins = 0
    total = 0
    for _, d in df.groupby(["counterfactual_group", "noise_kind", "placement"], sort=False):
        d = d.sort_values("snr_db")
        rows = d[["snr_db", score_col]].to_numpy(dtype=float)
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                low_snr, low_score = rows[i]
                high_snr, high_score = rows[j]
                if low_snr >= high_snr:
                    continue
                total += 1
                if low_score >= high_score:
                    wins += 1
    return {"n_pairs": int(total), "accuracy": float(wins / max(1, total))}


def split_counts(labels: pd.DataFrame, column: str) -> dict[str, dict[str, int]]:
    return {
        str(sp): {str(k): int(v) for k, v in d[column].value_counts(dropna=False).sort_index().to_dict().items()}
        for sp, d in labels.groupby("split")
    }


def role_mask(df: pd.DataFrame, column: str, value: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(False, index=df.index)
    return df[column].astype(str) == value


def load_optional_json(value: object) -> dict[str, Any] | None:
    if not value:
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = project_root() / path
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_optional_csv(value: object) -> pd.DataFrame | None:
    if not value:
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = project_root() / path
    if not path.exists():
        return None
    return pd.read_csv(path)


def render_console(summary: dict[str, Any]) -> str:
    lines = ["Factorial Local Quality Benchmark evaluation"]
    lines.append(f"Rows: {summary['dataset']['rows']}")
    lines.append(f"Measured-SNR oracle acc: {summary['dataset']['measured_snr_oracle_acc']:.4f}")
    tx = summary.get("transformer")
    if tx:
        a = tx["split_A_id_factorial_test"]
        lines.append(
            f"Transformer Split A test acc={a.get('acc', 0):.4f}, "
            f"macro_f1={a.get('macro_f1', 0):.4f}, medium_recall={a.get('per_class_recall', {}).get('medium', 0):.4f}"
        )
    return "\n".join(lines)


def render_markdown(summary: dict[str, Any]) -> str:
    ds = summary["dataset"]
    lines = [
        "# Factorial Local Quality Benchmark",
        "",
        f"Artifact dir: `{summary['artifact_dir']}`",
        f"Rows: `{ds['rows']}`",
        f"Counterfactual groups: `{ds['counterfactual_groups']}`",
        f"Measured-SNR oracle accuracy: `{ds['measured_snr_oracle_acc']:.4f}`",
        "",
        "## Dataset",
        "",
        f"- y_class counts: `{ds['split_y_class_counts']}`",
        f"- placement counts: `{ds['split_placement_counts']}`",
        f"- label subtype counts: `{ds['split_label_subtype_counts']}`",
        f"- label pairwise structure: `{ds['label_pairwise_structure']}`",
        "",
    ]
    tx = summary.get("transformer")
    if tx:
        lines.extend(
            [
                "## Transformer",
                "",
                metric_row("Split A ID factorial test", tx.get("split_A_id_factorial_test")),
                metric_row("Split C held-out noise=ma test", tx.get("split_C_heldout_noise_ma_test")),
                metric_row("Split D held-out SNR-profile test", tx.get("split_D_heldout_snr_profile_test")),
                f"- Split B counterfactual pairs: `{tx.get('split_B_counterfactual_pairs')}`",
                f"- Group consistency: `{tx.get('group_consistency')}`",
                f"- Local alignment: `{tx.get('local_alignment')}`",
                "",
                "### By Placement",
                "",
                "```json",
                json.dumps(tx.get("test_by_placement", {}), indent=2),
                "```",
                "",
            ]
        )
    sqi = summary.get("sqi_ml_summary")
    if sqi:
        svm = sqi.get("metrics", {}).get("svm_rbf", {}).get("test", {})
        mlp = sqi.get("metrics", {}).get("mlp", {}).get("test", {})
        lines.extend(
            [
                "## SQI-ML Baseline",
                "",
                f"- SVM test acc: `{svm.get('acc')}`, macro F1: `{svm.get('macro_f1')}`",
                f"- MLP test acc: `{mlp.get('acc')}`, macro F1: `{mlp.get('macro_f1')}`",
                "",
            ]
        )
    return "\n".join(lines)


def metric_row(name: str, metric: dict[str, Any] | None) -> str:
    if not metric or metric.get("n", 0) == 0:
        return f"- {name}: no rows"
    return (
        f"- {name}: n=`{metric['n']}`, acc=`{metric['acc']:.4f}`, "
        f"balanced_acc=`{metric['balanced_acc']:.4f}`, macro_f1=`{metric['macro_f1']:.4f}`, "
        f"medium_recall=`{metric['per_class_recall']['medium']:.4f}`"
    )


def _path(value: object, default: Path) -> Path:
    path = Path(str(value)) if value else default
    return path if path.is_absolute() else project_root() / path


def _rel(path: Path | None, root: Path) -> str:
    if path is None:
        return ""
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Factorial Local Quality Benchmark outputs.")
    parser.add_argument("--artifact_dir", default="outputs/transformer_factorial_local")
    parser.add_argument("--model_dir", default="")
    parser.add_argument("--sqi_summary", default="")
    parser.add_argument("--sqi_predictions", default="")
    parser.add_argument("--out_dir", default="")
    parser.add_argument("--batch_size", type=int, default=128)
    run(vars(parser.parse_args()))


if __name__ == "__main__":
    main()
