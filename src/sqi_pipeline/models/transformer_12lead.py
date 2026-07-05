from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader, Dataset


LEADS_12 = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class Config:
    split_csv: str = "outputs/sqi/splits/split_seta_seed0_balanced.csv"
    resampled_dir: str = "outputs/sqi/resampled_125"
    out_dir: str = "outputs/sqi/models/transformer_12lead"
    seed: int = 0
    epochs: int = 80
    patience: int = 14
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    d_model: int = 96
    n_heads: int = 4
    n_layers: int = 2
    ff_dim: int = 192
    dropout: float = 0.10
    pooling: str = "cls"
    label_smoothing: float = 0.0
    select_best_by: str = "val_acc"
    patch_size: int = 25
    stride: int = 10
    threshold_fixed: float = 0.5
    threshold_grid: int = 2001
    max_train_batches: int = 0
    num_workers: int = 0
    device: str = "auto"


class ECGDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, record_ids: list[str]) -> None:
        self.x = torch.from_numpy(x.astype(np.float32, copy=False))
        self.y = torch.from_numpy(y.astype(np.int64, copy=False))
        self.record_ids = record_ids

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


class TwelveLeadPatchTransformer(nn.Module):
    def __init__(
        self,
        *,
        n_leads: int = 12,
        n_samples: int = 1250,
        d_model: int = 96,
        n_heads: int = 4,
        n_layers: int = 2,
        ff_dim: int = 192,
        dropout: float = 0.10,
        pooling: str = "cls",
        patch_size: int = 25,
        stride: int = 10,
    ) -> None:
        super().__init__()
        if pooling not in {"cls", "mean", "cls_mean"}:
            raise ValueError(f"pooling must be cls, mean, or cls_mean; got {pooling}")
        self.pooling = pooling
        self.patch = nn.Conv1d(n_leads, d_model, kernel_size=patch_size, stride=stride)
        n_tokens = 1 + ((n_samples - patch_size) // stride + 1)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Parameter(torch.zeros(1, n_tokens + 1, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        head_in = d_model * 2 if pooling == "cls_mean" else d_model
        self.head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Dropout(dropout),
            nn.Linear(head_in, 2),
        )
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 12)
        h = self.patch(x.transpose(1, 2)).transpose(1, 2)
        cls = self.cls.expand(h.shape[0], -1, -1)
        h = torch.cat([cls, h], dim=1)
        h = h + self.pos[:, : h.shape[1], :]
        h = self.encoder(h)
        if self.pooling == "cls":
            g = h[:, 0, :]
        elif self.pooling == "mean":
            g = h[:, 1:, :].mean(dim=1)
        else:
            g = torch.cat([h[:, 0, :], h[:, 1:, :].mean(dim=1)], dim=1)
        return self.head(g)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def resolve_path(root: Path, p: str | Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else root / path


def rid_to_str(x: Any) -> str:
    if isinstance(x, (np.integer, int)):
        return str(int(x))
    if isinstance(x, (np.floating, float)) and np.isfinite(x):
        return str(int(x)) if float(x).is_integer() else str(x)
    s = str(x).strip()
    if s.endswith(".0") and s[:-2].isdigit():
        return s[:-2]
    return s


def load_split_data(cfg: Config) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    root = project_root()
    split_csv = resolve_path(root, cfg.split_csv)
    resampled_dir = resolve_path(root, cfg.resampled_dir)
    df = pd.read_csv(split_csv)
    df["record_id"] = df["record_id"].apply(rid_to_str)
    df["y01"] = (df["y"].astype(int) == 1).astype(np.int64)

    x_list: list[np.ndarray] = []
    missing: list[str] = []
    for rid in df["record_id"].astype(str).tolist():
        path = resampled_dir / f"{rid}.npz"
        if not path.exists():
            missing.append(rid)
            continue
        z = np.load(path, allow_pickle=True)
        sig = z["sig_125"].astype(np.float32)
        if sig.shape != (1250, 12):
            raise ValueError(f"{path}: expected sig_125 shape (1250, 12), got {sig.shape}")
        leads = list(z["leads"].tolist())
        if leads != LEADS_12:
            raise ValueError(f"{path}: unexpected lead order {leads}")
        x_list.append(sig)
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} resampled files, examples: {missing[:5]}")

    x = np.stack(x_list, axis=0)
    y = df["y01"].to_numpy(dtype=np.int64)
    return df, x, y


def standardize_from_train(x: np.ndarray, split: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    train = split == "train"
    mu = x[train].mean(axis=(0, 1), keepdims=True)
    sd = x[train].std(axis=(0, 1), keepdims=True)
    sd = np.maximum(sd, 1e-6)
    z = (x - mu) / sd
    stats = {
        "mean_per_lead": mu.reshape(-1).astype(float).tolist(),
        "std_per_lead": sd.reshape(-1).astype(float).tolist(),
        "lead_order": LEADS_12,
    }
    return z.astype(np.float32), stats


def make_loaders(
    df: pd.DataFrame,
    x: np.ndarray,
    y: np.ndarray,
    cfg: Config,
) -> tuple[dict[str, DataLoader], dict[str, ECGDataset]]:
    split = df["split"].astype(str).to_numpy()
    datasets: dict[str, ECGDataset] = {}
    loaders: dict[str, DataLoader] = {}
    for name, shuffle in [("train", True), ("val", False), ("test", False)]:
        idx = np.where(split == name)[0]
        rids = df.iloc[idx]["record_id"].astype(str).tolist()
        datasets[name] = ECGDataset(x[idx], y[idx], rids)
        loaders[name] = DataLoader(
            datasets[name],
            batch_size=cfg.batch_size,
            shuffle=shuffle,
            num_workers=cfg.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    return loaders, datasets


def binary_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> dict[str, Any]:
    y_true = y_true.astype(int).ravel()
    prob = prob.astype(np.float64).ravel()
    pred = (prob > float(threshold)).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    n = max(1, tp + tn + fp + fn)
    acc = (tp + tn) / n
    acceptable_recall = tp / max(1, tp + fn)
    unacceptable_recall = tn / max(1, tn + fp)
    try:
        auc = float(roc_auc_score(y_true, prob))
    except ValueError:
        auc = float("nan")
    return {
        "acc": float(acc),
        "se": float(unacceptable_recall),
        "sp": float(acceptable_recall),
        "acceptable_recall": float(acceptable_recall),
        "unacceptable_recall": float(unacceptable_recall),
        "auc": auc,
        "threshold": float(threshold),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def best_threshold(y_true: np.ndarray, prob: np.ndarray, n_grid: int) -> dict[str, Any]:
    best = {"threshold": 0.5, "acc": -1.0}
    for t in np.linspace(0.0, 1.0, int(n_grid)):
        m = binary_metrics(y_true, prob, float(t))
        if m["acc"] > float(best["acc"]):
            best = m
    return best


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys: list[np.ndarray] = []
    ps: list[np.ndarray] = []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        prob = torch.softmax(logits, dim=1)[:, 1]
        ys.append(yb.cpu().numpy())
        ps.append(prob.cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps)


def train_one(cfg: Config) -> dict[str, Any]:
    seed_everything(cfg.seed)
    root = project_root()
    out_dir = resolve_path(root, cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if cfg.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.device)

    df, x_raw, y = load_split_data(cfg)
    split = df["split"].astype(str).to_numpy()
    x, norm_stats = standardize_from_train(x_raw, split)
    loaders, datasets = make_loaders(df, x, y, cfg)

    model = TwelveLeadPatchTransformer(
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        ff_dim=cfg.ff_dim,
        dropout=cfg.dropout,
        pooling=cfg.pooling,
        patch_size=cfg.patch_size,
        stride=cfg.stride,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    if cfg.select_best_by not in {"val_acc", "val_auc"}:
        raise ValueError(f"select_best_by must be val_acc or val_auc; got {cfg.select_best_by}")
    best = {"epoch": 0, "val_acc": -math.inf, "val_auc": -math.inf, "score": (-math.inf, -math.inf), "state": None}
    history: list[dict[str, Any]] = []
    t0 = time.time()
    stale = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        losses: list[float] = []
        for step, (xb, yb) in enumerate(loaders["train"], start=1):
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
            if cfg.max_train_batches and step >= cfg.max_train_batches:
                break

        y_val, p_val = predict(model, loaders["val"], device)
        val_m = binary_metrics(y_val, p_val, cfg.threshold_fixed)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else float("nan"),
            "val_acc": val_m["acc"],
            "val_se": val_m["se"],
            "val_sp": val_m["sp"],
            "val_auc": val_m["auc"],
        }
        history.append(row)

        score = (row["val_auc"], row["val_acc"]) if cfg.select_best_by == "val_auc" else (row["val_acc"], row["val_auc"])
        improved = score > best["score"]
        if improved:
            best = {
                "epoch": epoch,
                "val_acc": row["val_acc"],
                "val_auc": row["val_auc"],
                "score": score,
                "state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            }
            stale = 0
        else:
            stale += 1
        print(
            f"epoch {epoch:03d} loss={row['train_loss']:.4f} "
            f"val_acc={row['val_acc']:.4f} val_auc={row['val_auc']:.4f}",
            flush=True,
        )
        if stale >= cfg.patience:
            break

    if best["state"] is None:
        raise RuntimeError("No checkpoint was selected.")
    model.load_state_dict(best["state"])

    split_metrics: dict[str, Any] = {}
    split_probs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in ["train", "val", "test"]:
        yy, pp = predict(model, loaders[name], device)
        split_probs[name] = (yy, pp)
        split_metrics[name] = binary_metrics(yy, pp, cfg.threshold_fixed)

    y_val, p_val = split_probs["val"]
    y_test, p_test = split_probs["test"]
    val_selected = best_threshold(y_val, p_val, cfg.threshold_grid)
    test_at_val_threshold = binary_metrics(y_test, p_test, val_selected["threshold"])
    test_oracle_maxacc = best_threshold(y_test, p_test, cfg.threshold_grid)

    baseline = load_baseline_comparison(root)
    metrics = {
        "step": "sqi_12lead_transformer",
        "task": "binary acceptable(+1) vs unacceptable(-1)",
        "config": asdict(cfg),
        "device": str(device),
        "n": {name: int(len(datasets[name])) for name in ["train", "val", "test"]},
        "label_counts": label_counts(df),
        "normalization": norm_stats,
        "best_epoch": int(best["epoch"]),
        "select_best_by": cfg.select_best_by,
        "train_time_s": float(time.time() - t0),
        "threshold_fixed": cfg.threshold_fixed,
        "fixed_threshold_metrics": split_metrics,
        "val_selected_threshold": val_selected,
        "test_at_val_selected_threshold": test_at_val_threshold,
        "test_oracle_maxacc": test_oracle_maxacc,
        "baseline_comparison": baseline,
    }

    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    save_predictions(df, split_probs, cfg.threshold_fixed, val_selected["threshold"], out_dir)
    torch.save(
        {
            "model_state_dict": best["state"],
            "config": asdict(cfg),
            "normalization": norm_stats,
            "best_epoch": int(best["epoch"]),
        },
        out_dir / "best_model.pt",
    )
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out_dir / "summary.md").write_text(render_summary(metrics), encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "test_acc": split_metrics["test"]["acc"]}, indent=2))
    return metrics


def label_counts(df: pd.DataFrame) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for split, sub in df.groupby("split"):
        counts = sub["y"].astype(int).value_counts().to_dict()
        out[str(split)] = {"unacceptable": int(counts.get(-1, 0)), "acceptable": int(counts.get(1, 0))}
    return out


def save_predictions(
    df: pd.DataFrame,
    split_probs: dict[str, tuple[np.ndarray, np.ndarray]],
    threshold_fixed: float,
    threshold_val: float,
    out_dir: Path,
) -> None:
    rows: list[pd.DataFrame] = []
    for split, (_, prob) in split_probs.items():
        sub = df[df["split"].astype(str) == split].copy().reset_index(drop=True)
        sub["prob_acceptable"] = prob
        sub["pred_fixed_y01"] = (prob > threshold_fixed).astype(np.int64)
        sub["pred_val_threshold_y01"] = (prob > threshold_val).astype(np.int64)
        rows.append(sub)
    pred = pd.concat(rows, axis=0, ignore_index=True)
    pred.to_csv(out_dir / "predictions.csv", index=False)


def load_baseline_comparison(root: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    svm_table = root / "outputs/sqi/models/svm/table6_12lead_combo_sqi_seed0.csv"
    if svm_table.exists():
        df = pd.read_csv(svm_table)
        row = df[df["Group"].astype(str).str.strip() == "All SQI"]
        if not row.empty:
            r = row.iloc[0]
            out["sqi_svm_all84"] = {
                "test_acc_fixed": float(r["Ac_test"]),
                "test_auc": float(r["AUC_test"]),
                "test_maxacc": float(r["maxAcc_test"]),
                "maxacc_threshold_test": float(r["maxAcc_thr_test"]),
            }

    mlp_json = root / "outputs/sqi/models/lm_mlp/lm_mlp_test_metrics_seed0.json"
    if mlp_json.exists():
        m = json.loads(mlp_json.read_text(encoding="utf-8"))
        out["sqi_lm_mlp_all84"] = {
            "test_acc_fixed": float(m["test_metrics_fixed"]["acc"]),
            "test_auc": float(m["test_metrics_fixed"]["auc"]),
            "test_maxacc": float(m["test_maxacc"]["acc"]),
            "maxacc_threshold_test": float(m["test_maxacc"]["threshold"]),
        }

    logreg_json = root / "outputs/sqi/models/baselines/logreg/logreg_test_metrics_seed0.json"
    if logreg_json.exists():
        m = json.loads(logreg_json.read_text(encoding="utf-8"))
        out["logreg_all84"] = {
            "test_acc_fixed": float(m["test_metrics_fixed"]["acc"]),
            "test_auc": float(m["test_metrics_fixed"]["auc"]),
            "test_maxacc": float(m["test_maxacc"]["acc"]),
            "maxacc_threshold_test": float(m["test_maxacc"]["threshold"]),
        }
    return out


def render_cm(m: dict[str, Any]) -> str:
    cm = m["confusion_matrix"]
    return f"tn={cm['tn']}, fp={cm['fp']}, fn={cm['fn']}, tp={cm['tp']}"


def render_summary(metrics: dict[str, Any]) -> str:
    fixed = metrics["fixed_threshold_metrics"]["test"]
    val_thr = metrics["test_at_val_selected_threshold"]
    oracle = metrics["test_oracle_maxacc"]
    lines = [
        "# SQI 12-Lead Raw Transformer",
        "",
        "This is a small standalone comparator for the classical SQI line. It uses",
        "the same balanced Set-A split and the same 12-lead 10 s signals at 125 Hz,",
        "but trains directly on raw waveforms instead of the 84 SQI summary features.",
        "",
        "Implementation file:",
        "",
        "```text",
        "src/sqi_pipeline/models/transformer_12lead.py",
        "```",
        "",
        "Outputs are intentionally kept in one directory:",
        "",
        "```text",
        f"{metrics['config']['out_dir']}",
        "```",
        "",
        "## Setup",
        "",
        f"- split: `{metrics['config']['split_csv']}`",
        f"- resampled signals: `{metrics['config']['resampled_dir']}`",
        f"- task: {metrics['task']}",
        f"- train / val / test: {metrics['n']['train']} / {metrics['n']['val']} / {metrics['n']['test']}",
        f"- best epoch: {metrics['best_epoch']}",
        f"- train time: {metrics['train_time_s']:.1f} s",
        "",
        (
            "Model: 12-lead Conv1d patch embedding + Transformer encoder + "
            f"`{metrics['config']['pooling']}` pooling + binary classifier."
        ),
        "",
        "## Result",
        "",
        "| Evaluation | Acc | Se | Sp | AUC | Threshold | Confusion Matrix |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
        (
            f"| fixed threshold | {fixed['acc']:.4f} | {fixed['se']:.4f} | {fixed['sp']:.4f} | "
            f"{fixed['auc']:.4f} | {fixed['threshold']:.4f} | {render_cm(fixed)} |"
        ),
        (
            f"| val-selected threshold | {val_thr['acc']:.4f} | {val_thr['se']:.4f} | {val_thr['sp']:.4f} | "
            f"{val_thr['auc']:.4f} | {val_thr['threshold']:.4f} | {render_cm(val_thr)} |"
        ),
        (
            f"| test oracle maxAcc | {oracle['acc']:.4f} | {oracle['se']:.4f} | {oracle['sp']:.4f} | "
            f"{oracle['auc']:.4f} | {oracle['threshold']:.4f} | {render_cm(oracle)} |"
        ),
        "",
        "## SQI-Line Comparison",
        "",
        "| Model | Input | Fixed Test Acc | Test AUC | Test maxAcc |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    base = metrics.get("baseline_comparison", {})
    for key, label in [
        ("sqi_svm_all84", "SVM-RBF"),
        ("sqi_lm_mlp_all84", "LM-MLP"),
        ("logreg_all84", "LogReg"),
    ]:
        if key in base:
            b = base[key]
            lines.append(
                f"| {label} | 12-lead 84 SQI | {b['test_acc_fixed']:.4f} | {b['test_auc']:.4f} | {b['test_maxacc']:.4f} |"
            )
    lines.append(
        f"| Raw Transformer | 12-lead waveform | {fixed['acc']:.4f} | {fixed['auc']:.4f} | {oracle['acc']:.4f} |"
    )
    lines.extend(
        [
            "",
            "## Readout",
            "",
            "Use the fixed-threshold result as the clean model comparison. The",
            "val-selected threshold is a light post-hoc calibration check, and the",
            "test oracle maxAcc is included only because the SQI tables also report",
            "maxAcc-style thresholds.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Train a small 12-lead raw transformer on the SQI pipeline split.")
    p.add_argument("--split_csv", default=Config.split_csv)
    p.add_argument("--resampled_dir", default=Config.resampled_dir)
    p.add_argument("--out_dir", default=Config.out_dir)
    p.add_argument("--seed", type=int, default=Config.seed)
    p.add_argument("--epochs", type=int, default=Config.epochs)
    p.add_argument("--patience", type=int, default=Config.patience)
    p.add_argument("--batch_size", type=int, default=Config.batch_size)
    p.add_argument("--lr", type=float, default=Config.lr)
    p.add_argument("--weight_decay", type=float, default=Config.weight_decay)
    p.add_argument("--d_model", type=int, default=Config.d_model)
    p.add_argument("--n_heads", type=int, default=Config.n_heads)
    p.add_argument("--n_layers", type=int, default=Config.n_layers)
    p.add_argument("--ff_dim", type=int, default=Config.ff_dim)
    p.add_argument("--dropout", type=float, default=Config.dropout)
    p.add_argument("--pooling", choices=["cls", "mean", "cls_mean"], default=Config.pooling)
    p.add_argument("--label_smoothing", type=float, default=Config.label_smoothing)
    p.add_argument("--select_best_by", choices=["val_acc", "val_auc"], default=Config.select_best_by)
    p.add_argument("--patch_size", type=int, default=Config.patch_size)
    p.add_argument("--stride", type=int, default=Config.stride)
    p.add_argument("--threshold_fixed", type=float, default=Config.threshold_fixed)
    p.add_argument("--threshold_grid", type=int, default=Config.threshold_grid)
    p.add_argument("--max_train_batches", type=int, default=Config.max_train_batches)
    p.add_argument("--num_workers", type=int, default=Config.num_workers)
    p.add_argument("--device", default=Config.device)
    return Config(**vars(p.parse_args()))


def main() -> None:
    cfg = parse_args()
    train_one(cfg)


if __name__ == "__main__":
    main()
