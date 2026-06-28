"""Transformer-family waveform-only search with SQI/geometry auxiliary targets.

No tabular feature is passed to the classifier.  The model sees waveform
channels only and learns to predict both the SQI class and the 47 SQI/geometry
targets from waveform representations.  This is the clean path for testing
whether a Transformer-like architecture can internalize the feature geometry.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from support_paths import ANALYSIS_DIR, OUT_ROOT, REPORT_DIR, REPORT_ROOT, ROOT, RUN_TAG  # noqa: E402

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


ARCH_SCRIPT = SCRIPT_DIR / "run_waveform_architecture_search.py"
NODE_ID = "N17043_gm_probe"
CLASS_TO_INT = {"good": 0, "medium": 1, "bad": 2}
REFERENCE_47 = {
    "name": "47-feature tabular current best",
    "original_test_acc": 0.963548425150407,
    "original_test_macro_f1": 0.9306830954417679,
    "original_test_good_recall": 0.9563186813186814,
    "original_test_medium_recall": 0.9728874830546769,
    "original_test_bad_recall": 0.927007299270073,
}

CANDIDATES = {
    "convtx_robust3_auxheavy": {
        "arch": "convtx",
        "channels": "robust3",
        "width": 96,
        "layers": 4,
        "heads": 4,
        "aux_weight": 0.80,
        "core_aux_weight": 0.45,
        "class_weight": [1.05, 1.25, 2.20],
        "lr": 5e-4,
        "seed": 20260671,
    },
    "multipatch_robust3_auxheavy": {
        "arch": "multipatch",
        "channels": "robust3",
        "width": 96,
        "layers": 4,
        "heads": 4,
        "aux_weight": 0.85,
        "core_aux_weight": 0.45,
        "class_weight": [1.05, 1.25, 2.20],
        "lr": 5e-4,
        "seed": 20260672,
    },
    "conformer_robust3_auxheavy": {
        "arch": "conformer",
        "channels": "robust3",
        "width": 96,
        "layers": 4,
        "heads": 4,
        "aux_weight": 0.75,
        "core_aux_weight": 0.45,
        "class_weight": [1.05, 1.25, 2.20],
        "lr": 5e-4,
        "seed": 20260673,
    },
    "convtx_raw1_auxheavy": {
        "arch": "convtx",
        "channels": "raw1",
        "width": 96,
        "layers": 4,
        "heads": 4,
        "aux_weight": 0.70,
        "core_aux_weight": 0.35,
        "class_weight": [1.05, 1.25, 2.20],
        "lr": 5e-4,
        "seed": 20260674,
    },
}


def load_arch_module() -> Any:
    spec = importlib.util.spec_from_file_location("run_waveform_architecture_search", ARCH_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {ARCH_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ARCH = load_arch_module()
GEOM = ARCH.GEOM
FEATURE_COLUMNS = list(ARCH.FEATURE_COLUMNS)
CORE_AUX_IDX = list(ARCH.CORE_AUX_IDX)


class AttentionPool(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(dim) * 0.02)
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        h = self.norm(tokens)
        score = torch.einsum("btd,d->bt", h, self.query) / math.sqrt(h.shape[-1])
        weight = torch.softmax(score, dim=1)
        return torch.einsum("bt,btd->bd", weight, tokens)


class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 512):
        super().__init__()
        pos = torch.arange(max_len).float()[:, None]
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe[None, :, :], persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.shape[1], :]


def make_encoder_layer(width: int, heads: int) -> nn.TransformerEncoderLayer:
    return nn.TransformerEncoderLayer(
        d_model=width,
        nhead=heads,
        dim_feedforward=width * 4,
        dropout=0.08,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )


class ConvSubsampleTransformer(nn.Module):
    def __init__(self, in_ch: int, width: int, layers: int, heads: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, width, kernel_size=15, stride=2, padding=7),
            nn.GroupNorm(max(1, min(8, width // 8)), width),
            nn.GELU(),
            nn.Conv1d(width, width, kernel_size=11, stride=2, padding=5),
            nn.GroupNorm(max(1, min(8, width // 8)), width),
            nn.GELU(),
            nn.Conv1d(width, width, kernel_size=9, stride=2, padding=4),
            nn.GELU(),
        )
        self.pos = PositionalEncoding(width, max_len=256)
        self.encoder = nn.TransformerEncoder(make_encoder_layer(width, heads), num_layers=layers)
        self.pool = AttentionPool(width)
        self.out_dim = width * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.stem(x).transpose(1, 2)
        tokens = self.encoder(self.pos(tokens))
        attn = self.pool(tokens)
        mean = tokens.mean(dim=1)
        return torch.cat([attn, mean], dim=1)


class MultiPatchTransformer(nn.Module):
    def __init__(self, in_ch: int, width: int, layers: int, heads: int):
        super().__init__()
        branch_width = width
        self.branches = nn.ModuleList(
            [
                nn.Conv1d(in_ch, branch_width, kernel_size=25, stride=10, padding=12),
                nn.Conv1d(in_ch, branch_width, kernel_size=51, stride=20, padding=25),
                nn.Conv1d(in_ch, branch_width, kernel_size=101, stride=40, padding=50),
            ]
        )
        self.branch_norm = nn.ModuleList([nn.LayerNorm(branch_width) for _ in self.branches])
        self.scale_embed = nn.Parameter(torch.randn(len(self.branches), branch_width) * 0.02)
        self.pos = PositionalEncoding(branch_width, max_len=256)
        self.encoder = nn.TransformerEncoder(make_encoder_layer(branch_width, heads), num_layers=layers)
        self.pool = AttentionPool(branch_width)
        self.out_dim = branch_width * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pieces = []
        for i, conv in enumerate(self.branches):
            t = conv(x).transpose(1, 2)
            t = self.branch_norm[i](t) + self.scale_embed[i][None, None, :]
            pieces.append(t)
        tokens = torch.cat(pieces, dim=1)
        tokens = self.encoder(self.pos(tokens))
        return torch.cat([self.pool(tokens), tokens.mean(dim=1)], dim=1)


class ConformerBlock(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        self.ff1 = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width * 4), nn.GELU(), nn.Dropout(0.08), nn.Linear(width * 4, width), nn.Dropout(0.08))
        self.attn_norm = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, heads, dropout=0.08, batch_first=True)
        self.conv_norm = nn.LayerNorm(width)
        self.dwconv = nn.Sequential(
            nn.Conv1d(width, width, kernel_size=17, padding=8, groups=width),
            nn.GELU(),
            nn.Conv1d(width, width, kernel_size=1),
            nn.Dropout(0.08),
        )
        self.ff2 = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width * 4), nn.GELU(), nn.Dropout(0.08), nn.Linear(width * 4, width), nn.Dropout(0.08))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + 0.5 * self.ff1(x)
        h = self.attn_norm(x)
        attn, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn
        h = self.conv_norm(x).transpose(1, 2)
        x = x + self.dwconv(h).transpose(1, 2)
        x = x + 0.5 * self.ff2(x)
        return x


class ConformerLite(nn.Module):
    def __init__(self, in_ch: int, width: int, layers: int, heads: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, width, kernel_size=19, stride=4, padding=9),
            nn.GroupNorm(max(1, min(8, width // 8)), width),
            nn.GELU(),
            nn.Conv1d(width, width, kernel_size=9, stride=2, padding=4),
            nn.GELU(),
        )
        self.pos = PositionalEncoding(width, max_len=256)
        self.blocks = nn.Sequential(*[ConformerBlock(width, heads) for _ in range(layers)])
        self.pool = AttentionPool(width)
        self.out_dim = width * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.stem(x).transpose(1, 2)
        tokens = self.blocks(self.pos(tokens))
        return torch.cat([self.pool(tokens), tokens.mean(dim=1)], dim=1)


class TransformerAuxClassifier(nn.Module):
    def __init__(self, cfg: dict[str, Any], in_ch: int):
        super().__init__()
        arch = str(cfg["arch"])
        width = int(cfg["width"])
        layers = int(cfg["layers"])
        heads = int(cfg["heads"])
        if arch == "convtx":
            self.encoder = ConvSubsampleTransformer(in_ch, width, layers, heads)
        elif arch == "multipatch":
            self.encoder = MultiPatchTransformer(in_ch, width, layers, heads)
        elif arch == "conformer":
            self.encoder = ConformerLite(in_ch, width, layers, heads)
        else:
            raise ValueError(f"unknown transformer arch {arch}")
        self.class_head = nn.Sequential(
            nn.LayerNorm(self.encoder.out_dim),
            nn.Linear(self.encoder.out_dim, 192),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(192, 96),
            nn.GELU(),
            nn.Dropout(0.08),
            nn.Linear(96, 3),
        )
        self.aux_head = nn.Sequential(
            nn.LayerNorm(self.encoder.out_dim),
            nn.Linear(self.encoder.out_dim, 192),
            nn.GELU(),
            nn.Dropout(0.08),
            nn.Linear(192, len(FEATURE_COLUMNS)),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.encoder(x)
        return {"logits": self.class_head(feat), "aux_pred": self.aux_head(feat), "features": feat}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--candidates", type=str, default=",".join(CANDIDATES.keys()))
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    names = [x.strip() for x in args.candidates.split(",") if x.strip()]
    rows: list[dict[str, Any]] = []
    summaries = []
    for name in names:
        if name not in CANDIDATES:
            raise ValueError(f"unknown candidate {name}; expected {sorted(CANDIDATES)}")
        summary = train_candidate(name, CANDIDATES[name], args)
        summaries.append(summary)
        rows.extend(summary["metric_rows"])
    metrics = pd.DataFrame(rows)
    metrics_path = ANALYSIS_DIR / "waveform_transformer_search_metrics.csv"
    summary_path = ANALYSIS_DIR / "waveform_transformer_search_summary.json"
    report_path = ANALYSIS_DIR / "waveform_transformer_search_report.md"
    report_copy = REPORT_DIR / report_path.name
    metrics.to_csv(metrics_path, index=False)
    payload = {
        "created_at": now(),
        "input_contract": "waveform only; SQI/geometry features are auxiliary targets, not inputs",
        "original_used_for_training": False,
        "reference_47": REFERENCE_47,
        "metrics_csv": str(metrics_path),
        "report": str(report_copy),
        "summaries": summaries,
    }
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")
    report = render_report(metrics, payload)
    report_path.write_text(report, encoding="utf-8")
    report_copy.write_text(report, encoding="utf-8")
    print(report)


def train_candidate(name: str, cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    print(f"training {name}: {cfg}", flush=True)
    torch.manual_seed(int(cfg["seed"]))
    np.random.seed(int(cfg["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = ARCH.SyntheticWaveDataset("train", str(cfg["channels"]))
    val_ds = ARCH.SyntheticWaveDataset("val", str(cfg["channels"]))
    test_ds = ARCH.SyntheticWaveDataset("test", str(cfg["channels"]))
    norm = ARCH.FeatureNorm(mean=train_ds.mean, std=train_ds.std)
    original_ds = ARCH.OriginalWaveDataset(norm, str(cfg["channels"]))
    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.num_workers, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.num_workers, pin_memory=pin)
    original_loader = DataLoader(original_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.num_workers, pin_memory=pin)
    model = TransformerAuxClassifier(cfg, int(train_ds.x.shape[1])).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=1e-4)
    class_weight = torch.tensor(cfg["class_weight"], dtype=torch.float32, device=device)
    best_state = None
    best_epoch = 0
    best_score = -1e9
    logs = []
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        losses = []
        for batch in train_loader:
            x = batch["x"].to(device=device, dtype=torch.float32)
            y = batch["y"].to(device=device)
            aux = batch["aux"].to(device=device, dtype=torch.float32)
            out = model(x)
            cls_loss = F.cross_entropy(out["logits"], y, weight=class_weight)
            aux_loss = F.smooth_l1_loss(out["aux_pred"], aux)
            core_loss = F.smooth_l1_loss(out["aux_pred"][:, CORE_AUX_IDX], aux[:, CORE_AUX_IDX]) if CORE_AUX_IDX else torch.zeros((), device=device)
            loss = cls_loss + float(cfg["aux_weight"]) * aux_loss + float(cfg["core_aux_weight"]) * core_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        val_rep, _ = eval_loader(model, val_loader, device)
        score = val_rep["acc"] + 0.20 * val_rep["macro_f1"] + 0.15 * min(val_rep["good_recall"], val_rep["medium_recall"], val_rep["bad_recall"]) - 0.02 * val_rep["aux_mae"]
        logs.append({"epoch": epoch, "train_loss": float(np.mean(losses)), **{f"val_{k}": v for k, v in val_rep.items() if k != "confusion_3x3"}})
        print(
            f"{name} epoch={epoch}/{args.epochs} loss={np.mean(losses):.4f} "
            f"val_acc={val_rep['acc']:.4f} recalls={val_rep['good_recall']:.3f}/{val_rep['medium_recall']:.3f}/{val_rep['bad_recall']:.3f} aux_mae={val_rep['aux_mae']:.3f}",
            flush=True,
        )
        if score > best_score:
            best_score = float(score)
            best_epoch = int(epoch)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state)
    val_rep, val_probs = eval_loader(model, val_loader, device)
    test_rep, test_probs = eval_loader(model, test_loader, device)
    orig_rep, orig_probs = eval_loader(model, original_loader, device)
    threshold = ARCH.calibrate_bad_threshold(val_ds.y, val_probs)
    metric_rows = [
        ARCH.metric_row(name, "synthetic_val", val_rep),
        ARCH.metric_row(name, "synthetic_test", test_rep),
        ARCH.metric_row(f"{name}_badcal", "synthetic_test", GEOM.metric_report(test_ds.y, ARCH.apply_bad_threshold(test_probs, threshold), test_probs)),
        ARCH.metric_row(name, "original_test_all_10s+", ARCH.bucket_report(original_ds, orig_probs, "original_test_all_10s+")),
        ARCH.metric_row(f"{name}_badcal", "original_test_all_10s+", ARCH.bucket_report(original_ds, orig_probs, "original_test_all_10s+", threshold)),
        ARCH.metric_row(name, "original_all_10s+", ARCH.bucket_report(original_ds, orig_probs, "original_all_10s+")),
        ARCH.metric_row(f"{name}_badcal", "original_all_10s+", ARCH.bucket_report(original_ds, orig_probs, "original_all_10s+", threshold)),
        ARCH.metric_row(name, "bad_core_nearboundary", ARCH.bucket_report(original_ds, orig_probs, "bad_core_nearboundary")),
        ARCH.metric_row(f"{name}_badcal", "bad_core_nearboundary", ARCH.bucket_report(original_ds, orig_probs, "bad_core_nearboundary", threshold)),
        ARCH.metric_row(name, "bad_outlier_stress", ARCH.bucket_report(original_ds, orig_probs, "bad_outlier_stress")),
        ARCH.metric_row(f"{name}_badcal", "bad_outlier_stress", ARCH.bucket_report(original_ds, orig_probs, "bad_outlier_stress", threshold)),
    ]
    run_dir = OUT_ROOT / "runs" / "waveform_transformer_search" / NODE_ID / name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "ckpt_best.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "candidate_config": cfg,
            "model_kind": "waveform_only_transformer_aux_sqi_geometry",
            "input_contract": "waveform only; SQI/geometry features are auxiliary targets",
            "feature_columns": FEATURE_COLUMNS,
            "feature_train_mean": train_ds.mean,
            "feature_train_std": train_ds.std,
            "best_epoch": best_epoch,
            "bad_threshold_trainval": threshold,
            "original_rows_used_for_training": False,
        },
        ckpt_path,
    )
    pd.DataFrame(logs).to_csv(run_dir / "train_log.csv", index=False)
    np.savez_compressed(run_dir / "synthetic_test_probs.npz", probs=test_probs)
    np.savez_compressed(run_dir / "original_probs.npz", probs=orig_probs)
    return {
        "candidate": name,
        "checkpoint": str(ckpt_path),
        "best_epoch": best_epoch,
        "bad_threshold_trainval": threshold,
        "synthetic_test": test_rep,
        "original_test_all_10s+": ARCH.bucket_report(original_ds, orig_probs, "original_test_all_10s+", threshold),
        "metric_rows": metric_rows,
    }


@torch.no_grad()
def eval_loader(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[dict[str, Any], np.ndarray]:
    model.eval()
    ys = []
    probs = []
    aux_abs = []
    for batch in loader:
        x = batch["x"].to(device=device, dtype=torch.float32)
        aux = batch["aux"].to(device=device, dtype=torch.float32)
        out = model(x)
        probs.append(F.softmax(out["logits"], dim=1).detach().cpu().numpy())
        aux_abs.append(torch.mean(torch.abs(out["aux_pred"] - aux), dim=0).detach().cpu().numpy())
        ys.append(batch["y"].numpy())
    y = np.concatenate(ys)
    p = np.concatenate(probs)
    rep = GEOM.metric_report(y, p.argmax(axis=1), p)
    rep["aux_mae"] = float(np.mean(np.stack(aux_abs)))
    rep["core_aux_mae"] = float(np.mean(np.stack(aux_abs)[:, CORE_AUX_IDX])) if CORE_AUX_IDX else rep["aux_mae"]
    return rep, p


def render_report(metrics: pd.DataFrame, payload: dict[str, Any]) -> str:
    lines = [
        "# Waveform-Only Transformer Search",
        "",
        "Only Transformer-family waveform models are tested here. Inputs are waveform channels only; SQI/geometry columns are auxiliary targets, never inference inputs.",
        "",
        "| Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m | Aux MAE |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    view = metrics[metrics["bucket"].isin(["synthetic_test", "original_test_all_10s+", "bad_core_nearboundary", "bad_outlier_stress"])].copy()
    for _, row in view.iterrows():
        lines.append(
            f"| {row['candidate']} | {row['bucket']} | {float(row['acc']):.6f} | {float(row['macro_f1']):.6f} | "
            f"{float(row['good_recall']):.6f} | {float(row['medium_recall']):.6f} | {float(row['bad_recall']):.6f} | "
            f"{int(row['good_to_medium'])} | {int(row['medium_to_good'])} | {int(row['bad_to_medium'])} | {float(row.get('aux_mae', 0.0)):.4f} |"
        )
    ref = payload["reference_47"]
    lines.extend(
        [
            "",
            "## Reference",
            "",
            f"- 47-feature tabular current best original test acc: `{ref['original_test_acc']:.6f}`; recalls good/medium/bad `{ref['original_test_good_recall']:.3f}/{ref['original_test_medium_recall']:.3f}/{ref['original_test_bad_recall']:.3f}`.",
            "",
            "## Candidate Configs",
            "",
        ]
    )
    for item in payload["summaries"]:
        lines.append(f"- `{item['candidate']}`: best_epoch={item['best_epoch']}, threshold={item['bad_threshold_trainval']:.2f}, checkpoint=`{item['checkpoint']}`")
    lines.extend(["", f"Metrics CSV: `{payload['metrics_csv']}`", ""])
    return "\n".join(lines)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return str(obj)


if __name__ == "__main__":
    main()
