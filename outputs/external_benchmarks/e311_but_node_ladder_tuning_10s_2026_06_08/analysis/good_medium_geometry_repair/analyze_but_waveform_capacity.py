"""BUT-only waveform Transformer capacity diagnostic.

This diagnostic is explicitly not a formal model claim. It trains a waveform
Transformer on BUT train rows and evaluates BUT test rows to separate
architecture capacity from PTB->BUT synthetic domain coverage.
"""

from __future__ import annotations

import importlib.util
import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


ROOT = Path(r"E:\GPTProject2\ecg")
RUN_TAG = "e311_but_node_ladder_tuning_10s_2026_06_08"
OUT_ROOT = ROOT / "outputs" / "external_benchmarks" / RUN_TAG
ANALYSIS_DIR = OUT_ROOT / "analysis" / "good_medium_geometry_repair"
RUN_ROOT = OUT_ROOT / "runs" / "waveform_geometry_student" / "N17043_gm_probe" / "search"
STUDENT_SCRIPT = ANALYSIS_DIR / "run_waveform_geometry_student.py"
CLASS_TO_INT = {"good": 0, "medium": 1, "bad": 2}


class SplitDataset(Dataset):
    def __init__(self, base: Any, split: str):
        self.base = base
        self.indices = np.flatnonzero(base.split == split).astype(np.int64)
        self.y = base.y[self.indices]
        self.split = base.split[self.indices]
        self.region = base.region[self.indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        return self.base[int(self.indices[i])]


def load_student_module() -> Any:
    spec = importlib.util.spec_from_file_location("run_waveform_geometry_student", STUDENT_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {STUDENT_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@torch.no_grad()
def predict_probs(model: torch.nn.Module, ds: Dataset, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    probs: list[np.ndarray] = []
    for batch in loader:
        x = batch["x"].to(device=device, dtype=torch.float32)
        out = model(x, mask_ratio=0.0)
        probs.append(torch.softmax(out["logits"], dim=1).detach().cpu().numpy())
    return np.concatenate(probs, axis=0)


def split_prediction_frame(atlas: pd.DataFrame, name: str, ds: SplitDataset, probs: np.ndarray) -> pd.DataFrame:
    pred = probs.argmax(axis=1).astype(np.int64)
    frame = atlas.iloc[ds.indices].copy().reset_index(drop=True)
    frame.insert(0, "capacity_bucket", name)
    frame["true_int"] = ds.y.astype(np.int64)
    frame["pred_int"] = pred
    inv = {v: k for k, v in CLASS_TO_INT.items()}
    frame["true_label"] = [inv[int(v)] for v in ds.y]
    frame["pred_label"] = [inv[int(v)] for v in pred]
    frame["correct"] = pred == ds.y
    frame["prob_good"] = probs[:, 0]
    frame["prob_medium"] = probs[:, 1]
    frame["prob_bad"] = probs[:, 2]
    return frame


def report(y: np.ndarray, probs: np.ndarray) -> dict[str, Any]:
    pred = probs.argmax(axis=1)
    conf = np.zeros((3, 3), dtype=np.int64)
    for t, p in zip(y.astype(np.int64), pred.astype(np.int64)):
        conf[t, p] += 1
    recall = np.diag(conf) / np.maximum(conf.sum(axis=1), 1)
    precision = np.diag(conf) / np.maximum(conf.sum(axis=0), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return {
        "acc": float((pred == y).mean()),
        "macro_f1": float(np.nanmean(f1)),
        "good_recall": float(recall[0]),
        "medium_recall": float(recall[1]),
        "bad_recall": float(recall[2]),
        "confusion_3x3": conf.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=str, default="predtop20_sqiquery_boundary_pretrain")
    parser.add_argument("--init", action="store_true", help="initialize from the candidate checkpoint before BUT-only fine-tune")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--tag", type=str, default="")
    args = parser.parse_args()
    mod = load_student_module()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 256 if device.type == "cuda" else 96
    # Reuse the strong waveform-only architecture config, but remove teacher
    # atlas/prototype side objectives from training.  Aux heads may exist as
    # internal branches, but 47 features are not used as loss targets here.
    ckpt = torch.load(RUN_ROOT / args.candidate / "ckpt_best.pt", map_location="cpu")
    cfg = dict(ckpt["candidate_config"])
    cfg.update(
        {
            "aux_weight": 0.0,
            "core_aux_weight": 0.0,
            "hard_aux_weight": 0.0,
            "top14_aux_weight": 0.0,
            "atlas_distill_weight": 0.0,
            "neighbor_weight": 0.0,
            "rank_weight": 0.0,
            "supcon_weight": 0.0,
            "bad_aux_weight": 0.0,
            "bad_specificity_weight": 0.0,
            "stress_aux_weight": 0.0,
            "waveform_proto_weight": 0.0,
            "embedding_proto_weight": 0.0,
            "lr": 2.5e-4,
            "seed": 20261010,
        }
    )
    torch.manual_seed(int(cfg["seed"]))
    synth_train = mod.ARCH.SyntheticWaveDataset("train", str(cfg["channels"]))
    norm = mod.ARCH.FeatureNorm(mean=synth_train.mean, std=synth_train.std)
    original = mod.ARCH.OriginalWaveDataset(norm, str(cfg["channels"]))
    train_ds = SplitDataset(original, "train")
    val_ds = SplitDataset(original, "val")
    test_ds = SplitDataset(original, "test")
    counts = np.bincount(train_ds.y, minlength=3).astype(np.float32)
    class_weight = torch.tensor(counts.sum() / np.maximum(counts, 1.0), dtype=torch.float32, device=device)
    class_weight = class_weight / class_weight.mean()
    model = mod.GeometryStudent(cfg, int(synth_train.x.shape[1])).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=1e-4)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    logs: list[dict[str, Any]] = []
    best_state = None
    best_score = -1e9
    if args.init:
        model.load_state_dict(ckpt["model_state"], strict=False)
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        losses = []
        for batch in train_loader:
            x = batch["x"].to(device=device, dtype=torch.float32)
            y = batch["y"].to(device=device)
            out = model(x, mask_ratio=0.0)
            ce = F.cross_entropy(out["logits"], y, weight=class_weight)
            opt.zero_grad(set_to_none=True)
            ce.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            losses.append(float(ce.detach().cpu()))
        val_probs = predict_probs(model, val_ds, device, batch_size * 2)
        val_rep = report(val_ds.y, val_probs)
        score = val_rep["acc"] + 0.20 * val_rep["macro_f1"] + 0.10 * min(val_rep["good_recall"], val_rep["medium_recall"], val_rep["bad_recall"])
        logs.append({"epoch": epoch, "loss": float(np.mean(losses)), **{f"val_{k}": v for k, v in val_rep.items() if k != "confusion_3x3"}})
        print(
            f"epoch={epoch}/{int(args.epochs)} loss={np.mean(losses):.4f} val_acc={val_rep['acc']:.4f} "
            f"recalls={val_rep['good_recall']:.3f}/{val_rep['medium_recall']:.3f}/{val_rep['bad_recall']:.3f}",
            flush=True,
        )
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    suffix = f"_{args.tag}" if args.tag else ""
    ckpt_out = ANALYSIS_DIR / f"but_waveform_capacity_best{suffix}.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "candidate_config": cfg,
            "source_candidate": args.candidate,
            "initialized_from_source": bool(args.init),
            "epochs": int(args.epochs),
            "input_contract": "BUT-only capacity diagnostic; waveform input only; not a formal model claim",
        },
        ckpt_out,
    )
    atlas = pd.read_csv(mod.ARCH.ORIGINAL_ATLAS) if hasattr(mod.ARCH, "ORIGINAL_ATLAS") else pd.read_csv(OUT_ROOT / "original_region_boundary" / "original_region_atlas.csv")
    rows = []
    pred_frames = []
    for name, ds in [("but_train", train_ds), ("but_val", val_ds), ("but_test", test_ds)]:
        probs = predict_probs(model, ds, device, batch_size * 2)
        pred_frames.append(split_prediction_frame(atlas, name, ds, probs))
        rep = report(ds.y, probs)
        rep["bucket"] = name
        rep["confusion_3x3"] = json.dumps(rep["confusion_3x3"])
        rows.append(rep)
    out_csv = ANALYSIS_DIR / f"but_waveform_capacity_metrics{suffix}.csv"
    log_csv = ANALYSIS_DIR / f"but_waveform_capacity_train_log{suffix}.csv"
    pred_csv = ANALYSIS_DIR / f"but_waveform_capacity_predictions{suffix}.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    pd.DataFrame(logs).to_csv(log_csv, index=False)
    pd.concat(pred_frames, ignore_index=True).to_csv(pred_csv, index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"wrote {out_csv}")
    print(f"wrote {pred_csv}")
    print(f"wrote {ckpt_out}")


if __name__ == "__main__":
    main()
