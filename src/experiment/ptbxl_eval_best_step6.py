from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

try:
    from src.models import ptbxl_step6_train_mtl_transformer as m
except ModuleNotFoundError:
    this_file = Path(__file__).resolve()
    root = this_file.parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from src.models import ptbxl_step6_train_mtl_transformer as m


def main() -> None:
    m.seed_all(m.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datasets, split_info = m.build_split_arrays()
    print(f"device={device}")
    for sp in ["train", "val", "test"]:
        print(f"[{sp}] n={split_info[sp]['n']} y_dist={split_info[sp]['y_dist_str']}")

    val_loader = m.DataLoader(
        datasets["val"],
        batch_size=m.BATCH_SIZE,
        shuffle=False,
        num_workers=m.NUM_WORKERS,
        pin_memory=m.PIN_MEMORY,
    )
    test_loader = m.DataLoader(
        datasets["test"],
        batch_size=m.BATCH_SIZE,
        shuffle=False,
        num_workers=m.NUM_WORKERS,
        pin_memory=m.PIN_MEMORY,
    )

    model, uw = m.build_model(device)

    ckpt_path = m.OUT_BEST
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Best checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    if "uw_state" in ckpt:
        uw.load_state_dict(ckpt["uw_state"], strict=True)

    # Keep outputs separate from training step6 artifacts.
    eval_dir = m.OUT_DIR / "eval_best"
    eval_debug_dir = eval_dir / "debug"
    eval_dir.mkdir(parents=True, exist_ok=True)
    eval_debug_dir.mkdir(parents=True, exist_ok=True)

    # test report
    test_report = m.eval_test_report(model, uw, test_loader, device)
    out_test = eval_dir / "test_report_best.json"
    with out_test.open("w", encoding="utf-8") as f:
        json.dump(test_report, f, ensure_ascii=False, indent=2)

    # optional extra val report for quick overfit check
    out_val = eval_dir / "val_report_best.json"
    val_report = m.eval_test_report(model, uw, val_loader, device)
    with out_val.open("w", encoding="utf-8") as f:
        json.dump(val_report, f, ensure_ascii=False, indent=2)

    # denoise examples by class
    m.export_denoise_examples_by_class(
        model, test_loader, device,
        eval_debug_dir / "denoise_examples_test.png",
        k_per_class=5,
    )
    m.export_denoise_examples_by_class(
        model, val_loader, device,
        eval_debug_dir / "denoise_examples_val.png",
        k_per_class=5,
    )

    print(f"[saved] {out_test}")
    print(f"[saved] {out_val}")
    print(f"[saved] {eval_debug_dir / 'denoise_examples_test.png'}")
    print(f"[saved] {eval_debug_dir / 'denoise_examples_val.png'}")
    print(f"[ckpt ] {ckpt_path}")

    d = test_report.get("denoise_metrics_by_class", {})
    if d:
        for k in ["good", "medium", "bad"]:
            if k in d:
                print(f"[test:{k}] snr_improve_db_mean={d[k].get('snr_improve_db_mean', None)}")


if __name__ == "__main__":
    main()
