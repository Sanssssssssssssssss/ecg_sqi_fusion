"""QRS-guided real-noise SNR continuation for BUT matching.

This runner builds on ``real_noise_snr_but_match_grid_10s``.  It keeps the
noise source real NSTDB EM/MA/BW, but adds a small QRS-aware shaping layer:
good keeps QRS relatively clean, medium preserves QRS while shifting real noise
toward non-QRS/T-ST detail, and bad can emphasize real noise around QRS or
slightly attenuate QRS.  This is meant to strengthen the expert-usability
boundary without returning to the earlier heavy hand-written artifact rules.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from src.transformer_pipeline.e311_uformer_eval import write_json
from src.transformer_pipeline.external_benchmarks import real_noise_snr_but_match_grid_10s as rn


RUN_TAG = "e311_real_noise_qrs_guided_10h_2026_06_04"
DEFAULT_OUT_ROOT = rn.ROOT / "outputs" / "external_benchmarks" / RUN_TAG
DEFAULT_REPORT_ROOT = rn.ROOT / "reports" / "external_benchmarks" / RUN_TAG
BASE_OUT_ROOT = rn.DEFAULT_OUT_ROOT
STATE_NAME = "qrs_guided_state.json"
SPEC_NAME = "qrs_guided_specs.json"
SUMMARY_NAME = "qrs_guided_training_summary.jsonl"

H_BAD_RESCUE_05 = {
    "acc": 0.8229,
    "balanced_acc": 0.8177,
    "macro_f1": 0.7454,
    "recall_good_medium_bad": [0.887, 0.773, 0.793],
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def update_state(path: Path, **kwargs: Any) -> None:
    state = read_json(path) if path.exists() else {"started_at": rn.now_iso()}
    state.update(kwargs)
    write_json(path, state)


def qrs_profiles() -> list[dict[str, Any]]:
    """Small QRS-aware profiles.  Values are deliberately modest."""

    return [
        {
            "name": "qrs_soft_boundary",
            "good_qrs_noise_gain": 0.35,
            "good_nonqrs_noise_gain": 1.00,
            "medium_qrs_noise_gain": 0.55,
            "medium_nonqrs_noise_gain": 1.35,
            "medium_tst_noise_gain": 1.45,
            "bad_qrs_noise_gain": 1.65,
            "bad_nonqrs_noise_gain": 1.00,
            "bad_tst_noise_gain": 1.15,
            "bad_qrs_atten": 0.05,
        },
        {
            "name": "medium_detail_qrs_preserved",
            "good_qrs_noise_gain": 0.30,
            "good_nonqrs_noise_gain": 0.95,
            "medium_qrs_noise_gain": 0.42,
            "medium_nonqrs_noise_gain": 1.65,
            "medium_tst_noise_gain": 1.80,
            "bad_qrs_noise_gain": 1.55,
            "bad_nonqrs_noise_gain": 0.95,
            "bad_tst_noise_gain": 1.10,
            "bad_qrs_atten": 0.08,
        },
        {
            "name": "bad_qrs_focus_light",
            "good_qrs_noise_gain": 0.45,
            "good_nonqrs_noise_gain": 1.00,
            "medium_qrs_noise_gain": 0.65,
            "medium_nonqrs_noise_gain": 1.25,
            "medium_tst_noise_gain": 1.35,
            "bad_qrs_noise_gain": 1.90,
            "bad_nonqrs_noise_gain": 0.90,
            "bad_tst_noise_gain": 1.05,
            "bad_qrs_atten": 0.08,
        },
        {
            "name": "bad_qrs_focus_strong",
            "good_qrs_noise_gain": 0.50,
            "good_nonqrs_noise_gain": 1.00,
            "medium_qrs_noise_gain": 0.70,
            "medium_nonqrs_noise_gain": 1.20,
            "medium_tst_noise_gain": 1.30,
            "bad_qrs_noise_gain": 2.25,
            "bad_nonqrs_noise_gain": 0.85,
            "bad_tst_noise_gain": 1.00,
            "bad_qrs_atten": 0.14,
        },
        {
            "name": "nonqrs_medium_bad_guard",
            "good_qrs_noise_gain": 0.42,
            "good_nonqrs_noise_gain": 0.95,
            "medium_qrs_noise_gain": 0.48,
            "medium_nonqrs_noise_gain": 1.85,
            "medium_tst_noise_gain": 1.95,
            "bad_qrs_noise_gain": 1.75,
            "bad_nonqrs_noise_gain": 1.05,
            "bad_tst_noise_gain": 1.10,
            "bad_qrs_atten": 0.06,
        },
        {
            "name": "qrs_boundary_balanced",
            "good_qrs_noise_gain": 0.40,
            "good_nonqrs_noise_gain": 1.00,
            "medium_qrs_noise_gain": 0.58,
            "medium_nonqrs_noise_gain": 1.45,
            "medium_tst_noise_gain": 1.55,
            "bad_qrs_noise_gain": 1.85,
            "bad_nonqrs_noise_gain": 0.95,
            "bad_tst_noise_gain": 1.10,
            "bad_qrs_atten": 0.10,
        },
        {
            "name": "good_qrs_cleaner",
            "good_qrs_noise_gain": 0.22,
            "good_nonqrs_noise_gain": 0.90,
            "medium_qrs_noise_gain": 0.52,
            "medium_nonqrs_noise_gain": 1.35,
            "medium_tst_noise_gain": 1.45,
            "bad_qrs_noise_gain": 1.70,
            "bad_nonqrs_noise_gain": 0.95,
            "bad_tst_noise_gain": 1.05,
            "bad_qrs_atten": 0.08,
        },
        {
            "name": "bad_atten_only_mild",
            "good_qrs_noise_gain": 0.55,
            "good_nonqrs_noise_gain": 1.00,
            "medium_qrs_noise_gain": 0.78,
            "medium_nonqrs_noise_gain": 1.18,
            "medium_tst_noise_gain": 1.25,
            "bad_qrs_noise_gain": 1.35,
            "bad_nonqrs_noise_gain": 1.00,
            "bad_tst_noise_gain": 1.05,
            "bad_qrs_atten": 0.16,
        },
    ]


def load_base_rows(base_out_root: Path, top_base: int) -> list[dict[str, Any]]:
    path = base_out_root / "distance_leaderboard.json"
    if not path.exists():
        raise FileNotFoundError(f"Base real-noise leaderboard missing: {path}")
    rows = read_json(path)["rows"]
    selected = rows[: int(top_base)]
    # Add explicit controls even if their distance rank is slightly worse.
    by_id = {str(row["variant_id"]): row for row in rows}
    for spec_id in (
        "single_bw_snr07_triangular",
        "single_bw_snr02_uniform",
        "blockwise_switch_snr02_uniform",
        "bw_heavy_snr01_uniform",
        "single_em_snr06_triangular",
        "single_ma_snr01_uniform",
    ):
        if spec_id in by_id and by_id[spec_id] not in selected:
            selected.append(by_id[spec_id])
    return selected[: max(int(top_base), len(selected))]


def make_specs(base_rows: list[dict[str, Any]], max_specs: int) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    profiles = qrs_profiles()
    for base in base_rows:
        base_spec = dict(base["spec"])
        for profile in profiles:
            qrs = dict(profile)
            name = str(qrs.pop("name"))
            spec = json.loads(json.dumps(base_spec))
            spec["id"] = f"{base_spec['id']}__{name}"
            spec["family"] = "real_noise_qrs_guided"
            spec["base_variant_id"] = base_spec["id"]
            spec["qrs"] = qrs
            spec["note"] = f"Real NSTDB SNR candidate plus light QRS-aware shaping profile {name}."
            specs.append(spec)
    return specs[: int(max_specs)]


def score_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    out_root = Path(args.out_root)
    report_root = Path(args.report_root)
    out_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)
    update_state(out_root / STATE_NAME, status="qrs_scan_running", stage="score_specs", updated_at=rn.now_iso())
    base_rows = load_base_rows(Path(args.base_out_root), int(args.top_base))
    specs = make_specs(base_rows, int(args.max_specs))
    write_json(out_root / SPEC_NAME, {"rows": specs, "base_rows": base_rows})
    write_json(report_root / SPEC_NAME, {"rows": specs})
    nstdb = rn.ensure_nstdb_tracks(Path(args.nstdb_root))
    ptb = rn.load_ptb_artifact(Path(args.source_artifact_dir))
    but_X, but_meta = rn.load_but(Path(args.but_protocol_dir))
    rng = pd.Series(range(len(but_meta))).sample(
        n=min(len(but_meta), int(args.but_feature_max_rows) * len(rn.CLASS_NAMES)),
        random_state=int(args.seed),
    ).sort_values()
    but_feat = rn.extract_features(but_X[rng.to_numpy()], but_meta.iloc[rng.to_numpy()].reset_index(drop=True), out_root / "but_morph_features_qrs_scan.csv", max_rows=0)
    rows: list[dict[str, Any]] = []
    for i, spec in enumerate(specs, start=1):
        update_state(out_root / STATE_NAME, status="qrs_scan_running", current=spec["id"], completed=i - 1, total=len(specs), updated_at=rn.now_iso())
        rows.append(rn.score_one_spec(args, spec, ptb, nstdb["tracks"], but_X, but_meta, but_feat))
    rows = sorted(rows, key=lambda row: float(row["overall_but_like_score"]))
    pd.DataFrame(
        [
            {
                "rank": i,
                "variant_id": row["variant_id"],
                "base_variant_id": row["spec"].get("base_variant_id", ""),
                "qrs_profile": str(row["variant_id"]).split("__")[-1],
                "overall_but_like_score": row["overall_but_like_score"],
                "morph_overall_distance": row["morph_overall_distance"],
                "morph_medium_distance": row["morph_medium_distance"],
                "sqi_overall_distance": row["sqi_overall_distance"],
                "domain_separability_bal_acc": row["domain_separability_bal_acc"],
                "variant_dir": row["variant_dir"],
            }
            for i, row in enumerate(rows, start=1)
        ]
    ).to_csv(out_root / "qrs_guided_distance_leaderboard.csv", index=False)
    write_json(out_root / "qrs_guided_distance_leaderboard.json", {"rows": rows})
    write_qrs_scan_report(args, rows)
    update_state(out_root / STATE_NAME, status="qrs_scan_complete", completed=len(rows), total=len(rows), best=rows[0] if rows else None, updated_at=rn.now_iso())
    return rows


def selected_for_training(args: argparse.Namespace, n: int) -> list[dict[str, Any]]:
    path = Path(args.out_root) / "qrs_guided_distance_leaderboard.json"
    rows = read_json(path)["rows"] if path.exists() else score_specs(args)
    selected = rows[: int(n)]
    # Ensure one high-distance-control from the old pure BW best profile gets trained.
    by_id = {str(row["variant_id"]): row for row in rows}
    control = "single_bw_snr07_triangular__qrs_boundary_balanced"
    if control in by_id and by_id[control] not in selected:
        selected.append(by_id[control])
    return selected[: int(n)]


def run_quick(args: argparse.Namespace) -> list[dict[str, Any]]:
    out_root = Path(args.out_root)
    update_state(out_root / STATE_NAME, status="qrs_quick_training", stage="quick_train", updated_at=rn.now_iso())
    selected = selected_for_training(args, int(args.top_quick))
    rows: list[dict[str, Any]] = []
    for i, row in enumerate(selected, start=1):
        update_state(out_root / STATE_NAME, status="qrs_quick_training", current=row["variant_id"], completed=i - 1, total=len(selected), updated_at=rn.now_iso())
        payload = rn.train_one(args, row, "quick")
        append_jsonl(out_root / SUMMARY_NAME, payload)
        rows.append(payload)
        write_training_report(args)
    update_state(out_root / STATE_NAME, status="qrs_quick_complete", n_quick=len(rows), updated_at=rn.now_iso())
    return rows


def run_full(args: argparse.Namespace) -> list[dict[str, Any]]:
    out_root = Path(args.out_root)
    update_state(out_root / STATE_NAME, status="qrs_full_training", stage="full_train", updated_at=rn.now_iso())
    rows_path = out_root / SUMMARY_NAME
    quick_rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines() if line.strip()] if rows_path.exists() else []

    def key(row: dict[str, Any]) -> tuple[float, float, float, float]:
        rep = row.get("but_10s_eval", {}).get("but_10s_test_report", {})
        rec = rep.get("recall_good_medium_bad", [0, 0, 0])
        return (float(rep.get("macro_f1", 0.0)), float(rep.get("balanced_acc", 0.0)), min(float(rec[1]), float(rec[2])), float(row.get("ptb_test_report", {}).get("acc", 0.0)))

    selected = sorted([r for r in quick_rows if r.get("mode") == "quick" and r.get("returncode") == 0], key=key, reverse=True)[: int(args.top_full)]
    full_rows: list[dict[str, Any]] = []
    for row in selected:
        row2 = {
            "variant_id": row["spec"]["id"],
            "spec": row["spec"],
            "variant_dir": row["variant_dir"],
            **row.get("distance", {}),
        }
        payload = rn.train_one(args, row2, "full")
        append_jsonl(out_root / SUMMARY_NAME, payload)
        full_rows.append(payload)
        write_training_report(args)
    update_state(out_root / STATE_NAME, status="qrs_full_complete", n_full=len(full_rows), updated_at=rn.now_iso())
    return full_rows


def write_qrs_scan_report(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    report_root = Path(args.report_root)
    report_root.mkdir(parents=True, exist_ok=True)
    lines = [
        "# QRS-Guided Real NSTDB Noise Scan",
        "",
        "This scan keeps real NSTDB EM/MA/BW as the only noise source and applies small QRS-aware noise placement/gain changes.",
        "",
        "| rank | variant | base | profile | score | morph | medium | SQI | domain |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for i, row in enumerate(rows[:30], start=1):
        spec = row["spec"]
        profile = str(row["variant_id"]).split("__")[-1]
        lines.append(
            f"| {i} | `{row['variant_id']}` | `{spec.get('base_variant_id', '')}` | {profile} | "
            f"{row['overall_but_like_score']:.4f} | {row['morph_overall_distance']:.4f} | "
            f"{row['morph_medium_distance']:.4f} | {row['sqi_overall_distance']:.4f} | {row['domain_separability_bal_acc']:.4f} |"
        )
    (report_root / "qrs_guided_scan_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_training_report(args: argparse.Namespace) -> None:
    report_root = Path(args.report_root)
    out_root = Path(args.out_root)
    rows: list[dict[str, Any]] = []
    path = out_root / SUMMARY_NAME
    if path.exists():
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def key(row: dict[str, Any]) -> tuple[float, float, float, float]:
        rep = row.get("but_10s_eval", {}).get("but_10s_test_report", {})
        rec = rep.get("recall_good_medium_bad", [0, 0, 0])
        return (float(rep.get("macro_f1", 0.0)), float(rep.get("balanced_acc", 0.0)), min(float(rec[1]), float(rec[2])), float(row.get("ptb_test_report", {}).get("acc", 0.0)))

    ranked = sorted(rows, key=key, reverse=True)
    lines = [
        "# QRS-Guided Real Noise 10h Training Summary",
        "",
        f"Reference h_bad_rescue_05: acc {H_BAD_RESCUE_05['acc']:.4f}, balanced {H_BAD_RESCUE_05['balanced_acc']:.4f}, macro-F1 {H_BAD_RESCUE_05['macro_f1']:.4f}.",
        "",
        "| rank | mode | variant | return | BUT acc | BUT bal | BUT macro | recalls G/M/B | PTB acc | PTB bad |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |",
    ]
    for i, row in enumerate(ranked, start=1):
        rep = row.get("but_10s_eval", {}).get("but_10s_test_report", {})
        rec = rep.get("recall_good_medium_bad", [0.0, 0.0, 0.0])
        ptb = row.get("ptb_test_report", {})
        ptb_rec = ptb.get("recall_good_medium_bad", [0.0, 0.0, 0.0])
        lines.append(
            f"| {i} | {row.get('mode')} | `{row.get('spec', {}).get('id')}` | {row.get('returncode')} | "
            f"{float(rep.get('acc', 0.0)):.4f} | {float(rep.get('balanced_acc', 0.0)):.4f} | {float(rep.get('macro_f1', 0.0)):.4f} | "
            f"{float(rec[0]):.3f}/{float(rec[1]):.3f}/{float(rec[2]):.3f} | "
            f"{float(ptb.get('acc', 0.0)):.4f} | {float(ptb_rec[2] if len(ptb_rec) > 2 else 0.0):.4f} |"
        )
    report_root.mkdir(parents=True, exist_ok=True)
    (report_root / "qrs_guided_training_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(report_root / "qrs_guided_training_summary.json", {"rows": ranked})


def run_all(args: argparse.Namespace) -> None:
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    Path(args.report_root).mkdir(parents=True, exist_ok=True)
    update_state(out_root / STATE_NAME, status="running", stage=args.stage, updated_at=rn.now_iso())
    if args.stage in {"all", "score_specs"}:
        score_specs(args)
    if args.stage in {"all", "quick_train"}:
        run_quick(args)
    if args.stage in {"all", "full_train"} and int(args.top_full) > 0:
        run_full(args)
    if args.stage == "report":
        rows = read_json(out_root / "qrs_guided_distance_leaderboard.json")["rows"] if (out_root / "qrs_guided_distance_leaderboard.json").exists() else []
        if rows:
            write_qrs_scan_report(args, rows)
        write_training_report(args)
    update_state(out_root / STATE_NAME, status="complete", stage=args.stage, updated_at=rn.now_iso())


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run QRS-guided real-noise SNR BUT matching.")
    parser.add_argument("--stage", choices=("score_specs", "quick_train", "full_train", "report", "all"), default="all")
    parser.add_argument("--out_root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--report_root", default=str(DEFAULT_REPORT_ROOT))
    parser.add_argument("--base_out_root", default=str(BASE_OUT_ROOT))
    parser.add_argument("--source_artifact_dir", default=str(rn.DEFAULT_SOURCE_ARTIFACT))
    parser.add_argument("--but_protocol_dir", default=str(rn.DEFAULT_BUT_PROTOCOL))
    parser.add_argument("--protocol_out_root", default=str(rn.PROTOCOL_OUT_ROOT))
    parser.add_argument("--protocol_report_root", default=str(rn.PROTOCOL_REPORT_ROOT))
    parser.add_argument("--nstdb_root", default=str(rn.DEFAULT_NSTDB_ROOT))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--top_base", type=int, default=12)
    parser.add_argument("--max_specs", type=int, default=72)
    parser.add_argument("--scan_sample_per_class", type=int, default=60)
    parser.add_argument("--but_feature_max_rows", type=int, default=1500)
    parser.add_argument("--sqi_rows_per_class", type=int, default=24)
    parser.add_argument("--top_quick", type=int, default=10)
    parser.add_argument("--top_full", type=int, default=4)
    parser.add_argument("--train_regenerate_full", type=int, default=1)
    parser.add_argument("--quick_epochs_stage1", type=int, default=4)
    parser.add_argument("--quick_epochs_stage2", type=int, default=4)
    parser.add_argument("--full_epochs_stage1", type=int, default=10)
    parser.add_argument("--full_epochs_stage2", type=int, default=8)
    parser.add_argument("--batch_size_stage1", type=int, default=24)
    parser.add_argument("--batch_size_stage2", type=int, default=32)
    parser.add_argument("--batch_size_eval", type=int, default=256)
    parser.add_argument("--lambda_den_stage2", type=float, default=0.5)
    parser.add_argument("--lambda_cls", type=float, default=18.0)
    parser.add_argument("--class_weight", default="1.00,1.40,1.70")
    parser.add_argument("--cpu", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_all(args)


if __name__ == "__main__":
    main()
