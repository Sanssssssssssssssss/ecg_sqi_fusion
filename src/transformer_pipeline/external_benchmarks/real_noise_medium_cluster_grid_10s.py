"""Medium-cluster real-noise/QRS grid for BUT 10s.

The previous QRS-guided run showed a useful bad boundary but weak BUT medium
recall.  This runner keeps the successful bad mechanism
(`blockwise real NSTDB noise + mild QRS attenuation`) and treats medium as an
independent cluster: QRS remains usable while non-QRS/P/T/ST detail becomes
unreliable.

Experiment-only: no sqi_pipeline or mainline checkpoint changes.
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


RUN_TAG = "e311_real_noise_medium_cluster_grid_10s_2026_06_05"
DEFAULT_OUT_ROOT = rn.ROOT / "outputs" / "external_benchmarks" / RUN_TAG
DEFAULT_REPORT_ROOT = rn.ROOT / "reports" / "external_benchmarks" / RUN_TAG
STATE_NAME = "medium_cluster_state.json"
SPEC_NAME = "medium_cluster_specs.json"
SUMMARY_NAME = "medium_cluster_training_summary.jsonl"

REFERENCE = {
    "h_bad_rescue_05": {"acc": 0.8229, "balanced_acc": 0.8177, "macro_f1": 0.7454, "recalls": [0.887, 0.773, 0.793]},
    "qrs_balanced_best_calibrated": {"balanced_acc": 0.7072, "macro_f1": 0.6820, "recalls": [0.878, 0.360, 0.883]},
    "qrs_balanced_best_raw": {"balanced_acc": 0.7145, "macro_f1": 0.6997, "recalls": [0.864, 0.436, 0.844]},
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
    if state.get("status") != "failed":
        state.pop("error", None)
    write_json(path, state)


def base_templates() -> list[dict[str, Any]]:
    return [
        {
            "id": "blockwise_snr02_uniform",
            "mix": {"name": "blockwise_switch", "mode": "blockwise", "choices": ["em", "ma", "bw"]},
            "snr": {"good": [14, 18], "medium": [6, 12], "bad": [-4, 2], "sampler": "uniform"},
        },
        {
            "id": "blockwise_snr06_triangular",
            "mix": {"name": "blockwise_switch", "mode": "blockwise", "choices": ["em", "ma", "bw"]},
            "snr": {"good": [12, 16], "medium": [6, 12], "bad": [-2, 4], "sampler": "triangular"},
        },
        {
            "id": "blockwise_medium_wide_uniform",
            "mix": {"name": "blockwise_switch", "mode": "blockwise", "choices": ["em", "ma", "bw"]},
            "snr": {"good": [14, 18], "medium": [4, 12], "bad": [-4, 2], "sampler": "uniform"},
        },
        {
            "id": "blockwise_medium_overlap_triangular",
            "mix": {"name": "blockwise_switch", "mode": "blockwise", "choices": ["em", "ma", "bw"]},
            "snr": {"good": [12, 18], "medium": [4, 10], "bad": [-2, 4], "sampler": "triangular"},
        },
        {
            "id": "single_bw_medium_control",
            "mix": {"name": "single_bw", "mode": "weighted", "weights": {"bw": 1.0}},
            "snr": {"good": [12, 16], "medium": [4, 10], "bad": [-2, 4], "sampler": "uniform"},
        },
    ]


def medium_profiles() -> list[dict[str, Any]]:
    # Bad side stays close to the best QRS-guided evidence.  Medium profiles vary
    # mostly in non-QRS/TST detail emphasis while keeping medium QRS relatively clean.
    common_bad = {
        "bad_qrs_noise_gain": 1.35,
        "bad_nonqrs_noise_gain": 1.00,
        "bad_tst_noise_gain": 1.05,
        "bad_qrs_atten": 0.16,
    }
    return [
        {
            "name": "m01_medium_detail_soft",
            "good_qrs_noise_gain": 0.50,
            "good_nonqrs_noise_gain": 0.90,
            "medium_qrs_noise_gain": 0.50,
            "medium_nonqrs_noise_gain": 1.55,
            "medium_tst_noise_gain": 1.85,
            **common_bad,
        },
        {
            "name": "m02_medium_detail_strong",
            "good_qrs_noise_gain": 0.50,
            "good_nonqrs_noise_gain": 0.85,
            "medium_qrs_noise_gain": 0.42,
            "medium_nonqrs_noise_gain": 2.05,
            "medium_tst_noise_gain": 2.55,
            **common_bad,
        },
        {
            "name": "m03_medium_tst_heavy",
            "good_qrs_noise_gain": 0.55,
            "good_nonqrs_noise_gain": 0.90,
            "medium_qrs_noise_gain": 0.48,
            "medium_nonqrs_noise_gain": 1.65,
            "medium_tst_noise_gain": 3.10,
            **common_bad,
        },
        {
            "name": "m04_medium_qrs_clean_extreme",
            "good_qrs_noise_gain": 0.48,
            "good_nonqrs_noise_gain": 0.85,
            "medium_qrs_noise_gain": 0.25,
            "medium_nonqrs_noise_gain": 2.20,
            "medium_tst_noise_gain": 2.70,
            **common_bad,
        },
        {
            "name": "m05_medium_good_overlap",
            "good_qrs_noise_gain": 0.65,
            "good_nonqrs_noise_gain": 1.05,
            "medium_qrs_noise_gain": 0.55,
            "medium_nonqrs_noise_gain": 1.80,
            "medium_tst_noise_gain": 2.10,
            **common_bad,
        },
        {
            "name": "m06_medium_bad_guard",
            "good_qrs_noise_gain": 0.55,
            "good_nonqrs_noise_gain": 0.95,
            "medium_qrs_noise_gain": 0.38,
            "medium_nonqrs_noise_gain": 2.35,
            "medium_tst_noise_gain": 2.85,
            "bad_qrs_noise_gain": 1.55,
            "bad_nonqrs_noise_gain": 0.95,
            "bad_tst_noise_gain": 1.00,
            "bad_qrs_atten": 0.18,
        },
        {
            "name": "m07_medium_raw_logit_probe",
            "good_qrs_noise_gain": 0.52,
            "good_nonqrs_noise_gain": 0.95,
            "medium_qrs_noise_gain": 0.62,
            "medium_nonqrs_noise_gain": 1.45,
            "medium_tst_noise_gain": 1.95,
            "bad_qrs_noise_gain": 1.25,
            "bad_nonqrs_noise_gain": 1.00,
            "bad_tst_noise_gain": 1.05,
            "bad_qrs_atten": 0.14,
        },
        {
            "name": "m08_medium_boundary_flat",
            "good_qrs_noise_gain": 0.45,
            "good_nonqrs_noise_gain": 0.82,
            "medium_qrs_noise_gain": 0.35,
            "medium_nonqrs_noise_gain": 2.60,
            "medium_tst_noise_gain": 2.60,
            **common_bad,
        },
    ]


def class_weights() -> list[str]:
    return [
        "1.00,1.55,1.70",
        "1.00,1.75,1.55",
        "1.00,1.90,1.45",
    ]


def make_specs(max_specs: int) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for base in base_templates():
        for profile in medium_profiles():
            for cw in class_weights():
                qrs = dict(profile)
                profile_name = qrs.pop("name")
                cw_tag = cw.replace(".", "p").replace(",", "_")
                specs.append(
                    {
                        "id": f"{base['id']}__{profile_name}__cw{cw_tag}",
                        "family": "real_noise_medium_cluster",
                        "mix": base["mix"],
                        "snr": base["snr"],
                        "class_weight": cw,
                        "qrs": qrs,
                        "base_variant_id": base["id"],
                        "medium_profile": profile_name,
                        "note": "Medium-focused real NSTDB grid: QRS usable, non-QRS/P/T/ST detail unreliable; bad keeps mild fatal-QRS mechanism.",
                    }
                )
    return specs[: int(max_specs)]


def medium_score(row: dict[str, Any]) -> float:
    # Lower is better.  Medium distance matters most, but overall/domain/SQI keep
    # the candidate from becoming an artificial PTB-only construct.
    return float(
        0.42 * row.get("morph_medium_distance", 1.0)
        + 0.24 * row.get("morph_overall_distance", 1.0)
        + 0.18 * row.get("sqi_overall_distance", 1.0)
        + 0.16 * row.get("domain_separability_bal_acc", 1.0)
    )


def score_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    out_root = Path(args.out_root)
    report_root = Path(args.report_root)
    out_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)
    update_state(out_root / STATE_NAME, status="scan_running", stage="score_specs", updated_at=rn.now_iso())
    nstdb = rn.ensure_nstdb_tracks(Path(args.nstdb_root))
    write_json(out_root / "nstdb_noise_audit.json", nstdb["audit"])
    ptb = rn.load_ptb_artifact(Path(args.source_artifact_dir))
    specs = make_specs(int(args.max_specs))
    write_json(out_root / SPEC_NAME, {"rows": specs})
    write_json(report_root / SPEC_NAME, {"rows": specs})
    but_X, but_meta = rn.load_but(Path(args.but_protocol_dir))
    rng = pd.Series(range(len(but_meta))).sample(
        n=min(len(but_meta), int(args.but_feature_max_rows) * len(rn.CLASS_NAMES)),
        random_state=int(args.seed),
    ).sort_values()
    but_feat = rn.extract_features(
        but_X[rng.to_numpy()],
        but_meta.iloc[rng.to_numpy()].reset_index(drop=True),
        out_root / "but_morph_features_medium_scan.csv",
        max_rows=0,
    )
    rows: list[dict[str, Any]] = []
    for i, spec in enumerate(specs, start=1):
        update_state(out_root / STATE_NAME, status="scan_running", current=spec["id"], completed=i - 1, total=len(specs), updated_at=rn.now_iso())
        row = rn.score_one_spec(args, spec, ptb, nstdb["tracks"], but_X, but_meta, but_feat)
        row["medium_cluster_score"] = medium_score(row)
        rows.append(row)
    rows = sorted(rows, key=lambda row: float(row["medium_cluster_score"]))
    df = pd.DataFrame(
        [
            {
                "rank": i,
                "variant_id": row["variant_id"],
                "base": row["spec"].get("base_variant_id"),
                "medium_profile": row["spec"].get("medium_profile"),
                "class_weight": row["spec"].get("class_weight"),
                "medium_cluster_score": row["medium_cluster_score"],
                "overall_but_like_score": row["overall_but_like_score"],
                "morph_overall_distance": row["morph_overall_distance"],
                "morph_medium_distance": row["morph_medium_distance"],
                "sqi_overall_distance": row["sqi_overall_distance"],
                "domain_separability_bal_acc": row["domain_separability_bal_acc"],
                "variant_dir": row["variant_dir"],
            }
            for i, row in enumerate(rows, start=1)
        ]
    )
    df.to_csv(out_root / "medium_cluster_distance_leaderboard.csv", index=False)
    df.to_csv(report_root / "medium_cluster_distance_leaderboard.csv", index=False)
    write_json(out_root / "medium_cluster_distance_leaderboard.json", {"rows": rows})
    write_json(report_root / "medium_cluster_distance_leaderboard.json", {"rows": rows[:40]})
    write_scan_report(args, rows)
    update_state(out_root / STATE_NAME, status="scan_complete", completed=len(rows), total=len(rows), best=rows[0] if rows else None, updated_at=rn.now_iso())
    return rows


def report_key(row: dict[str, Any]) -> tuple[float, float, float, float, float]:
    eval_payload = row.get("but_10s_eval", {})
    bal_cal = eval_payload.get("but_10s_balanced_test_report", {})
    bal_raw = eval_payload.get("but_10s_balanced_raw_report", {})
    candidates = [rep for rep in (bal_cal, bal_raw) if rep]
    if not candidates:
        candidates = [eval_payload.get("but_10s_test_report", {})]
    best = max(candidates, key=lambda rep: float(rep.get("macro_f1", 0.0)))
    rec = best.get("recall_good_medium_bad", [0, 0, 0])
    ptb = row.get("ptb_test_report", {})
    return (
        float(best.get("macro_f1", 0.0)),
        min(float(rec[1]), float(rec[2])),
        float(rec[1]),
        float(best.get("balanced_acc", best.get("acc", 0.0))),
        float(ptb.get("acc", 0.0)),
    )


def selected_for_training(args: argparse.Namespace, n: int) -> list[dict[str, Any]]:
    path = Path(args.out_root) / "medium_cluster_distance_leaderboard.json"
    rows = read_json(path)["rows"] if path.exists() else score_specs(args)
    selected = rows[: int(n)]
    by_id = {str(row["variant_id"]): row for row in rows}
    controls = [
        "blockwise_snr02_uniform__m02_medium_detail_strong__cw1p00_1p75_1p55",
        "blockwise_snr02_uniform__m04_medium_qrs_clean_extreme__cw1p00_1p90_1p45",
        "blockwise_medium_wide_uniform__m03_medium_tst_heavy__cw1p00_1p75_1p55",
        "blockwise_medium_overlap_triangular__m05_medium_good_overlap__cw1p00_1p75_1p55",
    ]
    for cid in controls:
        if cid in by_id and by_id[cid] not in selected:
            selected.append(by_id[cid])
    return selected[: int(n)]


def run_quick(args: argparse.Namespace) -> list[dict[str, Any]]:
    out_root = Path(args.out_root)
    update_state(out_root / STATE_NAME, status="quick_training", stage="quick_train", updated_at=rn.now_iso())
    selected = selected_for_training(args, int(args.top_quick))
    rows: list[dict[str, Any]] = []
    for i, row in enumerate(selected, start=1):
        update_state(out_root / STATE_NAME, status="quick_training", current=row["variant_id"], completed=i - 1, total=len(selected), updated_at=rn.now_iso())
        payload = rn.train_one(args, row, "quick")
        append_jsonl(out_root / SUMMARY_NAME, payload)
        rows.append(payload)
        write_training_report(args)
    update_state(out_root / STATE_NAME, status="quick_complete", n_quick=len(rows), updated_at=rn.now_iso())
    return rows


def run_full(args: argparse.Namespace) -> list[dict[str, Any]]:
    out_root = Path(args.out_root)
    update_state(out_root / STATE_NAME, status="full_training", stage="full_train", updated_at=rn.now_iso())
    path = out_root / SUMMARY_NAME
    quick_rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []
    selected = sorted([row for row in quick_rows if row.get("mode") == "quick" and row.get("returncode") == 0], key=report_key, reverse=True)[: int(args.top_full)]
    rows: list[dict[str, Any]] = []
    for i, row in enumerate(selected, start=1):
        update_state(out_root / STATE_NAME, status="full_training", current=row["spec"]["id"], completed=i - 1, total=len(selected), updated_at=rn.now_iso())
        row2 = {
            "variant_id": row["spec"]["id"],
            "spec": row["spec"],
            "variant_dir": row["variant_dir"],
            **row.get("distance", {}),
        }
        payload = rn.train_one(args, row2, "full")
        append_jsonl(out_root / SUMMARY_NAME, payload)
        rows.append(payload)
        write_training_report(args)
    update_state(out_root / STATE_NAME, status="full_complete", n_full=len(rows), updated_at=rn.now_iso())
    return rows


def write_scan_report(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    report_root = Path(args.report_root)
    report_root.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Medium-Cluster Real Noise Scan",
        "",
        "Lower medium_cluster_score means closer to BUT medium while keeping overall/SQI/domain distance in check.",
        "",
        "| rank | variant | base | profile | cw | medium score | medium dist | overall | SQI | domain |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for i, row in enumerate(rows[:40], start=1):
        spec = row["spec"]
        lines.append(
            f"| {i} | `{row['variant_id']}` | `{spec.get('base_variant_id')}` | `{spec.get('medium_profile')}` | `{spec.get('class_weight')}` | "
            f"{row['medium_cluster_score']:.4f} | {row['morph_medium_distance']:.4f} | {row['morph_overall_distance']:.4f} | "
            f"{row['sqi_overall_distance']:.4f} | {row['domain_separability_bal_acc']:.4f} |"
        )
    (report_root / "medium_cluster_scan_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_training_report(args: argparse.Namespace) -> None:
    out_root = Path(args.out_root)
    report_root = Path(args.report_root)
    rows = [json.loads(line) for line in (out_root / SUMMARY_NAME).read_text(encoding="utf-8").splitlines() if line.strip()] if (out_root / SUMMARY_NAME).exists() else []
    ranked = sorted(rows, key=report_key, reverse=True)
    lines = [
        "# Medium-Cluster Real Noise Training Summary",
        "",
        f"Reference qrs balanced best raw: macro-F1 {REFERENCE['qrs_balanced_best_raw']['macro_f1']:.4f}, recalls {REFERENCE['qrs_balanced_best_raw']['recalls']}.",
        f"Reference h_bad_rescue_05 original BUT: macro-F1 {REFERENCE['h_bad_rescue_05']['macro_f1']:.4f}, recalls {REFERENCE['h_bad_rescue_05']['recalls']}.",
        "",
        "| rank | mode | variant | return | bal cal macro | bal raw macro | raw recalls G/M/B | cal recalls G/M/B | orig macro | PTB acc | PTB bad |",
        "| --- | --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: |",
    ]
    for i, row in enumerate(ranked, start=1):
        ev = row.get("but_10s_eval", {})
        cal = ev.get("but_10s_balanced_test_report", {})
        raw = ev.get("but_10s_balanced_raw_report", {})
        orig = ev.get("but_10s_test_report", {})
        cal_rec = cal.get("recall_good_medium_bad", [0.0, 0.0, 0.0])
        raw_rec = raw.get("recall_good_medium_bad", [0.0, 0.0, 0.0])
        ptb = row.get("ptb_test_report", {})
        ptb_rec = ptb.get("recall_good_medium_bad", [0.0, 0.0, 0.0])
        lines.append(
            f"| {i} | {row.get('mode')} | `{row.get('spec', {}).get('id')}` | {row.get('returncode')} | "
            f"{float(cal.get('macro_f1', 0.0)):.4f} | {float(raw.get('macro_f1', 0.0)):.4f} | "
            f"{float(raw_rec[0]):.3f}/{float(raw_rec[1]):.3f}/{float(raw_rec[2]):.3f} | "
            f"{float(cal_rec[0]):.3f}/{float(cal_rec[1]):.3f}/{float(cal_rec[2]):.3f} | "
            f"{float(orig.get('macro_f1', 0.0)):.4f} | {float(ptb.get('acc', 0.0)):.4f} | {float(ptb_rec[2] if len(ptb_rec) > 2 else 0.0):.4f} |"
        )
    report_root.mkdir(parents=True, exist_ok=True)
    (report_root / "medium_cluster_training_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(report_root / "medium_cluster_training_summary.json", {"rows": ranked, "reference": REFERENCE})


def run_all(args: argparse.Namespace) -> None:
    out_root = Path(args.out_root)
    report_root = Path(args.report_root)
    out_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)
    update_state(out_root / STATE_NAME, status="running", stage=args.stage, updated_at=rn.now_iso())
    if args.stage in {"all", "score_specs"}:
        score_specs(args)
    if args.stage in {"all", "quick_train"}:
        run_quick(args)
    if args.stage in {"all", "full_train"} and int(args.top_full) > 0:
        run_full(args)
    if args.stage == "report":
        write_training_report(args)
    update_state(out_root / STATE_NAME, status="complete", stage=args.stage, updated_at=rn.now_iso())


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run medium-focused real-noise/QRS BUT grid.")
    parser.add_argument("--stage", choices=("score_specs", "quick_train", "full_train", "report", "all"), default="all")
    parser.add_argument("--out_root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--report_root", default=str(DEFAULT_REPORT_ROOT))
    parser.add_argument("--source_artifact_dir", default=str(rn.DEFAULT_SOURCE_ARTIFACT))
    parser.add_argument("--but_protocol_dir", default=str(rn.DEFAULT_BUT_PROTOCOL))
    parser.add_argument("--protocol_out_root", default=str(rn.PROTOCOL_OUT_ROOT))
    parser.add_argument("--nstdb_root", default=str(rn.DEFAULT_NSTDB_ROOT))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max_specs", type=int, default=60)
    parser.add_argument("--scan_sample_per_class", type=int, default=60)
    parser.add_argument("--but_feature_max_rows", type=int, default=1500)
    parser.add_argument("--sqi_rows_per_class", type=int, default=24)
    parser.add_argument("--top_quick", type=int, default=18)
    parser.add_argument("--top_full", type=int, default=6)
    parser.add_argument("--quick_epochs_stage1", type=int, default=4)
    parser.add_argument("--quick_epochs_stage2", type=int, default=4)
    parser.add_argument("--full_epochs_stage1", type=int, default=10)
    parser.add_argument("--full_epochs_stage2", type=int, default=8)
    parser.add_argument("--batch_size_stage1", type=int, default=24)
    parser.add_argument("--batch_size_stage2", type=int, default=32)
    parser.add_argument("--batch_size_eval", type=int, default=256)
    parser.add_argument("--lambda_den_stage2", type=float, default=0.5)
    parser.add_argument("--lambda_cls", type=float, default=18.0)
    parser.add_argument("--class_weight", default="1.00,1.55,1.70")
    parser.add_argument("--train_regenerate_full", type=int, default=1)
    parser.add_argument("--cpu", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    try:
        run_all(args)
        print(json.dumps({"status": "complete", "out_root": args.out_root}, ensure_ascii=False), flush=True)
    except Exception as exc:
        update_state(Path(args.out_root) / STATE_NAME, status="failed", error=str(exc), updated_at=rn.now_iso())
        raise


if __name__ == "__main__":
    main()
