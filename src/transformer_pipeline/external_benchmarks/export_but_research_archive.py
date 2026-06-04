"""Export a GitHub-readable archive for the E3.11f BUT experiments.

The local BUT work produced many outputs, checkpoints, NPZ files, and reports.
This exporter builds a light, source-controlled package that another chat can
read without local state:

* story and timeline markdown
* detailed synthetic rulebook
* compact metric tables and existing report summaries
* representative visual evidence

It intentionally does not copy raw data, checkpoints, NPZ, parquet, or outputs
wholesale.
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[3]
REPORTS_ROOT = ROOT / "reports" / "external_benchmarks"
OUTPUTS_ROOT = ROOT / "outputs" / "external_benchmarks"
ARCHIVE = REPORTS_ROOT / "e311_but_research_archive_2026_06_04"

BLOCKED_SUFFIXES = {".pt", ".pth", ".ckpt", ".npz", ".npy", ".parquet"}
MAX_COPY_BYTES = 2_000_000


EXPERIMENTS: list[dict[str, str]] = [
    {
        "tag": "e311_realdata_2026_06_02",
        "date": "2026-06-02",
        "phase": "external_realdata",
        "hypothesis": "Evaluate current Uformer mainline on real expert-labelled ECG quality data before tuning.",
        "rule": "No synthetic rule change. BUT QDB maps expert consensus class 1/2/3 to good/medium/bad under 10s@125Hz windows; CinC2017 maps '~' to noisy/bad and N/A/O to usable.",
        "recipe": "Zero-shot Uformer, validation-only threshold calibration, frozen feature probes/head-only baselines, runtime/visual preprocessing audits.",
        "result": "BUT zero-shot exposed severe domain shift and bad/medium boundary mismatch; CinC useful for noisy-vs-usable but not the main 3-class target.",
        "value": "Established BUT as the real-data target and showed synthetic PTB labels were not yet expert-usability aligned.",
        "recommendation": "Keep as external benchmark baseline.",
    },
    {
        "tag": "e311_but_protocol_adaptation_2026_06_03",
        "date": "2026-06-03",
        "phase": "protocol",
        "hypothesis": "Check whether 5s/10s protocol, label purity, and crop/ensemble choices cause the BUT boundary mismatch.",
        "rule": "Formal baseline remains 10s P1 center-label; 5s and purity protocols are diagnostic only.",
        "recipe": "Frozen Uformer features and lightweight probes across protocol variants; calibration is validation-only.",
        "result": "10s P1 retained as formal protocol; mismatch is not only a windowing artifact.",
        "value": "Locked protocol so later improvements are comparable.",
        "recommendation": "Use 10s P1 for all headline BUT metrics.",
    },
    {
        "tag": "e311_but_bad_boundary_10s_2026_06_03",
        "date": "2026-06-03",
        "phase": "bad_boundary",
        "hypothesis": "Synthetic bad should resemble BUT bad more through wearable contact/baseline/QRS unreliability than through generic SNR.",
        "rule": "Grid over flatline/contact loss, baseline step/dropout, clipping/low amplitude, spurious QRS bursts, and mixed wearable bad. Key anchor b10_all_bad_wearable mixes multiple bad mechanisms.",
        "recipe": "Quick Uformer training on generated PTB synthetic variants; evaluate PTB sanity and BUT 10s P1.",
        "result": "b10_all_bad_wearable became strict synthetic anchor: BUT acc 0.7735, balanced 0.8045, macro-F1 0.7238, recalls 0.824/0.724/0.866.",
        "value": "First useful generator direction; bad recall improved without relying purely on SNR.",
        "recommendation": "Keep b10 as baseline anchor.",
    },
    {
        "tag": "e311_but_bad_boundary_refine_10s_2026_06_03",
        "date": "2026-06-03",
        "phase": "bad_boundary_refine",
        "hypothesis": "Refine b10 around bad-prior/contact/QRS knobs to recover medium while preserving bad.",
        "rule": "Milder or stronger variants around b10: bad prior mild/strong, contact tuning, baseline-step/dropout adjustments.",
        "recipe": "Quick train, validation-only calibration, compare against b10.",
        "result": "Best refinement r08 raised good but did not beat b10 overall; no clean balanced/macro-F1 improvement.",
        "value": "Showed naive bad-boundary local tuning compresses medium.",
        "recommendation": "Do not expand this family blindly.",
    },
    {
        "tag": "e311_but_bad_boundary_full_confirm_10s_2026_06_03",
        "date": "2026-06-03",
        "phase": "full_confirm",
        "hypothesis": "Full 10+8 training might stabilize quick bad-boundary wins.",
        "rule": "Full confirmation for b10 and r08-style variants.",
        "recipe": "Stage1 denoiser 10 epochs + Stage2 head 8 epochs.",
        "result": "Full b10 had acc about 0.774 but balanced dropped to 0.721 and bad recall fell to 0.582; r08 bad collapsed further.",
        "value": "Revealed that full training can strengthen the wrong synthetic boundary.",
        "recommendation": "Use quick/probe before full; full only after BUT boundary is right.",
    },
    {
        "tag": "e311_but_boundary_head_adaptation_10s_2026_06_03",
        "date": "2026-06-03",
        "phase": "head_only",
        "hypothesis": "Frozen Uformer representations may contain enough information; BUT boundary may need calibration/head adaptation.",
        "rule": "No new generator. Freeze checkpoints/features and train lightweight BUT train/val heads only.",
        "recipe": "LogReg/SVM/MLP/head-only on frozen features; validation-only threshold/class-bias calibration.",
        "result": "Head-only adaptation could recover high bad recall but medium usually dropped, supporting a boundary/calibration issue.",
        "value": "Shows representation is not useless; labels require expert-boundary alignment.",
        "recommendation": "Keep as supervised adaptation evidence, not strict zero-shot.",
    },
    {
        "tag": "e311_but_generator_medium_bad_grid_10s_2026_06_03",
        "date": "2026-06-03",
        "phase": "medium_bad_generator",
        "hypothesis": "Move medium and bad separately instead of only strengthening bad.",
        "rule": "Medium local detail/P/T/ST unreliability, bad QRS/baseline/contact variants, plus mixed medium/bad pressure.",
        "recipe": "Generator-first quick grid, PTB/BUT evaluation.",
        "result": "Did not beat b10; medium and bad still traded off.",
        "value": "Confirmed medium is the hard boundary.",
        "recommendation": "Use as negative evidence.",
    },
    {
        "tag": "e311_but_generator_medium_mixture_grid_10s_2026_06_03",
        "date": "2026-06-03",
        "phase": "medium_mixture",
        "hypothesis": "Medium may be its own mixed cluster with QRS visible but details unreliable.",
        "rule": "Mixtures around medium QRS-visible families, local P/T/ST instability, and moderate bad pressure.",
        "recipe": "Quick grid with visual review.",
        "result": "mix04/mix05 gave medium clues; one variant improved balanced to about 0.811 but bad or macro did not cleanly beat b10.",
        "value": "Important clue that P/T/ST unreliability helps medium but weakens bad.",
        "recommendation": "Promote medium-as-independent-cluster hypothesis.",
    },
    {
        "tag": "e311_but_generator_morph_sweet_grid_10s_2026_06_03",
        "date": "2026-06-03",
        "phase": "morph_sweet",
        "hypothesis": "Visual morphology, not SNR, should guide the sweet spot.",
        "rule": "Local shape events, inverted pseudo-peaks, short baseline steps/ramps, P/T/ST instability with QRS preserved, dense QRS-confounding pseudo-peaks/contact for bad.",
        "recipe": "Small morphology quick grid.",
        "result": "Found evidence for medium/bad coexistence but no clean b10 improvement.",
        "value": "Shifted search from SNR to morphology/expert usability.",
        "recommendation": "Keep as design transition.",
    },
    {
        "tag": "e311_but_generator_morph_refine_grid_10s_2026_06_03",
        "date": "2026-06-03",
        "phase": "morph_refine",
        "hypothesis": "Refine morphology events around the sweet spot.",
        "rule": "Milder/stronger local morphology events and bad rescue pressure.",
        "recipe": "Quick train and visual review.",
        "result": "Did not beat b10 cleanly; refined events still compress one class.",
        "value": "Shows local morphology alone is insufficient.",
        "recommendation": "Need better class-boundary model.",
    },
    {
        "tag": "e311_but_generator_morph_v2_grid_10s_2026_06_03",
        "date": "2026-06-03",
        "phase": "morph_v2",
        "hypothesis": "Try morphology-only combinations with good mild wearable overlap, QRS-preserved medium, QRS-confounding bad.",
        "rule": "Medium-strong variants, bad-strong variants, softer good overlap.",
        "recipe": "Quick grid.",
        "result": "Medium-strong stole good or weakened bad; bad-strong compressed medium.",
        "value": "Clarified medium and bad are not a simple strength axis.",
        "recommendation": "Stop one-axis morphology scaling.",
    },
    {
        "tag": "e311_but_generator_morph_v3_grid_10s_2026_06_03",
        "date": "2026-06-03",
        "phase": "morph_v3",
        "hypothesis": "Small visual-evidence pass after v2 failure.",
        "rule": "b10 micro morphology, s02 bad with medium floor, mix05 medium with bad floor, no-flatline pseudo-QRS bad, medium guard with softer bad pressure.",
        "recipe": "Quick train.",
        "result": "No clean improvement; confirmed that increasing single specs is not enough.",
        "value": "Final negative evidence before larger analysis.",
        "recommendation": "Use data analysis, not blind expansion.",
    },
    {
        "tag": "e311_but_large_rule_grid_10s_2026_06_03",
        "date": "2026-06-03",
        "phase": "large_rule_grid",
        "hypothesis": "Large factorial rule grid can reveal which families are close to BUT.",
        "rule": "74 planned specs across medium QRS-visible families, bad QRS-unreliable families, good-not-pristine, contact/wearable, flatline/SNR controls.",
        "recipe": "Stopped at 42 quick rows; validation-only calibration; no full confirm until analysis.",
        "result": "Provided enough signal for morphology distance analysis; did not itself settle a winner.",
        "value": "Produced broad evidence linking rules to metrics.",
        "recommendation": "Use as seed pool for analysis.",
    },
    {
        "tag": "e311_but_morphology_analysis_10s_2026_06_03",
        "date": "2026-06-03",
        "phase": "morphology_analysis",
        "hypothesis": "Quantify BUT-vs-synthetic morphology distance to avoid visual-only guessing.",
        "rule": "No new generator. Extract shared features: QRS detectability, peak density, non-QRS detail energy, derivative/local baseline events, flat/contact/HF/baseline, RMS/low-amplitude.",
        "recipe": "Class-wise KS/Wasserstein-like distances and metric joins over completed variants.",
        "result": "Distance proxies help but are not sufficient; BUT medium is not merely middle SNR.",
        "value": "Produced domain gap tables, feature summaries, and distance-vs-metric plots.",
        "recommendation": "Use features to guide hypotheses, not to rank alone.",
    },
    {
        "tag": "e311_but_medium_cluster_analysis_10s_2026_06_03",
        "date": "2026-06-04",
        "phase": "medium_cluster_analysis",
        "hypothesis": "BUT medium may be an independent cluster rather than midpoint between good and bad.",
        "rule": "No generator. Analyze BUT class centroids, PCA, pairwise robust deltas, one-vs-rest feature importance, synthetic medium alignment.",
        "recipe": "Feature geometry and visual profiles.",
        "result": "Medium behaves like QRS-usable/detail-unreliable, not as linear interpolation between good and bad.",
        "value": "Strongly changed the generator direction.",
        "recommendation": "Design medium as independent expert-usability class.",
    },
    {
        "tag": "e311_but_morphology_guided_grid_10s_2026_06_03",
        "date": "2026-06-03",
        "phase": "morphology_guided_grid",
        "hypothesis": "Use morphology analysis and anchor rows to generate a guided grid.",
        "rule": "Medium rescue, bad rescue, coexistence anchors, negative controls, feature audit before training.",
        "recipe": "Smoke/quick/follow-up; feature_target_score only tie-breaker.",
        "result": "h_bad_rescue_05 became the strongest recent anchor: acc 0.8229, balanced 0.8177, macro-F1 0.7454, recalls 0.887/0.773/0.793.",
        "value": "Best strict synthetic evidence so far.",
        "recommendation": "Use h_bad_rescue_05 as best current rule anchor.",
    },
    {
        "tag": "e311_but_fatal_or_logic_grid_10s_2026_06_04",
        "date": "2026-06-04",
        "phase": "fatal_or_logic",
        "hypothesis": "good is AND(all critical dimensions good), bad is OR(any fatal dimension fails hard), medium is independent.",
        "rule": "Fatal-OR variants over qrs_confuse/contact/burst/morph/flat/lowamp with medium guard.",
        "recipe": "Quick and full confirm.",
        "result": "Best full fatal_or_qrs_confuse_06 had acc 0.8064, macro-F1 0.7319, recalls 0.842/0.796/0.603; bad weakened.",
        "value": "OR logic helps conceptually but shared spec still averages bad subtypes.",
        "recommendation": "Move to sample-level OR subtypes.",
    },
    {
        "tag": "e311_but_sample_or_subtype_grid_10s_2026_06_04",
        "date": "2026-06-04",
        "phase": "sample_or_subtype",
        "hypothesis": "Bad should be sample-level OR over fatal subtypes, not every bad sample receiving every artifact.",
        "rule": "Each bad sample gets primary and optional secondary subtype: qrs_confound/contact_flat/motion_burst/morph_break/clipping_lowamp/baseline_jump. Medium remains QRS-usable/detail-unreliable.",
        "recipe": "Smoke + 28 quick + top 6 full; subtype audits and galleries.",
        "result": "Quick best sample_or_13 macro-F1 0.7385; full best sample_or_03 macro-F1 0.6942. It did not beat h_bad_rescue_05.",
        "value": "Shows sample-level OR is not enough; full training reinforces wrong synthetic boundary.",
        "recommendation": "Next: diagnostic-usability bad with visible QRS but global unusability.",
    },
    {
        "tag": "e311_but_diagnostic_usability_grid_10s_2026_06_04",
        "date": "2026-06-04",
        "phase": "diagnostic_usability_latest",
        "hypothesis": "BUT labels are diagnostic usability boundaries: good all-critical usable, medium QRS-usable/detail-unreliable, bad can keep visible QRS while globally unanalyzable.",
        "rule": "Adds visible_qrs_global_noise subtype; defaults to quick/probe only with no full confirmation until quick beats anchors.",
        "recipe": "Smoke + 24 quick, top_full=0, validation-only calibration.",
        "result": "In progress at export time if not complete; partial rows included when available.",
        "value": "Current latest hypothesis after sample-level OR failure.",
        "recommendation": "Evaluate quick rows before any full training.",
    },
]


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_file(src: Path, dst: Path, *, max_bytes: int = MAX_COPY_BYTES) -> bool:
    if not src.exists() or not src.is_file():
        return False
    if src.suffix.lower() in BLOCKED_SUFFIXES:
        return False
    if src.stat().st_size > max_bytes:
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def copy_matching(src_dir: Path, dst_dir: Path, suffixes: Iterable[str], *, max_bytes: int = MAX_COPY_BYTES) -> list[str]:
    copied: list[str] = []
    if not src_dir.exists():
        return copied
    suffix_set = {s.lower() for s in suffixes}
    for src in sorted(src_dir.rglob("*")):
        if not src.is_file() or src.suffix.lower() not in suffix_set:
            continue
        rel = src.relative_to(src_dir)
        if copy_file(src, dst_dir / rel, max_bytes=max_bytes):
            copied.append(str(dst_dir / rel))
    return copied


def read_text_if_exists(path: Path, limit: int = 20_000) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[:limit]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def json_load(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def jsonl_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def report_metrics_from_row(row: dict[str, Any]) -> dict[str, Any]:
    rep = (row.get("but_10s_eval") or {}).get("but_10s_test_report") or {}
    rec = rep.get("recall_good_medium_bad") or [None, None, None]
    ptb = row.get("ptb_test_report") or {}
    ptb_rec = ptb.get("recall_good_medium_bad") or [None, None, None]
    den = row.get("ptb_denoise_metrics") or {}
    spec = row.get("spec") or {}
    return {
        "mode": row.get("mode"),
        "variant_id": spec.get("id"),
        "family": spec.get("family"),
        "class_weight": row.get("class_weight"),
        "but_acc": rep.get("acc"),
        "but_balanced_acc": rep.get("balanced_acc"),
        "but_macro_f1": rep.get("macro_f1"),
        "but_recall_good": rec[0],
        "but_recall_medium": rec[1],
        "but_recall_bad": rec[2],
        "min_medium_bad": min([x for x in [rec[1], rec[2]] if x is not None], default=None),
        "ptb_acc": ptb.get("acc"),
        "ptb_recall_bad": ptb_rec[2],
        "denoise_score": den.get("denoise_score"),
        "note": row.get("note"),
        "run_dir": row.get("run_dir"),
    }


def summarize_top_rows(tag: str, summary_name_candidates: list[str]) -> list[dict[str, Any]]:
    out_dir = OUTPUTS_ROOT / tag
    rows: list[dict[str, Any]] = []
    for name in summary_name_candidates:
        rows = jsonl_rows(out_dir / name)
        if rows:
            break
    metric_rows = [report_metrics_from_row(r) for r in rows if r.get("but_10s_eval")]
    metric_rows = [r for r in metric_rows if r.get("but_macro_f1") is not None]
    metric_rows.sort(key=lambda r: float(r.get("but_macro_f1") or 0), reverse=True)
    return metric_rows[:30]


def copy_selected_figures() -> list[dict[str, str]]:
    figure_specs = [
        ("but_processed_good", OUTPUTS_ROOT / "e311_realdata_2026_06_02/processed/butqdb/visuals/processed_good_gallery.png"),
        ("but_processed_medium", OUTPUTS_ROOT / "e311_realdata_2026_06_02/processed/butqdb/visuals/processed_medium_gallery.png"),
        ("but_processed_bad", OUTPUTS_ROOT / "e311_realdata_2026_06_02/processed/butqdb/visuals/processed_bad_gallery.png"),
        ("but_rms_distribution", OUTPUTS_ROOT / "e311_realdata_2026_06_02/processed/butqdb/visuals/raw_rms_distribution.png"),
        ("but_class_profile", OUTPUTS_ROOT / "e311_but_morphology_analysis_10s_2026_06_03/visuals/but_class_profiles.png"),
        ("but_feature_profile", OUTPUTS_ROOT / "e311_but_morphology_analysis_10s_2026_06_03/visuals/but_feature_profiles_by_class.png"),
        ("distance_vs_metric", OUTPUTS_ROOT / "e311_but_morphology_analysis_10s_2026_06_03/visuals/distance_vs_metrics.png"),
        ("synthetic_nearest_farthest", OUTPUTS_ROOT / "e311_but_morphology_analysis_10s_2026_06_03/visuals/synthetic_nearest_farthest.png"),
        ("medium_key_feature_boxplots", OUTPUTS_ROOT / "e311_but_medium_cluster_analysis_10s_2026_06_03/visuals/but_key_feature_boxplots.png"),
        ("medium_pca", OUTPUTS_ROOT / "e311_but_medium_cluster_analysis_10s_2026_06_03/visuals/but_pca_by_class.png"),
        ("current_grid_medium_bad_tradeoff", OUTPUTS_ROOT / "e311_but_medium_cluster_analysis_10s_2026_06_03/visuals/current_grid_medium_bad_tradeoff.png"),
        ("medium_vs_rest", OUTPUTS_ROOT / "e311_but_medium_cluster_analysis_10s_2026_06_03/visuals/medium_vs_rest_importance.png"),
        ("pairwise_feature_boundaries", OUTPUTS_ROOT / "e311_but_medium_cluster_analysis_10s_2026_06_03/visuals/pairwise_feature_boundaries.png"),
        ("synthetic_medium_alignment", OUTPUTS_ROOT / "e311_but_medium_cluster_analysis_10s_2026_06_03/visuals/synthetic_medium_alignment.png"),
        ("sample_or_medium_confused", OUTPUTS_ROOT / "e311_but_sample_or_subtype_grid_10s_2026_06_04/runs/full/sample_or_03_balanced_m04_s15/but_10s_eval/visuals/medium_confused.png"),
        ("sample_or_bad_missed", OUTPUTS_ROOT / "e311_but_sample_or_subtype_grid_10s_2026_06_04/runs/full/sample_or_03_balanced_m04_s15/but_10s_eval/visuals/bad_missed.png"),
        ("sample_or_correct_bad", OUTPUTS_ROOT / "e311_but_sample_or_subtype_grid_10s_2026_06_04/runs/full/sample_or_03_balanced_m04_s15/but_10s_eval/visuals/correct_bad.png"),
        ("sample_or_good_false_bad", OUTPUTS_ROOT / "e311_but_sample_or_subtype_grid_10s_2026_06_04/runs/full/sample_or_03_balanced_m04_s15/but_10s_eval/visuals/good_false_bad.png"),
        ("sample_or_subtype_gallery", OUTPUTS_ROOT / "e311_but_sample_or_subtype_grid_10s_2026_06_04/synthetic_variants/sample_or_03_balanced_m04_s15/visuals/bad_subtype_gallery.png"),
        ("diagnostic_subtype_gallery", OUTPUTS_ROOT / "e311_but_diagnostic_usability_grid_10s_2026_06_04/synthetic_variants/diagnostic_usability_05_good_strict_med_local_bad_visible/visuals/bad_subtype_gallery.png"),
    ]
    copied: list[dict[str, str]] = []
    for name, src in figure_specs:
        if not src.exists():
            continue
        ext = src.suffix.lower()
        dst = ARCHIVE / "figures" / f"{name}{ext}"
        if copy_file(src, dst, max_bytes=5_000_000):
            copied.append({"figure": name, "path": str(dst.relative_to(ARCHIVE)).replace("\\", "/"), "source": str(src.relative_to(ROOT)).replace("\\", "/")})
    return copied


def copy_reports_and_results() -> None:
    for exp in EXPERIMENTS:
        tag = exp["tag"]
        src_report = REPORTS_ROOT / tag
        dst_report = ARCHIVE / "results" / "per_experiment" / tag
        copy_matching(src_report, dst_report, {".md", ".csv", ".json"}, max_bytes=2_000_000)

    analysis_sources = [
        (
            OUTPUTS_ROOT / "e311_but_morphology_analysis_10s_2026_06_03",
            ARCHIVE / "analysis" / "morphology_analysis",
            [
                "analysis_audit.json",
                "but_morph_feature_summary.csv",
                "distance_metric_correlations.json",
                "grid_metric_distance_join.csv",
                "morph_distance_by_variant.csv",
                "morph_distance_by_feature.csv",
                "synthetic_morph_feature_summary.csv",
                "morphology_analysis_summary.md",
            ],
        ),
        (
            OUTPUTS_ROOT / "e311_but_medium_cluster_analysis_10s_2026_06_03",
            ARCHIVE / "analysis" / "medium_cluster_analysis",
            [
                "but_class_centroid_geometry.csv",
                "but_pca_projection.csv",
                "current_guided_grid_medium_boundary.csv",
                "data_audit.json",
                "feature_pairwise_robust_deltas.csv",
                "medium_cluster_analysis_report.md",
                "medium_cluster_geometry.json",
                "medium_one_vs_rest_feature_importance.csv",
                "model_separation_report.json",
                "synthetic_medium_alignment.csv",
            ],
        ),
    ]
    for src_dir, dst_dir, names in analysis_sources:
        for name in names:
            copy_file(src_dir / name, dst_dir / name, max_bytes=2_000_000)

    summary_map = {
        "e311_but_morphology_guided_grid_10s_2026_06_03": ["morphology_guided_summary.jsonl"],
        "e311_but_fatal_or_logic_grid_10s_2026_06_04": ["fatal_or_logic_summary.jsonl"],
        "e311_but_sample_or_subtype_grid_10s_2026_06_04": ["sample_or_subtype_summary.jsonl"],
        "e311_but_diagnostic_usability_grid_10s_2026_06_04": ["diagnostic_usability_summary.jsonl", "sample_or_subtype_summary.jsonl"],
    }
    all_top_rows: list[dict[str, Any]] = []
    for tag, candidates in summary_map.items():
        for row in summarize_top_rows(tag, candidates):
            row["experiment_tag"] = tag
            all_top_rows.append(row)
    write_csv(ARCHIVE / "results" / "top_variant_metrics.csv", all_top_rows)


def write_registry_and_metadata(figures: list[dict[str, str]]) -> None:
    write_csv(ARCHIVE / "results" / "experiment_registry.csv", EXPERIMENTS)
    (ARCHIVE / "figures" / "figure_manifest.json").write_text(json.dumps({"figures": figures}, indent=2), encoding="utf-8")
    diagnostic_state = json_load(OUTPUTS_ROOT / "e311_but_diagnostic_usability_grid_10s_2026_06_04/diagnostic_usability_state.json")
    metadata = {
        "archive_tag": ARCHIVE.name,
        "generated_from": {
            "reports_root": str(REPORTS_ROOT.relative_to(ROOT)).replace("\\", "/"),
            "outputs_root": str(OUTPUTS_ROOT.relative_to(ROOT)).replace("\\", "/"),
        },
        "formal_protocol": "BUT QDB 10s P1, class 1/2/3 -> good/medium/bad, validation-only threshold calibration.",
        "blocked_file_types": sorted(BLOCKED_SUFFIXES),
        "diagnostic_usability_state_at_export": diagnostic_state,
        "headline_anchors": {
            "b10_all_bad_wearable": {"acc": 0.7735, "balanced_acc": 0.8045, "macro_f1": 0.7238, "recall_good_medium_bad": [0.824, 0.724, 0.866]},
            "h_bad_rescue_05": {"acc": 0.8229, "balanced_acc": 0.8177, "macro_f1": 0.7454, "recall_good_medium_bad": [0.887, 0.773, 0.793]},
            "sample_or_13_quick": {"acc": 0.7322, "balanced_acc": 0.7700, "macro_f1": 0.7385, "recall_good_medium_bad": [0.935, 0.558, 0.818]},
            "sample_or_03_full": {"acc": 0.7133, "balanced_acc": 0.7611, "macro_f1": 0.6942, "recall_good_medium_bad": [0.876, 0.568, 0.839]},
        },
    }
    (ARCHIVE / "archive_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def write_readme() -> None:
    text = """# E3.11f BUT Synthetic-Rule Research Archive

This folder is the GitHub-readable handoff package for the E3.11f BUT QDB work. It is designed so another chat can understand the full research thread without local checkpoints or raw data.

## Why We Did This

The mainline E3.11f model was built on PTB-derived synthetic SQI labels. On internal PTB synthetic tests the Uformer representation and denoise-before-classifier mechanism performed strongly, but that alone does not prove the signal-quality labels match real expert judgement. BUT QDB is the key external check because it provides expert consensus ECG quality classes:

- BUT class 1 -> good: P/T/QRS visible and reliable enough for detailed analysis.
- BUT class 2 -> medium: QRS usable, but finer intervals/details become unreliable.
- BUT class 3 -> bad: signal unsuitable for further analysis.

The formal protocol in this archive is **BUT 10s P1**: 10-second windows, class 1/2/3 mapped directly to good/medium/bad, validation-only calibration, and test used only for reporting.

## Current Headline Conclusion

The old synthetic PTB rule was too close to a noise-strength/SNR axis. BUT behaves more like a **diagnostic-usability boundary**:

- good is an AND condition: all critical dimensions must be usable.
- medium is not a midpoint; it is a QRS-usable/detail-unreliable cluster.
- bad is an OR condition over fatal usability failures, and some bad examples can still have visible QRS while the whole strip is not analyzable.

The best strict synthetic anchor so far is `h_bad_rescue_05` from the morphology-guided grid: acc 0.8229, balanced 0.8177, macro-F1 0.7454, recalls 0.887/0.773/0.793. Later sample-level OR experiments were mechanism-informative but did not beat that anchor.

## How To Read This Folder

- `timeline.md` explains the order of experiments and why each one happened.
- `rulebook/` contains detailed synthetic data rules and failure-mode interpretations.
- `results/experiment_registry.csv` is the compact index of all experiment families.
- `results/top_variant_metrics.csv` is a metric table for the best/most relevant grid rows.
- `analysis/` contains BUT morphology statistics, medium-cluster analysis, and distance-vs-metric material.
- `figures/` contains representative visual evidence that motivated the rule changes.
- `next_hypotheses.md` summarizes what to try next.

## What Is Intentionally Not Included

No checkpoints, NPZ arrays, raw signals, parquet files, or large `outputs/` trees are committed here. The local machine keeps the full artifacts under `outputs/external_benchmarks/`; this archive keeps the analysis surface needed for GitHub review and another chat.
"""
    (ARCHIVE / "README.md").write_text(text, encoding="utf-8")


def write_timeline() -> None:
    lines = [
        "# BUT Experiment Timeline",
        "",
        "| Date | Phase | Experiment | Why it happened | Outcome |",
        "| --- | --- | --- | --- | --- |",
    ]
    for exp in EXPERIMENTS:
        lines.append(f"| {exp['date']} | {exp['phase']} | `{exp['tag']}` | {exp['hypothesis']} | {exp['result']} |")
    (ARCHIVE / "timeline.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_rulebook() -> None:
    rulebook = ARCHIVE / "rulebook"
    rulebook.mkdir(parents=True, exist_ok=True)
    overview = """# Synthetic Rulebook

## Base Setup Shared By Generator Experiments

- Source clean ECG: PTB-derived synthetic artifact used by E3.11f, 10s at 125Hz, split 10935/2184/2202.
- Output labels: good/medium/bad three-class SQI aligned to BUT class 1/2/3 semantics.
- Model recipe unless stated otherwise: Uformer1D residual denoiser + detached full-token classifier/SQI head.
- Formal external eval: BUT QDB 10s P1, validation-only threshold or bias calibration, test used only once for reporting.
- PTB sanity checks: internal PTB accuracy, bad recall, and denoise score are retained so external gains do not destroy the mainline.

## Rule Evolution In One Sentence

We moved from "bad means more noise" to "bad means any fatal diagnostic-usability failure", and from "medium is halfway between good and bad" to "medium is QRS usable but detail unreliable".
"""
    (rulebook / "README.md").write_text(overview, encoding="utf-8")

    detailed = """# Detailed Generator Rules

## 1. b10 / Bad-Boundary Wearable Rules

Purpose: make synthetic bad look like real wearable/BUT bad instead of simply low SNR.

Bad artifacts:
- contact loss and flat segments;
- baseline step/dropout;
- clipping and low-amplitude compression;
- spurious QRS-like bursts;
- mixed wearable motion/HF/baseline noise.

Medium rule at this stage was still too much like intermediate severity. b10 worked because it gave bad a realistic wearable flavor, but later analysis showed medium was still under-modeled.

## 2. Medium/Bad and Medium-Mixture Rules

Purpose: recover BUT class 2 by separating "detail unreliable" from "unusable".

Medium artifacts:
- QRS is mostly preserved;
- P/T/ST and non-QRS regions receive local unreliability;
- short baseline steps/ramps;
- inverted or pseudo local events;
- small local contact disturbances.

Bad artifacts:
- QRS-confounding pseudo-peaks;
- stronger contact or baseline failures;
- motion/HF bursts.

Finding: P/T/ST/local-detail events help medium, but if bad pressure is too weak, bad recall collapses; if bad pressure is too strong, medium is swallowed.

## 3. Morph Sweet / V2 / V3 Rules

Purpose: tune by visual morphology rather than SNR.

Rule axes:
- medium QRS preserve level;
- P/T/ST instability strength;
- local baseline steps/ramps;
- pseudo-peak density;
- contact-loss severity;
- good mild wearable overlap.

Finding: local morphology alone is not enough. Medium and bad cannot be described by one monotonic artifact strength axis.

## 4. Morphology-Guided Grid

Purpose: use BUT feature profiles and previous grid rows as priors.

Rule families:
- medium rescue: QRS visible, local detail unreliable;
- bad rescue: QRS-confounding/contact events, not pure flatline;
- coexistence anchors: interpolate b10, s02, mix05, good-not-pristine;
- negative controls: SNR/flatline-heavy rules.

Best anchor: h_bad_rescue_05 with acc 0.8229, balanced 0.8177, macro-F1 0.7454, recalls 0.887/0.773/0.793.

## 5. Fatal-OR Logic

Purpose: encode the user insight that one fatal dimension failing can make a strip bad.

Boundary:
- good = AND(all critical dimensions good);
- bad = OR(any fatal dimension fails hard);
- medium = independent QRS-usable/detail-unreliable cluster.

Problem: applying averaged fatal pressure to all bad examples still creates an unrealistic single bad distribution.

## 6. Sample-Level Fatal Subtype OR

Purpose: make each bad sample trigger one or two fatal subtypes instead of every bad artifact at once.

Bad subtypes:
- qrs_confound: attenuate/confuse QRS and insert QRS-like distractors;
- contact_flat: flat/contact-loss spans;
- motion_burst: burst/HF/wander motion noise;
- morph_break: local morphology damage in QRS/TST/critical regions;
- clipping_lowamp: clipping and amplitude compression;
- baseline_jump: short steps/ramps and baseline discontinuities.

Finding: this is mechanistically plausible but did not beat h_bad_rescue_05. Quick results were better than full; full training improved PTB/denoise but strengthened the wrong synthetic boundary.

## 7. Diagnostic-Usability Latest Hypothesis

Purpose: match BUT visual evidence that some bad examples retain visible QRS but remain globally unreliable.

New bad subtype:
- visible_qrs_global_noise: preserves much of QRS while adding global HF/burst/drift, repeated distractors, and unreliable background.

Boundary:
- good is strict: all critical dimensions clean enough;
- medium is QRS usable with local/detail unreliability;
- bad can be visible-QRS global unusability, contact failure, or QRS-confounding fatal failure.

This runner is intentionally quick/probe only. Full training is disabled by default until quick results beat anchors.
"""
    (rulebook / "detailed_generator_rules.md").write_text(detailed, encoding="utf-8")

    rows = []
    for exp in EXPERIMENTS:
        rows.append(
            {
                "tag": exp["tag"],
                "phase": exp["phase"],
                "hypothesis": exp["hypothesis"],
                "data_rule": exp["rule"],
                "training_recipe": exp["recipe"],
                "result": exp["result"],
                "value": exp["value"],
                "recommendation": exp["recommendation"],
            }
        )
    write_csv(rulebook / "experiment_rule_metadata.csv", rows)


def write_analysis_summary() -> None:
    text = """# BUT Statistics And Visual Analysis Summary

## Morphology Analysis

The morphology-analysis pass extracted comparable features from BUT and synthetic variants:

- QRS reliability: peak prominence, QRS-like density, missing/spurious peak proxies, slope/width.
- Detail reliability: non-QRS energy, local derivative anomalies, P/T/ST instability.
- Wearable/contact signals: flatline/contact spans, clipping, low amplitude, baseline wander, HF bursts.

The distance tables in `analysis/morphology_analysis/` show that feature similarity is useful but not sufficient. Better feature-target scores do not automatically produce better BUT metrics, especially when medium/bad boundaries are wrong.

## Medium Cluster Analysis

The medium analysis showed that BUT class 2 is not simply between class 1 and class 3. It behaves like its own cluster: QRS often remains visible, while detail reliability and local morphology become questionable. This finding is why later generators treat medium as independent instead of a milder bad.

## Visual Evidence

The most important visual examples are:

- `figures/sample_or_medium_confused.png`: many class-2 BUT windows have visible QRS and are predicted too good.
- `figures/sample_or_bad_missed.png`: some class-3 BUT windows keep visible QRS but are globally unreliable.
- `figures/sample_or_subtype_gallery.png`: synthetic bad subtype OR examples; visually plausible but still not enough.

## Current Interpretation

The problem is not "add more noise". The problem is aligning PTB synthetic labels to an expert usability boundary. The next promising direction is diagnostic-usability generation: visible-QRS global bad, independent medium, and strict good.
"""
    (ARCHIVE / "analysis" / "analysis_summary.md").write_text(text, encoding="utf-8")


def write_next_hypotheses() -> None:
    text = """# Next Hypotheses

1. **Diagnostic-usability bad, not all-destroyed bad.** BUT class 3 can keep visible QRS while the strip is globally unreliable. Continue testing `visible_qrs_global_noise` and avoid treating all bad examples as flat/QRS-erased.

2. **Medium is an independent cluster.** Keep QRS visible and introduce local detail unreliability. Do not tune medium by simply lowering SNR or weakening bad.

3. **Full training only after quick boundary works.** Prior full confirmations improved PTB and denoise but pulled the classifier toward the synthetic boundary, hurting BUT. Use quick/probe as a boundary discovery stage.

4. **Feature proxy is diagnostic, not decisive.** Morphology distance should explain hypotheses and veto obvious mismatches, but feature_target_score alone should not select models.

5. **Report zero-shot and supervised adaptation separately.** Head-only BUT adaptation can show representation transfer, but it is not strict synthetic zero-shot evidence.
"""
    (ARCHIVE / "next_hypotheses.md").write_text(text, encoding="utf-8")


def export() -> None:
    ensure_clean_dir(ARCHIVE)
    (ARCHIVE / "results").mkdir(parents=True, exist_ok=True)
    (ARCHIVE / "analysis").mkdir(parents=True, exist_ok=True)
    (ARCHIVE / "figures").mkdir(parents=True, exist_ok=True)

    copy_reports_and_results()
    figures = copy_selected_figures()
    write_registry_and_metadata(figures)
    write_readme()
    write_timeline()
    write_rulebook()
    write_analysis_summary()
    write_next_hypotheses()

    # Guardrail manifest: useful for review and easy to grep in CI/manual checks.
    forbidden = []
    for p in ARCHIVE.rglob("*"):
        if p.is_file() and p.suffix.lower() in BLOCKED_SUFFIXES:
            forbidden.append(str(p.relative_to(ARCHIVE)).replace("\\", "/"))
    (ARCHIVE / "archive_guardrails.json").write_text(
        json.dumps({"forbidden_files_found": forbidden, "file_count": sum(1 for p in ARCHIVE.rglob("*") if p.is_file())}, indent=2),
        encoding="utf-8",
    )
    if forbidden:
        raise RuntimeError(f"Archive contains forbidden file types: {forbidden}")


def main() -> None:
    export()
    print(json.dumps({"status": "ok", "archive": str(ARCHIVE)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
