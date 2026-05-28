from __future__ import annotations

from copy import deepcopy
from typing import Any


BASE: dict[str, Any] = {
    "seed": 1,
    "epochs": 24,
    "batch_size": 32,
    "lr": 6.25e-5,
    "weight_decay": 0.03,
    "dropout": 0.10,
    "input_mode": "raw",
    "head_type": "baseline",
    "multiscale": "none",
    "use_positional_embedding": True,
    "use_snr_head": True,
    "use_ordinal_head": False,
    "snr_weight": 0.05,
    "cls_weight": 1.0,
    "denoise_weight": 0.0,
    "level_weight": 0.0,
    "ordinal_weight": 0.0,
    "loss_type": "ce",
    "label_smoothing": 0.0,
    "focal_gamma": 1.5,
    "rdrop_alpha": 0.0,
    "sam_rho": 0.0,
    "uncertainty": False,
    "uncertainty_start_epoch": 6,
    "level_target_mode": "noisy_rr",
    "denoise_gate": "none",
    "description": "baseline clone: CLS+pos+D1 warm-start+SNR head",
}


def recipe(name: str, group: str, **updates: Any) -> dict[str, Any]:
    out = deepcopy(BASE)
    out.update(updates)
    out["name"] = name
    out["group"] = group
    return out


RECIPES: dict[str, list[dict[str, Any]]] = {
    "loss_conflict": [
        recipe("lc_ce_only", "loss_conflict", use_snr_head=False, snr_weight=0.0, description="CE only"),
        recipe("lc_ce_level", "loss_conflict", level_weight=1.0, description="CE + noisy-RR level"),
        recipe("lc_ce_denoise", "loss_conflict", denoise_weight=40.0, description="CE + denoise"),
        recipe(
            "lc_fixed_multitask",
            "loss_conflict",
            denoise_weight=40.0,
            level_weight=1.0,
            description="CE + denoise + level with fixed weights",
        ),
        recipe(
            "lc_uncert_multitask",
            "loss_conflict",
            denoise_weight=40.0,
            level_weight=1.0,
            uncertainty=True,
            uncertainty_start_epoch=6,
            description="CE + denoise + level with learned uncertainty weights after warmup",
        ),
    ],
    "head_reimpl": [
        recipe("hr_baseline_clone", "head_reimpl", description="strong baseline clone"),
        recipe(
            "hr_sqi_interpretable",
            "head_reimpl",
            head_type="sqi_interpretable",
            denoise_weight=20.0,
            level_weight=0.5,
            description="attention pooled tokens plus residual/level/SNR-style stats",
        ),
        recipe(
            "hr_local_quality_v2",
            "head_reimpl",
            head_type="local_quality_v2",
            denoise_weight=20.0,
            level_weight=0.5,
            description="classification sees local residual and predicted level patch summaries",
        ),
        recipe(
            "hr_sqi_local_combo",
            "head_reimpl",
            head_type="sqi_local",
            denoise_weight=20.0,
            level_weight=0.5,
            description="interpretable SQI head plus local-quality v2 features",
        ),
        recipe(
            "hr_multiscale_10_20_40",
            "head_reimpl",
            multiscale="10_20_40",
            description="parallel 10/20/40-ish patch tokens with light fusion",
        ),
    ],
    "target_gate_reimpl": [
        recipe(
            "tg_clean_rr_level",
            "target_gate_reimpl",
            level_weight=1.0,
            level_target_mode="clean_rr",
            description="level target uses clean R-peaks for RR boundaries",
        ),
        recipe(
            "tg_bad_fallback_level",
            "target_gate_reimpl",
            level_weight=1.0,
            level_target_mode="clean_rr_bad_fallback",
            description="bad or invalid RR samples still receive bad-level supervision",
        ),
        recipe(
            "tg_patch_residual_level",
            "target_gate_reimpl",
            level_weight=1.0,
            level_target_mode="patch_residual",
            description="patch residual target keeps supervision on hard bad cases",
        ),
        recipe(
            "tg_abs_topq_gate",
            "target_gate_reimpl",
            denoise_weight=40.0,
            denoise_gate="abs_topq_wavelet",
            description="denoise weighted around large clean high-frequency detail",
        ),
        recipe(
            "tg_qrs_gate",
            "target_gate_reimpl",
            denoise_weight=40.0,
            denoise_gate="qrs_mask",
            description="denoise weighted around clean QRS mask",
        ),
        recipe(
            "tg_level_weight_gate",
            "target_gate_reimpl",
            denoise_weight=40.0,
            level_weight=1.0,
            denoise_gate="level_weight",
            description="denoise weighted where level target says local quality is poor",
        ),
    ],
    "generalization_loss": [
        recipe("gl_label_smooth_005", "generalization_loss", label_smoothing=0.005, description="CE with light label smoothing"),
        recipe("gl_label_smooth_020", "generalization_loss", label_smoothing=0.020, description="CE with stronger label smoothing"),
        recipe("gl_focal_15", "generalization_loss", loss_type="focal", focal_gamma=1.5, description="focal loss gamma 1.5"),
        recipe("gl_focal_20", "generalization_loss", loss_type="focal", focal_gamma=2.0, description="focal loss gamma 2.0"),
        recipe(
            "gl_ordinal_ce",
            "generalization_loss",
            use_ordinal_head=True,
            ordinal_weight=0.05,
            description="CE plus CORAL-style ordinal auxiliary head",
        ),
        recipe("gl_rdrop", "generalization_loss", rdrop_alpha=0.5, description="R-Drop consistency regularization"),
        recipe("gl_sam", "generalization_loss", sam_rho=0.03, description="SAM sharpness-aware optimization"),
    ],
}


def list_recipes(group: str | None = None) -> list[dict[str, Any]]:
    groups = [group] if group else sorted(RECIPES)
    out: list[dict[str, Any]] = []
    for g in groups:
        if g not in RECIPES:
            raise KeyError(f"unknown group={g!r}; choices={sorted(RECIPES)}")
        out.extend(deepcopy(RECIPES[g]))
    return out


def get_recipe(*, group: str, task_id: int | None = None, name: str | None = None) -> dict[str, Any]:
    if name:
        for item in list_recipes(group):
            if item["name"] == name:
                return item
        raise KeyError(f"unknown recipe {name!r} in group {group!r}")
    if task_id is None:
        raise ValueError("task_id or name is required")
    items = list_recipes(group)
    if task_id < 0 or task_id >= len(items):
        raise IndexError(f"task_id={task_id} outside 0..{len(items)-1} for group={group!r}")
    return items[task_id]
