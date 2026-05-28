from __future__ import annotations

from copy import deepcopy
from typing import Any


BASE: dict[str, Any] = {
    "seed": 1,
    "epochs": 24,
    "batch_size": 32,
    "lr": 6.25e-5,
    "lr_eta_min": 4e-6,
    "weight_decay": 0.03,
    "dropout": 0.10,
    "input_mode": "raw",
    "head_type": "baseline",
    "multiscale": "none",
    "use_positional_embedding": True,
    "use_snr_head": True,
    "use_ordinal_head": False,
    "cls_hidden": 0,
    "snr_weight": 0.05,
    "cls_weight": 10.0,
    "class_weight_good": 1.0,
    "class_weight_medium": 1.0,
    "class_weight_bad": 1.0,
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
    "local_weight": 0.05,
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

    "focused_tuning": [
        recipe(
            "ft_local_l0025_nodense",
            "focused_tuning",
            head_type="local_quality_v2",
            denoise_weight=0.0,
            level_weight=0.0,
            local_weight=0.0025,
            description="local-quality head with very light synthetic local supervision",
        ),
        recipe(
            "ft_local_l005_nodense",
            "focused_tuning",
            head_type="local_quality_v2",
            denoise_weight=0.0,
            level_weight=0.0,
            local_weight=0.005,
            description="local-quality head with light synthetic local supervision",
        ),
        recipe(
            "ft_local_l0075_nodense",
            "focused_tuning",
            head_type="local_quality_v2",
            denoise_weight=0.0,
            level_weight=0.0,
            local_weight=0.0075,
            description="local-quality head aligned with the best mainline local-mask scale",
        ),
        recipe(
            "ft_cleanrr_l025",
            "focused_tuning",
            level_weight=0.25,
            level_target_mode="clean_rr",
            description="baseline head with low-weight clean-RR level supervision",
        ),
        recipe(
            "ft_badfallback_l025",
            "focused_tuning",
            level_weight=0.25,
            level_target_mode="clean_rr_bad_fallback",
            description="baseline head with low-weight clean-RR bad fallback supervision",
        ),
        recipe(
            "ft_local_cleanrr_l005_l025",
            "focused_tuning",
            head_type="local_quality_v2",
            denoise_weight=0.0,
            level_weight=0.25,
            local_weight=0.005,
            level_target_mode="clean_rr",
            description="light local-quality head plus low-weight clean-RR level supervision",
        ),
    ],
    "sqi_head_tuning": [
        recipe("sqi_input_cls", "sqi_head_tuning", head_type="sqi_input", description="CLS plus projected deterministic input SQI stats"),
        recipe("sqi_input_attn", "sqi_head_tuning", head_type="sqi_input_attn", description="encoder attention pool plus CLS plus projected input SQI stats"),
        recipe("sqi_input_cls_drop005", "sqi_head_tuning", head_type="sqi_input", dropout=0.05, description="input SQI stats with lower head/backbone dropout"),
        recipe("sqi_input_cls_snr010", "sqi_head_tuning", head_type="sqi_input", snr_weight=0.10, description="input SQI stats with stronger SNR auxiliary"),
        recipe("sqi_pred_nodense", "sqi_head_tuning", head_type="sqi_pred", description="CLS plus input stats plus trainable residual/level stats, no dense loss"),
        recipe("sqi_pred_den5", "sqi_head_tuning", head_type="sqi_pred", denoise_weight=5.0, description="predicted residual stats with light denoise supervision"),
        recipe("sqi_pred_den10_lvl025", "sqi_head_tuning", head_type="sqi_pred", denoise_weight=10.0, level_weight=0.25, description="predicted residual/level stats with light dense supervision"),
        recipe("sqi_pred_detach_den10_lvl025", "sqi_head_tuning", head_type="sqi_pred_detach", denoise_weight=10.0, level_weight=0.25, description="detached residual/level stats to stop classifier shortcut gradients"),
        recipe("sqi_pred_detach_cleanrr_l025", "sqi_head_tuning", head_type="sqi_pred_detach", denoise_weight=10.0, level_weight=0.25, level_target_mode="clean_rr", description="detached residual stats plus clean-RR level target"),
        recipe("sqi_mil_nodense", "sqi_head_tuning", head_type="sqi_mil", description="input/predicted SQI stats plus top-k patch MIL stats, no dense loss"),
        recipe("sqi_mil_den5", "sqi_head_tuning", head_type="sqi_mil", denoise_weight=5.0, description="MIL SQI stats with light denoise supervision"),
        recipe("sqi_mil_detach_den10_lvl025", "sqi_head_tuning", head_type="sqi_mil_detach", denoise_weight=10.0, level_weight=0.25, description="detached MIL SQI stats with light dense supervision"),
        recipe("sqi_mil_detach_cleanrr_l025", "sqi_head_tuning", head_type="sqi_mil_detach", denoise_weight=10.0, level_weight=0.25, level_target_mode="clean_rr", description="detached MIL SQI stats plus clean-RR level target"),
        recipe("sqi_mil_detach_cleanrr_l05", "sqi_head_tuning", head_type="sqi_mil_detach", denoise_weight=10.0, level_weight=0.50, level_target_mode="clean_rr", description="detached MIL SQI stats plus stronger clean-RR level target"),
        recipe("sqi_mil_detach_patchres_l025", "sqi_head_tuning", head_type="sqi_mil_detach", level_weight=0.25, level_target_mode="patch_residual", description="detached MIL SQI stats plus patch-residual level target"),
        recipe("sqi_mil_detach_ls005", "sqi_head_tuning", head_type="sqi_mil_detach", denoise_weight=10.0, level_weight=0.25, label_smoothing=0.005, description="detached MIL SQI stats with light label smoothing"),
    ],
    "sqi_head_mlp_tuning": [
        recipe("sqi_input_mlp64", "sqi_head_mlp_tuning", head_type="sqi_input", cls_hidden=64, description="input SQI stats with a one-hidden-layer classifier"),
        recipe("sqi_input_mlp128", "sqi_head_mlp_tuning", head_type="sqi_input", cls_hidden=128, description="input SQI stats with a wider simple classifier"),
        recipe("sqi_input_attn_mlp64", "sqi_head_mlp_tuning", head_type="sqi_input_attn", cls_hidden=64, description="attention tokens plus input SQI stats with a small classifier MLP"),
        recipe("sqi_input_mlp64_drop005", "sqi_head_mlp_tuning", head_type="sqi_input", cls_hidden=64, dropout=0.05, description="input SQI MLP with lower dropout"),
        recipe("sqi_input_mlp64_snr010", "sqi_head_mlp_tuning", head_type="sqi_input", cls_hidden=64, snr_weight=0.10, description="input SQI MLP with stronger SNR auxiliary"),
        recipe("sqi_pred_detach_mlp64_nodense", "sqi_head_mlp_tuning", head_type="sqi_pred_detach", cls_hidden=64, description="detached predicted residual/level SQI stats without dense losses"),
        recipe("sqi_pred_detach_mlp64_den5", "sqi_head_mlp_tuning", head_type="sqi_pred_detach", cls_hidden=64, denoise_weight=5.0, description="detached predicted SQI stats with light denoise"),
        recipe("sqi_pred_detach_mlp64_den10_l025", "sqi_head_mlp_tuning", head_type="sqi_pred_detach", cls_hidden=64, denoise_weight=10.0, level_weight=0.25, description="detached predicted SQI stats with light denoise and level"),
        recipe("sqi_mil_detach_mlp64_nodense", "sqi_head_mlp_tuning", head_type="sqi_mil_detach", cls_hidden=64, description="detached top-k local SQI stats without dense losses"),
        recipe("sqi_mil_detach_mlp64_den5", "sqi_head_mlp_tuning", head_type="sqi_mil_detach", cls_hidden=64, denoise_weight=5.0, description="detached top-k local SQI stats with light denoise"),
        recipe("sqi_mil_detach_mlp64_cleanrr_l025", "sqi_head_mlp_tuning", head_type="sqi_mil_detach", cls_hidden=64, denoise_weight=10.0, level_weight=0.25, level_target_mode="clean_rr", description="detached top-k local SQI stats with clean-RR level target"),
        recipe("sqi_mil_detach_mlp64_ls005", "sqi_head_mlp_tuning", head_type="sqi_mil_detach", cls_hidden=64, denoise_weight=10.0, level_weight=0.25, label_smoothing=0.005, description="detached top-k local SQI stats plus light label smoothing"),
    ],
    "sqi_mil_mlp_refined": [
        recipe("mil_cleanrr_l025_mw110", "sqi_mil_mlp_refined", head_type="sqi_mil_detach", cls_hidden=64, denoise_weight=10.0, level_weight=0.25, level_target_mode="clean_rr", class_weight_medium=1.10, description="best MIL SQI MLP plus mild medium class weight"),
        recipe("mil_cleanrr_l025_mw115", "sqi_mil_mlp_refined", head_type="sqi_mil_detach", cls_hidden=64, denoise_weight=10.0, level_weight=0.25, level_target_mode="clean_rr", class_weight_medium=1.15, description="best MIL SQI MLP plus mainline-style medium class weight"),
        recipe("mil_cleanrr_l025_mw125", "sqi_mil_mlp_refined", head_type="sqi_mil_detach", cls_hidden=64, denoise_weight=10.0, level_weight=0.25, level_target_mode="clean_rr", class_weight_medium=1.25, description="stronger medium class weight stress test"),
        recipe("mil_cleanrr_l015_mw115", "sqi_mil_mlp_refined", head_type="sqi_mil_detach", cls_hidden=64, denoise_weight=10.0, level_weight=0.15, level_target_mode="clean_rr", class_weight_medium=1.15, description="lower clean-RR level weight with medium class weight"),
        recipe("mil_cleanrr_l035_mw115", "sqi_mil_mlp_refined", head_type="sqi_mil_detach", cls_hidden=64, denoise_weight=10.0, level_weight=0.35, level_target_mode="clean_rr", class_weight_medium=1.15, description="higher clean-RR level weight with medium class weight"),
        recipe("mil_nodense_mw115", "sqi_mil_mlp_refined", head_type="sqi_mil_detach", cls_hidden=64, class_weight_medium=1.15, description="no dense loss but medium weighted"),
        recipe("mil_den5_mw115", "sqi_mil_mlp_refined", head_type="sqi_mil_detach", cls_hidden=64, denoise_weight=5.0, class_weight_medium=1.15, description="light denoise plus medium weighted"),
        recipe("mil_cleanrr_l025_ls002_mw115", "sqi_mil_mlp_refined", head_type="sqi_mil_detach", cls_hidden=64, denoise_weight=10.0, level_weight=0.25, level_target_mode="clean_rr", label_smoothing=0.002, class_weight_medium=1.15, description="tiny label smoothing with medium weight"),
        recipe("mil_cleanrr_l025_ls005_mw115", "sqi_mil_mlp_refined", head_type="sqi_mil_detach", cls_hidden=64, denoise_weight=10.0, level_weight=0.25, level_target_mode="clean_rr", label_smoothing=0.005, class_weight_medium=1.15, description="best label smoothing scale with medium weight"),
        recipe("mil_cleanrr_l025_ls010_mw115", "sqi_mil_mlp_refined", head_type="sqi_mil_detach", cls_hidden=64, denoise_weight=10.0, level_weight=0.25, level_target_mode="clean_rr", label_smoothing=0.010, class_weight_medium=1.15, description="larger label smoothing with medium weight"),
        recipe("mil_cleanrr_l025_drop005_mw115", "sqi_mil_mlp_refined", head_type="sqi_mil_detach", cls_hidden=64, denoise_weight=10.0, level_weight=0.25, level_target_mode="clean_rr", dropout=0.05, class_weight_medium=1.15, description="lower dropout around best MIL SQI MLP"),
        recipe("mil_cleanrr_l025_drop015_mw115", "sqi_mil_mlp_refined", head_type="sqi_mil_detach", cls_hidden=64, denoise_weight=10.0, level_weight=0.25, level_target_mode="clean_rr", dropout=0.15, class_weight_medium=1.15, description="higher dropout around best MIL SQI MLP"),
        recipe("mil_cleanrr_l025_h32_mw115", "sqi_mil_mlp_refined", head_type="sqi_mil_detach", cls_hidden=32, denoise_weight=10.0, level_weight=0.25, level_target_mode="clean_rr", class_weight_medium=1.15, description="smaller MLP hidden size"),
        recipe("mil_cleanrr_l025_h96_mw115", "sqi_mil_mlp_refined", head_type="sqi_mil_detach", cls_hidden=96, denoise_weight=10.0, level_weight=0.25, level_target_mode="clean_rr", class_weight_medium=1.15, description="slightly wider MLP hidden size"),
        recipe("mil_cleanrr_l025_lr4e5_mw115", "sqi_mil_mlp_refined", head_type="sqi_mil_detach", cls_hidden=64, denoise_weight=10.0, level_weight=0.25, level_target_mode="clean_rr", lr=4.0e-5, class_weight_medium=1.15, description="lower LR for reduced overfit"),
        recipe("mil_cleanrr_l025_snr010_mw115", "sqi_mil_mlp_refined", head_type="sqi_mil_detach", cls_hidden=64, denoise_weight=10.0, level_weight=0.25, level_target_mode="clean_rr", snr_weight=0.10, class_weight_medium=1.15, description="stronger SNR auxiliary with medium weight"),
    ],
    "sqi_residual_tuning": [
        recipe("sqi_resid_input", "sqi_residual_tuning", head_type="sqi_resid_input", description="pretrained CLS logits plus zero-init SQI input residual correction"),
        recipe("sqi_resid_input_drop005", "sqi_residual_tuning", head_type="sqi_resid_input", dropout=0.05, description="SQI residual correction with lower dropout"),
        recipe("sqi_resid_input_snr010", "sqi_residual_tuning", head_type="sqi_resid_input", snr_weight=0.10, description="SQI residual correction with stronger SNR auxiliary"),
        recipe("sqi_resid_pred_detach_nodense", "sqi_residual_tuning", head_type="sqi_resid_pred_detach", description="zero-init correction from detached predicted residual/level stats"),
        recipe("sqi_resid_pred_detach_den5", "sqi_residual_tuning", head_type="sqi_resid_pred_detach", denoise_weight=5.0, description="detached predicted SQI residual correction with light denoise"),
        recipe("sqi_resid_pred_detach_den10_l025", "sqi_residual_tuning", head_type="sqi_resid_pred_detach", denoise_weight=10.0, level_weight=0.25, description="detached predicted SQI correction with light dense supervision"),
        recipe("sqi_resid_mil_detach_nodense", "sqi_residual_tuning", head_type="sqi_resid_mil_detach", description="zero-init correction from detached top-k local SQI stats"),
        recipe("sqi_resid_mil_detach_den5", "sqi_residual_tuning", head_type="sqi_resid_mil_detach", denoise_weight=5.0, description="top-k local SQI residual correction with light denoise"),
        recipe("sqi_resid_mil_detach_cleanrr_l025", "sqi_residual_tuning", head_type="sqi_resid_mil_detach", denoise_weight=10.0, level_weight=0.25, level_target_mode="clean_rr", description="top-k local SQI correction plus clean-RR target"),
        recipe("sqi_resid_mil_detach_ls005", "sqi_residual_tuning", head_type="sqi_resid_mil_detach", denoise_weight=10.0, level_weight=0.25, label_smoothing=0.005, description="top-k local SQI correction plus light label smoothing"),
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
