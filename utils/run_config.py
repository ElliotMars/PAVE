"""Causal online-protocol validation and reproducible run metadata."""

from __future__ import annotations

import json
import math
import os
import subprocess
from enum import Enum
from pathlib import Path
from typing import Any


VALID_ONLINE_LEARNING_MODES = frozenset({"none", "full", "regressor"})
CAUSAL_PROTOCOL_ERROR = (
    "Causal online forecasting requires a delayed-feedback protocol. "
    "Enable --progressive_fb, --progressive_baseline_fb, or --delay_fb. "
    "Full-window immediate online updates are non-causal under "
    "rolling-origin evaluation."
)
PROGRESSIVE_BASELINE_METHODS = frozenset({"fsnet", "onenet", "dyname"})

PAPER_CRITICAL_FIELDS = (
    # Feedback protocol and model.
    "progressive_fb",
    "progressive_baseline_fb",
    "delay_fb",
    "online_learning",
    "method",
    "data",
    "data_path",
    "seq_len",
    "pred_len",
    "features",
    "num_experts",
    "expert_composition",
    "router_granularity",
    "top_k",
    "router_temperature",
    # Routing correction.
    "disable_online_correction",
    "correction_lr",
    "correction_decay",
    "correction_grad_clip",
    "correction_logit_clip",
    "router_entropy_weight",
    # Pretraining and online updates.
    "pretrain_mode",
    "expert_update_strategy",
    "learning_rate",
    "learning_rate_expert",
    "learning_rate_router",
    "online_lr_expert",
    "online_lr_router",
    "adaptive_controller",
    "expert_grad_clip",
    "router_grad_clip",
    # TSB.
    "disable_tsb",
    "disable_tsb_smoothing",
    "disable_tsb_conflict_filter",
    "tsb_alpha",
    "tsb_eps",
    "tsb_buffer_size",
    # Stable/Recovery memory.
    "stable_buffer_size",
    "recovery_buffer_size",
    "responsibility_threshold",
    "alignment_threshold",
    "credit_top_k",
    "buffer_duplicate_threshold",
    "recovery_failure_penalty",
    "recovery_degradation_margin",
    "disable_directional_recovery",
    "max_recovery_attempts",
    "recovery_batch_size",
    "recovery_loss_weight",
    "recovery_sketch_weight",
    "disable_recovery",
    "memory_refresh_interval",
    "promote_alignment_threshold",
    "promote_loss_threshold",
    "buffer_storage_dtype",
    # Capability and subspace.
    "capability_sketch_dim",
    "capability_sketch_seed",
    "subspace_scope",
    "subspace_rank",
    "subspace_max_rank",
    "subspace_energy_threshold",
    "subspace_refresh_interval",
    "subspace_min_samples",
    "subspace_eps",
    "subspace_lambda",
    "subspace_evidence_mass_scale",
    "subspace_gamma_min",
    "subspace_gamma_max",
    "disable_credit_weighted_subspace",
    # Diagnostics and comparators.
    "online_log_interval",
    "credit_diagnostic_buffer_size",
    "dynamic_comparator",
    "dynamic_comparator_max_switches",
    "dynamic_comparator_max_points",
    "strict_online_checks",
    "max_online_steps",
    # Reproducibility and device.
    "seed",
    "finetune_model_seed",
    "itr",
    "use_gpu",
    "gpu",
    "use_multi_gpu",
    "devices",
    "device_ids",
)


def _online_learning_mode(args: Any) -> str:
    mode = str(getattr(args, "online_learning", "none")).lower()
    if mode not in VALID_ONLINE_LEARNING_MODES:
        valid = ", ".join(sorted(VALID_ONLINE_LEARNING_MODES))
        raise ValueError(
            f"online_learning must be one of {{{valid}}}, got {mode!r}"
        )
    return mode


def validate_online_feedback_protocol(args: Any) -> str:
    """Fail fast for online updates that can consume a future target window."""

    mode = _online_learning_mode(args)
    if mode == "none":
        return "none_offline"
    progressive_baseline = bool(
        getattr(args, "progressive_baseline_fb", False)
    )
    if progressive_baseline:
        method = str(getattr(args, "method", "")).lower()
        if bool(getattr(args, "progressive_fb", False)):
            raise ValueError(
                "--progressive_baseline_fb is independent of --progressive_fb"
            )
        if method not in PROGRESSIVE_BASELINE_METHODS:
            supported = ", ".join(sorted(PROGRESSIVE_BASELINE_METHODS))
            raise ValueError(
                "--progressive_baseline_fb only supports "
                f"{{{supported}}}, got {method!r}"
            )
        return "progressive_baseline_control"
    if bool(getattr(args, "progressive_fb", False)):
        return "progressive"
    if bool(getattr(args, "delay_fb", False)):
        return "legacy_delayed"
    raise ValueError(CAUSAL_PROTOCOL_ERROR)


def _json_safe(value: Any) -> Any:
    """Convert whitelisted scalar configuration values without large dumps."""

    if value is None:
        return value
    if isinstance(value, Enum):
        return str(value.value)
    if isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if hasattr(value, "item") and type(value).__module__.startswith("numpy"):
        return _json_safe(value.item())
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }
    return f"<non-serializable:{type(value).__name__}>"


def _git_output(*arguments: str) -> str | None:
    repository = Path(__file__).resolve().parents[1]
    try:
        result = subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def read_git_metadata() -> dict[str, Any]:
    """Read repository identity without making metadata a run dependency."""

    commit = _git_output("rev-parse", "HEAD")
    status = _git_output("status", "--porcelain")
    return {
        "git_commit": commit or None,
        "git_dirty": None if status is None else bool(status),
    }


def _effective_update_strategy(args: Any) -> str | None:
    if not hasattr(args, "expert_update_strategy"):
        return None
    strategy = str(args.expert_update_strategy).lower()
    if bool(getattr(args, "disable_tsb", False)):
        if strategy == "tsb":
            return "plain"
        if strategy == "hybrid":
            return "subspace"
    return strategy


def build_run_config(
    args: Any,
    iteration_index: int | None = None,
) -> dict[str, Any]:
    """Build a small whitelist of paper-critical, JSON-safe configuration."""

    protocol = validate_online_feedback_protocol(args)
    config: dict[str, Any] = {"schema_version": 1}
    for field in PAPER_CRITICAL_FIELDS:
        if not hasattr(args, field):
            continue
        value = getattr(args, field)
        if field == "progressive_baseline_fb" and not bool(value):
            continue
        config[field] = _json_safe(value)

    effective_strategy = _effective_update_strategy(args)
    online_enabled = protocol != "none_offline"
    tsb_enabled = (
        online_enabled and effective_strategy in {"tsb", "hybrid"}
    )
    config.update(
        {
            "causal_feedback_protocol": protocol,
            "effective_expert_update_strategy": effective_strategy,
            "tsb_enabled": tsb_enabled,
            "tsb_smoothing_enabled": tsb_enabled
            and not bool(getattr(args, "disable_tsb_smoothing", False)),
            "tsb_conflict_filter_enabled": tsb_enabled
            and not bool(
                getattr(args, "disable_tsb_conflict_filter", False)
            ),
        }
    )

    is_pace = str(getattr(args, "method", "")).lower() == "multi_expert"
    if is_pace:
        direction_awareness_enabled = bool(
            online_enabled
            and not getattr(args, "disable_directional_recovery", False)
            and not getattr(args, "disable_version_awareness", False)
        )
        config.update(
            {
                "online_correction_enabled": online_enabled and not bool(
                    getattr(args, "disable_online_correction", False)
                ),
                "recovery_enabled": online_enabled and not bool(
                    getattr(args, "disable_recovery", False)
                ),
                "direction_awareness_enabled": direction_awareness_enabled,
                "directional_recovery_enabled": direction_awareness_enabled,
                "capability_rebase_enabled": direction_awareness_enabled,
                "capability_reference_loss_enabled": (
                    direction_awareness_enabled
                ),
                "credit_weighted_subspace": not bool(
                    getattr(args, "disable_credit_weighted_subspace", False)
                ),
            }
        )
    if protocol == "progressive_baseline_control":
        config["feedback_protocol"] = protocol
    if iteration_index is not None:
        config["iteration_index"] = int(iteration_index)
    if hasattr(args, "subspace_rank"):
        config["subspace_rank_mode"] = (
            "energy" if int(args.subspace_rank) == 0 else "fixed"
        )
    use_gpu = bool(getattr(args, "use_gpu", False))
    gpu = getattr(args, "gpu", None)
    config["device"] = (
        f"cuda:{gpu}" if use_gpu and gpu is not None
        else "cuda" if use_gpu
        else "cpu"
    )
    config.update(read_git_metadata())
    return _json_safe(config)


def save_run_config(
    args: Any,
    result_directory: str | os.PathLike[str],
    iteration_index: int | None = None,
) -> str:
    """Save run_config.json in one run's own result directory."""

    directory = Path(result_directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "run_config.json"
    config = build_run_config(args, iteration_index=iteration_index)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            config,
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")
    return str(path)


def annotate_run_config(
    config_path: str | os.PathLike[str], metadata: dict[str, Any]
) -> dict[str, Any]:
    """Merge post-run scalar protocol counts into an existing run config."""

    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    config.update(_json_safe(metadata))
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            config,
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")
    return config
