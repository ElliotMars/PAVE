"""Per-iteration result persistence and lightweight diagnostics aggregation.

The aggregate path intentionally reads only JSON summaries.  Step-level NPZ
files can be large on ECL and must remain isolated inside ``itr_<index>``.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np


_METRIC_SOURCES: dict[str, tuple[str, ...]] = {
    "online_mse": ("metrics", "online_mse", "mean"),
    "router_gap": ("metrics", "router_gap", "mean"),
    "responsibility_js_divergence": (
        "metrics",
        "responsibility_js_divergence",
        "mean",
    ),
    "ranking_reversal_rate": ("metrics", "ranking_reversal", "mean"),
    "capability_alignment": ("metrics", "capability_alignment", "mean"),
    "stable_buffer_average_size": ("metrics", "stable_buffer_size", "mean"),
    "recovery_buffer_average_size": (
        "metrics",
        "recovery_buffer_size",
        "mean",
    ),
    "recovery_success_rate": ("metrics", "recovery_success_rate", "mean"),
    "subspace_rank": ("metrics", "subspace_rank", "mean"),
    "subspace_captured_energy": (
        "metrics",
        "subspace_captured_energy",
        "mean",
    ),
    "z_norm": ("metrics", "z_norm", "mean"),
    "tsb_conflict_rate": ("metrics", "tsb_conflict_rate", "mean"),
    "mean_grad_cosine": (
        "tsb_gradient_diagnostics",
        "mean_grad_cosine",
    ),
    "gradient_conflict_rate": (
        "tsb_gradient_diagnostics",
        "conflict_rate",
    ),
    "mean_tsb_modification_ratio": (
        "tsb_gradient_diagnostics",
        "mean_tsb_modification_ratio",
    ),
}

_COUNTER_FIELDS = (
    "recovery_dropped_after_success",
    "promotion_count",
    "demotion_count",
    "drop_count",
)


def _nested(summary: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = summary
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    return value


def _scalar_mean(value: Any) -> float | None:
    """Convert a scalar/vector summary value to one finite iteration scalar."""

    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        return None
    return float(array.mean())


def _stats(values: list[float]) -> dict[str, Any] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
        "values": array.tolist(),
        "count": int(array.size),
    }


def annotate_iteration_summary(
    summary_path: str,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Merge run metadata into one already-written diagnostics summary."""

    with open(summary_path, "r", encoding="utf-8") as handle:
        summary = json.load(handle)
    summary["iteration"] = dict(metadata)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    return summary


def aggregate_online_diagnostics(
    iteration_summaries: Iterable[str | Mapping[str, Any]],
    output_path: str | None = None,
) -> dict[str, Any]:
    """Aggregate iteration JSON summaries without loading step-level NPZ data."""

    summaries: list[dict[str, Any]] = []
    for source in iteration_summaries:
        if isinstance(source, Mapping):
            summaries.append(dict(source))
        else:
            with open(source, "r", encoding="utf-8") as handle:
                summaries.append(json.load(handle))

    iterations = []
    for index, summary in enumerate(summaries):
        metadata = summary.get("iteration", {})
        iterations.append(
            {
                "iteration": int(metadata.get("index", index)),
                "seed": metadata.get("seed"),
                "num_records": int(summary.get("num_records", 0)),
                "completed_record_count": metadata.get(
                    "completed_record_count"
                ),
                "early_ended": metadata.get("early_ended"),
                "strict_checks_passed": metadata.get(
                    "strict_checks_passed"
                ),
            }
        )

    metrics: dict[str, Any] = {}
    for output_name, path in _METRIC_SOURCES.items():
        values = [
            scalar
            for summary in summaries
            if (scalar := _scalar_mean(_nested(summary, path))) is not None
        ]
        metrics[output_name] = _stats(values)

    counters: dict[str, Any] = {}
    for field in _COUNTER_FIELDS:
        values = [
            scalar
            for summary in summaries
            if (
                scalar := _scalar_mean(
                    _nested(summary, ("counters", field))
                )
            )
            is not None
        ]
        counters[field] = _stats(values)

    aggregate = {
        "num_iterations": len(summaries),
        "iterations": iterations,
        "metrics": metrics,
        "counters": counters,
    }
    if output_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(aggregate, handle, ensure_ascii=False, indent=2)
    return aggregate


def save_prediction_results(
    directory: str,
    metrics: Any,
    predictions: Any,
    targets: Any,
    mae_curve: Any,
    mse_curve: Any,
) -> None:
    """Save one iteration's prediction products using legacy filenames."""

    os.makedirs(directory, exist_ok=True)
    np.save(os.path.join(directory, "metrics.npy"), np.asarray(metrics))
    np.save(os.path.join(directory, "preds.npy"), np.asarray(predictions))
    np.save(os.path.join(directory, "trues.npy"), np.asarray(targets))
    np.save(os.path.join(directory, "mae.npy"), np.asarray(mae_curve))
    np.save(os.path.join(directory, "mse.npy"), np.asarray(mse_curve))
