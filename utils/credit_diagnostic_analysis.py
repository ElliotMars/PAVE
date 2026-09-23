"""Offline binned analysis for delayed credit-capability diagnostics."""

from __future__ import annotations

import argparse
import os
from typing import Sequence

import numpy as np


DEFAULT_UPDATE_DELTA_BINS = (0.0, 1.0, 2.0, 4.0, 8.0, 16.0, np.inf)
DEFAULT_ALIGNMENT_BINS = (0.0, 0.25, 0.5, 0.75, 1.0)


def _validated_edges(values: Sequence[float], name: str) -> np.ndarray:
    edges = np.asarray(values, dtype=np.float64)
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError(f"{name} must contain at least two edges")
    if np.isnan(edges).any() or not np.all(np.diff(edges) > 0):
        raise ValueError(f"{name} must be strictly increasing without NaN")
    return edges


def _bucket_indices(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    indices = np.searchsorted(edges, values, side="right") - 1
    indices[values == edges[-1]] = edges.size - 2
    valid = (values >= edges[0]) & (values <= edges[-1])
    indices[~valid] = -1
    return indices


def _binned_statistics(
    prefix: str,
    values: np.ndarray,
    edges: np.ndarray,
    metrics: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    indices = _bucket_indices(values, edges)
    num_bins = edges.size - 1
    result = {
        f"{prefix}_bin_edges": edges,
        f"{prefix}_count": np.zeros(num_bins, dtype=np.int64),
    }
    for metric_name in metrics:
        result[f"{prefix}_{metric_name}"] = np.full(
            num_bins, np.nan, dtype=np.float64
        )

    for bin_index in range(num_bins):
        mask = indices == bin_index
        count = int(mask.sum())
        result[f"{prefix}_count"][bin_index] = count
        if count == 0:
            continue
        for metric_name, metric_values in metrics.items():
            result[f"{prefix}_{metric_name}"][bin_index] = float(
                metric_values[mask].mean()
            )
    return result


def analyze_credit_diagnostics(
    input_path: str,
    output_path: str,
    update_delta_bins: Sequence[float] = DEFAULT_UPDATE_DELTA_BINS,
    alignment_bins: Sequence[float] = DEFAULT_ALIGNMENT_BINS,
) -> dict[str, np.ndarray]:
    """Read bounded record diagnostics and save plot-ready binned statistics."""

    with np.load(input_path, allow_pickle=False) as archive:
        required = {
            "expert_update_delta",
            "js_divergence",
            "ranking_reversal",
            "router_gap",
        }
        missing = sorted(required.difference(archive.files))
        if missing:
            raise KeyError(f"missing credit diagnostic fields: {missing}")
        expert_update_delta = np.asarray(
            archive["expert_update_delta"], dtype=np.float64
        )
        js_divergence = np.asarray(archive["js_divergence"], dtype=np.float64)
        ranking_reversal = np.asarray(
            archive["ranking_reversal"], dtype=np.float64
        )
        router_gap = np.asarray(archive["router_gap"], dtype=np.float64)
        if "mean_alignment" in archive.files:
            mean_alignment = np.asarray(
                archive["mean_alignment"], dtype=np.float64
            )
        elif "capability_alignment" in archive.files:
            capability_alignment = np.asarray(
                archive["capability_alignment"], dtype=np.float64
            )
            if capability_alignment.ndim != 2:
                raise ValueError("capability_alignment must have shape [N,E]")
            mean_alignment = capability_alignment.mean(axis=1)
        else:
            raise KeyError(
                "credit diagnostics require mean_alignment or capability_alignment"
            )

    fields = {
        "expert_update_delta": expert_update_delta,
        "js_divergence": js_divergence,
        "ranking_reversal": ranking_reversal,
        "mean_alignment": mean_alignment,
        "router_gap": router_gap,
    }
    lengths = {name: value.shape for name, value in fields.items()}
    if any(value.ndim != 1 for value in fields.values()):
        raise ValueError(f"record-level fields must be one-dimensional: {lengths}")
    num_records = expert_update_delta.size
    if any(value.size != num_records for value in fields.values()):
        raise ValueError(f"record-level fields must have equal length: {lengths}")
    if any(not np.isfinite(value).all() for value in fields.values()):
        raise FloatingPointError("credit diagnostics contain NaN or Inf")
    if np.any(expert_update_delta < 0):
        raise ValueError("expert_update_delta must be non-negative")
    if np.any((mean_alignment < 0) | (mean_alignment > 1)):
        raise ValueError("mean alignment must be in [0,1]")

    update_edges = _validated_edges(update_delta_bins, "update_delta_bins")
    alignment_edges = _validated_edges(alignment_bins, "alignment_bins")
    results = {
        "num_records": np.asarray(num_records, dtype=np.int64),
        **_binned_statistics(
            "update_delta",
            expert_update_delta,
            update_edges,
            {
                "mean_js": js_divergence,
                "reversal_rate": ranking_reversal,
                "mean_alignment": mean_alignment,
                "mean_router_gap": router_gap,
            },
        ),
        **_binned_statistics(
            "alignment",
            mean_alignment,
            alignment_edges,
            {
                "mean_js": js_divergence,
                "reversal_rate": ranking_reversal,
                "mean_router_gap": router_gap,
            },
        ),
    }
    directory = os.path.dirname(output_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    np.savez_compressed(output_path, **results)
    return results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Bin delayed credit-capability diagnostics without rerunning a model"
    )
    parser.add_argument("input_path", help="credit_diagnostics.npz")
    parser.add_argument("output_path", help="output plot-ready NPZ")
    parser.add_argument("--update_delta_bins", nargs="+", type=float)
    parser.add_argument("--alignment_bins", nargs="+", type=float)
    args = parser.parse_args(argv)
    analyze_credit_diagnostics(
        args.input_path,
        args.output_path,
        update_delta_bins=(
            args.update_delta_bins
            if args.update_delta_bins is not None
            else DEFAULT_UPDATE_DELTA_BINS
        ),
        alignment_bins=(
            args.alignment_bins
            if args.alignment_bins is not None
            else DEFAULT_ALIGNMENT_BINS
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
