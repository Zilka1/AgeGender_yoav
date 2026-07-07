"""Builds the parametric-vs-kNN and architecture-ablation comparison tables.

All functions here operate on already-computed metric dicts (produced by
``scripts/evaluate.py`` and ``scripts/build_knn_index.py``) -- nothing in
this module invents numbers; it only aggregates and tabulates real results.
"""

from __future__ import annotations

import pandas as pd


def build_parametric_vs_knn_table(parametric_metrics: dict, knn_metrics: dict) -> pd.DataFrame:
    """Side-by-side comparison table for the metrics listed in the spec.

    Expects both metric dicts to share the same keys (age_mae, age_rmse,
    interval_coverage, mean_interval_width, gender_accuracy,
    abstention_rate, mean_confidence, latency_ms_per_image).
    """
    keys = [
        "age_mae", "age_rmse", "interval_coverage", "mean_interval_width",
        "gender_accuracy", "abstention_rate", "mean_confidence", "latency_ms_per_image",
    ]
    rows = []
    for key in keys:
        rows.append(
            {
                "metric": key,
                "parametric": parametric_metrics.get(key),
                "knn": knn_metrics.get(key),
            }
        )
    return pd.DataFrame(rows)


def build_architecture_ablation_table(experiment_results: dict[str, dict]) -> pd.DataFrame:
    """One row per experiment (0, A-D), columns = parameter counts + key metrics.

    ``experiment_results`` maps experiment name -> a dict with keys
    ``parameter_breakdown`` (from ``ParameterBreakdown.as_dict()``),
    ``val_metrics`` / ``test_metrics``, and ``mean_epoch_time_seconds``.
    """
    rows = []
    for name, result in experiment_results.items():
        params = result.get("parameter_breakdown", {})
        metrics = result.get("test_metrics", result.get("val_metrics", {}))
        rows.append(
            {
                "experiment": name,
                "backbone_name": params.get("backbone_name"),
                "backbone_params": params.get("backbone_parameters"),
                "adapter_params": params.get("adapter_parameters"),
                "total_params": params.get("total_parameters"),
                "age_mae": metrics.get("age_mae"),
                "gender_accuracy": metrics.get("gender_accuracy"),
                "interval_coverage": metrics.get("interval_coverage"),
                "mean_epoch_time_seconds": result.get("mean_epoch_time_seconds"),
            }
        )
    return pd.DataFrame(rows)


def build_backbone_comparison_table(cnn_metrics: dict, resnet_metrics: dict) -> pd.DataFrame:
    """Side-by-side table for the plain-CNN-vs-Custom-ResNet-18 backbone comparison.

    Each of ``cnn_metrics`` / ``resnet_metrics`` is expected to already
    merge that experiment's parameter breakdown, timing, and test metrics
    into one flat dict (see ``scripts/generate_architecture_report.py``).
    Missing keys render as ``None`` rather than being fabricated.
    """
    keys = [
        ("backbone_name", "Backbone"),
        ("total_parameters", "Total parameters"),
        ("backbone_parameters", "Backbone parameters"),
        ("mean_epoch_time_seconds", "Mean epoch time (s)"),
        ("latency_ms_per_image", "Inference latency per image (ms)"),
        ("age_mae", "Age MAE"),
        ("age_rmse", "Age RMSE"),
        ("gender_accuracy", "Gender-label accuracy"),
        ("abstention_rate", "Abstention rate"),
        ("interval_coverage", "Raw interval coverage"),
        ("interval_coverage_calibrated", "Calibrated interval coverage"),
        ("mean_interval_width", "Mean interval width"),
    ]
    rows = []
    for key, label in keys:
        rows.append({"metric": label, "simple_cnn": cnn_metrics.get(key), "custom_resnet18": resnet_metrics.get(key)})
    return pd.DataFrame(rows)
