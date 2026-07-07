"""Cross-model backbone comparison analyses (clean-test summary, gender and
age selective-prediction / risk-coverage analysis, tail-error analysis, and
an honest conditional interpretation), used by ``scripts/compare_backbones.py``.

Every function here operates on already-computed per-sample prediction
arrays (the same ``preds`` dict shape ``scripts/evaluate.py:run_inference``
returns) -- nothing here re-runs a model or duplicates training/inference
logic. All numeric outputs are either directly measured or standard,
documented aggregate statistics (percentiles, AURC, bootstrap CIs); nothing
is fabricated, and the "is the added complexity justified" interpretation
is explicitly conditional on the measured numbers (see
``build_final_interpretation``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.metrics import (
    abstention_rate, age_error_percentiles, age_mae, age_r2, age_rmse, age_tail_error_rates,
    age_uncertainty_by_bucket, confidence_statistics, gender_accuracy, gender_coverage,
    gender_effective_accuracy, interval_coverage, mean_interval_width,
)
from src.evaluation.selective import (
    compute_aurc, paired_bootstrap_risk_diff_ci, risk_at_coverage, selective_risk_coverage_curve,
)

COMMON_COVERAGE_LEVELS = (0.80, 0.90, 0.95, 0.98)
AGE_ERROR_TAIL_THRESHOLDS = (5, 10, 15, 20)
DEVELOPMENTAL_AGE_BUCKETS = (
    (0, 13, "0-12"), (13, 20, "13-19"), (20, 35, "20-34"),
    (35, 50, "35-49"), (50, 65, "50-64"), (65, 200, "65+"),
)


def _gender_arrays(preds: dict, confidence_threshold: float) -> dict:
    mask = preds["gender_mask"].astype(bool)
    probs = preds["probs"][mask]
    y_true = preds["gender"][mask].astype(int)
    predicted = probs.argmax(axis=1)
    confidence = probs.max(axis=1)
    abstain = confidence < confidence_threshold
    return {"y_true": y_true, "predicted": predicted, "confidence": confidence, "abstain": abstain}


def build_clean_test_summary(
    model_name: str, preds: dict, confidence_threshold: float, calibration: dict | None = None,
    parameter_breakdown: dict | None = None, mean_epoch_time_seconds: float | None = None,
) -> dict:
    """One model's full clean-test-set summary row (Part B.1 of the backbone comparison spec)."""
    summary: dict = {"model": model_name}
    age_mask = preds["age_mask"].astype(bool)
    if age_mask.any():
        y_true = preds["age"][age_mask]
        q10, q50, q90 = preds["q10"][age_mask], preds["q50"][age_mask], preds["q90"][age_mask]
        summary.update({
            "age_mae": age_mae(y_true, q50),
            "age_rmse": age_rmse(y_true, q50),
            "age_r2": age_r2(y_true, q50),
            **{f"age_error_{k}": v for k, v in age_error_percentiles(y_true, q50).items()},
            **{f"age_error_frac_{k}": v for k, v in age_tail_error_rates(y_true, q50, AGE_ERROR_TAIL_THRESHOLDS).items()},
            "raw_interval_coverage": interval_coverage(y_true, q10, q90),
            "raw_interval_width": mean_interval_width(q10, q90),
        })
        if calibration is not None:
            from src.evaluation.calibration import apply_conformal_offset

            q10_cal, q90_cal = apply_conformal_offset(q10, q90, calibration["offset"])
            summary["calibrated_interval_coverage"] = interval_coverage(y_true, q10_cal, q90_cal)
            summary["calibrated_interval_width"] = mean_interval_width(q10_cal, q90_cal)

    gender_mask = preds["gender_mask"].astype(bool)
    if gender_mask.any():
        g = _gender_arrays(preds, confidence_threshold)
        summary.update({
            "gender_selective_accuracy": gender_accuracy(g["y_true"], g["predicted"], g["abstain"]),
            "gender_coverage": gender_coverage(g["abstain"]),
            "gender_abstention_rate": abstention_rate(g["abstain"]),
            "gender_effective_accuracy": gender_effective_accuracy(g["y_true"], g["predicted"], g["abstain"]),
            "gender_confidence_stats": confidence_statistics(g["confidence"]),
        })

    summary["latency_ms_per_image"] = preds.get("latency_ms_per_image")
    if parameter_breakdown:
        summary["total_parameters"] = parameter_breakdown.get("total_parameters")
        summary["backbone_parameters"] = parameter_breakdown.get("backbone_parameters")
    summary["mean_epoch_time_seconds"] = mean_epoch_time_seconds
    return summary


def build_clean_test_table(summaries: dict[str, dict]) -> pd.DataFrame:
    """One row per model; flattens nested confidence-stats dicts into top-level columns."""
    rows = []
    for name, summary in summaries.items():
        row = {k: v for k, v in summary.items() if not isinstance(v, dict)}
        for prefix, nested in summary.items():
            if isinstance(nested, dict):
                for k, v in nested.items():
                    row[f"{prefix}_{k}"] = v
        row["model"] = name
        rows.append(row)
    return pd.DataFrame(rows)


def build_gender_risk_coverage_analysis(
    models_preds: dict[str, dict], confidence_threshold: float, primary_model: str | None = None,
) -> dict:
    """Gender selective-risk-vs-coverage analysis across models (Part B.2).

    Returns ``{"curves": {model: (coverages, risks)}, "aurc": {model: float},
    "at_coverage": DataFrame, "pairwise_bootstrap": {model: ci_dict}}``.
    ``pairwise_bootstrap`` compares every non-primary model against
    ``primary_model`` (default: the first model) at each common coverage
    level, using the paired bootstrap (valid only when both models share
    the same index-aligned samples -- callers must pass predictions
    computed over the identical test set for every model).
    """
    curves, aurc, confidences, losses = {}, {}, {}, {}
    for name, preds in models_preds.items():
        g = _gender_arrays(preds, confidence_threshold)
        loss = (g["predicted"] != g["y_true"]).astype(float)
        coverages, risks = selective_risk_coverage_curve(g["confidence"], loss)
        curves[name] = (coverages, risks)
        aurc[name] = compute_aurc(coverages, risks)
        confidences[name], losses[name] = g["confidence"], loss

    at_coverage_rows = []
    for level in COMMON_COVERAGE_LEVELS:
        row = {"coverage": level}
        for name in models_preds:
            row[f"{name}_risk"] = risk_at_coverage(*curves[name], level)
        at_coverage_rows.append(row)
    at_coverage_table = pd.DataFrame(at_coverage_rows)

    pairwise_bootstrap = {}
    primary = primary_model or next(iter(models_preds))
    if primary in confidences:
        for name in models_preds:
            if name == primary:
                continue
            n = min(len(confidences[primary]), len(confidences[name]))
            if len(confidences[primary]) != len(confidences[name]):
                continue  # not index-aligned (different sample counts) -- skip rather than misuse the paired test
            pairwise_bootstrap[name] = {
                level: paired_bootstrap_risk_diff_ci(
                    confidences[primary][:n], losses[primary][:n],
                    confidences[name][:n], losses[name][:n], target_coverage=level,
                )
                for level in COMMON_COVERAGE_LEVELS
            }

    return {"curves": curves, "aurc": aurc, "at_coverage": at_coverage_table, "pairwise_bootstrap": pairwise_bootstrap}


def build_age_selective_analysis(models_preds: dict[str, dict], primary_model: str | None = None) -> dict:
    """Age selective-prediction analysis using interval width as the confidence score (Part B.3)."""
    mae_curves, rmse_curves, aurc, confidences, abs_errors = {}, {}, {}, {}, {}
    for name, preds in models_preds.items():
        mask = preds["age_mask"].astype(bool)
        y_true, q10, q50, q90 = preds["age"][mask], preds["q10"][mask], preds["q50"][mask], preds["q90"][mask]
        confidence = -(q90 - q10)  # narrower interval = higher confidence
        errors = np.abs(y_true - q50)
        mae_coverages, mae_risks = selective_risk_coverage_curve(confidence, errors)
        _, rmse_risks = selective_risk_coverage_curve(confidence, errors ** 2)
        rmse_risks = np.sqrt(rmse_risks)

        mae_curves[name] = (mae_coverages, mae_risks)
        rmse_curves[name] = (mae_coverages, rmse_risks)
        aurc[name] = compute_aurc(mae_coverages, mae_risks)
        confidences[name], abs_errors[name] = confidence, errors

    at_coverage_rows = []
    for level in COMMON_COVERAGE_LEVELS:
        row = {"coverage": level}
        for name in models_preds:
            row[f"{name}_mae"] = risk_at_coverage(*mae_curves[name], level)
        at_coverage_rows.append(row)
    at_coverage_table = pd.DataFrame(at_coverage_rows)

    pairwise_bootstrap = {}
    primary = primary_model or next(iter(models_preds))
    if primary in confidences:
        for name in models_preds:
            if name == primary or len(confidences[primary]) != len(confidences[name]):
                continue
            pairwise_bootstrap[name] = {
                level: paired_bootstrap_risk_diff_ci(
                    confidences[primary], abs_errors[primary], confidences[name], abs_errors[name],
                    target_coverage=level,
                )
                for level in COMMON_COVERAGE_LEVELS
            }

    return {
        "mae_curves": mae_curves, "rmse_curves": rmse_curves, "aurc": aurc,
        "at_coverage": at_coverage_table, "pairwise_bootstrap": pairwise_bootstrap,
    }


def build_tail_error_analysis(models_preds: dict[str, dict]) -> dict:
    """CDF data, tail-error-rate bars, and per-age-bucket MAE table across models (Part B.4)."""
    errors_by_model, tail_rates_by_model, bucket_tables = {}, {}, {}
    for name, preds in models_preds.items():
        mask = preds["age_mask"].astype(bool)
        y_true, q50, q10, q90 = preds["age"][mask], preds["q50"][mask], preds["q10"][mask], preds["q90"][mask]
        errors = np.abs(y_true - q50)
        errors_by_model[name] = errors
        tail_rates_by_model[name] = age_tail_error_rates(y_true, q50, AGE_ERROR_TAIL_THRESHOLDS)

        bucket_edges = [lo for lo, _, _ in DEVELOPMENTAL_AGE_BUCKETS] + [DEVELOPMENTAL_AGE_BUCKETS[-1][1]]
        raw_buckets = age_uncertainty_by_bucket(y_true, q10, q50, q90, bucket_edges=bucket_edges)
        relabeled = {label: raw_buckets[key] for (_, _, label), key in zip(DEVELOPMENTAL_AGE_BUCKETS, raw_buckets)}
        bucket_tables[name] = relabeled

    error_percentiles = {name: age_error_percentiles(preds["age"][preds["age_mask"].astype(bool)], preds["q50"][preds["age_mask"].astype(bool)]) for name, preds in models_preds.items()}

    bucket_rows = []
    for _, _, label in DEVELOPMENTAL_AGE_BUCKETS:
        row = {"age_bucket": label}
        for name in models_preds:
            stats = bucket_tables[name][label]
            row[f"{name}_count"] = stats["count"]
            row[f"{name}_mae"] = stats["mae"]
        bucket_rows.append(row)

    return {
        "errors_by_model": errors_by_model,
        "tail_rates_by_model": tail_rates_by_model,
        "error_percentiles": error_percentiles,
        "bucket_table": pd.DataFrame(bucket_rows),
    }


def build_final_interpretation(
    clean_summary_table: pd.DataFrame, gender_risk_analysis: dict, age_selective_analysis: dict,
    resnet_name: str, comparison_names: list[str],
) -> str:
    """Honest, conditional "is additional residual complexity justified?" narrative.

    Never asserts an advantage that isn't backed by the measured numbers,
    never treats a single-seed difference as decisive, and explicitly
    states the compact/plain alternative is preferred when results are
    tied or favor it -- the whole point of this analysis is to be capable
    of concluding *against* the residual architecture.

    ``gender_risk_analysis`` / ``age_selective_analysis`` must have been
    built with ``primary_model=resnet_name`` (see
    ``build_gender_risk_coverage_analysis`` / ``build_age_selective_analysis``),
    since their ``pairwise_bootstrap[other]`` entries are defined as
    ResNet-vs-``other`` (``risk_diff_b_minus_a = risk(other) - risk(resnet)``;
    positive means ResNet has *lower* risk, i.e. an advantage).
    """
    lines = ["## Is Additional Residual Complexity Justified?\n"]

    if resnet_name not in clean_summary_table["model"].values:
        return "\n".join(lines) + (
            "_Not available -- the ResNet checkpoint has not been evaluated in this run._\n"
        )

    resnet_row = clean_summary_table[clean_summary_table["model"] == resnet_name].iloc[0]
    findings = []
    decisive_advantage_found = False

    for other in comparison_names:
        if other == resnet_name or other not in clean_summary_table["model"].values:
            continue
        other_row = clean_summary_table[clean_summary_table["model"] == other].iloc[0]

        param_diff = resnet_row.get("total_parameters", 0) - other_row.get("total_parameters", 0)
        latency_diff = (resnet_row.get("latency_ms_per_image") or 0) - (other_row.get("latency_ms_per_image") or 0)

        gender_aurc_resnet = gender_risk_analysis.get("aurc", {}).get(resnet_name)
        gender_aurc_other = gender_risk_analysis.get("aurc", {}).get(other)
        age_aurc_resnet = age_selective_analysis.get("aurc", {}).get(resnet_name)
        age_aurc_other = age_selective_analysis.get("aurc", {}).get(other)

        # pairwise_bootstrap[other] compares (a=resnet, b=other): a positive
        # risk_diff_b_minus_a means "other" has higher risk than ResNet, i.e.
        # a ResNet advantage.
        gender_ci = gender_risk_analysis.get("pairwise_bootstrap", {}).get(other, {})
        age_ci = age_selective_analysis.get("pairwise_bootstrap", {}).get(other, {})
        gender_significant = any(ci.get("excludes_zero") and ci.get("risk_diff_b_minus_a", 0) > 0 for ci in gender_ci.values())
        age_significant = any(ci.get("excludes_zero") and ci.get("risk_diff_b_minus_a", 0) > 0 for ci in age_ci.values())

        resnet_lower_gender_aurc = gender_aurc_resnet is not None and gender_aurc_other is not None and gender_aurc_resnet < gender_aurc_other
        resnet_lower_age_aurc = age_aurc_resnet is not None and age_aurc_other is not None and age_aurc_resnet < age_aurc_other

        if (resnet_lower_gender_aurc and gender_significant) or (resnet_lower_age_aurc and age_significant):
            decisive_advantage_found = True
            findings.append(
                f"- vs. `{other}`: Custom ResNet-18 shows a statistically supported "
                f"(bootstrap CI excludes zero) reduction in selective risk "
                f"({'gender AURC' if resnet_lower_gender_aurc and gender_significant else 'age AURC'}), "
                f"at the cost of {int(param_diff):+,} parameters and {latency_diff:+.2f} ms/image. "
                "This is a plausible deployment scenario where the added residual "
                "complexity pays for itself -- e.g. when tail-risk/selective-prediction "
                "quality at high coverage matters more than raw parameter/latency cost.\n"
            )
        else:
            findings.append(
                f"- vs. `{other}`: no statistically supported ResNet advantage was found "
                f"in gender or age selective risk (AURC) at the coverage levels tested "
                f"({COMMON_COVERAGE_LEVELS}). Given ResNet costs {int(param_diff):+,} more "
                f"parameters and {latency_diff:+.2f} ms/image more latency, `{other}` is the "
                "preferred model for this dataset and training setup unless a specific "
                "downstream requirement (not evaluated here) favors ResNet.\n"
            )

    lines.extend(findings)
    lines.append(
        "\n**Caveat.** These conclusions reflect the seed(s), dataset, and coverage "
        "levels evaluated in this run only. A single-seed AURC difference, even if "
        "numerically in ResNet's favor, is not treated as decisive here unless the "
        "bootstrap CI excludes zero; see the mean +/- std table across seeds for "
        "additional evidence of stability before generalizing this conclusion.\n"
    )
    if not decisive_advantage_found:
        lines.append(
            "\n**Overall:** across the comparisons run, no measured evidence supports "
            "that the additional residual-connection complexity is justified for this "
            "dataset and training setup -- the compact/plain alternative(s) are at least "
            "as good on the metrics evaluated here.\n"
        )
    return "\n".join(lines)
