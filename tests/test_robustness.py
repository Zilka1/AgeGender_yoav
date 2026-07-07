"""Tests for deterministic robustness corruption functions.

Covers the full required set: blur, brightness, contrast, Gaussian
noise, JPEG compression, partial occlusion, resize degradation
(low_resolution), and grayscale conversion.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.evaluation.robustness import (
    CORRUPTION_NAMES, apply_corruption, build_robustness_diff_table, compute_degradation, gaussian_blur,
    gaussian_noise, grayscale, high_brightness, high_contrast, iter_corruption_configs, jpeg_compression,
    low_brightness, low_contrast, low_resolution, partial_crop, partial_occlusion,
)


def _sample_image(size=(64, 64)) -> Image.Image:
    rng = np.random.default_rng(0)
    array = rng.integers(0, 255, size=(size[1], size[0], 3), dtype=np.uint8)
    return Image.fromarray(array)


def test_all_required_corruption_types_are_registered():
    required = {
        "gaussian_blur", "gaussian_noise", "low_resolution", "jpeg_compression",
        "low_brightness", "high_brightness", "low_contrast", "high_contrast",
        "grayscale", "partial_occlusion",
    }
    assert required <= set(CORRUPTION_NAMES)


def test_each_corruption_preserves_image_size():
    image = _sample_image((80, 60))
    corruptions = [
        (gaussian_blur, 1.5), (gaussian_noise, 0.1), (low_resolution, 0.3), (jpeg_compression, 20),
        (low_brightness, 0.5), (high_brightness, 1.6), (low_contrast, 0.5), (high_contrast, 1.8),
        (grayscale, 0.7), (partial_occlusion, 0.2), (partial_crop, 0.2),
    ]
    for fn, param in corruptions:
        result = fn(image, param, seed=1)
        assert result.size == (80, 60), f"{fn.__name__} changed image size"


def test_grayscale_blend_factor_one_removes_all_color_variation():
    image = _sample_image()
    result = grayscale(image, blend_factor=1.0)
    array = np.asarray(result)
    # Fully desaturated: R, G, B channels should be identical per pixel.
    assert np.allclose(array[..., 0], array[..., 1])
    assert np.allclose(array[..., 1], array[..., 2])


def test_grayscale_blend_factor_zero_is_original_image():
    image = _sample_image()
    result = grayscale(image, blend_factor=0.0)
    assert np.array_equal(np.asarray(result), np.asarray(image.convert("RGB")))


def test_grayscale_clamps_out_of_range_blend_factor():
    image = _sample_image()
    over = grayscale(image, blend_factor=1.5)
    under = grayscale(image, blend_factor=-0.5)
    fully_gray = grayscale(image, blend_factor=1.0)
    original = np.asarray(image.convert("RGB"))
    assert np.array_equal(np.asarray(over), np.asarray(fully_gray))
    assert np.array_equal(np.asarray(under), original)


def test_low_contrast_and_high_contrast_move_in_opposite_directions():
    image = _sample_image()
    baseline_std = np.asarray(image.convert("L"), dtype=np.float64).std()
    low = np.asarray(low_contrast(image, 0.3).convert("L"), dtype=np.float64).std()
    high = np.asarray(high_contrast(image, 2.0).convert("L"), dtype=np.float64).std()
    assert low < baseline_std
    assert high > baseline_std


def test_apply_corruption_dispatches_new_corruption_types():
    image = _sample_image()
    for name, param in (("low_contrast", 0.5), ("high_contrast", 1.5), ("grayscale", 0.5)):
        result = apply_corruption(image, name, param, seed=0)
        assert result.size == image.size


def test_apply_corruption_rejects_unknown_name():
    import pytest

    with pytest.raises(ValueError):
        apply_corruption(_sample_image(), "not_a_real_corruption", 1.0)


def test_iter_corruption_configs_yields_new_corruption_types():
    robustness_cfg = {
        "corruptions": {
            "low_contrast": {"severities": [1, 2], "params": [0.7, 0.5]},
            "grayscale": {"severities": [1], "params": [0.4]},
        }
    }
    configs = list(iter_corruption_configs(robustness_cfg))
    names = {name for name, _, _ in configs}
    assert names == {"low_contrast", "grayscale"}
    assert len(configs) == 3


@pytest.mark.parametrize(
    "name,param",
    [
        ("gaussian_blur", 1.5), ("gaussian_noise", 0.1), ("low_resolution", 0.3), ("jpeg_compression", 20),
        ("low_brightness", 0.5), ("high_brightness", 1.6), ("low_contrast", 0.5), ("high_contrast", 1.8),
        ("grayscale", 0.7), ("partial_occlusion", 0.2), ("partial_crop", 0.2),
    ],
)
def test_corruption_is_deterministic_for_a_fixed_seed(name, param):
    """Every corruption must produce byte-identical output given the same
    seed -- required for a fair, reproducible robustness comparison across
    models (the same corrupted image must be shown to every model)."""
    image = _sample_image((48, 48))
    result_1 = apply_corruption(image, name, param, seed=7)
    result_2 = apply_corruption(image, name, param, seed=7)
    assert np.array_equal(np.asarray(result_1), np.asarray(result_2))


def test_corruption_with_randomness_differs_across_seeds():
    """Sanity check for the determinism test above: corruptions that use
    randomness (noise/occlusion/crop) must actually depend on the seed,
    otherwise the "same seed -> same output" test would be vacuous."""
    image = _sample_image((48, 48))
    result_a = apply_corruption(image, "gaussian_noise", 0.1, seed=1)
    result_b = apply_corruption(image, "gaussian_noise", 0.1, seed=2)
    assert not np.array_equal(np.asarray(result_a), np.asarray(result_b))


def _robustness_results_df():
    return pd.DataFrame([
        {"corruption": "clean", "severity": 0, "param": None, "age_mae": 5.0, "gender_accuracy": 0.95, "abstention_rate": 0.05},
        {"corruption": "gaussian_blur", "severity": 1, "param": 0.8, "age_mae": 6.0, "gender_accuracy": 0.90, "abstention_rate": 0.10},
        {"corruption": "gaussian_blur", "severity": 2, "param": 1.6, "age_mae": 8.0, "gender_accuracy": 0.80, "abstention_rate": 0.20},
    ])


def test_compute_degradation_adds_delta_and_pct_change_columns():
    df = compute_degradation(_robustness_results_df())
    clean_row = df[df["corruption"] == "clean"].iloc[0]
    assert clean_row["age_mae_delta"] == 0.0
    assert clean_row["age_mae_pct_change"] == 0.0

    blur_severity_2 = df[(df["corruption"] == "gaussian_blur") & (df["severity"] == 2)].iloc[0]
    assert blur_severity_2["age_mae_delta"] == pytest.approx(3.0)  # 8.0 - 5.0
    assert blur_severity_2["age_mae_pct_change"] == pytest.approx(60.0)  # 3.0 / 5.0 * 100
    assert blur_severity_2["gender_accuracy_delta"] == pytest.approx(-0.15)


def test_compute_degradation_raises_without_clean_baseline():
    df = _robustness_results_df()
    df = df[df["corruption"] != "clean"]
    with pytest.raises(ValueError):
        compute_degradation(df)


def test_build_robustness_diff_table_computes_direct_model_vs_model_difference():
    df_cnn = _robustness_results_df()
    df_resnet = _robustness_results_df().copy()
    df_resnet["age_mae"] = df_resnet["age_mae"] - 1.0  # ResNet uniformly 1 year better

    diff_table = build_robustness_diff_table({"simple_cnn": df_cnn, "custom_resnet18": df_resnet})
    assert len(diff_table) == 3
    row = diff_table[(diff_table["corruption"] == "gaussian_blur") & (diff_table["severity"] == 2)].iloc[0]
    assert row["simple_cnn_age_mae"] == pytest.approx(8.0)
    assert row["custom_resnet18_age_mae"] == pytest.approx(7.0)
    assert row["diff_age_mae_(custom_resnet18_minus_simple_cnn)"] == pytest.approx(-1.0)


def test_build_robustness_diff_table_requires_at_least_two_models():
    with pytest.raises(ValueError):
        build_robustness_diff_table({"only_one": _robustness_results_df()})
