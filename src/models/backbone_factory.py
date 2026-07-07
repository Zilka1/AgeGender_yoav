"""Backbone factory: builds the backbone named by ``model.backbone.name``.

Both supported backbones expose the same interface --
``embedding_dim`` (attribute), ``forward(x)``, ``forward_features(x)``
(returning ``layer1``-``layer4`` feature maps for Grad-CAM), and
``num_parameters()`` -- so callers (``MultiTaskFaceModel``, Grad-CAM,
progressive-freezing stage logic) do not need to know which one is
active. ``custom_resnet18`` (the project's main research backbone) is
the default; ``simple_cnn`` exists only as a controlled baseline (see
``configs/experiments.yaml: exp_0_simple_cnn_shared_adapters_learned_balance``).
"""

from __future__ import annotations

import torch.nn as nn

from src.models.custom_resnet import build_backbone as _build_custom_resnet18
from src.models.simple_cnn import build_simple_cnn as _build_simple_cnn

_BACKBONE_BUILDERS = {
    "custom_resnet18": _build_custom_resnet18,
    "simple_cnn": _build_simple_cnn,
}


def build_backbone(config: dict) -> nn.Module:
    """Build the backbone named by ``config["name"]`` (default ``"custom_resnet18"``)."""
    name = config.get("name", "custom_resnet18")
    if name not in _BACKBONE_BUILDERS:
        raise ValueError(f"Unknown backbone '{name}', expected one of {list(_BACKBONE_BUILDERS)}")
    return _BACKBONE_BUILDERS[name](config)
