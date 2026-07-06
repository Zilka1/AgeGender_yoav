"""Application state and FastAPI dependency wiring.

Holds a single in-process ``AppState`` with the loaded config, artifacts,
and predictor, so ``/admin/reload-models`` can reload everything without
restarting the process.
"""

from __future__ import annotations

import logging
import os

from src.inference.artifacts import load_all_artifacts
from src.inference.predictor import Predictor
from src.utils.config import CONFIG_DIR, load_config, resolve_device

logger = logging.getLogger(__name__)


class AppState:
    def __init__(self) -> None:
        self.config: dict = {}
        self.device: str = "cpu"
        self.predictor: Predictor | None = None

    def load(self) -> None:
        self.config = load_config(CONFIG_DIR / "api.yaml")
        api_config = self.config["api"]
        self.device = resolve_device(self.config.get("device", "auto"))

        # GENDER_LABEL_0 / GENDER_LABEL_1 env vars (see .env.example) take
        # priority over configs/api.yaml's gender_label_overrides when set,
        # letting a deployer rename the displayed dataset gender-label
        # classes (e.g. to match a specific dataset's own documented
        # convention) without editing YAML or retraining.
        env_overrides = [os.environ.get("GENDER_LABEL_0"), os.environ.get("GENDER_LABEL_1")]
        if any(env_overrides):
            base = api_config.get("gender_label_overrides") or [None, None]
            api_config["gender_label_overrides"] = [
                env_overrides[i] or (base[i] if i < len(base) else None) for i in range(2)
            ]

        artifacts = load_all_artifacts(api_config, self.device)
        self.predictor = Predictor(artifacts, api_config, self.device)
        if artifacts.warnings:
            for warning in artifacts.warnings:
                logger.warning(warning)


app_state = AppState()


def get_predictor() -> Predictor:
    if app_state.predictor is None:
        app_state.load()
    return app_state.predictor


def get_app_state() -> AppState:
    if app_state.predictor is None:
        app_state.load()
    return app_state
