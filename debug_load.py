import os
import sys
sys.path.append(os.getcwd())
from src.utils.config import load_config, CONFIG_DIR
from src.inference.artifacts import load_all_artifacts
import logging

logging.basicConfig(level=logging.DEBUG)

config = load_config(CONFIG_DIR / "api.yaml")
try:
    artifacts = load_all_artifacts(config["api"], device="cpu")
    print(f"Warnings: {artifacts.warnings}")
    print(f"Model loaded: {artifacts.model is not None}")
except Exception as e:
    import traceback
    traceback.print_exc()
