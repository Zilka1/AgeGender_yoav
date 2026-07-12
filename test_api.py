import os
import sys
from PIL import Image

sys.path.append(os.getcwd())
from src.utils.config import load_config, CONFIG_DIR
from src.inference.artifacts import load_all_artifacts
from src.inference.predictor import Predictor

config = load_config(CONFIG_DIR / "api.yaml")
artifacts = load_all_artifacts(config["api"], device="cpu")
predictor = Predictor(artifacts, config["api"], "cpu")

try:
    img = Image.open("lena.jpg")
    result = predictor.predict(img, include_gradcam=True, include_knn=False)
    print("Success")
except Exception as e:
    import traceback
    traceback.print_exc()
