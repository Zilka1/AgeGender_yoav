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

# Create a dummy image
predictor.enable_face_detection = False
img = Image.new('RGB', (200, 200), color='white')

try:
    print("Testing with dummy image...")
    result = predictor.predict(img, include_gradcam=True, include_knn=True)
    print(f"Face detected: {result.face_detected}")
except Exception as e:
    import traceback
    traceback.print_exc()

# Let's try with a real image from the data dir if possible
try:
    demo_img_path = "data/raw/utkface/100_1_0_20170110183726390.jpg.chip.jpg"
    if os.path.exists(demo_img_path):
        print("Testing with real image...")
        img = Image.open(demo_img_path)
        result = predictor.predict(img, include_gradcam=True, include_knn=True)
        print("Success with real image")
    else:
        print("No real image to test")
except Exception as e:
    import traceback
    traceback.print_exc()
