"""
Usage:
    python3 download_clip_weights.py
"""

import subprocess
import sys

MODEL_NAME = "openai/clip-vit-large-patch14"
CACHE_DIR = "models"

REQUIRED_PACKAGES = ["torch", "transformers", "huggingface_hub", "pillow"]


def install_dependencies():
    subprocess.check_call([sys.executable, "-m", "pip", "install", *REQUIRED_PACKAGES])


def download_weights():
    from transformers import CLIPModel, CLIPProcessor

    print(f"Downloading '{MODEL_NAME}' into ./{CACHE_DIR} ...")
    CLIPProcessor.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    CLIPModel.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    print("Done. Copy the './models' folder to the target machine's project directory.")


if __name__ == "__main__":
    install_dependencies()
    download_weights()

 
