"""
Usage:
    python3 download_siglip_so400m.py
"""

import os
import shutil
from transformers import SiglipModel, SiglipProcessor

MODEL_NAME = "google/siglip-so400m-patch14-384"
CACHE_DIR = "models_cache_temp"
TARGET_DIR = os.path.join("models", "google", "siglip-so400m-patch14-384")

def download_and_shard():
    print(f"Downloading '{MODEL_NAME}' into temporary cache '{CACHE_DIR}'...")
    model = SiglipModel.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    processor = SiglipProcessor.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)

    print(f"Saving sharded model (max_shard_size='1GB') to target directory '{TARGET_DIR}'...")
    model.save_pretrained(TARGET_DIR, max_shard_size="1GB")
    processor.save_pretrained(TARGET_DIR)

    print(f"Cleaning up temporary cache directory '{CACHE_DIR}'...")
    if os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)

    print("Done. Sharded model files have been successfully saved.")

if __name__ == "__main__":
    download_and_shard()
