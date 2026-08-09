"""
Usage:
    python3 download_colmodernvbert_base.py
"""

from huggingface_hub import snapshot_download

MODEL_NAME = "ModernVBERT/colmodernvbert-base"
CACHE_DIR = "models"

def download_weights():
    print(f"Downloading '{MODEL_NAME}' into ./{CACHE_DIR} ...")
    snapshot_download(repo_id=MODEL_NAME, cache_dir=CACHE_DIR)
    print("Done. Model files have been downloaded.")

if __name__ == "__main__":
    download_weights()
