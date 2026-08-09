"""
SigLIP Multimodal Inference Example.

Usage:
    python3 inference_siglip.py
"""

import torch
from PIL import Image
import requests
from transformers import SiglipModel, SiglipProcessor

MODEL_PATH = "models/google/siglip-so400m-patch14-384"

def run_inference():
    print(f"Loading local SigLIP model from '{MODEL_PATH}'...")
    model = SiglipModel.from_pretrained(MODEL_PATH)
    processor = SiglipProcessor.from_pretrained(MODEL_PATH)

    # Put model in evaluation mode and move to CPU (or GPU if available)
    # Default to CPU as requested (and to avoid VRAM OOM on smaller GPUs)
    device = "cpu"
    model = model.to(device)
    model.eval()
    print(f"Model loaded successfully on {device.upper()}.")

    # Load a sample image (e.g. two cats)
    image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    print(f"Fetching sample image from {image_url}...")
    try:
        image = Image.open(requests.get(image_url, stream=True).raw)
    except Exception as e:
        print(f"Failed to fetch image from URL: {e}")
        print("Creating a dummy image for testing instead...")
        image = Image.new("RGB", (384, 384), color="blue")

    # Define candidate text descriptions to compare against the image
    texts = [
        "a photo of two cats lying on a couch",
        "a photo of two dogs playing in the yard",
        "a chart or document page showing statistical tables"
    ]

    print("\nProcessing text and image inputs...")
    # NOTE: padding="max_length" is recommended as SigLIP was trained with it
    inputs = processor(
        text=texts,
        images=image,
        padding="max_length",
        return_tensors="pt"
    ).to(device)

    print("Running forward pass...")
    with torch.no_grad():
        outputs = model(**inputs)

    # 1. Getting similarity probabilities
    # SigLIP uses sigmoid activation for probability calibration rather than softmax
    logits_per_image = outputs.logits_per_image
    probs = torch.sigmoid(logits_per_image).cpu().numpy()[0]

    print("\n--- Zero-Shot Classification Results ---")
    for text, prob in zip(texts, probs):
        print(f"Probability: {prob * 100:.2f}% | Label: '{text}'")

    # 2. Getting raw embeddings
    # This demonstrates how both images and text map to the same shared embedding space
    image_features = outputs.image_embeds  # shape: [num_images, hidden_dim]
    text_features = outputs.text_embeds    # shape: [num_texts, hidden_dim]

    print(f"\n--- Multimodal Shared Embedding Space Details ---")
    print(f"Image Embedding shape: {list(image_features.shape)}")
    print(f"Text Embedding shape : {list(text_features.shape)}")
    print(f"Shared space size    : {image_features.shape[-1]} dimensions")

if __name__ == "__main__":
    run_inference()
