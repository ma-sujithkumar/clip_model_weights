---
name: modernvbert
description: >-
  Provides details, architecture info, and CPU loading instructions for the 
  ModernVBERT/colmodernvbert and ModernVBERT/colmodernvbert-base models.
---

# ModernVBERT / ColModernVBERT Model Skill

This skill provides workspace-specific details on how to load, run, and deploy the `ModernVBERT` model family cached in this repository, optimized for CPU inference.

---

## 1. Model Overview

* **Base Model**: `ModernVBERT/colmodernvbert-base` (~1.01 GB)
  * A 250M-parameter bidirectional encoder-only vision-language model.
* **LoRA Adapter**: `ModernVBERT/colmodernvbert` (~31.1 MB)
  * Fine-tuned variant using late interaction optimized for **Visual Document Retrieval (VDR)**.
* **Architecture**: Encoder-only model featuring bidirectional attention, designed to be highly compact and efficient for edge/CPU inference.

---

## 2. Workspace File Structure

The Hugging Face model cache is fully downloaded and tracked in this workspace via Git LFS:

* **Base Model Cache**: `models/models--ModernVBERT--colmodernvbert-base/`
* **LoRA Adapter Cache**: `models/models--ModernVBERT--colmodernvbert/`
* **Download Scripts**:
  * [download_colmodernvbert_base.py](file:///home/sujithma/clip/download_colmodernvbert_base.py)
  * [download_colmodernvbert.py](file:///home/sujithma/clip/download_colmodernvbert.py)

---

## 3. How to Load and Deploy on CPU

To load the model on a CPU-only deployment environment directly from the local cache:

```python
import torch
from transformers import AutoProcessor, AutoModel
from peft import PeftModel

CACHE_DIR = "models"
BASE_MODEL_NAME = "ModernVBERT/colmodernvbert-base"
ADAPTER_MODEL_NAME = "ModernVBERT/colmodernvbert"

print("Loading base model on CPU...")
# Load the base ModernVBERT encoder
base_model = AutoModel.from_pretrained(
    BASE_MODEL_NAME,
    cache_dir=CACHE_DIR,
    device_map="cpu",
    trust_remote_code=True
)

print("Applying LoRA adapter...")
# Wrap the base model with the ColModernVBERT LoRA adapter
model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_MODEL_NAME,
    cache_dir=CACHE_DIR
)

print("Loading processor...")
# Load the processor (tokenizers and preprocessors)
processor = AutoProcessor.from_pretrained(
    ADAPTER_MODEL_NAME,
    cache_dir=CACHE_DIR,
    trust_remote_code=True
)

model.eval()
print("Model loaded successfully on CPU and ready for inference!")
```

---

## 4. Deploying/Running Queries

For document retrieval and late interaction embeddings extraction:

```python
# Prepare inputs (example for queries or documents)
inputs = processor(
    text=["Query text or document context here"],
    images=None,  # Pass PIL Images if doing visual document encoding
    return_tensors="pt"
)

# Move inputs to CPU
inputs = {k: v.to("cpu") if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

# Extract multi-vector embeddings
with torch.no_grad():
    outputs = model(**inputs)
    # The embeddings are typically extracted from the last hidden states
    embeddings = outputs.last_hidden_state
```
