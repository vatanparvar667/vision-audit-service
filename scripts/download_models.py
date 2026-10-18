"""Download model weights once into ./models (after this the service needs no internet).

All models are Apache-2.0 licensed (commercial use allowed):
  - PekingU/rtdetr_v2_r50vd            people and everyday objects (COCO)
  - google/owlv2-base-patch16-ensemble open-vocabulary objects: banknotes, bags, documents, cups
  - Qwen/Qwen3-VL-4B-Instruct          yes/no verification of candidate events (loaded in 4-bit)
"""

import sys
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent.parent
MODELS = ["PekingU/rtdetr_v2_r50vd", "google/owlv2-base-patch16-ensemble", "Qwen/Qwen3-VL-4B-Instruct"]

if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "models"
    for repo in MODELS:
        path = snapshot_download(repo, local_dir=target / repo.split("/")[1])
        print("ok", repo, "->", path)
