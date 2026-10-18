"""Save Qwen3-VL-4B-Instruct pre-quantized to 4 bit (bitsandbytes NF4) next to the original weights.

The service then loads ~3 GB of 4-bit weights instead of reading the 8.9 GB bf16 checkpoint into RAM and
quantizing it on every start (peak RAM drops from ~9 GB to ~3-4 GB on a 16 GB laptop). Needs a CUDA GPU.
Usage: python scripts/quantize_vlm.py [MODELS_DIR]
"""

import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "models"
src, dst = root / "Qwen3-VL-4B-Instruct", root / "Qwen3-VL-4B-Instruct-bnb4"
t0 = time.time()
model = AutoModelForImageTextToText.from_pretrained(
    src, dtype=torch.bfloat16, device_map="cuda",
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                           bnb_4bit_compute_dtype=torch.bfloat16))
model.save_pretrained(dst)
AutoProcessor.from_pretrained(src).save_pretrained(dst)
size = sum(f.stat().st_size for f in dst.glob("*.safetensors")) / 1e9
print(f"saved {dst} ({size:.1f} GB) in {time.time() - t0:.0f}s")
