"""Local vision-language model (Qwen3-VL-4B-Instruct, Apache-2.0) used as a yes/no verifier.

Only the probability of the first answer token is used ("Yes" vs "No"), so one call costs a single
forward pass. The model never sees anything outside this machine.
"""

import json
import logging
import threading
from pathlib import Path

import numpy as np
import torch
from PIL import Image

log = logging.getLogger("vlm")


class YesNoVLM:
    def __init__(self, model_dir, device: str = "cuda"):
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

        self.device = device if torch.cuda.is_available() else "cpu"
        self.proc = AutoProcessor.from_pretrained(model_dir)
        kw = {"dtype": torch.bfloat16}
        prequantized = "quantization_config" in json.loads((Path(model_dir) / "config.json").read_text(encoding="utf-8"))
        if self.device.startswith("cuda"):
            kw["device_map"] = self.device
            if not prequantized:  # quantize on load; a pre-quantized folder carries its own 4-bit config
                kw["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
        self.model = AutoModelForImageTextToText.from_pretrained(model_dir, **kw).eval()
        tok = self.proc.tokenizer
        self.yes_ids = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ("Yes", "yes", " Yes", " yes")})
        self.no_ids = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ("No", "no", " No", " no")})
        self.lock = threading.Lock()

    @torch.inference_mode()
    def yes_prob(self, image: np.ndarray | Image.Image, question: str) -> float:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        else:
            image = image.copy()
        image.thumbnail((1280, 1280))  # large camera frames would explode the number of image tokens
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question.strip() + "\nAnswer with one word: Yes or No."},
        ]}]
        with self.lock:
            inputs = self.proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                                   return_dict=True, return_tensors="pt").to(self.model.device)
            logits = self.model(**inputs, logits_to_keep=1, use_cache=False).logits[0, -1].float()
        probs = torch.softmax(logits, dim=-1)
        y = float(probs[self.yes_ids].sum())
        n = float(probs[self.no_ids].sum())
        return y / (y + n) if y + n > 0 else 0.5


def crop(image: np.ndarray, box, pad: float = 0.15, min_size: int = 96) -> np.ndarray:
    """Crop with relative padding; small crops are enlarged so the VLM sees enough detail."""
    H, W = image.shape[:2]
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    x0, x1 = max(0, int(x0 - pad * w)), min(W, int(x1 + pad * w))
    y0, y1 = max(0, int(y0 - pad * h)), min(H, int(y1 + pad * h))
    out = image[y0:y1, x0:x1]
    if out.size == 0:
        return image
    short = min(out.shape[:2])
    if short < min_size:
        import cv2

        k = min_size / short
        out = cv2.resize(out, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)
    return out
