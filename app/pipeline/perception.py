"""Frame-level detectors: RT-DETRv2 (COCO classes) and OWLv2 (open-vocabulary cash-desk objects).

Both models are Apache-2.0 licensed and run locally from MODELS_DIR (no network access at runtime).
"""

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from app.pipeline.video import VideoClip

COCO_KEEP = {
    "person", "cup", "bottle", "wine glass", "cell phone", "handbag", "backpack", "suitcase",
    "laptop", "tv", "book", "chair", "dining table", "bowl", "sandwich", "donut", "cake", "banana",
    "apple", "orange", "pizza", "hot dog", "umbrella", "scissors", "keyboard",
}

# Open-vocabulary prompts; several phrasings per concept, merged after detection.
OWL_QUERIES = {
    "cash": ["banknotes", "a stack of paper money", "a bundle of cash"],
    "bag": ["a handbag", "a fabric bag", "a cash collection bag"],
    "document": ["a document folder", "a booklet", "papers on a desk"],
    "cup": ["a coffee cup", "a paper cup", "a drink bottle"],
    "clothes": ["a jacket", "a coat lying on a desk"],
    "phone": ["a mobile phone", "a smartphone in a hand"],
    "desk": ["a desk"],
    "safe": ["a metal safe"],
    "machine": ["a banknote counting machine"],
    "door": ["a glass door"],
}


# concepts re-checked on the enlarged desk crop (small objects the full-frame pass misses)
CROP_CONCEPTS = ["cash", "bag", "document", "cup", "clothes", "phone"]


def merge_owl(a: dict, b: dict) -> dict:
    """Union of two OWLv2 passes per sample; overlapping boxes of one concept keep the higher score."""
    return {i: nms_by_label(a.get(i, []) + b.get(i, []), iou_thr=0.5) for i in set(a) | set(b)}


def desk_crop_box(desk, width: int, height: int):
    """Desk region plus room above it (cash held over the desk), slightly wider than the desk."""
    if desk is None:
        return None
    x0, y0, x1, y1 = desk
    w, h = x1 - x0, y1 - y0
    return (max(0.0, x0 - 0.08 * w), max(0.0, y0 - 0.45 * h), min(float(width), x1 + 0.08 * w), min(float(height), y1))


@dataclass
class Det:
    label: str
    score: float
    box: tuple[float, float, float, float]  # x0, y0, x1, y1 in full-resolution pixels

    @property
    def w(self) -> float:
        return self.box[2] - self.box[0]

    @property
    def h(self) -> float:
        return self.box[3] - self.box[1]

    @property
    def cx(self) -> float:
        return (self.box[0] + self.box[2]) / 2

    @property
    def cy(self) -> float:
        return (self.box[1] + self.box[3]) / 2


class Perception:
    def __init__(self, models_dir, device: str = "cuda"):
        from transformers import (AutoImageProcessor, AutoModelForObjectDetection, Owlv2ForObjectDetection,
                                  Owlv2Processor)

        self.device = device if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        det_dir = models_dir / "rtdetr_v2_r50vd"
        self.det_proc = AutoImageProcessor.from_pretrained(det_dir)
        self.det = AutoModelForObjectDetection.from_pretrained(det_dir).to(self.device, self.dtype).eval()
        self.id2label = self.det.config.id2label
        owl_dir = models_dir / "owlv2-base-patch16-ensemble"
        self.owl_proc = Owlv2Processor.from_pretrained(owl_dir)
        self.owl = Owlv2ForObjectDetection.from_pretrained(owl_dir).to(self.device, self.dtype).eval()
        self._owl_texts = [q for qs in OWL_QUERIES.values() for q in qs]
        self._owl_concept = [c for c, qs in OWL_QUERIES.items() for _ in qs]
        self._owl_text_inputs = self.owl_proc(text=[self._owl_texts], return_tensors="pt").to(self.device)
        # second pass on an enlarged crop of the desk: small objects only
        self._crop_texts = [q for c in CROP_CONCEPTS for q in OWL_QUERIES[c]]
        self._crop_concept = [c for c in CROP_CONCEPTS for _ in OWL_QUERIES[c]]
        self._crop_text_inputs = self.owl_proc(text=[self._crop_texts], return_tensors="pt").to(self.device)
        # the same queries for a batch of two images (the frame and its mirror image)
        self._owl_text_inputs2 = self.owl_proc(text=[self._owl_texts] * 2, return_tensors="pt").to(self.device)
        self._crop_text_inputs2 = self.owl_proc(text=[self._crop_texts] * 2, return_tensors="pt").to(self.device)

    @torch.inference_mode()
    def detect_coco(self, clip: VideoClip, threshold: float = 0.3, batch: int = 8, progress=None) -> list[list[Det]]:
        H, W = clip.height, clip.width
        out_all: list[list[Det]] = []
        n = len(clip.samples)
        for b in range(0, n, batch):
            imgs = [clip.rgb(i) for i in range(b, min(n, b + batch))]
            inp = self.det_proc(images=imgs, return_tensors="pt").to(self.device)
            out = self.det(pixel_values=inp["pixel_values"].to(self.dtype))
            out.logits = out.logits.float()
            out.pred_boxes = out.pred_boxes.float()
            res = self.det_proc.post_process_object_detection(out, threshold=threshold, target_sizes=[(H, W)] * len(imgs))
            for r in res:
                dets = []
                for s, l, bx in zip(r["scores"].tolist(), r["labels"].tolist(), r["boxes"].tolist()):
                    name = self.id2label[l]
                    if name in COCO_KEEP:
                        dets.append(Det(name, float(s), tuple(float(v) for v in bx)))
                out_all.append(dets)
            if progress:
                progress(min(1.0, (b + batch) / n))
        return out_all

    @torch.inference_mode()
    def detect_owl(self, clip: VideoClip, indices: list[int], threshold: float = 0.12, progress=None,
                   crop: tuple | None = None, flip_tta: bool = True) -> dict[int, list[Det]]:
        """OWLv2 on selected samples. Returns concept-level detections (merged across phrasings, NMS by concept).
        With `crop` (full-frame box) only that region is analysed, enlarged to the model input, and only the
        small-object concepts are queried; boxes are returned in full-frame coordinates.
        With `flip_tta` the mirror image is analysed in the same batch and the better detection is kept: scores of
        small objects such as banknote bundles otherwise change a lot between a scene and its mirror view."""
        H, W = clip.height, clip.width
        if crop is not None:
            cx0, cy0 = max(0, int(crop[0])), max(0, int(crop[1]))
            cx1, cy1 = min(W, int(crop[2])), min(H, int(crop[3]))
            texts, concepts = (self._crop_text_inputs2 if flip_tta else self._crop_text_inputs), self._crop_concept
        else:
            cx0, cy0, cx1, cy1 = 0, 0, W, H
            texts, concepts = (self._owl_text_inputs2 if flip_tta else self._owl_text_inputs), self._owl_concept
        wc = cx1 - cx0
        S = max(cx1 - cx0, cy1 - cy0)  # OWLv2 pads to a square; boxes come back in padded coordinates
        result: dict[int, list[Det]] = {}
        for k, i in enumerate(indices):
            frame = clip.rgb(i)[cy0:cy1, cx0:cx1]
            imgs = [Image.fromarray(frame)] + ([Image.fromarray(np.ascontiguousarray(frame[:, ::-1]))] if flip_tta else [])
            px = self.owl_proc(images=imgs, return_tensors="pt")["pixel_values"].to(self.device, self.dtype)
            out = self.owl(input_ids=texts["input_ids"], attention_mask=texts["attention_mask"], pixel_values=px)
            out.logits = out.logits.float()
            out.pred_boxes = out.pred_boxes.float()
            res = self.owl_proc.post_process_grounded_object_detection(
                out, threshold=threshold, target_sizes=torch.tensor([[S, S]] * len(imgs), device=self.device))
            dets = []
            for flipped, r in enumerate(res):
                for sc, lb, bx in zip(r["scores"].tolist(), r["labels"].tolist(), r["boxes"].tolist()):
                    bx0, by0, bx1, by1 = bx
                    if flipped:  # back to the original orientation (the mirror image is padded on the right too)
                        bx0, bx1 = wc - bx1, wc - bx0
                    x0, y0, x1, y1 = bx0 + cx0, by0 + cy0, bx1 + cx0, by1 + cy0
                    box = (max(float(cx0), x0), max(float(cy0), y0), min(float(cx1), x1), min(float(cy1), y1))
                    if box[2] - box[0] < 2 or box[3] - box[1] < 2:
                        continue
                    dets.append(Det(concepts[lb], float(sc), box))
            result[i] = nms_by_label(dets, iou_thr=0.5)
            if progress:
                progress((k + 1) / len(indices))
        return result


def iou(a, b) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    if inter <= 0:
        return 0.0
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def overlap_frac(a, b) -> float:
    """Fraction of box a covered by box b."""
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area = max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))
    return inter / area


def nms_by_label(dets: list[Det], iou_thr: float = 0.5) -> list[Det]:
    keep: list[Det] = []
    for d in sorted(dets, key=lambda d: -d.score):
        if all(k.label != d.label or iou(k.box, d.box) < iou_thr for k in keep):
            keep.append(d)
    return keep


def boxes_array(dets: list[Det]) -> np.ndarray:
    return np.array([d.box for d in dets], dtype=np.float32).reshape(-1, 4)
