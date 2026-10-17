"""Video decoding: sequential read (seeking is unreliable on some .mov files) with uniform sampling."""

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class Sample:
    idx: int            # index among sampled frames
    frame_no: int       # index in the source video
    t: float            # seconds from the start
    jpeg: bytes         # full-resolution frame, kept compressed to save RAM
    small: np.ndarray   # downscaled grayscale for change analysis


@dataclass
class VideoClip:
    fps: float
    width: int
    height: int
    duration: float
    sample_fps: float
    samples: list[Sample] = field(default_factory=list)

    def rgb(self, i: int) -> np.ndarray:
        bgr = cv2.imdecode(np.frombuffer(self.samples[i].jpeg, np.uint8), cv2.IMREAD_COLOR)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def bgr(self, i: int) -> np.ndarray:
        return cv2.imdecode(np.frombuffer(self.samples[i].jpeg, np.uint8), cv2.IMREAD_COLOR)

    def nearest(self, t: float) -> int:
        return int(np.argmin([abs(s.t - t) for s in self.samples]))


SMALL_W = 232  # change-analysis resolution (half of the dataset width)


def read_video(path: str, sample_fps: float = 4.0, progress=None) -> VideoClip:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError("cannot open video (unsupported or corrupted file)")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if not 1.0 <= fps <= 240.0:
        fps = 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    step = max(1, int(round(fps / sample_fps)))
    clip = None
    frame_no = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_no % step == 0:
            h, w = frame.shape[:2]
            if clip is None:
                clip = VideoClip(fps=fps, width=w, height=h, duration=0.0, sample_fps=fps / step)
            small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (SMALL_W, int(round(h * SMALL_W / w))),
                               interpolation=cv2.INTER_AREA)
            ok_enc, enc = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            clip.samples.append(Sample(len(clip.samples), frame_no, frame_no / fps, enc.tobytes(), small))
            if progress and total and len(clip.samples) % 20 == 0:
                progress(frame_no / total)
        frame_no += 1
    cap.release()
    if clip is None or not clip.samples:
        raise ValueError("video has no decodable frames")
    clip.duration = frame_no / fps
    return clip
