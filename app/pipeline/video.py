"""Video decoding: sequential read (seeking is unreliable on some .mov files) with uniform sampling.

Skipped frames are only grabbed (no colour conversion). Kept frames are downscaled to MAX_SIDE and turned upright
from the rotation metadata; sample times come from the decoder timestamps."""

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

from app.pipeline.videoinfo import rotation

log = logging.getLogger("video")


@dataclass
class Sample:
    idx: int            # index among sampled frames
    frame_no: int       # index among decoded frames
    t: float            # seconds from the start
    jpeg: bytes         # upright frame (longer side <= MAX_SIDE), kept compressed to save RAM
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


SMALL_W = 232     # change-analysis resolution (half of the dataset width)
MAX_SIDE = 960    # longer side of kept frames (RT-DETR works at 640, OWLv2 at 960); the dataset is 464x848
MAX_FAILS = 300   # consecutive failed grabs = end of stream (a damaged stretch fails a few dozen times, then decodes)
ROTATE = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def _fit(frame: np.ndarray, rot: int) -> np.ndarray:
    h, w = frame.shape[:2]
    k = MAX_SIDE / max(h, w)
    if k < 1:
        frame = cv2.resize(frame, (max(1, round(w * k)), max(1, round(h * k))), interpolation=cv2.INTER_AREA)
    return cv2.rotate(frame, ROTATE[rot]) if rot else frame


def read_video(path: str, sample_fps: float = 4.0, progress=None) -> VideoClip:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError("cannot open video (unsupported or corrupted file)")
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)  # rotation is applied in _fit, after downscaling
    rot = rotation(str(path))
    if rot is None:  # no ffprobe / PyAV: OpenCV's FFmpeg backend reads the same metadata (clockwise degrees)
        rot = int(round(cap.get(cv2.CAP_PROP_ORIENTATION_META) / 90.0)) % 4 * 90
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if not 1.0 <= fps <= 240.0:
        fps = 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    step = max(1, int(round(fps / sample_fps)))
    clip = None
    frame_no = fails = skipped = stale = 0
    hi = float("-inf")  # running max of the decoder timestamps
    while fails < MAX_FAILS:
        if not cap.grab():  # end of stream or an undecodable packet; decoding resumes after damage
            fails += 1
            continue
        skipped, fails = skipped + fails, 0
        pts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if pts <= hi:  # repeated or stale timestamp (seen right after damaged packets)
            stale += 1
        hi = max(hi, pts)
        if frame_no % step == 0:
            ok, frame = cap.retrieve()
            if ok:
                frame = _fit(frame, rot)
                if clip is None:
                    h, w = frame.shape[:2]
                    clip = VideoClip(fps=fps, width=w, height=h, duration=0.0, sample_fps=fps / step)
                elif frame.shape[:2] != (clip.height, clip.width):  # resolution change mid-stream
                    frame = cv2.resize(frame, (clip.width, clip.height), interpolation=cv2.INTER_AREA)
                small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                                   (SMALL_W, int(round(clip.height * SMALL_W / clip.width))),
                                   interpolation=cv2.INTER_AREA)
                ok_enc, enc = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                clip.samples.append(Sample(len(clip.samples), frame_no, hi, enc.tobytes(), small))
                if progress and total and len(clip.samples) % 20 == 0:
                    progress(min(1.0, frame_no / total))
        frame_no += 1
    cap.release()
    if clip is None or not clip.samples:
        raise ValueError("video has no decodable frames")
    if hi > 0 and stale <= 2 + 0.01 * frame_no:  # decoder timestamps: correct for VFR, trimmed and damaged files
        for s in clip.samples:
            s.t = max(0.0, s.t)  # a frame with PTS 0 is reported with its DTS, which can be slightly negative
        clip.duration = hi + 1.0 / fps
    else:  # timestamps missing, constant or jumping back: nominal frame rate
        for s in clip.samples:
            s.t = s.frame_no / fps
        clip.duration = frame_no / fps
    if skipped:
        log.warning("%s: %d failed reads skipped, %d frames decoded", path, skipped, frame_no)
    if rot:
        log.info("%s: turned %d degrees clockwise to upright, %dx%d", path, rot, clip.width, clip.height)
    return clip
