"""Video decoding on synthetic clips (CPU only): rotation metadata, downscaling, damaged files, decoder timestamps."""

import importlib.util
import shutil
import struct
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.pipeline import video, videoinfo
from app.pipeline.video import SMALL_W, read_video


def upright(h=64, w=96):
    """Four coloured quadrants: any rotation of the picture is easy to tell apart."""
    img = np.zeros((h, w, 3), np.uint8)
    img[:h // 2, :w // 2] = (0, 0, 255)
    img[:h // 2, w // 2:] = (0, 255, 0)
    img[h // 2:, :w // 2] = (255, 0, 0)
    img[h // 2:, w // 2:] = 255
    return img


def write(path: Path, frames, fps=30.0) -> str:
    h, w = frames[0].shape[:2]
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    assert out.isOpened()
    for f in frames:
        out.write(f)
    out.release()
    return str(path)


def close(a, b) -> bool:
    return a.shape == b.shape and np.abs(a.astype(int) - b.astype(int)).mean() < 12


@pytest.fixture
def sideways(tmp_path) -> str:
    """Phone-style file: frames stored turned 90 degrees counter-clockwise, the display matrix turns them back."""
    frame = cv2.rotate(upright(), cv2.ROTATE_90_COUNTERCLOCKWISE)
    path = Path(write(tmp_path / "phone.mp4", [frame] * 30))
    data = bytearray(path.read_bytes())
    i = data.find(b"tkhd")
    assert data[i + 4] == 0  # version 0 box: the matrix starts 44 bytes after the type
    data[i + 44:i + 80] = struct.pack(">9i", 0, 1 << 16, 0, -(1 << 16), 0, 0, frame.shape[0] << 16, 0, 1 << 30)
    path.write_bytes(data)
    return str(path)


@pytest.mark.parametrize("source", ["probe", "opencv"])
def test_rotation_metadata_is_applied(sideways, monkeypatch, source):
    if source == "opencv":  # no ffprobe / PyAV, as in the Docker image
        monkeypatch.setattr(video, "rotation", lambda path: None)
    clip = read_video(sideways)
    assert (clip.width, clip.height) == (96, 64)
    assert close(clip.bgr(0), upright())
    assert clip.samples[0].small.shape == (int(round(64 * SMALL_W / 96)), SMALL_W)


@pytest.mark.parametrize("probe", [
    pytest.param(videoinfo._ffprobe, marks=pytest.mark.skipif(not shutil.which("ffprobe"), reason="no ffprobe")),
    pytest.param(videoinfo._pyav, marks=pytest.mark.skipif(not importlib.util.find_spec("av"), reason="no PyAV")),
])
def test_rotation_probes(sideways, tmp_path, probe):
    assert probe(sideways) == 90
    assert probe(write(tmp_path / "plain.mp4", [upright()] * 5)) == 0


def test_large_frames_are_downscaled(tmp_path):
    big = cv2.resize(upright(), (1280, 720), interpolation=cv2.INTER_NEAREST)
    clip = read_video(write(tmp_path / "hd.mp4", [big] * 10))
    assert (clip.width, clip.height) == (960, 540)
    assert all(clip.bgr(i).shape == (540, 960, 3) for i in range(len(clip.samples)))
    assert clip.samples[0].small.shape == (int(round(540 * SMALL_W / 960)), SMALL_W)
    assert close(clip.bgr(0), cv2.resize(upright(), (960, 540), interpolation=cv2.INTER_NEAREST))


def test_decoding_continues_after_damaged_packets(tmp_path):
    path = Path(write(tmp_path / "clip.mp4", [np.roll(upright(), 2 * i, axis=1) for i in range(90)]))  # 3 s
    data = bytearray(path.read_bytes())
    m = data.find(b"mdat")
    size = struct.unpack(">I", data[m - 4:m])[0]
    data[m + size // 2:m + size // 2 + size // 5] = bytes(size // 5)
    path.write_bytes(data)
    clip = read_video(str(path))
    t = [s.t for s in clip.samples]
    assert t == sorted(t)
    assert t[-1] > 2.5 and clip.duration == pytest.approx(3.0, abs=0.05)  # used to stop at the damage (1.5 s)
    assert max(np.diff(t)) > 0.5  # the lost stretch stays a gap instead of shifting later times


class FakeCapture:
    """Scripted capture: each entry is the timestamp (s) of a decoded frame, or None for a failed grab."""

    def __init__(self, script):
        self.script, self.pos, self.retrieved = list(script), 0.0, 0

    def isOpened(self):
        return True

    def set(self, prop, value):
        return False

    def get(self, prop):
        return {cv2.CAP_PROP_FPS: 30.0, cv2.CAP_PROP_POS_MSEC: self.pos * 1000.0}.get(prop, 0.0)

    def grab(self):
        t = self.script.pop(0) if self.script else None
        if t is None:
            return False
        self.pos = t
        return True

    def retrieve(self):
        self.retrieved += 1
        return True, np.zeros((48, 64, 3), np.uint8)

    def release(self):
        pass


@pytest.fixture
def fake(monkeypatch):
    def install(script):
        cap = FakeCapture(script)
        monkeypatch.setattr(video.cv2, "VideoCapture", lambda path: cap)
        monkeypatch.setattr(video, "rotation", lambda path: 0)
        return cap
    return install


def test_failed_reads_are_skipped_and_decoder_times_kept(fake):
    times = [i / 30 for i in range(60)]
    times[0] = -2 / 30  # OpenCV reports the first frame's DTS when its PTS is 0
    # frames 20-29 undecodable; the first frame after them repeats an old timestamp, as FFmpeg does
    cap = fake(times[:20] + [None] * 10 + [times[19]] + times[31:])
    clip = read_video("x.mov")
    assert cap.retrieved == len(clip.samples)  # skipped frames are only grabbed
    assert [s.t for s in clip.samples] == pytest.approx([0, 8 / 30, 16 / 30, 34 / 30, 42 / 30, 50 / 30, 58 / 30])
    assert clip.duration == pytest.approx(2.0)


def test_missing_timestamps_fall_back_to_frame_rate(fake):
    fake([0.0] * 40)
    clip = read_video("x.mov")
    assert [s.t for s in clip.samples] == pytest.approx([s.frame_no / 30 for s in clip.samples])
    assert clip.duration == pytest.approx(40 / 30)


def test_undecodable_video_is_a_clear_error(fake):
    fake([None] * 1000)
    with pytest.raises(ValueError, match="no decodable frames"):
        read_video("x.mov")


def test_not_a_video_is_a_clear_error(tmp_path):
    junk = tmp_path / "junk.mov"
    junk.write_bytes(b"\x00" * 4096)
    with pytest.raises(ValueError):
        read_video(str(junk))
