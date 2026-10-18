"""Display rotation from container metadata: phones store frames sideways plus a rotation tag / display matrix."""

import json
import shutil
import subprocess


def _snap(cw: float) -> int:
    return int(round(cw / 90.0)) % 4 * 90


def _ffprobe(path: str) -> int | None:
    exe = shutil.which("ffprobe")
    if exe is None:
        return None
    out = subprocess.run([exe, "-v", "error", "-select_streams", "v:0", "-show_streams", "-of", "json", path],
                         capture_output=True, timeout=30, check=True,
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
    streams = json.loads(out).get("streams")
    if not streams:
        return None
    for sd in streams[0].get("side_data_list", []):
        if "rotation" in sd:  # display matrix, counter-clockwise degrees
            return _snap(-float(sd["rotation"]))
    return _snap(float(streams[0].get("tags", {}).get("rotate", 0)))  # legacy tag, clockwise degrees


def _pyav(path: str) -> int | None:
    import av

    with av.open(path) as c:
        s = c.streams.video[0]
        if "rotate" in s.metadata:
            return _snap(float(s.metadata["rotate"]))
        return _snap(-next(c.decode(s)).rotation)


def rotation(path: str) -> int | None:
    """Clockwise degrees (0, 90, 180, 270) that turn decoded frames upright, as players show them.
    None when neither ffprobe nor PyAV could read the file."""
    for probe in (_ffprobe, _pyav):
        try:
            r = probe(path)
        except Exception:  # tool missing, timeout, unreadable file
            continue
        if r is not None:
            return r
    return None
