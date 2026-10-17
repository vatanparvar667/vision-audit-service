"""Web UI behaviour: tests/frontend_harness.js runs app/static/app.js in Node; skipped when Node is not installed."""

import shutil
import subprocess
from pathlib import Path

import pytest

HARNESS = Path(__file__).with_name("frontend_harness.js")
NODE = shutil.which("node")

SCENARIOS = [
    "fmtTimeRollsOver",
    "resetAbortsUpload",
    "abandonedUploadCannotHijackNextFile",
    "resetStopsPolling",
    "pollHttpErrorsSurface",
    "pollNetworkErrorsSurface",
    "pollRecoversAndResetsRetries",
    "headerDuration",
    "undecodableVideoStillGetsTimeline",
    "durationFromResult",
    "noStaleMarksAfterReset",
    "mobileTimecodeKeepsVideoInView",
    "filePickerWorksFromKeyboard",
]


@pytest.mark.skipif(NODE is None, reason="node is not installed")
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_web_ui(scenario):
    r = subprocess.run([NODE, str(HARNESS), scenario], capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
