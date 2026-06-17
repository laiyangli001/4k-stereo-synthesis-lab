import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "smoke" / "host_api_smoke.py"


def _run_smoke(*args: str) -> dict:
    proc = subprocess.run(
        [
            sys.executable,
            "-B",
            str(SCRIPT),
            "--device",
            "cpu",
            "--width",
            "64",
            "--height",
            "40",
            "--iters",
            "1",
            "--out",
            "-",
            *args,
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(proc.stdout)


def test_host_api_smoke_stereo_report_contract():
    report = _run_smoke("--preset", "cinema", "--output-format", "half_sbs")

    assert report["mode"] == "stereo"
    assert report["preset"] == "cinema"
    assert report["output_format"] == "half_sbs"
    assert report["screen_roll"] is None
    assert report["auto_depth"] is False
    assert report["rgb_shape"] == [1, 3, 40, 64]
    assert report["depth_shape"] == [1, 1, 40, 64]
    assert report["left_eye_shape"] == [1, 3, 40, 64]
    assert report["right_eye_shape"] == [1, 3, 40, 64]
    assert report["sbs_shape"] == [1, 3, 40, 64]
    assert report["mean_render_ms"] >= 0.0
    assert report["debug_info"]["backend"] == "quality_4k"


def test_host_api_smoke_openxr_report_contract():
    report = _run_smoke("--openxr", "--preset", "cinema", "--screen-roll", "0.25")

    assert report["mode"] == "openxr"
    assert report["preset"] == "cinema"
    assert report["output_format"] is None
    assert report["screen_roll"] == 0.25
    assert report["rgb_shape"] == [1, 3, 40, 64]
    assert report["depth_shape"] == [1, 1, 40, 64]
    assert report["left_eye_shape"] == [1, 3, 40, 64]
    assert report["right_eye_shape"] == [1, 3, 40, 64]
    assert report["sbs_shape"] is None
    assert report["mean_render_ms"] >= 0.0
    assert report["debug_info"]["backend"] == "openxr_roll_adaptive_grid_sample"
    assert report["debug_info"]["screen_roll"] == 0.25
