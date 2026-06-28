import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "smoke" / "d2s_depth_runtime_smoke.py"


def test_d2s_depth_runtime_smoke_queue_contract():
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
            "--target-height",
            "32",
            "--out",
            "-",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(proc.stdout)

    assert report["mode"] == "d2s_depth_runtime"
    assert report["raw_shape"] == [40, 64, 4]
    assert report["frame_rgb_shape"] == [32, 50, 3]
    assert report["runtime_rgb_shape"] == [1, 3, 32, 50]
    assert report["runtime_rgb_dtype"] == "torch.float32"
    assert 0.0 <= report["runtime_rgb_min"] <= report["runtime_rgb_max"] <= 1.0
    assert report["depth_shape"] == [1, 1, 32, 50]
    assert report["depth_dtype"] == "torch.float32"
    assert report["depth_min"] == 0.0
    assert report["depth_max"] == 1.0
    assert report["capture_timestamp_type"] == "float"
    assert report["queue_contract"] == "(frame_rgb, depth, capture_start_time)"
    assert report["provider_load_count"] == 1
    assert report["provider_predict_count"] == 1
    assert report["provider_info"]["provider"] == "fake"
    assert report["real_provider"] is False
    assert report["runtime_backend"] == "pytorch_cuda"


def test_d2s_depth_runtime_smoke_accepts_backend_cli_without_real_provider():
    proc = subprocess.run(
        [
            sys.executable,
            "-B",
            str(SCRIPT),
            "--device",
            "cpu",
            "--width",
            "16",
            "--height",
            "10",
            "--target-height",
            "8",
            "--backend",
            "onnx_cuda",
            "--out",
            "-",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(proc.stdout)

    assert report["real_provider"] is False
    assert report["runtime_backend"] == "onnx_cuda"
    assert report["provider_info"]["provider"] == "fake"
