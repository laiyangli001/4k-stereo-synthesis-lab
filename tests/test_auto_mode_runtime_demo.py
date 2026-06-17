import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "smoke" / "auto_mode_runtime_demo.py"


def _run_demo(*args: str) -> dict:
    proc = subprocess.run(
        [sys.executable, "-B", str(SCRIPT), "--out", "-", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(proc.stdout)


def test_manual_preset_does_not_start_auto_detection():
    report = _run_demo("--selected-preset", "game_low_latency")

    assert report["selected_preset"] == "game_low_latency"
    assert report["auto_detection_started"] is False
    assert report["timeline"] == []
    assert report["config"]["hole_fill"] == "fast"


def test_auto_preset_runs_debounced_runtime_timeline():
    report = _run_demo("--selected-preset", "auto", "--dt", "0.25")
    timeline = report["timeline"]

    assert report["auto_detection_started"] is True
    assert len(timeline) == 14
    assert timeline[0]["input_label"] == "video"
    assert timeline[3]["input_label"] == "game"
    assert any(row["resolved_preset"] == "game_low_latency" for row in timeline)
    assert timeline[-1]["input_label"] == "still"
    assert timeline[-1]["resolved_preset"] == "game_low_latency"
    assert "holding game_low_latency" in timeline[-1]["reason"]
