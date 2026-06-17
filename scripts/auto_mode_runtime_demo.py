from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _signal_sequence():
    from stereo_lab import AutoModeSignals

    video = AutoModeSignals(
        gpu_3d_util=0.06,
        gpu_video_decode_util=0.34,
        input_activity=0.03,
        idle_seconds=8.0,
        audio_active=True,
        fullscreen=True,
        foreground_process="BrowserVideo.exe",
        frame_motion_score=0.12,
        target_fps=60.0,
    )
    game = AutoModeSignals(
        gpu_3d_util=0.76,
        gpu_video_decode_util=0.02,
        input_activity=0.88,
        idle_seconds=0.1,
        audio_active=True,
        fullscreen=True,
        foreground_process="Unknown3DApp.exe",
        frame_motion_score=0.45,
        latency_pressure=0.8,
        target_fps=120.0,
    )
    still = AutoModeSignals(
        gpu_3d_util=0.02,
        gpu_video_decode_util=0.0,
        input_activity=0.0,
        idle_seconds=40.0,
        audio_active=False,
        fullscreen=False,
        foreground_process="Desktop.exe",
        frame_motion_score=0.01,
        still_duration_s=2.0,
        target_fps=60.0,
    )
    return [("video", video)] * 3 + [("game", game)] * 5 + [("still", still)] * 6


def _manual_report(preset: str, output_format: str) -> dict:
    from stereo_lab import auto_detection_required, stereo_config_for_preset

    config = stereo_config_for_preset(preset, output_format=output_format)
    return {
        "selected_preset": preset,
        "auto_detection_started": auto_detection_required(preset),
        "resolved_preset": preset,
        "config": {
            "backend": config.backend,
            "hole_fill": config.hole_fill,
            "output_format": config.output_format,
            "temporal": config.temporal,
        },
        "timeline": [],
    }


def _auto_report(output_format: str, dt_s: float) -> dict:
    from stereo_lab import AutoModeRuntime, auto_detection_required, stereo_config_for_preset

    runtime = AutoModeRuntime()
    timeline = []
    for idx, (label, signals) in enumerate(_signal_sequence()):
        decision = runtime.update(signals, dt_s=dt_s)
        config = stereo_config_for_preset(decision.preset, output_format=output_format)
        timeline.append(
            {
                "sample": idx,
                "input_label": label,
                "resolved_preset": decision.preset,
                "reason": decision.reason,
                "scores": decision.scores,
                "hold_remaining_s": runtime.state.hold_remaining_s,
                "candidate_preset": runtime.state.candidate_preset,
                "candidate_count": runtime.state.candidate_count,
                "config": {
                    "backend": config.backend,
                    "hole_fill": config.hole_fill,
                    "output_format": config.output_format,
                    "temporal": config.temporal,
                },
                "signals": asdict(signals),
            }
        )

    return {
        "selected_preset": "auto",
        "auto_detection_started": auto_detection_required("auto"),
        "dt_s": dt_s,
        "timeline": timeline,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Demonstrate host-side AutoModeRuntime integration with simulated async signal snapshots.")
    parser.add_argument("--selected-preset", default="auto", choices=["auto", "cinema", "game_low_latency", "still_image_hq", "debug_export"])
    parser.add_argument("--output-format", default="half_sbs")
    parser.add_argument("--dt", type=float, default=0.25)
    parser.add_argument("--out", default="-", help="JSON report path, or '-' to print without writing.")
    args = parser.parse_args()

    if args.selected_preset == "auto":
        report = _auto_report(args.output_format, args.dt)
    else:
        report = _manual_report(args.selected_preset, args.output_format)

    text = json.dumps(report, indent=2)
    if args.out != "-":
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
