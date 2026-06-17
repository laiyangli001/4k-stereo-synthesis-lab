from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    print("[1/4] importing torch ...", flush=True)
    import torch

    print("[2/4] importing stereo_lab ...", flush=True)
    from stereo_lab import StereoConfig, synthesize_stereo
    from stereo_lab.temporal import TemporalState

    print(f"[info] torch={torch.__version__} cuda={torch.cuda.is_available()}", flush=True)
    print("[3/4] running smoke cases ...", flush=True)

    rgb = torch.rand(1, 3, 32, 64)
    depth = torch.rand(1, 1, 32, 64)

    half = synthesize_stereo(rgb, depth, StereoConfig(backend="fast", output_format="half_sbs"))
    full = synthesize_stereo(rgb, depth, StereoConfig(backend="fast", output_format="full_sbs"))
    quality = synthesize_stereo(
        rgb,
        depth,
        StereoConfig(backend="quality_4k", layers=2, output_format="half_sbs", debug_output=True),
    )
    hq = synthesize_stereo(
        rgb,
        depth,
        StereoConfig(backend="hq_4k", layers=2, output_format="half_sbs", debug_output=True),
    )

    state = TemporalState()
    synthesize_stereo(rgb, depth, StereoConfig(backend="quality_4k", temporal=True), temporal_state=state)
    synthesize_stereo(rgb, depth, StereoConfig(backend="quality_4k", temporal=True), temporal_state=state)

    assert half.sbs.shape == (1, 3, 32, 64)
    assert full.sbs.shape == (1, 3, 32, 128)
    assert quality.debug_info["occlusion_mask"].shape == (1, 1, 32, 64)
    assert hq.debug_info["layers"] == 3
    print("[4/4] smoke ok", flush=True)


if __name__ == "__main__":
    main()
