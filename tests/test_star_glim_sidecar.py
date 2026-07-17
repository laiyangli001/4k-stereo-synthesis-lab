from __future__ import annotations

import json

from xr_viewer.panda_runtime import scene as panda_scene_module
from xr_viewer.panda_runtime.runtime import PandaFrameState
from xr_viewer.panda_runtime.scene import PandaAssetRef, PandaSceneGraph
from xr_viewer.panda_runtime.star_glim import load_star_glim_spec


def test_load_star_glim_spec_accepts_complete_asset_sidecar(tmp_path) -> None:
    asset_path = tmp_path / "environment.glb"
    asset_path.write_bytes(b"glTF")
    (tmp_path / "star_glim_stars.png").write_bytes(b"stars")
    (tmp_path / "star_glim_mask.png").write_bytes(b"mask")
    (tmp_path / "star_glim.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "effect": "star_glim",
                "node_name_patterns": ["*SkyBox*"],
                "stars_texture": "star_glim_stars.png",
                "mask_texture": "star_glim_mask.png",
                "intensity": 10.0,
                "speed": 0.001,
                "shine_speed": 0.5,
                "cell_density": 300.0,
                "cell_offset": 10.0,
                "cell_soft": 0.071,
                "cell_value": 0.0,
                "strength": 2.0,
            }
        ),
        encoding="utf-8",
    )

    spec = load_star_glim_spec(asset_path)

    assert spec is not None
    assert spec.node_name_patterns == ("*SkyBox*",)
    assert spec.stars_texture_path == tmp_path / "star_glim_stars.png"
    assert spec.mask_texture_path == tmp_path / "star_glim_mask.png"
    assert spec.intensity == 10.0
    assert spec.strength == 2.0


def test_load_star_glim_spec_requires_complete_local_inputs(tmp_path) -> None:
    asset_path = tmp_path / "environment.glb"
    asset_path.write_bytes(b"glTF")
    (tmp_path / "star_glim.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "effect": "star_glim",
                "node_name_patterns": ["*SkyBox*"],
                "stars_texture": "../outside.png",
                "mask_texture": "star_glim_mask.png",
                "intensity": 10.0,
                "speed": 0.001,
                "shine_speed": 0.5,
                "cell_density": 300.0,
                "cell_offset": 10.0,
                "cell_soft": 0.071,
                "cell_value": 0.0,
                "strength": 2.0,
            }
        ),
        encoding="utf-8",
    )

    assert load_star_glim_spec(asset_path) is None


def test_scene_binds_star_glim_once_and_advances_with_frame_clock(monkeypatch) -> None:
    scene = PandaSceneGraph(load_panda_assets=True)
    root = object()
    calls = []
    time_updates = []
    monkeypatch.setattr(
        scene,
        "_make_asset_ref",
        lambda role, path: (PandaAssetRef(role, path, loaded_with_panda=True), root, None),
    )
    monkeypatch.setattr(
        panda_scene_module,
        "apply_star_glim_sidecar",
        lambda path, bound_root, *, base_color_texture: calls.append((path, bound_root, base_color_texture)) or ("sky",),
    )
    monkeypatch.setattr(
        panda_scene_module,
        "set_star_glim_time",
        lambda targets, time_seconds: time_updates.append((targets, time_seconds)),
    )

    scene.load_environment("Artemis/environment.glb")
    scene.update_frame_state(PandaFrameState(animation_time_seconds=1.25))

    assert calls == [("Artemis/environment.glb", root, panda_scene_module._base_color_texture)]
    assert time_updates == [(("sky",), 1.25)]
