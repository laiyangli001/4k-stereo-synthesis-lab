from __future__ import annotations

import json
from pathlib import Path
import sys

from pygltflib import GLTF2

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from xr_viewer.panda3d_material_probe import (  # noqa: E402
    inspect_panda3d_materials,
    material_report_as_json,
)


ARTEMIS_ENVIRONMENT = ROOT / "src" / "xr_viewer" / "environments" / "Artemis" / "environment.glb"


def test_artemis_material_probe_records_phase0_semantics():
    report = inspect_panda3d_materials(ARTEMIS_ENVIRONMENT)

    assert report.material_count == 44
    assert report.image_count == 19
    assert report.texture_count == 19
    assert report.alpha_mode_counts == {"BLEND": 28, "OPAQUE": 16}
    assert report.double_sided_count == 0
    assert report.unlit_material_count == 44
    assert report.skybox_material_names == ("GLTF_UnlitSkybox_GLTF_Skybox_Composite_0",)
    assert len(report.transparent_material_names) == 28
    assert report.material_semantics_ready is True


def test_artemis_skybox_material_is_self_contained():
    gltf = GLTF2().load(str(ARTEMIS_ENVIRONMENT))
    skybox = next(
        material
        for material in gltf.materials or ()
        if "skybox" in (material.name or "").lower()
    )

    assert (skybox.alphaMode or "OPAQUE") == "OPAQUE"
    assert skybox.doubleSided is False
    assert "KHR_materials_unlit" in (skybox.extensions or {})


def test_material_probe_json_is_stable_and_cli_friendly():
    report = inspect_panda3d_materials(ARTEMIS_ENVIRONMENT)
    data = json.loads(material_report_as_json(report))

    assert data["material_semantics_ready"] is True
    assert data["skybox_material_names"] == ["GLTF_UnlitSkybox_GLTF_Skybox_Composite_0"]
    assert data["alpha_mode_counts"] == {"BLEND": 28, "OPAQUE": 16}
