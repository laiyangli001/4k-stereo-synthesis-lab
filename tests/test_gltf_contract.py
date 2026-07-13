import json
from pathlib import Path

import numpy as np
import pytest

import xr_viewer.gltf as gltf_package
import xr_viewer.gltf_contract as legacy_gltf_contract
import xr_viewer.gltf_loader as legacy_gltf_loader
from xr_viewer.gltf import contract as gltf_contract_module
from xr_viewer.gltf import loader as gltf_loader_module
from xr_viewer.gltf import render_plan as gltf_render_plan_module
from xr_viewer.gltf import validation as gltf_validation_module
from xr_viewer.controller_materials import (
    collect_controller_texture_requests,
    controller_texture_cache_key,
    prepare_controller_material,
)
from xr_viewer.gltf import (
    audit_gltf_extensions,
    diagnose_gltf_model,
    load_glb_model,
    load_gltf_scene,
    parse_gltf_material,
    summarize_gltf_scene,
)
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

from xr_viewer.gltf import (
    D3D11_VERTEX_OFFSETS_BYTES,
    D3D11_VERTEX_STRIDE_BYTES,
    GltfMaterial,
    GltfScene,
    OPENGL_VERTEX_FORMAT,
    TRANSPARENT_SORT_POLICY,
    TextureBinding,
    TextureTransform,
    VERTEX_FLOAT_COUNT,
    attach_primitive_contract,
    build_render_plan,
    classify_render_pass,
    render_pass_from_primitive,
    sort_transparent_primitives,
    transparent_sort_key,
    validate_mesh_contract,
)


def test_gltf_package_reexports_stable_loader_contract_and_render_plan_api():
    assert gltf_package.GltfMaterial is GltfMaterial
    assert gltf_package.GltfScene is GltfScene
    assert gltf_package.TextureBinding is TextureBinding
    assert gltf_package.load_glb_model is load_glb_model
    assert gltf_package.load_gltf_scene is load_gltf_scene
    assert gltf_package.parse_gltf_material is parse_gltf_material
    assert gltf_package.summarize_gltf_scene is summarize_gltf_scene
    assert gltf_package.build_render_plan is build_render_plan
    assert gltf_package.TRANSPARENT_SORT_POLICY == TRANSPARENT_SORT_POLICY
    assert legacy_gltf_loader.load_gltf_scene is load_gltf_scene
    assert legacy_gltf_contract.GltfMaterial is GltfMaterial
    assert gltf_contract_module.validate_mesh_contract is validate_mesh_contract
    assert gltf_loader_module.audit_gltf_extensions is audit_gltf_extensions
    assert gltf_render_plan_module.sort_transparent_primitives is sort_transparent_primitives
    assert gltf_validation_module.audit_gltf_extensions is audit_gltf_extensions

    scene = gltf_package.load_gltf_scene(SRC / "xr_viewer" / "environments" / "Bedroom" / "environment.glb")

    assert isinstance(scene, gltf_package.GltfScene)
    assert sum(len(indices) for indices in scene.render_plan.values()) == len(scene.primitives)


def _primitive(**overrides):
    primitive = {
        "vertices": np.zeros((3, 10), dtype=np.float32),
        "tangent": np.column_stack(
            [
                np.zeros((3, 3), dtype=np.float32),
                np.ones(3, dtype=np.float32),
            ]
        ).astype(np.float32),
        "indices": np.array([0, 1, 2], dtype=np.uint32),
        "base_color": np.array([0.2, 0.4, 0.6], dtype=np.float32),
        "base_alpha": 0.75,
        "alpha_mode": "OPAQUE",
        "tex_id": 3,
        "base_sampler": (9729, 9987, 10497, 33071),
        "base_texcoord": 1,
        "tex_offset": np.array([0.25, 0.5], dtype=np.float32),
        "tex_scale": np.array([2.0, 3.0], dtype=np.float32),
        "tex_rotation": 0.5,
        "normal_tex_id": 4,
        "normal_sampler": (9729, 9987, 10497, 10497),
        "normal_texcoord": 0,
        "node_name": "Node",
        "mesh_name": "Mesh",
    }
    primitive.update(overrides)
    return primitive


def test_mesh_layout_constants_match_all_backends():
    assert VERTEX_FLOAT_COUNT == 10
    assert OPENGL_VERTEX_FORMAT == "3f 3f 2f 2f"
    assert D3D11_VERTEX_OFFSETS_BYTES == (0, 12, 24, 32)
    assert D3D11_VERTEX_STRIDE_BYTES == 40


def test_validate_mesh_contract_rejects_legacy_eight_float_vertices():
    primitive = _primitive(vertices=np.zeros((3, 8), dtype=np.float32))

    with pytest.raises(ValueError, match=r"shape \(N, 10\)"):
        validate_mesh_contract(
            primitive["vertices"],
            primitive["tangent"],
            primitive["indices"],
        )


def test_primitive_contract_requires_explicit_material_contract():
    primitive = _primitive()

    with pytest.raises(ValueError, match="material_contract"):
        attach_primitive_contract(primitive)

    with pytest.raises(ValueError, match="material_contract"):
        render_pass_from_primitive(primitive)


@pytest.mark.parametrize(
    ("alpha_mode", "expected"),
    [
        ("OPAQUE", "opaque"),
        ("MASK", "mask"),
        ("BLEND", "transparent"),
    ],
)
def test_render_pass_classification_uses_material_alpha_mode(alpha_mode, expected):
    assert classify_render_pass(GltfMaterial(alpha_mode=alpha_mode)) == expected


def test_render_pass_classification_promotes_named_sky_geometry():
    material = GltfMaterial(alpha_mode="OPAQUE")

    assert classify_render_pass(material, mesh_name="SkyBox_Main") == "sky"
    assert classify_render_pass(material, node_name="background_sky_dome") == "sky"


def test_attach_primitive_contract_exposes_material_pass_and_bounds():
    primitive = _primitive(material_contract=GltfMaterial(alpha_mode="BLEND"))
    primitive["vertices"][:, :3] = np.array(
        [[-1.0, 0.0, 2.0], [3.0, 4.0, -2.0], [0.0, 1.0, 1.0]],
        dtype=np.float32,
    )

    contract = attach_primitive_contract(primitive)

    assert primitive["gltf_primitive"] is contract
    assert primitive["material_contract"] is contract.material
    assert primitive["render_pass"] == "transparent"
    assert contract.world_bounds[0] == pytest.approx((-1.0, 0.0, -2.0))
    assert contract.world_bounds[1] == pytest.approx((3.0, 4.0, 2.0))


def test_audit_gltf_extensions_reports_required_optional_and_nested_extensions():
    diagnostics = audit_gltf_extensions(
        {
            "extensionsUsed": ["KHR_materials_unlit", "VENDOR_optional"],
            "extensionsRequired": ["KHR_materials_unlit", "KHR_draco_mesh_compression"],
            "materials": [{"extensions": {"KHR_texture_transform": {}, "VENDOR_material": {}}}],
            "meshes": [{"primitives": [{"extensions": {"EXT_meshopt_compression": {}}}]}],
        }
    )

    assert diagnostics["extensionsRequired"] == ["KHR_draco_mesh_compression", "KHR_materials_unlit"]
    assert diagnostics["unsupportedRequired"] == ["KHR_draco_mesh_compression"]
    assert diagnostics["unsupportedOptional"] == [
        "EXT_meshopt_compression",
        "VENDOR_material",
        "VENDOR_optional",
    ]
    assert diagnostics["materialExtensions"] == ["KHR_texture_transform", "VENDOR_material"]
    assert diagnostics["primitiveExtensions"] == ["EXT_meshopt_compression"]


def test_load_glb_model_fails_fast_on_unsupported_required_extension(tmp_path):
    gltf_path = tmp_path / "unsupported_required.gltf"
    gltf_path.write_text(
        json.dumps(
            {
                "asset": {"version": "2.0"},
                "extensionsUsed": ["KHR_draco_mesh_compression"],
                "extensionsRequired": ["KHR_draco_mesh_compression"],
                "scenes": [{"nodes": []}],
                "scene": 0,
                "nodes": [],
                "buffers": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="KHR_draco_mesh_compression"):
        load_glb_model(gltf_path)


def test_build_render_plan_groups_primitive_indices_by_pass():
    primitives = [
        _primitive(material_contract=GltfMaterial(alpha_mode="OPAQUE")),
        _primitive(material_contract=GltfMaterial(alpha_mode="BLEND")),
        _primitive(material_contract=GltfMaterial(alpha_mode="MASK")),
        _primitive(material_contract=GltfMaterial(alpha_mode="OPAQUE"), mesh_name="SkyBox_Main"),
    ]
    for primitive in primitives:
        attach_primitive_contract(primitive)

    assert build_render_plan(primitives) == {
        "sky": (3,),
        "opaque": (0,),
        "mask": (2,),
        "transparent": (1,),
    }


def test_transparent_sort_policy_is_back_to_front_and_uses_model_matrix():
    near = {"name": "near", "sort_center_local": np.array([0.0, 0.0, -1.0], dtype=np.float32)}
    far = {"name": "far", "sort_center_local": np.array([0.0, 0.0, -3.0], dtype=np.float32)}
    moved = {
        "name": "moved",
        "sort_center_local": np.array([0.0, 0.0, 1.0], dtype=np.float32),
    }
    model = np.eye(4, dtype=np.float32)
    model[2, 3] = 5.0

    assert TRANSPARENT_SORT_POLICY == "back_to_front"
    assert transparent_sort_key(far, (0.0, 0.0, 0.0)) > transparent_sort_key(
        near,
        (0.0, 0.0, 0.0),
    )
    assert [
        prim["name"] for prim in sort_transparent_primitives([near, far], (0.0, 0.0, 0.0))
    ] == ["far", "near"]
    assert sort_transparent_primitives([near, moved], (0.0, 0.0, 0.0), model)[0] is moved


def test_summarize_gltf_scene_reports_counts_passes_and_bounds():
    primitive = _primitive(material_contract=GltfMaterial(alpha_mode="BLEND"))
    primitive["vertices"][:, :3] = np.array(
        [[-1.0, -2.0, 0.0], [2.0, 3.0, 4.0], [0.0, 1.0, -3.0]],
        dtype=np.float32,
    )
    attach_primitive_contract(primitive)

    summary = summarize_gltf_scene([primitive], [np.zeros((1, 1, 4), dtype=np.uint8)], [{"type": "point"}])

    assert summary["primitive_count"] == 1
    assert summary["texture_count"] == 1
    assert summary["light_count"] == 1
    assert summary["alpha_modes"] == {"BLEND": 1}
    assert summary["render_passes"] == {"transparent": 1}
    assert summary["vertex_widths"] == [10]
    assert summary["scene_bounds"][0] == pytest.approx((-1.0, -2.0, -3.0))
    assert summary["scene_bounds"][1] == pytest.approx((2.0, 3.0, 4.0))


def test_bedroom_diagnostics_smoke_matches_stable_contract():
    scene = load_gltf_scene(SRC / "xr_viewer" / "environments" / "Bedroom" / "environment.glb")
    summary = diagnose_gltf_model(SRC / "xr_viewer" / "environments" / "Bedroom" / "environment.glb")

    assert isinstance(scene, GltfScene)
    assert len(scene.primitives) > 0
    assert len(scene.textures) > 0
    assert scene.diagnostics["unsupportedRequired"] == []
    assert sum(len(indices) for indices in scene.render_plan.values()) == len(scene.primitives)
    assert summary["primitive_count"] > 0
    assert summary["texture_count"] > 0
    assert summary["vertex_widths"] == [10]
    assert summary["diagnostics"]["unsupportedRequired"] == []
    assert summary["render_plan"] == scene.render_plan
    assert sum(summary["render_passes"].values()) == summary["primitive_count"]


@pytest.mark.parametrize("glb_path", sorted((SRC / "xr_viewer" / "controllers").glob("*/*.glb")), ids=lambda p: str(p.relative_to(SRC)))
def test_controller_models_smoke_match_stable_contract(glb_path):
    primitives, textures, lights = load_glb_model(glb_path)
    summary = summarize_gltf_scene(primitives, textures, lights)

    assert summary["primitive_count"] > 0, glb_path
    assert summary["vertex_widths"] == [10], glb_path
    assert sum(summary["render_passes"].values()) == summary["primitive_count"], glb_path
    assert all("material_contract" in primitive for primitive in primitives), glb_path
    assert all("gltf_primitive" in primitive for primitive in primitives), glb_path


def test_controller_material_uses_contract_without_legacy_dict_fallback():
    material_contract = GltfMaterial(
        base_color=(0.1, 0.2, 0.3),
        base_alpha=0.4,
        alpha_mode="MASK",
        alpha_cutoff=0.25,
        double_sided=True,
        unlit=True,
        roughness=0.65,
        metallic=0.15,
        normal_scale=0.75,
        occlusion_strength=0.5,
        emissive_factor=(0.4, 0.5, 0.6),
        texture_slots={
            "base": TextureBinding(
                image_id=7,
                sampler=(9728, 9984, 33071, 33648),
                texcoord=1,
                transform=TextureTransform(offset=(0.2, 0.3), scale=(2.0, 3.0), rotation=0.5),
                color_space="srgb",
            ),
            "normal": TextureBinding(image_id=8, sampler=(9729, 9987, 10497, 10497), texcoord=2),
        },
    )
    material = prepare_controller_material(material_contract, "QUEST/left", {"diagnostics": {"materialMode": "opaque_unlit"}})

    assert material["base_color"] == pytest.approx((0.1, 0.2, 0.3))
    assert material["base_alpha"] == pytest.approx(0.4)
    assert material["alpha_mode"] == "MASK"
    assert material["alpha_mode_id"] == 1
    assert material["alpha_cutoff"] == pytest.approx(0.25)
    assert material["double_sided"] is True
    assert material["unlit"] is True
    assert material["roughness"] == pytest.approx(0.65)
    assert material["metallic"] == pytest.approx(0.15)
    assert material["normal_scale"] == pytest.approx(0.75)
    assert material["occlusion_strength"] == pytest.approx(0.5)
    assert material["emissive_factor"] == pytest.approx((0.4, 0.5, 0.6))
    assert material["tex_offset"] == pytest.approx((0.2, 0.3))
    assert material["tex_scale"] == pytest.approx((2.0, 3.0))
    assert material["tex_rotation"] == pytest.approx(0.5)
    assert material["base_texcoord"] == 1
    assert material["normal_texcoord"] == 2
    assert material["base_key"] == "QUEST/left:7:9728:9984:33071:33648"
    assert material["normal_key"] == "QUEST/left:8:9729:9987:10497:10497"
    assert collect_controller_texture_requests([material_contract]) == {
        (7, (9728, 9984, 33071, 33648)),
        (8, (9729, 9987, 10497, 10497)),
    }
    assert controller_texture_cache_key("QUEST/left", 7, (9728, 9984, 33071, 33648)) == "QUEST/left:7:9728:9984:33071:33648"


def test_controller_material_requires_contract():
    with pytest.raises(ValueError, match="GltfMaterial"):
        prepare_controller_material(_primitive(), "QUEST/left", {})

    with pytest.raises(ValueError, match="GltfMaterial"):
        collect_controller_texture_requests([_primitive()])


def test_parse_gltf_material_outputs_renderer_fields_and_transform():
    gltf = {
        "materials": [
            {
                "pbrMetallicRoughness": {
                    "baseColorFactor": [0.2, 0.4, 0.6, 0.8],
                    "baseColorTexture": {
                        "index": 0,
                        "texCoord": 1,
                        "extensions": {
                            "KHR_texture_transform": {
                                "offset": [0.25, 0.5],
                                "scale": [2.0, 3.0],
                                "rotation": 0.75,
                            }
                        },
                    },
                    "metallicFactor": 0.1,
                    "roughnessFactor": 0.9,
                },
                "normalTexture": {"index": 1, "scale": 0.5},
                "occlusionTexture": {"index": 2, "strength": 0.25},
                "emissiveFactor": [0.1, 0.2, 0.3],
                "emissiveTexture": {"index": 3},
                "alphaMode": "BLEND",
                "doubleSided": True,
                "extensions": {"KHR_materials_unlit": {}},
            }
        ]
    }
    textures = [
        np.zeros((1, 1, 4), dtype=np.uint8),
        np.zeros((1, 1, 4), dtype=np.uint8),
        np.zeros((1, 1, 4), dtype=np.uint8),
        np.zeros((1, 1, 4), dtype=np.uint8),
    ]

    fields = parse_gltf_material(
        gltf,
        0,
        tex_img_map={0: 0, 1: 1, 2: 2, 3: 3},
        tex_sampler_map={0: (9729, 9987, 33071, 10497)},
        all_textures=textures,
    )

    assert fields["tex_id"] == 0
    assert fields["base_color"] == pytest.approx((0.2, 0.4, 0.6))
    assert fields["base_alpha"] == pytest.approx(0.8)
    assert fields["base_sampler"] == (9729, 9987, 33071, 10497)
    assert fields["base_texcoord"] == 1
    assert fields["tex_offset"] == pytest.approx((0.25, 0.5))
    assert fields["tex_scale"] == pytest.approx((2.0, 3.0))
    assert fields["tex_rotation"] == pytest.approx(0.75)
    assert fields["normal_tex_id"] == 1
    assert fields["normal_scale"] == pytest.approx(0.5)
    assert fields["occlusion_tex_id"] == 2
    assert fields["occlusion_strength"] == pytest.approx(0.25)
    assert fields["emissive_tex_id"] == 3
    assert fields["emissive_factor"] == pytest.approx((0.1, 0.2, 0.3))
    assert fields["alpha_mode"] == "BLEND"
    assert fields["double_sided"] is True
    assert fields["unlit"] is True
    material = fields["material_contract"]
    assert isinstance(material, GltfMaterial)
    assert material.base_color == pytest.approx((0.2, 0.4, 0.6))
    assert material.base_alpha == pytest.approx(0.8)
    assert material.alpha_mode == "BLEND"
    assert material.double_sided is True
    assert material.unlit is True
    assert material.texture_slots["base"] == TextureBinding(
        image_id=0,
        sampler=(9729, 9987, 33071, 10497),
        texcoord=1,
        transform=material.texture_slots["base"].transform,
        color_space="srgb",
    )
    assert material.texture_slots["base"].transform.offset == pytest.approx((0.25, 0.5))
    assert material.texture_slots["base"].transform.scale == pytest.approx((2.0, 3.0))
    assert material.texture_slots["base"].transform.rotation == pytest.approx(0.75)
    assert material.texture_slots["normal"].color_space == "linear"
