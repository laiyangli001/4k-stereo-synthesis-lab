# glTF 2.0 Renderer Compliance Layer 实施计划

## 1. 背景

当前 OpenXR 房间模型和控制器模型已经改用 `pygltflib` 解析 glTF 2.0 文件结构，但显示是否正确仍取决于项目自己的 OpenGL / D3D11 上传和 shader 管线。

Don McCurdy glTF Viewer 能一步正确显示，是因为它背后是 three.js 的完整 glTF 渲染栈：GLTFLoader、scene graph、PBR material、透明 pass、色彩管理、tone mapping、sky/background 处理等。`pygltflib` 只解决文件结构解析，不等于完整 viewer。

Artemis 暴露的问题包括：

- preview 曾把 loader 输出的 10-float vertex contract 当 8-float 读取，导致放射状平面。
- preview 曾把 `alphaMode=BLEND` 当 opaque 绘制，导致透明面片遮挡。
- `KHR_materials_unlit: {}` 曾被错误当成 False。

因此后续目标应从单模型补丁改为建设 **glTF 2.0 renderer compliance layer**。

## 2. 目标

建立位于 glTF parser 与 OpenGL / D3D11 backend 之间的合规适配层：

```text
glTF 2.0 asset
  -> pygltflib parser
  -> glTF compliance adapter
  -> stable mesh/material/texture/render-pass contract
  -> OpenGL backend
  -> D3D11 backend
```

目标：

1. 解析层不再手写 GLB container / JSON schema。
2. mesh contract 唯一，preview、OpenGL、D3D11 全部一致。
3. material contract 唯一，OpenGL / D3D11 共用。
4. render pass 明确区分 opaque / mask / transparent / sky-background。
5. 色彩空间和 alpha 语义明确。
6. `extensionsRequired` 不支持时 fail fast，不静默错误显示。
7. 用合成测试和真实模型 smoke test 锁住边界。

## 3. 非目标

第一阶段不承诺完整替代 Don McCurdy glTF Viewer：

- 不一次性支持所有 `KHR_materials_*` 扩展。
- 不完整支持 animation、skin、morph target runtime 播放。
- 不支持 Draco / Meshopt，除非明确接入解码库。
- 不追求与 three.js 完全一致的 IBL / tone mapping。

但必须做到：不支持且影响正确性的 required extension 必须明确报错，不能错误渲染成看似可用的画面。

## 4. 当前问题清单

### 4.1 Mesh contract 漂移

当前 loader 输出可能是：

```text
position3 + normal3 + uv0 2 + uv1 2
```

部分 consumer 曾按：

```text
position3 + normal3 + uv0 2
```

读取，导致 vertex attribute 错位。

第一阶段固定 contract：

```text
position: float3
normal: float3
uv0: float2
uv1: float2
tangent: float4
```

### 4.2 Material contract 分裂

material 字段目前分散在 `gltf_loader.py`、`material_contract.py`、`controller_materials.py`、`environment_model.py`、`d3d11_native_renderer.py` 等路径中。

需要收敛为 renderer-facing material object，至少包含：

```text
base_color_factor
base_color_texture
metallic_factor
roughness_factor
metallic_roughness_texture
normal_texture
normal_scale
occlusion_texture
occlusion_strength
emissive_factor
emissive_texture
alpha_mode
alpha_cutoff
double_sided
unlit
texture_transforms
texcoord_set per texture slot
```

OpenGL 和 D3D11 不应各自解释 glTF material。

### 4.3 Render pass 缺失或不一致

最低 pass 要求：

```text
opaque pass:
  depth test on
  depth write on
  blend off

mask pass:
  depth test on
  depth write on
  blend off
  alpha cutoff discard

transparent pass:
  depth test on
  depth write off
  blend on
  source alpha / one minus source alpha
  ideally back-to-front sorted

sky/background pass:
  special handling
  avoid normal opaque geometry occlusion
```

preview、OpenGL runtime、D3D11 runtime 都必须遵守同一 pass 分类。

### 4.4 色彩空间不统一

需要明确：

- baseColor / emissive textures 是 sRGB 输入。
- normal / metallicRoughness / occlusion 是 linear data。
- shader 内部 lighting 使用 linear。
- 输出按 backend 目标格式决定 gamma / sRGB encode。

### 4.5 extension 策略不明确

| 类别 | 策略 |
|---|---|
| 已支持且 required 可接受 | 正常加载 |
| optional but unsupported | 警告并忽略 |
| required but unsupported | fail fast，停止加载 |

第一阶段明确支持：

- `KHR_materials_unlit`
- `KHR_texture_transform`
- `KHR_lights_punctual`

暂时 fail fast：

- `KHR_draco_mesh_compression`
- `EXT_meshopt_compression`
- `EXT_mesh_gpu_instancing`
- 其它会改变几何或材质关键语义的 `extensionsRequired`

## 5. 目标架构

建议新增或重组为：

```text
src/xr_viewer/gltf/
  __init__.py
  document.py          # pygltflib loading, buffers, images
  accessors.py         # accessor -> numpy
  scene.py             # scene graph, node matrices, mesh instances
  materials.py         # glTF material -> stable material contract
  primitives.py        # mesh primitive -> stable primitive contract
  render_plan.py       # opaque/mask/blend/sky pass classification
  validation.py        # extensionsRequired / asset sanity audit
```

如果暂时不拆目录，也至少应在 `gltf_loader.py` 内形成同样逻辑分区，并逐步迁移。

## 6. 数据 contract

建议定义 dataclass 或轻量 dict schema：

```python
GltfPrimitive:
    vertices: np.ndarray       # N x 10 float32
    tangent: np.ndarray        # N x 4 float32
    indices: np.ndarray        # M uint32
    material: GltfMaterial
    node_name: str
    mesh_name: str
    world_bounds: tuple[np.ndarray, np.ndarray]
    render_pass: Literal["opaque", "mask", "transparent", "sky"]
```

```python
GltfMaterial:
    base_color: tuple[float, float, float]
    base_alpha: float
    alpha_mode: Literal["OPAQUE", "MASK", "BLEND"]
    alpha_cutoff: float
    double_sided: bool
    unlit: bool
    texture_slots: dict[str, TextureBinding]
    roughness: float
    metallic: float
    normal_scale: float
    occlusion_strength: float
    emissive_factor: tuple[float, float, float]
```

```python
TextureBinding:
    image_id: int
    sampler: tuple[int, int, int, int]
    texcoord: int
    transform: TextureTransform
    color_space: Literal["srgb", "linear"]
```

## 7. Backend 接口

OpenGL 和 D3D11 backend 只接收 stable contract：

```text
load_gltf_scene(path)
  -> GltfScene(primitives, textures, lights, render_plan, diagnostics)

OpenGLEnvironmentUploader.upload(scene)
D3D11EnvironmentUploader.upload(scene)
PreviewUploader.upload(scene)
```

目标是让 preview、OpenGL、D3D11 对同一个模型使用同一个上传数据。

## 8. 实施阶段

### Phase 0：冻结当前事实和回归样例

任务：

1. 固定 Artemis 作为真实模型 smoke test。
2. 记录 primitive count、texture count、light count、alphaMode 分布、vertex layout width、local/world bounds。
3. 增加 loader diagnostics 输出函数。
4. preview / OpenGL / D3D11 打印同一个 active model summary。

验收：

```text
Artemis:
  44 primitives
  19 textures
  4 lights
  BLEND / OPAQUE distribution matches loader diagnostics
  vertex stride = 10
```

### Phase 1：统一 mesh contract

任务：

1. 明确 `vertices` 固定为 `N x 10 float32`。
2. tangent 固定为 `N x 4 float32`。
3. preview VAO、OpenGL env/controller VAO、D3D11 input layout 全部按 contract 读取。
4. 增加测试防止 8-float 读取 10-float 数据。

验收：

- 所有环境模型和控制器模型加载后 vertex shape 一致。
- preview 不再出现放射状平面。
- OpenGL / D3D11 渲染同一模型的几何 bounds 一致。

### Phase 2：统一 material contract

任务：

1. 集中 glTF material 解析：

```python
parse_gltf_material(gltf, material_index, texture_registry) -> GltfMaterial
```

2. OpenGL / D3D11 / controller / environment 不再分别解释 material 字段。
3. 明确 texture slot color space。
4. 统一 `KHR_materials_unlit` 为 extension presence。
5. 统一 `KHR_texture_transform` 到 texture binding。

验收：

- 控制器材质和房间材质走同一 material contract。
- OpenGL / D3D11 对 base color、unlit、alphaMode 的解释一致。
- 颜色发白回归测试继续通过。

### Phase 3：统一 render pass

任务：

1. 根据 material 分类：

```text
alphaMode OPAQUE -> opaque
alphaMode MASK -> mask
alphaMode BLEND -> transparent
sky/background naming or profile marker -> sky
```

2. OpenGL preview、OpenGL runtime、D3D11 runtime 都按 render plan 绘制。
3. transparent pass 关闭 depth write。
4. transparent pass 后续增加 back-to-front sorting。
5. SkyBox 不再作为普通 opaque geometry 误遮挡场景。

验收：

- Artemis transparent planes 不再遮挡为实心平面。
- preview 与 runtime 的 pass 分类一致。
- D3D11 / OpenGL 不因 BLEND primitive 改动导致材质差异。

### Phase 4：统一 color management

任务：

1. 明确 shader 输入纹理色彩空间。
2. OpenGL / D3D11 baseColor、emissive 做相同 sRGB->linear。
3. 输出 gamma / tone mapping 策略统一。
4. preview 可提供独立 exposure/gamma，但不能改变 material contract。

验收：

- 控制器按钮颜色在 OpenGL / D3D11 一致。
- 房间贴图颜色与标准 viewer 接近。
- 不再用 backend-specific 临时 gamma 补丁修颜色。

### Phase 5：extension audit 与 fail fast

任务：

1. `load_gltf_scene()` 返回 diagnostics：extensionsUsed、extensionsRequired、unsupportedRequired、unsupportedOptional、materialExtensions、primitiveExtensions。
2. required unsupported 时停止加载该模型，并打印明确原因。
3. optional unsupported 只警告。

验收：

- 带 Draco / Meshopt required 的模型不会错误显示。
- 日志能明确告诉用户需要转码或安装解码支持。

### Phase 6：真实模型回归集

建议模型集：

| 模型 | 覆盖点 |
|---|---|
| SimpleTriangle.glb | 最小 geometry |
| ExternalBuffer.gltf + .bin | 外部 buffer |
| UnlitEmptyExtension.glb | `KHR_materials_unlit: {}` |
| AlphaBlendPlanes.glb | BLEND pass |
| AlphaMaskFoliage.glb | MASK / alphaCutoff |
| TextureTransform.glb | `KHR_texture_transform` |
| Artemis | 真实复杂场景 |
| Bedroom | 当前生产房间 |
| Controller models | 控制器材质和动画节点 |

## 9. 验证命令

```powershell
src\python3\python.exe -m py_compile src\xr_viewer\gltf_loader.py src\tools\preview_room_layout.py
src\python3\python.exe -m pytest tests\test_keyboard_screen_preset.py -q
src\python3\python.exe -c "import sys, pytest; sys.path.insert(0, 'src'); raise SystemExit(pytest.main(['tests/test_openxr_runtime.py','-q']))"
```

Artemis smoke check：

```powershell
src\python3\python.exe -c "import sys; sys.path.insert(0,'src'); from xr_viewer.gltf_loader import load_glb_model; prims,tex,lights=load_glb_model('src/xr_viewer/environments/Artemis/environment.glb'); print(len(prims), len(tex), len(lights), prims[0]['vertices'].shape)"
```

preview 检查：

```powershell
src\python3\python.exe src\tools\preview_room_layout.py Artemis
```

## 10. 第一批落地任务

1. 新增 `gltf_contract.py` 或 `xr_viewer/gltf/` 包，定义 primitive/material/texture binding contract。
2. 把 `gltf_loader.py` 的 material 解析拆出为 `parse_gltf_material()`。
3. 把 preview、OpenGL、D3D11 的 vertex layout 检查写入测试。
4. 把 render pass 分类写成公共函数。
5. preview 改为调用公共 render plan，不再自己判断 alpha。
6. OpenGL runtime 环境模型改为调用公共 render plan。
7. D3D11 native renderer 改为调用同一 material/render pass contract。
8. 增加 Artemis / Bedroom / controller model diagnostics smoke tests。

## 11. 完成标准

1. 更换符合 glTF 2.0 core 的普通模型时，不再出现 vertex layout 错读导致的几何爆炸。
2. `alphaMode`、`doubleSided`、`unlit`、texture transform 在 preview / OpenGL / D3D11 一致。
3. unsupported required extensions 有明确日志和失败路径。
4. 用户不需要通过观察画面猜测是模型问题、loader 问题还是 render pass 问题。
5. 新模型接入流程变成：

```text
放入 environment.glb / profile.json
运行 diagnostics
若 unsupported required 为空，进入 preview
preview 与 OpenXR summary 一致
再做 profile 位置/灯光适配
```
