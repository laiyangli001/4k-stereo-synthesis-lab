# glTF 2.0 Core Renderer Compliance Layer 实施计划

## 1. 背景

当前 OpenXR 房间模型和控制器模型已经改用 `pygltflib` 解析 glTF 2.0 文件结构，但显示是否正确仍取决于项目自己的 OpenGL / D3D11 上传和 shader 管线。

Don McCurdy glTF Viewer 能一步正确显示，是因为它背后是 three.js 的完整 glTF 渲染栈：GLTFLoader、scene graph、animation mixer、skinning、morph target、PBR material、透明 pass、色彩管理、tone mapping、sky/background 处理等。`pygltflib` 只解决文件结构解析，不等于完整 viewer。

Artemis 暴露的问题包括：

- preview 曾把 loader 输出的 10-float vertex contract 当 8-float 读取，导致放射状平面。
- preview 曾把 `alphaMode=BLEND` 当 opaque 绘制，导致透明面片遮挡。
- `KHR_materials_unlit: {}` 曾被错误当成 False。
- 标准 GLB animation 能在 Don McCurdy viewer 播放，但本项目 loader 把 node transform 烘进静态 VBO，导致卫星/飞船动画不可见。

因此目标应从单模型补丁升级为 **glTF 2.0 core renderer compliance layer**。本文档覆盖 glTF 2.0 core 的静态与动态语义；Draco、Meshopt、BasisU、`KHR_materials_*` 等非 core 扩展按 extension 策略单独处理。

## 2. 目标

建立位于 glTF parser 与 OpenGL / D3D11 backend 之间的合规适配层：

```text
glTF 2.0 asset
  -> parser
  -> glTF core compliance adapter
  -> stable scene/mesh/material/animation/skin/morph contract
  -> OpenGL backend
  -> D3D11 backend
```

目标：

1. 解析层不再手写 GLB container / JSON schema。
2. mesh contract 唯一，preview、OpenGL、D3D11 全部一致。
3. material contract 唯一，OpenGL / D3D11 共用。
4. render pass 明确区分 opaque / mask / transparent / sky-background。
5. 色彩空间和 alpha 语义明确。
6. glTF core animation、skin、morph target、camera、sparse accessor 不再被静默忽略。
7. `extensionsRequired` 不支持时 fail fast，不静默错误显示。
8. 优先复用成熟 glTF runtime / renderer；自研只做必须的 adapter 和 fallback。
9. 用合成测试和真实模型 smoke test 锁住边界。

## 3. 范围边界

### 3.1 glTF 2.0 core 必须覆盖

- GLB / `.gltf` JSON / external buffer / data URI buffer。
- bufferView / accessor：`byteOffset`、`byteStride`、`componentType`、`type`、`normalized`、sparse accessor。
- scene / node hierarchy：active scene、多 scene、matrix、TRS、动态 world matrix 传播。
- mesh primitive：indices、attributes、morph targets。
- core PBR material：metallic-roughness、normal、occlusion、emissive、alphaMode、doubleSided。
- image / texture / sampler。
- animation：channels、samplers、translation、rotation、scale、weights，STEP / LINEAR / CUBICSPLINE。
- skinning：skins、joints、inverseBindMatrices、`JOINTS_0`、`WEIGHTS_0`。
- cameras：perspective / orthographic。

### 3.2 非 core 扩展

以下不是 glTF 2.0 core，不作为 core 合规完成条件；如果出现在 `extensionsRequired`，必须 fail fast：

- `KHR_draco_mesh_compression`
- `EXT_meshopt_compression`
- `KHR_texture_basisu`
- `EXT_mesh_gpu_instancing`
- 未明确支持的 `KHR_materials_*`

### 3.3 避免重复造轮子

完整 glTF viewer 是大型渲染栈，不应长期靠本项目零散补丁重写。

优先级：

1. 评估接入成熟 runtime renderer，例如 Filament `gltfio`。
2. Web preview / 对照验证继续使用 Don McCurdy viewer、Khronos Sample Viewer、three.js / Babylon.js。
3. parser 级库如 `pygltflib`、tinygltf、cgltf、Microsoft glTF-SDK 只能解决结构读取，不能替代 animation/skin/PBR runtime。
4. 自研路径只补当前 runtime 必需的最小合规面，并用 diagnostics 明确未支持项。

## 4. 当前问题清单

### 4.1 Mesh contract 漂移

当前 loader 输出固定目标：

```text
position: float3
normal: float3
uv0: float2
uv1: float2
tangent: float4
```

preview、OpenGL、D3D11 必须按同一 contract 读取。任何 consumer 按 8-float 读取 10-float 数据都属于合规失败。

### 4.2 Material contract 分裂

material 字段不应分散在 loader、controller、environment、D3D11 native renderer 中各自解释。

renderer-facing material 至少包含：

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
texture_transforms
texcoord_set per texture slot
```

`KHR_materials_unlit` 属于扩展，但已作为项目支持项进入 material contract。

### 4.3 Render pass 缺失或不一致

最低 pass 要求：

```text
opaque:
  depth test on
  depth write on
  blend off

mask:
  depth test on
  depth write on
  blend off
  alpha cutoff discard

transparent:
  depth test on
  depth write off
  blend on
  source alpha / one minus source alpha
  back-to-front sorted

sky/background:
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

### 4.5 Extension 策略不明确

| 类别 | 策略 |
|---|---|
| 已支持且 required 可接受 | 正常加载 |
| optional but unsupported | 警告并忽略 |
| required but unsupported | fail fast，停止加载 |

当前明确支持：

- `KHR_materials_unlit`
- `KHR_texture_transform`
- `KHR_lights_punctual`

暂时 fail fast：

- `KHR_draco_mesh_compression`
- `EXT_meshopt_compression`
- `EXT_mesh_gpu_instancing`
- 其它会改变几何或材质关键语义的 `extensionsRequired`

### 4.6 Core animation runtime 缺失

当前 loader 会在加载时把 node world matrix 烘进顶点 VBO。这个做法对静态模型可用，但对带 `animations` 的 glTF core 模型是错误的：

- animation channel 指向 node 的 `translation` / `rotation` / `scale` / `weights` 后，renderer 必须每帧采样并更新 node local matrix。
- animated node 的 mesh 顶点不能只按加载时的静态矩阵烘死。
- transparent sorting、bounds、camera、light 相关 world matrix 必须使用当前帧动态矩阵。

Artemis 卫星/飞船 GLB 已能在标准 viewer 中播放动画，暴露的就是这一缺口。

### 4.7 Skinning 缺失

glTF core skinning 需要：

- 读取 `skins`、`joints`、`inverseBindMatrices`。
- 读取 vertex attributes `JOINTS_0` / `WEIGHTS_0`。
- 每帧根据 joint world matrix 生成 joint matrix palette。
- OpenGL / D3D11 shader 统一消费 skinning 数据。

不支持 skinning 时，带骨骼动画的模型会静态错误或变形错误。

### 4.8 Morph target / weights 缺失

glTF core mesh target 需要：

- 读取 primitive `targets` 中的 POSITION / NORMAL / TANGENT delta。
- 支持 mesh / node 默认 `weights`。
- 支持 animation target path `weights`。
- shader 或 CPU fallback 应用 morph delta，并同步 normal/tangent 处理。

### 4.9 Sparse accessor / interleaved buffer / normalized attribute 缺失

accessor 层必须完整支持：

- sparse accessor 覆盖 base accessor 数据。
- interleaved bufferView `byteStride`。
- normalized integer attribute 到 float 的标准转换。
- 所有合法 componentType / accessor type 组合。
- accessor min/max 诊断与 bounds 验证。

### 4.10 Camera 和多 scene 语义缺失

glTF core 包含：

- 多 scene，active scene 默认规则。
- perspective / orthographic camera。
- camera node 的动态 transform。

当前 environment profile 自己定义视角，不等于支持 glTF camera。core 合规层应能解析 camera，并让 preview / diagnostics / 可选 viewer 路径消费。

## 5. 目标架构

建议模块边界：

```text
src/xr_viewer/gltf/
  __init__.py
  document.py          # GLB/gltf loading, buffers, images
  accessors.py         # accessor -> numpy, sparse/stride/normalized
  scene.py             # scene graph, node matrices, mesh instances
  animation.py         # animation channels/samplers and runtime sampling
  skinning.py          # skins, joints, inverse bind matrices, joint palettes
  morph.py             # morph targets and animated weights
  cameras.py           # perspective / orthographic cameras
  materials.py         # material -> stable material contract
  primitives.py        # mesh primitive -> stable primitive contract
  render_plan.py       # opaque/mask/blend/sky pass classification
  validation.py        # core diagnostics and extensionsRequired audit
```

## 6. 数据 contract

```python
GltfPrimitive:
    vertices: np.ndarray       # N x 10 float32, local mesh space for animated nodes
    tangent: np.ndarray        # N x 4 float32
    indices: np.ndarray        # M uint32
    material: GltfMaterial
    node_index: int
    node_name: str
    mesh_name: str
    local_bounds: tuple[np.ndarray, np.ndarray]
    world_bounds: tuple[np.ndarray, np.ndarray]  # current frame
    render_pass: Literal["opaque", "mask", "transparent", "sky"]
    skin_index: int | None
    morph_target_indices: tuple[int, ...]
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

```python
GltfNode:
    index: int
    name: str
    parent: int | None
    children: tuple[int, ...]
    mesh: int | None
    skin: int | None
    camera: int | None
    base_translation: np.ndarray
    base_rotation: np.ndarray       # quaternion x,y,z,w
    base_scale: np.ndarray
    base_matrix: np.ndarray | None
    local_matrix: np.ndarray        # current frame
    world_matrix: np.ndarray        # current frame
```

```python
GltfAnimation:
    name: str
    duration: float
    channels: tuple[GltfAnimationChannel, ...]

GltfAnimationChannel:
    target_node: int
    target_path: Literal["translation", "rotation", "scale", "weights"]
    interpolation: Literal["STEP", "LINEAR", "CUBICSPLINE"]
    input_times: np.ndarray
    output_values: np.ndarray
```

```python
GltfSkin:
    joints: tuple[int, ...]
    inverse_bind_matrices: np.ndarray
    joint_matrices: np.ndarray      # current frame palette
```

```python
GltfCamera:
    type: Literal["perspective", "orthographic"]
    node_index: int
    projection_params: dict
```

## 7. Backend 接口

```text
load_gltf_scene(path)
  -> GltfScene(nodes, primitives, textures, lights, cameras, animations, skins, render_plan, diagnostics)

GltfRuntimeState.update(time_seconds, animation_index)
OpenGLEnvironmentUploader.upload_static(scene)
OpenGLEnvironmentRenderer.draw(scene, runtime_state)
D3D11EnvironmentUploader.upload_static(scene)
D3D11EnvironmentRenderer.draw(scene, runtime_state)
PreviewUploader.upload_static(scene)
PreviewRenderer.draw(scene, runtime_state)
```

核心约束：

- 静态 mesh 可以走预烘 world transform 快路径。
- animated / skinned / morphed mesh 不能只烘成一次性 VBO。
- backend 必须支持 per-primitive model matrix，或等价的 dynamic instance matrix / dynamic VBO fallback。
- transparent sorting 和 bounds 必须能使用 runtime_state 的当前帧 world matrix。

## 8. 实施阶段

状态标记：

- `[x]` 已完成：当前代码和测试已覆盖该阶段主要验收项。
- `[~]` 部分完成：已有实现，但仍缺关键 runtime 语义、跨后端闭环或验收 fixture。
- `[ ]` 未完成：当前仓库没有对应核心实现。

当前核对依据：

- `src/xr_viewer/gltf/` 已拆出 document/accessors/scene/materials/primitives/render_plan/validation/color_management。
- `tests/test_gltf_contract.py` 已覆盖 stable contract、Artemis/Bedroom 静态 smoke、controller smoke、render pass、extension audit、color policy。
- 当前 loader 仍在加载时把 node world transform 烘进 `vertices`，没有 animation/skin/morph/camera runtime 模块。

### Phase 0：[~] 冻结当前事实和回归样例

任务：

1. [x] 固定 Artemis 作为真实模型 smoke test。
2. [~] 记录 primitive count、texture count、light count、animation count、channel count、alphaMode 分布、vertex layout width、local/world bounds。
3. [x] 增加 loader diagnostics 输出函数。
4. [~] preview / OpenGL / D3D11 打印同一个 active model summary。

当前状态：

- 已有 `diagnose_gltf_model()` / `summarize_gltf_scene()`，真实 Artemis/Bedroom smoke test 锁定 primitive、texture、light、alphaMode、render pass、vertex width。
- 未完成 animation/channel count 诊断；summary 也只报告 world-space scene bounds，没有 local/world 动态矩阵状态。
- OpenGL / D3D11 已使用共享 render/material contract，但“同一个 active model summary”日志闭环仍不完整。

验收：

```text
Artemis:
  primitive count stable
  texture count stable
  light count stable
  animation/channel count reported
  vertex stride = 10
```

### Phase 1：[x] 统一 mesh contract

任务：

1. [x] 明确 `vertices` 固定为 `N x 10 float32`。
2. [x] tangent 固定为 `N x 4 float32`。
3. [x] preview VAO、OpenGL env/controller VAO、D3D11 input layout 全部按 contract 读取。
4. [x] 增加测试防止 8-float 读取 10-float 数据。

当前状态：

- `GltfPrimitive` / `validate_mesh_contract()` 已固定 10-float vertex 和 4-float tangent。
- `test_validate_mesh_contract_rejects_legacy_eight_float_vertices()` 已防止回退到 8-float。
- `OPENGL_VERTEX_FORMAT`、`D3D11_VERTEX_STRIDE_BYTES`、`D3D11_VERTEX_OFFSETS_BYTES` 已作为共享常量测试。

验收：

- 所有环境模型和控制器模型加载后 vertex shape 一致。
- preview 不再出现放射状平面。
- OpenGL / D3D11 渲染同一模型的几何 bounds 一致。

### Phase 2：[~] 统一 material contract

任务：

1. [x] 集中 glTF material 解析：

```python
parse_gltf_material(gltf, material_index, texture_registry) -> GltfMaterial
```

2. [~] OpenGL / D3D11 / controller / environment 不再分别解释 material 字段。
3. [x] 明确 texture slot color space。
4. [x] 统一 `KHR_materials_unlit` 为 extension presence。
5. [x] 统一 `KHR_texture_transform` 到 texture binding。

当前状态：

- `parse_gltf_material()` 已输出 `GltfMaterial`，覆盖 base color、metallic/roughness、normal、occlusion、emissive、alphaMode、doubleSided、unlit、texture transform。
- controller material 已要求 `GltfMaterial` contract。
- OpenGL / D3D11 仍保留 backend 侧字段打包和 shader 参数解释，因此标为部分完成；需要继续收敛到单一 renderer-facing material contract。

验收：

- 控制器材质和房间材质走同一 material contract。
- OpenGL / D3D11 对 base color、unlit、alphaMode 的解释一致。
- 颜色发白回归测试继续通过。

### Phase 3：[~] 统一 render pass

任务：

1. [~] 根据 material 分类：

```text
alphaMode OPAQUE -> opaque
alphaMode MASK -> mask
alphaMode BLEND -> transparent
sky/background naming or profile marker -> sky
```

2. [x] OpenGL preview、OpenGL runtime、D3D11 runtime 都按 render plan 绘制。
3. [x] transparent pass 关闭 depth write。
4. [x] transparent pass 增加 back-to-front sorting。
5. [x] SkyBox 不再作为普通 opaque geometry 误遮挡场景。

当前状态：

- `classify_render_pass()`、`build_render_plan()`、`sort_transparent_primitives()` 已共享。
- preview、OpenGL environment renderer、D3D11 native renderer 已使用 sky/opaque/mask/transparent 分类和透明排序。
- SkyBox 当前依赖 profile 显式标记，不再靠命名启发式；因此文档里的“sky/background naming”部分未按原计划实现，实际策略是 profile marker。

验收：

- Artemis transparent planes 不再遮挡为实心平面。
- preview 与 runtime 的 pass 分类一致。
- D3D11 / OpenGL 不因 BLEND primitive 改动导致材质差异。

### Phase 4：[~] 统一 color management

任务：

1. [x] 明确 shader 输入纹理色彩空间。
2. [~] OpenGL / D3D11 baseColor、emissive 做相同 sRGB->linear。
3. [~] 输出 gamma / tone mapping 策略统一。
4. [~] preview 可提供独立 exposure/gamma，但不能改变 material contract。

当前状态：

- `color_management.py` 已定义 base/emissive 为 sRGB、normal/occlusion/mr 为 linear，并输出 diagnostics。
- D3D11 shader 侧已有 `gltfSrgbToLinear` / `gltfLinearToOutput` 相关测试。
- 仍需实际逐项确认 preview/OpenGL/D3D11 的 exposure/gamma/tone mapping 完全一致，不能只算 policy 层完成。

验收：

- 控制器按钮颜色在 OpenGL / D3D11 一致。
- 房间贴图颜色与标准 viewer 接近。
- 不再用 backend-specific 临时 gamma 补丁修颜色。

### Phase 5：[x] extension audit 与 fail fast

任务：

1. [x] `load_gltf_scene()` 返回 diagnostics：extensionsUsed、extensionsRequired、unsupportedRequired、unsupportedOptional、materialExtensions、primitiveExtensions。
2. [x] required unsupported 时停止加载该模型，并打印明确原因。
3. [x] optional unsupported 只警告。

当前状态：

- `audit_gltf_extensions()` / `raise_unsupported_required_extensions()` 已实现。
- Draco、Meshopt、GPU instancing required 路径有明确 remediation hint 和测试。

验收：

- 带 Draco / Meshopt required 的模型不会错误显示。
- 日志能明确告诉用户需要转码或安装解码支持。

### Phase 6：[~] 真实模型回归集

建议模型集：

| 状态 | 模型 | 覆盖点 |
|---|---|---|
| [ ] | SimpleTriangle.glb | 最小 geometry |
| [ ] | ExternalBuffer.gltf + .bin | 外部 buffer |
| [x] | UnlitEmptyExtension.glb | `KHR_materials_unlit: {}` |
| [x] | AlphaBlendPlanes.glb | BLEND pass |
| [~] | AlphaMaskFoliage.glb | MASK / alphaCutoff |
| [x] | TextureTransform.glb | `KHR_texture_transform` |
| [ ] | AnimatedTRS.glb | translation / rotation / scale animation |
| [ ] | AnimatedWeights.glb | morph target weights animation |
| [ ] | SimpleSkin.glb | joints / inverseBindMatrices / skinning |
| [ ] | SparseAccessor.glb | sparse accessor |
| [ ] | InterleavedAttributes.gltf | bufferView byteStride |
| [ ] | Cameras.glb | perspective / orthographic camera |
| [x] | Artemis | 真实复杂场景 |
| [x] | Bedroom | 当前生产房间 |
| [x] | Controller models | 控制器材质和动画节点 |

当前状态：

- 已有 Artemis、Bedroom、controller models 的真实 smoke。
- 合成 fixture 主要以内联 JSON/临时 glTF 覆盖 material/render/extension；没有看到标准命名的 animation/skin/morph/sparse/interleaved/camera fixture。

### Phase 7：[ ] Core animation runtime

任务：

1. [ ] 新增 `animation.py`，解析 `animations[*].channels` 和 `animations[*].samplers`。
2. [ ] 支持 target path：`translation`、`rotation`、`scale`、`weights`。
3. [ ] 支持 interpolation：`STEP`、`LINEAR`、`CUBICSPLINE`。
4. [ ] 每帧按 animation time 采样 node TRS / weights。
5. [ ] 采样后重建 node local matrix，并从 scene roots 向下更新 world matrix。
6. [ ] animated node primitive 不再把 node world matrix 永久烘进顶点。
7. [ ] OpenGL / D3D11 / preview 绘制时使用当前帧 model matrix。

当前状态：

- 当前 `load_glb_model()` 仍对 mesh primitive 应用 node world matrix，输出 world-space `vertices`。
- 没有 `animation.py`、`GltfAnimation`、runtime sampler 或 animated primitive local-space path。
- 现有 controller button press 逻辑是项目自定义控制器动画，不等于 glTF core animation runtime。

验收：

- Artemis GLB 中的 satellite / spaceship transform animation 在 preview、OpenGL、D3D11 可见。
- 合成 `AnimatedTRS.glb` 在 0s / half duration / loop boundary 的矩阵采样与预期一致。
- 不带 animation 的模型仍走静态快路径。

### Phase 8：[ ] Skinning

任务：

1. [ ] 解析 `skins`、`joints`、`inverseBindMatrices`。
2. [ ] 读取 `JOINTS_0` / `WEIGHTS_0`，并验证权重归一化。
3. [ ] 每帧计算 joint matrix palette。
4. [ ] OpenGL / D3D11 shader 增加 joint indices / weights input 和 joint palette uniform/structured buffer。
5. [ ] 没有 skin 的 primitive 不走 skinning shader 分支。

当前状态：

- 未看到 skinning 模块、joint palette、`JOINTS_0` / `WEIGHTS_0` 消费或 shader path。

验收：

- `SimpleSkin.glb` 骨骼动画姿态与标准 viewer 接近。
- 无 skin 模型的 vertex layout / shader path 不回退。

### Phase 9：[ ] Morph targets

任务：

1. [ ] 读取 primitive `targets` 的 POSITION / NORMAL / TANGENT delta。
2. [ ] 合并 mesh weights、node weights、animation weights。
3. [ ] OpenGL / D3D11 支持 morph target 应用；少量 target 可 shader path，多 target 可 CPU/dynamic VBO fallback。
4. [ ] morph 后 normal/tangent 保持可用，必要时重新归一化。

当前状态：

- 未看到 morph target parser、weights runtime 或 backend 应用路径。

验收：

- `AnimatedWeights.glb` 的形变动画可播放。
- morph target 缺失 normal/tangent 时有明确 fallback。

### Phase 10：[~] Accessor / buffer 完整性

任务：

1. [x] accessor 统一支持 componentType、type、count、byteOffset。
2. [x] bufferView 支持 byteStride/interleaved attributes。
3. [x] 支持 normalized integer attribute 转换。
4. [x] 支持 sparse accessor 覆盖。
5. [x] 对越界 bufferView/accessor fail fast。

当前状态：

- `_get_accessor()` 已支持 componentType/type/count/offset、interleaved `byteStride`、normalized integer、sparse accessor、越界报错。
- 但 Phase 6 中对应 `SparseAccessor.glb`、`InterleavedAttributes.gltf` 回归 fixture 还没落地，所以阶段整体标为部分完成。

验收：

- `SparseAccessor.glb`、`InterleavedAttributes.gltf` 正确加载。
- 访问越界的坏模型报明确错误，不产生随机几何。

### Phase 11：[~] Cameras 与多 scene

任务：

1. [x] 解析 active scene 和多 scene 列表。
2. [ ] 解析 perspective / orthographic camera。
3. [ ] camera node 使用 runtime world matrix。
4. [ ] preview 可以选择 glTF camera；OpenXR environment profile 可选择忽略或映射 camera，但 diagnostics 必须报告。

当前状态：

- `_iter_scene_mesh_nodes()` 已按 glTF active scene 选择 roots，能避免遍历非 active scene mesh。
- 未看到 camera parser、camera diagnostics、preview camera selection 或 OpenXR camera 映射。

验收：

- `Cameras.glb` diagnostics 能列出 camera 类型、FOV/ortho 参数和 world pose。
- active scene 选择符合 glTF 默认规则。

### Phase 12：[ ] Core compliance diagnostics

任务：

1. [ ] diagnostics 输出 core feature coverage：

```text
hasAnimations
hasSkins
hasMorphTargets
hasSparseAccessors
hasInterleavedAccessors
hasCameras
animatedNodeCount
skinnedPrimitiveCount
morphedPrimitiveCount
```

2. [ ] 对“core 但尚未支持”的语义禁止静默忽略；必须 warn 或 fail fast，取决于是否会错误显示。
3. [x] extension diagnostics 与 core diagnostics 分开，避免把 core 缺口误标成 extension 缺口。

当前状态：

- extension diagnostics 已独立实现。
- core feature coverage 尚未输出 animation/skin/morph/sparse/interleaved/camera 状态；带这些 core 语义的模型仍可能被当成普通静态模型处理。

验收：

- Artemis 动画 GLB diagnostics 明确显示 animation count / channel count / animated node count。
- 带 skin/morph/sparse 的模型在未支持路径不会被当作普通静态模型错误显示。

## 9. 验证命令

```powershell
src\python3\python.exe -m py_compile src\xr_viewer\gltf_loader.py src\xr_viewer\gltf\*.py src\tools\preview_room_layout.py
src\python3\python.exe -m pytest tests\test_keyboard_screen_preset.py tests\test_gltf_contract.py -q
src\python3\python.exe -c "import sys, pytest; sys.path.insert(0, 'src'); raise SystemExit(pytest.main(['tests/test_openxr_runtime.py','-q']))"
```

Artemis static smoke check：

```powershell
src\python3\python.exe -c "import sys; sys.path.insert(0,'src'); from xr_viewer.gltf import load_glb_model; prims,tex,lights=load_glb_model('src/xr_viewer/environments/Artemis/environment.glb'); print(len(prims), len(tex), len(lights), prims[0]['vertices'].shape)"
```

Artemis animation smoke check：

```powershell
src\python3\python.exe -c "from pygltflib import GLTF2; g=GLTF2().load('src/xr_viewer/environments/Artemis/environment.glb'); print(len(g.animations or []), sum(len(a.channels or []) for a in (g.animations or [])))"
```

preview 检查：

```powershell
src\python3\python.exe src\tools\preview_room_layout.py Artemis
```

## 10. 第一批落地任务

1. [x] 保留现有 `src/xr_viewer/gltf/` package，继续把 legacy `gltf_loader.py` 作为 facade。
2. [ ] 增加 core feature diagnostics：animation / skin / morph / sparse / interleaved / cameras。
3. [ ] 新增 animation parser 和 TRS sampling，先让 Artemis 标准 GLB 动画可见。
4. [ ] 把 animated primitive 从一次性 world-space VBO 改为 local-space VBO + runtime model matrix。
5. [~] preview、OpenGL、D3D11 支持 per-primitive model matrix。
6. [ ] 增加 sparse/interleaved accessor fixture，堵住 accessor 层 core 缺口。
7. [ ] 后续再接 skinning、morph target、camera 消费。
8. [ ] 评估 Filament `gltfio` 作为长期替代后端，避免继续自研完整 viewer。

当前优先级：

1. 先补 core feature diagnostics，避免 animation/skin/morph/sparse/camera 被静默忽略。
2. 再做 animation parser + local-space animated primitive path，这是 Artemis 卫星/飞船动画可见的前置条件。
3. sparse/interleaved accessor 代码已写，下一步应补 fixture 锁住行为。

## 11. 完成标准

1. 更换符合 glTF 2.0 core 的普通模型时，不再出现 vertex layout 错读导致的几何爆炸。
2. `alphaMode`、`doubleSided`、core metallic-roughness material、texture sampler、texture transform 在 preview / OpenGL / D3D11 一致。
3. glTF core animation 的 TRS / weights 可播放；animated node 不被静态烘死。
4. glTF core skinning 可播放；无 skin 模型保持静态快路径。
5. glTF core morph target 可播放；weights animation 可驱动 morph。
6. sparse accessor、interleaved buffer、normalized attribute 正确加载。
7. camera / active scene diagnostics 符合 glTF 默认规则。
8. unsupported required extensions 有明确日志和失败路径。
9. core 语义尚未支持时不能静默错误显示，必须 diagnostics warn 或 fail fast。
10. 用户不需要通过观察画面猜测是模型问题、loader 问题还是 render pass 问题。
11. 新模型接入流程变成：

```text
放入 environment.glb / profile.json
运行 diagnostics
若 unsupported required 为空，进入 preview
若含 animation/skin/morph/sparse/camera，diagnostics 明确显示支持状态
preview 与 OpenXR summary 一致
再做 profile 位置/灯光适配
```
