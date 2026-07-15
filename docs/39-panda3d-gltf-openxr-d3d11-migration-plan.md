# Panda3D glTF 场景运行时迁移方案：OpenGL 离屏渲染到 D3D11 OpenXR

## 1. 决策与目标

本方案用 Panda3D + `panda3d-gltf` 承担房间、天空盒、控制器等 `.glb` 场景的加载、scene graph、动画和 PBR 渲染；Panda3D 固定使用 OpenGL 离屏渲染，再把每眼结果交给现有 D3D11 OpenXR Projection swapchain。

它取代的是当前 `src/xr_viewer/gltf/`、`environment_model.py`、`environment_renderer.py` 及其 OpenGL/D3D11 场景上传/绘制分支，**不替换**以下已经验证的部分：

- 2D→3D 深度推理、双目生成和 latest-frame 调度。
- OpenXR session、视图定位、控制器输入、手柄射线和屏幕拖拽状态机。
- `D3D11NativeRenderer` 所有权下的 OpenXR D3D11 session / swapchain 创建与提交。

迁移目标是停止为完整 glTF 2.0 scene graph、动画、skin、morph 和 PBR 语义继续扩写项目自研 renderer，同时保留 D3D11 OpenXR 的稳定提交路径。

## 2. 现状与必须修正的前提

当前仓库已经有两条不同的 D3D11 相关能力，不能混为一谈：

| 能力 | 当前状态 | 迁移中的角色 |
|---|---|---|
| D3D11 OpenXR Projection swapchain | 已由 `D3D11NativeRenderer` / `ProjectionLayerPresenter` 使用 | 最终提交目标，继续保留 |
| OpenGL → D3D11 纹理互操作 | 已有 `WGL_NV_DX_interop2` 路径；将 D3D11 swapchain texture 注册为 GL texture/FBO 后直接绘制 | 首选桥接实验路径 |
| CUDA ↔ OpenGL runtime texture upload | 已存在 CUDA/GL 图像互操作相关路径 | 可复用的 GPU 侧资源管理经验 |
| CUDA ↔ D3D11 图像互操作 | 当前没有作为 Panda3D 输出桥的可用实现 | 仅作为 NV 互操作不可用时的第二阶段 native bridge |

因此，不能把“CUDA GL→D3D11 copy”描述成已有能力。当前 `core_d3d_interop.py` 的 D3D11 projection 互操作是 NVIDIA 的 `WGL_NV_DX_interop2`，并且失败时会跳过该 Projection 互操作路径。Panda3D POC 必须先证明其 OpenGL context 可安全参与这条路径。

Panda3D 官方支持的成熟图形后端为 OpenGL 与 Direct3D 9，没有 D3D11 后端；本方案强制 `load-display pandagl`，不尝试使用 `pandadx9`，更不修改 Panda3D 内核。

## 3. 目标架构

```text
existing capture / depth / stereo runtime
                 |
                 | latest completed screen frame
                 v
      Panda screen dynamic texture + scene NodePaths
                 |
                 | Panda3D OpenGL renderer
                 v
      left/right offscreen render targets (RGBA8 / optional HDR)
                 |
      +----------+------------------------------+
      |                                         |
      | preferred: WGL_NV_DX_interop2          | fallback: native CUDA bridge
      | lock D3D11 swapchain texture as GL FBO | CUDA GL image -> CUDA D3D11 image
      v                                         v
OpenXR-owned D3D11 left/right swapchain textures
                 |
                 v
       existing xrEndFrame Projection layer submit
```

Panda3D 的职责只到“画出当前帧的左右眼场景”。OpenXR 的 acquire/wait/release、layer composition、frame timing 和 D3D11 device 生命周期仍必须由现有 D3D11 backend 统一管理。这样不会出现两个模块同时拥有或释放同一 swapchain texture 的问题。

## 4. Panda3D 场景侧设计

### 4.1 初始化与资源边界

在新模块中使用无可见窗口的 Panda3D OpenGL pipe：

```python
load_prc_file_data("d2s-panda", """
window-type offscreen
load-display pandagl
audio-library-name null
sync-video false
show-frame-rate-meter false
""")
```

初始化后创建单一 `ShowBase`/`GraphicsEngine` 实例；它和全部 Panda `Texture`、`GraphicsBuffer`、`NodePath` 必须在同一个渲染线程使用。不得在 OpenXR 主线程和 Python worker 之间传递 Panda 对象。

- `panda3d-gltf` 负责 `environment.glb`、控制器 GLB 的加载；加载后保留根 `NodePath`，不要把 node transform 烘入顶点。
- 左右眼使用两个独立的 offscreen color target 与 camera，或使用一个明确验证过的 stereo buffer。每眼投影矩阵必须由同一帧的 `xrLocateViews` FOV 生成，不能沿用桌面 preview 的固定相机。
- 屏幕、控制器和环境均作为可独立更新的 `NodePath`。现有屏幕位姿、握持拖动和控制器 pose 继续在项目状态机中计算，再写入对应 `NodePath`。
- 动态屏幕内容使用 Panda `Texture`。上传端必须只消费最新已完成的双目帧；不得让 Panda render 等待推理或 CUDA 生产者。

### 4.2 glTF 功能验收，不先假定“黑盒完全正确”

`panda3d-gltf` 提供 glTF 2.0/GLB、动画转换和基础 PBR viewer；但项目仍需以实际资产验收，而不是把库名当作完整 core-compliance 证明。

最小资产集：

1. Artemis：天空盒、透明/双面材质、土星环、卫星/飞船 15 秒动画。
2. Bedroom：静态 PBR、纹理 sampler、lightmap/UV1。
3. 当前 Pico/Index/HP 控制器：透明按钮、normal/metallic-roughness/emissive。
4. Khronos 官方 animation、skinning、morph、sparse/interleaved accessor 和 camera fixture。

每个资产都要保存 Panda 截图、当前 renderer 截图、Khronos Sample Viewer 截图，并比对 animation channel 数、node transform、alpha 和颜色；差异必须记录为 loader/材质配置/桥接问题，不能再在单个模型上加入私有补丁。

### 4.3 天空盒与色彩

天空盒不应当作为普通不透明实体遮住房间。Panda 场景侧应将它做成 camera-relative 的 background/cubemap 或按 profile 显式定义：关闭深度写入、允许内侧可见、在实体前的 background pass 渲染。原 PNG、星空合成图、MIME 与导出质量问题仍属于 Unity→GLB 资产管线，Panda 只消费已验证的 GLB。

Panda 颜色管理必须单独验证：base color/emissive 按 sRGB 解释，normal、occlusion、metallic-roughness 按线性数据解释；最终 RT 格式、sRGB view 和 D3D11 swapchain format 组成一个端到端色彩契约。不能同时在 Panda 和 D3D11 中做两次 gamma/tone mapping。

## 5. OpenGL 到 D3D11 桥接方案

### 5.1 首选：复用 NV_DX_interop2 直接渲染

每个 OpenXR acquire 的 `(eye, image_index)` 对应一个长期缓存的 D3D11 texture。第一阶段复用现有注册/lock/unlock 逻辑，把它映射为 GL texture/FBO；Panda render 时将该 FBO 作为该眼的目标，而不是先渲染到 Panda texture 再复制。

这是最短路径，但只有同时满足下列条件才可采用：

1. Panda 的 OpenGL context 与 WGL registration context 兼容，且与 D3D11/OpenXR 使用同一 NVIDIA adapter。
2. Panda 可以在目标 FBO 被 lock 期间完成该眼全部绘制，随后完成 GPU flush/fence，再 unlock。
3. OpenXR `release_swapchain_image` 一定发生在 unlock 后；不能跨帧持有 runtime texture。
4. 左右眼、尺寸变化、session 重建和 device lost 都能正确注销和重新注册缓存。

该路径不是零风险：Panda3D 管理自己的 `GraphicsStateGuardian` 和 framebuffer 状态。若无法在不侵入 Panda 内部的前提下把 OpenXR texture 设为它的可靠 render target，则停止此路径，不通过裸 OpenGL 调用强行篡改 Panda state。

### 5.2 后备：CUDA GL→D3D11 native bridge

若 5.1 因 context/FBO 所有权失败，Panda 先渲染到它自己的 RGBA OpenGL texture；随后由一个小型原生 Windows DLL 完成：

1. `cudaGraphicsGLRegisterImage` 注册 Panda 左/右 GL color texture。
2. `cudaGraphicsD3D11RegisterResource` 注册当前 acquire 的 OpenXR D3D11 swapchain texture。
3. 同一 CUDA stream 中 map 两端 `cudaArray`，进行 device-to-device array copy 或 kernel copy。
4. 在 stream 完成后 unmap 两端，再由 D3D11/OpenXR release 该 image。

该 DLL 提供窄接口：`register_gl_texture`、`register_d3d11_texture`、`copy_eye`、`unregister/rebuild`。Python 只传纹理 ID、D3D11 pointer、尺寸和同步 token；不在 ctypes/Python 中手写 CUDA 结构体或让 CUDA resource 跨 session 存活。

CUDA bridge 的 POC 需逐项证明：同 adapter、RGBA 格式、行方向、MSAA resolve、resize、GPU fence 顺序和 device reset。它可以避免 CPU 回读，但并不自动等于零拷贝：GL 与 D3D11 是不同 API 资源，至少会有一次 GPU copy。

### 5.3 明确不采用的路径

- 不使用 CPU `glReadPixels`、PBO readback、PIL/Numpy 中转到 D3D11；它违反实时渲染目标。
- 不让 Panda3D 使用 Direct3D 9 再尝试 DX9→DX11 级联。
- 不修改 Panda3D 源码来增加 D3D11 backend。
- 不让 Panda 和现有 Moderngl renderer 同时渲染同一个环境；切换必须是 renderer ownership 的互斥选择。

## 6. 分阶段替换计划

### Phase 0：可行性闸门

当前状态（2026-07-15）：已新增 `src/tools/panda3d_gltf_probe.py`、`src/tools/panda3d_animation_screenshot_probe.py`、`src/tools/panda3d_material_probe.py`、`src/tools/panda3d_offscreen_probe.py`、`src/tools/panda3d_d3d11_interop_probe.py`、`xr_viewer.panda3d_probe`、`xr_viewer.panda3d_animation_screenshot_probe`、`xr_viewer.panda3d_material_probe`、`xr_viewer.panda3d_node_animation`、`xr_viewer.panda3d_offscreen_probe` 和 `xr_viewer.panda3d_d3d11_interop_probe`。本机使用 Panda3D 1.10.15、panda3d-gltf 1.3.0 检查当前 Artemis `environment.glb`，发现 GLB 本身有 **19** 个 animation、**38** 条 channel、**19** 个 animation target node，且这 **19** 个 target 全部属于 active scene。`panda3d-gltf` 原生载入结果仍为 **0** 个 `Character`、**0** 个 `AnimBundleNode`，但自定义 glTF node animation runtime 已能绑定 **19/19** 个 target node、采样 **38** 条 channel，并确认动画时长为 **15.0 秒**；`--strict-animation` 返回退出码 0。Phase 0 probe 现在通过 `GltfNodeAnimationPlayer` 按 **0.0 / 7.5 / 15.0 秒** 推进 Panda NodePath，并在 JSON 中记录采样节点与 transform 变化结果；动画截图 probe 可在相同采样时间输出 3 张 PNG，并记录每张截图的路径、SHA-256 和字节大小。材质语义 probe 已记录 Artemis **44** 个 material、**19** 个 image、**19** 个 texture，alpha 分布为 **BLEND 28 / MASK 1 / OPAQUE 15**，**44/44** 个 material 使用 `KHR_materials_unlit`，并定位天空盒材质 `GLTF_UnlitSkybox_GLTF_Skybox_Composite_0`。当前 HP/INDEX/PICO/QUEST/VIVE/YVR 左右手共 **12** 个控制器 GLB 均可由 Panda3D 加载，均为静态资产，node/geom 数量非零，animation runtime ready。Bedroom `environment.glb` 已清理一个越界 child node 引用（旧引用指向不存在的 node **206**），现在可无 warning 加载出 **416** 个 Panda node 和 **202** 个 Geom。Panda OpenGL offscreen 子闸门也已通过：`pandagl` 创建 64×64 render target，实际驱动为 **NVIDIA GeForce RTX 2060/PCIe/SSE2**，OpenGL **4.6.0 NVIDIA 596.36**，framebuffer 为 RGBA8 + depth24。Panda GL context 下的 D3D11/NV_DX readiness 已通过：D3D11 feature level **0xb000**，D3D11 adapter 枚举为 **NVIDIA GeForce RTX 2060**（vendor **0x10de**、device **0x1f03**、LUID **00000000:00087686**、dedicated VRAM **12646875136**），并确认 GL renderer 与 D3D11 adapter 名称匹配；`WGL_NV_DX_interop2` 函数可加载，`wglDXOpenDeviceNV` 可打开并关闭 D3D11 device；普通 **64×64 RGBA8 D3D11 Texture2D** 可注册为 GL texture、lock 成功，并通过 `GL_FRAMEBUFFER_COMPLETE`。Panda offscreen texture native id 已可获取，64×64 probe 记录 native id **1**。已加入 `D2S_PANDA3D_PHASE0_SWAPCHAIN_PROBE=1` 的真实 OpenXR D3D11 swapchain POC 路径：在 acquire/wait 后注册当前 swapchain texture、lock 为 GL FBO、画测试色块和三角形、unlock/release；Phase 0 仍需在头显/OpenXR runtime 下实际确认该路径无错误并可见。

- 安装锁定版本的 `panda3d` 与 `panda3d-gltf`，记录 Python ABI、GPU driver、Panda 版本和插件版本。
- 运行 `src/tools/panda3d_gltf_probe.py`：加载 Artemis/控制器并验证 glTF animation target 是否属于 active scene，以及 Panda runtime node 是否可驱动这些动画；Artemis 已由自定义 node animation runtime 通过，probe JSON 已记录 0.0/7.5/15.0 秒采样时间、采样节点和 transform changed 结果；`src/tools/panda3d_animation_screenshot_probe.py` 可保存 0.0/7.5/15.0 秒 PNG 截图并输出 SHA-256/大小摘要；`src/tools/panda3d_material_probe.py` 可输出 alpha、double-sided、unlit、texture/image 和 skybox material 摘要；12 个控制器 fixture 已通过，Bedroom missing-node warning 已通过清理越界 child 引用解决。
- 记录 Panda OpenGL vendor/renderer、Panda texture native handle 的可获取性、实际 offscreen texture format、同 adapter CUDA/D3D11 枚举结果；当前已记录 vendor/renderer/version、RGBA8/depth24 offscreen RT、Panda texture native id、D3D11 adapter description/vendor/device/LUID/VRAM、GL/D3D adapter 名称匹配、D3D11 feature level、NV_DX device open/close，以及普通 D3D11 Texture2D 的 register/lock/FBO complete。
- 使用现有 D3D11 OpenXR session 做 NV_DX interop POC；当前已验证普通 D3D11 texture 注册，并已提供 `D2S_PANDA3D_PHASE0_SWAPCHAIN_PROBE=1` 的真实 OpenXR swapchain 测试图路径。下一步需在头显运行时确认 acquire、注册、lock、渲染、unlock、release 全链路成功，先不加载真实 GLB。

闸门：任何一个 asset 的加载/动画/透明正确性失败，或 GL→D3D11 不能完成单帧 acquire-render-release，则保留当前 renderer，先修 POC，不开始替换。

### Phase 1：新 renderer 适配层，不改默认路径

当前状态（2026-07-15）：已新增 `src/xr_viewer/panda_runtime/` 的 import-light 适配层骨架，包含 `runtime.py`、`scene.py`、`stereo_targets.py`、`bridge.py`、`diagnostics.py`；`PandaSceneRenderer` 已定义 `load_environment`、`load_controller`、`update_frame_state`、`render_eyes`、`rebuild_targets`、`release` facade 契约。`scene.py` 默认只记录资产路径，启用 `load_panda_assets=True` 时会懒加载 `panda3d-gltf`、保留内部 root ownership，并记录 node/geom 计数，不向 facade 外暴露 `NodePath`。`stereo_targets.py` 默认只记录左右眼 target spec，启用 `create_panda_targets=True` 时会在单个 Panda `ShowBase` 下创建左右眼 offscreen buffer/texture，并记录 texture native id。`bridge.py` 已定义以 `(session_generation, eye, image_index, width, height, format)` 为边界的 `SwapchainResourceKey` 和明确的未实现失败契约，后续 NV_DX/CUDA bridge 必须复用该缓存策略。`diagnostics.py` 已可生成 runtime snapshot/JSON，汇总 scene assets、stereo target refs、bridge resource keys、最新 frame predicted display time 与 animation time。`PandaAnimationClock` 已加入 facade，`update_frame_state()` 会把 XR `predicted_display_time` 派生为从首帧起算且不倒退的 `animation_time_seconds`，并把同一个 bound frame snapshot 传给 scene 与 bridge；`PandaFrameState` 已增加 `frame_index`、`PandaEyeView` 和 `PandaPose` 契约，更新帧时会校验两眼 eye index 与同一 snapshot 边界，diagnostics 会记录 frame index、eye view count、controller count 和 screen pose presence。`PandaSceneGraph` 现在会从同一 frame snapshot 派生 `PandaSceneSnapshot`，记录 controller hands、screen pose、screen texture 和 eye view count；当 controller root 已由 Panda 加载时，会把 `PandaPose(position xyz, orientation xyzw)` 应用到对应 root 的 `set_pos_quat()`，并记录 applied controller hands；screen root 可通过 `attach_screen_root()` 接入，并在 frame update 时消费同一个 `screen_pose`；`PandaScreenTextureFrame` 已定义最新屏幕纹理快照，screen texture target 可通过 `attach_screen_texture_target()` 接入并在同一 frame snapshot 中更新，diagnostics 会记录尺寸、格式、native id 可用性和 applied 状态；`PandaControllerRay` 已定义手柄射线视觉快照，controller ray target 可通过 `attach_controller_ray_target()` 接入并随同一 frame snapshot 更新，diagnostics 会记录 ray hand count 和 applied hands。`PandaSceneGraph(load_panda_assets=True)` 会为含 glTF node animation 的资产创建 `GltfNodeAnimationPlayer`，并在 frame update 时用同一个 `animation_time_seconds` 驱动 Panda NodePath。diagnostics 现在记录每个 scene asset 的 animation channel/target/bound node/duration 摘要。`D2S_GLTF_RENDERER=native|panda3d` selector 已加入，默认仍为 `native`；在真实 OpenXR swapchain gate 未通过前，请求 `panda3d` 会记录原因并回退 native，不会替换现有 D3D11 native renderer。

新增 `src/xr_viewer/panda_runtime/`，建议边界如下：

```text
panda_runtime/
  runtime.py       # Panda process/thread lifecycle and renderer facade
  scene.py         # environment/controller/screen NodePath ownership
  stereo_targets.py# eye cameras, FOV/projection, render-target lifecycle
  bridge.py        # NV_DX bridge facade; optional native CUDA backend binding
  diagnostics.py   # asset/runtime/bridge summary and screenshots
```

定义与现有 viewer 脱钩的 `PandaSceneRenderer` 接口：`load_environment`、`load_controller`、`update_frame_state`、`render_eyes`、`rebuild_targets`、`release`。输入为已有的 head/eye/controller/screen pose snapshot，输出为“已填充的两个 D3D11 swapchain image”，不暴露 Panda `NodePath` 给外部。

新增 renderer 选择器，例如 `D2S_GLTF_RENDERER=native|panda3d`，默认 `native`。Panda 初始化失败、bridge 失败、设备变更时要打印一次明确原因并回退 native；不得静默输出黑屏。

### Phase 2：功能等价

- 先迁移 Artemis 和一个控制器，不迁移可选 glow/全景特效。
- 接入共享屏幕位姿、控制器 pose、手柄射线视觉和 screen texture 更新；手柄拖动逻辑仍在现有控制器状态机。当前 `PandaSceneGraph` 已把同帧 `controller_poses`、`controller_rays`、`screen_pose`、`screen_texture` 与 eye view count 收敛为 `PandaSceneSnapshot`，并能把已加载 controller root、已 attached controller ray target、已 attached screen root 与已 attached screen texture target 更新到对应同帧输入；下一步再接真实 Panda screen material/texture upload 和 controller ray geometry。
- 将 glTF animation clock 绑定到 XR predicted display time，避免每眼/每线程各走一个时钟；当前 facade 已完成 clock 派生，`PandaSceneGraph(load_panda_assets=True)` 已能创建并驱动真实 Panda node animation player，diagnostics 会记录 Artemis 38 channels / 19 targets / 19 bound nodes / 15.0 秒。
- 把 profile 保留为资产布局、尺度、sky/background、光照和默认视角配置；不要把 model-specific 动画逻辑重新塞回 profile。
- 以同一帧 snapshot 更新两眼相机与所有 NodePath，禁止左眼/右眼读到不同 controller/screen pose；当前 facade 已定义 `PandaFrameState(frame_index, eye_views, controller_poses, screen_pose)` 并校验两眼 eye index，diagnostics 可输出同帧 snapshot 摘要。

### Phase 3：接入与性能

- 优先打通 NV_DX 路径；仅在失败后启用 CUDA bridge，并在日志中记录实际桥接模式。
- 所有 swapchain 资源缓存以 `(session_generation, eye, image_index, width, height, format)` 为 key；session 重建先清资源，再创建。
- 统计每帧 Panda render、bridge、OpenXR acquire/release、submit 的 GPU/CPU 时间，区分首次资源创建与稳态。
- 让 Panda 使用最新已完成屏幕帧；旧帧可继续作为环境光或 screen texture，不能反压 capture/inference 队列。

### Phase 4：切换默认与清理

只有满足第 7 节全部验收条件后，才把 `panda3d` 设为默认。之后再删除被替代的环境/控制器自研上传和 shader 分支；`src/xr_viewer/gltf/`、`docs/38` 的 parser/diagnostics 可保留作资产审计工具，直到 Panda 路径的同等 diagnostics 完备。

## 7. 验收矩阵

| 项目 | 自动证据 | 真机证据 |
|---|---|---|
| GLB load | Artemis/Bedroom/控制器 fixture 成功，输出 primitive/node/animation 摘要 | 无缺 mesh/纹理 |
| 动画 | 0/7.5/15 秒三张截图，satellite/spaceship transform 不同 | 卫星连续绕行，无跳变 |
| 材质与天空盒 | alpha、double-sided、unlit、texture/image、skybox material 摘要；后续补 sRGB 截图差异阈值 | 土星环与星空同时可见，不被白球/黑盒遮挡 |
| XR 位姿 | 同一 snapshot 的两眼 view/projection 单测 | 头动稳定、控制器与射线对齐 |
| 屏幕交互 | 原有 grip drag 状态机回归测试 | 左右手柄拖动无漂移、无明显延迟 |
| D3D11 bridge | acquire → render → release 顺序、resize/session reset 测试 | Virtual Desktop 下左右眼稳定显示 |
| 性能 | 分开报告 Panda render/bridge/submit p50/p95 | 不低于当前 D3D11 native 目标帧率，且无 capture backpressure |
| 故障回退 | 强制 bridge 初始化失败后自动切回 native 并有原因日志 | 不黑屏、不崩溃 |

## 8. 与 docs/38 的关系

`docs/38-gltf-2-renderer-compliance-layer-plan.md` 仍描述当前自研 static contract 及其 animation/skinning/morph 等缺口。本方案是其“优先复用成熟 runtime/renderer”的具体候选实施，不宣称 Panda3D 已经替项目满足所有 glTF 2.0 core 细节。

在 Panda POC 通过前，docs/38 的未完成项不能标记为完成；在 Panda POC 通过后，应按真实 fixture 结果更新 docs/38，而不是按 Panda3D 依赖已安装更新状态。

## 9. 外部依据

- Panda3D 的 `GraphicsOutput.makeTextureBuffer()` 提供离屏 render-to-texture；纹理在不同 GraphicsStateGuardian 间的共享有明确限制：<https://docs.panda3d.org/1.10/python/reference/panda3d.core.GraphicsOutput>
- Panda3D 官方 renderer feature table 列出的 Windows DirectX 后端为 Direct3D 9，OpenGL 是功能最完整的后端：<https://docs.panda3d.org/1.10/python/programming/rendering-process/supported-renderer-features>
- `panda3d-gltf` 提供 glTF 2.0/GLB 加载、动画转换及 `gltf-viewer`，但仍应以本项目 fixture 验证扩展与视觉语义：<https://github.com/Moguri/panda3d-gltf>
