# Panda3D glTF 场景运行时迁移方案：OpenGL 离屏渲染到 D3D11 OpenXR

## 1. 决策与目标

本方案用 Panda3D + `panda3d-gltf` 承担房间、天空盒、控制器等 `.glb` 场景的加载、scene graph、动画和 PBR 渲染；Panda3D 固定使用 OpenGL 离屏渲染，输出左右眼 3D 模型层 GPU 画面，再交给项目现有 OpenXR renderer/compositor 合成。

它取代的是当前 `src/xr_viewer/gltf/`、`environment_model.py`、`environment_renderer.py` 及其 OpenGL/D3D11 场景上传/绘制分支，**不替换**以下已经验证的部分：

- 2D→3D 深度推理、双目生成和 latest-frame 调度。
- OpenXR session、视图定位、控制器输入、手柄射线和屏幕拖拽状态机。
- `D3D11NativeRenderer` 所有权下的 OpenXR D3D11 session / swapchain 创建与提交。
- 虚拟屏幕纹理上传、CUDA/GL GPU 零拷贝路径、手柄激光、键盘、FPS 面板、OSD/overlay，以及最终 Projection layer composition。

迁移目标是停止为完整 glTF 2.0 scene graph、动画、skin、morph 和 PBR 语义继续扩写项目自研 3D 模型 renderer，同时保留项目现有 GPU 上传、VR UI 合成和 OpenXR 稳定提交路径。

### 1.1 渲染所有权边界

Panda3D 是 **3D 模型层 renderer**，不是整个 OpenXR compositor。后续实现必须保持这个边界：

| 领域 | 所有者 | 说明 |
|---|---|---|
| glTF/GLB 加载、scene graph、动画、PBR/alpha/skybox、房间和手柄模型渲染 | Panda3D | 这是引入 Panda3D 的原因，避免继续自研完整 glTF 2.0 renderer |
| 2D→3D runtime、虚拟屏幕 GPU texture upload、CUDA/GL 零拷贝 | 项目现有 runtime/renderer | Panda3D 不接管 screen texture 生产、上传或消费；虚拟屏幕不进入 Panda runtime 合同 |
| 手柄激光、键盘、FPS 面板、OSD、Quad/overlay 策略 | 项目现有 OpenXR UI/compositor | 这些仍由我们自己的 shader/overlay 路径合成，避免把可控 UI 和零拷贝路径交给 Panda 黑盒 |
| OpenXR session、swapchain acquire/wait/release、`xrEndFrame` | 项目现有 OpenXR backend | Panda3D 不拥有 OpenXR lifecycle |

因此，目标不是让 Panda3D 画完整 VR UI 后直接提交 OpenXR，而是让 Panda3D 输出可被我们 GPU 侧采样/合成的左右眼 3D 模型层。运行时代码不得保留 Panda screen/ray NodePath、screen texture upload target、controller ray target 等类或自动绑定，以免后续误把虚拟屏幕/激光所有权迁入 Panda。

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
Panda3D glTF scene graph / animation / PBR
                 |
                 | Panda3D OpenGL renderer
                 v
      left/right 3D model layer textures (color + optional depth)
                 |
      +----------+------------------------------+
      |                                         |
      | preferred: shared GL / NV_DX bridge    | fallback: native CUDA bridge
      | expose as compositor-readable texture  | GPU copy, no CPU readback
      v                                         v
existing OpenXR renderer/compositor
                 |
                 | compose Panda 3D layer + virtual screen + laser + keyboard + FPS/OSD
                 v
OpenXR-owned left/right projection swapchain textures
                 |
                 v
existing xrEndFrame Projection layer submit

OpenXR OpenGL fallback, only when D3D11 is unavailable or explicitly disabled:
Panda3D OpenGL renderer -> compositor-readable GL texture -> existing OpenXR GL compositor -> xrEndFrame
```

Panda3D 的职责只到“画出当前帧的左右眼 3D 模型层”。虚拟屏幕、激光、键盘、FPS/OSD、layer composition、OpenXR acquire/wait/release、frame timing 和 D3D11/OpenGL device 生命周期仍必须由现有 OpenXR backend 统一管理。这样不会出现 Panda3D 与项目 compositor 同时拥有 UI、GPU 上传或 swapchain texture 的问题。

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
- 控制器和环境作为可独立更新的 `NodePath`。控制器 pose 继续在项目状态机中计算，再写入 Panda 控制器 `NodePath`。虚拟屏幕、激光、键盘和 OSD 不属于 Panda3D 所有权，也不得在 Panda runtime 中保留自动绑定类。
- 动态屏幕内容的生产、GPU 上传和零拷贝仍由项目现有 renderer 管理。Panda runtime 不创建 screen `NodePath`、不接收 screen texture、不调用 `setRamImage()` 上传虚拟屏幕。

### 4.2 glTF 功能验收，不先假定“黑盒完全正确”

`panda3d-gltf` 提供 glTF 2.0/GLB、动画转换和基础 PBR viewer；但项目仍需以实际资产验收，而不是把库名当作完整 core-compliance 证明。

最小资产集：

1. Artemis：天空盒、透明/双面材质、土星环、卫星/飞船 15 秒动画。
2. Bedroom：静态 PBR、纹理 sampler、lightmap/UV1。
3. 当前 Pico/Index/HP 控制器：透明按钮、normal/metallic-roughness/emissive。
4. Khronos 官方 animation、skinning、morph、sparse/interleaved accessor 和 camera fixture。

每个资产都要保存 Panda 截图、当前 renderer 截图、Khronos Sample Viewer 截图，并比对 animation channel 数、node transform、alpha 和颜色；差异必须记录为 loader/材质配置/桥接问题，不能再在单个模型上加入私有补丁。

### 4.2.1 Unity -> GLB -> OpenXR 坐标职责

环境坐标必须按以下顺序处理，禁止再用 profile 抵消错误的导出原点：

1. Unity 导出阶段先计算可见、非天空盒环境几何的水平包围盒中心，并把同一个反向 X/Z 平移应用到整个 GLB 导出根。网格、天空盒、动画节点、相机和灯光必须共享该根变换；Y 保持 Unity 世界高度/地面语义，不做垂直包围盒居中。
2. 归一化 GLB 加载后，其场景水平中心默认与 OpenXR reference-space 原点重合。`profile.model_position` 不得再保存数千单位的历史导出补偿，只允许表达有意的资产整体布局、尺度或小范围校正。
3. `preview_room_layout.py` 在归一化坐标系中编辑 `view_pose`/`view_poses` 和 screen 布局；座位位置属于 OpenXR reference-space 偏移，不得回写为模型根平移。
4. 运行时最后分别加载可选的模型布局参数和座位参数。Panda 只把 `model_position/rotation/scale` 应用到环境根，把 `view_pose` 应用到 OpenXR reference space，两者不得相互代偿。

旧 GLB 与旧 profile 必须成对迁移：先重新导出并静态确认应用节点变换后的非天空盒水平中心接近 `(0, 0)`，再删除旧 `model_position` 水平补偿，并用 preview 重新保存座位和 screen 坐标。不得在旧 GLB 尚未替换时提前清空 profile。

### 4.3 天空盒与色彩

天空盒语义必须优先固化在 Unity→GLB 资产管线：导出副本使用朝内法线与三角形绕序，材质为 `OPAQUE` + `KHR_materials_unlit`，从球体内部按标准 glTF 材质即可看见且不受房间灯光影响。Panda 只按 GLB 普通节点加载和渲染，不按节点名/profile 注入 alpha、double-sided、background bin 或 depth-write 私有覆盖。原 PNG、星空合成图、MIME、UV/朝向与导出质量问题都属于 Unity→GLB 资产管线；只有独立、标准化的 camera-relative background/cubemap 功能才允许作为未来通用 renderer 能力另行设计，不能为单个模型恢复私有补丁。

Unity -> GLB 材质分类必须是白名单规则：只有天空盒、明确具有发光纹理/非黑发光颜色或明确发光节点语义的对象、以及真正生成的 baked atlas/detail 材质使用 `KHR_materials_unlit`。普通房间、地面、看台、座椅、装饰、金属和普通透明材质必须导出为 glTF metallic-roughness PBR。仅启用 Unity `_EMISSION` keyword 不能证明材质发光；它可能是导入残留状态，不能据此把整个场景改成 unlit。baked 与 no-bake 导出必须执行同一分类规则。

Panda 颜色管理必须单独验证：base color/emissive 按 sRGB 解释，normal、occlusion、metallic-roughness 按线性数据解释。只有 baseColorTexture、没有 metallic-roughness/normal/emissive 贴图的 Artemis 环境表面不应强行走完整 metallic-roughness PBR；Panda runtime 应按 `preview_room_layout.py` 的 lit-diffuse 兼容语义渲染这类表面，只保留 Base Color 采样、profile ambient/head 光照、exposure/tone mapping，避免 simplepbr 对缺失 MR 贴图的采样产生黑色罩层或白色覆盖。最终 RT 格式、sRGB view 和 D3D11/OpenGL swapchain format 组成一个端到端色彩契约。不能同时在 Panda 和 D3D11/OpenGL 中做两次 gamma/tone mapping，也不能完全漏掉显示端 gamma/tonemap。Panda offscreen 目标必须申请 sRGB color buffer/texture；在 OpenGL fallback 中，Panda `render_frame()` 期间启用 `GL_FRAMEBUFFER_SRGB`，输出到 `Texture.F_srgb_alpha`，随后关闭该状态并把已编码的 GPU texture blit 到 OpenXR sRGB swapchain，避免线性颜色被当作显示颜色提交。当前待替换的 Artemis GLB 曾为 `lights=0` 且导出材质全部走 unlit 兼容路径，这是错误的资产基线。Panda runtime 已接入 profile ambient/head/fill lights：scene binding 从 viewer 读取 `_env_ambient_color`、`_env_head_light_color`、`_env_fill_lights`，把它们纳入幂等 binding key，并通过 `PandaSceneRenderer.configure_environment_lighting()` 重建 Panda `AmbientLight`、head-following `PointLight` 和 fill `PointLight`。Day/Night preset 或 profile 灯光变化后，下一帧会自动重新配置 Panda 灯光。PBR 还需要与 glTF viewer 类似的环境光基线：Panda 左右眼 offscreen target 初始化 simplepbr 时必须以各自的 eye `GraphicsBuffer` 和 eye camera 创建 pipeline，并提供程序生成的 neutral cubemap/env map，形成低频 IBL/反射输入；每帧渲染必须走 Panda `task_mgr.step()`，让 simplepbr update、camera_world_position 和 postprocess/tone mapping 对 OpenXR eye buffer 生效。profile 灯光安装阶段不得再调用 `set_shader_auto()` 覆盖 simplepbr 的 PBR shader，否则 neutral IBL 虽然创建成功也不会参与最终渲染。它是模型层渲染环境，不接管虚拟屏、键盘、激光、FPS/OSD 或最终 compositor。场景发黑不能只靠补 Panda 灯光解决；必须同时确认房间 PBR 材质确实有匹配的资产语义、只有上述白名单保留 unlit、GLB 灯光合理，并且 Panda/OpenXR 之间只做一次输出颜色变换；禁止恢复 Artemis 节点私有补丁。

### 4.4 Panda 稳态 GPU-only 规则

Panda3D 自己渲染 GLB 不等于全流程 strict zero-copy。GLB、纹理和顶点从磁盘进入 CPU 内存，再由 Panda 准备 `Geom`、`Texture`、`Material` 并上传到 GPU，这是加载期成本；每帧 CPU 做 scene graph、cull、state sorting 和动画采样也属于正常渲染调度。验收目标不是消除这些加载/调度成本，而是保证稳态帧循环没有像素级 CPU 回读或重复资源上传。

稳态规则：

- 房间、手柄、纹理、材质、render target、Panda `ShowBase` / `GraphicsBuffer` 必须缓存复用；不得每帧 reload GLB、重建 Panda texture/framebuffer 或重建模型资源。
- Panda 输出必须保持为 GPU texture/FBO，交给项目 compositor 采样或 GPU blit；禁止 `glReadPixels`、PBO readback、Numpy/PIL 中转后再上传。
- 动态虚拟屏幕不走 Panda CPU texture 上传路径；最终 screen texture upload 继续使用项目现有 CUDA/GL 或 D3D GPU 路径。Panda runtime 不保留 screen texture frame / upload target 类；`setRamImage()` / CPU bytes 上传只能出现在独立离线 probe，不进入 `src/xr_viewer/panda_runtime/`。
- Panda 只渲染 3D 模型层；虚拟屏幕、激光、键盘、FPS/OSD 不迁入 Panda 最终路径，避免破坏项目现有可控的 GPU 零拷贝合成链路。
- 优化 Panda 场景本身时，优先减少 alpha overdraw、合并静态节点、预生成 mipmap、限制阴影/灯光复杂度、只更新有动画的 `NodePath`，并用 Panda render/bridge/submit timing 区分加载期和稳态成本。

## 5. OpenGL 到 D3D11 桥接方案

### 5.0 GPU-only / zero-copy 硬性约束

Panda3D 可以做到项目要求的 GPU-only 3D 模型层输出，但这不是 Panda3D 默认自动保证的能力。Panda3D renderer 只有在满足下列条件之一时，才允许作为项目 OpenXR compositor 的有效 GPU 输入：

1. D3D11 主路径：`WGL_NV_DX_interop2` 成功，且 Panda 3D 模型层输出能以 GPU texture/FBO 形式交给现有 D3D11/OpenXR compositor 采样或 blit。
2. OpenGL fallback：Panda3D GL context 和 OpenXR compositor GL context 是同一个 context，或处于已验证的 WGL/GLX shared object namespace，使 Panda render target 能被项目 compositor GPU 侧采样或 blit。

以上是必要条件，必须通过真实 runtime 证据实现和验证。普通 D3D11 texture 的 register/lock/FBO complete 只能证明前置 readiness，不能等同于 Panda3D 3D 模型层已经能稳定进入项目 OpenXR compositor。若条件不成立，Panda3D path 必须明确报告原因并回退；禁止用 CPU readback、PBO readback、Numpy/PIL 中转或其它 CPU 图像搬运伪装成 fallback。`GLError 1282`、FBO incomplete、无法 lock/share texture、无法证明 shared context，都应视为 zero-copy gate 未通过，而不是资产或 glTF loader 问题。

### 5.1 首选：复用 NV_DX_interop2 直接渲染

每个 OpenXR acquire 的 `(eye, image_index)` 对应一个长期缓存的 D3D11 texture。第一阶段复用现有注册/lock/unlock 逻辑，把它映射为 GL texture/FBO；Panda render 时将该 FBO 作为该眼的目标，而不是先渲染到 Panda texture 再复制。

这是最短路径，但只有同时满足下列条件才可采用：

1. `WGL_NV_DX_interop2` 对当前 OpenXR D3D11 swapchain texture 注册成功，lock 后可作为 OpenGL FBO/texture 写入。
2. Panda 的 OpenGL context 与 WGL registration context 兼容，且与 D3D11/OpenXR 使用同一 NVIDIA adapter。
3. Panda 可以在目标 FBO 被 lock 期间完成该眼全部绘制，随后完成 GPU flush/fence，再 unlock。
4. OpenXR `release_swapchain_image` 一定发生在 unlock 后；不能跨帧持有 runtime texture。
5. 左右眼、尺寸变化、session 重建和 device lost 都能正确注销和重新注册缓存。

该路径不是零风险：Panda3D 管理自己的 `GraphicsStateGuardian` 和 framebuffer 状态。最佳结果是 Panda3D 在 lock 期间直接把当前眼画到 interop FBO/OpenXR swapchain texture；如果只能先渲染到 Panda offscreen color texture，再用 GPU blit 到 interop FBO，则仍属于 GPU-only、无 CPU copy，但不能标记为严格 zero-copy，只能作为过渡或性能可接受的 fallback 形态单独计时和验收。若无法在不侵入 Panda 内部的前提下把 OpenXR texture 设为它的可靠 render target，则停止此路径，不通过裸 OpenGL 调用强行篡改 Panda state。

### 5.2 后备：CUDA GL→D3D11 native bridge

若 5.1 因 context/FBO 所有权失败，Panda 先渲染到它自己的 RGBA OpenGL texture；随后由一个小型原生 Windows DLL 完成：

1. `cudaGraphicsGLRegisterImage` 注册 Panda 左/右 GL color texture。
2. `cudaGraphicsD3D11RegisterResource` 注册当前 acquire 的 OpenXR D3D11 swapchain texture。
3. 同一 CUDA stream 中 map 两端 `cudaArray`，进行 device-to-device array copy 或 kernel copy。
4. 在 stream 完成后 unmap 两端，再由 D3D11/OpenXR release 该 image。

该 DLL 提供窄接口：`register_gl_texture`、`register_d3d11_texture`、`copy_eye`、`unregister/rebuild`。Python 只传纹理 ID、D3D11 pointer、尺寸和同步 token；不在 ctypes/Python 中手写 CUDA 结构体或让 CUDA resource 跨 session 存活。

CUDA bridge 的 POC 需逐项证明：同 adapter、RGBA 格式、行方向、MSAA resolve、resize、GPU fence 顺序和 device reset。它可以避免 CPU 回读，但并不自动等于零拷贝：GL 与 D3D11 是不同 API 资源，至少会有一次 GPU copy。

### 5.3 OpenXR OpenGL fallback：Panda3D zero-copy projection path

主路径仍然是 D3D11 + `WGL_NV_DX_interop2`。当用户显式使用 OpenXR OpenGL backend，或 D3D11 backend 在当前 runtime/GPU 组合下不可用时，必须提供 Panda3D 的 OpenGL fallback path，避免退回自研 glTF renderer。

该 fallback 仍必须由 Panda3D 负责 glTF scene graph、动画、PBR/材质、环境/控制器 NodePath 和每眼相机；OpenXR 层只负责 acquire/wait/release swapchain image 与提交 Projection layer。它不是 CPU fallback，也不能使用 `glReadPixels`、PBO readback、PIL/Numpy 或任何 CPU 图像中转。

OpenGL fallback 的 zero-copy 合法实现只有两类：

1. Panda3D 直接在 OpenXR OpenGL session 使用的 context 中渲染，或由 Panda3D 创建并拥有该 context，再用它创建 OpenXR `GraphicsBindingOpenGL*KHR` session。
2. Panda3D context 与 OpenXR OpenGL context 处于同一个已验证的 WGL/GLX shared object namespace，使 Panda color target 与 OpenXR swapchain FBO/texture 可互相可见。

如果无法证明同 context 或 shared context 成立，OpenGL fallback 必须报告一次明确原因并回退 native/OpenGL renderer 或 D3D11 主路径；禁止通过 CPU copy 绕过。实现上必须想办法让 Panda3D 直接使用 OpenXR GL session context，或建立并验证可共享 OpenGL 对象的 context pair。OpenGL fallback 的验收证据必须至少包含：Panda GL vendor/renderer、OpenXR GL binding context、shared-context 验证、左右眼 acquire-render-release 顺序、无 `GLError 1282`、无 CPU readback 调用、动画连续播放。

Windows 当前实现（2026-07-16）：在创建 Panda eye texture 前，先保存 GLFW/OpenXR `HGLRC`；Panda `ShowBase(windowType="offscreen")` 创建其 `HGLRC` 后，立即调用 `wglShareLists(OpenXR_HGLRC, Panda_HGLRC)`，成功后才创建 Panda offscreen texture。预热完成必须切回 GLFW/OpenXR context，再以 `glIsTexture` 验证 Panda texture 可见性；共享失败或不可见时，禁止 acquire OpenXR swapchain image。每帧 Panda `GraphicsEngine.render_frame()` 前必须以保存的 Panda HDC/HGLRC 调用 `wglMakeCurrent`，渲染后才切回 GLFW/OpenXR context 做 GPU blit；不得在 OpenXR context 上直接驱动 Panda GSG。真机已证明共享和 acquire-render-release 成功，且 context 切换修复后不再出现 `GL 0x502`；随后灰屏被定位为坐标空间、profile transform 和真实 eye pose 缺失。OpenXR/OpenGL pose 现在按 `(x,y,z) -> (x,-z,y)` 及对应旋转基变换进入 Panda，环境根节点应用现有 model position/rotation/scale；当 viewer 没有自定义转换器时，adapter 直接从 `XrView.pose` 构建 eye model matrix。启动日志必须显示 `eye_poses=2` 才允许进入可见性验收。后续真机已确认房间和手柄模型可见，剩余缺失项曾收敛为 Artemis 天空盒。旧 GLB 将该节点导出为 `alphaMode=MASK`、朝外单面几何，因此依赖 native/profile 私有覆盖；Unity 导出器现已改为朝内法线/绕序并写出 `OPAQUE` + `KHR_materials_unlit`，重新导出的天空盒已在头显中可见。当前 GLB 静态检查为 **BLEND 28 / OPAQUE 16**，天空盒 `doubleSided=false` 且带 `KHR_materials_unlit`；直接 `panda3d-gltf` 加载后的天空盒 GeomState 只有材质和纹理，不含 `AlphaTestAttrib`、cull/background/depth-write override。Panda 运行时的 `configure_environment_skybox`、profile 节点匹配及相关诊断现已全部删除，无运行时天空盒特殊处理的真机测试已能显示天空盒，但出现间歇性均匀灰色多边形缺口。像素与网格检查证明灰块是 Panda clear color，天空球焊接后边界为 0、16128 个三角形全部朝内；根因是 Panda Lens 固定 `far=1000`，而偏心视点到天空球最远表面约 `1033.5`，部分三角形被远裁剪面截掉。当前 `PandaFrameSourceInput/PandaFrameState` 已传递 viewer/profile 的 near/far，每帧同步到两眼 Lens，启动日志输出 `clip=near/far`；Artemis 预期为 `clip=0.100/20000.0`。仍须真机确认灰色缺口消失。

### 5.4 明确不采用的路径

- 不使用 CPU `glReadPixels`、PBO readback、PIL/Numpy 中转到 D3D11 或 OpenGL swapchain；它违反实时渲染目标。
- 不让 Panda3D 使用 Direct3D 9 再尝试 DX9→DX11 级联。
- 不修改 Panda3D 源码来增加 D3D11 backend。
- 不让 Panda 和现有 Moderngl renderer 同时渲染同一个环境；切换必须是 renderer ownership 的互斥选择。

## 6. 分阶段替换计划

### Phase 0：可行性闸门

当前状态（2026-07-15）：已新增 `src/tools/panda3d_gltf_probe.py`、`src/tools/panda3d_animation_screenshot_probe.py`、`src/tools/panda3d_material_probe.py`、`src/tools/panda3d_offscreen_probe.py`、`src/tools/panda3d_d3d11_interop_probe.py`、`xr_viewer.panda3d_probe`、`xr_viewer.panda3d_animation_screenshot_probe`、`xr_viewer.panda3d_material_probe`、`xr_viewer.panda3d_node_animation`、`xr_viewer.panda3d_offscreen_probe` 和 `xr_viewer.panda3d_d3d11_interop_probe`。本机使用 Panda3D 1.10.15、panda3d-gltf 1.3.0 检查当前 Artemis `environment.glb`，发现 GLB 本身有 **19** 个 animation、**38** 条 channel、**19** 个 animation target node，且这 **19** 个 target 全部属于 active scene。`panda3d-gltf` 原生载入结果仍为 **0** 个 `Character`、**0** 个 `AnimBundleNode`，但自定义 glTF node animation runtime 已能绑定 **19/19** 个 target node、采样 **38** 条 channel，并确认动画时长为 **15.0 秒**；`--strict-animation` 返回退出码 0。Phase 0 probe 现在通过 `GltfNodeAnimationPlayer` 按 **0.0 / 7.5 / 15.0 秒** 推进 Panda NodePath，并在 JSON 中记录采样节点与 transform 变化结果；动画截图 probe 可在相同采样时间输出 3 张 PNG，并记录每张截图的路径、SHA-256 和字节大小。材质语义 probe 已记录 Artemis **44** 个 material、**19** 个 image、**19** 个 texture，alpha 分布为 **BLEND 28 / OPAQUE 16**，**44/44** 个 material 使用 `KHR_materials_unlit`，并定位天空盒材质 `GLTF_UnlitSkybox_GLTF_Skybox_Composite_0`。当前 HP/INDEX/PICO/QUEST/VIVE/YVR 左右手共 **12** 个控制器 GLB 均可由 Panda3D 加载，均为静态资产，node/geom 数量非零，animation runtime ready。Bedroom `environment.glb` 已清理一个越界 child node 引用（旧引用指向不存在的 node **206**），现在可无 warning 加载出 **416** 个 Panda node 和 **202** 个 Geom。Panda OpenGL offscreen 子闸门也已通过：`pandagl` 创建 64×64 render target，实际驱动为 **NVIDIA GeForce RTX 2060/PCIe/SSE2**，OpenGL **4.6.0 NVIDIA 596.36**，framebuffer 为 RGBA8 + depth24。Panda GL context 下的 D3D11/NV_DX readiness 已通过：D3D11 feature level **0xb000**，D3D11 adapter 枚举为 **NVIDIA GeForce RTX 2060**（vendor **0x10de**、device **0x1f03**、LUID **00000000:00087686**、dedicated VRAM **12646875136**），并确认 GL renderer 与 D3D11 adapter 名称匹配；`WGL_NV_DX_interop2` 函数可加载，`wglDXOpenDeviceNV` 可打开并关闭 D3D11 device；普通 **64×64 RGBA8 D3D11 Texture2D** 可注册为 GL texture、lock 成功，并通过 `GL_FRAMEBUFFER_COMPLETE`。Panda offscreen texture native id 已可获取，64×64 probe 记录 native id **1**。已加入 `D2S_PANDA3D_PHASE0_SWAPCHAIN_PROBE=1` 的真实 OpenXR D3D11 swapchain POC 路径：在 acquire/wait 后注册当前 swapchain texture、lock 为 GL FBO、画测试色块和三角形、unlock/release；Phase 0 仍需在头显/OpenXR runtime 下实际确认该路径无错误并可见。

- 安装锁定版本的 `panda3d` 与 `panda3d-gltf`，记录 Python ABI、GPU driver、Panda 版本和插件版本。
- 运行 `src/tools/panda3d_gltf_probe.py`：加载 Artemis/控制器并验证 glTF animation target 是否属于 active scene，以及 Panda runtime node 是否可驱动这些动画；Artemis 已由自定义 node animation runtime 通过，probe JSON 已记录 0.0/7.5/15.0 秒采样时间、采样节点和 transform changed 结果；`src/tools/panda3d_animation_screenshot_probe.py` 可保存 0.0/7.5/15.0 秒 PNG 截图并输出 SHA-256/大小摘要；`src/tools/panda3d_material_probe.py` 可输出 alpha、double-sided、unlit、texture/image 和 skybox material 摘要；12 个控制器 fixture 已通过，Bedroom missing-node warning 已通过清理越界 child 引用解决。
- 记录 Panda OpenGL vendor/renderer、Panda texture native handle 的可获取性、实际 offscreen texture format、同 adapter CUDA/D3D11 枚举结果；当前已记录 vendor/renderer/version、RGBA8/depth24 offscreen RT、Panda texture native id、D3D11 adapter description/vendor/device/LUID/VRAM、GL/D3D adapter 名称匹配、D3D11 feature level、NV_DX device open/close，以及普通 D3D11 Texture2D 的 register/lock/FBO complete。
- 使用现有 D3D11 OpenXR session 做 NV_DX interop POC；当前已验证普通 D3D11 texture 注册，并已提供 `D2S_PANDA3D_PHASE0_SWAPCHAIN_PROBE=1` 的真实 OpenXR swapchain 测试图路径。普通 texture 结果只能作为 readiness，不能代替真实 OpenXR swapchain gate。下一步需在头显运行时确认 acquire、注册、lock、渲染、unlock、release 全链路成功，先不加载真实 GLB。

闸门：任何一个 asset 的加载/动画/透明正确性失败，或 GL→D3D11 不能完成单帧 acquire-render-release，则保留当前 renderer，先修 POC，不开始替换。

### Phase 1：新 renderer 适配层，不改默认路径

当前状态（2026-07-16）：已新增 `src/xr_viewer/panda_runtime/` 的 import-light 适配层骨架，包含 `runtime.py`、`scene.py`、`stereo_targets.py`、`bridge.py`、`diagnostics.py`；`PandaSceneRenderer` 已定义 `load_environment`、`load_controller`、`update_frame_state`、`render_eyes`、`rebuild_targets`、`release` facade 契约。`scene.py` 默认只记录资产路径，启用 `load_panda_assets=True` 时会懒加载 `panda3d-gltf`、保留内部 root ownership，并记录 node/geom 计数，不向 facade 外暴露 `NodePath`。`stereo_targets.py` 默认只记录左右眼 target spec，启用 `create_panda_targets=True` 时会在单个 Panda `ShowBase` 下创建左右眼 offscreen buffer、texture、display region 和 camera，并记录 texture native id。`bridge.py` 已定义以 `(session_generation, eye, image_index, width, height, format)` 为边界的 `SwapchainResourceKey` 和明确的未实现失败契约，后续 NV_DX/CUDA bridge 必须复用该缓存策略。`PandaAnimationClock` 已加入 facade，`update_frame_state()` 会把 XR `predicted_display_time` 派生为从首帧起算且不倒退的 `animation_time_seconds`，并把同一个 bound frame snapshot 传给 scene 与 bridge；`PandaSceneRenderer.configure_animation()` 已提供 runtime 控制面，可在运行时设置播放速度、暂停、固定采样时间和循环开关。`PandaFrameState` 现在只携带 3D 模型层需要的 `frame_index`、`PandaEyeView`、`PandaPose` 和 `controller_poses`，更新帧时会校验两眼 eye index 与同一 snapshot 边界；`PandaSceneGraph` 只记录 scene assets、eye view count、controller hands、applied controller hands 和动画采样状态。已删除 Panda screen/ray 运行时类与绑定：`PandaScreenTextureFrame`、`PandaScreenTextureUploadTarget`、screen NodePath target、`PandaControllerRay`、controller ray target、`screen_pose`、`screen_texture`、`controller_rays` 不再属于 `panda_runtime` 合同。`sync_panda_scene_assets_from_viewer()` 只把 viewer 当前 `_env_model_path` 和当前 controller brand 的 `left.glb`/`right.glb` 幂等绑定到 Panda facade，并启用真实 Panda asset root ownership；天空盒完全使用 GLB 自带的 inward-facing geometry、`OPAQUE` 和 `KHR_materials_unlit` 语义，Panda runtime 不读取 profile 天空盒节点、不注入 RenderState；绑定失败只记录 `_panda_scene_binding_error`，不替换 native renderer。diagnostics 现在汇总 scene assets、stereo target refs、bridge resource keys、最新 frame predicted display time、animation time、eye view count、controller count 和动画 runtime 摘要；Artemis 运行时采样可观测为 38 channels / 19 bound nodes。`D2S_GLTF_RENDERER=native|panda3d` selector 已加入，默认仍为 `native`；在真实 OpenXR swapchain gate 未通过前，请求 `panda3d` 会记录原因并回退 native，不会替换现有 D3D11 native renderer。

新增 `src/xr_viewer/panda_runtime/`，建议边界如下：

```text
panda_runtime/
  runtime.py       # Panda process/thread lifecycle and renderer facade
  scene.py         # environment/controller NodePath ownership
  stereo_targets.py# eye cameras, FOV/projection, render-target lifecycle
  bridge.py        # NV_DX bridge facade; optional native CUDA backend binding
  diagnostics.py   # asset/runtime/bridge summary and screenshots
```

定义与现有 viewer 脱钩的 `PandaSceneRenderer` 接口：`load_environment`、`load_controller`、`update_frame_state`、`render_eyes`、`rebuild_targets`、`release`。输入为已有的 head/eye/controller pose snapshot，输出为“左右眼 Panda 3D 模型层 GPU resource”，再由项目 compositor 与虚拟屏幕、激光、键盘和 OSD 合成；不暴露 Panda `NodePath` 给外部。

新增 renderer 选择器，例如 `D2S_GLTF_RENDERER=native|panda3d`，默认 `native`。Panda 初始化失败、bridge 失败、设备变更时要打印一次明确原因并回退 native；不得静默输出黑屏。

### Phase 2：功能等价

- 先迁移 Artemis 和一个控制器，不迁移可选 glow/全景特效。
- 接入共享控制器 pose 和每眼 view/projection；手柄拖动、虚拟屏幕、手柄射线、键盘、FPS/OSD 仍在现有控制器状态机和 OpenXR UI/compositor 中。当前运行时已删除 `controller_rays`、`screen_pose`、`screen_texture`、screen NodePath 和 ray target 等 Panda 侧 API，后续不得重新引入到 `panda_runtime`。最终 Phase 2 的验收边界是：Panda 能渲染环境和手柄模型层，项目 compositor 能继续独立合成虚拟屏幕、激光、键盘和 OSD。`PandaFrameSourceInput` / `build_panda_frame_state()` 已提供从现有 OpenXR pose matrix 和 eye fov 进入 `PandaFrameState` 的 import-light adapter；`OpenXRFrameRenderer` 已在 `locate_views` 后、quad/projection 更新前调用该 adapter，并且只在请求 `D2S_GLTF_RENDERER=panda3d` 时把结果缓存到 viewer；请求 Panda 时会创建 import-light 的 `PandaSceneRenderer` facade，先把当前环境/控制器 GLB 真实加载并绑定到 Panda scene root，再把 cached frame state 交给 `update_frame_state()`；`ProjectionLayerPresenter` 已新增受 `panda3d_enabled` 保护的 `render_panda_bridge()` 调用点，会在 acquire/wait 后用左右眼 `SwapchainImageRef` 调用 `PandaSceneRenderer.render_eyes()`，并保证成功或失败都 release 已 acquire 的 swapchain image；NV_DX bridge 已接入 Panda offscreen target 到 OpenXR swapchain FBO 的 blit 路径，失败时仍会记录一次错误并回落 native，不替换默认提交路径。后续代码应把该 bridge 收敛为“Panda 3D 模型层输入项目 compositor”，而不是让 Panda 拥有完整 VR UI。
- 将 glTF animation clock 绑定到 XR predicted display time，避免每眼/每线程各走一个时钟；当前 facade 已完成 clock 派生，`PandaSceneGraph(load_panda_assets=True)` 已能创建并驱动真实 Panda node animation player，diagnostics 会记录 Artemis 38 channels / 19 targets / 19 bound nodes / 15.0 秒；Phase 0 runtime 控制面已支持 playback speed、pause、fixed sample time 和 loop 开关，便于在头显 gate 前后复现 0/7.5/15 秒采样与连续播放。
- 把 profile 保留为资产布局、尺度、sky/background、光照和默认视角配置；不要把 model-specific 动画逻辑重新塞回 profile。
- 以同一帧 snapshot 更新两眼相机与模型 NodePath，禁止左眼/右眼读到不同 controller pose；当前 facade 已定义 `PandaFrameState(frame_index, eye_views, controller_poses)` 并校验两眼 eye index，diagnostics 可输出同帧模型层 snapshot 摘要。

### Phase 3：接入与性能

- 优先打通 NV_DX 路径；仅在失败后启用 CUDA bridge，并在日志中记录实际桥接模式。当前已存在受保护的 Panda `render_eyes()` 调用点、失败回落 native 路径，以及 `PandaNvDxBridge` / `ViewerPandaNvDxInteropAdapter` concrete bridge shell；该 shell 复用现有 `_get_or_create_nv_interop_fbo()` 与 `_nv_dx_objects`，按左右眼 `SwapchainImageRef` 获取 FBO、lock/unlock NV_DX object。`PandaSceneGraph.render_to_framebuffers()` 已接入 Panda offscreen 左右眼 target：先把已加载环境和控制器 root 挂到当前 Panda `ShowBase.render`，把同帧 OpenXR eye pose/fov 同步到 Panda 左右眼 camera，再 `render_frame()`，最后把 Panda eye color texture 交给项目 OpenXR compositor 采样或 GPU blit。Panda runtime 不再有 screen/ray root 挂载能力；虚拟屏幕、激光、键盘、FPS/OSD 必须继续由项目 compositor 合成。`PandaSceneRenderer` 和 `ProjectionLayerPresenter` 现在会记录 Panda bridge 成功/失败次数、最后 bridge mode、左右眼 target size、image index、左右眼 rendered 状态和最后错误，便于真机 gate 后直接判断 acquire/render/release 证据。下一步仍需在头显/OpenXR runtime 下确认该路径真实可见、姿态矩阵正确且无显存/同步错误。
- OpenXR OpenGL backend 也必须有 Panda3D fallback，但只能作为 zero-copy fallback：Panda3D 必须使用 OpenXR compositor GL context 或已验证 shared context 输出项目 compositor 可见的 3D 模型层 texture；不能做 CPU readback、PBO readback 或 Numpy/PIL 中转。当前 Windows fallback 会在 Panda eye texture 创建前执行 `wglShareLists(OpenXR_HGLRC, Panda_HGLRC)`，预热后切回 OpenXR GL context 检查 Panda target texture id 是否可见；共享失败或不可见说明 context ownership/share 验证失败，必须在 acquire 前显式禁用 Panda OpenGL bridge 并回退 native/OpenGL renderer，不允许进入 `glFramebufferTexture2D` 后才失败或静默黑屏。
- 所有 swapchain 资源缓存以 `(session_generation, eye, image_index, width, height, format)` 为 key；session 重建先清资源，再创建。
- 统计每帧 Panda render、bridge、OpenXR acquire/release、submit 的 GPU/CPU 时间，区分首次资源创建与稳态。当前 Panda projection bridge 已记录 CPU 分段 timing：`acquire_wait`、`target_rebuild`、`bridge_render`、`release`、`total`，并通过 `_breakdown_add_time()` 进入现有 breakdown；GPU timing、submit p50/p95 汇总和首次资源创建/稳态区分仍需结合真机运行证据补齐。
- 禁止让 Panda 使用最新屏幕帧作为 runtime texture；屏幕帧只进入项目现有 compositor。若未来需要屏幕光照采样，必须以项目 GPU-only 光照/反射路径实现，不能通过 Panda screen texture 回传。

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
