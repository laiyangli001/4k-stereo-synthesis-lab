# Filament glTF 场景运行时实施计划：OpenXR D3D11 主路径与 OpenGL GPU-only fallback

## 1. 决策与目标

项目的完整 glTF 场景主路径改为 **Filament + gltfio**。Filament 负责 GLB 加载、资源上传、scene graph、PBR/unlit 材质、普通节点 TRS 动画、skin 和 morph 的运行时渲染；`gltfio::Animator` 是标准 glTF 动画的唯一播放实现。

本计划替代 `docs/39-panda3d-gltf-openxr-d3d11-migration-plan.md` 中 Panda3D 作为主 runtime 的目标架构。Panda3D 的探针、互操作实验和历史资料可以保留为诊断参考，但不得继续承担默认 GLB loader、动画播放器或 3D 模型层 renderer。

本计划不替换：

- 2D->3D 推理、双目生成、latest-frame 调度与 CUDA 输入链。
- OpenXR session、视图定位、swapchain acquire/wait/release、`xrEndFrame`。
- 虚拟屏幕 GPU texture upload、控制器状态机、激光、键盘、FPS/OSD、Quad/overlay 与最终 composition。
- 现有 D3D11 OpenXR session / swapchain 所有权。

目标是停止为 glTF scene graph、accessor/TRS 动画采样、PBR 和扩展语义维护自研或 Panda 私有实现，同时保持实时帧中无 CPU 像素回读、无 PBO/Numpy/PIL 中转、无每帧资源重建。

## 2. 所有权边界

| 领域 | 所有者 | 约束 |
|---|---|---|
| GLB 加载、资源解码、scene graph、PBR/unlit、节点/skin/morph 动画 | Filament `gltfio` | 使用 `AssetLoader`、`ResourceLoader`、`Animator`；禁止项目手写 accessor/TRS 采样器 |
| 每眼 3D 模型层 GPU 渲染 | Filament bridge | 只渲染环境和控制器模型层 |
| 2D->3D runtime、虚拟屏幕 CUDA/GL 或 D3D 上传 | 项目现有 runtime | 不进入 Filament texture upload 合同 |
| 激光、键盘、OSD、FPS、screen interaction | 现有 OpenXR UI/compositor | 不迁入 Filament scene |
| OpenXR lifecycle 和 Projection layer submit | 现有 OpenXR backend | Filament 不调用 `xrAcquireSwapchainImage` 或 `xrEndFrame` |

Filament bridge 的输入是同一帧 `xrLocateViews` 得到的左右眼 pose/FOV、控制器 pose、`XrTime` 和已 acquire 的目标资源；输出只能是已写入的左右眼 GPU target。虚拟屏幕、手柄射线、键盘和 OSD 保持由项目 compositor 合成。

## 3. 目标架构

```text
GLB / glTF assets
      |
      v
Filament gltfio: AssetLoader + ResourceLoader + Animator
      |
      v
Filament bridge: per-eye View / Camera / Scene / RenderTarget
      |
      +-- D3D11 primary: OpenXR D3D11 texture <-> WGL_NV_DX_interop2 <-> Filament GL target
      |
      +-- OpenGL fallback: verified same/shared OpenXR GL context -> Filament GL target
      v
existing OpenXR compositor: screen + laser + keyboard + OSD composition
      |
      v
OpenXR projection swapchain -> xrEndFrame
```

Filament 不拥有 OpenXR swapchain 生命周期。bridge 只在 acquire/wait 成功后对当帧指定眼和 image index 渲染，并在 compositor 完成 GPU fence/flush 所需操作后由既有 backend release。不得跨帧持有 runtime texture。

## 4. GPU-only 硬性闸门

Filament 能满足 GPU-only 3D 模型层路径，但不是默认自动成立。以下任一条件不成立时，Filament 不能成为默认 renderer：

1. **D3D11 主路径**：当前 OpenXR D3D11 swapchain texture 能经 `WGL_NV_DX_interop2` 注册、lock，并作为 Filament 可写的 OpenGL target；同一 NVIDIA adapter、FBO/RenderTarget 完整、unlock 发生在 release 前。
2. **OpenGL fallback**：Filament Engine 的 OpenGL context 与 OpenXR compositor context 是同一 context，或通过已验证的 WGL/GLX/EGL/CGL sharing 使 Filament target 在 compositor 中可见。
3. **稳态帧**：无 `glReadPixels`、PBO readback、CPU bitmap、Numpy/PIL、`Texture::setImage` 逐帧上传；GLB/纹理加载和 GPU 上传只发生在加载、重载或 session reset。
4. **同步**：同一预测显示时间的两眼使用同一 scene snapshot；任何 GPU fence/flush 在 swapchain release 前完成；不会跨 session generation 复用目标。

普通 D3D11 texture 注册成功不能代替真实 OpenXR swapchain 的 acquire -> render -> release gate。Engine 创建成功不能代替共享 object namespace 或目标 texture 可写性 gate。任一失败必须明确记录原因并回退 native renderer，禁止黑屏或 CPU fallback。

## 5. Bridge 设计

### 5.1 目录与跨平台构建

```text
native/filament/
  sdk/windows|macos|linux/v1.74.0/  # 本机构建依赖，不提交发布包
  bridge/                            # 共享 C++ bridge、CMake 与平台适配
src/xr_viewer/native/
  filament_bridge.dll                # Windows 发布产物
  libfilament_bridge.dylib           # macOS 发布产物
  libfilament_bridge.so              # Linux 发布产物
```

`native/filament/bridge/CMakeLists.txt` 是唯一构建定义。共享 C ABI 不暴露 Filament C++ 对象：

- `filament_bridge_create(shared_graphics_context)`：只允许在 OpenXR 图形 context 已 current 的线程调用。
- `filament_bridge_load_glb(bytes)`：加载 self-contained GLB，创建 `FilamentAsset` 与 GPU resources。
- `filament_bridge_apply_animations(time_seconds)`：对每个 glTF animation 调用 `Animator::applyAnimation`，再调用 `updateBoneMatrices`。
- 后续新增 `bind_eye_target`、`render_eye`、`release_eye_target`；参数仅包含 native GPU handle、尺寸、format、view/projection 和帧序号，绝不包含像素 buffer。

Windows、macOS、Linux 必须分别使用同版本官方 SDK 在本机或 CI 构建，不允许从 Windows 伪造 `.dylib`/`.so`。共享 asset/animation 代码不得包含 WGL、GLX、EGL 或 CGL 细节；这些放在平台 target adapter 中。

### 5.2 Filament runtime 生命周期

每个 OpenXR session generation 创建一个 bridge runtime：`Engine`、`Renderer`、`Scene`、左右眼 `View/Camera`、`AssetLoader`、`ResourceLoader`、material/texture provider。session 重建或 graphics device lost 时按依赖逆序销毁。

- 资产加载后调用 `asset->releaseSourceData()`，保留 GPU resource 与 `Animator`。
- 每帧以 XR predicted display time 的秒值驱动动画，不能把纳秒 `XrTime` 当秒传入。
- Artemis 的卫星和飞船是普通 glTF node TRS 动画，必须由 `Animator` 直接播放；不得转为 Skin/Character，也不得恢复 Python accessor/TRS sampler。
- 灯光、IBL、exposure、tone mapping 必须由通用 profile/资产语义驱动，不按 Artemis 节点名或材质名写私有覆盖。
- StarGlim sidecar 若保留，必须是独立的、资产声明的 GPU shader effect；time uniform 与同一 `animation_time_seconds` 同步，不影响 GLB 标准动画语义。

### 5.3 每眼 target 与颜色

每眼按当前 `XrView` 创建或更新 Filament `Camera` 的 projection 和 model matrix。目标缓存 key：

```text
(session_generation, backend, eye, image_index, width, height, format, sample_count)
```

目标变化、swapchain resize、format change、session reset 必须销毁并重建对应的 Filament texture/RenderTarget wrapper。color-space 是端到端合同：输入 glTF base color/emissive 的 sRGB 与 normal/MR/occlusion 线性语义由 gltfio 保持；Filament tone mapping 和 OpenXR sRGB swapchain 只允许一次显示变换。

## 6. 分阶段实施

### Phase 0：构建与 API gate

- 完成 Windows/macOS/Linux 官方 SDK 下载、SHA-256 校验和 CMake 构建矩阵。
- Windows 必须产出并加载 `src/xr_viewer/native/filament_bridge.dll`；macOS/Linux 由对应 runner 产出 `.dylib`/`.so`。
- Python `ctypes` loader 按 `sys.platform` 选择动态库，检查 ABI version 和必需符号；加载失败仅报告并保持 native renderer。
- 使用离线 Artemis GLB 验证 animation count、duration、0/7.5/15 秒 node transform 变化。此阶段不接入 OpenXR swapchain。

闸门：DLL/动态库缺失、ABI 不匹配、GLB 不能加载、动画数量/时长与 glTF JSON 不一致，均阻止下一阶段。

### Phase 1：context 与 target gate

- 在 OpenXR GL backend 的 context 已 current 时创建 Engine，记录 native context handle 和 Filament backend。
- 验证 Engine 创建前后 current context 不被替换；若使用 shared context，验证 target texture 的 object namespace 可见。
- 用实际 acquired OpenXR GL swapchain texture 建立 Filament attachment/RenderTarget，验证 FBO/RenderTarget 完整性，先清色或渲染测试三角形。
- D3D11 路径复用既有 NV_DX registration/lock/unlock 管理，但 Filament 必须在 lock 区间内写入 target；先做单眼 acquire-render-unlock-release POC。

闸门：任何 `GL_INVALID_OPERATION`、FBO incomplete、NV_DX lock/share 失败、release 前未 unlock、target 不可见，均只记录 bridge failed 并回退 native。

### Phase 2：Filament 场景与动画可见性

- 载入 Artemis 和一套控制器 GLB，添加到 Filament scene。
- 从同一 XR frame snapshot 更新左右眼相机、环境根变换和控制器 transform。
- 用 `Animator` 连续播放 Artemis；验证卫星/飞船平滑运动、天空盒、土星环、alpha/unlit 和 PBR fixture。
- 把 profile 保留为资产布局、尺度、视角和通用灯光配置，不保存模型私有动画补偿。

闸门：两眼 pose 不同帧、动画跳点、天空盒/alpha/PBR 明显错误或新出现 CPU transfer，阻止接入 compositor。

### Phase 3：接入 compositor 与性能

- `ProjectionLayerPresenter` 增加互斥 selector：`native|filament`；默认仍是 `native`。
- 成功时 Filament 覆盖环境/控制器 3D 模型层；screen、laser、keyboard、OSD 仍由既有 compositor 合成。
- 记录 `acquire_wait`、Filament render、bridge lock/unlock、release、submit 的 CPU 与 GPU timing，并分别报告首次创建和稳态 p50/p95。
- 强制 bridge 初始化/target bind 失败测试，确认会回退 native 而不黑屏、不泄露 swapchain image。

### Phase 4：默认切换与清理

只有第 7 节全部通过后，才将 `filament` 设为默认。之后删除被替代的自研环境/控制器 GLB renderer 和 Panda3D 主路径；保留 `docs/38` 与资产诊断工具直到 Filament 的等价诊断覆盖完成。

## 7. 验收矩阵

| 项目 | 自动证据 | 真机证据 |
|---|---|---|
| 多平台构建 | Windows DLL、macOS dylib、Linux so 的 CMake/CI artifact | 各平台启动可加载 bridge |
| GLB 与动画 | Artemis/Bedroom/控制器 fixture；animation count/duration/0-7.5-15s 摘要 | 卫星和飞船连续平滑运动 |
| 材质 | unlit、alpha、double-sided、PBR、sRGB fixture 截图 | 天空、土星环、房间和控制器无黑/白覆盖 |
| XR pose | 单帧两眼 view/projection、controller pose 单测 | 头动稳定、控制器对齐 |
| D3D11 bridge | acquire -> lock -> render -> unlock -> release 顺序和 reset 测试 | 左右眼稳定显示 |
| OpenGL fallback | same/shared context、target visibility、FBO complete | GL runtime 下无黑屏 |
| 性能与零拷贝 | 无 CPU-transfer telemetry；render/bridge/submit p50/p95 | 不低于 native 基线且无显存增长 |
| 回退 | 强制失败保留 native renderer | 不崩溃、不黑屏 |

## 8. 与现有文档的关系

- `docs/38-gltf-2-renderer-compliance-layer-plan.md`：继续作为 glTF 2.0 core contract、fixture 和覆盖缺口的审计基线。引入 Filament 不自动宣布 core-compliance 完成；必须用 fixture 逐项验证。
- `docs/39-panda3d-gltf-openxr-d3d11-migration-plan.md`：保留 Panda3D 方案、历史探针和 NV_DX 证据，但其主 renderer 结论被本计划取代。后续实现和状态更新以本文为准。
- 本文不把 SDK 可下载、DLL 可构建或 Engine 可创建等同于 OpenXR zero-copy 已通过；真实 swapchain gate 才是默认切换依据。

## 9. 当前状态（2026-07-18）

- Filament v1.74.0 Windows/macOS/Linux SDK 已下载并校验；CMake 跨平台构建定义已建立。
- Windows `src/xr_viewer/native/filament_bridge.dll` 已成功构建。
- bridge 已有 AssetLoader/ResourceLoader/Animator 的 C ABI 骨架，尚未完成每眼 RenderTarget、OpenXR context probe、swapchain bind、Python loader 或 presenter selector。
- 因此当前默认 renderer 不变；Panda3D 和 native 路径尚未被运行时替换。