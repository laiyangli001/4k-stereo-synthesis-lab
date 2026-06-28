# OpenXR 渲染路径报告

## 目的

本报告记录当前 Desktop2Stereo OpenXR 渲染路径的实际实现方式、与旧版 OpenXR 行为的关系，以及各立体模式现在对应的 OpenXR 输出路径。

当前目标已经从“设计接入方案”更新为“说明已接入路径”：

- 传统 / Fastest 保留旧版兼容的 OpenXR RGB+深度 shader 路径。
- 影院、游戏、静态图像和 debug/export 模式在 OpenXR 下使用完整立体合成 eyes 路径。
- `D2S_OPENXR_RUNTIME_DIRECT=0` 仍然可以强制 OpenXR 使用完整立体合成 eyes 路径。
- viewer 协议不变：根据 `runtime_output_format` 区分 RGB+深度和 runtime direct eyes。

## 旧版 Desktop2Stereo OpenXR 流程

旧版 `Desktop2Stereo_v2.5.0Beta` OpenXR 路径没有使用完整 `make_sbs()` 立体合成路径。

旧版 OpenXR 流程：

```text
capture
-> predict_depth(rgb)
-> depth_q.put(rgb, depth, timestamp)
-> OpenXRViewer.run(first_rgb, first_depth)
-> viewer 上传 RGB 纹理 + 深度纹理
-> OpenXR shader 从 RGB + 深度生成每眼视差
```

相关旧版行为：

- `main.py` 的 OpenXR 分支创建 `OpenXRViewer(ipd=IPD, depth_ratio=DEPTH_STRENGTH, ...)`。
- 将 `rgb` 和 `depth` 直接传入 `viewer.run()`。
- OpenXR viewer 在 `_update_frame(rgb, depth)` 中上传 RGB 和深度。
- viewer shader 使用深度参数创建立体视差。
- 旧版 `make_sbs(...)` 路径用于流媒体 / 非 OpenXR 输出，不用于 OpenXR。

因此，旧版 OpenXR 最准确的描述是 RGB+深度 shader 路径，而不是完整 SBS 合成路径。

## 当前 OpenXR 路径总览

当前代码库有三类 OpenXR 输出路径：

1. OpenXR RGB+深度 shader 路径
2. OpenXR prewarp eyes 路径
3. OpenXR full stereo synthesis eyes 路径

其中第 1 类是传统 / Fastest 的默认兼容路径，第 3 类是影院、游戏、静态图像和 debug/export 的当前质量路径。第 2 类仍是实验 / 兼容路径，不是主要模式映射目标。

## 1. OpenXR RGB+深度 shader 路径

这是传统 / Fastest 使用的旧版兼容路径，也是 `D2S_OPENXR_RUNTIME_DIRECT=1` 且当前 active preset 不属于完整合成 preset 时的 direct 路径。

```text
runtime:
RGB -> 深度模型 -> 深度后处理 -> OpenXRRuntimeResult(source_rgb, depth)

viewer:
source_rgb + depth -> OpenXR shader -> 头显
```

当前实现要点：

- `RuntimePipelineLoop.run()` 在 OpenXR direct 且未启用 full synthesis 时调用 `stereo_runtime.process_openxr_frame()`。
- runtime result 的 `debug_info["runtime_output_format"]` 为 `"openxr_rgb_depth"`。
- viewer 根据该 format 走 RGB+深度上传和 OpenXR shader 视差生成。
- `OpenXRViewer` 构造参数现在使用 `depth_strength`；旧的 `depth_ratio` 作为兼容 property 映射到 `depth_strength`。

该路径主要消费的参数：

- `depth_strength`
- `convergence`
- `ipd` / `ipd_mm`
- `stereo_scale`
- `max_shift_ratio`
- OpenXR viewer 侧屏幕和显示参数

该路径不完整消费的立体合成参数：

- `temporal_strength`
- `edge_threshold`
- `edge_dilation`
- `mask_feather_radius`
- `hole_fill_mode`
- `hole_fill_radius`
- `hole_fill_strength`

优势：

- 最接近旧版 OpenXR 行为。
- 延迟最低。
- 适合实时头显使用和交互式调参。
- 不需要在 runtime 端生成并传输左右眼完整图像。

劣势：

- 不调用完整 `stereo_runtime.synthesize_stereo()`。
- 补洞、边缘处理、遮罩羽化和时序合成不会完整影响头显输出。
- 不适合作为影院 / 静态图像等质量优先模式的最终输出路径。

正确用途：

- Traditional / Fastest。
- 旧版 OpenXR 行为兼容。
- 低延迟优先场景。
- OpenXR direct shader 调参。

## 2. OpenXR prewarp eyes 路径

该路径在 runtime 中先生成左右眼图像，再交给 viewer 上传。

```text
runtime:
RGB + depth -> render_openxr_stereo() -> left_eye + right_eye

viewer:
left_eye + right_eye -> runtime direct eye textures -> 头显
```

当前实现定位：

- `render_openxr_stereo()` 位于 `stereo_runtime.openxr_render`。
- 它使用 `OpenXRRenderConfig` 中的 `depth_strength`、`convergence`、`ipd`、`stereo_scale`、`max_shift_ratio` 和 `screen_roll`。
- 它不等同于完整 `synthesize_stereo()` 管线。
- 当前不是 GUI 立体 preset 的主映射路径。

优势：

- runtime 能控制 OpenXR warp。
- viewer 更接近左右眼纹理展示器。
- 可作为 RGB+深度 shader 不适用时的实验路径。

劣势：

- 不使用完整立体合成管线。
- 不完整消费影院 / 游戏 / 静态图像 preset 的补洞、边缘和时序参数。
- 比 RGB+深度路径更高带宽、更高开销。

正确用途：

- 兼容性实验。
- OpenXR warp 算法验证。
- 不作为当前质量 preset 的主要路径。

## 3. OpenXR full stereo synthesis eyes 路径

这是当前影院、游戏、静态图像和 debug/export 在 OpenXR 下使用的质量路径。

```text
runtime:
RGB -> 深度模型 -> stereo_runtime.synthesize_stereo()
-> StereoRuntimeResult(left_eye, right_eye, sbs, depth)
-> openxr_result_from_stereo_result()
-> OpenXRRuntimeResult(left_eye, right_eye, depth)

viewer:
left_eye + right_eye -> runtime direct eye textures -> 头显
```

当前实现要点：

- `RuntimePipelineLoop.run()` 在 OpenXR 且 full synthesis 启用时调用 `stereo_runtime.process_rgb_frame()`。
- `process_rgb_frame()` 生成普通 `StereoRuntimeResult`，包含 `left_eye`、`right_eye`、`sbs`、`depth`、`timing`、`provider_info` 和 `debug_info`。
- `openxr_result_from_stereo_result()` 将 `StereoRuntimeResult` 转换为 `OpenXRRuntimeResult`。
- 对 `quality_4k` / `hq_4k` 等已经生成全尺寸 `left_eye` / `right_eye` 的结果，转换函数保留全尺寸眼图，避免把 3840x2160 错降为 half-SBS 的 1920x2160 单眼纹理。
- 只有 `fast_plus` fused half-SBS 这类 left/right 不代表最终眼图的结果，才会拆分 `sbs`，并将 `runtime_output_pack_backend` 标记为 `split_half_sbs`。
- 转换后的 `debug_info["runtime_output_format"]` 为 `"openxr_full_synthesis_eyes"`。
- 转换函数保留 `timing`、`provider_info`、`depth`，并补充 `runtime_output_dtype`、`runtime_output_eye_size`、`runtime_output_display_size` 和 `runtime_output_pack_backend`。
- OpenXR viewer 初始化时优先使用 `runtime_output_display_size` 推导虚拟屏幕宽高；即使 fused half-SBS 单眼纹理是 1920x2160，也会按 3840x2160 的 16:9 显示尺寸建屏。
- OpenXR viewer 继续使用 runtime direct eyes 上传路径，不需要改 viewer 协议。

该路径消费的参数：

- `backend` / `Stereo Quality`：`fast`、`fast_plus`、`quality_4k`、`hq_4k`
- `depth_strength`
- `convergence`
- `ipd` / `ipd_mm`
- `stereo_scale`
- `max_shift_ratio`
- `foreground_scale`
- `depth_antialias_strength`
- `temporal_strength`
- `edge_threshold`
- `edge_dilation`
- `mask_feather_radius`
- `hole_fill_mode`
- `hole_fill_radius`
- `hole_fill_strength`
- output packing / runtime uint8 / fast_plus fused 相关参数

优势：

- OpenXR 现在能直接显示完整立体合成结果。
- 影院、游戏、静态图像 preset 的补洞、边缘、时序和输出打包逻辑能进入头显输出。
- 与非 OpenXR viewer 共享 `process_rgb_frame()` / `synthesize_stereo()` 质量路径。

劣势：

- 延迟和 GPU / 内存带宽开销高于 RGB+深度 shader 路径。
- 参数变化需要重新计算左右眼结果。
- 4K / HQ 配置需要按设备实际性能评估。

正确用途：

- Cinema。
- Game / Low Latency，配合 `fast_plus` 或低延迟 preset。
- Still Image / HQ。
- Debug / Export。
- 用户显式设置 `D2S_OPENXR_RUNTIME_DIRECT=0` 时的 OpenXR 输出。

## 当前路径选择逻辑

核心选择点在 `RuntimePipelineLoop.run()`：

```text
if run_mode == OpenXR and full_synthesis_enabled is false:
    process_openxr_frame()        # OpenXR RGB+depth
else:
    process_rgb_frame()           # normal full stereo synthesis
    if run_mode == OpenXR:
        openxr_result_from_stereo_result()
```

`full_synthesis_enabled` 的当前规则：

```text
run_mode != OpenXR
-> false

run_mode == OpenXR and openxr_runtime_direct == false
-> true

run_mode == OpenXR and stereo_active_preset in {
    cinema,
    game_low_latency,
    still_image_hq,
    debug_export,
}
-> true

其他 OpenXR 情况
-> false
```

因此，`D2S_OPENXR_RUNTIME_DIRECT=1` 仍是默认值，但不再代表所有 OpenXR preset 都强制走 RGB+深度；它只保留传统 / Fastest 等非质量 preset 的 direct shader 路径。影院、游戏和静态图像 preset 会因为 active preset 映射而进入完整合成 eyes 路径。

## 当前立体模式到 OpenXR 路径映射

| GUI / preset | runtime preset | OpenXR 路径 | 主要 runtime 调用 | 输出 format |
|---|---|---|---|---|
| Traditional / Fastest | `traditional_fastest` | RGB+深度 shader | `process_openxr_frame()` | `openxr_rgb_depth` |
| Cinema | `cinema` | full stereo synthesis eyes | `process_rgb_frame()` -> `openxr_result_from_stereo_result()` | `openxr_full_synthesis_eyes` |
| Game / Low Latency | `game_low_latency` | full stereo synthesis eyes | `process_rgb_frame()` -> `openxr_result_from_stereo_result()` | `openxr_full_synthesis_eyes` |
| Still Image / HQ | `still_image_hq` | full stereo synthesis eyes | `process_rgb_frame()` -> `openxr_result_from_stereo_result()` | `openxr_full_synthesis_eyes` |
| Debug / Export | `debug_export` | full stereo synthesis eyes | `process_rgb_frame()` -> `openxr_result_from_stereo_result()` | `openxr_full_synthesis_eyes` |
| Auto | `auto` resolves to active preset | 跟随 active preset | 跟随 active preset | 跟随 active preset |
| `D2S_OPENXR_RUNTIME_DIRECT=0` | 任意 | full stereo synthesis eyes | `process_rgb_frame()` -> `openxr_result_from_stereo_result()` | `openxr_full_synthesis_eyes` |

## Viewer 协议和上传路径

viewer 侧不需要新增协议。runtime queue 仍然发送：

```text
(runtime_result, capture_start_time)
```

viewer 根据 `runtime_result.debug_info["runtime_output_format"]` 分流：

- `openxr_rgb_depth`：上传 `source_rgb` + `depth`，由 OpenXR shader 生成视差。
- 其他 runtime direct eyes format，包括 `openxr_full_synthesis_eyes`：上传 `left_eye` + `right_eye`，直接作为头显左右眼纹理来源。

近期修复还统一了深度参数命名：

- OpenXR runtime 创建 viewer 时传入 `depth_strength=config.depth_strength`。
- `OpenXRViewerCore.depth_ratio` 保留为兼容 property，读写时映射到 `depth_strength`。
- overlay / environment profile 中旧的 `depth_ratio` 读取不会再导致 `AttributeError`。

## 补洞模式真实差异

当前 GUI 暴露的四个补洞模式并不是同一种高质量补洞策略的不同名称。按当前代码，真实差异如下：

| GUI 补洞模式 | 内部值 | 补洞技术 | 主要效果 | 性能影响预估 |
|---|---:|---|---|---|
| 柔和 / 低重影 | `soft_low_ghost` | `edge_aware_fill`，半径 `1`，强度 `0.6` | 轻补洞，少拉扯、少重影，但洞 / 边缘修复力度也最弱 | 最快。CUDA 可走 `triton_radius1`，通常低于标准 |
| 均衡 / 标准 | `balanced` | `edge_aware_fill`，半径 `3`，强度 `1.0` | 默认实时档，补洞更完整，仍保持实时性能 | 快。CUDA 可走 `triton_radius3`；实测 4K `quality_4k` 总合成约 `22-28ms` |
| 锐利 / 高细节 | `sharp_test` | `edge_aware_fill`，半径 `1`，强度 `1.0` | 比柔和更强，但采样半径小，尽量不大范围糊边；偏测试 / 对比档 | 很快。radius=1 采样范围小；若不满足专用 Triton 条件则回退 `torch_avg_pool` |
| 内容感知 / 最高质量 | `quality` | `directional_edge_aware_fill`，方向感知 + 深度 / shift 边缘判断 + UI 高频保护 + blur 混合 | 遮挡边缘质量最好，减少前景颜色拖进空洞，保护文字 / 高频边缘 | 最慢。当前是 torch 组合算子；实测 4K `quality_4k` 总合成约 `90-93ms` |

关键点：

- `柔和 / 均衡 / 锐利` 都是快速 `edge_aware_fill` 系列，只是 `radius` / `strength` 不同。
- `内容感知 / 最高质量` 才是 `directional_edge_aware_fill` 重策略。

更具体的参数差异：

```text
柔和: radius=1, strength=0.6
均衡: radius=3, strength=1.0
锐利: radius=1, strength=1.0
高质量: radius=3, strength=1.0 + directional/content-aware 额外逻辑
```

所以 OpenXR 实时建议默认使用 `均衡 / 标准`。只有专门对比画质，或者做静态图 / 导出时，才建议使用 `内容感知 / 最高质量`。

## 性能和质量取舍

| 路径 | 兼容旧版 | 完整合成 | 低延迟 | 使用补洞/边缘/时序 | 最佳用途 |
|---|---|---|---|---|---|
| OpenXR RGB+深度 shader | 是 | 否 | 最佳 | 否 | Traditional / Fastest，实时调参 |
| OpenXR prewarp eyes | 否 | 否 | 中等 | 否 | 兼容性 / 实验 |
| OpenXR full synthesis eyes | 否 | 是 | 较高开销 | 是 | Cinema、Game、Still Image、Debug/Export |

## 当前验证覆盖

与当前实现相关的回归测试覆盖了以下行为：

- `run_mode="OpenXR"` 且 `openxr_runtime_direct=True`、`stereo_active_preset="traditional_fastest"` 时调用 `process_openxr_frame()`，输出 `openxr_rgb_depth`。
- `run_mode="OpenXR"` 且 `openxr_runtime_direct=False` 时调用 `process_rgb_frame()`，并将结果转换为 `OpenXRRuntimeResult`。
- `cinema`、`game_low_latency`、`still_image_hq` 在 OpenXR direct 默认打开时仍进入 full synthesis eyes。
- `StereoRuntimeResult -> OpenXRRuntimeResult` 转换保留左右眼、深度、timing、provider_info，并设置 `openxr_full_synthesis_eyes`。
- `traditional_fastest` 是合法 runtime preset，并保持 OpenXR RGB+深度路径。
- OpenXR runtime viewer 构造使用 `depth_strength`，不再传旧的 `depth_ratio` 参数名。

最近验证命令：

```powershell
src\python3\python.exe -m py_compile src\stereo_runtime\runtime.py src\stereo_runtime\pipeline.py src\stereo_runtime\openxr_state.py src\stereo_runtime\session_helpers.py src\app_runtime\runtime_callbacks.py src\app_runtime\runtime_context.py src\xr_viewer\openxr_runtime.py src\xr_viewer\implementation.py

src\python3\python.exe -m pytest tests\test_runtime_pipeline.py tests\test_runtime_openxr.py tests\test_openxr_state.py tests\test_session_helpers.py tests\test_runtime_context.py tests\test_presets.py tests\test_adapter_config.py tests\test_openxr_runtime.py
```

## 后续建议

当前 OpenXR 渲染路径已经补齐。后续工作不应再把“full synthesis 尚未接入 OpenXR”作为已知缺口。

建议后续单独处理：

- GUI 控件可用性整理：在 RGB+深度 shader 路径下标记或隐藏不会生效的完整合成参数。
- 对 `cinema`、`game_low_latency`、`still_image_hq` 分别做真实头显帧时间基准测试。
- 按设备区分 full synthesis eyes 的默认质量级别，尤其是游戏 preset。
- 保留 `traditional_fastest` 作为旧版兼容和低延迟回退路径。
