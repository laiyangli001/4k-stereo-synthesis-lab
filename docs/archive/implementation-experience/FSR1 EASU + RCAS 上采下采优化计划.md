# FSR1 EASU + RCAS 上采/下采优化计划

## Summary
目标是把分辨率处理拆成两层：推理前只做“合理工作分辨率”的 GPU 缩放，保证深度稳定和性能；推理和 stereo warp 之后，在 Local Viewer 最终显示阶段加入 FSR1 EASU + RCAS pass，负责最终上采样和锐化。第一阶段只接 Local Viewer / 3D Monitor，OpenXR 暂不改。

## Key Changes
- **推理前缩放**
  - 保留 `Processing Resolution: Auto`：根据 `Stereo Output` 显示器和 `Display Mode` 计算推理/warp 工作高度。
  - 4K 输入 -> 2K/1K 输出：在 `depth.process()` 里先 GPU 下采样，再推理。
  - 1K/2K 输入 -> 4K 输出：不再为了推理强行放到 4K；默认仍按 Auto 的合理工作分辨率处理，避免深度模型吃锐化伪影。
  - 保留手动覆盖能力：`Processing Resolution: Auto/720/1080/1440/2160`，旧配置缺失时默认 Auto。

- **最终显示上采样**
  - 在 `viewer.py` 增加 offscreen render target：现有 stereo 渲染先画到 offscreen texture，再由最终 pass 输出到默认 framebuffer。
  - 增加 `FSR1 EASU + RCAS` 最终 pass：
    - 当 offscreen 分辨率低于窗口 framebuffer 时：执行 EASU 上采样，再 RCAS 锐化。
    - 当 offscreen 分辨率等于窗口 framebuffer 时：只执行 RCAS 或直接 bypass。
    - 当 offscreen 分辨率高于窗口 framebuffer 时：不跑 EASU，使用现有/线性下采样路径，避免 FSR1 被错误用于下采。
  - overlay 渲染保持在最终 pass 之后，确保 FPS/延迟文字不被 FSR 锐化或拉伸。

- **配置和 GUI**
  - 新增配置：
    - `Upscaler: Off | FSR1`
    - `Upscaler Sharpness: 0.0 - 1.0`
  - 默认：
    - `Upscaler: Off`
    - `Upscaler Sharpness: 0.35`
  - GUI 在 Local Viewer / 3D Monitor 模式显示该选项；OpenXR Link 下隐藏。
  - 中文：
    - `Upscaler:` -> `画面增强:`
    - `Off` -> `关闭`
    - `FSR1` 保持 `FSR1`
    - `Upscaler Sharpness:` -> `增强锐度:`

- **渲染行为**
  - FSR1 只处理最终合成画面，不参与深度推理输入。
  - 支持 `Full-SBS / Half-SBS / Full-TAB / Half-TAB / Mono / Depth Map`，但第一版重点验证 SBS/TAB。
  - CUDA-GL RGB/depth 上传路径不变。
  - CPU fallback 不变。
  - OpenXR D3D11 渲染链路不变。

## Implementation Details
- `utils.py`
  - 读取 `Upscaler` 和 `Upscaler Sharpness`，导出给 `main.py`。
  - 保持旧 `settings.yaml` 兼容，缺失字段使用默认值。
  - `diagnose_displays.py` 增加打印当前 `Processing Resolution`、`Upscaler`、`Sharpness`。

- `main.py`
  - 创建 `StereoWindow` 时传入 `upscaler` 和 `upscaler_sharpness`。
  - FPS breakdown 保持现有字段，必要时额外打印 final pass 时间，例如 `post=...ms`。

- `viewer.py`
  - 增加 FBO/texture 管理：
    - `_ensure_postprocess_target(width, height)`
    - `_begin_scene_target()`
    - `_end_scene_target_and_present()`
    - `_release_postprocess_resources()`
  - 将现有 `render()` 的主画面绘制包进 offscreen target。
  - 增加 FSR shader program：
    - EASU pass：低分辨率 offscreen texture -> intermediate texture / 或直接输出。
    - RCAS pass：EASU 结果 -> 默认 framebuffer。
  - 如果 `Upscaler=Off`，继续使用原有直接渲染路径，降低回归风险。
  - 如果 FBO/shader 初始化失败，自动降级到 Off，并打印一次 warning。

## Test Plan
- **配置兼容**
  - 旧 `settings.yaml` 无 `Upscaler` / `Sharpness` / `Processing Resolution` 时能启动。
  - GUI 保存后字段存在，重启后能恢复。
  - Local Viewer 显示选项，OpenXR Link 隐藏选项。

- **功能验证**
  - Local Viewer + Full-SBS + 4K 输出 + 1080p/1440p 输入：FSR1 开启后画面更清晰，FPS overlay 正常。
  - Half-SBS / Full-TAB / Half-TAB 不错位、不裁切、不黑屏。
  - `Upscaler=Off` 与现有画面行为一致。
  - FSR shader 初始化失败时自动回退，不影响主程序运行。

- **性能验证**
  - 记录同一场景：
    - Off
    - FSR1 Sharpness 0.25
    - FSR1 Sharpness 0.35
    - FSR1 Sharpness 0.5
  - 对比 FPS breakdown 中 `update/render/swap/post`。
  - 目标：4K 虚拟输出下仍尽量保持 90fps；如果 FSR1 影响明显，默认仍保持 Off。

- **画质验证**
  - 1K/2K 输入到 4K 输出：检查 UI文字、图片边缘、游戏画面、深度边缘。
  - Sharpness > 0.6 时重点检查光晕、边缘过锐、纹理闪烁。
  - Depth Map 模式确认深度图不被误作为推理输入重新处理。

## Assumptions
- FSR1 只作为最终显示增强，不承诺真实恢复 4K 原始细节。
- 第一版只实现 Local Viewer / 3D Monitor；OpenXR 不动。
- 默认 `Upscaler=Off`，避免改变老用户画面。
- 推荐测试起点：`FSR1 + Sharpness 0.35`。
- 下采样仍由推理前 `Processing Resolution` 和现有 GPU resize 负责；FSR1 不用于下采样。
