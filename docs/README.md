# 文档索引

本目录按“当前入口 / 架构与运行时 / 平台与兼容性 / 专项计划与调查 / benchmark 记录 / 历史归档”组织。交接给其他 Agent 时，优先阅读当前入口文档，不要从归档文档反推当前实现状态。

## 当前入口

| 文档 | 用途 |
|---|---|
| [00-api-handoff-progress.md](00-api-handoff-progress.md) | 当前唯一交接入口，记录项目状态、边界、验证命令和下一步 |
| [01-Realtime-2d-to-3d-specification.md](01-Realtime-2d-to-3d-specification.md) | Desktop2Stereo 当前正式运行时流程规范 |
| [02-desktop2stereo-engineering-design-specification.md](02-desktop2stereo-engineering-design-specification.md) | 基于运行时规范的工程实现、迁移、兼容清理和符合状态说明 |

## 架构与运行时

| 文档 | 用途 |
|---|---|
| [11-visual-regression-guide.md](11-visual-regression-guide.md) | 视觉回归输出图片含义、检查方法和固定样例说明 |
| [12-openxr-stereo-runtime-plan.md](12-openxr-stereo-runtime-plan.md) | OpenXR per-eye core 与 runtime 集成计划 |
| [13-realtime-stereo-parameter-guide.md](13-realtime-stereo-parameter-guide.md) | 当前实时立体参数测试与视觉回归调参指南 |
| [16-depth-py-migration-map.md](16-depth-py-migration-map.md) | legacy `depth.py` 功能迁移清单 |
| [17-multiplatform-provider-layout.md](17-multiplatform-provider-layout.md) | 多平台 depth provider 目录分层和 artifact 规划 |
| [18-host-bootstrap-device-flow.md](18-host-bootstrap-device-flow.md) | Host/GUI/capture bootstrap 设备检测与 runtime 参数传递流程 |
| [19-capture-architecture-flow.md](19-capture-architecture-flow.md) | capture 子系统架构、数据流和组件边界 |
| [20-openxr-gpu-glow-guide.md](20-openxr-gpu-glow-guide.md) | OpenXR GPU glow 的原理、调用链、配置和复杂效果扩展方法 |
| [37-openxr-projection-quad-layer-convergence-note.md](37-openxr-projection-quad-layer-convergence-note.md) | OpenXR projection 与 quad layer 架构收敛记录 |

## 平台与兼容性

| 文档 | 用途 |
|---|---|
| [27-vr-headset-focal-distance-reference.md](27-vr-headset-focal-distance-reference.md) | VR/AR 头显最佳虚拟屏幕距离与尺寸速查表 |
| [31-directml-fallback-and-d3d11-d3d12-bridge-survey.md](31-directml-fallback-and-d3d11-d3d12-bridge-survey.md) | ONNX Runtime DirectML 兜底与 D3D11-D3D12 零拷贝桥接调查 |
| [32-macos-zero-copy-capture-inference-survey.md](32-macos-zero-copy-capture-inference-survey.md) | macOS ScreenCaptureKit 到 MPS/CoreML 零拷贝实现调查 |
| [35-OpenXR_Asynchronous_Decoupled_Rendering_Architecture_Report.md](35-OpenXR_Asynchronous_Decoupled_Rendering_Architecture_Report.md) | OpenXR 异步渲染架构设计与实现报告 |
| [36-OpenXR_Asynchronous_Decoupled_Rendering_Implementation_Plan.md](36-OpenXR_Asynchronous_Decoupled_Rendering_Implementation_Plan.md) | OpenXR 异步解耦渲染重构实施计划 |

## 专项计划与调查

| 文档 | 用途 |
|---|---|
| [29-i18n-implementation-plan.md](29-i18n-implementation-plan.md) | Desktop2Stereo i18n / l10n 实现方案 |
| [30-jiggly-doodling-blum.md](30-jiggly-doodling-blum.md) | logging + Flet 原生日志和进度面板计划 |
| [33-quality-buffered-output-feasibility-report.md](33-quality-buffered-output-feasibility-report.md) | 高画质缓冲输出技术可行性报告 |
| [34-visiondepth3d-comparison-and-offline-workflow-report.md](34-visiondepth3d-comparison-and-offline-workflow-report.md) | VisionDepth3D 架构与离线工作流对比报告 |
| [38-gltf-2-renderer-compliance-layer-plan.md](38-gltf-2-renderer-compliance-layer-plan.md) | glTF 2.0 renderer compliance layer 计划，统一 parser、mesh/material contract、render pass 和 OpenGL/D3D11 行为 |
| [39-panda3d-gltf-openxr-d3d11-migration-plan.md](39-panda3d-gltf-openxr-d3d11-migration-plan.md) | Panda3D 接管 glTF 场景、OpenGL 离屏渲染并桥接至 D3D11 OpenXR 的可行性闸门与迁移计划 |
| [python-project-designguide.md](python-project-designguide.md) | Python 项目结构、依赖管理、测试和发布参考指南 |

## Benchmark 与优化记录

| 文档 | 用途 |
|---|---|
| [benchmark/07-depth-backend-benchmark.md](benchmark/07-depth-backend-benchmark.md) | depth backend、Python 环境、TensorRT/ONNX/PyTorch 性能对比 |
| [benchmark/08-synthesis-optimization-log.md](benchmark/08-synthesis-optimization-log.md) | synthesis / depth 优化历史和取舍记录 |
| [benchmark/10-rtx3090-fused-synthesis-results.md](benchmark/10-rtx3090-fused-synthesis-results.md) | RTX 3090 fused synthesis 正式结果 |

## 历史归档

`archive/` 保留早期讨论、方案评估、经验报告和边界定义，用于追溯背景，不作为当前实现状态的唯一依据：

- [archive/01-algorithm-survey.md](archive/01-algorithm-survey.md)
- [archive/02-4k-performance-budget.md](archive/02-4k-performance-budget.md)
- [archive/03-iw3-comparison.md](archive/03-iw3-comparison.md)
- [archive/04-implementation-plan.md](archive/04-implementation-plan.md)
- [archive/05-model-boundary.md](archive/05-model-boundary.md)
- [archive/implementation-experience/README.md](archive/implementation-experience/README.md)
- [archive/implementation-experience/4K 高质量立体生成算法实现计划书.md](archive/implementation-experience/4K 高质量立体生成算法实现计划书.md)
- [archive/implementation-experience/FSR1 EASU + RCAS 上采下采优化计划.md](archive/implementation-experience/FSR1 EASU + RCAS 上采下采优化计划.md)
- [archive/implementation-experience/14-host-api-preset-examples.md](archive/implementation-experience/14-host-api-preset-examples.md)
- [archive/implementation-experience/15-host-api-contract.md](archive/implementation-experience/15-host-api-contract.md)
- [archive/implementation-experience/21-openxr-ghosting-test-report.md](archive/implementation-experience/21-openxr-ghosting-test-report.md)
- [archive/implementation-experience/22-cinema-ipd64-production-sweep-guide.md](archive/implementation-experience/22-cinema-ipd64-production-sweep-guide.md)
- [archive/implementation-experience/23-saved-cinema-settings-samples-guide.md](archive/implementation-experience/23-saved-cinema-settings-samples-guide.md)
- [archive/implementation-experience/24-openxr-render-path-report.md](archive/implementation-experience/24-openxr-render-path-report.md)

## 编码提醒

中文 Markdown 使用 UTF-8。PowerShell 中读取中文文档时请显式指定：

```powershell
Get-Content docs\00-api-handoff-progress.md -Encoding UTF8
```
