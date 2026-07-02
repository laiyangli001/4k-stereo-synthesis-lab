# 文档索引

本目录按“当前入口 / 专项说明 / benchmark 记录 / 历史归档”组织。交接给其他 Agent 时，优先阅读当前入口文档，不要从归档文档反推当前实现状态。

## 当前入口

| 文档 | 用途 |
|---|---|
| [00-api-handoff-progress.md](00-api-handoff-progress.md) | 当前唯一交接入口，记录项目状态、边界、验证命令和下一步 |
| [01-Realtime-2d-to-3d-specification.md](01-Realtime-2d-to-3d-specification.md) | Desktop2Stereo 当前正式最终运行时流程规范；docs/25 已作废，以本文为准 |

## 专项说明

| 文档 | 用途 |
|---|---|
| [11-visual-regression-guide.md](11-visual-regression-guide.md) | 视觉回归输出图片含义、检查方法和固定样例说明 |
| [12-openxr-stereo-runtime-plan.md](12-openxr-stereo-runtime-plan.md) | OpenXR per-eye core 与未来 runtime 集成计划 |
| [13-realtime-stereo-parameter-guide.md](13-realtime-stereo-parameter-guide.md) | 当前实时立体参数测试与视觉回归调参指南 |
| [17-multiplatform-provider-layout.md](17-multiplatform-provider-layout.md) | 多平台 depth provider 目录分层和 artifact 规划 |
| [18-host-bootstrap-device-flow.md](18-host-bootstrap-device-flow.md) | Host/GUI/capture bootstrap 设备检测与 runtime 参数传递流程 |
| [20-openxr-gpu-glow-guide.md](20-openxr-gpu-glow-guide.md) | OpenXR GPU glow 的原理、调用链、配置和复杂效果扩展方法 |
| [01-desktop2stereo-engineering-design-specification.md](01-desktop2stereo-engineering-design-specification.md) | 基于 docs/01 的工程实现、迁移、兼容清理和符合状态说明 |
| [25-2d-to-3d-runtime-specification.md](25-2d-to-3d-runtime-specification.md) | 已作废的历史运行时规范；仅用于追溯背景，不作为当前裁决来源 |

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
