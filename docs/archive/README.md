# 历史文档归档

本目录保存已被Python Vulkan架构替代、已完成或仅用于历史研究的文档。归档内容不代表当前实现状态，也不能覆盖顶层规范。

当前有效文档请从 [../README.md](../README.md) 进入。

## 分类

| 目录 | 归档范围 |
|------|----------|
| `handoffs/` | 历史工作交接、临时 prompt 和阶段状态 |
| `legacy-runtime/` | 旧OpenGL/D3D11数据面、Panda3D、旧OpenXR和glTF迁移方案 |
| `completed-plans/` | 已完成或已失效的 GUI、日志、i18n 等实施计划 |
| `research/` | DirectML、macOS、缓冲输出、产品对比和通用 Python 工程调研 |
| `visual-regression/` | 基于旧 synthesis backend 的视觉回归说明 |
| `benchmark/` | 历史 depth/synthesis benchmark 和优化记录 |
| `implementation-experience/` | 早期 API、参数、OpenXR 测试和实现经验 |

根目录中的 `01` 至 `05` 是更早期的算法和实现调研，同样只保留作历史参考。

## 使用规则

- 查找当前系统行为：阅读 `../01-Realtime-2d-to-3d-specification.md`。
- 查找当前工程设计：阅读 `../02-desktop2stereo-engineering-design-specification.md`。
- 查找架构决策背景：阅读 `../01.D2S_Vulkan_Migration_Technical_Report.md`。
- 引用归档内容时必须注明“历史方案”，不得描述为当前主路径。
- 归档文件内部可能保留移动前路径和旧模块名，这是历史上下文的一部分。
