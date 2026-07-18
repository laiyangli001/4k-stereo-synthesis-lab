# 文档索引

`docs/` 顶层只保存当前有效的 Vulkan 架构、运行时规范和产品参考。阶段计划、旧运行时设计、调研报告、交接记录和历史 benchmark 统一保存在 [`archive/`](archive/README.md)，不得作为当前实现依据。

## 当前规范

| 文档 | 定位 |
|------|------|
| [01.D2S_Vulkan_Migration_Technical_Report.md](01.D2S_Vulkan_Migration_Technical_Report.md) | 从混合图形 API 转向 Vulkan 的技术决策与实施依据 |
| [01-Realtime-2d-to-3d-specification.md](01-Realtime-2d-to-3d-specification.md) | Vulkan 实时 2D 转 3D 系统行为、数据流和验收规格 |
| [02-desktop2stereo-engineering-design-specification.md](02-desktop2stereo-engineering-design-specification.md) | C++20/Vulkan/Filament/OpenXR 目标态工程设计 |

阅读顺序：先阅读迁移报告了解决策背景，再阅读 `01` 确认系统行为，最后使用 `02` 指导工程实现。

## 当前指南与参考

| 文档 | 定位 |
|------|------|
| [13-realtime-stereo-parameter-guide.md](13-realtime-stereo-parameter-guide.md) | 视差预算、深度强度、汇聚和视觉回归测试指南 |
| [27-vr-headset-focal-distance-reference.md](27-vr-headset-focal-distance-reference.md) | OpenXR 虚拟屏幕距离、尺寸和水平视角参考 |

参考文档不得覆盖 `01` 和 `02` 的规范性要求。设备数据或产品参数发生变化时，应单独验证并更新参考文档。

## 历史归档

归档内容按用途分类：

| 目录 | 内容 |
|------|------|
| [archive/handoffs/](archive/handoffs/) | 历史交接和临时任务提示 |
| [archive/legacy-runtime/](archive/legacy-runtime/) | Python、OpenGL、D3D11、Panda3D 和旧 OpenXR 架构 |
| [archive/completed-plans/](archive/completed-plans/) | 已完成或已被新架构替代的实施计划 |
| [archive/research/](archive/research/) | 平台、后端和功能可行性调研 |
| [archive/visual-regression/](archive/visual-regression/) | 旧合成路径的视觉回归说明 |
| [archive/benchmark/](archive/benchmark/) | 历史性能结果和优化记录 |
| [archive/implementation-experience/](archive/implementation-experience/) | 早期实现经验、测试报告和 Host API 资料 |

完整说明见 [archive/README.md](archive/README.md)。

## 维护规则

1. 顶层文档必须对应当前 Vulkan 主线，不新增迁移过程流水账。
2. 新的规范性结论优先写入 `01` 或 `02`，参数测试结论写入 `13`。
3. 已完成计划和被替代设计直接移动到 `archive/`，同时修正入口链接。
4. 归档文件保留原文用于追溯，不要求继续适配当前模块名。
5. 中文 Markdown 统一使用 UTF-8。
