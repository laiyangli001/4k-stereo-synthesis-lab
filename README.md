# 4K Stereo Synthesis Lab

`4k-stereo-synthesis-lab` 是 Desktop2Stereo v3.0beta 工作区，包含 Flet GUI、桌面捕获、OpenXR/XR viewer、streaming、depth provider、stereo synthesis、输出打包、preset、benchmark 和视觉回归工具。

当前主线目标是在 4K RGB + depth 输入下，提供稳定的 Desktop2Stereo host/runtime，并验证 layered / occlusion-aware / symmetric stereo synthesis 是否能超过简单 depth-shift 路线。

## 当前状态

- 4K `fast` baseline、`quality_4k`、`hq_4k` synthesis 已实现。
- Half-SBS、Full-SBS、TAB、mono、depth map、anaglyph、interleaved、Leia 输出已支持。
- Flet GUI、Windows/macOS/Linux 捕获后端、本地 viewer、OpenXR Link、XR 预览窗口已接入工作区。
- Distill-Any-Depth、InfiniDepth、Depth-Anything V1/V2/V3、Video-Depth-Anything、DepthPro 等模型通过统一 registry/provider 管理。
- NVIDIA CUDA/TensorRT、AMD ROCm/MIGraphX、Apple MPS/CoreML、Intel OpenVINO/XPU 等后端按平台和模型能力选择。
- Host preset/API 层已实现：`cinema`、`game_low_latency`、`still_image_hq`、`debug_export`、`auto`。
- OpenXR viewer 支持环境模型、XR 多指触控、控制器快捷键、会聚点和 VSync 设置。

## 硬边界

- 不为了性能降低 depth 推理分辨率。
- 不改变当前 Base 模型 `294x518` / `depth_resolution=518` 路径的语义。
- 不把模型、engine、outputs、Python 便携环境提交到 GitHub。
- 不把本地生成的模型加速产物写进其它 Desktop2Stereo 分支目录。

## 目录

| 路径 | 用途 |
|---|---|
| `src/gui/` | Flet GUI 与配置管理 |
| `src/capture/` | 桌面/窗口捕获后端与 capture runner |
| `src/stereo_runtime/` | 推荐对外导入包和当前核心实现，面向 Desktop2Stereo/host runtime |
| `src/viewer/` | 本地 stereo viewer |
| `src/xr_viewer/` | OpenXR viewer、环境模型和控制器交互 |
| `src/streaming/` | RTMP/HLS/MJPEG 等输出链路 |
| `tests/` | 单元测试和 smoke contract 测试 |
| `scripts/benchmark/` | 性能测试和 profile |
| `scripts/tools/` | ONNX 导出、深度图生成、对比、视觉回归工具 |
| `scripts/smoke/` | Host/API smoke 示例 |
| `scripts/examples/` | demo 和 OpenXR preview |
| `scripts/windows/` | 可见窗口 Windows 启动脚本 |
| `docs/` | 当前文档入口 |
| `docs/` | Vulkan 当前规范、工程设计和参考指南 |
| `docs/archive/` | 旧运行时、阶段计划、调研和 benchmark 归档 |

## 重要文档

- 文档总入口：[docs/README.md](docs/README.md)
- Vulkan 迁移报告：[docs/01.D2S_Vulkan_Migration_Technical_Report.md](docs/01.D2S_Vulkan_Migration_Technical_Report.md)
- 实时系统规格：[docs/01-Realtime-2d-to-3d-specification.md](docs/01-Realtime-2d-to-3d-specification.md)
- Vulkan 工程设计：[docs/02-desktop2stereo-engineering-design-specification.md](docs/02-desktop2stereo-engineering-design-specification.md)
- 立体参数指南：[docs/13-realtime-stereo-parameter-guide.md](docs/13-realtime-stereo-parameter-guide.md)
- 历史文档索引：[docs/archive/README.md](docs/archive/README.md)

## 启动

Windows GUI：

```powershell
src\python3\python.exe src\main.py
```

若使用仓库内批处理入口：

```powershell
src\main.bat
```

## 快速验证

```powershell
src\python3\python.exe -B -m pytest -q
```

Host API smoke：

```powershell
src\python3\python.exe -B scripts\smoke\host_api_smoke.py --preset cinema --output-format half_sbs --out -
```

4K end-to-end benchmark：

```powershell
src\python3\python.exe -B scripts\benchmark\bench_end_to_end_4k.py --rgb 4K.jpg --backend quality_4k --layers 2 --depth-backend tensorrt_native --output-format half_sbs --output-format full_sbs
```
