# API Handoff Progress

更新时间：2026-06-16  
当前提交：`5c5ac7b`  
项目目录：`D:\AI_2D_to_3D\4.LC700X_Desktop2Stereo\4k-stereo-synthesis-lab`

## 目标

本项目是独立的 4K stereo synthesis 实验仓库，目标是在不直接修改 `Desktop2Stereo` 主项目的前提下，验证一条更强的 4K 左右眼生成路线。

当前重点已经从单纯比较 SBS 输出，转为优先比较生成深度图，因为深度图最直观反映 3D 效果基础。后续 API 转换时，应把深度推理、立体合成、输出格式三个阶段拆开。

## 已实现功能

### 1. Stereo synthesis 主 API

文件：`src/stereo_lab/synthesis.py`

当前主入口：

```python
def synthesize_stereo(rgb, depth, config=None, temporal_state=None):
    """Return StereoResult(left_eye, right_eye, sbs, debug_info)."""
```

输入约定：

| 参数 | 形状 | 范围 | 说明 |
|---|---|---|---|
| `rgb` | `CHW` 或 `BCHW` | `0..1` | RGB 图像 tensor |
| `depth` | `HW` / `BHW` / `B1HW` | `0..1` | 深度图，1 表示近，0 表示远 |
| `config` | `StereoConfig` | - | 合成配置 |

输出：

| 字段 | 说明 |
|---|---|
| `left_eye` | 左眼图，保持原始单眼分辨率 |
| `right_eye` | 右眼图，保持原始单眼分辨率 |
| `sbs` | 拼接输出，支持 `half_sbs` 和 `full_sbs` |
| `debug_info` | 可选调试信息，例如 occlusion mask、shift map、layer count |

当前 backend：

| backend | 状态 | 说明 |
|---|---|---|
| `fast` | 已实现 | 类 Desktop2Stereo depth-shift baseline |
| `quality_4k` | 已实现原型 | 2-layer occlusion-aware synthesis |
| `hq_4k` | 已实现原型 | 默认至少 3-layer，预留 HQ 路线 |

### 2. 输出格式

文件：`src/stereo_lab/output.py`

已支持：

| 格式 | 输出尺寸 | 说明 |
|---|---|---|
| `half_sbs` | `W x H` | 左右眼各压缩到半宽后拼接 |
| `full_sbs` | `2W x H` | 左右眼保持原始宽度后拼接 |

算法内部始终保留左右眼原始分辨率，最后一步才决定 `half_sbs` 或 `full_sbs`。

### 3. Distill-Any-Depth-Base @ 518 深度推理

文件：`src/stereo_lab/depth_provider.py`

当前模型：

| 项 | 值 |
|---|---|
| 模型名 | `Distill-Any-Depth-Base` |
| Hugging Face ID | `lc700x/Distill-Any-Depth-Base-hf` |
| 深度输入分辨率 | `518` |
| 4K 16:9 patch 对齐输入 | `294x518` |
| patch size | `14` |
| 默认 cache | 项目自己的 `models/` |

当前入口：

```python
depth, provider_info = estimate_distill_any_depth_base_518(
    rgb,
    device="cuda",
    cache_dir=None,
    local_files_only=False,
    force_download=False,
)
```

注意：

- 默认会使用网络启用模式加载 Hugging Face 模型。
- 默认不再写入 `Desktop2Stereo` 的模型目录。
- `provider_info.to_report()` 会记录模型名、模型 ID、分辨率、cache 路径、加载模式。

### 4. 深度图生成与直观对比

文件：`scripts/generate_depth_map.py`

用途：从一张 RGB 图生成 Distill 深度图，优先用于判断 3D 效果基础。

示例：

```powershell
.\python3\python.exe scripts\generate_depth_map.py --rgb input.png --device cuda
```

可选参考深度图：

```powershell
.\python3\python.exe scripts\generate_depth_map.py --rgb input.png --reference-depth ref_depth.png --device cuda
```

输出目录默认：`outputs/depth_compare`

输出文件：

| 文件 | 说明 |
|---|---|
| `input_rgb.png` | 输入 RGB |
| `distill_base_518_depth.png` | 灰度深度图 |
| `distill_base_518_depth_color.png` | 彩色可视化深度图 |
| `reference_depth_matched.png` | 可选，尺寸匹配后的参考 depth |
| `reference_vs_distill_absdiff.png` | 可选，参考 depth 与 Distill depth 差异 |
| `depth_contact_sheet.png` | 汇总预览图 |
| `depth_report.json` | 模型与输出报告 |

可见窗口入口：

```text
scripts/run_visible_generate_depth.bat
```

使用方式：拖一张 RGB 图到 bat 上；也可附带一张 reference depth。

### 5. RGB 自动深度 + stereo 对比

文件：`scripts/compare_methods.py`

支持两种输入方式：

```powershell
# 使用已有 depth
.\python3\python.exe scripts\compare_methods.py --rgb input.png --depth depth.png --device cuda

# 自动用 Distill-Any-Depth-Base @ 518 生成 depth
.\python3\python.exe scripts\compare_methods.py --rgb input.png --auto-depth --depth-provider distill_base_518 --device cuda
```

输出目录默认：`outputs/compare`

新增输出：

| 文件 | 说明 |
|---|---|
| `used_depth.png` | 本次 stereo synthesis 实际使用的 depth |
| `*_left.png` | 各 backend 左眼图 |
| `*_right.png` | 各 backend 右眼图 |
| `*.png` | 各 backend SBS 图 |
| `*_occlusion_mask.png` | 可选，遮挡 mask |
| `contact_sheet.png` | 汇总预览图 |
| `report.json` | 记录模型、参数、输出尺寸、差异指标 |

可见窗口入口：

```text
scripts/run_visible_compare_rgb_auto_depth.bat
```

### 6. ONNX 导出

文件：`scripts/export_distill_base_onnx.py`

用途：把 `Distill-Any-Depth-Base` 导出为 Desktop2Stereo 兼容风格的 FP16 ONNX。

默认输出：

```text
models/models--lc700x--Distill-Any-Depth-Base-hf/model_fp16_294x518.onnx
```

默认参数：

| 项 | 值 |
|---|---|
| 输入名 | `pixel_values` |
| 输出名 | `predicted_depth` |
| dummy input | `1x3x294x518` |
| dtype | `fp16` |
| force download | 默认开启 |

命令：

```powershell
.\python3\python.exe scripts\export_distill_base_onnx.py --device cuda
```

可见窗口入口：

```text
scripts/run_visible_export_distill_base_onnx.bat
```

重要边界：

- 不覆盖 `Desktop2Stereo` 原模型目录。
- 输出和缓存都在本项目 `models/` 下。
- `models/` 已加入 `.gitignore`，不上传 GitHub。

### 7. 测试与验证

已做的轻量验证：

```powershell
.\python3\python.exe -B -c "import ast, pathlib; files=list(pathlib.Path('src').rglob('*.py'))+list(pathlib.Path('scripts').rglob('*.py'))+list(pathlib.Path('tests').rglob('*.py')); [ast.parse(p.read_text(encoding='utf-8'), filename=str(p)) for p in files]; print('syntax ok', len(files), 'files')"
```

说明：

- `compileall` 在当前环境会因为 `__pycache__` 权限问题失败。
- AST 语法检查通过，不写 `.pyc`。

## API 转换建议

建议对外暴露三个阶段接口。

### 1. Depth API

建议形式：

```python
def estimate_depth(rgb, config):
    """Return depth, depth_info."""
```

建议 config：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `provider` | `distill_base_518` | 当前只实现真实 Distill Base |
| `model_id` | `lc700x/Distill-Any-Depth-Base-hf` | 固定记录到报告 |
| `resolution` | `518` | 最长边 518 |
| `device` | `cuda` | 可退化到 CPU |
| `cache_dir` | `models/` | 项目内模型缓存 |
| `local_files_only` | `false` | 离线模式 |
| `force_download` | `false` | 强制重新下载 |
| `output_depth_size` | `input_size` | depth resize 回原图尺寸 |

### 2. Stereo API

建议形式：

```python
def synthesize_stereo(rgb, depth, config):
    """Return left_eye, right_eye, sbs, debug_info."""
```

现有 `StereoConfig` 可直接作为基础：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `backend` | `quality_4k` | `fast`, `quality_4k`, `hq_4k` |
| `layers` | `2` | Quality 默认 2，HQ 至少 3 |
| `occlusion` | `true` | 启用遮挡 mask |
| `symmetric` | `true` | 左右眼对称生成 |
| `hole_fill` | `edge_aware` | `none`, `fast`, `edge_aware` |
| `temporal` | `true` | 启用时序稳定 |
| `output_format` | `half_sbs` | `half_sbs`, `full_sbs` |
| `debug_output` | `false` | 是否返回 mask/shift 等 tensor |
| `depth_strength` | `2.0` | 深度强度 |
| `convergence` | `0.0` | 汇聚偏移 |
| `ipd` | `0.064` | 眼距参数 |
| `max_shift_ratio` | `0.05` | 最大位移比例 |

### 3. One-shot API

后续可以包装成：

```python
def rgb_to_stereo(rgb, depth_config, stereo_config):
    depth, depth_info = estimate_depth(rgb, depth_config)
    result = synthesize_stereo(rgb, depth, stereo_config)
    result.debug_info["depth_info"] = depth_info
    return result
```

这个接口适合给外部 Agent 或 Desktop2Stereo 集成层调用。

## 接下来要实现的功能

## 2026-06-16 追加进度

### 已完成：P0 深度图对比工作流第一步

- `scripts/generate_depth_map.py` 支持重复 `--provider`，当前可选 `distill_base_518` 和 `luma`。
- `scripts/generate_depth_map.py` 支持重复 `--reference-depth`，也支持 `name=path` 命名参考深度。
- `depth_report.json` 新增机器可读 `summary`，包含 depth histogram、edge density、foreground/background separation 等指标。
- `depth_report.json` 仍保留旧版 `depth_provider`、`outputs.depth`、`outputs.depth_color`、`outputs.depth_shape` 字段，避免旧脚本读取失败。
- 新增 `scripts/batch_generate_depth_maps.py`，支持 `--rgb` 多图或 `--rgb-dir` 批量生成每张图独立 depth 报告。
- 新增 `scripts/run_visible_batch_generate_depth.bat`，用于可见窗口批量生成 Distill depth。

验证：

```powershell
.\python3\python.exe -B -c "import ast, pathlib; files=list(pathlib.Path('src').rglob('*.py'))+list(pathlib.Path('scripts').rglob('*.py'))+list(pathlib.Path('tests').rglob('*.py')); [ast.parse(p.read_text(encoding='utf-8'), filename=str(p)) for p in files]; print('syntax ok', len(files), 'files')"
.\python3\python.exe -B -m pytest -q
.\python3\python.exe -B scripts\generate_depth_map.py --rgb outputs\demo\fast_half_sbs.png --provider luma --out-dir outputs\depth_luma_smoke --device cpu
.\python3\python.exe -B scripts\batch_generate_depth_maps.py --rgb outputs\demo\fast_half_sbs.png --provider luma --out-dir outputs\depth_batch_luma_smoke --device cpu
```

### 已完成：NVIDIA depth backend fallback 骨架

- 新增 `src/stereo_lab/depth_onnx_provider.py`。
- 新增 `distill_base_nvidia` provider，backend 优先级为：

```text
onnx_cuda -> pytorch_cuda
```

- `scripts/generate_depth_map.py`、`scripts/compare_methods.py`、`scripts/batch_generate_depth_maps.py` 已支持 `--provider distill_base_nvidia`。
- 可见窗口入口 `run_visible_generate_depth.bat` 和 `run_visible_batch_generate_depth.bat` 默认改为 NVIDIA fallback provider。
- `depth_report.json` 的 `depth_provider` 会记录：
  - `depth_backend`
  - `runtime`
  - `onnx_path`
  - `execution_provider`
  - `fallback_reason`

当前验证：

```powershell
.\python3\python.exe -B scripts\generate_depth_map.py --rgb outputs\demo\fast_half_sbs.png --provider distill_base_nvidia --out-dir outputs\depth_nvidia_provider_smoke --device cuda --depth-local-only
```

当前环境结果：

```json
{
  "depth_backend": "onnx_cuda",
  "runtime": "onnxruntime",
  "execution_provider": "CUDAExecutionProvider",
  "fallback_reason": null
}
```

说明：已通过 `onnxruntime-gpu[cuda,cudnn]` 补齐 ONNX Runtime GPU 依赖。pip 当前解析为 `onnxruntime-gpu 1.26.0` + CUDA 12.9/cuDNN 9 runtime，和项目现有 PyTorch CUDA 12.x 环境匹配。`distill_base_nvidia` 已能真正使用 `CUDAExecutionProvider`。

ONNX 与 PyTorch 对齐验证：

```powershell
.\python3\python.exe -B scripts\test_distill_base_onnx.py --rgb outputs\demo\fast_half_sbs.png --device cuda --out-dir outputs\onnx_distill_smoke
```

结果：

```json
{
  "providers_used": ["CUDAExecutionProvider", "CPUExecutionProvider"],
  "metrics": {
    "mae": 0.0013555270852521062,
    "mse": 0.000003387340484550805,
    "psnr": 54.701409339904785
  }
}
```

### 已完成：独立 CUDA 13 nightly 实验环境

目的：不污染稳定 `python3` 主环境，单独验证最新 NVIDIA 栈。

独立环境：

```text
python-cu13/
```

已安装并验证：

| 组件 | 版本 |
|---|---|
| Python | 3.14.6 |
| PyTorch | 2.14.0.dev20260615+cu130 |
| TorchVision | 0.29.0.dev20260615+cu130 |
| TorchAudio | 2.11.0.dev20260615+cu130 |
| ONNX Runtime GPU | 1.27.0 |
| Transformers | 5.12.1 |
| ONNX | 1.22.0 |

RTX 2060 实测：

```text
torch cuda: 13.0
cuda available: True
device: NVIDIA GeForce RTX 2060
```

ONNX CUDA 对齐测试：

```powershell
.\python-cu13\python.exe -B scripts\test_distill_base_onnx.py --rgb outputs\demo\fast_half_sbs.png --device cuda --out-dir outputs\onnx_distill_cu13_smoke
```

结果：

```json
{
  "providers_used": ["CUDAExecutionProvider", "CPUExecutionProvider"],
  "metrics": {
    "mae": 0.0010626752628013492,
    "mse": 0.0000019673220776894595,
    "psnr": 57.06124305725098
  }
}
```

复现脚本：

```powershell
.\scripts\setup_cuda13_nightly_env.ps1
.\scripts\run_cuda13_onnx_smoke.ps1
```

## 当前架构决策

### 1. 运行环境分层

当前保留两条 NVIDIA 环境线：

| 环境 | 目录 | 角色 | 状态 |
|---|---|---|---|
| 稳定主线 | `python3/` | CUDA 12.x，日常开发与稳定验证 | 已跑通 PyTorch CUDA、ONNX CUDA |
| 最新实验线 | `python-cu13/` | Python 3.14.6 + PyTorch nightly cu130 + ORT 1.27 | 已跑通 RTX 2060 + ONNX CUDA |

`python3/` 和 `python-cu13/` 都不上传 GitHub。`models/`、`outputs/`、`downloads/` 也不上传。

### 2. 模型与部署分层

建议最终发布拆成三类包：

| 包 | 是否带 PyTorch | 用途 |
|---|---|---|
| Runtime 包 | 否 | 只运行已准备好的 ONNX / TensorRT |
| Model Prep 包 | 是 | 用户首次下载模型、导出 ONNX、构建 TensorRT |
| Dev 包 | 是 | 开发、debug、benchmark、对齐测试 |

部署主路径建议：

```text
TensorRT FP16 -> ONNX CUDA -> PyTorch CUDA fallback
```

正式 runtime 可以不带 PyTorch；但如果要求用户本地下载模型并导出 ONNX/TensorRT，则需要单独提供 Model Prep 环境或脚本。

### 3. ONNX 与 TensorRT 的关系

- ONNX 不是替代 TensorRT，而是模型交换格式和 TensorRT fallback。
- ONNX Runtime 普通 `session.run()` 默认输出 CPU numpy；实时路径应补 `IOBinding`，避免 GPU->CPU 拷贝。
- TensorRT 更适合作为最终全 GPU pipeline 主路径。
- TensorRT build fail 不代表 ONNX CUDA 必然 fail；两者失败原因不同，因此 ONNX CUDA fallback 仍有价值。

## 剩余工作

### P0：深度图对比工作流完善

目标：先把“哪个 depth 更适合 3D”判断清楚。

已完成：

- 增加批量 depth 生成脚本。
- 支持同一 RGB 下多个 depth provider 的 contact sheet。
- 增加 depth histogram、edge consistency、foreground separation 等指标。
- 给 `depth_report.json` 增加机器可读 summary 字段。

待做：

- 支持把 iw3 / Desktop2Stereo 生成的 depth 或参考 depth 放进同一报告并形成固定评测集。
- 增加跨环境 depth 输出对比：`python3` 稳定线 vs `python-cu13` latest 线。
- 增加真实图片批量评估样例，不只用 demo 输出图。

### P0：TensorRT 路线

目标：给 `Distill-Any-Depth-Base @ 294x518 fp16` 建立独立 TensorRT 路径。

待做：

- 新增 `scripts/build_distill_base_trt.py`。
- 新增 `scripts/run_visible_build_distill_base_trt.bat`。
- 输出到：

```text
models/models--lc700x--Distill-Any-Depth-Base-hf/model_fp16_294x518.trt
```

- 新增 `src/stereo_lab/depth_trt_provider.py`。
- 实现优先 TensorRT、失败回退 ONNX CUDA、再回退 PyTorch CUDA 的 depth provider。
- 报告中记录 `depth_backend = tensorrt | onnx_cuda | pytorch_cuda`。
- 对比 `PyTorch CUDA / ONNX CUDA session.run / ONNX CUDA IOBinding / TensorRT FP16` 的单帧耗时、显存和端到端 FPS。

### P0：ONNX IOBinding

目标：验证 ONNX CUDA fallback 是否能避免 CPU copy。

待做：

- 在 `src/stereo_lab/depth_onnx_provider.py` 增加 IOBinding 路径。
- 报告中记录 `output_device = cuda | cpu`。
- benchmark 普通 `session.run()` 与 IOBinding 的耗时差异。
- 如果后续 stereo synthesis 仍是 PyTorch tensor path，研究 OrtValue/DLPack 或可接受的 GPU tensor 转换方式。

### P1：Quality 4K 算法增强

当前 `quality_4k` 是 2-layer 原型，还不是最终超越 iw3 的完整路线。

待做：

- 更严格的 occlusion-aware mask。
- 更稳定的 foreground/background layer 分离。
- hole fill 限制在遮挡和高梯度区域。
- 避免 UI 文字边缘出现糊边。
- 增加 debug 输出：layer weights、shift map、occlusion map。

### P1：HQ 4K 算法增强

当前 `hq_4k` 已经支持至少 3-layer，但时序和局部修复还比较轻。

待做：

- temporal cache 记录上一帧 depth/mask/warp。
- temporal strength 可调。
- 快速横向运动场景避免拖影。
- 预留局部 refinement 接口，但不要做全帧大模型 refinement。

### P1：真实 iw3 对比适配

当前仓库还没有真正调用 iw3 的 `row_flow_v3_sym` 生成同场景参考。

待做：

- 明确 iw3 输入输出格式。
- 生成同一 RGB/depth 下 iw3 或官方结果。
- 把 iw3 输出纳入 contact sheet 和报告。
- 注意不要把“不同 depth 模型差异”误判为“左右眼生成算法差异”。

### P2：Desktop2Stereo 集成评估

等实验链路稳定后再接回主项目。

待做：

- 设计最小集成 wrapper。
- 只把稳定 API 接到 Desktop2Stereo viewer。
- 不直接把实验脚本塞进主项目。
- 保留 `fast` 降级路径。

## 当前风险与注意事项

### 1. 不要污染 Desktop2Stereo 模型目录

本项目默认模型缓存和 ONNX 输出都应该在：

```text
4k-stereo-synthesis-lab/models/
```

不要默认写入：

```text
Desktop2Stereo_v2.4.2_Windows_NVIDIA/Desktop2Stereo/models/
```

### 2. 不要上传大模型文件

`.gitignore` 已包含：

```text
python3/
python-cu13/
models/
outputs/
downloads/
```

后续新增模型、ONNX、TRT、输出图，都应放在这些忽略目录内。

### 2.1 不要把运行库当作源码上传

当前 `python3/` 已超过 10GB，主要原因是 PyTorch CUDA、ONNX Runtime CUDA、TensorRT runtime 各自携带 GPU DLL。`python-cu13/` 也会很大。它们都只作为本机运行环境，不应进 Git。

`requirements*.txt` 记录环境依赖，运行库本体不上传。

### 3. 低配电脑首次导入很慢

用户明确要求：

- 长时间任务最好用可见 bat。
- 第一次 `torch` / CUDA / transformers 导入可能很慢。
- 不要在无反馈的后台长时间运行。

### 4. 现在的 `luma` depth 只能作为调试 fallback

`src/stereo_lab/auto_depth.py` 是伪深度，仅用于 UI / 流程调试。

正式评估时应使用：

```text
Distill-Any-Depth-Base @ 518
```

### 5. 当前还不能声称已经超过 iw3

已经实现的是实验链路和原型算法，不是最终画质结论。

必须等以下内容完成后再做结论：

- 固定同一 RGB/depth 输入。
- 生成 iw3 或官方参考输出。
- 对比 depth、左右眼、SBS、边缘、遮挡、时序。
- 4K 真实性能测试。

### 6. CUDA 13 nightly 只作为实验线

`python-cu13/` 已在 RTX 2060 上跑通 PyTorch nightly cu130 和 ONNX Runtime GPU 1.27，但它不是当前稳定主线。后续如果要把 CUDA 13 作为主线，必须重测：

- RTX 2060 最低档。
- RTX 3090 / RTX 5070 目标档。
- ONNX 导出一致性。
- TensorRT build / engine load。
- 端到端 depth + stereo synthesis 性能。

## 推荐下一步执行顺序

1. 实现 `depth_trt_provider.py` 和 `build_distill_base_trt.py`，形成 `TensorRT -> ONNX CUDA -> PyTorch CUDA`。
2. 给 ONNX provider 增加 IOBinding，验证 GPU-resident output。
3. 做 depth backend benchmark：PyTorch CUDA、ONNX CUDA、ONNX IOBinding、TensorRT FP16。
4. 把 depth provider API 稳定成 `estimate_depth(rgb, config)`。
5. 把 one-shot API 稳定成 `rgb_to_stereo(rgb, depth_config, stereo_config)`。
6. 用真实图片批量评估 Distill depth 是否符合 3D 直觉。
7. 增强 Quality 4K 的 occlusion/layer/hole fill。
8. 做 iw3 同场景对比。
9. 再评估是否接回 `Desktop2Stereo`。
