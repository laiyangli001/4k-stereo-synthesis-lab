# Depth Backend Benchmark Report

Date: 2026-06-16  
Primary test image: `4K.jpg` (`3840x2160`)  
Latest 4K report: `outputs/env_depth_backend_compare_4k/env_depth_backend_compare.md`  
720p reference report: `outputs/env_depth_backend_compare_reuse_session/env_depth_backend_compare.md`  
GPU: RTX 2060

## Summary

The previous benchmark result was misleading.

The old `309s` number was TensorRT engine build time, not inference time.  
The old `3.4s` number was repeated ORT TensorRT session creation / engine load time, not inference time.

The benchmark has been fixed to separate:

```text
setup_ms     = provider/session creation + TensorRT engine build/load
warmup_ms    = first calls using the same provider/session
inference_ms = repeated inference using the same provider/session
```

With provider/session reuse, TensorRT is now the fastest backend.

## TensorRT Runtime Fix

The earlier TensorRT failure was caused by missing DLL discovery:

```text
onnxruntime_providers_tensorrt.dll depends on nvinfer_10.dll
```

The DLL exists at:

```text
python3/Lib/site-packages/tensorrt_libs/
```

The project now auto-discovers and prepends this folder to `PATH` before creating the ORT TensorRT session.

Implementation:

```text
src/stereo_lab/depth_trt_provider.py
```

Key helper:

```python
ensure_tensorrt_dll_path()
```

## Native TensorRT Provider

Desktop2Stereo already uses a stronger native TensorRT path based on CUDA tensor pointers:

```python
context.set_tensor_address(input_name, tensor.data_ptr())
out = torch.empty(shape, device="cuda", dtype=dtype)
context.set_tensor_address(output_name, out.data_ptr())
context.execute_async_v3(stream_handle=torch.cuda.current_stream().cuda_stream)
```

This project now has its own native provider:

```text
src/stereo_lab/depth_trt_native_provider.py
```

Expected engine path:

```text
models/models--lc700x--Distill-Any-Depth-Base-hf/model_fp16_294x518.trt
```

Visible first-build helper:

```bat
scripts/windows/run_visible_build_native_tensorrt_engine.bat
```

Benchmark backend name:

```text
tensorrt_native
```

This is different from ONNX Runtime TensorRT EP. The native provider avoids the current ORT path's `torch.Tensor -> CPU numpy -> ORT` input conversion and `ORT output -> numpy -> torch.Tensor` output conversion.

## Backend Priority

`distill_base_nvidia` now targets:

```text
TensorRT EP -> ONNX CUDA IOBinding -> PyTorch CUDA
```

ONNX CUDA defaults to IOBinding.

## Environment Comparison

| Env | Torch | Torch CUDA | ONNX Runtime | TensorRT DLL source |
|---|---|---|---|---|
| `python3` | `2.7.1+cu128` | `12.8` | `1.26.0` | `python3/Lib/site-packages/tensorrt_libs` |
| `python-cu13` | `2.14.0.dev20260615+cu130` | `13.0` | `1.27.0` | reuses `python3/Lib/site-packages/tensorrt_libs` |

Both environments expose:

```text
TensorrtExecutionProvider
CUDAExecutionProvider
CPUExecutionProvider
```

## Corrected 720p Reference Results

Command:

```powershell
.\python3\python.exe -B scripts\benchmark\compare_python_env_depth_backends.py --rgb outputs\demo\fast_half_sbs.png --out-dir outputs\env_depth_backend_compare_reuse_session --warmup 2 --iters 5
```

Results:

| Env | Backend | Status | Setup ms | Warmup mean ms | Inference mean ms |
|---|---|---:|---:|---:|---:|
| `python3` | TensorRT EP | OK | `4363.944` | `110.387` | `14.417` |
| `python3` | ONNX CUDA IOBinding | OK | `1061.619` | `412.142` | `28.895` |
| `python3` | PyTorch CUDA | OK | `11765.374` | `123.420` | `44.999` |
| `python-cu13` | TensorRT EP | OK | `4221.489` | `68.487` | `14.888` |
| `python-cu13` | ONNX CUDA IOBinding | OK | `1076.401` | `357.801` | `29.321` |
| `python-cu13` | PyTorch CUDA | OK | `13439.888` | `170.471` | `45.095` |

## Real 4K Results

Command:

```powershell
.\python3\python.exe -B scripts\benchmark\compare_python_env_depth_backends.py --rgb 4K.jpg --out-dir outputs\env_depth_backend_compare_4k --warmup 2 --iters 10
```

Input:

```text
4K.jpg: 3840x2160 RGB
```

Results:

| Env | Backend | Status | Setup ms | Warmup mean ms | Inference mean ms | Inference median ms | Inference p90 ms |
|---|---|---:|---:|---:|---:|---:|---:|
| `python3` | TensorRT EP | OK | `4483.240` | `111.276` | `24.971` | `25.201` | `25.585` |
| `python3` | ONNX CUDA IOBinding | OK | `1068.342` | `394.309` | `37.483` | `37.967` | `40.058` |
| `python3` | PyTorch CUDA | OK | `10739.280` | `121.267` | `45.812` | `44.175` | `45.385` |
| `python-cu13` | TensorRT EP | OK | `4458.005` | `84.599` | `29.725` | `26.432` | `34.610` |
| `python-cu13` | ONNX CUDA IOBinding | OK | `1063.333` | `376.167` | `38.670` | `38.832` | `42.331` |
| `python-cu13` | PyTorch CUDA | OK | `9884.255` | `177.033` | `46.746` | `46.809` | `47.934` |

4K conclusion:

```text
python3 is the recommended main environment.
python-cu13 remains an experimental environment.
TensorRT EP is the fastest depth backend when provider/session is reused.
```

Environment delta on 4K:

| Backend | python3 mean | python-cu13 mean | Current judgment |
|---|---:|---:|---|
| TensorRT EP | `24.971 ms` | `29.725 ms` | `python3` faster; `python-cu13` had timing spikes |
| ONNX CUDA IOBinding | `37.483 ms` | `38.670 ms` | similar, `python3` slightly faster |
| PyTorch CUDA | `45.812 ms` | `46.746 ms` | similar, `python3` slightly faster |

## 4K Runtime Breakdown

Command:

```powershell
.\python3\python.exe -B scripts\benchmark\bench_depth_backends.py --rgb 4K.jpg --out-dir outputs\depth_backend_bench_profile_4k --warmup 1 --iters 3 --backend tensorrt --backend onnx_cuda_iobinding --backend pytorch_cuda
```

Results:

| Backend | Inference mean ms | Preprocess mean ms | Model mean ms | Postprocess mean ms |
|---|---:|---:|---:|---:|
| TensorRT EP | `25.130` | `6.382` | `12.577` | `6.061` |
| ONNX CUDA IOBinding | `40.418` | `6.851` | `25.981` | `7.479` |
| PyTorch CUDA | `42.999` | `8.219` | `33.622` | `1.064` |

Interpretation:

- TensorRT model execution is the largest single stage, but preprocess + postprocess is also nearly half of the total.
- For ORT backends, current preprocess includes `torch.Tensor -> CPU numpy` input conversion.
- For ORT backends, current model timing includes ORT run plus output retrieval.
- This confirms that lower-copy / zero-copy ORT input and output paths are worth investigating after the API is stable.

## Native TensorRT 4K Results

Command:

```powershell
.\python3\python.exe -B scripts\benchmark\bench_depth_backends.py --rgb 4K.jpg --out-dir outputs\depth_backend_bench_native_4k --warmup 1 --iters 5 --backend tensorrt_native --backend tensorrt --backend onnx_cuda_iobinding --backend pytorch_cuda
```

Input:

```text
4K.jpg: 3840x2160 RGB
```

Results:

| Backend | Setup ms | Inference mean ms | Preprocess mean ms | Model mean ms | Postprocess mean ms |
|---|---:|---:|---:|---:|---:|
| Native TensorRT | `762.838` | `19.697` | `7.134` | `11.436` | `1.064` |
| ONNX Runtime TensorRT EP | `4146.017` | `25.813` | `7.064` | `12.588` | `6.053` |
| ONNX CUDA IOBinding | `1068.964` | `39.217` | `6.578` | `25.861` | `5.780` |
| PyTorch CUDA | `11040.544` | `41.939` | `6.808` | `33.927` | `1.115` |

Native TensorRT conclusion:

```text
Native TensorRT is currently the fastest depth backend on RTX 2060.
It reduces 4K depth inference from ~25.8 ms with ORT TensorRT EP to ~19.7 ms.
The biggest improvement is postprocess/output handling: ~6.1 ms -> ~1.1 ms.
```

## ONNX CUDA DLPack 4K Results

Command:

```powershell
.\python3\python.exe -B scripts\benchmark\bench_depth_backends.py --rgb 4K.jpg --out-dir outputs\depth_backend_bench_dlpack_4k --warmup 1 --iters 5 --backend onnx_cuda_dlpack --backend onnx_cuda_iobinding --backend tensorrt_native
```

Results:

| Backend | Setup ms | Inference mean ms | Preprocess mean ms | Model mean ms | Postprocess mean ms |
|---|---:|---:|---:|---:|---:|
| ONNX CUDA DLPack | `1329.381` | `32.215` | `7.284` | `23.644` | `1.197` |
| ONNX CUDA IOBinding | `1000.853` | `39.379` | `6.479` | `25.869` | `5.999` |
| Native TensorRT | `733.199` | `15.287` | `5.218` | `9.034` | `0.970` |

ONNX DLPack conclusion:

```text
ONNX CUDA DLPack is a real lower-copy improvement over numpy-based IOBinding.
It reduces 4K depth inference from ~39.4 ms to ~32.2 ms.
The largest gain is output/postprocess: ~6.0 ms -> ~1.2 ms.
Native TensorRT remains much faster at ~15.3 ms in this run.
```

Zero-copy status by backend:

| Backend | Current copy behavior | Judgment |
|---|---|---|
| PyTorch CUDA | RGB tensor, model, depth tensor stay on CUDA after initial input conversion | Already GPU-resident for tensor inputs |
| ONNX CUDA IOBinding | Input/output pass through CPU numpy wrappers | Usable fallback, not zero-copy |
| ONNX CUDA DLPack | Torch CUDA tensor <-> ORT CUDA OrtValue through DLPack | Lower-copy fallback |
| Native TensorRT | TensorRT reads/writes PyTorch CUDA tensor memory via `data_ptr()` | Best current path |

## Native TensorRT Preallocated Output Buffer

Native TensorRT now reuses output CUDA tensors instead of allocating `torch.empty(...)` every frame.

Correctness check:

```powershell
.\python3\python.exe -B scripts\tools\check_native_tensorrt_consistency.py --rgb 4K.jpg --out outputs\native_tensorrt_consistency.json
```

Result:

| Metric | Value |
|---|---:|
| `depth_absdiff_mean` | `0.0` |
| `depth_absdiff_max` | `0.0` |

Benchmark:

```powershell
.\python3\python.exe -B scripts\benchmark\bench_depth_backends.py --rgb 4K.jpg --out-dir outputs\depth_backend_bench_native_prealloc_4k --warmup 2 --iters 10 --backend tensorrt_native
```

Result:

| Backend | Inference mean ms | Inference median ms | Inference p90 ms | Preprocess mean ms | Model mean ms | Postprocess mean ms |
|---|---:|---:|---:|---:|---:|---:|
| Native TensorRT prealloc | `17.775` | `17.933` | `18.851` | `6.089` | `10.544` | `1.082` |

Note:

```text
The preallocation change is numerically identical in this check.
Measured FPS still varies by GPU state / warmup, so use multi-run median for final claims.
```

## Native TensorRT CUDA Graph Probe

Command:

```powershell
.\python3\python.exe -B scripts\benchmark\bench_depth_backends.py --rgb 4K.jpg --out-dir outputs\depth_backend_bench_native_graph_4k --warmup 2 --iters 10 --backend tensorrt_native_graph --backend tensorrt_native
```

Results:

| Backend | Inference mean ms | Inference median ms | Inference p90 ms | Preprocess mean ms | Model mean ms | Postprocess mean ms |
|---|---:|---:|---:|---:|---:|---:|
| Native TensorRT CUDA Graph | `19.156` | `18.617` | `22.402` | `7.225` | `10.619` | `1.224` |
| Native TensorRT | `16.143` | `16.257` | `16.670` | `5.630` | `9.158` | `1.278` |

CUDA Graph conclusion:

```text
CUDA Graph capture runs, but the current implementation is not faster.
It is kept as an experimental backend, not the default path.
The likely reason is that current graph replay only covers TensorRT execute,
while the per-frame input copy, preprocess, and postprocess remain outside the graph.
```

Current rule:

```text
Do not change model resolution or model choice for this optimization.
CUDA Graph should only be revisited after fixed input/output buffers and possibly fused preprocess are available.
```

## Safe Preprocess Cache

The preprocess cache keeps the model input resolution and preprocessing semantics unchanged:

```text
4K RGB -> bicubic+antialias resize -> 294x518 -> ImageNet mean/std normalize -> fp16
```

Implemented cache:

- Reuse ImageNet mean/std CUDA tensors.
- Cache computed model input shape per source frame shape.
- Keep resize mode, antialias, normalize formula, model, and model resolution unchanged.

Correctness:

```powershell
.\python3\python.exe -B scripts\tools\check_native_tensorrt_consistency.py --rgb 4K.jpg --out outputs\native_tensorrt_consistency_preprocess_cache.json
```

| Metric | Value |
|---|---:|
| `depth_absdiff_mean` | `0.0` |
| `depth_absdiff_max` | `0.0` |

Benchmark:

```powershell
.\python3\python.exe -B scripts\benchmark\bench_depth_backends.py --rgb 4K.jpg --out-dir outputs\depth_backend_bench_preprocess_cache_4k --warmup 2 --iters 10 --backend tensorrt_native
```

| Backend | Inference mean ms | Inference median ms | Inference p90 ms | Preprocess mean ms | Model mean ms | Postprocess mean ms |
|---|---:|---:|---:|---:|---:|---:|
| Native TensorRT + preprocess cache | `17.586` | `17.904` | `18.430` | `5.679` | `10.644` | `1.186` |

Judgment:

```text
This is a safe optimization: same model resolution, same preprocessing semantics, identical depth output in the consistency check.
The speed gain is modest because the expensive part of preprocess is still bicubic+antialias resize.
```

## Final 4K Depth Backend Comparison

Command:

```powershell
.\python3\python.exe -B scripts\benchmark\bench_depth_backends.py --rgb 4K.jpg --out-dir outputs\depth_backend_final_compare_4k --warmup 3 --iters 20 --backend tensorrt_native --backend onnx_cuda_dlpack --backend onnx_cuda_iobinding --backend pytorch_cuda
```

Input:

```text
4K.jpg: 3840x2160 RGB
```

Results:

| Backend | Setup ms | Inference mean ms | Inference median ms | Inference p90 ms | Preprocess mean ms | Model mean ms | Postprocess mean ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| Native TensorRT | `856.475` | `16.425` | `15.641` | `18.482` | `5.412` | `9.835` | `1.104` |
| ONNX CUDA DLPack | `2955.727` | `27.706` | `26.287` | `32.192` | `5.270` | `21.334` | `1.020` |
| ONNX CUDA IOBinding | `2617.692` | `35.905` | `35.308` | `41.097` | `5.767` | `22.584` | `6.508` |
| PyTorch CUDA | `11588.347` | `42.324` | `41.060` | `48.238` | `5.409` | `35.832` | `0.993` |

Final judgment:

```text
Native TensorRT is the recommended primary depth backend.
ONNX CUDA DLPack is the recommended ONNX fallback.
ONNX CUDA IOBinding remains useful but is slower due to numpy-style output handling.
PyTorch CUDA remains the correctness fallback.
```

Quality constraint:

```text
All optimizations in this comparison keep the same model, same 294x518 inference resolution,
same resize mode, same antialias setting, and same normalize semantics.
```

## Distill-Any-Depth-Large 4K Limit Test

Model:

```text
Distill-Any-Depth-Large
model_id: xingyang1/Distill-Any-Depth-Large-hf
input: 1x3x294x518
dtype: fp16
```

Artifacts:

```text
models/models--xingyang1--Distill-Any-Depth-Large-hf/model_fp16_294x518.onnx
models/models--xingyang1--Distill-Any-Depth-Large-hf/model_fp16_294x518.trt
```

Command:

```powershell
.\python3\python.exe -B scripts\benchmark\bench_depth_backends.py --rgb 4K.jpg --out-dir outputs\depth_backend_large_4k --warmup 3 --iters 20 --backend tensorrt_native --backend onnx_cuda_dlpack --model-id xingyang1/Distill-Any-Depth-Large-hf --model-name Distill-Any-Depth-Large --onnx models\models--xingyang1--Distill-Any-Depth-Large-hf\model_fp16_294x518.onnx --trt-engine models\models--xingyang1--Distill-Any-Depth-Large-hf\model_fp16_294x518.trt
```

Results:

| Backend | Setup ms | Mean ms | Median ms | P90 ms | Mean FPS | Median FPS | Preprocess ms | Model ms | Postprocess ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Large Native TensorRT | `1843.979` | `33.616` | `33.717` | `34.370` | `29.75` | `29.66` | `5.458` | `26.929` | `1.155` |
| Large ONNX CUDA DLPack | `2560.461` | `62.590` | `60.506` | `64.332` | `15.98` | `16.53` | `5.810` | `54.729` | `1.955` |

Large judgment on RTX 2060:

```text
Distill-Any-Depth-Large @ 518 is usable for offline / quality evaluation.
On RTX 2060, Large native TensorRT is about 29-30 FPS for depth alone.
It is not suitable for 4K 60 FPS end-to-end SBS on this GPU.
High-end GPUs should be benchmarked separately.
```

## Interpretation

Provider/session reuse is mandatory.

Correct runtime pattern:

```python
provider = DistillAnyDepthBaseTensorRtOrt(...)
provider.load()  # once at startup

for frame in frames:
    depth = provider.predict(frame)
```

Current realtime API shape:

```python
from stereo_lab.depth_provider import DepthProviderConfig, create_depth_provider

provider = create_depth_provider(
    DepthProviderConfig(backend="tensorrt", device="cuda")
)
provider.load()

for rgb in frames:
    depth = provider.predict(rgb)
```

Incorrect runtime pattern:

```python
for frame in frames:
    provider = DistillAnyDepthBaseTensorRtOrt(...)
    depth = provider.predict(frame)
```

The previous benchmark accidentally followed the second pattern by constructing a new provider every iteration.

## Current Stability Judgment

| Backend | Stability | Speed on RTX 2060 | Notes |
|---|---|---:|---|
| TensorRT EP | Usable | ~`25 ms` on 4K input | Fastest after setup; setup/build must happen before realtime loop |
| ONNX CUDA IOBinding | Usable | ~`37-39 ms` on 4K input | Good fallback when TensorRT fails |
| PyTorch CUDA | Most stable | ~`46 ms` on 4K input | Correctness baseline and final fallback |

## Remaining Caveats

- Current timing still includes preprocessing and postprocessing.
- Input path still converts PyTorch tensor to numpy before ORT. This should be optimized later.
- TensorRT cache produced separate engine files for `python3` and `python-cu13`, likely due to ORT version differences:

```text
onnxruntime 1.26.0 -> one engine key
onnxruntime 1.27.0 -> another engine key
```

## Next Steps

1. Split benchmark into `preprocess_ms`, `model_ms`, and `postprocess_ms`.
2. Keep TensorRT provider/session alive in any realtime API.
3. Investigate zero-copy or lower-copy ORT input/output paths.
4. Run the same 4K benchmark on higher-end GPUs such as RTX 3090 / RTX 5070.
5. Decide whether ORT TensorRT EP is enough, or whether native TensorRT engine API is needed later.
