@echo off
setlocal
cd /d "%~dp0\.."
echo Building Distill-Any-Depth-Large native TensorRT engine. First build may take a long time.
echo.
python3\python.exe -B scripts\tools\build_native_tensorrt_engine.py ^
  --onnx models\models--xingyang1--Distill-Any-Depth-Large-hf\model_fp16_294x518.onnx ^
  --engine models\models--xingyang1--Distill-Any-Depth-Large-hf\model_fp16_294x518.trt ^
  %*
echo.
pause
