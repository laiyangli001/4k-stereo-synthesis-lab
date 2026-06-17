@echo off
chcp 65001>nul
setlocal

set "LAB_DIR=%~dp0..\.."
set "PYTHON_EXE=%LAB_DIR%\python3\python.exe"

if not exist "%PYTHON_EXE%" (
  echo [Error] Python not found: %PYTHON_EXE%
  pause
  exit /b 1
)

if "%~1"=="" (
  echo [Usage] Drag one RGB image onto this .bat, or run:
  echo         %~nx0 rgb.png
  pause
  exit /b 1
)

set "RGB=%~1"
set "OUT_DIR=%LAB_DIR%\outputs\env_depth_backend_compare"

title 4K Stereo Lab - Depth Backend Env Compare
echo [Info] RGB: %RGB%
echo [Info] Output: %OUT_DIR%
echo [Info] Environments: python3 and python-cu13
echo [Info] Backends: TensorRT, ONNX CUDA IOBinding, PyTorch CUDA
echo [Info] TensorRT first build may take several minutes.
echo.

pushd "%LAB_DIR%"
"%PYTHON_EXE%" "%LAB_DIR%\scripts\compare_python_env_depth_backends.py" --rgb "%RGB%" --out-dir "%OUT_DIR%" --warmup 1 --iters 3
set "EXIT_CODE=%ERRORLEVEL%"
if "%EXIT_CODE%"=="0" (
  echo.
  echo [Info] Opening benchmark report folder ...
  explorer "%OUT_DIR%"
)
popd

echo.
echo [Info] Exit code: %EXIT_CODE%
pause
exit /b %EXIT_CODE%
