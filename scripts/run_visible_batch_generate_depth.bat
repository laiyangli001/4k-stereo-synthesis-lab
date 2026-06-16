@echo off
chcp 65001>nul
setlocal

set "LAB_DIR=%~dp0.."
set "PYTHON_EXE=%LAB_DIR%\python3\python.exe"

if not exist "%PYTHON_EXE%" (
  echo [Error] Python not found: %PYTHON_EXE%
  echo [Hint] Copy Desktop2Stereo\python3 into this lab as python3 first.
  pause
  exit /b 1
)

if "%~1"=="" (
  echo [Usage] Drag a folder of RGB images onto this .bat, or run:
  echo         %~nx0 image_folder
  pause
  exit /b 1
)

set "RGB_DIR=%~1"
set "OUT_DIR=%LAB_DIR%\outputs\depth_batch"

title 4K Stereo Lab - Batch Distill Depth Maps
echo [Info] RGB folder: %RGB_DIR%
echo [Info] Output: %OUT_DIR%
echo [Info] Depth model: Distill-Any-Depth-Base @ 518
echo [Info] Backend priority: ONNX CUDA -^> PyTorch CUDA
echo [Info] First torch/CUDA import or model download may take several minutes.
echo.

pushd "%LAB_DIR%"
"%PYTHON_EXE%" "%LAB_DIR%\scripts\batch_generate_depth_maps.py" --rgb-dir "%RGB_DIR%" --out-dir "%OUT_DIR%" --provider distill_base_nvidia --device cuda
set "EXIT_CODE=%ERRORLEVEL%"
if "%EXIT_CODE%"=="0" (
  echo.
  echo [Info] Opening outputs\depth_batch ...
  explorer "%OUT_DIR%"
)
popd

echo.
echo [Info] Exit code: %EXIT_CODE%
pause
exit /b %EXIT_CODE%
