@echo off
chcp 65001>nul
setlocal

set "LAB_DIR=%~dp0..\.."
set "PYTHON_EXE=%LAB_DIR%\python3\python.exe"

if not exist "%PYTHON_EXE%" (
  echo [Error] Python not found: %PYTHON_EXE%
  echo [Hint] Copy Desktop2Stereo\python3 into this lab as python3 first.
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
set "OUT_DIR=%LAB_DIR%\outputs\compare_auto_depth"

title 4K Stereo Lab - Distill Auto Depth Compare
echo [Info] RGB: %RGB%
echo [Info] Output: %OUT_DIR%
echo [Info] Auto depth model: Distill-Any-Depth-Base @ 518
echo [Info] Model ID: lc700x/Distill-Any-Depth-Base-hf
echo [Info] Load mode: network-enabled Hugging Face download/cache
echo [Info] First torch/CUDA import may take several minutes on low-end machines.
echo.

pushd "%LAB_DIR%"
"%PYTHON_EXE%" "%LAB_DIR%\scripts\compare_methods.py" --rgb "%RGB%" --auto-depth --depth-provider distill_base_518 --out-dir "%OUT_DIR%" --output-format half_sbs --device cuda
set "EXIT_CODE=%ERRORLEVEL%"
if "%EXIT_CODE%"=="0" (
  echo.
  echo [Info] Opening outputs\compare_auto_depth ...
  explorer "%OUT_DIR%"
)
popd

echo.
echo [Info] Exit code: %EXIT_CODE%
pause
exit /b %EXIT_CODE%
