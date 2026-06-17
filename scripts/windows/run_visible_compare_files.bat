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
  echo [Usage] Drag RGB image and depth image onto this .bat, or run:
  echo         %~nx0 rgb.png depth.png
  pause
  exit /b 1
)

if "%~2"=="" (
  echo [Usage] Missing depth image.
  echo         %~nx0 rgb.png depth.png
  pause
  exit /b 1
)

set "RGB=%~1"
set "DEPTH=%~2"
set "OUT_DIR=%LAB_DIR%\outputs\compare_real"

title 4K Stereo Lab - Compare Files
echo [Info] RGB:   %RGB%
echo [Info] Depth: %DEPTH%
echo [Info] Output: %OUT_DIR%
echo [Info] First torch/CUDA import may take several minutes on low-end machines.
echo.

pushd "%LAB_DIR%"
"%PYTHON_EXE%" "%LAB_DIR%\scripts\tools\compare_methods.py" --rgb "%RGB%" --depth "%DEPTH%" --out-dir "%OUT_DIR%" --output-format half_sbs --device cuda
set "EXIT_CODE=%ERRORLEVEL%"
if "%EXIT_CODE%"=="0" (
  echo.
  echo [Info] Opening outputs\compare_real ...
  explorer "%OUT_DIR%"
)
popd

echo.
echo [Info] Exit code: %EXIT_CODE%
pause
exit /b %EXIT_CODE%
