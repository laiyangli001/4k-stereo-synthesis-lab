@echo off
chcp 65001>nul
setlocal

set "LAB_DIR=%~dp0..\.."
set "PYTHON_EXE=%LAB_DIR%\python3\python.exe"
set "RGB=%LAB_DIR%\outputs\demo\input_rgb.png"
set "DEPTH=%LAB_DIR%\outputs\demo\input_depth.png"

if not exist "%PYTHON_EXE%" (
  echo [Error] Python not found: %PYTHON_EXE%
  echo [Hint] Copy Desktop2Stereo\python3 into this lab as python3 first.
  pause
  exit /b 1
)

if not exist "%RGB%" (
  echo [Info] Demo input not found. Generating demo input first ...
  pushd "%LAB_DIR%"
  "%PYTHON_EXE%" "%LAB_DIR%\scripts\examples\demo_generate.py"
  popd
)

title 4K Stereo Lab - Compare Demo
echo [Info] RGB:   %RGB%
echo [Info] Depth: %DEPTH%
echo [Info] First torch/CUDA import may take several minutes on low-end machines.
echo.

pushd "%LAB_DIR%"
"%PYTHON_EXE%" "%LAB_DIR%\scripts\tools\compare_methods.py" --rgb "%RGB%" --depth "%DEPTH%" --out-dir "%LAB_DIR%\outputs\compare_demo" --output-format half_sbs --device cuda
set "EXIT_CODE=%ERRORLEVEL%"
if "%EXIT_CODE%"=="0" (
  echo.
  echo [Info] Opening outputs\compare_demo ...
  explorer "%LAB_DIR%\outputs\compare_demo"
)
popd

echo.
echo [Info] Exit code: %EXIT_CODE%
pause
exit /b %EXIT_CODE%
