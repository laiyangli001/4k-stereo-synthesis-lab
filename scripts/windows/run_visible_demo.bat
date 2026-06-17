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

title 4K Stereo Lab - Visible Demo
echo [Info] Using local Python:
echo        %PYTHON_EXE%
echo [Info] This demo writes PNG files to outputs\demo.
echo [Info] First torch/CUDA import may take several minutes on low-end machines.
echo.

pushd "%LAB_DIR%"
"%PYTHON_EXE%" "%LAB_DIR%\scripts\examples\demo_generate.py"
set "EXIT_CODE=%ERRORLEVEL%"
if "%EXIT_CODE%"=="0" (
  echo.
  echo [Info] Opening outputs\demo ...
  explorer "%LAB_DIR%\outputs\demo"
)
popd

echo.
echo [Info] Exit code: %EXIT_CODE%
pause
exit /b %EXIT_CODE%
