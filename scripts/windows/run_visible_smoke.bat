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

title 4K Stereo Lab - Smoke Test
echo [Info] Using Desktop2Stereo Python:
echo        %PYTHON_EXE%
echo [Info] First torch/CUDA import may take several minutes on low-end machines.
echo.

pushd "%LAB_DIR%"
"%PYTHON_EXE%" "%LAB_DIR%\scripts\smoke\smoke_test.py"
set "EXIT_CODE=%ERRORLEVEL%"
popd

echo.
echo [Info] Exit code: %EXIT_CODE%
pause
exit /b %EXIT_CODE%
