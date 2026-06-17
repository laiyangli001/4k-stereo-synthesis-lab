@echo off
chcp 65001 >nul
title Desktop2Stereo - 安装 Flet GUI 依赖

set PYTHON=%~dp0python3\python.exe

if not exist "%PYTHON%" (
    echo [错误] 未找到 Python: %PYTHON%
    pause
    exit /b 1
)

echo 使用 Python: %PYTHON%
%PYTHON% --version
echo.

:: 检查是否已安装
%PYTHON% -c "import flet; print('flet 已安装')" 2>nul && goto :done

echo 正在安装 flet（GUI 框架）...
%PYTHON% -m pip install flet -i https://mirrors.aliyun.com/pypi/simple/
if errorlevel 1 (
    %PYTHON% -m pip install flet
)
if errorlevel 1 (
    echo 安装失败，请检查网络连接
    pause
    exit /b 1
)

:done
echo.
echo 验证:
%PYTHON% -c "import flet; print('flet', flet.__version__)"
echo.
echo 运行: cd Desktop2Stereo ^&^& python3\python.exe gui.py
pause
