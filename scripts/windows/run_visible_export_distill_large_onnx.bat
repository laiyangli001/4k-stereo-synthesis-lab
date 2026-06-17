@echo off
setlocal
cd /d "%~dp0\.."
echo Exporting Distill-Any-Depth-Large ONNX. First download/export may take a long time.
echo.
python3\python.exe -B scripts\tools\export_distill_base_onnx.py ^
  --model-id xingyang1/Distill-Any-Depth-Large-hf ^
  --model-name Distill-Any-Depth-Large ^
  --dtype auto ^
  %*
echo.
pause
