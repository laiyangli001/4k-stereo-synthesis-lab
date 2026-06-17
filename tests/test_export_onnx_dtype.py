import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "tools"))

from export_distill_base_onnx import choose_export_dtype, default_output_path, probe_model_dtype


def test_choose_export_dtype_auto_cuda_defaults_fp16():
    dtype, name, reason = choose_export_dtype("lc700x/Distill-Any-Depth-Base-hf", torch.device("cuda"), "auto")

    assert dtype == torch.float16
    assert name == "fp16"
    assert "CUDA" in reason


def test_choose_export_dtype_auto_cpu_uses_fp32():
    dtype, name, reason = choose_export_dtype("lc700x/Distill-Any-Depth-Base-hf", torch.device("cpu"), "auto")

    assert dtype == torch.float32
    assert name == "fp32"
    assert "non-CUDA" in reason


def test_choose_export_dtype_force_fp32_keyword():
    dtype, name, reason = choose_export_dtype("lc700x/InfiniDepth-Large", torch.device("cuda"), "auto")

    assert dtype == torch.float32
    assert name == "fp32"
    assert "requires fp32" in reason


def test_default_output_path_uses_actual_dtype_name():
    path = default_output_path("xingyang1/Distill-Any-Depth-Large-hf", "fp16", 294, 518)

    assert path.name == "model_fp16_294x518.onnx"
    assert "models--xingyang1--Distill-Any-Depth-Large-hf" in str(path)


def test_probe_model_dtype_rejects_all_zero_output():
    class Output:
        predicted_depth = torch.zeros(1, 4, 4)

    class Model:
        def eval(self):
            return self

        def __call__(self, pixel_values):
            return Output()

    ok, reason = probe_model_dtype(Model(), device=torch.device("cpu"), dtype=torch.float32, height=4, width=4)

    assert ok is False
    assert "all zero" in reason


def test_probe_model_dtype_accepts_finite_dynamic_output():
    class Output:
        predicted_depth = torch.arange(16, dtype=torch.float32).view(1, 4, 4)

    class Model:
        def eval(self):
            return self

        def __call__(self, pixel_values):
            return Output()

    ok, reason = probe_model_dtype(Model(), device=torch.device("cpu"), dtype=torch.float32, height=4, width=4)

    assert ok is True
    assert "ok:" in reason
