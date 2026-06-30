import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stereo_runtime.model_artifacts import artifact_paths_for_model
from stereo_runtime.onnx_export import _quiet_onnx_export_warnings, choose_export_dtype, probe_model_dtype


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
    path = artifact_paths_for_model("xingyang1/Distill-Any-Depth-Large-hf", cache_dir=ROOT / "models").onnx_path_for_dtype("fp16")

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


def test_quiet_onnx_export_warnings_only_suppresses_known_export_noise():
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with _quiet_onnx_export_warnings():
            warnings.warn(
                "Converting a tensor to a Python boolean might cause the trace to be incorrect",
                torch.jit.TracerWarning,
            )
            warnings.warn(
                "ONNX export mode is set to TrainingMode.EVAL, but operator 'instance_norm' is set to train=True. Exporting with train=True.",
                UserWarning,
            )
            warnings.warn("real warning", UserWarning)

    assert [str(item.message) for item in caught] == ["real warning"]
