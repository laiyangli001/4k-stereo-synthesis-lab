"""Intel OpenVINO and XPU depth providers."""

from .pytorch_xpu import (
    DistillAnyDepthBaseXpu,
    GenericAutoDepthXpuProvider,
    create_pytorch_xpu_provider,
    is_xpu_torch_available,
    xpu_device_name,
)

__all__ = [
    "DistillAnyDepthBaseXpu",
    "GenericAutoDepthXpuProvider",
    "create_pytorch_xpu_provider",
    "is_xpu_torch_available",
    "xpu_device_name",
]
