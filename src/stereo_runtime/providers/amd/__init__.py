"""AMD ROCm and DirectML depth providers."""

from .pytorch_rocm import (
    DistillAnyDepthBaseRocm,
    GenericAutoDepthRocmProvider,
    create_pytorch_rocm_provider,
    is_rocm_torch_available,
    rocm_device_name,
)

__all__ = [
    "DistillAnyDepthBaseRocm",
    "GenericAutoDepthRocmProvider",
    "create_pytorch_rocm_provider",
    "is_rocm_torch_available",
    "rocm_device_name",
]
