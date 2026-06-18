"""OpenXR viewer split package with runtime-direct GPU interop support."""

from .base import OpenXRViewer as BaseOpenXRViewer
from .environment import OpenXRViewer as EnvironmentOpenXRViewer
from .implementation import OPENXR_AVAILABLE, OpenXRViewerCore, load_glb_model

OpenXRViewer = BaseOpenXRViewer

__all__ = [
    "OPENXR_AVAILABLE",
    "OpenXRViewer",
    "OpenXRViewerCore",
    "BaseOpenXRViewer",
    "EnvironmentOpenXRViewer",
    "load_glb_model",
]