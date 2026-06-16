"""4K stereo synthesis research prototypes."""

from .openxr_render import (
    OpenXREyeView,
    OpenXRFov,
    OpenXRRenderConfig,
    OpenXRScreenPose,
    OpenXRStereoResult,
    build_openxr_eye_mvp,
    is_pyopenxr_available,
    render_openxr_stereo,
)
from .synthesis import StereoConfig, StereoResult, synthesize_stereo

__all__ = [
    "OpenXREyeView",
    "OpenXRFov",
    "OpenXRRenderConfig",
    "OpenXRScreenPose",
    "OpenXRStereoResult",
    "StereoConfig",
    "StereoResult",
    "build_openxr_eye_mvp",
    "is_pyopenxr_available",
    "render_openxr_stereo",
    "synthesize_stereo",
]
