import threading

from streaming.audio import STEREO_MIX_NAMES
from streaming.config import DEFAULT_PORT

from stereo_runtime.model_capabilities import (
    COMPILE_FIX_KEYWORDS,
    DISABLE_COREML_KEYWORDS,
    DISABLE_CUDNN_KEYWORDS,
    DISABLE_OPENVINO_KEYWORDS,
    DISABLE_TRITON_KEYWORDS,
    DISABLE_TRT_KEYWORDS,
    FORCE_FP32_KEYWORDS,
    TRT_FIX_KEYWORDS,
)

# Global shutdown event
shutdown_event = threading.Event()

from viewer.assets import crop_icon, get_font_type


from .app_info import DEBUG, OS_NAME, VERSION
from .bootstrap import bootstrap_settings
from .display import (
    _get_device_name_from_mss_monitor,
    get_monitor_size,
)
from .device_runtime import resolve_device_runtime
from .network import get_local_ip
from .runtime_exports import resolve_runtime_exports
from .settings import read_yaml

settings = bootstrap_settings("settings.yaml", os_name=OS_NAME)
from viewer import (
    hide_window_from_capture,
    send_ctrl_cmd_f,
    set_window_to_bottom,
    show_window_in_capture,
)

_RUNTIME_EXPORTS = resolve_runtime_exports(settings, os_name=OS_NAME)
MODEL_MAPPING = _RUNTIME_EXPORTS.model_mapping
STREAM_QUALITY = _RUNTIME_EXPORTS.stream_quality
STREAM_PORT = _RUNTIME_EXPORTS.stream_port
LOCAL_IP = _RUNTIME_EXPORTS.local_ip
RUN_MODE = _RUNTIME_EXPORTS.run_mode
STREAM_MODE = _RUNTIME_EXPORTS.stream_mode
USE_3D_MONITOR = _RUNTIME_EXPORTS.use_3d_monitor
LOSSLESS_SCALING_SUPPORT = _RUNTIME_EXPORTS.lossless_scaling_support
MODEL = _RUNTIME_EXPORTS.model
MODEL_ID = _RUNTIME_EXPORTS.model_id
ALL_MODELS = _RUNTIME_EXPORTS.all_models
CACHE_PATH = _RUNTIME_EXPORTS.cache_path
DEPTH_RESOLUTION = _RUNTIME_EXPORTS.depth_resolution
DEVICE_ID = _RUNTIME_EXPORTS.device_id
FP16 = _RUNTIME_EXPORTS.fp16
MONITOR_INDEX = _RUNTIME_EXPORTS.monitor_index
DISPLAY_MODE = _RUNTIME_EXPORTS.display_mode
STEREO_DISPLAY_INDEX = _RUNTIME_EXPORTS.stereo_display_index
STEREO_DISPLAY_SELECTION = _RUNTIME_EXPORTS.stereo_display_selection
OUTPUT_RESOLUTION = _RUNTIME_EXPORTS.output_resolution
SHOW_FPS = _RUNTIME_EXPORTS.show_fps
DEPTH_STRENGTH = _RUNTIME_EXPORTS.depth_strength
IPD = _RUNTIME_EXPORTS.ipd
CONVERGENCE = _RUNTIME_EXPORTS.convergence
CAPTURE_MODE = _RUNTIME_EXPORTS.capture_mode
WINDOW_TITLE = _RUNTIME_EXPORTS.window_title
TARGET_FPS = _RUNTIME_EXPORTS.target_fps
FPS = _RUNTIME_EXPORTS.fps
FOREGROUND_SCALE = _RUNTIME_EXPORTS.foreground_scale
AA_STRENGTH = _RUNTIME_EXPORTS.aa_strength
USE_TORCH_COMPILE = _RUNTIME_EXPORTS.use_torch_compile
USE_TENSORRT = _RUNTIME_EXPORTS.use_tensorrt
RECOMPILE_TRT = _RUNTIME_EXPORTS.recompile_trt
USE_COREML = _RUNTIME_EXPORTS.use_coreml
RECOMPILE_COREML = _RUNTIME_EXPORTS.recompile_coreml
USE_OPENVINO = _RUNTIME_EXPORTS.use_openvino
RECOMPILE_OPENVINO = _RUNTIME_EXPORTS.recompile_openvino
CAPTURE_TOOL = _RUNTIME_EXPORTS.capture_tool
FILL_16_9 = _RUNTIME_EXPORTS.fill_16_9
LOCAL_VSYNC = _RUNTIME_EXPORTS.local_vsync
UPSCALER = _RUNTIME_EXPORTS.upscaler
UPSCALER_SHARPNESS = _RUNTIME_EXPORTS.upscaler_sharpness
FIX_VIEWER_ASPECT = _RUNTIME_EXPORTS.fix_viewer_aspect
STEREOMIX_DEVICE = _RUNTIME_EXPORTS.stereo_mix_device
STREAM_KEY = _RUNTIME_EXPORTS.stream_key
AUDIO_DELAY = _RUNTIME_EXPORTS.audio_delay
CRF = _RUNTIME_EXPORTS.crf
LANG = _RUNTIME_EXPORTS.language
ROWS = _RUNTIME_EXPORTS.controller_help_rows
ENV_ROWS = _RUNTIME_EXPORTS.environment_help_rows
CONTROLLER_MODEL = _RUNTIME_EXPORTS.controller_model
ENVIRONMENT_MODEL = _RUNTIME_EXPORTS.environment_model
XR_PREVIEW_WINDOW = _RUNTIME_EXPORTS.xr_preview_window

_DEVICE_RUNTIME = resolve_device_runtime(DEVICE_ID)
DEVICE = _DEVICE_RUNTIME.device
DEVICE_INFO = _DEVICE_RUNTIME.device_info
