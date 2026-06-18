import sys
import threading, time
import os, platform
import importlib.util
from pathlib import Path

# Debug Mode
DEBUG = False
# App Version
VERSION = "2.5.0Beta"
# Get OS name
OS_NAME = platform.system()
from streaming.audio import STEREO_MIX_NAMES
from streaming.config import DEFAULT_PORT, resolve_streaming_config

from stereo_runtime.model_capabilities import (
    COMPILE_FIX_KEYWORDS,
    DISABLE_COREML_KEYWORDS,
    DISABLE_CUDNN_KEYWORDS,
    DISABLE_OPENVINO_KEYWORDS,
    DISABLE_TRITON_KEYWORDS,
    DISABLE_TRT_KEYWORDS,
    FORCE_FP32_KEYWORDS,
    TRT_FIX_KEYWORDS,
    model_name_mapping,
)
from stereo_runtime.depth_settings import resolve_depth_settings

# Global shutdown event
shutdown_event = threading.Event()

from viewer.assets import crop_icon, get_font_type


from .display import (
    _get_device_name_from_mss_monitor,
    get_monitor_size,
)
from .network import configure_huggingface_endpoint, get_local_ip
from .run_mode import resolve_run_mode
from .settings import load_settings, read_yaml
from viewer.settings import resolve_viewer_settings

# load customized settings
settings = load_settings("settings.yaml")

# Ignore wanning for MPS
if OS_NAME == "Darwin":
    import os, warnings
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    warnings.filterwarnings(
        "ignore",
        message=".*aten::upsample_bicubic2d.out.*MPS backend.*",
        category=UserWarning)
from viewer.window_control import (
    hide_window_from_capture,
    send_ctrl_cmd_f,
    set_window_to_bottom,
    show_window_in_capture,
)
        
# Set Hugging Face environment variable
configure_huggingface_endpoint()

# Model Mapping Dict. Keep the Desktop2Stereo settings shape, but make
# stereo_runtime.model_registry the single source of truth for model names.
MODEL_MAPPING = model_name_mapping()

# Streamer Settings
_STREAMING_CONFIG = resolve_streaming_config(settings)
STREAM_QUALITY = _STREAMING_CONFIG.stream_quality
STREAM_PORT = _STREAMING_CONFIG.stream_port
LOCAL_IP = _STREAMING_CONFIG.local_ip

# Get settings
_RUN_MODE_CONFIG = resolve_run_mode(
    settings["Run Mode"],
    os_name=OS_NAME,
    fix_viewer_aspect=settings["Fix Viewer Aspect"],
    lossless_scaling_support=settings["Lossless Scaling Support"],
)
RUN_MODE = _RUN_MODE_CONFIG.run_mode
STREAM_MODE = _RUN_MODE_CONFIG.stream_mode
USE_3D_MONITOR = _RUN_MODE_CONFIG.use_3d_monitor
LOSSLESS_SCALING_SUPPORT = _RUN_MODE_CONFIG.lossless_scaling_support
_DEPTH_SETTINGS = resolve_depth_settings(settings)
MODEL = _DEPTH_SETTINGS.model
MODEL_ID = _DEPTH_SETTINGS.model_id
ALL_MODELS = _DEPTH_SETTINGS.all_models
CACHE_PATH = _DEPTH_SETTINGS.cache_path
DEPTH_RESOLUTION = _DEPTH_SETTINGS.depth_resolution
DEVICE_ID = _DEPTH_SETTINGS.device_id
FP16 = _DEPTH_SETTINGS.fp16
_VIEWER_SETTINGS = resolve_viewer_settings(settings)
MONITOR_INDEX = _VIEWER_SETTINGS.monitor_index
DISPLAY_MODE = _VIEWER_SETTINGS.display_mode
STEREO_DISPLAY_INDEX = _VIEWER_SETTINGS.stereo_display_index
STEREO_DISPLAY_SELECTION = _VIEWER_SETTINGS.stereo_display_selection
OUTPUT_RESOLUTION = _VIEWER_SETTINGS.output_resolution
SHOW_FPS = _VIEWER_SETTINGS.show_fps
DEPTH_STRENGTH = _VIEWER_SETTINGS.depth_strength
IPD = _VIEWER_SETTINGS.ipd
CONVERGENCE = _VIEWER_SETTINGS.convergence
CAPTURE_MODE = _VIEWER_SETTINGS.capture_mode
WINDOW_TITLE = _VIEWER_SETTINGS.window_title
TARGET_FPS = _VIEWER_SETTINGS.target_fps
FPS = _VIEWER_SETTINGS.fps

FOREGROUND_SCALE = _DEPTH_SETTINGS.foreground_scale
AA_STRENGTH = _DEPTH_SETTINGS.aa_strength
USE_TORCH_COMPILE = _DEPTH_SETTINGS.use_torch_compile
USE_TENSORRT = _DEPTH_SETTINGS.use_tensorrt
RECOMPILE_TRT = _DEPTH_SETTINGS.recompile_trt
USE_COREML = _DEPTH_SETTINGS.use_coreml
RECOMPILE_COREML = _DEPTH_SETTINGS.recompile_coreml
USE_OPENVINO = _DEPTH_SETTINGS.use_openvino
RECOMPILE_OPENVINO = _DEPTH_SETTINGS.recompile_openvino

def _load_capture_select():
    # Load the selector without importing capture.__init__, which still depends
    # on utils during the package migration.
    path = Path(__file__).resolve().parents[1] / "capture" / "capture_select.py"
    spec = importlib.util.spec_from_file_location("_d2s_capture_select", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_resolve_capture_tool = _load_capture_select().resolve_capture_tool

CAPTURE_TOOL = _resolve_capture_tool(settings["Capture Tool"])
FILL_16_9 = _VIEWER_SETTINGS.fill_16_9
LOCAL_VSYNC = _VIEWER_SETTINGS.local_vsync
UPSCALER = _VIEWER_SETTINGS.upscaler
UPSCALER_SHARPNESS = _VIEWER_SETTINGS.upscaler_sharpness
FIX_VIEWER_ASPECT = _RUN_MODE_CONFIG.fix_viewer_aspect
STEREOMIX_DEVICE = _STREAMING_CONFIG.stereo_mix_device
STREAM_KEY = _STREAMING_CONFIG.stream_key
AUDIO_DELAY = _STREAMING_CONFIG.audio_delay
CRF = _STREAMING_CONFIG.crf
LANG = settings["Language"]

from viewer.controller_help import get_controller_help_rows

ROWS, ENV_ROWS = get_controller_help_rows(LANG)

# Specify the Stereo Display for output
CONTROLLER_MODEL = _VIEWER_SETTINGS.controller_model
ENVIRONMENT_MODEL = _VIEWER_SETTINGS.environment_model
XR_PREVIEW_WINDOW = _VIEWER_SETTINGS.xr_preview_window

# Initialize Device
from .device import get_device
    
DEVICE, DEVICE_INFO = get_device(DEVICE_ID)
