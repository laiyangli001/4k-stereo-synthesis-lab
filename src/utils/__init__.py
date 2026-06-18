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
# Define StereoMix devices
STEREO_MIX_NAMES = [
# English
"stereo mix", "what you hear", "loopback", "system audio", "wave out mix", "mixed output",
# Chinese
"立体声混音", "您听到的声音", "环路", "系统音频", "波形输出混合", "混合输出",
# Japanese
"ステレオ ミックス", "ステレオミックス", "ループバック", "システムオーディオ", "ミックス出力",
# Spanish
"mezcla estéreo", "lo que escuchas", "bucle", "audio del sistema", "salida mixta",
# French
"mixage stéréo", "bouclage", "audio système", "sortie mixte",
# German
"stereomix", "was du hörst", "loopback", "systemaudio", "gemischte ausgabe",
# macOS specific
"blackhole", "loopback", "aggregate device", "multi-output device", "virtual desktop speakers", "remote sound",
# Linux specific
"monitor"
]

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

# Global shutdown event
shutdown_event = threading.Event()

from viewer.assets import crop_icon, get_font_type


from .display import (
    _get_device_name_from_mss_monitor,
    compute_output_resolution,
    get_fps,
    get_monitor_size,
)
from .network import configure_huggingface_endpoint, get_local_ip
from .settings import load_settings, read_yaml

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
DEFAULT_PORT = 1122
STREAM_QUALITY = settings["Stream Quality"]
STREAM_PORT = settings["Streamer Port"]
LOCAL_IP = get_local_ip()

# Get settings
RUN_MODE = settings["Run Mode"]
# Add for 3D monitor
USE_3D_MONITOR = False
STREAM_MODE = None

# Add for FrameGen
LOSSLESS_SCALING_SUPPORT = False
MODEL = settings["Depth Model"]
MODEL_ID = MODEL_MAPPING[MODEL]
ALL_MODELS = settings["Model List"]
CACHE_PATH = "./models"
DEPTH_RESOLUTION = settings["Depth Resolution"]
DEVICE_ID = settings["Computing Device"]
FP16 = settings["FP16"]
MONITOR_INDEX,  DISPLAY_MODE = settings["Monitor Index"], settings["Display Mode"]
STEREO_DISPLAY_INDEX = settings.get("Stereo Output")
STEREO_DISPLAY_SELECTION = False if not STEREO_DISPLAY_INDEX else True
OUTPUT_RESOLUTION = compute_output_resolution(
    settings.get("Processing Resolution", "Auto"),
    DISPLAY_MODE,
    MONITOR_INDEX,
    STEREO_DISPLAY_INDEX,
)
SHOW_FPS, DEPTH_STRENGTH = settings["Show FPS"], settings["Depth Strength"]
IPD = settings["IPD"]
CONVERGENCE = settings["Convergence"]
CAPTURE_MODE = settings["Capture Mode"]
WINDOW_TITLE = settings["Window Title"] if CAPTURE_MODE == "Window" else None
TARGET_FPS = int(settings.get("Target FPS", 0) or 0)
FPS = TARGET_FPS if 1 <= TARGET_FPS <= 240 else get_fps(WINDOW_TITLE, MONITOR_INDEX)

# Image Processing Parameters
FOREGROUND_SCALE = settings["Foreground Scale"] / 10 # 0-10
AA_STRENGTH = settings["Anti-aliasing"] * 2

# Experimental Settings
USE_TORCH_COMPILE = settings["torch.compile"]
USE_TENSORRT = settings["TensorRT"] # use TensorRT for CUDA
RECOMPILE_TRT = settings["Recompile TensorRT"] # recompile TensorRT engine

USE_COREML = settings["CoreML"] # use CoreML for MacOS
RECOMPILE_COREML = settings["Recompile CoreML"] # recompile CoreML mlpackage

USE_OPENVINO = settings["OpenVINO"]  # use OpenVINO for Intel
RECOMPILE_OPENVINO = settings["Recompile OpenVINO"] # recompile OpenVINO IR

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
FILL_16_9 = settings["Fill 16:9"]
LOCAL_VSYNC = settings.get("Local VSync", True)
UPSCALER = str(settings.get("Upscaler", "Off") or "Off")
if UPSCALER.strip().lower() in ("auto", "自动"):
    UPSCALER = "Auto"
elif UPSCALER.strip().lower() in ("off", "关闭"):
    UPSCALER = "Off"
elif UPSCALER.strip().lower() == "fsr1":
    UPSCALER = "FSR1"
try:
    UPSCALER_SHARPNESS = float(settings.get("Upscaler Sharpness", 0.35))
except (TypeError, ValueError):
    UPSCALER_SHARPNESS = 0.35
UPSCALER_SHARPNESS = max(0.0, min(1.0, UPSCALER_SHARPNESS))
FIX_VIEWER_ASPECT = True if RUN_MODE == "RTMP Streamer" else settings["Fix Viewer Aspect"] # Keep Viewer Aspect for RTMP with LOSSLESS_SCALING_SUPPORT
STEREOMIX_DEVICE = settings["Stereo Mix"] # RTMP StereoMix Device
STREAM_KEY = settings["Stream Key"]
AUDIO_DELAY = settings["Audio Delay"]
CRF = settings["CRF"]
LANG = settings["Language"]

from viewer.controller_help import get_controller_help_rows

ROWS, ENV_ROWS = get_controller_help_rows(LANG)

# Determin the run mode and stream mode
if RUN_MODE == "Local Viewer":
    RUN_MODE = "Viewer"
elif RUN_MODE == "3D Monitor" and OS_NAME == "Windows":
    RUN_MODE = "Viewer"
    USE_3D_MONITOR = True
elif RUN_MODE == "MJPEG Streamer":
    RUN_MODE = "Viewer"
    STREAM_MODE = "MJPEG" 
elif RUN_MODE == "RTMP Streamer":
    RUN_MODE = "Viewer"
    STREAM_MODE = "RTMP"
    if OS_NAME == "Windows":
        # Frame Generation Settings for RTMP, Local Viewer not requried
        LOSSLESS_SCALING_SUPPORT = settings["Lossless Scaling Support"]
elif RUN_MODE == "OpenXR Link":
    RUN_MODE = "OpenXR"
else:
    RUN_MODE = "Streamer"

# Specify the Stereo Display for output
CONTROLLER_MODEL = settings["Controller Model"]
ENVIRONMENT_MODEL = settings.get("Environment Model", "Default")
XR_PREVIEW_WINDOW = settings.get("XR Preview Window", True)

# Initialize Device
from .device import get_device
    
DEVICE, DEVICE_INFO = get_device(DEVICE_ID)
