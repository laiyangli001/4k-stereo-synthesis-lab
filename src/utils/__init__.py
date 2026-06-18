import sys, requests
import yaml, threading, time
import os, platform, socket
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

# Models with Disabled TRT 
DISABLE_TRT_KEYWORDS = [
    "dpt-hybrid-midas",
    "depthpro",
    "da3-giant",
    "da3nested-giant",
    "video-depth-anything",
]

TRT_FIX_KEYWORDS = [
    # DA3 models
    "depth-anything/DA3-SMALL",
    "depth-anything/DA3-BASE",
    "depth-anything/DA3-LARGE",
    # "depth-anything/DA3-GIANT",
    "depth-anything/DA3-LARGE-1.1",
    "depth-anything/DA3METRIC-LARGE",
    # "depth-anything/DA3NESTED-GIANT-LARGE",
    "depth-anything/DA3NESTED-GIANT-LARGE-1.1",
    "depth-anything/DA3MONO-LARGE",
    # Video-Depth-Anything
    "depth-anything/Video-Depth-Anything-Small",
    "depth-anything/Video-Depth-Anything-Base",
    "depth-anything/Video-Depth-Anything-Large",
    # Metric-Video-Depth-Anything
    "depth-anything/Metric-Video-Depth-Anything-Small",
    "depth-anything/Metric-Video-Depth-Anything-Base",
    "depth-anything/Metric-Video-Depth-Anything-Large",
    # Intel/zoedepth
    "Intel/zoedepth-nyu-kitti",
    # LC700X/InfiniDepth
    "lc700x/InfiniDepth-Small",
    "lc700x/InfiniDepth-SmallPlus",
    "lc700x/InfiniDepth-Base",
    "lc700x/InfiniDepth-Large",
]

FORCE_FP32_KEYWORDS = [
    # ZoeDepth models
    "Intel/zoedepth-nyu",
    "Intel/zoedepth-kitti",
]

COMPILE_FIX_KEYWORDS = [
    # Video-Depth-Anything
    "depth-anything/Video-Depth-Anything-Small",
    "depth-anything/Video-Depth-Anything-Base",
    "depth-anything/Video-Depth-Anything-Large",
    # Metric-Video-Depth-Anything
    "depth-anything/Metric-Video-Depth-Anything-Small",
    "depth-anything/Metric-Video-Depth-Anything-Base",
    "depth-anything/Metric-Video-Depth-Anything-Large",
]

# Models with Disabled CoreML 
DISABLE_COREML_KEYWORDS = [
    "video-depth-anything",
    "da3-",
    "da3nested",
    "dpt-beit",
    "zoedepth",
    "depthpro",
    "infinidepth",
]

# Models with Disabled OpenVINO 
DISABLE_OPENVINO_KEYWORDS = [
    "da3-",
    "dpt-hybrid-midas-hf",
]

# Disable CuDNN for RX 6000 and 5000 series GPUs
DISABLE_CUDNN_KEYWORDS = ["6950", "6900", "6850", "6800", "6750", "6700", "6650", "6600", "6550", "6500", "6400", "6300", "680", "6100", "5700", "5600", "5500", "5400", "5300", "520", "160", "AMD Radeon(TM) Graphics"]
# Disable Triton for RX 5000 series
DISABLE_TRITON_KEYWORDS = ["520", "160"]
# DISABLE_TRITON_KEYWORDS = ["5700", "5600", "5500", "5400", "5300", "520", "160"]

# Global shutdown event
shutdown_event = threading.Event()

from viewer.assets import crop_icon, get_font_type


from .display import (
    _get_device_name_from_mss_monitor,
    compute_output_resolution,
    get_fps,
    get_monitor_size,
)

def read_yaml(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except UnicodeDecodeError:
        # Fallback to try other common encodings if UTF-8 fails
        try:
            with open(path, "r", encoding="gbk") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            print(f"Failed to load settings.yaml with GBK encoding: {e}")
            return {}

def get_local_ip():
    """Return the local IP address by creating a UDP socket to a public IP."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            # doesn't need to be reachable
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"

# load customized settings
settings = read_yaml("settings.yaml")

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
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

def is_cn_ip():
    try:
        # Get your public IP
        ip = requests.get("https://api.ipify.org").text.strip()
        
        # Query geolocation info from ip-api.com
        response = requests.get(f"http://ip-api.com/json/{ip}", timeout=10)
        response.raise_for_status()
        
        data = response.json()
        country = data.get("country", "")
        
        # print(f"Your IP: {ip}, Country: {country}")
        return country == "China"
    except Exception as e:
        # print(f"Error checking IP location: {e}")
        return False

if is_cn_ip():
    os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
else:
    os.environ['HF_ENDPOINT'] = "https://huggingface.co"

# Model Mapping Dict. Keep the Desktop2Stereo settings shape, but make
# stereo_runtime.model_registry the single source of truth for model names.
from stereo_runtime.model_registry import ModelRegistry

MODEL_MAPPING = {
    spec.name: spec.model_id
    for spec in ModelRegistry.default().list()
}

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

# Handheld Controller Operation Guide for OpenXR Link, can be easily extended for in-game usage.
# ROWS is the normal no-room viewer guide. ENV_ROWS overrides only the room
# viewer guide, while shared controller mappings stay in the same shape.
if LANG == "CN":
    ROWS = [
        ("# 手柄操作指南", "", "", True),
        ("", "", "", False),

        ("[屏幕位置与姿态]", "", "", True),
        ("左握持 + 激光指屏幕", "按住移动或旋转", "屏幕垂直平移或90度旋转", False),
        ("右握持 + 激光指屏幕", "按住移动或旋转", "头部球面旋转或任意旋转", False),
        ("双握持 + 激光指屏幕", "按住移动", "双手中心点移动", False),
        ("左握持 + 左摇杆 左右", "按住推动", "屏幕偏摆旋转", False),
        ("左握持 + 左摇杆 前后", "按住推动", "屏幕俯仰旋转", False),
        ("右握持 + 右摇杆 左右", "按住推动", "屏幕尺寸调整", False),
        ("右握持 + 右摇杆 前后", "按住推动", "屏幕距离调整", False),
        ("", "", "", False),

        ("[预设与杂项]", "", "", True),
        ("左 Menu 键", "短按", "循环面板模式: 面向头/固定/隐藏", False),
        ("左 Y 键", "短按", "恢复默认居中显示", False),
        ("左 Y 键", "长按 1s", "循环屏幕预设", False),
        ("左 X 键", "短按", "显示/隐藏虚拟键盘", False),
        ("左 X 键", "长按 1s", "循环背景颜色", False),
        ("右 A 键", "短按", "屏幕曲面/平面切换", False),
        ("右 A 键", "长按 1s", "循环面板模式: 面向头/固定/隐藏", False),
        ("右 B 键", "短按", "切换VDXR绿屏/背景色", False),
        ("右 B 键", "长按 1s", "重置屏幕方向", False),
        ("", "", "", False),

        ("[深度与视觉]", "", "", True),
        ("右握持 + 左摇杆 前后", "按住推动", "调整深度强度", False),
        ("右握持 + 右摇杆按下", "按住单击", "重置深度强度", False),
        ("左握持 + 左摇杆按下", "按住单击", "切换2D和3D画面", False),
        ("", "", "", False),

        ("[鼠标与快捷键(激光指向屏幕时)]", "", "", True),
        ("右扳机", "全按单击", "鼠标左键单击", False),
        ("右扳机", "持续按住", "鼠标左键按住", False),
        ("左扳机", "全按单击", "鼠标右键单击", False),
        ("左扳机", "持续按住", "鼠标右键框选", False),
        ("右摇杆 前后", "按住推动", "鼠标滚轮滚动", False),
        ("左摇杆 前后", "按住推动", "键盘上下方向键", False),
        ("左摇杆 左右", "按住推动", "键盘左右方向键", False),
        ("左摇杆按下", "短按", "Ctrl+C", False),
        ("左摇杆按下", "长按 1s", "Ctrl+X", False),
        ("右摇杆按下", "短按", "Ctrl+V", False),
        ("右摇杆按下", "长按 1s", "回车键", False),
        ("", "", "", False),

        ("[虚拟键盘(仅键盘显示时)]", "", "", True),
        ("双握持 + 激光指键盘", "按住移动", "键盘绕头球面移动", False),
        ("右握持 + 左摇杆 左右", "按住推动", "键盘宽度缩放", False),
        ("右握持 + 左摇杆 前后", "按住推动", "键盘推拉距离", False),
        ("左握持 + 右摇杆 左右", "按住推动", "键盘偏摆偏移", False),
        ("左握持 + 右摇杆 前后", "按住推动", "键盘俯仰偏移", False),
        ("左握持 + 左摇杆按下", "按住移动", "键盘环绕头部位移", False),
        ("左/右扳机", "半按", "触发键盘按键", False),
        ("", "", "", False),

        ("[手柄模型校准(开发者)]", "", "", True),
        ("右 A+B 键", "同时按住 0.5s", "切换手柄品牌模型", False),
        ("右 A+B 键", "同时按住 5s", "进入/退出校准模式", False),
        ("右 B 键", "校准模式中单击", "保存校准并退出", False),
    ]
    ENV_ROWS = [
        ("# 房间模式手柄操作指南", "", "", True),
        ("", "", "", False),

        ("[房间与面板]", "", "", True),
        ("左 Menu 键", "短按", "循环状态面板: 面向头/固定/隐藏", False),
        ("左 Y 键", "短按", "重新居中到房间屏幕/默认位置", False),
        ("左 Y 键", "长按 1s", "切换房间环境", False),
        ("左 X 键", "短按", "显示/隐藏虚拟键盘", False),
        ("右 A 键", "长按 1s", "循环状态面板: 面向头/固定/隐藏", False),
        ("", "", "", False),

        ("[深度与视觉]", "", "", True),
        ("右握持 + 左摇杆 前后", "按住推动", "调整深度强度", False),
        ("右握持 + 右摇杆按下", "按住单击", "重置深度强度", False),
        ("左握持 + 左摇杆按下", "按住单击", "切换2D和3D画面", False),
        ("", "", "", False),

        ("[鼠标与快捷键(激光指向屏幕时)]", "", "", True),
        ("右扳机", "全按单击", "鼠标左键单击", False),
        ("右扳机", "持续按住", "鼠标左键按住", False),
        ("左扳机", "全按单击", "鼠标右键单击", False),
        ("左扳机", "持续按住", "鼠标右键框选", False),
        ("右摇杆 前后", "按住推动", "鼠标滚轮滚动", False),
        ("左摇杆 前后", "按住推动", "键盘上下方向键", False),
        ("左摇杆 左右", "按住推动", "键盘左右方向键", False),
        ("左摇杆按下", "短按", "Ctrl+C", False),
        ("左摇杆按下", "长按 1s", "Ctrl+X", False),
        ("右摇杆按下", "短按", "Ctrl+V", False),
        ("右摇杆按下", "长按 1s", "回车键", False),
        ("", "", "", False),

        ("[虚拟键盘(仅键盘显示时)]", "", "", True),
        ("单/双握持 + 激光指键盘", "按住移动", "键盘绕头球面移动", False),
        ("右握持 + 左摇杆 左右", "按住推动", "键盘宽度缩放", False),
        ("右握持 + 左摇杆 前后", "按住推动", "键盘推拉距离", False),
        ("左握持 + 右摇杆 左右", "按住推动", "键盘偏摆偏移", False),
        ("左握持 + 右摇杆 前后", "按住推动", "键盘俯仰偏移", False),
        ("左握持 + 左摇杆", "按住推动", "键盘环绕头部位移", False),
        ("左/右扳机", "半按", "触发键盘按键", False),
        ("", "", "", False),

        ("[手柄模型校准(开发者)]", "", "", True),
        ("右 A+B 键", "同时按住 0.5s", "切换手柄品牌模型", False),
        ("右 A+B 键", "同时按住 5s", "进入/退出校准模式", False),
        ("右 B 键", "校准模式中单击", "保存校准并退出", False),
    ]
else:
    ROWS = [
        ("# Controller Operation Guide", "", "", True),
        ("", "", "", False),

        ("[Screen Position & Orientation]", "", "", True),
        ("Left Grip + laser on screen", "Hold & move or twist", "Translate vertically or 90deg rotate", False),
        ("Right Grip + laser on screen", "Hold & move or twist", "Sphere-orbit or free rotate", False),
        ("Both Grips + laser on screen", "Hold & move", "Move by center of both hands", False),
        ("Left Grip + left stick L/R", "Hold & push", "Screen yaw rotation", False),
        ("Left Grip + left stick U/D", "Hold & push", "Screen pitch rotation", False),
        ("Right Grip + right stick L/R", "Hold & push", "Screen width adjustment", False),
        ("Right Grip + right stick U/D", "Hold & push", "Screen distance adjustment", False),
        ("", "", "", False),

        ("[Presets & Misc]", "", "", True),
        ("Left Menu button", "Short press", "Cycle panel: head-facing/fixed/hidden", False),
        ("Left Y button", "Short press", "Reset to default centered display", False),
        ("Left Y button", "Long press 1s", "Cycle screen presets", False),
        ("Left X button", "Short press", "Show/hide virtual keyboard", False),
        ("Left X button", "Long press 1s", "Cycle background color", False),
        ("Right A button", "Short press", "Toggle curved/flat screen", False),
        ("Right A button", "Long press 1s", "Cycle panel: head-facing/fixed/hidden", False),
        ("Right B button", "Short press", "Toggle VDXR green / background", False),
        ("Right B button", "Long press 1s", "Reset screen direction", False),
        ("", "", "", False),

        ("[Depth & Visual]", "", "", True),
        ("Right Grip + left stick U/D", "Hold & push", "Adjust depth strength", False),
        ("Right Grip + right stick press", "Hold & click", "Reset depth strength", False),
        ("Left Grip + left stick press", "Hold & click", "Toggle 2D / 3D", False),
        ("", "", "", False),

        ("[Mouse & Shortcuts (laser on screen)]", "", "", True),
        ("Right trigger", "Full press click", "Left mouse click", False),
        ("Right trigger", "Hold", "Left mouse button hold", False),
        ("Left trigger", "Full press click", "Right mouse click", False),
        ("Left trigger", "Hold", "Right mouse button drag", False),
        ("Right stick U/D", "Hold & push", "Mouse wheel scroll", False),
        ("Left stick U/D", "Hold & push", "Keyboard Up/Down arrows", False),
        ("Left stick L/R", "Hold & push", "Keyboard Left/Right arrows", False),
        ("Left stick press", "Short press", "Ctrl+C", False),
        ("Left stick press", "Long press 1s", "Ctrl+X", False),
        ("Right stick press", "Short press", "Ctrl+V", False),
        ("Right stick press", "Long press 1s", "Enter key", False),
        ("", "", "", False),

        ("[Virtual Keyboard (keyboard visible only)]", "", "", True),
        ("Both Grips + laser on keyboard", "Hold & move", "Orbit keyboard around head", False),
        ("Right Grip + left stick L/R", "Hold & push", "Keyboard width resize", False),
        ("Right Grip + left stick U/D", "Hold & push", "Keyboard push/pull distance", False),
        ("Left Grip + right stick L/R", "Hold & push", "Keyboard yaw offset", False),
        ("Left Grip + right stick U/D", "Hold & push", "Keyboard pitch offset", False),
        ("Left Grip + left stick press", "Hold & move", "Keyboard sphere orbit", False),
        ("Left/Right trigger", "Half press", "Trigger key on keyboard", False),
        ("", "", "", False),

        ("[Controller Calibration (Developer)]", "", "", True),
        ("Right A+B buttons", "Hold both 0.5s", "Switch controller brand model", False),
        ("Right A+B buttons", "Hold both 5s", "Enter/exit calibration mode", False),
        ("Right B button", "Click in calibration mode", "Save calibration & exit", False),
    ]
    ENV_ROWS = [
        ("# Room Controller Operation Guide", "", "", True),
        ("", "", "", False),

        ("[Room & Panel]", "", "", True),
        ("Left Menu button", "Short press", "Cycle status panel: head-facing/fixed/hidden", False),
        ("Left Y button", "Short press", "Recenter to room screen/default position", False),
        ("Left Y button", "Long press 1s", "Switch room environment", False),
        ("Left X button", "Short press", "Show/hide virtual keyboard", False),
        ("Right A button", "Long press 1s", "Cycle status panel: head-facing/fixed/hidden", False),
        ("", "", "", False),

        ("[Depth & Visual]", "", "", True),
        ("Right Grip + left stick U/D", "Hold & push", "Adjust depth strength", False),
        ("Right Grip + right stick press", "Hold & click", "Reset depth strength", False),
        ("Left Grip + left stick press", "Hold & click", "Toggle 2D / 3D", False),
        ("", "", "", False),

        ("[Mouse & Shortcuts (laser on screen)]", "", "", True),
        ("Right trigger", "Full press click", "Left mouse click", False),
        ("Right trigger", "Hold", "Left mouse button hold", False),
        ("Left trigger", "Full press click", "Right mouse click", False),
        ("Left trigger", "Hold", "Right mouse button drag", False),
        ("Right stick U/D", "Hold & push", "Mouse wheel scroll", False),
        ("Left stick U/D", "Hold & push", "Keyboard Up/Down arrows", False),
        ("Left stick L/R", "Hold & push", "Keyboard Left/Right arrows", False),
        ("Left stick press", "Short press", "Ctrl+C", False),
        ("Left stick press", "Long press 1s", "Ctrl+X", False),
        ("Right stick press", "Short press", "Ctrl+V", False),
        ("Right stick press", "Long press 1s", "Enter key", False),
        ("", "", "", False),

        ("[Virtual Keyboard (keyboard visible only)]", "", "", True),
        ("One/Both Grips + laser on keyboard", "Hold & move", "Orbit keyboard around head", False),
        ("Right Grip + left stick L/R", "Hold & push", "Keyboard width resize", False),
        ("Right Grip + left stick U/D", "Hold & push", "Keyboard push/pull distance", False),
        ("Left Grip + right stick L/R", "Hold & push", "Keyboard yaw offset", False),
        ("Left Grip + right stick U/D", "Hold & push", "Keyboard pitch offset", False),
        ("Left Grip + left stick", "Hold & push", "Keyboard sphere orbit", False),
        ("Left/Right trigger", "Half press", "Trigger key on keyboard", False),
        ("", "", "", False),

        ("[Controller Calibration (Developer)]", "", "", True),
        ("Right A+B buttons", "Hold both 0.5s", "Switch controller brand model", False),
        ("Right A+B buttons", "Hold both 5s", "Enter/exit calibration mode", False),
        ("Right B button", "Click in calibration mode", "Save calibration & exit", False),
    ]

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
