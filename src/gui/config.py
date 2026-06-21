import json
import os

import yaml

from utils import ALL_MODELS, DEFAULT_PORT

from .paths import BASE_DIR


_MODEL_SIZES = ["Small", "SmallPlus", "Base", "Large", "Giant"]
_SIZE_ORDER = {s: i for i, s in enumerate(_MODEL_SIZES)}


def parse_model_name(name):
    """Split model name into (family, size). DepthPro treated as Large."""
    parts = name.split("-")
    size_parts = []
    i = len(parts) - 1
    while i >= 0:
        matched = None
        for sz in _MODEL_SIZES:
            if parts[i].upper() == sz.upper():
                matched = sz
                break
        if matched:
            size_parts.insert(0, matched)
            i -= 1
        else:
            break
    if size_parts:
        family = "-".join(parts[:i + 1])
        size = "-".join(size_parts)
        return (family, size)
    return (name, "")


def build_family_size_map(model_list):
    """Returns (families_ordered, family_to_sizes) from list of full model names."""
    families = []
    family_to_sizes = {}
    for name in model_list:
        family, size = parse_model_name(name)
        if family not in family_to_sizes:
            family_to_sizes[family] = []
            families.append(family)
        if size and size not in family_to_sizes[family]:
            family_to_sizes[family].append(size)
    for family in family_to_sizes:
        family_to_sizes[family].sort(key=lambda s: _SIZE_ORDER.get(s, 99))
    return families, family_to_sizes


DEFAULT_MODEL_LIST = list(ALL_MODELS.keys())
DEFAULT_FAMILIES, FAMILY_TO_SIZES = build_family_size_map(DEFAULT_MODEL_LIST)
FAMILY_SIZE_TO_MODEL = {}
for name in DEFAULT_MODEL_LIST:
    f, s = parse_model_name(name)
    FAMILY_SIZE_TO_MODEL[(f, s)] = name


DEFAULTS = {
    "Capture Mode": "Monitor",
    "Monitor Index": 1,
    "Window Title": "",
    "Show FPS": False,
    "Model List": DEFAULT_MODEL_LIST,
    "Depth Model": DEFAULT_MODEL_LIST[0] if DEFAULT_MODEL_LIST else "",
    "Depth Strength": 2.0,
    "Depth Quick": "Standard",
    "Depth Resolution": 322,
    "Anti-aliasing": 2,
    "Depth Antialias Strength": 0.4,
    "Foreground Scale": 0.5,
    "IPD": 0.064,
    "Convergence": 0.0,
    "Stereo Scale": 0.5,
    "Stereo Preset": "cinema",
    "Stereo Quality": "quality_4k",
    "Max Shift Ratio": 0.05,
    "Temporal": True,
    "Temporal Strength": 0.7,
    "Auto Scene Reset": True,
    "Scene Reset Threshold": 0.22,
    "Reset Cooldown Frames": 3,
    "Edge Dilation": 2,
    "Edge Threshold": 0.04,
    "Cross Eyed": False,
    "Anaglyph Method": "red_cyan",
    "Display Mode": "Half-SBS",
    "FP16": False,
    "torch.compile": False,
    "TensorRT": False,
    "Recompile TensorRT": False,
    "CoreML": False,
    "Recompile CoreML": False,
    "MIGraphX": False,
    "Recompile MIGraphX": False,
    "Recompile OpenVINO": False,
    "Computing Device": 0,
    "Language": "EN",
    "Run Mode": "OpenXR Link",
    "XR Preview Window": True,
    "Local VSync": True,
    "Target FPS": 0,
    "Processing Resolution": "Auto",
    "Upscaler": "Off",
    "Upscaler Sharpness": 0.35,
    "Stream Protocol": "HLS",
    "Streamer Port": DEFAULT_PORT,
    "Stream Quality": 100,
    "Stream Key": "live",
    "Stereo Mix": None,
    "CRF": 20,
    "Audio Delay": -0.15,
    "Controller Model": "PICO",
    "Environment Model": "None",
    "Lossless Scaling Support": False,
    "Capture Tool": "none",
    "Fill 16:9": True,
    "Fix Viewer Aspect": False,
    "Stereo Output": None,
}


def discover_environment_keys():
    """Return canonical environment keys saved to settings.yaml."""
    env_base = os.path.join(BASE_DIR, "xr_viewer", "environments")
    options = ["None"]
    if not os.path.isdir(env_base):
        return options
    if os.path.exists(os.path.join(env_base, "environment.glb")):
        options.append("Default")
    room_dirs = []
    for name in os.listdir(env_base):
        room_dir = os.path.join(env_base, name)
        if os.path.isdir(room_dir) and not name.startswith(".") and os.path.isfile(os.path.join(room_dir, "environment.glb")):
            room_dirs.append(name)
    options.extend(sorted(room_dirs, key=str.lower))
    return options


def load_environment_display_names(keys=None):
    """Return per-folder display_name values from profile.json."""
    env_base = os.path.join(BASE_DIR, "xr_viewer", "environments")
    names_by_key = {}
    for key in keys or discover_environment_keys():
        if key in ("None", "Default"):
            continue
        names = {}
        profile_path = os.path.join(env_base, key, "profile.json")
        try:
            if os.path.isfile(profile_path):
                with open(profile_path, "r", encoding="utf-8-sig") as f:
                    raw = f.read().strip()
                if raw:
                    profile = json.loads(raw) or {}
                    display_name = profile.get("display_name") if isinstance(profile, dict) else None
                    if isinstance(display_name, dict):
                        names = {str(k): str(v) for k, v in display_name.items() if v}
                    elif isinstance(display_name, str) and display_name:
                        names = {"EN": display_name, "CN": display_name}
        except (OSError, ValueError):
            names = {}
        names_by_key[key] = names
    return names_by_key


def environment_display_label(key, lang="EN", names_by_key=None):
    """Map canonical environment key to the localized GUI label."""
    if key == "None":
        return "None"
    if key == "Default":
        return "默认" if lang == "CN" else "Default"
    names = (names_by_key or {}).get(key) or {}
    return names.get(lang) or names.get("EN") or key


def environment_key_from_label(label, lang="EN", keys=None, names_by_key=None):
    """Map a localized environment label back to its canonical key."""
    text = str(label or "").strip()
    keys = keys or discover_environment_keys()
    for key in keys:
        if environment_display_label(key, lang, names_by_key) == text:
            return key
    for key in keys:
        if key.lower() == text.lower():
            return key
    if text.lower() == "默认":
        return "Default"
    return "None" if "None" in keys else (keys[0] if keys else "None")


def get_environment_model_options(lang="EN", return_keys=False):
    """Return selectable room environment names for the GUI."""
    keys = discover_environment_keys()
    if return_keys:
        return keys
    names_by_key = load_environment_display_names(keys)
    return [environment_display_label(key, lang, names_by_key) for key in keys]


HAVE_YAML = True


def save_yaml(path, cfg):
    if not HAVE_YAML:
        return False, "PyYAML not installed"
    try:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
        os.replace(tmp, path)
        return True, ""
    except Exception as e:
        return False, str(e)
