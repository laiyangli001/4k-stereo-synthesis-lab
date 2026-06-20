import gettext
from types import MappingProxyType


DEFAULT_LOCALE = "EN"

MESSAGE_CATALOGS = {
    "EN": {
        "Monitor": "Input Monitor",
        "Window": "Input Window",
        "Refresh": "Refresh",
        "Show FPS": "Show FPS",
        "IPD (m):": "IPD (mm):",
        "Stereo Scale:": "Stereo Scale:",
        "Convergence:": "Convergence:",
        "Display Mode:": "Display Mode:",
        "Depth Model:": "Depth Model:",
        "Depth Strength:": "Depth Strength:",
        "Depth Quick:": "Depth Quick:",
        "Soft": "Soft",
        "Standard": "Standard",
        "Enhanced": "Enhanced",
        "Stereo Mode:": "Stereo Mode:",
        "Cinema": "Cinema",
        "Game / Low Latency": "Game / Low Latency",
        "Still Image / HQ": "Still Image / HQ",
        "Debug / Export": "Debug / Export",
        "Synthetic View:": "Synthetic View:",
        "fast": "Lowest",
        "fast_plus": "Medium",
        "quality_4k": "High",
        "hq_4k": "Highest",
        "Max Shift Ratio:": "Shift Ratio:",
        "Temporal Strength:": "Temporal Strength:",
        "Temporal": "Temporal",
        "Scene Threshold:": "Scene Threshold:",
        "Reset Cooldown:": "Reset Cooldown:",
        "Auto Scene Reset": "Auto Scene Reset",
        "Edge Dilation:": "Edge Dilation:",
        "Edge Threshold:": "Edge Threshold:",
        "On": "On",
        "Anaglyph:": "Anaglyph:",
        "Cross Eyed": "Cross Eyed",
        "Advanced Stereo": "Advanced Stereo",
        "Advanced Device Options": "Advanced Options",
        "Depth Resolution:": "Depth Resolution:",
        "Anti-aliasing:": "Anti-aliasing:",
        "Foreground Scale:": "Foreground Scale:",
        "FP16": "FP16",
        "Inference Acceleration:": "Acceleration:",
        "Recompile TensorRT": "Recompile TensorRT",
        "Recompile CoreML": "Recompile CoreML",
        "Recompile OpenVINO": "Recompile OpenVINO",
        "Stop": "Stop",
        "Computing Device:": "Computing Device:",
        "Reset": "Reset",
        "Run": "Run",
        "Set Language:": "Set Language:",
        "Error": "Error",
        "Warning": "Warning",
        "Saved": "Run Desktop2Stereo",
        "PyYAML not installed, cannot save YAML file.": "PyYAML not installed, cannot save YAML file.",
        "Settings saved to settings.yaml": "Settings saved to settings.yaml",
        "Failed to save settings.yaml:": "Failed to save settings.yaml:",
        "Could not retrieve monitor list.\nFalling back to indexes 1 and 2.": "Could not retrieve monitor list.\nFalling back to indexes 1 and 2.",
        "Loaded settings.yaml at startup": "Loaded settings.yaml at startup",
        "Running": "Running... (Hold ESC 3s to Stop)",
        "Stopped": "Stopped.",
        "Countdown": "Settings saved to settings.yaml, starting...",
        "A thread already running!": "A thread already running!",
        "No windows found": "No windows found",
        "Selected input window:": "Selected input window:",
        "Selected input monitor:": "Selected input monitor:",
        "Run Mode:": "Run Mode:",
        "Local Viewer": "Local Viewer",
        "Legacy Streamer": "Legacy Streamer",
        "MJPEG Streamer": "MJPEG Streamer",
        "RTMP Streamer": "RTMP Streamer",
        "Stream Protocol:": "Stream Protocol:",
        "Stream Key": "Stream Key:",
        "Stereo Mix": "Stereo Mix:",
        "CRF": "CRF:",
        "Audio Delay": "Audio Delay (s):",
        "Lossless Scaling Support": "LSFG",
        "3D Monitor": "3D Monitor",
        "OpenXR Link": "OpenXR Link",
        "XR Preview Window": "XR Preview Window",
        "Local VSync": "Local VSync",
        "Capture FPS:": "Capture FPS:",
        "Upscaler:": "Upscaler:",
        "Upscaler Sharpness:": "Sharpness:",
        "Auto": "Auto",
        "Off": "Off",
        "Streamer Port:": "Streamer Port:",
        "Streamer URL": "Streamer URL:",
       "Preview":"Preview",
        "Stream Quality:": "Stream Quality:",
        "Host": "Host:",
        "Invalid port number (1-65535)": "Invalid port number (must be between 1-65535)",
        "Invalid port number": "Port must be a number",
        "Please select a window before running in Window capture mode": "Please select a window before running in Window capture mode",
        "The selected window no longer exists. Please refresh and select a valid window.": "The selected window no longer exists. Please refresh and select a valid window.",
        "Failed to stop process on exit:": "Failed to stop process on exit:",
        "Failed to stop process:": "Failed to stop process:",
        "Failed to run process:": "Failed to run process:",
        "Failed to load settings.yaml:": "Failed to load settings.yaml:",
        "Opening URL in browser": "Opening URL in browser",
        "Controller:": "Controller:",
        "Environment:": "Room:",
        "Capture Tool:": "Capture Tool:",
        "Fill 16:9": "16:9",
        "Fix Viewer Aspect": "Fix Aspect",
        "Stereo Output:": "Stereo Output:",
        "Theme:": "Theme:",
        "DesktopDuplication selected: Window capture mode disabled.": "DesktopDuplication selected: Window capture mode disabled.",
        "torch.compile": "torch.compile",
        "TensorRT": "TensorRT",
        "CoreML": "CoreML",
        "OpenVINO": "OpenVINO",
        "tooltip_window": "Select a window to capture",
        "tooltip_depth_model": "Depth estimation model",
        "tooltip_depth_res": "Depth map resolution",
        "tooltip_convergence": "Stereo convergence",
        "tooltip_depth_strength": "Depth effect intensity",
        "tooltip_depth_quick": "Quick fixed depth presets for everyday use: Soft, Standard, or Enhanced",
        "tooltip_stereo_preset": "Auto switches scenes by weighted signals; manual presets force Cinema, Game, Still Image, or Debug behavior",
        "tooltip_stereo_quality": "Stereo synthesis backend: fast is lowest latency, quality_4k is balanced, hq_4k favors still-image quality",
        "tooltip_max_shift": "Maximum horizontal shift as a ratio of image width; higher values increase stereo separation",
        "tooltip_temporal_strength": "Temporal smoothing strength for stereo output; higher values reduce flicker but can add lag",
        "tooltip_temporal": "Enable temporal stabilization between frames",
        "tooltip_scene_reset": "Scene-change threshold for resetting temporal history; lower values reset more often",
        "tooltip_reset_cooldown": "Minimum frame cooldown between automatic temporal resets",
        "tooltip_auto_scene_reset": "Automatically reset temporal state when a scene cut is detected",
        "tooltip_edge_dilation": "Expands detected depth edges for occlusion handling",
        "tooltip_edge_threshold": "Depth edge sensitivity; lower values detect more edges",
        "tooltip_anaglyph": "Color pair used when Display Mode is Anaglyph",
        "tooltip_cross_eyed": "Swap left and right eyes for cross-eyed viewing",
        "tooltip_advanced_stereo": "Show expert stereo/runtime parameters. Leave off for the simplified everyday UI.",
        "tooltip_advanced_device_options": "Show capture frame pacing and image enhancement controls.",
        "tooltip_foreground_scale": "Foreground object scale",
        "tooltip_antialiasing": "Anti-aliasing level",
        "tooltip_ipd": "Interpupillary distance (mm)",
        "tooltip_stereo_scale": "Stereo strength multiplier applied to the physical IPD; lower values reduce parallax, higher values increase depth",
        "tooltip_device": "Inference device",
        "tooltip_capture_tool": "Capture backend",
        "tooltip_run_mode": "Output mode",
        "tooltip_display_mode": "Stereo display format",
        "tooltip_local_vsync": "Synchronize the local viewer to the display refresh rate",
        "tooltip_target_fps": "Override internal frame pacing. Auto uses the detected display refresh rate.",
        "tooltip_ctrl_model": "Controller model",
        "tooltip_env_model": "Room environment model",
        "tooltip_capture_mode": "Source: monitor or window",
        "tooltip_monitor": "Input monitor",
        "tooltip_stereo_monitor": "Stereo output monitor",
        "tooltip_lang": "Interface language",
        "tooltip_theme": "Color theme",
        "tooltip_stream_quality": "Encode quality",
        "tooltip_stream_proto": "Streaming protocol",
        "tooltip_audio": "Stereo mix device",
        "tooltip_stream_port": "Server port",
        "tooltip_stream_key": "Stream key",
        "tooltip_crf": "Quality factor (0-51)",
        "tooltip_audio_delay": "Audio offset (s)",
        "err_crf": "CRF must be between 0-51",
        "err_audio_delay": "Audio Delay must be between -10 and 10",
        "err_stream_key": "Stream Key can only contain letters, digits, underscore, hyphen, max 64 chars",
        "err_start_failed": "Start failed: {}",
        "esc_stop": "Hold ESC 3s — stopping!",
        "exited_with_code": "Exited with code {}",
        "failed_save_yaml": "Failed to save YAML: {}",
        "stereo_parameters_saved": "Stereo parameters saved",
        "invalid_url_scheme": "Invalid URL scheme: {}",
        "error_preview": "Failed to preview: {}",
        "url_copied": "URL copied to clipboard",
    },
    "CN": {
        "Monitor": "输入屏幕",
        "Window": "输入窗口",
        "Refresh": "刷新",
        "Show FPS": "显示帧率",
        "IPD (m):": "瞳距 (mm):",
        "Stereo Scale:": "立体缩放:",
        "Convergence:": "会聚点:",
        "Display Mode:": "显示模式:",
        "Depth Model:": "深度模型:",
        "Depth Strength:": "深度强度:",
        "Depth Quick:": "深度选项:",
        "Soft": "柔和",
        "Standard": "标准",
        "Enhanced": "增强",
        "Stereo Mode:": "立体模式:",
        "Cinema": "电影 / 偏均衡",
        "Game / Low Latency": "游戏 / 低延迟",
        "Still Image / HQ": "图片 / 高质量",
        "Debug / Export": "调试 / 导出",
        "Synthetic View:": "立体质量:",
        "fast": "最低",
        "fast_plus": "中等",
        "quality_4k": "较高",
        "hq_4k": "最高",
        "Max Shift Ratio:": "位移比例:",
        "Temporal Strength:": "时域强度:",
        "Temporal": "时域稳定",
        "Scene Threshold:": "场景阈值:",
        "Reset Cooldown:": "重置冷却:",
        "Auto Scene Reset": "自动场景重置",
        "Edge Dilation:": "边缘扩张:",
        "Edge Threshold:": "边缘阈值:",
        "On": "开启",
        "Anaglyph:": "红蓝模式:",
        "Cross Eyed": "交叉眼",
        "Advanced Stereo": "显示高级立体参数",
        "Advanced Device Options": "高级选项",
        "Depth Resolution:": "深度分辨率:",
        "Anti-aliasing:": "抗锯齿:",
        "Foreground Scale:": "前景缩放:",
        "FP16": "FP16",
        "Inference Acceleration:": "推理加速:",
        "Recompile TensorRT": "重译TensorRT",
        "Recompile CoreML": "重译CoreML",
        "Recompile OpenVINO": "重译OpenVINO",
        "Stop": "停止",
        "Computing Device:": "计算设备:",
        "Reset": "重置",
        "Run": "运行",
        "Set Language:": "设置语言:",
        "Error": "错误",
        "Warning": "警告",
        "Saved": "运行Desktop2Stereo",
        "PyYAML not installed, cannot save YAML file.": "未安装PyYAML，无法保存YAML文件。",
        "Settings saved to settings.yaml": "设置已保存到 settings.yaml",
        "Failed to save settings.yaml:": "保存 settings.yaml 失败：",
        "Could not retrieve monitor list.\nFalling back to indexes 1 and 2.": "无法获取显示器列表。\n回退到索引1和2。",
        "Loaded settings.yaml at startup": "启动时已加载 settings.yaml",
        "Running": "运行中...（长按ESC 3秒停止）",
        "Stopped": "已停止。",
        "Countdown": "设置已保存到 settings.yaml，启动中...",
        "A thread already running!": "一个进程已经运行！",
        "No windows found": "未找到窗口",
        "Selected input window:": "已选择输入窗口:",
        "Selected input monitor:": "已选择输入显示器 :",
        "Run Mode:": "运行模式:",
        "Local Viewer": "本地查看",
        "Legacy Streamer": "旧网络推流",
        "MJPEG Streamer": "MJPEG推流",
        "RTMP Streamer": "RTMP推流",
        "Stream Protocol:": "流协议:",
        "Stream Key": "推流密钥:",
        "Stereo Mix": "混音设备:",
        "CRF": "恒定质量:",
        "Audio Delay": "音频延迟 (秒):",
        "system": "系统",
        "blue": "蓝色",
        "green": "绿色",
        "red": "红色",
        "purple": "紫色",
        "orange": "橙色",
        "teal": "青色",
        "pink": "粉色",
        "grey": "灰色",
        "Lossless Scaling Support": "小黄鸭",
        "3D Monitor": "3D显示器",
        "OpenXR Link": "OpenXR串流",
        "Local VSync": "本地垂直同步",
        "Capture FPS:": "捕获帧率:",
        "Upscaler:": "画面增强:",
        "Upscaler Sharpness:": "增强锐度:",
        "Auto": "自动",
        "Off": "关闭",
        "Streamer Port:": "推流端口:",
        "Streamer URL": "推流网址:",
        "Preview": "预览",
        "Stream Quality:": "推流质量:",
        "Host": "主机:",
        "Invalid port number (1-65535)": "端口号无效 (必须介于1-65535之间)",
        "Invalid port number": "端口必须是数字",
        "Please select a window before running in Window capture mode": "请在窗口捕获模式下选择一个窗口再运行",
        "The selected window no longer exists. Please refresh and select a valid window.": "所选窗口已不存在。请刷新并选择一个有效的窗口。",
        "Failed to stop process on exit:": "退出时停止进程失败：",
        "Failed to stop process:": "停止进程失败：",
        "Failed to run process:": "运行进程失败：",
        "Failed to load settings.yaml:": "加载 settings.yaml 失败：",
        "Opening URL in browser": "正在浏览器中打开网址",
        "Controller:": "手柄模型：",
        "Environment:": "房间模型：",
        "Capture Tool:": "捕获工具:",
        "Fill 16:9": "16:9",
        "Fix Viewer Aspect": "锁定比例",
        "Stereo Output:": "立体输出:",
        "Theme:": "主题颜色:",
        "DesktopDuplication selected: Window capture mode disabled.": "已选择DesktopDuplication：窗口捕获模式已禁用。",
        "torch.compile": "torch.compile",
        "TensorRT": "TensorRT",
        "CoreML": "CoreML",
        "OpenVINO": "OpenVINO",
        "tooltip_window": "选择要捕获的窗口",
        "tooltip_depth_model": "选择深度估计模型",
        "tooltip_depth_res": "深度图分辨率",
        "tooltip_convergence": "立体会聚点",
        "tooltip_depth_strength": "深度效果强度",
        "tooltip_depth_quick": "给普通用户使用的固定深度档位：柔和、标准、增强",
        "tooltip_stereo_preset": "自动模式会按场景信号加权切换；手动模式固定为电影、游戏、图片或调试行为",
        "tooltip_stereo_quality": "立体合成后端：fast 低延迟，quality_4k 均衡，hq_4k 偏图片高质量",
        "tooltip_max_shift": "水平位移占画面宽度的比例；越高立体分离越强",
        "tooltip_temporal_strength": "时域平滑强度；越高越稳定，但可能增加拖影或延迟",
        "tooltip_temporal": "启用帧间时域稳定",
        "tooltip_scene_reset": "场景变化重置阈值；越低越容易触发重置",
        "tooltip_reset_cooldown": "两次自动时域重置之间的最小帧数间隔",
        "tooltip_auto_scene_reset": "检测到场景切换时自动重置时域历史",
        "tooltip_edge_dilation": "扩张深度边缘区域，用于遮挡和补洞处理",
        "tooltip_edge_threshold": "深度边缘检测敏感度；越低检测到的边缘越多",
        "tooltip_anaglyph": "显示模式为红蓝/补色时使用的颜色组合",
        "tooltip_cross_eyed": "交换左右眼，用于交叉眼观看",
        "tooltip_advanced_stereo": "显示专家级立体和运行时参数；普通使用建议保持关闭。",
        "tooltip_advanced_device_options": "显示捕获帧率、本地垂直同步、画面增强和增强锐度。",
        "tooltip_foreground_scale": "前景缩放比例",
        "tooltip_antialiasing": "抗锯齿级别",
        "tooltip_ipd": "瞳距（毫米）",
        "tooltip_stereo_scale": "作用在物理 IPD 上的立体强度倍率；数值越低视差越小，数值越高深度越强",
        "tooltip_device": "计算设备",
        "tooltip_capture_tool": "捕获后端",
        "tooltip_run_mode": "输出模式",
        "tooltip_display_mode": "立体显示格式",
        "tooltip_local_vsync": "将本地查看窗口同步到显示器刷新率，关闭可用于帧率对比测试",
        "tooltip_target_fps": "覆盖内部帧率节流；自动使用检测到的显示器刷新率",
        "tooltip_ctrl_model": "手柄型号",
        "tooltip_env_model": "房间环境模型",
        "tooltip_capture_mode": "捕获源：屏幕或窗口",
        "tooltip_monitor": "输入显示器",
        "tooltip_stereo_monitor": "立体输出显示器",
        "tooltip_lang": "界面语言",
        "tooltip_theme": "主题颜色",
        "tooltip_stream_quality": "编码质量",
        "tooltip_stream_proto": "推流协议",
        "tooltip_audio": "混音设备",
        "tooltip_stream_port": "推流端口",
        "tooltip_stream_key": "推流密钥",
        "tooltip_crf": "质量因子 (0-51)",
        "tooltip_audio_delay": "音频偏移（秒）",
        "err_crf": "CRF 必须是 0-51 之间的整数",
        "err_audio_delay": "Audio Delay 必须是 -10 到 10 之间的数值",
        "err_stream_key": "Stream Key 只能包含字母、数字、下划线和连字符，最长 64 字符",
        "err_start_failed": "启动失败: {}",
        "esc_stop": "长按ESC 3秒停止",
        "exited_with_code": "退出码 {}",
        "failed_save_yaml": "保存 YAML 失败: {}",
        "stereo_parameters_saved": "立体参数已保存",
        "invalid_url_scheme": "无效 URL 协议: {}",
        "error_preview": "打开浏览器失败: {}",
        "url_copied": "已复制网址到剪贴板",
    }
}

LOCALE_ALIASES = MappingProxyType({
    "EN": "EN",
    "EN_US": "EN",
    "EN-US": "EN",
    "CN": "CN",
    "ZH": "CN",
    "ZH_CN": "CN",
    "ZH-CN": "CN",
    "ZH_HANS": "CN",
    "ZH-HANS": "CN",
})

SUPPORTED_LOCALES = tuple(MESSAGE_CATALOGS.keys())
UI_MESSAGES = MESSAGE_CATALOGS
UI_TEXTS = UI_MESSAGES


class CatalogTranslation(gettext.NullTranslations):
    def __init__(self, catalog, fallback=None):
        super().__init__()
        self._catalog = catalog
        self._fallback = fallback

    def gettext(self, message):
        if message in self._catalog:
            return self._catalog[message]
        if self._fallback is not None:
            return self._fallback.gettext(message)
        return message

    def ngettext(self, msgid1, msgid2, n):
        message = msgid1 if n == 1 else msgid2
        return self.gettext(message)


_EN_TRANSLATION = CatalogTranslation(MESSAGE_CATALOGS["EN"])
_LOCALE_TRANSLATIONS = {
    lang: CatalogTranslation(catalog, fallback=_EN_TRANSLATION if lang != "EN" else None)
    for lang, catalog in MESSAGE_CATALOGS.items()
}


def normalize_locale(locale):
    key = str(locale or DEFAULT_LOCALE).replace(" ", "_").upper()
    return LOCALE_ALIASES.get(key, key if key in MESSAGE_CATALOGS else DEFAULT_LOCALE)


def is_supported_locale(locale):
    return normalize_locale(locale) in MESSAGE_CATALOGS


def get_translation(locale=DEFAULT_LOCALE):
    return _LOCALE_TRANSLATIONS[normalize_locale(locale)]


def get_messages(locale=DEFAULT_LOCALE):
    return MESSAGE_CATALOGS[normalize_locale(locale)]


def gettext_for(locale, message):
    return get_translation(locale).gettext(message)

STEREO_QUALITY_KEYS = ("fast", "fast_plus", "quality_4k", "hq_4k")


def stereo_quality_options(locale=DEFAULT_LOCALE):
    messages = get_messages(locale)
    return [messages[key] for key in STEREO_QUALITY_KEYS]


def stereo_quality_to_display(value, locale=DEFAULT_LOCALE):
    key = str(value or "quality_4k")
    messages = get_messages(locale)
    return messages.get(key, messages["quality_4k"])

def display_to_stereo_quality(value):
    text = str(value or "")
    for key in STEREO_QUALITY_KEYS:
        if text == key:
            return key
        for locale in SUPPORTED_LOCALES:
            if text == get_messages(locale).get(key):
                return key
    return "quality_4k"

