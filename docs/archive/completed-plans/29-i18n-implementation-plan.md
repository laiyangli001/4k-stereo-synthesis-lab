# i18n / l10n 实现方案 — Desktop2Stereo

## 现状分析

### 已有基础（存量）
- `src/gui/localization.py` 定义了 `UI_MESSAGES = {"EN": {...}, "CN": {...}}` 字典
- GUI 中已使用 `UI_MESSAGES[self.locale]["key"]` 模式做界面翻译
- 语言切换入口：GUI 顶部的 `lang_dd` 下拉框
- locale 存储在 `self.locale`，默认 `"EN"`，保存到 `settings.yaml` 的 `Language` 字段

### 缺口（需要补充）
- 子进程（main.py）没有访问 `UI_MESSAGES` 的能力（它不 import GUI 模块）
- 没有任何运行时日志/状态消息的翻译机制
- 翻译数据只在 GUI 进程内存中，子进程不知道当前语言
- 没有统一的 `t(key, locale)` 函数，各处直接引用 `UI_MESSAGES` 字典

---

## 架构设计

```
src/utils/i18n.py          ← 共享翻译模块（GUI + 子进程共同引用）
src/gui/localization.py    ← 从 utils.i18n 导入基础数据，合并 GUI 专属项
src/main.py / src/stereo_runtime/  ← 子进程代码通过 from utils.i18n import t, status_log 使用
```

### locale 传递方式

**子进程通过环境变量获取 locale**（不依赖 settings.yaml 解析）：

```
GUI 进程（process.py）:
  env["DESKTOP2STEREO_LOCALE"] = self.locale
  subprocess.Popen(..., env=env)

子进程（main.py）:
  _locale = os.environ.get("DESKTOP2STEREO_LOCALE", "EN")
```

为什么不用 settings.yaml？避免循环导入依赖（`utils.settings` 可能依赖 `utils.i18n`），且环境变量更轻量、零解析开销。`get_locale_from_settings()` 保留作为兜底降级。

---

## 核心代码

### 1. 新建 `src/utils/i18n.py`

```python
"""Shared i18n module for Desktop2Stereo. Accessible from both GUI and subprocess.
MESSAGES is read-only — never modify at runtime.
"""
from __future__ import annotations
import logging
import os
from typing import Any

# ── 翻译数据（只读常量） ──

MESSAGES: dict[str, dict[str, str]] = {
    "EN": {
        # ── 关键运行状态（status_log 用） ──
        "Preparing environment": "📦 Preparing environment...",
        "Preparing Flet package": "📦 Preparing Flet GUI package...",
        "Checking model cache": "🔍 Checking model cache...",
        "Downloading model": "⬇️ Downloading AI model {model} from {endpoint}...",
        "Downloading model (first time)": "⬇️ Downloading AI model {model} (first time, may take several minutes)...",
        "Exporting ONNX": "⚙️ Exporting ONNX file: {filename}",
        "Building TensorRT engine": "🔧 Building TensorRT engine (this may take a while)...",
        "Starting capture": "🚀 Starting capture...",
        "Loading model": "📤 Loading model {model}...",
        "Running": "✅ Running",
        "Stopped": "⏹ Stopped",
        "Error occurred": "❌ Error occurred",
        "Fatal error": "❌ Fatal error: {error}",
        "Shutting down": "🛑 Shutting down...",
        "Runtime preparation": "🔧 Runtime preparation: checking {component}",
        "Ready": "✅ Ready",

        # ── 子进程输出标记 ──
        "capture_started": "Capture started (monitor #{index})",
        "capture_stopped": "Capture stopped",
        "depth_loaded": "Depth model loaded: {model}",
        "trt_loaded": "TensorRT engine loaded: {path}",

        # ── 错误/用户提示 ──
        "bug_report_copied": "📋 Bug report copied to clipboard!",
        "no_pyperclip": "pyperclip not installed",
    },
    "CN": {
        "Preparing environment": "📦 正在准备运行环境...",
        "Preparing Flet package": "📦 正在准备 Flet GUI 包...",
        "Checking model cache": "🔍 正在检查模型缓存...",
        "Downloading model": "⬇️ 正在下载AI模型 {model}（来自 {endpoint}）...",
        "Downloading model (first time)": "⬇️ 首次下载AI模型 {model}，可能需要几分钟...",
        "Exporting ONNX": "⚙️ 正在导出ONNX文件：{filename}",
        "Building TensorRT engine": "🔧 正在编译TensorRT引擎（可能需要较长时间）...",
        "Starting capture": "🚀 正在启动采集...",
        "Loading model": "📤 正在加载模型 {model}...",
        "Running": "✅ 运行中",
        "Stopped": "⏹ 已停止",
        "Error occurred": "❌ 出现异常",
        "Fatal error": "❌ 致命错误：{error}",
        "Shutting down": "🛑 正在关闭...",
        "Runtime preparation": "🔧 运行准备：检查 {component}",
        "Ready": "✅ 准备就绪",

        "capture_started": "采集已启动（监视器 #{index}）",
        "capture_stopped": "采集已停止",
        "depth_loaded": "深度模型已加载：{model}",
        "trt_loaded": "TensorRT 引擎已加载：{path}",

        "bug_report_copied": "📋 异常报告已复制到剪贴板！",
        "no_pyperclip": "未安装 pyperclip",
    },
}

_SUPPORTED_LOCALES = ("EN", "CN")
_DEFAULT_LOCALE = "EN"


def t(key: str, locale: str | None = None, **kwargs: Any) -> str:
    """Translate key to locale, with optional format args.

    降级链: 目标 locale → EN → 原始 key
    格式化失败时，附加未格式化参数方便调试，避免静默漏参数。
    """
    locale = locale or _DEFAULT_LOCALE
    msg_map = MESSAGES.get(locale, MESSAGES[_DEFAULT_LOCALE])
    template = msg_map.get(key)
    if template is None:
        template = MESSAGES[_DEFAULT_LOCALE].get(key, key)
    if kwargs:
        try:
            return template.format(**kwargs)
        except (KeyError, ValueError):
            return f"{template} [unformatted: {kwargs}]"
    return template


def supported_locales() -> tuple[str, ...]:
    return _SUPPORTED_LOCALES


def is_supported_locale(code: str) -> bool:
    return code in _SUPPORTED_LOCALES


def _resolve_locale() -> str:
    """确定当前 locale：环境变量 → 兜底 EN。"""
    return os.environ.get("DESKTOP2STEREO_LOCALE", _DEFAULT_LOCALE)


def status_log(key: str, level: int = logging.INFO, **kwargs: Any) -> None:
    """记录翻译后的状态消息。自动获取 locale，避免调用方传参。

    level 参数支持灵活级别：错误场景用 logging.ERROR，面板会高亮。
    惰性获取 logger，避免模块级导入副作用。
    """
    locale = _resolve_locale()
    msg = t(key, locale, **kwargs)
    logging.getLogger("status").log(level, msg)


def locale_for_gui(locale: str | None) -> str:
    """GUI 显示用辅助函数：验证 locale 是否受支持。"""
    return locale if locale in _SUPPORTED_LOCALES else _DEFAULT_LOCALE
```

### 2. 改造 `src/gui/localization.py`

```python
"""GUI localization — runtime labels only. Re-exports shared data from utils.i18n."""
from utils.i18n import MESSAGES as _BASE, is_supported_locale

# GUI 专属翻译（不与 _BASE 共享命名空间，以减少意外覆盖风险）
_GUI_ONLY: dict[str, dict[str, str]] = {
    "EN": {
        "Depth Model:": "Depth Model:",
        "Run Mode:": "Run Mode:",
        "Run": "Run",
        "Stop": "Stop",
        # ... 保留所有现有 GUI 翻译项
    },
    "CN": {
        "Depth Model:": "深度模型：",
        "Run Mode:": "运行模式：",
        "Run": "启动",
        "Stop": "停止",
        # ... 保留所有现有 GUI 翻译项
    },
}

# 合并：_GUI_ONLY 覆盖 _BASE 同名键（如 "Running"/"Stopped"）
UI_MESSAGES: dict[str, dict[str, str]] = {}
for locale in ("EN", "CN"):
    merged = {}
    merged.update(_BASE.get(locale, {}))
    merged.update(_GUI_ONLY.get(locale, {}))
    UI_MESSAGES[locale] = merged
```

---

## 使用方式

### GUI 进程（`src/gui/process.py`）

```python
# 静态 UI 文本（不变）
self.run_btn.content.value = UI_MESSAGES[self.locale]["Run"]

# 运行时状态消息（通过 status_log）
from utils.i18n import status_log

# 内部包装，保持风格统一
def _status(self, key, **kwargs):
    import os
    os.environ["DESKTOP2STEREO_LOCALE"] = self.locale  # 确保环境变量
    status_log(key, **kwargs)
```

### 子进程（`src/main.py`）

```python
from utils.i18n import status_log

# 环境变量由 GUI 在启动子进程时设置
# os.environ["DESKTOP2STEREO_LOCALE"] 已可用

status_log("Starting capture")
status_log("Loading model", model=model_name)
status_log("Downloading model (first time)", model=model_id)

# 错误场景指定 level
status_log("Fatal error", level=logging.ERROR, error=str(exc))
```

### 子进程初始化（`_countdown_and_run` 启动子进程时）

```python
# src/gui/process.py 的 _countdown_and_run()
child_env = os.environ.copy()
child_env["DESKTOP2STEREO_LOCALE"] = self.locale  # 传递 locale
child_env["D2S_FORCE_TQDM"] = "1"
self.process = await asyncio.create_subprocess_exec(
    *child_args,
    stdout=asyncio.subprocess.PIPE,
    stderr=asyncio.subprocess.STDOUT,
    env=child_env,
)
```

---

## 翻译键命名约定

| 模式 | 示例 | 用途 |
|------|------|------|
| `Pascal 短句` | `"Downloading model"` | 运行状态消息（status_log） |
| `snake_case` | `"bug_report_copied"` | 用户提示/错误消息 |
| `"标签名:"` | `"Depth Model:"` | GUI 界面标签（localization.py） |

- 键名用英文，不包含运行时参数
- 需要插入变量的键，用 `{placeholder}` 语法
- **技术细节日志不经过翻译**，直接 `logger.debug("原始英文")`

---

## 完整性校验

建议在测试套件中添加：

```python
def test_locale_completeness():
    """所有 locale 必须包含相同键集合，避免翻译遗漏。"""
    en_keys = set(MESSAGES["EN"].keys())
    for loc in MESSAGES:
        missing = en_keys - set(MESSAGES[loc].keys())
        extra = set(MESSAGES[loc].keys()) - en_keys
        if missing:
            raise ValueError(f"Locale {loc} missing keys: {missing}")
        if extra:
            # warning，不阻断——GUI 专属键可能在 EN 中不存在
            print(f"Warning: {loc} has extra keys: {extra}")
```

---

## 文件清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `src/utils/i18n.py` | **新建** | `t()` + `status_log()` + `_resolve_locale()` — 全部模块级导入无副作用 |
| `src/gui/localization.py` | **重构** | 改为从 `utils.i18n` 导入基础数据，合并 GUI 专属项 |
| `src/gui/process.py` | 修改 | `_countdown_and_run()` 设置 `DESKTOP2STEREO_LOCALE` 环境变量 |
| `src/gui/builders.py` | 不改 | 只负责 UI 构造 |
| `src/main.py` | 小改 | `status_log()` 替换 `progress_write("[Main] ...")` |
| `src/stereo_runtime/*.py` | 小改 | 关键进度消息用 `status_log` 包裹 |

---

## 不修改的文件

- `src/gui/config_mgr.py` — 不涉及消息输出
- `src/gui/handlers.py` — 替换 print 即可，不涉及 i18n 逻辑
- `src/stereo_runtime/progress.py` — 进度条机制不受影响

---

## 验证方法

| 验证项 | 方法 |
|--------|------|
| EN 模式状态消息 | English 模式下 Run，状态消息是否英文带 emoji |
| CN 模式状态消息 | 简体中文模式下 Run，状态消息是否中文带 emoji |
| 技术细节保持英文 | 面板中 DEBUG 日志保持英文不翻译 |
| 子进程跟随语言 | 子进程下载/导出/编译消息是否跟随 GUI 语言 |
| GUI 界面翻译 | 语言切换后，所有 UI 标签正常切换 |
| 缺失键降级 | 调用 `t("nonexistent_key")` 返回 `"nonexistent_key"` |
| 格式化失败降级 | 调用 `t("Downloading model")` 无 kwargs，返回模板原文 |
| locale 环境变量 | 检查子进程启动时 `DESKTOP2STEREO_LOCALE` 是否正确设置 |
| 语言动态切换 | GUI 语言切换后，新产生的状态消息用新语言（旧消息保持原样） |
