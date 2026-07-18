# 计划：logging + Flet 原生日志和进度面板

## Context

- CMD 输出黑白杂乱，无层次
- GUI 只有单行状态栏，看不到完整日志
- `_TeeStream` 拦截 stdout/stderr 但无结构分级
- 子进程输出无颜色无层次
- **需要给普通用户看的运行状态窗口**，让用户知道程序卡在哪一步
- **出异常时一键反馈**，自动收集日志+系统信息
- **关键步骤支持中英文**，技术细节用英文原文

目标：Python logging 分级 + Flet 原生日志着色 + Flet 原生运行进度面板

---

## 全量 print() 普查

### GUI 进程（需要 logger.info/warning/error）

| 位置 | 当前代码 | 建议级别 | 备注 |
|------|---------|---------|------|
| `gui/process.py` | `print("[Main] Initializing...")` | INFO | 关键步骤 |
| `gui/process.py` | `print("[Main] Stopped")` | INFO | 关键步骤 |
| `gui/handlers.py` | `print(f"[Warning] _safe_update failed: {e}")` | WARNING | 技术细节 |
| `gui/flet_runtime.py` | `print("[Flet GUI] ...")` x 5 | INFO/ERROR | 关键步骤（首次运行解压） |

### 子进程（通过管道捕获，全部用 child_logger）

| 位置 | 当前代码 | 建议级别 | 备注 |
|------|---------|---------|------|
| `main.py` | `progress_write("[Main] ...")` | status | **关键步骤** |
| `stereo_runtime/pipeline.py` | `print("[RuntimePipeline] ...")` | INFO/ERROR | 运行状态 |
| `capture/backends/windows_capture_event.py` | `print("[WindowsCaptureCUDA] ...")` | DEBUG | 调测用 |
| `capture/backends/windows_capture_event.py` | `print("[keyboard] ...")` | ERROR | 异常 |
| `viewer/window_utils.py` | `print("Linux window listing error:", e)` | WARNING | 异常 |
| `stereo_runtime/tensorrt_native.py` | 待查 | INFO | TRT 编译进度 |

---

## 关键步骤（i18n + 进度条）

### 阶段① — 首次运行环境准备
```
[Flet GUI] 检测到系统: {os}
[Flet GUI] 正在准备 Flet GUI 包...
[Flet GUI] 未找到匹配的 Flet GUI 包，请开启 VPN 重试
```
→ 用 `activity_progress` 显示解压进度
→ i18n: `UI_MESSAGES[locale]["Preparing Flet package..."]`

### 阶段② — 模型下载
```
[Main] Runtime preparation: checking depth model {model_id}
[Main] Checking local depth model cache...
[Main] Depth model cache hit:
[Main] Depth model not found in local cache; preparing download from {endpoint}
[Main] Preparing depth model download... First download may take several minutes.
[Main] Download {model_id} [{bar}] {percent}% ...
```
→ 使用 `DownloadProgress` 输出结构化 `[D2S_PROGRESS]` 事件，由 GUI 用 `ft.ProgressBar` 展示
→ 但消息都是英文，需要 i18n 包裹关键消息
→ i18n: `UI_MESSAGES[locale]["Downloading model..."]`

### 阶段③ — ONNX 导出
```
[Main] Loading model for ONNX export: {model_id}...
[Main] Probing ONNX export dtype: {dtype}...
[Main] Exporting ONNX: {filename}...
```
→ **已经有 `activity_progress`**
→ i18n: `UI_MESSAGES[locale]["Exporting ONNX..."]`

### 阶段④ — TensorRT 引擎编译
```
(prepare_model_artifacts 内)
[Main] Runtime preparation ████████░░ 3/4  building TensorRT engine
```
→ **已经有 `stage_progress`**
→ 编译过程可能 5-30 分钟，必须展示进度
→ i18n: `UI_MESSAGES[locale]["Building TensorRT engine..."]`

### 阶段⑤ — 每日运行时
```
[Main] Initializing Desktop2Stereo Local Viewer...
[Main] Runtime preparation ████████░░ 4/4  ready
[Main] Starting capture...
▶ 运行中...
[Main] Stopped
```
→ i18n: `UI_MESSAGES[locale]["Starting..."]`
→ 运行时显示 FPS 状态

### 阶段⑥ — 错误状态
```
[process_runtime_loop] Fatal: CUDA out of memory
[process_runtime_loop] Error: FileNotFoundError: model not found
[Main] Keyboard interrupt received, shutting down...
```
→ ERROR 级别
→ 触发"反馈异常"按钮显示
→ 错误消息需要 i18n

---

## 实施步骤

### Step 1：使用 Flet 原生控件
```bash
不引入终端样式依赖；GUI 样式由 Flet 控件负责。
```

### Step 2：新建 `src/gui/log_handler.py`

```python
import logging
import queue
from collections import deque

class GuiLogHandler(logging.Handler):
    """结构化日志入队：(levelno, asctime, formatted)"""

    def __init__(self, maxlen: int = 2000):
        super().__init__()
        self.queue = queue.Queue()
        self.cache = deque(maxlen=maxlen)

    def emit(self, record):
        try:
            asctime = self.formatTime(record, self.formatter.datefmt or "%H:%M:%S")
            formatted = self.format(record)
            self.queue.put((record.levelno, asctime, formatted))
            self.cache.append((record.levelno, asctime, formatted))
        except Exception:
            self.handleError(record)
```

### Step 3：重构 logging 配置

移除 `_TeeStream`，改为三路 handler：

| Handler | 目的地 | 级别 |
|---------|--------|------|
| `StreamHandler` | CMD 控制台（纯文本） | DEBUG |
| `FileHandler` | `desktop2stereo.log`（w 模式） | DEBUG |
| `GuiLogHandler` | 队列 → Flet 面板 | INFO |

```python
def _setup_logging():
    root = logging.getLogger()
    if root.handlers:  # 防重复
        return

    # StreamHandler — CMD 纯文本输出
    import logging
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(message)s",
        handlers=[logging.StreamHandler(console_stream),
                              show_path=False, omit_repeated_times=False)]
    )

    # FileHandler — 文件日志（每次启动清空）
    fh = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s", "%H:%M:%S"))
    root.addHandler(fh)

    # GuiLogHandler — GUI 面板
    gui_handler = GuiLogHandler(maxlen=2000)
    gui_handler.setLevel(logging.DEBUG)
    gui_handler.setFormatter(logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s", "%H:%M:%S"))
    root.addHandler(gui_handler)

    # StreamToLogger — 兜底漏网 print
    sys.stdout = StreamToLogger(logging.getLogger("stdout"))
```

### Step 4：替换 print() → logging

**GUI 进程**（`logger = logging.getLogger(__name__)`）：
- `gui/process.py`: 6 处 print → `logger.info/warning`
- `gui/handlers.py`: 1 处 print → `logger.warning`
- `gui/flet_runtime.py`: 5 处 print → `logger.info/error`

**子进程输出**（`_pump_child_output` → `child_logger.info`）：
- 自动识别 `Error`/`Traceback` → `child_logger.error`
- 自动识别 `Warning` → `child_logger.warning`
- 其余 → `child_logger.info`
- 保留行缓冲处理不完整行

### Step 5：logger 分层约定

| Logger | 用途 | 语言 | 面板样式 | Emoji |
|--------|------|------|---------|-------|
| `status` | 关键用户可见步骤 | i18n（中/英） | 粗体+高亮 | ✅ |
| `child` | 子进程原始输出 | 英文原文 | 默认 | ❌ |
| `gui.*` | GUI 操作日志 | 英文 | 默认 | ❌ |
| 其他 | 技术细节 | 英文 | 灰色 | ❌ |

关键步骤加 emoji 前缀（通过 `status` logger 发出）：
- `📦 正在准备运行环境...`
- `⬇️ 正在下载模型权重...`
- `⚙️ 正在导出 ONNX...`
- `🔧 正在编译 TensorRT 引擎...`
- `🚀 正在启动...`
- `✅ 运行中`
- `⏹ 已停止`
- `❌ 出现错误`

### Step 6：GUI 运行状态面板

标题栏动态更新：
```
● 正在初始化...       → 灰色 + 旋转
● 下载模型中...       → 蓝色 + 进度条
● 编译引擎中...       → 蓝色 + 进度条
● 运行中              → 绿色
⚠️ 出现异常，请查看日志 → 红色
⏹ 已停止              → 灰色
```

控件：
- 折叠/展开按钮
- 状态标题（动态更新）
- 级别过滤下拉框（ALL/DEBUG/INFO/WARNING/ERROR）
- 清除按钮
- `🐛 反馈异常` 按钮（ERROR 时闪烁）
- `ft.ListView` 日志主体（自动滚动）

颜色映射 + emoji 前缀：
```python
def _log_color(self, levelno):
    if levelno >= logging.ERROR:   return "red"
    if levelno >= logging.WARNING: return "#FFA500"
    if levelno >= logging.INFO:    return None
    return "grey"

def _log_emoji(self, logger_name, levelno):
    if logger_name == "status":
        return "🟢 " if levelno == logging.INFO else "🔴 "
    return ""
```

### Step 7：日志轮询任务

```python
async def _poll_log_queue(self):
    while not self._closed:
        if self.log_listview is None:
            await asyncio.sleep(0.1)
            continue
        count = 0
        try:
            while count < 100:  # 单次最多 100 条
                levelno, asctime, formatted = self.gui_log_handler.queue.get_nowait()
                color = self._log_color(levelno)
                self.log_listview.controls.append(
                    ft.Text(formatted, color=color, size=12))
                count += 1
        except queue.Empty:
            pass
        if count > 0:
            if len(self.log_listview.controls) > 1000:
                self.log_listview.controls = self.log_listview.controls[-500:]
            self.log_listview.auto_scroll = True
            self.log_listview.update()
        await asyncio.sleep(0.1)
```

### Step 8：i18n 关键步骤

在 `src/gui/localization.py` 添加翻译条目：
```python
UI_MESSAGES = {
    "EN": {
        "Downloading model...": "⬇️ Downloading AI model...",
        "Exporting ONNX...": "⚙️ Exporting ONNX file...",
        "Building TensorRT engine...": "🔧 Building TensorRT engine (this may take a while)...",
        "Starting capture...": "🚀 Starting capture...",
        "Running": "✅ Running",
        "Stopped": "⏹ Stopped",
        "Error occurred": "❌ Error occurred",
        "Bug report copied!": "Bug report copied to clipboard!",
    },
    "CN": {
        "Downloading model...": "⬇️ 正在下载AI模型...",
        "Exporting ONNX...": "⚙️ 正在导出ONNX文件...",
        "Building TensorRT engine...": "🔧 正在编译TensorRT引擎（可能需要较长时间）...",
        "Starting capture...": "🚀 正在启动采集...",
        "Running": "✅ 运行中",
        "Stopped": "⏹ 已停止",
        "Error occurred": "❌ 出现异常",
        "Bug report copied!": "异常报告已复制到剪贴板！",
    },
}
```

### Step 9：一键反馈异常

按钮“🐛 反馈异常”，点击后：

1. 收集系统信息（OS + 版本 + 设备 + 模型）
2. 收集最后 200 条日志
3. 收集当前配置
4. 复制到剪贴板

```python
def on_report_issue(self, e):
    import platform, datetime, json
    try:
        import pyperclip
    except ImportError:
        self.set_status("pyperclip not installed")
        return

    report = f"""=== Desktop2Stereo Bug Report ===
Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Version: {VERSION}
OS: {platform.platform()}
Device: {self.device_dd.value}
Run Mode: {self.run_mode_key}
Depth Model: {self.current_model_name}

=== Last 200 log lines ===
"""
    for _, asctime, formatted in list(self.gui_log_handler.cache)[-200:]:
        report += f"[{asctime}] {formatted.split('] ', 1)[-1]}\n"

    report += f"\n=== Config ===\n{json.dumps(self._config, indent=2, ensure_ascii=False)}"
    pyperclip.copy(report)
    self.set_status(UI_MESSAGES[self.locale]["Bug report copied!"])
```

**反馈途径建议**：初期用剪贴板复制，用户可以粘贴到 QQ群 / GitHub Issues。后期可以集成 `gh` CLI 自动创建 GitHub Issue。

---

## 涉及的文件

| 文件 | 操作 | 说明 |
|------|------|------|
| `src/gui/log_handler.py` | **新建** | `GuiLogHandler` |
| `src/gui/process.py` | **大幅修改** | 重构 logging 配置，移除 _TeeStream，改造 _pump_child_output |
| `src/gui/builders.py` | **修改** | 添加运行状态面板 UI |
| `src/gui/handlers.py` | 小改 | `print()` → `logger.warning()` |
| `src/gui/localization.py` | **修改** | 添加关键步骤 i18n 条目 |
| `src/gui/flet_runtime.py` | 小改 | `print()` → `logger.info/error()` |

---

## 验证方法

| 验证项 | 方法 |
|--------|------|
| GUI 原生着色 | 启动 GUI，观察 Flet 日志行按级别和下载状态着色 |
| GUI 状态面板 | 底部面板启动日志显示，标题栏状态动态变化 |
| 颜色编码 | DEBUG=灰、INFO=默认、WARNING=橙、ERROR=红 |
| Emoji 前缀 | status logger 消息带 🟢🔴 等 emoji |
| 异常标题栏 | 制造 ERROR，标题栏变红，反馈按钮出现 |
| 级别过滤 | 切换 Filter=WARNING，仅显示 WARNING+ |
| 子进程日志 | Run 后子进程输出显示在面板中 |
| i18n 关键步骤 | 切换 EN/CN，关键消息跟随翻译，技术细节保持英文 |
| 进度条 | 模型下载/ONNX导出时显示进度 |
| 一键反馈 | 点击"反馈异常"，粘贴到文本编辑器验证格式 |
| 文件日志 | 每次启动清空不累积，格式正确 |
| 自动裁剪 | 超 1000 行时裁剪到 500 |
| 清除功能 | 面板+缓存+队列全部清空 |
