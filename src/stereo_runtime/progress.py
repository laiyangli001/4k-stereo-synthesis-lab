from __future__ import annotations

import os
import sys
import threading
import time
from contextlib import contextmanager

from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)


def supports_live_progress(stream=None) -> bool:
    if os.environ.get("D2S_FORCE_RICH_PROGRESS", "").strip().lower() in {"1", "true", "yes", "on"}:
        return True
    stream = sys.stdout if stream is None else stream
    try:
        return bool(stream.isatty())
    except Exception:
        return False


def _forced_live_progress() -> bool:
    return os.environ.get("D2S_FORCE_RICH_PROGRESS", "").strip().lower() in {"1", "true", "yes", "on"}


def progress_write(message: str, *, leading_newline: bool = False) -> None:
    text = str(message)
    if leading_newline:
        text = "\n" + text
    Console(file=sys.stdout, force_terminal=False, color_system=None).print(text)


class _NullProgress:
    def __init__(self, desc: str = "") -> None:
        self.desc = desc
        self.n = 0
        self.total = None

    def update(self, amount=1):
        self.n += amount

    def set_description(self, desc):
        self.desc = str(desc or "")

    def set_postfix_str(self, _value, refresh=True):
        return None

    def refresh(self):
        return None

    def close(self):
        return None


class _StageLogProgress(_NullProgress):
    def __init__(self, desc: str, total: int) -> None:
        super().__init__(desc)
        self.total = total
        self._last_postfix = None

    def update(self, amount=1):
        self.n += amount

    def set_postfix_str(self, value, refresh=True):
        postfix = str(value or "").strip()
        if postfix and postfix != self._last_postfix:
            print(f"[Main] {self.desc}: {postfix}", flush=True)
            self._last_postfix = postfix


def create_download_progress(*, console=None, transient=False):
    return Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        console=console or Console(file=sys.stdout, force_terminal=False, color_system=None),
        transient=transient,
    )


def create_activity_progress(*, console=None, transient=False):
    return Progress(
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console or Console(file=sys.stdout, force_terminal=False, color_system=None),
        transient=transient,
    )


class _RichProgressAdapter:
    """Bridge external update callbacks to Rich Progress."""

    def __init__(self, *args, **kwargs):
        self.total = kwargs.get("total", args[0] if args else None)
        self.desc = str(kwargs.get("desc") or "")
        self.n = 0
        self.leave = bool(kwargs.get("leave", False))
        self._progress = create_download_progress(transient=not self.leave) if self.total is not None else create_activity_progress(transient=True)
        self._task = self._progress.add_task(self.desc, total=self.total)
        self._started = False

    def __enter__(self):
        self._start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    @property
    def fp(self):
        return sys.stdout

    def update(self, amount=1):
        self._start()
        self.n += amount
        self._progress.update(self._task, completed=self.n)
        return None

    def set_description(self, desc, refresh=True):
        self._start()
        self.desc = str(desc or "")
        self._progress.update(self._task, description=self.desc)
        return None

    def set_postfix_str(self, value, refresh=True):
        self._start()
        postfix = str(value or "").strip()
        if postfix:
            self.desc = f"{self.desc}: {postfix}" if self.desc else postfix
            self._progress.update(self._task, description=self.desc)
        return None

    def refresh(self):
        self._start()
        self._progress.refresh()
        return None

    def close(self):
        if self._started:
            self._progress.stop()
        return None

    def _start(self):
        if not self._started:
            self._progress.start()
            self._started = True


def make_rich_progress(*args, **kwargs):
    return _RichProgressAdapter(*args, **kwargs)


@contextmanager
def activity_progress(desc: str, *, interval_s: float = 0.2):
    """Show a live activity line for long operations without real percent callbacks."""
    if not supports_live_progress():
        print(f"[Main] {desc}...", flush=True)
        yield _NullProgress(desc)
        return

    stop_event = threading.Event()
    bar = make_rich_progress(
        total=None,
        desc=f"[Main] {desc}",
        bar_format="{desc} | {elapsed} elapsed",
        leave=False,
    )

    def _pulse() -> None:
        while not stop_event.wait(interval_s):
            bar.update(1)

    thread = threading.Thread(target=_pulse, name=f"Progress:{desc}", daemon=True)
    thread.start()
    try:
        yield bar
    finally:
        stop_event.set()
        thread.join(timeout=1.0)
        bar.close()


@contextmanager
def stage_progress(desc: str, total: int):
    if not supports_live_progress():
        yield _StageLogProgress(desc, total)
        return

    bar = make_rich_progress(
        total=total,
        desc=f"[Main] {desc}",
        bar_format="{desc} [{bar}] {n_fmt}/{total_fmt} {postfix}",
        leave=False,
    )
    try:
        yield bar
    finally:
        bar.close()

@contextmanager
def file_size_progress(desc: str, path, *, total_bytes: int, interval_s: float = 0.2):
    """Track a file's byte size as an approximate progress bar."""
    from pathlib import Path

    target = Path(path)
    total = max(1, int(total_bytes or 1))
    if not supports_live_progress():
        started = time.perf_counter()
        print(f"[Main] {desc}...", flush=True)
        try:
            yield _NullProgress(desc)
        finally:
            try:
                size = target.stat().st_size if target.exists() else 0
            except OSError:
                size = 0
            size = min(size, total)
            percent = (size / total) * 100.0 if total else 100.0
            print(f"[Main] {desc} {percent:6.2f}%  {size}/{total}  {time.perf_counter() - started:0.1f}s", flush=True)
        return

    stop_event = threading.Event()
    bar = make_rich_progress(
        total=total,
        desc=f"[Main] {desc}",
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        bar_format="{desc} {bar} {percentage:6.2f}%  {n_fmt}/{total_fmt}  {elapsed}",
        leave=True,
    )
    last_size = 0

    def _poll() -> None:
        nonlocal last_size
        while not stop_event.wait(interval_s):
            try:
                size = target.stat().st_size if target.exists() else 0
            except OSError:
                size = last_size
            size = min(size, total)
            if size > last_size:
                bar.update(size - last_size)
                last_size = size
            else:
                bar.refresh()

    thread = threading.Thread(target=_poll, name=f"Progress:{desc}", daemon=True)
    thread.start()
    try:
        yield bar
    finally:
        stop_event.set()
        thread.join(timeout=1.0)
        try:
            size = target.stat().st_size if target.exists() else last_size
        except OSError:
            size = last_size
        size = min(size, total)
        if size > last_size:
            bar.update(size - last_size)
            last_size = size
        bar.close()


def write_bytes_with_progress(path, data, desc: str, *, chunk_size: int = 8 * 1024 * 1024):
    """Write bytes with a real byte-count progress bar."""
    from pathlib import Path

    target = Path(path)
    try:
        blob = memoryview(data)
    except TypeError:
        blob = memoryview(bytes(data))
    total = len(blob)
    if not supports_live_progress():
        with target.open("wb") as file:
            for offset in range(0, total, chunk_size):
                file.write(blob[offset:offset + chunk_size])
        print(f"[Main] {desc} 100.00%  {total}/{total}", flush=True)
        return

    bar = make_rich_progress(
        total=total,
        desc=f"[Main] {desc}",
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        bar_format="{desc} {bar} {percentage:6.2f}%  {n_fmt}/{total_fmt}",
        leave=True,
    )
    try:
        with target.open("wb") as file:
            for offset in range(0, total, chunk_size):
                chunk = blob[offset:offset + chunk_size]
                file.write(chunk)
                bar.update(len(chunk))
    finally:
        bar.close()
