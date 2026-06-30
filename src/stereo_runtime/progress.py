from __future__ import annotations

import os
import sys
import threading
import time
from contextlib import contextmanager


def supports_live_progress(stream=None) -> bool:
    if os.environ.get("D2S_FORCE_TQDM", "").strip().lower() in {"1", "true", "yes", "on"}:
        return True
    stream = sys.stdout if stream is None else stream
    try:
        return bool(stream.isatty())
    except Exception:
        return False


def _forced_live_progress() -> bool:
    return os.environ.get("D2S_FORCE_TQDM", "").strip().lower() in {"1", "true", "yes", "on"}


def _forced_progress_ncols() -> int:
    try:
        return max(60, min(120, int(os.environ.get("D2S_TQDM_NCOLS", "79"))))
    except ValueError:
        return 79


def progress_write(message: str, *, leading_newline: bool = False) -> None:
    text = str(message)
    if leading_newline:
        text = "\n" + text
    if supports_live_progress():
        from tqdm import tqdm

        tqdm.write(text, file=sys.stdout)
        return
    print(text, flush=True)


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


def make_tqdm_progress(*args, **kwargs):
    """Create a tqdm progress bar with Desktop2Stereo console defaults."""
    from tqdm import tqdm

    kwargs.setdefault("file", sys.stdout)
    kwargs.setdefault("ascii", True)
    if _forced_live_progress():
        kwargs.setdefault("dynamic_ncols", False)
        kwargs.setdefault("ncols", _forced_progress_ncols())
    else:
        kwargs.setdefault("dynamic_ncols", True)
    kwargs.setdefault("mininterval", 0.1)
    return tqdm(*args, **kwargs)


@contextmanager
def activity_progress(desc: str, *, interval_s: float = 0.2):
    """Show a live activity line for long operations without real percent callbacks."""
    if not supports_live_progress():
        print(f"[Main] {desc}...", flush=True)
        yield _NullProgress(desc)
        return

    stop_event = threading.Event()
    bar = make_tqdm_progress(
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

    bar = make_tqdm_progress(
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
    bar = make_tqdm_progress(
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

    bar = make_tqdm_progress(
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
