import sys
import types

from capture.dxgi import DxgiDuplicationFrameSource


class FakeDxgiDuplicationSession:
    instances = []

    def __init__(self, monitor_index):
        self.monitor_index = monitor_index
        self.acquire_timeouts = []
        self.recreated = False
        self.closed = False
        FakeDxgiDuplicationSession.instances.append(self)

    def acquire_frame(self, timeout_ms):
        self.acquire_timeouts.append(timeout_ms)
        return f"frame:{timeout_ms}"

    def acquire_gpu_frame(self, timeout_ms=16, shared=True):
        return f"gpu:{timeout_ms}:{shared}"

    def switch_monitor(self, monitor_index):
        self.monitor_index = monitor_index

    def recreate(self):
        self.recreated = True

    def close(self):
        self.closed = True


def test_dxgi_duplication_frame_source_wraps_session_lifecycle(monkeypatch):
    module = types.ModuleType("capture.dxgi.windows_capture")
    module.DxgiDuplicationSession = FakeDxgiDuplicationSession
    monkeypatch.setitem(sys.modules, "capture.dxgi.windows_capture", module)
    FakeDxgiDuplicationSession.instances.clear()

    source = DxgiDuplicationFrameSource(monitor_index=2)

    assert source.session is FakeDxgiDuplicationSession.instances[0]
    assert source.monitor_index == 2
    assert source.supports_gpu_frames is True
    assert source.acquire_frame(timeout_ms=17) == "frame:17"
    assert source.acquire_gpu_frame(timeout_ms=18, shared=False) == "gpu:18:False"
    assert source.session.acquire_timeouts == [17]

    source.switch_monitor(3)
    assert source.monitor_index == 3
    assert source.session.monitor_index == 3

    source.recreate()
    assert source.session.recreated is True

    source.close()
    assert source.session.closed is True
