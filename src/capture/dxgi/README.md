# DXGI capture helpers

This package contains project-facing DXGI Desktop Duplication helpers.

It keeps the project adapter plus the editable native binding source used to
build the DXGI-enabled `windows_capture` package. The built `.pyd` is also kept
here so project code can import the DXGI-enabled build without relying on
site-packages.

```text
capture.dxgi.DxgiDuplicationFrameSource
capture.dxgi.windows_capture
```

Important files:

```text
windows_capture/__init__.py
windows_capture/windows_capture.pyd
native/lib.rs
native/windows_capture/__init__.py
```

`native/lib.rs` is the modified `windows-capture-python/src/lib.rs` source that
adds the GPU-backed DXGI frame API. `native/windows_capture/__init__.py` is the
matching Python wrapper source. Runtime imports use the bundled
`capture.dxgi.windows_capture` package.
