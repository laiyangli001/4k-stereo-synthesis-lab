# Filament bridge

This directory contains the portable C++ bridge. It loads GLB assets through
Filament gltfio and applies standard glTF animation channels through Animator.
It never accepts or returns CPU pixel buffers.

Each target platform needs its matching official Filament SDK archive. Configure
CMake with `FILAMENT_SDK_ROOT` pointing at that extracted archive. The generated
library is placed in `src/xr_viewer/native` for packaging:

```text
Windows: filament_bridge.dll
macOS:   libfilament_bridge.dylib
Linux:   libfilament_bridge.so
```

OpenXR context and swapchain binding is deliberately platform-specific and is
not implemented by this portable asset and animation layer.