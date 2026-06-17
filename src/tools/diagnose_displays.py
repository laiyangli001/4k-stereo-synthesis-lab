import os
import sys


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT_DIR)
sys.path.insert(0, ROOT_DIR)


def main():
    from utils import (
        MONITOR_INDEX,
        STEREO_DISPLAY_INDEX,
        STEREO_DISPLAY_SELECTION,
        TARGET_FPS,
        FPS,
        OUTPUT_RESOLUTION,
        UPSCALER,
        UPSCALER_SHARPNESS,
        CAPTURE_TOOL,
        RUN_MODE,
        _get_device_name_from_mss_monitor,
    )

    print("=== Desktop2Stereo Display Diagnosis ===")
    print(f"Run Mode: {RUN_MODE}")
    print(f"Capture Tool: {CAPTURE_TOOL}")
    print(f"Input Monitor Index (MSS): {MONITOR_INDEX}")
    print(f"Stereo Output Selection: {STEREO_DISPLAY_SELECTION}")
    print(f"Stereo Output Index (MSS): {STEREO_DISPLAY_INDEX}")
    print(f"Target FPS: {TARGET_FPS}")
    print(f"Effective FPS: {FPS}")
    print(f"Processing Resolution: {OUTPUT_RESOLUTION}")
    print(f"Upscaler: {UPSCALER} sharpness={UPSCALER_SHARPNESS:.2f}")
    print()

    try:
        import mss
        print("=== MSS monitors ===")
        with mss.mss() as sct:
            for i, mon in enumerate(sct.monitors):
                tag = []
                if i == MONITOR_INDEX:
                    tag.append("INPUT")
                if STEREO_DISPLAY_SELECTION and i == STEREO_DISPLAY_INDEX:
                    tag.append("OUTPUT")
                print(
                    f"MSS {i}: left={mon['left']} top={mon['top']} "
                    f"size={mon['width']}x{mon['height']} {' '.join(tag)}"
                )
    except Exception as exc:
        print(f"MSS error: {type(exc).__name__}: {exc}")
    print()

    if os.name == "nt":
        try:
            import win32api
            print("=== Win32 display monitors ===")
            for idx, item in enumerate(win32api.EnumDisplayMonitors()):
                hmon, _hdc, rect = item
                info = win32api.GetMonitorInfo(hmon)
                device = info.get("Device")
                settings = win32api.EnumDisplaySettings(device, -1)
                print(
                    f"WIN {idx}: device={device} rect={rect} "
                    f"freq={settings.DisplayFrequency}Hz "
                    f"mode={settings.PelsWidth}x{settings.PelsHeight}"
                )
            print()
            print("=== MSS to Win32 mapping ===")
            try:
                input_device = _get_device_name_from_mss_monitor(MONITOR_INDEX)
                input_settings = win32api.EnumDisplaySettings(input_device, -1)
                print(
                    f"Input MSS {MONITOR_INDEX} -> {input_device} "
                    f"{input_settings.PelsWidth}x{input_settings.PelsHeight}@{input_settings.DisplayFrequency}Hz"
                )
            except Exception as exc:
                print(f"Input mapping error: {type(exc).__name__}: {exc}")
            if STEREO_DISPLAY_SELECTION:
                try:
                    output_device = _get_device_name_from_mss_monitor(STEREO_DISPLAY_INDEX)
                    output_settings = win32api.EnumDisplaySettings(output_device, -1)
                    print(
                        f"Output MSS {STEREO_DISPLAY_INDEX} -> {output_device} "
                        f"{output_settings.PelsWidth}x{output_settings.PelsHeight}@{output_settings.DisplayFrequency}Hz"
                    )
                except Exception as exc:
                    print(f"Output mapping error: {type(exc).__name__}: {exc}")
        except Exception as exc:
            print(f"Win32 error: {type(exc).__name__}: {exc}")
    print()

    try:
        import glfw
        print("=== GLFW monitors ===")
        if not glfw.init():
            print("GLFW init failed")
            return
        try:
            for i, mon in enumerate(glfw.get_monitors()):
                x, y = glfw.get_monitor_pos(mon)
                vm = glfw.get_video_mode(mon)
                print(
                    f"GLFW {i}: pos=({x},{y}) "
                    f"size={vm.size.width}x{vm.size.height}@{vm.refresh_rate}Hz"
                )
        finally:
            glfw.terminate()
    except Exception as exc:
        print(f"GLFW error: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
