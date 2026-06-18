def normalize_upscaler(value) -> str:
    upscaler = str(value or "Off")
    lowered = upscaler.strip().lower()
    if lowered in ("auto", "自动"):
        return "Auto"
    if lowered in ("off", "关闭"):
        return "Off"
    if lowered == "fsr1":
        return "FSR1"
    return upscaler


def normalize_upscaler_sharpness(value, default: float = 0.35) -> float:
    try:
        sharpness = float(value)
    except (TypeError, ValueError):
        sharpness = default
    return max(0.0, min(1.0, sharpness))
