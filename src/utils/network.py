import os
import socket

import requests


def get_local_ip():
    """Return the local IP address by creating a UDP socket to a public IP."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            # The remote address does not need to be reachable for getsockname().
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"


def is_cn_ip():
    """Dual-probe detection: only returns False when *both* endpoints are reachable.
    Exception → True (safe side, triggers HF mirror fallback)."""
    google_ok = False
    hf_ok = False

    try:
        requests.get("https://www.google.com", timeout=5)
        google_ok = True
    except Exception:
        google_ok = False

    try:
        requests.get("https://huggingface.co", timeout=5)
        hf_ok = True
    except Exception:
        hf_ok = False

    return not (google_ok and hf_ok)


def configure_huggingface_endpoint():
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
    if is_cn_ip():
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    else:
        os.environ["HF_ENDPOINT"] = "https://huggingface.co"
    return os.environ["HF_ENDPOINT"]
