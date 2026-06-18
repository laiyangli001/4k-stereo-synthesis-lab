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
    try:
        ip = requests.get("https://api.ipify.org").text.strip()
        response = requests.get(f"http://ip-api.com/json/{ip}", timeout=10)
        response.raise_for_status()
        data = response.json()
        return data.get("country", "") == "China"
    except Exception:
        return False


def configure_huggingface_endpoint():
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    if is_cn_ip():
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    else:
        os.environ["HF_ENDPOINT"] = "https://huggingface.co"
    return os.environ["HF_ENDPOINT"]
