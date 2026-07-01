import os
import socket


def get_local_ip():
    """Return the local IP address by creating a UDP socket to a public IP."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            # The remote address does not need to be reachable for getsockname().
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"


_HF_ENDPOINT_DEFAULT = "https://hf-mirror.com"


def configure_huggingface_endpoint(async_probe=True):
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
    return os.environ.setdefault("HF_ENDPOINT", _HF_ENDPOINT_DEFAULT)
