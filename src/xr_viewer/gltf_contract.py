"""Compatibility shim for the glTF renderer-facing contract.

New code should import from ``xr_viewer.gltf`` or ``xr_viewer.gltf.contract``.
"""

from .gltf.contract import *  # noqa: F401,F403
from .gltf.contract import __all__
