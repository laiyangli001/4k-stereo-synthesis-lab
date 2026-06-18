from __future__ import annotations

import os
import warnings


def configure_platform_environment(os_name: str) -> None:
    if os_name != "Darwin":
        return

    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    warnings.filterwarnings(
        "ignore",
        message=".*aten::upsample_bicubic2d.out.*MPS backend.*",
        category=UserWarning,
    )
