from __future__ import annotations

import os
import re


LUNA_MODEL_PATTERN = re.compile(r"^gpt-\d+(?:\.\d+)?-luna$", re.IGNORECASE)


def configured_luna_model() -> str:
    """Read the one deployment supplied Luna mapping and fail closed if absent."""
    model = os.environ.get("LUNA_AUTO_MODEL", "").strip().lower()
    if not LUNA_MODEL_PATTERN.fullmatch(model):
        raise RuntimeError("LUNA_AUTO_MODEL must be set to the approved concrete Luna model")
    return model
