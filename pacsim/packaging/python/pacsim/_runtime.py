"""Runtime helpers for packaged pacsim wheels."""

from __future__ import annotations

import os
from pathlib import Path


def bundled_swiftshader_icd_path() -> str | None:
    """Return the bundled SwiftShader ICD JSON path, when present."""

    path = Path(__file__).resolve().parent / "swiftshader" / "vk_swiftshader_icd.json"
    return str(path) if path.is_file() else None


def configure_bundled_swiftshader(*, force: bool = False) -> str | None:
    """Expose the bundled SwiftShader ICD through environment variables.

    pacsim's Vulkan renderer discovers the package-local ICD automatically during
    software fallback. This helper is mainly useful for callers that need Vulkan
    tooling or child processes to see the same bundled ICD.
    """

    path = bundled_swiftshader_icd_path()
    if path is None:
        return None

    os.environ["PACSIM_SWIFTSHADER_ICD"] = path
    if force:
        os.environ["VK_ICD_FILENAMES"] = path
    return path
