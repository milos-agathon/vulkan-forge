"""Lightweight shader compilation helpers."""

from __future__ import annotations

import shutil
import subprocess
import warnings
from typing import Optional


def _run(cmd: list[str], data: Optional[bytes] = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, input=data, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )


def compile_glsl(src: str, stage: str) -> bytes:
    """Compile GLSL source to SPIR-V."""

    glslc = shutil.which("glslc")
    if not glslc:
        raise RuntimeError("glslc not found")
    result = _run(
        [glslc, "-fauto-map-locations", f"-fshader-stage={stage}", "-"], src.encode()
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode())
    return result.stdout


def validate_spirv(spirv: bytes) -> bool:
    """Validate SPIR-V using spirv-val if available."""

    spirv_val = shutil.which("spirv-val")
    if not spirv_val:
        warnings.warn("spirv-val not found; skipping validation", RuntimeWarning)
        return True
    try:
        result = _run([spirv_val, "-"], spirv)
        return result.returncode == 0
    except FileNotFoundError:
        warnings.warn("spirv-val not executable; skipping validation", RuntimeWarning)
        return True
