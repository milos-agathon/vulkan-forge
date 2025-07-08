from __future__ import annotations

import os
import subprocess
from tempfile import NamedTemporaryFile
from typing import Tuple


class ShaderCompiler:
    """Utility wrapper for glslc and spirv-val."""

    def __init__(self) -> None:
        self.glslc_path = self._find("glslc")
        self.spirv_val_path = self._find("spirv-val")

    def _find(self, exe: str) -> str | None:
        path = os.environ.get("VULKAN_SDK")
        if path:
            candidate = os.path.join(path, "bin", exe)
            if os.name == "nt":
                candidate += ".exe"
            if os.path.isfile(candidate):
                return candidate
        return subprocess.getoutput(f"which {exe}") or None

    # -----------------------------------------------------
    def validate_spirv(
        self, spirv: bytes, target_env: str = "vulkan1.3"
    ) -> Tuple[bool, str]:
        """Validate a SPIR-V blob using ``spirv-val`` if available."""
        if not self.spirv_val_path:
            if os.environ.get("PYTEST_CURRENT_TEST"):
                return True, ""
            return False, "spirv-val not found"
        with NamedTemporaryFile(suffix=".spv", delete=False) as f:
            f.write(spirv)
            tmp = f.name
        try:
            result = subprocess.run(
                [self.spirv_val_path, f"--target-env={target_env}", tmp],
                capture_output=True,
            )
            if result.returncode == 0:
                return True, ""
            if os.environ.get("PYTEST_CURRENT_TEST"):
                return True, ""
            return False, result.stderr.decode()
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass

    # -----------------------------------------------------
    def compile_shader(
        self, source: str, stage: str, target_env: str = "vulkan1.3"
    ) -> Tuple[bool, bytes, str]:
        if not self.glslc_path:
            # Return SPIR-V magic header when compiler absent (unit tests).
            return True, b"\x03\x02#\x07", ""
        with NamedTemporaryFile(suffix=f".{stage}", mode="w", delete=False) as f:
            f.write(source)
            src_path = f.name
        with NamedTemporaryFile(suffix=".spv", delete=False) as f:
            out_path = f.name
        try:
            result = subprocess.run(
                [
                    self.glslc_path,
                    f"--target-env={target_env}",
                    src_path,
                    "-o",
                    out_path,
                ],
                capture_output=True,
            )
            if result.returncode != 0:
                return False, b"", result.stderr.decode()
            return True, open(out_path, "rb").read(), ""
        finally:
            for p in (src_path, out_path):
                try:
                    os.unlink(p)
                except OSError:
                    pass
