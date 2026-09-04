from __future__ import annotations

import os

import forge3d as f3d


def require_qualifying_gpu(reason: str) -> dict:
    """Require a physical GPU, skipping only outside the qualifying lane."""
    err = f3d.native_import_error()
    if err is not None:
        if os.environ.get("FORGE3D_RUN_OBLIQUA_GPU"):
            raise AssertionError(
                f"UNVERIFIED/ABSENT: native module unavailable for {reason}: {err}"
            )
        import pytest

        pytest.skip(f"UNVERIFIED/ABSENT: native module unavailable for {reason}")
    probe = f3d.device_probe(os.environ.get("WGPU_BACKEND"))
    bad = (
        probe.get("status") != "ok"
        or probe.get("software_fallback")
        or str(probe.get("device_type", "")).upper() == "CPU"
    )
    if os.environ.get("FORGE3D_RUN_OBLIQUA_GPU"):
        assert not bad, f"UNVERIFIED/ABSENT: no qualifying GPU for {reason}: {probe}"
    elif bad:
        import pytest

        pytest.skip(f"UNVERIFIED/ABSENT: no qualifying GPU for {reason}")
    return probe
