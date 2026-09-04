"""Fail-closed selectors for backend-specific visual baselines."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping


ALLOWED_GOLDEN_VARIANTS = frozenset({"metal", "nvidia-vulkan"})


def selected_golden_variant(env_name: str, *, implicit_metal: bool) -> str | None:
    """Return a validated explicit variant, or the optional Metal diagnostic.

    NVIDIA selection is always explicit because a generic Vulkan adapter is not
    proof that the render came from the protected NVIDIA acceptance runner.
    """
    variant = os.environ.get(env_name)
    if variant is not None:
        if variant not in ALLOWED_GOLDEN_VARIANTS:
            raise ValueError(f"Unknown golden variant for {env_name}: {variant!r}")
        backend = os.environ.get("WGPU_BACKEND", "").lower()
        expected_backend = "vulkan" if variant == "nvidia-vulkan" else "metal"
        if backend != expected_backend:
            raise ValueError(
                f"Golden variant {variant!r} requires WGPU_BACKEND="
                f"{expected_backend!r}, got {backend!r}"
            )
        return variant
    if implicit_metal and os.environ.get("WGPU_BACKEND", "").lower() == "metal":
        return "metal"
    return None


def selected_golden_path(
    golden_dir: Path,
    stem: str,
    env_name: str,
    *,
    implicit_metal: bool,
) -> Path:
    """Build a golden path only after validating its backend selector."""
    variant = selected_golden_variant(env_name, implicit_metal=implicit_metal)
    suffix = f".{variant}" if variant is not None else ""
    return golden_dir / f"{stem}{suffix}.png"


def nvidia_vulkan_golden_selected(env_name: str) -> bool:
    """Whether this process explicitly selected the protected NVIDIA variant."""
    return selected_golden_variant(env_name, implicit_metal=False) == "nvidia-vulkan"


def assert_nvidia_vulkan_golden_adapter(env_name: str, probe: Mapping[str, object]) -> None:
    """Bind an NVIDIA variant comparison to the renderer process's adapter."""
    if not nvidia_vulkan_golden_selected(env_name):
        return

    assert probe.get("status") == "ok", f"NVIDIA golden adapter probe failed: {probe!r}"
    assert str(probe.get("backend", "")).lower() == "vulkan"
    assert str(probe.get("device_type", "")).lower() == "discretegpu"
    assert int(probe.get("vendor", 0)) == 0x10DE
    assert "nvidia" in str(probe.get("name", "")).lower()
    assert probe.get("software_fallback") is False
