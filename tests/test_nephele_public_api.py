from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from forge3d.media import MediaError, Medium, render_volumetric_reference
from forge3d.terrain_params import AovSettings
from forge3d.terrain_params import make_terrain_params_config
from forge3d.viewer import ViewerHandle
import forge3d.media as media_module
import forge3d as f3d
from forge3d._native import NATIVE_AVAILABLE


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIAGNOSTIC_FIELDS = (
    "majorant_proof",
    "majorant_valid",
    "sample_count",
    "step_count",
    "temporal_history_decision",
    "temporal_history_reason",
    "host_visible_bytes",
    "froxel_device_local_bytes",
    "density_device_local_bytes",
    "majorant_device_local_bytes",
    "staging_readback_bytes",
    "adapter",
    "backend",
    "driver",
    "source_revision",
    "executed_multi_scatter",
    "single_scatter_luminance",
    "multiple_scatter_luminance",
    "energy_accounting_residual",
)
REALTIME_DIAGNOSTIC_FIELDS = REFERENCE_DIAGNOSTIC_FIELDS + (
    "sun_transmittance_method",
    "sun_transmittance_bias",
    "sun_transmittance_max_segment_length",
    "sun_transmittance_executed_steps",
    "sun_transmittance_max_abs_error",
)


class _NativeMedium:
    def __init__(self, version: int = 0) -> None:
        self.sigma_a = (0.1, 0.2, 0.3)
        self.sigma_s = (0.4, 0.5, 0.6)
        self.sigma_t = (0.5, 0.7, 0.9)
        self.version = version
        self.identity = "canonical-medium-id"

    def to_dict(self) -> dict[str, Any]:
        return {
            "sigma_a": list(self.sigma_a),
            "sigma_s": list(self.sigma_s),
            "phase": "Isotropic",
            "density": {
                "Homogeneous": {
                    "authored_density": 0.75,
                    "mapping": {"physical_density_per_authored_unit": 2.0},
                }
            },
            "version": self.version,
        }


def _fake_native(reference_result: dict[str, Any] | None = None) -> Any:
    class _MediumFactory:
        @staticmethod
        def homogeneous(*_args: Any, version: int = 0, **_kwargs: Any) -> _NativeMedium:
            return _NativeMedium(version)

        @staticmethod
        def grid3d(*_args: Any, version: int = 0, **_kwargs: Any) -> _NativeMedium:
            return _NativeMedium(version)

        @staticmethod
        def perlin_worley(*_args: Any, version: int = 0, **_kwargs: Any) -> _NativeMedium:
            return _NativeMedium(version)

    def _render(*args: Any, **kwargs: Any) -> dict[str, Any]:
        if reference_result is not None:
            reference_result["args"] = args
            reference_result["kwargs"] = kwargs
        shape = (2, 3, 3)
        diagnostics = {
            field: (
                True
                if field in {"majorant_valid", "executed_multi_scatter"}
                else 1
                if field.endswith("_bytes")
                or field.endswith("_count")
                else field
            )
            for field in REFERENCE_DIAGNOSTIC_FIELDS
        }
        return {
            "beauty": np.ones(shape, dtype=np.float32),
            "transmittance": np.ones(shape, dtype=np.float32),
            "in_scatter": np.ones(shape, dtype=np.float32),
            "cloud_shadow": np.ones(shape, dtype=np.float32),
            "optical_depth": np.ones(shape, dtype=np.float32),
            "terrain_slice": np.ones((2, 3), dtype=np.float32),
            "diagnostics": diagnostics,
        }

    return SimpleNamespace(Medium=_MediumFactory, _render_volumetric_reference=_render)


def test_medium_and_integrated_reference_execute_the_public_wrapper(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    monkeypatch.setattr(media_module, "_native_module", lambda: _fake_native(captured))
    medium = Medium.homogeneous((0.1, 0.2, 0.3), (0.4, 0.5, 0.6), 0.75, version=9)

    result = render_volumetric_reference(
        medium,
        np.zeros((4, 5), dtype=np.float64),
        3,
        2,
        {"origin": (0.0, 5.0, 10.0), "look_at": (0.0, 0.0, 0.0)},
        samples_per_pixel=4,
        homogeneous_medium_reach=100.0,
        seed=17,
    )

    assert medium.identity == "canonical-medium-id"
    assert medium.version == 9
    assert captured["args"][0] is medium._native
    assert captured["args"][1].dtype == np.float32
    assert captured["args"][1].flags.c_contiguous
    assert captured["args"][2:4] == (3, 2)
    assert captured["kwargs"]["samples_per_pixel"] == 4
    assert captured["kwargs"]["homogeneous_medium_reach"] == 100.0
    assert captured["kwargs"]["clip"] == (0.1, 6000.0)
    assert captured["kwargs"]["seed"] == 17
    assert {
        "beauty",
        "transmittance",
        "in_scatter",
        "cloud_shadow",
        "optical_depth",
        "terrain_slice",
    } <= result.keys()
    assert result["beauty"].shape == (2, 3, 3)
    assert result["terrain_slice"].shape == (2, 3)
    assert tuple(result["diagnostics"]) == REFERENCE_DIAGNOSTIC_FIELDS


def test_medium_validation_errors_remain_structured(monkeypatch: pytest.MonkeyPatch) -> None:
    class _RejectingMedium:
        @staticmethod
        def homogeneous(*_args: Any, **_kwargs: Any) -> Any:
            raise ValueError("sigma_s must be non-negative")

    monkeypatch.setattr(
        media_module,
        "_native_module",
        lambda: SimpleNamespace(Medium=_RejectingMedium),
    )
    with pytest.raises(MediaError, match="sigma_s must be non-negative"):
        Medium((0.0, 0.0, 0.0), (-1.0, 0.0, 0.0))


def test_viewer_media_attach_remove_and_type_error_execute(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(media_module, "_native_module", lambda: _fake_native())
    medium = Medium((0.1, 0.2, 0.3), (0.4, 0.5, 0.6), 0.75, version=9)
    handle = object.__new__(ViewerHandle)
    sent: list[dict[str, Any]] = []
    handle._send_command = lambda command: sent.append(command) or {"ok": True}  # type: ignore[method-assign]

    handle.set_media(medium)
    handle.set_media(None)

    assert sent[0] == {"cmd": "set_media", "media": medium.to_dict()}
    assert sent[1] == {"cmd": "set_media", "media": None}
    with pytest.raises(TypeError, match="forge3d.media.Medium"):
        handle.set_media(object())  # type: ignore[arg-type]

    diagnostics = {field: field for field in REALTIME_DIAGNOSTIC_FIELDS}
    handle._send_command = lambda _command: {  # type: ignore[method-assign]
        "ok": True,
        "stats": {
            "media_diagnostics": diagnostics,
            "media_render_error": "media_render_failed: injected failure",
        },
    }
    stats = handle.get_stats()
    assert stats["media_diagnostics"] == diagnostics
    assert stats["media_render_error"].startswith("media_render_failed:")


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="requires the matching native extension")
def test_native_terrain_params_media_roundtrip_returns_public_wrapper() -> None:
    medium = Medium((0.1, 0.2, 0.3), (0.4, 0.5, 0.6), 0.75, version=9)
    config = make_terrain_params_config(
        size_px=(64, 64),
        render_scale=1.0,
        terrain_span=10.0,
        msaa_samples=1,
        z_scale=1.0,
        exposure=1.0,
        domain=(0.0, 1.0),
        media=medium,
    )
    params = f3d.TerrainRenderParams(config)
    returned = params.media

    assert isinstance(returned, Medium)
    assert returned._native is medium._native
    assert returned.identity == medium.identity
    assert returned.to_dict() == medium.to_dict()

    config.media = None
    assert f3d.TerrainRenderParams(config).media is None

    handle = object.__new__(ViewerHandle)
    sent: list[dict[str, Any]] = []
    handle._send_command = lambda command: sent.append(command) or {"ok": True}  # type: ignore[method-assign]
    handle.set_media(returned)
    handle.set_media(None)
    assert sent == [
        {"cmd": "set_media", "media": medium.to_dict()},
        {"cmd": "set_media", "media": None},
    ]


def test_media_aov_settings_execute() -> None:
    settings = AovSettings(
        enabled=True,
        albedo=False,
        normal=False,
        depth=False,
        transmittance=True,
        in_scatter=True,
        cloud_shadow=True,
        optical_depth=True,
    )
    assert settings.any_enabled
    assert AovSettings(enabled=True, albedo=False, normal=False, depth=False, source_id=True).any_enabled
    with pytest.raises(ValueError, match="format"):
        AovSettings(format="jpeg")

def test_media_public_inventory_stubs_and_aov_methods_are_complete() -> None:
    module = ast.parse((ROOT / "python/forge3d/media.py").read_text(encoding="utf-8"))
    classes = {node.name for node in module.body if isinstance(node, ast.ClassDef)}
    functions = {node.name for node in module.body if isinstance(node, ast.FunctionDef)}
    assert {"Medium", "MediaError"} <= classes
    assert "render_volumetric_reference" in functions

    stubs = (ROOT / "python/forge3d/__init__.pyi").read_text(encoding="utf-8")
    viewer_stubs = (ROOT / "python/forge3d/viewer.pyi").read_text(encoding="utf-8")
    media_stubs = (ROOT / "python/forge3d/media.pyi").read_text(encoding="utf-8")
    assert "def set_media(self, media: Medium | None)" in viewer_stubs
    media_stub_module = ast.parse(media_stubs)
    reference_stub = next(
        node
        for node in media_stub_module.body
        if isinstance(node, ast.ClassDef) and node.name == "ReferenceMediaDiagnostics"
    )
    reference_types = {
        node.target.id: ast.unparse(node.annotation)
        for node in reference_stub.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }
    assert reference_types == {
        "majorant_proof": "Any",
        "majorant_valid": "bool",
        "sample_count": "int",
        "step_count": "int",
        "temporal_history_decision": "str",
        "temporal_history_reason": "str",
        "host_visible_bytes": "int",
        "froxel_device_local_bytes": "int",
        "density_device_local_bytes": "int",
        "majorant_device_local_bytes": "int",
        "staging_readback_bytes": "int",
        "adapter": "str",
        "backend": "str",
        "driver": "str",
        "source_revision": "str",
        "executed_multi_scatter": "bool",
        "single_scatter_luminance": "float | None",
        "multiple_scatter_luminance": "float | None",
        "energy_accounting_residual": "float | None",
    }
    realtime_stub = next(
        node
        for node in media_stub_module.body
        if isinstance(node, ast.ClassDef) and node.name == "RealtimeMediaDiagnostics"
    )
    assert [base.id for base in realtime_stub.bases if isinstance(base, ast.Name)] == [
        "ReferenceMediaDiagnostics"
    ]
    realtime_types = {
        node.target.id: ast.unparse(node.annotation)
        for node in realtime_stub.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }
    assert realtime_types == {
        "sun_transmittance_method": "str",
        "sun_transmittance_bias": "str",
        "sun_transmittance_max_segment_length": "float | None",
        "sun_transmittance_executed_steps": "int",
        "sun_transmittance_max_abs_error": "float",
    }
    reference_result_stub = next(
        node
        for node in media_stub_module.body
        if isinstance(node, ast.ClassDef) and node.name == "VolumetricReferenceResult"
    )
    diagnostics_annotation = next(
        node.annotation
        for node in reference_result_stub.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "diagnostics"
    )
    assert ast.unparse(diagnostics_annotation) == "ReferenceMediaDiagnostics"
    for name in ("transmittance", "in_scatter", "cloud_shadow", "optical_depth"):
        assert f"def has_{name}(self) -> bool" in stubs
        assert f"def {name}(self) -> np.ndarray" in stubs
        assert f"def save_{name}(self, path: PathLikeStr)" in stubs


def test_native_media_aov_and_diagnostic_names_are_registered() -> None:
    aov_source = (ROOT / "src/path_tracing/aov.rs").read_text(encoding="utf-8")
    native_source = (ROOT / "src/media_py.rs").read_text(encoding="utf-8")
    realtime_source = (ROOT / "src/terrain/renderer/media.rs").read_text(encoding="utf-8")
    for name in ("transmittance", "in_scatter", "cloud_shadow", "optical_depth"):
        assert f'"{name}"' in aov_source
        assert f'"{name}"' in native_source
    for name in REFERENCE_DIAGNOSTIC_FIELDS:
        assert f'"{name}"' in native_source
    for name in REALTIME_DIAGNOSTIC_FIELDS:
        assert f'"{name}"' in realtime_source


def test_live_viewer_paths_prepare_terrain_trace_before_encode() -> None:
    for relative in (
        "src/viewer/terrain/render/screen/effects.rs",
        "src/viewer/terrain/render/offscreen/effects.rs",
    ):
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert source.index("pass.prepare_viewer_terrain_trace(") < source.index(
            "let output = pass.encode("
        )


def test_executed_beer_residual_is_independently_derived_and_nonzero() -> None:
    product = np.float32(1.0)
    integrated = np.float32(0.0)
    step = np.float32(0.37)
    for extinction in np.asarray([0.2, 0.5, 1.0], dtype=np.float32):
        product = np.float32(product * np.exp(np.float32(-extinction * step)))
        integrated = np.float32(integrated + extinction * step)
    residual = abs(float(product - np.exp(np.float32(-integrated))))
    assert 0.0 < residual <= float(np.finfo(np.float32).eps)
    shader = (ROOT / "src/shaders/nephele_froxel.wgsl").read_text(encoding="utf-8")
    assert "abs(t.x-exp(-tau.x))" in shader


def test_temporal_depth_uses_one_depth32_code_not_exact_float_equality() -> None:
    depth = np.float32(0.5)
    adjacent = np.nextafter(depth, np.float32(1.0), dtype=np.float32)
    assert int(adjacent.view(np.uint32)) - int(depth.view(np.uint32)) == 1
    assert abs(float(depth) - 0.51) > float(adjacent - depth)
    shader = (ROOT / "src/shaders/nephele_froxel.wgsl").read_text(encoding="utf-8")
    assert "max(a,b)-min(a,b)<=1u" in shader
    assert "depth32_history_matches(old_depth,projected_old_depth)" in shader
    assert "old_depth==projected_old_depth" not in shader
