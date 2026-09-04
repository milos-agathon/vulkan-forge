"""Behavioral proof for the public M-06 Python viewer contracts."""

from __future__ import annotations

import io
import math
import os
from pathlib import Path

import pytest

import forge3d
from forge3d import viewer_ipc
from forge3d.viewer import ViewerHandle
from tests import test_m06_full_geospatial_viewer as m06_viewer
from tests import test_terrain_viewer_pbr as terrain_viewer
from tests import test_vector_overlay_rendering as vector_viewer
from tests.test_terrain_viewer_pbr import _drain_viewer_stdout


VIEWER_BINARY_HELPERS = (
    (m06_viewer, m06_viewer._viewer_binary, "ROOT"),
    (terrain_viewer, terrain_viewer.find_viewer_binary, "PROJECT_ROOT"),
    (vector_viewer, vector_viewer.find_viewer_binary, "PROJECT_ROOT"),
)


def test_shared_viewer_launcher_honors_exact_binary_override(tmp_path, monkeypatch):
    binary = tmp_path / "interactive_viewer.exe"
    binary.write_bytes(b"fresh viewer")
    monkeypatch.setenv("FORGE3D_VIEWER_BINARY", str(binary))
    assert Path(viewer_ipc.find_viewer_binary()) == binary

    missing = tmp_path / "missing-viewer.exe"
    monkeypatch.setenv("FORGE3D_VIEWER_BINARY", str(missing))
    with pytest.raises(FileNotFoundError, match="FORGE3D_VIEWER_BINARY does not exist"):
        viewer_ipc.find_viewer_binary()

    monkeypatch.setenv("FORGE3D_VIEWER_BINARY", "")
    with pytest.raises(ValueError, match="FORGE3D_VIEWER_BINARY must not be empty"):
        viewer_ipc.find_viewer_binary()


def _capturing_handle() -> tuple[ViewerHandle, list[dict]]:
    handle = object.__new__(ViewerHandle)
    captured: list[dict] = []

    def send(command: dict) -> dict:
        captured.append(command)
        return {"ok": True, "id": command.get("id")}

    handle._send_command = send  # type: ignore[method-assign]
    return handle, captured


@pytest.mark.parametrize(("module", "helper", "_root_attr"), VIEWER_BINARY_HELPERS)
def test_required_viewer_binary_honors_override_and_rejects_stale_or_empty_path(
    tmp_path, monkeypatch, module, helper, _root_attr
):
    binary = tmp_path / "interactive_viewer.exe"
    binary.write_bytes(b"fresh viewer")
    monkeypatch.setenv("FORGE3D_VIEWER_BINARY", str(binary))
    assert helper() == binary

    missing = tmp_path / "missing-viewer.exe"
    monkeypatch.setenv("FORGE3D_VIEWER_BINARY", str(missing))
    with pytest.raises(AssertionError, match="FORGE3D_VIEWER_BINARY does not exist"):
        helper()

    monkeypatch.setenv("FORGE3D_VIEWER_BINARY", "")
    with pytest.raises(AssertionError, match="FORGE3D_VIEWER_BINARY must not be empty"):
        helper()


@pytest.mark.parametrize(("module", "helper", "root_attr"), VIEWER_BINARY_HELPERS)
def test_required_viewer_binary_falls_back_only_when_override_is_unset(
    tmp_path, monkeypatch, module, helper, root_attr
):
    monkeypatch.delenv("FORGE3D_VIEWER_BINARY", raising=False)
    monkeypatch.setattr(module, root_attr, tmp_path)
    suffix = ".exe" if os.name == "nt" else ""
    binary = tmp_path / "target" / "release" / f"interactive_viewer{suffix}"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"fallback viewer")
    assert helper() == binary


def test_post_ready_viewer_output_is_drained_with_stable_prefix(capsys):
    _drain_viewer_stdout(io.StringIO("first line\nsecond line\n"))
    assert capsys.readouterr().out == (
        "[forge3d-viewer] first line\n[forge3d-viewer] second line\n"
    )


def test_authoritative_aliases_are_exported_from_package_and_viewer_module():
    assert forge3d.WorldPosition is not None
    assert forge3d.VectorOverlayVertex is not None
    assert forge3d.NormalizedExtent is not None


def test_vector_helper_preserves_earth_scale_submillimetre_xyz_and_large_u32_id():
    handle, captured = _capturing_handle()
    x = 6_378_137.0 + 0.000_25
    created = handle.add_vector_overlay(
        "precision",
        [(x, 2.0, 3.0, 1.0, 0.5, 0.0, 1.0, 16_777_217)],
        [0],
        primitive="points",
    )
    assert created == 1
    assert captured[0]["vertices"][0][0] == x
    assert captured[0]["vertices"][0][7] == 16_777_217
    assert handle._next_public_vector_overlay_id == 2


@pytest.mark.parametrize(
    "row",
    [
        (0.0,) * 7,
        (0.0,) * 9,
        (math.nan, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1),
        (0.0, 0.0, 0.0, 1e300, 1.0, 1.0, 1.0, 1),
        (0.0, 0.0, 0.0, -0.1, 1.0, 1.0, 1.0, 1),
        (0.0, 0.0, 0.0, 1.1, 1.0, 1.0, 1.0, 1),
        (0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.5),
        (0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, -1),
        (0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2**32),
    ],
)
def test_vector_rejection_is_synchronous_and_allocator_atomic(row):
    handle, captured = _capturing_handle()
    with pytest.raises((TypeError, ValueError)):
        handle.add_vector_overlay("bad", [row], [0], primitive="points")
    assert captured == []
    assert getattr(handle, "_next_public_vector_overlay_id", 1) == 1


def test_normalized_extent_rejects_non_normalized_or_empty_ranges_before_ipc():
    handle, captured = _capturing_handle()
    with pytest.raises(ValueError):
        handle.load_overlay("bad", "overlay.png", extent=(0.0, 0.0, 2.0, 1.0))
    with pytest.raises(ValueError):
        handle.load_overlay("bad", "overlay.png", extent=(0.5, 0.0, 0.5, 1.0))
    assert captured == []


def test_high_level_pick_is_execution_correlated_and_returns_absolute_results():
    handle = object.__new__(ViewerHandle)
    commands: list[dict] = []
    responses = iter(
        (
            {"ok": True, "pick_events": [{"screen_pos": [1, 2], "results": []}]},
            {"ok": True},
            {
                "ok": True,
                "pick_events": [
                    {
                        "screen_pos": [40, 50],
                        "results": [
                            {
                                "kind": "Object",
                                "world_pos": [6_378_137.000_25, 2.0, 3.0],
                            }
                        ],
                    }
                ],
            },
        )
    )

    def send(command: dict) -> dict:
        commands.append(command)
        return next(responses)

    handle._send_command = send  # type: ignore[method-assign]
    results = handle.pick_at(40, 50, shift=True)
    assert [command["cmd"] for command in commands] == [
        "poll_pick_events",
        "pick_at",
        "poll_pick_events",
    ]
    assert commands[1] == {
        "cmd": "pick_at",
        "x": 40,
        "y": 50,
        "shift": True,
        "ctrl": False,
    }
    assert results[0]["world_pos"][0] == 6_378_137.000_25


def test_manual_label_update_uses_the_execution_correlated_command():
    handle, captured = _capturing_handle()
    handle.update_labels()
    assert captured == [{"cmd": "update_labels"}]
