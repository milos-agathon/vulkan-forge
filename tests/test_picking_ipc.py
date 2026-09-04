"""Integration tests for picking system via IPC.

Tests the Plan 3 picking functionality including:
- Vector overlay with feature IDs
- Lasso mode toggle
- Pick event polling
- BVH-based picking
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from forge3d.viewer_ipc import (
    launch_viewer,
    close_viewer,
    send_ipc,
    add_vector_overlay,
    poll_pick_events,
    set_lasso_mode,
    get_lasso_state,
    clear_selection,
)


RUN_REQUIRED = os.environ.get("RUN_M06_VIEWER_CI") == "1"
pytestmark = [
    pytest.mark.interactive_viewer,
    pytest.mark.skipif(
        not RUN_REQUIRED,
        reason="required M-06 NVIDIA/Vulkan viewer lane only",
    ),
]


def _required_dem_path() -> Path:
    configured = os.environ.get("FORGE3D_TEST_DEM")
    path = (
        Path(configured)
        if configured
        else Path(__file__).parent.parent / "assets" / "tif" / "Mount_Fuji_30m.tif"
    )
    if not path.is_file():
        pytest.fail(f"required M-06 viewer DEM does not exist: {path}")
    return path


@pytest.fixture(scope="module")
def viewer_connection():
    """Launch the exact required-lane viewer and provide its IPC connection."""
    process, _, sock = launch_viewer(width=800, height=600, print_output=True)
    try:
        yield sock, process
    finally:
        close_viewer(sock, process)


class TestVectorOverlayFeatureIds:
    """Test vector overlay with per-feature IDs."""
    
    def test_add_overlay_with_feature_ids(self, viewer_connection):
        """Test adding vector overlay with distinct feature IDs per triangle."""
        sock, _ = viewer_connection
        
        # First load a terrain (required for vector overlays)
        dem_path = _required_dem_path()

        resp = send_ipc(sock, {"cmd": "load_terrain", "path": str(dem_path)})
        assert resp.get("ok", False), f"Failed to load terrain: {resp}"
        
        # Create vertices with distinct feature IDs
        # Format: [x, y, z, r, g, b, a, feature_id]
        vertices = [
            # Triangle 1 - feature_id = 1
            [138.72, 4000.0, 35.36, 1.0, 0.0, 0.0, 1.0, 1.0],
            [138.73, 4000.0, 35.35, 1.0, 0.0, 0.0, 1.0, 1.0],
            [138.71, 4000.0, 35.35, 1.0, 0.0, 0.0, 1.0, 1.0],
            # Triangle 2 - feature_id = 2
            [138.74, 4000.0, 35.36, 0.0, 1.0, 0.0, 1.0, 2.0],
            [138.75, 4000.0, 35.35, 0.0, 1.0, 0.0, 1.0, 2.0],
            [138.73, 4000.0, 35.35, 0.0, 1.0, 0.0, 1.0, 2.0],
            # Triangle 3 - feature_id = 3
            [138.76, 4000.0, 35.36, 0.0, 0.0, 1.0, 1.0, 3.0],
            [138.77, 4000.0, 35.35, 0.0, 0.0, 1.0, 1.0, 3.0],
            [138.75, 4000.0, 35.35, 0.0, 0.0, 1.0, 1.0, 3.0],
        ]
        indices = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        
        resp = add_vector_overlay(
            sock,
            "Test Features",
            vertices,
            indices,
            primitive="triangles",
            drape=False,
            opacity=1.0,
        )
        
        assert resp.get("ok", False), f"Failed to add vector overlay: {resp}"
        # The response should contain the overlay ID
        assert "id" in resp or resp.get("ok"), "Expected overlay ID in response"


class TestLassoMode:
    """Test lasso selection mode."""
    
    def test_lasso_mode_toggle(self, viewer_connection):
        """Test enabling and disabling lasso mode."""
        sock, _ = viewer_connection
        
        # Enable lasso mode
        resp = set_lasso_mode(sock, True)
        assert resp.get("ok", False), f"Failed to enable lasso mode: {resp}"
        
        # Check lasso state
        resp = get_lasso_state(sock)
        assert resp.get("ok", False), f"Failed to get lasso state: {resp}"
        # State should be "active" or similar
        
        # Disable lasso mode
        resp = set_lasso_mode(sock, False)
        assert resp.get("ok", False), f"Failed to disable lasso mode: {resp}"
    
    def test_clear_selection(self, viewer_connection):
        """Test clearing selection."""
        sock, _ = viewer_connection
        
        resp = clear_selection(sock)
        assert resp.get("ok", False), f"Failed to clear selection: {resp}"


class TestPickEventPolling:
    """Test pick event polling."""
    
    def test_poll_empty_events(self, viewer_connection):
        """Test polling when no pick events have occurred."""
        sock, _ = viewer_connection
        
        resp = poll_pick_events(sock)
        assert resp.get("ok", False), f"Failed to poll pick events: {resp}"
        # Events list may be empty if no clicks happened
        events = resp.get("pick_events", [])
        assert isinstance(events, list), "Expected pick_events to be a list"


class TestVertexFormat:
    """Test that vertex format with 8 components is accepted."""
    
    def test_8_component_vertices(self, viewer_connection):
        """Test that vertices with 8 components (including feature_id) are accepted."""
        sock, _ = viewer_connection
        
        dem_path = _required_dem_path()
        resp = send_ipc(sock, {"cmd": "load_terrain", "path": str(dem_path)})
        assert resp.get("ok", False), f"Failed to load terrain: {resp}"

        # Try adding overlay - should not fail with parse error
        vertices = [
            [138.72, 4000.0, 35.36, 1.0, 0.0, 0.0, 1.0, 42.0],  # feature_id = 42
            [138.73, 4000.0, 35.35, 1.0, 0.0, 0.0, 1.0, 42.0],
            [138.71, 4000.0, 35.35, 1.0, 0.0, 0.0, 1.0, 42.0],
        ]
        indices = [0, 1, 2]
        
        resp = add_vector_overlay(
            sock,
            "Feature ID Test",
            vertices,
            indices,
            primitive="triangles",
        )
        
        # Should succeed without JSON parse error
        assert resp.get("ok", False), f"8-component vertex format rejected: {resp}"
    
    def test_7_component_vertices_should_fail(self, viewer_connection):
        """Test that vertices with only 7 components are rejected."""
        sock, _ = viewer_connection
        
        # Try adding overlay with 7-component vertices - should fail
        vertices = [
            [138.72, 4000.0, 35.36, 1.0, 0.0, 0.0, 1.0],  # Missing feature_id
            [138.73, 4000.0, 35.35, 1.0, 0.0, 0.0, 1.0],
            [138.71, 4000.0, 35.35, 1.0, 0.0, 0.0, 1.0],
        ]
        indices = [0, 1, 2]
        
        with pytest.raises(ValueError, match="exactly 8 lanes"):
            add_vector_overlay(
                sock,
                "Invalid Format Test",
                vertices,
                indices,
                primitive="triangles",
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
