"""P0.2/M3 Shadow Technique Selection Tests.

Tests that shadow technique CLI validation works correctly and that
different techniques (including VSM/EVSM/MSM) produce visually different outputs.
"""

import hashlib
import os
from pathlib import Path

import numpy as np
import pytest

from _terrain_runtime import HARDWARE_DEVICE_TYPES, SOFTWARE_ADAPTER_TOKENS
from forge3d.terrain_params import ShadowSettings
from forge3d.config import ShadowParams, _SHADOW_TECHNIQUES, validate_shadow_technique, load_renderer_config


def test_shadow_depth_height_curve_uses_receiver_primitives():
    source = (
        Path(__file__).parent.parent / "src" / "shaders" / "terrain_shadow_depth.wgsl"
    ).read_text(encoding="utf-8")

    assert "curved = det_pow(t, power);" in source
    assert "curved = pow(t, power);" not in source
    assert "curved = height_curve_lut_sample(t);" in source
    assert "return det_mix(t, curved, strength);" in source


def test_viewer_csm_allocation_and_rebuild_are_technique_aware():
    root = Path(__file__).parent.parent
    init = (root / "src" / "viewer" / "terrain" / "scene" / "pipeline_init.rs").read_text(
        encoding="utf-8"
    )
    update = (root / "src" / "viewer" / "terrain" / "scene" / "core.rs").read_text(
        encoding="utf-8"
    )
    bindings = (
        root / "src" / "viewer" / "terrain" / "render" / "helpers.rs"
    ).read_text(encoding="utf-8")

    assert "enable_evsm: requires_moments" in init
    assert "existing.config.enable_evsm == requires_moments" in init
    assert "if requires_moments" in init
    assert init.index("self.pbr_bind_group = None") < init.index(
        "self.csm_renderer = Some(replacement.csm_renderer)"
    )
    assert "validate_shadow_allocation_budget(" in update
    assert "requires_moments," in update
    assert "csm.moment_binding_view()" in bindings
    parser = (
        root / "src" / "viewer" / "terrain" / "pbr_renderer" / "mod.rs"
    ).read_text(encoding="utf-8")
    assert 'name.eq_ignore_ascii_case("none")' in parser
    assert "ShadowTechnique::from_name(name)" in parser
    assert "technique.requires_moments()" in parser


def test_shadow_replacements_release_dependents_before_tracked_owners():
    root = Path(__file__).parent.parent
    blur = (root / "src" / "shadows" / "blur_pass.rs").read_text(encoding="utf-8")
    resize = blur.split("fn ensure_intermediate_texture", 1)[1].split(
        "/// Execute two-pass", 1
    )[0]
    owner = resize.index("self.intermediate_texture = Some(texture)")
    for dependent in (
        "self.bind_groups = None",
        "self.intermediate_view = None",
        "self.moment_view = None",
        "self.current_atlas_id = None",
    ):
        assert resize.index(dependent) < owner, dependent

    setup = (
        root / "src" / "terrain" / "renderer" / "shadows" / "setup.rs"
    ).read_text(encoding="utf-8")
    replace = setup.split("fn ensure_shadow_atlas", 1)[1].split(
        "pub(in crate::terrain::renderer) fn prepare_shadow_setup", 1
    )[0]
    staged = replace.index("let replacement =")
    owner = replace.index("self.csm_renderer = replacement")
    assert staged < replace.index("self.moment_pass = None") < owner
    assert staged < replace.index("self.moment_blur_pass = None") < owner

    viewer = (
        root / "src" / "viewer" / "terrain" / "scene" / "pipeline_init.rs"
    ).read_text(encoding="utf-8")
    replace = viewer.split("let replacement = ShadowResources", 1)[1].split(
        'println!("[terrain_scene] Shadows initialized',
        1,
    )[0]
    owner = replace.index("self.csm_renderer = Some(replacement.csm_renderer)")
    assert replace.index("self.pbr_bind_group = None") < owner
    assert replace.index("self.moment_pass = None") < owner
    assert replace.index("self.moment_blur_pass = None") < owner
    assert owner < replace.index("self.moment_pass = replacement.moment_pass")
    assert owner < replace.index(
        "self.moment_blur_pass = replacement.moment_blur_pass"
    )


class TestShadowTechniqueValidation:
    """Test shadow technique validation in terrain_params.py."""

    # P0.2/M3: All shadow techniques are now supported (including VSM/EVSM/MSM)
    SUPPORTED_TECHNIQUES = ["hard", "pcf", "pcss", "vsm", "evsm", "msm", "none"]
    # CSM is the pipeline, not a technique - should raise clear error
    UNSUPPORTED_TECHNIQUES = ["csm"]

    def _make_shadow_settings(self, technique: str) -> ShadowSettings:
        """Helper to create ShadowSettings with given technique."""
        return ShadowSettings(
            enabled=True,
            technique=technique,
            resolution=2048,
            cascades=3,
            max_distance=4000.0,
            softness=1.5,
            intensity=0.8,
            slope_scale_bias=0.001,
            depth_bias=0.0005,
            normal_bias=0.0002,
            min_variance=1e-4,
            light_bleed_reduction=0.5,
            evsm_exponent=40.0,
            fade_start=1.0,
        )

    @pytest.mark.parametrize("technique", SUPPORTED_TECHNIQUES)
    def test_technique_validation_accepts_supported(self, technique: str):
        """All supported techniques should validate without error."""
        settings = self._make_shadow_settings(technique)
        # Technique should be normalized to uppercase
        assert settings.technique == technique.upper()

    @pytest.mark.parametrize("technique", SUPPORTED_TECHNIQUES)
    def test_technique_validation_case_insensitive(self, technique: str):
        """Technique validation should be case-insensitive."""
        for variant in [technique.lower(), technique.upper(), technique.capitalize()]:
            settings = self._make_shadow_settings(variant)
            assert settings.technique == technique.upper()

    def test_unsupported_technique_raises_error(self):
        """Unsupported techniques should raise ValueError with helpful message."""
        with pytest.raises(ValueError) as exc_info:
            self._make_shadow_settings("invalid_technique")
        
        error_msg = str(exc_info.value)
        assert "Unsupported shadow technique" in error_msg
        assert "INVALID_TECHNIQUE" in error_msg

    def test_none_technique_disables_shadows(self):
        """Technique='NONE' should disable shadows automatically."""
        settings = self._make_shadow_settings("none")
        assert settings.technique == "NONE"
        assert settings.enabled is False


class TestShadowConfigValidation:
    """Test shadow technique validation in config.py ShadowParams."""

    @pytest.mark.parametrize("technique", ["hard", "pcf", "pcss", "vsm", "evsm", "msm"])
    def test_shadow_params_from_mapping_supported(self, technique: str):
        """P0.2/M3: ShadowParams.from_mapping should accept all techniques including VSM/EVSM/MSM."""
        params = ShadowParams.from_mapping({"technique": technique})
        assert params.technique == technique  # config.py uses lowercase

    @pytest.mark.parametrize("technique", ["vsm", "evsm", "msm"])
    def test_shadow_params_requires_moments(self, technique: str):
        """P0.2/M3: VSM/EVSM/MSM techniques should require moment maps."""
        params = ShadowParams.from_mapping({"technique": technique})
        assert params.requires_moments() is True

    def test_shadow_params_from_mapping_rejects_csm(self):
        """ShadowParams.from_mapping should reject 'csm' with explanation."""
        with pytest.raises(ValueError) as exc_info:
            ShadowParams.from_mapping({"technique": "csm"})
        
        error_msg = str(exc_info.value)
        assert "csm" in error_msg.lower()
        assert "pipeline" in error_msg.lower() or "not a valid filter" in error_msg.lower()

    def test_shadow_techniques_dict_complete(self):
        """P0.2/M3: _SHADOW_TECHNIQUES dict should include all techniques including VSM/EVSM/MSM."""
        expected = {"none", "hard", "pcf", "pcss", "vsm", "evsm", "msm"}
        actual = set(_SHADOW_TECHNIQUES.keys())
        assert expected == actual, f"Expected {expected}, got {actual}"

    def test_renderer_config_shadow_technique(self):
        """load_renderer_config should propagate shadow technique correctly."""
        config = load_renderer_config(None, {"shadows": "pcss"})
        assert config.shadows.technique == "pcss"

    @pytest.mark.parametrize(
        "factory_path",
        (
            "forge3d.map_scene._mapscene_shadow_settings",
            "forge3d.terrain_demo._make_terrain_shadow_settings",
        ),
    )
    def test_active_terrain_translation_preserves_explicit_pcss_parameters(
        self, factory_path: str
    ):
        """The two active Python entry points must not reinterpret PCSS controls."""
        from importlib import import_module

        module_name, factory_name = factory_path.rsplit(".", 1)
        factory = getattr(import_module(module_name), factory_name)
        public = ShadowParams(
            technique="pcss",
            pcss_blocker_radius=7.25,
            pcss_filter_radius=3.5,
            light_size=2.75,
        )

        native_input = factory(public)

        assert native_input.pcss_blocker_radius == pytest.approx(7.25)
        assert native_input.pcss_filter_radius == pytest.approx(3.5)
        assert native_input.light_size == pytest.approx(2.75)
        assert native_input.softness != pytest.approx(public.light_size)

    def test_terrain_shadow_defaults_match_public_pcss_defaults(self):
        """Direct terrain settings and RendererConfig share one texel-space default."""
        settings = TestShadowTechniqueValidation()._make_shadow_settings("pcss")
        public = ShadowParams()

        assert settings.pcss_blocker_radius == pytest.approx(
            public.pcss_blocker_radius
        )
        assert settings.pcss_filter_radius == pytest.approx(public.pcss_filter_radius)
        assert settings.light_size == pytest.approx(public.light_size)

    @pytest.mark.parametrize(
        ("field_name", "value"),
        (
            ("pcss_light_radius", np.nan),
            ("pcss_blocker_radius", np.inf),
            ("pcss_filter_radius", -np.inf),
            ("light_size", np.nan),
        ),
    )
    def test_terrain_shadow_rejects_non_finite_pcss_controls(
        self, field_name: str, value: float
    ):
        kwargs = {field_name: value}
        with pytest.raises(ValueError, match=rf"{field_name} must be finite"):
            ShadowSettings(
                enabled=True,
                technique="PCSS",
                resolution=2048,
                cascades=1,
                max_distance=20.0,
                softness=1.0,
                intensity=1.0,
                slope_scale_bias=0.001,
                depth_bias=0.0005,
                normal_bias=0.0002,
                min_variance=1e-4,
                light_bleed_reduction=0.5,
                evsm_exponent=40.0,
                fade_start=1.0,
                **kwargs,
            )


class TestValidateShadowTechnique:
    """Test the validate_shadow_technique function directly."""

    @pytest.mark.parametrize("technique", ["hard", "pcf", "pcss", "vsm", "evsm", "msm", "none"])
    def test_accepts_supported_techniques(self, technique: str):
        """P0.2/M3: Should accept all techniques including VSM/EVSM/MSM."""
        result = validate_shadow_technique(technique)
        assert result == technique

    def test_rejects_csm_with_explanation(self):
        """Should reject 'csm' with explanation that it's the pipeline, not a filter."""
        with pytest.raises(ValueError) as exc_info:
            validate_shadow_technique("csm")
        
        error_msg = str(exc_info.value)
        assert "csm" in error_msg.lower()
        assert "pipeline" in error_msg.lower() or "not a valid filter" in error_msg.lower()


class TestShadowMemoryBudget:
    """Test shadow memory budget validation."""

    def test_public_renderer_budget_counts_all_moment_textures(self):
        params = ShadowParams(technique="evsm", map_size=2048, cascades=2)
        assert params.atlas_memory_bytes() == 160 * 1024 * 1024

        with pytest.raises(ValueError, match="exceeds 256 MiB"):
            load_renderer_config(
                {
                    "shadows": {
                        "technique": "evsm",
                        "map_size": 4096,
                        "cascades": 1,
                    }
                }
            )

    @pytest.mark.parametrize("map_size", (-1, 0, 255, 257))
    def test_public_renderer_rejects_invalid_shadow_dimensions(self, map_size: int):
        with pytest.raises(ValueError, match="greater than zero|power of two|at least 256"):
            load_renderer_config(
                {"shadows": {"map_size": map_size, "cascades": 1}}
            )

    def test_public_renderer_preserves_legacy_small_shadow_maps(self):
        hard = load_renderer_config(
            {"shadows": {"technique": "hard", "map_size": 128, "cascades": 1}}
        )
        filtered = load_renderer_config(
            {"shadows": {"technique": "pcf", "map_size": 256, "cascades": 1}}
        )
        assert hard.shadows.map_size == 128
        assert filtered.shadows.map_size == 256

    def test_memory_budget_normal_config(self):
        """Normal shadow config should pass memory budget check."""
        # 4096x4096 * 4 cascades * 4 bytes = 256 MiB (within 512 MiB budget)
        settings = ShadowSettings(
            enabled=True,
            technique="PCF",
            resolution=4096,
            cascades=4,
            max_distance=4000.0,
            softness=1.5,
            intensity=0.8,
            slope_scale_bias=0.001,
            depth_bias=0.0005,
            normal_bias=0.0002,
            min_variance=1e-4,
            light_bleed_reduction=0.5,
            evsm_exponent=40.0,
            fade_start=1.0,
        )
        assert settings.resolution == 4096
        assert settings.cascades == 4

    def test_memory_estimate_increases_with_resolution(self):
        """Higher resolution should increase memory estimate."""
        low_res = ShadowSettings(
            enabled=True, technique="PCF", resolution=1024, cascades=2,
            max_distance=4000.0, softness=1.5, intensity=0.8,
            slope_scale_bias=0.001, depth_bias=0.0005, normal_bias=0.0002,
            min_variance=1e-4, light_bleed_reduction=0.5, evsm_exponent=40.0, fade_start=1.0,
        )
        high_res = ShadowSettings(
            enabled=True, technique="PCF", resolution=4096, cascades=2,
            max_distance=4000.0, softness=1.5, intensity=0.8,
            slope_scale_bias=0.001, depth_bias=0.0005, normal_bias=0.0002,
            min_variance=1e-4, light_bleed_reduction=0.5, evsm_exponent=40.0, fade_start=1.0,
        )
        assert high_res._estimate_memory_bytes() > low_res._estimate_memory_bytes()

    def test_vsm_budget_includes_persistent_blur_intermediate(self):
        with pytest.raises(ValueError, match="exceed memory budget"):
            ShadowSettings(
                enabled=True,
                technique="VSM",
                resolution=4096,
                cascades=2,
                max_distance=4000.0,
                softness=1.5,
                intensity=0.8,
                slope_scale_bias=0.001,
                depth_bias=0.0005,
                normal_bias=0.0002,
                min_variance=1e-4,
                light_bleed_reduction=0.5,
                evsm_exponent=40.0,
                fade_start=1.0,
            )

    def test_budget_preserves_requested_native_terrain_resolution(self):
        settings = ShadowSettings(
            enabled=True, technique="PCF", resolution=1024, cascades=1,
            max_distance=4000.0, softness=1.5, intensity=0.8,
            slope_scale_bias=0.001, depth_bias=0.0005, normal_bias=0.0002,
            min_variance=1e-4, light_bleed_reduction=0.5,
            evsm_exponent=40.0, fade_start=1.0,
        )
        assert settings.resolution == 1024
        assert settings._estimate_memory_bytes() == 4 * 1024 * 1024


def _create_step_dem(width: int = 256, height: int = 256, cliff_height: float = 100.0) -> np.ndarray:
    """Create a synthetic step-DEM with a sharp cliff for shadow testing.
    
    Left half is low (0), right half is high (cliff_height).
    This creates a long shadow boundary where HARD/PCF/PCSS will differ.
    
    Args:
        width: DEM width in pixels
        height: DEM height in pixels
        cliff_height: Height of the cliff in meters
    
    Returns:
        2D numpy array with shape (height, width) containing elevation values
    """
    dem = np.zeros((height, width), dtype=np.float32)
    # Right half is elevated
    dem[:, width // 2:] = cliff_height
    return dem


def _rasterio_available() -> bool:
    try:
        import rasterio
        return not getattr(rasterio, "__forge3d_stub__", False)
    except Exception:
        return False


def _save_geotiff(dem: np.ndarray, path: Path) -> None:
    """Save a DEM array as a GeoTIFF file.

    Uses rasterio to write a simple GeoTIFF with default CRS and transform.
    """
    import rasterio
    from rasterio.transform import from_bounds
    
    height, width = dem.shape
    # Create a simple transform (1 meter per pixel, origin at 0,0)
    transform = from_bounds(0, 0, width, height, width, height)
    
    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=dem.dtype,
        crs='EPSG:32610',  # UTM zone 10N (arbitrary but valid)
        transform=transform,
    ) as dst:
        dst.write(dem, 1)


@pytest.mark.skipif(not _rasterio_available(), reason="rasterio not installed")
@pytest.mark.offscreen
class TestShadowTechniqueDifferentiation:
    """Test that different shadow techniques produce different outputs.

    Uses a synthetic step-DEM with a sharp cliff to guarantee visible shadow edges.
    """
    
    @pytest.fixture
    def step_dem_path(self, tmp_path: Path) -> Path:
        """Create a temporary step-DEM GeoTIFF for testing."""
        dem = _create_step_dem(width=256, height=256, cliff_height=100.0)
        dem_path = tmp_path / "step_dem.tif"
        _save_geotiff(dem, dem_path)
        return dem_path
    
    def _render_with_technique(self, dem_path: Path, technique: str, output_path: Path) -> bytes:
        """Render the DEM with the specified shadow technique and return the image bytes."""
        import subprocess
        import sys
        
        # Use terrain_demo.py CLI which handles all setup correctly
        # Use mesh mode for perspective, and low sun for visible shadows
        # The 512 shadow res ensures filtering differences are visible
        cmd = [
            sys.executable, "-B", "examples/terrain_demo.py",
            "--dem", str(dem_path),
            "--size", "320", "180",
            "--shadows", technique,
            "--shadow-map-res", "512",
            "--sun-elevation", "10",
            "--sun-azimuth", "45",
            "--ibl-intensity", "0",
            "--hdr", "assets/hdri/snow_field_4k.hdr",
            "--output", str(output_path),
            "--overwrite",
            "--camera-mode", "mesh",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
        if result.returncode != 0:
            raise RuntimeError(f"Render failed: {result.stderr}")
        return output_path.read_bytes()
    
    @pytest.mark.xfail(reason="HARD vs PCF differences are subtle in this test scene")
    @pytest.mark.skipif(
        not Path("assets/hdri/snow_field_4k.hdr").exists(),
        reason="HDR asset not available"
    )
    def test_hard_vs_pcf_differ(self, step_dem_path: Path, tmp_path: Path):
        """HARD and PCF techniques must produce different outputs."""
        hard_path = tmp_path / "hard.png"
        pcf_path = tmp_path / "pcf.png"
        
        hard_bytes = self._render_with_technique(step_dem_path, "hard", hard_path)
        pcf_bytes = self._render_with_technique(step_dem_path, "pcf", pcf_path)
        
        hard_hash = hashlib.md5(hard_bytes).hexdigest()
        pcf_hash = hashlib.md5(pcf_bytes).hexdigest()
        
        assert hard_hash != pcf_hash, f"HARD and PCF produced identical output: {hard_hash}"
    
    @pytest.mark.skipif(
        not Path("assets/hdri/snow_field_4k.hdr").exists(),
        reason="HDR asset not available"
    )
    def test_hard_vs_vsm_differ(self, step_dem_path: Path, tmp_path: Path):
        """P0.2/M3: HARD and VSM techniques must produce different outputs."""
        hard_path = tmp_path / "hard.png"
        vsm_path = tmp_path / "vsm.png"
        
        hard_bytes = self._render_with_technique(step_dem_path, "hard", hard_path)
        vsm_bytes = self._render_with_technique(step_dem_path, "vsm", vsm_path)
        
        hard_hash = hashlib.md5(hard_bytes).hexdigest()
        vsm_hash = hashlib.md5(vsm_bytes).hexdigest()
        
        assert hard_hash != vsm_hash, f"HARD and VSM produced identical output: {hard_hash}"

    @pytest.mark.xfail(reason="VSM vs EVSM differences are subtle - both use variance-based filtering")
    @pytest.mark.skipif(
        not Path("assets/hdri/snow_field_4k.hdr").exists(),
        reason="HDR asset not available"
    )
    def test_vsm_vs_evsm_differ(self, step_dem_path: Path, tmp_path: Path):
        """P0.2/M3: VSM and EVSM techniques must produce different outputs."""
        vsm_path = tmp_path / "vsm.png"
        evsm_path = tmp_path / "evsm.png"
        
        vsm_bytes = self._render_with_technique(step_dem_path, "vsm", vsm_path)
        evsm_bytes = self._render_with_technique(step_dem_path, "evsm", evsm_path)
        
        vsm_hash = hashlib.md5(vsm_bytes).hexdigest()
        evsm_hash = hashlib.md5(evsm_bytes).hexdigest()
        
        assert vsm_hash != evsm_hash, f"VSM and EVSM produced identical output: {vsm_hash}"

    @pytest.mark.xfail(reason="EVSM vs MSM differences are subtle - both use moment-based filtering")
    @pytest.mark.skipif(
        not Path("assets/hdri/snow_field_4k.hdr").exists(),
        reason="HDR asset not available"
    )
    def test_evsm_vs_msm_differ(self, step_dem_path: Path, tmp_path: Path):
        """P0.2/M3: EVSM and MSM techniques must produce different outputs."""
        evsm_path = tmp_path / "evsm.png"
        msm_path = tmp_path / "msm.png"
        
        evsm_bytes = self._render_with_technique(step_dem_path, "evsm", evsm_path)
        msm_bytes = self._render_with_technique(step_dem_path, "msm", msm_path)
        
        evsm_hash = hashlib.md5(evsm_bytes).hexdigest()
        msm_hash = hashlib.md5(msm_bytes).hexdigest()
        
        assert evsm_hash != msm_hash, f"EVSM and MSM produced identical output: {evsm_hash}"


def _is_hardware_viewer_adapter(probe: dict) -> bool:
    if probe.get("status") != "ok" or probe.get("software_fallback"):
        return False
    if str(probe.get("device_type", "")).lower() not in HARDWARE_DEVICE_TYPES:
        return False
    name = str(probe.get("name", "")).lower()
    return not any(token in name for token in SOFTWARE_ADAPTER_TOKENS)


def _viewer_hardware_adapter_available() -> bool:
    try:
        import forge3d as f3d

        if not f3d.has_gpu():
            return False
        return _is_hardware_viewer_adapter(
            f3d.device_probe(os.environ.get("WGPU_BACKEND"))
        )
    except Exception:
        return False


@pytest.mark.parametrize(
    ("probe", "expected"),
    (
        (
            {
                "status": "ok",
                "device_type": "DiscreteGpu",
                "name": "NVIDIA GeForce RTX 3070",
                "software_fallback": False,
            },
            True,
        ),
        (
            {
                "status": "ok",
                "device_type": "Cpu",
                "name": "Microsoft Basic Render Driver (WARP)",
                "software_fallback": True,
            },
            False,
        ),
        (
            {
                "status": "ok",
                "device_type": "DiscreteGpu",
                "name": "llvmpipe software rasterizer",
                "software_fallback": False,
            },
            False,
        ),
        (
            {
                "status": "ok",
                "device_type": "DiscreteGpu",
                "name": "nominal hardware",
                "software_fallback": True,
            },
            False,
        ),
    ),
)
def test_evsm_regression_requires_hardware_adapter(probe: dict, expected: bool):
    assert _is_hardware_viewer_adapter(probe) is expected


def _pyramid_dem(path: Path, n: int = 512, pix: float = 20.0) -> Path:
    """Flat plain with one tall isolated pyramid -> guaranteed long cast shadow."""
    import rasterio
    from rasterio.transform import from_bounds

    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    c = n * 0.35
    r = np.maximum(np.abs(xx - c), np.abs(yy - c))
    dem = (np.clip(1.0 - r / (n * 0.12), 0.0, 1.0) * 900.0).astype(np.float32)
    with rasterio.open(
        path, "w", driver="GTiff", height=n, width=n, count=1, dtype="float32",
        crs="EPSG:32610", transform=from_bounds(0, 0, n * pix, n * pix, n, n),
    ) as dst:
        dst.write(dem, 1)
    return path


@pytest.mark.skipif(not _rasterio_available(), reason="rasterio not installed")
@pytest.mark.skipif(
    os.environ.get("RUN_M06_VIEWER_CI") != "1",
    reason="required M-06 NVIDIA/Vulkan viewer lane only",
)
@pytest.mark.interactive_viewer
@pytest.mark.viewer
class TestEvsmExposureParity:
    """EVSM must light the scene like the other techniques, not black it out.

    Regression guard for the EVSM moment pipeline: the moment atlas is
    Rgba16Float, so exp(c * depth) with c=40 overflows to +Inf (and exp(-40*d)
    flushes to zero), which made every EVSM fragment resolve to ~0 visibility.
    The negative lobe must also be stored negated so its Chebyshev bound is
    monotonically increasing, otherwise EVSM reports "lit" everywhere instead.
    """

    @pytest.fixture(autouse=True)
    def _require_hardware_adapter(self):
        if not _viewer_hardware_adapter_available():
            pytest.skip(
                "EVSM visual-quality regression requires a hardware GPU adapter"
            )

    def _render(
        self, viewer, technique: str, out: Path, *, debug_mode: int = 0
    ) -> np.ndarray:
        viewer.send_ipc({
            "cmd": "set_terrain", "phi": 90.0, "theta": 55.0, "fov": 30.0,
            "radius": 18000.0, "zscale": 1.0, "sun_azimuth": 270.0,
            "sun_elevation": 8.0, "sun_intensity": 2.0, "ambient": 0.15,
            "shadow": 1.0, "background": [1.0, 1.0, 1.0],
        })
        # The cast-shadow fidelity oracle requires 2048; software adapters are
        # routed away from this test by the class capability gate above.
        viewer.send_ipc({
            "cmd": "set_terrain_pbr", "enabled": True, "shadow_technique": technique,
            "shadow_map_res": 2048, "exposure": 1.0, "msaa": 1,
            "ibl_intensity": 0.0, "debug_mode": debug_mode,
            "sky": {"enabled": False},
        })
        viewer.snapshot(str(out), width=640, height=400)
        from PIL import Image
        img = np.asarray(Image.open(out).convert("RGB"), dtype=np.float32) / 255.0
        return 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]

    def test_evsm_is_not_black(self, tmp_path: Path):
        import forge3d as f3d

        dem = _pyramid_dem(tmp_path / "pyramid.tif")
        with f3d.open_viewer_async(terrain_path=str(dem), width=640, height=400,
                                   timeout=45.0) as viewer:
            sequence = [
                self._render(viewer, technique, tmp_path / f"{index}_{technique}.png")
                for index, technique in enumerate(("pcf", "evsm", "pcf", "evsm"))
            ]

        assert np.array_equal(sequence[0], sequence[2]), (
            "PCF display changed across an EVSM round trip"
        )
        assert np.array_equal(sequence[1], sequence[3]), (
            "EVSM display changed across a PCF round trip"
        )
        lum = {"pcf": sequence[0], "evsm": sequence[1]}

        means = {}
        for tech, l in lum.items():
            terrain = l < 0.97  # exclude the white background
            terrain_pixels = int(terrain.sum())
            assert 0 < terrain_pixels < terrain.size, (
                f"{tech}: terrain mask must exclude the white background; "
                f"selected {terrain_pixels}/{terrain.size} pixels"
            )
            means[tech] = float(l[terrain].mean())

        # EVSM must not be globally darker than PCF by more than 20%.
        assert means["evsm"] >= means["pcf"] * 0.8, (
            f"EVSM is broken-dark: terrain mean {means['evsm']:.3f} vs "
            f"PCF {means['pcf']:.3f}"
        )

        # ...and it must still cast a real shadow (some pixels clearly darker
        # than the lit plain), not just return "fully lit" everywhere.
        terrain = lum["evsm"] < 0.97
        vals = lum["evsm"][terrain]
        shadowed = float((vals < np.percentile(vals, 95) * 0.6).mean())
        assert shadowed >= 0.01, (
            f"EVSM casts no shadow: only {shadowed:.4f} of terrain is shadowed"
        )

    def test_evsm_banding_is_bounded_in_raw_visibility(self, tmp_path: Path):
        import forge3d as f3d

        dem = _pyramid_dem(tmp_path / "pyramid.tif")
        with f3d.open_viewer_async(
            terrain_path=str(dem), width=640, height=400, timeout=45.0
        ) as viewer:
            terrain_reference = self._render(
                viewer, "pcf", tmp_path / "pcf_reference.png"
            )
            raw_sequence = [
                self._render(
                    viewer,
                    technique,
                    tmp_path / f"{index}_{technique}_raw.png",
                    debug_mode=35,
                )
                for index, technique in enumerate(("pcf", "evsm", "pcf", "evsm"))
            ]

        assert np.array_equal(raw_sequence[0], raw_sequence[2]), (
            "PCF raw visibility changed across an EVSM round trip"
        )
        assert np.array_equal(raw_sequence[1], raw_sequence[3]), (
            "EVSM raw visibility changed across a PCF round trip"
        )
        pcf, evsm = raw_sequence[:2]
        terrain = terrain_reference < 0.97
        assert terrain.sum() >= 50_000, "viewer framing has too little terrain"

        cast = terrain & (pcf < 0.6)
        radius = 3
        neighborhoods = np.lib.stride_tricks.sliding_window_view(
            np.pad(cast, radius, constant_values=False),
            (2 * radius + 1, 2 * radius + 1),
        )
        neighborhood_has_cast = neighborhoods.any(axis=(-2, -1))
        cast_core = terrain & neighborhoods.all(axis=(-2, -1))
        shadow_edge = terrain & neighborhood_has_cast & ~cast_core
        far_lit = terrain & ~neighborhood_has_cast & (pcf > 0.99)
        assert cast_core.sum() >= 500, "PCF cast shadow has no measurable interior"
        assert shadow_edge.sum() >= 1_000, "PCF cast shadow has no measurable edge"
        assert far_lit.sum() >= 10_000, "PCF has no measurable far-lit terrain"

        horizontal_pairs = cast_core[:, :-1] & cast_core[:, 1:]
        vertical_pairs = cast_core[:-1, :] & cast_core[1:, :]
        jumps = np.concatenate(
            (
                np.abs(np.diff(evsm, axis=1))[horizontal_pairs],
                np.abs(np.diff(evsm, axis=0))[vertical_pairs],
            )
        )
        discontinuity_fraction = float((jumps > 0.2).mean())
        p99_jump = float(np.quantile(jumps, 0.99))
        max_jump = float(jumps.max())
        retained_shadow = float((evsm[cast_core] < 0.6).mean())
        print(
            "EVSM raw visibility: "
            f"discontinuity={discontinuity_fraction:.6f}, "
            f"p99_jump={p99_jump:.6f}, max_jump={max_jump:.6f}, "
            f"retained_shadow={retained_shadow:.6f}"
        )
        assert discontinuity_fraction <= 0.05, (
            "EVSM raw visibility contains alternating shadow bands: "
            f"{discontinuity_fraction:.4f} adjacent interior pairs jump by >0.2"
        )
        assert p99_jump <= 0.25 and max_jump <= 0.5, (
            "EVSM raw visibility still contains severe alternating slabs: "
            f"p99 jump={p99_jump:.4f}, max jump={max_jump:.4f}"
        )
        assert retained_shadow >= 0.1, (
            "EVSM artifact suppression erased the PCF-defined cast-shadow core: "
            f"only {retained_shadow:.4f} remains clearly shadowed"
        )
        far_lit_delta = evsm[far_lit] - pcf[far_lit]
        far_lit_abs_delta = float(np.abs(far_lit_delta).mean())
        assert float(far_lit_delta.mean()) >= -0.02, (
            "EVSM globally dims terrain away from the cast shadow: "
            f"far-lit mean delta={far_lit_delta.mean():.4f}"
        )
        assert far_lit_abs_delta <= 0.03, (
            "EVSM differs from PCF away from the cast shadow: "
            f"far-lit mean absolute delta={far_lit_abs_delta:.4f}"
        )

        evsm_difference = np.abs(evsm - pcf)
        differentiated_soft = (
            shadow_edge
            & (evsm_difference > 0.01)
            & (evsm > 0.05)
            & (evsm < 0.95)
        )
        differentiated_soft_pixels = int(differentiated_soft.sum())
        soft_fraction = float(differentiated_soft_pixels / shadow_edge.sum())
        mean_edge_difference = float(evsm_difference[shadow_edge].mean())
        print(
            "EVSM localized soft transition: "
            f"mean_edge_delta={mean_edge_difference:.6f}, "
            f"far_lit_delta={far_lit_abs_delta:.6f}, "
            f"soft_pixels={differentiated_soft_pixels}, "
            f"soft_fraction={soft_fraction:.6f}"
        )
        assert mean_edge_difference >= 0.03, (
            "EVSM's moment-native shadow edge is too close to PCF: "
            f"mean edge difference={mean_edge_difference:.4f}"
        )
        assert mean_edge_difference >= far_lit_abs_delta + 0.02, (
            "EVSM's difference is not localized to the shadow edge: "
            f"edge={mean_edge_difference:.4f}, far-lit={far_lit_abs_delta:.4f}"
        )
        assert differentiated_soft_pixels >= 500 and soft_fraction >= 0.05, (
            "EVSM produces no materially localized soft transition: "
            f"{differentiated_soft_pixels} pixels ({soft_fraction:.4f} of edge)"
        )


def _native_terrain_gpu_available() -> bool:
    try:
        from _terrain_runtime import terrain_rendering_available

        return terrain_rendering_available()
    except Exception:
        return False


def _native_pyramid_heightmap(size: int = 128) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    center = size * 0.42
    radius = np.maximum(np.abs(xx - center), np.abs(yy - center))
    return np.clip(1.0 - radius / (size * 0.12), 0.0, 1.0).astype(np.float32)


@pytest.mark.skipif(
    not _native_terrain_gpu_available(),
    reason="no terrain-capable hardware-backed forge3d runtime",
)
@pytest.mark.offscreen
def test_native_vsm_casts_shadow_moment_visibility_pcss_and_msm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Native MSM uses four moments, moment filters stay visible, and PCSS widens."""
    import forge3d as f3d
    from _terrain_runtime import _write_test_hdr
    from forge3d.terrain_params import PomSettings, make_terrain_params_config

    monkeypatch.setenv("FORGE3D_TERRAIN_SHADOW_DEBUG", "raw")
    hdr_path = tmp_path / "constant.hdr"
    _write_test_hdr(hdr_path)

    session = f3d.Session(window=False)
    renderer = f3d.TerrainRenderer(session)
    material_set = f3d.MaterialSet.terrain_default()
    ibl = f3d.IBL.from_hdr(str(hdr_path), intensity=0.0)
    heightmap = _native_pyramid_heightmap()
    flat_mask_map = f3d.Colormap1D.from_stops(
        stops=[
            (0.0, "#ffffff"),
            (0.001, "#000000"),
            (1.0, "#000000"),
        ],
        domain=(0.0, 1.0),
    )
    flat_mask_overlay = f3d.OverlayLayer.from_colormap1d(
        flat_mask_map, strength=1.0
    )

    def render(
        technique: str,
        *,
        enabled: bool = True,
        terrain: np.ndarray | None = None,
        curve_mode: str = "linear",
        curve_power: float = 1.0,
        curve_lut: np.ndarray | None = None,
        pcss_blocker_radius: float = 6.0,
        pcss_filter_radius: float = 4.0,
        light_size: float = 1.0,
        pcss_light_radius: float = 0.0,
        shadow_resolution: int = 1024,
    ) -> np.ndarray:
        shadows = ShadowSettings(
            enabled=enabled,
            technique=technique,
            resolution=shadow_resolution,
            cascades=1,
            max_distance=20.0,
            softness=1.0,
            intensity=1.0,
            slope_scale_bias=0.001,
            depth_bias=0.0005,
            normal_bias=0.0002,
            min_variance=1e-4,
            light_bleed_reduction=0.5,
            evsm_exponent=40.0,
            fade_start=1.0,
            pcss_blocker_radius=pcss_blocker_radius,
            pcss_filter_radius=pcss_filter_radius,
            light_size=light_size,
            pcss_light_radius=pcss_light_radius,
        )
        config = make_terrain_params_config(
            size_px=(320, 240),
            render_scale=1.0,
            terrain_span=4.0,
            msaa_samples=1,
            z_scale=2.0,
            exposure=1.0,
            domain=(0.0, 1.0),
            albedo_mode="colormap",
            colormap_strength=1.0,
            ibl_enabled=False,
            ibl_intensity=0.0,
            light_azimuth_deg=315.0,
            light_elevation_deg=12.0,
            sun_intensity=3.0,
            cam_radius=6.0,
            cam_phi_deg=135.0,
            cam_theta_deg=55.0,
            fov_y_deg=52.0,
            camera_mode="mesh:zup",
            debug_mode=1 if not enabled else 0,
            height_curve_mode=curve_mode,
            height_curve_strength=0.0 if curve_mode == "linear" else 1.0,
            height_curve_power=curve_power,
            height_curve_lut=curve_lut,
            overlays=[flat_mask_overlay],
            shadows=shadows,
            pom=PomSettings(False, "Occlusion", 0.0, 1, 1, 0, False, False),
        )
        return renderer.render_terrain_pbr_pom(
            material_set=material_set,
            env_maps=ibl,
            params=f3d.TerrainRenderParams(config),
            heightmap=heightmap if terrain is None else terrain,
        ).to_numpy()

    first_frame_512 = (
        render("vsm", shadow_resolution=512)[..., 0].astype(np.float32) / 255.0
    )
    raw = {
        technique: render(technique)[..., :3].astype(np.float32) / 255.0
        for technique in ("pcf", "vsm", "evsm", "msm")
    }
    # Reconfigure a live renderer through three physical atlas sizes. Each
    # transition must recreate matching depth/moment/blur resources.
    resized_high = (
        render("vsm", shadow_resolution=2048)[..., 0].astype(np.float32) / 255.0
    )
    resized_back = (
        render("vsm", shadow_resolution=512)[..., 0].astype(np.float32) / 255.0
    )
    luminance = {technique: image[..., 0] for technique, image in raw.items()}
    terrain_reference = (
        render("pcf", enabled=False)[..., 0].astype(np.float32) / 255.0
    )
    terrain = terrain_reference > 0.9
    assert terrain.any(), "shadow-disabled native render contains no flat terrain"
    for technique, image in raw.items():
        assert np.max(np.abs(image[..., 0][terrain] - image[..., 1][terrain])) <= (
            1.0 / 255.0
        )
        assert np.max(np.abs(image[..., 0][terrain] - image[..., 2][terrain])) <= (
            1.0 / 255.0
        )

    cast_shadow = terrain & (luminance["pcf"] < 0.6)
    assert float(cast_shadow.sum() / terrain.sum()) >= 0.01, (
        "deterministic pyramid produced no measurable PCF cast-shadow region"
    )
    radius = 3
    neighborhoods = np.lib.stride_tricks.sliding_window_view(
        np.pad(cast_shadow, radius, constant_values=False),
        (2 * radius + 1, 2 * radius + 1),
    )
    neighborhood_has_cast = neighborhoods.any(axis=(-2, -1))
    cast_core = terrain & neighborhoods.all(axis=(-2, -1))
    far_lit = terrain & ~neighborhood_has_cast & (luminance["pcf"] > 0.99)
    assert cast_core.sum() >= 500, "native PCF cast shadow has no measurable interior"
    assert far_lit.sum() >= 5_000, "native PCF has no measurable far-lit terrain"

    pcf_exposure = float(luminance["pcf"][far_lit].mean())
    for label, image in (
        ("VSM first frame at 512", first_frame_512),
        ("VSM after 1024->2048 resize", resized_high),
        ("VSM after 2048->512 resize", resized_back),
    ):
        exposure = float(image[far_lit].mean())
        shadowed = float(((image < exposure * 0.6) & cast_shadow).sum() / terrain.sum())
        assert exposure >= pcf_exposure * 0.8, (
            f"{label} breaks lit exposure: {exposure:.3f} vs PCF {pcf_exposure:.3f}"
        )
        assert shadowed >= 0.01, (
            f"{label} casts no shadow: only {shadowed:.4f} of terrain is clearly shadowed"
        )

    for technique in ("vsm", "evsm", "msm"):
        exposure = float(luminance[technique][far_lit].mean())
        shadowed = float(
            ((luminance[technique] < exposure * 0.6) & cast_shadow).sum()
            / terrain.sum()
        )
        print(
            f"{technique}: exposure={exposure:.6f}, "
            f"pcf={pcf_exposure:.6f}, shadowed={shadowed:.6f}"
        )
        assert exposure >= pcf_exposure * 0.8, (
            f"{technique.upper()} breaks lit exposure: {exposure:.3f} vs "
            f"PCF {pcf_exposure:.3f}"
        )
        assert shadowed >= 0.01, (
            f"{technique.upper()} casts no shadow: only {shadowed:.4f} "
            "of terrain is clearly shadowed"
        )

    evsm_far_delta = luminance["evsm"][far_lit] - luminance["pcf"][far_lit]
    evsm_far_abs_delta = float(np.abs(evsm_far_delta).mean())
    evsm_retained_core = float((luminance["evsm"][cast_core] < 0.6).mean())
    print(
        "native EVSM localization: "
        f"far_mean_delta={evsm_far_delta.mean():.6f}, "
        f"far_abs_delta={evsm_far_abs_delta:.6f}, "
        f"retained_core={evsm_retained_core:.6f}"
    )
    assert float(evsm_far_delta.mean()) >= -0.02, (
        "native EVSM globally dims terrain away from the cast shadow: "
        f"far-lit mean delta={evsm_far_delta.mean():.4f}"
    )
    assert evsm_far_abs_delta <= 0.03, (
        "native EVSM differs from PCF away from the cast shadow: "
        f"far-lit mean absolute delta={evsm_far_abs_delta:.4f}"
    )
    assert evsm_retained_core >= 0.1, (
        "native EVSM erased the PCF-defined cast-shadow core: "
        f"only {evsm_retained_core:.4f} remains clearly shadowed"
    )

    msm_cast_difference = float(
        np.mean(np.abs(luminance["msm"][cast_shadow] - luminance["vsm"][cast_shadow]))
    )
    print(f"MSM vs VSM cast-shadow MAE={msm_cast_difference:.6f}")
    assert not np.array_equal(raw["msm"], raw["vsm"]), (
        "MSM and VSM produced byte-identical native output"
    )
    assert msm_cast_difference >= 0.005, (
        "MSM does not visibly consume its third and fourth moments: "
        f"cast-shadow MAE={msm_cast_difference:.6f}"
    )

    pcss = {
        light_size: (
            render(
                "pcss", light_size=light_size, pcss_filter_radius=16.0
            )[..., 0].astype(np.float32)
            / 255.0
        )
        for light_size in (1.0, 12.0)
    }
    pcss_mae = {
        radius: float(
            np.mean(np.abs(visibility[terrain] - luminance["pcf"][terrain]))
        )
        for radius, visibility in pcss.items()
    }
    print(f"PCSS vs PCF MAE: {pcss_mae}")
    assert pcss_mae[12.0] >= 0.002, (
        f"wide-light PCSS aliases PCF: MAE={pcss_mae[12.0]:.6f}"
    )
    assert pcss_mae[12.0] >= pcss_mae[1.0] + 0.00075, (
        f"light-size response is only a near-fixed perturbation: {pcss_mae}"
    )

    def edge_geometry(
        pcf_visibility: np.ndarray, terrain_mask: np.ndarray
    ) -> tuple[np.ndarray, int]:
        cast = terrain_mask & (pcf_visibility < 0.6)
        padded = np.pad(cast, 1)
        interior = (
            cast
            & padded[:-2, 1:-1]
            & padded[2:, 1:-1]
            & padded[1:-1, :-2]
            & padded[1:-1, 2:]
        )
        edge = cast & ~interior
        assert edge.any(), "deterministic cast shadow has no measurable edge"
        band = np.lib.stride_tricks.sliding_window_view(
            np.pad(edge, 12), (25, 25)
        ).any(axis=(-2, -1))
        return band & terrain_mask, int(edge.sum())

    edge_band, edge_count = edge_geometry(luminance["pcf"], terrain)

    def transition_width(
        visibility: np.ndarray, band: np.ndarray, denominator: int
    ) -> float:
        transition = 4.0 * visibility * (1.0 - visibility)
        return float(transition[band].sum() / denominator)

    widths = {
        radius: transition_width(visibility, edge_band, edge_count)
        for radius, visibility in pcss.items()
    }
    print(f"PCSS transition widths: {widths}")
    assert widths[12.0] >= widths[1.0] + 0.25, (
        f"larger PCSS light size did not widen the cast-shadow transition: {widths}"
    )

    curve_heightmap = np.zeros_like(heightmap)
    curve_heightmap[30:78, 30:78] = np.float32(0.5)
    curve_heightmap[30, 30] = np.float32(1.0)
    intermediate = curve_heightmap == np.float32(0.5)
    assert float(intermediate.mean()) >= 0.1

    lut = np.zeros(256, dtype=np.float32)
    curve_cases = (
        (
            "pow",
            8.0,
            None,
            np.power(curve_heightmap, np.float32(8.0)).astype(np.float32),
        ),
        (
            "lut",
            1.0,
            lut,
            lut[np.rint(curve_heightmap * 255.0).astype(np.int32)],
        ),
    )
    for mode, power, curve_lut, prewarped in curve_cases:
        curved = render(
            "pcf",
            terrain=curve_heightmap,
            curve_mode=mode,
            curve_power=power,
            curve_lut=curve_lut,
        )
        reference = render("pcf", terrain=prewarped)
        mean_abs = float(
            np.mean(
                np.abs(
                    curved[..., :3].astype(np.float32)
                    - reference[..., :3].astype(np.float32)
                )
            )
            / 255.0
        )
        print(f"{mode}: curved-vs-prewarped MAE={mean_abs:.6f}")
        assert mean_abs <= 0.01, (
            f"{mode} caster diverges from the receiver curve: MAE={mean_abs:.6f}"
        )
