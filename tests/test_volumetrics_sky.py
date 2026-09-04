"""M6: Tests for Volumetrics and Sky settings.

Tests the volumetrics and sky settings configuration.
"""
import pytest

from forge3d.terrain_params import (
    DensityVolumeSettings,
    VolumetricsSettings,
    SkySettings,
    localized_haze_volume,
    make_terrain_params_config,
    plume_volume,
    valley_fog_volume,
)


class TestVolumetricsSettings:
    """Tests for VolumetricsSettings dataclass."""

    def test_volumetrics_settings_default(self):
        """VolumetricsSettings should be disabled by default."""
        settings = VolumetricsSettings()
        assert settings.enabled is False
        assert settings.mode == "uniform"
        assert settings.density == 0.01
        assert settings.scattering == 0.5
        assert settings.absorption == 0.1
        assert settings.phase_g == 0.0
        assert settings.light_shafts is False
        assert settings.use_shadows is True
        assert settings.half_res is False
        assert settings.density_volumes == ()
        assert settings.uses_density_volumes is False

    def test_volumetrics_settings_enabled(self):
        """VolumetricsSettings can be enabled with custom parameters."""
        settings = VolumetricsSettings(
            enabled=True,
            mode="height",
            density=0.02,
            light_shafts=True,
        )
        assert settings.enabled is True
        assert settings.mode == "height"
        assert settings.density == 0.02
        assert settings.light_shafts is True

    def test_volumetrics_mode_validation(self):
        """VolumetricsSettings validates mode field."""
        # Valid modes
        VolumetricsSettings(mode="uniform")
        VolumetricsSettings(mode="height")
        VolumetricsSettings(mode="exponential")

        # Invalid mode
        with pytest.raises(ValueError, match="mode must be one of"):
            VolumetricsSettings(mode="invalid")

    def test_volumetrics_density_validation(self):
        """VolumetricsSettings validates density field."""
        VolumetricsSettings(density=0.0)
        VolumetricsSettings(density=1.0)

        with pytest.raises(ValueError, match="density must be >= 0"):
            VolumetricsSettings(density=-0.1)

    def test_volumetrics_scattering_validation(self):
        """VolumetricsSettings validates scattering field."""
        VolumetricsSettings(scattering=0.0)
        VolumetricsSettings(scattering=1.0)

        with pytest.raises(ValueError, match="scattering must be in"):
            VolumetricsSettings(scattering=-0.1)
        with pytest.raises(ValueError, match="scattering must be in"):
            VolumetricsSettings(scattering=1.1)

    def test_volumetrics_absorption_validation(self):
        """VolumetricsSettings validates absorption field."""
        VolumetricsSettings(absorption=0.0)
        VolumetricsSettings(absorption=1.0)

        with pytest.raises(ValueError, match="absorption must be in"):
            VolumetricsSettings(absorption=-0.1)
        with pytest.raises(ValueError, match="absorption must be in"):
            VolumetricsSettings(absorption=1.1)

    def test_volumetrics_phase_g_validation(self):
        """VolumetricsSettings validates phase_g field."""
        VolumetricsSettings(phase_g=-1.0)
        VolumetricsSettings(phase_g=0.0)
        VolumetricsSettings(phase_g=1.0)

        with pytest.raises(ValueError, match="phase_g must be in"):
            VolumetricsSettings(phase_g=-1.1)
        with pytest.raises(ValueError, match="phase_g must be in"):
            VolumetricsSettings(phase_g=1.1)

    def test_volumetrics_shaft_samples_validation(self):
        """VolumetricsSettings validates shaft_samples field."""
        VolumetricsSettings(shaft_samples=8)
        VolumetricsSettings(shaft_samples=64)
        VolumetricsSettings(shaft_samples=128)

        with pytest.raises(ValueError, match="shaft_samples must be in"):
            VolumetricsSettings(shaft_samples=4)
        with pytest.raises(ValueError, match="shaft_samples must be in"):
            VolumetricsSettings(shaft_samples=256)

    def test_volumetrics_has_light_shafts_property(self):
        """has_light_shafts property returns correct values."""
        settings = VolumetricsSettings(light_shafts=False)
        assert settings.has_light_shafts is False

        settings = VolumetricsSettings(light_shafts=True, shaft_intensity=0.0)
        assert settings.has_light_shafts is False

        settings = VolumetricsSettings(light_shafts=True, shaft_intensity=1.0)
        assert settings.has_light_shafts is True

    def test_volumetrics_density_volumes_validate_and_export_for_viewer(self):
        """VolumetricsSettings enforces TV6 limits and exports viewer IPC payloads."""
        fog = valley_fog_volume(center=(32.0, 18.0, 40.0), size=(64.0, 24.0, 72.0), seed=3)
        settings = VolumetricsSettings(
            enabled=True,
            mode="height",
            light_shafts=True,
            shaft_samples=48,
            density_volumes=(fog,),
        )

        assert settings.uses_density_volumes is True
        viewer_payload = settings.to_viewer_dict()
        assert viewer_payload["steps"] == 48
        assert viewer_payload["density_volumes"][0]["preset"] == "valley_fog"
        assert viewer_payload["density_volumes"][0]["seed"] == 3

        too_many = tuple(
            DensityVolumeSettings(center=(float(i), 0.0, 0.0))
            for i in range(DensityVolumeSettings.MAX_ACTIVE_VOLUMES + 1)
        )
        with pytest.raises(ValueError, match="supports at most"):
            VolumetricsSettings(density_volumes=too_many)


class TestDensityVolumeSettings:
    """Tests for TV6 density volume configuration."""

    def test_density_volume_settings_default(self):
        """DensityVolumeSettings should expose a stable bounded default."""
        settings = DensityVolumeSettings()
        assert settings.preset == "valley_fog"
        assert settings.center == (0.0, 0.0, 0.0)
        assert settings.size == (128.0, 64.0, 128.0)
        assert settings.resolution == (64, 32, 64)
        assert settings.density_scale == 1.0
        assert settings.wind == (0.25, 1.0, 0.0)

    def test_density_volume_settings_validation(self):
        """DensityVolumeSettings validates preset, bounds, and seed."""
        with pytest.raises(ValueError, match="preset must be one of"):
            DensityVolumeSettings(preset="steam")
        with pytest.raises(ValueError, match=r"size.x must be > 0"):
            DensityVolumeSettings(size=(0.0, 64.0, 64.0))
        with pytest.raises(ValueError, match=r"resolution.x must be in \[8, 96\]"):
            DensityVolumeSettings(resolution=(4, 32, 32))
        with pytest.raises(ValueError, match="seed must be >= 0"):
            DensityVolumeSettings(seed=-1)

    def test_density_volume_helper_constructors(self):
        """TV6 helper constructors should stamp the correct preset wiring."""
        fog = valley_fog_volume(center=(50.0, 12.0, 60.0), size=(180.0, 48.0, 180.0), seed=7)
        assert fog.preset == "valley_fog"
        assert fog.floor_offset == 4.0
        assert fog.seed == 7

        plume = plume_volume(
            center=(64.0, 120.0, 64.0),
            size=(56.0, 220.0, 56.0),
            plume_spread=0.55,
            wind=(0.4, 1.0, -0.2),
        )
        assert plume.preset == "plume"
        assert plume.plume_spread == 0.55
        assert plume.wind == (0.4, 1.0, -0.2)

        haze = localized_haze_volume(
            center=(72.0, 80.0, 44.0),
            size=(96.0, 40.0, 96.0),
            ceiling=0.7,
            density_scale=0.9,
        )
        assert haze.preset == "localized_haze"
        assert haze.ceiling == 0.7
        assert haze.density_scale == 0.9


class TestSkySettings:
    """Tests for SkySettings dataclass."""

    def test_sky_settings_default(self):
        """SkySettings should be disabled by default."""
        settings = SkySettings()
        assert settings.enabled is False
        assert settings.model == "hosek-wilkie"
        assert settings.turbidity == 2.0
        assert settings.ground_albedo == 0.3
        assert settings.sun_intensity == 1.0
        assert settings.sun_size == 1.0
        assert settings.aerial_perspective is True
        assert settings.aerial_density == 1.0
        assert settings.sky_exposure == 1.0

    def test_sky_settings_enabled(self):
        """SkySettings can be enabled with custom parameters."""
        settings = SkySettings(
            enabled=True,
            turbidity=4.0,
            sun_intensity=2.0,
        )
        assert settings.enabled is True
        assert settings.turbidity == 4.0
        assert settings.sun_intensity == 2.0

    def test_sky_model_validation(self):
        """SkySettings validates explicit sky model selection."""
        SkySettings(model="hosek-wilkie")
        SkySettings(model="preetham")
        SkySettings(model="approximate")

        with pytest.raises(ValueError, match="model must be one of"):
            SkySettings(model="rayleigh")

    def test_sky_turbidity_validation(self):
        """SkySettings validates turbidity field."""
        SkySettings(turbidity=1.0)
        SkySettings(turbidity=5.0)
        SkySettings(turbidity=10.0)

        with pytest.raises(ValueError, match="turbidity must be in"):
            SkySettings(turbidity=0.5)
        with pytest.raises(ValueError, match="turbidity must be in"):
            SkySettings(turbidity=11.0)

    def test_sky_ground_albedo_validation(self):
        """SkySettings validates ground_albedo field."""
        SkySettings(ground_albedo=0.0)
        SkySettings(ground_albedo=0.5)
        SkySettings(ground_albedo=1.0)

        with pytest.raises(ValueError, match="ground_albedo must be in"):
            SkySettings(ground_albedo=-0.1)
        with pytest.raises(ValueError, match="ground_albedo must be in"):
            SkySettings(ground_albedo=1.1)

    def test_sky_sun_intensity_validation(self):
        """SkySettings validates sun_intensity field."""
        SkySettings(sun_intensity=0.0)
        SkySettings(sun_intensity=2.0)

        with pytest.raises(ValueError, match="sun_intensity"):
            SkySettings(sun_intensity=-0.1)

    def test_sky_has_aerial_perspective_property(self):
        """has_aerial_perspective property returns correct values."""
        settings = SkySettings(aerial_perspective=False)
        assert settings.has_aerial_perspective is False

        settings = SkySettings(aerial_perspective=True, aerial_density=0.0)
        assert settings.has_aerial_perspective is False

        settings = SkySettings(aerial_perspective=True, aerial_density=1.0)
        assert settings.has_aerial_perspective is True


class TestTerrainRenderParamsWithVolumetricsSky:
    """Tests for TerrainRenderParams with volumetrics and sky settings."""

    def test_terrain_params_default_volumetrics(self):
        """TerrainRenderParams should have disabled volumetrics by default."""
        params = make_terrain_params_config(
            size_px=(256, 256),
            render_scale=1.0,
            terrain_span=1000.0,
            msaa_samples=1,
            z_scale=1.0,
            exposure=1.0,
            domain=(0.0, 100.0),
        )
        assert params.volumetrics is not None
        assert params.volumetrics.enabled is False

    def test_terrain_params_default_sky(self):
        """TerrainRenderParams should have disabled sky by default."""
        params = make_terrain_params_config(
            size_px=(256, 256),
            render_scale=1.0,
            terrain_span=1000.0,
            msaa_samples=1,
            z_scale=1.0,
            exposure=1.0,
            domain=(0.0, 100.0),
        )
        assert params.sky is not None
        assert params.sky.enabled is False

    def test_terrain_params_with_volumetrics_enabled(self):
        """TerrainRenderParams can have volumetrics enabled."""
        vol = VolumetricsSettings(
            enabled=True,
            mode="height",
            density=0.02,
            light_shafts=True,
        )
        params = make_terrain_params_config(
            size_px=(256, 256),
            render_scale=1.0,
            terrain_span=1000.0,
            msaa_samples=1,
            z_scale=1.0,
            exposure=1.0,
            domain=(0.0, 100.0),
            volumetrics=vol,
        )
        assert params.volumetrics is not None
        assert params.volumetrics.enabled is True
        assert params.volumetrics.mode == "height"
        assert params.volumetrics.light_shafts is True

    def test_terrain_params_with_sky_enabled(self):
        """TerrainRenderParams can have sky enabled."""
        sky = SkySettings(
            enabled=True,
            turbidity=4.0,
            sun_intensity=2.0,
        )
        params = make_terrain_params_config(
            size_px=(256, 256),
            render_scale=1.0,
            terrain_span=1000.0,
            msaa_samples=1,
            z_scale=1.0,
            exposure=1.0,
            domain=(0.0, 100.0),
            sky=sky,
        )
        assert params.sky is not None
        assert params.sky.enabled is True
        assert params.sky.turbidity == 4.0
        assert params.sky.sun_intensity == 2.0


class TestVolumetricsPhysics:
    """Tests for volumetrics physics."""

    def test_fog_modes(self):
        """Test different fog density modes."""
        uniform = VolumetricsSettings(mode="uniform", density=0.01)
        assert uniform.mode == "uniform"

        height = VolumetricsSettings(mode="height", density=0.01, height_falloff=0.2)
        assert height.mode == "height"

        exponential = VolumetricsSettings(mode="exponential", density=0.01)
        assert exponential.mode == "exponential"

    def test_phase_function_directions(self):
        """Test Henyey-Greenstein phase function parameters."""
        # Isotropic scattering
        iso = VolumetricsSettings(phase_g=0.0)
        assert iso.phase_g == 0.0

        # Forward scattering (like fog/haze)
        forward = VolumetricsSettings(phase_g=0.8)
        assert forward.phase_g == 0.8

        # Back scattering
        back = VolumetricsSettings(phase_g=-0.5)
        assert back.phase_g == -0.5


class TestSkyPhysics:
    """Tests for sky physics."""

    def test_turbidity_atmospheric_conditions(self):
        """Test turbidity values for different conditions."""
        # Clear sky
        clear = SkySettings(turbidity=1.5)
        assert clear.turbidity == 1.5

        # Average conditions
        average = SkySettings(turbidity=3.0)
        assert average.turbidity == 3.0

        # Hazy/polluted
        hazy = SkySettings(turbidity=8.0)
        assert hazy.turbidity == 8.0

    def test_aerial_perspective_distance(self):
        """Test aerial perspective density settings."""
        # Weak aerial perspective
        weak = SkySettings(aerial_perspective=True, aerial_density=0.5)
        assert weak.has_aerial_perspective is True

        # Strong aerial perspective
        strong = SkySettings(aerial_perspective=True, aerial_density=2.0)
        assert strong.has_aerial_perspective is True

        # Disabled
        off = SkySettings(aerial_perspective=False)
        assert off.has_aerial_perspective is False
