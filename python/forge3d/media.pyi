from typing import Any, Mapping, Sequence, TypedDict

class ReferenceMediaDiagnostics(TypedDict):
    majorant_proof: Any
    majorant_valid: bool
    sample_count: int
    step_count: int
    temporal_history_decision: str
    temporal_history_reason: str
    host_visible_bytes: int
    froxel_device_local_bytes: int
    density_device_local_bytes: int
    majorant_device_local_bytes: int
    staging_readback_bytes: int
    adapter: str
    backend: str
    driver: str
    source_revision: str
    executed_multi_scatter: bool
    single_scatter_luminance: float | None
    multiple_scatter_luminance: float | None
    energy_accounting_residual: float | None

class RealtimeMediaDiagnostics(ReferenceMediaDiagnostics):
    sun_transmittance_method: str
    sun_transmittance_bias: str
    sun_transmittance_max_segment_length: float | None
    sun_transmittance_executed_steps: int
    sun_transmittance_max_abs_error: float

class VolumetricReferenceResult(TypedDict):
    beauty: Any
    transmittance: Any
    in_scatter: Any
    cloud_shadow: Any
    optical_depth: Any
    terrain_slice: Any
    terrain_hit: Any
    media_lighting_visibility: Any
    camera_contract: Mapping[str, Any]
    diagnostics: ReferenceMediaDiagnostics

class MediaError(ValueError): ...

class Medium:
    def __init__(self, sigma_a: Sequence[float], sigma_s: Sequence[float], density: float = ..., *, phase: str = ..., g: float = ..., density_scale: float = ..., version: int = ...) -> None: ...
    @classmethod
    def homogeneous(cls, sigma_a: Sequence[float], sigma_s: Sequence[float], density: float = ..., *, phase: str = ..., g: float = ..., density_scale: float = ..., version: int = ...) -> Medium: ...
    @classmethod
    def grid3d(cls, sigma_a: Sequence[float], sigma_s: Sequence[float], density: Any, bounds: tuple[Sequence[float], Sequence[float]], *, phase: str = ..., g: float = ..., density_scale: float = ..., version: int = ...) -> Medium: ...
    @classmethod
    def perlin_worley(cls, sigma_a: Sequence[float], sigma_s: Sequence[float], bounds: tuple[Sequence[float], Sequence[float]], *, frequency: float, octaves: int, seed: int, worley_weight: float, phase: str = ..., g: float = ..., density_scale: float = ..., version: int = ...) -> Medium: ...
    @property
    def sigma_a(self) -> tuple[float, float, float]: ...
    @property
    def sigma_s(self) -> tuple[float, float, float]: ...
    @property
    def sigma_t(self) -> tuple[float, float, float]: ...
    @property
    def version(self) -> int: ...
    @property
    def identity(self) -> str: ...
    def to_dict(self) -> dict[str, Any]: ...

def render_volumetric_reference(medium: Medium, heightmap: Any, width: int, height: int, camera: Mapping[str, Any], *, spacing: tuple[float, float] = ..., exaggeration: float = ..., albedo: tuple[float, float, float] = ..., sun_azimuth_deg: float = ..., sun_elevation_deg: float = ..., sun_intensity: float = ..., sun_color: tuple[float, float, float] = ..., environment_intensity: float = ..., samples_per_pixel: int = ..., homogeneous_medium_reach: float, clip: tuple[float, float] = ..., seed: int = ..., certificate: bool | str = ..., cache: Any = ...) -> VolumetricReferenceResult: ...
