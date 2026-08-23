from pathlib import Path
from collections.abc import Mapping
from typing import Any, Callable, Optional, Sequence, TypeAlias

from .bundle import LoadedBundle
from .diagnostics import Diagnostic
from .geo import SolarTime

WorldPosition: TypeAlias = tuple[float, float, float]
VectorOverlayVertex: TypeAlias = tuple[float, float, float, float, float, float, float, int]
NormalizedExtent: TypeAlias = tuple[float, float, float, float]


class LabelBatchResult:
    ids: list[Optional[int]]
    diagnostics: list[Diagnostic]


class LabelOperationResult:
    ok: bool
    diagnostics: list[Diagnostic]
    state: dict[str, Any]


class ViewerError(Exception): ...


class ViewerHandle:
    def __init__(
        self,
        process: Any,
        host: str,
        port: int,
        timeout: float = ...,
        cleanup_paths: Optional[list[Path]] = ...,
    ) -> None: ...
    def send_ipc(self, cmd: dict[str, Any]) -> dict[str, Any]: ...
    def load_label_atlas(
        self,
        atlas_png_path: str | Path,
        metrics_json_path: str | Path,
    ) -> LabelOperationResult: ...
    def add_label(
        self,
        text: str,
        world_pos: WorldPosition,
        size: Optional[float] = ...,
        color: Optional[tuple[float, float, float, float]] = ...,
        halo_color: Optional[tuple[float, float, float, float]] = ...,
        halo_width: Optional[float] = ...,
        priority: Optional[int] = ...,
        min_zoom: Optional[float] = ...,
        max_zoom: Optional[float] = ...,
        offset: Optional[tuple[float, float]] = ...,
        rotation: Optional[float] = ...,
        underline: Optional[bool] = ...,
        small_caps: Optional[bool] = ...,
        leader: Optional[bool] = ...,
        horizon_fade_angle: Optional[float] = ...,
    ) -> int | LabelOperationResult: ...
    def add_labels(self, labels: Sequence[dict[str, Any]]) -> LabelBatchResult: ...
    def add_line_label(
        self,
        text: str,
        polyline: Sequence[WorldPosition],
        size: Optional[float] = ...,
        color: Optional[tuple[float, float, float, float]] = ...,
        halo_color: Optional[tuple[float, float, float, float]] = ...,
        halo_width: Optional[float] = ...,
        priority: Optional[int] = ...,
        placement: str = ...,
        repeat_distance: Optional[float] = ...,
        min_zoom: Optional[float] = ...,
        max_zoom: Optional[float] = ...,
        terrain_mode: Optional[str] = ...,
    ) -> int | LabelOperationResult: ...
    def add_curved_label(
        self,
        text: str,
        path: Sequence[WorldPosition],
        *,
        size: Optional[float] = ...,
        color: Optional[tuple[float, float, float, float]] = ...,
        halo_color: Optional[tuple[float, float, float, float]] = ...,
        halo_width: Optional[float] = ...,
        priority: Optional[int] = ...,
        tracking: Optional[float] = ...,
        center_on_path: Optional[bool] = ...,
    ) -> LabelOperationResult: ...
    def add_callout(
        self,
        text: str,
        anchor: WorldPosition,
        offset: tuple[float, float] = ...,
        background_color: Optional[tuple[float, float, float, float]] = ...,
        border_color: Optional[tuple[float, float, float, float]] = ...,
        border_width: Optional[float] = ...,
        corner_radius: Optional[float] = ...,
        padding: Optional[float] = ...,
        text_size: Optional[float] = ...,
        text_color: Optional[tuple[float, float, float, float]] = ...,
    ) -> int | LabelOperationResult: ...
    def add_vector_overlay(
        self,
        name: str,
        vertices: Sequence[VectorOverlayVertex],
        indices: Sequence[int],
        primitive: str = ...,
        drape: bool = ...,
        drape_offset: float = ...,
        opacity: float = ...,
        depth_bias: float = ...,
        line_width: float = ...,
        point_size: float = ...,
        z_order: int = ...,
    ) -> int: ...
    def set_labels_enabled(self, enabled: bool) -> LabelOperationResult: ...
    def clear_labels(self) -> LabelOperationResult: ...
    def remove_label(self, label_id: int) -> LabelOperationResult: ...
    def set_label_typography(
        self,
        *,
        tracking: Optional[float] = ...,
        kerning: Optional[bool] = ...,
        line_height: Optional[float] = ...,
        word_spacing: Optional[float] = ...,
    ) -> LabelOperationResult: ...
    def set_declutter_algorithm(
        self,
        algorithm: str,
        *,
        seed: Optional[int] = ...,
        max_iterations: Optional[int] = ...,
    ) -> LabelOperationResult: ...
    def label_configuration_state(self) -> dict[str, Any]: ...
    def get_stats(self) -> dict[str, Any]: ...
    def poll_pick_events(self) -> list[dict[str, Any]]: ...
    def pick_at(
        self,
        x: int,
        y: int,
        *,
        shift: bool = ...,
        ctrl: bool = ...,
    ) -> list[dict[str, Any]]: ...
    def update_labels(self) -> None: ...
    def get_terrain_volumetrics_report(self) -> dict[str, Any]: ...
    def load_obj(self, path: str | Path) -> None: ...
    def load_gltf(self, path: str | Path) -> None: ...
    def load_terrain(self, path: str | Path) -> None: ...
    def load_bundle(
        self,
        path_or_bundle: str | Path | LoadedBundle,
        variant_id: Optional[str] = ...,
    ) -> LoadedBundle: ...
    def load_overlay(
        self,
        name: str,
        path: str | Path,
        extent: Optional[NormalizedExtent] = ...,
        opacity: Optional[float] = ...,
        z_order: Optional[int] = ...,
        preserve_colors: Optional[bool] = ...,
    ) -> None: ...
    def load_point_cloud(
        self,
        path: str | Path,
        point_size: float = ...,
        max_points: int = ...,
        color_mode: Optional[str] = ...,
    ) -> None: ...
    def set_point_cloud_params(
        self,
        point_size: Optional[float] = ...,
        visible: Optional[bool] = ...,
        color_mode: Optional[str] = ...,
    ) -> None: ...
    def set_transform(
        self,
        translation: Optional[WorldPosition] = ...,
        rotation_quat: Optional[tuple[float, float, float, float]] = ...,
        scale: Optional[tuple[float, float, float]] = ...,
    ) -> None: ...
    def set_camera_lookat(
        self,
        eye: WorldPosition,
        target: WorldPosition,
        up: WorldPosition = ...,
    ) -> None: ...
    def set_fov(self, deg: float) -> None: ...
    def set_sun(self, azimuth_deg: float, elevation_deg: float) -> None: ...
    def set_sun_time(
        self,
        solar_time: SolarTime | Mapping[str, object],
        intensity: float = ...,
    ) -> None: ...
    def set_ibl(self, path: str | Path, intensity: float = ...) -> None: ...
    def set_z_scale(self, value: float) -> None: ...
    def set_terrain_scatter(self, batches: list[dict[str, Any]]) -> None: ...
    def clear_terrain_scatter(self) -> None: ...
    def list_scene_variants(self) -> list[dict[str, Any]]: ...
    def list_review_layers(self) -> list[dict[str, Any]]: ...
    def get_active_scene_variant(self) -> Optional[str]: ...
    def apply_scene_variant(self, variant_id: str) -> None: ...
    def set_review_layer_visible(self, layer_id: str, visible: bool) -> None: ...
    def set_orbit_camera(
        self,
        phi_deg: float,
        theta_deg: float,
        radius: float,
        fov_deg: Optional[float] = ...,
        target: Optional[WorldPosition] = ...,
    ) -> None: ...
    def snapshot(
        self,
        path: str | Path,
        width: Optional[int] = ...,
        height: Optional[int] = ...,
    ) -> None: ...
    def render_animation(
        self,
        animation: Any,
        output_dir: str | Path,
        fps: int = ...,
        width: Optional[int] = ...,
        height: Optional[int] = ...,
        progress_callback: Optional[Callable[[int, int], None]] = ...,
    ) -> None: ...
    def close(self) -> None: ...
    def __enter__(self) -> "ViewerHandle": ...
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None: ...
    @property
    def port(self) -> int: ...
    @property
    def is_running(self) -> bool: ...


def set_msaa(samples: int) -> int: ...


def open_viewer_async(
    width: int = ...,
    height: int = ...,
    title: str = ...,
    obj_path: Optional[str | Path] = ...,
    gltf_path: Optional[str | Path] = ...,
    terrain_path: Optional[str | Path] = ...,
    fov_deg: float = ...,
    timeout: float = ...,
    ipc_host: str = ...,
    ipc_port: int = ...,
) -> ViewerHandle: ...


def open_viewer(
    width: int = ...,
    height: int = ...,
    title: str = ...,
    obj_path: Optional[str | Path] = ...,
    gltf_path: Optional[str | Path] = ...,
    fov_deg: float = ...,
    snapshot_path: Optional[str] = ...,
    snapshot_width: Optional[int] = ...,
    snapshot_height: Optional[int] = ...,
    initial_commands: Optional[Sequence[str]] = ...,
) -> int: ...
