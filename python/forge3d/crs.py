# python/forge3d/crs.py
# CRS reprojection utilities on the built-in pure-Rust MENSURA engine
# RELEVANT FILES: src/gis/crs.rs, src/geo/projections/mod.rs, python/forge3d/render.py

"""
CRS (Coordinate Reference System) utilities for forge3d.

Provides reprojection of coordinates between different coordinate systems.
Coordinate transforms use the built-in pure-Rust MENSURA projection engine
(``forge3d._forge3d.CrsTransform``) exclusively; an unsupported CRS raises
rather than silently falling back to pyproj. pyproj, if installed, is used
only for WKT/EPSG metadata lookups, never as a transform backend.

Example:
    >>> from forge3d.crs import transform_coords, reproject_geom
    >>> import numpy as np
    >>>
    >>> # Transform WGS84 coordinates to UTM zone 54N
    >>> wgs84 = np.array([[138.73, 35.36], [138.74, 35.37]])
    >>> utm = transform_coords(wgs84, "EPSG:4326", "EPSG:32654")
    >>> print(utm)  # UTM coordinates in meters
"""

from __future__ import annotations

from typing import Any, Optional, Union, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from shapely.geometry.base import BaseGeometry

# Try to import pyproj for fallback
# We require pyproj >= 2.0 for Transformer.from_crs() API
try:
    import pyproj
    # Check for modern API (pyproj >= 2.0)
    if hasattr(pyproj, 'Transformer'):
        HAS_PYPROJ = True
        HAS_PYPROJ_LEGACY = False
    elif hasattr(pyproj, 'Proj'):
        # Old pyproj API (< 2.0)
        HAS_PYPROJ = False
        HAS_PYPROJ_LEGACY = True
    else:
        HAS_PYPROJ = False
        HAS_PYPROJ_LEGACY = False
except ImportError:
    HAS_PYPROJ = False
    HAS_PYPROJ_LEGACY = False
    pyproj = None  # type: ignore

# The built-in pure-Rust MENSURA projection engine ships in the wheel and is
# always available for coordinate transforms; the optional `proj` Cargo feature
# is only a dev-time differential oracle and is never a runtime fallback. pyproj
# (imported above) is used only for WKT/EPSG metadata lookups.


def _crs_transform(from_crs, to_crs, *, always_xy: bool = True):
    """Build a native pure-Rust CRS transformer, raising on unsupported pairs.

    This is the single MENSURA transform entry point for `forge3d.crs`. No
    pyproj fallback: an unsupported CRS raises InvalidCrs/BackendUnavailable.
    """
    from forge3d import _forge3d as _native

    return _native.CrsTransform.from_crs(from_crs, to_crs, always_xy=always_xy)


def proj_available() -> bool:
    """Report availability of the shipped pure-Rust CRS transform engine.

    MENSURA always provides the curated native transform surface. Unsupported
    CRS pairs raise explicitly; availability does not imply that every EPSG or
    free-form pipeline is supported.
    """
    return True


def transform_coords(
    coords: np.ndarray,
    from_crs: str,
    to_crs: str,
    *,
    always_xy: bool = True,
) -> np.ndarray:
    """Transform an array of coordinates from one CRS to another.

    Parameters
    ----------
    coords : np.ndarray
        Array of shape (N, 2) with [x, y] or [lon, lat] coordinates.
    from_crs : str
        Source CRS as EPSG code (e.g., "EPSG:4326") or PROJ string.
    to_crs : str
        Target CRS as EPSG code or PROJ string.
    always_xy : bool
        If True, coordinates are always interpreted as (x, y) / (lon, lat).
        This matches the pyproj always_xy=True convention. Default: True.

    Returns
    -------
    np.ndarray
        Transformed coordinates with shape (N, 2).

    Raises
    ------
    ValueError
        If ``coords`` does not have shape (N, 2).
    RuntimeError
        If the source or destination CRS is unsupported by the built-in
        MENSURA engine. The transform never silently falls back to pyproj.

    Example
    -------
    >>> wgs84 = np.array([[138.73, 35.36]])  # lon, lat
    >>> utm = transform_coords(wgs84, "EPSG:4326", "EPSG:32654")
    >>> print(utm)  # [[x_utm, y_utm]]
    """
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"coords must have shape (N, 2), got {coords.shape}")

    if len(coords) == 0:
        return coords.copy()

    # Skip if same CRS
    if _crs_equal(from_crs, to_crs):
        return coords.copy()

    # MENSURA: transform through the built-in pure-Rust engine only. An
    # unsupported/unregistered CRS raises rather than silently falling back to
    # pyproj. This is the same native-only path as ``reproject_geom``.
    transformer = _crs_transform(from_crs, to_crs, always_xy=always_xy)
    result = np.empty_like(coords)
    for i in range(coords.shape[0]):
        tx, ty = transformer.transform_point(float(coords[i, 0]), float(coords[i, 1]))
        result[i, 0] = tx
        result[i, 1] = ty
    return result


def reproject_geom(
    geom: BaseGeometry,
    from_crs: str,
    to_crs: str,
) -> BaseGeometry:
    """Reproject a Shapely geometry from one CRS to another.

    Parameters
    ----------
    geom : BaseGeometry
        Shapely geometry (Point, LineString, Polygon, etc.).
    from_crs : str
        Source CRS as EPSG code or PROJ string.
    to_crs : str
        Target CRS as EPSG code or PROJ string.

    Returns
    -------
    BaseGeometry
        Reprojected geometry of the same type.

    Example
    -------
    >>> from shapely.geometry import Point
    >>> pt_wgs84 = Point(138.73, 35.36)
    >>> pt_utm = reproject_geom(pt_wgs84, "EPSG:4326", "EPSG:32654")
    """
    try:
        from shapely import ops as shapely_ops
    except ImportError as e:
        raise ImportError("shapely is required for geometry reprojection") from e

    # Skip if same CRS
    if _crs_equal(from_crs, to_crs):
        return geom

    # MENSURA: reproject through the built-in pure-Rust engine only (no pyproj).
    transformer = _crs_transform(from_crs, to_crs, always_xy=True)

    def _transform(xs, ys):
        rx: list[float] = []
        ry: list[float] = []
        for x, y in zip(xs, ys):
            tx, ty = transformer.transform_point(float(x), float(y))
            rx.append(tx)
            ry.append(ty)
        return rx, ry

    return shapely_ops.transform(_transform, geom)


def parse_crs_from_wkt(wkt: str) -> Optional[str]:
    """Parse a CRS from WKT string and return EPSG code if possible.

    Parameters
    ----------
    wkt : str
        Well-Known Text CRS definition.

    Returns
    -------
    Optional[str]
        EPSG code (e.g., "EPSG:4326") if recognized, or the original WKT.
    """
    if not HAS_PYPROJ and not HAS_PYPROJ_LEGACY:
        return None

    if HAS_PYPROJ:
        try:
            crs = pyproj.CRS.from_wkt(wkt)
            epsg = crs.to_epsg()
            if epsg:
                return f"EPSG:{epsg}"
            return wkt
        except Exception:
            return None

    # Legacy pyproj doesn't have CRS.from_wkt
    return None


def crs_to_epsg(crs: Union[str, "pyproj.CRS"]) -> Optional[int]:
    """Convert a CRS to its EPSG code if possible.

    Parameters
    ----------
    crs : str or pyproj.CRS
        CRS as string or pyproj CRS object.

    Returns
    -------
    Optional[int]
        EPSG code or None if not determinable.
    """
    if isinstance(crs, str):
        if crs.upper().startswith("EPSG:"):
            try:
                return int(crs.split(":")[1])
            except (ValueError, IndexError):
                return None

        if HAS_PYPROJ:
            try:
                return pyproj.CRS.from_user_input(crs).to_epsg()
            except Exception:
                return None
        # Legacy pyproj doesn't have CRS.from_user_input
        return None

    if HAS_PYPROJ and hasattr(crs, "to_epsg"):
        return crs.to_epsg()

    return None


def _crs_equal(crs1: str, crs2: str) -> bool:
    """Check if two CRS strings refer to the same coordinate system."""
    if crs1 == crs2:
        return True

    # Compare EPSG codes
    code1 = crs_to_epsg(crs1)
    code2 = crs_to_epsg(crs2)
    if code1 is not None and code2 is not None:
        return code1 == code2

    return False


def get_crs_from_rasterio(path: str) -> Optional[str]:
    """Get CRS from a rasterio dataset (GeoTIFF, etc.).

    Parameters
    ----------
    path : str
        Path to raster file.

    Returns
    -------
    Optional[str]
        EPSG code or WKT, or None if not determinable.
    """
    try:
        import rasterio
        with rasterio.open(path) as ds:
            if ds.crs is None:
                return None
            # Try to get EPSG code
            epsg = ds.crs.to_epsg()
            if epsg:
                return f"EPSG:{epsg}"
            # Fall back to WKT
            return ds.crs.to_wkt()
    except Exception:
        return None


def get_crs_from_geopandas(path: str, layer: Optional[str] = None) -> Optional[str]:
    """Get CRS from a geopandas-readable file.

    Parameters
    ----------
    path : str
        Path to vector file (GeoJSON, Shapefile, GeoPackage, etc.).
    layer : Optional[str]
        Layer name for multi-layer datasets.

    Returns
    -------
    Optional[str]
        EPSG code or WKT, or None if not determinable.
    """
    try:
        import geopandas as gpd
        gdf = gpd.read_file(path, layer=layer) if layer else gpd.read_file(path)
        if gdf.crs is None:
            return None
        epsg = gdf.crs.to_epsg()
        if epsg:
            return f"EPSG:{epsg}"
        return gdf.crs.to_wkt()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# MENSURA: EGM96 geoid, Karney geodesics, geodetic <-> ECEF (pure-Rust native)
# ---------------------------------------------------------------------------

def _require_geodesy_native() -> Any:
    try:
        from forge3d import _forge3d as _native
    except ImportError as exc:  # pragma: no cover - native build required
        raise RuntimeError(
            "forge3d native extension is required for forge3d.crs geodesy functions"
        ) from exc
    return _native


def body_info(name: str) -> dict[str, Any]:
    """Return registered planetary datum constants with explicit units."""
    return _require_geodesy_native().body_info(name)


def geoid_undulation(lat: float, lon: float) -> float:
    """EGM96 geoid undulation N(lat, lon) in metres.

    Degree/order-120 spherical-harmonic synthesis following NGA's F477
    reference convention (WGS84 ellipsoid, correction model, -0.53 m
    zero-degree term).
    """
    return _require_geodesy_native().geoid_undulation(lat, lon)


def areoid_undulation(lat: float, lon: float) -> float:
    """GMM3 Mars areoid undulation above its reference ellipsoid, metres."""
    return _require_geodesy_native().areoid_undulation(lat, lon)


def orthometric_to_ellipsoidal(h_orthometric: float, lat: float, lon: float) -> float:
    """Convert an orthometric (EGM96) height to ellipsoidal: h = H + N."""
    return _require_geodesy_native().orthometric_to_ellipsoidal(h_orthometric, lat, lon)


def ellipsoidal_to_orthometric(h_ellipsoidal: float, lat: float, lon: float) -> float:
    """Convert an ellipsoidal height to orthometric (EGM96): H = h - N."""
    return _require_geodesy_native().ellipsoidal_to_orthometric(h_ellipsoidal, lat, lon)


def geodesic_inverse(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
    *,
    body: str = "earth",
) -> dict[str, float]:
    """Karney inverse geodesic on Earth, Moon, or Mars.

    Returns {"s12": metres, "azi1": deg, "azi2": deg, "a12": deg}.
    """
    return _require_geodesy_native().geodesic_inverse(
        lat1, lon1, lat2, lon2, body=body
    )


def geodesic_direct(
    lat1: float,
    lon1: float,
    azi1: float,
    s12: float,
    *,
    body: str = "earth",
) -> dict[str, float]:
    """Karney direct geodesic on Earth, Moon, or Mars.

    Returns {"lat2": deg, "lon2": deg, "azi2": deg, "a12": deg}.
    """
    return _require_geodesy_native().geodesic_direct(
        lat1, lon1, azi1, s12, body=body
    )


def wgs84_to_ecef(lon: float, lat: float, h: float = 0.0) -> tuple[float, float, float]:
    """WGS84 geodetic (degrees, ELLIPSOIDAL height metres) -> ECEF metres (f64)."""
    return _require_geodesy_native().wgs84_to_ecef(lon, lat, h)


def ecef_to_wgs84(x: float, y: float, z: float) -> tuple[float, float, float]:
    """ECEF metres -> WGS84 geodetic (lon deg, lat deg, ellipsoidal height m)."""
    return _require_geodesy_native().ecef_to_wgs84(x, y, z)


def dem_orthometric_to_ellipsoidal(
    dem: Any, bounds: tuple[float, float, float, float]
) -> np.ndarray:
    """Per-pixel orthometric (EGM96) -> ellipsoidal DEM conversion.

    ``bounds`` is (left, bottom, right, top) in EPSG:4326 degrees; pixel
    centres are sampled. Returns a float64 array of h = H + N(lat, lon).
    """
    dem64 = np.ascontiguousarray(np.asarray(dem, dtype=np.float64))
    return _require_geodesy_native().dem_orthometric_to_ellipsoidal(dem64, tuple(bounds))


__all__ = [
    "proj_available",
    "transform_coords",
    "reproject_geom",
    "parse_crs_from_wkt",
    "crs_to_epsg",
    "get_crs_from_rasterio",
    "get_crs_from_geopandas",
    "body_info",
    "areoid_undulation",
    "geoid_undulation",
    "orthometric_to_ellipsoidal",
    "ellipsoidal_to_orthometric",
    "geodesic_inverse",
    "geodesic_direct",
    "wgs84_to_ecef",
    "ecef_to_wgs84",
    "dem_orthometric_to_ellipsoidal",
]
