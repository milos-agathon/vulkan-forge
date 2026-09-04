use std::collections::HashMap;

use crate::gis::affine::validate_bounds_tuple;
use crate::gis::error::{GisError, GisResult};
use crate::gis::raster_write::CrsSpec;
use crate::gis::types::{RasterBounds, RasterInfo, RasterWarning, WARNING_MISSING_CRS};

const WEB_MERCATOR_MAX_LAT: f64 = 85.051_128_78;

#[derive(Debug, Clone)]
pub struct CrsInspection {
    pub source_kind: String,
    pub missing: bool,
    pub wkt: Option<String>,
    pub authority: Option<HashMap<String, String>>,
    pub axis_order: Option<String>,
    pub axis_order_policy: String,
    pub warnings: Vec<RasterWarning>,
}

pub fn inspect_info_crs(info: &RasterInfo, source_kind: &str) -> GisResult<CrsInspection> {
    let warnings = info
        .warnings
        .iter()
        .filter(|warning| warning.code == WARNING_MISSING_CRS)
        .cloned()
        .collect::<Vec<_>>();
    Ok(CrsInspection {
        source_kind: source_kind.to_string(),
        missing: info.crs_wkt.is_none() && info.crs_authority.is_none(),
        wkt: info.crs_wkt.clone(),
        authority: info.crs_authority.clone(),
        axis_order: info.crs_authority.as_ref().map(|_| "xy".to_string()),
        axis_order_policy: "traditional_gis_xy".to_string(),
        warnings,
    })
}

pub fn inspect_crs_spec(spec: &CrsSpec, source_kind: &str) -> CrsInspection {
    CrsInspection {
        source_kind: source_kind.to_string(),
        missing: false,
        wkt: spec.wkt.clone(),
        authority: authority_map(spec),
        axis_order: Some("xy".to_string()),
        axis_order_policy: "traditional_gis_xy".to_string(),
        warnings: Vec::new(),
    }
}

pub fn require_crs(info: &RasterInfo) -> GisResult<CrsSpec> {
    CrsSpec::from_raster_info(info)
        .ok_or_else(|| GisError::MissingCrs("missing_crs: raster has no CRS metadata".to_string()))
}

pub fn crs_equal(left: &CrsSpec, right: &CrsSpec) -> bool {
    left.equivalent_to(right)
}

pub fn authority_map(spec: &CrsSpec) -> Option<HashMap<String, String>> {
    let (name, code) = spec.authority.as_ref()?;
    let mut authority = HashMap::new();
    authority.insert("name".to_string(), name.to_ascii_uppercase());
    authority.insert("code".to_string(), code.to_string());
    Some(authority)
}

pub fn epsg_code(spec: &CrsSpec) -> Option<u32> {
    if let Some((name, code)) = &spec.authority {
        if name.eq_ignore_ascii_case("EPSG") {
            return code.parse().ok();
        }
    }
    if spec
        .wkt
        .as_deref()
        .is_some_and(|wkt| wkt.to_ascii_uppercase().contains("WGS 84"))
    {
        return Some(4326);
    }
    None
}

pub fn iau_code(spec: &CrsSpec) -> Option<u32> {
    let (name, code) = spec.authority.as_ref()?;
    name.eq_ignore_ascii_case("IAU")
        .then(|| code.parse().ok())
        .flatten()
}

fn iau_body(code: u32) -> GisResult<&'static crate::geo::body::Body> {
    if crate::geo::projections::iau_is_mars_ographic(code) {
        return Err(GisError::InvalidCrs(format!(
            "unsupported IAU:{code} Mars planetographic/ographic west-positive CRS; \
             SELENE supports planetocentric +East only"
        )));
    }
    crate::geo::projections::iau_body(code)
        .ok_or_else(|| GisError::InvalidCrs(format!("unsupported IAU code {code}")))
}

fn body_fixed_code(spec: &CrsSpec) -> Option<u32> {
    let (name, code) = spec.authority.as_ref()?;
    name.eq_ignore_ascii_case("FORGE3D")
        .then(|| code.parse().ok())
        .flatten()
}

fn body_fixed_body(code: u32) -> GisResult<&'static crate::geo::body::Body> {
    match code {
        301 => Ok(&crate::geo::body::MOON),
        499 => Ok(&crate::geo::body::MARS),
        _ => Err(GisError::InvalidCrs(format!(
            "unsupported FORGE3D body-fixed code {code}"
        ))),
    }
}

fn explicit_planetocentric_projection(
    spec: &CrsSpec,
) -> Option<crate::geo::projections::ProjectionDefinition> {
    use crate::geo::projections::ProjectionDefinition as P;
    match spec.projection? {
        projection @ (P::PlanetocentricEquirectangular(_)
        | P::PlanetocentricPolarStereographicA(_)) => Some(projection),
        _ => None,
    }
}

fn validate_planetocentric_projection(
    code: u32,
    projection: crate::geo::projections::ProjectionDefinition,
) -> GisResult<&'static crate::geo::body::Body> {
    let body = iau_body(code)?;
    let expected = crate::geo::projections::iau_ellipsoid(body, code)
        .ok_or_else(|| GisError::InvalidCrs(format!("unsupported IAU code {code}")))?;
    let actual = planetocentric_projection_ellipsoid(projection)?;
    if !ellipsoids_equivalent(actual, expected) {
        return Err(GisError::InvalidCrs(format!(
            "body mismatch: explicit projection ellipsoid does not match {} IAU:{code}",
            body.name
        )));
    }
    Ok(body)
}

fn planetocentric_projection_ellipsoid(
    projection: crate::geo::projections::ProjectionDefinition,
) -> GisResult<crate::geo::projections::Ellipsoid> {
    use crate::geo::projections::ProjectionDefinition as P;
    match projection {
        P::PlanetocentricEquirectangular(projection) => Ok(projection.ellipsoid),
        P::PlanetocentricPolarStereographicA(projection) => Ok(projection.ellipsoid),
        _ => Err(GisError::InvalidCrs(
            "custom projection is not planetocentric".to_string(),
        )),
    }
}

fn planetocentric_projection_body(
    projection: crate::geo::projections::ProjectionDefinition,
) -> GisResult<&'static crate::geo::body::Body> {
    let ellipsoid = planetocentric_projection_ellipsoid(projection)?;
    let mars_sphere = crate::geo::projections::Ellipsoid {
        a: crate::geo::body::MARS.ellipsoid.a,
        f: 0.0,
    };
    if ellipsoids_equivalent(ellipsoid, crate::geo::body::MOON.ellipsoid) {
        Ok(&crate::geo::body::MOON)
    } else if ellipsoids_equivalent(ellipsoid, crate::geo::body::MARS.ellipsoid)
        || ellipsoids_equivalent(ellipsoid, mars_sphere)
    {
        Ok(&crate::geo::body::MARS)
    } else {
        Err(GisError::InvalidCrs(
            "explicit planetocentric projection ellipsoid does not match a registered body"
                .to_string(),
        ))
    }
}

fn ellipsoids_equivalent(
    left: crate::geo::projections::Ellipsoid,
    right: crate::geo::projections::Ellipsoid,
) -> bool {
    (left.a - right.a).abs() <= left.a.abs().max(right.a.abs()) * 1e-12
        && (left.f - right.f).abs() <= 1e-12
}

/// Kind of an EPSG CRS for geographic-versus-planar dispatch. `CrsSpec`
/// parsing admits the whole EPSG 4000-4999 block plus the curated projection
/// table, and that block is NOT homogeneous: alongside geographic 2D CRSs it
/// contains projected members (4087/4088 World Equidistant Cylindrical, the
/// CGCS2000/Xian80 Gauss-Kruger zones, ...) and geocentric/3D members
/// (4978 WGS84 geocentric, 4936 ETRS89 geocentric, ...). Classification is
/// therefore an explicit curated table, never a numeric-range heuristic; a
/// code the table does not know is `Unclassified` and callers must raise a
/// stable error rather than guess.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum EpsgCrsKind {
    /// EPSG:4326 — the one geographic CRS with geodesic/dateline support.
    GeographicWgs84,
    /// A known geographic (degree-axis) CRS other than EPSG:4326.
    Geographic,
    /// A known geocentric (3D Cartesian) CRS; meaningless for 2D geometry.
    Geocentric,
    /// A known projected CRS (planar, CRS units).
    Projected,
    /// Parseable but not in the curated classification table.
    Unclassified,
}

pub(crate) fn epsg_crs_kind(code: u32) -> EpsgCrsKind {
    match code {
        4326 => EpsgCrsKind::GeographicWgs84,
        // Projected members of the 4000-4999 parse block.
        4087 | 4088 => EpsgCrsKind::Projected,
        // Geocentric members of the parse block (WGS84 and ETRS89).
        4936 | 4978 => EpsgCrsKind::Geocentric,
        // Curated common geographic CRSs: RGF93, ED50, ETRS89, NAD27, NAD83,
        // OSGB36, GDA94, CGCS2000, JGD2000, NAD83(CSRS), SIRGAS2000,
        // NAD83(NSRS2007), ETRS89 3D, WGS84 3D.
        4171 | 4230 | 4258 | 4267 | 4269 | 4277 | 4283 | 4490 | 4612 | 4617 | 4674 | 4759
        | 4937 | 4979 => EpsgCrsKind::Geographic,
        code if crate::geo::projections::epsg_projection_definition(code).is_some() => {
            EpsgCrsKind::Projected
        }
        _ => EpsgCrsKind::Unclassified,
    }
}

pub fn canonical_label(spec: &CrsSpec) -> GisResult<String> {
    if let Some(projection) = spec.projection {
        return Ok(format!("forge3d:{projection:?}"));
    }
    if let Some((name, code)) = &spec.authority {
        return Ok(format!("{}:{code}", name.to_ascii_uppercase()));
    }
    if let Some(wkt) = &spec.wkt {
        return Ok(wkt.clone());
    }
    Err(GisError::MissingCrs(
        "missing_crs: CRS definition is empty".to_string(),
    ))
}

pub fn transform_bounds(
    bounds: RasterBounds,
    src: &CrsSpec,
    dst: &CrsSpec,
) -> GisResult<RasterBounds> {
    // densify = 1 samples exactly the four corners (legacy behaviour).
    transform_bounds_densified(bounds, src, dst, 1)
}

/// Transform bounds by densifying each edge into `densify` segments before
/// reprojecting — required for correctness under any projection with curved
/// parallels/meridians, where the extremum of an edge lies between corners.
pub fn transform_bounds_densified(
    bounds: RasterBounds,
    src: &CrsSpec,
    dst: &CrsSpec,
    densify: u32,
) -> GisResult<RasterBounds> {
    if crs_equal(src, dst) {
        return Ok(bounds);
    }
    if densify == 0 {
        return Err(GisError::InvalidArgument(
            "invalid_argument: transform_bounds densify must be at least 1".to_string(),
        ));
    }
    let n = densify as usize;
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    let mut fold = |x: f64, y: f64| -> GisResult<()> {
        let (tx, ty) = transform_point(x, y, src, dst)?;
        min_x = min_x.min(tx);
        max_x = max_x.max(tx);
        min_y = min_y.min(ty);
        max_y = max_y.max(ty);
        Ok(())
    };
    for i in 0..=n {
        let t = i as f64 / n as f64;
        let x = bounds.left + t * (bounds.right - bounds.left);
        let y = bounds.bottom + t * (bounds.top - bounds.bottom);
        fold(x, bounds.bottom)?;
        fold(x, bounds.top)?;
        fold(bounds.left, y)?;
        fold(bounds.right, y)?;
    }
    validate_bounds_tuple((min_x, min_y, max_x, max_y), false)
}

/// Check whether a CRS pair is dispatchable by the built-in engine without
/// evaluating any coordinate.
pub fn transform_pair_supported(src: &CrsSpec, dst: &CrsSpec) -> GisResult<()> {
    use crate::geo::projections::epsg_projection_definition;
    for code in [iau_code(src), iau_code(dst)].into_iter().flatten() {
        iau_body(code)?;
    }
    for code in [body_fixed_code(src), body_fixed_code(dst)]
        .into_iter()
        .flatten()
    {
        body_fixed_body(code)?;
    }
    if crs_equal(src, dst) {
        return Ok(());
    }
    if let (Some(projection), Some(code)) = (explicit_planetocentric_projection(src), iau_code(dst))
    {
        validate_planetocentric_projection(code, projection)?;
        return Ok(());
    }
    if let (Some(code), Some(projection)) = (iau_code(src), explicit_planetocentric_projection(dst))
    {
        validate_planetocentric_projection(code, projection)?;
        return Ok(());
    }
    let projection_and_fixed = match (
        explicit_planetocentric_projection(src),
        body_fixed_code(dst),
        body_fixed_code(src),
        explicit_planetocentric_projection(dst),
    ) {
        (Some(projection), Some(fixed), None, None)
        | (None, None, Some(fixed), Some(projection)) => Some((projection, fixed)),
        _ => None,
    };
    if let Some((projection, fixed)) = projection_and_fixed {
        let projection_body = planetocentric_projection_body(projection)?;
        let fixed_body = body_fixed_body(fixed)?;
        if projection_body == fixed_body {
            return Ok(());
        }
        return Err(GisError::InvalidCrs(format!(
            "body mismatch: {} projection cannot transform to {} body-fixed CRS",
            projection_body.name, fixed_body.name
        )));
    }
    let (src_planetocentric, dst_planetocentric) = (
        explicit_planetocentric_projection(src),
        explicit_planetocentric_projection(dst),
    );
    if src_planetocentric.is_some() || dst_planetocentric.is_some() {
        if let (Some(source), Some(target)) = (src_planetocentric, dst_planetocentric) {
            let source_body = planetocentric_projection_body(source)?;
            let target_body = planetocentric_projection_body(target)?;
            if source_body == target_body {
                return Ok(());
            }
            return Err(GisError::InvalidCrs(format!(
                "body mismatch: {} projection cannot transform to {} projection",
                source_body.name, target_body.name
            )));
        }
        return Err(GisError::InvalidCrs(
            "planetocentric custom projections cannot be mixed with Earth or untyped custom CRSs"
                .to_string(),
        ));
    }
    if src.projection.is_some() || dst.projection.is_some() {
        if iau_code(src).is_some()
            || iau_code(dst).is_some()
            || body_fixed_code(src).is_some()
            || body_fixed_code(dst).is_some()
        {
            return Err(GisError::InvalidCrs(
                "planetary IAU/body-fixed CRSs cannot be mixed with custom Earth projections"
                    .to_string(),
            ));
        }
        return Ok(());
    }
    match (
        iau_code(src),
        body_fixed_code(src),
        iau_code(dst),
        body_fixed_code(dst),
    ) {
        (Some(iau), None, None, Some(fixed)) | (None, Some(fixed), Some(iau), None) => {
            let geographic_body = iau_body(iau)?;
            let fixed_body = body_fixed_body(fixed)?;
            if geographic_body == fixed_body {
                return Ok(());
            }
            return Err(GisError::InvalidCrs(format!(
                "body mismatch: {} planetary CRS cannot transform to {} body-fixed CRS",
                geographic_body.name, fixed_body.name
            )));
        }
        (None, Some(source), None, Some(target)) => {
            return Err(GisError::InvalidCrs(format!(
                "body mismatch: {} body-fixed CRS cannot transform to {} body-fixed CRS",
                body_fixed_body(source)?.name,
                body_fixed_body(target)?.name
            )));
        }
        _ => {}
    }
    match (iau_code(src), iau_code(dst)) {
        (Some(source), Some(target)) => {
            let source_body = iau_body(source)?;
            let target_body = iau_body(target)?;
            if source_body == target_body {
                return Ok(());
            }
            return Err(GisError::InvalidCrs(format!(
                "body mismatch: {} CRS {} cannot transform to {} CRS {}",
                source_body.name,
                canonical_label(src)?,
                target_body.name,
                canonical_label(dst)?
            )));
        }
        (Some(code), None) | (None, Some(code)) => {
            let body = iau_body(code)?;
            return Err(GisError::InvalidCrs(format!(
                "body mismatch: {} planetary CRS cannot transform to an Earth EPSG CRS",
                body.name
            )));
        }
        (None, None) => {}
    }
    let is_geocentric_pair = matches!(
        (epsg_code(src), epsg_code(dst)),
        (Some(4326 | 4979), Some(4978)) | (Some(4978), Some(4326 | 4979))
    );
    if is_geocentric_pair {
        return Ok(());
    }
    let ok = |code: u32| code == 4326 || epsg_projection_definition(code).is_some();
    match (epsg_code(src), epsg_code(dst)) {
        (Some(s), Some(d)) if ok(s) && ok(d) => Ok(()),
        _ => Err(GisError::BackendUnavailable(format!(
            "BackendUnavailable: CRS transform {} to {} requires an unavailable PROJ backend",
            canonical_label(src)?,
            canonical_label(dst)?
        ))),
    }
}

/// Transform geographic/projected 3D coordinates to/from body-fixed Cartesian
/// coordinates for WGS84 Earth or a registered planetocentric IAU body.
/// Planar-only calls remain on [`transform_point`].
pub fn transform_point3(
    x: f64,
    y: f64,
    z: f64,
    src: &CrsSpec,
    dst: &CrsSpec,
) -> GisResult<(f64, f64, f64)> {
    use crate::geo::projections::geocentric::{
        ecef_to_planetocentric, planetocentric_to_ecef, wgs84_ecef_to_geodetic,
        wgs84_geodetic_to_ecef,
    };
    use crate::geo::projections::ProjError;
    use glam::DVec3;

    if !x.is_finite() || !y.is_finite() || !z.is_finite() {
        return Err(GisError::TransformFailed(
            "coordinate values must be finite".to_string(),
        ));
    }
    if crs_equal(src, dst) {
        return Ok((x, y, z));
    }
    let map_err = |error: ProjError| GisError::TransformFailed(error.to_string());
    if let (Some(projection), Some(fixed)) = (
        explicit_planetocentric_projection(src),
        body_fixed_code(dst),
    ) {
        let projection_body = planetocentric_projection_body(projection)?;
        let fixed_body = body_fixed_body(fixed)?;
        if projection_body != fixed_body {
            return Err(GisError::InvalidCrs(format!(
                "body mismatch: {} projection cannot transform to {} body-fixed CRS",
                projection_body.name, fixed_body.name
            )));
        }
        let ellipsoid = planetocentric_projection_ellipsoid(projection)?;
        let (lon, lat) = projection.inverse(x, y).map_err(map_err)?;
        let point = planetocentric_to_ecef(&ellipsoid, lon, lat, z).map_err(map_err)?;
        return Ok((point.x, point.y, point.z));
    }
    if let (Some(fixed), Some(projection)) = (
        body_fixed_code(src),
        explicit_planetocentric_projection(dst),
    ) {
        let fixed_body = body_fixed_body(fixed)?;
        let projection_body = planetocentric_projection_body(projection)?;
        if fixed_body != projection_body {
            return Err(GisError::InvalidCrs(format!(
                "body mismatch: {} body-fixed CRS cannot transform to {} projection",
                fixed_body.name, projection_body.name
            )));
        }
        let ellipsoid = planetocentric_projection_ellipsoid(projection)?;
        let (lon, lat, height) =
            ecef_to_planetocentric(&ellipsoid, DVec3::new(x, y, z)).map_err(map_err)?;
        let (out_x, out_y) = projection.forward(lon, lat).map_err(map_err)?;
        return Ok((out_x, out_y, height));
    }
    match (
        iau_code(src),
        body_fixed_code(src),
        iau_code(dst),
        body_fixed_code(dst),
    ) {
        (Some(source), None, None, Some(target)) => {
            let source_body = iau_body(source)?;
            let target_body = body_fixed_body(target)?;
            if source_body != target_body {
                return Err(GisError::InvalidCrs(format!(
                    "body mismatch: {} cannot transform to {}",
                    source_body.name, target_body.name
                )));
            }
            let (lon, lat) = if let Some(projection) =
                crate::geo::projections::iau_projection(source_body, source)
            {
                projection.inverse(x, y).map_err(map_err)?
            } else if crate::geo::projections::iau_geographic(source_body, source) {
                (x, y)
            } else {
                return Err(GisError::InvalidCrs(format!(
                    "unsupported IAU code {source}"
                )));
            };
            let ellipsoid = crate::geo::projections::iau_ellipsoid(source_body, source)
                .ok_or_else(|| GisError::InvalidCrs(format!("unsupported IAU code {source}")))?;
            let point = planetocentric_to_ecef(&ellipsoid, lon, lat, z).map_err(map_err)?;
            return Ok((point.x, point.y, point.z));
        }
        (None, Some(source), Some(target), None) => {
            let source_body = body_fixed_body(source)?;
            let target_body = iau_body(target)?;
            if source_body != target_body {
                return Err(GisError::InvalidCrs(format!(
                    "body mismatch: {} cannot transform to {}",
                    source_body.name, target_body.name
                )));
            }
            let ellipsoid = crate::geo::projections::iau_ellipsoid(target_body, target)
                .ok_or_else(|| GisError::InvalidCrs(format!("unsupported IAU code {target}")))?;
            let (lon, lat, height) =
                ecef_to_planetocentric(&ellipsoid, DVec3::new(x, y, z)).map_err(map_err)?;
            let (out_x, out_y) = if let Some(projection) =
                crate::geo::projections::iau_projection(target_body, target)
            {
                projection.forward(lon, lat).map_err(map_err)?
            } else if crate::geo::projections::iau_geographic(target_body, target) {
                (lon, lat)
            } else {
                return Err(GisError::InvalidCrs(format!(
                    "unsupported IAU code {target}"
                )));
            };
            return Ok((out_x, out_y, height));
        }
        _ => {}
    }
    match (epsg_code(src), epsg_code(dst)) {
        (Some(4326 | 4979), Some(4978)) => {
            let point = wgs84_geodetic_to_ecef(x, y, z)
                .map_err(|error| GisError::TransformFailed(error.to_string()))?;
            Ok((point.x, point.y, point.z))
        }
        (Some(4978), Some(4326 | 4979)) => wgs84_ecef_to_geodetic(DVec3::new(x, y, z))
            .map_err(|error| GisError::TransformFailed(error.to_string())),
        _ => Err(GisError::BackendUnavailable(format!(
            "BackendUnavailable: CRS transform {} to {} has no built-in 3D path",
            canonical_label(src)?,
            canonical_label(dst)?
        ))),
    }
}

/// Dispatch a coordinate through the built-in pure-Rust projection engine
/// (src/geo/projections) via the authoritative `epsg_projection_definition`
/// table (4326 geographic, plus 3857, WGS84 UTM zones, 3395, 2154, 5070, 5041,
/// and 5042), or an explicit `CrsSpec.projection`. Anything else is an explicit
/// `BackendUnavailable` — the engine never silently passes coordinates through.
pub fn transform_point(x: f64, y: f64, src: &CrsSpec, dst: &CrsSpec) -> GisResult<(f64, f64)> {
    use crate::geo::projections::{epsg_projection_definition, ProjError, ProjectionDefinition};

    if !x.is_finite() || !y.is_finite() {
        return Err(GisError::TransformFailed(
            "coordinate values must be finite".to_string(),
        ));
    }
    transform_pair_supported(src, dst)?;
    if crs_equal(src, dst) {
        return Ok((x, y));
    }
    let unsupported = |src: &CrsSpec, dst: &CrsSpec| -> GisResult<(f64, f64)> {
        Err(GisError::BackendUnavailable(format!(
            "BackendUnavailable: CRS transform {} to {} requires an unavailable PROJ backend",
            canonical_label(src)?,
            canonical_label(dst)?
        )))
    };
    let map_err = |err: ProjError| -> GisError {
        match err {
            ProjError::Domain(msg) => GisError::InvalidBounds(msg),
            other => GisError::TransformFailed(other.to_string()),
        }
    };
    if let (Some(source), Some(target)) = (iau_code(src), iau_code(dst)) {
        let source_body = iau_body(source)?;
        let target_body = iau_body(target)?;
        if source_body != target_body {
            return Err(GisError::InvalidCrs(format!(
                "body mismatch: {} CRS {} cannot transform to {} CRS {}",
                source_body.name,
                canonical_label(src)?,
                target_body.name,
                canonical_label(dst)?
            )));
        }
        let (lon, lat) = if let Some(projection) =
            crate::geo::projections::iau_projection(source_body, source)
        {
            projection.inverse(x, y).map_err(map_err)?
        } else if crate::geo::projections::iau_geographic(source_body, source) {
            (x, y)
        } else {
            return unsupported(src, dst);
        };
        return if let Some(projection) =
            crate::geo::projections::iau_projection(target_body, target)
        {
            projection.forward(lon, lat).map_err(map_err)
        } else if crate::geo::projections::iau_geographic(target_body, target) {
            Ok((lon, lat))
        } else {
            unsupported(src, dst)
        };
    }
    if let (Some(projection), Some(target)) =
        (explicit_planetocentric_projection(src), iau_code(dst))
    {
        let body = validate_planetocentric_projection(target, projection)?;
        let (lon, lat) = projection.inverse(x, y).map_err(map_err)?;
        return if let Some(target_projection) =
            crate::geo::projections::iau_projection(body, target)
        {
            target_projection.forward(lon, lat).map_err(map_err)
        } else if crate::geo::projections::iau_geographic(body, target) {
            Ok((lon, lat))
        } else {
            unsupported(src, dst)
        };
    }
    if let (Some(source), Some(projection)) =
        (iau_code(src), explicit_planetocentric_projection(dst))
    {
        let body = validate_planetocentric_projection(source, projection)?;
        let (lon, lat) = if let Some(source_projection) =
            crate::geo::projections::iau_projection(body, source)
        {
            source_projection.inverse(x, y).map_err(map_err)?
        } else if crate::geo::projections::iau_geographic(body, source) {
            (x, y)
        } else {
            return unsupported(src, dst);
        };
        return projection.forward(lon, lat).map_err(map_err);
    }
    if (iau_code(src).is_some() && body_fixed_code(dst).is_some())
        || (body_fixed_code(src).is_some() && iau_code(dst).is_some())
    {
        return Err(GisError::BackendUnavailable(
            "planetary geographic/body-fixed conversion requires transform_point3".to_string(),
        ));
    }
    if iau_code(src).is_some() || iau_code(dst).is_some() {
        return Err(GisError::InvalidCrs(
            "body mismatch: planetary IAU and Earth EPSG CRSs cannot be mixed".to_string(),
        ));
    }
    let (src_code, dst_code) = (epsg_code(src), epsg_code(dst));
    // An explicit projection definition on the CrsSpec wins; otherwise resolve
    // the EPSG code through the one authoritative projection table. 4326 stays
    // geographic (a `None` here, handled by the passthrough branch below).
    let resolve = |spec: &CrsSpec, code: Option<u32>| -> Option<ProjectionDefinition> {
        spec.projection
            .or_else(|| code.and_then(epsg_projection_definition))
    };
    // Route through geographic WGS84: src → 4326 → dst.
    let (lon, lat) = if let Some(projection) = resolve(src, src_code) {
        projection.inverse(x, y).map_err(map_err)?
    } else if src_code == Some(4326) {
        (x, y)
    } else {
        return unsupported(src, dst);
    };
    if let Some(projection) = resolve(dst, dst_code) {
        projection.forward(lon, lat).map_err(map_err)
    } else if dst_code == Some(4326) {
        Ok((lon, lat))
    } else {
        unsupported(src, dst)
    }
}

pub fn web_mercator_bounds(bounds: RasterBounds, src: &CrsSpec) -> GisResult<RasterBounds> {
    let src_epsg = epsg_code(src);
    validate_bounds_tuple(bounds.tuple(), src_epsg == Some(4326))?;
    if src_epsg == Some(4326) {
        if bounds.left > bounds.right {
            return Err(GisError::InvalidBounds(
                "antimeridian_bounds_unsupported: split antimeridian bounds before transforming"
                    .to_string(),
            ));
        }
        if bounds.bottom < -WEB_MERCATOR_MAX_LAT || bounds.top > WEB_MERCATOR_MAX_LAT {
            return Err(GisError::InvalidBounds(format!(
                "invalid_latitude_range: Web Mercator supports latitudes within +/-{WEB_MERCATOR_MAX_LAT}"
            )));
        }
    }
    let dst = CrsSpec::from_string("EPSG:3857".to_string())?;
    transform_bounds(bounds, src, &dst)
}

pub fn parse_crs_string(value: String) -> GisResult<CrsSpec> {
    if let Some(code) = parse_epsg_code(&value) {
        return CrsSpec::from_string(format!("EPSG:{code}"));
    }
    CrsSpec::from_string(value)
}

fn parse_epsg_code(crs: &str) -> Option<u32> {
    let trimmed = crs.trim();
    let (authority, code) = trimmed.split_once(':')?;
    if authority.eq_ignore_ascii_case("EPSG") {
        code.parse::<u32>().ok()
    } else {
        None
    }
}

#[cfg(feature = "extension-module")]
use pyo3::prelude::*;
#[cfg(feature = "extension-module")]
use pyo3::types::PyAny;

#[cfg_attr(
    feature = "extension-module",
    pyo3::pyclass(module = "forge3d._forge3d", name = "CrsTransform")
)]
#[derive(Debug, Clone)]
pub struct CrsTransform {
    pub src: CrsSpec,
    pub dst: CrsSpec,
    pub axis_order_policy: String,
}

impl CrsTransform {
    pub fn new(src: CrsSpec, dst: CrsSpec, always_xy: bool) -> GisResult<Self> {
        // Check dispatchability once so unsupported pairs fail at creation,
        // not first use. (A numeric probe point would false-negative on
        // domain limits, e.g. UTM-south coordinates outside Web Mercator's
        // latitude range.)
        transform_pair_supported(&src, &dst)?;
        Ok(Self {
            src,
            dst,
            axis_order_policy: if always_xy { "always_xy" } else { "native" }.to_string(),
        })
    }
}

#[cfg(feature = "extension-module")]
#[pyo3::pymethods]
impl CrsTransform {
    #[staticmethod]
    #[pyo3(signature = (src_crs, dst_crs, *, always_xy = true))]
    fn from_crs(
        src_crs: &Bound<'_, PyAny>,
        dst_crs: &Bound<'_, PyAny>,
        always_xy: bool,
    ) -> PyResult<Self> {
        Self::new(
            crate::gis::extract_required_crs(Some(src_crs))?,
            crate::gis::extract_required_crs(Some(dst_crs))?,
            always_xy,
        )
        .map_err(Into::into)
    }

    #[getter]
    fn src_crs(&self) -> PyResult<String> {
        canonical_label(&self.src).map_err(Into::into)
    }

    #[getter]
    fn dst_crs(&self) -> PyResult<String> {
        canonical_label(&self.dst).map_err(Into::into)
    }

    #[getter]
    fn src_authority(&self) -> Option<HashMap<String, String>> {
        authority_map(&self.src)
    }

    #[getter]
    fn dst_authority(&self) -> Option<HashMap<String, String>> {
        authority_map(&self.dst)
    }

    #[getter]
    fn axis_order_policy(&self) -> String {
        self.axis_order_policy.clone()
    }

    fn transform_point(&self, x: f64, y: f64) -> PyResult<(f64, f64)> {
        transform_point(x, y, &self.src, &self.dst).map_err(Into::into)
    }

    fn transform_point3(&self, x: f64, y: f64, z: f64) -> PyResult<(f64, f64, f64)> {
        transform_point3(x, y, z, &self.src, &self.dst).map_err(Into::into)
    }

    fn transform_bounds(&self, bounds: (f64, f64, f64, f64)) -> PyResult<(f64, f64, f64, f64)> {
        let bounds = validate_bounds_tuple(bounds, false)?;
        transform_bounds(bounds, &self.src, &self.dst)
            .map(|bounds| bounds.tuple())
            .map_err(Into::into)
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "CrsTransform(src_crs={:?}, dst_crs={:?}, axis_order_policy={:?})",
            canonical_label(&self.src)?,
            canonical_label(&self.dst)?,
            self.axis_order_policy
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(code: &str) -> CrsSpec {
        CrsSpec::from_string(code.to_string()).expect("valid CRS code")
    }

    /// Forward WGS84 lon/lat into `code`, inverse back, return recovered lon/lat.
    fn round_trip(code: &str, lon: f64, lat: f64) -> (f64, f64) {
        let wgs84 = spec("EPSG:4326");
        let proj = spec(code);
        let (e, n) = transform_point(lon, lat, &wgs84, &proj).expect("forward");
        transform_point(e, n, &proj, &wgs84).expect("inverse")
    }

    #[test]
    fn every_curated_projection_is_reachable_by_epsg_code_and_round_trips() {
        // (code, interior lon, interior lat)
        let cases = [
            ("EPSG:3857", 12.5, 41.9),   // Web Mercator (Rome)
            ("EPSG:32633", 15.0, 45.0),  // UTM 33N
            ("EPSG:32733", 15.0, -30.0), // UTM 33S (southern false northing)
            ("EPSG:3395", 10.0, 45.0),   // World Mercator (variant A)
            ("EPSG:2154", 2.5, 47.0),    // Lambert-93 (LCC 2SP, France)
            ("EPSG:5070", -96.0, 38.0),  // Conus Albers (AEA)
            ("EPSG:5041", 30.0, 85.0),   // UPS North (Polar Stereographic A)
            ("EPSG:5042", 30.0, -85.0),  // UPS South (Polar Stereographic A)
        ];
        for (code, lon, lat) in cases {
            let (rlon, rlat) = round_trip(code, lon, lat);
            let residual = (rlon - lon).abs().max((rlat - lat).abs());
            assert!(
                residual < 1e-9,
                "{code} round-trip residual {residual:.3e} deg exceeds 1e-9"
            );
        }
    }

    #[test]
    fn unsupported_epsg_pair_raises_rather_than_passing_through() {
        // EPSG:4269 (NAD83 geographic) parses (4000-4999) but has no built-in
        // path to a projected CRS: it must raise, never pass coordinates through.
        let src = spec("EPSG:4326");
        let dst = spec("EPSG:4269");
        assert!(matches!(
            transform_point(1.0, 2.0, &src, &dst),
            Err(GisError::BackendUnavailable(_))
        ));
    }

    #[test]
    fn non_finite_input_is_rejected() {
        let src = spec("EPSG:4326");
        let dst = spec("EPSG:3857");
        assert!(transform_point(f64::NAN, 2.0, &src, &dst).is_err());
    }

    #[test]
    fn identity_transform_returns_input() {
        let s = spec("EPSG:3857");
        assert_eq!(
            transform_point(100.0, 200.0, &s, &s).unwrap(),
            (100.0, 200.0)
        );
    }

    #[test]
    fn internal_body_fixed_tags_parse_only_for_known_bodies() {
        let moon_fixed = spec("FORGE3D:301");
        transform_pair_supported(&moon_fixed, &moon_fixed).unwrap();
        assert_eq!(canonical_label(&moon_fixed).unwrap(), "FORGE3D:301");
        assert!(CrsSpec::from_string("FORGE3D:999".to_string()).is_err());
    }

    #[test]
    fn raster_metadata_cannot_bypass_iau_convention_validation() {
        let ographic = CrsSpec {
            authority: Some(("IAU".to_string(), "49901".to_string())),
            wkt: None,
            projection: None,
        };
        assert!(transform_pair_supported(&ographic, &ographic)
            .unwrap_err()
            .to_string()
            .contains("planetocentric +East"));
    }

    #[test]
    fn planetary_crs_cannot_construct_with_custom_earth_projection() {
        let moon = spec("IAU:30100");
        let custom =
            CrsSpec::from_projection(crate::geo::projections::ProjectionDefinition::WebMercator);
        assert!(transform_pair_supported(&moon, &custom).is_err());
        let fixed = spec("FORGE3D:301");
        assert!(transform_pair_supported(&fixed, &custom).is_err());
    }

    #[test]
    fn explicit_planetocentric_projections_reject_untyped_and_mismatched_bodies() {
        use crate::geo::projections::{eqc::Equirectangular, Ellipsoid, ProjectionDefinition};
        let projection = |ellipsoid| {
            CrsSpec::from_projection(ProjectionDefinition::PlanetocentricEquirectangular(
                Equirectangular {
                    ellipsoid,
                    lat0_deg: 0.0,
                    lon0_deg: 0.0,
                    lat_ts_deg: 0.0,
                    false_easting: 0.0,
                    false_northing: 0.0,
                },
            ))
        };
        let mars = projection(crate::geo::body::MARS.ellipsoid);
        let moon = projection(crate::geo::body::MOON.ellipsoid);
        assert!(transform_pair_supported(&mars, &spec("EPSG:4326")).is_err());
        assert!(transform_pair_supported(&mars, &moon).is_err());

        let bogus_mars = projection(Ellipsoid {
            a: crate::geo::body::MARS.ellipsoid.a,
            f: 0.5,
        });
        assert!(transform_pair_supported(&bogus_mars, &spec("FORGE3D:499")).is_err());
    }

    #[test]
    fn explicit_planetocentric_projection_routes_through_matching_iau_body() {
        let mars = spec("IAU:49902");
        let explicit = CrsSpec::from_projection(
            crate::geo::projections::ProjectionDefinition::PlanetocentricEquirectangular(
                crate::geo::projections::eqc::Equirectangular {
                    ellipsoid: crate::geo::body::MARS.ellipsoid,
                    lat0_deg: 0.0,
                    lon0_deg: 0.0,
                    lat_ts_deg: 0.0,
                    false_easting: 0.0,
                    false_northing: 0.0,
                },
            ),
        );
        transform_pair_supported(&mars, &explicit).unwrap();
        let projected = transform_point(10.0, 20.0, &mars, &explicit).unwrap();
        assert!((projected.0 - 592_746.975_233_062_2).abs() < 1e-6);
        assert!((projected.1 - 1_198_439.570_168_155_2).abs() < 1e-6);
        let recovered = transform_point(projected.0, projected.1, &explicit, &mars).unwrap();
        assert!((recovered.0 - 10.0).abs() < 1e-12);
        assert!((recovered.1 - 20.0).abs() < 1e-12);

        let fixed = spec("FORGE3D:499");
        transform_pair_supported(&explicit, &fixed).unwrap();
        let xyz = transform_point3(projected.0, projected.1, 1_234.5, &explicit, &fixed).unwrap();
        let recovered = transform_point3(xyz.0, xyz.1, xyz.2, &fixed, &explicit).unwrap();
        assert!((recovered.0 - projected.0).abs() < 1e-6);
        assert!((recovered.1 - projected.1).abs() < 1e-6);
        assert!((recovered.2 - 1_234.5).abs() < 1e-6);
    }

    #[test]
    fn spherical_iau_projection_definition_is_accepted_as_explicit_planetocentric() {
        let moon = spec("IAU:30100");
        let projection =
            crate::geo::projections::iau_projection(&crate::geo::body::MOON, 30110).unwrap();
        let explicit = CrsSpec::from_projection(projection);
        transform_pair_supported(&moon, &explicit).unwrap();
        let projected = transform_point(10.0, 20.0, &moon, &explicit).unwrap();
        let recovered = transform_point(projected.0, projected.1, &explicit, &moon).unwrap();
        assert!((recovered.0 - 10.0).abs() < 1e-12);
        assert!((recovered.1 - 20.0).abs() < 1e-12);

        let fixed = spec("FORGE3D:301");
        transform_pair_supported(&explicit, &fixed).unwrap();
        let xyz = transform_point3(projected.0, projected.1, 100.0, &explicit, &fixed).unwrap();
        let recovered = transform_point3(xyz.0, xyz.1, xyz.2, &fixed, &explicit).unwrap();
        assert!((recovered.0 - projected.0).abs() < 1e-6);
        assert!((recovered.1 - projected.1).abs() < 1e-6);
        assert!((recovered.2 - 100.0).abs() < 1e-6);
    }

    #[test]
    fn projection_domain_edges_are_handled() {
        let wgs84 = spec("EPSG:4326");
        // North pole -> UPS North (Polar Stereographic A) maps to the false
        // origin (2e6, 2e6), and inverts back to lat = 90.
        let ups_n = spec("EPSG:5041");
        let (e, n) = transform_point(0.0, 90.0, &wgs84, &ups_n).expect("pole forward");
        assert!((e - 2_000_000.0).abs() < 1e-3, "pole easting {e}");
        assert!((n - 2_000_000.0).abs() < 1e-3, "pole northing {n}");
        let (_lon, lat) = transform_point(e, n, &ups_n, &wgs84).expect("pole inverse");
        assert!((lat - 90.0).abs() < 1e-9, "recovered pole lat {lat}");

        // UTM 33N central meridian (lon 15) at the equator sits at the false
        // easting with zero northing.
        let utm33n = spec("EPSG:32633");
        let (e, n) = transform_point(15.0, 0.0, &wgs84, &utm33n).expect("cm forward");
        assert!(
            (e - 500_000.0).abs() < 1e-3 && n.abs() < 1e-3,
            "cm ({e},{n})"
        );

        // Southern UTM carries the 10,000,000 m false northing.
        let utm33s = spec("EPSG:32733");
        let (_e, n) = transform_point(15.0, -0.001, &wgs84, &utm33s).expect("south forward");
        assert!(n > 9_999_000.0, "southern false northing {n}");

        // Non-finite input is rejected, never projected.
        assert!(transform_point(f64::INFINITY, 0.0, &wgs84, &utm33n).is_err());
        assert!(transform_point(0.0, f64::NAN, &wgs84, &utm33n).is_err());

        // An out-of-zone longitude (30 deg from the central meridian) still
        // yields a finite, distorted result rather than NaN or a panic.
        let (e, n) = transform_point(45.0, 10.0, &wgs84, &utm33n).expect("out-of-zone");
        assert!(e.is_finite() && n.is_finite(), "out-of-zone ({e},{n})");
    }

    #[test]
    fn projection_roundtrip_residual_per_method_is_far_under_1mm() {
        // MENSURA M-02 evidence: forward then inverse over an interior grid for
        // every curated method, reporting the worst closure residual PER METHOD
        // in millimetres. This is self-consistency, not external G7-2
        // conformance (the kernels' own worked-example unit tests assert that,
        // bounded by the published reference precision); it proves each method
        // is internally consistent orders of magnitude below 1 mm.
        const DEG_TO_M: f64 = 111_320.0; // nominal metres/degree
                                         // (code, centre lon, centre lat, lon span, lat span)
        let cases: &[(&str, f64, f64, f64, f64)] = &[
            ("EPSG:3857", 0.0, 0.0, 300.0, 140.0),
            ("EPSG:32633", 15.0, 0.0, 6.0, 160.0),
            ("EPSG:3395", 0.0, 0.0, 300.0, 140.0),
            ("EPSG:2154", 3.0, 46.5, 10.0, 8.0),
            ("EPSG:5070", -96.0, 38.0, 50.0, 20.0),
            ("EPSG:5041", 0.0, 85.0, 300.0, 8.0),
            ("EPSG:5042", 0.0, -85.0, 300.0, 8.0),
        ];
        let wgs84 = spec("EPSG:4326");
        for (code, lon0, lat0, dlon, dlat) in cases {
            let proj = spec(code);
            let mut worst_mm = 0.0f64;
            for i in 0..=6 {
                for j in 0..=6 {
                    let lon = lon0 - dlon / 2.0 + dlon * i as f64 / 6.0;
                    let lat = lat0 - dlat / 2.0 + dlat * j as f64 / 6.0;
                    if lat.abs() > 89.5 {
                        continue;
                    }
                    if let Ok((e, n)) = transform_point(lon, lat, &wgs84, &proj) {
                        if let Ok((rlon, rlat)) = transform_point(e, n, &proj, &wgs84) {
                            let dm = (rlon - lon).abs().max((rlat - lat).abs()) * DEG_TO_M * 1000.0;
                            worst_mm = worst_mm.max(dm);
                        }
                    }
                }
            }
            println!("{code}: worst round-trip residual {worst_mm:.6e} mm");
            assert!(
                worst_mm < 1.0,
                "{code} round-trip residual {worst_mm} mm exceeds 1 mm"
            );
        }
    }
}
