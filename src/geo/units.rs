// src/geo/units.rs
// MENSURA: phantom-typed geodetic scalar algebra.
// Unit, height-system, CRS, and epoch mismatches are compile errors, not runtime bugs.
// RELEVANT FILES: src/geo/projections/mod.rs, src/geo/geoid.rs, src/camera/anchor.rs
//
//! The only sanctioned way to turn a world coordinate into render-space `f32`
//! is `crate::camera::anchor::Anchor::to_render_f32` — a grep-gated invariant
//! (tests/test_world_coord_f32_gate.py).
//!
//! Same-tag arithmetic compiles (this example also proves the imports and the
//! public validated constructors used by the `compile_fail` blocks below are
//! valid, so those blocks fail for the right reason — the type mismatch, not a
//! visibility error). The unchecked `geographic`/`ecef` constructors are
//! crate-private; the public surface builds coordinates through
//! `try_geographic`/`try_ecef`:
//!
//! ```
//! use forge3d::geo::units::{Angle, Coord, Degree, Ecef, Height, Itrf2014, Length, Metre, Wgs84};
//! let d = Length::<Metre>::new(2.0) + Length::<Metre>::new(0.5);
//! assert_eq!(d.value(), 2.5);
//! let a = Angle::<Degree>::new(1.5);
//! let h = Height::<forge3d::geo::units::Ellipsoidal>::new(10.0);
//! let c = Coord::<Wgs84, Itrf2014>::try_geographic(a, Angle::new(52.5), h).unwrap();
//! let e = Coord::<Ecef, Itrf2014>::try_ecef(4.0e6, 1.0e6, 4.8e6).unwrap();
//! let off = e - e;
//! assert_eq!((off.dx, off.dy, off.dz), (0.0, 0.0, 0.0));
//! let _ = (c.lon(), c.lat(), c.height());
//! ```
//!
//! # Uncompilable errors (rustdoc `compile_fail` proofs)
//!
//! Adding a length to an angle does not compile:
//!
//! ```compile_fail
//! use forge3d::geo::units::{Angle, Degree, Length, Metre};
//! let l = Length::<Metre>::new(1.0);
//! let a = Angle::<Degree>::new(1.0);
//! let _ = l + a; // ERROR: no `Add<Angle<Degree>>` for `Length<Metre>`
//! ```
//!
//! Mixing height systems does not compile — go through
//! `forge3d::geo::geoid::orthometric_to_ellipsoidal`:
//!
//! ```compile_fail
//! use forge3d::geo::units::{Egm96, Ellipsoidal, Height, Orthometric};
//! let e = Height::<Ellipsoidal>::new(100.0);
//! let o = Height::<Orthometric<Egm96>>::new(100.0);
//! let _ = e - o; // ERROR: height systems differ
//! ```
//!
//! Differencing coordinates from different epochs does not compile — go
//! through `forge3d::geo::units::epoch_transform`:
//!
//! ```compile_fail
//! use forge3d::geo::units::{Angle, Coord, Height, Itrf2000, Itrf2014, Wgs84};
//! let a = Coord::<Wgs84, Itrf2014>::try_geographic(
//!     Angle::new(13.4), Angle::new(52.5), Height::new(0.0)).unwrap();
//! let b = Coord::<Wgs84, Itrf2000>::try_geographic(
//!     Angle::new(13.4), Angle::new(52.5), Height::new(0.0)).unwrap();
//! let _ = a - b; // ERROR: reference epochs differ
//! ```
//!
//! Differencing coordinates from different CRSs does not compile:
//!
//! ```compile_fail
//! use forge3d::geo::units::{Angle, Coord, Ecef, Height, Itrf2014, Wgs84};
//! let a = Coord::<Wgs84, Itrf2014>::try_geographic(
//!     Angle::new(13.4), Angle::new(52.5), Height::new(0.0)).unwrap();
//! let b = Coord::<Ecef, Itrf2014>::try_ecef(4.0e6, 1.0e6, 4.8e6).unwrap();
//! let _ = a - b; // ERROR: CRS tags differ
//! ```
//!
//! Casting a world coordinate to `f32` without an `Anchor` does not compile —
//! accessors return typed scalars, never a bare `f64`:
//!
//! ```compile_fail
//! use forge3d::geo::units::{Angle, Coord, Height, Itrf2014, Wgs84};
//! let c = Coord::<Wgs84, Itrf2014>::try_geographic(
//!     Angle::new(13.4), Angle::new(52.5), Height::new(0.0)).unwrap();
//! let _ = c.lon() as f32; // ERROR: `Angle<Degree>` is not a primitive
//! ```
//!
//! Earth and Mars coordinates cannot be differenced:
//!
//! ```compile_fail
//! use forge3d::geo::units::{
//!     Angle, Coord, EllipsoidalOf, Height, Itrf2014, Mars, MarsIau2000, Wgs84,
//! };
//! let earth = Coord::<Wgs84, Itrf2014>::try_geographic(
//!     Angle::new(0.0), Angle::new(0.0), Height::new(0.0)).unwrap();
//! let mars = Coord::<MarsIau2000, Itrf2014>::try_geographic(
//!     Angle::new(0.0), Angle::new(0.0),
//!     Height::<EllipsoidalOf<Mars>>::new(0.0)).unwrap();
//! let _ = earth - mars; // ERROR: body-bearing CRS tags differ
//! ```
//!
//! EGM96 cannot be applied to a Mars ellipsoidal height:
//!
//! ```compile_fail
//! use forge3d::geo::geoid::ellipsoidal_to_orthometric;
//! use forge3d::geo::units::{Angle, EllipsoidalOf, Height, Mars};
//! let mars_height = Height::<EllipsoidalOf<Mars>>::new(21_900.0);
//! let _ = ellipsoidal_to_orthometric(
//!     mars_height, Angle::new(18.65), Angle::new(226.2));
//! // ERROR: EGM96 conversion requires an Earth ellipsoidal height
//! ```
//!
//! An Earth EPSG-tagged raster cannot be passed where a lunar raster is
//! expected (or vice versa):
//!
//! ```compile_fail
//! use core::marker::PhantomData;
//! use forge3d::geo::units::{CrsTag, MoonMe2000, Wgs84};
//! struct Raster<C: CrsTag>(PhantomData<C>);
//! fn open_earth_raster(_: Raster<Wgs84>) {}
//! let lunar = Raster::<MoonMe2000>(PhantomData);
//! open_earth_raster(lunar); // ERROR: lunar and Earth CRS tags differ
//! ```
//!
//! The unchecked `geographic`/`ecef` fast paths are crate-private, so a
//! downstream crate cannot bypass validation to construct a non-finite or
//! otherwise invalid typed coordinate — the only public path is `try_geographic`
//! / `try_ecef`, which reject bad input:
//!
//! ```compile_fail
//! use forge3d::geo::units::{Angle, Coord, Height, Itrf2014, Wgs84};
//! // ERROR: `geographic` is private outside the `forge3d` crate.
//! let _ = Coord::<Wgs84, Itrf2014>::geographic(
//!     Angle::new(f64::NAN), Angle::new(52.5), Height::new(0.0));
//! ```
use core::marker::PhantomData;
use core::ops::{Add, Div, Mul, Neg, Sub};

use glam::DVec3;

pub use super::body::{BodyTag, Earth, Mars, Moon};

mod sealed {
    pub trait Sealed {}
}
use sealed::Sealed;

// ---------------------------------------------------------------------------
// Length
// ---------------------------------------------------------------------------

/// Marker trait for linear units.
pub trait LengthUnit: Sealed + Copy + core::fmt::Debug + 'static {
    const METRES_PER_UNIT: f64;
    const SYMBOL: &'static str;
}

/// SI metre.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Metre {}
/// International foot (exactly 0.3048 m).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Foot {}
/// US survey foot (exactly 1200/3937 m).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum USSurveyFoot {}

impl Sealed for Metre {}
impl Sealed for Foot {}
impl Sealed for USSurveyFoot {}
impl LengthUnit for Metre {
    const METRES_PER_UNIT: f64 = 1.0;
    const SYMBOL: &'static str = "m";
}
impl LengthUnit for Foot {
    const METRES_PER_UNIT: f64 = 0.3048;
    const SYMBOL: &'static str = "ft";
}
impl LengthUnit for USSurveyFoot {
    const METRES_PER_UNIT: f64 = 1200.0 / 3937.0;
    const SYMBOL: &'static str = "ftUS";
}

/// A length in unit `U`. Arithmetic is only defined between identical units.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct Length<U: LengthUnit> {
    value: f64,
    _unit: PhantomData<U>,
}

impl<U: LengthUnit> Length<U> {
    pub const fn new(value: f64) -> Self {
        Self {
            value,
            _unit: PhantomData,
        }
    }
    /// Numeric value in this unit.
    pub const fn value(self) -> f64 {
        self.value
    }
    /// Explicit unit conversion.
    pub fn to<V: LengthUnit>(self) -> Length<V> {
        Length::new(self.value * U::METRES_PER_UNIT / V::METRES_PER_UNIT)
    }
    pub fn abs(self) -> Self {
        Self::new(self.value.abs())
    }
}

impl<U: LengthUnit> Add for Length<U> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::new(self.value + rhs.value)
    }
}
impl<U: LengthUnit> Sub for Length<U> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.value - rhs.value)
    }
}
impl<U: LengthUnit> Neg for Length<U> {
    type Output = Self;
    fn neg(self) -> Self {
        Self::new(-self.value)
    }
}
impl<U: LengthUnit> Mul<f64> for Length<U> {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self {
        Self::new(self.value * rhs)
    }
}
impl<U: LengthUnit> Div<f64> for Length<U> {
    type Output = Self;
    fn div(self, rhs: f64) -> Self {
        Self::new(self.value / rhs)
    }
}

// ---------------------------------------------------------------------------
// Angle
// ---------------------------------------------------------------------------

/// Marker trait for angular units.
pub trait AngleUnit: Sealed + Copy + core::fmt::Debug + 'static {
    const RADIANS_PER_UNIT: f64;
    const SYMBOL: &'static str;
}

/// Sexagesimal degree.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Degree {}
/// SI radian.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Radian {}
/// Gradian (400 per turn).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Grad {}

impl Sealed for Degree {}
impl Sealed for Radian {}
impl Sealed for Grad {}
impl AngleUnit for Degree {
    const RADIANS_PER_UNIT: f64 = core::f64::consts::PI / 180.0;
    const SYMBOL: &'static str = "deg";
}
impl AngleUnit for Radian {
    const RADIANS_PER_UNIT: f64 = 1.0;
    const SYMBOL: &'static str = "rad";
}
impl AngleUnit for Grad {
    const RADIANS_PER_UNIT: f64 = core::f64::consts::PI / 200.0;
    const SYMBOL: &'static str = "gon";
}

/// An angle in unit `U`. Arithmetic is only defined between identical units.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct Angle<U: AngleUnit> {
    value: f64,
    _unit: PhantomData<U>,
}

impl<U: AngleUnit> Angle<U> {
    pub const fn new(value: f64) -> Self {
        Self {
            value,
            _unit: PhantomData,
        }
    }
    pub const fn value(self) -> f64 {
        self.value
    }
    pub fn to<V: AngleUnit>(self) -> Angle<V> {
        Angle::new(self.value * U::RADIANS_PER_UNIT / V::RADIANS_PER_UNIT)
    }
    pub fn radians(self) -> f64 {
        self.value * U::RADIANS_PER_UNIT
    }
    pub fn abs(self) -> Self {
        Self::new(self.value.abs())
    }
}

impl<U: AngleUnit> Add for Angle<U> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::new(self.value + rhs.value)
    }
}
impl<U: AngleUnit> Sub for Angle<U> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.value - rhs.value)
    }
}
impl<U: AngleUnit> Neg for Angle<U> {
    type Output = Self;
    fn neg(self) -> Self {
        Self::new(-self.value)
    }
}
impl<U: AngleUnit> Mul<f64> for Angle<U> {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self {
        Self::new(self.value * rhs)
    }
}
impl<U: AngleUnit> Div<f64> for Angle<U> {
    type Output = Self;
    fn div(self, rhs: f64) -> Self {
        Self::new(self.value / rhs)
    }
}

// ---------------------------------------------------------------------------
// Height systems
// ---------------------------------------------------------------------------

/// Marker trait for geoid models a height can be referenced to.
pub trait GeoidModel: Sealed + Copy + core::fmt::Debug + 'static {
    type Body: BodyTag;
    const NAME: &'static str;
}

/// The EGM96 geoid (see `crate::geo::geoid`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Egm96 {}
impl Sealed for Egm96 {}
impl GeoidModel for Egm96 {
    type Body = Earth;
    const NAME: &'static str = "EGM96";
}

/// The GMM3-derived Mars areoid (see `crate::geo::geoid`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MarsAreoid {}
impl Sealed for MarsAreoid {}
impl GeoidModel for MarsAreoid {
    type Body = Mars;
    const NAME: &'static str = "Mars areoid";
}

/// Marker trait for vertical reference systems.
pub trait HeightSystem: Sealed + Copy + core::fmt::Debug + 'static {
    type Body: BodyTag;
    const NAME: &'static str;
}

/// Height above body `B`'s reference ellipsoid.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EllipsoidalOf<B: BodyTag> {
    _body: PhantomData<B>,
}
impl<B: BodyTag> Sealed for EllipsoidalOf<B> {}
impl<B: BodyTag> HeightSystem for EllipsoidalOf<B> {
    type Body = B;
    const NAME: &'static str = "ellipsoidal";
}
/// Backward-compatible WGS84 ellipsoidal height.
pub type Ellipsoidal = EllipsoidalOf<Earth>;

/// Height above the geoid `G` (what a DEM usually encodes).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Orthometric<G: GeoidModel> {
    _geoid: PhantomData<G>,
}
impl<G: GeoidModel> Sealed for Orthometric<G> {}
impl<G: GeoidModel> HeightSystem for Orthometric<G> {
    type Body = G::Body;
    const NAME: &'static str = "orthometric";
}

/// Height above a nautical chart datum (type slot only; no conversions shipped).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChartDatumOf<B: BodyTag> {
    _body: PhantomData<B>,
}
impl<B: BodyTag> Sealed for ChartDatumOf<B> {}
impl<B: BodyTag> HeightSystem for ChartDatumOf<B> {
    type Body = B;
    const NAME: &'static str = "chart_datum";
}
pub type ChartDatum = ChartDatumOf<Earth>;

/// A height in metres referenced to system `S`. Heights in different systems
/// cannot be mixed; convert via `crate::geo::geoid`.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct Height<S: HeightSystem> {
    metres: f64,
    _system: PhantomData<S>,
}

impl<S: HeightSystem> Height<S> {
    pub const fn new(metres: f64) -> Self {
        Self {
            metres,
            _system: PhantomData,
        }
    }
    pub const fn metres(self) -> f64 {
        self.metres
    }
}

/// Raising/lowering a height by a metric length stays in the same system.
impl<S: HeightSystem> Add<Length<Metre>> for Height<S> {
    type Output = Self;
    fn add(self, rhs: Length<Metre>) -> Self {
        Self::new(self.metres + rhs.value())
    }
}
impl<S: HeightSystem> Sub<Length<Metre>> for Height<S> {
    type Output = Self;
    fn sub(self, rhs: Length<Metre>) -> Self {
        Self::new(self.metres - rhs.value())
    }
}
/// The difference of two heights in the SAME system is a length.
impl<S: HeightSystem> Sub for Height<S> {
    type Output = Length<Metre>;
    fn sub(self, rhs: Self) -> Length<Metre> {
        Length::new(self.metres - rhs.metres)
    }
}

// ---------------------------------------------------------------------------
// CRS and epoch tags
// ---------------------------------------------------------------------------

/// Marker trait for coordinate reference systems a `Coord` can be tagged with.
pub trait CrsTag: Sealed + Copy + core::fmt::Debug + 'static {
    type Body: BodyTag;
    const AUTHORITY: &'static str;
    const CODE: u32;
    const NAME: &'static str;
}

/// Geographic WGS84 (EPSG:4326), lon/lat degrees + ellipsoidal height.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Wgs84 {}
/// Spherical Web Mercator (EPSG:3857), metres.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WebMercator {}
/// Moon ME2000 planetocentric longitude/latitude, positive east.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MoonMe2000 {}
/// Mars IAU2000 planetocentric longitude/latitude, positive east.
///
/// Planetographic latitude is deliberately unsupported in SELENE v1.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MarsIau2000 {}
/// Body-centred, body-fixed geocentric metres for body `B`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EcefOf<B: BodyTag> {
    _body: PhantomData<B>,
}
/// Backward-compatible Earth-centred earth-fixed geocentric frame.
pub type Ecef = EcefOf<Earth>;

impl Sealed for Wgs84 {}
impl Sealed for WebMercator {}
impl Sealed for MoonMe2000 {}
impl Sealed for MarsIau2000 {}
impl<B: BodyTag> Sealed for EcefOf<B> {}
impl CrsTag for Wgs84 {
    type Body = Earth;
    const AUTHORITY: &'static str = "EPSG";
    const CODE: u32 = 4326;
    const NAME: &'static str = "WGS 84";
}
impl CrsTag for WebMercator {
    type Body = Earth;
    const AUTHORITY: &'static str = "EPSG";
    const CODE: u32 = 3857;
    const NAME: &'static str = "WGS 84 / Pseudo-Mercator";
}
impl CrsTag for MoonMe2000 {
    type Body = Moon;
    const AUTHORITY: &'static str = "IAU";
    const CODE: u32 = 30100;
    const NAME: &'static str = "Moon (2015) planetocentric +East";
}
impl CrsTag for MarsIau2000 {
    type Body = Mars;
    const AUTHORITY: &'static str = "IAU";
    const CODE: u32 = 49902;
    const NAME: &'static str = "Mars (2015) planetocentric +East";
}
impl<B: BodyTag> CrsTag for EcefOf<B> {
    type Body = B;
    const AUTHORITY: &'static str = B::BODY_FIXED_AUTHORITY;
    const CODE: u32 = B::BODY_FIXED_CODE;
    const NAME: &'static str = B::BODY_FIXED_NAME;
}

/// Marker trait for reference-frame realization epochs (ITRF sense).
pub trait EpochTag: Sealed + Copy + core::fmt::Debug + 'static {
    const NAME: &'static str;
}

/// ITRF2014 realization.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Itrf2014 {}
/// ITRF2008 realization.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Itrf2008 {}
/// ITRF2000 realization.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Itrf2000 {}

impl Sealed for Itrf2014 {}
impl Sealed for Itrf2008 {}
impl Sealed for Itrf2000 {}
impl EpochTag for Itrf2014 {
    const NAME: &'static str = "ITRF2014";
}
impl EpochTag for Itrf2008 {
    const NAME: &'static str = "ITRF2008";
}
impl EpochTag for Itrf2000 {
    const NAME: &'static str = "ITRF2000";
}

// ---------------------------------------------------------------------------
// Coord
// ---------------------------------------------------------------------------

/// Marker for CRSs whose axes are geographic lon/lat degrees.
pub trait GeographicCrs: CrsTag {}
impl GeographicCrs for Wgs84 {}
impl GeographicCrs for MoonMe2000 {}
impl GeographicCrs for MarsIau2000 {}

/// Marker for CRSs whose axes are projected/geocentric metres.
pub trait MetricCrs: CrsTag {}
impl MetricCrs for WebMercator {}
impl<B: BodyTag> MetricCrs for EcefOf<B> {}

/// A position tagged with its CRS and its reference epoch. Coordinates from
/// different CRSs or epochs cannot be differenced; convert first.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Coord<C: CrsTag, E: EpochTag> {
    x: f64,
    y: f64,
    z: f64,
    _crs: PhantomData<C>,
    _epoch: PhantomData<E>,
}

impl<C: GeographicCrs, E: EpochTag> Coord<C, E> {
    /// Build a geographic coordinate from typed lon/lat/height.
    ///
    /// This constructor does not validate its inputs (it stays `const`-friendly
    /// and is used by internal fast paths). It is **crate-private**: every public
    /// trust boundary must go through [`Coord::try_geographic`], which rejects
    /// non-finite values and out-of-range latitudes before any numerical work, so
    /// no non-finite/invalid typed coordinate can be constructed from outside the
    /// crate.
    pub(crate) fn geographic(
        lon: Angle<Degree>,
        lat: Angle<Degree>,
        h: Height<EllipsoidalOf<C::Body>>,
    ) -> Self {
        Self {
            x: lon.value(),
            y: lat.value(),
            z: h.metres(),
            _crs: PhantomData,
            _epoch: PhantomData,
        }
    }

    /// Validated geographic constructor for trust boundaries: rejects
    /// non-finite longitude/latitude/height and latitudes outside [-90, 90].
    /// Longitude is not wrapped here (that is a topology decision made later);
    /// it is only required to be finite.
    pub fn try_geographic(
        lon: Angle<Degree>,
        lat: Angle<Degree>,
        h: Height<EllipsoidalOf<C::Body>>,
    ) -> Result<Self, GeoInputError> {
        if !lon.value().is_finite() {
            return Err(GeoInputError::NotFinite("longitude"));
        }
        if !lat.value().is_finite() {
            return Err(GeoInputError::NotFinite("latitude"));
        }
        if !h.metres().is_finite() {
            return Err(GeoInputError::NotFinite("height"));
        }
        if lat.value().abs() > 90.0 {
            return Err(GeoInputError::LatitudeOutOfRange(lat.value()));
        }
        Ok(Self::geographic(lon, lat, h))
    }
    pub fn lon(&self) -> Angle<Degree> {
        Angle::new(self.x)
    }
    pub fn lat(&self) -> Angle<Degree> {
        Angle::new(self.y)
    }
    pub fn height(&self) -> Height<EllipsoidalOf<C::Body>> {
        Height::new(self.z)
    }
}

impl<B: BodyTag, E: EpochTag> Coord<EcefOf<B>, E> {
    /// Build a geocentric coordinate from raw ECEF metres.
    ///
    /// **Crate-private** unchecked fast path: every public trust boundary must go
    /// through [`Coord::try_ecef`], which rejects non-finite components before any
    /// numerical work.
    pub(crate) fn ecef(x: f64, y: f64, z: f64) -> Self {
        Self {
            x,
            y,
            z,
            _crs: PhantomData,
            _epoch: PhantomData,
        }
    }

    /// Validated geocentric constructor for trust boundaries: rejects
    /// non-finite ECEF components before any numerical work.
    pub fn try_ecef(x: f64, y: f64, z: f64) -> Result<Self, GeoInputError> {
        if !x.is_finite() || !y.is_finite() || !z.is_finite() {
            return Err(GeoInputError::NotFinite("ecef component"));
        }
        Ok(Self::ecef(x, y, z))
    }
    pub fn x(&self) -> Length<Metre> {
        Length::new(self.x)
    }
    pub fn y(&self) -> Length<Metre> {
        Length::new(self.y)
    }
    pub fn z(&self) -> Length<Metre> {
        Length::new(self.z)
    }
}

impl<C: CrsTag, E: EpochTag> Coord<C, E> {
    /// Raw f64 triple, crate-internal. The public surface only exposes typed
    /// scalars; the render-space `f32` cliff lives solely in
    /// `crate::camera::anchor::Anchor::to_render_f32`.
    pub(crate) fn raw(&self) -> DVec3 {
        DVec3::new(self.x, self.y, self.z)
    }

    pub(crate) fn from_raw(v: DVec3) -> Self {
        Self {
            x: v.x,
            y: v.y,
            z: v.z,
            _crs: PhantomData,
            _epoch: PhantomData,
        }
    }
}

/// Metric offset between two coordinates of identical CRS and epoch.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CoordOffset<C: CrsTag> {
    pub dx: f64,
    pub dy: f64,
    pub dz: f64,
    _crs: PhantomData<C>,
}

impl<C: CrsTag, E: EpochTag> Sub for Coord<C, E> {
    type Output = CoordOffset<C>;
    fn sub(self, rhs: Self) -> CoordOffset<C> {
        CoordOffset {
            dx: self.x - rhs.x,
            dy: self.y - rhs.y,
            dz: self.z - rhs.z,
            _crs: PhantomData,
        }
    }
}

// ---------------------------------------------------------------------------
// Datum / epoch transforms (WGS84 ↔ ITRF-family Helmert path only)
// ---------------------------------------------------------------------------

/// Error raised by the validated (`try_*`) geodetic constructors and
/// transforms when an input is unfit for numerical work.
#[derive(thiserror::Error, Debug, Clone, PartialEq)]
pub enum GeoInputError {
    /// A required scalar was NaN or infinite.
    #[error("non-finite {0}")]
    NotFinite(&'static str),
    /// A latitude fell outside the closed interval [-90, 90] degrees.
    #[error("latitude {0} out of range [-90, 90]")]
    LatitudeOutOfRange(f64),
    /// A Helmert parameter was NaN or infinite.
    #[error("non-finite Helmert parameter")]
    NonFiniteHelmert,
    /// A coordinate epoch (decimal year) differed from the frame's reference
    /// epoch, and no station-velocity model is shipped to bridge the gap.
    #[error("coordinate epoch {got} unsupported: shipped ITRF Helmerts are velocity-free and valid only at reference epoch {reference}")]
    NonReferenceEpoch { got: f64, reference: f64 },
}

/// Reference epoch (decimal year) at which the shipped ITRF Helmert parameters
/// are defined (Altamimi et al. 2016, *ITRF2014: A new release of the
/// International Terrestrial Reference Frame*, J. Geophys. Res., Table 1).
///
/// Frame realization and coordinate epoch are distinct concepts. The
/// [`EpochTag`] type parameter on a [`Coord`] names the *frame realization*
/// (ITRF2000 / ITRF2008 / ITRF2014). The *coordinate epoch* — the time of
/// observation at which a station's position is expressed — is a separate
/// scalar. The 7-parameter Helmerts shipped here are velocity-free: they are
/// exact at [`ITRF_REFERENCE_EPOCH`] and ignore tectonic plate motion (up to a
/// few cm/decade) at any other coordinate epoch. forge3d ships no
/// station-velocity/plate-motion model, so a transform requested at a
/// non-reference coordinate epoch is rejected rather than silently applied
/// (see [`epoch_transform_at`]); it is never treated as an identity.
pub const ITRF_REFERENCE_EPOCH: f64 = 2010.0;

/// A 7-parameter Helmert transform (position-vector convention, small angles).
/// Translations in metres, rotations in radians, `scale_ppb` in parts-per-billion.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Helmert {
    pub tx: f64,
    pub ty: f64,
    pub tz: f64,
    pub rx: f64,
    pub ry: f64,
    pub rz: f64,
    pub scale_ppb: f64,
}

impl Helmert {
    pub const IDENTITY: Helmert = Helmert {
        tx: 0.0,
        ty: 0.0,
        tz: 0.0,
        rx: 0.0,
        ry: 0.0,
        rz: 0.0,
        scale_ppb: 0.0,
    };

    /// Apply to a geocentric position (position-vector rotation convention).
    pub fn apply(&self, p: DVec3) -> DVec3 {
        let s = 1.0 + self.scale_ppb * 1e-9;
        DVec3::new(
            self.tx + s * (p.x - self.rz * p.y + self.ry * p.z),
            self.ty + s * (self.rz * p.x + p.y - self.rx * p.z),
            self.tz + s * (-self.ry * p.x + self.rx * p.y + p.z),
        )
    }

    /// Exact-enough inverse for small-angle Helmert (negated parameters).
    pub fn inverse(&self) -> Helmert {
        Helmert {
            tx: -self.tx,
            ty: -self.ty,
            tz: -self.tz,
            rx: -self.rx,
            ry: -self.ry,
            rz: -self.rz,
            scale_ppb: -self.scale_ppb,
        }
    }

    /// True when every parameter is finite (guard before numerical use).
    pub fn is_finite(&self) -> bool {
        self.tx.is_finite()
            && self.ty.is_finite()
            && self.tz.is_finite()
            && self.rx.is_finite()
            && self.ry.is_finite()
            && self.rz.is_finite()
            && self.scale_ppb.is_finite()
    }
}

/// A typed datum/epoch transform: the only sanctioned way to move a `Coord`
/// between epoch tags. Only the WGS84/ITRF-family Helmert path ships; grid
/// shifts (NTv2/NADCON) are deliberately out of scope.
pub trait DatumTransform<From: EpochTag, To: EpochTag> {
    fn helmert(&self) -> Helmert;
}

// The four published directions below all derive from a single authoritative
// row of Altamimi et al. 2016, Table 1 (ITRF2014 → ITRFyyyy at epoch 2010.0,
// units: mm / ppb / mas, rotations zero for these pairs). Reverse directions
// are the negated parameters (exact here because all rotations are zero).
//
//   ITRF2014 → ITRF2008 : T = ( 1.6,  1.9,  2.4) mm, D = -0.02 ppb
//   ITRF2014 → ITRF2000 : T = ( 0.7,  1.2, -26.1) mm, D =  2.12 ppb

/// ITRF2014 → ITRF2008 Helmert (Altamimi et al. 2016, Table 1, epoch 2010.0).
#[derive(Clone, Copy, Debug, Default)]
pub struct Itrf2014ToItrf2008;
impl DatumTransform<Itrf2014, Itrf2008> for Itrf2014ToItrf2008 {
    fn helmert(&self) -> Helmert {
        Helmert {
            tx: 1.6e-3,
            ty: 1.9e-3,
            tz: 2.4e-3,
            rx: 0.0,
            ry: 0.0,
            rz: 0.0,
            scale_ppb: -0.02,
        }
    }
}

/// ITRF2008 → ITRF2014 Helmert (reverse of [`Itrf2014ToItrf2008`]).
#[derive(Clone, Copy, Debug, Default)]
pub struct Itrf2008ToItrf2014;
impl DatumTransform<Itrf2008, Itrf2014> for Itrf2008ToItrf2014 {
    fn helmert(&self) -> Helmert {
        Itrf2014ToItrf2008.helmert().inverse()
    }
}

/// ITRF2014 → ITRF2000 Helmert (Altamimi et al. 2016, Table 1, epoch 2010.0).
#[derive(Clone, Copy, Debug, Default)]
pub struct Itrf2014ToItrf2000;
impl DatumTransform<Itrf2014, Itrf2000> for Itrf2014ToItrf2000 {
    fn helmert(&self) -> Helmert {
        Helmert {
            tx: 0.7e-3,
            ty: 1.2e-3,
            tz: -26.1e-3,
            rx: 0.0,
            ry: 0.0,
            rz: 0.0,
            scale_ppb: 2.12,
        }
    }
}

/// ITRF2000 → ITRF2014 Helmert (reverse of [`Itrf2014ToItrf2000`]; equal to the
/// IERS/IGN published ITRF2000→ITRF2014 direction, T = (-0.7, -1.2, 26.1) mm,
/// D = -2.12 ppb, R = 0).
#[derive(Clone, Copy, Debug, Default)]
pub struct Itrf2000ToItrf2014;
impl DatumTransform<Itrf2000, Itrf2014> for Itrf2000ToItrf2014 {
    fn helmert(&self) -> Helmert {
        Itrf2014ToItrf2000.helmert().inverse()
    }
}

/// Move a geocentric coordinate between frame realizations through a typed
/// transform. This applies the velocity-free Helmert unconditionally; it is
/// only correct at [`ITRF_REFERENCE_EPOCH`]. Prefer [`epoch_transform_at`] at a
/// trust boundary, which validates the coordinate epoch and inputs.
pub fn epoch_transform<From, To, T>(coord: Coord<Ecef, From>, transform: &T) -> Coord<Ecef, To>
where
    From: EpochTag,
    To: EpochTag,
    T: DatumTransform<From, To>,
{
    Coord::from_raw(transform.helmert().apply(coord.raw()))
}

/// Validated frame-realization transform at an explicit coordinate epoch.
///
/// Rejects non-finite coordinates or Helmert parameters, and — because no
/// station-velocity model is shipped — rejects any `coordinate_epoch` that is
/// not [`ITRF_REFERENCE_EPOCH`] instead of silently ignoring plate motion. This
/// makes an unsupported epoch an explicit error, never an identity.
pub fn epoch_transform_at<From, To, T>(
    coord: Coord<Ecef, From>,
    transform: &T,
    coordinate_epoch: f64,
) -> Result<Coord<Ecef, To>, GeoInputError>
where
    From: EpochTag,
    To: EpochTag,
    T: DatumTransform<From, To>,
{
    if !coordinate_epoch.is_finite() {
        return Err(GeoInputError::NotFinite("coordinate epoch"));
    }
    if (coordinate_epoch - ITRF_REFERENCE_EPOCH).abs() > f64::EPSILON {
        return Err(GeoInputError::NonReferenceEpoch {
            got: coordinate_epoch,
            reference: ITRF_REFERENCE_EPOCH,
        });
    }
    let raw = coord.raw();
    if !raw.x.is_finite() || !raw.y.is_finite() || !raw.z.is_finite() {
        return Err(GeoInputError::NotFinite("ecef component"));
    }
    let helmert = transform.helmert();
    if !helmert.is_finite() {
        return Err(GeoInputError::NonFiniteHelmert);
    }
    Ok(Coord::from_raw(helmert.apply(raw)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn planetary_geographic_tags_use_iau_without_body_fixed_impersonation() {
        assert_eq!(Wgs84::AUTHORITY, "EPSG");
        assert_eq!(Wgs84::CODE, 4326);
        assert_eq!(MoonMe2000::AUTHORITY, "IAU");
        assert_eq!(MoonMe2000::CODE, 30100);
        assert_eq!(MarsIau2000::AUTHORITY, "IAU");
        assert_eq!(MarsIau2000::CODE, 49902);
        assert_eq!(EcefOf::<Moon>::AUTHORITY, "FORGE3D");
        assert_eq!(EcefOf::<Moon>::CODE, 301);
        assert_eq!(EcefOf::<Mars>::AUTHORITY, "FORGE3D");
        assert_eq!(EcefOf::<Mars>::CODE, 499);
    }

    #[test]
    fn length_unit_conversions_are_exact() {
        let ft = Length::<USSurveyFoot>::new(3937.0);
        assert!((ft.to::<Metre>().value() - 1200.0).abs() < 1e-9);
        let m = Length::<Metre>::new(0.3048);
        assert!((m.to::<Foot>().value() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn angle_unit_conversions_are_exact() {
        let d = Angle::<Degree>::new(180.0);
        assert!((d.to::<Radian>().value() - core::f64::consts::PI).abs() < 1e-15);
        assert!((d.to::<Grad>().value() - 200.0).abs() < 1e-12);
    }

    #[test]
    fn same_system_height_difference_is_a_length() {
        let a = Height::<Ellipsoidal>::new(120.5);
        let b = Height::<Ellipsoidal>::new(100.0);
        assert!(((a - b).value() - 20.5).abs() < 1e-12);
    }

    #[test]
    fn same_tag_coord_difference_compiles_and_is_metric() {
        let a = Coord::<Ecef, Itrf2014>::ecef(1.0, 2.0, 3.0);
        let b = Coord::<Ecef, Itrf2014>::ecef(0.5, 0.5, 0.5);
        let d = a - b;
        assert_eq!((d.dx, d.dy, d.dz), (0.5, 1.5, 2.5));
    }

    #[test]
    fn helmert_roundtrip_closes_below_a_tenth_of_a_millimetre() {
        let t = Itrf2000ToItrf2014.helmert();
        let p = DVec3::new(4_027_894.006, 307_045.600, 4_919_474.910);
        let back = t.inverse().apply(t.apply(p));
        assert!((back - p).length() < 1e-4);
    }

    #[test]
    fn epoch_transform_changes_the_tag() {
        let c2000 = Coord::<Ecef, Itrf2000>::ecef(4_027_894.006, 307_045.600, 4_919_474.910);
        let c2014: Coord<Ecef, Itrf2014> = epoch_transform(c2000, &Itrf2000ToItrf2014);
        // ΔZ = tz + (scale − 1)·Z = 26.1 mm − 2.12 ppb × 4.919e6 m.
        let expected = 0.0261 - 2.12e-9 * 4_919_474.910;
        assert!((c2014.z().value() - c2000.raw().z - expected).abs() < 1e-9);
    }

    // A representative geocentric point (Potsdam-ish) used across round trips.
    const P: DVec3 = DVec3::new(3_800_641.0, 882_005.0, 5_028_791.0);

    #[test]
    fn every_shipped_itrf_direction_round_trips_below_a_tenth_of_a_millimetre() {
        // 2014 → 2008 → 2014
        let a = Coord::<Ecef, Itrf2014>::ecef(P.x, P.y, P.z);
        let b: Coord<Ecef, Itrf2008> = epoch_transform(a, &Itrf2014ToItrf2008);
        let c: Coord<Ecef, Itrf2014> = epoch_transform(b, &Itrf2008ToItrf2014);
        assert!((c.raw() - a.raw()).length() < 1e-4);
        // 2014 → 2000 → 2014
        let d: Coord<Ecef, Itrf2000> = epoch_transform(a, &Itrf2014ToItrf2000);
        let e: Coord<Ecef, Itrf2014> = epoch_transform(d, &Itrf2000ToItrf2014);
        assert!((e.raw() - a.raw()).length() < 1e-4);
    }

    #[test]
    fn reverse_transforms_are_exact_inverses_of_forward() {
        assert_eq!(
            Itrf2008ToItrf2014.helmert(),
            Itrf2014ToItrf2008.helmert().inverse()
        );
        assert_eq!(
            Itrf2000ToItrf2014.helmert(),
            Itrf2014ToItrf2000.helmert().inverse()
        );
    }

    #[test]
    fn itrf2014_to_itrf2008_matches_published_translation() {
        // Pure translation dominates at these magnitudes: ΔX ≈ 1.6 mm, etc.
        let a = Coord::<Ecef, Itrf2014>::ecef(P.x, P.y, P.z);
        let b: Coord<Ecef, Itrf2008> = epoch_transform(a, &Itrf2014ToItrf2008);
        let off = b.raw() - a.raw();
        // Translation + scale only; scale term ~ -0.02 ppb × 6.4e6 m ≈ -0.13 mm.
        assert!((off.x - (1.6e-3 - 0.02e-9 * P.x)).abs() < 1e-9);
        assert!((off.z - (2.4e-3 - 0.02e-9 * P.z)).abs() < 1e-9);
    }

    #[test]
    fn try_geographic_rejects_bad_inputs_and_accepts_good_ones() {
        let ok = Coord::<Wgs84, Itrf2014>::try_geographic(
            Angle::new(13.4),
            Angle::new(52.5),
            Height::new(38.0),
        );
        assert!(ok.is_ok());
        assert_eq!(
            Coord::<Wgs84, Itrf2014>::try_geographic(
                Angle::new(f64::NAN),
                Angle::new(52.5),
                Height::new(0.0)
            ),
            Err(GeoInputError::NotFinite("longitude"))
        );
        assert!(matches!(
            Coord::<Wgs84, Itrf2014>::try_geographic(
                Angle::new(13.4),
                Angle::new(95.0),
                Height::new(0.0)
            ),
            Err(GeoInputError::LatitudeOutOfRange(_))
        ));
        assert_eq!(
            Coord::<Wgs84, Itrf2014>::try_geographic(
                Angle::new(13.4),
                Angle::new(52.5),
                Height::new(f64::INFINITY)
            ),
            Err(GeoInputError::NotFinite("height"))
        );
    }

    #[test]
    fn try_ecef_rejects_non_finite() {
        assert!(Coord::<Ecef, Itrf2014>::try_ecef(P.x, P.y, P.z).is_ok());
        assert_eq!(
            Coord::<Ecef, Itrf2014>::try_ecef(P.x, f64::NAN, P.z),
            Err(GeoInputError::NotFinite("ecef component"))
        );
    }

    #[test]
    fn epoch_transform_at_rejects_non_reference_epoch() {
        let a = Coord::<Ecef, Itrf2000>::ecef(P.x, P.y, P.z);
        // At the reference epoch it matches the unchecked transform.
        let checked: Coord<Ecef, Itrf2014> =
            epoch_transform_at(a, &Itrf2000ToItrf2014, ITRF_REFERENCE_EPOCH).unwrap();
        let unchecked: Coord<Ecef, Itrf2014> = epoch_transform(a, &Itrf2000ToItrf2014);
        assert_eq!(checked.raw(), unchecked.raw());
        // At any other coordinate epoch it refuses rather than ignoring drift.
        assert!(matches!(
            epoch_transform_at::<Itrf2000, Itrf2014, _>(a, &Itrf2000ToItrf2014, 2020.0),
            Err(GeoInputError::NonReferenceEpoch { .. })
        ));
    }
}
