use half::f16;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::{MediaError, Medium};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Bounds3 {
    pub min: [f32; 3],
    pub max: [f32; 3],
}

impl Bounds3 {
    pub fn validate(self) -> Result<(), MediaError> {
        if (0..3).all(|axis| {
            self.min[axis].is_finite()
                && self.max[axis].is_finite()
                && self.min[axis] < self.max[axis]
        }) {
            Ok(())
        } else {
            Err(MediaError::InvalidDensity(
                "bounds must be finite and have positive extent".into(),
            ))
        }
    }

    pub fn contains(self, point: [f32; 3]) -> bool {
        (0..3).all(|axis| point[axis] >= self.min[axis] && point[axis] <= self.max[axis])
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SpatialTransform {
    pub bounds: Bounds3,
}

impl SpatialTransform {
    pub fn world_to_unit(self, point: [f32; 3]) -> Option<[f32; 3]> {
        if !self.bounds.contains(point) {
            return None;
        }
        Some(std::array::from_fn(|axis| {
            (point[axis] - self.bounds.min[axis]) / (self.bounds.max[axis] - self.bounds.min[axis])
        }))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DensityMapping {
    pub physical_density_per_authored_unit: f32,
}

impl DensityMapping {
    pub fn validate(self) -> Result<(), MediaError> {
        if self.physical_density_per_authored_unit.is_finite()
            && self.physical_density_per_authored_unit >= 0.0
        {
            Ok(())
        } else {
            Err(MediaError::InvalidDensity(
                "density mapping must be finite and non-negative".into(),
            ))
        }
    }

    pub fn to_physical(self, authored_density: f32) -> f32 {
        authored_density * self.physical_density_per_authored_unit
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Homogeneous {
    pub authored_density: f32,
    pub mapping: DensityMapping,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PerlinWorley {
    pub transform: SpatialTransform,
    pub frequency: f32,
    pub octaves: u32,
    pub seed: u64,
    pub worley_weight: f32,
    pub mapping: DensityMapping,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Grid3D {
    transform: SpatialTransform,
    dimensions: [u32; 3],
    /// Exact R16Float x-major, then y, then z samples used by transport.
    r16_density: Vec<u16>,
    mapping: DensityMapping,
}

impl Grid3D {
    pub fn new(
        transform: SpatialTransform,
        dimensions: [u32; 3],
        authored_density: Vec<f32>,
        mapping: DensityMapping,
    ) -> Result<Self, MediaError> {
        if authored_density
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(MediaError::InvalidDensity(
                "grid density must be finite and non-negative".into(),
            ));
        }
        Self::from_r16_bits(
            transform,
            dimensions,
            authored_density
                .into_iter()
                .map(|value| f16::from_f32(value).to_bits())
                .collect(),
            mapping,
        )
    }

    pub fn from_r16_bits(
        transform: SpatialTransform,
        dimensions: [u32; 3],
        r16_density: Vec<u16>,
        mapping: DensityMapping,
    ) -> Result<Self, MediaError> {
        let grid = Self {
            transform,
            dimensions,
            r16_density,
            mapping,
        };
        grid.validate()?;
        Ok(grid)
    }

    pub fn transform(&self) -> SpatialTransform {
        self.transform
    }

    pub fn dimensions(&self) -> [u32; 3] {
        self.dimensions
    }

    pub fn mapping(&self) -> DensityMapping {
        self.mapping
    }

    pub fn r16_density_bits(&self) -> &[u16] {
        &self.r16_density
    }

    pub fn dequantized_density(&self) -> Vec<f32> {
        self.r16_density
            .iter()
            .map(|bits| f16::from_bits(*bits).to_f32())
            .collect()
    }

    fn node(&self, index: [u32; 3]) -> f32 {
        f16::from_bits(self.r16_density[flat_index(self.dimensions, index)]).to_f32()
    }

    fn validate(&self) -> Result<(), MediaError> {
        self.transform.bounds.validate()?;
        self.mapping.validate()?;
        if self.dimensions.contains(&0) {
            return Err(MediaError::InvalidDensity(
                "grid dimensions must be nonzero".into(),
            ));
        }
        let count = self.dimensions.into_iter().try_fold(1usize, |n, axis| {
            n.checked_mul(axis as usize).ok_or_else(|| {
                MediaError::InvalidDensity("grid dimensions overflow address space".into())
            })
        })?;
        if self.r16_density.len() != count {
            return Err(MediaError::InvalidDensity(format!(
                "grid has {} R16 values but dimensions require {count}",
                self.r16_density.len()
            )));
        }
        if self.r16_density.iter().any(|bits| {
            let value = f16::from_bits(*bits).to_f32();
            !value.is_finite() || value < 0.0
        }) {
            return Err(MediaError::InvalidDensity(
                "R16 grid density must be finite and non-negative".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DensityField {
    Homogeneous(Homogeneous),
    PerlinWorley(PerlinWorley),
    Grid3D(Grid3D),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DensityIdentity {
    pub version: u64,
    pub digest: [u8; 32],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MediumIdentity {
    pub version: u64,
    pub digest: [u8; 32],
}

impl DensityField {
    pub fn validate(&self) -> Result<(), MediaError> {
        match self {
            Self::Homogeneous(field) => {
                validate_authored(field.authored_density)?;
                field.mapping.validate()
            }
            Self::PerlinWorley(field) => {
                field.transform.bounds.validate()?;
                field.mapping.validate()?;
                if !field.frequency.is_finite() || field.frequency <= 0.0 {
                    return Err(MediaError::InvalidDensity(
                        "Perlin-Worley frequency must be finite and positive".into(),
                    ));
                }
                if field.octaves == 0 {
                    return Err(MediaError::InvalidDensity(
                        "Perlin-Worley must have at least one octave".into(),
                    ));
                }
                let mut frequency = field.frequency;
                for _ in 1..field.octaves {
                    frequency *= 2.0;
                    if !frequency.is_finite() {
                        return Err(MediaError::InvalidDensity(
                            "Perlin-Worley octave frequency overflows".into(),
                        ));
                    }
                }
                if !field.worley_weight.is_finite() || !(0.0..=1.0).contains(&field.worley_weight) {
                    return Err(MediaError::InvalidDensity(
                        "Perlin-Worley blend weight must lie in [0, 1]".into(),
                    ));
                }
                Ok(())
            }
            Self::Grid3D(field) => field.validate(),
        }
    }

    pub fn authored_density(&self, point: [f32; 3]) -> f32 {
        match self {
            Self::Homogeneous(field) => field.authored_density,
            Self::PerlinWorley(field) => field
                .transform
                .world_to_unit(point)
                .map(|unit| perlin_worley(field, unit))
                .unwrap_or(0.0),
            Self::Grid3D(field) => field
                .transform
                .world_to_unit(point)
                .map(|unit| trilinear(field, unit))
                .unwrap_or(0.0),
        }
    }

    pub fn physical_density(&self, point: [f32; 3]) -> f32 {
        match self {
            Self::Homogeneous(field) => field.mapping.to_physical(field.authored_density),
            Self::PerlinWorley(field) => field.mapping.to_physical(self.authored_density(point)),
            Self::Grid3D(field) => field.mapping.to_physical(self.authored_density(point)),
        }
    }

    pub(crate) fn maximum_physical_density(&self) -> Result<f32, MediaError> {
        let authored = match self {
            Self::Homogeneous(field) => field.authored_density,
            Self::PerlinWorley(_) => 1.0,
            Self::Grid3D(grid) => trilinear_outward_upper(
                grid.r16_density
                    .iter()
                    .map(|bits| f16::from_bits(*bits).to_f32())
                    .fold(0.0, f32::max),
            ),
        };
        outward_mul_nonnegative(authored, self.mapping().physical_density_per_authored_unit)
    }

    pub fn identity(&self, version: u64) -> DensityIdentity {
        let bytes = serde_json::to_vec(self).expect("density serialization cannot fail");
        DensityIdentity {
            version,
            digest: Sha256::digest(bytes).into(),
        }
    }

    pub(crate) fn mapping(&self) -> DensityMapping {
        match self {
            Self::Homogeneous(value) => value.mapping,
            Self::PerlinWorley(value) => value.mapping,
            Self::Grid3D(value) => value.mapping,
        }
    }
}

pub(super) fn medium_identity(medium: &Medium, version: u64) -> MediumIdentity {
    let bytes = serde_json::to_vec(medium).expect("validated medium serialization cannot fail");
    MediumIdentity {
        version,
        digest: Sha256::digest(bytes).into(),
    }
}

fn validate_authored(value: f32) -> Result<(), MediaError> {
    if value.is_finite() && value >= 0.0 {
        Ok(())
    } else {
        Err(MediaError::InvalidDensity(
            "authored density must be finite and non-negative".into(),
        ))
    }
}

fn trilinear(grid: &Grid3D, unit: [f32; 3]) -> f32 {
    // Matches normalized, clamp-to-edge GPU linear sampling: texel centers
    // are at (i + 0.5) / dimension.
    let coordinate: [f32; 3] =
        std::array::from_fn(|axis| unit[axis].clamp(0.0, 1.0) * grid.dimensions[axis] as f32 - 0.5);
    let lower_i: [i64; 3] = std::array::from_fn(|axis| coordinate[axis].floor() as i64);
    let lower: [u32; 3] =
        std::array::from_fn(|axis| lower_i[axis].clamp(0, grid.dimensions[axis] as i64 - 1) as u32);
    let upper: [u32; 3] = std::array::from_fn(|axis| {
        (lower_i[axis] + 1).clamp(0, grid.dimensions[axis] as i64 - 1) as u32
    });
    let fraction: [f32; 3] =
        std::array::from_fn(|axis| coordinate[axis] - coordinate[axis].floor());
    let mut value = 0.0;
    for z in 0..2 {
        for y in 0..2 {
            for x in 0..2 {
                let index = [
                    if x == 0 { lower[0] } else { upper[0] },
                    if y == 0 { lower[1] } else { upper[1] },
                    if z == 0 { lower[2] } else { upper[2] },
                ];
                let weight = if x == 0 {
                    1.0 - fraction[0]
                } else {
                    fraction[0]
                } * if y == 0 {
                    1.0 - fraction[1]
                } else {
                    fraction[1]
                } * if z == 0 {
                    1.0 - fraction[2]
                } else {
                    fraction[2]
                };
                value += grid.node(index) * weight;
            }
        }
    }
    value
}

fn flat_index(dimensions: [u32; 3], index: [u32; 3]) -> usize {
    index[0] as usize
        + dimensions[0] as usize * (index[1] as usize + dimensions[1] as usize * index[2] as usize)
}

fn perlin_worley(field: &PerlinWorley, unit: [f32; 3]) -> f32 {
    let mut frequency = field.frequency;
    let mut amplitude = 1.0;
    let mut total = 0.0;
    let mut normalization = 0.0;
    for octave in 0..field.octaves {
        let p = unit.map(|v| v * frequency);
        let perlin = perlin_noise(p, mix64(field.seed ^ octave as u64));
        let worley = 1.0 - worley_distance(p, mix64(field.seed.wrapping_add(octave as u64)));
        total += ((1.0 - field.worley_weight) * perlin + field.worley_weight * worley) * amplitude;
        normalization += amplitude;
        frequency *= 2.0;
        amplitude *= 0.5;
    }
    (total / normalization).clamp(0.0, 1.0)
}

fn perlin_noise(p: [f32; 3], seed: u64) -> f32 {
    let base = p.map(|v| v.floor() as i32);
    let t = p.map(|v| {
        let f = v - v.floor();
        f * f * f * (f * (f * 6.0 - 15.0) + 10.0)
    });
    let mut value = 0.0;
    for z in 0..2 {
        for y in 0..2 {
            for x in 0..2 {
                let h = lattice(seed, [base[0] + x, base[1] + y, base[2] + z]);
                let gradient = PERLIN_GRADIENTS[(h % PERLIN_GRADIENTS.len() as u64) as usize];
                let displacement = [
                    p[0] - (base[0] + x) as f32,
                    p[1] - (base[1] + y) as f32,
                    p[2] - (base[2] + z) as f32,
                ];
                let contribution = gradient[0] * displacement[0]
                    + gradient[1] * displacement[1]
                    + gradient[2] * displacement[2];
                let weight = if x == 0 { 1.0 - t[0] } else { t[0] }
                    * if y == 0 { 1.0 - t[1] } else { t[1] }
                    * if z == 0 { 1.0 - t[2] } else { t[2] };
                value += contribution * weight;
            }
        }
    }
    (0.5 + 0.5 * value).clamp(0.0, 1.0)
}

const PERLIN_GRADIENTS: [[f32; 3]; 12] = [
    [1.0, 1.0, 0.0],
    [-1.0, 1.0, 0.0],
    [1.0, -1.0, 0.0],
    [-1.0, -1.0, 0.0],
    [1.0, 0.0, 1.0],
    [-1.0, 0.0, 1.0],
    [1.0, 0.0, -1.0],
    [-1.0, 0.0, -1.0],
    [0.0, 1.0, 1.0],
    [0.0, -1.0, 1.0],
    [0.0, 1.0, -1.0],
    [0.0, -1.0, -1.0],
];

fn worley_distance(p: [f32; 3], seed: u64) -> f32 {
    let base = p.map(|v| v.floor() as i32);
    let mut nearest = f32::INFINITY;
    for z in -1..=1 {
        for y in -1..=1 {
            for x in -1..=1 {
                let cell = [base[0] + x, base[1] + y, base[2] + z];
                let h = lattice(seed, cell);
                let feature = [
                    cell[0] as f32 + unit_float(h),
                    cell[1] as f32 + unit_float(mix64(h)),
                    cell[2] as f32 + unit_float(mix64(mix64(h))),
                ];
                let distance = ((p[0] - feature[0]).powi(2)
                    + (p[1] - feature[1]).powi(2)
                    + (p[2] - feature[2]).powi(2))
                .sqrt();
                nearest = nearest.min(distance);
            }
        }
    }
    (nearest / 3.0f32.sqrt()).min(1.0)
}

fn lattice(seed: u64, p: [i32; 3]) -> u64 {
    mix64(
        seed ^ (p[0] as u32 as u64).wrapping_mul(0x9e3779b185ebca87)
            ^ (p[1] as u32 as u64).wrapping_mul(0xc2b2ae3d27d4eb4f)
            ^ (p[2] as u32 as u64).wrapping_mul(0x165667b19e3779f9),
    )
}

fn mix64(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58476d1ce4e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d049bb133111eb);
    value ^ (value >> 31)
}

fn unit_float(value: u64) -> f32 {
    ((value >> 40) as f32) * (1.0 / (1u32 << 24) as f32)
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MajorantProof {
    /// Field and bound are the same constant.
    ExactConstant,
    /// The evaluator's final clamp proves this bound over its continuous domain.
    ProceduralCodomain { authored_upper_bound_bits: u32 },
    /// GPU-linear interpolation is a convex combination of sampled R16 nodes;
    /// each spatial cell bounds its 3x3x3 possible neighborhood. The bound
    /// includes the derived error envelope for the evaluator's complement,
    /// multiply, and add operations, then rounds both extinction products
    /// outward.
    TrilinearConvexHull,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MajorantGrid {
    bounds: Option<Bounds3>,
    dimensions: [u32; 3],
    extinction: Vec<f32>,
    proof: MajorantProof,
    density_identity: DensityIdentity,
    medium_identity: MediumIdentity,
}

impl MajorantGrid {
    pub fn construct(medium: &Medium, version: u64) -> Result<Self, MediaError> {
        medium.validate()?;
        let sigma_t_max = medium.sigma_t().max_component();
        let density_identity = medium.density().identity(version);
        let extinction_identity = medium.identity(version);
        let (bounds, dimensions, authored, proof) = match medium.density() {
            DensityField::Homogeneous(field) => (
                None,
                [1, 1, 1],
                vec![field.authored_density],
                MajorantProof::ExactConstant,
            ),
            DensityField::PerlinWorley(field) => (
                Some(field.transform.bounds),
                [1, 1, 1],
                vec![1.0],
                MajorantProof::ProceduralCodomain {
                    authored_upper_bound_bits: 1.0f32.to_bits(),
                },
            ),
            DensityField::Grid3D(grid) => {
                let dimensions = grid.dimensions;
                let mut maxima =
                    Vec::with_capacity(dimensions.into_iter().map(|v| v as usize).product());
                for z in 0..dimensions[2] {
                    for y in 0..dimensions[1] {
                        for x in 0..dimensions[0] {
                            let mut maximum = 0.0f32;
                            for dz in -1..=1 {
                                for dy in -1..=1 {
                                    for dx in -1..=1 {
                                        let index = [
                                            (x as i64 + dx).clamp(0, grid.dimensions[0] as i64 - 1)
                                                as u32,
                                            (y as i64 + dy).clamp(0, grid.dimensions[1] as i64 - 1)
                                                as u32,
                                            (z as i64 + dz).clamp(0, grid.dimensions[2] as i64 - 1)
                                                as u32,
                                        ];
                                        maximum = maximum.max(grid.node(index));
                                    }
                                }
                            }
                            maxima.push(trilinear_outward_upper(maximum));
                        }
                    }
                }
                (
                    Some(grid.transform.bounds),
                    dimensions,
                    maxima,
                    MajorantProof::TrilinearConvexHull,
                )
            }
        };
        let mapping = medium.density().mapping();
        let extinction = authored
            .into_iter()
            .map(|v| {
                outward_mul_nonnegative(v, mapping.physical_density_per_authored_unit)
                    .and_then(|physical| outward_mul_nonnegative(physical, sigma_t_max))
            })
            .collect::<Result<Vec<_>, _>>()?;
        if extinction.iter().any(|v| !v.is_finite() || *v < 0.0) {
            return Err(MediaError::InvalidMajorant(
                "density/extinction product is non-finite".into(),
            ));
        }
        Ok(Self {
            bounds,
            dimensions,
            extinction,
            proof,
            density_identity,
            medium_identity: extinction_identity,
        })
    }

    pub fn validate_for(&self, medium: &Medium) -> Result<(), MediaError> {
        medium.validate()?;
        if self.medium_identity != medium.identity(self.medium_identity.version) {
            return Err(MediaError::InvalidMajorant(
                "majorant belongs to a different extinction model".into(),
            ));
        }
        let expected = Self::construct(medium, self.medium_identity.version)?;
        if self != &expected {
            return Err(MediaError::InvalidMajorant(
                "majorant structure or proof is not canonical".into(),
            ));
        }
        Ok(())
    }

    pub fn bounds(&self) -> Option<Bounds3> {
        self.bounds
    }

    pub fn dimensions(&self) -> [u32; 3] {
        self.dimensions
    }

    pub fn extinction_cells(&self) -> &[f32] {
        &self.extinction
    }

    pub fn proof(&self) -> &MajorantProof {
        &self.proof
    }

    pub fn density_identity(&self) -> DensityIdentity {
        self.density_identity
    }

    pub fn medium_identity(&self) -> MediumIdentity {
        self.medium_identity
    }

    pub fn query(&self, point: [f32; 3]) -> f32 {
        let Some(bounds) = self.bounds else {
            return self.extinction[0];
        };
        let Some(unit) = (SpatialTransform { bounds }).world_to_unit(point) else {
            return 0.0;
        };
        let cell: [u32; 3] = std::array::from_fn(|axis| {
            (unit[axis] * self.dimensions[axis] as f32)
                .floor()
                .min((self.dimensions[axis] - 1) as f32) as u32
        });
        self.extinction[flat_index(self.dimensions, cell)]
    }

    pub fn global(&self) -> f32 {
        self.extinction.iter().copied().fold(0.0, f32::max)
    }

    pub fn verify_points(
        &self,
        medium: &Medium,
        points: impl IntoIterator<Item = [f32; 3]>,
    ) -> Result<(), MediaError> {
        for point in points {
            let exact = medium.extinction_at(point).max_component();
            let bound = self.query(point);
            if exact > bound {
                return Err(MediaError::InvalidMajorant(format!(
                    "extinction {exact} exceeds bound {bound} at {point:?}"
                )));
            }
        }
        Ok(())
    }
}

fn next_up_nonnegative(value: f32) -> f32 {
    if value == 0.0 || !value.is_finite() {
        value
    } else {
        f32::from_bits(value.to_bits() + 1)
    }
}

fn outward_mul_nonnegative(a: f32, b: f32) -> Result<f32, MediaError> {
    let exact = a as f64 * b as f64;
    let mut rounded = a * b;
    if !exact.is_finite() || !rounded.is_finite() {
        return Err(MediaError::InvalidMajorant(
            "density/extinction product is non-finite".into(),
        ));
    }
    if (rounded as f64) < exact {
        rounded = next_up_nonnegative(rounded);
    }
    Ok(rounded)
}

fn trilinear_outward_upper(maximum: f32) -> f32 {
    if maximum == 0.0 {
        return 0.0;
    }
    // A path through the exact evaluator has at most 13 rounded operations:
    // three complements, two weight products, one value product, and seven
    // nontrivial additions. For non-negative IEEE-754 operations, gamma(13)
    // bounds their accumulated relative error. The eta term also covers
    // gradual-underflow absolute error.
    const OPERATIONS: f64 = 13.0;
    const UNIT_ROUNDOFF: f64 = 1.0 / ((1u64 << 24) as f64);
    const HALF_MIN_SUBNORMAL: f64 = f32::from_bits(1) as f64 * 0.5;
    let denominator = 1.0 - OPERATIONS * UNIT_ROUNDOFF;
    let gamma = OPERATIONS * UNIT_ROUNDOFF / denominator;
    let exact_upper =
        maximum as f64 * (1.0 + gamma) + OPERATIONS * HALF_MIN_SUBNORMAL / denominator;
    let mut rounded = exact_upper as f32;
    if (rounded as f64) < exact_upper {
        rounded = next_up_nonnegative(rounded);
    }
    rounded
}
