//! Optional `geo` parity and performance oracle for non-degenerate inputs.

use std::time::Instant;

use geo::{algorithm::Validation, BooleanOps};

use super::verification::{area, pair, perimeter, Rng};
use super::{overlay, BooleanOp, MultiPolygon};

pub(super) struct OracleReport {
    pub cases: u64,
    pub disagreements: u64,
    pub exact_ms: f64,
    pub geo_ms: f64,
    pub ratio: f64,
}

fn measure<T>(f: impl FnOnce() -> T) -> (T, f64) {
    let start = Instant::now();
    let value = f();
    (value, start.elapsed().as_secs_f64() * 1_000.0)
}

fn median3(mut samples: [f64; 3]) -> f64 {
    samples.sort_by(f64::total_cmp);
    samples[1]
}

fn to_geo(value: &MultiPolygon) -> geo::MultiPolygon<f64> {
    geo::MultiPolygon(
        value
            .0
            .iter()
            .map(|polygon| {
                let line = |ring: &super::Ring| {
                    geo::LineString::from(
                        ring.iter()
                            .map(|point| (point.x, point.y))
                            .collect::<Vec<_>>(),
                    )
                };
                geo::Polygon::new(
                    line(&polygon.exterior),
                    polygon.holes.iter().map(line).collect(),
                )
            })
            .collect(),
    )
}

fn from_geo(value: &geo::MultiPolygon<f64>) -> MultiPolygon {
    MultiPolygon(
        value
            .0
            .iter()
            .map(|polygon| {
                let ring = |line: &geo::LineString<f64>| {
                    line.0
                        .iter()
                        .map(|coord| super::Point::new(coord.x, coord.y))
                        .collect::<Vec<_>>()
                };
                super::Polygon {
                    exterior: ring(polygon.exterior()),
                    holes: polygon.interiors().iter().map(ring).collect(),
                }
            })
            .collect(),
    )
}

pub(super) fn run(cases: u64, seed: u64) -> OracleReport {
    let mut rng = Rng(seed.max(1));
    let corpus = (0..cases)
        .map(|index| pair(index * 6, &mut rng))
        .collect::<Vec<_>>();
    let exact_run = || {
        corpus
            .iter()
            .map(|(left, right)| {
                (
                    overlay(left, right, BooleanOp::Union).unwrap(),
                    overlay(left, right, BooleanOp::Intersection).unwrap(),
                )
            })
            .collect::<Vec<_>>()
    };
    let geo_run = || {
        // Time the former runtime seam, not just `geo`'s internal sweep: the
        // old path converted both inputs into geo-types and converted both
        // results back before returning them to forge3d callers.
        corpus
            .iter()
            .map(|(left, right)| {
                let left = to_geo(left);
                let right = to_geo(right);
                assert!(left.is_valid() && right.is_valid());
                let union = left.union(&right);
                assert!(union.is_valid());
                assert!(left.is_valid() && right.is_valid());
                let intersection = left.intersection(&right);
                assert!(intersection.is_valid());
                (from_geo(&union), from_geo(&intersection))
            })
            .collect::<Vec<_>>()
    };

    // A one-shot wall-clock ratio is too sensitive to scheduler jitter for a
    // hard cross-platform gate. Keep the 3x contract, but use an interleaved
    // median of three complete runs so one noisy sample cannot decide it.
    let (exact, exact_0) = measure(&exact_run);
    let (geo, geo_0) = measure(&geo_run);
    let (_, geo_1) = measure(&geo_run);
    let (_, exact_1) = measure(&exact_run);
    let (_, exact_2) = measure(&exact_run);
    let (_, geo_2) = measure(&geo_run);
    let exact_ms = median3([exact_0, exact_1, exact_2]);
    let geo_ms = median3([geo_0, geo_1, geo_2]);

    let mut disagreements = 0;
    for ((left, right), ((union, intersection), geo_result)) in
        corpus.iter().zip(exact.iter().zip(geo.iter()))
    {
        let delta = union.snap_motion_bound.max(intersection.snap_motion_bound);
        let common = perimeter(left) + perimeter(right);
        let union_bound =
            delta * (common + perimeter(&union.geometry)) + std::f64::consts::PI * delta * delta;
        let intersection_bound = delta * (common + perimeter(&intersection.geometry))
            + std::f64::consts::PI * delta * delta;
        let union_error = (area(&union.geometry) - area(&geo_result.0)).abs();
        let intersection_error = (area(&intersection.geometry) - area(&geo_result.1)).abs();
        if union_error > union_bound || intersection_error > intersection_bound {
            disagreements += 1;
        }
    }
    OracleReport {
        cases,
        disagreements,
        exact_ms,
        geo_ms,
        ratio: exact_ms / geo_ms.max(f64::MIN_POSITIVE),
    }
}
