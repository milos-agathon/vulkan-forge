use super::*;
use std::f32::consts::PI;

fn mapping(scale: f32) -> DensityMapping {
    DensityMapping {
        physical_density_per_authored_unit: scale,
    }
}

fn homogeneous(value: f32) -> DensityField {
    DensityField::Homogeneous(Homogeneous {
        authored_density: value,
        mapping: mapping(1.0),
    })
}

fn medium(density: DensityField) -> Medium {
    Medium::new([0.1, 0.2, 0.3], [0.4, 0.5, 0.6], Phase::Isotropic, density).unwrap()
}

#[test]
fn validates_rgb_phase_and_analytic_transmittance() {
    assert!(Medium::new(
        [-0.1, 0.0, 0.0],
        [0.0; 3],
        Phase::Isotropic,
        homogeneous(1.0)
    )
    .is_err());
    assert!(Phase::henyey_greenstein(-1.0).is_err());
    assert!(Phase::henyey_greenstein(1.0).is_err());
    let medium = medium(homogeneous(2.0));
    let actual = medium.homogeneous_transmittance(3.0).unwrap();
    for (value, sigma) in actual
        .components()
        .into_iter()
        .zip(medium.sigma_t().components())
    {
        assert!((value - (-sigma * 6.0).exp()).abs() < 1.0e-6);
    }
}

#[test]
fn homogeneous_single_scatter_slab_is_an_independent_closed_form_oracle() {
    let medium = Medium::new([0.25; 3], [0.75; 3], Phase::Isotropic, homogeneous(2.0)).unwrap();
    let distance = 1.5;
    let actual = medium
        .homogeneous_single_scatter_slab(distance, 0.3, [4.0; 3])
        .unwrap();
    let expected = 4.0 * 0.75 * 2.0 * (1.0 / (4.0 * PI)) * distance * (-2.0f32 * distance).exp();
    for value in actual.components() {
        assert!((value - expected).abs() < 1.0e-7);
    }
}

#[test]
fn hg_evaluation_and_sampling_are_consistent() {
    for g in [-0.8, 0.0, 0.8] {
        let phase = Phase::henyey_greenstein(g).unwrap();
        let sample = phase.sample([0.0, 0.0, 1.0], [0.37, 0.73]).unwrap();
        assert!((sample.pdf - phase.evaluate(sample.direction[2]).unwrap()).abs() < 2.0e-6);
        assert_eq!(sample.value, sample.pdf);
    }
    let phase = Phase::henyey_greenstein(0.65).unwrap();
    let count = 16_384u64;
    let mean = (0..count)
        .map(|pixel| {
            let id = SampleIdentity {
                frame: 0,
                pixel,
                sample: 0,
                bounce: 0,
            };
            phase
                .sample([0.0, 0.0, 1.0], [id.uniform(0), id.uniform(1)])
                .unwrap()
                .direction[2] as f64
        })
        .sum::<f64>()
        / count as f64;
    assert!((mean - 0.65).abs() < 6.0 / (count as f64).sqrt());
    let phase = Phase::henyey_greenstein(5.0e-4).unwrap();
    let count = 65_536u64;
    let mean = (0..count)
        .map(|pixel| {
            let id = SampleIdentity {
                frame: 4,
                pixel,
                sample: 9,
                bounce: 1,
            };
            let sample = phase
                .sample([0.0, 0.0, 1.0], [id.uniform(0), id.uniform(1)])
                .unwrap();
            assert!((sample.pdf - phase.evaluate(sample.direction[2]).unwrap()).abs() < 2.0e-6);
            sample.direction[2] as f64
        })
        .sum::<f64>()
        / count as f64;
    assert!((mean - 5.0e-4).abs() < 6.0 / (count as f64).sqrt());

    let phase = Phase::henyey_greenstein(1.0e-20).unwrap();
    let low = phase.sample([0.0, 0.0, 1.0], [0.1, 0.25]).unwrap();
    let high = phase.sample([0.0, 0.0, 1.0], [0.9, 0.25]).unwrap();
    assert!(low.direction[2] < high.direction[2]);
    for sample in [low, high] {
        assert!((sample.pdf - phase.evaluate(sample.direction[2]).unwrap()).abs() < 2.0e-6);
    }
}

#[test]
fn hg_is_finite_at_representable_anisotropy_limits() {
    for (g, cosine) in [
        (f32::from_bits(1.0f32.to_bits() - 1), 1.0),
        (f32::from_bits((-1.0f32).to_bits() - 1), -1.0),
    ] {
        let phase = Phase::henyey_greenstein(g).unwrap();
        let value = phase.evaluate(cosine).unwrap();
        assert!(value.is_finite() && value > 0.0);
        for u in [[0.0, 0.0], [0.5, 0.5], [0.999_999_94, 0.25]] {
            let sample = phase.sample([0.0, 0.0, 1.0], u).unwrap();
            assert!(sample.value.is_finite() && sample.pdf.is_finite());
            assert!(sample.direction.iter().all(|value| value.is_finite()));
        }
    }
    assert!(Phase::Isotropic.evaluate(f32::NAN).is_err());
}

#[test]
fn grid_majorant_bounds_interpolation_and_boundaries() {
    let grid = Grid3D::new(
        SpatialTransform {
            bounds: Bounds3 {
                min: [-1.0, -2.0, -3.0],
                max: [1.0, 2.0, 3.0],
            },
        },
        [3, 2, 2],
        vec![0.0, 4.0, 1.0, 2.0, 0.5, 3.0, 1.0, 8.0, 2.0, 4.0, 1.0, 6.0],
        mapping(0.25),
    )
    .unwrap();
    let medium = medium(DensityField::Grid3D(grid));
    let majorant = MajorantGrid::construct(&medium, 7).unwrap();
    assert_eq!(majorant.proof(), &MajorantProof::TrilinearConvexHull);
    let points = (0..=12).flat_map(|z| {
        (0..=12).flat_map(move |y| {
            (0..=24).map(move |x| {
                [
                    -1.0 + 2.0 * x as f32 / 24.0,
                    -2.0 + 4.0 * y as f32 / 12.0,
                    -3.0 + 6.0 * z as f32 / 12.0,
                ]
            })
        })
    });
    majorant.verify_points(&medium, points).unwrap();
    for point in [[-1.0, -2.0, -3.0], [0.0, 0.0, 0.0], [1.0, 2.0, 3.0]] {
        assert!(majorant.query(point) >= medium.extinction_at(point).max_component());
    }
}

#[test]
fn grid_uses_exact_r16_sampling_coordinates_and_boundaries() {
    let transform = SpatialTransform {
        bounds: Bounds3 {
            min: [0.0; 3],
            max: [1.0; 3],
        },
    };
    let rounded = Grid3D::new(transform, [1, 1, 1], vec![0.5003], mapping(1.0)).unwrap();
    let quantized = rounded.dequantized_density()[0];
    assert!(quantized > 0.5003);
    assert_eq!(quantized, half::f16::from_f32(0.5003).to_f32());
    let field = DensityField::Grid3D(
        Grid3D::new(transform, [2, 1, 1], vec![0.0, 1.0], mapping(1.0)).unwrap(),
    );
    assert_eq!(field.authored_density([0.0, 0.5, 0.5]), 0.0);
    assert_eq!(field.authored_density([0.25, 0.5, 0.5]), 0.0);
    assert!((field.authored_density([0.5, 0.5, 0.5]) - 0.5).abs() < f32::EPSILON);
    assert_eq!(field.authored_density([0.75, 0.5, 0.5]), 1.0);
    assert_eq!(field.authored_density([1.0, 0.5, 0.5]), 1.0);
}

#[test]
fn grid_majorant_covers_r16_interpolation_rounding_counterexample() {
    let transform = SpatialTransform {
        bounds: Bounds3 {
            min: [0.0; 3],
            max: [1.0; 3],
        },
    };
    let authored = 2.0f32.powi(-14);
    let density = DensityField::Grid3D(
        Grid3D::new(transform, [2, 2, 2], vec![authored; 8], mapping(1.0)).unwrap(),
    );
    let point = [0.4532737136, 0.2600572407, 0.4152496457];
    assert!(density.authored_density(point) > authored);
    let medium = Medium::new([0.0; 3], [1.0; 3], Phase::Isotropic, density).unwrap();
    let majorant = MajorantGrid::construct(&medium, 17).unwrap();
    majorant
        .verify_points(
            &medium,
            [
                point,
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [0.25, 0.25, 0.25],
                [0.5, 0.5, 0.5],
                [0.75, 0.75, 0.75],
            ],
        )
        .unwrap();
}

#[test]
fn procedural_identity_and_bound_are_deterministic() {
    let density = DensityField::PerlinWorley(PerlinWorley {
        transform: SpatialTransform {
            bounds: Bounds3 {
                min: [0.0; 3],
                max: [1.0; 3],
            },
        },
        frequency: 3.25,
        octaves: 4,
        seed: 42,
        worley_weight: 0.6,
        mapping: mapping(2.0),
    });
    let medium = medium(density.clone());
    assert_eq!(density.identity(3), density.identity(3));
    assert_ne!(density.identity(3), density.identity(4));
    let majorant = MajorantGrid::construct(&medium, 3).unwrap();
    let points = (0..1000).map(|pixel| {
        let id = SampleIdentity {
            frame: 1,
            pixel,
            sample: 2,
            bounce: 0,
        };
        [id.uniform(0), id.uniform(1), id.uniform(2)]
    });
    majorant.verify_points(&medium, points).unwrap();
}

#[test]
fn majorant_construction_fails_closed_on_overflow() {
    assert!(Medium::new(
        [f32::MAX; 3],
        [f32::MAX; 3],
        Phase::Isotropic,
        homogeneous(1.0)
    )
    .is_err());
    assert!(Medium::new(
        [0.5; 3],
        [0.5; 3],
        Phase::Isotropic,
        DensityField::Homogeneous(Homogeneous {
            authored_density: 2.0,
            mapping: mapping(f32::MAX),
        }),
    )
    .is_err());
}

#[test]
fn medium_rejects_extinction_or_oracle_overflow() {
    assert!(Medium::new([f32::MAX; 3], [0.0; 3], Phase::Isotropic, homogeneous(2.0),).is_err());
    let medium = Medium::new([0.0; 3], [1.0; 3], Phase::Isotropic, homogeneous(1.0)).unwrap();
    assert!(medium
        .extinction_at([0.0; 3])
        .components()
        .iter()
        .all(|v| v.is_finite()));
    let oracle = medium
        .homogeneous_single_scatter_slab(f32::MAX, 0.0, [f32::MAX; 3])
        .unwrap();
    assert!(oracle.components().iter().all(|value| value.is_finite()));
    assert!(medium
        .homogeneous_single_scatter_slab(1.0, 0.0, [f32::NAN; 3])
        .is_err());
}

#[test]
fn tracking_and_roulette_are_deterministic() {
    let medium = Medium::new([0.0; 3], [1.0; 3], Phase::Isotropic, homogeneous(1.0)).unwrap();
    let majorant = MajorantGrid::construct(&medium, 0).unwrap();
    let context = TrackingContext::new(medium, majorant).unwrap();
    let ray = Ray {
        origin: [0.0; 3],
        direction: [1.0, 0.0, 0.0],
    };
    let id = SampleIdentity {
        frame: 9,
        pixel: 8,
        sample: 7,
        bounce: 6,
    };
    assert_eq!(
        ratio_track(&context, ray, 1.5, id).unwrap(),
        ratio_track(&context, ray, 1.5, id).unwrap()
    );
    assert_eq!(
        delta_track(&context, ray, 100.0, 1, id)
            .unwrap()
            .unwrap()
            .sample_identity,
        id
    );
    assert_eq!(
        russian_roulette(Rgb::new([0.25; 3], "throughput").unwrap(), 0.1, 0.0).unwrap(),
        Some(4.0)
    );
    assert_eq!(
        russian_roulette(Rgb::new([0.25; 3], "throughput").unwrap(), 0.5, 0.0).unwrap(),
        None
    );
}

#[test]
fn stale_zero_majorant_is_rejected_before_early_return() {
    let density = homogeneous(1.0);
    let zero = Medium::new([0.0; 3], [0.0; 3], Phase::Isotropic, density.clone()).unwrap();
    let stale = MajorantGrid::construct(&zero, 4).unwrap();
    let nonzero = Medium::new([0.0; 3], [1.0; 3], Phase::Isotropic, density).unwrap();
    let result = TrackingContext::new(nonzero, stale);
    assert!(matches!(result, Err(MediaError::InvalidMajorant(_))));
}

#[test]
fn environment_pdf_is_normalized_and_matches_samples() {
    let radiance = vec![
        Rgb::new([1.0, 0.0, 0.0], "env").unwrap(),
        Rgb::new([0.0, 2.0, 0.0], "env").unwrap(),
        Rgb::new([0.0, 0.0, 3.0], "env").unwrap(),
        Rgb::new([4.0; 3], "env").unwrap(),
        Rgb::new([0.5; 3], "env").unwrap(),
        Rgb::new([2.0; 3], "env").unwrap(),
        Rgb::new([1.0; 3], "env").unwrap(),
        Rgb::new([3.0; 3], "env").unwrap(),
    ];
    let distribution = EnvironmentDistribution::new(4, 2, radiance).unwrap();
    assert!((distribution.probability_sum() - 1.0).abs() < 1.0e-12);
    for pixel in 0..64 {
        let id = SampleIdentity {
            frame: 0,
            pixel,
            sample: 0,
            bounce: 0,
        };
        let sample = distribution
            .sample([id.uniform(0), id.uniform(1), id.uniform(2), id.uniform(3)])
            .unwrap();
        assert!((sample.pdf_solid_angle - distribution.pdf(sample.direction)).abs() < 1.0e-6);
    }
}

#[test]
fn polar_environment_uses_f64_solid_angles() {
    let distribution = EnvironmentDistribution::new(1, 16_384, vec![Rgb::ONE; 16_384]).unwrap();
    assert!((distribution.probability_sum() - 1.0).abs() < 2.0e-12);
    assert!((distribution.solid_angle_sum() - 4.0 * std::f64::consts::PI).abs() < 2.0e-12);
    for direction in [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]] {
        let pdf = distribution.pdf(direction);
        assert!(pdf.is_finite() && pdf > 0.0);
        assert!((pdf - (1.0 / (4.0 * PI))).abs() < 2.0e-6);
    }
}

#[test]
fn polar_environment_sample_pdf_preserves_unequal_first_rows() {
    let mut radiance = vec![Rgb::ZERO; 16_384];
    radiance[0] = Rgb::ONE;
    radiance[1] = Rgb::new([2.0; 3], "env").unwrap();
    let distribution = EnvironmentDistribution::new(1, 16_384, radiance).unwrap();
    let row0 = distribution.sample([0.05, 0.0, 0.5, 0.25]).unwrap();
    let row1 = distribution.sample([0.5, 0.0, 0.25, 0.25]).unwrap();
    assert_eq!(row0.texel, [0, 0]);
    assert_eq!(row1.texel, [0, 1]);
    assert_ne!(row0.pdf_solid_angle, row1.pdf_solid_angle);
    assert_eq!(row0.pdf_solid_angle, distribution.pdf(row0.direction));
    assert_eq!(row1.pdf_solid_angle, distribution.pdf(row1.direction));
}

#[test]
fn environment_boundaries_share_the_sample_pdf_measure() {
    let radiance = (0..32)
        .map(|index| Rgb::new([index as f32 + 1.0; 3], "env").unwrap())
        .collect();
    let distribution = EnvironmentDistribution::new(4, 8, radiance).unwrap();
    for u in [
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.999_999_94, 0.999_999_94],
        [0.999_999_94; 4],
    ] {
        let sample = distribution.sample(u).unwrap();
        assert_eq!(sample.pdf_solid_angle, distribution.pdf(sample.direction));
        assert_eq!(sample.radiance, distribution.radiance(sample.direction));
        assert!(sample.pdf_solid_angle.is_finite() && sample.pdf_solid_angle > 0.0);
    }
}

#[test]
fn environment_rejects_positive_pdfs_that_f32_cannot_represent() {
    let tiny = Rgb::new([f32::MIN_POSITIVE; 3], "env").unwrap();
    let large = Rgb::new([f32::MAX / 4.0; 3], "env").unwrap();
    assert!(EnvironmentDistribution::new(2, 1, vec![tiny, large]).is_err());
}

#[test]
fn power_weights_sum_and_sun_stays_delta() {
    for (a, b) in [
        (0.0, 0.0),
        (0.0, 2.0),
        (3.0, 0.0),
        (0.2, 4.0),
        (f32::MAX, f32::MAX),
        (1.0e-30, 1.0e-23),
    ] {
        let (wa, wb) = power_heuristic(a, b).unwrap();
        assert!((wa + wb - 1.0).abs() < 2.0e-6);
    }
    let sun = DirectionalSun::new([0.0, 2.0, 0.0], [10.0; 3]).unwrap();
    assert_eq!(sun.continuous_pdf([0.0, 1.0, 0.0]), None);
    assert_eq!(sun.mis_weight(), 1.0);
    assert!((Phase::Isotropic.evaluate(0.0).unwrap() - 1.0 / (4.0 * PI)).abs() < f32::EPSILON);
}

#[test]
fn serde_round_trip_retains_validated_payload() {
    let original = medium(homogeneous(1.25));
    let decoded: Medium = serde_json::from_str(&serde_json::to_string(&original).unwrap()).unwrap();
    decoded.validate().unwrap();
    assert_eq!(decoded, original);
    assert!(serde_json::from_str::<Rgb>("[-1.0,0.0,0.0]").is_err());
    let mut invalid = serde_json::to_value(&original).unwrap();
    invalid["phase"] = serde_json::json!({"HenyeyGreenstein": {"g": 1.0}});
    assert!(serde_json::from_value::<Medium>(invalid).is_err());
}

#[test]
fn medium_identity_is_complete_public_and_versioned() {
    let base = Medium::new(
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        Phase::henyey_greenstein(0.25).unwrap(),
        homogeneous(0.75),
    )
    .unwrap();
    assert_eq!(base.identity(11), base.identity(11));
    assert_ne!(base.identity(11), base.identity(12));
    for changed in [
        Medium::new(
            [0.11, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            Phase::henyey_greenstein(0.25).unwrap(),
            homogeneous(0.75),
        )
        .unwrap(),
        Medium::new(
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            Phase::henyey_greenstein(0.5).unwrap(),
            homogeneous(0.75),
        )
        .unwrap(),
        Medium::new(
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            Phase::henyey_greenstein(0.25).unwrap(),
            homogeneous(0.5),
        )
        .unwrap(),
    ] {
        assert_ne!(base.identity(11), changed.identity(11));
    }
    let majorant = MajorantGrid::construct(&base, 11).unwrap();
    assert_eq!(majorant.medium_identity(), base.identity(11));
}

struct EmptyReferenceScene;

impl ReferenceScene for EmptyReferenceScene {
    fn intersect(
        &self,
        _ray: Ray,
        _maximum_distance: f32,
    ) -> Result<Option<ReferenceSurfaceHit>, MediaError> {
        Ok(None)
    }

    fn occluded(&self, _ray: Ray, _maximum_distance: f32) -> Result<bool, MediaError> {
        Ok(false)
    }

    fn geometry_reach(&self, _ray: Ray) -> Result<f32, MediaError> {
        Ok(10.0)
    }

    fn medium_interval(
        &self,
        _ray: Ray,
        maximum_distance: f32,
    ) -> Result<Option<ReferenceMediumInterval>, MediaError> {
        Ok(Some(ReferenceMediumInterval {
            start: 0.0,
            end: maximum_distance,
        }))
    }
}

#[test]
fn integrated_reference_is_deterministic_and_unbiased_for_vacuum_environment() {
    let medium = Medium::new([0.0; 3], [0.0; 3], Phase::Isotropic, homogeneous(1.0)).unwrap();
    let context =
        TrackingContext::new(medium.clone(), MajorantGrid::construct(&medium, 0).unwrap()).unwrap();
    let environment = EnvironmentDistribution::new(2, 1, vec![Rgb::ONE; 2]).unwrap();
    let sun = DirectionalSun::new([0.0, 1.0, 0.0], [0.0; 3]).unwrap();
    let config = ReferenceTransportConfig {
        roulette_start_bounce: 3,
        roulette_minimum_probability: 0.0,
    };
    let sample_count = 4096;
    let mut mean = [0.0; 3];
    for sample in 0..sample_count {
        let identity = SampleIdentity {
            frame: 0,
            pixel: 0,
            sample,
            bounce: 0,
        };
        let actual = trace_reference_sample(
            &context,
            &EmptyReferenceScene,
            &environment,
            sun,
            Ray {
                origin: [0.0; 3],
                direction: [0.0, 0.0, 1.0],
            },
            identity,
            config,
        )
        .unwrap();
        assert_eq!(
            actual,
            trace_reference_sample(
                &context,
                &EmptyReferenceScene,
                &environment,
                sun,
                Ray {
                    origin: [0.0; 3],
                    direction: [0.0, 0.0, 1.0],
                },
                identity,
                config,
            )
            .unwrap()
        );
        for (sum, value) in mean.iter_mut().zip(actual.radiance.components()) {
            *sum += value / sample_count as f32;
        }
    }
    for value in mean {
        assert!((value - 1.0).abs() < 0.06, "channel mean {value}");
    }
}

#[test]
fn reference_channel_stratifies_every_consecutive_sample_triple() {
    let medium = Medium::new([0.0; 3], [0.0; 3], Phase::Isotropic, homogeneous(1.0)).unwrap();
    let context =
        TrackingContext::new(medium.clone(), MajorantGrid::construct(&medium, 0).unwrap()).unwrap();
    let environment = EnvironmentDistribution::new(1, 1, vec![Rgb::ONE]).unwrap();
    let sun = DirectionalSun::new([0.0, 1.0, 0.0], [0.0; 3]).unwrap();
    let config = ReferenceTransportConfig {
        roulette_start_bounce: 0,
        roulette_minimum_probability: 0.0,
    };
    let channels = (0..18)
        .map(|sample| {
            trace_reference_sample(
                &context,
                &EmptyReferenceScene,
                &environment,
                sun,
                Ray {
                    origin: [0.0; 3],
                    direction: [0.0, 0.0, 1.0],
                },
                SampleIdentity {
                    frame: 9,
                    pixel: 7,
                    sample,
                    bounce: 0,
                },
                config,
            )
            .unwrap()
            .spectral_channel
        })
        .collect::<Vec<_>>();
    for triple in channels.chunks_exact(3) {
        let mut sorted = triple.to_vec();
        sorted.sort_unstable();
        assert_eq!(sorted, [0, 1, 2]);
    }
    let first_channels = (0..64)
        .map(|pixel| {
            trace_reference_sample(
                &context,
                &EmptyReferenceScene,
                &environment,
                sun,
                Ray {
                    origin: [0.0; 3],
                    direction: [0.0, 0.0, 1.0],
                },
                SampleIdentity {
                    frame: 9,
                    pixel,
                    sample: 0,
                    bounce: 0,
                },
                config,
            )
            .unwrap()
            .spectral_channel
        })
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(first_channels, [0, 1, 2].into_iter().collect());
}

#[test]
fn deterministic_uniform_never_rounds_to_one() {
    const SPECTRAL_CHANNEL_DIMENSION: u64 = 0x4e45_5048_454c_4500;
    let reproducer = SampleIdentity {
        frame: 0,
        pixel: 0,
        sample: 12_069_728_697_810_569_649,
        bounce: 0,
    };
    let value = reproducer.uniform(SPECTRAL_CHANNEL_DIMENSION);
    assert!((0.0..1.0).contains(&value), "uniform returned {value}");
    assert!((value * 3.0).floor() < 3.0);

    for sample in 0..65_536 {
        let value = SampleIdentity {
            frame: 17,
            pixel: sample,
            sample,
            bounce: 23,
        }
        .uniform(SPECTRAL_CHANNEL_DIMENSION);
        assert!((0.0..1.0).contains(&value));
    }
}

struct SurfaceBehindBoundedMedium;

impl ReferenceScene for SurfaceBehindBoundedMedium {
    fn intersect(
        &self,
        ray: Ray,
        maximum_distance: f32,
    ) -> Result<Option<ReferenceSurfaceHit>, MediaError> {
        Ok((maximum_distance >= 2.0).then_some(ReferenceSurfaceHit {
            distance: 2.0,
            position: [
                ray.origin[0] + 2.0 * ray.direction[0],
                ray.origin[1] + 2.0 * ray.direction[1],
                ray.origin[2] + 2.0 * ray.direction[2],
            ],
            normal: [0.0, 0.0, -1.0],
            albedo: Rgb::ONE,
        }))
    }

    fn occluded(&self, _ray: Ray, _maximum_distance: f32) -> Result<bool, MediaError> {
        Ok(false)
    }

    fn geometry_reach(&self, _ray: Ray) -> Result<f32, MediaError> {
        Ok(10.0)
    }

    fn medium_interval(
        &self,
        _ray: Ray,
        maximum_distance: f32,
    ) -> Result<Option<ReferenceMediumInterval>, MediaError> {
        Ok(Some(ReferenceMediumInterval {
            start: 0.0,
            end: maximum_distance.min(1.0),
        }))
    }
}

#[test]
fn non_vacuum_transport_tracks_before_independent_surface_reach() {
    let medium = Medium::new([1.0e6; 3], [0.0; 3], Phase::Isotropic, homogeneous(1.0)).unwrap();
    let context =
        TrackingContext::new(medium.clone(), MajorantGrid::construct(&medium, 0).unwrap()).unwrap();
    let sample = trace_reference_sample(
        &context,
        &SurfaceBehindBoundedMedium,
        &EnvironmentDistribution::new(1, 1, vec![Rgb::ONE]).unwrap(),
        DirectionalSun::new([0.0, 1.0, 0.0], [0.0; 3]).unwrap(),
        Ray {
            origin: [0.0; 3],
            direction: [0.0, 0.0, 1.0],
        },
        SampleIdentity {
            frame: 0,
            pixel: 0,
            sample: 0,
            bounce: 0,
        },
        ReferenceTransportConfig {
            roulette_start_bounce: 0,
            roulette_minimum_probability: 0.0,
        },
    )
    .unwrap();
    assert_eq!(sample.collision_count, 1);
    assert_eq!(sample.surface_count, 0);
    assert_eq!(sample.radiance, Rgb::ZERO);
}
