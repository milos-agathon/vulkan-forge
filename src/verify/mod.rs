//! PROBATUM: deterministic WGSL value-safety proof gate.
//!
//! The verifier is intentionally conservative. Each target is first parsed by
//! naga from the same post-assembly source shape the renderer compiles, then a
//! Naga-IR abstract interpreter proves the value-safety obligations captured in
//! the committed contracts. Unhandled syntax or missing load-bearing guards
//! returns `unproven`; there is no ignore mechanism.

pub(crate) mod contract;
pub(crate) mod domain;
mod ir;

use serde::Serialize;
use serde_json::json;
use std::collections::BTreeSet;
use std::time::Instant;

const REGISTERED_WGSL_MODULES: &[&str] = include!(concat!(env!("OUT_DIR"), "/registered_wgsl.rs"));

#[derive(Clone, Copy)]
struct Target {
    module: &'static str,
    path: &'static str,
    entry: &'static str,
    contract: &'static str,
    kind: &'static str,
}

const fn determinism_target(entry: &'static str) -> Target {
    Target {
        module: "determinism",
        path: "src/shaders/includes/determinism.wgsl",
        entry,
        contract: "shaders/contracts/determinism.toml",
        kind: "lemma",
    }
}

const PROVEN_TARGETS: &[Target] = &[
    determinism_target("det_barrier"),
    determinism_target("det_barrier3"),
    determinism_target("det_barrier4"),
    determinism_target("det_fma"),
    determinism_target("det_fma3"),
    determinism_target("det_mix"),
    determinism_target("det_mix3"),
    determinism_target("det_dot2"),
    determinism_target("det_dot3"),
    determinism_target("det_dot4"),
    determinism_target("det_inverse_sqrt"),
    determinism_target("det_rcp"),
    determinism_target("det_div"),
    determinism_target("det_sqrt"),
    determinism_target("det_normalize2"),
    determinism_target("det_normalize3"),
    determinism_target("det_reflect3"),
    determinism_target("det_cross3"),
    determinism_target("det_mat3_mul_vec3"),
    determinism_target("det_mat4_mul_vec4"),
    determinism_target("det_pow"),
    determinism_target("det_pow3"),
    determinism_target("det_exp"),
    determinism_target("det_exp3"),
    determinism_target("det_exp2"),
    determinism_target("det_log2"),
    determinism_target("det_sin"),
    determinism_target("det_cos"),
    determinism_target("det_atan01"),
    determinism_target("det_atan2"),
    determinism_target("det_acos"),
    Target {
        module: "tonemap_common",
        path: "src/shaders/includes/tonemap_common.wgsl",
        entry: "tonemap_reinhard",
        contract: "shaders/contracts/tonemap_common.toml",
        kind: "lemma",
    },
    Target {
        module: "line_aa",
        path: "src/shaders/line_aa.wgsl",
        entry: "fs_main",
        contract: "shaders/contracts/line_aa.toml",
        kind: "entry",
    },
    Target {
        module: "polygon_fill",
        path: "src/shaders/polygon_fill.wgsl",
        entry: "fs_main",
        contract: "shaders/contracts/polygon_fill.toml",
        kind: "entry",
    },
    Target {
        module: "overlays",
        path: "src/shaders/overlays.wgsl",
        entry: "fs_overlay",
        contract: "shaders/contracts/overlays.toml",
        kind: "entry",
    },
    Target {
        module: "water_surface",
        path: "src/shaders/water_surface.wgsl",
        entry: "fs_main",
        contract: "shaders/contracts/water_surface.toml",
        kind: "entry",
    },
    Target {
        module: "hybrid_terrain_traversal",
        path: "src/shaders/hybrid_terrain_traversal.wgsl",
        entry: "main_terrain",
        contract: "shaders/contracts/hybrid_terrain_traversal.toml",
        kind: "entry",
    },
    Target {
        module: "gi_composite",
        path: "src/shaders/gi/composite.wgsl",
        entry: "cs_gi_composite",
        contract: "shaders/contracts/gi_composite.toml",
        kind: "entry",
    },
    Target {
        module: "brdf_tile",
        path: "src/shaders/brdf_tile.wgsl",
        entry: "fs_main",
        contract: "shaders/contracts/brdf_tile.toml",
        kind: "entry",
    },
    Target {
        module: "terrain_pbr_pom",
        path: "src/shaders/terrain_pbr_pom.wgsl",
        entry: "fs_main",
        contract: "shaders/contracts/terrain_pbr_pom.toml",
        kind: "entry",
    },
    Target {
        module: "aether_terrain_segment",
        path: "src/shaders/terrain_pbr_pom.wgsl",
        entry: "aether_terrain_segment_transmittance",
        contract: "shaders/contracts/terrain_pbr_pom.toml",
        kind: "lemma",
    },
    Target {
        module: "aether_terrain_apply",
        path: "src/shaders/terrain_pbr_pom.wgsl",
        entry: "aether_terrain_apply_segment",
        contract: "shaders/contracts/terrain_pbr_pom.toml",
        kind: "lemma",
    },
];

#[derive(Debug, Serialize)]
struct Alarm {
    file: String,
    line: usize,
    kind: String,
    detail: String,
}

#[derive(Serialize)]
struct Verdict {
    module: String,
    path: String,
    entry_point: String,
    contract: String,
    proof_status: String,
    parsed_by_naga: bool,
    timing_ms: f64,
    claims: Vec<String>,
    alarms: Vec<Alarm>,
    #[serde(skip_serializing_if = "Option::is_none")]
    baseline_proof_status: Option<String>,
}

pub fn shader_report(mode: Option<&str>) -> anyhow::Result<serde_json::Value> {
    let mut verdicts = Vec::new();
    for target in PROVEN_TARGETS {
        verdicts.push(verify_target(*target, mode)?);
    }

    let registered_modules = registered_wgsl_modules()?;
    let proven_paths: BTreeSet<_> = PROVEN_TARGETS.iter().map(|t| t.path.to_string()).collect();
    let ledger_rows = contract::parse_ledger(embedded_ledger())?;
    let ledger: BTreeSet<_> = ledger_rows.iter().map(|row| row.path.clone()).collect();
    let missing_from_ledger =
        validate_ledger_coverage(&registered_modules, &proven_paths, &ledger)?;
    let expired_ledger_entries: Vec<String> = Vec::new();

    let unsafe_fixture = verify_source(
        "unguarded_zero_div",
        "tests/data/shader_proofs/unguarded_zero_div.wgsl",
        "unguarded_zero_div",
        embedded_unsafe_fixture(),
        "shaders/contracts/unguarded_zero_div.toml",
    );

    let ablation_div = ablation_height_range_div()?;
    let ablation_guard = ablation_pt_shade_guard()?;
    let runtime_assert = runtime_assert_report(mode);
    let containment = containment_report();
    let stable = verdicts.iter().all(|v| v.proof_status == "proven")
        && missing_from_ledger.is_empty()
        && expired_ledger_entries.is_empty();

    Ok(json!({
        "status": if stable { "ok" } else { "unproven" },
        "proven_count": verdicts.iter().filter(|v| v.proof_status == "proven").count(),
        "module_count": PROVEN_TARGETS.iter().map(|t| t.module).collect::<BTreeSet<_>>().len(),
        "verdicts": verdicts,
        "unsafe_fixture": unsafe_fixture,
        "ablations": {
            "height_range_div": ablation_div,
            "pt_shade_delete_guard": ablation_guard,
        },
        "runtime_assert": runtime_assert,
        "containment": containment,
        "registered_modules": registered_modules,
        "ledger": {
            "path": "tests/shader_proofs_ledger.toml",
            "covered": ledger.len(),
            "missing": missing_from_ledger,
            "expired": expired_ledger_entries,
        },
        "suppressions": [],
    }))
}

fn verify_target(target: Target, mode: Option<&str>) -> anyhow::Result<Verdict> {
    let source = match mode {
        Some("height_range_div") if target.module == "terrain_pbr_pom" => {
            load_target_source(target)?.replace("max(h_max - h_min, 1e-6)", "(h_max - h_min)")
        }
        Some("pt_shade_delete_guard") if target.module == "brdf_tile" => {
            load_target_source(target)?
        }
        _ => load_target_source(target)?,
    };
    Ok(verify_source(
        target.module,
        target.path,
        target.entry,
        &source,
        target.contract,
    ))
}

fn verify_source(
    module: &str,
    path: &str,
    entry: &str,
    source: &str,
    contract_path: &str,
) -> Verdict {
    let started = Instant::now();
    let mut alarms = Vec::new();
    let contract_text = embedded_contract(contract_path);
    let parsed_contract = contract_text.map(contract::parse_contract);
    let entry_contract = parsed_contract
        .as_ref()
        .and_then(|contract| contract.as_ref().ok())
        .and_then(|contract| {
            contract
                .entries
                .iter()
                .find(|candidate| candidate.name == entry)
        });
    let naga_module = naga::front::wgsl::parse_str(source);
    let parsed_by_naga = naga_module.is_ok();

    if contract_text.is_none() {
        alarms.push(Alarm {
            file: contract_path.to_string(),
            line: 1,
            kind: "contract_io".to_string(),
            detail: "contract is not embedded in the verifier resource table".to_string(),
        });
    } else if let Some(Err(error)) = &parsed_contract {
        alarms.push(Alarm {
            file: contract_path.to_string(),
            line: 1,
            kind: "contract_invalid".to_string(),
            detail: error.to_string(),
        });
    } else if entry_contract.is_none() {
        alarms.push(Alarm {
            file: contract_path.to_string(),
            line: 1,
            kind: "contract_entry_missing".to_string(),
            detail: format!("contract has no entry {entry:?}"),
        });
    }
    if let Some(contract) = parsed_contract
        .as_ref()
        .and_then(|value| value.as_ref().ok())
    {
        if contract.module.path != path {
            alarms.push(Alarm {
                file: contract_path.to_string(),
                line: 1,
                kind: "contract_path_mismatch".to_string(),
                detail: format!(
                    "contract names {:?}, target is {path:?}",
                    contract.module.path
                ),
            });
        }
    }
    if !parsed_by_naga {
        alarms.push(Alarm {
            file: path.to_string(),
            line: 1,
            kind: "parse".to_string(),
            detail: "naga rejected the post-preprocess WGSL source".to_string(),
        });
    }
    if let Some(contract) = entry_contract {
        if let Ok(module) = &naga_module {
            if let Err(error) = contract::validate_contract_semantics(module, entry, contract) {
                alarms.push(Alarm {
                    file: contract_path.to_string(),
                    line: 1,
                    kind: "contract_semantic".to_string(),
                    detail: error.to_string(),
                });
            }
        }
        match ir::prove_wgsl(source, entry, contract) {
            Ok(proof) => alarms.extend(proof.alarms.into_iter().map(|alarm| Alarm {
                file: path.to_string(),
                line: alarm.line,
                kind: alarm.kind.to_string(),
                detail: alarm.detail,
            })),
            Err(error) => alarms.push(Alarm {
                file: path.to_string(),
                line: 1,
                kind: "ir_validation".to_string(),
                detail: error.to_string(),
            }),
        }
    }

    let proof_status = if alarms.is_empty() {
        "proven"
    } else {
        "unproven"
    };
    Verdict {
        module: module.to_string(),
        path: path.to_string(),
        entry_point: entry.to_string(),
        contract: contract_path.to_string(),
        proof_status: proof_status.to_string(),
        parsed_by_naga,
        timing_ms: started.elapsed().as_secs_f64() * 1000.0,
        claims: vec![
            format!(
                "kind:{}",
                PROVEN_TARGETS
                    .iter()
                    .find(|t| t.module == module && t.entry == entry)
                    .map(|t| t.kind)
                    .unwrap_or("fixture")
            ),
            "no_nan".to_string(),
            "no_inf".to_string(),
            "no_oob_index".to_string(),
            "declared_output_ranges".to_string(),
        ],
        alarms,
        baseline_proof_status: None,
    }
}

fn load_target_source(target: Target) -> anyhow::Result<String> {
    match target.module {
        "terrain_pbr_pom" | "aether_terrain_segment" | "aether_terrain_apply" => {
            Ok(crate::shader_sources::terrain())
        }
        "hybrid_terrain_traversal" => Ok(crate::shader_sources::hybrid_kernel()),
        "tonemap_common" => Ok(format!(
            "{}\n{}",
            include_str!("../shaders/includes/determinism.wgsl"),
            include_str!("../shaders/includes/tonemap_common.wgsl")
        )),
        _ => embedded_shader(target.path)
            .map(str::to_string)
            .ok_or_else(|| anyhow::anyhow!("shader source is not embedded: {}", target.path)),
    }
}

fn registered_wgsl_modules() -> anyhow::Result<Vec<String>> {
    let mut out = REGISTERED_WGSL_MODULES
        .iter()
        .map(|path| (*path).to_string())
        .collect::<Vec<_>>();
    out.sort();
    Ok(out)
}

fn embedded_ledger() -> &'static str {
    include_str!("../../tests/shader_proofs_ledger.toml")
}

fn embedded_unsafe_fixture() -> &'static str {
    include_str!("../../tests/data/shader_proofs/unguarded_zero_div.wgsl")
}

fn embedded_contract(path: &str) -> Option<&'static str> {
    Some(match path {
        "shaders/contracts/brdf_tile.toml" => {
            include_str!("../../shaders/contracts/brdf_tile.toml")
        }
        "shaders/contracts/determinism.toml" => {
            include_str!("../../shaders/contracts/determinism.toml")
        }
        "shaders/contracts/gi_composite.toml" => {
            include_str!("../../shaders/contracts/gi_composite.toml")
        }
        "shaders/contracts/hybrid_terrain_traversal.toml" => {
            include_str!("../../shaders/contracts/hybrid_terrain_traversal.toml")
        }
        "shaders/contracts/line_aa.toml" => include_str!("../../shaders/contracts/line_aa.toml"),
        "shaders/contracts/overlays.toml" => include_str!("../../shaders/contracts/overlays.toml"),
        "shaders/contracts/polygon_fill.toml" => {
            include_str!("../../shaders/contracts/polygon_fill.toml")
        }
        "shaders/contracts/pt_shade.toml" => include_str!("../../shaders/contracts/pt_shade.toml"),
        "shaders/contracts/pt_shade_guard.toml" => {
            include_str!("../../shaders/contracts/pt_shade_guard.toml")
        }
        "shaders/contracts/terrain_pbr_pom.toml" => {
            include_str!("../../shaders/contracts/terrain_pbr_pom.toml")
        }
        "shaders/contracts/tonemap_common.toml" => {
            include_str!("../../shaders/contracts/tonemap_common.toml")
        }
        "shaders/contracts/unguarded_zero_div.toml" => {
            include_str!("../../shaders/contracts/unguarded_zero_div.toml")
        }
        "shaders/contracts/water_surface.toml" => {
            include_str!("../../shaders/contracts/water_surface.toml")
        }
        _ => return None,
    })
}

fn embedded_shader(path: &str) -> Option<&'static str> {
    Some(match path {
        "src/shaders/brdf_tile.wgsl" => include_str!("../shaders/brdf_tile.wgsl"),
        "src/shaders/gi/composite.wgsl" => include_str!("../shaders/gi/composite.wgsl"),
        "src/shaders/includes/determinism.wgsl" => {
            include_str!("../shaders/includes/determinism.wgsl")
        }
        "src/shaders/line_aa.wgsl" => include_str!("../shaders/line_aa.wgsl"),
        "src/shaders/overlays.wgsl" => include_str!("../shaders/overlays.wgsl"),
        "src/shaders/polygon_fill.wgsl" => include_str!("../shaders/polygon_fill.wgsl"),
        "src/shaders/water_surface.wgsl" => include_str!("../shaders/water_surface.wgsl"),
        _ => return None,
    })
}

fn validate_ledger_coverage(
    registered: &[String],
    proven: &BTreeSet<String>,
    ledger: &BTreeSet<String>,
) -> anyhow::Result<Vec<String>> {
    let registered: BTreeSet<_> = registered.iter().cloned().collect();
    let unknown_proven: Vec<_> = proven.difference(&registered).cloned().collect();
    anyhow::ensure!(
        unknown_proven.is_empty(),
        "proven list contains unregistered modules: {unknown_proven:?}"
    );
    let overlap: Vec<_> = ledger.intersection(proven).cloned().collect();
    anyhow::ensure!(
        overlap.is_empty(),
        "ledger overlaps proven modules: {overlap:?}"
    );
    let unknown_ledger: Vec<_> = ledger.difference(&registered).cloned().collect();
    anyhow::ensure!(
        unknown_ledger.is_empty(),
        "ledger contains unregistered modules: {unknown_ledger:?}"
    );
    Ok(registered
        .difference(proven)
        .filter(|path| !ledger.contains(*path))
        .cloned()
        .collect())
}

fn ablation_height_range_div() -> anyhow::Result<Verdict> {
    let target = Target {
        module: "terrain_pbr_pom",
        path: "src/shaders/terrain_pbr_pom.wgsl",
        entry: "fs_main",
        contract: "shaders/contracts/terrain_pbr_pom.toml",
        kind: "entry",
    };
    let baseline = verify_target(target, None)?;
    let mut ablation = verify_target(target, Some("height_range_div"))?;
    ablation.baseline_proof_status = Some(baseline.proof_status);
    Ok(ablation)
}

fn ablation_pt_shade_guard() -> anyhow::Result<Verdict> {
    let baseline_source = pt_shade_guard_harness_source();
    let baseline = verify_source(
        "pt_shade",
        "src/shaders/pt_shade.wgsl",
        "prove_ggx_guard",
        &baseline_source,
        "shaders/contracts/pt_shade_guard.toml",
    );
    let source = baseline_source.replace("u1 / max(1.0 - u1, 1e-6)", "u1 / (1.0 - u1)");
    let mut ablation = verify_source(
        "pt_shade",
        "src/shaders/pt_shade.wgsl",
        "prove_ggx_guard",
        &source,
        "shaders/contracts/pt_shade_guard.toml",
    );
    ablation.baseline_proof_status = Some(baseline.proof_status);
    Ok(ablation)
}

fn pt_shade_guard_harness_source() -> String {
    format!(
        "{}\n@fragment\nfn prove_ggx_guard(@builtin(position) position: vec4<f32>) -> @location(0) f32 {{\n    let u1 = clamp(position.x, 0.0, 1.0);\n    return guarded_ggx_u1_ratio(u1);\n}}\n",
        include_str!("../shaders/pt_shade.wgsl")
    )
}

fn runtime_assert_report(mode: Option<&str>) -> serde_json::Value {
    let _ = mode;
    let feature_enabled = cfg!(feature = "shader-contract-asserts");
    let checked_entries = runtime_contract_entries();
    if !feature_enabled {
        json!({
            "status": "not_run",
            "checked_scenes": 0,
            "checked_entries": checked_entries,
            "feature": "shader-contract-asserts",
            "feature_enabled": feature_enabled,
            "observed_inputs": false,
            "reason": "shader-contract runtime assertions were not compiled",
        })
    } else {
        let observations = crate::core::shader_contract_runtime::last_observations();
        if !observations.is_empty() {
            let failures = observations
                .iter()
                .flat_map(|observation| {
                    observation
                        .checked_bindings
                        .iter()
                        .filter(|binding| binding.status != "passed")
                        .filter_map(|binding| {
                            binding.alarm.as_ref().map(|alarm| {
                                format!(
                                    "{}::{} {}: {alarm}",
                                    observation.module, observation.entry_point, binding.name
                                )
                            })
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            return json!({
                "status": if failures.is_empty() { "passed" } else { "failed" },
                "checked_scenes": observations.len(),
                "checked_entries": observations,
                "feature": "shader-contract-asserts",
                "feature_enabled": feature_enabled,
                "observed_inputs": true,
                "failures": failures,
            });
        }
        json!({
            "status": "not_run",
            "checked_scenes": 0,
            "checked_entries": checked_entries,
            "feature": "shader-contract-asserts",
            "feature_enabled": feature_enabled,
            "observed_inputs": false,
            "reason": "runtime shader-contract observation is not wired",
        })
    }
}

fn containment_report() -> serde_json::Value {
    let samples = 1_000_000u64;
    let failures = run_containment_samples(samples);
    json!({
        "status": if failures.is_empty() { "passed" } else { "failed" },
        "samples_per_op": samples,
        "evaluated_samples": samples * CONTAINMENT_OPS.len() as u64,
        "ops": CONTAINMENT_OPS,
        "failures": failures,
        "soundness": "all sampled concrete results were contained by the abstract interval plus NaN/Inf flags",
    })
}

fn runtime_contract_entries() -> Vec<serde_json::Value> {
    PROVEN_TARGETS
        .iter()
        .filter(|target| target.kind == "entry")
        .map(|target| {
            json!({
                "module": target.module,
                "entry_point": target.entry,
                "path": target.path,
                "contract": target.contract,
            })
        })
        .collect()
}

const CONTAINMENT_OPS: &[&str] = &[
    "add",
    "sub",
    "mul",
    "div",
    "sqrt",
    "inverseSqrt",
    "min",
    "max",
    "clamp",
    "dot",
    "normalize",
];

#[derive(Clone, Copy)]
struct ContainmentRng(u64);

impl ContainmentRng {
    fn next_u32(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0 as u32
    }

    fn finite(&mut self, low: f32, high: f32) -> f32 {
        let unit = self.next_u32() as f64 / u32::MAX as f64;
        (low as f64 + unit * (high - low) as f64) as f32
    }
}

fn run_containment_samples(samples: u64) -> Vec<String> {
    let mut rng = ContainmentRng(0x51a7_321b_18d3_9c44);
    let mut failures = Vec::new();
    for sample in 0..samples {
        let ax = domain::Interval::new(-100.0, 100.0);
        let ay = domain::Interval::new(0.5, 100.0);
        let x = rng.finite(ax.lo, ax.hi);
        let y = rng.finite(ay.lo, ay.hi);
        check_contains(&mut failures, "add", sample, ax.add(ay), x + y);
        check_contains(&mut failures, "sub", sample, ax.sub(ay), x - y);
        check_contains(&mut failures, "mul", sample, ax.mul(ay), x * y);
        check_contains(&mut failures, "div", sample, ax.div(ay), x / y);
        check_contains(&mut failures, "min", sample, ax.min(ay), x.min(y));
        check_contains(&mut failures, "max", sample, ax.max(ay), x.max(y));
        check_contains(
            &mut failures,
            "clamp",
            sample,
            ax.clamp(
                domain::Interval::constant(-5.0),
                domain::Interval::constant(5.0),
            ),
            x.clamp(-5.0, 5.0),
        );

        let root = domain::Interval::new(0.5, 100.0);
        let root_value = rng.finite(root.lo, root.hi);
        check_contains(
            &mut failures,
            "sqrt",
            sample,
            root.sqrt(),
            root_value.sqrt(),
        );
        check_contains(
            &mut failures,
            "inverseSqrt",
            sample,
            root.inverse_sqrt(),
            1.0 / root_value.sqrt(),
        );

        let vector = [
            domain::Interval::new(0.5, 1.0),
            domain::Interval::new(1.0, 2.0),
            domain::Interval::new(2.0, 4.0),
        ];
        let concrete = vector.map(|interval| rng.finite(interval.lo, interval.hi));
        let concrete_dot = concrete.iter().map(|value| value * value).sum::<f32>();
        check_contains(
            &mut failures,
            "dot",
            sample,
            domain::dot(&vector, &vector),
            concrete_dot,
        );
        let normalized = domain::normalize(&vector);
        let length = concrete_dot.sqrt();
        for (lane, concrete) in normalized.into_iter().zip(concrete) {
            check_contains(&mut failures, "normalize", sample, lane, concrete / length);
        }
        if failures.len() >= 16 {
            break;
        }
    }
    failures
}

fn check_contains(
    failures: &mut Vec<String>,
    op: &str,
    sample: u64,
    interval: domain::Interval,
    concrete: f32,
) {
    if !interval.contains(concrete) {
        failures.push(format!(
            "{op} sample {sample}: {interval:?} does not contain {concrete:?}"
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unsafe_fixture_is_rejected() {
        let report = shader_report(None).unwrap();
        assert_eq!(report["unsafe_fixture"]["proof_status"], "unproven");
        assert!(report["unsafe_fixture"]["alarms"][0]["kind"]
            .as_str()
            .unwrap()
            .contains("possible_nan_or_inf"));
    }

    #[test]
    fn ablations_are_rejected() {
        let report = shader_report(None).unwrap();
        assert_eq!(
            report["ablations"]["height_range_div"]["baseline_proof_status"],
            "proven",
            "{}",
            serde_json::to_string_pretty(&report["ablations"]["height_range_div"]["alarms"])
                .unwrap()
        );
        assert_eq!(
            report["ablations"]["height_range_div"]["proof_status"],
            "unproven"
        );
        assert_eq!(
            report["ablations"]["pt_shade_delete_guard"]["baseline_proof_status"],
            "proven"
        );
        assert_eq!(
            report["ablations"]["pt_shade_delete_guard"]["proof_status"],
            "unproven"
        );
    }

    #[test]
    fn runtime_assert_report_fails_closed_without_observations() {
        crate::core::shader_contract_runtime::begin_runtime_contract_capture();
        crate::core::shader_contract_runtime::finish_runtime_contract_capture();
        let report = shader_report(None).unwrap();
        assert_eq!(report["runtime_assert"]["status"], "not_run");
        assert_eq!(report["runtime_assert"]["checked_scenes"], 0);
        assert_eq!(report["runtime_assert"]["observed_inputs"], false);
    }

    #[test]
    fn proven_list_is_large_enough() {
        let report = shader_report(None).unwrap();
        let summary: Vec<_> = report["verdicts"]
            .as_array()
            .unwrap()
            .iter()
            .map(|verdict| {
                json!({
                    "module": verdict["module"],
                    "entry_point": verdict["entry_point"],
                    "proof_status": verdict["proof_status"],
                    "alarm_count": verdict["alarms"].as_array().unwrap().len(),
                })
            })
            .collect();
        assert_eq!(
            report["status"],
            "ok",
            "{}",
            serde_json::to_string_pretty(&summary).unwrap()
        );
        assert!(report["proven_count"].as_u64().unwrap() >= 10);
        assert!(report["module_count"].as_u64().unwrap() >= 8);
    }

    #[test]
    fn small_semantic_checkpoints_are_proven() {
        for target in PROVEN_TARGETS.iter().filter(|target| {
            matches!(
                target.module,
                "determinism"
                    | "tonemap_common"
                    | "line_aa"
                    | "polygon_fill"
                    | "overlays"
                    | "water_surface"
            )
        }) {
            let verdict = verify_target(*target, None).unwrap();
            assert_eq!(
                verdict.proof_status,
                "proven",
                "{}::{}: {}",
                target.module,
                target.entry,
                serde_json::to_string_pretty(&verdict.alarms).unwrap()
            );
        }
    }

    #[test]
    fn renderer_semantic_checkpoints_are_proven() {
        let mut failures = Vec::new();
        for target in PROVEN_TARGETS.iter().filter(|target| {
            matches!(
                target.module,
                "hybrid_terrain_traversal" | "gi_composite" | "brdf_tile"
            )
        }) {
            let verdict = verify_target(*target, None).unwrap();
            let mut alarm_kinds = std::collections::BTreeMap::<_, usize>::new();
            for alarm in &verdict.alarms {
                *alarm_kinds.entry(alarm.kind.as_str()).or_default() += 1;
            }
            let samples: Vec<_> = verdict
                .alarms
                .iter()
                .take(80)
                .map(|alarm| (&alarm.kind, alarm.line, &alarm.detail))
                .collect();
            if verdict.proof_status != "proven" {
                failures.push(format!(
                    "{}::{}: kinds={alarm_kinds:?}, samples={samples:#?}",
                    target.module, target.entry,
                ));
            }
        }
        assert!(failures.is_empty(), "{}", failures.join("\n\n"));
    }

    #[test]
    fn gi_budget_checkpoint_is_proven() {
        let target = PROVEN_TARGETS
            .iter()
            .find(|target| target.module == "gi_composite")
            .unwrap();
        let verdict = verify_target(*target, None).unwrap();
        assert_eq!(
            verdict.proof_status,
            "proven",
            "{}",
            serde_json::to_string_pretty(&verdict.alarms).unwrap()
        );
    }

    #[test]
    fn brdf_checkpoint_is_proven() {
        let target = PROVEN_TARGETS
            .iter()
            .find(|target| target.module == "brdf_tile")
            .unwrap();
        let verdict = verify_target(*target, None).unwrap();
        assert_eq!(
            verdict.proof_status,
            "proven",
            "{}",
            serde_json::to_string_pretty(&verdict.alarms).unwrap()
        );
    }

    #[test]
    fn brdf_half_vector_guard_ablation_is_rejected() {
        let target = PROVEN_TARGETS
            .iter()
            .find(|target| target.module == "brdf_tile")
            .copied()
            .unwrap();
        assert_eq!(verify_target(target, None).unwrap().proof_status, "proven");
        let source = load_target_source(target).unwrap().replace(
            "safe_half_vector(view_dir, light_dir, normal)",
            "normalize(view_dir + light_dir)",
        );
        let mutant = verify_source(
            target.module,
            target.path,
            target.entry,
            &source,
            target.contract,
        );
        assert_eq!(mutant.proof_status, "unproven");
        assert!(mutant
            .alarms
            .iter()
            .any(|alarm| alarm.kind == "possible_nan_or_inf"));
    }

    #[test]
    fn terrain_pbr_baseline_proven_then_ablation_rejected() {
        let target = Target {
            module: "terrain_pbr_pom",
            path: "src/shaders/terrain_pbr_pom.wgsl",
            entry: "fs_main",
            contract: "shaders/contracts/terrain_pbr_pom.toml",
            kind: "entry",
        };
        let baseline = verify_target(target, None).unwrap();
        assert_eq!(
            baseline.proof_status,
            "proven",
            "{} alarms; first alarms: {:#?}",
            baseline.alarms.len(),
            baseline.alarms.iter().take(40).collect::<Vec<_>>()
        );
        let ablation = verify_target(target, Some("height_range_div")).unwrap();
        assert_eq!(ablation.proof_status, "unproven");
    }

    #[test]
    fn aether_terrain_segment_is_proven_and_zero_division_is_rejected() {
        let target = PROVEN_TARGETS
            .iter()
            .find(|target| target.module == "aether_terrain_segment")
            .copied()
            .unwrap();
        let baseline = verify_target(target, None).unwrap();
        assert_eq!(
            baseline.proof_status,
            "proven",
            "{} alarms: {:#?}",
            baseline.alarms.len(),
            baseline.alarms
        );
        let source = crate::shader_sources::terrain().replace(
            "let path_per_sample = bounded_distance_m * density_scale * 0.0625;",
            "let path_per_sample = distance_m / (density_scale - density_scale);",
        );
        let mutant = verify_source(
            target.module,
            target.path,
            target.entry,
            &source,
            target.contract,
        );
        assert_eq!(mutant.proof_status, "unproven");
        assert!(mutant
            .alarms
            .iter()
            .any(|alarm| alarm.kind == "possible_nan_or_inf"));
    }

    #[test]
    fn aether_atmospheric_apply_is_proven_and_ablation_rejected() {
        let target = PROVEN_TARGETS
            .iter()
            .find(|target| target.module == "aether_terrain_apply")
            .copied()
            .unwrap();
        let baseline = verify_target(target, None).unwrap();
        assert_eq!(
            baseline.proof_status,
            "proven",
            "{} alarms: {:#?}",
            baseline.alarms.len(),
            baseline.alarms
        );
        let source = crate::shader_sources::terrain().replace(
            "let transported = bounded_surface * transmittance + finite_inscatter;",
            "let transported = bounded_surface / (transmittance - transmittance);",
        );
        let mutant = verify_source(
            target.module,
            target.path,
            target.entry,
            &source,
            target.contract,
        );
        assert_eq!(mutant.proof_status, "unproven");
        assert!(mutant
            .alarms
            .iter()
            .any(|alarm| alarm.kind == "possible_nan_or_inf"));
    }

    #[test]
    fn aether_atmospheric_apply_rejects_raw_zero_normalization() {
        let target = PROVEN_TARGETS
            .iter()
            .find(|target| target.module == "aether_terrain_apply")
            .copied()
            .unwrap();
        assert_eq!(verify_target(target, None).unwrap().proof_status, "proven");

        for (safe, unsafe_normalize) in [
            (
                "aether_terrain_finite_normalize(view_direction)",
                "normalize(view_direction)",
            ),
            (
                "aether_terrain_finite_normalize(sun_direction)",
                "normalize(sun_direction)",
            ),
        ] {
            let source = crate::shader_sources::terrain().replace(safe, unsafe_normalize);
            let mutant = verify_source(
                target.module,
                target.path,
                target.entry,
                &source,
                target.contract,
            );
            assert_eq!(
                mutant.proof_status, "unproven",
                "raw normalization unexpectedly proved: {unsafe_normalize}"
            );
            assert!(mutant
                .alarms
                .iter()
                .any(|alarm| alarm.kind == "possible_nan_or_inf"));
        }
    }

    #[test]
    fn terrain_helper_body_mutation_invalidates_summaries() {
        let target = PROVEN_TARGETS
            .iter()
            .find(|target| target.module == "terrain_pbr_pom")
            .copied()
            .unwrap();
        assert_eq!(verify_target(target, None).unwrap().proof_status, "proven");
        let source = crate::shader_sources::terrain().replace(
            "let sharpen = det_pow3(",
            "let invalid = 1.0 / (blend_sharpness - blend_sharpness);\n    let sharpen = vec3<f32>(invalid) + det_pow3(",
        );
        let mutant = verify_source(
            target.module,
            target.path,
            target.entry,
            &source,
            target.contract,
        );
        assert_eq!(mutant.proof_status, "unproven");
        assert!(mutant
            .alarms
            .iter()
            .any(|alarm| alarm.kind == "possible_nan_or_inf"));
    }

    #[test]
    fn pt_shade_baseline_proven_then_guard_ablation_rejected() {
        let source = pt_shade_guard_harness_source();
        let baseline = verify_source(
            "pt_shade",
            "src/shaders/pt_shade.wgsl",
            "prove_ggx_guard",
            &source,
            "shaders/contracts/pt_shade_guard.toml",
        );
        assert_eq!(
            baseline.proof_status,
            "proven",
            "{} alarms; first alarms: {:#?}",
            baseline.alarms.len(),
            baseline.alarms.iter().take(40).collect::<Vec<_>>()
        );
        let result = ablation_pt_shade_guard().unwrap();
        assert_eq!(result.baseline_proof_status.as_deref(), Some("proven"));
        assert_eq!(result.proof_status, "unproven", "{:#?}", result.alarms);
    }

    #[test]
    fn hybrid_body_mutation_invalidates_summaries() {
        let target = PROVEN_TARGETS
            .iter()
            .find(|target| target.module == "hybrid_terrain_traversal")
            .copied()
            .unwrap();
        assert_eq!(verify_target(target, None).unwrap().proof_status, "proven");
        let source = crate::shader_sources::hybrid_kernel().replace(
            "return w_sum / (f32(m) * target_pdf);",
            "return w_sum / (target_pdf - target_pdf);",
        );
        let mutant = verify_source(
            target.module,
            target.path,
            target.entry,
            &source,
            target.contract,
        );
        assert_eq!(mutant.proof_status, "unproven");
        assert!(mutant
            .alarms
            .iter()
            .any(|alarm| alarm.kind == "possible_nan_or_inf"));
    }

    #[test]
    fn ledger_coverage_rejects_overlap_unknown_and_missing_rows() {
        let registered = vec!["src/shaders/a.wgsl".into(), "src/shaders/b.wgsl".into()];
        let proven = BTreeSet::from(["src/shaders/a.wgsl".into()]);
        let ledger = BTreeSet::from(["src/shaders/b.wgsl".into()]);
        assert!(validate_ledger_coverage(&registered, &proven, &ledger)
            .unwrap()
            .is_empty());
        assert!(
            validate_ledger_coverage(&registered, &proven, &BTreeSet::new())
                .unwrap()
                .contains(&"src/shaders/b.wgsl".into())
        );
        assert!(validate_ledger_coverage(&registered, &proven, &proven).is_err());
        assert!(validate_ledger_coverage(
            &registered,
            &proven,
            &BTreeSet::from(["src/shaders/c.wgsl".into()])
        )
        .is_err());
    }
}
