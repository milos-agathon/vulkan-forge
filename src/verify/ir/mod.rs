mod engine;
mod eval;
mod expr;
mod ops;
mod value;

use super::contract::{EntryContract, InputContract};
use super::domain::Interval;
use engine::{Evaluator, FunctionRef};
use naga::Handle;
use value::Value;

#[derive(Debug)]
pub(super) struct Proof {
    pub alarms: Vec<ProofAlarm>,
}

#[derive(Debug)]
pub(super) struct ProofAlarm {
    pub line: usize,
    pub kind: &'static str,
    pub detail: String,
}

pub(super) fn prove_wgsl(
    source: &str,
    entry: &str,
    contract: &EntryContract,
) -> anyhow::Result<Proof> {
    let module = naga::front::wgsl::parse_str(source)?;
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)?;
    let function = module
        .entry_points
        .iter()
        .position(|candidate| candidate.name == entry)
        .map(FunctionRef::Entry)
        .or_else(|| {
            module
                .functions
                .iter()
                .find(|(_, candidate)| candidate.name.as_deref() == Some(entry))
                .map(|(handle, _)| FunctionRef::Regular(handle))
        })
        .ok_or_else(|| anyhow::anyhow!("Naga module has no function {entry:?}"))?;
    let root = function.function(&module);
    let args = root
        .arguments
        .iter()
        .map(|argument| {
            seed_value(
                &module,
                argument.ty,
                argument.name.as_deref().unwrap_or(""),
                contract,
                None,
            )
        })
        .collect();
    let initializer_evaluator = Evaluator {
        source,
        module: &module,
        info: &info,
        contract,
        alarms: Vec::new(),
    };
    let globals = module
        .global_variables
        .iter()
        .map(|(_, global)| {
            let name = global.name.as_deref().unwrap_or("");
            let fallback = || seed_value(&module, global.ty, name, contract, None);
            if contract.input(name).is_some() {
                return fallback();
            }
            match global
                .init
                .map(|initializer| initializer_evaluator.eval_const(initializer))
            {
                Some(value) if !matches!(value, Value::Opaque) => value,
                _ => fallback(),
            }
        })
        .collect();
    let arg_abs_min = root
        .arguments
        .iter()
        .map(|argument| {
            let name = argument.name.as_deref().unwrap_or("");
            contract
                .invariants
                .iter()
                .find_map(|invariant| match invariant {
                    super::contract::InvariantContract::AbsGreaterEqual { value, minimum }
                        if value == name =>
                    {
                        Some(*minimum)
                    }
                    _ => None,
                })
        })
        .collect();
    let arg_norm_min = root
        .arguments
        .iter()
        .map(|argument| {
            let name = argument.name.as_deref().unwrap_or("");
            contract
                .invariants
                .iter()
                .find_map(|invariant| match invariant {
                    super::contract::InvariantContract::NormGreaterEqual { value, minimum }
                        if value == name =>
                    {
                        Some(*minimum)
                    }
                    _ => None,
                })
        })
        .collect();
    let mut evaluator = Evaluator {
        source,
        module: &module,
        info: &info,
        contract,
        alarms: Vec::new(),
    };
    let flow =
        evaluator.run_function(function, args, arg_abs_min, arg_norm_min, globals, vec![])?;
    check_return_outputs(&mut evaluator, function, &flow);
    check_global_outputs(&mut evaluator, &flow);
    Ok(Proof {
        alarms: evaluator.alarms,
    })
}

fn check_return_outputs(evaluator: &mut Evaluator<'_>, function: FunctionRef, flow: &engine::Flow) {
    let root = function.function(evaluator.module);
    let Some(result) = &root.result else { return };
    let output_names = match (&evaluator.module.types[result.ty].inner, function) {
        (naga::TypeInner::Struct { members, .. }, _) => {
            members.iter().map(|member| member.name.clone()).collect()
        }
        (_, FunctionRef::Entry(_)) => vec![Some(binding_name(result.binding.as_ref()))],
        _ => vec![Some("return".into())],
    };
    for returned in &flow.returns {
        let Some(value) = &returned.value else {
            continue;
        };
        let values = match value {
            Value::Composite(values) if output_names.len() > 1 => values.clone(),
            value => vec![value.clone()],
        };
        for (name, value) in output_names.iter().zip(values) {
            let Some(output) = name.as_deref().and_then(|name| {
                evaluator
                    .contract
                    .outputs
                    .iter()
                    .find(|range| range.name == name)
            }) else {
                continue;
            };
            let within = if output.allow_nan {
                value.within_allow_nan(output.min, output.max)
            } else {
                value.within(output.min, output.max)
            };
            if !within {
                evaluator.push_alarm(
                    returned.line,
                    "output_range",
                    &format!(
                        "{} value {value:?} is outside [{}, {}]",
                        output.name, output.min, output.max
                    ),
                );
            }
        }
    }
}

fn check_global_outputs(evaluator: &mut Evaluator<'_>, flow: &engine::Flow) {
    let states = flow
        .returns
        .iter()
        .map(|returned| (&returned.frame, returned.line))
        .chain(flow.normal.iter().map(|frame| (frame, 1)));
    for (frame, line) in states {
        for (index, (_, global)) in evaluator.module.global_variables.iter().enumerate() {
            let Some(output) = global.name.as_deref().and_then(|name| {
                evaluator
                    .contract
                    .outputs
                    .iter()
                    .find(|range| range.name == name)
            }) else {
                continue;
            };
            let within = if output.allow_nan {
                frame.globals[index].within_allow_nan(output.min, output.max)
            } else {
                frame.globals[index].within(output.min, output.max)
            };
            if !within {
                evaluator.push_alarm(
                    line,
                    "output_range",
                    &format!(
                        "{} value {:?} is outside [{}, {}]",
                        output.name, frame.globals[index], output.min, output.max
                    ),
                );
            }
        }
    }
}

fn binding_name(binding: Option<&naga::Binding>) -> String {
    match binding {
        Some(naga::Binding::Location { location, .. }) => format!("location{location}"),
        Some(naga::Binding::BuiltIn(builtin)) => format!("builtin:{builtin:?}"),
        None => "return".into(),
    }
}

fn input_range(input: &InputContract) -> Option<(f32, f32)> {
    match input {
        InputContract::Value(range)
        | InputContract::Uniform(range)
        | InputContract::Texture { range, .. }
        | InputContract::Buffer { range, .. }
        | InputContract::BufferField(range) => Some((range.min, range.max)),
        _ => None,
    }
}

fn seed_value(
    module: &naga::Module,
    ty: Handle<naga::Type>,
    name: &str,
    contract: &EntryContract,
    inherited: Option<(f32, f32)>,
) -> Value {
    if let Some(input) = contract.input(name) {
        match input {
            InputContract::Texture { range, dimensions } => {
                let sample_scalar = Value::Float(Interval::new(range.min, range.max));
                let sample = match module.types[ty].inner {
                    naga::TypeInner::Image {
                        class: naga::ImageClass::Depth { .. },
                        ..
                    } => sample_scalar,
                    _ => sample_scalar.splat(naga::VectorSize::Quad),
                };
                return Value::Image {
                    name: name.to_string(),
                    sample: Box::new(sample),
                    dimensions: dimensions
                        .iter()
                        .map(|dimension| match dimension {
                            super::contract::DimensionContract::Minimum(minimum) => {
                                (*minimum as u64, u32::MAX as u64)
                            }
                            super::contract::DimensionContract::Symbol(symbol) => contract
                                .input(symbol)
                                .and_then(input_range)
                                .map_or((1, u32::MAX as u64), |(lo, hi)| {
                                    (lo.max(0.0) as u64, hi.max(0.0) as u64)
                                }),
                        })
                        .collect(),
                };
            }
            InputContract::Sampler(_) => return Value::Sampler,
            InputContract::Buffer { range, length } => {
                if let naga::TypeInner::Array { base, size, .. } = module.types[ty].inner {
                    let length = match (size, length) {
                        (naga::ArraySize::Constant(size), _) => Some(size.get() as u64),
                        (_, super::contract::BufferLength::Fixed(length)) => Some(*length as u64),
                        _ => None,
                    };
                    return Value::Array {
                        element: Box::new(seed_value(
                            module,
                            base,
                            &format!("{name}[]"),
                            contract,
                            Some((range.min, range.max)),
                        )),
                        length,
                    };
                }
            }
            _ => {}
        }
    }
    if let Some(output) = contract.outputs.iter().find(|output| output.name == name) {
        if let naga::TypeInner::Image {
            dim,
            arrayed,
            class,
        } = module.types[ty].inner
        {
            let scalar = Value::Float(Interval::new(output.min, output.max));
            let sample = match class {
                naga::ImageClass::Depth { .. } => scalar,
                _ => scalar.splat(naga::VectorSize::Quad),
            };
            return Value::Image {
                name: name.to_string(),
                sample: Box::new(sample),
                dimensions: output_dimensions(contract, name, dim, arrayed),
            };
        }
    }
    let range = contract.input(name).and_then(input_range).or(inherited);
    match &module.types[ty].inner {
        naga::TypeInner::Struct { members, .. } => Value::Composite(
            members
                .iter()
                .map(|member| {
                    let member_name = member.name.as_deref().map_or_else(
                        || name.to_string(),
                        |member| {
                            if name.is_empty() {
                                member.to_string()
                            } else {
                                format!("{name}.{member}")
                            }
                        },
                    );
                    seed_value(module, member.ty, &member_name, contract, range)
                })
                .collect(),
        ),
        naga::TypeInner::Vector { size, scalar } => {
            let component_names = ["x", "y", "z", "w"];
            Value::Composite(
                component_names
                    .iter()
                    .take(value::lanes(*size))
                    .map(|component| {
                        let component_name = format!("{name}.{component}");
                        let component_range = contract
                            .input(&component_name)
                            .and_then(input_range)
                            .or(range);
                        match scalar.kind {
                            naga::ScalarKind::Float => Value::Float(
                                component_range.map_or_else(Interval::unknown, |(lo, hi)| {
                                    Interval::new(lo, hi)
                                }),
                            ),
                            naga::ScalarKind::Uint => Value::Int {
                                lo: component_range.map_or(0, |(lo, _)| lo.max(0.0) as i64),
                                hi: component_range.map_or(u32::MAX as i64, |(_, hi)| {
                                    hi.min(u32::MAX as f32) as i64
                                }),
                            },
                            naga::ScalarKind::Sint => Value::Int {
                                lo: component_range.map_or(i32::MIN as i64, |(lo, _)| {
                                    lo.max(i32::MIN as f32) as i64
                                }),
                                hi: component_range.map_or(i32::MAX as i64, |(_, hi)| {
                                    hi.min(i32::MAX as f32) as i64
                                }),
                            },
                            _ => Value::Opaque,
                        }
                    })
                    .collect(),
            )
        }
        _ => range.map_or_else(
            || Value::unknown(module, &module.types[ty].inner),
            |(lo, hi)| Value::from_range(module, ty, lo, hi),
        ),
    }
}

fn output_dimensions(
    contract: &EntryContract,
    name: &str,
    dim: naga::ImageDimension,
    arrayed: bool,
) -> Vec<(u64, u64)> {
    let input_dimensions = |input: &InputContract| match input {
        InputContract::Texture { dimensions, .. } => Some(
            dimensions
                .iter()
                .map(|dimension| match dimension {
                    super::contract::DimensionContract::Minimum(minimum) => {
                        (*minimum as u64, u32::MAX as u64)
                    }
                    super::contract::DimensionContract::Symbol(symbol) => contract
                        .input(symbol)
                        .and_then(input_range)
                        .map_or((1, u32::MAX as u64), |(lo, hi)| {
                            (lo.max(1.0) as u64, hi.max(1.0) as u64)
                        }),
                })
                .collect::<Vec<_>>(),
        ),
        _ => None,
    };
    for invariant in &contract.invariants {
        if let super::contract::InvariantContract::SameDimensions { left, right } = invariant {
            let other = if left == name {
                Some(right)
            } else if right == name {
                Some(left)
            } else {
                None
            };
            if let Some(dimensions) = other
                .and_then(|other| contract.input(other))
                .and_then(input_dimensions)
            {
                return dimensions;
            }
        }
        if let super::contract::InvariantContract::DimensionsCover {
            texture,
            width,
            height,
        } = invariant
        {
            if texture == name {
                return [width, height]
                    .into_iter()
                    .map(|symbol| {
                        contract
                            .input(symbol)
                            .and_then(input_range)
                            .map_or((1, u32::MAX as u64), |(lo, hi)| {
                                (lo.max(1.0) as u64, hi.max(1.0) as u64)
                            })
                    })
                    .collect();
            }
        }
    }
    let spatial = match dim {
        naga::ImageDimension::D1 => 1,
        naga::ImageDimension::D2 => 2,
        naga::ImageDimension::D3 | naga::ImageDimension::Cube => 3,
    } + usize::from(arrayed);
    vec![(1, u32::MAX as u64); spatial]
}

#[cfg(test)]
mod tests;
