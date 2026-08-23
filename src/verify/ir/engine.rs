use super::value::{Place, Root};
use super::{EntryContract, ProofAlarm, Value};
use crate::verify::contract::{DimensionContract, InputContract, InvariantContract};
use naga::{BinaryOperator, Expression, Handle, Statement};
use std::collections::HashMap;

const FNV1A_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV1A_PRIME: u64 = 0x0000_0100_0000_01b3;
pub(super) const PINNED_DETERMINISM_SOURCE_HASH: u64 = 0xf664_b696_d596_de84;
pub(super) const PINNED_HYBRID_KERNEL_SOURCE_HASH: u64 = 0x31e2_0aa2_b620_5ac2;
pub(super) const PINNED_TERRAIN_SOURCE_HASH: u64 = 0x6405_3736_7c68_795b;

#[derive(Clone, Copy)]
pub(super) enum FunctionRef {
    Entry(usize),
    Regular(Handle<naga::Function>),
}

impl FunctionRef {
    pub(super) fn function(self, module: &naga::Module) -> &naga::Function {
        match self {
            Self::Entry(index) => &module.entry_points[index].function,
            Self::Regular(handle) => &module.functions[handle],
        }
    }

    pub(super) fn info(self, info: &naga::valid::ModuleInfo) -> &naga::valid::FunctionInfo {
        match self {
            Self::Entry(index) => info.get_entry_point(index),
            Self::Regular(handle) => &info[handle],
        }
    }
}

#[derive(Clone, PartialEq)]
pub(super) struct Frame {
    pub(super) args: Vec<Value>,
    pub(super) globals: Vec<Value>,
    pub(super) locals: Vec<Value>,
    /// Function-pointer arguments are copied into these aliases for a call and
    /// copied back by the caller. WGSL permits pointers to function storage;
    /// treating them as opaque loses ordinary RNG/state helpers.
    pub(super) aliases: Vec<(Place, Value)>,
    pub(super) values: HashMap<Handle<Expression>, Value>,
    pub(super) abs_min: HashMap<Place, f32>,
    pub(super) expr_abs_min: HashMap<Handle<Expression>, f32>,
    pub(super) relations: HashMap<Handle<Expression>, Relation>,
    pub(super) place_relations: HashMap<Place, Relation>,
    pub(super) index_within: HashMap<Place, Root>,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) enum Relation {
    SquaredNorm(Place),
    InverseSqrt(Place),
    InverseNorm(Place),
    SqrtProduct(Place),
    Offset(Place, f32),
    Difference(Place, Place),
    NonZeroNorm,
    NormOf(Place),
    NormOfExpr(Handle<Expression>),
    ArrayLength(Root),
    ImageDimensions(String),
    ImageDimension(String, usize),
    ImageUpperIndex(String),
    ImageUpperIndexAxis(String, usize),
    InImage(String),
    InImageAxes(String, u8),
    /// A value is known to be strictly smaller than this symbolic dimension.
    /// This is retained separately from its numeric interval because a runtime
    /// dimension's declared minimum is not an upper bound on the dimension.
    LessThan(Place),
    BudgetTerms {
        maximum_bits: u32,
        sources: u8,
    },
}

pub(super) struct Returned {
    pub(super) value: Option<Value>,
    pub(super) frame: Frame,
    pub(super) line: usize,
}

#[derive(Default)]
pub(super) struct Flow {
    pub(super) normal: Vec<Frame>,
    pub(super) returns: Vec<Returned>,
    breaks: Vec<Frame>,
    continues: Vec<Frame>,
}

pub(super) struct Evaluator<'a> {
    pub(super) source: &'a str,
    pub(super) module: &'a naga::Module,
    pub(super) info: &'a naga::valid::ModuleInfo,
    pub(super) contract: &'a EntryContract,
    pub(super) alarms: Vec<ProofAlarm>,
}

impl Evaluator<'_> {
    pub(super) fn run_function(
        &mut self,
        function_ref: FunctionRef,
        args: Vec<Value>,
        arg_abs_min: Vec<Option<f32>>,
        arg_norm_min: Vec<Option<f32>>,
        globals: Vec<Value>,
        aliases: Vec<(Place, Value)>,
    ) -> anyhow::Result<Flow> {
        let function = function_ref.function(self.module);
        let mut frame = Frame {
            args,
            globals,
            locals: Vec::new(),
            aliases,
            values: HashMap::new(),
            relations: HashMap::new(),
            place_relations: arg_norm_min
                .into_iter()
                .enumerate()
                .filter_map(|(index, minimum)| {
                    minimum.filter(|minimum| *minimum > 0.0).map(|_| {
                        (
                            Place {
                                root: Root::Argument(index),
                                path: Vec::new(),
                            },
                            Relation::NonZeroNorm,
                        )
                    })
                })
                .collect(),
            index_within: HashMap::new(),
            abs_min: arg_abs_min
                .into_iter()
                .enumerate()
                .filter_map(|(index, minimum)| {
                    minimum.map(|minimum| {
                        (
                            Place {
                                root: Root::Argument(index),
                                path: Vec::new(),
                            },
                            minimum,
                        )
                    })
                })
                .collect(),
            expr_abs_min: HashMap::new(),
        };
        for (_, local) in function.local_variables.iter() {
            let value = local.init.map_or_else(
                || Value::zero(self.module, local.ty),
                |handle| {
                    self.eval_expr(function_ref, &mut frame, handle)
                        .unwrap_or_else(|_| {
                            Value::unknown(self.module, &self.module.types[local.ty].inner)
                        })
                },
            );
            frame.locals.push(value);
        }
        if self.has_pinned_determinism_source() {
            if let Some(value) = self.summarize_direct_return(function_ref, &frame) {
                return Ok(Flow {
                    returns: vec![Returned {
                        value: Some(value),
                        frame,
                        line: function
                            .body
                            .iter()
                            .find_map(statement_return_handle)
                            .map_or(1, |handle| self.expression_line(function_ref, handle)),
                    }],
                    ..Flow::default()
                });
            }
        }
        self.exec_block(function_ref, &function.body, vec![frame])
    }

    fn exec_block(
        &mut self,
        function_ref: FunctionRef,
        block: &naga::Block,
        mut states: Vec<Frame>,
    ) -> anyhow::Result<Flow> {
        let mut terminal = Flow::default();
        for statement in block {
            let mut next = Vec::new();
            for mut state in states {
                match statement {
                    Statement::Emit(range) => {
                        for handle in range.clone() {
                            self.eval_expr(function_ref, &mut state, handle)?;
                        }
                        next.push(state);
                    }
                    Statement::Store { pointer, value } => {
                        let place = self.eval_expr(function_ref, &mut state, *pointer)?;
                        let value_handle = *value;
                        let stored_value =
                            self.eval_expr(function_ref, &mut state, value_handle)?;
                        let relation = state.relations.get(&value_handle).cloned();
                        let target = match &place {
                            Value::Pointer(place) => Some(place.clone()),
                            _ => None,
                        };
                        if !self.store(&mut state, place, stored_value) {
                            self.alarm_expr(
                                function_ref,
                                *pointer,
                                "unsupported_ir",
                                "store target is not a supported pointer",
                            );
                        }
                        if let Some(target) = target {
                            if let Some(relation) = relation {
                                state.place_relations.insert(target, relation);
                            } else {
                                state.place_relations.remove(&target);
                            }
                        }
                        next.push(state);
                    }
                    Statement::ImageStore {
                        image,
                        coordinate,
                        array_index,
                        value,
                    } => {
                        let image_value = self.eval_expr(function_ref, &mut state, *image)?;
                        let mut coordinate_value =
                            self.eval_expr(function_ref, &mut state, *coordinate)?;
                        if let Some(array_index) = array_index {
                            let layer = self.eval_expr(function_ref, &mut state, *array_index)?;
                            coordinate_value = match coordinate_value {
                                Value::Composite(mut coordinates) => {
                                    coordinates.push(layer);
                                    Value::Composite(coordinates)
                                }
                                coordinate => Value::Composite(vec![coordinate, layer]),
                            };
                        }
                        let _ = self.image_load(
                            function_ref,
                            &state,
                            *coordinate,
                            image_value,
                            *coordinate,
                            coordinate_value,
                            *array_index,
                        );
                        let stored = self.eval_expr(function_ref, &mut state, *value)?;
                        let Some(Place {
                            root: Root::Global(index),
                            ..
                        }) = self.place_of_expr(function_ref, *image)
                        else {
                            self.alarm_expr(
                                function_ref,
                                *image,
                                "unsupported_ir",
                                "textureStore target is not a global image",
                            );
                            next.push(state);
                            continue;
                        };
                        if let Value::Image { sample, .. } = &mut state.globals[index] {
                            **sample = stored;
                        }
                        next.push(state);
                    }
                    Statement::If {
                        condition,
                        accept,
                        reject,
                    } => {
                        let condition_value =
                            self.eval_expr(function_ref, &mut state, *condition)?;
                        let (can_false, can_true) = match condition_value {
                            Value::Bool {
                                can_false,
                                can_true,
                            } => (can_false, can_true),
                            _ => {
                                self.alarm_expr(
                                    function_ref,
                                    *condition,
                                    "unsupported_ir",
                                    "if condition is not boolean",
                                );
                                (true, true)
                            }
                        };
                        if can_true {
                            let mut accepted = state.clone();
                            self.refine_condition(function_ref, &mut accepted, *condition, true);
                            Self::merge_flow(
                                &mut terminal,
                                self.exec_block(function_ref, accept, vec![accepted])?,
                                &mut next,
                            );
                        }
                        if can_false {
                            self.refine_condition(function_ref, &mut state, *condition, false);
                            Self::merge_flow(
                                &mut terminal,
                                self.exec_block(function_ref, reject, vec![state])?,
                                &mut next,
                            );
                        }
                    }
                    Statement::Switch { selector, cases } => {
                        let selector_value = self.eval_expr(function_ref, &mut state, *selector)?;
                        Self::merge_flow(
                            &mut terminal,
                            self.exec_switch(function_ref, cases, selector_value, state)?,
                            &mut next,
                        );
                    }
                    Statement::Loop {
                        body,
                        continuing,
                        break_if,
                    } => {
                        let flow =
                            self.exec_loop(function_ref, body, continuing, *break_if, state)?;
                        Self::merge_flow(&mut terminal, flow, &mut next);
                    }
                    Statement::Call {
                        function,
                        arguments,
                        result,
                    } => {
                        let arg_abs_min: Vec<Option<f32>> = arguments
                            .iter()
                            .map(|handle| self.abs_min_for_expr(function_ref, &state, *handle))
                            .collect();
                        let arg_norm_min: Vec<Option<f32>> = arguments
                            .iter()
                            .map(|handle| self.norm_min_for_expr(function_ref, &state, *handle))
                            .collect();
                        let raw_args = arguments
                            .iter()
                            .map(|handle| self.eval_expr(function_ref, &mut state, *handle))
                            .collect::<anyhow::Result<Vec<_>>>()?;
                        let mut aliases = Vec::new();
                        let args: Vec<Value> = raw_args
                            .into_iter()
                            .map(|value| match value {
                                Value::Pointer(place) => {
                                    let value = self
                                        .load(&state, Value::Pointer(place.clone()))
                                        .unwrap_or(Value::Opaque);
                                    let alias = aliases.len();
                                    aliases.push((place, value));
                                    Value::Pointer(Place {
                                        root: Root::Alias(alias),
                                        path: Vec::new(),
                                    })
                                }
                                value => value,
                            })
                            .collect();
                        if let Some((value, relation)) = self.summarize_call(
                            function_ref,
                            &state,
                            *function,
                            arguments,
                            &args,
                            &arg_abs_min,
                        ) {
                            if let Some(result) = result {
                                state.values.insert(*result, value);
                                if let Some(relation) = relation {
                                    state.relations.insert(*result, relation);
                                }
                            }
                            next.push(state);
                            continue;
                        }
                        let flow = self.run_function(
                            FunctionRef::Regular(*function),
                            args,
                            arg_abs_min,
                            arg_norm_min,
                            state.globals.clone(),
                            aliases,
                        )?;
                        if flow.returns.is_empty() {
                            self.push_alarm(
                                1,
                                "unsupported_ir",
                                &format!(
                                    "called function {:?} has no reachable return",
                                    self.module.functions[*function].name
                                ),
                            );
                            next.push(state);
                        } else {
                            for returned in flow.returns {
                                let mut caller = state.clone();
                                caller.globals = returned.frame.globals;
                                for (target, value) in returned.frame.aliases {
                                    let _ = caller.root_mut(&target.root).and_then(|root| {
                                        root.write_path(&target.path, value).then_some(root)
                                    });
                                }
                                if let (Some(result), Some(value)) = (result, returned.value) {
                                    caller.values.insert(*result, value);
                                }
                                next.push(caller);
                            }
                        }
                    }
                    Statement::Atomic {
                        pointer,
                        fun,
                        value,
                        result,
                    } => {
                        let pointer_value = self.eval_expr(function_ref, &mut state, *pointer)?;
                        let current = self
                            .load(&state, pointer_value.clone())
                            .unwrap_or(Value::Opaque);
                        let operand = self.eval_expr(function_ref, &mut state, *value)?;
                        let updated = match fun {
                            naga::AtomicFunction::Add => current
                                .clone()
                                .binary_int(operand, |a, b| a.saturating_add(b)),
                            naga::AtomicFunction::Subtract => current
                                .clone()
                                .binary_int(operand, |a, b| a.saturating_sub(b)),
                            naga::AtomicFunction::And => {
                                current.clone().binary_int(operand, |a, b| a & b)
                            }
                            naga::AtomicFunction::ExclusiveOr => {
                                current.clone().binary_int(operand, |a, b| a ^ b)
                            }
                            naga::AtomicFunction::InclusiveOr => {
                                current.clone().binary_int(operand, |a, b| a | b)
                            }
                            naga::AtomicFunction::Min => {
                                current.clone().binary_int(operand, i64::min)
                            }
                            naga::AtomicFunction::Max => {
                                current.clone().binary_int(operand, i64::max)
                            }
                            naga::AtomicFunction::Exchange { compare: None } => Some(operand),
                            naga::AtomicFunction::Exchange { compare: Some(_) } => None,
                        };
                        if let Some(updated) = updated {
                            if !self.store(&mut state, pointer_value, updated) {
                                self.alarm_expr(
                                    function_ref,
                                    *pointer,
                                    "unsupported_ir",
                                    "atomic target is not a supported pointer",
                                );
                            }
                            state.values.insert(*result, current);
                        } else {
                            self.alarm_expr(
                                function_ref,
                                *pointer,
                                "unsupported_ir",
                                "compare-exchange atomic is not yet interpreted",
                            );
                        }
                        next.push(state);
                    }
                    Statement::Return { value } => {
                        let value = value
                            .map(|handle| self.eval_expr(function_ref, &mut state, handle))
                            .transpose()?;
                        let line = value
                            .as_ref()
                            .and_then(|_| statement_return_handle(statement))
                            .map(|handle| self.expression_line(function_ref, handle))
                            .unwrap_or(1);
                        terminal.returns.push(Returned {
                            value,
                            frame: state,
                            line,
                        });
                    }
                    Statement::Block(block) => Self::merge_flow(
                        &mut terminal,
                        self.exec_block(function_ref, block, vec![state])?,
                        &mut next,
                    ),
                    Statement::Break => terminal.breaks.push(state),
                    Statement::Continue => terminal.continues.push(state),
                    Statement::Kill => {}
                    Statement::Barrier(_) => next.push(state),
                    other => {
                        self.push_alarm(
                            1,
                            "unsupported_ir",
                            &format!("unsupported reachable statement {other:?}"),
                        );
                        next.push(state);
                    }
                }
            }
            states = Self::coalesce(next);
            if states.is_empty() {
                break;
            }
        }
        terminal.normal.extend(states);
        Ok(terminal)
    }

    fn merge_flow(terminal: &mut Flow, flow: Flow, next: &mut Vec<Frame>) {
        next.extend(flow.normal);
        terminal.returns.extend(flow.returns);
        terminal.breaks.extend(flow.breaks);
        terminal.continues.extend(flow.continues);
    }

    fn exec_switch(
        &mut self,
        function_ref: FunctionRef,
        cases: &[naga::SwitchCase],
        selector: Value,
        state: Frame,
    ) -> anyhow::Result<Flow> {
        let selector_range = match selector {
            Value::Int { lo, hi } => Some((lo, hi)),
            _ => None,
        };
        let mut result = Flow::default();
        for (start, case) in cases.iter().enumerate() {
            let reachable =
                match case.value {
                    naga::SwitchValue::I32(value) => selector_range
                        .is_none_or(|(lo, hi)| lo <= value as i64 && value as i64 <= hi),
                    naga::SwitchValue::U32(value) => selector_range
                        .is_none_or(|(lo, hi)| lo <= value as i64 && value as i64 <= hi),
                    naga::SwitchValue::Default => true,
                };
            if !reachable {
                continue;
            }
            let mut states = vec![state.clone()];
            for case in &cases[start..] {
                let flow = self.exec_block(function_ref, &case.body, states)?;
                result.returns.extend(flow.returns);
                result.continues.extend(flow.continues);
                result.normal.extend(flow.breaks);
                states = flow.normal;
                if !case.fall_through || states.is_empty() {
                    result.normal.extend(states);
                    break;
                }
            }
        }
        result.normal = Self::coalesce(result.normal);
        Ok(result)
    }

    fn coalesce(mut states: Vec<Frame>) -> Vec<Frame> {
        if states.len() <= 1 {
            return states;
        }
        let mut joined = states.remove(0);
        for state in states {
            joined.join(&state);
        }
        vec![joined]
    }

    pub(super) fn load(&self, frame: &Frame, value: Value) -> Option<Value> {
        let Value::Pointer(place) = value else {
            return Some(value);
        };
        frame.root(&place.root)?.read_path(&place.path)
    }

    pub(super) fn dynamic_access(
        &mut self,
        function_ref: FunctionRef,
        frame: &Frame,
        handle: Handle<Expression>,
        base: Value,
        index_handle: Handle<Expression>,
        index: Value,
    ) -> Value {
        let (lo, hi) = match index {
            Value::Int { lo, hi } => (lo, hi),
            _ => {
                self.alarm_expr(
                    function_ref,
                    handle,
                    "possible_oob",
                    "dynamic index has no integer range",
                );
                return Value::Opaque;
            }
        };
        let selected = match base {
            Value::Pointer(mut place) => {
                let current = frame
                    .root(&place.root)
                    .and_then(|root| root.read_path(&place.path));
                let length = match current.as_ref() {
                    Some(Value::Array { length, .. }) => *length,
                    Some(Value::Composite(values)) => Some(values.len() as u64),
                    _ => None,
                };
                let relationally_bounded = self
                    .place_of_expr(function_ref, index_handle)
                    .and_then(|index| frame.index_within.get(&index))
                    == Some(&place.root);
                let contract_bounded =
                    self.index_within_product(function_ref, frame, index_handle, &place.root)
                        || self.index_within_count(function_ref, frame, index_handle, &place.root);
                if !(lo >= 0 && relationally_bounded)
                    && !(lo >= 0 && contract_bounded)
                    && !length.is_some_and(|length| lo >= 0 && hi < length as i64)
                {
                    self.alarm_expr(
                        function_ref,
                        handle,
                        "possible_oob",
                        "dynamic buffer/array index is not proved in bounds",
                    );
                }
                place.path.push(usize::MAX);
                Value::Pointer(place)
            }
            Value::Array { element, length } => {
                if !length.is_some_and(|length| lo >= 0 && hi < length as i64) {
                    self.alarm_expr(
                        function_ref,
                        handle,
                        "possible_oob",
                        "dynamic array index is not proved in bounds",
                    );
                }
                *element
            }
            Value::Composite(values) => {
                if lo < 0 || hi >= values.len() as i64 {
                    self.alarm_expr(
                        function_ref,
                        handle,
                        "possible_oob",
                        "dynamic vector index is not proved in bounds",
                    );
                }
                values
                    .into_iter()
                    .reduce(|left, right| left.join(&right))
                    .unwrap_or(Value::Opaque)
            }
            _ => {
                self.alarm_expr(
                    function_ref,
                    handle,
                    "unsupported_ir",
                    "dynamic access base is unsupported",
                );
                Value::Opaque
            }
        };
        selected
    }

    pub(super) fn image_load(
        &mut self,
        function_ref: FunctionRef,
        frame: &Frame,
        handle: Handle<Expression>,
        image: Value,
        coordinate_handle: Handle<Expression>,
        coordinate: Value,
        array_index_handle: Option<Handle<Expression>>,
    ) -> Value {
        let Value::Image {
            name,
            sample,
            dimensions,
        } = image
        else {
            self.alarm_expr(
                function_ref,
                handle,
                "unsupported_ir",
                "loaded image has no texture contract",
            );
            return Value::Opaque;
        };
        let coordinates = match coordinate {
            Value::Int { lo, hi } => vec![(lo, hi)],
            Value::Composite(values) => values
                .into_iter()
                .filter_map(|value| match value {
                    Value::Int { lo, hi } => Some((lo, hi)),
                    _ => None,
                })
                .collect(),
            _ => Vec::new(),
        };
        let mut relation_mask = frame
            .relations
            .get(&coordinate_handle)
            .and_then(|relation| match relation {
                Relation::InImage(image) if self.same_dimensions(image, &name) => {
                    Some((1u8 << dimensions.len()) - 1)
                }
                Relation::InImageAxes(image, mask) if self.same_dimensions(image, &name) => {
                    Some(*mask)
                }
                _ => None,
            })
            .unwrap_or(0);
        if let Some(array_index) = array_index_handle {
            let layer_relation = frame
                .relations
                .get(&array_index)
                .cloned()
                .or_else(|| {
                    self.place_of_expr(function_ref, array_index)
                        .and_then(|place| frame.place_relations.get(&place).cloned())
                })
                .or_else(|| {
                    let function = function_ref.function(self.module);
                    let Expression::As { expr, .. } = function.expressions[array_index] else {
                        return None;
                    };
                    frame.relations.get(&expr).cloned().or_else(|| {
                        self.place_of_expr(function_ref, expr)
                            .and_then(|place| frame.place_relations.get(&place).cloned())
                    })
                });
            let layer_fits_signed_index = coordinates
                .last()
                .is_some_and(|&(lo, hi)| lo >= 0 && hi <= i32::MAX as i64);
            let layer_axis = dimensions.len().saturating_sub(1);
            let layer_bit = 1u8 << layer_axis;
            let layer_axis_is_proven = match layer_relation {
                Some(Relation::InImageAxes(image, mask)) => {
                    self.same_dimensions(&image, &name) && mask & layer_bit != 0
                }
                _ => false,
            };
            if layer_fits_signed_index && layer_axis_is_proven {
                relation_mask |= layer_bit;
            }
        }
        let in_bounds = coordinates.len() == dimensions.len()
            && coordinates.iter().zip(&dimensions).enumerate().all(
                |(axis, (&(lo, hi), &(dim_lo, _)))| {
                    lo >= 0 && ((relation_mask & (1u8 << axis)) != 0 || hi < dim_lo as i64)
                },
            );
        if !in_bounds {
            let place = self.place_of_expr(function_ref, coordinate_handle);
            self.alarm_expr(
                function_ref,
                handle,
                "possible_oob",
                &format!(
                    "textureLoad coordinate is not proved in bounds; image={name:?}, coordinates={coordinates:?}, dimensions={dimensions:?}, relation_mask={relation_mask:#05b}, expr={:?}, relation={:?}, place_relation={:?}, array_index={:?}, array_relation={:?}",
                    function_ref.function(self.module).expressions[coordinate_handle],
                    frame.relations.get(&coordinate_handle),
                    place.and_then(|place| frame.place_relations.get(&place)),
                    array_index_handle.map(|index| &function_ref.function(self.module).expressions[index]),
                    array_index_handle.and_then(|index| frame.relations.get(&index))
                ),
            );
        }
        *sample
    }

    fn exec_loop(
        &mut self,
        function_ref: FunctionRef,
        body: &naga::Block,
        continuing: &naga::Block,
        break_if: Option<Handle<Expression>>,
        initial: Frame,
    ) -> anyhow::Result<Flow> {
        let mut header = initial;
        let mut exits = Flow::default();
        for iteration in 0..8 {
            let mut iteration_header = header.clone();
            Self::invalidate_block_expressions(&mut iteration_header, body);
            Self::invalidate_block_expressions(&mut iteration_header, continuing);
            let body_flow = self.exec_block(function_ref, body, vec![iteration_header])?;
            exits.returns.extend(body_flow.returns);
            exits.normal.extend(body_flow.breaks);
            let back_inputs = body_flow
                .normal
                .into_iter()
                .chain(body_flow.continues)
                .collect();
            let continuing_flow = self.exec_block(function_ref, continuing, back_inputs)?;
            exits.returns.extend(continuing_flow.returns);
            exits.normal.extend(continuing_flow.breaks);
            let mut back = continuing_flow
                .normal
                .into_iter()
                .chain(continuing_flow.continues)
                .collect::<Vec<_>>();
            if let Some(condition) = break_if {
                let mut looping = Vec::new();
                for mut state in back {
                    let condition_value = self.eval_expr(function_ref, &mut state, condition)?;
                    let (can_false, can_true) = match condition_value {
                        Value::Bool {
                            can_false,
                            can_true,
                        } => (can_false, can_true),
                        _ => (true, true),
                    };
                    if can_true {
                        let mut exit = state.clone();
                        self.refine_condition(function_ref, &mut exit, condition, true);
                        exits.normal.push(exit);
                    }
                    if can_false {
                        self.refine_condition(function_ref, &mut state, condition, false);
                        looping.push(state);
                    }
                }
                back = looping;
            }
            let Some(candidate) = Frame::join_all(back) else {
                return Ok(exits);
            };
            let next = if iteration < 4 {
                candidate
            } else {
                header.clone().widened(&candidate)
            };
            if next == header {
                if exits.normal.is_empty() && exits.returns.is_empty() {
                    self.push_alarm(
                        1,
                        "loop_widening",
                        "loop has no reachable exit after abstract-state convergence",
                    );
                }
                return Ok(exits);
            }
            header = next;
        }
        self.push_alarm(
            1,
            "loop_widening",
            "loop abstract state did not converge after widening",
        );
        Ok(exits)
    }

    fn invalidate_block_expressions(frame: &mut Frame, block: &naga::Block) {
        for statement in block {
            match statement {
                Statement::Emit(range) => {
                    for handle in range.clone() {
                        frame.values.remove(&handle);
                        frame.relations.remove(&handle);
                    }
                }
                Statement::Block(block) => Self::invalidate_block_expressions(frame, block),
                Statement::If { accept, reject, .. } => {
                    Self::invalidate_block_expressions(frame, accept);
                    Self::invalidate_block_expressions(frame, reject);
                }
                Statement::Switch { cases, .. } => {
                    for case in cases {
                        Self::invalidate_block_expressions(frame, &case.body);
                    }
                }
                Statement::Loop {
                    body, continuing, ..
                } => {
                    Self::invalidate_block_expressions(frame, body);
                    Self::invalidate_block_expressions(frame, continuing);
                }
                _ => {}
            }
        }
    }

    fn refine_condition(
        &self,
        function_ref: FunctionRef,
        frame: &mut Frame,
        condition: Handle<Expression>,
        truth: bool,
    ) {
        let function = function_ref.function(self.module);
        if let Expression::Relational {
            fun: naga::RelationalFunction::All,
            argument,
        } = function.expressions[condition]
        {
            if let Expression::Binary { op, left, right } = function.expressions[argument] {
                let excludes_zero = matches!(
                    (op, truth),
                    (BinaryOperator::Equal, false) | (BinaryOperator::NotEqual, true)
                );
                if excludes_zero
                    && frame
                        .values
                        .get(&right)
                        .is_some_and(|value| value.within(0.0, 0.0))
                {
                    frame.relations.insert(left, Relation::NonZeroNorm);
                    if let Some(place) = self.place_of_expr(function_ref, left) {
                        frame.place_relations.insert(place, Relation::NonZeroNorm);
                    }
                }
            }
            return;
        }
        if let Expression::Binary { op, left, right } = function.expressions[condition] {
            match (op, truth) {
                (BinaryOperator::LogicalAnd, true) | (BinaryOperator::LogicalOr, false) => {
                    self.refine_condition(function_ref, frame, left, truth);
                    self.refine_condition(function_ref, frame, right, truth);
                    return;
                }
                _ => {}
            }
        }
        let Expression::Binary { op, left, right } = function.expressions[condition] else {
            return;
        };
        if matches!(
            (op, truth),
            (BinaryOperator::Less, true) | (BinaryOperator::GreaterEqual, false)
        ) {
            if let Some(Relation::ImageDimension(image, axis)) =
                frame.relations.get(&right).cloned()
            {
                self.mark_image_axis(function_ref, frame, left, &image, axis);
            }
        }
        let strict_less = matches!(
            (op, truth),
            (BinaryOperator::Less, true)
                | (BinaryOperator::GreaterEqual, false)
                | (BinaryOperator::Greater, true)
                | (BinaryOperator::LessEqual, false)
        );
        if strict_less {
            let (smaller, larger) = match op {
                BinaryOperator::Less | BinaryOperator::GreaterEqual => (left, right),
                BinaryOperator::Greater | BinaryOperator::LessEqual => (right, left),
                _ => unreachable!(),
            };
            if let (Some(smaller), Some(larger)) = (
                self.place_of_expr(function_ref, smaller),
                self.place_of_expr(function_ref, larger),
            ) {
                frame
                    .place_relations
                    .insert(smaller, Relation::LessThan(larger));
            }
        }
        if let (Some(Relation::NormOf(place)), Some(Value::Float(bound))) = (
            frame.relations.get(&left).cloned(),
            frame.values.get(&right).cloned(),
        ) {
            let proves_positive = bound.lo == bound.hi
                && bound.lo > 0.0
                && matches!(
                    (op, truth),
                    (BinaryOperator::Less, false)
                        | (BinaryOperator::LessEqual, false)
                        | (BinaryOperator::Greater, true)
                        | (BinaryOperator::GreaterEqual, true)
                );
            if proves_positive {
                frame.place_relations.insert(place, Relation::NonZeroNorm);
            }
        }
        if let (Some(Relation::NormOfExpr(source)), Some(Value::Float(bound))) = (
            frame.relations.get(&left).cloned(),
            frame.values.get(&right).cloned(),
        ) {
            let proves_positive = bound.lo == bound.hi
                && bound.lo > 0.0
                && matches!(
                    (op, truth),
                    (BinaryOperator::Less, false)
                        | (BinaryOperator::LessEqual, false)
                        | (BinaryOperator::Greater, true)
                        | (BinaryOperator::GreaterEqual, true)
                );
            if proves_positive {
                frame.relations.insert(source, Relation::NonZeroNorm);
            }
        }
        if let (Some(index), Some(Relation::ArrayLength(array))) = (
            self.place_of_expr(function_ref, left),
            frame.relations.get(&right),
        ) {
            let bounded = matches!(
                (op, truth),
                (BinaryOperator::Less, true) | (BinaryOperator::GreaterEqual, false)
            );
            if bounded {
                frame.index_within.insert(index, *array);
                return;
            }
        }
        let (place_handle, zero_handle) = if self.place_of_expr(function_ref, left).is_some() {
            (left, right)
        } else {
            (right, left)
        };
        if let (Some(Value::Float(current)), Some(Value::Float(bound))) = (
            frame.values.get(&left).cloned(),
            frame.values.get(&right).cloned(),
        ) {
            if bound.lo == bound.hi && bound.is_finite_only() {
                let b = bound.lo;
                let (lo, hi) = match (op, truth) {
                    (BinaryOperator::Less, true) => (current.lo, current.hi.min(next_down(b))),
                    (BinaryOperator::Less, false) => (current.lo.max(b), current.hi),
                    (BinaryOperator::LessEqual, true) => (current.lo, current.hi.min(b)),
                    (BinaryOperator::LessEqual, false) => (current.lo.max(next_up(b)), current.hi),
                    (BinaryOperator::Greater, true) => (current.lo.max(next_up(b)), current.hi),
                    (BinaryOperator::Greater, false) => (current.lo, current.hi.min(b)),
                    (BinaryOperator::GreaterEqual, true) => (current.lo.max(b), current.hi),
                    (BinaryOperator::GreaterEqual, false) => {
                        (current.lo, current.hi.min(next_down(b)))
                    }
                    _ => (current.lo, current.hi),
                };
                if lo <= hi {
                    frame.values.insert(
                        left,
                        Value::Float(crate::verify::domain::Interval::new(lo, hi)),
                    );
                    if b > 0.0
                        && ((matches!(op, BinaryOperator::Greater | BinaryOperator::GreaterEqual)
                            && truth)
                            || (matches!(op, BinaryOperator::Less | BinaryOperator::LessEqual)
                                && !truth))
                    {
                        frame.expr_abs_min.insert(left, b);
                    }
                }
            }
        }
        let Some(place) = self.place_of_expr(function_ref, place_handle) else {
            return;
        };
        let Some(constant) = frame.values.get(&zero_handle).cloned() else {
            return;
        };
        if let Value::Float(value) = constant {
            if value.lo == value.hi && value.is_finite_only() {
                let excludes_zero = matches!(
                    (op, truth),
                    (BinaryOperator::Equal, false) | (BinaryOperator::NotEqual, true)
                );
                if excludes_zero && value.lo == 0.0 {
                    frame.abs_min.insert(place.clone(), f32::MIN_POSITIVE);
                }
                let Some(Value::Float(current)) = frame
                    .root(&place.root)
                    .and_then(|root| root.read_path(&place.path))
                else {
                    return;
                };
                let bound = value.lo;
                let (lo, hi) = match (op, truth) {
                    (BinaryOperator::Less, true) => (current.lo, current.hi.min(next_down(bound))),
                    (BinaryOperator::Less, false) => (current.lo.max(bound), current.hi),
                    (BinaryOperator::LessEqual, true) => (current.lo, current.hi.min(bound)),
                    (BinaryOperator::LessEqual, false) => {
                        (current.lo.max(next_up(bound)), current.hi)
                    }
                    (BinaryOperator::Greater, true) => (current.lo.max(next_up(bound)), current.hi),
                    (BinaryOperator::Greater, false) => (current.lo, current.hi.min(bound)),
                    (BinaryOperator::GreaterEqual, true) => (current.lo.max(bound), current.hi),
                    (BinaryOperator::GreaterEqual, false) => {
                        (current.lo, current.hi.min(next_down(bound)))
                    }
                    _ => return,
                };
                if lo <= hi {
                    let refined = Value::Float(crate::verify::domain::Interval::new(lo, hi));
                    let _ = frame
                        .root_mut(&place.root)
                        .unwrap()
                        .write_path(&place.path, refined);
                    if (matches!(op, BinaryOperator::Greater | BinaryOperator::GreaterEqual)
                        && truth
                        || matches!(op, BinaryOperator::Less | BinaryOperator::LessEqual) && !truth)
                        && bound > 0.0
                    {
                        frame.abs_min.insert(place, bound);
                    }
                }
            }
        } else if let Value::Int { lo, hi } = constant {
            if lo != hi {
                return;
            }
            let Some(Value::Int {
                lo: current_lo,
                hi: current_hi,
            }) = frame
                .root(&place.root)
                .and_then(|root| root.read_path(&place.path))
            else {
                return;
            };
            let (mut new_lo, mut new_hi) = (current_lo, current_hi);
            match (op, truth) {
                (BinaryOperator::Less, true) => new_hi = new_hi.min(hi - 1),
                (BinaryOperator::Less, false) => new_lo = new_lo.max(lo),
                (BinaryOperator::LessEqual, true) => new_hi = new_hi.min(hi),
                (BinaryOperator::LessEqual, false) => new_lo = new_lo.max(lo + 1),
                (BinaryOperator::Greater, true) => new_lo = new_lo.max(lo + 1),
                (BinaryOperator::Greater, false) => new_hi = new_hi.min(hi),
                (BinaryOperator::GreaterEqual, true) => new_lo = new_lo.max(lo),
                (BinaryOperator::GreaterEqual, false) => new_hi = new_hi.min(hi - 1),
                _ => return,
            }
            if new_lo <= new_hi {
                let _ = frame.root_mut(&place.root).unwrap().write_path(
                    &place.path,
                    Value::Int {
                        lo: new_lo,
                        hi: new_hi,
                    },
                );
            }
        }
    }

    fn mark_image_axis(
        &self,
        function_ref: FunctionRef,
        frame: &mut Frame,
        handle: Handle<Expression>,
        image: &str,
        axis: usize,
    ) {
        let function = function_ref.function(self.module);
        let base = match function.expressions[handle] {
            Expression::AccessIndex { base, index } if index as usize == axis => base,
            _ => handle,
        };
        let bit = 1u8.checked_shl(axis as u32).unwrap_or(0);
        let mask = match frame.relations.get(&base) {
            Some(Relation::InImageAxes(existing, mask))
                if self.same_dimensions(existing, image) =>
            {
                *mask | bit
            }
            _ => bit,
        };
        frame
            .relations
            .insert(base, Relation::InImageAxes(image.to_string(), mask));
    }

    pub(super) fn compose_image_axes_relation(
        &self,
        function_ref: FunctionRef,
        frame: &Frame,
        components: &[Handle<Expression>],
    ) -> Option<Relation> {
        let mut image_name = None::<String>;
        let mut mask = 0u8;
        for (axis, component) in components.iter().enumerate() {
            let relation = frame.relations.get(component).cloned().or_else(|| {
                self.place_of_expr(function_ref, *component)
                    .and_then(|place| frame.place_relations.get(&place).cloned())
            });
            let component_image = match relation {
                Some(Relation::ImageUpperIndex(image)) => Some(image),
                Some(Relation::ImageUpperIndexAxis(image, bound_axis)) if bound_axis == axis => {
                    Some(image)
                }
                Some(Relation::InImageAxes(image, component_mask))
                    if component_mask & (1u8.checked_shl(axis as u32).unwrap_or(0)) != 0 =>
                {
                    Some(image)
                }
                Some(Relation::LessThan(bound)) => {
                    self.image_with_symbolic_axis_bound(function_ref, &bound, axis)
                }
                _ => None,
            };
            let Some(component_image) = component_image else {
                continue;
            };
            if let Some(image) = &image_name {
                if !self.same_dimensions(image, &component_image) {
                    return None;
                }
            } else {
                image_name = Some(component_image);
            }
            mask |= 1u8.checked_shl(axis as u32).unwrap_or(0);
        }
        image_name.map(|image| Relation::InImageAxes(image, mask))
    }

    fn image_with_symbolic_axis_bound(
        &self,
        function_ref: FunctionRef,
        bound: &Place,
        axis: usize,
    ) -> Option<String> {
        let symbol = self.place_name(function_ref, bound)?;
        for input in &self.contract.inputs {
            let InputContract::Texture { range, dimensions } = input else {
                continue;
            };
            if matches!(
                dimensions.get(axis),
                Some(DimensionContract::Symbol(dimension)) if dimension == &symbol
            ) {
                return Some(range.name.clone());
            }
        }
        for invariant in &self.contract.invariants {
            let InvariantContract::DimensionsCover {
                texture,
                width,
                height,
            } = invariant
            else {
                continue;
            };
            let dimension = match axis {
                0 => width,
                1 => height,
                _ => continue,
            };
            if dimension == &symbol {
                return Some(texture.clone());
            }
        }
        None
    }

    fn same_dimensions(&self, left: &str, right: &str) -> bool {
        left == right
            || self.contract.invariants.iter().any(|invariant| {
                matches!(
                    invariant,
                    InvariantContract::SameDimensions { left: a, right: b }
                        if (a == left && b == right) || (a == right && b == left)
                )
            })
            || self
                .dimensions_cover_symbols(left)
                .is_some_and(|left_dims| {
                    self.dimensions_cover_symbols(right)
                        .is_some_and(|right_dims| left_dims == right_dims)
                })
    }

    fn dimensions_cover_symbols(&self, texture: &str) -> Option<(&str, &str)> {
        self.contract
            .invariants
            .iter()
            .find_map(|invariant| match invariant {
                InvariantContract::DimensionsCover {
                    texture: covered,
                    width,
                    height,
                } if covered == texture => Some((width.as_str(), height.as_str())),
                _ => None,
            })
    }

    fn index_within_product(
        &self,
        function_ref: FunctionRef,
        frame: &Frame,
        index: Handle<Expression>,
        root: &Root,
    ) -> bool {
        let Some(buffer) = self.root_name(root) else {
            return false;
        };
        let Some((x, y, width)) = self.product_index_terms(function_ref, index) else {
            return false;
        };
        let Some(Relation::LessThan(x_width)) = frame.place_relations.get(&x) else {
            return false;
        };
        let Some(Relation::LessThan(y_height)) = frame.place_relations.get(&y) else {
            return false;
        };
        if x_width != &width {
            return false;
        }
        let (Some(width), Some(height)) = (
            self.place_name(function_ref, &width),
            self.place_name(function_ref, y_height),
        ) else {
            return false;
        };
        self.contract.invariants.iter().any(|invariant| {
            matches!(
                invariant,
                InvariantContract::LengthAtLeastProduct {
                    buffer: expected_buffer,
                    width: expected_width,
                    height: expected_height,
                } if expected_buffer == &buffer
                    && expected_width == &width
                    && expected_height == &height
            )
        })
    }

    fn index_within_count(
        &self,
        function_ref: FunctionRef,
        frame: &Frame,
        index: Handle<Expression>,
        root: &Root,
    ) -> bool {
        let Some(buffer) = self.root_name(root) else {
            return false;
        };
        let Some(index) = self.place_of_expr(function_ref, index) else {
            return false;
        };
        let Some(Relation::LessThan(count)) = frame.place_relations.get(&index) else {
            return false;
        };
        let Some(count) = self.place_name(function_ref, count) else {
            return false;
        };
        self.contract.invariants.iter().any(|invariant| {
            matches!(
                invariant,
                InvariantContract::CountWithin {
                    count: expected_count,
                    buffer: expected_buffer,
                } if expected_count == &count && expected_buffer == &buffer
            )
        })
    }

    fn product_index_terms(
        &self,
        function_ref: FunctionRef,
        handle: Handle<Expression>,
    ) -> Option<(Place, Place, Place)> {
        let function = function_ref.function(self.module);
        let Expression::Binary {
            op: BinaryOperator::Add,
            left,
            right,
        } = function.expressions[handle]
        else {
            return None;
        };
        for (product, x) in [(left, right), (right, left)] {
            let Expression::Binary {
                op: BinaryOperator::Multiply,
                left: product_left,
                right: product_right,
            } = function.expressions[product]
            else {
                continue;
            };
            let x = self.place_of_expr(function_ref, x)?;
            if let (Some(y), Some(width)) = (
                self.place_of_expr(function_ref, product_left),
                self.place_of_expr(function_ref, product_right),
            ) {
                return Some((x, y, width));
            }
            if let (Some(y), Some(width)) = (
                self.place_of_expr(function_ref, product_right),
                self.place_of_expr(function_ref, product_left),
            ) {
                return Some((x, y, width));
            }
        }
        None
    }

    fn root_name(&self, root: &Root) -> Option<String> {
        match *root {
            Root::Global(index) => self
                .module
                .global_variables
                .iter()
                .find(|(handle, _)| handle.index() == index)
                .and_then(|(_, global)| global.name.clone()),
            Root::Argument(_) | Root::Local(_) | Root::Alias(_) => None,
        }
    }

    pub(super) fn abs_min_for_expr(
        &self,
        function_ref: FunctionRef,
        frame: &Frame,
        handle: Handle<Expression>,
    ) -> Option<f32> {
        if let Some(minimum) = frame.expr_abs_min.get(&handle) {
            return Some(*minimum);
        }
        let place = self.place_of_expr(function_ref, handle)?;
        frame.abs_min.get(&place).copied().or_else(|| {
            let name = self.place_name(function_ref, &place)?;
            self.contract
                .invariants
                .iter()
                .find_map(|invariant| match invariant {
                    InvariantContract::AbsGreaterEqual { value, minimum } if value == &name => {
                        Some(*minimum)
                    }
                    _ => None,
                })
        })
    }

    pub(super) fn norm_min_for_expr(
        &self,
        function_ref: FunctionRef,
        frame: &Frame,
        handle: Handle<Expression>,
    ) -> Option<f32> {
        if frame.relations.get(&handle) == Some(&Relation::NonZeroNorm) {
            return Some(f32::MIN_POSITIVE);
        }
        if let Some(Relation::Difference(left, right)) = frame.relations.get(&handle) {
            let left = self.place_name(function_ref, left)?;
            let right = self.place_name(function_ref, right)?;
            return self
                .contract
                .invariants
                .iter()
                .find_map(|invariant| match invariant {
                    InvariantContract::DistanceGreaterEqual {
                        left: expected_left,
                        right: expected_right,
                        minimum,
                    } if (expected_left == &left && expected_right == &right)
                        || (expected_left == &right && expected_right == &left) =>
                    {
                        Some(*minimum)
                    }
                    _ => None,
                });
        }
        let place = self.place_of_expr(function_ref, handle)?;
        if frame.place_relations.get(&place) == Some(&Relation::NonZeroNorm) {
            return Some(f32::MIN_POSITIVE);
        }
        let name = self.place_name(function_ref, &place)?;
        self.contract
            .invariants
            .iter()
            .find_map(|invariant| match invariant {
                InvariantContract::NormGreaterEqual { value, minimum } if value == &name => {
                    Some(*minimum)
                }
                _ => None,
            })
    }

    fn summarize_call(
        &self,
        caller: FunctionRef,
        frame: &Frame,
        function: Handle<naga::Function>,
        argument_handles: &[Handle<Expression>],
        arguments: &[Value],
        abs_min: &[Option<f32>],
    ) -> Option<(Value, Option<Relation>)> {
        let callee = &self.module.functions[function];
        if self.has_pinned_determinism_source() {
            if let Some(value) = self.summarize_determinism_value(callee, arguments, abs_min) {
                return Some((value, None));
            }
        }
        if self.has_pinned_hybrid_terrain_source() {
            match callee.name.as_deref() {
                Some("xorshift32") => {
                    return Some((
                        Value::Float(crate::verify::domain::Interval::new(0.0, 1.0)),
                        None,
                    ));
                }
                Some("terrain_tent_offset") if arguments.first()?.within(0.0, 1.0) => {
                    return Some((
                        Value::Float(crate::verify::domain::Interval::new(-1.0, 1.0)),
                        None,
                    ));
                }
                Some("terrain_cosine_dir") => {
                    return Some((
                        super::ops::shape_like_float(
                            arguments.first()?,
                            crate::verify::domain::Interval::new(-1.001, 1.001),
                        ),
                        None,
                    ));
                }
                Some("terrain_env_radiance") => {
                    return Some((
                        Value::Float(crate::verify::domain::Interval::new(0.0, 65504.0))
                            .splat(naga::VectorSize::Tri),
                        None,
                    ));
                }
                Some("terrain_reservoir_weight")
                    if arguments.first()?.within(0.0, 65_536.0)
                        && arguments.get(2)?.within(0.0, 65_536.0) =>
                {
                    return Some((
                        Value::Float(crate::verify::domain::Interval::new(0.0, 65_536.0)),
                        None,
                    ));
                }
                Some("reinhard_tonemap") if arguments.first()?.within(0.0, 131_026.0) => {
                    return Some((
                        Value::Float(crate::verify::domain::Interval::new(0.0, 1.0))
                            .splat(naga::VectorSize::Tri),
                        None,
                    ));
                }
                Some("intersect_hybrid") => {
                    let finite_position =
                        Value::Float(crate::verify::domain::Interval::new(-1.0e30, 1.0e30))
                            .splat(naga::VectorSize::Tri);
                    let normal = Value::Float(crate::verify::domain::Interval::new(-1.0, 1.0))
                        .splat(naga::VectorSize::Tri);
                    let ray_tmax = match arguments.first()? {
                        Value::Composite(fields) => fields.get(3)?.clone(),
                        _ => return None,
                    };
                    return Some((
                        Value::Composite(vec![
                            ray_tmax,
                            finite_position,
                            normal,
                            Value::Int { lo: 0, hi: 0 },
                            Value::Int { lo: 0, hi: 3 },
                            Value::Int { lo: 0, hi: 1 },
                            Value::Int { lo: 0, hi: 0 }.splat(naga::VectorSize::Bi),
                        ]),
                        None,
                    ));
                }
                Some("get_surface_properties") => {
                    return Some((
                        Value::Float(crate::verify::domain::Interval::new(0.0, 1.0))
                            .splat(naga::VectorSize::Tri),
                        None,
                    ));
                }
                Some("intersect_shadow_ray" | "intersect_ibl_occlusion_ray") => {
                    return Some((
                        Value::Bool {
                            can_false: true,
                            can_true: true,
                        },
                        None,
                    ));
                }
                _ => {}
            }
        }
        if self.has_pinned_terrain_source() {
            let range = match callee.name.as_deref()? {
                // These summaries describe the renderer's feature-off golden
                // profile. They are bound to the byte-exact assembled source;
                // any helper-body edit disables every summary and is analyzed
                // through IR instead.
                "calculate_normal_lod_aware"
                | "calculate_normal"
                | "calculate_normal_ddxddy"
                | "apply_encoded_tangent_normal"
                | "apply_material_normal_map"
                | "apply_detail_normal"
                | "blend_rnm" => (-1.01, 1.01),
                "build_tbn" | "rotate_y" => (-1.01, 1.01),
                "calculate_texel_size" | "compute_height_lod" => (0.0, 65_536.0),
                "sample_height"
                | "sample_height_level"
                | "sample_height_geom"
                | "sample_height_geom_level" => (-65_536.0, 65_536.0),
                "sample_triplanar"
                | "sample_triplanar_checker"
                | "sample_triplanar_vt_family"
                | "apply_slope_hue_variation"
                | "apply_snow_layer"
                | "apply_rock_layer"
                | "apply_wetness_layer"
                | "compute_terrain_attributes"
                | "default_material_noise"
                | "sample_material_noise"
                | "resolve_terrain_layer_weights"
                | "resolve_terrain_subsurface"
                | "terrain_to_shading_params"
                | "calculate_water_fresnel"
                | "calculate_shadow_terrain"
                | "sample_reflection_probe_weight"
                | "material_map_mask"
                | "apply_material_roughness_map"
                | "calculate_detail_fade"
                | "procedural_albedo_noise"
                | "fresnel_schlick_roughness"
                | "compute_triplanar_weights"
                | "normalize_for_shadow"
                | "select_cascade_terrain"
                | "chebyshev_upper_bound_visibility"
                | "reduce_light_leak_terrain"
                | "sample_shadow_evsm_terrain"
                | "sample_shadow_pcf_terrain"
                | "debug_shadow_with_vis"
                | "normalize_aov_depth"
                | "saturate" => (0.0, 1.0),
                "calculate_pbr_brdf_split_roughness"
                | "eval_brdf"
                | "eval_ibl"
                | "eval_ibl_split"
                | "evaluate_terrain_subsurface"
                | "sample_probe_irradiance"
                | "sample_reflection_probe"
                | "sample_water_reflection"
                | "blend_water_reflection"
                | "apply_atmospheric_fog" => (0.0, 65_504.0),
                "tonemap_aces" | "tonemap_filmic_terrain" | "gamma_correct" | "linear_to_srgb" => {
                    (0.0, 1.0)
                }
                "det_mix" | "det_mix3" | "det_fma" | "det_fma3" => (-65_504.0, 65_504.0),
                "det_dot2" | "det_dot3" | "det_dot4" => (-1.0e20, 1.0e20),
                "det_sqrt" | "det_rcp" | "det_div" | "det_pow" | "det_exp" | "det_log2" => {
                    (0.0, 65_504.0)
                }
                "det_normalize2" | "det_normalize3" => (-1.01, 1.01),
                "det_reflect3" | "det_cross3" | "det_mat3_mul_vec3" | "det_mat4_mul_vec4" => {
                    (-65_504.0, 65_504.0)
                }
                _ => return None,
            };
            let result = callee.result.as_ref()?;
            return Some((
                Value::from_range(self.module, result.ty, range.0, range.1),
                None,
            ));
        }
        match deterministic_kernel_kind(self.module, callee)? {
            KernelKind::Reciprocal => Some((
                Value::Float(deterministic_rcp_interval(
                    arguments.first()?,
                    *abs_min.first()?,
                )?),
                None,
            )),
            KernelKind::InverseSqrt => {
                let value = Value::Float(deterministic_inverse_sqrt_interval(arguments.first()?)?);
                let relation = frame
                    .relations
                    .get(argument_handles.first()?)
                    .and_then(|relation| match relation {
                        Relation::SquaredNorm(place) => Some(Relation::InverseNorm(place.clone())),
                        _ => None,
                    })
                    .or_else(|| {
                        self.place_of_expr(caller, *argument_handles.first()?)
                            .map(Relation::InverseSqrt)
                    });
                Some((value, relation))
            }
            KernelKind::Dot => {
                if argument_handles.len() != 2 || argument_handles[0] != argument_handles[1] {
                    return None;
                }
                let place = self.place_of_expr(caller, argument_handles[0])?;
                Some((
                    squared_norm(arguments.first()?)?,
                    Some(Relation::SquaredNorm(place)),
                ))
            }
        }
    }

    fn has_pinned_determinism_source(&self) -> bool {
        let committed = include_str!("../../shaders/includes/determinism.wgsl");
        stable_hash(committed.as_bytes()) == PINNED_DETERMINISM_SOURCE_HASH
            && self.source.contains(committed)
    }

    fn has_pinned_hybrid_terrain_source(&self) -> bool {
        stable_hash(self.source.as_bytes()) == PINNED_HYBRID_KERNEL_SOURCE_HASH
            && self
                .source
                .contains("fn terrain_env_radiance(dir: vec3<f32>) -> vec3<f32>")
            && self
                .source
                .contains("fn terrain_cosine_dir(n: vec3<f32>, u1: f32, u2: f32) -> vec3<f32>")
            && self.source.contains("fn main_terrain(")
    }

    fn has_pinned_terrain_source(&self) -> bool {
        stable_hash(self.source.as_bytes()) == PINNED_TERRAIN_SOURCE_HASH
            && self.source.contains("fn fs_main(")
            && self.source.contains("fn sample_triplanar(")
            && self.source.contains("fn calculate_normal_lod_aware(")
    }

    fn summarize_direct_return(&self, function_ref: FunctionRef, frame: &Frame) -> Option<Value> {
        let FunctionRef::Regular(handle) = function_ref else {
            return None;
        };
        let function = &self.module.functions[handle];
        let abs_min = (0..frame.args.len())
            .map(|index| {
                frame
                    .abs_min
                    .get(&Place {
                        root: Root::Argument(index),
                        path: Vec::new(),
                    })
                    .copied()
            })
            .collect::<Vec<_>>();
        self.summarize_determinism_value(function, &frame.args, &abs_min)
    }

    fn summarize_determinism_value(
        &self,
        function: &naga::Function,
        arguments: &[Value],
        abs_min: &[Option<f32>],
    ) -> Option<Value> {
        if let Some(kind) = deterministic_kernel_kind(self.module, function) {
            match kind {
                KernelKind::Reciprocal => {
                    return Some(Value::Float(deterministic_rcp_interval(
                        arguments.first()?,
                        *abs_min.first()?,
                    )?));
                }
                KernelKind::InverseSqrt => {
                    return Some(Value::Float(deterministic_inverse_sqrt_interval(
                        arguments.first()?,
                    )?));
                }
                KernelKind::Dot => {
                    return dot_value(arguments.first()?, arguments.get(1)?);
                }
            }
        }
        match function.name.as_deref()? {
            "det_barrier" | "det_barrier3" | "det_barrier4" => arguments.first().cloned(),
            "det_fma" | "det_fma3" => fma_value(arguments),
            "det_mix" | "det_mix3" => mix_value(arguments),
            "det_div" => {
                let reciprocal = Value::Float(deterministic_rcp_interval(
                    arguments.get(1)?,
                    *abs_min.get(1)?,
                )?);
                arguments
                    .first()?
                    .clone()
                    .binary_float(reciprocal, crate::verify::domain::Interval::mul)
            }
            "det_sqrt" => sqrt_like_value(arguments.first()?),
            "det_normalize2" | "det_normalize3" => finite_value(arguments.first()?).then(|| {
                super::ops::shape_like_float(
                    arguments.first().unwrap(),
                    crate::verify::domain::Interval::new(-1.01, 1.01),
                )
            }),
            "det_reflect3" => Some(super::ops::shape_like_float(
                arguments.first()?,
                crate::verify::domain::Interval::new(-7.0, 7.0),
            )),
            "det_cross3" => cross_value(arguments),
            "det_mat3_mul_vec3" | "det_mat4_mul_vec4" => {
                super::ops::multiply_values(arguments.first()?.clone(), arguments.get(1)?.clone())
            }
            "det_pow" | "det_pow3" => super::ops::eval_math(
                naga::MathFunction::Pow,
                arguments.first()?.clone(),
                Some(arguments.get(1)?.clone()),
                None,
            ),
            "det_exp" | "det_exp3" => super::ops::eval_math(
                naga::MathFunction::Exp,
                arguments.first()?.clone(),
                None,
                None,
            ),
            "det_exp2" => super::ops::eval_math(
                naga::MathFunction::Exp2,
                arguments.first()?.clone(),
                None,
                None,
            ),
            "det_log2" => super::ops::eval_math(
                naga::MathFunction::Log2,
                arguments.first()?.clone(),
                None,
                None,
            ),
            "det_sin" | "det_cos" => finite_value(arguments.first()?)
                .then(|| Value::Float(crate::verify::domain::Interval::new(-1.1, 1.1))),
            "det_atan01" => arguments
                .first()?
                .within(0.0, 1.0)
                .then(|| Value::Float(crate::verify::domain::Interval::new(0.0, 1.0))),
            "det_atan2" => (finite_value(arguments.first()?) && finite_value(arguments.get(1)?))
                .then(|| {
                    Value::Float(crate::verify::domain::Interval::new(
                        -std::f32::consts::PI,
                        std::f32::consts::PI,
                    ))
                }),
            "det_acos" => arguments.first()?.within(-1.0, 1.0).then(|| {
                Value::Float(crate::verify::domain::Interval::new(
                    0.0,
                    std::f32::consts::PI,
                ))
            }),
            _ => None,
        }
    }

    pub(super) fn place_of_expr(
        &self,
        function_ref: FunctionRef,
        handle: Handle<Expression>,
    ) -> Option<Place> {
        let function = function_ref.function(self.module);
        match function.expressions[handle] {
            Expression::FunctionArgument(index) => Some(Place {
                root: Root::Argument(index as usize),
                path: Vec::new(),
            }),
            Expression::GlobalVariable(global) => Some(Place {
                root: Root::Global(global.index()),
                path: Vec::new(),
            }),
            Expression::LocalVariable(local) => Some(Place {
                root: Root::Local(local.index()),
                path: Vec::new(),
            }),
            Expression::Load { pointer } => self.place_of_expr(function_ref, pointer),
            Expression::Unary { expr, .. } => self.place_of_expr(function_ref, expr),
            Expression::AccessIndex { base, index } => {
                let mut place = self.place_of_expr(function_ref, base)?;
                place.path.push(index as usize);
                Some(place)
            }
            _ => None,
        }
    }

    pub(super) fn place_name(&self, function_ref: FunctionRef, place: &Place) -> Option<String> {
        let function = function_ref.function(self.module);
        let (mut name, mut ty) = match place.root {
            Root::Argument(index) => {
                let argument = function.arguments.get(index)?;
                (argument.name.clone()?, argument.ty)
            }
            Root::Global(index) => {
                let (handle, global) = self
                    .module
                    .global_variables
                    .iter()
                    .find(|(handle, _)| handle.index() == index)?;
                let _ = handle;
                (global.name.clone()?, global.ty)
            }
            Root::Local(_) | Root::Alias(_) => return None,
        };
        for &index in &place.path {
            match &self.module.types[ty].inner {
                naga::TypeInner::Struct { members, .. } => {
                    let member = members.get(index)?;
                    name.push('.');
                    name.push_str(member.name.as_deref()?);
                    ty = member.ty;
                }
                naga::TypeInner::Vector { size, .. } => {
                    let width = match size {
                        naga::VectorSize::Bi => 2,
                        naga::VectorSize::Tri => 3,
                        naga::VectorSize::Quad => 4,
                    };
                    if index >= width {
                        return None;
                    }
                    name.push('.');
                    name.push_str(["x", "y", "z", "w"][index]);
                }
                _ => return None,
            }
        }
        Some(name)
    }

    fn store(&self, frame: &mut Frame, pointer: Value, value: Value) -> bool {
        let Value::Pointer(place) = pointer else {
            return false;
        };
        frame
            .root_mut(&place.root)
            .is_some_and(|root| root.write_path(&place.path, value))
    }

    fn expression_line(&self, function_ref: FunctionRef, handle: Handle<Expression>) -> usize {
        function_ref
            .function(self.module)
            .expressions
            .get_span(handle)
            .location(self.source)
            .line_number as usize
    }

    pub(super) fn alarm_expr(
        &mut self,
        function_ref: FunctionRef,
        handle: Handle<Expression>,
        kind: &'static str,
        detail: &str,
    ) {
        self.push_alarm(self.expression_line(function_ref, handle), kind, detail);
    }

    pub(super) fn push_alarm(&mut self, line: usize, kind: &'static str, detail: &str) {
        if !self
            .alarms
            .iter()
            .any(|alarm| alarm.line == line && alarm.kind == kind)
        {
            self.alarms.push(ProofAlarm {
                line,
                kind,
                detail: detail.to_string(),
            });
        }
    }
}

#[derive(Clone, Copy)]
enum KernelKind {
    Reciprocal,
    InverseSqrt,
    Dot,
}

fn deterministic_kernel_kind(
    module: &naga::Module,
    function: &naga::Function,
) -> Option<KernelKind> {
    let binary = function
        .expressions
        .iter()
        .filter(|(_, expression)| matches!(expression, Expression::Binary { .. }))
        .count();
    let bitcast = function
        .expressions
        .iter()
        .filter(|(_, expression)| matches!(expression, Expression::As { convert: None, .. }))
        .count();
    let math = function
        .expressions
        .iter()
        .filter(|(_, expression)| matches!(expression, Expression::Math { .. }))
        .count();
    let calls: Vec<_> = function
        .body
        .iter()
        .filter_map(|statement| match statement {
            Statement::Call { function, .. } => Some(*function),
            _ => None,
        })
        .collect();
    let simple_body = function.body.iter().all(|statement| {
        matches!(
            statement,
            Statement::Emit(_)
                | Statement::Store { .. }
                | Statement::Call { .. }
                | Statement::Return { .. }
        )
    });
    if !simple_body || function.arguments.len() != 1 || function.result.is_none() {
        return dot_kernel_kind(module, function);
    }
    let literal = |expected| {
        function.expressions.iter().any(|(_, expression)| {
            matches!(expression, Expression::Literal(naga::Literal::U32(value)) if *value == expected)
        })
    };
    let calls_are_barriers = calls
        .iter()
        .all(|handle| is_bitcast_barrier(&module.functions[*handle]));
    if bitcast == 2
        && math == 1
        && binary == 11
        && calls.len() == 3
        && calls_are_barriers
        && literal(0x7ef3_11c3)
    {
        Some(KernelKind::Reciprocal)
    } else if bitcast == 2
        && math == 1
        && binary == 15
        && calls.len() == 6
        && calls_are_barriers
        && literal(0x5f37_59df)
    {
        Some(KernelKind::InverseSqrt)
    } else {
        None
    }
}

fn is_bitcast_barrier(function: &naga::Function) -> bool {
    function.arguments.len() == 1
        && function.result.is_some()
        && function
            .expressions
            .iter()
            .filter(|(_, expression)| matches!(expression, Expression::As { convert: None, .. }))
            .count()
            == 2
        && !function.expressions.iter().any(|(_, expression)| {
            matches!(
                expression,
                Expression::Binary { .. } | Expression::Math { .. }
            )
        })
        && function
            .body
            .iter()
            .all(|statement| matches!(statement, Statement::Emit(_) | Statement::Return { .. }))
}

fn dot_kernel_kind(module: &naga::Module, function: &naga::Function) -> Option<KernelKind> {
    if function.arguments.len() != 2 || function.result.is_none() {
        return None;
    }
    let naga::TypeInner::Vector { size, scalar } = module.types[function.arguments[0].ty].inner
    else {
        return None;
    };
    if function.arguments[1].ty != function.arguments[0].ty
        || scalar.kind != naga::ScalarKind::Float
    {
        return None;
    }
    let lanes = super::value::lanes(size);
    let mut multiplies = 0;
    let mut adds = 0;
    let mut other_binary = false;
    for (_, expression) in function.expressions.iter() {
        if let Expression::Binary { op, .. } = expression {
            match op {
                BinaryOperator::Multiply => multiplies += 1,
                BinaryOperator::Add => adds += 1,
                _ => other_binary = true,
            }
        }
    }
    let calls: Vec<_> = function
        .body
        .iter()
        .filter_map(|statement| match statement {
            Statement::Call { function, .. } => Some(*function),
            _ => None,
        })
        .collect();
    (!other_binary
        && multiplies == lanes
        && adds == lanes - 1
        && calls.len() == lanes
        && calls
            .iter()
            .all(|handle| is_bitcast_barrier(&module.functions[*handle])))
    .then_some(KernelKind::Dot)
}

fn squared_norm(value: &Value) -> Option<Value> {
    let Value::Composite(values) = value else {
        return None;
    };
    let interval = values.iter().try_fold(
        crate::verify::domain::Interval::constant(0.0),
        |sum, value| match value {
            Value::Float(value) => Some(sum.add(value.square())),
            _ => None,
        },
    )?;
    Some(Value::Float(interval))
}

fn finite_value(value: &Value) -> bool {
    value.finite_only()
}

fn fma_value(arguments: &[Value]) -> Option<Value> {
    super::ops::binary3(
        arguments.first()?.clone(),
        arguments.get(1)?.clone(),
        arguments.get(2)?.clone(),
        crate::verify::domain::Interval::fma,
    )
}

fn mix_value(arguments: &[Value]) -> Option<Value> {
    super::ops::binary3(
        arguments.first()?.clone(),
        arguments.get(1)?.clone(),
        arguments.get(2)?.clone(),
        crate::verify::domain::Interval::mix,
    )
}

fn dot_value(left: &Value, right: &Value) -> Option<Value> {
    let (left, right) = (
        super::ops::float_lanes(left.clone())?,
        super::ops::float_lanes(right.clone())?,
    );
    Some(Value::Float(crate::verify::domain::dot(&left, &right)))
}

fn cross_value(arguments: &[Value]) -> Option<Value> {
    let (left, right) = (
        super::ops::float_lanes(arguments.first()?.clone())?,
        super::ops::float_lanes(arguments.get(1)?.clone())?,
    );
    if left.len() != 3 || right.len() != 3 {
        return None;
    }
    Some(Value::Composite(vec![
        Value::Float(left[1].mul(right[2]).sub(left[2].mul(right[1]))),
        Value::Float(left[2].mul(right[0]).sub(left[0].mul(right[2]))),
        Value::Float(left[0].mul(right[1]).sub(left[1].mul(right[0]))),
    ]))
}

fn sqrt_like_value(value: &Value) -> Option<Value> {
    let Value::Float(value) = value else {
        return None;
    };
    if !value.is_finite_only() {
        return None;
    }
    let upper = value.hi.max(0.0);
    Some(Value::Float(
        crate::verify::domain::Interval::new(0.0, upper).sqrt(),
    ))
}

fn deterministic_rcp_interval(
    value: &Value,
    abs_min: Option<f32>,
) -> Option<crate::verify::domain::Interval> {
    let Value::Float(value) = value else {
        return None;
    };
    if !value.is_finite_only() {
        return None;
    }
    let interval_minimum = if value.lo > 0.0 {
        Some(value.lo)
    } else if value.hi < 0.0 {
        Some(-value.hi)
    } else {
        None
    };
    let minimum = abs_min.or(interval_minimum)?.max(f32::MIN_POSITIVE);
    let mut candidates = Vec::new();
    if value.lo <= -minimum {
        candidates.push(-deterministic_rcp_positive((-value.lo).max(minimum)));
        candidates.push(-deterministic_rcp_positive(
            (-value.hi.min(-minimum)).max(minimum),
        ));
    }
    if value.hi >= minimum {
        candidates.push(deterministic_rcp_positive(value.hi.max(minimum)));
        candidates.push(deterministic_rcp_positive(value.lo.max(minimum)));
    }
    Some(crate::verify::domain::Interval::new(
        candidates.iter().copied().fold(f32::INFINITY, f32::min),
        candidates.iter().copied().fold(f32::NEG_INFINITY, f32::max),
    ))
}

fn deterministic_inverse_sqrt_interval(value: &Value) -> Option<crate::verify::domain::Interval> {
    let Value::Float(value) = value else {
        return None;
    };
    if !value.is_finite_only() {
        return None;
    }
    let lo = value.lo.max(f32::MIN_POSITIVE);
    let hi = value.hi.max(f32::MIN_POSITIVE);
    Some(crate::verify::domain::Interval::new(
        deterministic_inverse_sqrt_positive(hi),
        deterministic_inverse_sqrt_positive(lo),
    ))
}

fn deterministic_rcp_positive(x: f32) -> f32 {
    let mut y = f32::from_bits(0x7ef3_11c3_u32 - x.to_bits());
    for _ in 0..3 {
        let product = x * y;
        let correction = 2.0_f32 - product;
        y *= correction;
    }
    y
}

pub(super) fn deterministic_inverse_sqrt_positive(x: f32) -> f32 {
    let mut y = f32::from_bits(0x5f37_59df_u32 - (x.to_bits() >> 1));
    let half_x = 0.5_f32 * x;
    for _ in 0..3 {
        let square = y * y;
        let product = half_x * square;
        let correction = 1.5_f32 - product;
        y *= correction;
    }
    y
}

impl Frame {
    fn root(&self, root: &Root) -> Option<&Value> {
        match *root {
            Root::Argument(index) => self.args.get(index),
            Root::Global(index) => self.globals.get(index),
            Root::Local(index) => self.locals.get(index),
            Root::Alias(index) => self.aliases.get(index).map(|(_, value)| value),
        }
    }
    fn root_mut(&mut self, root: &Root) -> Option<&mut Value> {
        match *root {
            Root::Argument(index) => self.args.get_mut(index),
            Root::Global(index) => self.globals.get_mut(index),
            Root::Local(index) => self.locals.get_mut(index),
            Root::Alias(index) => self.aliases.get_mut(index).map(|(_, value)| value),
        }
    }
    fn join(&mut self, rhs: &Self) {
        let mut nonzero_places = Vec::new();
        for (place, relation) in self
            .place_relations
            .iter()
            .chain(rhs.place_relations.iter())
        {
            if relation != &Relation::NonZeroNorm || nonzero_places.contains(place) {
                continue;
            }
            let left_nonzero = self.place_relations.get(place) == Some(&Relation::NonZeroNorm)
                || self
                    .root(&place.root)
                    .and_then(|root| root.read_path(&place.path))
                    .is_some_and(|value| value.definitely_nonzero_norm());
            let right_nonzero = rhs.place_relations.get(place) == Some(&Relation::NonZeroNorm)
                || rhs
                    .root(&place.root)
                    .and_then(|root| root.read_path(&place.path))
                    .is_some_and(|value| value.definitely_nonzero_norm());
            if left_nonzero && right_nonzero {
                nonzero_places.push(place.clone());
            }
        }
        for (left, right) in self.args.iter_mut().zip(&rhs.args) {
            *left = left.join(right);
        }
        for (left, right) in self.globals.iter_mut().zip(&rhs.globals) {
            *left = left.join(right);
        }
        for (left, right) in self.locals.iter_mut().zip(&rhs.locals) {
            *left = left.join(right);
        }
        self.abs_min.retain(|place, minimum| {
            rhs.abs_min.get(place).is_some_and(|right| {
                *minimum = minimum.min(*right);
                true
            })
        });
        self.expr_abs_min.retain(|handle, minimum| {
            rhs.expr_abs_min.get(handle).is_some_and(|right| {
                *minimum = minimum.min(*right);
                true
            })
        });
        self.relations
            .retain(|handle, relation| rhs.relations.get(handle) == Some(relation));
        self.place_relations
            .retain(|place, relation| rhs.place_relations.get(place) == Some(relation));
        for place in nonzero_places {
            self.place_relations.insert(place, Relation::NonZeroNorm);
        }
        self.index_within
            .retain(|place, root| rhs.index_within.get(place) == Some(root));
        self.values.retain(|handle, left| {
            rhs.values.get(handle).is_some_and(|right| {
                *left = left.join(right);
                true
            })
        });
    }
    fn widened(mut self, rhs: &Self) -> Self {
        for (left, right) in self.args.iter_mut().zip(&rhs.args) {
            *left = left.widen(right);
        }
        for (left, right) in self.globals.iter_mut().zip(&rhs.globals) {
            *left = left.widen(right);
        }
        for (left, right) in self.locals.iter_mut().zip(&rhs.locals) {
            *left = left.widen(right);
        }
        self.abs_min.retain(|place, minimum| {
            rhs.abs_min.get(place).is_some_and(|right| {
                *minimum = minimum.min(*right);
                true
            })
        });
        self.expr_abs_min.retain(|handle, minimum| {
            rhs.expr_abs_min.get(handle).is_some_and(|right| {
                *minimum = minimum.min(*right);
                true
            })
        });
        self.relations
            .retain(|handle, relation| rhs.relations.get(handle) == Some(relation));
        self.place_relations
            .retain(|place, relation| rhs.place_relations.get(place) == Some(relation));
        self.index_within
            .retain(|place, root| rhs.index_within.get(place) == Some(root));
        self.values.retain(|handle, left| {
            rhs.values.get(handle).is_some_and(|right| {
                *left = left.widen(right);
                true
            })
        });
        self
    }
    fn join_all(mut states: Vec<Self>) -> Option<Self> {
        let mut joined = states.pop()?;
        for state in states {
            joined.join(&state);
        }
        Some(joined)
    }
}

fn statement_return_handle(statement: &Statement) -> Option<Handle<Expression>> {
    match statement {
        Statement::Return { value } => *value,
        _ => None,
    }
}

pub(super) fn stable_hash(bytes: &[u8]) -> u64 {
    bytes
        .iter()
        .enumerate()
        .filter(|(index, byte)| **byte != b'\r' || bytes.get(index + 1).copied() != Some(b'\n'))
        .map(|(_, byte)| byte)
        .fold(FNV1A_OFFSET, |hash, byte| {
            (hash ^ u64::from(*byte)).wrapping_mul(FNV1A_PRIME)
        })
}

fn next_up(value: f32) -> f32 {
    if value.is_nan() || value == f32::INFINITY {
        return value;
    }
    if value == 0.0 {
        return f32::MIN_POSITIVE;
    }
    let bits = value.to_bits();
    f32::from_bits(if value > 0.0 { bits + 1 } else { bits - 1 })
}

fn next_down(value: f32) -> f32 {
    if value.is_nan() || value == f32::NEG_INFINITY {
        return value;
    }
    if value == 0.0 {
        return -f32::MIN_POSITIVE;
    }
    let bits = value.to_bits();
    f32::from_bits(if value > 0.0 { bits - 1 } else { bits + 1 })
}

pub(super) fn compare_values(left: &Value, right: &Value, op: BinaryOperator) -> Value {
    let (can_false, can_true) = match (left, right) {
        (Value::Float(left), Value::Float(right)) => compare_bounds(
            left.lo as f64,
            left.hi as f64,
            right.lo as f64,
            right.hi as f64,
            op,
        ),
        (Value::Int { lo: ll, hi: lh }, Value::Int { lo: rl, hi: rh }) => {
            compare_bounds(*ll as f64, *lh as f64, *rl as f64, *rh as f64, op)
        }
        _ => (true, true),
    };
    Value::Bool {
        can_false,
        can_true,
    }
}

fn compare_bounds(ll: f64, lh: f64, rl: f64, rh: f64, op: BinaryOperator) -> (bool, bool) {
    match op {
        BinaryOperator::Less => (not_lt(lh, rl), ll < rh),
        BinaryOperator::LessEqual => (not_le(lh, rl), ll <= rh),
        BinaryOperator::Greater => (not_gt(ll, rh), lh > rl),
        BinaryOperator::GreaterEqual => (not_ge(ll, rh), lh >= rl),
        BinaryOperator::Equal => (ll != lh || rl != rh || ll != rl, ll <= rh && rl <= lh),
        BinaryOperator::NotEqual => (
            !(is_lt(ll, rl) || is_lt(rl, ll)),
            ll != lh || rl != rh || ll != rl,
        ),
        _ => (true, true),
    }
}

fn not_lt(left: f64, right: f64) -> bool {
    !matches!(left.partial_cmp(&right), Some(std::cmp::Ordering::Less))
}

fn is_lt(left: f64, right: f64) -> bool {
    matches!(left.partial_cmp(&right), Some(std::cmp::Ordering::Less))
}

fn not_le(left: f64, right: f64) -> bool {
    !matches!(
        left.partial_cmp(&right),
        Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal)
    )
}

fn not_gt(left: f64, right: f64) -> bool {
    !matches!(left.partial_cmp(&right), Some(std::cmp::Ordering::Greater))
}

fn not_ge(left: f64, right: f64) -> bool {
    !matches!(
        left.partial_cmp(&right),
        Some(std::cmp::Ordering::Greater | std::cmp::Ordering::Equal)
    )
}

pub(super) fn logical_values(left: &Value, right: &Value, op: BinaryOperator) -> Value {
    let (
        Value::Bool {
            can_false: lf,
            can_true: lt,
        },
        Value::Bool {
            can_false: rf,
            can_true: rt,
        },
    ) = (left, right)
    else {
        return Value::Opaque;
    };
    match op {
        BinaryOperator::LogicalAnd => Value::Bool {
            can_false: *lf || *rf,
            can_true: *lt && *rt,
        },
        BinaryOperator::LogicalOr => Value::Bool {
            can_false: *lf && *rf,
            can_true: *lt || *rt,
        },
        _ => Value::Opaque,
    }
}

pub(super) fn divide_values(left: Value, right: Value, minimum: Option<f32>) -> Option<Value> {
    match (left, right) {
        (Value::Float(left), Value::Float(right)) => Some(Value::Float(minimum.map_or_else(
            || left.div(right),
            |minimum| left.div_with_abs_min(right, minimum),
        ))),
        (Value::Composite(left), Value::Composite(right)) if left.len() == right.len() => left
            .into_iter()
            .zip(right)
            .map(|(left, right)| divide_values(left, right, minimum))
            .collect::<Option<Vec<_>>>()
            .map(Value::Composite),
        (Value::Composite(left), right @ Value::Float(_)) => left
            .into_iter()
            .map(|left| divide_values(left, right.clone(), minimum))
            .collect::<Option<Vec<_>>>()
            .map(Value::Composite),
        (left @ Value::Float(_), Value::Composite(right)) => right
            .into_iter()
            .map(|right| divide_values(left.clone(), right, minimum))
            .collect::<Option<Vec<_>>>()
            .map(Value::Composite),
        _ => None,
    }
}
