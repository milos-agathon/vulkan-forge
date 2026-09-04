use super::engine::{
    compare_values, divide_values, logical_values, Evaluator, Frame, FunctionRef, Relation,
};
use super::ops::*;
use super::value::{self, Place, Root, Value};
use crate::verify::contract::InvariantContract;
use crate::verify::domain::Interval;
use naga::{BinaryOperator, Expression, Handle, Literal, UnaryOperator};

impl Evaluator<'_> {
    pub(super) fn eval_expr(
        &mut self,
        function_ref: FunctionRef,
        frame: &mut Frame,
        handle: Handle<Expression>,
    ) -> anyhow::Result<Value> {
        if let Some(value) = frame.values.get(&handle) {
            return Ok(value.clone());
        }
        let function = function_ref.function(self.module);
        let value = match &function.expressions[handle] {
            Expression::Literal(Literal::F32(value)) => Value::Float(Interval::constant(*value)),
            Expression::Literal(Literal::AbstractFloat(value)) => {
                Value::Float(Interval::constant(*value as f32))
            }
            Expression::Literal(Literal::U32(value)) => Value::Int {
                lo: *value as i64,
                hi: *value as i64,
            },
            Expression::Literal(Literal::I32(value)) => Value::Int {
                lo: *value as i64,
                hi: *value as i64,
            },
            Expression::Literal(Literal::Bool(value)) => Value::Bool {
                can_false: !*value,
                can_true: *value,
            },
            Expression::Constant(constant) => {
                self.eval_const(self.module.constants[*constant].init)
            }
            Expression::FunctionArgument(index) => frame.args[*index as usize].clone(),
            Expression::LocalVariable(local) => Value::Pointer(Place {
                root: Root::Local(local.index()),
                path: Vec::new(),
            }),
            Expression::GlobalVariable(global) => {
                if self.module.global_variables[*global].space == naga::AddressSpace::Handle {
                    frame.globals[global.index()].clone()
                } else {
                    Value::Pointer(Place {
                        root: Root::Global(global.index()),
                        path: Vec::new(),
                    })
                }
            }
            Expression::Load { pointer } => {
                let pointer = self.eval_expr(function_ref, frame, *pointer)?;
                if let Value::Pointer(place) = &pointer {
                    if let Some(relation) = frame.place_relations.get(place).cloned() {
                        frame.relations.insert(handle, relation);
                    }
                }
                self.load(frame, pointer).unwrap_or(Value::Opaque)
            }
            Expression::AccessIndex { base, index } => {
                let value = self
                    .eval_expr(function_ref, frame, *base)?
                    .access_index(*index as usize)
                    .unwrap_or(Value::Opaque);
                if let Some(Relation::ImageDimensions(name)) = frame.relations.get(base).cloned() {
                    frame
                        .relations
                        .insert(handle, Relation::ImageDimension(name, *index as usize));
                }
                if let Some(relation @ Relation::BudgetTerms { .. }) =
                    frame.relations.get(base).cloned()
                {
                    frame.relations.insert(handle, relation);
                }
                value
            }
            Expression::Access { base, index } => {
                let base_value = self.eval_expr(function_ref, frame, *base)?;
                let index_value = self.eval_expr(function_ref, frame, *index)?;
                self.dynamic_access(function_ref, frame, handle, base_value, *index, index_value)
            }
            Expression::Splat { size, value } => {
                self.eval_expr(function_ref, frame, *value)?.splat(*size)
            }
            Expression::Compose { ty, components } => {
                let values = components
                    .iter()
                    .map(|component| self.eval_expr(function_ref, frame, *component))
                    .collect::<anyhow::Result<Vec<_>>>()?;
                if matches!(self.module.types[*ty].inner, naga::TypeInner::Vector { .. }) {
                    if let Some(relation) =
                        self.compose_image_axes_relation(function_ref, frame, components)
                    {
                        frame.relations.insert(handle, relation);
                    }
                    Value::Composite(
                        values
                            .into_iter()
                            .flat_map(|value| match value {
                                Value::Composite(values) => values,
                                value => vec![value],
                            })
                            .collect(),
                    )
                } else {
                    Value::Composite(values)
                }
            }
            Expression::Unary {
                op: UnaryOperator::Negate,
                expr,
            } => self
                .eval_expr(function_ref, frame, *expr)?
                .unary_float(Interval::negate)
                .unwrap_or(Value::Opaque),
            Expression::Unary {
                op: UnaryOperator::LogicalNot,
                expr,
            } => match self.eval_expr(function_ref, frame, *expr)? {
                Value::Bool {
                    can_false,
                    can_true,
                } => Value::Bool {
                    can_false: can_true,
                    can_true: can_false,
                },
                _ => Value::Opaque,
            },
            Expression::Binary { op, left, right } => {
                let left_handle = *left;
                let right_handle = *right;
                let left = self.eval_expr(function_ref, frame, left_handle)?;
                let right = self.eval_expr(function_ref, frame, right_handle)?;
                let budget_relation = self.budget_binary_relation(
                    frame,
                    *op,
                    left_handle,
                    right_handle,
                    &left,
                    &right,
                );
                let image_upper_relation = if *op == BinaryOperator::Multiply {
                    [(left_handle, &right), (right_handle, &left)]
                        .into_iter()
                        .find_map(|(source, factor)| {
                            factor
                                .within(0.0, 1.0)
                                .then(|| frame.relations.get(&source).cloned())?
                        })
                        .filter(|relation| {
                            matches!(
                                relation,
                                Relation::ImageUpperIndex(_) | Relation::ImageUpperIndexAxis(_, _)
                            )
                        })
                } else {
                    None
                };
                let mut result = match *op {
                    BinaryOperator::Add => {
                        if let Some(relation) = self.offset_relation(
                            function_ref,
                            left_handle,
                            right_handle,
                            &left,
                            &right,
                        ) {
                            frame.relations.insert(handle, relation);
                        }
                        self.correlated_unit_lerp(function_ref, frame, left_handle, right_handle)
                            .or_else(|| left.clone().binary_float(right.clone(), Interval::add))
                            .or_else(|| left.binary_int(right, |a, b| a.saturating_add(b)))
                    }
                    BinaryOperator::Subtract => {
                        if let (Some(left), Some(right)) = (
                            self.place_of_expr(function_ref, left_handle),
                            self.place_of_expr(function_ref, right_handle),
                        ) {
                            frame
                                .relations
                                .insert(handle, Relation::Difference(left, right));
                        }
                        if is_uniform_one(&right) {
                            match frame.relations.get(&left_handle).cloned() {
                                Some(Relation::ImageDimensions(name)) => {
                                    frame
                                        .relations
                                        .insert(handle, Relation::ImageUpperIndex(name));
                                }
                                Some(Relation::ImageDimension(name, axis)) => {
                                    frame
                                        .relations
                                        .insert(handle, Relation::ImageUpperIndexAxis(name, axis));
                                }
                                _ => {}
                            }
                        }
                        left.clone()
                            .binary_float(right.clone(), Interval::sub)
                            .or_else(|| left.binary_int(right, |a, b| a.saturating_sub(b)))
                    }
                    BinaryOperator::Multiply => {
                        let (correlated, relation) = self.correlated_multiply(
                            function_ref,
                            frame,
                            left_handle,
                            right_handle,
                            &left,
                            &right,
                        );
                        if let Some(relation) = relation {
                            frame.relations.insert(handle, relation);
                        }
                        correlated
                            .or_else(|| multiply_values(left.clone(), right.clone()))
                            .or_else(|| left.binary_int(right, |a, b| a.saturating_mul(b)))
                    }
                    BinaryOperator::Divide => {
                        let minimum = self.abs_min_for_expr(function_ref, frame, right_handle);
                        self.correlated_divide(
                            function_ref,
                            frame,
                            left_handle,
                            right_handle,
                            &left,
                            &right,
                        )
                        .or_else(|| divide_values(left.clone(), right.clone(), minimum))
                        .or_else(|| eval_int_binary(left, right, *op))
                    }
                    BinaryOperator::Modulo => eval_int_binary(left, right, *op),
                    BinaryOperator::Less
                    | BinaryOperator::LessEqual
                    | BinaryOperator::Greater
                    | BinaryOperator::GreaterEqual
                    | BinaryOperator::Equal
                    | BinaryOperator::NotEqual => Some(compare_values(&left, &right, *op)),
                    BinaryOperator::LogicalAnd | BinaryOperator::LogicalOr => {
                        Some(logical_values(&left, &right, *op))
                    }
                    BinaryOperator::And
                    | BinaryOperator::ExclusiveOr
                    | BinaryOperator::InclusiveOr
                    | BinaryOperator::ShiftLeft
                    | BinaryOperator::ShiftRight => {
                        eval_int_binary(left, right, *op).or(Some(Value::Int {
                            lo: 0,
                            hi: u32::MAX as i64,
                        }))
                    }
                }
                .unwrap_or(Value::Opaque);
                if let Some(relation) = budget_relation {
                    result = cap_budget(result, &relation);
                    frame.relations.insert(handle, relation);
                }
                if let Some(relation) = image_upper_relation {
                    frame.relations.insert(handle, relation);
                }
                if !result.finite_only() {
                    self.alarm_expr(
                        function_ref,
                        handle,
                        "possible_nan_or_inf",
                        &format!("arithmetic result may be NaN or infinity: {result:?}"),
                    );
                }
                result
            }
            Expression::Math {
                fun,
                arg,
                arg1,
                arg2,
                ..
            } => {
                let arg_handle = *arg;
                let arg = self.eval_expr(function_ref, frame, arg_handle)?;
                let arg1_handle = *arg1;
                let arg2_handle = *arg2;
                let arg1 = arg1_handle
                    .map(|handle| self.eval_expr(function_ref, frame, handle))
                    .transpose()?;
                let arg2 = arg2_handle
                    .map(|handle| self.eval_expr(function_ref, frame, handle))
                    .transpose()?;
                let image_upper_relation = match *fun {
                    naga::MathFunction::Max => {
                        if arg1.as_ref().is_some_and(|value| value.within(0.0, 0.0)) {
                            frame.relations.get(&arg_handle).cloned()
                        } else if arg.within(0.0, 0.0) {
                            arg1_handle.and_then(|handle| frame.relations.get(&handle).cloned())
                        } else {
                            None
                        }
                    }
                    naga::MathFunction::Round
                    | naga::MathFunction::Floor
                    | naga::MathFunction::Trunc => frame.relations.get(&arg_handle).cloned(),
                    _ => None,
                }
                .filter(|relation| {
                    matches!(
                        relation,
                        Relation::ImageUpperIndex(_) | Relation::ImageUpperIndexAxis(_, _)
                    )
                });
                let clamp_relation = if *fun == naga::MathFunction::Clamp
                    && arg1.as_ref().is_some_and(|low| low.within(0.0, 0.0))
                {
                    arg2_handle.and_then(|upper| match frame.relations.get(&upper) {
                        Some(Relation::ImageUpperIndex(name)) => {
                            Some(Relation::InImage(name.clone()))
                        }
                        Some(Relation::ImageUpperIndexAxis(name, axis)) => Some(
                            Relation::InImageAxes(name.clone(), 1u8.checked_shl(*axis as u32)?),
                        ),
                        _ => None,
                    })
                } else {
                    None
                };
                let ordered_clamp = *fun == naga::MathFunction::Clamp
                    && arg1_handle.zip(arg2_handle).is_some_and(|(low, high)| {
                        self.contract_orders_bounds(function_ref, low, high)
                    });
                let mut result = if ordered_clamp {
                    binary3(
                        arg.clone(),
                        arg1.clone().unwrap(),
                        arg2.clone().unwrap(),
                        |value, low, high| value.max(low).min(high),
                    )
                } else {
                    eval_math(*fun, arg.clone(), arg1, arg2)
                }
                .unwrap_or_else(|| {
                    self.alarm_expr(
                        function_ref,
                        handle,
                        "unsupported_ir",
                        &format!("unsupported math function {fun:?}"),
                    );
                    Value::unknown(
                        self.module,
                        function_ref.info(self.info)[handle]
                            .ty
                            .inner_with(&self.module.types),
                    )
                });
                if let Some(relation) =
                    self.budget_math_relation(frame, *fun, arg_handle, arg1_handle, arg2_handle)
                {
                    result = cap_budget(result, &relation);
                    frame.relations.insert(handle, relation);
                }
                if let Some(relation) = image_upper_relation {
                    frame.relations.insert(handle, relation);
                }
                if *fun == naga::MathFunction::Abs {
                    if let Some(minimum) = self.abs_min_for_expr(function_ref, frame, arg_handle) {
                        result = result
                            .clone()
                            .unary_float(|value| value.max(Interval::constant(minimum)))
                            .unwrap_or(result);
                    }
                }
                if *fun == naga::MathFunction::Normalize
                    && (arg.definitely_nonzero_norm()
                        || self
                            .norm_min_for_expr(function_ref, frame, arg_handle)
                            .is_some_and(|minimum| minimum > 0.0))
                    && (!result.finite_only() || !result.within(-1.001, 1.001))
                {
                    result = shape_like_float(&arg, Interval::new(-1.001, 1.001));
                    frame.relations.insert(handle, Relation::NonZeroNorm);
                }
                if *fun == naga::MathFunction::Length {
                    if let Some(place) = self.place_of_expr(function_ref, arg_handle) {
                        frame.relations.insert(handle, Relation::NormOf(place));
                    } else {
                        frame
                            .relations
                            .insert(handle, Relation::NormOfExpr(arg_handle));
                    }
                }
                if let Some(relation) = clamp_relation {
                    frame.relations.insert(handle, relation);
                }
                if !result.finite_only() {
                    self.alarm_expr(
                        function_ref,
                        handle,
                        "possible_nan_or_inf",
                        &format!("math result may be NaN or infinity: {result:?}"),
                    );
                }
                result
            }
            Expression::Select {
                condition,
                accept,
                reject,
            } => {
                let (condition_handle, accept_handle, reject_handle) =
                    (*condition, *accept, *reject);
                let condition = self.eval_expr(function_ref, frame, condition_handle)?;
                let accept = self.eval_expr(function_ref, frame, accept_handle)?;
                let reject = self.eval_expr(function_ref, frame, reject_handle)?;
                if let Some(value) = self.correlated_select(
                    function_ref,
                    frame,
                    condition_handle,
                    accept_handle,
                    reject_handle,
                    &accept,
                    &reject,
                ) {
                    value
                } else {
                    match condition {
                        Value::Bool {
                            can_false: false,
                            can_true: true,
                        } => accept,
                        Value::Bool {
                            can_false: true,
                            can_true: false,
                        } => reject,
                        _ => accept.join(&reject),
                    }
                }
            }
            Expression::Swizzle {
                size,
                vector,
                pattern,
            } => {
                let Value::Composite(values) = self.eval_expr(function_ref, frame, *vector)? else {
                    return Ok(Value::Opaque);
                };
                let selected = pattern
                    .iter()
                    .take(value::lanes(*size))
                    .map(|component| values.get(*component as usize).cloned())
                    .collect::<Option<Vec<_>>>();
                let value = selected.map(Value::Composite).unwrap_or_else(|| {
                    self.alarm_expr(
                        function_ref,
                        handle,
                        "unsupported_ir",
                        "swizzle component has no abstract lane",
                    );
                    Value::Opaque
                });
                if let Some(relation @ Relation::BudgetTerms { .. }) =
                    frame.relations.get(vector).cloned()
                {
                    frame.relations.insert(handle, relation);
                }
                value
            }
            Expression::As {
                expr,
                kind,
                convert,
            } => {
                if let Some(relation) = frame.relations.get(expr).cloned().or_else(|| {
                    self.place_of_expr(function_ref, *expr)
                        .and_then(|place| frame.place_relations.get(&place).cloned())
                }) {
                    frame.relations.insert(handle, relation);
                }
                if convert.is_none() {
                    if let Expression::As {
                        expr: original,
                        convert: None,
                        ..
                    } = function.expressions[*expr]
                    {
                        self.eval_expr(function_ref, frame, original)?
                    } else {
                        bitcast_value(self.eval_expr(function_ref, frame, *expr)?, *kind)
                    }
                } else {
                    convert_value(self.eval_expr(function_ref, frame, *expr)?, *kind)
                }
            }
            Expression::Relational { fun, argument } => {
                let argument = self.eval_expr(function_ref, frame, *argument)?;
                eval_relational(*fun, argument)
            }
            Expression::Derivative { expr, .. } => {
                derivative_value(self.eval_expr(function_ref, frame, *expr)?)
            }
            Expression::ImageSample { image, .. } => {
                match self.eval_expr(function_ref, frame, *image)? {
                    Value::Image { name, sample, .. } => {
                        if let Some(relation) = self.budget_term(&name) {
                            frame.relations.insert(handle, relation);
                        }
                        *sample
                    }
                    _ => {
                        self.alarm_expr(
                            function_ref,
                            handle,
                            "unsupported_ir",
                            "sampled value has no texture contract",
                        );
                        Value::unknown(
                            self.module,
                            function_ref.info(self.info)[handle]
                                .ty
                                .inner_with(&self.module.types),
                        )
                    }
                }
            }
            Expression::ImageLoad {
                image,
                coordinate,
                array_index,
                ..
            } => {
                let coordinate_handle = *coordinate;
                let image = self.eval_expr(function_ref, frame, *image)?;
                if let Value::Image { name, .. } = &image {
                    if let Some(relation) = self.budget_term(name) {
                        frame.relations.insert(handle, relation);
                    }
                }
                let mut coordinate = self.eval_expr(function_ref, frame, coordinate_handle)?;
                if let Some(array_index) = array_index {
                    let layer = self.eval_expr(function_ref, frame, *array_index)?;
                    coordinate = match coordinate {
                        Value::Composite(mut coordinates) => {
                            coordinates.push(layer);
                            Value::Composite(coordinates)
                        }
                        coordinate => Value::Composite(vec![coordinate, layer]),
                    };
                }
                self.image_load(
                    function_ref,
                    frame,
                    handle,
                    image,
                    coordinate_handle,
                    coordinate,
                    *array_index,
                )
            }
            Expression::ImageQuery {
                image,
                query: naga::ImageQuery::Size { .. },
            } => match self.eval_expr(function_ref, frame, *image)? {
                Value::Image {
                    name, dimensions, ..
                } => {
                    frame
                        .relations
                        .insert(handle, Relation::ImageDimensions(name));
                    let values = dimensions
                        .into_iter()
                        .map(|(lo, hi)| Value::Int {
                            lo: lo as i64,
                            hi: hi as i64,
                        })
                        .collect::<Vec<_>>();
                    if values.len() == 1 {
                        values.into_iter().next().unwrap()
                    } else {
                        Value::Composite(values)
                    }
                }
                _ => Value::Opaque,
            },
            Expression::ImageQuery {
                query: naga::ImageQuery::NumLevels,
                ..
            } => Value::Int { lo: 1, hi: 32 },
            Expression::ImageQuery {
                image,
                query: naga::ImageQuery::NumLayers,
            } => match self.eval_expr(function_ref, frame, *image)? {
                Value::Image {
                    name, dimensions, ..
                } => {
                    let axis = dimensions.len().saturating_sub(1);
                    frame
                        .relations
                        .insert(handle, Relation::ImageDimension(name, axis));
                    dimensions
                        .get(axis)
                        .map(|&(lo, hi)| Value::Int {
                            lo: lo as i64,
                            hi: hi as i64,
                        })
                        .unwrap_or(Value::Opaque)
                }
                _ => Value::Opaque,
            },
            Expression::ImageQuery {
                query: naga::ImageQuery::NumSamples,
                ..
            } => Value::Int {
                lo: 1,
                hi: u32::MAX as i64,
            },
            Expression::ArrayLength(pointer) => {
                let pointer = self.eval_expr(function_ref, frame, *pointer)?;
                if let Value::Pointer(place) = &pointer {
                    frame
                        .relations
                        .insert(handle, Relation::ArrayLength(place.root));
                }
                match self.load(frame, pointer) {
                    Some(Value::Array {
                        length: Some(length),
                        ..
                    }) => Value::Int {
                        lo: length as i64,
                        hi: length as i64,
                    },
                    _ => Value::Int {
                        lo: 0,
                        hi: u32::MAX as i64,
                    },
                }
            }
            Expression::CallResult(_) => {
                frame.values.get(&handle).cloned().unwrap_or(Value::Opaque)
            }
            Expression::ZeroValue(ty) => Value::zero(self.module, *ty),
            other => {
                self.alarm_expr(
                    function_ref,
                    handle,
                    "unsupported_ir",
                    &format!("unsupported reachable expression {other:?}"),
                );
                Value::unknown(
                    self.module,
                    function_ref.info(self.info)[handle]
                        .ty
                        .inner_with(&self.module.types),
                )
            }
        };
        frame.values.insert(handle, value.clone());
        Ok(value)
    }

    fn contract_orders_bounds(
        &self,
        function_ref: FunctionRef,
        low: Handle<Expression>,
        high: Handle<Expression>,
    ) -> bool {
        let Some(low) = self
            .place_of_expr(function_ref, low)
            .and_then(|place| self.place_name(function_ref, &place))
        else {
            return false;
        };
        let Some(high) = self
            .place_of_expr(function_ref, high)
            .and_then(|place| self.place_name(function_ref, &place))
        else {
            return false;
        };
        self.contract.invariants.iter().any(|invariant| {
            matches!(
                invariant,
                InvariantContract::DifferenceGreaterEqual {
                    left,
                    right,
                    minimum,
                } if left == &high && right == &low && *minimum >= 0.0
            )
        })
    }

    fn correlated_multiply(
        &self,
        function_ref: FunctionRef,
        frame: &Frame,
        left_handle: Handle<Expression>,
        right_handle: Handle<Expression>,
        left: &Value,
        right: &Value,
    ) -> (Option<Value>, Option<Relation>) {
        if left_handle == right_handle {
            return (square_value(left), None);
        }
        for (relation_handle, source_handle, relation_value, source_value) in [
            (left_handle, right_handle, left, right),
            (right_handle, left_handle, right, left),
        ] {
            let Some(relation) = frame.relations.get(&relation_handle) else {
                continue;
            };
            let Some(source) = self.place_of_expr(function_ref, source_handle) else {
                continue;
            };
            match relation {
                Relation::InverseNorm(expected) if expected == &source => {
                    return (
                        Some(shape_like_float(source_value, Interval::new(-1.001, 1.001))),
                        None,
                    );
                }
                Relation::InverseSqrt(expected) if expected == &source => {
                    let Value::Float(input) = source_value else {
                        continue;
                    };
                    let Value::Float(scale) = relation_value else {
                        continue;
                    };
                    let lower = if input.lo < 0.0 {
                        input.lo as f64 * scale.hi as f64
                    } else {
                        0.0
                    };
                    let upper_input = input.hi.max(f32::MIN_POSITIVE);
                    let upper = if input.hi <= 0.0 {
                        0.0
                    } else {
                        (upper_input
                            * super::engine::deterministic_inverse_sqrt_positive(upper_input))
                            as f64
                    };
                    return (
                        Some(Value::Float(Interval::rounded_bounds(lower, upper))),
                        Some(Relation::SqrtProduct(source)),
                    );
                }
                _ => {}
            }
        }
        (None, None)
    }

    fn correlated_unit_lerp(
        &self,
        function_ref: FunctionRef,
        frame: &Frame,
        left: Handle<Expression>,
        right: Handle<Expression>,
    ) -> Option<Value> {
        let function = function_ref.function(self.module);
        let (product, one) = if frame.values.get(&right)?.within(1.0, 1.0) {
            (left, right)
        } else if frame.values.get(&left)?.within(1.0, 1.0) {
            (right, left)
        } else {
            return None;
        };
        let Expression::Binary {
            op: BinaryOperator::Multiply,
            left: product_left,
            right: product_right,
        } = function.expressions[product]
        else {
            return None;
        };
        let (factor, endpoint) = [(product_left, product_right), (product_right, product_left)]
            .into_iter()
            .find_map(|(factor, delta)| {
                let Expression::Binary {
                    op: BinaryOperator::Subtract,
                    left: endpoint,
                    right: delta_one,
                } = function.expressions[delta]
                else {
                    return None;
                };
                (frame.values.get(&factor)?.within(0.0, 1.0)
                    && frame.values.get(&delta_one)?.within(1.0, 1.0))
                .then_some((factor, endpoint))
            })?;
        let factor = frame.values.get(&factor)?.clone();
        let endpoint = frame.values.get(&endpoint)?.clone();
        let one = frame.values.get(&one)?.clone();
        binary3(one, endpoint, factor, Interval::mix)
    }

    fn budget_term(&self, image: &str) -> Option<Relation> {
        self.contract.invariants.iter().find_map(|invariant| {
            let InvariantContract::AdditiveTextureBudget {
                diffuse,
                indirect,
                specular,
                reflection,
                maximum,
                ..
            } = invariant
            else {
                return None;
            };
            [diffuse, indirect, specular, reflection]
                .iter()
                .position(|source| source.as_str() == image)
                .map(|index| Relation::BudgetTerms {
                    maximum_bits: maximum.to_bits(),
                    sources: 1 << index,
                })
        })
    }

    fn budget_binary_relation(
        &self,
        frame: &Frame,
        op: BinaryOperator,
        left_handle: Handle<Expression>,
        right_handle: Handle<Expression>,
        left: &Value,
        right: &Value,
    ) -> Option<Relation> {
        let left_relation = frame.relations.get(&left_handle);
        let right_relation = frame.relations.get(&right_handle);
        match op {
            BinaryOperator::Add => merge_budget(left_relation, right_relation),
            BinaryOperator::Multiply => {
                if right.within(0.0, 1.0) {
                    left_relation.cloned().filter(is_budget)
                } else if left.within(0.0, 1.0) {
                    right_relation.cloned().filter(is_budget)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    fn budget_math_relation(
        &self,
        frame: &Frame,
        fun: naga::MathFunction,
        arg: Handle<Expression>,
        arg1: Option<Handle<Expression>>,
        _arg2: Option<Handle<Expression>>,
    ) -> Option<Relation> {
        match fun {
            naga::MathFunction::Max
            | naga::MathFunction::Min
            | naga::MathFunction::Clamp
            | naga::MathFunction::Saturate => frame.relations.get(&arg).cloned().filter(is_budget),
            naga::MathFunction::Mix => merge_budget(
                frame.relations.get(&arg),
                arg1.and_then(|handle| frame.relations.get(&handle)),
            ),
            _ => None,
        }
    }

    fn offset_relation(
        &self,
        function_ref: FunctionRef,
        left_handle: Handle<Expression>,
        right_handle: Handle<Expression>,
        left: &Value,
        right: &Value,
    ) -> Option<Relation> {
        [(left_handle, right), (right_handle, left)]
            .into_iter()
            .find_map(|(source_handle, offset)| {
                Some(Relation::Offset(
                    self.place_of_expr(function_ref, source_handle)?,
                    uniform_float_constant(offset)?,
                ))
            })
    }

    fn correlated_divide(
        &self,
        function_ref: FunctionRef,
        frame: &Frame,
        numerator_handle: Handle<Expression>,
        denominator_handle: Handle<Expression>,
        numerator: &Value,
        denominator: &Value,
    ) -> Option<Value> {
        let function = function_ref.function(self.module);
        if let Expression::Binary {
            op: BinaryOperator::Add,
            left,
            right,
        } = function.expressions[denominator_handle]
        {
            let residual = if left == numerator_handle {
                frame.values.get(&right)
            } else if right == numerator_handle {
                frame.values.get(&left)
            } else {
                None
            };
            if let (
                Value::Float(numerator),
                Some(Value::Float(residual)),
                Value::Float(denominator),
            ) = (numerator, residual, denominator)
            {
                if numerator.is_finite_only()
                    && residual.is_finite_only()
                    && numerator.lo >= 0.0
                    && residual.lo >= 0.0
                {
                    return Some(Value::Float(
                        numerator.fraction_with_nonnegative_residual(*residual, *denominator),
                    ));
                }
            }
        }
        let numerator_place = self.place_of_expr(function_ref, numerator_handle)?;
        let Relation::Offset(source, offset) = frame.relations.get(&denominator_handle)? else {
            return None;
        };
        if source != &numerator_place {
            return None;
        }
        numerator
            .clone()
            .unary_float(|value| value.ratio_with_positive_offset(*offset))
    }

    #[allow(clippy::too_many_arguments)]
    fn correlated_select(
        &self,
        function_ref: FunctionRef,
        frame: &Frame,
        condition: Handle<Expression>,
        accept_handle: Handle<Expression>,
        reject_handle: Handle<Expression>,
        accept: &Value,
        reject: &Value,
    ) -> Option<Value> {
        let function = function_ref.function(self.module);
        let Expression::Binary { op, left, right } = function.expressions[condition] else {
            return None;
        };
        let (source_handle, zero_handle, source_on_left) =
            if self.place_of_expr(function_ref, left).is_some() {
                (left, right, true)
            } else {
                (right, left, false)
            };
        let source = self.place_of_expr(function_ref, source_handle)?;
        let zero = frame.values.get(&zero_handle)?;
        if !zero.within(0.0, 0.0) {
            return None;
        }
        let positive_on_true = matches!(
            (op, source_on_left),
            (BinaryOperator::Greater, true)
                | (BinaryOperator::GreaterEqual, true)
                | (BinaryOperator::Less, false)
                | (BinaryOperator::LessEqual, false)
        );
        let (relation_handle, relation_value, other) = if positive_on_true {
            (accept_handle, accept, reject)
        } else {
            (reject_handle, reject, accept)
        };
        if !other.within(0.0, 0.0)
            || frame.relations.get(&relation_handle) != Some(&Relation::SqrtProduct(source))
        {
            return None;
        }
        relation_value
            .clone()
            .unary_float(|value| value.max(Interval::constant(0.0)))
    }
}

fn square_value(value: &Value) -> Option<Value> {
    match value {
        Value::Float(value) => Some(Value::Float(value.square())),
        Value::Composite(values) => values
            .iter()
            .map(square_value)
            .collect::<Option<Vec<_>>>()
            .map(Value::Composite),
        _ => None,
    }
}

fn uniform_float_constant(value: &Value) -> Option<f32> {
    match value {
        Value::Float(value) if value.lo == value.hi && value.is_finite_only() => Some(value.lo),
        Value::Composite(values) => {
            let first = uniform_float_constant(values.first()?)?;
            values
                .iter()
                .all(|value| uniform_float_constant(value) == Some(first))
                .then_some(first)
        }
        _ => None,
    }
}

fn is_uniform_one(value: &Value) -> bool {
    match value {
        Value::Float(value) => value.lo == 1.0 && value.hi == 1.0,
        Value::Int { lo: 1, hi: 1 } => true,
        Value::Composite(values) => !values.is_empty() && values.iter().all(is_uniform_one),
        _ => false,
    }
}

fn is_budget(relation: &Relation) -> bool {
    matches!(relation, Relation::BudgetTerms { .. })
}

fn merge_budget(left: Option<&Relation>, right: Option<&Relation>) -> Option<Relation> {
    match (left, right) {
        (
            Some(Relation::BudgetTerms {
                maximum_bits: left_max,
                sources: left_sources,
            }),
            Some(Relation::BudgetTerms {
                maximum_bits: right_max,
                sources: right_sources,
            }),
        ) if left_max == right_max && left_sources & right_sources == 0 => {
            Some(Relation::BudgetTerms {
                maximum_bits: *left_max,
                sources: left_sources | right_sources,
            })
        }
        (Some(relation @ Relation::BudgetTerms { .. }), None)
        | (None, Some(relation @ Relation::BudgetTerms { .. })) => Some(relation.clone()),
        _ => None,
    }
}

fn cap_budget(value: Value, relation: &Relation) -> Value {
    let Relation::BudgetTerms { maximum_bits, .. } = relation else {
        return value;
    };
    let maximum = f32::from_bits(*maximum_bits);
    value
        .clone()
        .unary_float(|interval| Interval::new(interval.lo.max(0.0), interval.hi.min(maximum)))
        .unwrap_or(value)
}
