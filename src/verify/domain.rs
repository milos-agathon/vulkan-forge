//! Conservative scalar intervals used component-wise for WGSL values.
#![cfg_attr(not(test), allow(dead_code))]
// ponytail: the follow-on verifier interpreter will consume this module; remove the allowance then.

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Interval {
    pub(crate) lo: f32,
    pub(crate) hi: f32,
    pub(crate) may_nan: bool,
    pub(crate) may_pos_inf: bool,
    pub(crate) may_neg_inf: bool,
    pub(crate) may_neg_zero: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Comparison {
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
}

// Vulkan permits about 2.5 ULP for native f32 divide/sqrt/inverseSqrt
// (see shaders/includes/determinism.wgsl); three outward ULPs cover that contract.
const NATIVE_OP_ULPS: usize = 3;
const DIV_DENOM_MAX: f32 = f32::from_bits(0x7e80_0000); // 2^126

impl Interval {
    pub(crate) fn new(lo: f32, hi: f32) -> Self {
        assert!(lo.is_finite() && hi.is_finite() && lo <= hi);
        Self {
            lo,
            hi,
            may_nan: false,
            may_pos_inf: false,
            may_neg_inf: false,
            may_neg_zero: lo <= 0.0 && hi >= 0.0,
        }
    }

    pub(crate) fn constant(value: f32) -> Self {
        if value.is_nan() {
            return Self::exceptional(true, false, false, false);
        }
        if value == f32::INFINITY {
            return Self::exceptional(false, true, false, false);
        }
        if value == f32::NEG_INFINITY {
            return Self::exceptional(false, false, true, false);
        }
        Self {
            lo: value,
            hi: value,
            may_nan: false,
            may_pos_inf: false,
            may_neg_inf: false,
            may_neg_zero: value == 0.0,
        }
    }

    fn exceptional(nan: bool, pos_inf: bool, neg_inf: bool, neg_zero: bool) -> Self {
        Self {
            lo: f32::INFINITY,
            hi: f32::NEG_INFINITY,
            may_nan: nan,
            may_pos_inf: pos_inf,
            may_neg_inf: neg_inf,
            may_neg_zero: neg_zero,
        }
    }

    fn top(nan: bool) -> Self {
        Self {
            lo: -f32::MAX,
            hi: f32::MAX,
            may_nan: nan,
            may_pos_inf: true,
            may_neg_inf: true,
            may_neg_zero: true,
        }
    }

    pub(crate) fn unknown() -> Self {
        Self::top(true)
    }

    pub(crate) fn is_finite_only(self) -> bool {
        self.has_finite() && !self.may_nan && !self.may_pos_inf && !self.may_neg_inf
    }

    pub(crate) fn is_within(self, lo: f32, hi: f32) -> bool {
        self.is_finite_only() && self.lo >= lo && self.hi <= hi
    }

    pub(crate) fn negate(self) -> Self {
        if self.has_infinity() || self.may_nan {
            return Self::top(true);
        }
        bounds(-(self.hi as f64), -(self.lo as f64), false)
    }

    pub(crate) fn abs(self) -> Self {
        if self.has_infinity() || self.may_nan {
            return Self::top(true);
        }
        if self.lo >= 0.0 {
            self
        } else if self.hi <= 0.0 {
            self.negate()
        } else {
            bounds(
                0.0,
                (self.lo as f64).abs().max((self.hi as f64).abs()),
                false,
            )
        }
    }

    pub(crate) fn sign(self) -> Self {
        if self.has_infinity() || self.may_nan {
            return Self::top(true);
        }
        Self::new(
            if self.lo < 0.0 { -1.0 } else { 0.0 },
            if self.hi > 0.0 { 1.0 } else { 0.0 },
        )
    }

    pub(crate) fn square(self) -> Self {
        if self.may_nan || self.has_infinity() {
            return Self::top(true);
        }
        let maximum = (self.lo as f64).abs().max((self.hi as f64).abs());
        let minimum = if self.lo <= 0.0 && self.hi >= 0.0 {
            0.0
        } else {
            (self.lo as f64).abs().min((self.hi as f64).abs())
        };
        bounds(minimum * minimum, maximum * maximum, false)
    }

    pub(crate) fn rounded_bounds(lo: f64, hi: f64) -> Self {
        bounds(lo, hi, false)
    }

    fn has_finite(self) -> bool {
        self.lo <= self.hi
    }

    fn has_infinity(self) -> bool {
        self.may_pos_inf || self.may_neg_inf
    }

    fn may_zero(self) -> bool {
        self.may_neg_zero || (self.has_finite() && self.lo <= 0.0 && self.hi >= 0.0)
    }

    fn may_subnormal(self) -> bool {
        self.has_finite()
            && ((self.hi > 0.0 && self.lo < f32::MIN_POSITIVE)
                || (self.lo < 0.0 && self.hi > -f32::MIN_POSITIVE))
    }

    fn ftz_subnormal_hull(self) -> Option<Self> {
        if !self.may_subnormal() {
            return None;
        }
        let max_subnormal = next_down(f32::MIN_POSITIVE);
        let has_positive = self.hi > 0.0 && self.lo < f32::MIN_POSITIVE;
        let has_negative = self.lo < 0.0 && self.hi > -f32::MIN_POSITIVE;
        Some(Self::new(
            if has_negative {
                self.lo.max(-max_subnormal)
            } else {
                0.0
            },
            if has_positive {
                self.hi.min(max_subnormal)
            } else {
                -0.0
            },
        ))
    }

    fn with_input_ftz(mut self) -> Self {
        if !self.has_finite() {
            return self;
        }
        if self.lo > 0.0 && self.lo < f32::MIN_POSITIVE {
            self.lo = 0.0;
            self.may_neg_zero = true;
        }
        if self.hi < 0.0 && self.hi > -f32::MIN_POSITIVE {
            self.hi = -0.0;
            self.may_neg_zero = true;
        }
        self.may_neg_zero |= self.lo <= 0.0 && self.hi >= 0.0;
        self
    }

    pub(crate) fn has_exceptional_values(&self) -> bool {
        self.may_nan || self.may_pos_inf || self.may_neg_inf || self.may_neg_zero
    }

    pub(crate) fn contains(self, value: f32) -> bool {
        if value.is_nan() {
            self.may_nan
        } else if value == f32::INFINITY {
            self.may_pos_inf
        } else if value == f32::NEG_INFINITY {
            self.may_neg_inf
        } else if value == 0.0 && value.is_sign_negative() {
            self.may_neg_zero
        } else {
            self.has_finite() && self.lo <= value && value <= self.hi
        }
    }

    pub(crate) fn add(self, rhs: Self) -> Self {
        let self_ = self.with_input_ftz();
        let rhs = rhs.with_input_ftz();
        if self_.has_infinity() || rhs.has_infinity() {
            return Self::top(true);
        }
        bounds(
            self_.lo as f64 + rhs.lo as f64,
            self_.hi as f64 + rhs.hi as f64,
            self_.may_nan || rhs.may_nan,
        )
    }

    pub(crate) fn sub(self, rhs: Self) -> Self {
        let self_ = self.with_input_ftz();
        let rhs = rhs.with_input_ftz();
        if self_.has_infinity() || rhs.has_infinity() {
            return Self::top(true);
        }
        bounds(
            self_.lo as f64 - rhs.hi as f64,
            self_.hi as f64 - rhs.lo as f64,
            self_.may_nan || rhs.may_nan,
        )
    }

    pub(crate) fn mul(self, rhs: Self) -> Self {
        let self_ = self.with_input_ftz();
        let rhs = rhs.with_input_ftz();
        if self_.has_infinity() || rhs.has_infinity() {
            return Self::top(true);
        }
        let products = [
            self_.lo as f64 * rhs.lo as f64,
            self_.lo as f64 * rhs.hi as f64,
            self_.hi as f64 * rhs.lo as f64,
            self_.hi as f64 * rhs.hi as f64,
        ];
        from_candidates(&products, self_.may_nan || rhs.may_nan)
    }

    pub(crate) fn div(self, rhs: Self) -> Self {
        let self_ = self.with_input_ftz();
        let rhs = rhs.with_input_ftz();
        if self_.may_nan
            || rhs.may_nan
            || self_.has_infinity()
            || rhs.has_infinity()
            || rhs.may_zero()
            || !denominator_has_defined_accuracy(rhs)
        {
            return Self::top(true);
        }
        let quotients = [
            self_.lo as f64 / rhs.lo as f64,
            self_.lo as f64 / rhs.hi as f64,
            self_.hi as f64 / rhs.lo as f64,
            self_.hi as f64 / rhs.hi as f64,
        ];
        expand_ulps(
            from_candidates(&quotients, self_.may_nan || rhs.may_nan),
            NATIVE_OP_ULPS,
        )
    }

    pub(crate) fn div_with_abs_min(self, rhs: Self, minimum: f32) -> Self {
        if !minimum.is_finite() || minimum <= 0.0 || rhs.may_nan || rhs.has_infinity() {
            return self.div(rhs);
        }
        let mut result = None;
        if rhs.has_finite() && rhs.lo <= -minimum {
            let negative = Self::new(rhs.lo, rhs.hi.min(-minimum));
            result = Some(self.div(negative));
        }
        if rhs.has_finite() && rhs.hi >= minimum {
            let positive = Self::new(rhs.lo.max(minimum), rhs.hi);
            result = Some(result.map_or_else(
                || self.div(positive),
                |value: Self| value.join(self.div(positive)),
            ));
        }
        result.unwrap_or_else(|| Self::top(true))
    }

    pub(crate) fn ratio_with_positive_offset(self, offset: f32) -> Self {
        if !self.is_finite_only() || self.lo < 0.0 || !offset.is_finite() || offset <= 0.0 {
            return Self::top(true);
        }
        expand_ulps(
            bounds(
                self.lo as f64 / (self.lo as f64 + offset as f64),
                self.hi as f64 / (self.hi as f64 + offset as f64),
                false,
            ),
            NATIVE_OP_ULPS,
        )
    }

    pub(crate) fn fraction_with_nonnegative_residual(
        self,
        residual: Self,
        denominator: Self,
    ) -> Self {
        if !self.is_finite_only()
            || !residual.is_finite_only()
            || self.lo < 0.0
            || residual.lo < 0.0
        {
            return Self::top(true);
        }
        let denominator = denominator.with_input_ftz();
        if !denominator.is_finite_only()
            || denominator.lo < f32::MIN_POSITIVE
            || denominator.hi > DIV_DENOM_MAX
        {
            return Self::top(true);
        }
        expand_ulps(Self::new(0.0, 1.0), NATIVE_OP_ULPS)
    }

    pub(crate) fn sqrt(self) -> Self {
        let input = self.with_input_ftz();
        if input.may_nan || input.has_infinity() || !input.has_finite() || input.lo < 0.0 {
            return Self::top(true);
        }
        let direct = input.nonnegative_sqrt();
        if input.lo > 0.0 {
            direct.join(Self::constant(1.0).div(input.inverse_sqrt()))
        } else {
            direct
        }
    }

    pub(crate) fn nonnegative_sqrt(self) -> Self {
        if self.may_nan || self.has_infinity() || !self.has_finite() || self.lo < 0.0 {
            return Self::top(true);
        }
        expand_ulps(
            bounds((self.lo as f64).sqrt(), (self.hi as f64).sqrt(), false),
            NATIVE_OP_ULPS,
        )
    }

    pub(crate) fn inverse_sqrt(self) -> Self {
        let input = self.with_input_ftz();
        if input.may_nan
            || input.has_infinity()
            || !input.has_finite()
            || input.lo < 0.0
            || input.may_zero()
        {
            return Self::top(true);
        }
        expand_ulps(
            bounds(
                1.0 / (input.hi as f64).sqrt(),
                1.0 / (input.lo as f64).sqrt(),
                false,
            ),
            NATIVE_OP_ULPS,
        )
    }

    pub(crate) fn min(self, rhs: Self) -> Self {
        let may_choose_either_subnormal = self.may_subnormal() && rhs.may_subnormal();
        let self_ = self.with_input_ftz();
        let rhs = rhs.with_input_ftz();
        if self_.may_nan || rhs.may_nan || self_.has_infinity() || rhs.has_infinity() {
            return Self::top(true);
        }
        if !self_.has_finite() || !rhs.has_finite() {
            return self_.join(rhs);
        }
        let result = bounds(
            self_.lo.min(rhs.lo) as f64,
            self_.hi.min(rhs.hi) as f64,
            self_.may_nan || rhs.may_nan,
        );
        if may_choose_either_subnormal {
            result.join(self_).join(rhs)
        } else {
            result
        }
    }

    pub(crate) fn max(self, rhs: Self) -> Self {
        let may_choose_either_subnormal = self.may_subnormal() && rhs.may_subnormal();
        let self_ = self.with_input_ftz();
        let rhs = rhs.with_input_ftz();
        if self_.may_nan || rhs.may_nan || self_.has_infinity() || rhs.has_infinity() {
            return Self::top(true);
        }
        if !self_.has_finite() || !rhs.has_finite() {
            return self_.join(rhs);
        }
        let result = bounds(
            self_.lo.max(rhs.lo) as f64,
            self_.hi.max(rhs.hi) as f64,
            self_.may_nan || rhs.may_nan,
        );
        if may_choose_either_subnormal {
            result.join(self_).join(rhs)
        } else {
            result
        }
    }

    pub(crate) fn clamp(self, low: Self, high: Self) -> Self {
        if !low.has_finite() || !high.has_finite() || low.hi > high.lo {
            return Self::top(true);
        }
        self.max(low).min(high)
    }

    pub(crate) fn mix(self, rhs: Self, factor: Self) -> Self {
        if !self.is_finite_only() || !rhs.is_finite_only() || !factor.is_finite_only() {
            return Self::top(true);
        }
        let mut candidates = [0.0; 8];
        let mut index = 0;
        for left in [self.lo, self.hi] {
            for right in [rhs.lo, rhs.hi] {
                for factor in [factor.lo, factor.hi] {
                    candidates[index] =
                        left as f64 * (1.0 - factor as f64) + right as f64 * factor as f64;
                    index += 1;
                }
            }
        }
        from_candidates(&candidates, false)
    }

    pub(crate) fn select(on_false: Self, on_true: Self, condition: Option<bool>) -> Self {
        match condition {
            Some(false) => on_false,
            Some(true) => on_true,
            None => on_false.join(on_true),
        }
    }

    pub(crate) fn fma(self, multiplier: Self, addend: Self) -> Self {
        let self_ = self.with_input_ftz();
        let multiplier = multiplier.with_input_ftz();
        let addend = addend.with_input_ftz();
        if self_.has_infinity() || multiplier.has_infinity() || addend.has_infinity() {
            return Self::top(true);
        }
        let mut candidates = [0.0; 8];
        let mut index = 0;
        for left in [self_.lo, self_.hi] {
            for right in [multiplier.lo, multiplier.hi] {
                for addend in [addend.lo, addend.hi] {
                    candidates[index] = left as f64 * right as f64 + addend as f64;
                    index += 1;
                }
            }
        }
        let fused = from_candidates(
            &candidates,
            self_.may_nan || multiplier.may_nan || addend.may_nan,
        );
        let decomposed = self_.mul(multiplier).add(addend);
        fused.join(decomposed)
    }

    pub(crate) fn join(self, rhs: Self) -> Self {
        let (lo, hi) = match (self.has_finite(), rhs.has_finite()) {
            (true, true) => (self.lo.min(rhs.lo), self.hi.max(rhs.hi)),
            (true, false) => (self.lo, self.hi),
            (false, true) => (rhs.lo, rhs.hi),
            (false, false) => (f32::INFINITY, f32::NEG_INFINITY),
        };
        Self {
            lo,
            hi,
            may_nan: self.may_nan || rhs.may_nan,
            may_pos_inf: self.may_pos_inf || rhs.may_pos_inf,
            may_neg_inf: self.may_neg_inf || rhs.may_neg_inf,
            may_neg_zero: self.may_neg_zero || rhs.may_neg_zero || (lo <= 0.0 && hi >= 0.0),
        }
    }

    pub(crate) fn widen(self, next: Self) -> Self {
        if !self.has_finite() {
            return self.join(next);
        }
        let joined = self.join(next);
        Self {
            lo: if next.has_finite() && next.lo < self.lo {
                -f32::MAX
            } else {
                joined.lo
            },
            hi: if next.has_finite() && next.hi > self.hi {
                f32::MAX
            } else {
                joined.hi
            },
            ..joined
        }
    }

    pub(crate) fn refine(
        self,
        rhs: Self,
        comparison: Comparison,
        truth: bool,
    ) -> Option<(Self, Self)> {
        let self_ = self.with_input_ftz();
        let rhs = rhs.with_input_ftz();
        let left_ftz = self
            .ftz_subnormal_hull()
            .filter(|&ftz| comparison_branch_feasible(ftz, rhs, comparison, truth));
        let right_ftz = rhs
            .ftz_subnormal_hull()
            .filter(|&ftz| comparison_branch_feasible(self_, ftz, comparison, truth));
        let refined = if !truth
            && (self_.may_nan || rhs.may_nan)
            && matches!(
                comparison,
                Comparison::Lt | Comparison::Le | Comparison::Gt | Comparison::Ge | Comparison::Eq
            ) {
            Some((self_, rhs))
        } else {
            match (comparison, truth) {
                (Comparison::Lt, true) => refine_ordered(self_, rhs, true),
                (Comparison::Lt, false) => refine_ordered(rhs, self_, false).map(swap),
                (Comparison::Le, true) => refine_ordered(self_, rhs, false),
                (Comparison::Le, false) => refine_ordered(rhs, self_, true).map(swap),
                (Comparison::Gt, truth) => rhs.refine(self_, Comparison::Lt, truth).map(swap),
                (Comparison::Ge, truth) => rhs.refine(self_, Comparison::Le, truth).map(swap),
                (Comparison::Eq, true) | (Comparison::Ne, false) => {
                    let lo = self_.lo.max(rhs.lo);
                    let hi = self_.hi.min(rhs.hi);
                    let mut left = self_.with_bounds(lo, hi);
                    let mut right = rhs.with_bounds(lo, hi);
                    left.may_nan = false;
                    right.may_nan = false;
                    viable(left, right)
                }
                (Comparison::Eq, false) | (Comparison::Ne, true) => Some((self_, rhs)),
            }
        };
        refined.map(|(left, right)| {
            (
                left_ftz.map_or(left, |ftz| left.join(ftz)),
                right_ftz.map_or(right, |ftz| right.join(ftz)),
            )
        })
    }

    fn with_bounds(mut self, lo: f32, hi: f32) -> Self {
        self.lo = lo;
        self.hi = hi;
        self.may_neg_zero &= lo <= 0.0 && hi >= 0.0;
        self
    }
}

pub(crate) fn dot(left: &[Interval], right: &[Interval]) -> Interval {
    assert_eq!(left.len(), right.len());
    left.iter()
        .zip(right)
        .fold(Interval::constant(0.0), |sum, (&left, &right)| {
            sum.add(left.mul(right))
        })
}

pub(crate) fn normalize(value: &[Interval]) -> Vec<Interval> {
    let length = dot(value, value).sqrt();
    value
        .iter()
        .map(|component| component.div(length))
        .collect()
}

fn bounds(lo: f64, hi: f64, may_nan: bool) -> Interval {
    if may_nan
        || lo.is_nan()
        || hi.is_nan()
        || lo > hi
        || lo < -(f32::MAX as f64)
        || hi > f32::MAX as f64
    {
        return Interval::top(true);
    }
    let lo = lower_f32(lo);
    let hi = upper_f32(hi);
    include_result_ftz(Interval {
        lo,
        hi,
        may_nan: false,
        may_pos_inf: false,
        may_neg_inf: false,
        may_neg_zero: lo <= 0.0 && hi >= 0.0,
    })
}

fn from_candidates(values: &[f64], may_nan: bool) -> Interval {
    let lo = values.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    bounds(lo, hi, may_nan)
}

fn lower_f32(value: f64) -> f32 {
    let rounded = value as f32;
    if rounded as f64 > value {
        next_down(rounded)
    } else {
        rounded
    }
}

fn upper_f32(value: f64) -> f32 {
    let rounded = value as f32;
    if (rounded as f64) < value {
        next_up(rounded)
    } else {
        rounded
    }
}

fn include_result_ftz(mut interval: Interval) -> Interval {
    if !interval.has_finite() {
        return interval;
    }
    if interval.lo > 0.0 && interval.lo < f32::MIN_POSITIVE {
        interval.lo = 0.0;
        interval.may_neg_zero = true;
    }
    if interval.hi < 0.0 && interval.hi > -f32::MIN_POSITIVE {
        interval.hi = -0.0;
        interval.may_neg_zero = true;
    }
    interval.may_neg_zero |= interval.lo <= 0.0 && interval.hi >= 0.0;
    interval
}

fn expand_ulps(mut interval: Interval, count: usize) -> Interval {
    if !interval.has_finite() {
        return interval;
    }
    for _ in 0..count {
        if interval.lo != 0.0 {
            let lo = next_down(interval.lo);
            if lo == f32::NEG_INFINITY {
                return Interval::top(true);
            } else {
                interval.lo = lo;
            }
        }
        if interval.hi != 0.0 {
            let hi = next_up(interval.hi);
            if hi == f32::INFINITY {
                return Interval::top(true);
            } else {
                interval.hi = hi;
            }
        }
    }
    include_result_ftz(interval)
}

fn denominator_has_defined_accuracy(denominator: Interval) -> bool {
    if !denominator.has_finite() {
        return false;
    }
    (denominator.lo >= f32::MIN_POSITIVE && denominator.hi <= DIV_DENOM_MAX)
        || (denominator.lo >= -DIV_DENOM_MAX && denominator.hi <= -f32::MIN_POSITIVE)
}

fn next_down(value: f32) -> f32 {
    if value == f32::NEG_INFINITY {
        value
    } else if value == 0.0 {
        -f32::from_bits(1)
    } else if value > 0.0 {
        f32::from_bits(value.to_bits() - 1)
    } else {
        f32::from_bits(value.to_bits() + 1)
    }
}

fn next_up(value: f32) -> f32 {
    if value == f32::INFINITY {
        value
    } else if value == 0.0 {
        f32::from_bits(1)
    } else if value > 0.0 {
        f32::from_bits(value.to_bits() + 1)
    } else {
        f32::from_bits(value.to_bits() - 1)
    }
}

fn refine_ordered(left: Interval, right: Interval, strict: bool) -> Option<(Interval, Interval)> {
    if left.has_infinity() || right.has_infinity() {
        let mut left = left;
        let mut right = right;
        left.may_nan = false;
        right.may_nan = false;
        return viable(left, right);
    }
    let left_hi = if strict {
        left.hi.min(next_down(right.hi))
    } else {
        left.hi.min(right.hi)
    };
    let right_lo = if strict {
        right.lo.max(next_up(left.lo))
    } else {
        right.lo.max(left.lo)
    };
    let mut left = left.with_bounds(left.lo, left_hi);
    let mut right = right.with_bounds(right_lo, right.hi);
    left.may_nan = false;
    right.may_nan = false;
    viable(left, right)
}

fn comparison_branch_feasible(
    left: Interval,
    right: Interval,
    comparison: Comparison,
    truth: bool,
) -> bool {
    if (left.may_nan || right.may_nan)
        && match comparison {
            Comparison::Ne => truth,
            Comparison::Eq => !truth,
            _ => !truth,
        }
    {
        return true;
    }
    if left.has_infinity() || right.has_infinity() {
        return true;
    }
    if !left.has_finite() || !right.has_finite() {
        return left.may_neg_zero || right.may_neg_zero;
    }
    match (comparison, truth) {
        (Comparison::Lt, true) => left.lo < right.hi,
        (Comparison::Lt, false) => left.hi >= right.lo,
        (Comparison::Le, true) => left.lo <= right.hi,
        (Comparison::Le, false) => left.hi > right.lo,
        (Comparison::Gt, true) => left.hi > right.lo,
        (Comparison::Gt, false) => left.lo <= right.hi,
        (Comparison::Ge, true) => left.hi >= right.lo,
        (Comparison::Ge, false) => left.lo < right.hi,
        (Comparison::Eq, true) | (Comparison::Ne, false) => {
            left.lo <= right.hi && right.lo <= left.hi
        }
        (Comparison::Eq, false) | (Comparison::Ne, true) => {
            left.lo != left.hi || right.lo != right.hi || left.lo != right.lo
        }
    }
}

fn viable(left: Interval, right: Interval) -> Option<(Interval, Interval)> {
    let left_exists = left.has_finite() || left.has_infinity() || left.may_neg_zero;
    let right_exists = right.has_finite() || right.has_infinity() || right.may_neg_zero;
    (left_exists && right_exists).then_some((left, right))
}

fn swap((left, right): (Interval, Interval)) -> (Interval, Interval) {
    (right, left)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Copy)]
    struct Rng(u64);

    impl Rng {
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

    fn assert_contains(interval: Interval, concrete: f32) {
        assert!(
            interval.contains(concrete),
            "{interval:?} does not contain {concrete:?}"
        );
    }

    fn assert_top(interval: Interval) {
        assert_eq!((interval.lo, interval.hi), (-f32::MAX, f32::MAX));
        assert!(interval.may_nan);
        assert!(interval.may_pos_inf);
        assert!(interval.may_neg_inf);
        assert!(interval.may_neg_zero);
    }

    fn assert_not_top(interval: Interval) {
        assert_ne!((interval.lo, interval.hi), (-f32::MAX, f32::MAX));
    }

    fn directed_ulp(value: f32, upward: bool) -> f32 {
        if upward {
            step_up(value, NATIVE_OP_ULPS)
        } else {
            step_down(value, NATIVE_OP_ULPS)
        }
    }

    fn inverse_sqrt_alternative(value: f32, upward: bool) -> f32 {
        directed_ulp((1.0_f64 / (value as f64).sqrt()) as f32, upward)
    }

    fn division_alternative(numerator: f32, denominator: f32, upward: bool) -> f32 {
        directed_ulp((numerator as f64 / denominator as f64) as f32, upward)
    }

    fn inherited_sqrt_alternative(value: f32, choices: u32) -> f32 {
        let reciprocal_root = inverse_sqrt_alternative(value, choices & 1 != 0);
        division_alternative(1.0, reciprocal_root, choices & 2 != 0)
    }

    fn step_down(mut value: f32, count: usize) -> f32 {
        for _ in 0..count {
            value = next_down(value);
        }
        value
    }

    fn step_up(mut value: f32, count: usize) -> f32 {
        for _ in 0..count {
            value = next_up(value);
        }
        value
    }

    #[test]
    fn tracks_exceptional_values_and_negative_zero() {
        let nan = Interval::constant(f32::NAN);
        assert!(nan.may_nan);
        assert!(Interval::constant(f32::INFINITY).may_pos_inf);
        assert!(Interval::constant(f32::NEG_INFINITY).may_neg_inf);
        assert!(Interval::constant(-0.0).may_neg_zero);
        assert!(Interval::new(-0.0, 1.0).may_neg_zero);
        assert!(!Interval::new(1.0, 2.0).has_exceptional_values());
        assert_contains(nan.min(Interval::constant(2.0)), 2.0);
        assert_contains(nan.max(Interval::constant(2.0)), 2.0);
    }

    #[test]
    fn scalar_transfers_cover_the_required_operations() {
        let x = Interval::new(2.0, 4.0);
        let y = Interval::new(5.0, 8.0);
        assert_contains(x.add(y), 12.0);
        assert_contains(x.sub(y), -6.0);
        assert_contains(x.mul(y), 32.0);
        assert_contains(x.div(y), 0.25);
        assert_contains(x.sqrt(), 2.0);
        assert_contains(x.inverse_sqrt(), 1.0 / 2.0_f32.sqrt());
        assert_contains(x.min(y), 4.0);
        assert_contains(x.max(y), 8.0);
        assert_contains(
            y.clamp(Interval::constant(6.0), Interval::constant(7.0)),
            7.0,
        );
        assert_contains(x.mix(y, Interval::constant(0.25)), 5.0);
        assert_eq!(Interval::select(x, y, Some(false)), x);
        assert_eq!(Interval::select(x, y, Some(true)), y);
        assert_eq!(Interval::select(x, y, None), x.join(y));
        assert_contains(x.fma(y, Interval::constant(1.0)), 33.0);
    }

    #[test]
    fn dot_and_normalize_are_component_wise() {
        let vector = [
            Interval::constant(3.0),
            Interval::constant(4.0),
            Interval::constant(0.0),
        ];
        assert_contains(dot(&vector, &vector), 25.0);
        let normalized = normalize(&vector);
        assert_contains(normalized[0], 0.6);
        assert_contains(normalized[1], 0.8);
        assert_contains(normalized[2], 0.0);
    }

    #[test]
    fn branch_refinement_join_and_widen_are_conservative() {
        let left = Interval::new(-10.0, 10.0);
        let right = Interval::new(0.0, 20.0);
        let (left_true, right_true) = left.refine(right, Comparison::Lt, true).unwrap();
        assert!(left_true.hi < right.hi);
        assert!(right_true.lo > left.lo);
        assert_contains(left_true, 9.0);
        assert_contains(right_true, 10.0);

        let (left_false, right_false) = left.refine(right, Comparison::Lt, false).unwrap();
        assert_contains(left_false, 5.0);
        assert_contains(right_false, 5.0);

        let (equal_left, equal_right) = left.refine(right, Comparison::Eq, true).unwrap();
        assert!(equal_left.lo < 0.0 && equal_left.lo > -f32::MIN_POSITIVE);
        assert_eq!(equal_left.hi, 10.0);
        assert_contains(equal_left, -f32::from_bits(1));
        assert_eq!((equal_right.lo, equal_right.hi), (0.0, 10.0));
        assert!(equal_left.may_neg_zero);
        assert!(equal_right.may_neg_zero);
        assert!(Interval::new(2.0, 3.0)
            .refine(Interval::new(0.0, 1.0), Comparison::Le, true)
            .is_none());
        assert!(left.refine(right, Comparison::Gt, true).is_some());
        assert!(left.refine(right, Comparison::Ge, false).is_some());
        assert!(left.refine(right, Comparison::Ne, true).is_some());

        let (nan_left, _) = Interval::constant(f32::NAN)
            .refine(Interval::constant(1.0), Comparison::Lt, false)
            .expect("NaN makes an ordered comparison false");
        assert!(nan_left.may_nan);

        assert_eq!(left.join(right), Interval::new(-10.0, 20.0));
        assert!(
            Interval::constant(-1.0)
                .join(Interval::constant(1.0))
                .may_neg_zero
        );
        let widened = Interval::new(-1.0, 1.0).widen(Interval::new(-2.0, 2.0));
        assert_eq!((widened.lo, widened.hi), (-f32::MAX, f32::MAX));
    }

    #[test]
    fn unsafe_operations_remain_unproven() {
        let division = Interval::new(-1.0, 1.0).div(Interval::new(-1.0, 1.0));
        assert!(division.may_nan && division.may_pos_inf && division.may_neg_inf);
        assert!(Interval::new(-1.0, 4.0).sqrt().may_nan);
        assert!(Interval::new(0.0, 4.0).inverse_sqrt().may_pos_inf);
        assert!(normalize(&[Interval::constant(0.0); 3])
            .iter()
            .all(Interval::has_exceptional_values));
    }

    #[test]
    fn native_ops_include_the_vulkan_accuracy_envelope() {
        let div_exact = 1.0_f32 / 3.0;
        let div = Interval::constant(1.0).div(Interval::constant(3.0));
        assert!(div.lo <= step_down(div_exact, 3));
        assert!(div.hi >= step_up(div_exact, 3));

        let sqrt_exact = 2.0_f32.sqrt();
        let sqrt = Interval::constant(2.0).sqrt();
        assert!(sqrt.lo <= step_down(sqrt_exact, 3));
        assert!(sqrt.hi >= step_up(sqrt_exact, 3));

        let inverse_exact = 1.0 / 2.0_f32.sqrt();
        let inverse = Interval::constant(2.0).inverse_sqrt();
        assert!(inverse.lo <= step_down(inverse_exact, 3));
        assert!(inverse.hi >= step_up(inverse_exact, 3));
    }

    #[test]
    fn subnormal_inputs_and_results_include_ftz_zeros() {
        let positive_subnormal = f32::from_bits(1);
        let negative_subnormal = -positive_subnormal;
        assert_contains(
            Interval::constant(positive_subnormal).add(Interval::constant(0.0)),
            0.0,
        );
        assert_contains(
            Interval::constant(negative_subnormal).add(Interval::constant(0.0)),
            -0.0,
        );
        assert_contains(
            Interval::constant(f32::MIN_POSITIVE).mul(Interval::constant(0.5)),
            0.0,
        );
        assert_contains(
            Interval::constant(-f32::MIN_POSITIVE).mul(Interval::constant(0.5)),
            -0.0,
        );
        assert!(
            Interval::constant(1.0)
                .div(Interval::constant(positive_subnormal))
                .may_pos_inf
        );
    }

    #[test]
    fn widening_preserves_exceptional_only_states() {
        for (exceptional, expected) in [
            (
                Interval::exceptional(true, false, false, false),
                (true, false, false, false),
            ),
            (
                Interval::exceptional(false, true, false, false),
                (false, true, false, false),
            ),
            (
                Interval::exceptional(false, false, true, false),
                (false, false, true, false),
            ),
            (
                Interval::exceptional(false, false, false, true),
                (false, false, false, true),
            ),
        ] {
            let widened = exceptional.widen(Interval::new(1.0, 2.0));
            assert_eq!(
                (
                    widened.may_nan,
                    widened.may_pos_inf,
                    widened.may_neg_inf,
                    widened.may_neg_zero,
                ),
                expected,
            );
            assert_contains(widened, 1.5);
        }
    }

    #[test]
    fn ordered_refinement_preserves_feasible_infinite_branches() {
        let zero = Interval::constant(0.0);
        let positive_inf = Interval::constant(f32::INFINITY);
        let negative_inf = Interval::constant(f32::NEG_INFINITY);

        let (left, right) = zero.refine(positive_inf, Comparison::Lt, true).unwrap();
        assert_contains(left, 0.0);
        assert_contains(right, f32::INFINITY);

        let (left, right) = negative_inf.refine(zero, Comparison::Lt, true).unwrap();
        assert_contains(left, f32::NEG_INFINITY);
        assert_contains(right, 0.0);
    }

    #[test]
    fn exceptional_boundaries_remain_contained_and_unproven() {
        let spanning_zero = Interval::new(-1.0, 1.0);
        let near_zero = Interval::new(f32::from_bits(1), f32::MIN_POSITIVE);
        assert!(Interval::constant(1.0).div(spanning_zero).may_nan);
        assert!(Interval::constant(1.0).div(near_zero).may_pos_inf);
        assert!(Interval::new(-4.0, -1.0).sqrt().may_nan);
        assert!(Interval::new(-4.0, -1.0).inverse_sqrt().may_nan);
        assert!(
            Interval::constant(f32::MAX)
                .mul(Interval::constant(2.0))
                .may_pos_inf
        );
        assert!(
            Interval::constant(f32::NAN)
                .add(Interval::constant(f32::INFINITY))
                .may_nan
        );
    }

    #[test]
    fn division_fails_closed_outside_its_accuracy_range() {
        let denominator_max = f32::from_bits(0x7e80_0000); // 2^126
        let valid_low = Interval::constant(1.0).div(Interval::constant(f32::MIN_POSITIVE));
        let valid_high = Interval::constant(1.0).div(Interval::constant(denominator_max));
        let valid_negative_low =
            Interval::constant(1.0).div(Interval::constant(-f32::MIN_POSITIVE));
        let valid_negative_high = Interval::constant(1.0).div(Interval::constant(-denominator_max));
        assert_ne!((valid_low.lo, valid_low.hi), (-f32::MAX, f32::MAX));
        assert_ne!((valid_high.lo, valid_high.hi), (-f32::MAX, f32::MAX));
        assert_ne!(
            (valid_negative_low.lo, valid_negative_low.hi),
            (-f32::MAX, f32::MAX)
        );
        assert_ne!(
            (valid_negative_high.lo, valid_negative_high.hi),
            (-f32::MAX, f32::MAX)
        );

        assert_top(Interval::constant(1.0).div(Interval::constant(next_down(f32::MIN_POSITIVE))));
        assert_top(Interval::constant(1.0).div(Interval::constant(next_up(denominator_max))));
        assert_top(Interval::constant(1.0).div(Interval::constant(next_down(-denominator_max))));
        assert_top(
            Interval::constant(1.0).div(Interval::new(denominator_max, next_up(denominator_max))),
        );
        assert_top(Interval::constant(1.0).div(Interval::new(
            next_down(f32::MIN_POSITIVE),
            f32::MIN_POSITIVE,
        )));
    }

    #[test]
    fn sqrt_contains_the_inherited_reciprocal_root_envelope() {
        let x = 2.0_f32;
        let reciprocal_root = (1.0_f64 / (x as f64).sqrt()) as f32;
        let reciprocal_low = step_down(reciprocal_root, 3);
        let reciprocal_high = step_up(reciprocal_root, 3);
        let inherited_low = step_down((1.0 / reciprocal_high) as f32, 3);
        let inherited_high = step_up((1.0 / reciprocal_low) as f32, 3);
        let sqrt = Interval::constant(x).sqrt();
        assert!(sqrt.lo <= inherited_low);
        assert!(sqrt.hi >= inherited_high);

        let independent_oracle = (1.0_f64 / (3.0_f64).sqrt()) as f32;
        let inverse = Interval::constant(3.0).inverse_sqrt();
        assert!(inverse.lo <= step_down(independent_oracle, 3));
        assert!(inverse.hi >= step_up(independent_oracle, 3));

        for boundary in [f32::MIN_POSITIVE, f32::MAX] {
            assert_contains(Interval::constant(boundary).sqrt(), boundary.sqrt());
        }
    }

    #[test]
    fn fma_joins_fused_and_decomposed_results() {
        let a = next_up(1.0);
        let b = next_down(1.0);
        let c = -1.0;
        let fused = a.mul_add(b, c);
        let decomposed = a * b + c;
        assert_ne!(fused, decomposed);

        let result = Interval::constant(a).fma(Interval::constant(b), Interval::constant(c));
        assert_contains(result, fused);
        assert_contains(result, decomposed);
    }

    #[test]
    fn refinement_keeps_subnormal_ftz_comparison_alternatives() {
        let positive = Interval::constant(f32::from_bits(1));
        let negative = Interval::constant(-f32::from_bits(1));
        let zero = Interval::constant(0.0);

        for (left, original, comparison, truth) in [
            (positive, f32::from_bits(1), Comparison::Eq, true),
            (positive, f32::from_bits(1), Comparison::Gt, true),
            (positive, f32::from_bits(1), Comparison::Gt, false),
            (negative, -f32::from_bits(1), Comparison::Eq, true),
            (negative, -f32::from_bits(1), Comparison::Lt, true),
            (negative, -f32::from_bits(1), Comparison::Lt, false),
        ] {
            let (refined_left, refined_zero) = left
                .refine(zero, comparison, truth)
                .expect("the FTZ comparison branch remains reachable");
            for alternative in [original, 0.0, -0.0] {
                assert_contains(refined_left, alternative);
            }
            assert_contains(refined_zero, 0.0);
            assert_contains(refined_zero, -0.0);
        }
    }

    #[test]
    fn refinement_restores_subnormals_only_when_the_branch_can_use_them() {
        let (left, right) = Interval::new(-1.0, 10.0)
            .refine(Interval::new(5.0, 20.0), Comparison::Eq, true)
            .unwrap();
        assert_eq!((left.lo, left.hi), (5.0, 10.0));
        assert_eq!((right.lo, right.hi), (5.0, 10.0));
    }

    #[test]
    fn add_and_sub_round_large_exponent_gaps_to_the_large_operand() {
        let tiny = Interval::constant(f32::MIN_POSITIVE);
        let one = Interval::constant(1.0);
        assert_eq!(one.sub(tiny), one);
        assert_eq!(one.add(tiny), one);
    }

    #[test]
    fn zero_and_ftz_preserve_both_zero_signs() {
        assert!(Interval::constant(0.0).contains(-0.0));
        assert!(Interval::new(0.0, 1.0).may_neg_zero);
        let inverse = Interval::constant(0.0).inverse_sqrt();
        assert!(inverse.may_pos_inf && inverse.may_neg_inf);
    }

    #[test]
    fn invalid_results_retain_indeterminate_finite_values_after_guards() {
        let overflow = Interval::constant(f32::MAX).mul(Interval::constant(2.0));
        assert_top(overflow);
        let invalid_sqrt = Interval::new(-4.0, -1.0).sqrt();
        assert_top(invalid_sqrt);
        assert_top(Interval::constant(f32::NAN).add(Interval::constant(1.0)));

        let (guarded, _) = overflow
            .refine(overflow, Comparison::Ne, false)
            .expect("x != x false edge remains reachable for indeterminate finite values");
        assert_eq!((guarded.lo, guarded.hi), (-f32::MAX, f32::MAX));
    }

    #[test]
    fn subnormal_min_max_and_clamp_keep_permitted_operands() {
        let one = f32::from_bits(1);
        let two = f32::from_bits(2);
        let three = f32::from_bits(3);

        assert_contains(Interval::constant(two).min(Interval::constant(one)), two);
        assert_contains(Interval::constant(-two).max(Interval::constant(-one)), -two);

        let positive =
            Interval::constant(two).clamp(Interval::constant(one), Interval::constant(three));
        for legal in [0.0, -0.0, one, two, three] {
            assert_contains(positive, legal);
        }

        let negative =
            Interval::constant(-two).clamp(Interval::constant(-three), Interval::constant(-one));
        for legal in [0.0, -0.0, -one, -two, -three] {
            assert_contains(negative, legal);
        }
    }

    #[test]
    fn one_million_deterministic_samples_are_contained() {
        let mut rng = Rng(0x4d59_5df4_d0f3_3173);
        for sample in 0..1_000_000 {
            let (ax, x, ay, y) = match sample & 0x3fff {
                0 => (
                    Interval::new(-f32::MIN_POSITIVE, f32::MIN_POSITIVE),
                    f32::from_bits(1),
                    Interval::new(-1.0, 1.0),
                    -0.0,
                ),
                1 => (
                    Interval::new(f32::MAX / 2.0, f32::MAX),
                    f32::MAX,
                    Interval::new(2.0, 3.0),
                    3.0,
                ),
                2 => (
                    Interval::new(-1.0, 1.0).join(Interval::constant(f32::NAN)),
                    f32::NAN,
                    Interval::new(1.0, 2.0).join(Interval::constant(f32::INFINITY)),
                    f32::INFINITY,
                ),
                3 => (
                    Interval::new(-4.0, -1.0),
                    -2.0,
                    Interval::new(f32::from_bits(1), f32::MIN_POSITIVE),
                    f32::from_bits(1),
                ),
                _ => {
                    let ax = Interval::new(rng.finite(-100.0, -1.0), rng.finite(1.0, 100.0));
                    let ay = Interval::new(rng.finite(0.25, 10.0), rng.finite(10.0, 100.0));
                    let x = rng.finite(ax.lo, ax.hi);
                    let y = rng.finite(ay.lo, ay.hi);
                    (ax, x, ay, y)
                }
            };
            let at = Interval::new(0.0, 1.0);
            let t = rng.finite(at.lo, at.hi);
            let root_interval = Interval::new(rng.finite(0.5, 2.0), rng.finite(2.5, 10.0));
            let root_value = rng.finite(root_interval.lo, root_interval.hi);

            assert_contains(ax.add(ay), x + y);
            assert_contains(ax.sub(ay), x - y);
            assert_contains(ax.mul(ay), x * y);
            assert_contains(ax.div(ay), x / y);
            let abstract_sqrt = root_interval.sqrt();
            let abstract_inverse_sqrt = root_interval.inverse_sqrt();
            assert_not_top(abstract_sqrt);
            assert_not_top(abstract_inverse_sqrt);
            assert_contains(
                abstract_sqrt,
                inherited_sqrt_alternative(root_value, rng.next_u32()),
            );
            assert_contains(
                abstract_inverse_sqrt,
                inverse_sqrt_alternative(root_value, rng.next_u32() & 1 != 0),
            );
            assert_contains(ax.min(ay), x.min(y));
            assert_contains(ax.max(ay), x.max(y));
            assert_contains(
                ax.clamp(Interval::constant(-5.0), Interval::constant(5.0)),
                x.clamp(-5.0, 5.0),
            );
            assert_contains(ax.mix(ay, at), x * (1.0 - t) + y * t);
            assert_contains(
                Interval::select(ax, ay, None),
                if rng.next_u32() & 1 == 0 { x } else { y },
            );
            assert_contains(ax.fma(ay, at), x.mul_add(y, t));

            let joined = ax.join(ay);
            assert_contains(joined, x);
            assert_contains(joined, y);
            let widened = ax.widen(ay);
            assert_contains(widened, x);
            assert_contains(widened, y);
            let (refined_x, refined_y) = ax
                .refine(ay, Comparison::Lt, x < y)
                .expect("the sampled comparison must remain feasible");
            assert_contains(refined_x, x);
            assert_contains(refined_y, y);

            let abstracted = [
                Interval::new(0.5, 1.0),
                Interval::new(1.0, 2.0),
                Interval::new(2.0, 4.0),
            ];
            let concrete = abstracted.map(|interval| rng.finite(interval.lo, interval.hi));
            let concrete_dot = concrete.iter().map(|v| v * v).sum::<f32>();
            assert_contains(dot(&abstracted, &abstracted), concrete_dot);
            let length = inherited_sqrt_alternative(concrete_dot, rng.next_u32());
            let normalized = normalize(&abstracted);
            for (component, concrete) in normalized.into_iter().zip(concrete) {
                assert_not_top(component);
                assert_contains(
                    component,
                    division_alternative(concrete, length, rng.next_u32() & 1 != 0),
                );
            }

            if sample & 0x3fff == 0 {
                let subnormal = Interval::constant(f32::from_bits(1));
                assert_contains(subnormal.sqrt(), 0.0);
                assert!(subnormal.inverse_sqrt().may_pos_inf);
            }
        }
    }
}
