use crate::formula;
use std::sync::Arc;

pub(super) enum Forbidden {}

pub(super) enum Simple {}
pub(super) enum Multistage {}
pub(super) enum VaryingSlope {}

pub(super) enum FixedEffect {}

pub(super) struct Formula {
    pub(super) left: Expression<Multistage>,
    pub(super) right: Expression<Multistage>,
    pub(super) fixed_effects: Option<Expression<VaryingSlope>>,
}
pub(super) enum MultistageOperand {
    Term(formula::Term),
    Formula(Box<Formula>),
}

pub(super) struct Grouped<E> {
    pub(super) group: Box<E>,
    pub(super) intercept: bool,
    pub(super) first: Arc<formula::Factor>,
    pub(super) rest: Vec<Arc<formula::Factor>>,
}

#[derive(Clone, Debug)]
pub(super) enum Stepwise {
    Sw,
    Sw0,
    Csw,
    Csw0,
}

impl Stepwise {
    fn expand(
        &self,
        arguments: impl IntoIterator<Item = Vec<formula::Expression>>,
    ) -> Vec<formula::Expression> {
        let mut alternatives = match self {
            Self::Sw | Self::Sw0 => arguments.into_iter().flatten().collect::<Vec<_>>(),

            Self::Csw | Self::Csw0 => {
                let mut prefixes = vec![formula::Expression::from_iter([])];
                let mut alternatives = Vec::new();

                for argument in arguments {
                    prefixes =
                        super::combine_alternatives(&prefixes, &argument, |prefix, value| {
                            prefix.add(value)
                        });

                    alternatives.extend(prefixes.iter().cloned());
                }

                alternatives
            }
        };

        if matches!(self, Self::Sw0 | Self::Csw0) {
            alternatives.insert(0, formula::Expression::from_iter([]));
        }

        alternatives
    }
}

pub(super) struct StepwiseCall<E> {
    pub(super) operator: Stepwise,
    pub(super) first: Box<E>,
    pub(super) rest: Vec<E>,
}

/// Defines the node payloads that an [`Expression`] of this kind can store.
///
/// An associated type of [`Forbidden`] makes the corresponding variant
/// unconstructible. Whether an operator is valid for particular operands is
/// determined by [`Construct`].
pub(super) trait Kind: Sized {
    type Operand;
    type Addition;
    type Subtraction;
    type Negation;
    type Interaction;
    type FullInteraction;
    type Nesting;
    type Grouped;
    type Stepwise;
}

impl Kind for Simple {
    type Operand = crate::formula::Term;
    type Addition = Binary<Expression<Self>>;
    type Subtraction = Binary<Expression<Self>>;
    type Negation = Box<Expression<Self>>;
    type Interaction = Binary<Expression<Self>>;
    type FullInteraction = Binary<Expression<Self>>;
    type Nesting = Binary<Expression<Self>>;
    type Grouped = Forbidden;
    type Stepwise = StepwiseCall<Expression<Self>>;
}

impl Kind for Multistage {
    type Operand = MultistageOperand;
    type Addition = Binary<Expression<Self>>;
    type Subtraction = Binary<Expression<Self>, Expression<Simple>>;
    type Negation = Box<Expression<Simple>>;
    type Interaction = Binary<Expression<Simple>>;
    type FullInteraction = Binary<Expression<Simple>>;
    type Nesting = Binary<Expression<Simple>>;
    type Grouped = Forbidden;
    type Stepwise = StepwiseCall<Expression<Self>>;
}

impl Kind for VaryingSlope {
    type Operand = Expression<FixedEffect>;
    type Addition = Binary<Expression<Self>>;
    type Subtraction = Forbidden;
    type Negation = Forbidden;
    type Interaction = Forbidden;
    type FullInteraction = Forbidden;
    type Nesting = Forbidden;
    type Grouped = Grouped<Expression<FixedEffect>>;
    type Stepwise = StepwiseCall<Expression<Self>>;
}

impl Kind for FixedEffect {
    type Operand = String;
    type Addition = Binary<Expression<Self>>;
    type Subtraction = Forbidden;
    type Negation = Forbidden;
    type Interaction = Binary<Expression<Self>>;
    type FullInteraction = Forbidden;
    type Nesting = Forbidden;
    type Grouped = Forbidden;
    type Stepwise = StepwiseCall<Expression<Self>>;
}

pub(super) enum Expression<K: Kind> {
    Empty,
    Operand(K::Operand),
    Add(K::Addition),
    Subtract(K::Subtraction),
    Negate(K::Negation),
    Interact(K::Interaction),
    InteractFully(K::FullInteraction),
    Nesting(K::Nesting),
    Parenthesized(Box<Self>),
    Stepwise(K::Stepwise),
    Grouped(K::Grouped),
}

impl<K: Construct> Expression<K> {
    pub(super) fn term(term: formula::Term) -> Result<Self, ConstructionError> {
        K::term(term)
    }

    pub(super) fn multistage(formula: Formula) -> Result<Self, ConstructionError> {
        K::multistage(formula)
    }

    pub(super) fn add(left: Self, right: Self) -> Result<Self, ConstructionError> {
        K::add(left, right)
    }

    pub(super) fn subtract(left: Self, right: Self) -> Result<Self, ConstructionError> {
        K::subtract(left, right)
    }

    pub(super) fn negate(self) -> Result<Self, ConstructionError> {
        K::negate(self)
    }

    pub(super) fn interact(left: Self, right: Self) -> Result<Self, ConstructionError> {
        K::interact(left, right)
    }

    pub(super) fn interact_fully(left: Self, right: Self) -> Result<Self, ConstructionError> {
        K::interact_fully(left, right)
    }

    pub(super) fn nest(left: Self, right: Self) -> Result<Self, ConstructionError> {
        K::nest(left, right)
    }

    pub(super) fn grouped(
        self,
        intercept: bool,
        first: Arc<formula::Factor>,
        rest: Vec<Arc<formula::Factor>>,
    ) -> Result<Self, ConstructionError> {
        K::grouped(self, intercept, first, rest)
    }

    pub(super) fn stepwise(call: StepwiseCall<Self>) -> Result<Self, ConstructionError> {
        K::stepwise(call)
    }
}

impl Expression<Simple> {
    fn expand_operand(term: &formula::Term) -> Vec<formula::Expression> {
        vec![formula::Expression::from(term.clone())]
    }

    pub(super) fn expand(&self) -> Vec<formula::Expression> {
        expand_arithmetic(self, Self::expand_operand)
    }

    pub(super) fn apply_to(&self, accumulated: formula::Expression) -> Vec<formula::Expression> {
        apply_arithmetic(self, accumulated, Self::expand_operand)
    }

    fn subtract_from(&self, accumulated: formula::Expression) -> Vec<formula::Expression> {
        if let Self::Negate(value) = self {
            value.apply_to(accumulated)
        } else {
            self.expand()
                .into_iter()
                .map(|value| accumulated.clone().subtract(value))
                .collect()
        }
    }
}

impl Expression<Multistage> {
    fn expand_operand(operand: &MultistageOperand) -> Vec<formula::Expression> {
        match operand {
            MultistageOperand::Term(term) => vec![formula::Expression::from(term.clone())],
            MultistageOperand::Formula(child) => child
                .expand()
                .into_iter()
                .map(|child| formula::Expression::new([], vec![child]))
                .collect(),
        }
    }

    pub(super) fn expand(&self) -> Vec<formula::Expression> {
        expand_arithmetic(self, Self::expand_operand)
    }

    pub(super) fn apply_to(&self, accumulated: formula::Expression) -> Vec<formula::Expression> {
        apply_arithmetic(self, accumulated, Self::expand_operand)
    }

    pub(super) fn into_simple(self) -> Result<Expression<Simple>, ConstructionError> {
        Ok(match self {
            Self::Empty => Expression::Empty,
            Self::Operand(MultistageOperand::Term(term)) => Expression::Operand(term),
            Self::Operand(MultistageOperand::Formula(_)) => {
                return Err(ConstructionError::MultistageArithmetic);
            }
            Self::Add(node) => Expression::Add(Binary {
                left: Box::new((*node.left).into_simple()?),
                right: Box::new((*node.right).into_simple()?),
            }),
            Self::Subtract(node) => Expression::Subtract(Binary {
                left: Box::new((*node.left).into_simple()?),
                right: node.right,
            }),
            Self::Parenthesized(value) => {
                Expression::Parenthesized(Box::new((*value).into_simple()?))
            }
            Self::Stepwise(call) => Expression::Stepwise(StepwiseCall {
                operator: call.operator,
                first: Box::new((*call.first).into_simple()?),
                rest: call
                    .rest
                    .into_iter()
                    .map(Self::into_simple)
                    .collect::<Result<Vec<_>, _>>()?,
            }),
            Self::Negate(value) => Expression::Negate(value),
            Self::Interact(node) => Expression::Interact(node),
            Self::InteractFully(node) => Expression::InteractFully(node),
            Self::Nesting(node) => Expression::Nesting(node),
            Self::Grouped(impossible) => match impossible {},
        })
    }
}

impl Expression<VaryingSlope> {
    pub(super) fn into_fixed_effect(self) -> Result<Expression<FixedEffect>, ConstructionError> {
        Ok(match self {
            Self::Empty => Expression::Empty,
            Self::Operand(group) => group,
            Self::Add(node) => Expression::Add(Binary::new(
                (*node.left).into_fixed_effect()?,
                (*node.right).into_fixed_effect()?,
            )),
            Self::Parenthesized(value) => {
                Expression::Parenthesized(Box::new((*value).into_fixed_effect()?))
            }
            Self::Stepwise(call) => Expression::Stepwise(StepwiseCall {
                operator: call.operator,
                first: Box::new((*call.first).into_fixed_effect()?),
                rest: call
                    .rest
                    .into_iter()
                    .map(Self::into_fixed_effect)
                    .collect::<Result<Vec<_>, _>>()?,
            }),
            Self::Subtract(impossible) => match impossible {},
            Self::Negate(impossible) => match impossible {},
            Self::Interact(impossible) => match impossible {},
            Self::InteractFully(impossible) => match impossible {},
            Self::Nesting(impossible) => match impossible {},
            Self::Grouped(_) => {
                return Err(ConstructionError::ExpectedFixedEffectGroup);
            }
        })
    }

    pub(super) fn expand(&self) -> Vec<formula::Expression> {
        match self {
            Self::Empty => vec![formula::Expression::from_iter([])],
            Self::Operand(group) => group
                .expand()
                .into_iter()
                .map(|expression| formula::encode_fixed_effects(expression, true, []))
                .collect(),
            Self::Add(node) => super::combine_alternatives(
                &node.left.expand(),
                &node.right.expand(),
                |left, right| left.add(right),
            ),
            Self::Parenthesized(value) => value.expand(),
            Self::Stepwise(call) => call.operator.expand(
                std::iter::once(call.first.as_ref())
                    .chain(call.rest.iter())
                    .map(Self::expand),
            ),
            Self::Grouped(node) => node
                .group
                .expand()
                .into_iter()
                .map(|expression| {
                    let loadings = std::iter::once(&node.first).chain(&node.rest).cloned();
                    formula::encode_fixed_effects(expression, node.intercept, loadings)
                })
                .collect(),

            Self::Subtract(impossible) => match *impossible {},
            Self::Negate(impossible) => match *impossible {},
            Self::Interact(impossible) => match *impossible {},
            Self::InteractFully(impossible) => match *impossible {},
            Self::Nesting(impossible) => match *impossible {},
        }
    }
}

impl Expression<FixedEffect> {
    pub(super) fn expand(&self) -> Vec<formula::Expression> {
        match self {
            Self::Empty => vec![formula::Expression::from_iter([])],

            Self::Operand(name) => vec![formula::Expression::from(formula::Term::from(
                formula::Factor::from(name.as_str()),
            ))],

            Self::Add(node) => super::combine_alternatives(
                &node.left.expand(),
                &node.right.expand(),
                |left, right| left.add(right),
            ),

            Self::Interact(node) => super::combine_alternatives(
                &node.left.expand(),
                &node.right.expand(),
                |left, right| left.interact(&right),
            ),

            Self::Parenthesized(value) => value.expand(),

            Self::Stepwise(call) => call.operator.expand(
                std::iter::once(call.first.as_ref())
                    .chain(call.rest.iter())
                    .map(Self::expand),
            ),

            Self::Subtract(impossible) => match *impossible {},
            Self::Negate(impossible) => match *impossible {},
            Self::InteractFully(impossible) => match *impossible {},
            Self::Nesting(impossible) => match *impossible {},
            Self::Grouped(impossible) => match *impossible {},
        }
    }
}

fn expand_arithmetic<K>(
    expression: &Expression<K>,
    expand_operand: fn(&K::Operand) -> Vec<formula::Expression>,
) -> Vec<formula::Expression>
where
    K: Kind<
            Addition = Binary<Expression<K>>,
            Subtraction = Binary<Expression<K>, Expression<Simple>>,
            Negation = Box<Expression<Simple>>,
            Interaction = Binary<Expression<Simple>>,
            FullInteraction = Binary<Expression<Simple>>,
            Nesting = Binary<Expression<Simple>>,
            Grouped = Forbidden,
            Stepwise = StepwiseCall<Expression<K>>,
        >,
{
    match expression {
        Expression::Empty => vec![formula::Expression::from_iter([])],
        Expression::Operand(operand) => expand_operand(operand),
        Expression::Add(_) | Expression::Subtract(_) | Expression::Negate(_) => apply_arithmetic(
            expression,
            formula::Expression::from_iter([]),
            expand_operand,
        ),
        Expression::Interact(node) => {
            super::combine_alternatives(&node.left.expand(), &node.right.expand(), |left, right| {
                left.interact(&right)
            })
        }
        Expression::InteractFully(node) => {
            super::combine_alternatives(&node.left.expand(), &node.right.expand(), |left, right| {
                left.interact_fully(right)
            })
        }
        Expression::Nesting(node) => {
            super::combine_alternatives(&node.left.expand(), &node.right.expand(), |left, right| {
                left.nest(right)
            })
        }
        Expression::Parenthesized(value) => expand_arithmetic(value, expand_operand),
        Expression::Stepwise(call) => call.operator.expand(
            std::iter::once(call.first.as_ref())
                .chain(call.rest.iter())
                .map(|argument| expand_arithmetic(argument, expand_operand)),
        ),
        Expression::Grouped(impossible) => match *impossible {},
    }
}

fn apply_arithmetic<K>(
    expression: &Expression<K>,
    accumulated: formula::Expression,
    expand_operand: fn(&K::Operand) -> Vec<formula::Expression>,
) -> Vec<formula::Expression>
where
    K: Kind<
            Addition = Binary<Expression<K>>,
            Subtraction = Binary<Expression<K>, Expression<Simple>>,
            Negation = Box<Expression<Simple>>,
            Interaction = Binary<Expression<Simple>>,
            FullInteraction = Binary<Expression<Simple>>,
            Nesting = Binary<Expression<Simple>>,
            Grouped = Forbidden,
            Stepwise = StepwiseCall<Expression<K>>,
        >,
{
    match expression {
        Expression::Add(node) => apply_arithmetic(&node.left, accumulated, expand_operand)
            .into_iter()
            .flat_map(|value| apply_arithmetic(&node.right, value, expand_operand))
            .collect(),
        Expression::Subtract(node) => apply_arithmetic(&node.left, accumulated, expand_operand)
            .into_iter()
            .flat_map(|value| node.right.subtract_from(value))
            .collect(),
        Expression::Negate(value) => value.subtract_from(accumulated),
        _ => expand_arithmetic(expression, expand_operand)
            .into_iter()
            .map(|value| accumulated.clone().add(value))
            .collect(),
    }
}

impl Formula {
    pub(super) fn expand(&self) -> Vec<formula::Formula> {
        let left = self.left.expand();
        let right = self
            .right
            .apply_to(formula::Expression::from(formula::Term::intercept()));
        let fixed_effects = self
            .fixed_effects
            .as_ref()
            .map(Expression::<VaryingSlope>::expand)
            .unwrap_or_else(|| vec![formula::Expression::from_iter([])]);

        let sides = super::combine_alternatives(&left, &right, |left, right| (left, right));
        super::combine_alternatives(&sides, &fixed_effects, |(left, right), fixed_effects| {
            let right = if fixed_effects.has_fixed_effect_intercept() {
                right.without_intercept()
            } else {
                right
            };

            formula::Formula {
                expressions: std::collections::BTreeMap::from([
                    (formula::Role::Dependent, left),
                    (formula::Role::Independent, right),
                    (formula::Role::FixedEffect, fixed_effects),
                ]),
            }
        })
    }
}

/// Validates parsed operands and constructs expressions with the requested meaning.
///
/// Each method must reject unsupported operands or return an expression allowed
/// by [`Kind`]. The result need not use the variant named after the operator:
/// construction can convert operands into a narrower kind and wrap the result.
///
/// For example, `VaryingSlope::interact` requires operands convertible to
/// `Expression<FixedEffect>`. It constructs an inner grouping interaction and
/// returns it as an `Operand`, while `VaryingSlope::Interaction` is `Forbidden`.
pub(super) trait Construct: Kind {
    fn term(term: formula::Term) -> Result<Expression<Self>, ConstructionError>;

    fn multistage(formula: Formula) -> Result<Expression<Self>, ConstructionError>;

    fn add(
        left: Expression<Self>,
        right: Expression<Self>,
    ) -> Result<Expression<Self>, ConstructionError>;

    fn subtract(
        left: Expression<Self>,
        right: Expression<Self>,
    ) -> Result<Expression<Self>, ConstructionError>;

    fn negate(value: Expression<Self>) -> Result<Expression<Self>, ConstructionError>;

    fn interact(
        left: Expression<Self>,
        right: Expression<Self>,
    ) -> Result<Expression<Self>, ConstructionError>;

    fn interact_fully(
        left: Expression<Self>,
        right: Expression<Self>,
    ) -> Result<Expression<Self>, ConstructionError>;

    fn nest(
        left: Expression<Self>,
        right: Expression<Self>,
    ) -> Result<Expression<Self>, ConstructionError>;

    fn grouped(
        _group: Expression<Self>,
        _intercept: bool,
        _first: Arc<formula::Factor>,
        _rest: Vec<Arc<formula::Factor>>,
    ) -> Result<Expression<Self>, ConstructionError> {
        Err(ConstructionError::GroupedNotAllowed)
    }

    fn stepwise(
        call: StepwiseCall<Expression<Self>>,
    ) -> Result<Expression<Self>, ConstructionError>;
}

impl Construct for Multistage {
    fn term(term: formula::Term) -> Result<Expression<Self>, ConstructionError> {
        Ok(Expression::Operand(MultistageOperand::Term(term)))
    }

    fn multistage(formula: Formula) -> Result<Expression<Self>, ConstructionError> {
        Ok(Expression::Operand(MultistageOperand::Formula(Box::new(
            formula,
        ))))
    }

    fn add(
        left: Expression<Self>,
        right: Expression<Self>,
    ) -> Result<Expression<Self>, ConstructionError> {
        Ok(Expression::Add(Binary::new(left, right)))
    }

    fn subtract(
        left: Expression<Self>,
        right: Expression<Self>,
    ) -> Result<Expression<Self>, ConstructionError> {
        Ok(Expression::Subtract(Binary::new(
            left,
            right.into_simple()?,
        )))
    }

    fn negate(value: Expression<Self>) -> Result<Expression<Self>, ConstructionError> {
        Ok(Expression::Negate(Box::new(value.into_simple()?)))
    }

    fn interact(
        left: Expression<Self>,
        right: Expression<Self>,
    ) -> Result<Expression<Self>, ConstructionError> {
        Ok(Expression::Interact(Binary::new(
            left.into_simple()?,
            right.into_simple()?,
        )))
    }

    fn interact_fully(
        left: Expression<Self>,
        right: Expression<Self>,
    ) -> Result<Expression<Self>, ConstructionError> {
        Ok(Expression::InteractFully(Binary::new(
            left.into_simple()?,
            right.into_simple()?,
        )))
    }

    fn nest(
        left: Expression<Self>,
        right: Expression<Self>,
    ) -> Result<Expression<Self>, ConstructionError> {
        Ok(Expression::Nesting(Binary::new(
            left.into_simple()?,
            right.into_simple()?,
        )))
    }

    fn stepwise(
        call: StepwiseCall<Expression<Self>>,
    ) -> Result<Expression<Self>, ConstructionError> {
        Ok(Expression::Stepwise(call))
    }
}

impl Construct for VaryingSlope {
    fn term(term: formula::Term) -> Result<Expression<Self>, ConstructionError> {
        Expression::<FixedEffect>::term(term).map(Expression::Operand)
    }

    fn multistage(_: Formula) -> Result<Expression<Self>, ConstructionError> {
        Err(ConstructionError::MultistageNotAllowed)
    }

    fn add(
        left: Expression<Self>,
        right: Expression<Self>,
    ) -> Result<Expression<Self>, ConstructionError> {
        Ok(Expression::Add(Binary::new(left, right)))
    }

    fn subtract(
        _: Expression<Self>,
        _: Expression<Self>,
    ) -> Result<Expression<Self>, ConstructionError> {
        Err(ConstructionError::InvalidOperator {
            operator: "-",
            context: "varying-slope expressions",
        })
    }

    fn negate(_: Expression<Self>) -> Result<Expression<Self>, ConstructionError> {
        Err(ConstructionError::InvalidOperator {
            operator: "-",
            context: "varying-slope expressions",
        })
    }

    fn interact(
        left: Expression<Self>,
        right: Expression<Self>,
    ) -> Result<Expression<Self>, ConstructionError> {
        let group = Expression::<FixedEffect>::interact(
            left.into_fixed_effect()?,
            right.into_fixed_effect()?,
        )?;
        Ok(Expression::Operand(group))
    }

    fn interact_fully(
        _: Expression<Self>,
        _: Expression<Self>,
    ) -> Result<Expression<Self>, ConstructionError> {
        Err(ConstructionError::InvalidOperator {
            operator: "*",
            context: "varying-slope expressions",
        })
    }

    fn nest(
        _: Expression<Self>,
        _: Expression<Self>,
    ) -> Result<Expression<Self>, ConstructionError> {
        Err(ConstructionError::InvalidOperator {
            operator: "/",
            context: "varying-slope expressions",
        })
    }

    fn grouped(
        group: Expression<Self>,
        intercept: bool,
        first: Arc<formula::Factor>,
        rest: Vec<Arc<formula::Factor>>,
    ) -> Result<Expression<Self>, ConstructionError> {
        let group = group.into_fixed_effect()?;

        Ok(Expression::Grouped(Grouped {
            group: Box::new(group),
            intercept,
            first,
            rest,
        }))
    }

    fn stepwise(
        call: StepwiseCall<Expression<Self>>,
    ) -> Result<Expression<Self>, ConstructionError> {
        Ok(Expression::Stepwise(call))
    }
}

impl Construct for FixedEffect {
    fn term(term: formula::Term) -> Result<Expression<Self>, ConstructionError> {
        if let Some(crate::formula::FactorKind::Variable(variable)) =
            term.as_factor().map(|factor| &factor.kind)
        {
            Ok(Expression::Operand(variable.definition().name().to_owned()))
        } else {
            Err(ConstructionError::ExpectedFixedEffectColumn)
        }
    }

    fn multistage(_: Formula) -> Result<Expression<Self>, ConstructionError> {
        Err(ConstructionError::MultistageNotAllowed)
    }

    fn add(
        left: Expression<Self>,
        right: Expression<Self>,
    ) -> Result<Expression<Self>, ConstructionError> {
        Ok(Expression::Add(Binary::new(left, right)))
    }

    fn subtract(
        _: Expression<Self>,
        _: Expression<Self>,
    ) -> Result<Expression<Self>, ConstructionError> {
        Err(ConstructionError::InvalidOperator {
            operator: "-",
            context: "fixed-effect expressions",
        })
    }

    fn negate(_: Expression<Self>) -> Result<Expression<Self>, ConstructionError> {
        Err(ConstructionError::InvalidOperator {
            operator: "-",
            context: "fixed-effect expressions",
        })
    }

    fn interact(
        left: Expression<Self>,
        right: Expression<Self>,
    ) -> Result<Expression<Self>, ConstructionError> {
        Ok(Expression::Interact(Binary::new(left, right)))
    }

    fn interact_fully(
        _: Expression<Self>,
        _: Expression<Self>,
    ) -> Result<Expression<Self>, ConstructionError> {
        Err(ConstructionError::InvalidOperator {
            operator: "*",
            context: "fixed-effect expressions",
        })
    }

    fn nest(
        _: Expression<Self>,
        _: Expression<Self>,
    ) -> Result<Expression<Self>, ConstructionError> {
        Err(ConstructionError::InvalidOperator {
            operator: "/",
            context: "fixed-effect expressions",
        })
    }

    fn stepwise(
        call: StepwiseCall<Expression<Self>>,
    ) -> Result<Expression<Self>, ConstructionError> {
        Ok(Expression::Stepwise(call))
    }
}

pub(super) struct Binary<L, R = L> {
    pub(super) left: Box<L>,
    pub(super) right: Box<R>,
}

impl<L, R> Binary<L, R> {
    fn new(left: L, right: R) -> Self {
        Self {
            left: Box::new(left),
            right: Box::new(right),
        }
    }
}

#[derive(Debug, PartialEq)]
pub(super) enum ConstructionError {
    MultistageArithmetic,
    MultistageNotAllowed,
    ExpectedFixedEffectColumn,
    InvalidOperator {
        operator: &'static str,
        context: &'static str,
    },
    ExpectedFixedEffectGroup,
    GroupedNotAllowed,
}

impl std::fmt::Display for ConstructionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MultistageArithmetic => f.write_str("multistage formulas may only be added"),
            Self::MultistageNotAllowed => f.write_str("multistage formulas are not allowed here"),
            Self::ExpectedFixedEffectColumn => {
                f.write_str("fixed effects require column references")
            }
            Self::InvalidOperator { operator, context } => {
                write!(f, "operator {operator} is not supported in {context}")
            }
            Self::ExpectedFixedEffectGroup => {
                f.write_str("expected a grouping expression without loadings")
            }
            Self::GroupedNotAllowed => f.write_str("grouped loadings are not allowed here"),
        }
    }
}

impl std::error::Error for ConstructionError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formula::Term;

    mod conversion {
        use super::*;

        fn term() -> Expression<Multistage> {
            Expression::Operand(MultistageOperand::Term(Term::intercept()))
        }

        fn nested() -> Expression<Multistage> {
            Expression::Operand(MultistageOperand::Formula(Box::new(Formula {
                left: term(),
                right: term(),
                fixed_effects: None,
            })))
        }

        #[test]
        fn conversion_preserves_grouped_subtraction() {
            let expression = Expression::Parenthesized(Box::new(Expression::Subtract(Binary {
                left: Box::new(term()),
                right: Box::new(Expression::Operand(Term::intercept())),
            })));

            let converted = expression.into_simple().unwrap();
            let Expression::Parenthesized(value) = converted else {
                panic!("expected group");
            };
            let Expression::Subtract(node) = *value else {
                panic!("expected subtraction");
            };
            assert!(matches!(*node.left, Expression::Operand(_)));
            assert!(matches!(*node.right, Expression::Operand(_)));
        }

        #[test]
        fn conversion_rejects_nested_formulas_inside_containers() {
            let expressions = [
                nested(),
                Expression::Parenthesized(Box::new(nested())),
                Expression::Add(Binary {
                    left: Box::new(term()),
                    right: Box::new(nested()),
                }),
                Expression::Stepwise(StepwiseCall {
                    operator: Stepwise::Sw,
                    first: Box::new(term()),
                    rest: vec![nested()],
                }),
            ];

            for expression in expressions {
                assert!(matches!(
                    expression.into_simple(),
                    Err(ConstructionError::MultistageArithmetic)
                ));
            }
        }
    }

    mod fixed_effects {
        use super::*;
        use crate::formula::Factor;

        #[test]
        fn operands_require_single_columns() {
            for name in ["firm", "1"] {
                let expression =
                    Expression::<FixedEffect>::term(Term::from(Factor::from(name))).unwrap();

                assert!(matches!(
                    expression,
                    Expression::Operand(actual) if actual == name
                ));
            }

            let product: Term = [Factor::from("firm"), Factor::from("year")]
                .into_iter()
                .collect();

            for term in [Term::intercept(), product] {
                assert!(matches!(
                    Expression::<FixedEffect>::term(term),
                    Err(ConstructionError::ExpectedFixedEffectColumn)
                ));
            }
        }

        #[test]
        fn operators_are_restricted() {
            type Expr = Expression<FixedEffect>;
            type BinaryConstructor = fn(Expr, Expr) -> Result<Expr, ConstructionError>;

            let column = || Expr::Operand("firm".to_owned());

            assert!(matches!(Expr::add(column(), column()), Ok(Expr::Add(_))));
            assert!(matches!(
                Expr::interact(column(), column()),
                Ok(Expr::Interact(_))
            ));

            let forbidden: [(BinaryConstructor, &str); 3] = [
                (Expr::subtract, "-"),
                (Expr::interact_fully, "*"),
                (Expr::nest, "/"),
            ];

            for (construct, operator) in forbidden {
                assert!(matches!(
                    construct(column(), column()),
                    Err(ConstructionError::InvalidOperator {
                        operator: actual,
                        context: "fixed-effect expressions",
                    }) if actual == operator
                ));
            }

            assert!(matches!(
                column().negate(),
                Err(ConstructionError::InvalidOperator {
                    operator: "-",
                    context: "fixed-effect expressions",
                })
            ));
        }
    }
}
