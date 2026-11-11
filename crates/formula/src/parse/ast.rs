use crate::formula;

pub(super) enum Forbidden {}

pub(super) enum Simple {}
pub(super) enum Multistage {}
pub(super) enum FixedEffect {}

pub(super) struct Formula {
    pub(super) left: Expression<Multistage>,
    pub(super) right: Expression<Multistage>,
    pub(super) fixed_effects: Option<Expression<FixedEffect>>,
}
pub(super) enum MultistageOperand {
    Term(crate::formula::Term),
    Formula(Box<Formula>),
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

pub(super) trait Kind: Sized {
    type Operand;
    type Addition;
    type Subtraction;
    type Negation;
    type Interaction;
    type FullInteraction;
    type Nesting;
}

impl Kind for Simple {
    type Operand = crate::formula::Term;
    type Addition = Binary<Expression<Self>>;
    type Subtraction = Binary<Expression<Self>>;
    type Negation = Box<Expression<Self>>;
    type Interaction = Binary<Expression<Self>>;
    type FullInteraction = Binary<Expression<Self>>;
    type Nesting = Binary<Expression<Self>>;
}

impl Kind for Multistage {
    type Operand = MultistageOperand;
    type Addition = Binary<Expression<Self>>;
    type Subtraction = Binary<Expression<Self>, Expression<Simple>>;
    type Negation = Box<Expression<Simple>>;
    type Interaction = Binary<Expression<Simple>>;
    type FullInteraction = Binary<Expression<Simple>>;
    type Nesting = Binary<Expression<Simple>>;
}

impl Kind for FixedEffect {
    type Operand = String;
    type Addition = Binary<Expression<Self>>;
    type Subtraction = Forbidden;
    type Negation = Forbidden;
    type Interaction = Binary<Expression<Self>>;
    type FullInteraction = Forbidden;
    type Nesting = Forbidden;
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
    Group(Box<Self>),
    Stepwise(StepwiseCall<Self>),
}

impl<K: Construct> Expression<K> {
    pub(super) fn term(term: crate::formula::Term) -> Result<Self, ConstructionError> {
        K::term(term).map(Self::Operand)
    }

    pub(super) fn multistage(formula: Formula) -> Result<Self, ConstructionError> {
        K::multistage(formula).map(Self::Operand)
    }

    pub(super) fn add(left: Self, right: Self) -> Result<Self, ConstructionError> {
        K::add(left, right).map(Self::Add)
    }

    pub(super) fn subtract(left: Self, right: Self) -> Result<Self, ConstructionError> {
        K::subtract(left, right).map(Self::Subtract)
    }

    pub(super) fn negate(self) -> Result<Self, ConstructionError> {
        K::negate(self).map(Self::Negate)
    }

    pub(super) fn interact(left: Self, right: Self) -> Result<Self, ConstructionError> {
        K::interact(left, right).map(Self::Interact)
    }

    pub(super) fn interact_fully(left: Self, right: Self) -> Result<Self, ConstructionError> {
        K::interact_fully(left, right).map(Self::InteractFully)
    }

    pub(super) fn nest(left: Self, right: Self) -> Result<Self, ConstructionError> {
        K::nest(left, right).map(Self::Nesting)
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
            Self::Group(value) => Expression::Group(Box::new((*value).into_simple()?)),
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
        })
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

            Self::Group(value) => value.expand(),

            Self::Stepwise(call) => call.operator.expand(
                std::iter::once(call.first.as_ref())
                    .chain(call.rest.iter())
                    .map(Self::expand),
            ),

            Self::Subtract(impossible) => match *impossible {},
            Self::Negate(impossible) => match *impossible {},
            Self::InteractFully(impossible) => match *impossible {},
            Self::Nesting(impossible) => match *impossible {},
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
        Expression::Group(value) => expand_arithmetic(value, expand_operand),
        Expression::Stepwise(call) => call.operator.expand(
            std::iter::once(call.first.as_ref())
                .chain(call.rest.iter())
                .map(|argument| expand_arithmetic(argument, expand_operand)),
        ),
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
            .map(Expression::<FixedEffect>::expand)
            .unwrap_or_else(|| vec![formula::Expression::from_iter([])]);

        let sides = super::combine_alternatives(&left, &right, |left, right| (left, right));
        super::combine_alternatives(&sides, &fixed_effects, |(left, right), fixed_effects| {
            let fixed_effects = formula::encode_fixed_effects(fixed_effects);
            let right = if fixed_effects.is_empty() {
                right
            } else {
                right.without_intercept()
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

pub(super) trait Construct: Kind {
    fn term(term: crate::formula::Term) -> Result<Self::Operand, ConstructionError>;

    fn multistage(formula: Formula) -> Result<Self::Operand, ConstructionError>;

    fn add(
        left: Expression<Self>,
        right: Expression<Self>,
    ) -> Result<Self::Addition, ConstructionError>;

    fn subtract(
        left: Expression<Self>,
        right: Expression<Self>,
    ) -> Result<Self::Subtraction, ConstructionError>;

    fn negate(value: Expression<Self>) -> Result<Self::Negation, ConstructionError>;

    fn interact(
        left: Expression<Self>,
        right: Expression<Self>,
    ) -> Result<Self::Interaction, ConstructionError>;

    fn interact_fully(
        left: Expression<Self>,
        right: Expression<Self>,
    ) -> Result<Self::FullInteraction, ConstructionError>;

    fn nest(
        left: Expression<Self>,
        right: Expression<Self>,
    ) -> Result<Self::Nesting, ConstructionError>;
}

impl Construct for Multistage {
    fn term(term: crate::formula::Term) -> Result<Self::Operand, ConstructionError> {
        Ok(MultistageOperand::Term(term))
    }

    fn multistage(formula: Formula) -> Result<Self::Operand, ConstructionError> {
        Ok(MultistageOperand::Formula(Box::new(formula)))
    }

    fn add(
        left: Expression<Self>,
        right: Expression<Self>,
    ) -> Result<Self::Addition, ConstructionError> {
        Ok(Binary::new(left, right))
    }

    fn subtract(
        left: Expression<Self>,
        right: Expression<Self>,
    ) -> Result<Self::Subtraction, ConstructionError> {
        Ok(Binary::new(left, right.into_simple()?))
    }

    fn negate(value: Expression<Self>) -> Result<Self::Negation, ConstructionError> {
        Ok(Box::new(value.into_simple()?))
    }

    fn interact(
        left: Expression<Self>,
        right: Expression<Self>,
    ) -> Result<Self::Interaction, ConstructionError> {
        Ok(Binary::new(left.into_simple()?, right.into_simple()?))
    }

    fn interact_fully(
        left: Expression<Self>,
        right: Expression<Self>,
    ) -> Result<Self::FullInteraction, ConstructionError> {
        Ok(Binary::new(left.into_simple()?, right.into_simple()?))
    }

    fn nest(
        left: Expression<Self>,
        right: Expression<Self>,
    ) -> Result<Self::Nesting, ConstructionError> {
        Ok(Binary::new(left.into_simple()?, right.into_simple()?))
    }
}

impl Construct for FixedEffect {
    fn term(term: crate::formula::Term) -> Result<Self::Operand, ConstructionError> {
        if let Some(crate::formula::FactorKind::Variable(variable)) =
            term.as_factor().map(|factor| &factor.kind)
        {
            Ok(variable.definition().name().to_owned())
        } else {
            Err(ConstructionError::ExpectedFixedEffectColumn)
        }
    }

    fn multistage(_: Formula) -> Result<Self::Operand, ConstructionError> {
        Err(ConstructionError::MultistageNotAllowed)
    }

    fn add(
        left: Expression<Self>,
        right: Expression<Self>,
    ) -> Result<Self::Addition, ConstructionError> {
        Ok(Binary::new(left, right))
    }

    fn subtract(
        _: Expression<Self>,
        _: Expression<Self>,
    ) -> Result<Self::Subtraction, ConstructionError> {
        Err(ConstructionError::FixedEffectOperator("-"))
    }

    fn negate(_: Expression<Self>) -> Result<Self::Negation, ConstructionError> {
        Err(ConstructionError::FixedEffectOperator("-"))
    }

    fn interact(
        left: Expression<Self>,
        right: Expression<Self>,
    ) -> Result<Self::Interaction, ConstructionError> {
        Ok(Binary::new(left, right))
    }

    fn interact_fully(
        _: Expression<Self>,
        _: Expression<Self>,
    ) -> Result<Self::FullInteraction, ConstructionError> {
        Err(ConstructionError::FixedEffectOperator("*"))
    }

    fn nest(_: Expression<Self>, _: Expression<Self>) -> Result<Self::Nesting, ConstructionError> {
        Err(ConstructionError::FixedEffectOperator("/"))
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
    FixedEffectOperator(&'static str),
}

impl std::fmt::Display for ConstructionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MultistageArithmetic => f.write_str("multistage formulas may only be added"),
            Self::MultistageNotAllowed => f.write_str("multistage formulas are not allowed here"),
            Self::ExpectedFixedEffectColumn => {
                f.write_str("fixed effects require column references")
            }
            Self::FixedEffectOperator(operator) => {
                write!(f, "operator {operator} is not supported in fixed effects")
            }
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
            let expression = Expression::Group(Box::new(Expression::Subtract(Binary {
                left: Box::new(term()),
                right: Box::new(Expression::Operand(Term::intercept())),
            })));

            let converted = expression.into_simple().unwrap();
            let Expression::Group(value) = converted else {
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
                Expression::Group(Box::new(nested())),
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
                    Err(ConstructionError::FixedEffectOperator(actual))
                        if actual == operator
                ));
            }

            assert!(matches!(
                column().negate(),
                Err(ConstructionError::FixedEffectOperator("-"))
            ));
        }
    }
}
