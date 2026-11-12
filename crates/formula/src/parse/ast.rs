use crate::formula;
use std::sync::Arc;

pub(super) enum Forbidden {}

pub(super) enum Simple {}
pub(super) enum Multistage {}
pub(super) enum TermQuery {}
pub(super) enum VaryingSlope {}
pub(super) enum FixedEffect {}

pub(super) struct Grouped<E> {
    pub(super) group: Box<E>,
    pub(super) intercept: bool,
    pub(super) first: Arc<formula::Factor>,
    pub(super) rest: Vec<Arc<formula::Factor>>,
}

pub(super) struct StepwiseCall<E> {
    pub(super) operator: StepwiseOperator,
    pub(super) first: Box<E>,
    pub(super) rest: Vec<E>,
}

#[derive(Clone, Debug)]
pub(super) enum StepwiseOperator {
    Sw,
    Sw0,
    Csw,
    Csw0,
}

impl StepwiseOperator {
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
                        super::cartesian_product_with(&prefixes, &argument, |prefix, value| {
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

pub(super) trait Kind: Sized {
    type Operand: Construct<formula::Term>;
    type Multistage: Construct<Vec<formula::Formula>>;
    type Addition: Construct<(Expression<Self>, Expression<Self>)>;
    type Subtraction: Construct<(Expression<Self>, Expression<Self>)>;
    type Negation: Construct<Expression<Self>>;
    type Interaction: Construct<(Expression<Self>, Expression<Self>)>;
    type FullInteraction: Construct<(Expression<Self>, Expression<Self>)>;
    type Nesting: Construct<(Expression<Self>, Expression<Self>)>;
    type Grouped: Construct<Grouped<Expression<Self>>>;
    type Stepwise: Construct<StepwiseCall<Expression<Self>>>;

    fn interact(
        left: Expression<Self>,
        right: Expression<Self>,
    ) -> Result<Expression<Self>, ConstructionError> {
        Self::Interaction::construct((left, right)).map(Expression::Interact)
    }
}

impl Kind for Simple {
    type Operand = formula::Term;
    type Multistage = Forbidden;
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
    type Operand = formula::Term;
    type Multistage = Vec<formula::Formula>;
    type Addition = Binary<Expression<Self>>;
    type Subtraction = Binary<Expression<Self>, Expression<Simple>>;
    type Negation = Box<Expression<Simple>>;
    type Interaction = Binary<Expression<Simple>>;
    type FullInteraction = Binary<Expression<Simple>>;
    type Nesting = Binary<Expression<Simple>>;
    type Grouped = Forbidden;
    type Stepwise = StepwiseCall<Expression<Self>>;
}

impl Kind for TermQuery {
    type Operand = formula::Term;
    type Multistage = Forbidden;
    type Addition = Binary<Expression<Self>>;
    type Subtraction = Binary<Expression<Self>>;
    type Negation = Box<Expression<Self>>;
    type Interaction = Binary<Expression<Self>>;
    type FullInteraction = Binary<Expression<Self>>;
    type Nesting = Binary<Expression<Self>>;
    type Grouped = Forbidden;
    type Stepwise = Forbidden;
}

impl Kind for VaryingSlope {
    type Operand = Expression<FixedEffect>;
    type Multistage = Forbidden;
    type Addition = Binary<Expression<Self>>;
    type Subtraction = Forbidden;
    type Negation = Forbidden;
    type Interaction = Forbidden;
    type FullInteraction = Forbidden;
    type Nesting = Forbidden;
    type Grouped = Grouped<Expression<FixedEffect>>;
    type Stepwise = StepwiseCall<Expression<Self>>;

    fn interact(
        left: Expression<Self>,
        right: Expression<Self>,
    ) -> Result<Expression<Self>, ConstructionError> {
        Expression::<FixedEffect>::interact(
            Expression::<FixedEffect>::construct(left)?,
            Expression::<FixedEffect>::construct(right)?,
        )
        .map(Expression::Operand)
    }
}

impl Kind for FixedEffect {
    type Operand = formula::VariableDefinition;
    type Multistage = Forbidden;
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
    Operand(K::Operand),
    Multistage(K::Multistage),
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

impl<K: Kind> Expression<K> {
    pub(super) fn term(term: formula::Term) -> Result<Self, ConstructionError> {
        K::Operand::construct(term).map(Self::Operand)
    }

    pub(super) fn multistage(formulas: Vec<formula::Formula>) -> Result<Self, ConstructionError> {
        K::Multistage::construct(formulas).map(Self::Multistage)
    }

    pub(super) fn add(left: Self, right: Self) -> Result<Self, ConstructionError> {
        K::Addition::construct((left, right)).map(Self::Add)
    }

    pub(super) fn subtract(left: Self, right: Self) -> Result<Self, ConstructionError> {
        K::Subtraction::construct((left, right)).map(Self::Subtract)
    }

    pub(super) fn negate(self) -> Result<Self, ConstructionError> {
        K::Negation::construct(self).map(Self::Negate)
    }

    pub(super) fn interact(left: Self, right: Self) -> Result<Self, ConstructionError> {
        K::interact(left, right)
    }

    pub(super) fn interact_fully(left: Self, right: Self) -> Result<Self, ConstructionError> {
        K::FullInteraction::construct((left, right)).map(Self::InteractFully)
    }

    pub(super) fn nest(left: Self, right: Self) -> Result<Self, ConstructionError> {
        K::Nesting::construct((left, right)).map(Self::Nesting)
    }

    pub(super) fn grouped(
        self,
        intercept: bool,
        first: Arc<formula::Factor>,
        rest: Vec<Arc<formula::Factor>>,
    ) -> Result<Self, ConstructionError> {
        K::Grouped::construct(Grouped {
            group: Box::new(self),
            intercept,
            first,
            rest,
        })
        .map(Self::Grouped)
    }

    pub(super) fn stepwise(call: StepwiseCall<Self>) -> Result<Self, ConstructionError> {
        K::Stepwise::construct(call).map(Self::Stepwise)
    }
}

/// Describes the expressions supported by arithmetic expansion.
pub(super) trait Arithmetic:
    Kind<
        Addition = Binary<Expression<Self>>,
        Subtraction = Binary<Expression<Self>, Expression<Self::Other>>,
        Negation = Box<Expression<Self::Other>>,
        Interaction = Binary<Expression<Self::Other>>,
        FullInteraction = Binary<Expression<Self::Other>>,
        Nesting = Binary<Expression<Self::Other>>,
        Grouped = Forbidden,
    >
{
    /// Kind used for the other arithmetic operand.
    type Other: Arithmetic;

    fn expand_operand(operand: &Self::Operand) -> Vec<formula::Expression>;

    fn expand_stepwise(call: &Self::Stepwise) -> Vec<formula::Expression>;

    fn expand_multistage(value: &Self::Multistage) -> Vec<formula::Expression>;
}

impl Arithmetic for Simple {
    type Other = Self;

    fn expand_operand(term: &formula::Term) -> Vec<formula::Expression> {
        vec![formula::Expression::from(term.clone())]
    }

    fn expand_stepwise(call: &StepwiseCall<Expression<Self>>) -> Vec<formula::Expression> {
        call.operator.expand(
            std::iter::once(call.first.as_ref())
                .chain(call.rest.iter())
                .map(Expression::<Self>::expand),
        )
    }

    fn expand_multistage(value: &Forbidden) -> Vec<formula::Expression> {
        match *value {}
    }
}

impl Arithmetic for Multistage {
    type Other = Simple;

    fn expand_operand(term: &formula::Term) -> Vec<formula::Expression> {
        vec![formula::Expression::from(term.clone())]
    }

    fn expand_stepwise(call: &StepwiseCall<Expression<Self>>) -> Vec<formula::Expression> {
        call.operator.expand(
            std::iter::once(call.first.as_ref())
                .chain(call.rest.iter())
                .map(Expression::<Self>::expand),
        )
    }

    fn expand_multistage(children: &Vec<formula::Formula>) -> Vec<formula::Expression> {
        children
            .iter()
            .cloned()
            .map(|child| formula::Expression::new([], vec![child]))
            .collect()
    }
}

impl Arithmetic for TermQuery {
    type Other = Self;

    fn expand_operand(term: &formula::Term) -> Vec<formula::Expression> {
        vec![formula::Expression::from(term.clone())]
    }

    fn expand_stepwise(call: &Forbidden) -> Vec<formula::Expression> {
        match *call {}
    }

    fn expand_multistage(value: &Forbidden) -> Vec<formula::Expression> {
        match *value {}
    }
}

impl<K: Arithmetic> Expression<K> {
    pub(super) fn expand(&self) -> Vec<formula::Expression> {
        expand_arithmetic(self)
    }

    pub(super) fn apply_to(&self, accumulated: formula::Expression) -> Vec<formula::Expression> {
        apply_arithmetic(self, accumulated)
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

impl Expression<VaryingSlope> {
    pub(super) fn expand(&self) -> Vec<formula::Expression> {
        match self {
            Self::Operand(group) => group
                .expand()
                .into_iter()
                .map(|expression| formula::encode_fixed_effects(expression, true, []))
                .collect(),
            Self::Add(node) => super::cartesian_product_with(
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
            Self::Multistage(impossible) => match *impossible {},
        }
    }
}

impl Expression<FixedEffect> {
    pub(super) fn expand(&self) -> Vec<formula::Expression> {
        match self {
            Self::Operand(variable) => vec![formula::Expression::from(formula::Term::from(
                formula::Factor::from(variable.name()),
            ))],

            Self::Add(node) => super::cartesian_product_with(
                &node.left.expand(),
                &node.right.expand(),
                |left, right| left.add(right),
            ),

            Self::Interact(node) => super::cartesian_product_with(
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
            Self::Multistage(impossible) => match *impossible {},
        }
    }
}

fn expand_arithmetic<K>(expression: &Expression<K>) -> Vec<formula::Expression>
where
    K: Arithmetic,
{
    match expression {
        Expression::Operand(operand) => K::expand_operand(operand),
        Expression::Multistage(child) => K::expand_multistage(child),
        Expression::Add(_) | Expression::Subtract(_) | Expression::Negate(_) => {
            apply_arithmetic(expression, formula::Expression::from_iter([]))
        }
        Expression::Interact(node) => super::cartesian_product_with(
            &node.left.expand(),
            &node.right.expand(),
            |left, right| left.interact(&right),
        ),
        Expression::InteractFully(node) => super::cartesian_product_with(
            &node.left.expand(),
            &node.right.expand(),
            |left, right| left.interact_fully(right),
        ),
        Expression::Nesting(node) => super::cartesian_product_with(
            &node.left.expand(),
            &node.right.expand(),
            |left, right| left.nest(right),
        ),
        Expression::Parenthesized(value) => expand_arithmetic(value),
        Expression::Stepwise(call) => K::expand_stepwise(call),
        Expression::Grouped(impossible) => match *impossible {},
    }
}

fn apply_arithmetic<K>(
    expression: &Expression<K>,
    accumulated: formula::Expression,
) -> Vec<formula::Expression>
where
    K: Arithmetic,
{
    match expression {
        Expression::Add(node) => apply_arithmetic(&node.left, accumulated)
            .into_iter()
            .flat_map(|value| apply_arithmetic(&node.right, value))
            .collect(),
        Expression::Subtract(node) => apply_arithmetic(&node.left, accumulated)
            .into_iter()
            .flat_map(|value| node.right.subtract_from(value))
            .collect(),
        Expression::Negate(value) => value.subtract_from(accumulated),
        _ => expand_arithmetic(expression)
            .into_iter()
            .map(|value| accumulated.clone().add(value))
            .collect(),
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

impl<Input, L, R> Construct<(Input, Input)> for Binary<L, R>
where
    L: Construct<Input>,
    R: Construct<Input>,
{
    fn construct((left, right): (Input, Input)) -> Result<Self, ConstructionError> {
        Ok(Self::new(L::construct(left)?, R::construct(right)?))
    }
}

pub(super) trait Construct<Input>: Sized {
    fn construct(input: Input) -> Result<Self, ConstructionError>;
}

impl<Input> Construct<Input> for Forbidden {
    fn construct(_input: Input) -> Result<Self, ConstructionError> {
        Err(ConstructionError::Unsupported)
    }
}

impl<Input, E: Construct<Input>> Construct<Input> for Box<E> {
    fn construct(input: Input) -> Result<Self, ConstructionError> {
        E::construct(input).map(Box::new)
    }
}

impl<E> Construct<StepwiseCall<E>> for StepwiseCall<E> {
    fn construct(call: Self) -> Result<Self, ConstructionError> {
        Ok(call)
    }
}

impl<Input, E: Construct<Input>> Construct<Grouped<Input>> for Grouped<E> {
    fn construct(input: Grouped<Input>) -> Result<Self, ConstructionError> {
        Ok(Self {
            group: Box::new(E::construct(*input.group)?),
            intercept: input.intercept,
            first: input.first,
            rest: input.rest,
        })
    }
}

impl Construct<formula::Term> for formula::Term {
    fn construct(term: Self) -> Result<Self, ConstructionError> {
        Ok(term)
    }
}

impl<K: Kind> Construct<Expression<K>> for Expression<K> {
    fn construct(value: Self) -> Result<Self, ConstructionError> {
        Ok(value)
    }
}

impl Construct<Vec<formula::Formula>> for Vec<formula::Formula> {
    fn construct(formulas: Self) -> Result<Self, ConstructionError> {
        Ok(formulas)
    }
}

impl Construct<formula::Term> for formula::VariableDefinition {
    fn construct(term: formula::Term) -> Result<Self, ConstructionError> {
        match term.as_factor().map(|factor| &factor.kind) {
            Some(formula::FactorKind::Variable(variable)) => Ok(variable.definition().clone()),
            _ => Err(ConstructionError::ExpectedFixedEffectColumn),
        }
    }
}

impl Construct<formula::Term> for Expression<FixedEffect> {
    fn construct(term: formula::Term) -> Result<Self, ConstructionError> {
        Self::term(term)
    }
}

impl Construct<Expression<Multistage>> for Expression<Simple> {
    fn construct(value: Expression<Multistage>) -> Result<Self, ConstructionError> {
        Ok(match value {
            Expression::Operand(term) => Self::Operand(term),
            Expression::Multistage(_) => {
                return Err(ConstructionError::MultistageArithmetic);
            }
            Expression::Add(node) => Self::Add(Binary::construct((*node.left, *node.right))?),
            Expression::Subtract(node) => {
                Self::Subtract(Binary::new(Self::construct(*node.left)?, *node.right))
            }
            Expression::Parenthesized(value) => Self::Parenthesized(Box::construct(*value)?),
            Expression::Stepwise(call) => Self::Stepwise(StepwiseCall {
                operator: call.operator,
                first: Box::construct(*call.first)?,
                rest: call
                    .rest
                    .into_iter()
                    .map(Self::construct)
                    .collect::<Result<Vec<_>, _>>()?,
            }),
            Expression::Negate(value) => Self::Negate(value),
            Expression::Interact(node) => Self::Interact(node),
            Expression::InteractFully(node) => Self::InteractFully(node),
            Expression::Nesting(node) => Self::Nesting(node),
            Expression::Grouped(impossible) => match impossible {},
        })
    }
}

impl Construct<Expression<VaryingSlope>> for Expression<FixedEffect> {
    fn construct(value: Expression<VaryingSlope>) -> Result<Self, ConstructionError> {
        Ok(match value {
            Expression::Operand(group) => group,
            Expression::Multistage(impossible) => match impossible {},
            Expression::Add(node) => Self::Add(Binary::construct((*node.left, *node.right))?),
            Expression::Parenthesized(value) => Self::Parenthesized(Box::construct(*value)?),
            Expression::Stepwise(call) => Self::Stepwise(StepwiseCall {
                operator: call.operator,
                first: Box::construct(*call.first)?,
                rest: call
                    .rest
                    .into_iter()
                    .map(Self::construct)
                    .collect::<Result<Vec<_>, _>>()?,
            }),
            Expression::Subtract(impossible) => match impossible {},
            Expression::Negate(impossible) => match impossible {},
            Expression::Interact(impossible) => match impossible {},
            Expression::InteractFully(impossible) => match impossible {},
            Expression::Nesting(impossible) => match impossible {},
            Expression::Grouped(_) => {
                return Err(ConstructionError::ExpectedFixedEffectGroup);
            }
        })
    }
}

#[derive(Debug, PartialEq)]
pub(super) enum ConstructionError {
    Unsupported,
    MultistageArithmetic,
    ExpectedFixedEffectColumn,
    ExpectedFixedEffectGroup,
}

impl std::fmt::Display for ConstructionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unsupported => f.write_str("operator is not supported"),
            Self::MultistageArithmetic => f.write_str("multistage formulas may only be added"),
            Self::ExpectedFixedEffectColumn => {
                f.write_str("fixed effects require column references")
            }
            Self::ExpectedFixedEffectGroup => {
                f.write_str("expected a grouping expression without loadings")
            }
        }
    }
}

impl std::error::Error for ConstructionError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formula::Term;

    mod term_query {
        use super::*;
        use crate::formula::Factor;
        use winnow::Parser;
        use winnow::error::ErrMode;

        type Expr = Expression<TermQuery>;

        fn term() -> Expr {
            Expr::term(Term::from(Factor::from("x"))).unwrap()
        }

        #[test]
        fn query_normalization_matches_simple_expressions() {
            for source in [
                "x",
                "log(x)",
                "C(group)",
                "x + x + z",
                "x - z",
                "x - x",
                "-x",
                "--x",
                "x - -z",
                "x:z",
                "x*z",
                "x/z",
                "(x + z)*w",
                "x - (z + w)",
                "x - (z - w)",
                "0 + x",
                "x - 1 - 0",
                "(0 + x)",
                "x + (-1 + z)",
            ] {
                let mut query_cursor = source;
                let query = crate::parse::parse_node::<TermQuery>(&mut query_cursor).unwrap();
                let mut formula_cursor = source;
                let existing = Expression::<Simple>::construct(
                    crate::parse::parse_node::<Multistage>(&mut formula_cursor).unwrap(),
                )
                .unwrap();

                assert!(query_cursor.is_empty(), "{source}");
                assert!(formula_cursor.is_empty(), "{source}");
                assert_eq!(query.expand(), existing.expand(), "{source}");
                assert_eq!(
                    query.apply_to(formula::Expression::from(Term::intercept())),
                    existing.apply_to(formula::Expression::from(Term::intercept())),
                    "with intercept: {source}"
                );
            }
        }

        #[test]
        fn constructors_preserve_arithmetic_nodes() {
            assert!(matches!(term(), Expr::Operand(_)));
            assert!(matches!(Expr::add(term(), term()), Ok(Expr::Add(_))));
            assert!(matches!(
                Expr::subtract(term(), term()),
                Ok(Expr::Subtract(_))
            ));
            assert!(matches!(Expr::negate(term()), Ok(Expr::Negate(_))));
            assert!(matches!(
                Expr::interact(term(), term()),
                Ok(Expr::Interact(_))
            ));
            assert!(matches!(
                Expr::interact_fully(term(), term()),
                Ok(Expr::InteractFully(_))
            ));
            assert!(matches!(Expr::nest(term(), term()), Ok(Expr::Nesting(_))));
        }

        #[test]
        fn constructors_reject_child_formulas_and_all_stepwise_operators() {
            let formulas = crate::parse::parse_formula_ast.parse("y ~ x").unwrap();
            assert!(matches!(
                Expr::multistage(formulas),
                Err(ConstructionError::Unsupported)
            ));

            for operator in [
                StepwiseOperator::Sw,
                StepwiseOperator::Sw0,
                StepwiseOperator::Csw,
                StepwiseOperator::Csw0,
            ] {
                assert!(matches!(
                    Expr::stepwise(StepwiseCall {
                        operator,
                        first: Box::new(term()),
                        rest: vec![],
                    }),
                    Err(ConstructionError::Unsupported)
                ));
            }
            assert_eq!(
                ConstructionError::Unsupported.to_string(),
                "operator is not supported"
            );
        }

        #[test]
        fn parser_accepts_supported_expressions() {
            for source in [
                "x", "log(x)", "x + z", "x - z", "-x", "x:z", "x*z", "x/z", "(x + z)",
            ] {
                let mut cursor = source;
                assert!(
                    crate::parse::parse_node::<TermQuery>(&mut cursor).is_ok(),
                    "{source}"
                );
                assert!(cursor.is_empty(), "{source}");
            }
        }

        #[test]
        fn parser_commits_to_rejecting_nested_stepwise_calls() {
            for operator in ["sw", "sw0", "csw", "csw0"] {
                for source in [
                    format!("{operator}(x)"),
                    format!("{operator}(x, z)"),
                    format!("({operator}(x))"),
                    format!("x + {operator}(z)"),
                    format!("x - {operator}(z)"),
                    format!("-{operator}(x)"),
                    format!("x:{operator}(z)"),
                    format!("x*{operator}(z)"),
                    format!("x/{operator}(z)"),
                ] {
                    let mut cursor = source.as_str();
                    match crate::parse::parse_node::<TermQuery>(&mut cursor) {
                        Err(ErrMode::Cut(error)) => assert!(
                            error.to_string().contains("operator is not supported"),
                            "{source}: {error}"
                        ),
                        _ => panic!("expected committed stepwise construction error for {source}"),
                    }
                }
            }
        }

        #[test]
        fn parser_rejects_nested_child_formulas() {
            for source in ["[x ~ z]", "([x ~ z])", "x + [z ~ w]", "x:[z ~ w]"] {
                let mut cursor = source;
                match crate::parse::parse_node::<TermQuery>(&mut cursor) {
                    Err(ErrMode::Cut(error)) => assert!(
                        error.to_string().contains("operator is not supported"),
                        "{source}: {error}"
                    ),
                    _ => panic!("expected committed child-formula construction error for {source}"),
                }
            }
        }
    }

    mod conversion {
        use super::*;
        use winnow::Parser;

        fn term() -> Expression<Multistage> {
            Expression::Operand(Term::intercept())
        }

        fn nested() -> Expression<Multistage> {
            Expression::Multistage(crate::parse::parse_formula_ast.parse("y ~ x").unwrap())
        }

        #[test]
        fn conversion_preserves_grouped_subtraction() {
            let expression: Expression<Multistage> =
                Expression::Parenthesized(Box::new(Expression::Subtract(Binary {
                    left: Box::new(term()),
                    right: Box::new(Expression::Operand(Term::intercept())),
                })));

            let converted = Expression::<Simple>::construct(expression).unwrap();
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
                    operator: StepwiseOperator::Sw,
                    first: Box::new(term()),
                    rest: vec![nested()],
                }),
            ];

            for expression in expressions {
                assert!(matches!(
                    Expression::<Simple>::construct(expression),
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
                    Expression::Operand(actual) if actual.name() == name
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

            let column = || Expr::term(Term::from(Factor::from("firm"))).unwrap();

            assert!(matches!(Expr::add(column(), column()), Ok(Expr::Add(_))));
            assert!(matches!(
                Expr::interact(column(), column()),
                Ok(Expr::Interact(_))
            ));

            let forbidden: [BinaryConstructor; 3] =
                [Expr::subtract, Expr::interact_fully, Expr::nest];

            for construct in forbidden {
                assert!(matches!(
                    construct(column(), column()),
                    Err(ConstructionError::Unsupported)
                ));
            }

            assert!(matches!(
                column().negate(),
                Err(ConstructionError::Unsupported)
            ));
        }
    }
}
