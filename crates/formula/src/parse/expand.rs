use crate::parse::ast::{
    ExpressionAst, ExpressionKind, FixedEffectGroup, Forbidden, FormulaAst, ParseKind, ParsedNode,
    StepwiseOperator,
};
use crate::{Factor, formula};

fn cartesian_product_with<L: Clone, R: Clone, O>(
    left: &[L],
    right: &[R],
    combine: impl Fn(L, R) -> O,
) -> Vec<O> {
    let mut alternatives = Vec::new();
    for left in left {
        for right in right {
            alternatives.push(combine(left.clone(), right.clone()));
        }
    }
    alternatives
}

impl FormulaAst {
    pub(in crate::parse) fn expand(&self) -> Vec<crate::Formula> {
        let left = self.left.expand();

        let right = self
            .right
            .apply_to(formula::Expression::from(formula::Term::intercept()));

        let fixed_effects = self
            .fixed_effects
            .as_ref()
            .map(|expression| expression.expand())
            .unwrap_or_else(|| vec![formula::Expression::from_iter([])]);

        let left_right = cartesian_product_with(&left, &right, |left, right| (left, right));

        cartesian_product_with(
            &left_right,
            &fixed_effects,
            |(left, right), fixed_effects| {
                let right = if fixed_effects.has_fixed_effect_intercept() {
                    right.without_intercept()
                } else {
                    right
                };

                formula::Formula {
                    expressions: std::collections::BTreeMap::from([
                        (crate::Role::Dependent, left),
                        (crate::Role::Independent, right),
                        (crate::Role::FixedEffect, fixed_effects),
                    ]),
                }
            },
        )
    }
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
                    prefixes = cartesian_product_with(&prefixes, &argument, |prefix, value| {
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

pub(super) trait OperandExpansion {
    fn expand_operand(&self) -> Vec<formula::Expression>;

    fn apply_operand(&self, accumulated: formula::Expression) -> Vec<formula::Expression> {
        self.expand_operand()
            .into_iter()
            .map(|value| accumulated.clone().add(value))
            .collect()
    }

    fn subtract_operand(&self, accumulated: formula::Expression) -> Vec<formula::Expression> {
        self.expand_operand()
            .into_iter()
            .map(|value| accumulated.clone().subtract(value))
            .collect()
    }
}

impl OperandExpansion for formula::Term {
    fn expand_operand(&self) -> Vec<formula::Expression> {
        vec![formula::Expression::from(self.clone())]
    }
}

impl OperandExpansion for formula::VariableDefinition {
    fn expand_operand(&self) -> Vec<formula::Expression> {
        vec![formula::Expression::from(formula::Term::from(
            Factor::from(self.name()),
        ))]
    }
}

impl OperandExpansion for Forbidden {
    fn expand_operand(&self) -> Vec<formula::Expression> {
        match *self {}
    }
}

impl<K: ExpressionKind> OperandExpansion for FixedEffectGroup<K>
where
    K::Operand: OperandExpansion,
{
    fn expand_operand(&self) -> Vec<formula::Expression> {
        self.group
            .expand()
            .into_iter()
            .map(|expression| {
                formula::encode_fixed_effects(
                    expression,
                    self.intercept,
                    self.loadings.iter().cloned(),
                )
            })
            .collect()
    }
}

impl<K: ExpressionKind> OperandExpansion for ExpressionAst<K>
where
    K::Operand: OperandExpansion,
{
    fn expand_operand(&self) -> Vec<formula::Expression> {
        self.expand()
    }

    fn apply_operand(&self, accumulated: formula::Expression) -> Vec<formula::Expression> {
        self.apply_to(accumulated)
    }

    fn subtract_operand(&self, accumulated: formula::Expression) -> Vec<formula::Expression> {
        self.subtract_from(accumulated)
    }
}

impl<K: ExpressionKind> ExpressionAst<K>
where
    K::Operand: OperandExpansion,
{
    pub(in crate::parse) fn expand(&self) -> Vec<formula::Expression> {
        match self {
            Self::Operand(operand) => operand.expand_operand(),

            Self::Add(_, _) | Self::Subtract(_, _) | Self::Negate(_, _) => {
                self.apply_to(formula::Expression::from_iter([]))
            }

            Self::Interact(_, operands) => cartesian_product_with(
                &operands.left.expand(),
                &operands.right.expand(),
                |left, right| left.interact(&right),
            ),

            Self::InteractFully(_, operands) => cartesian_product_with(
                &operands.left.expand(),
                &operands.right.expand(),
                |left, right| left.interact_fully(right),
            ),

            Self::Nesting(_, operands) => cartesian_product_with(
                &operands.left.expand(),
                &operands.right.expand(),
                |left, right| left.nest(right),
            ),

            Self::Parenthesized(expression) => expression.expand(),

            Self::Stepwise(_, call) => call.operator.expand(
                std::iter::once(&call.first)
                    .chain(call.rest.iter())
                    .map(Self::expand),
            ),

            Self::Child(_, formula) => formula
                .expand()
                .into_iter()
                .map(|child| formula::Expression::new([], vec![child]))
                .collect(),
        }
    }

    fn apply_to(&self, accumulated: formula::Expression) -> Vec<formula::Expression> {
        match self {
            Self::Operand(operand) => operand.apply_operand(accumulated),

            Self::Add(_, operands) => operands
                .left
                .apply_to(accumulated)
                .into_iter()
                .flat_map(|value| operands.right.apply_to(value))
                .collect(),

            Self::Subtract(_, operands) => operands
                .left
                .apply_to(accumulated)
                .into_iter()
                .flat_map(|value| operands.right.subtract_from(value))
                .collect(),

            Self::Negate(_, expression) => expression.subtract_from(accumulated),

            _ => self
                .expand()
                .into_iter()
                .map(|value| accumulated.clone().add(value))
                .collect(),
        }
    }

    fn subtract_from(&self, accumulated: formula::Expression) -> Vec<formula::Expression> {
        match self {
            Self::Operand(operand) => operand.subtract_operand(accumulated),

            Self::Negate(_, expression) => expression.apply_to(accumulated),

            _ => self
                .expand()
                .into_iter()
                .map(|value| accumulated.clone().subtract(value))
                .collect(),
        }
    }
}

impl<K: ParseKind> ParsedNode<K>
where
    K::Operand: OperandExpansion,
    <K::Embedded as ExpressionKind>::Operand: OperandExpansion,
{
    pub(in crate::parse) fn expand(&self) -> Vec<formula::Expression> {
        match self {
            Self::Flat(expression) => expression.expand(),
            Self::Embedded(expression) => expression.expand(),
        }
    }

    pub(in crate::parse) fn apply_to(
        &self,
        accumulated: formula::Expression,
    ) -> Vec<formula::Expression> {
        match self {
            Self::Flat(expression) => expression.apply_to(accumulated),
            Self::Embedded(expression) => expression.apply_to(accumulated),
        }
    }
}
