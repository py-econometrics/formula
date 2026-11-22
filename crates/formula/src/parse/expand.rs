use crate::formula::VaryingSlopes;
use crate::parse::ast::{ExpressionAst, Forbidden, FormulaAst, Layer, StepwiseOperator};
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
                let mut steps = Vec::new();

                for argument in arguments {
                    prefixes = cartesian_product_with(&prefixes, &argument, |prefix, value| {
                        prefix.add(value)
                    });

                    steps.extend(prefixes.iter().cloned());
                }

                steps
            }

            Self::Mvsw => {
                let mut steps = vec![formula::Expression::from_iter([])];
                for argument in arguments {
                    let included = cartesian_product_with(&steps, &argument, |prefix, value| {
                        prefix.add(value)
                    });

                    steps.extend(included);
                }
                steps
            }
        };

        if matches!(self, Self::Sw0 | Self::Csw0) {
            alternatives.insert(0, formula::Expression::from_iter([]));
        }

        alternatives
    }
}

pub(super) trait AtomExpansion {
    fn expand_atom(&self) -> Vec<formula::Expression>;
}

impl AtomExpansion for formula::Term {
    fn expand_atom(&self) -> Vec<formula::Expression> {
        vec![formula::Expression::from(self.clone())]
    }
}

impl AtomExpansion for formula::VariableDefinition {
    fn expand_atom(&self) -> Vec<formula::Expression> {
        vec![formula::Expression::from(formula::Term::from(
            Factor::from(self.name()),
        ))]
    }
}

impl AtomExpansion for Forbidden {
    fn expand_atom(&self) -> Vec<formula::Expression> {
        match *self {}
    }
}

impl AtomExpansion for Box<FormulaAst> {
    fn expand_atom(&self) -> Vec<formula::Expression> {
        FormulaAst::expand(self)
            .into_iter()
            .map(|child| formula::Expression::new([], vec![child]))
            .collect()
    }
}

pub(super) trait Expand {
    fn expand(&self) -> Vec<formula::Expression>;
    fn apply_to(&self, accumulated: formula::Expression) -> Vec<formula::Expression>;
    fn subtract_from(&self, accumulated: formula::Expression) -> Vec<formula::Expression>;
}

impl Expand for Forbidden {
    fn expand(&self) -> Vec<formula::Expression> {
        match *self {}
    }
    fn apply_to(&self, _: formula::Expression) -> Vec<formula::Expression> {
        match *self {}
    }
    fn subtract_from(&self, _: formula::Expression) -> Vec<formula::Expression> {
        match *self {}
    }
}

/// Describes how an inner expression expands in its layer.
pub(super) trait EmbeddedExpansion: Sized {
    fn expand(inner: &impl Expand, annotation: Option<&Self>) -> Vec<formula::Expression>;

    fn apply(
        inner: &impl Expand,
        annotation: Option<&Self>,
        accumulated: formula::Expression,
    ) -> Vec<formula::Expression> {
        Self::expand(inner, annotation)
            .into_iter()
            .map(|value| accumulated.clone().add(value))
            .collect()
    }

    fn subtract(
        inner: &impl Expand,
        annotation: Option<&Self>,
        accumulated: formula::Expression,
    ) -> Vec<formula::Expression> {
        Self::expand(inner, annotation)
            .into_iter()
            .map(|value| accumulated.clone().subtract(value))
            .collect()
    }
}

impl EmbeddedExpansion for Forbidden {
    fn expand(inner: &impl Expand, _: Option<&Self>) -> Vec<formula::Expression> {
        inner.expand()
    }

    fn apply(
        inner: &impl Expand,
        _: Option<&Self>,
        accumulated: formula::Expression,
    ) -> Vec<formula::Expression> {
        inner.apply_to(accumulated)
    }

    fn subtract(
        inner: &impl Expand,
        _: Option<&Self>,
        accumulated: formula::Expression,
    ) -> Vec<formula::Expression> {
        inner.subtract_from(accumulated)
    }
}

impl EmbeddedExpansion for VaryingSlopes {
    fn expand(inner: &impl Expand, slopes: Option<&Self>) -> Vec<formula::Expression> {
        inner
            .expand()
            .into_iter()
            .map(|group| formula::encode_fixed_effects(group, slopes))
            .collect()
    }
}

impl<L: Layer> Expand for ExpressionAst<L>
where
    L::Atom: AtomExpansion,
    L::Inner: Expand,
    L::Annotation: EmbeddedExpansion,
{
    fn expand(&self) -> Vec<formula::Expression> {
        match self {
            Self::Atom(atom) => atom.expand_atom(),
            Self::Embedded(inner, annotation) => L::Annotation::expand(inner, annotation.as_ref()),

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
        }
    }

    fn apply_to(&self, accumulated: formula::Expression) -> Vec<formula::Expression> {
        match self {
            Self::Embedded(inner, annotation) => {
                L::Annotation::apply(inner, annotation.as_ref(), accumulated)
            }

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
            Self::Embedded(inner, annotation) => {
                L::Annotation::subtract(inner, annotation.as_ref(), accumulated)
            }

            Self::Negate(_, expression) => expression.apply_to(accumulated),

            _ => self
                .expand()
                .into_iter()
                .map(|value| accumulated.clone().subtract(value))
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::ast::{Binary, MultistageFormula, Permitted, SimpleFormula};

    type Ast = ExpressionAst<SimpleFormula>;

    fn term(name: &str) -> formula::Term {
        formula::Term::from(Factor::from(name))
    }

    fn atom(name: &str) -> Ast {
        Ast::Atom(term(name))
    }

    fn negate(value: Ast) -> Ast {
        Ast::Negate(Permitted, Box::new(value))
    }

    fn subtract(left: Ast, right: Ast) -> Ast {
        Ast::Subtract(Permitted, Binary::new(left, right))
    }

    fn parenthesized(value: Ast) -> Ast {
        Ast::Parenthesized(Box::new(value))
    }

    fn value(terms: impl IntoIterator<Item = formula::Term>) -> formula::Expression {
        terms.into_iter().collect()
    }

    #[test]
    fn negation_and_parentheses_preserve_subtraction_scope() {
        for (node, local, applied) in [
            (
                negate(negate(atom("x"))),
                value([term("x")]),
                value([formula::Term::intercept(), term("x")]),
            ),
            (
                negate(parenthesized(negate(atom("x")))),
                value([]),
                value([formula::Term::intercept()]),
            ),
            (
                subtract(atom("x"), negate(atom("z"))),
                value([term("x"), term("z")]),
                value([formula::Term::intercept(), term("x"), term("z")]),
            ),
            (
                subtract(atom("x"), parenthesized(negate(atom("z")))),
                value([term("x")]),
                value([formula::Term::intercept(), term("x")]),
            ),
        ] {
            assert_eq!(node.expand(), [local]);
            assert_eq!(
                node.apply_to(value([formula::Term::intercept()])),
                [applied]
            );
        }
    }

    #[test]
    fn embedded_simple_expressions_are_transparent_until_parenthesized() {
        type Outer = ExpressionAst<MultistageFormula>;
        let seed = value([formula::Term::intercept()]);
        let node = Outer::embed(subtract(atom("x"), Ast::Atom(formula::Term::intercept())));
        assert_eq!(node.apply_to(seed.clone()), [value([term("x")])]);

        let grouped = Outer::Parenthesized(Box::new(node));
        assert_eq!(
            grouped.apply_to(seed.clone()),
            [value([formula::Term::intercept(), term("x")])]
        );

        // Subtracting an embedded negation adds its operand to the accumulator.
        let negated = Outer::embed(negate(atom("x")));
        assert_eq!(
            negated.subtract_from(seed),
            [value([formula::Term::intercept(), term("x")])]
        );
    }
}
