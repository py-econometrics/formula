//! Direct readers for typed operand values. Layer transitions are AST structure,
//! not parser callbacks or conversions from a second operand enum.
use super::ast::{
    ConstructionError, ExpressionAst, ExpressionKind, FixedEffect, Forbidden, FormulaAst, Loaded,
};
use super::{construction_result, parse_factor, parse_formula_ast, parse_loadings};
use crate::formula::VariableDefinition;
use crate::{Factor, Term};
use std::sync::Arc;
use winnow::ascii::multispace0;
use winnow::combinator::{Postfix, alt, backtrack_err, cut_err, fail, preceded, terminated};
use winnow::error::{ContextError, ErrMode, StrContext, StrContextValue};
use winnow::token::take_while;
use winnow::{ModalResult, Parser};
const LOADING_PRECEDENCE: i64 = 25;

/// Read an operand value. The flag records the shorthand `0` for `-1`;
/// negation is applied by the containing expression, according to its permissions.
pub(super) trait ReadValue: Sized {
    fn read(cursor: &mut &str) -> ModalResult<(Self, bool)>;
}
impl ReadValue for Forbidden {
    fn read(cursor: &mut &str) -> ModalResult<(Self, bool)> {
        fail.parse_next(cursor)
    }
}
impl ReadValue for Term {
    fn read(cursor: &mut &str) -> ModalResult<(Self, bool)> {
        alt((
            take_while(1.., super::is_identifier_character)
                .verify(|name: &str| matches!(name, "0" | "1"))
                .map(|name| (Term::intercept(), name == "0")),
            parse_factor.map(|factor| (Term::from(factor), false)),
        ))
        .parse_next(cursor)
    }
}
impl ReadValue for VariableDefinition {
    fn read(cursor: &mut &str) -> ModalResult<(Self, bool)> {
        let (term, negate) = Term::read(cursor)?;
        construction_result(
            cursor,
            Self::try_from(term).map_err(|_| ConstructionError::Unsupported),
        )
        .map(|value| (value, negate))
    }
}
impl ReadValue for FormulaAst {
    fn read(cursor: &mut &str) -> ModalResult<(Self, bool)> {
        preceded(
            '[',
            cut_err(terminated(
                parse_formula_ast,
                ']'.context(StrContext::Expected(StrContextValue::Description(
                    "']' to close the nested formula",
                ))),
            )),
        )
        .map(|value| (value, false))
        .parse_next(cursor)
    }
}
impl<I: ParseLayer> ReadValue for ExpressionAst<I> {
    fn read(cursor: &mut &str) -> ModalResult<(Self, bool)> {
        super::parse_operand::<I>(cursor).map(|value| (value, false))
    }
}
impl<I: ParseLayer> ReadValue for FixedEffect<I> {
    fn read(cursor: &mut &str) -> ModalResult<(Self, bool)> {
        ExpressionAst::<I>::read(cursor).map(|(group, _)| (Self { group }, false))
    }
}
impl<I: ParseLayer> ReadValue for Loaded<FixedEffect<I>> {
    fn read(cursor: &mut &str) -> ModalResult<(Self, bool)> {
        // Read the entire nested value, just like a child formula. The group's
        // tighter operators belong inside it; sums remain in the enclosing layer.
        // Until '[' is found this is speculative, including grouped/stepwise input.
        let group = backtrack_err(|cursor: &mut &str| {
            super::parse_node_at::<I>(cursor, LOADING_PRECEDENCE + 1)
        })
        .parse_next(cursor)?;
        let intercept = alt(("[[".value(false), '['.value(true))).parse_next(cursor)?;
        let (first_loading, rest_loadings) = loading_tail(cursor, intercept)?;
        Ok((
            Self {
                group: FixedEffect { group },
                intercept,
                first_loading,
                rest_loadings,
            },
            false,
        ))
    }
}

pub(super) trait ParseLayer: ExpressionKind<Operand: ReadValue, Nested: ReadValue> {}
impl<K: ExpressionKind<Operand: ReadValue, Nested: ReadValue>> ParseLayer for K {}

pub(super) fn nested<K: ParseLayer>(cursor: &mut &str) -> ModalResult<ExpressionAst<K>> {
    let (nested, negate) = K::Nested::read(cursor)?;
    let value = ExpressionAst::Nested(Box::new(nested));
    if negate {
        construction_result(cursor, value.negate())
    } else {
        Ok(value)
    }
}
pub(super) fn primary<K: ParseLayer>(cursor: &mut &str) -> ModalResult<ExpressionAst<K>> {
    let (operand, negate) = K::Operand::read(cursor)?;
    let value = ExpressionAst::Operand(operand);
    if negate {
        construction_result(cursor, value.negate())
    } else {
        Ok(value)
    }
}

fn loading_tail(
    cursor: &mut &str,
    intercept: bool,
) -> ModalResult<(Arc<Factor>, Vec<Arc<Factor>>)> {
    if intercept {
        cut_err(terminated(parse_loadings, (']', multispace0))).parse_next(cursor)
    } else {
        cut_err(terminated(parse_loadings, ("]]", multispace0))).parse_next(cursor)
    }
}

// These two diagnostics are shared by every layer. Supported productions have
// already been read as typed values; a remaining occurrence is inadmissible.
pub(super) fn reject_child<K: ParseLayer>(cursor: &mut &str) -> ModalResult<ExpressionAst<K>> {
    FormulaAst::read(cursor)?;
    construction_result(cursor, Err(ConstructionError::Unsupported))
}
type PostfixRule<'i, K> = Postfix<&'i str, ExpressionAst<K>, ErrMode<ContextError>>;
pub(super) fn postfix<'i, K: ParseLayer>(cursor: &mut &'i str) -> ModalResult<PostfixRule<'i, K>> {
    alt((
        "[[".value(Postfix(LOADING_PRECEDENCE, reject_loading::<K, false>)),
        '['.value(Postfix(LOADING_PRECEDENCE, reject_loading::<K, true>)),
    ))
    .parse_next(cursor)
}
fn reject_loading<K: ParseLayer, const INTERCEPT: bool>(
    cursor: &mut &str,
    _: ExpressionAst<K>,
) -> ModalResult<ExpressionAst<K>> {
    loading_tail(cursor, INTERCEPT)?;
    construction_result(cursor, Err(ConstructionError::Unsupported))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::ast::{
        FixedEffectExpression, FixedEffectOperations, FormulaOperations, Layer,
        MultistageOperations, Permitted, SimpleFormulaExpression,
    };
    enum Wrapped {
        Child(FormulaAst),
        Loaded(Loaded<FixedEffect<SimpleFormulaExpression>>),
    }
    impl ReadValue for Wrapped {
        fn read(cursor: &mut &str) -> ModalResult<(Self, bool)> {
            alt((
                FormulaAst::read.map(|(child, negate)| (Self::Child(child), negate)),
                Loaded::<FixedEffect<SimpleFormulaExpression>>::read
                    .map(|(loaded, negate)| (Self::Loaded(loaded), negate)),
            ))
            .parse_next(cursor)
        }
    }
    type Additive = Layer<
        ExpressionAst<SimpleFormulaExpression>,
        SimpleFormulaExpression,
        MultistageOperations,
        Permitted,
        Wrapped,
    >;
    type Algebraic = Layer<
        ExpressionAst<SimpleFormulaExpression>,
        SimpleFormulaExpression,
        FormulaOperations,
        Permitted,
        Wrapped,
    >;

    #[test]
    fn typed_operands_are_reusable_across_operator_vocabularies() {
        for source in ["x*y + [a ~ z]", "x*y + z[t]", "x*y + (z:w)[t]"] {
            let ExpressionAst::Add(_, pair) =
                super::super::parse_node::<Additive>.parse(source).unwrap()
            else {
                panic!("expected outer addition: {source}");
            };
            let ExpressionAst::Operand(inner) = *pair.left else {
                panic!("inner subtree");
            };
            assert!(matches!(inner, ExpressionAst::InteractFully(_, _)));
        }
        assert!(
            super::super::parse_node::<Additive>
                .parse("[a ~ x] * y[t]")
                .is_err()
        );
        let ExpressionAst::InteractFully(_, pair) = super::super::parse_node::<Algebraic>
            .parse("[a ~ x] * y[t]")
            .unwrap()
        else {
            panic!("outer interaction");
        };
        let ExpressionAst::Nested(left) = *pair.left else {
            panic!("nested child");
        };
        let Wrapped::Child(child) = *left else {
            panic!("child value");
        };
        assert!(matches!(child.right, ExpressionAst::Operand(_)));
    }

    #[test]
    fn typed_loading_preserves_binding_and_recursive_grammar() {
        let ExpressionAst::Add(_, pair) = super::super::parse_node::<Additive>
            .parse("x + y:z[t]")
            .unwrap()
        else {
            panic!("outer addition");
        };
        let ExpressionAst::Nested(wrapped) = *pair.right else {
            panic!("nested value");
        };
        let Wrapped::Loaded(loaded) = *wrapped else {
            panic!("loaded value");
        };
        assert!(matches!(loaded.group.group, ExpressionAst::Interact(_, _)));
        assert!(matches!(
            super::super::parse_node::<Additive>
                .parse("sw([a ~ x], (y:z)[t])")
                .unwrap(),
            ExpressionAst::Stepwise(_, _)
        ));
        for source in ["[a ~ x][t]", "x[t][u]", "([a ~ x] + y)[t]", "sw(x[t], y):z"] {
            assert!(
                super::super::parse_node::<Additive>.parse(source).is_err(),
                "{source}"
            );
        }
    }

    #[test]
    fn another_layer_reuses_the_entire_fixed_effect_grammar() {
        type Envelope = Layer<
            ExpressionAst<FixedEffectExpression>,
            FixedEffectExpression,
            FixedEffectOperations,
            Permitted,
        >;
        for source in [
            "firm:year + sector[t]",
            "sw(firm, year)[t]",
            "(firm[t] + year)",
        ] {
            let expression = super::super::parse_node::<Envelope>.parse(source).unwrap();
            assert!(expression.into_inner().is_ok());
        }
    }
}
