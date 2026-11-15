//! Operand grammars and their composition into an expression layer.

use super::ast::{
    ConstructionError, ExpressionAst, ExpressionKind, FixedEffectGroup, Forbidden,
    MultistageOperand, ParsedNode,
};
use super::{construction_result, parse_factor, parse_formula_ast, parse_loadings};
use crate::formula::VariableDefinition;
use crate::{Factor, Term};
use std::sync::Arc;
use winnow::ascii::multispace0;
use winnow::combinator::{Postfix, alt, cut_err, fail, peek, preceded, terminated};
use winnow::error::{ContextError, ErrMode, StrContext, StrContextValue};
use winnow::token::take_while;
use winnow::{ModalResult, Parser};

/// `0` spells a negated intercept. Applying the negation is expression
/// construction, so operand parsing records it without constructing a node.
pub(super) struct Primary<O> {
    operand: O,
    negate: bool,
}

impl<O> From<O> for Primary<O> {
    fn from(operand: O) -> Self {
        Self {
            operand,
            negate: false,
        }
    }
}

/// Parse an operand without knowing its containing layer or operator vocabulary.
/// A postfix constructs this operand from an expression in the inner layer.
pub(super) trait OperandSyntax<I: ExpressionKind>: Sized {
    fn primary(cursor: &mut &str) -> ModalResult<Primary<Self>> {
        fail.parse_next(cursor)
    }

    fn postfix(cursor: &mut &str) -> ModalResult<OperandPostfix<I, Self>> {
        fail.parse_next(cursor)
    }
}

impl OperandSyntax<Forbidden> for Forbidden {}

impl OperandSyntax<Forbidden> for Term {
    fn primary(cursor: &mut &str) -> ModalResult<Primary<Self>> {
        alt((
            take_while(1.., super::is_identifier_character)
                .verify(|name: &str| matches!(name, "0" | "1"))
                .map(|name| Primary {
                    operand: Term::intercept(),
                    negate: name == "0",
                }),
            parse_factor.map(|factor| Primary::from(Term::from(factor))),
        ))
        .parse_next(cursor)
    }
}

impl OperandSyntax<Forbidden> for VariableDefinition {
    fn primary(cursor: &mut &str) -> ModalResult<Primary<Self>> {
        let primary = Term::primary(cursor)?;
        let operand = construction_result(
            cursor,
            Self::try_from(primary.operand).map_err(|_| ConstructionError::Unsupported),
        )?;
        Ok(Primary {
            operand,
            negate: primary.negate,
        })
    }
}

impl<I: ExpressionKind> OperandSyntax<I> for MultistageOperand<I> {
    fn primary(cursor: &mut &str) -> ModalResult<Primary<Self>> {
        child_formula(cursor).map(|formula| Self::Child(Box::new(formula)).into())
    }
}

/// Unlike a Pratt postfix, this constructs an operand from an inner expression.
/// The shared adapter handles the change between expression layers.
pub(super) struct OperandPostfix<I: ExpressionKind, O> {
    precedence: i64,
    finish: fn(&mut &str, ExpressionAst<I>) -> ModalResult<O>,
}

impl<I: ExpressionKind> OperandSyntax<I> for FixedEffectGroup<I> {
    fn postfix(cursor: &mut &str) -> ModalResult<OperandPostfix<I, Self>> {
        alt((
            "[[".map(|_| OperandPostfix {
                precedence: 25,
                finish: loaded_group::<I, false>,
            }),
            '['.map(|_| OperandPostfix {
                precedence: 25,
                finish: loaded_group::<I, true>,
            }),
        ))
        .parse_next(cursor)
    }
}

fn loaded_group<I: ExpressionKind, const INTERCEPT: bool>(
    cursor: &mut &str,
    group: ExpressionAst<I>,
) -> ModalResult<FixedEffectGroup<I>> {
    let loadings = loading_tail::<INTERCEPT>(cursor)?;
    Ok(FixedEffectGroup {
        group,
        intercept: INTERCEPT,
        loadings,
    })
}

/// A shorthand for the operand-parser bounds, derived for every eligible kind.
/// There are no grammar choices or per-layer implementations here.
/// ParsedNode tracks one promotion boundary, so its inner layer must be a leaf;
/// arbitrary stacks would require recursive parsing state, not more grammars.
pub(super) trait ParseLayer:
    ExpressionKind<
        Operand: OperandSyntax<Self::Inner>,
        Inner: ExpressionKind<Inner = Forbidden, Operand: OperandSyntax<Forbidden>>,
    >
{
}

impl<K> ParseLayer for K
where
    K: ExpressionKind<Operand: OperandSyntax<K::Inner>>,
    K::Inner: ExpressionKind<Inner = Forbidden, Operand: OperandSyntax<Forbidden>>,
{
}

pub(super) fn primary<K: ParseLayer>(cursor: &mut &str) -> ModalResult<ParsedNode<K>> {
    let outer = K::Operand::primary.map(|primary| {
        (
            ParsedNode::Outer(ExpressionAst::Operand(primary.operand)),
            primary.negate,
        )
    });
    let inner = <K::Inner as ExpressionKind>::Operand::primary.map(|primary| {
        (
            ParsedNode::Inner(ExpressionAst::Operand(primary.operand)),
            primary.negate,
        )
    });
    let reject_child = |cursor: &mut &str| {
        child_formula(cursor)?;
        construction_result(cursor, Err(ConstructionError::Unsupported))
    };
    let (node, negate) = alt((outer, inner, reject_child)).parse_next(cursor)?;
    if negate {
        construction_result(cursor, node.negate())
    } else {
        Ok(node)
    }
}

type PostfixRule<'i, K> = Postfix<&'i str, ParsedNode<K>, ErrMode<ContextError>>;

pub(super) fn postfix<'i, K: ParseLayer>(cursor: &mut &'i str) -> ModalResult<PostfixRule<'i, K>> {
    // Winnow's postfix callback is a function pointer with the same input and
    // output type. Peek at the operand rule here, then read it in the callback
    // that adapts the inner expression to an outer operand.
    let supported = peek(K::Operand::postfix).map(|rule| {
        Postfix(rule.precedence, |cursor, node: ParsedNode<K>| {
            let rule = K::Operand::postfix(cursor)?;
            let inner = construction_result(cursor, node.into_inner())?;
            (rule.finish)(cursor, inner)
                .map(|operand| ParsedNode::Outer(ExpressionAst::Operand(operand)))
        })
    });
    // Recognize unsupported loading syntax too, retaining its syntax diagnostics.
    alt((
        supported,
        "[[".value(Postfix(25, reject_loadings::<K, false>)),
        '['.value(Postfix(25, reject_loadings::<K, true>)),
    ))
    .parse_next(cursor)
}

fn child_formula(cursor: &mut &str) -> ModalResult<super::ast::FormulaAst> {
    preceded(
        '[',
        cut_err(terminated(
            parse_formula_ast,
            ']'.context(StrContext::Expected(StrContextValue::Description(
                "']' to close the nested formula",
            ))),
        )),
    )
    .parse_next(cursor)
}

fn loading_tail<const INTERCEPT: bool>(cursor: &mut &str) -> ModalResult<Vec<Arc<Factor>>> {
    let (first, rest) = if INTERCEPT {
        cut_err(terminated(parse_loadings, (']', multispace0))).parse_next(cursor)?
    } else {
        cut_err(terminated(parse_loadings, ("]]", multispace0))).parse_next(cursor)?
    };
    Ok(std::iter::once(first).chain(rest).collect())
}

fn reject_loadings<K: ExpressionKind, const INTERCEPT: bool>(
    cursor: &mut &str,
    _node: ParsedNode<K>,
) -> ModalResult<ParsedNode<K>> {
    loading_tail::<INTERCEPT>(cursor)?;
    construction_result(cursor, Err(ConstructionError::Unsupported))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::ast::{
        FormulaOperations, Layer, MultistageOperations, Permitted, SimpleFormulaExpression,
    };

    struct Wrapped(ExpressionAst<SimpleFormulaExpression>);

    impl From<ExpressionAst<SimpleFormulaExpression>> for Wrapped {
        fn from(inner: ExpressionAst<SimpleFormulaExpression>) -> Self {
            Self(inner)
        }
    }

    // These parsers know neither the outer expression kind nor ParsedNode.
    impl OperandSyntax<SimpleFormulaExpression> for Wrapped {
        fn primary(cursor: &mut &str) -> ModalResult<Primary<Self>> {
            let factor = preceded('@', cut_err(parse_factor)).parse_next(cursor)?;
            Ok(Self(ExpressionAst::Operand(Term::from(factor))).into())
        }

        fn postfix(
            cursor: &mut &str,
        ) -> ModalResult<OperandPostfix<SimpleFormulaExpression, Self>> {
            '!'.map(|_| OperandPostfix {
                precedence: 25,
                finish: |_, inner| Ok(Self(inner)),
            })
            .parse_next(cursor)
        }
    }

    type Additive = Layer<Wrapped, SimpleFormulaExpression, MultistageOperations, Permitted>;
    type Algebraic = Layer<Wrapped, SimpleFormulaExpression, FormulaOperations, Permitted>;

    #[test]
    fn operand_grammar_is_reusable_across_operator_vocabularies() {
        for source in ["x*y + @z", "x*y + z!", "x*y + (z:w)!"] {
            let ParsedNode::Outer(ExpressionAst::Add(_, pair)) =
                super::super::parse_node::<Additive>.parse(source).unwrap()
            else {
                panic!("expected outer addition: {source}");
            };
            let ExpressionAst::Operand(Wrapped(inner)) = *pair.left else {
                panic!("expected embedded inner interaction");
            };
            assert!(matches!(inner, ExpressionAst::InteractFully(_, _)));
        }

        assert!(
            super::super::parse_node::<Additive>
                .parse("@x * @y")
                .is_err()
        );
        assert!(matches!(
            super::super::parse_node::<Algebraic>
                .parse("@x * @y")
                .unwrap(),
            ParsedNode::Outer(ExpressionAst::InteractFully(_, _))
        ));
    }

    #[test]
    fn custom_postfix_obeys_binding_power_and_recursive_grammar() {
        let ParsedNode::Outer(ExpressionAst::Add(_, pair)) = super::super::parse_node::<Additive>
            .parse("x + y:z!")
            .unwrap()
        else {
            panic!("expected outer addition");
        };
        let ExpressionAst::Operand(Wrapped(ExpressionAst::Interact(_, _))) = *pair.right else {
            panic!("postfix must wrap the whole interaction, but not the addition");
        };

        assert!(matches!(
            super::super::parse_node::<Additive>
                .parse("sw(@x, (y:z)!)")
                .unwrap(),
            ParsedNode::Outer(ExpressionAst::Stepwise(_, _))
        ));
        // The postfix accepts an inner expression, not an already outer operand.
        assert!(super::super::parse_node::<Additive>.parse("@x!").is_err());
    }
}
