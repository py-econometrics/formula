//! Syntax-specific operand and postfix parsers for the shared Pratt engine.

use super::ast::{
    ConstructionError, ExpressionAst, ExpressionKind, FixedEffectGroup, Forbidden,
    GroupingOperations, Layer, ParseLayers, ParsedNode, Permission, SimpleFormulaExpression,
    SingleFormulaExpression,
};
use super::{construction_result, parse_formula_ast, parse_loadings};
use crate::{Factor, Term};
use std::sync::Arc;
use winnow::ascii::multispace0;
use winnow::combinator::{Postfix, alt, cut_err, preceded, terminated};
use winnow::error::{ContextError, ErrMode, StrContext, StrContextValue};
use winnow::{ModalResult, Parser};

const GROUPED_PRECEDENCE: i64 = 25;
type PostfixRule<'i, K> = Postfix<&'i str, ParsedNode<K>, ErrMode<ContextError>>;

/// The Pratt engine delegates syntax-specific construction to these parsers.
/// Unsupported forms are consumed before rejection to retain syntax diagnostics.
pub(super) trait Syntax: ParseLayers<Operand: TryFrom<Term>> {
    fn special_operand(cursor: &mut &str) -> ModalResult<ParsedNode<Self>> {
        child_formula(cursor)?;
        construction_result(cursor, Err(ConstructionError::Unsupported))
    }

    fn postfix<'i>(cursor: &mut &'i str) -> ModalResult<PostfixRule<'i, Self>> {
        alt((
            "[[".value(Postfix(GROUPED_PRECEDENCE, reject_loadings::<Self, false>)),
            '['.value(Postfix(GROUPED_PRECEDENCE, reject_loadings::<Self, true>)),
        ))
        .parse_next(cursor)
    }
}

impl Syntax for SingleFormulaExpression {}

impl Syntax for SimpleFormulaExpression {
    fn special_operand(cursor: &mut &str) -> ModalResult<ParsedNode<Self>> {
        let formula = child_formula(cursor)?;
        Ok(ParsedNode::Embedded(ExpressionAst::Operand(formula.into())))
    }
}

impl<S: Permission> Syntax
    for Layer<crate::formula::VariableDefinition, Forbidden, GroupingOperations, S>
{
    fn postfix<'i>(cursor: &mut &'i str) -> ModalResult<PostfixRule<'i, Self>> {
        alt((
            "[[".value(Postfix(GROUPED_PRECEDENCE, loaded_group::<Self, false>)),
            '['.value(Postfix(GROUPED_PRECEDENCE, loaded_group::<Self, true>)),
        ))
        .parse_next(cursor)
    }
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

fn reject_loadings<K: ParseLayers, const INTERCEPT: bool>(
    cursor: &mut &str,
    _node: ParsedNode<K>,
) -> ModalResult<ParsedNode<K>> {
    loading_tail::<INTERCEPT>(cursor)?;
    construction_result(cursor, Err(ConstructionError::Unsupported))
}

fn loaded_group<K: ParseLayers, const INTERCEPT: bool>(
    cursor: &mut &str,
    node: ParsedNode<K>,
) -> ModalResult<ParsedNode<K>>
where
    K::Embedded: ExpressionKind<Operand = FixedEffectGroup<K>>,
{
    let loadings = loading_tail::<INTERCEPT>(cursor)?;
    let group = construction_result(cursor, node.into_flat())?;
    Ok(ParsedNode::Embedded(ExpressionAst::Operand(
        FixedEffectGroup {
            group,
            intercept: INTERCEPT,
            loadings,
        },
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::ast::FormulaOperations;

    // This syntax introduces a new operand and punctuation, without teaching
    // the AST or Pratt engine about either one.
    struct Custom;
    struct Wrapped(ExpressionAst<Custom>);

    impl ExpressionKind for Custom {
        type Operand = Term;
        type Inner = Forbidden;
        type Operators = FormulaOperations;
        type Stepwise = Forbidden;
    }

    impl From<ExpressionAst<Custom>> for Wrapped {
        fn from(expression: ExpressionAst<Custom>) -> Self {
            Self(expression)
        }
    }

    impl ParseLayers for Custom {
        type Embedded = Layer<Wrapped, Self, FormulaOperations, Forbidden>;
    }

    impl Syntax for Custom {
        fn special_operand(cursor: &mut &str) -> ModalResult<ParsedNode<Self>> {
            let factor = preceded('@', cut_err(crate::parse::parse_factor)).parse_next(cursor)?;
            let inner = ExpressionAst::<Self>::Operand(Term::from(factor));
            Ok(ParsedNode::Embedded(inner.into()))
        }

        fn postfix<'i>(cursor: &mut &'i str) -> ModalResult<PostfixRule<'i, Self>> {
            '!'.value(Postfix(25, |cursor, node: ParsedNode<Self>| {
                construction_result(cursor, node.into_embedded()).map(ParsedNode::Embedded)
            }))
            .parse_next(cursor)
        }
    }

    #[test]
    fn new_operand_and_postfix_share_the_pratt_engine() {
        let ParsedNode::Embedded(ExpressionAst::Add(_, pair)) = crate::parse::parse_node::<Custom>
            .parse("x*y + @z!")
            .unwrap()
        else {
            panic!("expected outer addition");
        };
        let ExpressionAst::Operand(Wrapped(inner)) = *pair.left else {
            panic!("expected embedded inner interaction");
        };
        assert!(matches!(inner, ExpressionAst::InteractFully(_, _)));
        let ExpressionAst::Operand(Wrapped(ExpressionAst::Operand(term))) = *pair.right else {
            panic!("expected custom operand");
        };
        assert_eq!(term, Term::from(Factor::from("z")));
    }
}
