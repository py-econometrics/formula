//! Recognize the language once, then construct typed operands through TryFrom.
//! The input enum describes syntax, not the AST's available layer types.
use super::ast::{
    ConstructionError, ExpressionAst, ExpressionKind, FixedEffect, Forbidden, FormulaAst,
    LayerOperand, Loaded,
};
use super::{construction_result, parse_factor, parse_formula_ast, parse_loadings};
use crate::Term;
use crate::formula::VariableDefinition;
use winnow::ascii::multispace0;
use winnow::combinator::{Postfix, alt, cut_err, preceded, terminated};
use winnow::error::{ContextError, ErrMode, StrContext, StrContextValue};
use winnow::token::take_while;
use winnow::{ModalResult, Parser};

/// A recognized operand production, before admission to a typed AST.
/// Rejected inputs are returned intact, allowing an atom to be tried in the
/// inner layer without cloning it or parsing its tokens again.
pub(super) enum ParsedOperand<I: ExpressionKind> {
    Term(Term),
    Child(FormulaAst),
    Loaded(Loaded<FixedEffect<I>>),
}

impl<I: ExpressionKind> TryFrom<ParsedOperand<I>> for Term {
    type Error = ParsedOperand<I>;
    fn try_from(input: ParsedOperand<I>) -> Result<Self, Self::Error> {
        match input {
            ParsedOperand::Term(term) => Ok(term),
            other => Err(other),
        }
    }
}
impl<I: ExpressionKind> TryFrom<ParsedOperand<I>> for VariableDefinition {
    type Error = ParsedOperand<I>;
    fn try_from(input: ParsedOperand<I>) -> Result<Self, Self::Error> {
        match input {
            ParsedOperand::Term(term) => {
                Self::try_from(term.clone()).map_err(|_| ParsedOperand::Term(term))
            }
            other => Err(other),
        }
    }
}
impl<I: ExpressionKind> TryFrom<ParsedOperand<I>> for FormulaAst {
    type Error = ParsedOperand<I>;
    fn try_from(input: ParsedOperand<I>) -> Result<Self, Self::Error> {
        match input {
            ParsedOperand::Child(child) => Ok(child),
            other => Err(other),
        }
    }
}
impl<I: ExpressionKind> TryFrom<ParsedOperand<I>> for Loaded<FixedEffect<I>> {
    type Error = ParsedOperand<I>;
    fn try_from(input: ParsedOperand<I>) -> Result<Self, Self::Error> {
        match input {
            ParsedOperand::Loaded(loaded) => Ok(loaded),
            other => Err(other),
        }
    }
}
impl<I: ExpressionKind> TryFrom<ParsedOperand<I>> for Forbidden {
    type Error = ParsedOperand<I>;
    fn try_from(input: ParsedOperand<I>) -> Result<Self, Self::Error> {
        Err(input)
    }
}
impl TryFrom<Term> for Forbidden {
    type Error = ConstructionError;
    fn try_from(_: Term) -> Result<Self, Self::Error> {
        Err(ConstructionError::Unsupported)
    }
}
impl<I: ExpressionKind, B, N> TryFrom<ParsedOperand<I>> for LayerOperand<B, N>
where
    N: TryFrom<ParsedOperand<I>, Error = ParsedOperand<I>>,
{
    type Error = ParsedOperand<I>;
    fn try_from(input: ParsedOperand<I>) -> Result<Self, Self::Error> {
        N::try_from(input).map(|nested| Self::Nested(Box::new(nested)))
    }
}

/// One constructor interface, independent of how many productions the language
/// recognizes. As before, the parser carries one inner-to-outer boundary.
pub(super) trait ParseLayer:
    ExpressionKind<
        Operand: TryFrom<ParsedOperand<Self::Inner>, Error = ParsedOperand<Self::Inner>>,
        Inner: ExpressionKind<Inner = Forbidden, Operand: TryFrom<Term>>,
    >
{
}
impl<K> ParseLayer for K where
    K: ExpressionKind<
            Operand: TryFrom<ParsedOperand<K::Inner>, Error = ParsedOperand<K::Inner>>,
            Inner: ExpressionKind<Inner = Forbidden, Operand: TryFrom<Term>>,
        >
{
}

/// All productions use the same admission path. Only an ordinary atom can start
/// in the inner layer; structured operands must inhabit the requested layer.
fn construct<K: ParseLayer>(
    input: ParsedOperand<K::Inner>,
) -> Result<ExpressionAst<K>, ConstructionError> {
    match K::Operand::try_from(input) {
        Ok(operand) => Ok(ExpressionAst::Operand(operand)),
        Err(ParsedOperand::Term(term)) => ExpressionAst::<K::Inner>::try_operand(term)
            .and_then(ExpressionAst::<K>::try_operand::<ExpressionAst<K::Inner>>),
        Err(_) => Err(ConstructionError::Unsupported),
    }
}

pub(super) fn primary<K: ParseLayer>(cursor: &mut &str) -> ModalResult<ExpressionAst<K>> {
    let child = preceded(
        '[',
        cut_err(terminated(
            parse_formula_ast,
            ']'.context(StrContext::Expected(StrContextValue::Description(
                "']' to close the nested formula",
            ))),
        )),
    )
    .map(|child| (ParsedOperand::Child(child), false));
    let intercept = take_while(1.., super::is_identifier_character)
        .verify(|name: &str| matches!(name, "0" | "1"))
        .map(|name| (ParsedOperand::Term(Term::intercept()), name == "0"));
    let factor = parse_factor.map(|factor| (ParsedOperand::Term(Term::from(factor)), false));
    let (input, negate) = alt((child, intercept, factor)).parse_next(cursor)?;
    let node = construction_result(cursor, construct::<K>(input))?;
    if negate {
        construction_result(cursor, node.negate())
    } else {
        Ok(node)
    }
}

type PostfixRule<'i, K> = Postfix<&'i str, ExpressionAst<K>, ErrMode<ContextError>>;
pub(super) fn postfix<'i, K: ParseLayer>(cursor: &mut &'i str) -> ModalResult<PostfixRule<'i, K>> {
    alt((
        "[[".value(Postfix(25, loading::<K, false>)),
        '['.value(Postfix(25, loading::<K, true>)),
    ))
    .parse_next(cursor)
}

fn loading<K: ParseLayer, const INTERCEPT: bool>(
    cursor: &mut &str,
    node: ExpressionAst<K>,
) -> ModalResult<ExpressionAst<K>> {
    let (first_loading, rest_loadings) = if INTERCEPT {
        cut_err(terminated(parse_loadings, (']', multispace0))).parse_next(cursor)?
    } else {
        cut_err(terminated(parse_loadings, ("]]", multispace0))).parse_next(cursor)?
    };
    let inner = construction_result(cursor, node.into_inner())?;
    let input = ParsedOperand::Loaded(Loaded {
        group: inner.into(),
        intercept: INTERCEPT,
        first_loading,
        rest_loadings,
    });
    construction_result(cursor, construct::<K>(input))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::ast::{
        FormulaOperations, Layer, MultistageOperations, Permitted, SimpleFormulaExpression,
    };

    // A new operand type admits BOTH existing structured productions. It adds no
    // grammar declarations or parser callbacks; only ordinary value conversions.
    enum Wrapped {
        Child(FormulaAst),
        Loaded(Loaded<FixedEffect<SimpleFormulaExpression>>),
    }
    impl TryFrom<ParsedOperand<SimpleFormulaExpression>> for Wrapped {
        type Error = ParsedOperand<SimpleFormulaExpression>;
        fn try_from(input: ParsedOperand<SimpleFormulaExpression>) -> Result<Self, Self::Error> {
            match input {
                ParsedOperand::Child(child) => Ok(Self::Child(child)),
                ParsedOperand::Loaded(loaded) => Ok(Self::Loaded(loaded)),
                other => Err(other),
            }
        }
    }
    type Operand = LayerOperand<ExpressionAst<SimpleFormulaExpression>, Wrapped>;
    type Additive = Layer<Operand, SimpleFormulaExpression, MultistageOperations, Permitted>;
    type Algebraic = Layer<Operand, SimpleFormulaExpression, FormulaOperations, Permitted>;

    #[test]
    fn operand_admission_is_reusable_across_operator_vocabularies() {
        for source in ["x*y + [a ~ z]", "x*y + z[t]", "x*y + (z:w)[t]"] {
            let ExpressionAst::Add(_, pair) =
                super::super::parse_node::<Additive>.parse(source).unwrap()
            else {
                panic!("expected outer addition: {source}");
            };
            let ExpressionAst::Operand(LayerOperand::Expression(inner)) = *pair.left else {
                panic!("expected preserved inner subtree");
            };
            assert!(matches!(inner, ExpressionAst::InteractFully(_, _)));
        }
        assert!(
            super::super::parse_node::<Additive>
                .parse("[a ~ x] * y[t]")
                .is_err()
        );
        let ExpressionAst::InteractFully(_, operands) = super::super::parse_node::<Algebraic>
            .parse("[a ~ x] * y[t]")
            .unwrap()
        else {
            panic!("expected outer interaction");
        };
        let ExpressionAst::Operand(LayerOperand::Nested(left)) = *operands.left else {
            panic!("expected admitted child");
        };
        let Wrapped::Child(child) = *left else {
            panic!("expected child payload");
        };
        assert!(matches!(
            child.right,
            ExpressionAst::Operand(LayerOperand::Expression(_))
        ));
    }

    #[test]
    fn loading_admission_obeys_binding_power_and_recursive_grammar() {
        let ExpressionAst::Add(_, pair) = super::super::parse_node::<Additive>
            .parse("x + y:z[t]")
            .unwrap()
        else {
            panic!("expected outer addition");
        };
        let ExpressionAst::Operand(LayerOperand::Nested(wrapped)) = *pair.right else {
            panic!("expected constructed nested operand");
        };
        let Wrapped::Loaded(loaded) = *wrapped else {
            panic!("expected loaded payload");
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
}
