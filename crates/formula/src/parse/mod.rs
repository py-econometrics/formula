mod ast;
mod expand;

use crate::Role;
use crate::fit::Fit;
use crate::formula::{Expression, FactorKind, Term, VariableDefinition, VaryingSlopes};
use crate::formula::{Factor, Formula, Formulas, push_unique};
use crate::parse::ast::{Compose, StepwiseOperator};
use crate::parse::expand::Expand;
use crate::transform::{
    CategoricalDefinition, ExpressionTransform, LevelSelection, Levels, ParameterValue,
    PolynomialDefinition, StandardizeDefinition, Transform,
};
use polars::prelude::{PlSmallStr, Scalar};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use winnow::ascii::{Caseless, multispace0};
use winnow::combinator::Infix::{Left, Right};
use winnow::combinator::{
    Postfix, Prefix, alt, cut_err, delimited, eof, expression, fail, opt, peek, preceded, repeat,
    separated, terminated,
};
use winnow::error::{
    ContextError, ErrMode, FromExternalError, ParseError, StrContext, StrContextValue,
};
use winnow::token::{one_of, take_while};
use winnow::{ModalResult, Parser};

pub(crate) const FORMULA_SEPARATOR: char = '~';
pub(crate) const PIPE_SEPARATOR: char = '|';
const SUM_PRECEDENCE: i64 = 10;
const PRODUCT_PRECEDENCE: i64 = 20;
const ANNOTATION_PRECEDENCE: i64 = 25;
const INTERACTION_PRECEDENCE: i64 = 30;
const POWER_PRECEDENCE: i64 = 35;
const PREFIX_PRECEDENCE: i64 = 40;

#[derive(Debug)]
struct TransformCall {
    name: String,
    arguments: Vec<TransformCallArgument>,
}

#[derive(Debug)]
enum TransformCallArgument {
    Positional(Factor),
    Keyword { name: String, value: ParameterValue },
}

#[derive(Debug)]
enum TransformCallError {
    UnknownTransform(String),
    DuplicateKeyword {
        name: String,
    },
    DuplicateName {
        name: String,
    },
    PositionalAfterKeyword {
        position: usize,
    },
    UnexpectedKeywords {
        transform: &'static str,
        keywords: Vec<String>,
    },
    InvalidParameter {
        transform: &'static str,
        message: String,
    },
}

impl std::fmt::Display for TransformCallError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DuplicateKeyword { name } => {
                write!(f, "duplicate keyword `{name}`")
            }
            Self::DuplicateName { name } => {
                write!(f, "duplicate name '{name}'")
            }
            Self::PositionalAfterKeyword { position } => {
                write!(
                    f,
                    "positional argument {position} follows a keyword argument"
                )
            }
            Self::UnknownTransform(name) => write!(f, "unknown transform `{name}`"),
            Self::UnexpectedKeywords {
                transform,
                keywords,
            } => write!(
                f,
                "unexpected keywords for transform `{transform}`: {}",
                keywords.join(", ")
            ),
            Self::InvalidParameter { transform, message } => {
                write!(
                    f,
                    "invalid parameter for transform `{transform}`: {message}"
                )
            }
        }
    }
}

impl std::error::Error for TransformCallError {}

pub(crate) fn is_identifier_character(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

fn is_argument_boundary(c: char) -> bool {
    c.is_whitespace() || matches!(c, ',' | ')' | ']' | '}')
}

fn parse_bare_token<'i>(cursor: &mut &'i str) -> ModalResult<&'i str> {
    take_while(1.., |c: char| !is_argument_boundary(c)).parse_next(cursor)
}

pub(crate) fn parse_scalar(cursor: &mut &str) -> ModalResult<Scalar> {
    terminated(
        alt((
            parse_literal_string,
            Caseless("true").value(Scalar::from(true)),
            Caseless("false").value(Scalar::from(false)),
            parse_literal_number,
            fail.context(StrContext::Expected(StrContextValue::Description(
                "a scalar value",
            ))),
        )),
        parse_argument_boundary,
    )
    .parse_next(cursor)
}

fn parse_parameter_value(cursor: &mut &str) -> ModalResult<ParameterValue> {
    alt((
        parse_parameter_map.map(ParameterValue::Map),
        parse_parameter_list.map(ParameterValue::List),
        parse_scalar.map(ParameterValue::Scalar),
    ))
    .parse_next(cursor)
}

fn parse_parameter_list(cursor: &mut &str) -> ModalResult<Vec<Scalar>> {
    preceded(
        ('[', multispace0),
        cut_err(terminated(
            separated(0.., parse_scalar, (multispace0, ',', multispace0)),
            (multispace0, ']'),
        )),
    )
    .parse_next(cursor)
}

fn parse_parameter_map(cursor: &mut &str) -> ModalResult<Vec<(PlSmallStr, ParameterValue)>> {
    let entry = (
        parse_single_quoted.context(StrContext::Expected(StrContextValue::Description(
            "a single-quoted name",
        ))),
        cut_err(preceded(
            (
                multispace0,
                ':'.context(StrContext::Expected(StrContextValue::CharLiteral(':'))),
                multispace0,
            ),
            parse_parameter_value,
        )),
    )
        .map(|(name, value)| (PlSmallStr::from(name), value));

    preceded(
        ('{', multispace0),
        cut_err(
            alt((
                '}'.value(Vec::new()),
                terminated(
                    separated(1.., entry, (multispace0, ',', multispace0)),
                    (
                        multispace0,
                        '}'.context(StrContext::Expected(StrContextValue::CharLiteral('}'))),
                    ),
                ),
            ))
            .try_map(|entries: Vec<(PlSmallStr, ParameterValue)>| {
                let mut names = BTreeSet::new();
                match entries.iter().find(|(name, _)| !names.insert(name)) {
                    Some((name, _)) => Err(TransformCallError::DuplicateName {
                        name: name.to_string(),
                    }),
                    None => Ok(entries),
                }
            }),
        ),
    )
    .parse_next(cursor)
}

fn parse_single_quoted<'i>(cursor: &mut &'i str) -> ModalResult<&'i str> {
    preceded(
        '\'',
        cut_err(terminated(
            take_while(0.., |c: char| c != '\''),
            '\''.context(StrContext::Expected(StrContextValue::Description(
                "a closing single quote",
            ))),
        )),
    )
    .parse_next(cursor)
}

fn parse_literal_string(cursor: &mut &str) -> ModalResult<Scalar> {
    parse_single_quoted
        .map(|text: &str| Scalar::from(PlSmallStr::from(text)))
        .parse_next(cursor)
}

fn parse_quoted(cursor: &mut &str) -> ModalResult<String> {
    preceded(
        '`',
        cut_err(terminated(
            repeat(
                0..,
                alt(("``".value("`"), take_while(1.., |c: char| c != '`'))),
            ),
            '`'.context(StrContext::Expected(StrContextValue::Description(
                "a closing backtick",
            ))),
        )),
    )
    .parse_next(cursor)
}

fn has_float_marker(token: &str) -> bool {
    token.contains('.') || token.contains('e') || token.contains('E')
}

fn parse_literal_number(cursor: &mut &str) -> ModalResult<Scalar> {
    alt((
        parse_bare_token
            .verify(|token: &str| !has_float_marker(token))
            .try_map(str::parse::<i64>)
            .map(Scalar::from),
        parse_bare_token
            .verify(|token: &str| has_float_marker(token))
            .try_map(str::parse::<f64>)
            .verify(|value: &f64| value.is_finite())
            .map(Scalar::from),
    ))
    .parse_next(cursor)
}

fn parse_argument_boundary(cursor: &mut &str) -> ModalResult<()> {
    peek(alt((one_of(is_argument_boundary).void(), eof.void()))).parse_next(cursor)
}

fn parse_loadings(cursor: &mut &str) -> ModalResult<(Arc<Factor>, Vec<Arc<Factor>>)> {
    multispace0.parse_next(cursor)?;

    let first = cut_err(parse_transform_argument_positional.context(StrContext::Label("loading")))
        .map(Arc::new)
        .parse_next(cursor)?;

    let rest = repeat(
        0..,
        preceded(
            (multispace0, ',', multispace0),
            cut_err(parse_transform_argument_positional.context(StrContext::Label("loading")))
                .map(Arc::new),
        ),
    )
    .parse_next(cursor)?;

    multispace0.parse_next(cursor)?;

    Ok((first, rest))
}

fn parse_factor(cursor: &mut &str) -> ModalResult<Factor> {
    alt((parse_transform_call, parse_column)).parse_next(cursor)
}

fn parse_column(cursor: &mut &str) -> ModalResult<Factor> {
    alt((
        parse_quoted,
        take_while(1.., is_identifier_character).map(str::to_owned),
    ))
    .map(|name: String| Factor::from(name.as_str()))
    .parse_next(cursor)
}

fn parse_transform_call(cursor: &mut &str) -> ModalResult<Factor> {
    let name = terminated(take_while(1.., is_identifier_character), '(').parse_next(cursor)?;

    cut_err(
        parse_transform_arguments
            .map(|arguments| TransformCall {
                name: name.to_owned(),
                arguments,
            })
            .try_map(Factor::try_from),
    )
    .parse_next(cursor)
}

fn parse_transform_arguments(cursor: &mut &str) -> ModalResult<Vec<TransformCallArgument>> {
    multispace0.parse_next(cursor)?;

    if cursor.starts_with(')') {
        ')'.parse_next(cursor)?;
        return Ok(Vec::new());
    }

    let arguments = separated(
        1..,
        cut_err(parse_transform_argument),
        (multispace0, ',', multispace0),
    )
    .parse_next(cursor)?;

    multispace0.parse_next(cursor)?;

    ')'.context(StrContext::Expected(StrContextValue::Description(
        "',' or ')' after an argument",
    )))
    .parse_next(cursor)?;

    Ok(arguments)
}

fn parse_transform_argument(cursor: &mut &str) -> ModalResult<TransformCallArgument> {
    alt((
        parse_transform_argument_keyword,
        parse_transform_argument_positional.map(TransformCallArgument::Positional),
        fail.context(StrContext::Expected(StrContextValue::Description(
            "a transform argument",
        ))),
    ))
    .parse_next(cursor)
}

fn parse_transform_argument_positional(cursor: &mut &str) -> ModalResult<Factor> {
    alt((
        terminated(parse_literal_number, parse_argument_boundary).map(|value| Factor {
            kind: FactorKind::Literal(value),
        }),
        parse_factor,
    ))
    .parse_next(cursor)
}

fn parse_transform_argument_keyword(cursor: &mut &str) -> ModalResult<TransformCallArgument> {
    let name = terminated(
        take_while(1.., is_identifier_character),
        (multispace0, '=', multispace0),
    )
    .parse_next(cursor)?;

    let value = cut_err(parse_parameter_value.context(StrContext::Label("keyword value")))
        .parse_next(cursor)?;

    Ok(TransformCallArgument::Keyword {
        name: name.to_owned(),
        value,
    })
}

#[derive(Debug, PartialEq, Eq)]
enum ConstructionError {
    Unsupported,
}

impl std::fmt::Display for ConstructionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unsupported => f.write_str("operation is not supported"),
        }
    }
}

impl std::error::Error for ConstructionError {}

fn construction_result<T>(cursor: &mut &str, value: Option<T>) -> ModalResult<T> {
    value.ok_or_else(|| {
        ErrMode::Cut(ContextError::from_external_error(
            &*cursor,
            ConstructionError::Unsupported,
        ))
    })
}

#[cfg(test)]
pub(crate) fn parse_expression(cursor: &mut &str) -> ModalResult<Vec<Expression>> {
    parse_node::<ast::MultistageFormula>
        .map(|node| node.expand())
        .parse_next(cursor)
}

fn parse_node<L: ParseLayer>(cursor: &mut &str) -> ModalResult<ast::ExpressionAst<L>> {
    expression(parse_operand::<L>)
        .prefix(
            preceded(multispace0, '-')
                .value(Prefix(
                    PREFIX_PRECEDENCE,
                    |cursor, node: ast::ExpressionAst<L>| {
                        construction_result(cursor, node.negate().ok())
                    },
                ))
                .context(StrContext::Expected(StrContextValue::Description(
                    "an expression",
                ))),
        )
        .infix(alt((
            '+'.value(Left(SUM_PRECEDENCE, |cursor, left, right| {
                construction_result(
                    cursor,
                    ast::ExpressionAst::<L>::binary(ast::BinaryOperator::Add, left, right).ok(),
                )
            })),
            '-'.value(Left(SUM_PRECEDENCE, |cursor, left, right| {
                construction_result(
                    cursor,
                    ast::ExpressionAst::<L>::binary(ast::BinaryOperator::Subtract, left, right)
                        .ok(),
                )
            })),
            '*'.value(Left(PRODUCT_PRECEDENCE, |cursor, left, right| {
                construction_result(
                    cursor,
                    ast::ExpressionAst::<L>::binary(
                        ast::BinaryOperator::InteractFully,
                        left,
                        right,
                    )
                    .ok(),
                )
            })),
            '/'.value(Left(PRODUCT_PRECEDENCE, |cursor, left, right| {
                construction_result(
                    cursor,
                    ast::ExpressionAst::<L>::binary(ast::BinaryOperator::Nest, left, right).ok(),
                )
            })),
            ':'.value(Right(INTERACTION_PRECEDENCE, |cursor, left, right| {
                construction_result(
                    cursor,
                    ast::ExpressionAst::<L>::binary(ast::BinaryOperator::Interact, left, right)
                        .ok(),
                )
            })),
        )))
        .postfix(postfix::<L>)
        .parse_next(cursor)
}

fn parse_stepwise_arguments<L: ParseLayer>(
    cursor: &mut &str,
) -> ModalResult<(ast::ExpressionAst<L>, Vec<ast::ExpressionAst<L>>)> {
    multispace0.parse_next(cursor)?;

    let first = cut_err(parse_node::<L>).parse_next(cursor)?;

    let rest = repeat(
        0..,
        preceded((multispace0, ',', multispace0), cut_err(parse_node::<L>)),
    )
    .parse_next(cursor)?;

    multispace0.parse_next(cursor)?;

    ')'.context(StrContext::Expected(StrContextValue::Description(
        "',' or ')' after an expression",
    )))
    .parse_next(cursor)?;

    Ok((first, rest))
}

fn parse_stepwise<L: ParseLayer>(cursor: &mut &str) -> ModalResult<ast::ExpressionAst<L>> {
    let operator = terminated(
        alt((
            "sw0".value(StepwiseOperator::Sw0),
            "sw".value(StepwiseOperator::Sw),
            "csw0".value(StepwiseOperator::Csw0),
            "csw".value(StepwiseOperator::Csw),
            "mvsw".value(StepwiseOperator::Mvsw),
        )),
        '(',
    )
    .parse_next(cursor)?;

    let (first, rest) = cut_err(parse_stepwise_arguments::<L>).parse_next(cursor)?;

    construction_result(
        cursor,
        ast::ExpressionAst::<L>::stepwise(operator, first, rest).ok(),
    )
}

fn parse_node_parenthesized<L: ParseLayer>(
    cursor: &mut &str,
) -> ModalResult<ast::ExpressionAst<L>> {
    '('.parse_next(cursor)?;

    let node = cut_err(parse_node::<L>).parse_next(cursor)?;

    cut_err(
        ')'.context(StrContext::Expected(StrContextValue::Description(
            "')' to close the parenthesized expression",
        ))),
    )
    .parse_next(cursor)?;

    Ok(node.parenthesized())
}

fn parse_operand<L: ParseLayer>(cursor: &mut &str) -> ModalResult<ast::ExpressionAst<L>> {
    delimited(
        multispace0,
        alt((
            parse_node_parenthesized::<L>,
            parse_stepwise::<L>,
            ast::ExpressionAst::<L>::atom,
        )),
        multispace0,
    )
    .parse_next(cursor)
}

/// Generic description of how the `winnow` parser converts strings to the typed AST
trait ParseAtom: Sized {
    fn parse(cursor: &mut &str) -> ModalResult<(Self, bool)>;
}

impl ParseAtom for ast::Forbidden {
    fn parse(cursor: &mut &str) -> ModalResult<(Self, bool)> {
        fail.parse_next(cursor)
    }
}

impl ParseAtom for Term {
    fn parse(cursor: &mut &str) -> ModalResult<(Self, bool)> {
        alt((
            take_while(1.., is_identifier_character)
                .verify(|name: &str| matches!(name, "0" | "1"))
                .map(|name| (Term::intercept(), name == "0")),
            parse_factor.map(|factor| (Term::from(factor), false)),
        ))
        .parse_next(cursor)
    }
}

impl ParseAtom for VariableDefinition {
    fn parse(cursor: &mut &str) -> ModalResult<(Self, bool)> {
        let (term, negate) = Term::parse(cursor)?;
        construction_result(cursor, Self::try_from(term).ok()).map(|value| (value, negate))
    }
}

impl ParseAtom for Box<ast::FormulaAst> {
    fn parse(cursor: &mut &str) -> ModalResult<(Self, bool)> {
        preceded(
            '[',
            cut_err(terminated(
                parse_nested_formula_ast,
                ']'.context(StrContext::Expected(StrContextValue::Description(
                    "']' to close the nested formula",
                ))),
            )),
        )
        .map(|formula| (Box::new(formula), false))
        .parse_next(cursor)
    }
}

trait ParseAnnotation: Sized {
    fn starts(input: &str) -> bool;
    fn parse(cursor: &mut &str) -> ModalResult<Self>;
}

impl ParseAnnotation for ast::Forbidden {
    fn starts(_: &str) -> bool {
        false
    }
    fn parse(cursor: &mut &str) -> ModalResult<Self> {
        fail.parse_next(cursor)
    }
}

impl ParseAnnotation for VaryingSlopes {
    fn starts(input: &str) -> bool {
        input.starts_with('[')
    }
    fn parse(cursor: &mut &str) -> ModalResult<Self> {
        let intercept = alt(("[[".value(false), '['.value(true))).parse_next(cursor)?;
        let close = if intercept { "]" } else { "]]" };
        let (first, rest) =
            cut_err(terminated(parse_loadings, (close, multispace0))).parse_next(cursor)?;
        Ok(Self {
            intercept,
            first,
            rest,
        })
    }
}

// Alias for [`ast::Layer`] for more compact type bounds in `parse_node` etc
trait ParseLayer:
    ast::Layer<Atom: ParseAtom, Inner: ParseExpression, Annotation: ParseAnnotation>
{
}
impl<L> ParseLayer for L where
    L: ast::Layer<Atom: ParseAtom, Inner: ParseExpression, Annotation: ParseAnnotation>
{
}

trait ParseExpression: Compose {
    fn atom(cursor: &mut &str) -> ModalResult<Self>;
    fn annotation_starts(input: &str) -> bool;
    fn annotation(self, cursor: &mut &str) -> ModalResult<Self>;
}

impl ParseExpression for ast::Forbidden {
    fn atom(cursor: &mut &str) -> ModalResult<Self> {
        fail.parse_next(cursor)
    }
    fn annotation_starts(_: &str) -> bool {
        false
    }
    fn annotation(self, _: &mut &str) -> ModalResult<Self> {
        match self {}
    }
}

impl<L: ParseLayer> ParseExpression for ast::ExpressionAst<L> {
    fn atom(cursor: &mut &str) -> ModalResult<Self> {
        match opt(L::Atom::parse).parse_next(cursor)? {
            Some((atom, false)) => Ok(Self::Atom(atom)),
            Some((atom, true)) => construction_result(cursor, Self::Atom(atom).negate().ok()),
            None => L::Inner::atom(cursor).map(Self::embed),
        }
    }

    fn annotation_starts(input: &str) -> bool {
        L::Annotation::starts(input) || L::Inner::annotation_starts(input)
    }

    fn annotation(self, cursor: &mut &str) -> ModalResult<Self> {
        if L::Annotation::starts(cursor) {
            let annotation = L::Annotation::parse(cursor)?;
            return construction_result(cursor, self.annotate(annotation));
        }
        match self.into_inner() {
            Ok(inner) => inner.annotation(cursor).map(Self::embed),
            Err(_) => construction_result(cursor, None),
        }
    }
}

fn postfix<'i, L: ParseLayer>(
    cursor: &mut &'i str,
) -> ModalResult<Postfix<&'i str, ast::ExpressionAst<L>, ErrMode<ContextError>>> {
    if ast::ExpressionAst::<L>::annotation_starts(cursor) {
        Ok(Postfix(ANNOTATION_PRECEDENCE, |cursor, node| {
            node.annotation(cursor)
        }))
    } else {
        '^'.value(Postfix(
            POWER_PRECEDENCE,
            |cursor, base: ast::ExpressionAst<L>| {
                let exponent = cut_err(preceded(multispace0, parse_exponent)).parse_next(cursor)?;
                multispace0.parse_next(cursor)?;
                construction_result(cursor, base.power(exponent).ok())
            },
        ))
        .parse_next(cursor)
    }
}

/// A positive integer exponent of `^`.
fn parse_exponent(cursor: &mut &str) -> ModalResult<u32> {
    take_while(1.., |c: char| c.is_ascii_digit())
        .try_map(str::parse::<u32>)
        .verify(|exponent| *exponent >= 1)
        .context(StrContext::Expected(StrContextValue::Description(
            "a positive integer exponent",
        )))
        .parse_next(cursor)
}

fn parse_formula_ast(cursor: &mut &str) -> ModalResult<ast::FormulaAst> {
    let first = preceded(multispace0, parse_node::<ast::MultistageFormula>).parse_next(cursor)?;

    // Without `~`, the first expression is the right-hand side and there is no response
    let (left, right) = if opt(FORMULA_SEPARATOR).parse_next(cursor)?.is_some() {
        let right = preceded(
            multispace0,
            cut_err(
                parse_node::<ast::MultistageFormula>
                    .context(StrContext::Label("right-hand expression")),
            ),
        )
        .parse_next(cursor)?;
        (Some(first), right)
    } else {
        (None, first)
    };

    let fixed_effects = opt(preceded(
        PIPE_SEPARATOR,
        cut_err(
            parse_node::<ast::FixedEffects>.context(StrContext::Label("fixed-effect expression")),
        ),
    ))
    .parse_next(cursor)?;

    Ok(ast::FormulaAst {
        left,
        right,
        fixed_effects,
    })
}

fn parse_nested_formula_ast(cursor: &mut &str) -> ModalResult<ast::FormulaAst> {
    let formula = parse_formula_ast(cursor)?;
    if formula.left.is_none() {
        return fail
            .context(StrContext::Expected(StrContextValue::CharLiteral(
                FORMULA_SEPARATOR,
            )))
            .parse_next(cursor);
    }
    Ok(formula)
}

pub(crate) type ParseResult<'i, T> = Result<T, ParseError<&'i str, ContextError>>;

pub fn parse(source: &str) -> ParseResult<'_, Formulas> {
    parse_many(std::iter::once(source))
}

pub fn parse_many<'a>(input: impl IntoIterator<Item = &'a str>) -> ParseResult<'a, Formulas> {
    let mut formulas: Vec<Formula> = Vec::new();

    for source in input {
        let expanded = terminated(
            parse_formula_ast,
            eof.context(StrContext::Expected(StrContextValue::Description(
                "a formula operator or end of input",
            ))),
        )
        .map(|node| node.expand())
        .parse(source)?;

        for formula in expanded {
            push_unique(&mut formulas, formula);
        }
    }

    Ok(Formulas::new(formulas))
}

pub(crate) fn parse_term(source: &str, role: Role) -> ParseResult<'_, Expression> {
    let parser = |cursor: &mut &str| -> ModalResult<Expression> {
        let mut alternatives = match role {
            Role::Dependent | Role::Independent => {
                parse_node::<ast::SingleSimpleFormula>(cursor)?.expand()
            }
            Role::FixedEffect => {
                let node = parse_node::<ast::SingleFixedEffects>(cursor)?;
                node.expand()
            }
        };

        Ok(alternatives.remove(0))
    };

    delimited(multispace0, parser, (multispace0, eof)).parse(source)
}

impl TryFrom<TransformCall> for Factor {
    type Error = TransformCallError;

    fn try_from(call: TransformCall) -> std::result::Result<Self, Self::Error> {
        let mut arguments = Vec::new();
        let mut keywords = BTreeMap::new();
        let mut saw_keyword = false;

        for (index, argument) in call.arguments.into_iter().enumerate() {
            match argument {
                TransformCallArgument::Positional(argument) if !saw_keyword => {
                    arguments.push(Arc::new(argument));
                }
                TransformCallArgument::Positional(_) => {
                    return Err(TransformCallError::PositionalAfterKeyword {
                        position: index + 1,
                    });
                }
                TransformCallArgument::Keyword { name, value } => {
                    saw_keyword = true;
                    if keywords.contains_key(&name) {
                        return Err(TransformCallError::DuplicateKeyword { name });
                    }
                    keywords.insert(name, value);
                }
            }
        }

        let transform = match call.name.as_str() {
            "log" => {
                reject_keywords("log", keywords)?;
                Transform::Expression(ExpressionTransform::LOG)
            }
            "exp" => {
                reject_keywords("exp", keywords)?;
                Transform::Expression(ExpressionTransform::EXP)
            }
            "sqrt" => {
                reject_keywords("sqrt", keywords)?;
                Transform::Expression(ExpressionTransform::SQRT)
            }
            "add" => {
                reject_keywords("add", keywords)?;
                Transform::Expression(ExpressionTransform::ADD)
            }
            "sub" => {
                reject_keywords("sub", keywords)?;
                Transform::Expression(ExpressionTransform::SUB)
            }
            "mul" => {
                reject_keywords("mul", keywords)?;
                Transform::Expression(ExpressionTransform::MUL)
            }
            "div" => {
                reject_keywords("div", keywords)?;
                Transform::Expression(ExpressionTransform::DIV)
            }
            "pow" => {
                reject_keywords("pow", keywords)?;
                Transform::Expression(ExpressionTransform::POW)
            }
            "center" => {
                reject_keywords("center", keywords)?;
                Transform::Standardize(Fit::Unfitted(StandardizeDefinition::CENTER))
            }
            "scale" => {
                reject_keywords("scale", keywords)?;
                Transform::Standardize(Fit::Unfitted(StandardizeDefinition::SCALE))
            }
            "C" => {
                // `omit=[]` omits nothing, so it is normalised away; we do this here to support
                // a non-empty `keep` with an `empty` omit
                let omit = keywords
                    .remove(CategoricalDefinition::OMIT_PARAMETER)
                    .filter(
                        |omit| !matches!(omit, ParameterValue::List(levels) if levels.is_empty()),
                    );
                let keep = keywords.remove(CategoricalDefinition::KEEP_PARAMETER);
                let bins = keywords.remove(CategoricalDefinition::BIN_PARAMETER);
                reject_keywords("C", keywords)?;
                let bins = match bins {
                    None => None,
                    // `bin={}` merges nothing, so it is normalised away
                    Some(ParameterValue::Map(bins)) if bins.is_empty() => None,
                    Some(ParameterValue::Map(bins)) => Some(
                        bins.into_iter()
                            .map(|(name, members)| match Levels::try_from(members) {
                                // An empty bin would encode a column without any levels
                                Ok(Levels::List(levels)) if levels.is_empty() => {
                                    Err(TransformCallError::InvalidParameter {
                                        transform: "C",
                                        message: format!(
                                            "bin '{name}' must name at least one level"
                                        ),
                                    })
                                }
                                Ok(members) => Ok((name, members)),
                                Err(_) => Err(TransformCallError::InvalidParameter {
                                    transform: "C",
                                    message: format!("bin '{name}' must be a scalar or a list"),
                                }),
                            })
                            .collect::<Result<Vec<_>, _>>()?,
                    ),
                    Some(_) => {
                        return Err(TransformCallError::InvalidParameter {
                            transform: "C",
                            message: "`bin` must be a dict".into(),
                        });
                    }
                };
                let selection = match (omit, keep) {
                    (None, None) => None,
                    (Some(_), Some(_)) => {
                        return Err(TransformCallError::InvalidParameter {
                            transform: "C",
                            message: "`omit` and `keep` are mutually exclusive".into(),
                        });
                    }
                    (Some(omit), None) => match Levels::try_from(omit) {
                        Ok(omit) => Some(LevelSelection::Omit(omit)),
                        Err(_) => {
                            return Err(TransformCallError::InvalidParameter {
                                transform: "C",
                                message: "`omit` must be a scalar or a list".into(),
                            });
                        }
                    },
                    (None, Some(keep)) => match Levels::try_from(keep) {
                        // Keeping no levels would leave a term without columns
                        Ok(Levels::List(levels)) if levels.is_empty() => {
                            return Err(TransformCallError::InvalidParameter {
                                transform: "C",
                                message: "`keep` must name at least one level".into(),
                            });
                        }
                        Ok(keep) => Some(LevelSelection::Keep(keep)),
                        Err(_) => {
                            return Err(TransformCallError::InvalidParameter {
                                transform: "C",
                                message: "`keep` must be a scalar or a list".into(),
                            });
                        }
                    },
                };
                Transform::C(Fit::Unfitted(CategoricalDefinition::new(selection, bins)))
            }
            "poly" => {
                let keyword = keywords.remove(PolynomialDefinition::DEGREE_PARAMETER);
                let positional = match arguments.as_slice() {
                    // No positional argument
                    [_] => None,
                    // A single positional literal can be interpreted as degree
                    [_, second] if second.literal().is_some() => second.literal().cloned(),
                    // All other cases (e.g., positional non-literal arguments) are not supported
                    _ => {
                        return Err(TransformCallError::InvalidParameter {
                            transform: "poly",
                            message: "expected one input and an optional positive integer degree"
                                .into(),
                        });
                    }
                };
                let degree = match (positional, keyword) {
                    // Cannot specify degree both positionally and as keyword
                    (Some(_), Some(_)) => {
                        return Err(TransformCallError::InvalidParameter {
                            transform: "poly",
                            message: "the degree is given both by position and as `degree`".into(),
                        });
                    }
                    // Degree is specified positionally
                    (Some(value), None) => {
                        arguments.pop();
                        Some(ParameterValue::Scalar(value))
                    }
                    // Degree is specified as keyword
                    (None, keyword) => keyword,
                };

                let raw = keywords.remove(PolynomialDefinition::RAW_PARAMETER);
                reject_keywords("poly", keywords)?;

                let definition = PolynomialDefinition::new(degree, raw).map_err(|error| {
                    TransformCallError::InvalidParameter {
                        transform: "poly",
                        message: error.to_string(),
                    }
                })?;

                Transform::Poly(Fit::Unfitted(definition))
            }
            _ => return Err(TransformCallError::UnknownTransform(call.name)),
        };

        Ok(Self {
            kind: FactorKind::Call {
                transform,
                arguments,
            },
        })
    }
}

fn reject_keywords(
    transform: &'static str,
    keywords: BTreeMap<String, ParameterValue>,
) -> Result<(), TransformCallError> {
    if keywords.is_empty() {
        Ok(())
    } else {
        Err(TransformCallError::UnexpectedKeywords {
            transform,
            keywords: keywords.into_keys().collect(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formula::Role;
    use crate::parse::ast as typed;

    #[test]
    fn parameter_list_parser_preserves_each_elements_scalar_type() {
        assert_eq!(
            parse_parameter_value
                .parse("[ 'a', 2, 3.5, true ]")
                .unwrap(),
            ParameterValue::List(vec![
                Scalar::from(PlSmallStr::from_static("a")),
                Scalar::from(2_i64),
                Scalar::from(3.5_f64),
                Scalar::from(true),
            ])
        );
        assert_eq!(
            parse_parameter_value.parse("[ ]").unwrap(),
            ParameterValue::List(vec![])
        );
        for source in ["['a'", "['a',]", "[, 'a']", "[['a']]", "['a' 'b']"] {
            assert!(parse_parameter_value.parse(source).is_err(), "{source}");
        }
    }

    #[test]
    fn parameter_map_parser_preserves_written_order() {
        let text = |value| Scalar::from(PlSmallStr::from_static(value));
        assert_eq!(
            parse_parameter_value
                .parse("{ 'b' : [ 1 ], 'a': 'x', 'c': [], 'd': {'e': 2.5} }")
                .unwrap(),
            ParameterValue::Map(vec![
                (
                    PlSmallStr::from_static("b"),
                    ParameterValue::List(vec![Scalar::from(1_i64)])
                ),
                (
                    PlSmallStr::from_static("a"),
                    ParameterValue::Scalar(text("x"))
                ),
                (PlSmallStr::from_static("c"), ParameterValue::List(vec![])),
                (
                    PlSmallStr::from_static("d"),
                    ParameterValue::Map(vec![(
                        PlSmallStr::from_static("e"),
                        ParameterValue::Scalar(Scalar::from(2.5_f64))
                    )])
                ),
            ])
        );
        assert_eq!(
            parse_parameter_value.parse("{ }").unwrap(),
            ParameterValue::Map(vec![])
        );
        for (source, message) in [
            ("{'a': ['b']", "expected `}`"),
            ("{'a': ['b'],}", "expected `}`"),
            ("{'a' ['b']}", "expected `:`"),
            ("{'a': }", "expected a scalar value"),
            ("{1: ['a']}", "expected a single-quoted name"),
            ("{a: ['b']}", "expected a single-quoted name"),
            ("{'a': ['b'], 'a': ['c']}", "duplicate name 'a'"),
            ("{'a': {'b': 1, 'b': 2}}", "duplicate name 'b'"),
        ] {
            let error = parse_parameter_value.parse(source).unwrap_err().to_string();
            assert!(error.contains(message), "{source}: {error}");
        }
    }

    #[test]
    fn transform_keywords_reject_dict_values() {
        for (source, message) in [
            (
                "y ~ C(x, omit={'a': ['b']})",
                "`omit` must be a scalar or a list",
            ),
            (
                "y ~ C(x, keep={'a': ['b']})",
                "`keep` must be a scalar or a list",
            ),
            (
                "y ~ poly(x, degree={'a': [1]})",
                "`degree` must be a positive integer",
            ),
            ("y ~ poly(x, raw={'a': [true]})", "`raw` must be a boolean"),
        ] {
            let error = parse(source).unwrap_err().to_string();
            assert!(error.contains(message), "{source}: {error}");
        }
    }

    #[test]
    fn powers_expand_to_interactions_up_to_their_degree() {
        for (source, expected) in [
            ("y ~ (a + b)^2", "y ~ a + b + a:b"),
            ("y ~ (a + b + c)^2", "y ~ a + b + c + a:b + a:c + b:c"),
            (
                "y ~ (a + b + c)^3",
                "y ~ (a + b + c)*(a + b + c)*(a + b + c)",
            ),
            ("y ~ (a + b)^1", "y ~ a + b"),
            ("y ~ (a + b)^4294967295", "y ~ a + b + a:b"),
            ("y ~ (a + b) ^ 2 + x", "y ~ a + b + a:b + x"),
            ("y ~ a^2", "y ~ a"),
            // `^` binds more tightly than `:` and `*`.
            ("y ~ x:(a + b)^2", "y ~ x:a + x:b + x:a:b"),
            ("y ~ x*(a + b)^2", "y ~ x*(a + b + a:b)"),
            ("y ~ (a + b)^2 - a:b", "y ~ a + b"),
            ("y ~ sw(a, b)^2", "y ~ sw(a, b)"),
            ("y ~ 0 + (a + b)^2", "y ~ 0 + a + b + a:b"),
        ] {
            assert_eq!(parse(source).unwrap(), parse(expected).unwrap(), "{source}");
        }
        for source in [
            "y ~ (a + b)^0",
            "y ~ (a + b)^x",
            "y ~ (a + b)^-1",
            "y ~ (a + b)^2.5",
            "y ~ (a + b)^",
            "y ~ x | (f + g)^2",
            "y ~ (x + [z ~ w])^2",
        ] {
            assert!(parse(source).is_err(), "{source}");
        }
    }

    #[test]
    fn polynomial_degree_may_be_positional() {
        for (source, expected) in [
            ("y ~ poly(x, 2)", "y ~ poly(x, degree=2)"),
            (
                "y ~ poly(x, 3, raw=true)",
                "y ~ poly(x, degree=3, raw=true)",
            ),
        ] {
            assert_eq!(parse(source).unwrap(), parse(expected).unwrap(), "{source}");
        }
        for (source, message) in [
            (
                "y ~ poly(x, 2, degree=2)",
                "both by position and as `degree`",
            ),
            ("y ~ poly(x, 2.5)", "`degree` must be a positive integer"),
            (
                "y ~ poly(x, z)",
                "expected one input and an optional positive integer degree",
            ),
        ] {
            let error = match parse(source) {
                Err(error) => error.to_string(),
                Ok(formulas) => formulas
                    .iter()
                    .next()
                    .unwrap()
                    .validate()
                    .unwrap_err()
                    .to_string(),
            };
            assert!(error.contains(message), "{source}: {error}");
        }
    }

    #[test]
    fn categorical_omit_and_keep_are_mutually_exclusive() {
        for source in [
            "y ~ C(x, omit='a', keep=['b'])",
            "y ~ C(x, keep=['b'], omit='a')",
        ] {
            let error = parse(source).unwrap_err().to_string();
            assert!(
                error.contains("`omit` and `keep` are mutually exclusive"),
                "{source}: {error}"
            );
        }
    }

    #[test]
    fn categorical_empty_omit_is_normalised_away() {
        for (source, expected) in [
            ("y ~ C(x, omit=[])", "y ~ 1 + C(x)"),
            ("y ~ C(x, omit=[], keep=['b'])", "y ~ 1 + C(x, keep=['b'])"),
            ("y ~ C(x, keep=['b'], omit=[])", "y ~ 1 + C(x, keep=['b'])"),
        ] {
            let formulas = parse(source).unwrap();
            assert_eq!(formulas, parse(expected).unwrap(), "{source}");
            let rendered = formulas.iter().map(ToString::to_string).collect::<Vec<_>>();
            assert_eq!(rendered, [expected], "{source}");
        }
    }

    #[test]
    fn categorical_bins_display_first_and_normalise_empty_dicts_away() {
        for (source, expected) in [
            (
                "y ~ C(x, omit='a', bin={'bc': ['b', 'c'], 'd': 'd'})",
                "y ~ 1 + C(x, bin={'bc': ['b', 'c'], 'd': 'd'}, omit='a')",
            ),
            ("y ~ C(x, bin={})", "y ~ 1 + C(x)"),
            ("y ~ C(x, bin={}, omit=[])", "y ~ 1 + C(x)"),
        ] {
            let formulas = parse(source).unwrap();
            assert_eq!(formulas, parse(expected).unwrap(), "{source}");
            let rendered = formulas.iter().map(ToString::to_string).collect::<Vec<_>>();
            assert_eq!(rendered, [expected], "{source}");
        }
    }

    #[test]
    fn categorical_rejects_malformed_bins() {
        for (source, message) in [
            ("y ~ C(x, bin='a')", "`bin` must be a dict"),
            ("y ~ C(x, bin=['a'])", "`bin` must be a dict"),
            (
                "y ~ C(x, bin={'b': []})",
                "bin 'b' must name at least one level",
            ),
            (
                "y ~ C(x, bin={'b': {'c': 'a'}})",
                "bin 'b' must be a scalar or a list",
            ),
        ] {
            let error = parse(source).unwrap_err().to_string();
            assert!(error.contains(message), "{source}: {error}");
        }
    }

    #[test]
    fn categorical_rejects_empty_keep() {
        for source in ["y ~ C(x, keep=[])", "y ~ C(x, omit=[], keep=[])"] {
            let error = parse(source).unwrap_err().to_string();
            assert!(
                error.contains("`keep` must name at least one level"),
                "{source}: {error}"
            );
        }
    }

    #[test]
    fn right_hand_sides_alone_have_no_response() {
        for (source, expected) in [
            ("0 + log(z)", "0 ~ 0 + log(z)"),
            ("x | f", "0 ~ x | f"),
            ("sw(x, z)", "0 ~ sw(x, z)"),
        ] {
            assert_eq!(parse(source).unwrap(), parse(expected).unwrap(), "{source}");
        }
        let formulas = parse("0 + log(z)").unwrap();
        assert!(formulas.get_by_str("log(z) - 1").unwrap().is_some());
    }

    #[test]
    fn ordinary_term_queries_reject_single_argument_stepwise_calls() {
        for role in [Role::Dependent, Role::Independent] {
            let error = parse_term("sw(x)", role).unwrap_err();
            assert_eq!(
                construction_error(error.inner()),
                &ConstructionError::Unsupported,
                "{role:?}: {error}"
            );
        }
    }

    mod varying_slopes {
        use super::*;

        #[test]
        fn normalization_preserves_grouping_and_deduplicates_terms() {
            for (source, expected) in [
                ("firm[time, time]", "firm[time]"),
                ("firm[time] + firm[time]", "firm[time]"),
                ("firm[[time, exposure, time]]", "firm[[time, exposure]]"),
                ("firm[time, exposure, time]", "firm[time, exposure]"),
                ("firm:year[time] + sector", "(firm:year)[time] + sector"),
                ("(firm + year)[time]", "firm[time] + year[time]"),
                (
                    "firm[[log(time), add(time, 2), log(time)]]",
                    "firm[[log(time), add(time, 2)]]",
                ),
                (
                    " `firm name` : year [[ time , exposure ]] ",
                    "(`firm name`:year)[[time, exposure]]",
                ),
            ] {
                assert_eq!(
                    parse(&format!("y ~ x | {source}")).unwrap(),
                    parse(&format!("y ~ x | {expected}")).unwrap(),
                    "{source}",
                );
            }
        }

        #[test]
        fn normalization_preserves_first_occurrence_order_and_round_trips() {
            let specification =
                parse("y ~ x | firm[[exposure, time, exposure]] + firm + firm[time]").unwrap();
            let formula = specification.iter().next().unwrap();
            let rendered = formula.to_string();
            assert_eq!(
                rendered,
                "y ~ x - 1 | firm[[exposure, time]] + firm + firm[time]",
            );
            assert_eq!(parse(&rendered).unwrap(), specification);
        }

        #[test]
        fn numeric_loadings_end_at_closing_brackets_and_preserve_literal_types() {
            for (source, value, intercept) in [
                ("firm[2]", Scalar::from(2_i64), true),
                ("firm[[2]]", Scalar::from(2_i64), false),
                ("firm[[2.0]]", Scalar::from(2_f64), false),
                ("firm[[-2.5]]", Scalar::from(-2.5_f64), false),
            ] {
                let node = parse_node::<typed::FixedEffects>.parse(source).unwrap();
                let typed::ExpressionAst::Embedded(_, Some(slopes)) = node else {
                    panic!("expected varying slopes for {source}");
                };
                assert!(slopes.rest.is_empty());
                let actual = slopes.first.literal().expect("expected a numeric literal");
                assert_eq!(actual.dtype(), value.dtype(), "{source}");
                assert_eq!(actual, &value, "{source}");
                assert_eq!(slopes.intercept, intercept, "{source}");
            }
        }

        #[test]
        fn global_intercept_depends_on_group_intercepts_and_explicit_removal() {
            for (source, intercept) in [
                ("y ~ x | firm[time]", false),
                ("y ~ x | firm[time, exposure]", false),
                ("y ~ x | firm[[time]]", true),
                ("y ~ x | firm:year[[time]]", true),
                ("y ~ x - 1 | firm[[time]]", false),
                ("y ~ x + 1 | firm[time]", false),
                ("y ~ x | firm[[time]] + year", false),
                ("y ~ x | firm[[time]] + year[[exposure]]", true),
            ] {
                let specification = parse(source).unwrap();
                let formula = specification.iter().next().unwrap();
                let expected = if intercept {
                    expression(&[&[], &["x"]])
                } else {
                    expression(&[&["x"]])
                };
                assert_eq!(
                    formula.expressions[&Role::Independent],
                    expected,
                    "{source}"
                );
            }
        }

        #[test]
        fn stepwise_expansion_preserves_loadings_and_local_intercept_policy() {
            for (source, expected) in [
                (
                    "y ~ x | sw(firm[time], year[[time]])",
                    vec!["y ~ x | firm[time]", "y ~ x | year[[time]]"],
                ),
                (
                    "y ~ x | sw0(firm[time])",
                    vec!["y ~ x", "y ~ x | firm[time]"],
                ),
                (
                    "y ~ x | csw(firm[[time]], year[exposure])",
                    vec![
                        "y ~ x | firm[[time]]",
                        "y ~ x | firm[[time]] + year[exposure]",
                    ],
                ),
                (
                    "y ~ x | sw(firm, year)[time]",
                    vec!["y ~ x | firm[time]", "y ~ x | year[time]"],
                ),
            ] {
                assert_eq!(
                    parse(source).unwrap(),
                    parse_many(expected).unwrap(),
                    "{source}"
                );
            }
        }

        #[test]
        fn malformed_or_unsupported_loading_syntax_is_rejected() {
            for source in [
                "firm[]",
                "firm[[]]",
                "firm[time,]",
                "firm[,time]",
                "firm[[time,]]",
                "firm[[,time]]",
                "firm[time",
                "firm[[time]",
                "firm[time]]",
                "firm[[time]]]",
                "firm[time + exposure]",
                "firm[time:exposure]",
                "firm[time[exposure]]",
                "firm[sw(time, exposure)]",
                "log(firm)[time]",
            ] {
                let source = format!("y ~ x | {source}");
                assert!(parse(&source).is_err(), "{source}");
            }
        }

        #[test]
        fn loadings_cannot_be_reused_as_grouping_operands() {
            for source in [
                "firm[[time]]:year",
                "firm:(year[[time]])",
                "(firm[[time]] + year):sector",
                "sw(firm, year[[time]]):sector",
                "firm[[time]][exposure]",
                "(firm[[time]])[exposure]",
            ] {
                let source = format!("y ~ x | {source}");
                let error = parse(&source).unwrap_err();
                assert_eq!(
                    construction_error(error.inner()),
                    &ConstructionError::Unsupported,
                    "{source}: {error}",
                );
            }
        }

        #[test]
        fn loadings_are_restricted_to_the_fixed_effect_pipe() {
            // Outside the fixed-effect section a bracket after a term is not
            // part of the grammar, so parsing stops at it.
            for (source, offset) in [
                ("y ~ x[time]", 5),
                ("y ~ x[[time]]", 5),
                ("y[time] ~ x", 1),
                ("y ~ [a ~ b[time]]", 10),
            ] {
                let error = parse(source).unwrap_err();
                assert_eq!(error.offset(), offset, "{source}: {error}");
            }
        }
    }

    mod multistage {
        use super::*;

        fn formula(source: &str) -> Formula {
            let specification = parse(source).unwrap();
            assert_eq!(specification.len(), 1, "{source}");
            specification.iter().next().unwrap().clone()
        }

        fn attach(mut parent: Formula, role: Role, children: Vec<Formula>) -> Formula {
            let expression = parent.expressions.remove(&role).unwrap();
            parent
                .expressions
                .insert(role, expression.add(Expression::new([], children)));
            parent
        }

        #[test]
        fn intercept_defaults_and_controls_are_local() {
            for (source, outer, child) in [
                ("y ~ x + [a ~ z]", "y ~ x", "a ~ z"),
                ("y ~ x - 1 + [a ~ z]", "y ~ x - 1", "a ~ z"),
                ("y ~ x + [a ~ z - 1]", "y ~ x", "a ~ z - 1"),
                ("y ~ x + [a ~ z | firm]", "y ~ x", "a ~ z | firm"),
                ("y ~ x + [a ~ z] | year", "y ~ x | year", "a ~ z"),
            ] {
                assert_eq!(
                    formula(source),
                    attach(formula(outer), Role::Independent, vec![formula(child)]),
                    "{source}"
                );
            }
        }

        #[test]
        fn siblings_and_descendants_retain_local_expressions() {
            let child = attach(formula("a ~ z"), Role::Independent, vec![formula("b ~ w")]);
            let expected = attach(
                formula("y ~ x"),
                Role::Independent,
                vec![child, formula("c ~ v")],
            );
            assert_eq!(formula("y ~ x + [a ~ z + [b ~ w]] + [c ~ v]"), expected);

            let source = "y ~ x + [log(a) + b:c ~ C(g) + poly(z, degree=2) | firm]";
            let expected = attach(
                formula("y ~ x"),
                Role::Independent,
                vec![formula("log(a) + b:c ~ C(g) + poly(z, degree=2) | firm")],
            );
            assert_eq!(formula(source), expected);
        }

        #[test]
        fn attachment_preserves_the_containing_role() {
            let child = formula("a ~ z");
            let dependent = attach(formula("0 ~ x"), Role::Dependent, vec![child.clone()]);
            let independent = attach(formula("0 ~ x"), Role::Independent, vec![child]);
            assert_eq!(formula("[a ~ z] ~ x"), dependent);
            assert_eq!(formula("0 ~ x + [a ~ z]"), independent);
            assert_ne!(dependent, independent);
        }

        #[test]
        fn subtraction_changes_only_local_terms() {
            for (source, outer) in [
                ("y ~ x + [a ~ z] - a", "y ~ x"),
                ("y ~ x + a + [a ~ z] - a", "y ~ x"),
                ("y ~ x + [a ~ z] - x", "y ~ 1"),
                ("y ~ x + [a ~ z] - (x + 1)", "y ~ 0"),
            ] {
                assert_eq!(
                    formula(source),
                    attach(formula(outer), Role::Independent, vec![formula("a ~ z")]),
                    "{source}"
                );
            }
        }

        #[test]
        fn subtraction_removes_whole_matching_children_without_entering_them() {
            for (source, expected) in [
                ("y ~ x + [a ~ z] - [a ~ z]", "y ~ x"),
                ("y ~ x + [a ~ z] + [b ~ w] - (x + [a ~ z])", "y ~ [b ~ w]"),
                (
                    "y ~ x + [a ~ z + [b ~ w]] - [b ~ w]",
                    "y ~ x + [a ~ z + [b ~ w]]",
                ),
                ("y ~ x + [a ~ z + [b ~ w]] - [a ~ z + [b ~ w]]", "y ~ x"),
            ] {
                assert_eq!(formula(source), formula(expected), "{source}");
            }
        }

        #[test]
        fn subtraction_expands_child_alternatives_before_matching() {
            assert_eq!(
                parse("y ~ [a ~ sw(x, z)] - [a ~ x]").unwrap(),
                parse_many(["y ~ 1", "y ~ [a ~ z]"]).unwrap(),
            );
        }

        #[test]
        fn stepwise_expansion_preserves_parent_child_order() {
            let actual = parse("y ~ sw(x, w) + [sw(a, b) ~ sw(z, v)] + [c ~ q]").unwrap();
            let mut expected = Vec::new();
            for outer in ["x", "w"] {
                for dependent in ["a", "b"] {
                    for independent in ["z", "v"] {
                        expected.push(attach(
                            formula(&format!("y ~ {outer}")),
                            Role::Independent,
                            vec![
                                formula(&format!("{dependent} ~ {independent}")),
                                formula("c ~ q"),
                            ],
                        ));
                    }
                }
            }
            assert_eq!(actual, expected.into_iter().collect::<Formulas>());
        }

        #[test]
        fn stepwise_zero_and_cumulative_children_preserve_intercept_removal() {
            let parent = formula("y ~ x - 1");
            let first = attach(parent.clone(), Role::Independent, vec![formula("a ~ z")]);
            let both = attach(first.clone(), Role::Independent, vec![formula("b ~ w")]);
            assert_eq!(
                parse("y ~ x - 1 + sw0([a ~ z])").unwrap(),
                [parent, first.clone()].into_iter().collect::<Formulas>()
            );
            assert_eq!(
                parse("y ~ x - 1 + csw([a ~ z], [b ~ w])").unwrap(),
                [first, both].into_iter().collect::<Formulas>()
            );
        }
    }

    fn construction_error(error: &ContextError) -> &ConstructionError {
        error
            .cause()
            .and_then(|cause| cause.downcast_ref::<ConstructionError>())
            .expect("expected an AST construction error")
    }

    mod syntax {
        use super::*;
        use typed::{ExpressionAst as Ast, SimpleFormula};

        /// The inner expression of an unannotated embedding.
        fn inner<L: typed::Layer>(node: Ast<L>) -> L::Inner {
            node.into_inner()
                .unwrap_or_else(|_| panic!("expected an unannotated embedding"))
        }

        fn simple(source: &str) -> Ast<SimpleFormula> {
            inner(
                parse_node::<typed::MultistageFormula>
                    .parse(source)
                    .unwrap(),
            )
        }

        fn assert_term(node: &Ast<SimpleFormula>, name: &str) {
            let Ast::Atom(term) = node else {
                panic!("expected term {name}");
            };
            assert_eq!(term, &Term::from(Factor::from(name)));
        }

        #[test]
        fn parsing_preserves_formula_sections_and_default_group_settings() {
            let formula = parse_formula_ast.parse("y ~ x | firm").unwrap();
            assert_term(&inner(formula.left.unwrap()), "y");
            assert_term(&inner(formula.right), "x");
            let Ast::Atom(column) = inner(formula.fixed_effects.unwrap()) else {
                panic!("expected a grouping column")
            };
            assert_eq!(column.name(), "firm");
        }

        #[test]
        fn child_formulas_remain_unexpanded() {
            let Ast::Atom(child) = parse_node::<typed::MultistageFormula>
                .parse("[a ~ sw(x, z)]")
                .unwrap()
            else {
                panic!("expected a child formula");
            };
            let Ast::Stepwise(_, _, first, rest) = inner(child.right) else {
                panic!("expected an unexpanded stepwise call");
            };
            assert_term(&first, "x");
            assert_eq!(rest.len(), 1);
            assert_term(&rest[0], "z");
        }

        #[test]
        fn grouping_column_names_are_preserved() {
            for (source, name) in [
                ("firm", "firm"),
                ("`log(x)`", "log(x)"),
                ("`0`", "0"),
                ("`1`", "1"),
                ("0x", "0x"),
                ("1x", "1x"),
            ] {
                let Ast::Atom(column) =
                    inner(parse_node::<typed::FixedEffects>.parse(source).unwrap())
                else {
                    panic!("expected a grouping column for {source}");
                };
                assert_eq!(column.name(), name, "{source}");
            }
        }

        #[test]
        fn loading_postfix_wraps_the_entire_grouping_interaction() {
            let Ast::Embedded(Ast::Interact(_, operands), Some(slopes)) =
                parse_node::<typed::FixedEffects>
                    .parse("firm:year[time]")
                    .unwrap()
            else {
                panic!("expected varying slopes on the whole interaction");
            };
            let (Ast::Atom(left), Ast::Atom(right)) = (*operands.left, *operands.right) else {
                panic!("expected grouping columns");
            };
            assert_eq!((left.name(), right.name()), ("firm", "year"));
            assert_eq!(
                slopes.loadings().cloned().collect::<Vec<_>>(),
                [Arc::new(Factor::from("time"))]
            );
            assert!(slopes.intercept);
        }

        #[test]
        fn parsing_preserves_subtraction() {
            let Ast::Subtract(_, operands) = simple("x - z") else {
                panic!("expected subtraction");
            };
            assert_term(&operands.left, "x");
            assert_term(&operands.right, "z");
        }

        #[test]
        fn parsing_does_not_cancel_double_negation() {
            let Ast::Negate(_, outer) = simple("--x") else {
                panic!("expected outer negation");
            };
            let Ast::Negate(_, inner) = *outer else {
                panic!("expected inner negation");
            };
            assert_term(&inner, "x");
        }

        #[test]
        fn parsing_preserves_grouped_negation_in_subtraction() {
            let Ast::Subtract(_, operands) = simple("x - (-z)") else {
                panic!("expected subtraction");
            };
            assert_term(&operands.left, "x");
            let Ast::Parenthesized(group) = *operands.right else {
                panic!("expected parentheses");
            };
            let Ast::Negate(_, value) = *group else {
                panic!("expected grouped negation");
            };
            assert_term(&value, "z");
        }

        #[test]
        fn parsing_preserves_stepwise_argument_scope() {
            let Ast::Stepwise(_, operator, first, rest) = simple("sw((x - 1), z)") else {
                panic!("expected stepwise call");
            };
            assert!(matches!(operator, StepwiseOperator::Sw));
            assert!(matches!(*first, Ast::Parenthesized(_)));
            assert_eq!(rest.len(), 1);
            assert_term(&rest[0], "z");
        }
    }

    #[test]
    fn fixed_effect_expressions_reject_intercept_literals_before_expansion() {
        for source in [
            "0",
            "1",
            "-0",
            "-1",
            "f + 0",
            "f - 1",
            "f + (1)",
            "(0 + f)",
            "f:1",
            "1/f",
            "sw(f, 0)",
            "csw(f, 1)",
            "sw0(0, f)",
            "csw0(f, 1)",
            "f + 1 - 1",
        ] {
            let formula = format!("y ~ x | {source}");
            let error = parse_formula_ast
                .parse(&formula)
                .err()
                .expect("expected fixed-effect construction failure");
            assert!(
                construction_error(error.inner()) == &ConstructionError::Unsupported,
                "{formula}: {error}"
            );
            assert!(
                error
                    .inner()
                    .context()
                    .any(|context| context == &StrContext::Label("fixed-effect expression")),
                "{formula}: {error}"
            );
        }
    }

    #[test]
    fn fixed_effect_expressions_report_unsupported_operators() {
        for source in [
            "firm - year",
            "-firm",
            "firm * year",
            "firm / year",
            "(firm - year)",
            "sw(firm, year * worker)",
        ] {
            let formula = format!("y ~ x | {source}");
            let error = parse_formula_ast
                .parse(&formula)
                .err()
                .expect("expected fixed-effect construction failure");
            assert_eq!(
                construction_error(error.inner()),
                &ConstructionError::Unsupported,
                "{formula}: {error}",
            );
            assert!(
                error
                    .inner()
                    .context()
                    .any(|context| { context == &StrContext::Label("fixed-effect expression") })
            );
        }
    }

    #[test]
    fn fixed_effect_expressions_reject_transform_calls_before_expansion() {
        for source in [
            "log(x)",
            "(f + log(x))",
            "sw(f, log(x))",
            "f + log(x) - log(x)",
        ] {
            let formula = format!("y ~ x | {source}");
            assert!(parse_formula_ast.parse(&formula).is_err(), "{formula}");
        }
    }

    #[test]
    fn fixed_effect_pipe_expands_groupings_and_stepwise_alternatives() {
        let cases: &[(&str, &[&str])] = &[
            ("y ~ x | (f1 + f2):f3", &["y ~ x | f1:f3 + f2:f3"]),
            ("y ~ x | sw0(f)", &["y ~ x", "y ~ x | f"]),
            (
                "y ~ x | csw0(f1, f2)",
                &["y ~ x", "y ~ x | f1", "y ~ x | f1 + f2"],
            ),
            (
                "y ~ sw(x, z) | sw0(f1):f2",
                &["y ~ x", "y ~ x | f1:f2", "y ~ z", "y ~ z | f1:f2"],
            ),
        ];
        for &(source, expanded) in cases {
            assert_eq!(
                parse(source).unwrap(),
                parse_many(expanded.iter().copied()).unwrap(),
                "{source}"
            );
        }
    }

    #[test]
    fn polynomial_defaults_and_keyword_order_preserve_identity() {
        for (source, equivalent) in [
            ("y ~ poly(x)", "y ~ poly(x, degree=1, raw=false)"),
            ("y ~ poly(x, degree=3)", "y ~ poly(x, raw=false, degree=3)"),
            (
                "y ~ poly(x, degree=3, raw=true)",
                "y ~ poly(x, raw=true, degree=3)",
            ),
            (
                "y ~ poly( x , raw = TRUE , degree = 3 )",
                "y ~ poly(x, degree=3, raw=true)",
            ),
        ] {
            assert_eq!(
                parse(source).unwrap(),
                parse(equivalent).unwrap(),
                "{source}"
            );
        }
        assert_ne!(
            parse("y ~ poly(x, degree=2)").unwrap(),
            parse("y ~ poly(x, degree=3)").unwrap(),
        );
        assert_ne!(
            parse("y ~ poly(x, raw=false)").unwrap(),
            parse("y ~ poly(x, raw=true)").unwrap(),
        );
    }

    #[test]
    fn polynomial_calls_construct_the_configured_transform() {
        for (source, degree, raw) in [
            ("poly(x)", 1_i64, false),
            ("poly(x, degree=3)", 3, false),
            ("poly(x, degree=2, raw=true)", 2, true),
        ] {
            let expected = Factor {
                kind: FactorKind::Call {
                    transform: Transform::Poly(Fit::Unfitted(
                        PolynomialDefinition::new(
                            Some(ParameterValue::Scalar(Scalar::from(degree))),
                            Some(ParameterValue::Scalar(Scalar::from(raw))),
                        )
                        .unwrap(),
                    )),
                    arguments: vec![Arc::new(Factor::from("x"))],
                },
            };
            assert_eq!(parse_factor.parse(source).unwrap(), expected, "{source}");
        }
    }

    #[test]
    fn invalid_polynomial_parameters_identify_the_transform_and_parameter() {
        for (source, parameter) in [
            ("y ~ poly(x, degree=0)", "degree"),
            ("y ~ poly(x, degree=3.0)", "degree"),
            ("y ~ poly(x, raw=1)", "raw"),
        ] {
            let error = parse(source).unwrap_err();
            let cause = error
                .inner()
                .cause()
                .and_then(|cause| cause.downcast_ref::<TransformCallError>());
            assert!(
                matches!(cause,
                    Some(TransformCallError::InvalidParameter { transform: "poly", message })
                        if message.contains(parameter)
                ),
                "{source}: {error}"
            );
        }
    }

    fn single_expression(source: &str) -> Expression {
        let [expression]: [_; 1] = parse_expression
            .parse(source)
            .unwrap()
            .try_into()
            .expect("expected one expression");
        expression
    }

    fn expression(terms: &[&[&str]]) -> Expression {
        terms
            .iter()
            .map(|factors| factors.iter().copied().map(Factor::from).collect::<Term>())
            .collect()
    }

    #[test]
    fn positional_numbers_preserve_numeric_types_and_values() {
        for (source, value) in [
            ("2", Scalar::from(2_i64)),
            ("2.0", Scalar::from(2.0_f64)),
            ("-2", Scalar::from(-2_i64)),
            ("-0.5", Scalar::from(-0.5_f64)),
            ("1e3", Scalar::from(1000.0_f64)),
            ("1.2E-3", Scalar::from(0.0012_f64)),
        ] {
            assert_eq!(
                parse_transform_argument_positional.parse(source).unwrap(),
                Factor {
                    kind: FactorKind::Literal(value)
                },
                "{source}"
            );
        }
    }

    #[test]
    fn positional_numbers_work_in_nested_calls() {
        let expected = Factor {
            kind: FactorKind::Call {
                transform: Transform::Expression(ExpressionTransform::MUL),
                arguments: vec![
                    Arc::new(Factor {
                        kind: FactorKind::Call {
                            transform: Transform::Expression(ExpressionTransform::ADD),
                            arguments: vec![
                                Arc::new(Factor::from("x")),
                                Arc::new(Factor {
                                    kind: FactorKind::Literal(Scalar::from(-2_i64)),
                                }),
                            ],
                        },
                    }),
                    Arc::new(Factor {
                        kind: FactorKind::Literal(Scalar::from(3.0_f64)),
                    }),
                ],
            },
        };
        assert_eq!(
            parse_factor.parse("mul(add(x, -2), 3.0)").unwrap(),
            expected
        );
    }

    #[test]
    fn positional_numbers_remain_distinct_from_columns_and_intercepts() {
        assert_eq!(
            parse_transform_argument_positional.parse("`2`").unwrap(),
            Factor::from("2"),
        );
        assert_eq!(
            parse_transform_argument_positional.parse("2x").unwrap(),
            Factor::from("2x"),
        );
        assert_ne!(
            parse_expression.parse("mul(x, 2)").unwrap(),
            parse_expression.parse("mul(x, `2`)").unwrap(),
        );
        assert_eq!(
            parse_expression.parse("1").unwrap(),
            vec![Expression::from(Term::intercept())],
        );
        assert_eq!(
            parse_expression.parse("1 - 1").unwrap(),
            vec![Expression::from_iter([])],
        );
    }

    #[test]
    fn quoted_columns_decode_literal_names() {
        for (source, name) in [
            ("`household income`", "household income"),
            ("`a/b`", "a/b"),
            ("`a+b-c*d:e~f,(g)=h`", "a+b-c*d:e~f,(g)=h"),
            ("`收入 €`", "收入 €"),
            ("` leading and trailing `", " leading and trailing "),
            ("``", ""),
            ("`a``b`", "a`b"),
            ("````", "`"),
            ("`1`", "1"),
        ] {
            assert_eq!(
                parse_column.parse(source).unwrap(),
                Factor::from(name),
                "{source}"
            );
        }
    }

    #[test]
    fn quoted_columns_work_throughout_formulas() {
        let call = |transform, arguments: Vec<Factor>| Factor {
            kind: FactorKind::Call {
                transform,
                arguments: arguments.into_iter().map(Arc::new).collect(),
            },
        };
        let nested = call(
            Transform::Expression(ExpressionTransform::LOG),
            vec![call(
                Transform::Expression(ExpressionTransform::SQRT),
                vec![Factor::from("household income")],
            )],
        );
        let categorical = call(
            Transform::C(Fit::Unfitted(CategoricalDefinition::new(
                Some(LevelSelection::Omit(Levels::Scalar(Scalar::from(
                    PlSmallStr::from_static("a"),
                )))),
                None,
            ))),
            vec![Factor::from("group name")],
        );

        for (source, right) in [
            (
                "`outcome value` ~ `household income`",
                vec![expression(&[&["household income"]])],
            ),
            (
                "`outcome value` ~ log(sqrt(`household income`))",
                vec![Expression::from(Term::from(nested))],
            ),
            (
                "`outcome value` ~ C(`group name`, omit='a')",
                vec![Expression::from(Term::from(categorical))],
            ),
            (
                "`outcome value` ~ `income-growth`:`tax.rate` + `income-growth`*`a+b`",
                vec![expression(&[
                    &["income-growth", "tax.rate"],
                    &["income-growth"],
                    &["a+b"],
                    &["income-growth", "a+b"],
                ])],
            ),
            (
                "`outcome value` ~ sw(`household income`, `a``b`)",
                vec![
                    expression(&[&["household income"]]),
                    expression(&[&["a`b"]]),
                ],
            ),
        ] {
            let expected = Formulas::new(
                right
                    .into_iter()
                    .map(|right| Formula {
                        expressions: BTreeMap::from([
                            (Role::Dependent, expression(&[&["outcome value"]])),
                            (
                                Role::Independent,
                                Expression::from(Term::intercept()).add(right),
                            ),
                            (Role::FixedEffect, Expression::from_iter([])),
                        ]),
                    })
                    .collect(),
            );
            assert_eq!(parse(source).unwrap(), expected, "{source}");
        }
    }

    #[test]
    fn quoted_and_unquoted_columns_share_identity() {
        assert_eq!(parse("y ~ x + `x`").unwrap(), parse("y ~ x").unwrap());
    }

    #[test]
    fn quoted_numbers_are_columns_not_intercepts_or_literals() {
        let zero = parse_expression.parse("`0`").unwrap();
        assert_eq!(zero, vec![expression(&[&["0"]])]);
        assert_ne!(zero, parse_expression.parse("0").unwrap());
        let column = parse_expression.parse("`1`").unwrap();
        assert_eq!(column, vec![expression(&[&["1"]])]);
        assert_ne!(column, parse_expression.parse("1").unwrap());
        assert_ne!(
            parse_expression.parse("log(`1`)").unwrap(),
            parse_expression.parse("log(1)").unwrap(),
        );
    }

    #[test]
    fn parse_many_accepts_arrays_and_borrowed_iterators() {
        let source = String::from("y ~ sw(x, z)");
        let expected = parse(&source).unwrap();
        assert_eq!(parse_many([source.as_str()]).unwrap(), expected);

        let borrowed = [source.as_str()];
        let slice: &[&str] = &borrowed;
        assert_eq!(parse_many(slice.iter().copied()).unwrap(), expected);

        let sources = [source, String::from("y ~ x"), String::from("y ~ w")];
        let actual = parse_many(sources.iter().map(String::as_str)).unwrap();
        let expected = parse_many(["y ~ x", "y ~ z", "y ~ w"]).unwrap();
        assert_eq!(actual, expected);
    }

    #[test]
    fn parse_many_handles_empty_input() {
        let specification = parse_many(std::iter::empty()).unwrap();
        assert!(specification.is_empty());
    }

    #[test]
    fn multiverse_stepwise_expands_argument_subsets() {
        let cases: &[(&str, &[&str])] = &[
            ("y ~ mvsw(x)", &["y~1", "y~x"]),
            (
                "y ~ mvsw(a, b, c, d)",
                &[
                    "y~1",
                    "y~a",
                    "y~b",
                    "y~c",
                    "y~d",
                    "y~a+b",
                    "y~a+c",
                    "y~a+d",
                    "y~b+c",
                    "y~b+d",
                    "y~c+d",
                    "y~a+b+c",
                    "y~a+b+d",
                    "y~a+c+d",
                    "y~b+c+d",
                    "y~a+b+c+d",
                ],
            ),
            ("y ~ mvsw(x + z, w)", &["y~1", "y~x+z", "y~w", "y~x+z+w"]),
            (
                "y ~ mvsw(sw(x, z), w)",
                &["y~1", "y~x", "y~z", "y~w", "y~x+w", "y~z+w"],
            ),
            (
                "y ~ sw(mvsw(x, z), w)",
                &["y~1", "y~x", "y~z", "y~x+z", "y~w"],
            ),
            ("y ~ mvsw(x, x)", &["y~1", "y~x"]),
            (
                "y ~ a - 1 + mvsw(x, z)",
                &["y~a-1", "y~a+x-1", "y~a+z-1", "y~a+x+z-1"],
            ),
            ("y ~ mvsw(x, z):w", &["y~1", "y~x:w", "y~z:w", "y~x:w+z:w"]),
            (
                "y ~ x | mvsw(firm, year)",
                &["y~x", "y~x|firm", "y~x|year", "y~x|firm+year"],
            ),
        ];

        for &(source, expected) in cases {
            let actual = parse(source).unwrap();
            let expected = parse_many(expected.iter().copied()).unwrap();
            assert_eq!(actual.len(), expected.len(), "{source}");
            for formula in &expected {
                assert!(actual.iter().any(|value| value == formula), "{source}");
            }
        }
    }

    #[test]
    fn stepwise_operators_expand_in_order() {
        let cases: &[(&str, &[&str])] = &[
            ("sw(x, z)", &["x", "z"]),
            ("sw0(x, z)", &["1-1", "x", "z"]),
            ("csw(x, z)", &["x", "x + z"]),
            ("csw0(x, z)", &["1-1", "x", "x + z"]),
            ("sw(x)", &["x"]),
            ("sw( x + y,\n log(z) )", &["x + y", "log(z)"]),
            ("sw(x, sw(y, z))", &["x", "y", "z"]),
            (
                "csw(sw(x, y), sw(z, w))",
                &["x", "y", "x+z", "x+w", "y+z", "y+w"],
            ),
            ("sw(csw(x, y), z)", &["x", "x+y", "z"]),
            ("csw0(x + y, z)", &["1-1", "x+y", "x+y+z"]),
        ];

        for &(source, expected) in cases {
            let expected = expected
                .iter()
                .copied()
                .map(single_expression)
                .collect::<Vec<_>>();
            assert_eq!(
                parse_expression.parse(source).unwrap(),
                expected,
                "{source}"
            );
        }
    }

    #[test]
    fn stepwise_expansion_preserves_operators_and_precedence() {
        let cases: &[(&str, &[&str])] = &[
            ("a + sw(x:y):sw(w, v)", &["a+x:y:w", "a+x:y:v"]),
            ("sw(x, z) + sw(w, v)", &["x+w", "x+v", "z+w", "z+v"]),
            ("sw(x+y, z):w", &["x:w+y:w", "z:w"]),
            ("a - sw(x, a)", &["a", "a-a"]),
            ("sw(x, y)*z", &["x*z", "y*z"]),
            ("sw0(x)*z", &["z", "x*z"]),
            ("sw0(x):z", &["1-1", "x:z"]),
            ("z:sw0(x)", &["1-1", "z:x"]),
            ("csw0(x, y):z", &["1-1", "x:z", "x:z+y:z"]),
            ("sw0(a)/b", &["1-1", "a+a:b"]),
            ("sw(a, b)/c", &["a+a:c", "b+b:c"]),
            ("(a + sw(x, y)):z", &["a:z+x:z", "a:z+y:z"]),
            ("a + sw(x, y)*z", &["a+x+z+x:z", "a+y+z+y:z"]),
        ];

        for &(source, expected) in cases {
            let expected = expected
                .iter()
                .copied()
                .map(single_expression)
                .collect::<Vec<_>>();
            assert_eq!(
                parse_expression.parse(source).unwrap(),
                expected,
                "{source}"
            );
        }
    }

    #[test]
    fn multiple_estimation_expands_both_sides_with_the_rhs_seed() {
        let cases: &[(&str, &[&str])] = &[
            (
                "sw(y1, y2) ~ a + sw(x, z)",
                &["y1~a+x", "y1~a+z", "y2~a+x", "y2~a+z"],
            ),
            ("sw(y1, y2):w ~ x", &["y1:w~x", "y2:w~x"]),
            ("y ~ sw0(x, z)", &["y~1", "y~x", "y~z"]),
            ("y ~ csw0(x, z) - 1", &["y~1-1", "y~x-1", "y~x+z-1"]),
            ("y ~ sw0(x, z):w", &["y~1", "y~x:w", "y~z:w"]),
            ("y ~ sw0(x):z - 1", &["y~1-1", "y~x:z-1"]),
            ("y ~ csw0(x, z):w - 1", &["y~1-1", "y~x:w-1", "y~x:w+z:w-1"]),
            ("y ~ sw(a, b)/c", &["y~a+a:c", "y~b+b:c"]),
            ("y ~ sw(1, x - 1)", &["y~1", "y~x"]),
            ("y ~ sw(0 + x, z)", &["y~x", "y~z"]),
        ];

        for &(source, expected) in cases {
            assert_eq!(
                parse(source).unwrap(),
                parse_many(expected.iter().copied()).unwrap(),
                "{source}"
            );
        }
    }

    #[test]
    fn zero_stepwise_alternatives_preserve_intercept_removal() {
        for (source, expanded) in [
            (
                "y ~ x - 1 + sw0(z, w)",
                ["y ~ x - 1", "y ~ x + z - 1", "y ~ x + w - 1"],
            ),
            (
                "y ~ x - 1 + csw0(z, w)",
                ["y ~ x - 1", "y ~ x + z - 1", "y ~ x + z + w - 1"],
            ),
        ] {
            assert_eq!(
                parse(source).unwrap(),
                parse_many(expanded).unwrap(),
                "{source}"
            );
        }
    }

    #[test]
    fn multiple_estimation_deduplicates_complete_formulas_stably() {
        let cases: &[(&str, &[&str])] = &[
            ("y ~ x + sw(x, x+x, z)", &["y~x", "y~x+z"]),
            ("sw(y1, y1, y2) ~ sw(x, x)", &["y1~x", "y2~x"]),
            ("y ~ sw(x:z, z:x)", &["y~x:z"]),
            ("y ~ sw(x+z, z+x)", &["y~x+z"]),
            ("y ~ sw(z+x, x+z)", &["y~z+x"]),
            ("sw(y, y-1) ~ x", &["y~x"]),
        ];

        for &(source, expected) in cases {
            assert_eq!(
                parse(source).unwrap(),
                parse_many(expected.iter().copied()).unwrap(),
                "{source}"
            );
        }
    }

    #[test]
    fn scalars_preserve_types_and_values() {
        let cases = [
            ("'a'", Scalar::from(PlSmallStr::from_static("a"))),
            ("''", Scalar::from(PlSmallStr::from_static(""))),
            ("'true'", Scalar::from(PlSmallStr::from_static("true"))),
            ("'a b'", Scalar::from(PlSmallStr::from_static("a b"))),
            ("true", Scalar::from(true)),
            ("TRUE", Scalar::from(true)),
            ("False", Scalar::from(false)),
            ("-12", Scalar::from(-12_i64)),
            ("1.5", Scalar::from(1.5_f64)),
            ("1e3", Scalar::from(1000.0_f64)),
            ("1.2E-3", Scalar::from(0.0012_f64)),
        ];

        for (source, expected) in cases {
            let actual = parse_scalar.parse(source).expect(source);
            assert_eq!(actual.dtype(), expected.dtype(), "{source}");
            assert_eq!(actual, expected, "{source}");
        }
    }

    #[test]
    fn scalar_boundaries_are_checked_without_consuming_delimiters() {
        let cases = [
            ("true,rest", Scalar::from(true), ",rest"),
            ("1.5)", Scalar::from(1.5_f64), ")"),
            ("1 rest", Scalar::from(1_i64), " rest"),
            ("false", Scalar::from(false), ""),
        ];

        for (source, expected, remainder) in cases {
            let mut cursor = source;
            assert_eq!(parse_scalar.parse_next(&mut cursor).unwrap(), expected);
            assert_eq!(cursor, remainder, "{source}");
        }
    }

    #[test]
    fn transforms_preserve_arguments_order_and_allowed_whitespace() {
        let categorical = |reference: Option<Levels>| {
            Transform::C(Fit::Unfitted(CategoricalDefinition::new(
                reference.map(LevelSelection::Omit),
                None,
            )))
        };
        let cases = [
            (
                "log(x)",
                Factor {
                    kind: FactorKind::Call {
                        transform: Transform::Expression(ExpressionTransform::LOG),
                        arguments: vec![Arc::new(Factor::from("x"))],
                    },
                },
            ),
            (
                "exp(x)",
                Factor {
                    kind: FactorKind::Call {
                        transform: Transform::Expression(ExpressionTransform::EXP),
                        arguments: vec![Arc::new(Factor::from("x"))],
                    },
                },
            ),
            (
                "C(group)",
                Factor {
                    kind: FactorKind::Call {
                        transform: categorical(None),
                        arguments: vec![Arc::new(Factor::from("group"))],
                    },
                },
            ),
            (
                "C(group, omit='a')",
                Factor {
                    kind: FactorKind::Call {
                        transform: categorical(Some(Levels::Scalar(Scalar::from(
                            PlSmallStr::from_static("a"),
                        )))),
                        arguments: vec![Arc::new(Factor::from("group"))],
                    },
                },
            ),
        ];

        for (source, expected) in cases {
            assert_eq!(
                parse_factor.parse(source).expect(source),
                expected,
                "{source}"
            );
        }

        let expected = Expression::from(Term::from(Factor {
            kind: FactorKind::Call {
                transform: categorical(Some(Levels::Scalar(Scalar::from(1_i64)))),
                arguments: vec![Arc::new(Factor::from("x"))],
            },
        }));
        for source in [
            "C(x, omit=1)",
            "C(x, omit= 1)",
            "C( x , omit = 1 )",
            " \n C(\tx,\nomit\t=\t1\n) \t",
        ] {
            assert_eq!(
                parse_expression.parse(source).expect(source),
                vec![expected.clone()],
                "{source}"
            );
        }
    }

    #[test]
    fn stepwise_name_prefixes_remain_valid_column_names() {
        for name in [
            "sw",
            "sw_value",
            "sw0_value",
            "csw_value",
            "csw0_value",
            "mvsw",
            "mvsw_value",
        ] {
            assert_eq!(
                parse_expression.parse(name).unwrap(),
                vec![expression(&[&[name]])],
                "{name}"
            );
        }
        assert_eq!(
            parse("sw ~ csw0 + sw0x").unwrap(),
            Formulas::new(vec![Formula {
                expressions: BTreeMap::from([
                    (Role::Dependent, expression(&[&["sw"]])),
                    (Role::Independent, expression(&[&[], &["csw0"], &["sw0x"]])),
                    (Role::FixedEffect, Expression::from_iter([])),
                ]),
            }]),
        );
    }

    #[test]
    fn expression_operators_preserve_precedence_grouping_and_deduplication() {
        let cases = [
            ("x*y", expression(&[&["x"], &["y"], &["x", "y"]])),
            ("x+y:z", expression(&[&["x"], &["y", "z"]])),
            ("(x+y):z", expression(&[&["x", "z"], &["y", "z"]])),
            (
                "x*y:z",
                expression(&[&["x"], &["y", "z"], &["x", "y", "z"]]),
            ),
            ("x+y*z", expression(&[&["x"], &["y"], &["z"], &["y", "z"]])),
            ("x+y-y", expression(&[&["x"]])),
            ("x+x", expression(&[&["x"]])),
            ("x:x", expression(&[&["x"]])),
            ("a:b + b:a", expression(&[&["a", "b"]])),
            ("a:b - b:a", expression(&[])),
            ("a*b*a", expression(&[&["a"], &["b"], &["a", "b"]])),
            (
                "a*b/c",
                expression(&[&["a"], &["b"], &["a", "b"], &["a", "b", "c"]]),
            ),
            (
                "a/b*c",
                expression(&[&["a"], &["a", "b"], &["c"], &["a", "c"], &["a", "b", "c"]]),
            ),
            (
                "a*(b/c)",
                expression(&[&["a"], &["b"], &["b", "c"], &["a", "b"], &["a", "b", "c"]]),
            ),
            ("1", expression(&[&[]])),
            ("1 + x + 1", expression(&[&[], &["x"]])),
            ("1:x", expression(&[&["x"]])),
            ("-1*x", expression(&[&["x"]])),
            ("0*x", expression(&[&["x"]])),
            ("(-1)*x", expression(&[&["x"]])),
            ("-(1*x)", expression(&[])),
            ("x*-1*z", expression(&[&["x"], &["z"], &["x", "z"]])),
        ];

        for (source, expected) in cases {
            assert_eq!(
                parse_expression.parse(source).expect(source),
                vec![expected],
                "{source}"
            );
        }
    }

    #[test]
    fn nesting_expands_grouped_operands() {
        for (source, expanded) in [
            ("a/b", "a + a:b"),
            ("(a+b)/c", "a + b + a:b:c"),
            ("a/(b+c)", "a + a:b + a:c"),
            ("(a+b)/(c+d)", "a + b + a:b:c + a:b:d"),
            ("a/b/c", "a + a:b + a:b:c"),
            ("a/a", "a"),
            ("a/b:c", "a + a:b:c"),
            ("a+b/c", "a + b + b:c"),
        ] {
            let [expected]: [_; 1] = parse_expression
                .parse(expanded)
                .unwrap()
                .try_into()
                .expect("expected one expression");
            assert_eq!(
                parse_expression.parse(source).unwrap(),
                vec![expected],
                "{source}"
            );
        }
    }

    #[test]
    fn nesting_with_empty_operands_does_not_add_terms() {
        assert_eq!(
            parse_expression.parse("(a-a)/b").unwrap(),
            vec![expression(&[])]
        );
        assert_eq!(
            parse_expression.parse("a/(b-b)").unwrap(),
            vec![expression(&[&["a"]])]
        );
    }

    #[test]
    fn nesting_keeps_intercept_subtraction_local_to_its_expression() {
        for (source, expanded, expected) in [
            (
                "y ~ a/b - 1",
                "y ~ a + a:b - 1",
                expression(&[&["a"], &["a", "b"]]),
            ),
            (
                "y ~ (a - 1)/b",
                "y ~ a + a:b",
                expression(&[&[], &["a"], &["a", "b"]]),
            ),
            (
                "y ~ a/(b - 1)",
                "y ~ a + a:b",
                expression(&[&[], &["a"], &["a", "b"]]),
            ),
        ] {
            let actual = parse(source).unwrap();
            assert_eq!(actual, parse(expanded).unwrap(), "{source}");
            assert_eq!(
                actual.iter().next().unwrap().expressions[&Role::Independent],
                expected,
                "{source}"
            );
        }
    }

    #[test]
    fn quoted_slashes_remain_part_of_column_names() {
        for (source, expected) in [
            ("`a/b`", expression(&[&["a/b"]])),
            ("`a/b`/c", expression(&[&["a/b"], &["a/b", "c"]])),
            ("a/`b/c`", expression(&[&["a"], &["a", "b/c"]])),
        ] {
            assert_eq!(
                parse_expression.parse(source).unwrap(),
                vec![expected],
                "{source}"
            );
        }
    }

    #[test]
    fn formulas_assign_roles_to_expressions() {
        let expected = Formula {
            expressions: BTreeMap::from([
                (Role::Dependent, expression(&[&["y"]])),
                (Role::Independent, expression(&[&[], &["x"], &["z"]])),
                (Role::FixedEffect, Expression::from_iter([])),
            ]),
        };

        for source in ["y~x+z", "y ~ x + z", " \n y \t~\n x + z \t"] {
            assert_eq!(
                parse(source).expect(source).iter().next().unwrap().clone(),
                expected,
                "{source}"
            );
        }

        assert_eq!(
            parse_expression.parse("x + z").unwrap(),
            vec![expression(&[&["x"], &["z"]])]
        );
    }

    #[test]
    fn transform_composition_preserves_inner_to_outer_order() {
        let cases = [
            (
                "log(C(x))",
                Factor {
                    kind: FactorKind::Call {
                        transform: Transform::Expression(ExpressionTransform::LOG),
                        arguments: vec![Arc::new(Factor {
                            kind: FactorKind::Call {
                                transform: Transform::C(Fit::Unfitted(CategoricalDefinition::new(
                                    None, None,
                                ))),
                                arguments: vec![Arc::new(Factor::from("x"))],
                            },
                        })],
                    },
                },
            ),
            (
                "C(log(x))",
                Factor {
                    kind: FactorKind::Call {
                        transform: Transform::C(Fit::Unfitted(CategoricalDefinition::new(
                            None, None,
                        ))),
                        arguments: vec![Arc::new(Factor {
                            kind: FactorKind::Call {
                                transform: Transform::Expression(ExpressionTransform::LOG),
                                arguments: vec![Arc::new(Factor::from("x"))],
                            },
                        })],
                    },
                },
            ),
            (
                "log(exp(x))",
                Factor {
                    kind: FactorKind::Call {
                        transform: Transform::Expression(ExpressionTransform::LOG),
                        arguments: vec![Arc::new(Factor {
                            kind: FactorKind::Call {
                                transform: Transform::Expression(ExpressionTransform::EXP),
                                arguments: vec![Arc::new(Factor::from("x"))],
                            },
                        })],
                    },
                },
            ),
        ];

        for (source, factor) in cases {
            let expected = Expression::from(Term::from(factor));
            assert_eq!(
                parse_expression.parse(source).expect(source),
                vec![expected],
                "{source}"
            );
        }
    }

    #[test]
    fn leading_subtraction_uses_the_initial_intercept() {
        for (source, explicit) in [
            ("y ~ -x + x*z", "y ~ 1 - x + x*z"),
            ("y ~ -(x + 1)", "y ~ 1 - (x + 1)"),
            ("y ~ -1x", "y ~ 1 - 1x"),
        ] {
            assert_eq!(parse(source).unwrap(), parse(explicit).unwrap(), "{source}");
        }
    }

    #[test]
    fn leading_subtraction_is_local_in_unseeded_contexts() {
        assert_eq!(
            parse_expression.parse("-1 + x").unwrap(),
            vec![expression(&[&["x"]])],
        );

        for (source, equivalent) in [
            ("y ~ (-1 + x)", "y ~ x"),
            ("y ~ (-x + z)", "y ~ z"),
            ("y ~ sw(-1 + x, z)", "y ~ sw(x, z)"),
        ] {
            assert_eq!(
                parse(source).unwrap(),
                parse(equivalent).unwrap(),
                "{source}",
            );
        }
    }

    mod errors {
        use super::*;

        #[test]
        fn multistage_arithmetic_is_rejected_before_expansion() {
            for source in [
                "y ~ x * [a ~ z]",
                "y ~ [a ~ z] * x",
                "y ~ x : [a ~ z]",
                "y ~ [a ~ z] : x",
                "y ~ x / [a ~ z]",
                "y ~ [a ~ z] / x",
                "y ~ -[a ~ z]",
                "y ~ --[a ~ z]",
                "y ~ x - -[a ~ z]",
                "y ~ x * ([a ~ z] + w)",
                "y ~ ([a ~ z] + w) : x",
                "y ~ -([a ~ z] + w)",
                "y ~ x * sw(w, [a ~ z])",
                "y ~ x / sw0([a ~ z])",
                "y ~ -sw([a ~ z], w)",
                "y ~ x + [a ~ z * [b ~ w]]",
            ] {
                // Construction rejects operations forbidden for multistage expressions.
                let error = parse_formula_ast
                    .parse(source)
                    .err()
                    .expect("expected construction failure");
                assert!(
                    construction_error(error.inner()) == &ConstructionError::Unsupported,
                    "{source}: {error}"
                );
            }
        }

        #[test]
        fn fixed_effect_expressions_reject_multistage_blocks_in_any_grouping() {
            for source in [
                "y ~ x | [a ~ z]",
                "y ~ x | firm + [a ~ z]",
                "y ~ x | ([a ~ z])",
                "y ~ x | sw(firm, [a ~ z])",
                "y ~ x | csw0(firm, [a ~ z])",
                "y ~ x + [a ~ z | [b ~ w]]",
            ] {
                assert!(parse_formula_ast.parse(source).is_err(), "{source}");
            }
        }

        #[test]
        fn missing_multistage_closing_brackets_report_the_expected_delimiter() {
            for source in [
                "y ~ x + [a ~ z",
                "y ~ [a ~ z | firm",
                "y ~ [a ~ z + [b ~ w]",
            ] {
                let error = parse(source).unwrap_err();
                assert!(
                    error.inner().context().any(|context| {
                        context
                            == &StrContext::Expected(StrContextValue::Description(
                                "']' to close the nested formula",
                            ))
                    }),
                    "{source}: {error}"
                );
            }
        }

        #[test]
        fn malformed_multistage_formulas_are_rejected() {
            for source in [
                "y ~ []",
                "y ~ [a]",
                "y ~ [~ z]",
                "y ~ [a ~]",
                "y ~ [a ~ z]]",
                "y ~ [[a ~ z]]",
                "y ~ [a ~ z)",
            ] {
                assert!(parse(source).is_err(), "{source}");
            }
        }

        #[test]
        fn formula_boundaries_report_context_and_offsets() {
            for (source, expected, offset) in [
                (
                    "~ x",
                    StrContext::Expected(StrContextValue::Description("an expression")),
                    0,
                ),
                (
                    "y x",
                    StrContext::Expected(StrContextValue::Description(
                        "a formula operator or end of input",
                    )),
                    2,
                ),
                ("y ~", StrContext::Label("right-hand expression"), 3),
                ("y ~   ", StrContext::Label("right-hand expression"), 6),
                (
                    "y ~ x C(group)",
                    StrContext::Expected(StrContextValue::Description(
                        "a formula operator or end of input",
                    )),
                    6,
                ),
                (
                    "y ~ x)",
                    StrContext::Expected(StrContextValue::Description(
                        "a formula operator or end of input",
                    )),
                    5,
                ),
            ] {
                let error = parse(source).unwrap_err();
                assert_eq!(error.offset(), offset, "{source}");
                assert!(
                    error.inner().context().any(|context| context == &expected),
                    "{source}"
                );
            }
        }

        #[test]
        fn missing_operands_and_parentheses_report_specific_expectations() {
            for (source, expected, offset) in [
                ("y ~ x +", "an expression", 7),
                ("y ~ ()", "an expression", 5),
                ("y ~ (x + )", "an expression", 9),
                ("y ~ (x", "')' to close the parenthesized expression", 6),
                (
                    "y ~ (x + z",
                    "')' to close the parenthesized expression",
                    10,
                ),
                ("y ~ (C(x", "',' or ')' after an argument", 8),
                ("y ~ sw(C(x", "',' or ')' after an argument", 10),
            ] {
                let error = parse(source).unwrap_err();
                assert_eq!(error.offset(), offset, "{source}");
                let expectations = error
                    .inner()
                    .context()
                    .filter_map(|context| match context {
                        StrContext::Expected(value) => Some(value),
                        _ => None,
                    })
                    .collect::<Vec<_>>();
                assert_eq!(
                    expectations,
                    vec![&StrContextValue::Description(expected)],
                    "{source}"
                );
            }
        }

        #[test]
        fn unsupported_keywords_preserve_the_transform_and_keyword_names() {
            for (source, transform, keyword) in [
                ("y ~ C(x, unknown=1)", "C", "unknown"),
                ("y ~ poly(x, unknown=1)", "poly", "unknown"),
                ("y ~ C(log(x, base=2))", "log", "base"),
            ] {
                let error = parse(source).unwrap_err();
                let cause = error
                    .inner()
                    .cause()
                    .and_then(|cause| cause.downcast_ref::<TransformCallError>());
                assert!(
                    matches!(
                        cause,
                        Some(TransformCallError::UnexpectedKeywords {
                            transform: actual_transform,
                            keywords,
                        }) if *actual_transform == transform && keywords == &[keyword]
                    ),
                    "{source}"
                );
                assert!(
                    !error
                        .inner()
                        .context()
                        .any(|context| matches!(context, StrContext::Expected(_))),
                    "{source}"
                );
            }
        }

        #[test]
        fn parse_many_preserves_the_failing_source_and_diagnostic() {
            for failing in ["y ~", "y ~ sw(x,)", "y ~ unknown(x)"] {
                let sources = ["y ~ x", failing, "y ~ z"];
                let error = parse_many(sources).unwrap_err();
                let standalone = parse(failing).unwrap_err();

                assert_eq!(*error.input(), failing);
                assert_eq!(error.offset(), standalone.offset(), "{failing}");
                assert_eq!(error.inner(), standalone.inner(), "{failing}");
            }
        }

        #[test]
        fn diagnostics_use_byte_offsets_with_unicode_and_newlines() {
            let source = "y ~ (\n `収入` + )";
            let error = parse(source).unwrap_err();
            let offset = source.find(')').unwrap();

            assert_eq!(error.offset(), offset);
            assert!(offset > source[..offset].chars().count());
            assert!(error.inner().context().any(|context| {
                context == &StrContext::Expected(StrContextValue::Description("an expression"))
            }));
        }

        #[test]
        fn call_syntax_errors_precede_call_construction_errors() {
            let source = "y ~ unknown(x";
            let error = parse(source).unwrap_err();

            assert_eq!(error.offset(), source.len());
            assert!(error.inner().cause().is_none());
            assert!(error.inner().context().any(|context| {
                context
                    == &StrContext::Expected(StrContextValue::Description(
                        "',' or ')' after an argument",
                    ))
            }));
        }

        #[test]
        fn unterminated_quoted_columns_are_rejected() {
            for source in ["`", "`income", "`a``", "```", "log(`income)", "x + `income"] {
                assert!(parse_expression.parse(source).is_err(), "{source}");
            }
        }

        #[test]
        fn stepwise_calls_reject_malformed_arguments_and_name_spacing() {
            for source in [
                "y ~ sw( )",
                "y ~ sw(,x)",
                "y ~ sw(x,,y)",
                "y ~ sw(x))",
                "y ~ sw(x +)",
                "y ~ sw(x, sw())",
                "y ~ log(sw(x, z))",
                "y ~ sw(x, reference=1)",
                "y ~ mvsw()",
                "y ~ mvsw(,x)",
                "y ~ mvsw(x,,y)",
                "y ~ mvsw(x +)",
            ] {
                assert!(parse(source).is_err(), "{source}");
            }
            for name in ["sw", "sw0", "csw", "csw0", "mvsw"] {
                let source = format!("y ~ {name} (x)");
                assert!(parse(&source).is_err(), "{source}");
            }
        }

        #[test]
        fn invalid_transform_calls_are_rejected() {
            for source in ["C (x)", "C(x, omit=1 2)", "C(x, omit=tr ue)", "F(x)"] {
                assert!(parse_expression.parse(source).is_err(), "{source}");
            }
        }

        #[test]
        fn duplicate_keywords_identify_the_keyword() {
            for (source, keyword) in [
                ("y ~ C(x, omit='a', omit='b')", "omit"),
                ("y ~ poly(x, degree=2, degree=3)", "degree"),
            ] {
                let error = parse(source).unwrap_err();
                let cause = error
                    .inner()
                    .cause()
                    .and_then(|cause| cause.downcast_ref::<TransformCallError>());

                assert!(
                    matches!(
                        cause,
                        Some(TransformCallError::DuplicateKeyword { name }) if name == keyword
                    ),
                    "{source}"
                );
            }
        }

        #[test]
        fn positional_arguments_after_keywords_identify_the_position() {
            for (source, expected) in [("y ~ C(x, omit='a', y)", 3), ("y ~ poly(degree=2, x)", 2)] {
                let error = parse(source).unwrap_err();
                let cause = error
                    .inner()
                    .cause()
                    .and_then(|cause| cause.downcast_ref::<TransformCallError>());

                assert!(
                    matches!(
                        cause,
                        Some(TransformCallError::PositionalAfterKeyword { position }) if *position == expected
                    ),
                    "{source}"
                );
            }
        }

        #[test]
        fn unknown_transforms_identify_the_name() {
            for source in [
                "y ~ unknown(x)",
                "y ~ C(unknown(x))",
                "y ~ sw(unknown(x))",
                "y ~ (unknown(x))",
            ] {
                let error = parse(source).unwrap_err();
                let cause = error
                    .inner()
                    .cause()
                    .and_then(|cause| cause.downcast_ref::<TransformCallError>());

                assert!(
                    matches!(
                        cause,
                        Some(TransformCallError::UnknownTransform(name))
                            if name == "unknown"
                    ),
                    "{source}"
                );
                assert!(
                    !error
                        .inner()
                        .context()
                        .any(|context| matches!(context, StrContext::Expected(_))),
                    "{source}"
                );
            }
        }

        #[test]
        fn missing_keyword_value_reports_its_context_and_offset() {
            let source = "y ~ C(x, omit=)";
            let error = parse(source).unwrap_err();

            assert_eq!(error.offset(), source.find(')').unwrap());
            assert!(
                error
                    .inner()
                    .context()
                    .any(|context| context == &StrContext::Label("keyword value"))
            );
            assert!(error.inner().context().any(|context| {
                context == &StrContext::Expected(StrContextValue::Description("a scalar value"))
            }));
        }

        #[test]
        fn unterminated_quotes_report_the_expected_delimiter() {
            for (source, expected) in [
                ("y ~ C(x, omit='a)", "a closing single quote"),
                ("y ~ `household income", "a closing backtick"),
                ("y ~ C(`household income)", "a closing backtick"),
            ] {
                let error = parse(source).unwrap_err();

                assert_eq!(error.offset(), source.len(), "{source}");
                assert!(
                    error.inner().context().any(|context| {
                        context == &StrContext::Expected(StrContextValue::Description(expected))
                    }),
                    "{source}"
                );
                for generic in ["a scalar value", "a transform argument"] {
                    assert!(
                        !error.inner().context().any(|context| {
                            context == &StrContext::Expected(StrContextValue::Description(generic))
                        }),
                        "{source}: unexpected generic expectation {generic}"
                    );
                }
            }
        }

        #[test]
        fn malformed_argument_lists_report_expectations_and_offsets() {
            for (source, expected, offset) in [
                ("y ~ C(x,)", "a transform argument", 8),
                ("y ~ C(x,,y)", "a transform argument", 8),
                ("y ~ C(x y)", "',' or ')' after an argument", 8),
                ("y ~ C(x", "',' or ')' after an argument", 7),
            ] {
                let error = parse(source).unwrap_err();

                assert_eq!(error.offset(), offset, "{source}");
                assert!(
                    error.inner().context().any(|context| {
                        context == &StrContext::Expected(StrContextValue::Description(expected))
                    }),
                    "{source}"
                );
            }
        }

        #[test]
        fn malformed_stepwise_arguments_report_expectations_and_offsets() {
            for (source, expected, offset) in [
                ("y ~ sw()", "an expression", 7),
                ("y ~ sw(x,)", "an expression", 9),
                ("y ~ sw(x y)", "',' or ')' after an expression", 9),
                ("y ~ sw(x", "',' or ')' after an expression", 8),
            ] {
                let error = parse(source).unwrap_err();

                assert_eq!(error.offset(), offset, "{source}");
                assert!(
                    error.inner().context().any(|context| {
                        context == &StrContext::Expected(StrContextValue::Description(expected))
                    }),
                    "{source}"
                );
            }
        }

        #[test]
        fn formula_parser_rejects_incomplete_or_trailing_input() {
            for source in [
                "",
                " ",
                "y ~",
                "~ x",
                "y ~ x ~ z",
                "y ~ x C(group)",
                "y ~ (x + z",
                "y ~ x)",
                "y ~ x +",
                "y ~ x ?",
                "y ~ a/",
                "y ~ /a",
                "y ~ x |",
                "y ~ x | f | g",
            ] {
                assert!(parse(source).is_err(), "{source:?}");
            }
        }

        #[test]
        fn expression_parser_rejects_incomplete_or_trailing_input() {
            for source in [
                "",
                " ",
                "y ~ x",
                "x C(group)",
                "(x + z",
                "x)",
                "x +",
                "x ?",
                "a/",
                "/a",
            ] {
                assert!(parse_expression.parse(source).is_err(), "{source:?}");
            }
        }

        #[test]
        fn malformed_numeric_arguments_are_rejected() {
            for source in ["mul(x, 2.0x)", "mul(x, -)", "mul(x, 1.2.3)"] {
                assert!(parse_expression.parse(source).is_err(), "{source}");
            }
        }

        #[test]
        fn malformed_scalars_are_rejected() {
            for source in [
                "truex",
                "'a'x",
                "12abc",
                "'unterminated",
                "9223372036854775808",
                "1 2",
                "tr ue",
            ] {
                assert!(parse_scalar.parse(source).is_err(), "{source}");
            }
        }
    }
}

#[cfg(test)]
mod layers {
    use super::*;

    // Layers that were never declared in the AST: child formulas *and*
    // varying slopes over simple expressions, under two operator vocabularies.
    enum Additive {}
    impl ast::Layer for Additive {
        type Atom = Box<ast::FormulaAst>;
        type Operators = ast::MultistageOperations;
        type Inner = ast::ExpressionAst<ast::SimpleFormula>;
        type Annotation = VaryingSlopes;
        type Stepwise = ast::Permitted;
    }

    enum Algebraic {}
    impl ast::Layer for Algebraic {
        type Atom = Box<ast::FormulaAst>;
        type Operators = ast::FormulaOperations;
        type Inner = ast::ExpressionAst<ast::SimpleFormula>;
        type Annotation = VaryingSlopes;
        type Stepwise = ast::Permitted;
    }

    #[test]
    fn atoms_and_annotations_are_reusable_across_operator_vocabularies() {
        for source in ["x*y + [a ~ z]", "x*y + z[t]", "x*y + (z:w)[t]"] {
            let ast::ExpressionAst::Add(_, pair) = parse_node::<Additive>.parse(source).unwrap()
            else {
                panic!("expected outer addition: {source}");
            };
            let ast::ExpressionAst::Embedded(inner, None) = *pair.left else {
                panic!("inner subtree");
            };
            assert!(matches!(inner, ast::ExpressionAst::InteractFully(_, _)));
        }
        assert!(parse_node::<Additive>.parse("[a ~ x] * y[t]").is_err());
        let ast::ExpressionAst::InteractFully(_, pair) =
            parse_node::<Algebraic>.parse("[a ~ x] * y[t]").unwrap()
        else {
            panic!("outer interaction");
        };
        assert!(matches!(*pair.left, ast::ExpressionAst::Atom(_)));
        assert!(matches!(
            *pair.right,
            ast::ExpressionAst::Embedded(_, Some(_))
        ));
    }

    #[test]
    fn annotation_binds_as_a_pratt_postfix() {
        let ast::ExpressionAst::Add(_, pair) = parse_node::<Additive>.parse("x + y:z[t]").unwrap()
        else {
            panic!("outer addition");
        };
        let ast::ExpressionAst::Embedded(group, Some(_)) = *pair.right else {
            panic!("loaded group");
        };
        assert!(matches!(group, ast::ExpressionAst::Interact(_, _)));
        assert!(matches!(
            parse_node::<Additive>
                .parse("sw([a ~ x], (y:z)[t])")
                .unwrap(),
            ast::ExpressionAst::Stepwise(..)
        ));
        for source in ["[a ~ x][t]", "x[t][u]", "([a ~ x] + y)[t]", "sw(x[t], y):z"] {
            assert!(parse_node::<Additive>.parse(source).is_err(), "{source}");
        }
    }

    #[test]
    fn a_third_layer_reuses_the_entire_fixed_effect_grammar() {
        enum Envelope {}
        impl ast::Layer for Envelope {
            type Atom = ast::Forbidden;
            type Operators = ast::FixedEffectOperations;
            type Inner = ast::ExpressionAst<ast::FixedEffects>;
            type Annotation = ast::Forbidden;
            type Stepwise = ast::Permitted;
        }
        for source in [
            "firm:year + sector[t]",
            "sw(firm, year)[t]",
            "(firm[t] + year)",
        ] {
            let expression = parse_node::<Envelope>.parse(source).unwrap();
            assert!(expression.into_inner().is_ok(), "{source}");
        }
    }

    #[test]
    fn mixed_layer_example_keeps_inner_interaction_and_outer_parentheses() {
        assert_eq!(
            parse("out ~ x*y + ([x ~ z] + w)").unwrap(),
            parse("out ~ x + y + x:y + w + [x ~ z]").unwrap()
        );
        let node = parse_node::<ast::MultistageFormula>
            .parse("x*y + ([x ~ z] + w)")
            .unwrap();
        let ast::ExpressionAst::Add(_, pair) = node else {
            panic!("outer addition");
        };
        let ast::ExpressionAst::Embedded(inner, None) = *pair.left else {
            panic!("inner expression operand");
        };
        assert!(matches!(inner, ast::ExpressionAst::InteractFully(_, _)));
        assert!(matches!(*pair.right, ast::ExpressionAst::Parenthesized(_)));
    }
}
