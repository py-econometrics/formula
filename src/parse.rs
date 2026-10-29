use crate::fit::Fit;
use crate::formula::{Expression, Factor, Formula, Specification, Term, push_unique};
use crate::transform::{
    CategoricalDefinition, ExpressionTransform, IntegerEncodingDefinition, PolynomialDefinition,
    Transform,
};
use polars::prelude::{PlSmallStr, Scalar};
use std::collections::BTreeMap;
use std::sync::Arc;
use winnow::ascii::{Caseless, multispace0};
use winnow::combinator::Infix::{Left, Right};
use winnow::combinator::{
    alt, cut_err, delimited, eof, expression, fail, peek, preceded, repeat, separated, terminated,
};
use winnow::error::{StrContext, StrContextValue};
use winnow::token::{one_of, take_while};
use winnow::{ModalResult, Parser};

#[derive(Debug)]
struct TransformCall {
    name: String,
    arguments: Vec<TransformCallArgument>,
}

#[derive(Debug)]
enum TransformCallArgument {
    Positional(Factor),
    Keyword { name: String, value: Scalar },
}

#[derive(Debug)]
enum TransformCallError {
    UnknownTransform(String),
    DuplicateKeyword {
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

fn is_identifier_character(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

fn is_argument_boundary(c: char) -> bool {
    c.is_whitespace() || matches!(c, ',' | ')')
}

fn parse_bare_token<'i>(cursor: &mut &'i str) -> ModalResult<&'i str> {
    take_while(1.., |c: char| !is_argument_boundary(c)).parse_next(cursor)
}

fn parse_scalar(cursor: &mut &str) -> ModalResult<Scalar> {
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

fn parse_literal_string(cursor: &mut &str) -> ModalResult<Scalar> {
    preceded(
        '\'',
        cut_err(terminated(
            take_while(0.., |c: char| c != '\''),
            '\''.context(StrContext::Expected(StrContextValue::Description(
                "a closing single quote",
            ))),
        )),
    )
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

fn parse_intercept(cursor: &mut &str) -> ModalResult<Expression> {
    take_while(1.., is_identifier_character)
        .verify(|name: &str| name == "1")
        .value(Expression::from(Term::intercept()))
        .parse_next(cursor)
}

// Support a leading suppressed intercept, e.g., `y ~ -1 + x`
fn parse_suppressed_intercept(cursor: &mut &str) -> ModalResult<Expression> {
    preceded(('-', multispace0), parse_intercept)
        .map(|intercept| Expression::from_iter([]).subtract(intercept))
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
            .map(Scalar::from),
    ))
    .parse_next(cursor)
}

fn parse_argument_boundary(cursor: &mut &str) -> ModalResult<()> {
    peek(alt((one_of(is_argument_boundary).void(), eof.void()))).parse_next(cursor)
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
        terminated(parse_literal_number, parse_argument_boundary).map(Factor::Literal),
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

    let value =
        cut_err(parse_scalar.context(StrContext::Label("keyword value"))).parse_next(cursor)?;

    Ok(TransformCallArgument::Keyword {
        name: name.to_owned(),
        value,
    })
}

fn combine_expressions(
    left: &[Expression],
    right: &[Expression],
    combine: impl Fn(Expression, Expression) -> Expression,
) -> Vec<Expression> {
    let mut expressions = Vec::new();
    for left_expression in left {
        for right_expression in right {
            expressions.push(combine(left_expression.clone(), right_expression.clone()));
        }
    }
    expressions
}

pub(crate) fn parse_expression(cursor: &mut &str) -> ModalResult<Vec<Expression>> {
    expression(parse_operand)
        .infix(alt((
            '+'.value(Left(
                10,
                |_, left: Vec<Expression>, right: Vec<Expression>| {
                    Ok(combine_expressions(&left, &right, |left, right| {
                        left.add(right)
                    }))
                },
            )),
            '-'.value(Left(
                10,
                |_, left: Vec<Expression>, right: Vec<Expression>| {
                    Ok(combine_expressions(&left, &right, |left, right| {
                        left.subtract(right)
                    }))
                },
            )),
            '*'.value(Left(
                20,
                |_, left: Vec<Expression>, right: Vec<Expression>| {
                    Ok(combine_expressions(&left, &right, |left, right| {
                        left.interact_fully(right)
                    }))
                },
            )),
            '/'.value(Left(
                20,
                |_, left: Vec<Expression>, right: Vec<Expression>| {
                    Ok(combine_expressions(&left, &right, |left, right| {
                        left.nest(right)
                    }))
                },
            )),
            ':'.value(Right(
                30,
                |_, left: Vec<Expression>, right: Vec<Expression>| {
                    Ok(combine_expressions(&left, &right, |left, right| {
                        left.interact(&right)
                    }))
                },
            )),
        )))
        .parse_next(cursor)
}

fn parse_stepwise_arguments(cursor: &mut &str) -> ModalResult<Vec<Vec<Expression>>> {
    multispace0.parse_next(cursor)?;

    let arguments = separated(
        1..,
        cut_err(parse_expression),
        (multispace0, ',', multispace0),
    )
    .parse_next(cursor)?;

    multispace0.parse_next(cursor)?;

    ')'.context(StrContext::Expected(StrContextValue::Description(
        "',' or ')' after an expression",
    )))
    .parse_next(cursor)?;

    Ok(arguments)
}

fn expand_cumulative(arguments: Vec<Vec<Expression>>) -> Vec<Expression> {
    let mut prefixes = vec![Expression::from_iter([])];
    let mut expressions = Vec::new();

    for argument in arguments {
        prefixes = combine_expressions(&prefixes, &argument, |left, right| left.add(right));
        expressions.extend(prefixes.iter().cloned());
    }

    expressions
}

#[derive(Clone)]
enum Stepwise {
    Sw,
    Sw0,
    Csw,
    Csw0,
}

fn parse_stepwise(cursor: &mut &str) -> ModalResult<Vec<Expression>> {
    let operator = terminated(
        alt((
            "sw0".value(Stepwise::Sw0),
            "sw".value(Stepwise::Sw),
            "csw0".value(Stepwise::Csw0),
            "csw".value(Stepwise::Csw),
        )),
        '(',
    )
    .parse_next(cursor)?;

    let arguments = cut_err(parse_stepwise_arguments).parse_next(cursor)?;

    let mut expressions = match operator {
        Stepwise::Sw | Stepwise::Sw0 => arguments.into_iter().flatten().collect::<Vec<_>>(),
        Stepwise::Csw | Stepwise::Csw0 => expand_cumulative(arguments),
    };

    if matches!(operator, Stepwise::Sw0 | Stepwise::Csw0) {
        expressions.insert(0, Expression::from_iter([]));
    }

    Ok(expressions)
}

fn parse_expression_parenthesized(cursor: &mut &str) -> ModalResult<Vec<Expression>> {
    '('.parse_next(cursor)?;

    let expressions = cut_err(parse_expression).parse_next(cursor)?;

    cut_err(
        ')'.context(StrContext::Expected(StrContextValue::Description(
            "')' to close the parenthesized expression",
        ))),
    )
    .parse_next(cursor)?;

    Ok(expressions)
}

fn parse_operand(cursor: &mut &str) -> ModalResult<Vec<Expression>> {
    delimited(
        multispace0,
        alt((
            parse_expression_parenthesized,
            parse_stepwise,
            alt((
                parse_suppressed_intercept,
                parse_intercept,
                parse_factor.map(Term::from).map(Expression::from),
            ))
            .map(|expression| vec![expression]),
            cut_err(
                fail.context(StrContext::Expected(StrContextValue::Description(
                    "an expression",
                ))),
            ),
        )),
        multispace0,
    )
    .parse_next(cursor)
}

fn parse_formula(cursor: &mut &str) -> ModalResult<Vec<Formula>> {
    multispace0.parse_next(cursor)?;

    let left = parse_expression
        .context(StrContext::Label("left-hand expression"))
        .parse_next(cursor)?;

    '~'.context(StrContext::Expected(StrContextValue::CharLiteral('~')))
        .parse_next(cursor)?;

    multispace0.parse_next(cursor)?;

    let right = cut_err(parse_expression.context(StrContext::Label("right-hand expression")))
        .parse_next(cursor)?;

    let left = left
        .into_iter()
        // By default no intercept on the left-hand expression of a formula
        .map(|expression| expression.resolve_intercept(false))
        .collect::<Vec<_>>();

    let right = right
        .into_iter()
        // By default add intercept on the right-hand expression of a formula
        .map(|expression| expression.resolve_intercept(true))
        .collect::<Vec<_>>();

    let mut formulas = Vec::new();
    for left in &left {
        for right in &right {
            formulas.push(Formula {
                left: left.clone(),
                right: right.clone(),
            });
        }
    }

    Ok(formulas)
}

pub(crate) type ParseResult<'i, T> =
    std::result::Result<T, winnow::error::ParseError<&'i str, winnow::error::ContextError>>;

pub fn parse(source: &str) -> ParseResult<'_, Specification> {
    parse_many(std::iter::once(source))
}

pub fn parse_many<'a>(input: impl IntoIterator<Item = &'a str>) -> ParseResult<'a, Specification> {
    let mut formulas: Vec<Formula> = Vec::new();

    for source in input {
        let expanded = terminated(
            parse_formula,
            eof.context(StrContext::Expected(StrContextValue::Description(
                "a formula operator or end of input",
            ))),
        )
        .parse(source)?;

        for formula in expanded {
            push_unique(&mut formulas, formula);
        }
    }

    Ok(Specification::new(formulas))
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
            "C" => {
                let reference = keywords.remove("reference");
                reject_keywords("C", keywords)?;
                Transform::C(Fit::Unfitted(CategoricalDefinition::new(reference)))
            }
            "F" => {
                reject_keywords("F", keywords)?;
                Transform::F(Fit::Unfitted(IntegerEncodingDefinition))
            }
            "poly" => {
                let degree = keywords.remove("degree");
                let raw = keywords.remove("raw");
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

        Ok(Self::Call {
            transform,
            arguments,
        })
    }
}

fn reject_keywords(
    transform: &'static str,
    keywords: BTreeMap<String, Scalar>,
) -> std::result::Result<(), TransformCallError> {
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
            let expected = Factor::Call {
                transform: Transform::Poly(Fit::Unfitted(
                    PolynomialDefinition::new(Some(Scalar::from(degree)), Some(Scalar::from(raw)))
                        .unwrap(),
                )),
                arguments: vec![Arc::new(Factor::from("x"))],
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
                Factor::Literal(value),
                "{source}"
            );
        }
    }

    #[test]
    fn positional_numbers_work_in_nested_calls() {
        let expected = Factor::Call {
            transform: Transform::Expression(ExpressionTransform::MUL),
            arguments: vec![
                Arc::new(Factor::Call {
                    transform: Transform::Expression(ExpressionTransform::ADD),
                    arguments: vec![
                        Arc::new(Factor::from("x")),
                        Arc::new(Factor::Literal(Scalar::from(-2_i64))),
                    ],
                }),
                Arc::new(Factor::Literal(Scalar::from(3.0_f64))),
            ],
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
            parse_expression.parse("-1").unwrap(),
            vec![Expression::from_iter([]).subtract(Expression::from(Term::intercept()))],
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
        let call = |transform, arguments: Vec<Factor>| Factor::Call {
            transform,
            arguments: arguments.into_iter().map(Arc::new).collect(),
        };
        let nested = call(
            Transform::Expression(ExpressionTransform::LOG),
            vec![call(
                Transform::Expression(ExpressionTransform::SQRT),
                vec![Factor::from("household income")],
            )],
        );
        let categorical = call(
            Transform::C(Fit::Unfitted(CategoricalDefinition::new(Some(
                Scalar::from(PlSmallStr::from_static("a")),
            )))),
            vec![Factor::from("group name")],
        );
        let integer = call(
            Transform::F(Fit::Unfitted(IntegerEncodingDefinition)),
            vec![Factor::from("income-growth"), Factor::from("tax.rate")],
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
                "`outcome value` ~ C(`group name`, reference='a')",
                vec![Expression::from(Term::from(categorical))],
            ),
            (
                "`outcome value` ~ F(`income-growth`, `tax.rate`)",
                vec![Expression::from(Term::from(integer))],
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
            let expected = Specification::new(
                right
                    .into_iter()
                    .map(|right| Formula {
                        left: expression(&[&["outcome value"]]),
                        right: right.resolve_intercept(true),
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
        assert_eq!(actual.formulas(), expected.formulas());
    }

    #[test]
    fn parse_many_handles_empty_input() {
        let specification = parse_many(std::iter::empty()).unwrap();
        assert!(specification.formulas().is_empty());
    }

    #[test]
    fn stepwise_operators_expand_in_order() {
        let cases: &[(&str, &[&str])] = &[
            ("sw(x, z)", &["x", "z"]),
            ("sw0(x, z)", &["x-x", "x", "z"]),
            ("csw(x, z)", &["x", "x + z"]),
            ("csw0(x, z)", &["x-x", "x", "x + z"]),
            ("sw(x)", &["x"]),
            ("sw( x + y,\n log(z) )", &["x + y", "log(z)"]),
            ("sw(x, sw(y, z))", &["x", "y", "z"]),
            (
                "csw(sw(x, y), sw(z, w))",
                &["x", "y", "x+z", "x+w", "y+z", "y+w"],
            ),
            ("sw(csw(x, y), z)", &["x", "x+y", "z"]),
            ("csw0(x + y, z)", &["x-x", "x+y", "x+y+z"]),
        ];

        for &(source, expected) in cases {
            let expected = expected
                .iter()
                .map(|source| {
                    let [expected]: [_; 1] = parse_expression
                        .parse(source)
                        .unwrap()
                        .try_into()
                        .expect("expected one expression");
                    expected
                })
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
            ("sw(a, b)/c", &["a+a:c", "b+b:c"]),
            ("(a + sw(x, y)):z", &["a:z+x:z", "a:z+y:z"]),
            ("a + sw(x, y)*z", &["a+x+z+x:z", "a+y+z+y:z"]),
        ];

        for &(source, expected) in cases {
            let expected = expected
                .iter()
                .map(|source| {
                    let [expected]: [_; 1] = parse_expression
                        .parse(source)
                        .unwrap()
                        .try_into()
                        .expect("expected one expression");
                    expected
                })
                .collect::<Vec<_>>();
            assert_eq!(
                parse_expression.parse(source).unwrap(),
                expected,
                "{source}"
            );
        }
    }

    #[test]
    fn multiple_estimation_expands_both_sides_and_resolves_intercepts() {
        let cases: &[(&str, &[&str])] = &[
            (
                "sw(y1, y2) ~ a + sw(x, z)",
                &["y1~a+x", "y1~a+z", "y2~a+x", "y2~a+z"],
            ),
            ("sw(y1, y2):w ~ x", &["y1:w~x", "y2:w~x"]),
            ("y ~ sw0(x, z)", &["y~1", "y~x", "y~z"]),
            ("y ~ csw0(x, z) - 1", &["y~1-1", "y~x-1", "y~x+z-1"]),
            ("y ~ sw0(x, z):w", &["y~1", "y~x:w", "y~z:w"]),
            ("y ~ sw(a, b)/c", &["y~a+a:c", "y~b+b:c"]),
            ("y ~ sw(1, x - 1)", &["y~1", "y~x-1"]),
        ];

        for &(source, expected) in cases {
            let expected = expected
                .iter()
                .map(|source| parse(source).unwrap().formulas()[0].clone())
                .collect::<Vec<_>>();
            assert_eq!(
                super::parse(source).unwrap(),
                Specification::new(expected),
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
            ("y ~ sw(x+z, z+x)", &["y~x+z", "y~z+x"]),
            ("sw(y, y-1) ~ x", &["y~x", "y-1~x"]),
        ];

        for &(source, expected) in cases {
            let expected = expected
                .iter()
                .map(|source| parse(source).unwrap().formulas()[0].clone())
                .collect::<Vec<_>>();
            assert_eq!(
                super::parse(source).unwrap(),
                Specification::new(expected),
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
        let categorical =
            |reference| Transform::C(Fit::Unfitted(CategoricalDefinition::new(reference)));
        let cases = [
            (
                "log(x)",
                Factor::Call {
                    transform: Transform::Expression(ExpressionTransform::LOG),
                    arguments: vec![Arc::new(Factor::from("x"))],
                },
            ),
            (
                "exp(x)",
                Factor::Call {
                    transform: Transform::Expression(ExpressionTransform::EXP),
                    arguments: vec![Arc::new(Factor::from("x"))],
                },
            ),
            (
                "C(group)",
                Factor::Call {
                    transform: categorical(None),
                    arguments: vec![Arc::new(Factor::from("group"))],
                },
            ),
            (
                "C(group, reference='a')",
                Factor::Call {
                    transform: categorical(Some(Scalar::from(PlSmallStr::from_static("a")))),
                    arguments: vec![Arc::new(Factor::from("group"))],
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

        let expected = Expression::from(Term::from(Factor::Call {
            transform: categorical(Some(Scalar::from(1_i64))),
            arguments: vec![Arc::new(Factor::from("x"))],
        }));
        for source in [
            "C(x, reference=1)",
            "C(x, reference= 1)",
            "C( x , reference = 1 )",
            " \n C(\tx,\nreference\t=\t1\n) \t",
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
        for name in ["sw", "sw_value", "sw0_value", "csw_value", "csw0_value"] {
            assert_eq!(
                parse_expression.parse(name).unwrap(),
                vec![expression(&[&[name]])],
                "{name}"
            );
        }
        assert_eq!(
            parse("sw ~ csw0 + sw0x").unwrap(),
            Specification::new(vec![Formula {
                left: expression(&[&["sw"]]),
                right: expression(&[&["csw0"], &["sw0x"]]).resolve_intercept(true),
            }]),
        );
    }

    #[test]
    fn integer_encoding_preserves_all_positional_arguments() {
        let cases = [
            ("F(x)", vec![Factor::from("x")]),
            ("F(x, x)", vec![Factor::from("x"), Factor::from("x")]),
            (
                "F(f1, f2, f3, f4)",
                vec![
                    Factor::from("f1"),
                    Factor::from("f2"),
                    Factor::from("f3"),
                    Factor::from("f4"),
                ],
            ),
            (
                "F(log(x), z)",
                vec![
                    Factor::Call {
                        transform: Transform::Expression(ExpressionTransform::LOG),
                        arguments: vec![Arc::new(Factor::from("x"))],
                    },
                    Factor::from("z"),
                ],
            ),
        ];

        for (source, arguments) in cases {
            let expected = Factor::Call {
                transform: Transform::F(Fit::Unfitted(IntegerEncodingDefinition)),
                arguments: arguments.into_iter().map(Arc::new).collect(),
            };
            assert_eq!(
                parse_factor.parse(source).expect(source),
                expected,
                "{source}"
            );
        }
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
    fn nesting_preserves_intercept_suppression() {
        for (source, expanded) in [
            ("y ~ a/b - 1", "y ~ a + a:b - 1"),
            ("y ~ (a - 1)/b", "y ~ a + a:b - 1"),
            ("y ~ a/(b - 1)", "y ~ a + a:b - 1"),
        ] {
            let actual = parse(source).unwrap();
            assert_eq!(actual, parse(expanded).unwrap(), "{source}");
            assert_eq!(
                actual.formulas()[0].right,
                expression(&[&["a"], &["a", "b"]]).subtract(Expression::from(Term::intercept())),
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
    fn formulas_keep_dependent_and_independent_expressions_separate() {
        let expected = Formula {
            left: expression(&[&["y"]]),
            right: expression(&[&[], &["x"], &["z"]]),
        };

        for source in ["y~x+z", "y ~ x + z", " \n y \t~\n x + z \t"] {
            assert_eq!(
                parse(source).expect(source).formulas()[0].clone(),
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
                Factor::Call {
                    transform: Transform::Expression(ExpressionTransform::LOG),
                    arguments: vec![Arc::new(Factor::Call {
                        transform: Transform::C(Fit::Unfitted(CategoricalDefinition::new(None))),
                        arguments: vec![Arc::new(Factor::from("x"))],
                    })],
                },
            ),
            (
                "C(log(x))",
                Factor::Call {
                    transform: Transform::C(Fit::Unfitted(CategoricalDefinition::new(None))),
                    arguments: vec![Arc::new(Factor::Call {
                        transform: Transform::Expression(ExpressionTransform::LOG),
                        arguments: vec![Arc::new(Factor::from("x"))],
                    })],
                },
            ),
            (
                "log(exp(x))",
                Factor::Call {
                    transform: Transform::Expression(ExpressionTransform::LOG),
                    arguments: vec![Arc::new(Factor::Call {
                        transform: Transform::Expression(ExpressionTransform::EXP),
                        arguments: vec![Arc::new(Factor::from("x"))],
                    })],
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

    mod errors {
        use super::*;

        #[test]
        fn formula_boundaries_report_context_and_offsets() {
            for (source, expected, offset) in [
                ("~ x", StrContext::Label("left-hand expression"), 0),
                (
                    "y x",
                    StrContext::Expected(StrContextValue::CharLiteral('~')),
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
                ("y ~ F(x, reference=1)", "F", "reference"),
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
            assert!(parse("y ~ `income").is_err());
        }

        #[test]
        fn stepwise_calls_reject_malformed_arguments_and_name_spacing() {
            for name in ["sw", "sw0", "csw", "csw0"] {
                for arguments in [
                    "()", "( )", "(,x)", "(x,)", "(x,,y)", "(x", "(x))", "(x +)", "(x y)", " (x)",
                ] {
                    let source = format!("y ~ {name}{arguments}");
                    assert!(super::parse(&source).is_err(), "{source}");
                }
            }
            for source in [
                "y ~ sw(x, sw())",
                "y ~ log(sw(x, z))",
                "y ~ sw(x, reference=1)",
            ] {
                assert!(super::parse(source).is_err(), "{source}");
            }
        }

        #[test]
        fn invalid_transform_calls_are_rejected() {
            for source in [
                "unknown(x)",
                "C (x)",
                "log(x, reference=1)",
                "C(reference=1, x)",
                "C(x, reference=1, reference=2)",
                "C(x, unknown=1)",
                "C(x, reference=1 2)",
                "C(x, reference=tr ue)",
                "F(x, unknown=1)",
            ] {
                assert!(parse_expression.parse(source).is_err(), "{source}");
            }
        }

        #[test]
        fn duplicate_keywords_identify_the_keyword() {
            for (source, keyword) in [
                ("y ~ C(x, reference='a', reference='b')", "reference"),
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
            for (source, expected) in [
                ("y ~ C(x, reference='a', y)", 3),
                ("y ~ poly(degree=2, x)", 2),
            ] {
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
            let source = "y ~ C(x, reference=)";
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
                ("y ~ C(x, reference='a)", "a closing single quote"),
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
                "x",
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
        fn leading_minus_only_accepts_intercept() {
            for source in ["y ~ -x + x*z", "y ~ -(x + 1)", "y ~ -1x"] {
                assert!(parse(source).is_err(), "{source}");
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
