use crate::fit::Fit;
use crate::formula::{Expression, Factor, Formula, Specification, Term, push_unique};
use crate::transform::{
    CategoricalDefinition, ExpressionTransform, IntegerEncodingDefinition, Transform,
};
use polars::prelude::{PlSmallStr, Scalar};
use std::collections::BTreeMap;
use std::sync::Arc;
use winnow::Parser;
use winnow::Result;
use winnow::ascii::{Caseless, multispace0};
use winnow::combinator::Infix::{Left, Right};
use winnow::combinator::{
    alt, delimited, eof, expression, peek, preceded, repeat, separated, separated_pair, terminated,
};
use winnow::token::{one_of, take_while};

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
enum TransformError {
    InvalidCall,
    Unknown(String),
    UnexpectedKeywords {
        transform: &'static str,
        keywords: Vec<String>,
    },
}

impl std::fmt::Display for TransformError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidCall => f.write_str("invalid transform call"),
            Self::Unknown(name) => write!(f, "unknown transform `{name}`"),
            Self::UnexpectedKeywords {
                transform,
                keywords,
            } => write!(
                f,
                "unexpected keywords for transform `{transform}`: {}",
                keywords.join(", ")
            ),
        }
    }
}

impl std::error::Error for TransformError {}

fn is_identifier_character(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

fn is_argument_boundary(c: char) -> bool {
    c.is_whitespace() || matches!(c, ',' | ')')
}

fn parse_bare_token<'i>(cursor: &mut &'i str) -> Result<&'i str> {
    take_while(1.., |c: char| !is_argument_boundary(c)).parse_next(cursor)
}

fn parse_scalar(cursor: &mut &str) -> Result<Scalar> {
    terminated(
        alt((
            parse_literal_string,
            Caseless("true").value(Scalar::from(true)),
            Caseless("false").value(Scalar::from(false)),
            parse_literal_number,
        )),
        parse_argument_boundary,
    )
    .parse_next(cursor)
}

fn parse_literal_string(cursor: &mut &str) -> Result<Scalar> {
    delimited('\'', take_while(0.., |c: char| c != '\''), '\'')
        .map(|text: &str| Scalar::from(PlSmallStr::from(text)))
        .parse_next(cursor)
}

fn parse_intercept(cursor: &mut &str) -> Result<Expression> {
    take_while(1.., is_identifier_character)
        .verify(|name: &str| name == "1")
        .value(Expression::from(Term::intercept()))
        .parse_next(cursor)
}

// Support a leading suppressed intercept, e.g., `y ~ -1 + x`
fn parse_suppressed_intercept(cursor: &mut &str) -> Result<Expression> {
    preceded(('-', multispace0), parse_intercept)
        .map(|intercept| Expression::from_iter([]).subtract(intercept))
        .parse_next(cursor)
}

fn has_float_marker(token: &str) -> bool {
    token.contains('.') || token.contains('e') || token.contains('E')
}

fn parse_literal_number(cursor: &mut &str) -> Result<Scalar> {
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

fn parse_argument_boundary(cursor: &mut &str) -> Result<()> {
    peek(alt((one_of(is_argument_boundary).void(), eof.void()))).parse_next(cursor)
}

fn parse_factor(cursor: &mut &str) -> Result<Factor> {
    alt((parse_transform, parse_column)).parse_next(cursor)
}

fn parse_column(cursor: &mut &str) -> Result<Factor> {
    alt((
        parse_quoted,
        take_while(1.., is_identifier_character).map(str::to_owned),
    ))
    .map(|name: String| Factor::from(name.as_str()))
    .parse_next(cursor)
}

fn parse_quoted(cursor: &mut &str) -> Result<String> {
    delimited(
        '`',
        repeat(
            0.., // Allow the empty string as a column name
            // Allow backtick-escaped backticks
            alt(("``".value("`"), take_while(1.., |c: char| c != '`'))),
        ),
        '`',
    )
    .parse_next(cursor)
}

fn parse_transform(cursor: &mut &str) -> Result<Factor> {
    parse_transform_call
        .try_map(Factor::try_from)
        .parse_next(cursor)
}

fn parse_transform_call(cursor: &mut &str) -> Result<TransformCall> {
    (
        take_while(1.., is_identifier_character),
        delimited(
            ('(', multispace0),
            separated(
                0..,
                parse_transform_call_arguments,
                (multispace0, ',', multispace0),
            ),
            (multispace0, ')'),
        ),
    )
        .map(
            |(name, arguments): (&str, Vec<TransformCallArgument>)| TransformCall {
                name: name.to_owned(),
                arguments,
            },
        )
        .parse_next(cursor)
}

fn parse_transform_call_arguments(cursor: &mut &str) -> Result<TransformCallArgument> {
    alt((
        parse_transform_call_keyword_arguments,
        parse_positional_argument.map(TransformCallArgument::Positional),
    ))
    .parse_next(cursor)
}

fn parse_positional_argument(cursor: &mut &str) -> Result<Factor> {
    alt((
        terminated(parse_literal_number, parse_argument_boundary).map(Factor::Literal),
        parse_factor,
    ))
    .parse_next(cursor)
}

fn parse_transform_call_keyword_arguments(cursor: &mut &str) -> Result<TransformCallArgument> {
    separated_pair(
        take_while(1.., is_identifier_character),
        (multispace0, '=', multispace0),
        parse_scalar,
    )
    .map(
        |(name, value): (&str, Scalar)| TransformCallArgument::Keyword {
            name: name.to_owned(),
            value,
        },
    )
    .parse_next(cursor)
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

fn parse_expression_inner(cursor: &mut &str) -> Result<Vec<Expression>> {
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

fn parse_stepwise_arguments(cursor: &mut &str) -> Result<Vec<Vec<Expression>>> {
    delimited(
        '(',
        separated(1.., parse_expression_inner, (multispace0, ',', multispace0)),
        (multispace0, ')'),
    )
    .parse_next(cursor)
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

fn parse_stepwise(cursor: &mut &str) -> Result<Vec<Expression>> {
    let operator = alt((
        "sw0".value(Stepwise::Sw0),
        "sw".value(Stepwise::Sw),
        "csw0".value(Stepwise::Csw0),
        "csw".value(Stepwise::Csw),
    ))
    .parse_next(cursor)?;
    let arguments = parse_stepwise_arguments.parse_next(cursor)?;

    let mut expressions = match operator {
        Stepwise::Sw | Stepwise::Sw0 => arguments.into_iter().flatten().collect::<Vec<_>>(),
        Stepwise::Csw | Stepwise::Csw0 => expand_cumulative(arguments),
    };

    if matches!(operator, Stepwise::Sw0 | Stepwise::Csw0) {
        expressions.insert(0, Expression::from_iter([]));
    }

    Ok(expressions)
}

fn parse_operand(cursor: &mut &str) -> Result<Vec<Expression>> {
    delimited(
        multispace0,
        alt((
            delimited('(', parse_expression_inner, ')'),
            parse_stepwise,
            alt((
                parse_suppressed_intercept,
                parse_intercept,
                parse_factor.map(Term::from).map(Expression::from),
            ))
            .map(|expression| vec![expression]),
        )),
        multispace0,
    )
    .parse_next(cursor)
}

fn parse_formula(cursor: &mut &str) -> Result<Vec<Formula>> {
    separated_pair(parse_expression_inner, '~', parse_expression_inner)
        .map(|(dependent, independent)| {
            let dependent = dependent
                .into_iter()
                // The left-hand side of a formula doesn't attract an intercept by default
                .map(|expression| expression.resolve_intercept(false))
                .collect::<Vec<_>>();
            let independent = independent
                .into_iter()
                // The right-hand side of a formula attracts an intercept by default
                .map(|expression| expression.resolve_intercept(true))
                .collect::<Vec<_>>();

            let mut formulas: Vec<Formula> = Vec::new();
            for dependent in &dependent {
                for independent in &independent {
                    formulas.push(Formula {
                        left: dependent.clone(),
                        right: independent.clone(),
                    });
                }
            }
            formulas
        })
        .parse_next(cursor)
}

pub(crate) type ParseResult<'i, T> =
    std::result::Result<T, winnow::error::ParseError<&'i str, winnow::error::ContextError>>;

pub fn parse(source: &str) -> ParseResult<'_, Specification> {
    parse_many(std::iter::once(source))
}

pub fn parse_many<'a>(input: impl IntoIterator<Item = &'a str>) -> ParseResult<'a, Specification> {
    let mut formulas: Vec<Formula> = Vec::new();

    for source in input {
        let expanded = terminated(parse_formula, eof).parse(source)?;
        for formula in expanded {
            push_unique(&mut formulas, formula);
        }
    }

    Ok(Specification::new(formulas))
}

#[cfg(test)]
pub(crate) fn parse_expression(source: &str) -> ParseResult<'_, Vec<Expression>> {
    terminated(parse_expression_inner, eof).parse(source)
}

#[cfg(test)]
pub(crate) fn parse_one_expression(source: &str) -> ParseResult<'_, Expression> {
    parse_expression(source).map(|expressions| {
        let [expression]: [Expression; 1] = expressions
            .try_into()
            .expect("expected exactly one expression");
        expression
    })
}

impl TryFrom<TransformCall> for Factor {
    type Error = TransformError;

    fn try_from(call: TransformCall) -> std::result::Result<Self, Self::Error> {
        let mut arguments = Vec::new();
        let mut keywords = BTreeMap::new();
        let mut saw_keyword = false;

        for argument in call.arguments {
            match argument {
                TransformCallArgument::Positional(argument) if !saw_keyword => {
                    arguments.push(Arc::new(argument));
                }
                TransformCallArgument::Positional(_) => {
                    return Err(TransformError::InvalidCall);
                }
                TransformCallArgument::Keyword { name, value } => {
                    saw_keyword = true;
                    if keywords.insert(name, value).is_some() {
                        return Err(TransformError::InvalidCall);
                    }
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
            _ => return Err(TransformError::Unknown(call.name)),
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
) -> std::result::Result<(), TransformError> {
    if keywords.is_empty() {
        Ok(())
    } else {
        Err(TransformError::UnexpectedKeywords {
            transform,
            keywords: keywords.into_keys().collect(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
                parse_positional_argument.parse(source).unwrap(),
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
            parse_positional_argument.parse("`2`").unwrap(),
            Factor::from("2"),
        );
        assert_eq!(
            parse_positional_argument.parse("2x").unwrap(),
            Factor::from("2x"),
        );
        assert_ne!(
            parse_one_expression("mul(x, 2)").unwrap(),
            parse_one_expression("mul(x, `2`)").unwrap(),
        );
        assert_eq!(
            parse_one_expression("1").unwrap(),
            Expression::from(Term::intercept()),
        );
        assert_eq!(
            parse_one_expression("-1").unwrap(),
            Expression::from_iter([]).subtract(Expression::from(Term::intercept())),
        );
        for source in ["mul(x, 2.0x)", "mul(x, -)", "mul(x, 1.2.3)"] {
            assert!(parse_one_expression(source).is_err(), "{source}");
        }
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
        let column = parse_one_expression("`1`").unwrap();
        assert_eq!(column, expression(&[&["1"]]));
        assert_ne!(column, parse_one_expression("1").unwrap());
        assert_ne!(
            parse_one_expression("log(`1`)").unwrap(),
            parse_one_expression("log(1)").unwrap(),
        );
    }

    #[test]
    fn unterminated_quoted_columns_are_rejected() {
        for source in ["`", "`income", "`a``", "```", "log(`income)", "x + `income"] {
            assert!(parse_one_expression(source).is_err(), "{source}");
        }
        assert!(parse("y ~ `income").is_err());
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
    fn parse_many_handles_empty_input_and_later_errors() {
        let specification = parse_many(std::iter::empty()).unwrap();
        assert!(specification.formulas().is_empty());

        let sources = ["y ~ x", "y ~"];
        let error = parse_many(sources).unwrap_err();
        assert_eq!(*error.input(), sources[1]);
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
                .map(|source| parse_one_expression(source).unwrap())
                .collect::<Vec<_>>();
            assert_eq!(
                super::parse_expression(source).unwrap(),
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
                .map(|source| parse_one_expression(source).unwrap())
                .collect::<Vec<_>>();
            assert_eq!(
                super::parse_expression(source).unwrap(),
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
        // Names are special only when immediately followed by their argument list.
        assert_eq!(
            super::parse("sw ~ csw0 + sw0x").unwrap(),
            Specification::new(vec![
                parse("sw ~ csw0 + sw0x").unwrap().formulas()[0].clone()
            ]),
        );
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
                parse_one_expression(source).expect(source),
                expected,
                "{source}"
            );
        }

        assert!(parse_one_expression("C (x)").is_err());
    }

    #[test]
    fn invalid_transform_calls_are_rejected() {
        for source in [
            "unknown(x)",
            "log(x, reference=1)",
            "C(reference=1, x)",
            "C(x, reference=1, reference=2)",
            "C(x, unknown=1)",
            "C(x, reference=1 2)",
            "C(x, reference=tr ue)",
            "F(x, unknown=1)",
        ] {
            assert!(parse_one_expression(source).is_err(), "{source}");
        }
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
                parse_one_expression(source).expect(source),
                expected,
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
            assert_eq!(
                parse_one_expression(source).unwrap(),
                parse_one_expression(expanded).unwrap(),
                "{source}"
            );
        }
    }

    #[test]
    fn nesting_with_empty_operands_does_not_add_terms() {
        assert_eq!(parse_one_expression("(a-a)/b").unwrap(), expression(&[]));
        assert_eq!(
            parse_one_expression("a/(b-b)").unwrap(),
            expression(&[&["a"]])
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
            assert_eq!(parse_one_expression(source).unwrap(), expected, "{source}");
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
            parse_one_expression("x + z").unwrap(),
            expression(&[&["x"], &["z"]])
        );
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
            assert!(parse_one_expression(source).is_err(), "{source:?}");
        }
    }

    #[test]
    fn leading_minus_only_accepts_intercept() {
        for source in ["y ~ -x + x*z", "y ~ -(x + 1)", "y ~ -1x"] {
            assert!(parse(source).is_err(), "{source}");
        }
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
                parse_one_expression(source).expect(source),
                expected,
                "{source}"
            );
        }
    }
}
