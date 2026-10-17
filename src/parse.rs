use crate::formula::{Expression, Factor, Formula, Term};
use crate::transform::{
    CategoricalTransform, ExpressionTransform, IntegerEncodingTransform, Transform,
};
use polars::prelude::{PlSmallStr, Scalar};
use std::collections::BTreeMap;
use winnow::Parser;
use winnow::Result;
use winnow::ascii::{Caseless, multispace0};
use winnow::combinator::Infix::{Left, Right};
use winnow::combinator::{
    alt, delimited, eof, expression, peek, separated, separated_pair, terminated,
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
    alt((
        parse_transform,
        take_while(1.., is_identifier_character).map(Factor::from),
    ))
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
        parse_factor.map(TransformCallArgument::Positional),
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

fn parse_expression_inner(cursor: &mut &str) -> Result<Expression> {
    expression(parse_operand)
        .infix(alt((
            '+'.value(Left(10, |_, left: Expression, right: Expression| {
                Ok(left.add(right))
            })),
            '-'.value(Left(10, |_, left: Expression, right: Expression| {
                Ok(left.subtract(right))
            })),
            '*'.value(Right(20, |_, left: Expression, right: Expression| {
                Ok(left.interact_fully(right))
            })),
            ':'.value(Right(30, |_, left: Expression, right: Expression| {
                Ok(left.interact(&right))
            })),
        )))
        .parse_next(cursor)
}

fn parse_operand(cursor: &mut &str) -> Result<Expression> {
    delimited(
        multispace0,
        alt((
            delimited('(', parse_expression_inner, ')'),
            parse_factor.map(Term::from).map(Expression::from),
        )),
        multispace0,
    )
    .parse_next(cursor)
}

fn parse_formula(cursor: &mut &str) -> Result<Formula> {
    separated_pair(parse_expression_inner, '~', parse_expression_inner)
        .map(|(dependent, independent)| Formula {
            dependent,
            independent,
        })
        .parse_next(cursor)
}

type ParseResult<'i, T> =
    std::result::Result<T, winnow::error::ParseError<&'i str, winnow::error::ContextError>>;

pub fn parse(source: &str) -> ParseResult<'_, Formula> {
    terminated(parse_formula, eof).parse(source)
}

pub fn parse_expression(source: &str) -> ParseResult<'_, Expression> {
    terminated(parse_expression_inner, eof).parse(source)
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
                    arguments.push(argument);
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
            "C" => {
                let reference = keywords.remove("reference");
                reject_keywords("C", keywords)?;
                Transform::C(CategoricalTransform::new(reference))
            }
            "F" => {
                reject_keywords("F", keywords)?;
                Transform::F(IntegerEncodingTransform::default())
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
        let categorical = |reference| Transform::C(CategoricalTransform::new(reference));
        let cases = [
            (
                "log(x)",
                Factor::Call {
                    transform: Transform::Expression(ExpressionTransform::LOG),
                    arguments: vec![Factor::from("x")],
                },
            ),
            (
                "exp(x)",
                Factor::Call {
                    transform: Transform::Expression(ExpressionTransform::EXP),
                    arguments: vec![Factor::from("x")],
                },
            ),
            (
                "C(group)",
                Factor::Call {
                    transform: categorical(None),
                    arguments: vec![Factor::from("group")],
                },
            ),
            (
                "C(group, reference='a')",
                Factor::Call {
                    transform: categorical(Some(Scalar::from(PlSmallStr::from_static("a")))),
                    arguments: vec![Factor::from("group")],
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
            arguments: vec![Factor::from("x")],
        }));
        for source in [
            "C(x, reference=1)",
            "C(x, reference= 1)",
            "C( x , reference = 1 )",
            " \n C(\tx,\nreference\t=\t1\n) \t",
        ] {
            assert_eq!(
                parse_expression(source).expect(source),
                expected,
                "{source}"
            );
        }

        assert!(parse_expression("C (x)").is_err());
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
            assert!(parse_expression(source).is_err(), "{source}");
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
                        arguments: vec![Factor::from("x")],
                    },
                    Factor::from("z"),
                ],
            ),
        ];

        for (source, arguments) in cases {
            let expected = Factor::Call {
                transform: Transform::F(IntegerEncodingTransform::default()),
                arguments,
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
        ];

        for (source, expected) in cases {
            assert_eq!(
                parse_expression(source).expect(source),
                expected,
                "{source}"
            );
        }
    }

    #[test]
    fn formulas_keep_dependent_and_independent_expressions_separate() {
        let expected = Formula {
            dependent: expression(&[&["y"]]),
            independent: expression(&[&["x"], &["z"]]),
        };

        for source in ["y~x+z", "y ~ x + z", " \n y \t~\n x + z \t"] {
            assert_eq!(parse(source).expect(source), expected, "{source}");
        }

        assert_eq!(parse_expression("x + z").unwrap(), expected.independent);
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
        ] {
            assert!(parse(source).is_err(), "{source:?}");
        }
    }

    #[test]
    fn expression_parser_rejects_incomplete_or_trailing_input() {
        for source in ["", " ", "y ~ x", "x C(group)", "(x + z", "x)", "x +", "x ?"] {
            assert!(parse_expression(source).is_err(), "{source:?}");
        }
    }

    #[test]
    fn transform_composition_preserves_inner_to_outer_order() {
        let cases = [
            (
                "log(C(x))",
                Factor::Call {
                    transform: Transform::Expression(ExpressionTransform::LOG),
                    arguments: vec![Factor::Call {
                        transform: Transform::C(CategoricalTransform::new(None)),
                        arguments: vec![Factor::from("x")],
                    }],
                },
            ),
            (
                "C(log(x))",
                Factor::Call {
                    transform: Transform::C(CategoricalTransform::new(None)),
                    arguments: vec![Factor::Call {
                        transform: Transform::Expression(ExpressionTransform::LOG),
                        arguments: vec![Factor::from("x")],
                    }],
                },
            ),
            (
                "log(exp(x))",
                Factor::Call {
                    transform: Transform::Expression(ExpressionTransform::LOG),
                    arguments: vec![Factor::Call {
                        transform: Transform::Expression(ExpressionTransform::EXP),
                        arguments: vec![Factor::from("x")],
                    }],
                },
            ),
        ];

        for (source, factor) in cases {
            let expected = Expression::from(Term::from(factor));
            assert_eq!(
                parse_expression(source).expect(source),
                expected,
                "{source}"
            );
        }
    }
}
