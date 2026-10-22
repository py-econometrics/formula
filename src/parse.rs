use crate::formula::{Expression, Factor, Formula, Specification, Term, push_unique};
use crate::transform::{
    CategoricalDefinition, ExpressionTransform, Fit, IntegerEncodingDefinition, Transform,
};
use polars::prelude::{PlSmallStr, Scalar};
use std::collections::BTreeMap;
use std::sync::Arc;
use winnow::Parser;
use winnow::Result;
use winnow::ascii::{Caseless, multispace0};
use winnow::combinator::Infix::{Left, Right};
use winnow::combinator::{
    alt, delimited, eof, expression, peek, preceded, separated, separated_pair, terminated,
};
use winnow::token::{one_of, take_while};

pub trait FormulaInput {
    fn sources(&self) -> impl Iterator<Item = &str>;
}

impl FormulaInput for str {
    fn sources(&self) -> impl Iterator<Item = &str> {
        std::iter::once(self)
    }
}

impl FormulaInput for String {
    fn sources(&self) -> impl Iterator<Item = &str> {
        std::iter::once(self.as_str())
    }
}

impl<S: AsRef<str>> FormulaInput for Vec<S> {
    fn sources(&self) -> impl Iterator<Item = &str> {
        self.iter().map(|source| source.as_ref())
    }
}

impl<S: AsRef<str>, const N: usize> FormulaInput for [S; N] {
    fn sources(&self) -> impl Iterator<Item = &str> {
        self.iter().map(|source| source.as_ref())
    }
}

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
            '*'.value(Right(
                20,
                |_, left: Vec<Expression>, right: Vec<Expression>| {
                    Ok(combine_expressions(&left, &right, |left, right| {
                        left.interact_fully(right)
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

fn parse_sw(cursor: &mut &str) -> Result<Vec<Expression>> {
    preceded("sw", parse_stepwise_arguments)
        .map(|arguments: Vec<Vec<Expression>>| arguments.into_iter().flatten().collect())
        .parse_next(cursor)
}

fn parse_sw0(cursor: &mut &str) -> Result<Vec<Expression>> {
    preceded("sw0", parse_stepwise_arguments)
        .map(|arguments: Vec<Vec<Expression>>| {
            // Add zero (empty) step
            let mut expressions = vec![Expression::from_iter([])];
            expressions.extend(arguments.into_iter().flatten());
            expressions
        })
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

fn parse_csw(cursor: &mut &str) -> Result<Vec<Expression>> {
    preceded("csw", parse_stepwise_arguments)
        .map(expand_cumulative)
        .parse_next(cursor)
}

fn parse_csw0(cursor: &mut &str) -> Result<Vec<Expression>> {
    preceded("csw0", parse_stepwise_arguments)
        .map(|arguments| {
            // Add zero (empty) step
            let mut expressions = vec![Expression::from_iter([])];
            expressions.extend(expand_cumulative(arguments));
            expressions
        })
        .parse_next(cursor)
}

fn parse_stepwise(cursor: &mut &str) -> Result<Vec<Expression>> {
    alt((parse_sw0, parse_sw, parse_csw0, parse_csw)).parse_next(cursor)
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
                    // Avoid retaining duplicate formulas
                    push_unique(
                        &mut formulas,
                        Formula {
                            dependent: dependent.clone(),
                            independent: independent.clone(),
                        },
                    );
                }
            }
            formulas
        })
        .parse_next(cursor)
}

pub(crate) type ParseResult<'i, T> =
    std::result::Result<T, winnow::error::ParseError<&'i str, winnow::error::ContextError>>;

pub fn parse<T: FormulaInput + ?Sized>(input: &T) -> ParseResult<'_, Specification> {
    let mut formulas: Vec<Formula> = Vec::new();

    for source in input.sources() {
        let expanded = terminated(parse_formula, eof).parse(source)?;
        for formula in expanded {
            push_unique(&mut formulas, formula);
        }
    }

    Ok(Specification::new(formulas))
}

pub(crate) fn parse_one(source: &str) -> ParseResult<'_, Formula> {
    terminated(
        parse_formula.verify_map(|formulas: Vec<Formula>| {
            let [formula]: [Formula; 1] = formulas.try_into().ok()?;
            Some(formula)
        }),
        eof,
    )
    .parse(source)
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
    fn borrowed_input_forms_produce_the_same_specification() {
        let source = String::from("y ~ sw(x, z)");
        let expected = super::parse(source.as_str()).unwrap();
        assert_eq!(super::parse(&source).unwrap(), expected);
        assert_eq!(super::parse(&vec![source.as_str()]).unwrap(), expected);
        assert_eq!(super::parse(&vec![source.clone()]).unwrap(), expected);
        assert_eq!(super::parse(&[source.as_str()]).unwrap(), expected);
        let owned_array = [source.clone()];
        assert_eq!(super::parse(&owned_array).unwrap(), expected);

        let sources = vec![source, String::from("y ~ x"), String::from("y ~ w")];
        let expected = ["y ~ x", "y ~ z", "y ~ w"]
            .map(|source| parse_one(source).unwrap())
            .to_vec();
        assert_eq!(
            super::parse(&sources).unwrap(),
            Specification::new(expected)
        );
    }

    #[test]
    fn borrowed_inputs_handle_empty_collections_and_later_errors() {
        let empty: Vec<String> = Vec::new();
        assert_eq!(
            super::parse(&empty).unwrap(),
            Specification::new(Vec::new())
        );

        let sources = vec![String::from("y ~ x"), String::from("y ~")];
        let error = super::parse(&sources).unwrap_err();
        assert_eq!(*error.input(), sources[1]);
        assert!(parse_one("y ~ sw(x, z)").is_err());
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
            ("y ~ sw(1, x - 1)", &["y~1", "y~x-1"]),
        ];

        for &(source, expected) in cases {
            let expected = expected
                .iter()
                .map(|source| parse_one(source).unwrap())
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
                .map(|source| parse_one(source).unwrap())
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
            Specification::new(vec![parse_one("sw ~ csw0 + sw0x").unwrap()]),
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
    fn formulas_keep_dependent_and_independent_expressions_separate() {
        let expected = Formula {
            dependent: expression(&[&["y"]]),
            independent: expression(&[&[], &["x"], &["z"]]),
        };

        for source in ["y~x+z", "y ~ x + z", " \n y \t~\n x + z \t"] {
            assert_eq!(parse_one(source).expect(source), expected, "{source}");
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
        ] {
            assert!(parse_one(source).is_err(), "{source:?}");
        }
    }

    #[test]
    fn expression_parser_rejects_incomplete_or_trailing_input() {
        for source in ["", " ", "y ~ x", "x C(group)", "(x + z", "x)", "x +", "x ?"] {
            assert!(parse_one_expression(source).is_err(), "{source:?}");
        }
    }

    #[test]
    fn leading_minus_only_accepts_intercept() {
        for source in ["y ~ -x + x*z", "y ~ -(x + 1)", "y ~ -1x"] {
            assert!(parse_one(source).is_err(), "{source}");
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
    #[test]
    fn addition_combines_expression_collections() {
        // Ordinary syntax: one expression on each side produces one expression.
        let left = [parse_one_expression("x").unwrap()];
        let right = [parse_one_expression("y").unwrap()];

        let actual = combine_expressions(&left, &right, |left, right| left.add(right));

        assert_eq!(actual, vec![parse_one_expression("x + y").unwrap()]);

        // Multiple expressions: left order first, then right order.
        let left = [
            parse_one_expression("x").unwrap(),
            parse_one_expression("z").unwrap(),
        ];
        let right = [
            parse_one_expression("w").unwrap(),
            parse_one_expression("v").unwrap(),
        ];

        let actual = combine_expressions(&left, &right, |left, right| left.add(right));

        let expected = ["x + w", "x + v", "z + w", "z + v"]
            .map(|source| parse_one_expression(source).unwrap())
            .to_vec();

        assert_eq!(actual, expected);
    }

    #[test]
    fn interaction_distributes_within_each_expression() {
        let left = [
            parse_one_expression("x + y").unwrap(),
            parse_one_expression("z").unwrap(),
        ];
        let right = [parse_one_expression("w").unwrap()];

        let actual = combine_expressions(&left, &right, |left, right| left.interact(&right));

        let expected = ["x:w + y:w", "z:w"]
            .map(|source| parse_one_expression(source).unwrap())
            .to_vec();

        assert_eq!(actual, expected);
    }
}
