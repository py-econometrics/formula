mod ast;

use crate::fit::Fit;
#[cfg(test)]
use crate::formula::Expression;
use crate::formula::{Factor, Formula, Formulas, Term, push_unique};
use crate::parse::ast::Stepwise;
use crate::transform::{
    CategoricalDefinition, ExpressionTransform, PolynomialDefinition, Transform,
};
use polars::prelude::{PlSmallStr, Scalar};
use std::collections::BTreeMap;
use std::sync::Arc;
use winnow::ascii::{Caseless, multispace0};
use winnow::combinator::Infix::{Left, Right};
use winnow::combinator::{
    Prefix, alt, cut_err, delimited, eof, expression, fail, opt, peek, preceded, repeat, separated,
    terminated,
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
const INTERACTION_PRECEDENCE: i64 = 30;
const PREFIX_PRECEDENCE: i64 = 40;

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

pub(crate) fn is_identifier_character(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

fn is_argument_boundary(c: char) -> bool {
    c.is_whitespace() || matches!(c, ',' | ')')
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

fn parse_intercept<K: ast::Construct>(cursor: &mut &str) -> ModalResult<ast::Expression<K>> {
    let name = take_while(1.., is_identifier_character)
        .verify(|name: &str| matches!(name, "0" | "1"))
        .parse_next(cursor)?;

    let node = construction_result(cursor, ast::Expression::<K>::term(Term::intercept()))?;

    if name == "0" {
        construction_result(cursor, node.negate())
    } else {
        Ok(node)
    }
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

fn combine_alternatives<L: Clone, R: Clone, O>(
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

fn construction_result<T>(
    cursor: &mut &str,
    result: Result<T, ast::ConstructionError>,
) -> ModalResult<T> {
    result.map_err(|error| ErrMode::Cut(ContextError::from_external_error(&*cursor, error)))
}

#[cfg(test)]
pub(crate) fn parse_expression(cursor: &mut &str) -> ModalResult<Vec<Expression>> {
    parse_node::<ast::Multistage>
        .map(|node| node.expand())
        .parse_next(cursor)
}

fn parse_node<K: ast::Construct>(cursor: &mut &str) -> ModalResult<ast::Expression<K>> {
    expression(parse_operand::<K>)
        .prefix(
            preceded(multispace0, '-')
                .value(Prefix(
                    PREFIX_PRECEDENCE,
                    |cursor, node: ast::Expression<K>| construction_result(cursor, node.negate()),
                ))
                .context(StrContext::Expected(StrContextValue::Description(
                    "an expression",
                ))),
        )
        .infix(alt((
            '+'.value(Left(SUM_PRECEDENCE, |cursor, left, right| {
                construction_result(cursor, ast::Expression::<K>::add(left, right))
            })),
            '-'.value(Left(SUM_PRECEDENCE, |cursor, left, right| {
                construction_result(cursor, ast::Expression::<K>::subtract(left, right))
            })),
            '*'.value(Left(PRODUCT_PRECEDENCE, |cursor, left, right| {
                construction_result(cursor, ast::Expression::<K>::interact_fully(left, right))
            })),
            '/'.value(Left(PRODUCT_PRECEDENCE, |cursor, left, right| {
                construction_result(cursor, ast::Expression::<K>::nest(left, right))
            })),
            ':'.value(Right(INTERACTION_PRECEDENCE, |cursor, left, right| {
                construction_result(cursor, ast::Expression::<K>::interact(left, right))
            })),
        )))
        .parse_next(cursor)
}

fn parse_stepwise_arguments<K: ast::Construct>(
    cursor: &mut &str,
) -> ModalResult<(ast::Expression<K>, Vec<ast::Expression<K>>)> {
    multispace0.parse_next(cursor)?;

    let first = cut_err(parse_node::<K>).parse_next(cursor)?;

    let rest = repeat(
        0..,
        preceded((multispace0, ',', multispace0), cut_err(parse_node::<K>)),
    )
    .parse_next(cursor)?;

    multispace0.parse_next(cursor)?;

    ')'.context(StrContext::Expected(StrContextValue::Description(
        "',' or ')' after an expression",
    )))
    .parse_next(cursor)?;

    Ok((first, rest))
}

fn parse_stepwise<K: ast::Construct>(cursor: &mut &str) -> ModalResult<ast::Expression<K>> {
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

    let (first, rest) = cut_err(parse_stepwise_arguments::<K>).parse_next(cursor)?;

    Ok(ast::Expression::Stepwise(ast::StepwiseCall {
        operator,
        first: Box::new(first),
        rest,
    }))
}

fn parse_node_parenthesized<K: ast::Construct>(
    cursor: &mut &str,
) -> ModalResult<ast::Expression<K>> {
    '('.parse_next(cursor)?;

    let node = cut_err(parse_node::<K>).parse_next(cursor)?;

    cut_err(
        ')'.context(StrContext::Expected(StrContextValue::Description(
            "')' to close the parenthesized expression",
        ))),
    )
    .parse_next(cursor)?;

    Ok(ast::Expression::Group(Box::new(node)))
}

fn parse_operand<K: ast::Construct>(cursor: &mut &str) -> ModalResult<ast::Expression<K>> {
    let factor = |cursor: &mut &str| {
        let factor = parse_factor.parse_next(cursor)?;
        construction_result(cursor, ast::Expression::<K>::term(Term::from(factor)))
    };

    delimited(
        multispace0,
        alt((
            parse_node_parenthesized::<K>,
            parse_stepwise::<K>,
            parse_multistage::<K>,
            parse_intercept::<K>,
            factor,
        )),
        multispace0,
    )
    .parse_next(cursor)
}

fn parse_multistage<K: ast::Construct>(cursor: &mut &str) -> ModalResult<ast::Expression<K>> {
    let formula = preceded(
        '[',
        cut_err(terminated(
            parse_formula_ast,
            ']'.context(StrContext::Expected(StrContextValue::Description(
                "']' to close the nested formula",
            ))),
        )),
    )
    .parse_next(cursor)?;

    construction_result(cursor, ast::Expression::<K>::multistage(formula))
}

fn parse_formula_ast(cursor: &mut &str) -> ModalResult<ast::Formula> {
    let left = preceded(
        multispace0,
        parse_node::<ast::Multistage>.context(StrContext::Label("left-hand expression")),
    )
    .parse_next(cursor)?;

    '~'.context(StrContext::Expected(StrContextValue::CharLiteral(
        FORMULA_SEPARATOR,
    )))
    .parse_next(cursor)?;

    let right = preceded(
        multispace0,
        cut_err(parse_node::<ast::Multistage>.context(StrContext::Label("right-hand expression"))),
    )
    .parse_next(cursor)?;

    let fixed_effects = opt(preceded(
        PIPE_SEPARATOR,
        cut_err(
            parse_node::<ast::FixedEffect>.context(StrContext::Label("fixed-effect expression")),
        ),
    ))
    .parse_next(cursor)?;

    Ok(ast::Formula {
        left,
        right,
        fixed_effects,
    })
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
        .map(|ast| ast.expand())
        .parse(source)?;

        for formula in expanded {
            push_unique(&mut formulas, formula);
        }
    }

    Ok(Formulas::new(formulas))
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
    use crate::formula::Role;
    use crate::parse::ast as typed;

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

    fn parse_fixed_effects(cursor: &mut &str) -> ModalResult<Vec<Expression>> {
        parse_node::<typed::FixedEffect>
            .map(|node| node.expand())
            .parse_next(cursor)
    }

    fn construction_error(error: &ContextError) -> &typed::ConstructionError {
        error
            .cause()
            .and_then(|cause| cause.downcast_ref::<typed::ConstructionError>())
            .expect("expected an AST construction error")
    }

    mod syntax {
        use super::*;
        use typed::{Expression as Ast, Multistage, MultistageOperand, Simple};

        fn assert_term(node: &Ast<Multistage>, name: &str) {
            let Ast::Operand(MultistageOperand::Term(term)) = node else {
                panic!("expected multistage-kind term {name}");
            };
            assert_eq!(term, &Term::from(Factor::from(name)));
        }

        fn assert_simple_term(node: &Ast<Simple>, name: &str) {
            let Ast::Operand(term) = node else {
                panic!("expected simple term {name}");
            };
            assert_eq!(term, &Term::from(Factor::from(name)));
        }

        #[test]
        fn parsing_preserves_formula_sections() {
            let ast = parse_formula_ast.parse("y ~ x | firm").unwrap();
            assert_term(&ast.left, "y");
            assert_term(&ast.right, "x");
            assert!(matches!(
                ast.fixed_effects,
                Some(Ast::Operand(name)) if name == "firm"
            ));
        }

        #[test]
        fn parsing_preserves_subtraction() {
            let ast = parse_node::<Multistage>.parse("x - z").unwrap();
            let Ast::Subtract(node) = ast else {
                panic!("expected subtraction");
            };
            assert_term(&node.left, "x");
            assert_simple_term(&node.right, "z");
        }

        #[test]
        fn parsing_does_not_cancel_double_negation() {
            let ast = parse_node::<Multistage>.parse("--x").unwrap();
            let Ast::Negate(outer) = ast else {
                panic!("expected outer negation");
            };
            let Ast::Negate(inner) = outer.as_ref() else {
                panic!("expected inner negation");
            };
            assert_simple_term(inner, "x");
        }

        #[test]
        fn parsing_preserves_grouped_negation_in_subtraction() {
            let ast = parse_node::<Multistage>.parse("x - (-z)").unwrap();
            let Ast::Subtract(node) = ast else {
                panic!("expected subtraction");
            };
            assert_term(&node.left, "x");
            let Ast::Group(group) = node.right.as_ref() else {
                panic!("expected group");
            };
            let Ast::Negate(value) = group.as_ref() else {
                panic!("expected grouped negation");
            };
            assert_simple_term(value, "z");
        }

        #[test]
        fn stepwise_expansion_preserves_formula_and_argument_scope() {
            let ast = parse_formula_ast.parse("y ~ sw((x - 1), z)").unwrap();
            assert_term(&ast.left, "y");
            let Ast::Stepwise(call) = &ast.right else {
                panic!("expected stepwise call");
            };
            assert!(matches!(call.operator, Stepwise::Sw));
            assert!(matches!(call.first.as_ref(), Ast::Group(_)));
            assert_eq!(call.rest.len(), 1);
            assert_term(&call.rest[0], "z");

            let expected = parse_many(["y ~ x", "y ~ z"]).unwrap();
            assert_eq!(ast.expand().into_iter().collect::<Formulas>(), expected);
        }

        #[test]
        fn zero_step_expansion_produces_an_empty_alternative() {
            for source in ["sw0(x)", "csw0(x)"] {
                let ast = parse_node::<Multistage>.parse(source).unwrap();
                assert_eq!(ast.expand(), [expression(&[]), expression(&[&["x"]])]);
            }
        }

        #[test]
        fn cumulative_expansion_preserves_prefix_order_and_local_scope() {
            let ast = parse_node::<Multistage>
                .parse("csw(sw(a, b), sw(c, d))")
                .unwrap();
            let expected: &[&[&str]] = &[
                &["a"],
                &["b"],
                &["a", "c"],
                &["a", "d"],
                &["b", "c"],
                &["b", "d"],
            ];
            let expected = expected
                .iter()
                .map(|names| {
                    names
                        .iter()
                        .map(|name| Term::from(Factor::from(*name)))
                        .collect::<Expression>()
                })
                .collect::<Vec<_>>();
            assert_eq!(ast.expand(), expected);

            // The first argument removes its own intercept, not the RHS seed.
            assert_eq!(
                parse("y ~ csw(sw(a - 1, b - 1), sw(c, d))").unwrap(),
                parse_many([
                    "y ~ a",
                    "y ~ b",
                    "y ~ a + c",
                    "y ~ a + d",
                    "y ~ b + c",
                    "y ~ b + d",
                ])
                .unwrap(),
            );
        }
    }

    #[test]
    fn fixed_effect_expressions_preserve_grouping_and_stepwise_expansion() {
        for (source, expected) in [
            ("f1 + f1:f2", vec![expression(&[&["f1"], &["f1", "f2"]])]),
            (
                "(f1 + f2):f3",
                vec![expression(&[&["f1", "f3"], &["f2", "f3"]])],
            ),
            (
                "sw0(f1):f2",
                vec![expression(&[]), expression(&[&["f1", "f2"]])],
            ),
            ("`log(x)`", vec![expression(&[&["log(x)"]])]),
            ("`0`", vec![expression(&[&["0"]])]),
            ("`1`", vec![expression(&[&["1"]])]),
            ("0x + 1x", vec![expression(&[&["0x"], &["1x"]])]),
        ] {
            assert_eq!(
                parse_fixed_effects.parse(source).unwrap(),
                expected,
                "{source}"
            );
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
            let error = parse_fixed_effects.parse(source).unwrap_err();
            assert!(
                construction_error(error.inner())
                    == &typed::ConstructionError::ExpectedFixedEffectColumn,
                "{source}: {error}"
            );

            let formula = format!("y ~ x | {source}");
            let error = parse(&formula).unwrap_err();
            assert!(
                construction_error(error.inner())
                    == &typed::ConstructionError::ExpectedFixedEffectColumn,
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
        for (source, operator) in [
            ("firm - year", "-"),
            ("-firm", "-"),
            ("firm * year", "*"),
            ("firm / year", "/"),
            ("(firm - year)", "-"),
            ("sw(firm, year * worker)", "*"),
        ] {
            let error = parse_fixed_effects.parse(source).unwrap_err();
            assert_eq!(
                construction_error(error.inner()),
                &typed::ConstructionError::FixedEffectOperator(operator),
                "{source}: {error}",
            );

            let formula = format!("y ~ x | {source}");
            let error = parse(&formula).unwrap_err();
            assert_eq!(
                construction_error(error.inner()),
                &typed::ConstructionError::FixedEffectOperator(operator),
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
            assert!(parse_fixed_effects.parse(source).is_err(), "{source}");
            let formula = format!("y ~ x | {source}");
            assert!(parse(&formula).is_err(), "{formula}");
        }
    }

    #[test]
    fn fixed_effect_pipe_expands_stepwise_alternatives_across_roles() {
        let cases: &[(&str, &[&str])] = &[
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
            let expected = expected
                .iter()
                .map(|source| parse(source).unwrap().iter().next().unwrap().clone())
                .collect::<Vec<_>>();
            assert_eq!(
                super::parse(source).unwrap(),
                Formulas::new(expected),
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
            let expected = expected
                .iter()
                .map(|source| parse(source).unwrap().iter().next().unwrap().clone())
                .collect::<Vec<_>>();
            assert_eq!(
                super::parse(source).unwrap(),
                Formulas::new(expected),
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

    mod expansion {
        use super::*;

        #[test]
        fn local_expansion_and_application_preserve_sign_and_group_scope() {
            for (source, local, applied) in [
                ("--x", expression(&[&["x"]]), expression(&[&[], &["x"]])),
                ("-(-x)", expression(&[]), expression(&[&[]])),
                (
                    "x - -z",
                    expression(&[&["x"], &["z"]]),
                    expression(&[&[], &["x"], &["z"]]),
                ),
                (
                    "x - (-z)",
                    expression(&[&["x"]]),
                    expression(&[&[], &["x"]]),
                ),
                (
                    "x - 1 - -1",
                    expression(&[&["x"], &[]]),
                    expression(&[&["x"], &[]]),
                ),
                ("x - 1 - (-1)", expression(&[&["x"]]), expression(&[&["x"]])),
            ] {
                let node = parse_node::<typed::Multistage>.parse(source).unwrap();
                assert_eq!(node.expand(), [local], "{source}");
                assert_eq!(
                    node.apply_to(Expression::from(Term::intercept())),
                    [applied],
                    "{source}",
                );
            }
        }

        #[test]
        fn stepwise_expands_directly_without_an_intermediate_ast() {
            let stepwise = parse_node::<typed::Multistage>.parse("sw(x, z)").unwrap();
            assert_eq!(
                stepwise.expand(),
                [expression(&[&["x"]]), expression(&[&["z"]])]
            );
        }

        #[test]
        fn expansion_preserves_signs_and_grouping() {
            type Ast = typed::Expression<typed::Multistage>;
            let x = || Ast::term(Term::from(Factor::from("x"))).unwrap();
            let intercept = || Ast::term(Term::intercept()).unwrap();
            let removed = || Ast::subtract(x(), intercept()).unwrap();
            let seed = Expression::from(Term::intercept());

            assert_eq!(removed().apply_to(seed.clone()), [expression(&[&["x"]])]);

            let grouped = Ast::Group(Box::new(removed()));
            assert_eq!(grouped.apply_to(seed.clone()), [expression(&[&[], &["x"]])]);

            let restored = Ast::add(removed(), intercept()).unwrap();
            assert_eq!(restored.apply_to(seed), [expression(&[&["x"], &[]])]);

            assert_eq!(
                intercept().negate().unwrap().negate().unwrap().expand(),
                [expression(&[&[]])],
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
                "y ~ x - [a ~ z]",
                "y ~ -[a ~ z]",
                "y ~ --[a ~ z]",
                "y ~ x - -[a ~ z]",
                "y ~ x * ([a ~ z] + w)",
                "y ~ ([a ~ z] + w) : x",
                "y ~ x - (w + [a ~ z])",
                "y ~ -([a ~ z] + w)",
                "y ~ x * sw(w, [a ~ z])",
                "y ~ x / sw0([a ~ z])",
                "y ~ x - csw(w, [a ~ z])",
                "y ~ -sw([a ~ z], w)",
                "y ~ x + [a ~ z * [b ~ w]]",
            ] {
                // Test the syntax parser directly: evaluation is not involved.
                let error = parse_formula_ast
                    .parse(source)
                    .err()
                    .expect("expected construction failure");
                assert!(
                    construction_error(error.inner())
                        == &typed::ConstructionError::MultistageArithmetic,
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
                "F(x)",
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
