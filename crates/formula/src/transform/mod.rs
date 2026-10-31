mod categorical;
mod expression;
mod input;
mod integer;
mod polynomial;

pub(crate) use categorical::CategoricalDefinition;
pub(crate) use expression::ExpressionTransform;
pub use input::ArgumentCount;
pub(crate) use input::validate_numeric_input;
pub(crate) use integer::IntegerEncodingDefinition;
pub(crate) use polynomial::PolynomialDefinition;

use crate::fit::{Fit, FitError};
use crate::transform::categorical::CategoricalState;
use crate::transform::integer::IntegerEncodingState;
use crate::transform::polynomial::PolynomialState;
use polars::error::PolarsError;
use polars::prelude::{Column, DataFrame, PlSmallStr};

fn call_name<'a>(name: &str, arguments: impl IntoIterator<Item = &'a str>) -> PlSmallStr {
    let arguments = arguments.into_iter().collect::<Vec<_>>().join(", ");
    format!("{name}({arguments})").into()
}

#[derive(Debug, PartialEq)]
pub enum Transform {
    Expression(ExpressionTransform),
    C(Fit<CategoricalDefinition, CategoricalState>),
    F(Fit<IntegerEncodingDefinition, IntegerEncodingState>),
    Poly(Fit<PolynomialDefinition, PolynomialState>),
}

pub(crate) trait TransformDefinition {
    fn name(&self) -> &'static str;

    fn call_name(&self, arguments: &[&str]) -> PlSmallStr {
        call_name(self.name(), arguments.iter().copied())
    }

    fn expands(&self) -> bool;

    fn argument_count(&self) -> ArgumentCount;
}

impl Transform {
    fn definition(&self) -> &dyn TransformDefinition {
        match self {
            Self::Expression(expression) => expression,
            Self::C(categorical) => categorical.definition(),
            Self::F(integer) => integer.definition(),
            Self::Poly(polynomial) => polynomial.definition(),
        }
    }

    pub(crate) fn expands(&self) -> bool {
        self.definition().expands()
    }

    pub(crate) fn argument_count(&self) -> ArgumentCount {
        self.definition().argument_count()
    }

    pub(crate) fn call_name(&self, arguments: &[&str]) -> PlSmallStr {
        self.definition().call_name(arguments)
    }

    pub(crate) fn fit(&self, inputs: &[Column]) -> Result<Self, TransformError> {
        #[cfg(test)]
        TRANSFORM_CALLS.update(|(fits, transforms)| (fits + 1, transforms));

        match self {
            Self::Expression(expression) => Ok(Self::Expression(expression.fit())),
            Self::C(categorical) => Ok(Self::C(categorical.fit(inputs)?)),
            Self::F(integer) => Ok(Self::F(integer.fit(inputs)?)),
            Self::Poly(polynomial) => Ok(Self::Poly(polynomial.fit(inputs)?)),
        }
    }

    pub(crate) fn transform(
        &self,
        inputs: &[Column],
    ) -> Result<DataFrame, FitError<TransformError>> {
        #[cfg(test)]
        TRANSFORM_CALLS.update(|(fits, transforms)| (fits, transforms + 1));

        match self {
            Self::Expression(expression) => {
                expression.transform(inputs).map_err(FitError::Evaluation)
            }
            Self::C(categorical) => categorical.transform(inputs),
            Self::F(integer) => integer.transform(inputs),
            Self::Poly(polynomial) => polynomial.transform(inputs),
        }
    }
}

#[cfg(test)]
std::thread_local! {
    /// Counts (fit calls, transform calls) on the current test thread.
    pub(crate) static TRANSFORM_CALLS: std::cell::Cell<(usize, usize)> =
        const { std::cell::Cell::new((0, 0)) };
}

#[derive(Debug)]
pub enum TransformError {
    InputCountMismatch {
        expected: ArgumentCount,
        actual: usize,
    },
    InputNamesMismatch {
        expected: Vec<PlSmallStr>,
        actual: Vec<PlSmallStr>,
    },
    InputTypeMismatch {
        name: PlSmallStr,
        expected: String,
        actual: String,
    },
    NonNumericInput {
        name: PlSmallStr,
        actual: String,
    },
    NumericalError(String),
    InvalidParameter(String),
    Polars(PolarsError),
}

impl From<PolarsError> for TransformError {
    fn from(error: PolarsError) -> Self {
        Self::Polars(error)
    }
}

impl std::error::Error for TransformError {}

impl std::fmt::Display for TransformError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InputCountMismatch { expected, actual } => {
                write!(
                    f,
                    "number of input columns: expected {expected}, received {actual}"
                )
            }
            Self::InputNamesMismatch { expected, actual } => {
                write!(f, "expected input names {expected:?}, received {actual:?}")
            }
            Self::InputTypeMismatch {
                name,
                expected,
                actual,
            } => {
                write!(f, "input `{name}`: expected {expected}, received {actual}")
            }
            Self::NonNumericInput { name, actual } => write!(
                f,
                "column `{name}`: expected a primitive numeric dtype, received {actual}",
            ),
            Self::NumericalError(message) => f.write_str(message),
            Self::InvalidParameter(message) => f.write_str(message),
            Self::Polars(error) => error.fmt(f),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fit::Fitted;
    use polars::prelude::{PlSmallStr, Scalar};

    #[test]
    fn fit_identity_depends_on_definition_not_fitted_state() -> Result<(), TransformError> {
        let definition =
            CategoricalDefinition::new(Some(Scalar::from(PlSmallStr::from_static("a"))));
        let first_input = [Column::new("group".into(), ["a", "b"])];
        let second_input = [Column::new("group".into(), ["a", "c"])];
        let first = Fit::Fitted(Fitted::fit(definition.clone(), &first_input)?);
        let second = Fit::Fitted(Fitted::fit(definition.clone(), &second_input)?);

        assert_eq!(first, second);
        assert_eq!(first, Fit::Unfitted(definition));
        assert_ne!(first, Fit::Unfitted(CategoricalDefinition::new(None)));

        let definition = IntegerEncodingDefinition;
        let first = Fit::Fitted(Fitted::fit(definition.clone(), &first_input)?);
        let second = Fit::Fitted(Fitted::fit(definition.clone(), &second_input)?);
        assert_eq!(first, second);
        assert_eq!(first, Fit::Unfitted(definition));
        Ok(())
    }

    #[test]
    fn transform_enum_delegates_integer_operations() -> Result<(), TransformError> {
        let inputs = [Column::new("group".into(), ["a", "b", "a"])];
        let transform = Transform::F(Fit::Unfitted(IntegerEncodingDefinition));

        assert!(!transform.expands());
        assert_eq!(transform.argument_count(), ArgumentCount::AtLeast(1));
        assert!(matches!(
            transform.transform(&inputs),
            Err(FitError::NotFitted)
        ));

        let transform = transform.fit(&inputs)?;
        let output = transform.transform(&inputs).unwrap();
        assert_eq!(output.get_column_names(), ["group"]);
        assert_eq!(
            output[0].u32()?.iter().collect::<Vec<_>>(),
            vec![Some(0), Some(1), Some(0)]
        );
        Ok(())
    }
}
