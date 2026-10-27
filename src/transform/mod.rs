mod categorical;
mod expression;
mod input;
mod integer;

pub(crate) use categorical::CategoricalDefinition;
pub(crate) use expression::ExpressionTransform;
pub use input::ArgumentCount;
pub(crate) use input::validate_numeric_input;
pub(crate) use integer::IntegerEncodingDefinition;

use crate::fit::{Fit, FitError};
use crate::transform::categorical::CategoricalState;
use crate::transform::integer::IntegerEncodingState;
use polars::error::PolarsError;
use polars::prelude::{Column, DataFrame, DataType, PlSmallStr};

fn call_name<'a>(name: &str, arguments: impl IntoIterator<Item = &'a str>) -> PlSmallStr {
    let arguments = arguments.into_iter().collect::<Vec<_>>().join(", ");
    format!("{name}({arguments})").into()
}

#[derive(Debug, PartialEq)]
pub enum Transform {
    Expression(ExpressionTransform),
    C(Fit<CategoricalDefinition, CategoricalState>),
    F(Fit<IntegerEncodingDefinition, IntegerEncodingState>),
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
    pub(crate) fn expands(&self) -> bool {
        match self {
            Self::Expression(expression) => expression.expands(),
            Self::C(categorical) => categorical.definition().expands(),
            Self::F(integer) => integer.definition().expands(),
        }
    }

    pub(crate) fn argument_count(&self) -> ArgumentCount {
        match self {
            Self::Expression(expression) => expression.argument_count(),
            Self::C(categorical) => categorical.definition().argument_count(),
            Self::F(integer) => integer.definition().argument_count(),
        }
    }

    pub(crate) fn call_name(&self, arguments: &[&str]) -> PlSmallStr {
        match self {
            Self::Expression(expression) => expression.call_name(arguments),
            Self::C(categorical) => categorical.definition().call_name(arguments),
            Self::F(integer) => integer.definition().call_name(arguments),
        }
    }

    pub(crate) fn fit(&self, inputs: &[Column]) -> Result<Self, TransformError> {
        #[cfg(test)]
        TRANSFORM_CALLS.update(|(fits, transforms)| (fits + 1, transforms));

        match self {
            Self::Expression(expression) => Ok(Self::Expression(expression.fit())),
            Self::C(categorical) => Ok(Self::C(categorical.fit(inputs)?)),
            Self::F(integer) => Ok(Self::F(integer.fit(inputs)?)),
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
    InputCountMismatch(String),
    InputNamesMismatch(String),
    InputTypeMismatch(String),
    NonNumericInput(String),
    InvalidParameter(String),
    Polars(PolarsError),
}

impl TransformError {
    pub(crate) fn input_count_mismatch(expected: ArgumentCount, actual: usize) -> Self {
        let (requirement, count) = match expected {
            ArgumentCount::Exactly(count) => ("exactly", count),
            ArgumentCount::AtLeast(count) => ("at least", count),
        };
        let noun = if count == 1 { "column" } else { "columns" };
        Self::InputCountMismatch(format!(
            "expected {requirement} {count} input {noun}, received {actual}",
        ))
    }

    pub(crate) fn input_names_mismatch(expected: &[PlSmallStr], inputs: &[Column]) -> Self {
        let actual = inputs.iter().map(|input| input.name()).collect::<Vec<_>>();
        Self::InputNamesMismatch(format!(
            "expected input names {expected:?}, received {actual:?}",
        ))
    }

    pub(crate) fn input_type_mismatch(input: &Column, expected: &DataType) -> Self {
        Self::InputTypeMismatch(format!(
            "input `{}`: expected {}, received {}",
            input.name(),
            expected,
            input.dtype(),
        ))
    }

    pub(crate) fn non_numeric_input(input: &Column) -> Self {
        Self::NonNumericInput(format!(
            "column `{}`: expected a primitive numeric dtype, received {}",
            input.name(),
            input.dtype(),
        ))
    }
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
            Self::InputCountMismatch(message)
            | Self::InputNamesMismatch(message)
            | Self::InputTypeMismatch(message)
            | Self::NonNumericInput(message)
            | Self::InvalidParameter(message) => f.write_str(message),
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
        assert_eq!(output.get_column_names(), ["F(group)"]);
        assert_eq!(
            output[0].u32()?.iter().collect::<Vec<_>>(),
            vec![Some(0), Some(1), Some(0)]
        );
        Ok(())
    }
}
