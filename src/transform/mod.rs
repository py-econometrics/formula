mod categorical;
mod expression;
mod input;
mod integer;

pub(crate) use categorical::CategoricalDefinition;
pub(crate) use expression::ExpressionTransform;
pub use input::ArgumentCount;
pub(crate) use input::validate_numeric_input;
pub(crate) use integer::IntegerEncodingDefinition;

use crate::fit::{Fit, FitError, Fitted};
use crate::transform::categorical::CategoricalState;
use crate::transform::integer::IntegerEncodingState;
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
}

impl<D: TransformDefinition> Fitted<D, D::State> {
    pub(crate) fn fit(definition: D, inputs: &[Column]) -> Result<Self, TransformError> {
        let state = definition.fit(inputs)?;
        Ok(Self::new(definition, state))
    }

    pub(crate) fn transform(&self, inputs: &[Column]) -> Result<DataFrame, TransformError> {
        self.definition.transform(&self.state, inputs)
    }
}

impl<D: TransformDefinition + Clone> Fit<D, D::State> {
    pub(crate) fn expands(&self) -> bool {
        self.definition().expands()
    }

    pub(crate) fn argument_count(&self) -> ArgumentCount {
        self.definition().argument_count()
    }

    pub(crate) fn fit(&self, inputs: &[Column]) -> Result<Self, TransformError> {
        let fitted = Fitted::fit(self.definition().clone(), inputs)?;
        Ok(Self::Fitted(fitted))
    }

    pub(crate) fn transform(
        &self,
        inputs: &[Column],
    ) -> Result<DataFrame, FitError<TransformError>> {
        match self {
            Self::Unfitted(_) => Err(FitError::NotFitted),
            Self::Fitted(operation) => operation.transform(inputs).map_err(FitError::Definition),
        }
    }
}

pub(crate) trait TransformDefinition {
    /// Data learned during fitting.
    type State;

    fn expands(&self) -> bool;

    fn argument_count(&self) -> ArgumentCount;

    fn fit(&self, inputs: &[Column]) -> Result<Self::State, TransformError>;

    fn transform(
        &self,
        state: &Self::State,
        inputs: &[Column],
    ) -> Result<DataFrame, TransformError>;
}

impl Transform {
    pub(crate) fn expands(&self) -> bool {
        match self {
            Self::Expression(expression) => expression.expands(),
            Self::C(categorical) => categorical.expands(),
            Self::F(integer) => integer.expands(),
        }
    }

    pub(crate) fn argument_count(&self) -> ArgumentCount {
        match self {
            Self::Expression(expression) => expression.argument_count(),
            Self::C(categorical) => categorical.argument_count(),
            Self::F(integer) => integer.argument_count(),
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
                expression.transform(inputs).map_err(FitError::Definition)
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
    InputTypeMismatch,
    InvalidParameter(String),
    Polars(PolarsError),
    DtypeMismatch(String),
    NonNumericInput(String),
}

impl From<PolarsError> for TransformError {
    fn from(error: PolarsError) -> Self {
        Self::Polars(error)
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
