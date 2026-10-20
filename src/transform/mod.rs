mod categorical;
mod expression;
mod input;
mod integer;

pub(crate) use categorical::CategoricalDefinition;
pub(crate) use expression::ExpressionTransform;
pub use input::ArgumentCount;
pub(crate) use integer::IntegerEncodingDefinition;

use polars::error::PolarsError;
use polars::prelude::{DataFrame, PlSmallStr, Series};
use std::fmt;

fn call_name<'a>(name: &str, arguments: impl IntoIterator<Item = &'a str>) -> PlSmallStr {
    let arguments = arguments.into_iter().collect::<Vec<_>>().join(", ");
    format!("{name}({arguments})").into()
}

#[derive(Debug, PartialEq)]
pub enum Transform {
    Expression(ExpressionTransform),
    C(Fit<CategoricalDefinition>),
    F(Fit<IntegerEncodingDefinition>),
}

pub(crate) trait TransformDefinition {
    /// Data learned during fitting.
    type Fit;

    fn expands(&self) -> bool;

    fn argument_count(&self) -> ArgumentCount;

    fn fit(&self, inputs: &[Series]) -> Result<Self::Fit, TransformError>;

    fn transform(&self, fit: &Self::Fit, inputs: &[Series]) -> Result<DataFrame, TransformError>;
}

pub(crate) enum Fit<D: TransformDefinition> {
    Unfitted(D),
    Fitted(Fitted<D>),
}

/// A transform definition and its fitted state.
pub(crate) struct Fitted<D: TransformDefinition> {
    definition: D,
    fit: D::Fit,
}

impl<D: TransformDefinition> Fitted<D> {
    pub(crate) fn new(definition: D, inputs: &[Series]) -> Result<Self, TransformError> {
        let fit = definition.fit(inputs)?;
        Ok(Self { definition, fit })
    }

    pub(crate) fn transform(&self, inputs: &[Series]) -> Result<DataFrame, TransformError> {
        self.definition.transform(&self.fit, inputs)
    }
}

impl<D> fmt::Debug for Fitted<D>
where
    D: TransformDefinition + fmt::Debug,
    D::Fit: fmt::Debug,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Fitted")
            .field("definition", &self.definition)
            .field("fit", &self.fit)
            .finish()
    }
}

impl<D: TransformDefinition> Fit<D> {
    pub(crate) fn definition(&self) -> &D {
        match self {
            Self::Unfitted(definition) => definition,
            Self::Fitted(operation) => &operation.definition,
        }
    }
}

impl<D> fmt::Debug for Fit<D>
where
    D: TransformDefinition + fmt::Debug,
    D::Fit: fmt::Debug,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unfitted(definition) => {
                formatter.debug_tuple("Unfitted").field(definition).finish()
            }
            Self::Fitted(operation) => formatter.debug_tuple("Fitted").field(operation).finish(),
        }
    }
}

// Factor identity describes the definition, not the data used for fitting.
impl<D: TransformDefinition + PartialEq> PartialEq for Fit<D> {
    fn eq(&self, other: &Self) -> bool {
        self.definition() == other.definition()
    }
}

impl<D: TransformDefinition + Clone> Fit<D> {
    pub(crate) fn expands(&self) -> bool {
        self.definition().expands()
    }

    pub(crate) fn argument_count(&self) -> ArgumentCount {
        self.definition().argument_count()
    }

    pub(crate) fn fit(&self, inputs: &[Series]) -> Result<Self, TransformError> {
        let fitted = Fitted::new(self.definition().clone(), inputs)?;
        Ok(Self::Fitted(fitted))
    }

    pub(crate) fn transform(&self, inputs: &[Series]) -> Result<DataFrame, TransformError> {
        match self {
            Self::Unfitted(_) => Err(TransformError::NotFitted),
            Self::Fitted(operation) => operation.transform(inputs),
        }
    }
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

    pub(crate) fn fit(&self, inputs: &[Series]) -> Result<Self, TransformError> {
        #[cfg(test)]
        TRANSFORM_CALLS.update(|(fits, transforms)| (fits + 1, transforms));

        match self {
            Self::Expression(expression) => Ok(Self::Expression(expression.fit())),
            Self::C(categorical) => Ok(Self::C(categorical.fit(inputs)?)),
            Self::F(integer) => Ok(Self::F(integer.fit(inputs)?)),
        }
    }

    pub(crate) fn transform(&self, inputs: &[Series]) -> Result<DataFrame, TransformError> {
        #[cfg(test)]
        TRANSFORM_CALLS.update(|(fits, transforms)| (fits, transforms + 1));

        match self {
            Self::Expression(expression) => expression.transform(inputs),
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
    NotFitted,
    InputTypeMismatch,
    InvalidParameter(String),
    Polars(PolarsError),
}

impl From<PolarsError> for TransformError {
    fn from(error: PolarsError) -> Self {
        Self::Polars(error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use polars::prelude::{NamedFrom, PlSmallStr, Scalar};

    #[test]
    fn fit_identity_depends_on_definition_not_fitted_state() -> Result<(), TransformError> {
        let definition =
            CategoricalDefinition::new(Some(Scalar::from(PlSmallStr::from_static("a"))));
        let first_input = [Series::new("group".into(), ["a", "b"])];
        let second_input = [Series::new("group".into(), ["a", "c"])];
        let first = Fit::Fitted(Fitted::new(definition.clone(), &first_input)?);
        let second = Fit::Fitted(Fitted::new(definition.clone(), &second_input)?);

        assert_eq!(first, second);
        assert_eq!(first, Fit::Unfitted(definition));
        assert_ne!(first, Fit::Unfitted(CategoricalDefinition::new(None)));

        let definition = IntegerEncodingDefinition;
        let first = Fit::Fitted(Fitted::new(definition.clone(), &first_input)?);
        let second = Fit::Fitted(Fitted::new(definition.clone(), &second_input)?);
        assert_eq!(first, second);
        assert_eq!(first, Fit::Unfitted(definition));
        Ok(())
    }

    #[test]
    fn transform_enum_delegates_integer_operations() -> Result<(), TransformError> {
        let inputs = [Series::new("group".into(), ["a", "b", "a"])];
        let transform = Transform::F(Fit::Unfitted(IntegerEncodingDefinition));

        assert!(!transform.expands());
        assert_eq!(transform.argument_count(), ArgumentCount::AtLeast(1));
        assert!(matches!(
            transform.transform(&inputs),
            Err(TransformError::NotFitted)
        ));

        let transform = transform.fit(&inputs)?;
        let output = transform.transform(&inputs)?;
        assert_eq!(output.get_column_names(), ["F(group)"]);
        assert_eq!(
            output[0].u32()?.iter().collect::<Vec<_>>(),
            vec![Some(0), Some(1), Some(0)]
        );
        Ok(())
    }
}
