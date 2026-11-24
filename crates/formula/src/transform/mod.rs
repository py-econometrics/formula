mod categorical;
mod codes;
mod expression;
mod grouped;
mod input;
mod integer;
mod polynomial;
mod standardize;
pub mod state;

pub(crate) use categorical::CategoricalDefinition;
#[cfg(test)]
pub(crate) use categorical::{LevelSelection, Levels};
pub(crate) use expression::ExpressionTransform;
pub(crate) use grouped::GroupedTransform;
pub use input::ArgumentCount;
use input::PreparedInput;
pub(crate) use input::validate_numeric_field;
pub(crate) use integer::IntegerEncodingDefinition;
pub(crate) use polynomial::PolynomialDefinition;
pub(crate) use standardize::StandardizeDefinition;
use standardize::StandardizeState;
pub use state::{StateValue, TransformState};
use std::collections::BTreeMap;

use crate::fit::{Fit, FitError};
use crate::formula::{ComponentKey, MaterializedColumns};
use crate::transform::categorical::CategoricalState;
pub use crate::transform::integer::IntegerEncodingState;
use crate::transform::polynomial::PolynomialState;
use crate::transform::state::ExportState;
use polars::error::PolarsError;
use polars::prelude::{Column, DataFrame, PlSmallStr, Scalar};

fn call_name<'a>(name: &str, arguments: impl IntoIterator<Item = &'a str>) -> PlSmallStr {
    let arguments = arguments.into_iter().collect::<Vec<_>>().join(", ");
    format!("{name}({arguments})").into()
}

#[derive(Clone, Debug, PartialEq)]
pub enum ParameterValue {
    Scalar(Scalar),
    List(Vec<Scalar>),
    Map(Vec<(PlSmallStr, ParameterValue)>),
}

#[derive(Clone, Debug)]
pub(crate) enum TransformOutput {
    Columns(MaterializedColumns<ComponentKey>),
    /// Indicator columns, each true where a row's code is the code of its column
    Indicator {
        columns: MaterializedColumns<ComponentKey>,
        codes: Column,
        count: u32,
    },
}

impl TransformOutput {
    pub(crate) fn columns(&self) -> &MaterializedColumns<ComponentKey> {
        match self {
            Self::Columns(columns) | Self::Indicator { columns, .. } => columns,
        }
    }

    pub(crate) fn frame(&self) -> &DataFrame {
        self.columns().frame()
    }
}

#[derive(Debug, PartialEq)]
pub(crate) enum Transform {
    Expression(ExpressionTransform),
    C(Fit<CategoricalDefinition, CategoricalState>),
    Poly(Fit<PolynomialDefinition, PolynomialState>),
    Standardize(Fit<StandardizeDefinition, StandardizeState>),
    F(Fit<IntegerEncodingDefinition, IntegerEncodingState>),
    G(GroupedTransform),
}

pub(crate) trait TransformDefinition {
    fn name(&self) -> &'static str;

    fn call_name(&self, arguments: &[&str]) -> PlSmallStr {
        call_name(self.name(), arguments.iter().copied())
    }

    fn expands(&self) -> bool;

    fn argument_count(&self) -> ArgumentCount;

    fn deduplicate_inputs(&self) -> bool {
        false
    }

    fn parameters(&self) -> BTreeMap<&'static str, ParameterValue> {
        BTreeMap::new()
    }
}

impl Transform {
    fn definition(&self) -> &dyn TransformDefinition {
        match self {
            Self::Expression(expression) => expression,
            Self::C(categorical) => categorical.definition(),
            Self::Poly(polynomial) => polynomial.definition(),
            Self::Standardize(standardize) => standardize.definition(),
            Self::F(integer) => integer.definition(),
            Self::G(grouped) => grouped,
        }
    }

    pub(crate) fn name(&self) -> &'static str {
        self.definition().name()
    }

    pub(crate) fn expands(&self) -> bool {
        self.definition().expands()
    }

    pub(crate) fn argument_count(&self) -> ArgumentCount {
        self.definition().argument_count()
    }

    pub(crate) fn parameters(&self) -> BTreeMap<&'static str, ParameterValue> {
        self.definition().parameters()
    }

    pub(crate) fn call_name(&self, arguments: &[&str]) -> PlSmallStr {
        self.definition().call_name(arguments)
    }

    fn input(&self, inputs: &[Column]) -> Result<PreparedInput, TransformError> {
        PreparedInput::new(inputs, self.definition().deduplicate_inputs())
    }

    pub(crate) fn fit(&self, inputs: &[Column]) -> Result<Self, TransformError> {
        #[cfg(test)]
        TRANSFORM_CALLS.update(|(fits, transforms)| (fits + 1, transforms));
        match self {
            Self::C(categorical) => {
                let inputs = self.input(inputs)?;
                Ok(Self::C(categorical.fit(&inputs)?))
            }
            Self::Poly(polynomial) => {
                let inputs = self.input(inputs)?;
                Ok(Self::Poly(polynomial.fit(&inputs)?))
            }
            Self::Standardize(standardize) => {
                let inputs = self.input(inputs)?;
                Ok(Self::Standardize(standardize.fit(&inputs)?))
            }
            Self::F(integer) => {
                let inputs = self.input(inputs)?;
                Ok(Self::F(integer.fit(&inputs)?))
            }

            Self::Expression(expression) => Ok(Self::Expression(expression.clone())),
            Self::G(grouped) => Ok(Self::G(grouped.clone())),
        }
    }

    pub(crate) fn transform(
        &self,
        inputs: &[Column],
    ) -> Result<TransformOutput, FitError<TransformError>> {
        #[cfg(test)]
        TRANSFORM_CALLS.update(|(fits, transforms)| (fits, transforms + 1));

        let inputs = self.input(inputs).map_err(FitError::Transform)?;
        match self {
            Self::Expression(expression) => {
                expression.transform(&inputs).map_err(FitError::Transform)
            }
            Self::C(categorical) => categorical.transform(&inputs),
            Self::Poly(polynomial) => polynomial.transform(&inputs),
            Self::Standardize(standardize) => standardize.transform(&inputs),
            Self::F(integer) => integer.transform(&inputs),
            Self::G(grouped) => grouped.transform(&inputs).map_err(FitError::Transform),
        }
    }

    pub(crate) fn state(&self) -> Option<TransformState> {
        match self {
            Self::C(categorical) => categorical.state().map(|_| TransformState::default()),
            Self::Poly(polynomial) => polynomial.state().map(|_| TransformState::default()),
            Self::Standardize(standardize) => {
                standardize.state().map(|_| TransformState::default())
            }
            Self::F(integer) => integer.state().map(ExportState::export_state),

            Self::Expression(_) => Some(TransformState::default()),
            Self::G(_) => Some(TransformState::default()),
        }
    }
}

#[cfg(test)]
thread_local! {
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

    fn prepare(inputs: &[Column]) -> PreparedInput {
        PreparedInput::new(inputs, false).unwrap()
    }

    #[test]
    fn fit_identity_depends_on_definition_not_fitted_state() -> Result<(), TransformError> {
        let definition = CategoricalDefinition::new(
            Some(LevelSelection::Omit(Levels::Scalar(Scalar::from(
                PlSmallStr::from_static("a"),
            )))),
            None,
        );
        let first_input = prepare(&[Column::new("group".into(), ["a", "b"])]);
        let second_input = prepare(&[Column::new("group".into(), ["a", "c"])]);
        let first = Fit::Fitted(Fitted::fit(definition.clone(), &first_input)?);
        let second = Fit::Fitted(Fitted::fit(definition.clone(), &second_input)?);

        assert_eq!(first, second);
        assert_eq!(first, Fit::Unfitted(definition));
        assert_ne!(first, Fit::Unfitted(CategoricalDefinition::new(None, None)));

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

        assert!(transform.state().is_none());
        assert!(!transform.expands());
        assert_eq!(transform.argument_count(), ArgumentCount::AtLeast(1));
        assert!(matches!(
            transform.transform(&inputs),
            Err(FitError::NotFitted)
        ));

        let transform = transform.fit(&inputs)?;
        let state = transform.state().expect("fitted encoder exposes its state");
        assert_eq!(state.iter().count(), 2);
        assert!(matches!(state.get("codes"), Some(StateValue::DataFrame(_))));
        assert!(matches!(
            state.get("levels"),
            Some(StateValue::DataFrame(_))
        ));
        let output = transform.transform(&inputs).unwrap();
        assert_eq!(output.frame().get_column_names(), ["group"]);
        assert_eq!(
            output.frame()[0].u32()?.iter().collect::<Vec<_>>(),
            vec![Some(0), Some(1), Some(0)]
        );
        Ok(())
    }

    #[test]
    fn state_distinguishes_unfitted_transforms_from_empty_snapshots() -> Result<(), TransformError>
    {
        let inputs = [Column::new("x".into(), [1.0, 2.0, 1.0])];
        for transform in [
            Transform::C(Fit::Unfitted(CategoricalDefinition::new(None, None))),
            Transform::Poly(Fit::Unfitted(PolynomialDefinition::from_parameters(
                None,
                Some(ParameterValue::Scalar(Scalar::from(true))),
            )?)),
        ] {
            assert!(transform.state().is_none());
            let fitted = transform.fit(&inputs)?;
            assert!(
                fitted
                    .state()
                    .expect("fitted transform has a snapshot")
                    .iter()
                    .next()
                    .is_none()
            );
        }
        for transform in [
            Transform::Expression(ExpressionTransform::LOG),
            Transform::G(GroupedTransform::new(false)),
        ] {
            assert!(
                transform
                    .state()
                    .expect("stateless transform has an empty snapshot")
                    .iter()
                    .next()
                    .is_none()
            );
        }
        Ok(())
    }
}
