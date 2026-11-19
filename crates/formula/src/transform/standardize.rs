//! `center(x)` and `scale(x)`, which learn the mean (and standard deviation)
//! of `x` when fitting and reuse them when transforming new data.

use super::input::{PreparedInput, validate_numeric_field};
use super::{ArgumentCount, TransformDefinition, TransformError, call_name};
use crate::fit::Definition;
use crate::formula::{ComponentKey, MaterializedColumns};
use polars::prelude::{ChunkAgg, ChunkVar, DataType, Field, IntoLazy, col, lit};

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct StandardizeDefinition {
    scale: bool,
}

impl StandardizeDefinition {
    /// `center(x)`: subtract the fitted mean.
    pub(crate) const CENTER: Self = Self { scale: false };
    /// `scale(x)`: subtract the fitted mean and divide by the fitted sample
    /// standard deviation, as R's `scale()`.
    pub(crate) const SCALE: Self = Self { scale: true };
}

#[derive(Debug)]
pub(crate) struct StandardizeState {
    field: Field,
    mean: f64,
    // 1 for `center(x)`
    deviation: f64,
}

impl TransformDefinition for StandardizeDefinition {
    fn name(&self) -> &'static str {
        if self.scale { "scale" } else { "center" }
    }

    fn expands(&self) -> bool {
        false
    }

    fn argument_count(&self) -> ArgumentCount {
        ArgumentCount::Exactly(1)
    }
}

impl Definition<StandardizeState> for StandardizeDefinition {
    type Input = PreparedInput;
    type Output = MaterializedColumns<ComponentKey>;
    type Error = TransformError;

    fn fit(&self, inputs: &PreparedInput) -> Result<StandardizeState, TransformError> {
        let (field, input) = inputs.single()?;
        validate_numeric_field(field)?;
        let input = input.cast(&DataType::Float64)?;
        // Nonfinite values give nonfinite outputs, which fitting with missing
        // values dropped then leaves out. They do not enter the statistics.
        let input = input.filter(&input.is_finite()?)?;
        let values = input.f64()?;
        let mean = values.mean().ok_or_else(|| {
            TransformError::NumericalError(format!(
                "{}: no values to compute the mean of",
                self.name()
            ))
        })?;
        let deviation = if self.scale {
            let deviation = values.std(1).unwrap_or(f64::NAN);
            if !(deviation.is_finite() && deviation > 0.0) {
                return Err(TransformError::NumericalError(
                    "scale: the standard deviation must be positive and finite".into(),
                ));
            }
            deviation
        } else {
            1.0
        };
        Ok(StandardizeState {
            field: field.clone(),
            mean,
            deviation,
        })
    }

    fn validate(
        &self,
        state: &StandardizeState,
        inputs: &PreparedInput,
    ) -> Result<(), TransformError> {
        let (field, _) = inputs.single()?;
        if field.dtype() != state.field.dtype() {
            return Err(TransformError::InputTypeMismatch {
                name: field.name().clone(),
                expected: state.field.dtype().to_string(),
                actual: field.dtype().to_string(),
            });
        }
        Ok(())
    }

    fn transform(
        &self,
        state: &StandardizeState,
        inputs: &PreparedInput,
    ) -> Result<Self::Output, TransformError> {
        let (field, input) = inputs.single()?;
        let name = call_name(self.name(), [field.name().as_str()]);
        let expression = ((col(input.name().clone()).cast(DataType::Float64) - lit(state.mean))
            / lit(state.deviation))
        .alias(name);
        let output = inputs.frame.clone().lazy().select([expression]).collect()?;
        Ok(MaterializedColumns::single_column(
            output,
            ComponentKey::None,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fit::Fitted;
    use polars::prelude::Column;

    fn prepare(values: &[f64]) -> PreparedInput {
        PreparedInput::new(&[Column::new("x".into(), values)], false).unwrap()
    }

    fn output(
        fitted: &Fitted<StandardizeDefinition, StandardizeState>,
        values: &[f64],
    ) -> Vec<f64> {
        let output = fitted.transform(&prepare(values)).unwrap();
        let (frame, _) = output.into_parts();
        frame.columns()[0]
            .f64()
            .unwrap()
            .into_no_null_iter()
            .collect()
    }

    #[test]
    fn standardizing_transforms_take_one_input_and_keep_one_column() -> Result<(), TransformError> {
        for definition in [StandardizeDefinition::CENTER, StandardizeDefinition::SCALE] {
            assert!(!definition.expands());
            assert_eq!(definition.argument_count(), ArgumentCount::Exactly(1));
            let transform =
                crate::transform::Transform::Standardize(crate::fit::Fit::Unfitted(definition));
            assert!(transform.state().is_none());
            let input = [Column::new("x".into(), [1.0, 2.0, 4.0])];
            assert!(transform.fit(&input)?.state().is_some());
        }
        Ok(())
    }

    #[test]
    fn statistics_are_learned_once_and_reused() -> Result<(), TransformError> {
        let center = Fitted::fit(StandardizeDefinition::CENTER, &prepare(&[1.0, 2.0, 6.0]))?;
        assert_eq!(output(&center, &[1.0, 3.0]), [-2.0, 0.0]);
        let scale = Fitted::fit(StandardizeDefinition::SCALE, &prepare(&[1.0, 2.0, 3.0]))?;
        assert_eq!(output(&scale, &[1.0, 2.0, 4.0]), [-1.0, 0.0, 2.0]);
        Ok(())
    }

    #[test]
    fn degenerate_inputs_are_rejected() {
        for (definition, values) in [
            (StandardizeDefinition::SCALE, &[2.0, 2.0][..]),
            (StandardizeDefinition::SCALE, &[2.0][..]),
            (StandardizeDefinition::CENTER, &[][..]),
            (
                StandardizeDefinition::CENTER,
                &[f64::INFINITY, f64::NAN][..],
            ),
        ] {
            assert!(
                Fitted::fit(definition, &prepare(values)).is_err(),
                "{values:?}"
            );
        }
    }
}
