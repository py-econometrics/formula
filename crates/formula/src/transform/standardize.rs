//! `center(x)` and `scale(x)`, which learn the mean (and standard deviation)
//! of `x` when fitting and reuse them when transforming new data.

use super::input::{PreparedInput, validate_numeric_field};
use super::{ArgumentCount, TransformDefinition, TransformError, TransformOutput, call_name};
use crate::fit::Definition;
use crate::formula::{ComponentKey, MaterializedColumns};
use polars::prelude::{Field, IntoLazy, NULL, col, lit, when};

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
    // `None` for `center(x)`
    deviation: Option<f64>,
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
    type Output = TransformOutput;
    type Error = TransformError;

    fn fit(&self, inputs: &PreparedInput) -> Result<StandardizeState, TransformError> {
        let (field, input) = inputs.single()?;
        validate_numeric_field(field)?;
        // Skip non-finite values
        let input = input.as_materialized_series();
        let values = input.filter(&input.is_finite()?)?;
        let mean = values.mean().ok_or_else(|| {
            TransformError::NumericalError("no finite values to compute the mean of".into())
        })?;
        if !mean.is_finite() {
            return Err(TransformError::NumericalError(
                "the mean must be finite".into(),
            ));
        }
        let deviation = if self.scale {
            // Constant columns may end up with non-zero deviation due to rounding
            if values.min_reduce()? == values.max_reduce()? {
                return Err(TransformError::NumericalError(
                    "the input is constant".into(),
                ));
            }
            let deviation = values
                .std(1)
                .filter(|deviation| deviation.is_finite() && *deviation > 0.0)
                .ok_or_else(|| {
                    TransformError::NumericalError(
                        "the standard deviation must be positive and finite".into(),
                    )
                })?;
            Some(deviation)
        } else {
            None
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
        if field.name() != state.field.name() {
            return Err(TransformError::InputNamesMismatch {
                expected: vec![state.field.name().clone()],
                actual: vec![field.name().clone()],
            });
        }
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
        // Non-finite values become missing
        let column = col(input.name().clone());
        let finite = when(column.clone().is_finite())
            .then(column)
            .otherwise(lit(NULL));
        let centered = finite - lit(state.mean);
        let expression = match state.deviation {
            Some(deviation) => centered / lit(deviation),
            None => centered,
        }
        .alias(name);
        let output = inputs.frame.clone().lazy().select([expression]).collect()?;
        Ok(TransformOutput::Columns(
            MaterializedColumns::single_column(output, ComponentKey::None),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fit::Fitted;
    use polars::prelude::{Column, DataType, PlSmallStr};

    fn prepare(values: &[f64]) -> PreparedInput {
        PreparedInput::new(&[Column::new("x".into(), values)], false).unwrap()
    }

    fn output(
        fitted: &Fitted<StandardizeDefinition, StandardizeState>,
        values: &[f64],
    ) -> Vec<f64> {
        let output = fitted.transform(&prepare(values)).unwrap();
        output.frame().columns()[0]
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
            (StandardizeDefinition::CENTER, &[f64::INFINITY][..]),
        ] {
            assert!(
                Fitted::fit(definition, &prepare(values)).is_err(),
                "{values:?}"
            );
        }
    }

    #[test]
    fn scale_rejection_names_the_cause() {
        for (values, cause) in [
            // The fitted deviation of [0.1; 3] is about 1.7e-17, not zero.
            (&[0.1, 0.1, 0.1][..], "constant"),
            (&[2.2; 100][..], "constant"),
            // The infinite value is dropped, leaving one value.
            (&[1.0, f64::INFINITY][..], "constant"),
            (&[f64::MAX, f64::MAX][..], "mean"),
            (&[f64::NAN, f64::INFINITY][..], "no finite values"),
            // Distinct values whose variance overflows or underflows.
            (&[-f64::MAX, f64::MAX][..], "standard deviation"),
            (&[1e-200, 2e-200][..], "standard deviation"),
        ] {
            assert!(
                matches!(
                    Fitted::fit(StandardizeDefinition::SCALE, &prepare(values)),
                    Err(TransformError::NumericalError(message)) if message.contains(cause)
                ),
                "{values:?}"
            );
        }
    }

    #[test]
    fn nonfinite_values_are_missing() -> Result<(), TransformError> {
        let inputs = PreparedInput::new(
            &[Column::new(
                "x".into(),
                [
                    Some(1.0),
                    None,
                    Some(f64::NAN),
                    Some(3.0),
                    Some(f64::INFINITY),
                    Some(f64::NEG_INFINITY),
                    Some(5.0),
                ],
            )],
            false,
        )?;
        for (definition, expected) in [
            (
                StandardizeDefinition::CENTER,
                [Some(-2.0), None, None, Some(0.0), None, None, Some(2.0)],
            ),
            (
                StandardizeDefinition::SCALE,
                [Some(-1.0), None, None, Some(0.0), None, None, Some(1.0)],
            ),
        ] {
            let output = Fitted::fit(definition, &inputs)?.transform(&inputs)?;
            let actual = output.frame().columns()[0]
                .f64()?
                .iter()
                .collect::<Vec<_>>();
            assert_eq!(actual, expected);
        }
        Ok(())
    }

    #[test]
    fn integers_become_floats_and_float_types_are_kept() -> Result<(), TransformError> {
        for (input, dtype) in [
            (Column::new("x".into(), [1_i64, 3, 5]), DataType::Float64),
            (Column::new("x".into(), [1_u8, 3, 5]), DataType::Float64),
            (
                Column::new("x".into(), [1.0_f32, 3.0, 5.0]),
                DataType::Float32,
            ),
        ] {
            let inputs = PreparedInput::new(&[input], false)?;
            for (definition, expected) in [
                (StandardizeDefinition::CENTER, [-2.0, 0.0, 2.0]),
                (StandardizeDefinition::SCALE, [-1.0, 0.0, 1.0]),
            ] {
                let output = Fitted::fit(definition, &inputs)?.transform(&inputs)?;
                let column = &output.frame().columns()[0];
                assert_eq!(column.dtype(), &dtype);
                let actual = column
                    .cast(&DataType::Float64)?
                    .f64()?
                    .into_no_null_iter()
                    .collect::<Vec<_>>();
                assert_eq!(actual, expected);
            }
        }
        Ok(())
    }

    #[test]
    fn prediction_rejects_changed_names_and_dtypes() -> Result<(), TransformError> {
        for definition in [StandardizeDefinition::CENTER, StandardizeDefinition::SCALE] {
            let fitted = Fitted::fit(definition, &prepare(&[1.0, 2.0, 3.0]))?;
            let renamed = PreparedInput::new(&[Column::new("z".into(), [1.0])], false)?;
            assert!(matches!(
                fitted.transform(&renamed),
                Err(TransformError::InputNamesMismatch { expected, actual })
                    if expected == vec![PlSmallStr::from_static("x")]
                        && actual == vec![PlSmallStr::from_static("z")]
            ));
            let retyped = PreparedInput::new(&[Column::new("x".into(), [1_i64])], false)?;
            assert!(matches!(
                fitted.transform(&retyped),
                Err(TransformError::InputTypeMismatch { name, expected, actual })
                    if name.as_str() == "x" && expected == DataType::Float64.to_string()
                        && actual == DataType::Int64.to_string()
            ));
        }
        Ok(())
    }
}
