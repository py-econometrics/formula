use crate::fit::Definition;
use crate::transform::input::{argument_column_name, input_frame, validate_single_input};
use crate::transform::{TransformDefinition, call_name, validate_numeric_input};
use crate::{ArgumentCount, TransformError};
use polars::prelude::{
    AnyValue, Column, DataFrame, DataType, Expr, Field, Float64Chunked, IntoColumn, IntoLazy, NULL,
    PlSmallStr, Scalar, col, lit, when,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PolynomialDefinition {
    degree: usize,
    raw: bool,
}

impl PolynomialDefinition {
    pub(crate) fn new(degree: Option<Scalar>, raw: Option<Scalar>) -> Result<Self, TransformError> {
        let degree = degree.unwrap_or_else(|| Scalar::from(1_i64));
        let raw = raw.unwrap_or_else(|| Scalar::from(false));

        let AnyValue::Int64(degree) = degree.as_any_value() else {
            return Err(TransformError::InvalidParameter(
                "`degree` must be a positive integer".into(),
            ));
        };
        if degree <= 0 {
            return Err(TransformError::InvalidParameter(
                "`degree` must be a positive integer".into(),
            ));
        }
        let Ok(degree) = usize::try_from(degree) else {
            return Err(TransformError::InvalidParameter(
                "`degree` must be a positive integer".into(),
            ));
        };

        let AnyValue::Boolean(raw) = raw.as_any_value() else {
            return Err(TransformError::InvalidParameter(
                "`raw` must be a boolean".into(),
            ));
        };

        Ok(Self { degree, raw })
    }
    fn output_name(&self, input_name: &str, exponent: usize) -> PlSmallStr {
        let call = self.call_name(&[input_name]);
        format!("{call}[{exponent}]").into()
    }
}

impl TransformDefinition for PolynomialDefinition {
    fn name(&self) -> &'static str {
        "poly"
    }

    fn expands(&self) -> bool {
        true
    }

    fn argument_count(&self) -> ArgumentCount {
        ArgumentCount::Exactly(1)
    }

    fn call_name(&self, arguments: &[&str]) -> PlSmallStr {
        let degree = format!("degree={}", self.degree);
        let raw = format!("raw={}", self.raw);

        call_name(
            self.name(),
            arguments
                .iter()
                .copied()
                .chain([degree.as_str(), raw.as_str()]),
        )
    }
}

#[derive(Debug)]
enum PolynomialBasis {
    Raw,
    Orthogonal(OrthogonalBasis),
}

#[derive(Debug)]
struct OrthogonalBasis {
    // The fitted constant q0 = 1 / sqrt(number of finite training values).
    initial_value: f64,
    steps: Vec<RecurrenceStep>,
}

// q_next = ((x - alpha) * q_current - beta * q_previous) / norm.
#[derive(Debug)]
struct RecurrenceStep {
    alpha: f64,
    beta: f64,
    norm: f64,
}

impl PolynomialBasis {
    fn evaluate(&self, frame: DataFrame, degree: usize) -> Result<Vec<Column>, TransformError> {
        let name = argument_column_name(0);
        let argument = finite_values(col(name.clone()));

        match self {
            Self::Raw => {
                let prepared = frame.lazy().select([argument]).collect()?;
                let values = prepared.column(name.as_str())?;
                let mut columns = Vec::with_capacity(degree);
                columns.push(values.clone());

                // Reuse each power to compute the next with one multiplication.
                for exponent in 2..=degree {
                    columns.push((&columns[exponent - 2] * values)?);
                }

                Ok(columns)
            }

            Self::Orthogonal(basis) => {
                let prepared = frame.lazy().select([argument]).collect()?;
                basis.evaluate(prepared.column(name.as_str())?)
            }
        }
    }
}

impl OrthogonalBasis {
    fn evaluate(&self, values: &Column) -> Result<Vec<Column>, TransformError> {
        let values = values.f64()?.rechunk();
        let array = values
            .downcast_iter()
            .next()
            .expect("rechunked input has one array");
        let values = array.values().as_slice();
        let mut columns: Vec<Float64Chunked> = Vec::with_capacity(self.steps.len());

        // Compute on value buffers; apply the input null mask to each output afterward.
        for (index, step) in self.steps.iter().enumerate() {
            let degree = index + 1;
            let next = match degree {
                1 => values
                    .iter()
                    .map(|x| ((x - step.alpha) * self.initial_value) / step.norm)
                    .collect(),
                2 => values
                    .iter()
                    .zip(columns[0].cont_slice()?)
                    .map(|(x, current)| {
                        ((x - step.alpha) * current - step.beta * self.initial_value) / step.norm
                    })
                    .collect(),
                _ => {
                    let current = columns[degree - 2].cont_slice()?;
                    let previous = columns[degree - 3].cont_slice()?;

                    values
                        .iter()
                        .zip(current)
                        .zip(previous)
                        .map(|((x, current), previous)| {
                            ((x - step.alpha) * current - step.beta * previous) / step.norm
                        })
                        .collect()
                }
            };

            columns.push(Float64Chunked::from_vec("".into(), next));
        }

        Ok(columns
            .into_iter()
            .map(|column| {
                column
                    .with_validity(array.validity().cloned())
                    .into_column()
            })
            .collect())
    }

    fn training_frame(input: &Column, degree: usize) -> Result<DataFrame, TransformError> {
        let name = argument_column_name(0);
        let frame = input_frame(std::slice::from_ref(input))?
            .lazy()
            .select([finite_values(col(name.clone()))])
            .drop_nulls(None)
            .collect()?;

        let distinct = frame.column(name.as_str())?.n_unique()?;

        if degree >= distinct {
            return Err(TransformError::NumericalError(format!(
                "orthogonal polynomial degree {degree} requires at least {} distinct finite values; found {distinct}",
                degree.saturating_add(1),
            )));
        }

        Ok(frame)
    }

    fn fit(input: &Column, degree: usize) -> Result<Self, TransformError> {
        let frame = Self::training_frame(input, degree)?;
        let values = frame
            .column(argument_column_name(0).as_str())?
            .f64()?
            .rechunk();
        let values = values.cont_slice()?;
        let initial_value = 1.0 / (values.len() as f64).sqrt();
        let mut previous = vec![0.0; values.len()];
        let mut current = vec![initial_value; values.len()];
        let mut beta = 0.0_f64;
        let mut steps = Vec::with_capacity(degree);

        for exponent in 1..=degree {
            let alpha = sum_in_lanes(
                values
                    .iter()
                    .zip(&current)
                    .map(|(value, current)| value * current * current),
            );
            if !alpha.is_finite() || !beta.is_finite() {
                return Err(TransformError::NumericalError(format!(
                    "invalid orthogonal polynomial coefficients at degree {exponent}"
                )));
            }

            // Overwrite the previous vector with the next recurrence residual
            for ((previous, current), value) in previous.iter_mut().zip(&current).zip(values) {
                *previous = (value - alpha) * current - beta * *previous;
            }
            let next_norm2 = sum_in_lanes(previous.iter().map(|value| value * value));

            if !next_norm2.is_finite() || next_norm2 <= 0.0 {
                return Err(TransformError::NumericalError(format!(
                    "invalid orthogonal polynomial normalization at degree {exponent}"
                )));
            }

            let norm = next_norm2.sqrt();
            steps.push(RecurrenceStep { alpha, beta, norm });
            // Keep both recurrence vectors normalized between iterations
            for value in &mut previous {
                *value /= norm;
            }
            // Advance the recurrence by swapping buffers instead of allocating
            std::mem::swap(&mut previous, &mut current);
            beta = norm;
        }

        Ok(Self {
            initial_value,
            steps,
        })
    }
}

#[derive(Debug)]
pub(crate) struct PolynomialState {
    field: Field,
    basis: PolynomialBasis,
}

impl Definition<PolynomialState> for PolynomialDefinition {
    type Input = [Column];
    type Output = DataFrame;
    type Error = TransformError;

    fn validate(&self, state: &PolynomialState, inputs: &[Column]) -> Result<(), TransformError> {
        let input = validate_single_input(inputs)?;

        if input.name() != state.field.name() {
            return Err(TransformError::InputNamesMismatch {
                expected: vec![state.field.name().clone()],
                actual: vec![input.name().clone()],
            });
        }

        if input.dtype() != state.field.dtype() {
            return Err(TransformError::InputTypeMismatch {
                name: input.name().clone(),
                expected: state.field.dtype().to_string(),
                actual: input.dtype().to_string(),
            });
        }

        Ok(())
    }

    fn fit(&self, inputs: &[Column]) -> Result<PolynomialState, TransformError> {
        let input = validate_single_input(inputs)?;
        validate_numeric_input(input)?;

        let basis = if self.raw {
            PolynomialBasis::Raw
        } else {
            PolynomialBasis::Orthogonal(OrthogonalBasis::fit(input, self.degree)?)
        };

        Ok(PolynomialState {
            field: input.field().into_owned(),
            basis,
        })
    }

    fn transform(
        &self,
        state: &PolynomialState,
        inputs: &[Column],
    ) -> Result<DataFrame, TransformError> {
        let frame = input_frame(inputs)?;
        let height = frame.height();
        let columns = state
            .basis
            .evaluate(frame, self.degree)?
            .into_iter()
            .enumerate()
            .map(|(index, column)| {
                column.with_name(self.output_name(state.field.name().as_str(), index + 1))
            })
            .collect::<Vec<_>>();

        Ok(DataFrame::new(height, columns)?)
    }
}

fn finite_values(input: Expr) -> Expr {
    let input = input.cast(DataType::Float64);

    when(input.clone().is_finite())
        .then(input)
        .otherwise(lit(NULL).cast(DataType::Float64))
}

// Use four partial sums so the CPU can overlap additions.
fn sum_in_lanes(mut values: impl Iterator<Item = f64>) -> f64 {
    let mut sums = [0.0; 4];
    'input: loop {
        for sum in &mut sums {
            let Some(value) = values.next() else {
                break 'input;
            };
            *sum += value;
        }
    }
    sums.into_iter().sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fit::Fitted;

    #[test]
    fn constructor_parameters_default_independently() -> Result<(), TransformError> {
        for (degree, raw, expected_degree, expected_raw) in [
            (None, None, 1, false),
            (Some(Scalar::from(3_i64)), None, 3, false),
            (None, Some(Scalar::from(true)), 1, true),
        ] {
            let definition = PolynomialDefinition::new(degree, raw)?;
            assert_eq!(definition.degree, expected_degree);
            assert_eq!(definition.raw, expected_raw);
        }
        Ok(())
    }

    #[test]
    fn constructor_rejects_nonpositive_and_noninteger_degrees() {
        for degree in [
            Scalar::from(0_i64),
            Scalar::from(-1_i64),
            Scalar::from(i64::MIN),
            Scalar::from(3.0_f64),
            Scalar::from(2.5_f64),
            Scalar::from(f64::NAN),
            Scalar::from(f64::INFINITY),
            Scalar::from(true),
            Scalar::from(PlSmallStr::from_static("3")),
            Scalar::new(DataType::Int64, AnyValue::Null),
        ] {
            assert!(matches!(
                PolynomialDefinition::new(Some(degree), None),
                Err(TransformError::InvalidParameter(message))
                    if message == "`degree` must be a positive integer"
            ));
        }
    }

    #[test]
    fn constructor_rejects_nonboolean_raw_values() {
        for raw in [
            Scalar::from(0_i64),
            Scalar::from(1_i64),
            Scalar::from(1.0_f64),
            Scalar::from(PlSmallStr::from_static("true")),
            Scalar::new(DataType::Boolean, AnyValue::Null),
        ] {
            assert!(matches!(
                PolynomialDefinition::new(None, Some(raw)),
                Err(TransformError::InvalidParameter(message))
                    if message == "`raw` must be a boolean"
            ));
        }
    }

    // Absolute tolerance for these small, well-conditioned fixtures.
    const ORTHOGONAL_TOLERANCE: f64 = 1e-10;

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < ORTHOGONAL_TOLERANCE,
            "expected {expected}, received {actual}"
        );
    }

    fn assert_polynomial_schema(frame: &DataFrame, height: usize, degree: usize, raw: bool) {
        assert_eq!(frame.shape(), (height, degree));
        for (index, column) in frame.columns().iter().enumerate() {
            assert_eq!(column.dtype(), &DataType::Float64);
            assert_eq!(
                column.name().as_str(),
                format!("poly(x, degree={degree}, raw={raw})[{}]", index + 1)
            );
        }
    }

    #[test]
    fn orthogonal_basis_matches_hand_calculated_fixture() -> Result<(), TransformError> {
        let training = [Column::new("x".into(), [-1.0, 0.0, 1.0])];
        let fitted = Fitted::fit(
            PolynomialDefinition::new(Some(Scalar::from(2_i64)), Some(Scalar::from(false)))?,
            &training,
        )?;
        let output = fitted.transform(&training)?;
        assert_polynomial_schema(&output, 3, 2, false);

        // q1 = x / sqrt(2), q2 = (3*x^2 - 2) / sqrt(6).
        for (column, expected) in output.columns().iter().zip([
            [-1.0 / 2.0_f64.sqrt(), 0.0, 1.0 / 2.0_f64.sqrt()],
            [
                1.0 / 6.0_f64.sqrt(),
                -2.0 / 6.0_f64.sqrt(),
                1.0 / 6.0_f64.sqrt(),
            ],
        ]) {
            for (actual, expected) in column.f64()?.iter().zip(expected) {
                assert_close(actual.expect("finite training value"), expected);
            }
        }
        Ok(())
    }

    #[test]
    fn orthogonal_training_columns_are_centered_and_orthonormal() -> Result<(), TransformError> {
        for (degree, values) in [
            (3, vec![-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]),
            (3, vec![0.0, 0.0, 0.2, 1.0, 2.0, 4.0, 7.0]),
            (3, vec![98.0, 99.0, 100.0, 101.0, 103.0, 105.0]),
            (10, (0..20).map(|i| i as f64 * 1e20).collect()),
            (10, (0..20).map(|i| i as f64 * 1e-20).collect()),
        ] {
            let training = [Column::new("x".into(), values)];
            let fitted = Fitted::fit(
                PolynomialDefinition::new(Some(Scalar::from(degree as i64)), None)?,
                &training,
            )?;
            let output = fitted.transform(&training)?;
            assert_polynomial_schema(&output, training[0].len(), degree, false);
            let columns = output
                .columns()
                .iter()
                .map(|column| {
                    column
                        .f64()
                        .unwrap()
                        .iter()
                        .map(Option::unwrap)
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();

            for (index, column) in columns.iter().enumerate() {
                assert_close(column.iter().sum::<f64>(), 0.0);
                for (other_index, other) in columns.iter().enumerate() {
                    let product = column.iter().zip(other).map(|(a, b)| a * b).sum::<f64>();
                    assert_close(product, if index == other_index { 1.0 } else { 0.0 });
                }
            }
        }
        Ok(())
    }

    #[test]
    fn orthogonal_prediction_reuses_training_basis() -> Result<(), TransformError> {
        let training = [Column::new("x".into(), [9.0, 10.0, 11.0])];
        let fitted = Fitted::fit(
            PolynomialDefinition::new(Some(Scalar::from(2_i64)), Some(Scalar::from(false)))?,
            &training,
        )?;
        let prediction = [Column::new("x".into(), [12.0])];
        for _ in 0..2 {
            let output = fitted.transform(&prediction)?;
            assert_polynomial_schema(&output, 1, 2, false);
            assert_close(output[0].f64()?.get(0).unwrap(), 2.0 / 2.0_f64.sqrt());
            assert_close(output[1].f64()?.get(0).unwrap(), 10.0 / 6.0_f64.sqrt());
        }
        Ok(())
    }

    #[test]
    fn orthogonal_prediction_matches_for_subsets_and_concatenated_batches()
    -> Result<(), TransformError> {
        let fitted = Fitted::fit(
            PolynomialDefinition::new(Some(Scalar::from(3_i64)), None)?,
            &[Column::new("x".into(), [-2.0, -1.0, 0.0, 1.0, 2.0])],
        )?;
        let values = [Some(-1.0), None, Some(0.0), Some(1.0)];
        let expected = fitted.transform(&[Column::new("x".into(), values)])?;
        let subset = Column::new(
            "x".into(),
            [
                Some(42.0),
                Some(-1.0),
                None,
                Some(0.0),
                Some(1.0),
                Some(42.0),
            ],
        )
        .slice(1, values.len());
        let mut batches = Column::new("x".into(), [Some(-1.0), None]);
        batches.append(&Column::new("x".into(), [Some(0.0), Some(1.0)]))?;
        assert!(batches.as_materialized_series().n_chunks() > 1);

        for (case, input) in [("subset", subset), ("concatenated batches", batches)] {
            assert!(
                fitted.transform(&[input])?.equals_missing(&expected),
                "{case}"
            );
        }
        Ok(())
    }

    #[test]
    fn orthogonal_fit_ignores_missing_values_and_prediction_preserves_rows()
    -> Result<(), TransformError> {
        let clean = [Column::new("x".into(), [-1.0, 0.0, 1.0])];
        let missing = [Column::new(
            "x".into(),
            [
                Some(-1.0),
                None,
                Some(f64::NAN),
                Some(0.0),
                Some(f64::INFINITY),
                Some(1.0),
                Some(f64::NEG_INFINITY),
            ],
        )];
        let definition =
            PolynomialDefinition::new(Some(Scalar::from(2_i64)), Some(Scalar::from(false)))?;
        let clean_fit = Fitted::fit(definition.clone(), &clean)?;
        let missing_fit = Fitted::fit(definition, &missing)?;
        let expected = clean_fit.transform(&clean)?;

        for fitted in [&clean_fit, &missing_fit] {
            let output = fitted.transform(&missing)?;
            assert_polynomial_schema(&output, 7, 2, false);
            for (column, expected) in output.columns().iter().zip(expected.columns()) {
                assert_eq!(column.null_count(), 4);
                for row in [1, 2, 4, 6] {
                    assert_eq!(column.f64()?.get(row), None);
                }
                for (row, expected_row) in [(0, 0), (3, 1), (5, 2)] {
                    assert_close(
                        column.f64()?.get(row).unwrap(),
                        expected.f64()?.get(expected_row).unwrap(),
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn orthogonal_fit_rejects_insufficient_distinct_finite_values() -> Result<(), TransformError> {
        for (degree, values) in [
            (1, vec![]),
            (1, vec![None, None]),
            (
                1,
                vec![Some(f64::NAN), Some(f64::INFINITY), Some(f64::NEG_INFINITY)],
            ),
            (1, vec![Some(2.0), Some(2.0), Some(2.0)]),
            (
                2,
                vec![
                    Some(0.0),
                    Some(1.0),
                    Some(0.0),
                    Some(1.0),
                    None,
                    Some(f64::NAN),
                ],
            ),
        ] {
            assert!(matches!(
                Fitted::fit(
                    PolynomialDefinition::new(
                        Some(Scalar::from(i64::from(degree))),
                        Some(Scalar::from(false))
                    )?,
                    &[Column::new("x".into(), values)]
                ),
                Err(TransformError::NumericalError(_))
            ));
        }
        Ok(())
    }

    #[test]
    fn polynomial_empty_and_all_missing_predictions_retain_schema() -> Result<(), TransformError> {
        let training = [Column::new("x".into(), [-1.0, 0.0, 1.0])];
        for raw in [false, true] {
            let fitted = Fitted::fit(
                PolynomialDefinition::new(Some(Scalar::from(2_i64)), Some(Scalar::from(raw)))?,
                &training,
            )?;
            for values in [
                vec![],
                vec![None, None],
                vec![Some(f64::NAN), Some(f64::INFINITY), Some(f64::NEG_INFINITY)],
            ] {
                let height = values.len();
                let output = fitted.transform(&[Column::new("x".into(), values)])?;
                assert_polynomial_schema(&output, height, 2, raw);
                for column in output.columns() {
                    assert_eq!(column.null_count(), height, "raw={raw}");
                }
            }
        }
        Ok(())
    }

    #[test]
    fn orthogonal_fit_rejects_numerically_invalid_coefficients_and_norms()
    -> Result<(), TransformError> {
        for values in [
            vec![f64::MAX, f64::MAX / 2.0, f64::MAX / 4.0],
            vec![-f64::MAX, 0.0, f64::MAX],
            vec![-1e-200, 0.0, 1e-200],
        ] {
            assert!(matches!(
                Fitted::fit(
                    PolynomialDefinition::new(
                        Some(Scalar::from(1_i64)),
                        Some(Scalar::from(false))
                    )?,
                    &[Column::new("x".into(), values)]
                ),
                Err(TransformError::NumericalError(_))
            ));
        }
        Ok(())
    }

    #[test]
    fn raw_powers_preserve_values_order_and_schema_on_prediction() -> Result<(), TransformError> {
        let training = [Column::new("x".into(), [1_i64, 2])];
        let fitted = Fitted::fit(
            PolynomialDefinition::new(Some(Scalar::from(3_i64)), Some(Scalar::from(true)))?,
            &training,
        )?;
        let prediction = [Column::new("x".into(), [2_i64, -3, 0])];

        for _ in 0..2 {
            let output = fitted.transform(&prediction)?;
            assert_polynomial_schema(&output, 3, 3, true);
            for (column, expected) in output.columns().iter().zip([
                [Some(2.0), Some(-3.0), Some(0.0)],
                [Some(4.0), Some(9.0), Some(0.0)],
                [Some(8.0), Some(-27.0), Some(0.0)],
            ]) {
                assert_eq!(column.f64()?.iter().collect::<Vec<_>>(), expected);
            }
        }
        Ok(())
    }

    #[test]
    fn raw_powers_mask_nonfinite_and_null_inputs() -> Result<(), TransformError> {
        let inputs = [Column::new(
            "x".into(),
            [
                Some(2.0),
                None,
                Some(f64::NAN),
                Some(f64::INFINITY),
                Some(f64::NEG_INFINITY),
                Some(-0.5),
            ],
        )];
        let fitted = Fitted::fit(
            PolynomialDefinition::new(Some(Scalar::from(2_i64)), Some(Scalar::from(true)))?,
            &inputs,
        )?;
        let output = fitted.transform(&inputs)?;
        assert_polynomial_schema(&output, 6, 2, true);
        for (column, expected) in output.columns().iter().zip([
            [Some(2.0), None, None, None, None, Some(-0.5)],
            [Some(4.0), None, None, None, None, Some(0.25)],
        ]) {
            assert_eq!(column.f64()?.iter().collect::<Vec<_>>(), expected);
        }
        Ok(())
    }

    #[test]
    fn raw_fit_allows_empty_and_all_missing_training_data() -> Result<(), TransformError> {
        for values in [
            vec![],
            vec![None, None],
            vec![Some(f64::NAN), Some(f64::INFINITY), Some(f64::NEG_INFINITY)],
        ] {
            let fitted = Fitted::fit(
                PolynomialDefinition::new(Some(Scalar::from(2_i64)), Some(Scalar::from(true)))?,
                &[Column::new("x".into(), values)],
            )?;
            let output = fitted.transform(&[Column::new("x".into(), [3.0])])?;
            assert_polynomial_schema(&output, 1, 2, true);
            assert_eq!(output[0].f64()?.get(0), Some(3.0));
            assert_eq!(output[1].f64()?.get(0), Some(9.0));
        }
        Ok(())
    }

    #[test]
    fn raw_overflow_propagates_as_infinity() -> Result<(), TransformError> {
        let inputs = [Column::new("x".into(), [f64::MAX, -f64::MAX])];
        let fitted = Fitted::fit(
            PolynomialDefinition::new(Some(Scalar::from(3_i64)), Some(Scalar::from(true)))?,
            &inputs,
        )?;
        let output = fitted.transform(&inputs)?;
        assert_polynomial_schema(&output, 2, 3, true);
        assert_eq!(output[0].f64()?.get(0), Some(f64::MAX));
        assert_eq!(output[1].f64()?.get(0), Some(f64::INFINITY));
        assert_eq!(output[1].f64()?.get(1), Some(f64::INFINITY));
        assert_eq!(output[2].f64()?.get(1), Some(f64::NEG_INFINITY));
        Ok(())
    }

    #[test]
    fn polynomial_fit_and_prediction_require_exactly_one_input() -> Result<(), TransformError> {
        let inputs = [
            Column::new("x".into(), [1.0, 2.0, 3.0]),
            Column::new("z".into(), [4.0, 5.0, 6.0]),
        ];
        for raw in [false, true] {
            let definition =
                PolynomialDefinition::new(Some(Scalar::from(2_i64)), Some(Scalar::from(raw)))?;
            let fitted = Fitted::fit(definition.clone(), &inputs[..1])?;
            for invalid in [&inputs[..0], &inputs[..]] {
                for result in [
                    definition.fit(invalid).map(|_| ()),
                    fitted.transform(invalid).map(|_| ()),
                ] {
                    assert!(
                        matches!(
                            result,
                            Err(TransformError::InputCountMismatch {
                                expected: ArgumentCount::Exactly(1), actual,
                            }) if actual == invalid.len()
                        ),
                        "raw={raw}"
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn polynomial_fit_rejects_nonnumeric_input() -> Result<(), TransformError> {
        for raw in [false, true] {
            let definition = PolynomialDefinition::new(None, Some(Scalar::from(raw)))?;
            for input in [
                Column::new("x".into(), ["a"]),
                Column::new("x".into(), [true]),
            ] {
                assert!(
                    matches!(
                        definition.fit(&[input]),
                        Err(TransformError::NonNumericInput { .. })
                    ),
                    "raw={raw}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn polynomial_prediction_rejects_changed_names_and_dtypes() -> Result<(), TransformError> {
        let training = [Column::new("x".into(), [1.0, 2.0, 3.0])];
        for raw in [false, true] {
            let fitted = Fitted::fit(
                PolynomialDefinition::new(Some(Scalar::from(2_i64)), Some(Scalar::from(raw)))?,
                &training,
            )?;
            assert!(
                matches!(
                    fitted.transform(&[Column::new("z".into(), [1.0])]),
                    Err(TransformError::InputNamesMismatch { expected, actual })
                        if expected == vec![PlSmallStr::from_static("x")]
                            && actual == vec![PlSmallStr::from_static("z")]
                ),
                "raw={raw}"
            );
            assert!(
                matches!(
                    fitted.transform(&[Column::new("x".into(), [1_i64])]),
                    Err(TransformError::InputTypeMismatch { name, expected, actual })
                        if name.as_str() == "x" && expected == DataType::Float64.to_string()
                            && actual == DataType::Int64.to_string()
                ),
                "raw={raw}"
            );
        }
        Ok(())
    }

    #[test]
    fn polynomial_expands_and_requires_exactly_one_argument() -> Result<(), TransformError> {
        for raw in [false, true] {
            let definition =
                PolynomialDefinition::new(Some(Scalar::from(1_i64)), Some(Scalar::from(raw)))?;
            assert!(definition.expands());
            assert_eq!(definition.argument_count(), ArgumentCount::Exactly(1));
        }
        Ok(())
    }

    #[test]
    fn call_names_include_configuration_and_preserve_input_labels() -> Result<(), TransformError> {
        for (degree, raw, input, expected) in [
            (1, false, "x", "poly(x, degree=1, raw=false)"),
            (3, true, "x", "poly(x, degree=3, raw=true)"),
            (3, false, "log(x)", "poly(log(x), degree=3, raw=false)"),
        ] {
            let definition = PolynomialDefinition::new(
                Some(Scalar::from(i64::from(degree))),
                Some(Scalar::from(raw)),
            )?;
            assert_eq!(definition.call_name(&[input]).as_str(), expected);
        }
        Ok(())
    }
}
