use crate::fit::Definition;
use crate::transform::input::{argument_column_name, input_frame, validate_single_input};
use crate::transform::{TransformDefinition, call_name, validate_numeric_input};
use crate::{ArgumentCount, TransformError};
use polars::prelude::{
    AnyValue, Column, DataFrame, DataType, Expr, Field, IntoLazy, NULL, PlSmallStr, Scalar, col,
    lit, when,
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
    steps: Vec<RecurrenceStep>,
}

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
        let mut previous = Column::new_scalar("previous".into(), Scalar::from(0.0), values.len());
        let mut current = Column::new_scalar("current".into(), Scalar::from(1.0), values.len());
        let mut columns = Vec::with_capacity(self.steps.len());

        for step in &self.steps {
            let centered = values - step.alpha;
            let leading = (&centered * &current)?;
            let correction = &previous * step.beta;
            let next = (&leading - &correction)?;

            columns.push(&next / step.norm);

            previous = current;
            current = next;
        }

        Ok(columns)
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
        let values = frame.column(argument_column_name(0).as_str())?;

        // Initialise starting polynomials p0 = 1 and p-1 = 0
        let mut previous = Column::new_scalar("previous".into(), Scalar::from(0.0), frame.height());
        let mut current = Column::new_scalar("current".into(), Scalar::from(1.0), frame.height());
        // Initialise values
        let mut current_norm2 = frame.height() as f64;
        let mut previous_norm2 = 0.0;
        let mut steps = Vec::with_capacity(degree);

        for exponent in 1..=degree {
            let squared = (&current * &current)?;
            let weighted_values = (values * &squared)?;
            let weighted_sum = weighted_values.sum_reduce()?.value().try_extract::<f64>()?;

            // Update alpha and beta
            let alpha = weighted_sum / current_norm2;
            let beta = if exponent == 1 {
                0.0
            } else {
                current_norm2 / previous_norm2
            };
            if !alpha.is_finite() || !beta.is_finite() {
                return Err(TransformError::NumericalError(format!(
                    "invalid orthogonal polynomial coefficients at degree {exponent}"
                )));
            }

            // Compute the next unnormalized polynomial using the three-term recurrence.
            let centered = values - alpha;
            let leading = (&centered * &current)?;
            let correction = &previous * beta;
            let next = (&leading - &correction)?;
            let next_norm2 = (&next * &next)?
                .sum_reduce()?
                .value()
                .try_extract::<f64>()?;

            if !next_norm2.is_finite() || next_norm2 <= 0.0 {
                return Err(TransformError::NumericalError(format!(
                    "invalid orthogonal polynomial normalization at degree {exponent}"
                )));
            }

            steps.push(RecurrenceStep {
                alpha,
                beta,
                norm: next_norm2.sqrt(),
            });

            previous = current;
            current = next;
            previous_norm2 = current_norm2;
            current_norm2 = next_norm2;
        }

        Ok(Self { steps })
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fit::Fitted;

    #[test]
    fn constructor_defaults_to_degree_one_and_orthogonal_mode() -> Result<(), TransformError> {
        let definition = PolynomialDefinition::new(None, None)?;
        assert_eq!(definition.degree, 1);
        assert!(!definition.raw);
        assert_eq!(
            definition,
            PolynomialDefinition::new(Some(Scalar::from(1_i64)), Some(Scalar::from(false)),)?
        );
        Ok(())
    }

    #[test]
    fn constructor_parameters_default_independently() -> Result<(), TransformError> {
        let definition = PolynomialDefinition::new(Some(Scalar::from(3_i64)), None)?;
        assert_eq!(definition.degree, 3);
        assert!(!definition.raw);

        let definition = PolynomialDefinition::new(None, Some(Scalar::from(true)))?;
        assert_eq!(definition.degree, 1);
        assert!(definition.raw);
        Ok(())
    }

    #[test]
    fn constructor_rejects_negative_and_noninteger_degrees() {
        for degree in [
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
            for raw in [false, true] {
                assert!(matches!(
                    PolynomialDefinition::new(
                        Some(degree.clone()),
                        Some(Scalar::from(raw)),
                    ),
                    Err(TransformError::InvalidParameter(message))
                        if message == "`degree` must be a positive integer"
                ));
            }
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

    fn assert_orthogonal_schema(frame: &DataFrame, height: usize, degree: usize) {
        assert_eq!(frame.shape(), (height, degree));
        for (index, column) in frame.columns().iter().enumerate() {
            assert_eq!(column.dtype(), &DataType::Float64);
            assert_eq!(
                column.name().as_str(),
                format!("poly(x, degree={degree}, raw=false)[{}]", index + 1)
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
        assert_orthogonal_schema(&output, 3, 2);

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
        for values in [
            vec![-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
            vec![0.0, 0.0, 0.2, 1.0, 2.0, 4.0, 7.0],
            vec![98.0, 99.0, 100.0, 101.0, 103.0, 105.0],
        ] {
            let training = [Column::new("x".into(), values)];
            let fitted = Fitted::fit(
                PolynomialDefinition::new(Some(Scalar::from(3_i64)), Some(Scalar::from(false)))?,
                &training,
            )?;
            let output = fitted.transform(&training)?;
            assert_orthogonal_schema(&output, training[0].len(), 3);
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
            assert_orthogonal_schema(&output, 1, 2);
            assert_close(output[0].f64()?.get(0).unwrap(), 2.0 / 2.0_f64.sqrt());
            assert_close(output[1].f64()?.get(0).unwrap(), 10.0 / 6.0_f64.sqrt());
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
            assert_orthogonal_schema(&output, 7, 2);
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
    fn orthogonal_empty_and_all_missing_predictions_retain_schema() -> Result<(), TransformError> {
        let training = [Column::new("x".into(), [-1.0, 0.0, 1.0])];
        let fitted = Fitted::fit(
            PolynomialDefinition::new(Some(Scalar::from(2_i64)), Some(Scalar::from(false)))?,
            &training,
        )?;
        for values in [
            vec![],
            vec![None, None],
            vec![Some(f64::NAN), Some(f64::INFINITY), Some(f64::NEG_INFINITY)],
        ] {
            let height = values.len();
            let output = fitted.transform(&[Column::new("x".into(), values)])?;
            assert_orthogonal_schema(&output, height, 2);
            for column in output.columns() {
                assert_eq!(column.null_count(), height);
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

    fn assert_raw_schema(frame: &DataFrame, height: usize, degree: usize) {
        assert_eq!(frame.shape(), (height, degree));
        for (index, column) in frame.columns().iter().enumerate() {
            assert_eq!(column.dtype(), &DataType::Float64);
            assert_eq!(
                column.name().as_str(),
                format!("poly(x, degree={degree}, raw=true)[{}]", index + 1)
            );
        }
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
            assert_raw_schema(&output, 3, 3);
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
        assert_raw_schema(&output, 6, 2);
        for (column, expected) in output.columns().iter().zip([
            [Some(2.0), None, None, None, None, Some(-0.5)],
            [Some(4.0), None, None, None, None, Some(0.25)],
        ]) {
            assert_eq!(column.f64()?.iter().collect::<Vec<_>>(), expected);
        }
        Ok(())
    }

    #[test]
    fn raw_fit_and_prediction_allow_empty_and_all_missing_data() -> Result<(), TransformError> {
        let cases = [
            vec![],
            vec![None, None],
            vec![Some(f64::NAN), Some(f64::INFINITY), Some(f64::NEG_INFINITY)],
        ];
        for training_values in &cases {
            let training = [Column::new("x".into(), training_values.clone())];
            let fitted = Fitted::fit(
                PolynomialDefinition::new(Some(Scalar::from(2_i64)), Some(Scalar::from(true)))?,
                &training,
            )?;
            for prediction_values in &cases {
                let prediction = [Column::new("x".into(), prediction_values.clone())];
                let output = fitted.transform(&prediction)?;
                assert_raw_schema(&output, prediction_values.len(), 2);
                for column in output.columns() {
                    assert_eq!(column.null_count(), prediction_values.len());
                }
            }
            let output = fitted.transform(&[Column::new("x".into(), [3.0])])?;
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
        assert_raw_schema(&output, 2, 3);
        assert_eq!(output[0].f64()?.get(0), Some(f64::MAX));
        assert_eq!(output[1].f64()?.get(0), Some(f64::INFINITY));
        assert_eq!(output[1].f64()?.get(1), Some(f64::INFINITY));
        assert_eq!(output[2].f64()?.get(1), Some(f64::NEG_INFINITY));
        Ok(())
    }

    #[test]
    fn raw_fit_and_prediction_require_exactly_one_input() -> Result<(), TransformError> {
        let definition =
            PolynomialDefinition::new(Some(Scalar::from(2_i64)), Some(Scalar::from(true)))?;
        let inputs = [
            Column::new("x".into(), [1.0]),
            Column::new("z".into(), [2.0]),
        ];
        let fitted = Fitted::fit(definition.clone(), &inputs[..1])?;
        for invalid in [&inputs[..0], &inputs[..]] {
            for result in [
                definition.fit(invalid).map(|_| ()),
                fitted.transform(invalid).map(|_| ()),
            ] {
                assert!(matches!(result, Err(TransformError::InputCountMismatch {
                    expected: ArgumentCount::Exactly(1), actual,
                }) if actual == invalid.len()));
            }
        }
        Ok(())
    }

    #[test]
    fn raw_fit_rejects_nonnumeric_input() -> Result<(), TransformError> {
        let definition =
            PolynomialDefinition::new(Some(Scalar::from(2_i64)), Some(Scalar::from(true)))?;
        for input in [
            Column::new("x".into(), ["a"]),
            Column::new("x".into(), [true]),
        ] {
            assert!(matches!(
                definition.fit(&[input]),
                Err(TransformError::NonNumericInput { .. })
            ));
        }
        Ok(())
    }

    #[test]
    fn raw_prediction_rejects_changed_names_and_dtypes() -> Result<(), TransformError> {
        let training = [Column::new("x".into(), [1.0])];
        let fitted = Fitted::fit(
            PolynomialDefinition::new(Some(Scalar::from(2_i64)), Some(Scalar::from(true)))?,
            &training,
        )?;
        assert!(matches!(
            fitted.transform(&[Column::new("z".into(), [1.0])]),
            Err(TransformError::InputNamesMismatch { expected, actual })
                if expected == vec![PlSmallStr::from_static("x")]
                    && actual == vec![PlSmallStr::from_static("z")]
        ));
        assert!(matches!(
            fitted.transform(&[Column::new("x".into(), [1_i64])]),
            Err(TransformError::InputTypeMismatch { name, expected, actual })
                if name.as_str() == "x" && expected == DataType::Float64.to_string()
                    && actual == DataType::Int64.to_string()
        ));
        Ok(())
    }

    #[test]
    fn zero_degree_is_rejected_in_both_modes() {
        for raw in [false, true] {
            assert!(matches!(
                PolynomialDefinition::new(Some(Scalar::from(0_i64)), Some(Scalar::from(raw))),
                Err(TransformError::InvalidParameter(_))
            ));
        }
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

    #[test]
    fn identity_depends_on_degree_and_raw_mode() -> Result<(), TransformError> {
        let definition =
            PolynomialDefinition::new(Some(Scalar::from(3_i64)), Some(Scalar::from(false)))?;
        assert_eq!(
            definition,
            PolynomialDefinition::new(Some(Scalar::from(3_i64)), Some(Scalar::from(false)))?
        );
        assert_ne!(
            definition,
            PolynomialDefinition::new(Some(Scalar::from(2_i64)), Some(Scalar::from(false)))?
        );
        assert_ne!(
            definition,
            PolynomialDefinition::new(Some(Scalar::from(3_i64)), Some(Scalar::from(true)))?
        );
        Ok(())
    }
}
