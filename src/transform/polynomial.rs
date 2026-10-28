use crate::fit::Definition;
use crate::transform::input::{argument_column_name, input_frame, validate_single_input};
use crate::transform::{TransformDefinition, call_name, validate_numeric_input};
use crate::{ArgumentCount, TransformError};
use polars::datatypes::DataType;
use polars::frame::DataFrame;
use polars::prelude::{Column, Expr, Field, IntoLazy, NULL, PlSmallStr, col, lit, when};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PolynomialDefinition {
    degree: usize,
    raw: bool,
}

impl PolynomialDefinition {
    pub(crate) fn new(degree: usize, raw: bool) -> Result<Self, TransformError> {
        if degree == 0 {
            return Err(TransformError::InvalidParameter(
                "`degree` must be a positive integer".into(),
            ));
        }
        Ok(Self { degree, raw })
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
}

impl PolynomialBasis {
    fn expressions(&self, input: Expr, degree: usize) -> Vec<Expr> {
        match self {
            Self::Raw => (1..=degree)
                .map(|exponent| input.clone().pow(lit(exponent as u64)))
                .collect(),
        }
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

        if !self.raw {
            return Err(TransformError::InvalidParameter(
                "orthogonal polynomial encoding is not implemented yet".into(),
            ));
        }

        Ok(PolynomialState {
            field: input.field().into_owned(),
            basis: PolynomialBasis::Raw,
        })
    }

    fn transform(
        &self,
        state: &PolynomialState,
        inputs: &[Column],
    ) -> Result<DataFrame, TransformError> {
        let call = self.call_name(&[state.field.name().as_str()]);
        let frame = input_frame(inputs)?;
        let argument = finite_values(col(argument_column_name(0)));

        let expressions = state
            .basis
            .expressions(argument, self.degree)
            .into_iter()
            .enumerate()
            .map(|(index, expression)| expression.alias(format!("{call}[{}]", index + 1)))
            .collect::<Vec<_>>();

        Ok(frame.lazy().select(expressions).collect()?)
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
        let fitted = Fitted::fit(PolynomialDefinition::new(3, true)?, &training)?;
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
        let fitted = Fitted::fit(PolynomialDefinition::new(2, true)?, &inputs)?;
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
            let fitted = Fitted::fit(PolynomialDefinition::new(2, true)?, &training)?;
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
        let fitted = Fitted::fit(PolynomialDefinition::new(3, true)?, &inputs)?;
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
        let definition = PolynomialDefinition::new(2, true)?;
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
        let definition = PolynomialDefinition::new(2, true)?;
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
        let fitted = Fitted::fit(PolynomialDefinition::new(2, true)?, &training)?;
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
                PolynomialDefinition::new(0, raw),
                Err(TransformError::InvalidParameter(_))
            ));
        }
    }

    #[test]
    fn polynomial_expands_exactly_one_argument() -> Result<(), TransformError> {
        for raw in [false, true] {
            let definition = PolynomialDefinition::new(1, raw)?;
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
            let definition = PolynomialDefinition::new(degree, raw)?;
            assert_eq!(definition.call_name(&[input]).as_str(), expected);
        }
        Ok(())
    }

    #[test]
    fn identity_depends_on_degree_and_raw_mode() -> Result<(), TransformError> {
        let definition = PolynomialDefinition::new(3, false)?;
        assert_eq!(definition, PolynomialDefinition::new(3, false)?);
        assert_ne!(definition, PolynomialDefinition::new(2, false)?);
        assert_ne!(definition, PolynomialDefinition::new(3, true)?);
        Ok(())
    }
}
