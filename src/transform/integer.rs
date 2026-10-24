use super::input::{argument_column_name, input_frame};
use super::{ArgumentCount, TransformDefinition, TransformError, call_name};
use polars::frame::UniqueKeepStrategy;
use polars::prelude::{
    DataFrame, DataType, IntoLazy, JoinType, MaintainOrderJoin, PlSmallStr, Series, all_horizontal,
    col,
};

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct IntegerEncodingDefinition;

#[derive(Debug)]
pub(crate) struct IntegerEncodingState {
    names: Vec<PlSmallStr>,
    mapping: DataFrame,
}

impl IntegerEncodingDefinition {
    const CODE_COLUMN: &'static str = "__code";
}

impl TransformDefinition for IntegerEncodingDefinition {
    type State = IntegerEncodingState;

    fn expands(&self) -> bool {
        false
    }

    fn argument_count(&self) -> ArgumentCount {
        ArgumentCount::AtLeast(1)
    }

    fn fit(&self, inputs: &[Series]) -> Result<Self::State, TransformError> {
        let mapping = input_frame(inputs)?
            .lazy()
            .drop_nulls(None)
            .unique_stable(None, UniqueKeepStrategy::First);
        // Prevent encoding nonfinite floats as a category
        let finite = inputs
            .iter()
            .enumerate()
            .filter(|(_, input)| input.dtype().is_float())
            .map(|(index, _)| col(argument_column_name(index)).is_finite())
            .collect::<Vec<_>>();
        let mapping = if finite.is_empty() {
            mapping
        } else {
            mapping.filter(all_horizontal(finite)?)
        };

        Ok(IntegerEncodingState {
            names: inputs.iter().map(|input| input.name().clone()).collect(),
            mapping: mapping
                .with_row_index(Self::CODE_COLUMN, None)
                .with_columns([col(Self::CODE_COLUMN).strict_cast(DataType::UInt32)])
                .collect()?,
        })
    }

    fn transform(
        &self,
        state: &Self::State,
        inputs: &[Series],
    ) -> Result<DataFrame, TransformError> {
        let mapping = &state.mapping;
        if !inputs
            .iter()
            .map(|input| input.name())
            .eq(state.names.iter())
        {
            return Err(TransformError::InvalidParameter(
                "input column names differ from the fitted encoding".into(),
            ));
        }

        let input_frame = input_frame(inputs)?;
        for column in input_frame.columns() {
            if column.dtype() != mapping.column(column.name())?.dtype() {
                return Err(TransformError::InputTypeMismatch);
            }
        }
        let keys = input_frame
            .columns()
            .iter()
            .map(|column| col(column.name().clone()))
            .collect::<Vec<_>>();

        let output_name = call_name("F", state.names.iter().map(|name| name.as_str()));

        Ok(input_frame
            .lazy()
            .join_builder()
            .with(mapping.clone().lazy())
            .how(JoinType::Left)
            .on(keys)
            .join_nulls(false)
            .maintain_order(MaintainOrderJoin::Left)
            .finish()
            .select([col(IntegerEncodingDefinition::CODE_COLUMN).alias(output_name)])
            .collect()?)
    }
}

#[cfg(test)]
mod tests {
    use super::super::input::argument_column_name;
    use super::*;
    use crate::fit::{Fit, Fitted};
    use polars::error::PolarsError;
    use polars::prelude::NamedFrom;

    fn assert_integer_output(actual: &DataFrame, expected: &[Option<u32>]) {
        assert_eq!(actual.shape(), (expected.len(), 1));
        let column = &actual.columns()[0];
        assert_eq!(column.dtype(), &DataType::UInt32);
        let values = column
            .u32()
            .expect("UInt32 output column")
            .iter()
            .collect::<Vec<_>>();
        assert_eq!(values, expected);
    }

    #[test]
    fn integer_encoding_requires_fit() {
        let transform = Fit::Unfitted(IntegerEncodingDefinition);
        let input = [Series::new("f1".into(), ["a", "b"])];
        assert!(matches!(
            transform.transform(&input),
            Err(TransformError::NotFitted)
        ));
    }

    #[test]
    fn integer_definition_produces_independent_fits() -> Result<(), TransformError> {
        let definition = IntegerEncodingDefinition;
        let first_input = [Series::new("group".into(), ["a", "b"])];
        let second_input = [Series::new("group".into(), ["b", "a"])];
        let first = definition.fit(&first_input)?;
        let second = definition.fit(&second_input)?;

        assert_integer_output(
            &definition.transform(&first, &first_input)?,
            &[Some(0), Some(1)],
        );
        assert_integer_output(
            &definition.transform(&second, &first_input)?,
            &[Some(1), Some(0)],
        );
        assert_integer_output(
            &definition.transform(&first, &first_input)?,
            &[Some(0), Some(1)],
        );

        Ok(())
    }

    #[test]
    fn integer_encoding_suorts_multiple_inputs() -> Result<(), TransformError> {
        let input = [
            Series::new("f1".into(), ["a", "a", "a", "b", "a"]),
            Series::new("f2".into(), [1_i32, 1, 2, 1, 1]),
            Series::new("f3".into(), [true, false, true, true, true]),
        ];
        let cases = [
            (1, [Some(0), Some(0), Some(0), Some(1), Some(0)]),
            (2, [Some(0), Some(0), Some(1), Some(2), Some(0)]),
            (3, [Some(0), Some(1), Some(2), Some(3), Some(0)]),
        ];

        for (count, expected) in cases {
            let definition = IntegerEncodingDefinition;
            assert!(!definition.expands());
            assert_eq!(definition.argument_count(), ArgumentCount::AtLeast(1));
            let transform = definition.fit(&input[..count])?;
            assert_integer_output(
                &definition.transform(&transform, &input[..count])?,
                &expected,
            );
        }
        Ok(())
    }

    #[test]
    fn integer_encoding_reuses_tuple_codes_and_preserves_row_order() -> Result<(), TransformError> {
        let training = [
            Series::new("f1".into(), ["b", "a", "b", "c", "a"]),
            Series::new("f2".into(), [1_i32, 1, 2, 1, 1]),
        ];
        let input = [
            Series::new("f1".into(), ["c", "b", "a", "b", "a"]),
            Series::new("f2".into(), [1_i32, 2, 1, 1, 2]),
        ];
        let definition = IntegerEncodingDefinition;
        let transform = definition.fit(&training)?;
        let fitted_names = transform.names.clone();
        let fitted_mapping = transform.mapping.clone();

        for _ in 0..2 {
            let output = definition.transform(&transform, &input)?;
            // Codes are returned in order input data
            assert_integer_output(&output, &[Some(3), Some(2), Some(1), Some(0), None]);
            assert_eq!(output.columns()[0].name().as_str(), "F(f1, f2)");
            assert_eq!(transform.names, fitted_names);
            assert!(transform.mapping.equals_missing(&fitted_mapping));
        }
        Ok(())
    }

    #[test]
    fn integer_encoding_preserves_nulls_and_marks_unseen_tuples_null() -> Result<(), TransformError>
    {
        let training = [
            Series::new(
                "f1".into(),
                [Some("a"), None, Some("b"), Some("a"), Some("b")],
            ),
            Series::new("f2".into(), [Some(1_i32), Some(1), None, None, Some(2)]),
        ];
        let input = [
            Series::new(
                "f1".into(),
                [Some("a"), Some("b"), Some("c"), None, Some("a"), Some("b")],
            ),
            Series::new(
                "f2".into(),
                [Some(1_i32), Some(2), Some(1), Some(1), None, None],
            ),
        ];
        let definition = IntegerEncodingDefinition;
        let transform = definition.fit(&training)?;
        assert_integer_output(
            &definition.transform(&transform, &input)?,
            &[Some(0), Some(1), None, None, None, None],
        );
        Ok(())
    }

    #[test]
    fn integer_encoding_excludes_nonfinite_keys() -> Result<(), TransformError> {
        let input = Series::new(
            "group".into(),
            [
                Some(2.0_f64),
                Some(f64::NAN),
                Some(f64::INFINITY),
                Some(1.0),
                Some(f64::NEG_INFINITY),
                Some(2.0),
                None,
            ],
        );
        for dtype in [DataType::Float32, DataType::Float64] {
            let inputs = [input.cast(&dtype)?];
            let definition = IntegerEncodingDefinition;
            let state = definition.fit(&inputs)?;
            assert_integer_output(
                &definition.transform(&state, &inputs)?,
                &[Some(0), None, None, Some(1), None, Some(0), None],
            );
        }
        Ok(())
    }

    #[test]
    fn integer_encoding_excludes_tuples_with_any_nonfinite_component() -> Result<(), TransformError>
    {
        for dtype in [DataType::Float32, DataType::Float64] {
            let inputs = [
                Series::new(
                    "first".into(),
                    [
                        2.0,
                        f64::NAN,
                        2.0,
                        f64::INFINITY,
                        1.0,
                        f64::NEG_INFINITY,
                        2.0,
                        2.0,
                        2.0,
                        2.0,
                    ],
                )
                .cast(&dtype)?,
                Series::new(
                    "second".into(),
                    [
                        3.0,
                        3.0,
                        f64::NAN,
                        3.0,
                        4.0,
                        3.0,
                        f64::INFINITY,
                        f64::NEG_INFINITY,
                        3.0,
                        4.0,
                    ],
                )
                .cast(&dtype)?,
            ];
            let definition = IntegerEncodingDefinition;
            let state = definition.fit(&inputs)?;
            assert_integer_output(
                &definition.transform(&state, &inputs)?,
                &[
                    Some(0),
                    None,
                    None,
                    None,
                    Some(1),
                    None,
                    None,
                    None,
                    Some(0),
                    Some(2),
                ],
            );
        }
        Ok(())
    }

    #[test]
    fn integer_encoding_handles_all_nonfinite_training_data() -> Result<(), TransformError> {
        for dtype in [DataType::Float32, DataType::Float64] {
            let training =
                [
                    Series::new("group".into(), [f64::NAN, f64::INFINITY, f64::NEG_INFINITY])
                        .cast(&dtype)?,
                ];
            let inputs = [Series::new(
                "group".into(),
                [
                    Some(1.0),
                    Some(f64::NAN),
                    Some(f64::INFINITY),
                    Some(f64::NEG_INFINITY),
                    None,
                ],
            )
            .cast(&dtype)?];
            let definition = IntegerEncodingDefinition;
            let state = definition.fit(&training)?;
            assert_integer_output(
                &definition.transform(&state, &inputs)?,
                &[None, None, None, None, None],
            );
        }
        Ok(())
    }

    #[test]
    fn integer_encoding_rejects_changed_names_order_or_count() -> Result<(), TransformError> {
        let training = [
            Series::new("f1".into(), ["a", "b"]),
            Series::new("f2".into(), [1_i32, 2]),
        ];
        let definition = IntegerEncodingDefinition;
        let transform = definition.fit(&training)?;
        let cases = [
            vec![Series::new("other".into(), ["a", "b"]), training[1].clone()],
            vec![training[1].clone(), training[0].clone()],
            vec![training[0].clone()],
            vec![],
            vec![
                training[0].clone(),
                training[1].clone(),
                Series::new("f3".into(), [true, false]),
            ],
        ];

        for input in cases {
            assert!(
                matches!(
                    definition.transform(&transform, &input),
                    Err(TransformError::InvalidParameter(_))
                ),
                "{input:?}"
            );
        }
        Ok(())
    }

    #[test]
    fn integer_encoding_rejects_changed_input_dtypes() -> Result<(), TransformError> {
        let training = [
            Series::new("f1".into(), [1_i32, 2]),
            Series::new("f2".into(), ["a", "b"]),
        ];
        let definition = IntegerEncodingDefinition;
        let transform = definition.fit(&training)?;
        let cases = [
            [Series::new("f1".into(), [1_i64, 2]), training[1].clone()],
            [training[0].clone(), Series::new("f2".into(), [1_i32, 2])],
        ];

        for input in cases {
            assert!(matches!(
                definition.transform(&transform, &input),
                Err(TransformError::InputTypeMismatch)
            ));
        }
        Ok(())
    }

    #[test]
    fn integer_encoding_accepts_repeated_arguments_and_reserved_input_names()
    -> Result<(), TransformError> {
        let column = Series::new(
            IntegerEncodingDefinition::CODE_COLUMN.into(),
            ["a", "b", "a"],
        );
        let cases = [
            (
                [column.clone(), column.clone()],
                [Some(0), Some(1), Some(0)],
            ),
            (
                [column, Series::new(argument_column_name(0), [1_i32, 1, 2])],
                [Some(0), Some(1), Some(2)],
            ),
        ];

        for (input, expected) in cases {
            let definition = IntegerEncodingDefinition;
            let transform = definition.fit(&input)?;
            assert_integer_output(&definition.transform(&transform, &input)?, &expected);
        }
        Ok(())
    }

    #[test]
    fn integer_encoding_handles_empty_and_all_null_training_data() -> Result<(), TransformError> {
        let empty = [Series::new("f1".into(), Vec::<i32>::new())];
        let all_null = [Series::new("f1".into(), [None::<i32>, None])];
        let known_dtype = [Series::new("f1".into(), [1_i32, 2])];

        for training in [empty, all_null] {
            let definition = IntegerEncodingDefinition;
            let transform = definition.fit(&training)?;
            assert_integer_output(
                &definition.transform(&transform, &training)?,
                &vec![None; training[0].len()],
            );
            assert_integer_output(
                &definition.transform(&transform, &known_dtype)?,
                &[None, None],
            );
        }
        Ok(())
    }

    #[test]
    fn integer_encoding_refit_replaces_mapping_and_input_names() -> Result<(), TransformError> {
        let first = [Series::new("f1".into(), ["a", "b"])];
        let second = [Series::new("other".into(), ["c", "a"])];
        let input = [Series::new("other".into(), ["b", "c", "a"])];
        let transform = Fit::Fitted(Fitted::fit(IntegerEncodingDefinition, &first)?);
        let transform = transform.fit(&second)?;

        assert_integer_output(&transform.transform(&input)?, &[None, Some(0), Some(1)]);
        assert!(matches!(
            transform.transform(&first),
            Err(TransformError::InvalidParameter(_))
        ));
        Ok(())
    }

    #[test]
    fn integer_encoding_rejects_invalid_fit_inputs_without_replacing_state()
    -> Result<(), TransformError> {
        let training = [Series::new("f1".into(), ["a", "b"])];
        let transform = Fit::Fitted(Fitted::fit(IntegerEncodingDefinition, &training)?);
        let original = transform.transform(&training)?;
        assert!(matches!(
            transform.fit(&[]),
            Err(TransformError::InvalidParameter(_))
        ));

        let unequal_lengths = [training[0].clone(), Series::new("f2".into(), [1_i32])];
        assert!(matches!(
            transform.fit(&unequal_lengths),
            Err(TransformError::Polars(PolarsError::ShapeMismatch(_)))
        ));
        assert!(transform.transform(&training)?.equals_missing(&original));
        assert_integer_output(&transform.transform(&training)?, &[Some(0), Some(1)]);
        Ok(())
    }
}
