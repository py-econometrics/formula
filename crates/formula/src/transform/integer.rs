use super::input::{PreparedInput, argument_column_name};
use super::{ArgumentCount, TransformDefinition, TransformError};
use crate::fit::Definition;
use crate::formula::{ComponentKey, MaterializedColumns};
use polars::frame::UniqueKeepStrategy;
use polars::prelude::{
    Column, DataFrame, DataType, Field, IntoLazy, IntoSeries, JoinType, MaintainOrderJoin, NULL,
    all_horizontal, col, dtype_cols, lit, when,
};

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct IntegerEncodingDefinition;

#[derive(Debug)]
pub(crate) struct IntegerEncodingState {
    fields: Vec<Field>,
    encoding: IntegerEncoding,
}

#[derive(Debug)]
enum IntegerEncoding {
    Mapping(DataFrame),
    Native(Column),
}

impl IntegerEncodingDefinition {
    const CODE_COLUMN: &'static str = "__code";
}

impl TransformDefinition for IntegerEncodingDefinition {
    fn expands(&self) -> bool {
        false
    }

    fn argument_count(&self) -> ArgumentCount {
        ArgumentCount::AtLeast(1)
    }

    fn deduplicate_inputs(&self) -> bool {
        true
    }

    fn name(&self) -> &'static str {
        "F"
    }
}

impl Definition<IntegerEncodingState> for IntegerEncodingDefinition {
    type Input = PreparedInput;
    type Output = MaterializedColumns<ComponentKey>;
    type Error = TransformError;

    fn validate(
        &self,
        state: &IntegerEncodingState,
        inputs: &PreparedInput,
    ) -> Result<(), TransformError> {
        if inputs.fields.len() != state.fields.len() {
            return Err(TransformError::InputCountMismatch {
                expected: ArgumentCount::Exactly(state.fields.len()),
                actual: inputs.fields.len(),
            });
        }

        if !inputs
            .fields
            .iter()
            .map(|input| input.name())
            .eq(state.fields.iter().map(Field::name))
        {
            return Err(TransformError::InputNamesMismatch {
                expected: state
                    .fields
                    .iter()
                    .map(|field| field.name().clone())
                    .collect(),
                actual: inputs
                    .fields
                    .iter()
                    .map(|input| input.name().clone())
                    .collect(),
            });
        }

        for (input, expected) in inputs.fields.iter().zip(&state.fields) {
            if input.dtype() != expected.dtype() {
                return Err(TransformError::InputTypeMismatch {
                    name: input.name().clone(),
                    expected: expected.dtype().to_string(),
                    actual: input.dtype().to_string(),
                });
            }
        }

        Ok(())
    }

    fn fit(&self, inputs: &PreparedInput) -> Result<IntegerEncodingState, TransformError> {
        if inputs.fields.is_empty() {
            return Err(TransformError::InputCountMismatch {
                expected: self.argument_count(),
                actual: 0,
            });
        }
        // Use native codes for a single column in Polars' Categorical/Enum dtype
        if let [input] = inputs.frame.columns()
            && matches!(
                input.dtype(),
                DataType::Categorical(_, _) | DataType::Enum(_, _)
            )
        {
            return Ok(IntegerEncodingState {
                fields: inputs.fields.clone(),
                encoding: IntegerEncoding::Native(input.drop_nulls().unique_stable()?),
            });
        }

        // Prevent encoding nonfinite floats as a category
        let finite = dtype_cols([DataType::Float32, DataType::Float64])
            .as_selector()
            .as_expr()
            .is_finite();
        let mapping = inputs
            .frame
            .clone()
            .lazy()
            .drop_nulls(None)
            .filter(all_horizontal([finite])?)
            .unique_stable(None, UniqueKeepStrategy::First);

        Ok(IntegerEncodingState {
            fields: inputs.fields.clone(),
            encoding: IntegerEncoding::Mapping(
                mapping
                    .with_row_index(Self::CODE_COLUMN, None)
                    .with_columns([col(Self::CODE_COLUMN).strict_cast(DataType::UInt32)])
                    .collect()?,
            ),
        })
    }

    fn transform(
        &self,
        state: &IntegerEncodingState,
        inputs: &PreparedInput,
    ) -> Result<Self::Output, TransformError> {
        let output_name = state
            .fields
            .iter()
            .map(|field| field.name().as_str())
            .collect::<Vec<_>>()
            .join(":");

        let frame = match &state.encoding {
            IntegerEncoding::Native(levels) => {
                let argument = col(argument_column_name(0));
                // Check if levels were observed during fitting
                let observed = argument
                    .clone()
                    .is_in(lit(levels.implode()?.into_series()), false);
                inputs
                    .frame
                    .clone()
                    .lazy()
                    .select([when(observed)
                        .then(argument.to_physical().cast(DataType::UInt32))
                        .otherwise(lit(NULL))
                        .alias(output_name)])
                    .collect()?
            }
            IntegerEncoding::Mapping(mapping) => {
                // Collect level combinations observed during fitting
                let level_keys = inputs
                    .frame
                    .columns()
                    .iter()
                    .map(|column| col(column.name().clone()))
                    .collect::<Vec<_>>();
                // Join observed levels in input frame on to fitted levels
                inputs
                    .frame
                    .clone()
                    .lazy()
                    .join_builder()
                    .with(mapping.clone().lazy())
                    .how(JoinType::Left)
                    .on(level_keys)
                    .join_nulls(false)
                    .maintain_order(MaintainOrderJoin::Left)
                    .finish()
                    .select([col(IntegerEncodingDefinition::CODE_COLUMN).alias(output_name)])
                    .collect()?
            }
        };

        Ok(MaterializedColumns::single_column(
            frame,
            ComponentKey::None,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::super::input::argument_column_name;
    use super::*;
    use crate::fit::{Fit, Fitted};
    use crate::transform::Transform;
    use polars::error::PolarsError;
    use polars::prelude::{CategoricalPhysical, Categories, FrozenCategories};

    fn prepare(inputs: &[Column]) -> PreparedInput {
        PreparedInput::new(inputs, true).unwrap()
    }

    #[test]
    fn prepared_inputs_are_shared_by_fitting_validation_and_transformation()
    -> Result<(), TransformError> {
        let x = Column::new("x".into(), [1.0, f64::NAN, 2.0]);
        let y = Column::new("y".into(), [10_i32, 20, 30]);
        let repeated = [x.clone(), x.clone(), y.clone()];
        let unique = [x, y];
        let fitted = crate::transform::Transform::F(Fit::Unfitted(IntegerEncodingDefinition))
            .fit(&repeated)?;

        let repeated_output = fitted.transform(&repeated).unwrap();
        let unique_output = fitted.transform(&unique).unwrap();
        assert_eq!(repeated_output.frame().get_column_names(), ["x:y"]);
        assert_integer_output(repeated_output.frame(), &[Some(0), None, Some(1)]);
        assert!(
            repeated_output
                .frame()
                .equals_missing(unique_output.frame())
        );

        let changed = [unique[0].clone(), Column::new("y".into(), [10_i64, 20, 30])];
        assert!(matches!(
            fitted.transform(&changed),
            Err(crate::FitError::Transform(TransformError::InputTypeMismatch { name, .. }))
                if name.as_str() == "y"
        ));
        Ok(())
    }

    #[test]
    fn deduplicated_native_inputs_keep_physical_codes() -> Result<(), TransformError> {
        for dtype in native_dtypes() {
            let x = Column::new("x".into(), ["d", "b", "d"]).cast(&dtype)?;
            let inputs = [x.clone(), x.clone()];
            let fitted = crate::transform::Transform::F(Fit::Unfitted(IntegerEncodingDefinition))
                .fit(&inputs)?;
            let output = fitted.transform(&inputs).unwrap();
            assert_eq!(output.frame().get_column_names(), ["x"]);
            assert_integer_output(output.frame(), &[Some(3), Some(1), Some(3)]);
            assert!(
                output
                    .frame()
                    .equals_missing(fitted.transform(&[x]).unwrap().frame())
            );
        }
        Ok(())
    }

    fn native_dtypes() -> Vec<DataType> {
        let categorical = DataType::from_categories(Categories::random(
            "formula-tests".into(),
            CategoricalPhysical::U32,
        ));
        for label in ["a", "b", "c", "d"] {
            categorical
                .cat_mapping()
                .unwrap()
                .insert_cat(label)
                .unwrap();
        }
        vec![
            categorical,
            DataType::from_frozen_categories(FrozenCategories::new(["a", "b", "c", "d"]).unwrap()),
        ]
    }

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
        let transform = Transform::F(Fit::Unfitted(IntegerEncodingDefinition));
        let input = [Column::new("f1".into(), ["a", "b"])];
        assert!(matches!(
            transform.transform(&input),
            Err(crate::FitError::NotFitted)
        ));
    }

    #[test]
    fn integer_definition_produces_independent_fits() -> Result<(), TransformError> {
        let definition = IntegerEncodingDefinition;
        let first_input = prepare(&[Column::new("group".into(), ["a", "b"])]);
        let second_input = prepare(&[Column::new("group".into(), ["b", "a"])]);
        let first = definition.fit(&first_input)?;
        let second = definition.fit(&second_input)?;

        assert_integer_output(
            definition.transform(&first, &first_input)?.frame(),
            &[Some(0), Some(1)],
        );
        assert_integer_output(
            definition.transform(&second, &first_input)?.frame(),
            &[Some(1), Some(0)],
        );
        assert_integer_output(
            definition.transform(&first, &first_input)?.frame(),
            &[Some(0), Some(1)],
        );

        Ok(())
    }

    #[test]
    fn integer_encoding_supports_multiple_inputs() -> Result<(), TransformError> {
        let input = [
            Column::new("f1".into(), ["a", "a", "a", "b", "a"]),
            Column::new("f2".into(), [1_i32, 1, 2, 1, 1]),
            Column::new("f3".into(), [true, false, true, true, true]),
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
            let transform = definition.fit(&prepare(&input[..count]))?;
            assert_integer_output(
                definition
                    .transform(&transform, &prepare(&input[..count]))?
                    .frame(),
                &expected,
            );
        }
        Ok(())
    }

    #[test]
    fn integer_encoding_reuses_tuple_codes_and_preserves_row_order() -> Result<(), TransformError> {
        let training = prepare(&[
            Column::new("f1".into(), ["b", "a", "b", "c", "a"]),
            Column::new("f2".into(), [1_i32, 1, 2, 1, 1]),
        ]);
        let input = prepare(&[
            Column::new("f1".into(), ["c", "b", "a", "b", "a"]),
            Column::new("f2".into(), [1_i32, 2, 1, 1, 2]),
        ]);
        let definition = IntegerEncodingDefinition;
        let transform = definition.fit(&training)?;
        let fitted_fields = transform.fields.clone();
        let IntegerEncoding::Mapping(mapping) = &transform.encoding else {
            panic!("expected combination mapping");
        };
        let fitted_mapping = mapping.clone();

        for _ in 0..2 {
            let output = definition.transform(&transform, &input)?;
            // Codes are returned in order input data
            assert_integer_output(output.frame(), &[Some(3), Some(2), Some(1), Some(0), None]);
            assert_eq!(output.frame().columns()[0].name().as_str(), "f1:f2");
            assert_eq!(transform.fields, fitted_fields);
            assert!(mapping.equals_missing(&fitted_mapping));
        }
        Ok(())
    }

    #[test]
    fn native_encoding_preserves_codes_and_masks_unobserved_levels() -> Result<(), TransformError> {
        for dtype in native_dtypes() {
            let training =
                prepare(&[
                    Column::new("x".into(), [Some("d"), None, Some("b"), Some("d")])
                        .cast(&dtype)?,
                ]);
            let fitted = Fitted::fit(IntegerEncodingDefinition, &training)?;
            let IntegerEncoding::Native(levels) = &fitted.state.encoding else {
                panic!("expected native encoding");
            };
            assert_eq!(levels.len(), 2);
            assert_eq!(levels.null_count(), 0);

            let input = prepare(&[Column::new(
                "x".into(),
                [Some("b"), Some("a"), None, Some("d"), Some("c"), Some("b")],
            )
            .cast(&dtype)?]);
            let output = fitted.transform(&input)?;
            assert_integer_output(
                output.frame(),
                &[Some(1), None, None, Some(3), None, Some(1)],
            );
            assert_eq!(output.frame().columns()[0].name().as_str(), "x");
        }
        Ok(())
    }

    #[test]
    fn native_encoding_handles_empty_and_all_null_inputs() -> Result<(), TransformError> {
        for dtype in native_dtypes() {
            for values in [vec![], vec![None::<&str>, None]] {
                let training = prepare(&[Column::new("x".into(), values).cast(&dtype)?]);
                let fitted = Fitted::fit(IntegerEncodingDefinition, &training)?;
                let input = prepare(&[Column::new("x".into(), [Some("a"), None]).cast(&dtype)?]);
                assert_integer_output(fitted.transform(&input)?.frame(), &[None, None]);
            }

            let training = [Column::new("x".into(), ["a"]).cast(&dtype)?];
            let prepared_training = prepare(&training);
            let fitted = Fitted::fit(IntegerEncodingDefinition, &prepared_training)?;
            let empty = prepare(&[training[0].slice(0, 0)]);
            assert_integer_output(fitted.transform(&empty)?.frame(), &[]);
        }
        Ok(())
    }

    #[test]
    fn native_encoding_does_not_learn_levels_from_dictionary_growth() -> Result<(), TransformError>
    {
        let dtype = native_dtypes().remove(0);
        let training = prepare(&[Column::new("x".into(), ["b"]).cast(&dtype)?]);
        let fitted = Fitted::fit(IntegerEncodingDefinition, &training)?;
        let input = prepare(&[Column::new("x".into(), ["new", "b", "a"]).cast(&dtype)?]);

        for _ in 0..2 {
            assert_integer_output(fitted.transform(&input)?.frame(), &[None, Some(1), None]);
        }
        Ok(())
    }

    #[test]
    fn native_encoding_rejects_incompatible_dictionaries() -> Result<(), TransformError> {
        for (dtype, incompatible) in native_dtypes().into_iter().zip([
            DataType::from_categories(Categories::random(
                "formula-tests".into(),
                CategoricalPhysical::U32,
            )),
            DataType::from_frozen_categories(FrozenCategories::new(["b", "a", "c", "d"])?),
        ]) {
            let training = prepare(&[Column::new("x".into(), ["a", "b"]).cast(&dtype)?]);
            let fitted = Fitted::fit(IntegerEncodingDefinition, &training)?;
            let input = prepare(&[Column::new("x".into(), ["a", "b"]).cast(&incompatible)?]);
            assert!(matches!(
                fitted.transform(&input),
                Err(TransformError::InputTypeMismatch { .. })
            ));
        }
        Ok(())
    }

    #[test]
    fn enum_encoding_widens_physical_codes_to_uint32() -> Result<(), TransformError> {
        for count in [256, 65_536] {
            let labels = (0..count).map(|i| format!("level{i}")).collect::<Vec<_>>();
            let dtype = DataType::from_frozen_categories(FrozenCategories::new(
                labels.iter().map(String::as_str),
            )?);
            let training =
                prepare(&[Column::new("x".into(), [labels[count - 1].as_str()]).cast(&dtype)?]);
            let fitted = Fitted::fit(IntegerEncodingDefinition, &training)?;
            assert_integer_output(
                fitted.transform(&training)?.frame(),
                &[Some((count - 1) as u32)],
            );
        }
        Ok(())
    }

    #[test]
    fn native_columns_in_combinations_use_observed_tuple_codes() -> Result<(), TransformError> {
        for dtype in native_dtypes() {
            let training = prepare(&[
                Column::new("x".into(), ["d", "b", "d"]).cast(&dtype)?,
                Column::new("n".into(), [1_i32, 1, 2]),
            ]);
            let fitted = Fitted::fit(IntegerEncodingDefinition, &training)?;
            let input = prepare(&[
                Column::new(
                    "x".into(),
                    [Some("d"), Some("b"), Some("a"), None, Some("b")],
                )
                .cast(&dtype)?,
                Column::new("n".into(), [2_i32, 2, 1, 1, 1]),
            ]);
            assert_integer_output(
                fitted.transform(&input)?.frame(),
                &[Some(2), None, None, None, Some(1)],
            );
        }
        Ok(())
    }

    #[test]
    fn combinations_support_two_native_columns() -> Result<(), TransformError> {
        for left_dtype in native_dtypes() {
            for right_dtype in native_dtypes() {
                let training = prepare(&[
                    Column::new("x".into(), ["d", "b", "d"]).cast(&left_dtype)?,
                    Column::new("z".into(), ["a", "a", "c"]).cast(&right_dtype)?,
                ]);
                let fitted = Fitted::fit(IntegerEncodingDefinition, &training)?;
                let input = prepare(&[
                    Column::new("x".into(), ["d", "b", "a", "b"]).cast(&left_dtype)?,
                    Column::new("z".into(), ["c", "c", "a", "a"]).cast(&right_dtype)?,
                ]);
                assert_integer_output(
                    fitted.transform(&input)?.frame(),
                    &[Some(2), None, None, Some(1)],
                );
            }
        }
        Ok(())
    }

    #[test]
    fn integer_encoding_preserves_nulls_and_marks_unseen_tuples_null() -> Result<(), TransformError>
    {
        let training = prepare(&[
            Column::new(
                "f1".into(),
                [Some("a"), None, Some("b"), Some("a"), Some("b")],
            ),
            Column::new("f2".into(), [Some(1_i32), Some(1), None, None, Some(2)]),
        ]);
        let input = prepare(&[
            Column::new(
                "f1".into(),
                [Some("a"), Some("b"), Some("c"), None, Some("a"), Some("b")],
            ),
            Column::new(
                "f2".into(),
                [Some(1_i32), Some(2), Some(1), Some(1), None, None],
            ),
        ]);
        let definition = IntegerEncodingDefinition;
        let transform = definition.fit(&training)?;
        assert_integer_output(
            definition.transform(&transform, &input)?.frame(),
            &[Some(0), Some(1), None, None, None, None],
        );
        Ok(())
    }

    #[test]
    fn integer_encoding_excludes_nonfinite_keys() -> Result<(), TransformError> {
        let input = Column::new(
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
            let inputs = prepare(&[input.cast(&dtype)?]);
            let definition = IntegerEncodingDefinition;
            let state = definition.fit(&inputs)?;
            assert_integer_output(
                definition.transform(&state, &inputs)?.frame(),
                &[Some(0), None, None, Some(1), None, Some(0), None],
            );
        }
        Ok(())
    }

    #[test]
    fn integer_encoding_excludes_tuples_with_any_nonfinite_component() -> Result<(), TransformError>
    {
        for dtype in [DataType::Float32, DataType::Float64] {
            let inputs = prepare(&[
                Column::new(
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
                Column::new(
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
            ]);
            let definition = IntegerEncodingDefinition;
            let state = definition.fit(&inputs)?;
            assert_integer_output(
                definition.transform(&state, &inputs)?.frame(),
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
            let training = prepare(&[Column::new(
                "group".into(),
                [f64::NAN, f64::INFINITY, f64::NEG_INFINITY],
            )
            .cast(&dtype)?]);
            let inputs = prepare(&[Column::new(
                "group".into(),
                [
                    Some(1.0),
                    Some(f64::NAN),
                    Some(f64::INFINITY),
                    Some(f64::NEG_INFINITY),
                    None,
                ],
            )
            .cast(&dtype)?]);
            let definition = IntegerEncodingDefinition;
            let state = definition.fit(&training)?;
            assert_integer_output(
                definition.transform(&state, &inputs)?.frame(),
                &[None, None, None, None, None],
            );
        }
        Ok(())
    }

    #[test]
    fn integer_encoding_rejects_changed_names_or_order() -> Result<(), TransformError> {
        let training = [
            Column::new("f1".into(), ["a", "b"]),
            Column::new("f2".into(), [1_i32, 2]),
        ];
        let prepared_training = prepare(&training);
        let definition = IntegerEncodingDefinition;
        let transform = definition.fit(&prepared_training)?;
        let cases = [
            (
                vec![Column::new("other".into(), ["a", "b"]), training[1].clone()],
                "expected input names [\"f1\", \"f2\"], received [\"other\", \"f2\"]",
            ),
            (
                vec![training[1].clone(), training[0].clone()],
                "expected input names [\"f1\", \"f2\"], received [\"f2\", \"f1\"]",
            ),
        ];

        for (input, expected) in cases {
            let prepared_input = prepare(&input);
            let Err(error @ TransformError::InputNamesMismatch { .. }) =
                definition.validate(&transform, &prepared_input)
            else {
                panic!("expected InputNamesMismatch for {input:?}");
            };
            assert_eq!(error.to_string(), expected);
        }
        Ok(())
    }

    #[test]
    fn integer_encoding_rejects_changed_input_count() -> Result<(), TransformError> {
        let training = [
            Column::new("f1".into(), ["a", "b"]),
            Column::new("f2".into(), [1_i32, 2]),
        ];
        let prepared_training = prepare(&training);
        let definition = IntegerEncodingDefinition;
        let transform = definition.fit(&prepared_training)?;
        let cases = [
            vec![training[0].clone()],
            vec![],
            vec![
                training[0].clone(),
                training[1].clone(),
                Column::new("f3".into(), [true, false]),
            ],
        ];

        for input in cases {
            let prepared_input = prepare(&input);
            let Err(error @ TransformError::InputCountMismatch { .. }) =
                definition.validate(&transform, &prepared_input)
            else {
                panic!("expected InputCountMismatch for {input:?}");
            };
            assert_eq!(
                error.to_string(),
                format!(
                    "number of input columns: expected {}, received {}",
                    ArgumentCount::Exactly(2),
                    input.len()
                ),
            );
        }
        Ok(())
    }

    #[test]
    fn integer_encoding_rejects_changed_input_dtypes() -> Result<(), TransformError> {
        let training = [
            Column::new("f1".into(), [1_i32, 2]),
            Column::new("f2".into(), ["a", "b"]),
        ];
        let prepared_training = prepare(&training);
        let definition = IntegerEncodingDefinition;
        let transform = definition.fit(&prepared_training)?;
        let cases = [
            [Column::new("f1".into(), [1_i64, 2]), training[1].clone()],
            [training[0].clone(), Column::new("f2".into(), [1_i32, 2])],
        ];

        for (index, input) in cases.into_iter().enumerate() {
            let prepared_input = prepare(&input);
            let Err(error @ TransformError::InputTypeMismatch { .. }) =
                definition.validate(&transform, &prepared_input)
            else {
                panic!("expected InputTypeMismatch for {input:?}");
            };
            assert_eq!(
                error.to_string(),
                format!(
                    "input `{}`: expected {}, received {}",
                    training[index].name(),
                    training[index].dtype(),
                    input[index].dtype(),
                ),
            );
        }
        Ok(())
    }

    #[test]
    fn integer_encoding_accepts_repeated_arguments_and_reserved_input_names()
    -> Result<(), TransformError> {
        let column = Column::new(
            IntegerEncodingDefinition::CODE_COLUMN.into(),
            ["a", "b", "a"],
        );
        let cases = [
            (
                [column.clone(), column.clone()],
                [Some(0), Some(1), Some(0)],
            ),
            (
                [column, Column::new(argument_column_name(0), [1_i32, 1, 2])],
                [Some(0), Some(1), Some(2)],
            ),
        ];

        for (input, expected) in cases {
            let prepared_input = prepare(&input);
            let definition = IntegerEncodingDefinition;
            let transform = definition.fit(&prepared_input)?;
            assert_integer_output(
                definition.transform(&transform, &prepared_input)?.frame(),
                &expected,
            );
        }
        Ok(())
    }

    #[test]
    fn integer_encoding_handles_empty_and_all_null_training_data() -> Result<(), TransformError> {
        let empty = [Column::new("f1".into(), Vec::<i32>::new())];
        let all_null = [Column::new("f1".into(), [None::<i32>, None])];
        let known_dtype = prepare(&[Column::new("f1".into(), [1_i32, 2])]);

        for training in [empty, all_null] {
            let prepared_training = prepare(&training);
            let definition = IntegerEncodingDefinition;
            let transform = definition.fit(&prepared_training)?;
            assert_integer_output(
                definition
                    .transform(&transform, &prepared_training)?
                    .frame(),
                &vec![None; training[0].len()],
            );
            assert_integer_output(
                definition.transform(&transform, &known_dtype)?.frame(),
                &[None, None],
            );
        }
        Ok(())
    }

    #[test]
    fn integer_encoding_refit_replaces_mapping_and_input_names() -> Result<(), TransformError> {
        let first = [Column::new("f1".into(), ["a", "b"])];
        let second = [Column::new("other".into(), ["c", "a"])];
        let input = [Column::new("other".into(), ["b", "c", "a"])];
        let transform = Transform::F(Fit::Unfitted(IntegerEncodingDefinition)).fit(&first)?;
        let transform = transform.fit(&second)?;

        assert_integer_output(
            transform.transform(&input).unwrap().frame(),
            &[None, Some(0), Some(1)],
        );
        let Err(crate::FitError::Transform(error @ TransformError::InputNamesMismatch { .. })) =
            transform.transform(&first)
        else {
            panic!("expected InputNamesMismatch");
        };
        assert_eq!(
            error.to_string(),
            "expected input names [\"other\"], received [\"f1\"]",
        );
        Ok(())
    }

    #[test]
    fn integer_encoding_rejects_invalid_fit_inputs_without_replacing_state()
    -> Result<(), TransformError> {
        let training = [Column::new("f1".into(), ["a", "b"])];
        let transform = crate::transform::Transform::F(Fit::Unfitted(IntegerEncodingDefinition))
            .fit(&training)?;
        let original = transform.transform(&training).unwrap();
        let Err(error @ TransformError::InputCountMismatch { .. }) = transform.fit(&[]) else {
            panic!("expected InputCountMismatch");
        };
        assert_eq!(
            error.to_string(),
            "number of input columns: expected at least 1, received 0"
        );

        let unequal_lengths = [training[0].clone(), Column::new("f2".into(), [1_i32])];
        assert!(matches!(
            transform.fit(&unequal_lengths),
            Err(TransformError::Polars(PolarsError::ShapeMismatch(_)))
        ));
        assert!(
            transform
                .transform(&training)
                .unwrap()
                .frame()
                .equals_missing(original.frame())
        );
        assert_integer_output(
            transform.transform(&training).unwrap().frame(),
            &[Some(0), Some(1)],
        );
        Ok(())
    }
}
