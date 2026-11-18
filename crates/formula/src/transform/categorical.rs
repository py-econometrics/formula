use super::input::PreparedInput;
use super::{ArgumentCount, ParameterValue, TransformDefinition, TransformError, call_name};
use crate::fit::Definition;
use crate::formula::{ComponentKey, MaterializedColumns};
use crate::rank::Scope;
use polars::chunked_array::cast::CastOptions;
use polars::error::PolarsResult;
use polars::prelude::{
    AnyValue, Column, DataFrame, DataType, IntoLazy, IntoSeries, NULL, PlSmallStr, Scalar,
    SortOptions, col, lit, when,
};
use std::collections::BTreeMap;

/// How a term materialises the levels of a categorical factor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Contrast {
    /// One indicator per level, named `prefix[level]`.
    Full,
    /// Every level but the reference, named `prefix[T.level]` (treatment contrasts).
    Reduced,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct CategoricalDefinition {
    reference: Option<ParameterValue>,
}

impl CategoricalDefinition {
    pub(crate) const OMIT_PARAMETER: &str = "omit";

    pub(crate) fn new(reference: Option<ParameterValue>) -> Self {
        Self { reference }
    }

    fn reference_name(value: &Scalar) -> String {
        match value.as_any_value() {
            AnyValue::String(value) => format!("'{value}'"),
            AnyValue::StringOwned(value) => format!("'{value}'"),
            _ => Self::level_name(value),
        }
    }

    fn output_name(&self, input_name: &str, level: &Scalar) -> PlSmallStr {
        Self::column_name(&self.call_name(&[input_name]), level, Contrast::Full)
    }

    /// Names the indicator column of `level` after `prefix`, such as `C(a)` or a bare `a`.
    pub(crate) fn column_name(prefix: &str, level: &Scalar, contrast: Contrast) -> PlSmallStr {
        let marker = match contrast {
            Contrast::Full => "",
            Contrast::Reduced => "T.",
        };

        format!("{prefix}[{marker}{}]", Self::level_name(level)).into()
    }

    /// Selects indicator columns by their positions in the fitted levels.
    pub(crate) fn code(
        state: &CategoricalState,
        columns: MaterializedColumns<ComponentKey>,
        prefix: &str,
        scope: &Scope,
    ) -> Result<MaterializedColumns<ComponentKey>, TransformError> {
        let contrast = match scope {
            Scope::Reduced => Contrast::Reduced,
            Scope::Full | Scope::Levels(_) => Contrast::Full,
            Scope::Omitted => unreachable!("omitted factors have no columns in this group"),
        };
        let (frame, metadata) = columns.into_parts();
        let retained_levels =
            (0..state.level_count()).filter(|index| !state.omitted_indices().contains(index));
        let (columns, metadata): (Vec<_>, Vec<_>) = retained_levels
            .zip(frame.columns().iter().zip(metadata))
            .filter(|(index, _)| match scope {
                Scope::Reduced => *index != 0,
                Scope::Levels(selected) => selected.contains(index),
                _ => true,
            })
            .map(|(_, (column, key))| {
                let ComponentKey::Scalar(level) = &key else {
                    unreachable!("categorical columns are keyed by their level");
                };
                let name = Self::column_name(prefix, level, contrast);
                (column.clone().with_name(name), key)
            })
            .unzip();

        Ok(MaterializedColumns::new(
            DataFrame::new(frame.height(), columns)?,
            metadata,
        ))
    }

    fn level_name(level: &Scalar) -> String {
        match level.as_any_value() {
            AnyValue::Boolean(true) => "True".into(),
            AnyValue::Boolean(false) => "False".into(),
            AnyValue::Float32(value) => format!("{value:?}"),
            AnyValue::Float64(value) => format!("{value:?}"),
            value => value.str_value().into_owned(),
        }
    }

    /// Converts the reference value to the input dtype.
    ///
    /// String references for categorical and enum inputs are resolved through the
    /// existing dictionary without inserting new levels. Numeric references may
    /// convert between integer widths or between float widths; other dtype
    /// combinations must match exactly. Nonfinite numeric references are rejected.
    ///
    /// Membership in the observed fitting levels is checked separately in `fit`.
    fn convert_reference(reference: &Scalar, dtype: &DataType) -> Result<Scalar, TransformError> {
        // Get the dtype of the reference provided
        let source = reference.dtype();
        // Check compatibility with and convert to dtype
        let reference = match (source, dtype) {
            // Source and target dtype match
            (source, target) if source == target => reference.clone(),
            // Target is Categorical/Enum; check if reference is in observed levels
            (DataType::String, DataType::Categorical(_, mapping) | DataType::Enum(_, mapping)) => {
                let value = reference.as_any_value();
                let label = value.str_value();
                let code = mapping.get_cat(label.as_ref()).ok_or_else(|| {
                    TransformError::InvalidParameter(
                        "`omit` value is not among the fitted levels".into(),
                    )
                })?;

                let value = match dtype {
                    DataType::Categorical(_, _) => {
                        AnyValue::CategoricalOwned(code, mapping.clone())
                    }
                    DataType::Enum(_, _) => AnyValue::EnumOwned(code, mapping.clone()),
                    _ => unreachable!("dtype is Categorical or Enum"),
                };

                Scalar::new(dtype.clone(), value)
            }
            // Reference/target are integers; we cast here compatible integers
            (source, target) if source.is_integer() && target.is_integer() => reference
                .clone()
                .cast_with_options(target, CastOptions::Strict)?,
            // Reference/target are floats; we cast here compatible floats
            (source, target) if source.is_float() && target.is_float() => reference
                .clone()
                .cast_with_options(target, CastOptions::Strict)?,
            // Reference not compatible with target dtype
            _ => {
                return Err(TransformError::InvalidParameter(format!(
                    "`omit` dtype {source:?} is incompatible with input dtype {dtype:?}"
                )));
            }
        };

        if reference.dtype().is_float()
            && !reference.as_any_value().try_extract::<f64>()?.is_finite()
        {
            return Err(TransformError::InvalidParameter(
                "`omit` must be finite after conversion to the input dtype".into(),
            ));
        }

        Ok(reference)
    }
}

impl TransformDefinition for CategoricalDefinition {
    fn expands(&self) -> bool {
        true
    }

    fn argument_count(&self) -> ArgumentCount {
        ArgumentCount::Exactly(1)
    }

    fn name(&self) -> &'static str {
        "C"
    }

    fn parameters(&self) -> BTreeMap<&'static str, ParameterValue> {
        self.reference
            .as_ref()
            .map(|reference| (Self::OMIT_PARAMETER, reference.clone()))
            .into_iter()
            .collect()
    }

    fn call_name(&self, arguments: &[&str]) -> PlSmallStr {
        let reference_argument = self.reference.as_ref().map(|reference| {
            let value = match reference {
                ParameterValue::Scalar(value) => Self::reference_name(value),
                ParameterValue::List(values) => {
                    let names = values.iter().map(Self::reference_name).collect::<Vec<_>>();
                    format!("[{}]", names.join(", "))
                }
            };

            format!("{}={value}", Self::OMIT_PARAMETER)
        });

        call_name(
            self.name(),
            arguments
                .iter()
                .copied()
                .chain(reference_argument.as_deref()),
        )
    }
}

impl Definition<CategoricalState> for CategoricalDefinition {
    type Input = PreparedInput;
    type Output = MaterializedColumns<ComponentKey>;
    type Error = TransformError;

    fn validate(
        &self,
        state: &CategoricalState,
        inputs: &PreparedInput,
    ) -> Result<(), TransformError> {
        let (field, input) = inputs.single()?;

        if input.dtype() != state.levels.dtype() {
            return Err(TransformError::InputTypeMismatch {
                name: field.name().clone(),
                expected: state.levels.dtype().to_string(),
                actual: input.dtype().to_string(),
            });
        }

        Ok(())
    }

    fn fit(&self, inputs: &PreparedInput) -> Result<CategoricalState, TransformError> {
        let (_, input) = inputs.single()?;
        let levels = input.drop_nulls().unique()?;
        // Prevent encoding nonfinite floats as a category
        let levels = if levels.dtype().is_float() {
            levels.filter(&levels.is_finite()?)?
        } else {
            levels
        };
        // Sort for a deterministic order: numbers by value, strings and categoricals
        // by label, and enums by category order.
        let levels = levels.sort(SortOptions::default())?;

        let references: &[Scalar] = match &self.reference {
            None => &[],
            Some(ParameterValue::Scalar(value)) => std::slice::from_ref(value),
            Some(ParameterValue::List(values)) => values.as_slice(),
        };
        let mut reference_indices = Vec::new();

        for reference in references {
            let reference = Self::convert_reference(reference, levels.dtype())?;

            let index = (0..levels.len())
                .find_map(|index| match levels.get(index) {
                    Ok(level) if level == reference.as_any_value() => Some(Ok(index)),
                    Ok(_) => None,
                    Err(error) => Some(Err(error)),
                })
                .transpose()?
                .ok_or_else(|| {
                    TransformError::InvalidParameter(
                        "`omit` value is not among the fitted levels".into(),
                    )
                })?;

            if !reference_indices.contains(&index) {
                reference_indices.push(index);
            }
        }

        Ok(CategoricalState {
            levels,
            reference_indices,
        })
    }

    fn transform(
        &self,
        state: &CategoricalState,
        inputs: &PreparedInput,
    ) -> Result<Self::Output, TransformError> {
        let (field, input) = inputs.single()?;
        let levels = &state.levels;
        let frame = &inputs.frame;

        let argument = col(frame.columns()[0].name().clone());
        let fitted_levels = (0..levels.len())
            .map(|i| {
                Ok(Scalar::new(
                    levels.dtype().clone(),
                    levels.get(i)?.into_static(),
                ))
            })
            .collect::<PolarsResult<Vec<_>>>()?;

        // Direct `or` comparisons are cheaper for small sets;
        // `is_in`'s hashing scales better as the number of observed levels grows.
        let level_was_seen = if levels.len() <= 8 {
            fitted_levels
                .iter()
                .map(|level| argument.clone().eq_missing(lit(level.clone())))
                .reduce(|left, right| left.or(right))
                .unwrap_or(lit(false))
        } else {
            argument
                .clone()
                .is_in(lit(levels.implode()?.into_series()), false)
        };

        // Keep every fitted level except those explicitly omitted.
        let (expressions, metadata): (Vec<_>, Vec<ComponentKey>) = fitted_levels
            .into_iter()
            .enumerate()
            .filter(|(index, _)| !state.reference_indices.contains(index))
            .map(|(_, level)| {
                let column_name = self.output_name(field.name(), &level);

                let expression = when(argument.clone().is_null())
                    .then(lit(NULL))
                    .when(level_was_seen.clone())
                    .then(argument.clone().eq(lit(level.clone())))
                    .otherwise(lit(NULL))
                    .alias(column_name);

                (expression, ComponentKey::Scalar(level))
            })
            .unzip();
        // Preserve input row count even when expression is empty
        let result = if expressions.is_empty() {
            DataFrame::empty_with_height(input.len())
        } else {
            frame.clone().lazy().select(expressions).collect()?
        };

        Ok(MaterializedColumns::new(result, metadata))
    }
}

#[derive(Debug)]
pub(crate) struct CategoricalState {
    levels: Column,
    reference_indices: Vec<usize>,
}

impl CategoricalState {
    pub(crate) fn level_count(&self) -> usize {
        self.levels.len()
    }

    pub(crate) fn omitted_indices(&self) -> &[usize] {
        &self.reference_indices
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fit::{Fit, Fitted};
    use crate::transform::Transform;
    use polars::prelude::{CategoricalPhysical, Categories, FrozenCategories};

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

    fn prepare(inputs: &[Column]) -> PreparedInput {
        PreparedInput::new(inputs, false).unwrap()
    }

    #[test]
    fn native_levels_are_sorted_and_keep_label_names() -> Result<(), TransformError> {
        for dtype in native_dtypes() {
            let training = [Column::new("x".into(), ["d", "b", "d"]).cast(&dtype)?];
            let prepared_training = prepare(&training);
            let fitted = Fitted::fit(CategoricalDefinition::new(None), &prepared_training)?;
            let output = fitted.transform(&prepared_training)?;

            assert_eq!(output.frame().get_column_names(), ["C(x)[b]", "C(x)[d]"]);
            assert_output(
                &fitted.definition,
                &fitted.state,
                &training,
                output.frame(),
                &[
                    &[Some(false), Some(true), Some(false)],
                    &[Some(true), Some(false), Some(true)],
                ],
            );
        }
        Ok(())
    }

    #[test]
    fn levels_sort_by_value_label_or_enum_category_order() -> Result<(), TransformError> {
        let categorical = DataType::from_categories(Categories::random(
            "formula-tests".into(),
            CategoricalPhysical::U32,
        ));
        // Dictionary order differs from label order.
        for label in ["b", "c", "a"] {
            categorical
                .cat_mapping()
                .unwrap()
                .insert_cat(label)
                .unwrap();
        }
        let enumeration =
            DataType::from_frozen_categories(FrozenCategories::new(["c", "a", "b"]).unwrap());
        let labels = Column::new("x".into(), ["b", "a", "c", "a"]);

        for (input, expected) in [
            (
                Column::new("x".into(), [10_i64, 9, 2, 9]),
                ["C(x)[2]", "C(x)[9]", "C(x)[10]"],
            ),
            (
                Column::new("x".into(), [10.5_f64, 9.0, 2.0]),
                ["C(x)[2.0]", "C(x)[9.0]", "C(x)[10.5]"],
            ),
            (
                Column::new("x".into(), ["b", "B", "a"]),
                ["C(x)[B]", "C(x)[a]", "C(x)[b]"],
            ),
            (labels.clone(), ["C(x)[a]", "C(x)[b]", "C(x)[c]"]),
            (
                labels.cast(&categorical)?,
                ["C(x)[a]", "C(x)[b]", "C(x)[c]"],
            ),
            (
                labels.cast(&enumeration)?,
                ["C(x)[c]", "C(x)[a]", "C(x)[b]"],
            ),
        ] {
            let prepared = prepare(&[input]);
            let fitted = Fitted::fit(CategoricalDefinition::new(None), &prepared)?;
            assert_eq!(
                fitted.transform(&prepared)?.frame().get_column_names(),
                expected
            );
        }
        Ok(())
    }

    #[test]
    fn column_names_follow_contrast_with_any_prefix() {
        let level = Scalar::from(PlSmallStr::from_static("q"));
        assert_eq!(
            CategoricalDefinition::column_name("a", &level, Contrast::Full),
            "a[q]"
        );
        assert_eq!(
            CategoricalDefinition::column_name("C(a)", &level, Contrast::Reduced),
            "C(a)[T.q]"
        );
    }

    #[test]
    fn native_unknown_levels_and_nulls_produce_null_indicators() -> Result<(), TransformError> {
        for dtype in native_dtypes() {
            let training = prepare(&[Column::new("x".into(), ["d", "b"]).cast(&dtype)?]);
            let fitted = Fitted::fit(CategoricalDefinition::new(None), &training)?;
            let input = [Column::new(
                "x".into(),
                [Some("b"), Some("a"), None, Some("d"), Some("c")],
            )
            .cast(&dtype)?];
            let prepared_input = prepare(&input);
            assert_output(
                &fitted.definition,
                &fitted.state,
                &input,
                fitted.transform(&prepared_input)?.frame(),
                &[
                    &[Some(true), None, None, Some(false), None],
                    &[Some(false), None, None, Some(true), None],
                ],
            );
        }
        Ok(())
    }

    /// Codes the transform output of `fitted` with a reduced contrast.
    fn reduced(
        fitted: &Fitted<CategoricalDefinition, CategoricalState>,
        input_name: &str,
        output: MaterializedColumns<ComponentKey>,
    ) -> Result<MaterializedColumns<ComponentKey>, TransformError> {
        let prefix = fitted.definition.call_name(&[input_name]);
        CategoricalDefinition::code(&fitted.state, output, &prefix, &Scope::Reduced)
    }

    #[test]
    fn native_reference_is_omitted_without_treating_unknown_levels_as_reference()
    -> Result<(), TransformError> {
        for dtype in native_dtypes() {
            let training = prepare(&[Column::new("x".into(), ["d", "b"]).cast(&dtype)?]);
            let reference = Scalar::from(PlSmallStr::from_static("b"));
            let fitted = Fitted::fit(
                CategoricalDefinition::new(Some(ParameterValue::Scalar(reference))),
                &training,
            )?;
            let input = [
                Column::new("x".into(), [Some("b"), Some("a"), None, Some("d")]).cast(&dtype)?,
            ];
            let prepared_input = prepare(&input);
            let output = fitted.transform(&prepared_input)?;
            assert_eq!(output.frame().get_column_names(), ["C(x, omit='b')[d]"]);
            assert_output(
                &fitted.definition,
                &fitted.state,
                &input,
                output.frame(),
                &[&[Some(false), None, None, Some(true)]],
            );
        }
        Ok(())
    }

    #[test]
    fn reduced_coding_omits_the_first_level_by_default() -> Result<(), TransformError> {
        let input = prepare(&[Column::new("x".into(), ["c", "a", "b"])]);
        let fitted = Fitted::fit(CategoricalDefinition::new(None), &input)?;
        let output = reduced(&fitted, "x", fitted.transform(&input)?)?;
        assert_eq!(
            output.frame().get_column_names(),
            ["C(x)[T.b]", "C(x)[T.c]"]
        );
        assert_eq!(
            output.into_parts().1,
            [
                ComponentKey::Scalar(Scalar::from(PlSmallStr::from_static("b"))),
                ComponentKey::Scalar(Scalar::from(PlSmallStr::from_static("c"))),
            ]
        );
        Ok(())
    }

    #[test]
    fn native_reference_must_be_observed_not_just_declared() -> Result<(), TransformError> {
        for dtype in native_dtypes() {
            let training = prepare(&[Column::new("x".into(), ["d", "b"]).cast(&dtype)?]);
            let reference = Scalar::from(PlSmallStr::from_static("a"));
            let error = Fitted::fit(
                CategoricalDefinition::new(Some(ParameterValue::Scalar(reference))),
                &training,
            )
            .unwrap_err();
            assert!(matches!(error, TransformError::InvalidParameter(_)));
            assert_eq!(
                error.to_string(),
                "`omit` value is not among the fitted levels"
            );
        }
        Ok(())
    }

    #[test]
    fn native_missing_reference_does_not_extend_dictionary() -> Result<(), TransformError> {
        for dtype in native_dtypes() {
            let training = prepare(&[Column::new("x".into(), ["d", "b"]).cast(&dtype)?]);
            let mapping = dtype.cat_mapping().unwrap();
            assert!(mapping.get_cat("missing").is_none());
            let reference = Scalar::from(PlSmallStr::from_static("missing"));
            assert!(matches!(
                Fitted::fit(
                    CategoricalDefinition::new(Some(ParameterValue::Scalar(reference))),
                    &training
                ),
                Err(TransformError::InvalidParameter(_))
            ));
            assert!(mapping.get_cat("missing").is_none());
        }
        Ok(())
    }

    #[test]
    fn categorical_dictionary_growth_preserves_fitted_schema() -> Result<(), TransformError> {
        let dtype = native_dtypes().remove(0);
        let training = prepare(&[Column::new("x".into(), ["d", "b"]).cast(&dtype)?]);
        let fitted = Fitted::fit(CategoricalDefinition::new(None), &training)?;
        let original = fitted.transform(&training)?;
        let input = [Column::new("x".into(), ["new", "b", "a"]).cast(&dtype)?];
        let prepared_input = prepare(&input);

        for _ in 0..2 {
            let output = fitted.transform(&prepared_input)?;
            assert_eq!(
                output.frame().get_column_names(),
                original.frame().get_column_names()
            );
            assert_output(
                &fitted.definition,
                &fitted.state,
                &input,
                output.frame(),
                &[&[None, Some(true), None], &[None, Some(false), None]],
            );
            assert!(
                fitted
                    .transform(&training)?
                    .frame()
                    .equals_missing(original.frame())
            );
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
            let fitted = Fitted::fit(CategoricalDefinition::new(None), &training)?;
            let input = prepare(&[Column::new("x".into(), ["a", "b"]).cast(&incompatible)?]);
            assert!(matches!(
                fitted.transform(&input),
                Err(TransformError::InputTypeMismatch { .. })
            ));
        }
        Ok(())
    }

    #[test]
    fn native_empty_and_all_null_training_preserve_prediction_height() -> Result<(), TransformError>
    {
        for dtype in native_dtypes() {
            for values in [vec![], vec![None::<&str>, None]] {
                let training = prepare(&[Column::new("x".into(), values).cast(&dtype)?]);
                let fitted = Fitted::fit(CategoricalDefinition::new(None), &training)?;
                let input =
                    prepare(&[Column::new("x".into(), [Some("a"), None, Some("b")]).cast(&dtype)?]);
                assert_eq!(fitted.transform(&input)?.frame().shape(), (3, 0));
            }
        }
        Ok(())
    }

    #[test]
    fn native_empty_and_all_null_predictions_preserve_fitted_schema() -> Result<(), TransformError>
    {
        for dtype in native_dtypes() {
            let training = prepare(&[Column::new("x".into(), ["b", "d"]).cast(&dtype)?]);
            let fitted = Fitted::fit(CategoricalDefinition::new(None), &training)?;
            let original = fitted.transform(&training)?;
            for values in [vec![], vec![None::<&str>, None]] {
                let height = values.len();
                let input = prepare(&[Column::new("x".into(), values).cast(&dtype)?]);
                let output = fitted.transform(&input)?;
                assert_eq!(output.frame().shape(), (height, 2));
                assert_eq!(
                    output.frame().get_column_names(),
                    original.frame().get_column_names()
                );
                for column in output.frame().columns() {
                    assert_eq!(column.dtype(), &DataType::Boolean);
                    assert_eq!(column.null_count(), height);
                }
            }
        }
        Ok(())
    }

    #[test]
    fn native_reference_only_encoding_preserves_prediction_height() -> Result<(), TransformError> {
        for dtype in native_dtypes() {
            let training = prepare(&[Column::new("x".into(), ["b", "b"]).cast(&dtype)?]);
            let reference = Scalar::from(PlSmallStr::from_static("b"));
            let fitted = Fitted::fit(
                CategoricalDefinition::new(Some(ParameterValue::Scalar(reference))),
                &training,
            )?;
            let input =
                prepare(&[Column::new("x".into(), [Some("b"), Some("a"), None]).cast(&dtype)?]);
            assert_eq!(fitted.transform(&input)?.frame().shape(), (3, 0));
        }
        Ok(())
    }

    #[test]
    fn reference_only_encoding_preserves_input_height() -> Result<(), TransformError> {
        let inputs = prepare(&[Column::new("group".into(), [1_i32, 1, 1])]);
        let definition =
            CategoricalDefinition::new(Some(ParameterValue::Scalar(Scalar::from(1_i64))));
        let transform = definition.fit(&inputs)?;
        assert_eq!(
            definition.transform(&transform, &inputs)?.frame().shape(),
            (3, 0)
        );
        Ok(())
    }

    #[test]
    fn empty_fitted_levels_preserve_nonempty_input_height() -> Result<(), TransformError> {
        let inputs = [Column::new("group".into(), [7_i32, 8])];
        for training in [
            Column::new("group".into(), Vec::<i32>::new()),
            Column::new("group".into(), [None::<i32>, None]),
            Column::new("group".into(), [f32::NAN, f32::INFINITY, f32::NEG_INFINITY]),
            Column::new("group".into(), [f64::NAN, f64::INFINITY, f64::NEG_INFINITY]),
        ] {
            let inputs = prepare(&[inputs[0].cast(training.dtype())?]);
            let definition = CategoricalDefinition::new(None);
            let transform = definition.fit(&prepare(&[training]))?;
            assert_eq!(
                definition.transform(&transform, &inputs)?.frame().shape(),
                (2, 0)
            );
        }
        Ok(())
    }

    #[test]
    fn categorical_excludes_nonfinite_levels() -> Result<(), TransformError> {
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
            let inputs = [input.cast(&dtype)?];
            let prepared_inputs = prepare(&inputs);
            let definition = CategoricalDefinition::new(None);
            let state = definition.fit(&prepared_inputs)?;
            let levels = state.levels.cast(&DataType::Float64)?;
            assert_eq!(
                levels.f64()?.iter().collect::<Vec<_>>(),
                vec![Some(1.0), Some(2.0)]
            );
            assert_eq!(state.levels.dtype(), &dtype);
            assert_output(
                &definition,
                &state,
                &inputs,
                definition.transform(&state, &prepared_inputs)?.frame(),
                &[
                    &[Some(false), None, None, Some(true), None, Some(false), None],
                    &[Some(true), None, None, Some(false), None, Some(true), None],
                ],
            );
        }
        Ok(())
    }

    #[test]
    fn transform_enum_delegates_categorical_operations() -> Result<(), TransformError> {
        let input = [Column::new("group".into(), ["a", "b"])];
        let reference = Scalar::from(PlSmallStr::from_static("a"));
        let transform = Transform::C(Fit::Unfitted(CategoricalDefinition::new(Some(
            ParameterValue::Scalar(reference),
        ))));

        assert!(transform.expands());
        assert!(matches!(
            transform.transform(&input),
            Err(crate::FitError::NotFitted)
        ));
        let transform = transform.fit(&input)?;
        let output = transform.transform(&input).unwrap();
        let Transform::C(Fit::Fitted(categorical)) = &transform else {
            unreachable!();
        };
        assert_output(
            &categorical.definition,
            &categorical.state,
            &input,
            output.frame(),
            &[&[Some(false), Some(true)]],
        );
        Ok(())
    }

    fn assert_output(
        definition: &CategoricalDefinition,
        transform: &CategoricalState,
        input: &[Column],
        actual: &DataFrame,
        expected: &[&[Option<bool>]],
    ) {
        let height = expected.first().map_or(0, |column| column.len());
        assert_eq!(actual.shape(), (height, expected.len()));

        // The transform keeps every level, the reference included.
        let levels = &transform.levels;
        let names = (0..levels.len())
            .filter(|index| !transform.reference_indices.contains(index))
            .map(|index| {
                Scalar::new(
                    levels.dtype().clone(),
                    levels.get(index).unwrap().into_static(),
                )
            })
            .map(|level| definition.output_name(input[0].name(), &level))
            .collect::<Vec<_>>();

        for (index, (column, expected_values)) in actual.columns().iter().zip(expected).enumerate()
        {
            assert_eq!(column.name(), &names[index]);
            assert_eq!(column.dtype(), &DataType::Boolean);
            let values: Vec<_> = column
                .bool()
                .expect("Boolean output column")
                .iter()
                .collect();
            assert_eq!(
                values.as_slice(),
                *expected_values,
                "column {}",
                column.name()
            );
        }
    }

    #[test]
    fn categorical_requires_fit() {
        let reference = Scalar::from(PlSmallStr::from_static("missing"));
        let transform = Fit::Unfitted(CategoricalDefinition::new(Some(ParameterValue::Scalar(
            reference,
        ))));
        let input = prepare(&[Column::new("group".into(), ["a", "b"])]);

        assert!(matches!(
            transform.transform(&input),
            Err(crate::FitError::NotFitted)
        ));
        assert!(matches!(
            transform.fit(&input),
            Err(TransformError::InvalidParameter(_))
        ));
        assert!(matches!(
            transform.transform(&input),
            Err(crate::FitError::NotFitted)
        ));
    }

    #[test]
    fn categorical_definition_produces_independent_fits() -> Result<(), TransformError> {
        let definition = CategoricalDefinition::new(Some(ParameterValue::Scalar(Scalar::from(
            PlSmallStr::from_static("a"),
        ))));
        let first_input = prepare(&[Column::new("group".into(), ["a", "b"])]);
        let second_input = prepare(&[Column::new("group".into(), ["a", "c"])]);
        let input = [Column::new("group".into(), ["a", "b", "c"])];
        let prepared_input = prepare(&input);
        let first = definition.fit(&first_input)?;
        let second = definition.fit(&second_input)?;

        assert_output(
            &definition,
            &first,
            &input,
            definition.transform(&first, &prepared_input)?.frame(),
            &[&[Some(false), Some(true), None]],
        );
        assert_output(
            &definition,
            &second,
            &input,
            definition.transform(&second, &prepared_input)?.frame(),
            &[&[Some(false), None, Some(true)]],
        );

        Ok(())
    }

    #[test]
    fn categorical_failed_refit_preserves_fitted_levels() -> Result<(), TransformError> {
        let definition = CategoricalDefinition::new(Some(ParameterValue::Scalar(Scalar::from(
            PlSmallStr::from_static("a"),
        ))));
        let training = prepare(&[Column::new("group".into(), ["a", "b"])]);
        let invalid = prepare(&[Column::new("group".into(), ["b", "c"])]);
        let transform = Fit::Fitted(Fitted::fit(definition.clone(), &training)?);
        let original = transform.transform(&training).unwrap();

        assert!(matches!(
            transform.fit(&invalid),
            Err(TransformError::InvalidParameter(_))
        ));
        assert!(
            transform
                .transform(&training)
                .unwrap()
                .frame()
                .equals_missing(original.frame())
        );
        assert_eq!(transform.definition(), &definition);
        Ok(())
    }

    #[test]
    fn categorical_sorts_levels_and_preserves_fitted_schema() -> Result<(), TransformError> {
        // Exercise small sets and both sides of the eight-level lookup boundary.
        let levels = ["b", "a", "c", "d", "e", "f", "g", "h", "i"];
        for count in [3, 8, 9] {
            let mut values = vec![Some("b"), Some("a"), Some("b"), None];
            values.extend(levels[2..count].iter().copied().map(Some));
            let training = prepare(&[Column::new("group".into(), values)]);
            // Prediction omits fitted levels, but their columns must remain.
            let input = [Column::new(
                "group".into(),
                [Some("a"), Some("b"), Some("unseen"), None],
            )];
            let prepared_input = prepare(&input);
            let definition = CategoricalDefinition::new(None);
            let fitted = Fitted::fit(definition, &training)?;
            let output = fitted.transform(&prepared_input)?;
            let mut expected = vec![
                [Some(true), Some(false), None, None],
                [Some(false), Some(true), None, None],
            ];
            expected.extend((2..count).map(|_| [Some(false), Some(false), None, None]));
            let expected = expected
                .iter()
                .map(|values| values.as_slice())
                .collect::<Vec<_>>();
            assert_output(
                &fitted.definition,
                &fitted.state,
                &input,
                output.frame(),
                &expected,
            );
            let mut sorted = levels[..count].to_vec();
            sorted.sort_unstable();
            let names = sorted
                .iter()
                .map(|level| format!("C(group)[{level}]"))
                .collect::<Vec<_>>();
            assert_eq!(
                output
                    .frame()
                    .get_column_names()
                    .iter()
                    .map(|name| name.as_str())
                    .collect::<Vec<_>>(),
                names
            );
        }
        Ok(())
    }

    #[test]
    fn categorical_omits_reference_and_preserves_nulls() -> Result<(), TransformError> {
        // Keep the same reference and unknown/null semantics for both lookups.
        let levels = ["a", "b", "c", "d", "e", "f", "g", "h", "i"];
        for count in [3, 8, 9] {
            let mut values = levels[..count]
                .iter()
                .copied()
                .map(Some)
                .collect::<Vec<_>>();
            values.push(None);
            let training = prepare(&[Column::new("group".into(), values)]);
            let input = [Column::new(
                "group".into(),
                [Some("a"), Some("b"), Some("c"), Some("unseen"), None],
            )];
            let prepared_input = prepare(&input);
            let reference = Scalar::from(PlSmallStr::from_static("a"));
            let fitted = Fitted::fit(
                CategoricalDefinition::new(Some(ParameterValue::Scalar(reference))),
                &training,
            )?;
            let output = fitted.transform(&prepared_input)?;
            let mut expected = vec![
                [Some(false), Some(true), Some(false), None, None],
                [Some(false), Some(false), Some(true), None, None],
            ];
            expected
                .extend((3..count).map(|_| [Some(false), Some(false), Some(false), None, None]));
            for (column, expected) in output.frame().columns().iter().zip(&expected) {
                assert_eq!(column.bool()?.iter().collect::<Vec<_>>(), expected);
            }
            assert_eq!(output.frame().width(), expected.len());
            let names = levels[1..count]
                .iter()
                .map(|level| format!("C(group, omit='a')[{level}]"))
                .collect::<Vec<_>>();
            assert_eq!(
                output
                    .frame()
                    .get_column_names()
                    .iter()
                    .map(|name| name.as_str())
                    .collect::<Vec<_>>(),
                names
            );
        }
        Ok(())
    }

    #[test]
    fn categorical_accepts_integer_reference_across_widths() -> Result<(), TransformError> {
        let inputs = [
            [Column::new("group".into(), [1_i32, 2, 1])],
            [Column::new("group".into(), [1_i64, 2, 1])],
            [Column::new("group".into(), [1_u32, 2, 1])],
            [Column::new("group".into(), [1_u64, 2, 1])],
        ];
        for input in inputs {
            let prepared_input = prepare(&input);
            let definition =
                CategoricalDefinition::new(Some(ParameterValue::Scalar(Scalar::from(1_i64))));
            let transform = definition.fit(&prepared_input)?;
            assert_output(
                &definition,
                &transform,
                &input,
                definition.transform(&transform, &prepared_input)?.frame(),
                &[&[Some(false), Some(true), Some(false)]],
            );
        }
        Ok(())
    }

    #[test]
    fn categorical_resolves_float_reference_to_column_precision() -> Result<(), TransformError> {
        let input = [Column::new("group".into(), [0.1_f32, 0.2, 0.1])];
        let prepared_input = prepare(&input);
        let definition =
            CategoricalDefinition::new(Some(ParameterValue::Scalar(Scalar::from(0.1_f64))));
        let transform = definition.fit(&prepared_input)?;

        assert_output(
            &definition,
            &transform,
            &input,
            definition.transform(&transform, &prepared_input)?.frame(),
            &[&[Some(false), Some(true), Some(false)]],
        );
        Ok(())
    }

    #[test]
    fn categorical_supports_boolean_reference() -> Result<(), TransformError> {
        let input = [Column::new("group".into(), [Some(true), Some(false), None])];
        let prepared_input = prepare(&input);
        let definition =
            CategoricalDefinition::new(Some(ParameterValue::Scalar(Scalar::from(false))));
        let transform = definition.fit(&prepared_input)?;

        assert_output(
            &definition,
            &transform,
            &input,
            definition.transform(&transform, &prepared_input)?.frame(),
            &[&[Some(true), Some(false), None]],
        );
        Ok(())
    }

    #[test]
    fn categorical_rejects_missing_reference() {
        let input = prepare(&[Column::new("group".into(), ["a", "b"])]);
        let reference = Scalar::from(PlSmallStr::from_static("missing"));
        let definition = CategoricalDefinition::new(Some(ParameterValue::Scalar(reference)));

        assert!(matches!(
            definition.fit(&input),
            Err(TransformError::InvalidParameter(_))
        ));
    }

    #[test]
    fn categorical_rejects_reference_type_changes() {
        let cases = [
            (
                Scalar::from(1_i64),
                [Column::new("group".into(), [1.0_f64])],
            ),
            (
                Scalar::from(1.0_f64),
                [Column::new("group".into(), [1_i32])],
            ),
            (Scalar::from(true), [Column::new("group".into(), [1_i32])]),
            (Scalar::from(1_i64), [Column::new("group".into(), [true])]),
            (
                Scalar::from(PlSmallStr::from_static("1")),
                [Column::new("group".into(), [1_i32])],
            ),
        ];

        for (reference, input) in cases {
            let prepared_input = prepare(&input);
            let definition = CategoricalDefinition::new(Some(ParameterValue::Scalar(reference)));
            assert!(matches!(
                definition.fit(&prepared_input),
                Err(TransformError::InvalidParameter(_))
            ));
        }
    }

    #[test]
    fn categorical_rejects_out_of_range_integer_reference() {
        let cases = [
            (
                Scalar::from(-1_i64),
                [Column::new("group".into(), [0_u32, 1])],
            ),
            (
                Scalar::from(i64::MAX),
                [Column::new("group".into(), [0_i32, 1])],
            ),
        ];

        for (reference, input) in cases {
            let prepared_input = prepare(&input);
            let definition = CategoricalDefinition::new(Some(ParameterValue::Scalar(reference)));
            assert!(matches!(
                definition.fit(&prepared_input),
                Err(TransformError::Polars(_))
            ));
        }
    }

    #[test]
    fn categorical_rejects_nonfinite_float_reference() {
        let input = prepare(&[Column::new("group".into(), [0.1_f32, 0.2])]);

        for reference in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, f64::MAX] {
            let definition =
                CategoricalDefinition::new(Some(ParameterValue::Scalar(Scalar::from(reference))));
            assert!(matches!(
                definition.fit(&input),
                Err(TransformError::InvalidParameter(_))
            ));
        }
    }

    #[test]
    fn categorical_rejects_changed_input_dtype() -> Result<(), TransformError> {
        let training = prepare(&[Column::new("group".into(), [1_i32, 2])]);
        let input = prepare(&[Column::new("group".into(), [1_i64, 2])]);

        for reference in [None, Some(Scalar::from(1_i64))] {
            let definition = CategoricalDefinition::new(reference.map(ParameterValue::Scalar));
            let transform = definition.fit(&training)?;
            let Err(error @ TransformError::InputTypeMismatch { .. }) =
                definition.validate(&transform, &input)
            else {
                panic!("expected InputTypeMismatch");
            };
            assert_eq!(
                error.to_string(),
                format!(
                    "input `group`: expected {}, received {}",
                    DataType::Int32,
                    DataType::Int64,
                ),
            );
        }
        Ok(())
    }

    #[test]
    fn categorical_refit_replaces_fitted_levels() -> Result<(), TransformError> {
        let first = prepare(&[Column::new("group".into(), ["a", "b"])]);
        let second = prepare(&[Column::new("group".into(), ["b", "c"])]);
        let input = [Column::new("group".into(), ["a", "b", "c"])];
        let prepared_input = prepare(&input);
        let definition = CategoricalDefinition::new(None);
        let original = Fit::Fitted(Fitted::fit(definition, &first)?);
        let original_output = original.transform(&prepared_input).unwrap();
        let transform = original.fit(&second)?;
        assert!(
            original
                .transform(&prepared_input)
                .unwrap()
                .frame()
                .equals_missing(original_output.frame())
        );

        let Fit::Fitted(transform) = transform else {
            panic!("expected a fitted categorical operation");
        };

        assert_output(
            &transform.definition,
            &transform.state,
            &input,
            transform.transform(&prepared_input)?.frame(),
            &[
                &[None, Some(true), Some(false)],
                &[None, Some(false), Some(true)],
            ],
        );
        Ok(())
    }
}
