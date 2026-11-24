use super::input::PreparedInput;
use super::{ArgumentCount, ParameterValue, TransformDefinition, TransformError, call_name};
use crate::fit::Definition;
use crate::formula::{ComponentKey, MaterializedColumns};
use polars::chunked_array::cast::CastOptions;
use polars::error::PolarsResult;
use polars::prelude::{
    AnyValue, DataFrame, DataType, IntoLazy, IntoSeries, NULL, PlSmallStr, Scalar, Series,
    SortOptions, col, lit, when,
};
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Levels {
    Scalar(Scalar),
    List(Vec<Scalar>),
}

impl Levels {
    fn as_slice(&self) -> &[Scalar] {
        match self {
            Self::Scalar(level) => std::slice::from_ref(level),
            Self::List(levels) => levels,
        }
    }
}

impl TryFrom<ParameterValue> for Levels {
    type Error = ParameterValue;

    fn try_from(value: ParameterValue) -> Result<Self, Self::Error> {
        match value {
            ParameterValue::Scalar(level) => Ok(Self::Scalar(level)),
            ParameterValue::List(levels) => Ok(Self::List(levels)),
            value => Err(value),
        }
    }
}

impl From<Levels> for ParameterValue {
    fn from(levels: Levels) -> Self {
        match levels {
            Levels::Scalar(level) => Self::Scalar(level),
            Levels::List(levels) => Self::List(levels),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum LevelSelection {
    Omit(Levels),
    Keep(Levels),
}

impl LevelSelection {
    fn parameter(&self) -> (&'static str, &Levels) {
        match self {
            Self::Omit(levels) => (CategoricalDefinition::OMIT_PARAMETER, levels),
            Self::Keep(levels) => (CategoricalDefinition::KEEP_PARAMETER, levels),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct CategoricalDefinition {
    selection: Option<LevelSelection>,
    bins: Option<Vec<(PlSmallStr, Levels)>>,
}

impl CategoricalDefinition {
    pub(crate) const OMIT_PARAMETER: &str = "omit";
    pub(crate) const KEEP_PARAMETER: &str = "keep";
    pub(crate) const BIN_PARAMETER: &str = "bin";

    pub(crate) fn new(
        selection: Option<LevelSelection>,
        bins: Option<Vec<(PlSmallStr, Levels)>>,
    ) -> Self {
        // `bin={}` merges nothing, so an empty dict means no `bin`
        let bins = bins.filter(|bins| !bins.is_empty());
        Self { selection, bins }
    }

    fn levels_name(levels: &Levels) -> String {
        match levels {
            Levels::Scalar(level) => Self::value_name(level),
            Levels::List(levels) => {
                let names = levels.iter().map(Self::value_name).collect::<Vec<_>>();
                format!("[{}]", names.join(", "))
            }
        }
    }

    fn value_name(value: &Scalar) -> String {
        match value.as_any_value() {
            AnyValue::String(value) => format!("'{value}'"),
            AnyValue::StringOwned(value) => format!("'{value}'"),
            _ => Self::level_name(value),
        }
    }

    fn output_name(&self, input_name: &str, level: &Scalar) -> PlSmallStr {
        let call = self.call_name(&[input_name]);
        format!("{call}[{}]", Self::level_name(level)).into()
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

    /// Converts a level named by `parameter` to the input dtype.
    ///
    /// String levels for categorical and enum inputs are resolved through the
    /// existing dictionary without inserting new levels. Numeric levels may
    /// convert between integer widths or between float widths; other dtype
    /// combinations must match exactly. Nonfinite numeric levels are rejected.
    ///
    /// Membership in the observed fitting levels is checked separately in `fit`.
    fn convert_level(
        parameter: &str,
        level: &Scalar,
        dtype: &DataType,
    ) -> Result<Scalar, TransformError> {
        // Get the dtype of the level provided
        let source = level.dtype();
        // Check compatibility with and convert to dtype
        let level = match (source, dtype) {
            // Source and target dtype match
            (source, target) if source == target => level.clone(),
            // Target is Categorical/Enum; check if level is in observed levels
            (DataType::String, DataType::Categorical(_, mapping) | DataType::Enum(_, mapping)) => {
                let value = level.as_any_value();
                let label = value.str_value();
                let code = mapping.get_cat(label.as_ref()).ok_or_else(|| {
                    TransformError::InvalidParameter(format!(
                        "`{parameter}` value is not among the fitted levels"
                    ))
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
            // Level/target are integers; we cast compatible integers here
            (source, target) if source.is_integer() && target.is_integer() => level
                .clone()
                .cast_with_options(target, CastOptions::Strict)?,
            // Level/target are floats; we cast compatible floats here
            (source, target) if source.is_float() && target.is_float() => level
                .clone()
                .cast_with_options(target, CastOptions::Strict)?,
            // Level not compatible with target dtype
            _ => {
                return Err(TransformError::InvalidParameter(format!(
                    "`{parameter}` dtype {source:?} is incompatible with input dtype {dtype:?}"
                )));
            }
        };

        if level.dtype().is_float() && !level.as_any_value().try_extract::<f64>()?.is_finite() {
            return Err(TransformError::InvalidParameter(format!(
                "`{parameter}` must be finite after conversion to the input dtype"
            )));
        }

        Ok(level)
    }

    /// Return `value`'s position in `levels`
    fn level_index(
        parameter: &str,
        value: &Scalar,
        levels: &Series,
    ) -> Result<usize, TransformError> {
        let value = Self::convert_level(parameter, value, levels.dtype())?;
        (0..levels.len())
            .find_map(|index| match levels.get(index) {
                Ok(level) if level == value.as_any_value() => Some(Ok(index)),
                Ok(_) => None,
                Err(error) => Some(Err(error)),
            })
            .transpose()?
            .ok_or_else(|| {
                TransformError::InvalidParameter(format!(
                    "`{parameter}` value is not among the fitted levels"
                ))
            })
    }

    fn fitted_levels(levels: &Series) -> PolarsResult<Vec<Scalar>> {
        (0..levels.len())
            .map(|index| {
                Ok(Scalar::new(
                    levels.dtype().clone(),
                    levels.get(index)?.into_static(),
                ))
            })
            .collect()
    }

    /// Builds the one-hot encoding: one column per unbinned level or bin
    fn bin_levels(
        bins: &[(PlSmallStr, Levels)],
        levels: &Series,
    ) -> Result<(Vec<EncodedColumn>, Vec<usize>), TransformError> {
        let fitted_levels = Self::fitted_levels(levels)?;
        // Populate mapping of level index to bin
        let mut level_bins: Vec<Option<usize>> = vec![None; levels.len()];
        for (bin, (name, members)) in bins.iter().enumerate() {
            for member in members.as_slice() {
                let index = Self::level_index(Self::BIN_PARAMETER, member, levels)?;
                match level_bins[index] {
                    None => level_bins[index] = Some(bin),
                    // A level repeated within its own bin is allowed
                    Some(other) if other == bin => {}
                    Some(other) => {
                        let (other_name, _) = &bins[other];
                        return Err(TransformError::InvalidParameter(format!(
                            "level {} is in bins '{other_name}' and '{name}'",
                            Self::level_name(&fitted_levels[index]),
                        )));
                    }
                }
            }
        }
        // A bin can't be named like an unbinned level
        for (name, _) in bins {
            if (0..levels.len()).any(|index| {
                level_bins[index].is_none()
                    && Self::level_name(&fitted_levels[index]) == name.as_str()
            }) {
                return Err(TransformError::InvalidParameter(format!(
                    "bin '{name}' has the same name as an unbinned level"
                )));
            }
        }
        // Construct columns for one-hot encoding
        let mut columns = Vec::new();
        for index in 0..levels.len() {
            match level_bins[index] {
                None => columns.push(EncodedColumn::Level(index)),
                // The bin's column was already added at one of its earlier levels
                Some(bin) if level_bins[..index].contains(&Some(bin)) => {}
                // Add a bin's column at the position of its first level
                Some(bin) => columns.push(EncodedColumn::Bin {
                    name: bins[bin].0.clone(),
                    levels: (index..levels.len())
                        .filter(|member| level_bins[*member] == Some(bin))
                        .collect(),
                }),
            }
        }
        // Construct the mapping of a level to the column of the one-hot encoding
        let mut level_columns = vec![0; levels.len()];
        for (column, encoded) in columns.iter().enumerate() {
            for level in encoded.levels() {
                level_columns[*level] = column;
            }
        }
        Ok((columns, level_columns))
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
        self.selection
            .iter()
            .map(|selection| {
                let (name, levels) = selection.parameter();
                (name, levels.clone().into())
            })
            .chain(self.bins.iter().map(|bins| {
                let bins = bins
                    .iter()
                    .map(|(name, levels)| (name.clone(), levels.clone().into()))
                    .collect();
                (Self::BIN_PARAMETER, ParameterValue::Map(bins))
            }))
            .collect()
    }

    fn call_name(&self, arguments: &[&str]) -> PlSmallStr {
        // `bin` is applied before `omit` or `keep`, so it is shown first
        let bin_argument = self.bins.as_ref().map(|bins| {
            let bins = bins
                .iter()
                .map(|(name, levels)| format!("'{name}': {}", Self::levels_name(levels)))
                .collect::<Vec<_>>();
            format!("{}={{{}}}", Self::BIN_PARAMETER, bins.join(", "))
        });
        let selection_argument = self.selection.as_ref().map(|selection| {
            let (name, levels) = selection.parameter();
            format!("{name}={}", Self::levels_name(levels))
        });

        call_name(
            self.name(),
            arguments
                .iter()
                .copied()
                .chain(bin_argument.as_deref())
                .chain(selection_argument.as_deref()),
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
        // Sort levels: numbers by value, strings and categoricals by label, enums by category
        let levels = input
            .drop_nulls()
            .unique()?
            .sort(SortOptions::default())?
            .take_materialized_series();
        // Prevent encoding nonfinite floats as a category
        let levels = if levels.dtype().is_float() {
            levels.filter(&levels.is_finite()?)?
        } else {
            levels
        };

        let (columns, level_columns) = match &self.bins {
            None => (
                (0..levels.len()).map(EncodedColumn::Level).collect(),
                (0..levels.len()).collect(),
            ),
            Some(bins) => Self::bin_levels(bins, &levels)?,
        };

        let Some(selection) = &self.selection else {
            return Ok(CategoricalState {
                levels,
                columns,
                omitted_indices: Vec::new(),
            });
        };
        let (parameter, selected) = selection.parameter();
        let mut selected_indices = Vec::new();
        for value in selected.as_slice() {
            let named_bin = columns.iter().position(|column| {
                matches!(column, EncodedColumn::Bin { name, .. }
                    if value.dtype() == &DataType::String
                        && value.as_any_value().str_value() == name.as_str())
            });
            let column = match named_bin {
                Some(column) => column,
                None => {
                    let column = level_columns[Self::level_index(parameter, value, &levels)?];
                    if let EncodedColumn::Bin { name, .. } = &columns[column] {
                        return Err(TransformError::InvalidParameter(format!(
                            "`{parameter}` value is merged into bin '{name}'"
                        )));
                    }
                    column
                }
            };

            if !selected_indices.contains(&column) {
                selected_indices.push(column);
            }
        }

        // `keep` omits every column it doesn't name
        let omitted_indices = if let LevelSelection::Keep(_) = selection {
            (0..columns.len())
                .filter(|index| !selected_indices.contains(index))
                .collect()
        } else {
            selected_indices
        };

        Ok(CategoricalState {
            levels,
            columns,
            omitted_indices,
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
        let fitted_levels = Self::fitted_levels(levels)?;

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

        let (expressions, metadata): (Vec<_>, Vec<ComponentKey>) = state
            .columns
            .iter()
            .enumerate()
            .filter(|(index, _)| !state.omitted_indices.contains(index))
            .map(|(_, column)| {
                let key = column.key(&fitted_levels);
                let column_name = self.output_name(field.name(), &key);
                // Check if input is part of the (encoded) levels: one level or a bin
                let indicator = column
                    .levels()
                    .iter()
                    .map(|index| argument.clone().eq(lit(fitted_levels[*index].clone())))
                    .reduce(|left, right| left.or(right))
                    .unwrap_or(lit(false));

                let expression = when(argument.clone().is_null())
                    .then(lit(NULL))
                    .when(level_was_seen.clone())
                    .then(indicator)
                    .otherwise(lit(NULL))
                    .alias(column_name);

                (expression, ComponentKey::Scalar(key))
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
enum EncodedColumn {
    Level(usize),
    Bin {
        name: PlSmallStr,
        levels: Vec<usize>,
    },
}

impl EncodedColumn {
    fn key(&self, fitted_levels: &[Scalar]) -> Scalar {
        match self {
            Self::Level(index) => fitted_levels[*index].clone(),
            Self::Bin { name, .. } => Scalar::from(name.clone()),
        }
    }

    fn levels(&self) -> &[usize] {
        match self {
            Self::Level(index) => std::slice::from_ref(index),
            Self::Bin { levels, .. } => levels,
        }
    }
}

#[derive(Debug)]
pub(crate) struct CategoricalState {
    levels: Series,
    columns: Vec<EncodedColumn>,
    omitted_indices: Vec<usize>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fit::{Fit, Fitted};
    use crate::transform::Transform;
    use polars::prelude::{CategoricalPhysical, Categories, Column, FrozenCategories};

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
    fn native_levels_keep_label_names() -> Result<(), TransformError> {
        for dtype in native_dtypes() {
            let training = [Column::new("x".into(), ["b", "d", "b"]).cast(&dtype)?];
            let prepared_training = prepare(&training);
            let fitted = Fitted::fit(CategoricalDefinition::new(None, None), &prepared_training)?;
            let output = fitted.transform(&prepared_training)?;

            assert_eq!(output.frame().get_column_names(), ["C(x)[b]", "C(x)[d]"]);
            assert_output(
                &fitted.definition,
                &fitted.state,
                &training,
                output.frame(),
                &[
                    &[Some(true), Some(false), Some(true)],
                    &[Some(false), Some(true), Some(false)],
                ],
            );
        }
        Ok(())
    }

    #[test]
    fn levels_sort_by_value_label_or_enum_category_order() -> Result<(), TransformError> {
        // Dictionary order differs from label order
        let categorical = DataType::from_categories(Categories::random(
            "formula-tests".into(),
            CategoricalPhysical::U32,
        ));
        for label in ["c", "b", "a"] {
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
                labels.cast(&categorical)?,
                ["C(x)[a]", "C(x)[b]", "C(x)[c]"],
            ),
            (
                labels.cast(&enumeration)?,
                ["C(x)[c]", "C(x)[a]", "C(x)[b]"],
            ),
        ] {
            let prepared = prepare(&[input]);
            let fitted = Fitted::fit(CategoricalDefinition::new(None, None), &prepared)?;
            assert_eq!(
                fitted.transform(&prepared)?.frame().get_column_names(),
                expected
            );
        }
        Ok(())
    }

    #[test]
    fn native_unknown_levels_and_nulls_produce_null_indicators() -> Result<(), TransformError> {
        for dtype in native_dtypes() {
            let training = prepare(&[Column::new("x".into(), ["b", "d"]).cast(&dtype)?]);
            let fitted = Fitted::fit(CategoricalDefinition::new(None, None), &training)?;
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

    #[test]
    fn native_reference_is_omitted_without_treating_unknown_levels_as_reference()
    -> Result<(), TransformError> {
        for dtype in native_dtypes() {
            let training = prepare(&[Column::new("x".into(), ["d", "b"]).cast(&dtype)?]);
            let reference = Scalar::from(PlSmallStr::from_static("b"));
            let fitted = Fitted::fit(
                CategoricalDefinition::new(
                    Some(LevelSelection::Omit(Levels::Scalar(reference))),
                    None,
                ),
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
    fn native_reference_must_be_observed_not_just_declared() -> Result<(), TransformError> {
        for dtype in native_dtypes() {
            let training = prepare(&[Column::new("x".into(), ["d", "b"]).cast(&dtype)?]);
            let reference = Scalar::from(PlSmallStr::from_static("a"));
            let error = Fitted::fit(
                CategoricalDefinition::new(
                    Some(LevelSelection::Omit(Levels::Scalar(reference))),
                    None,
                ),
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
                    CategoricalDefinition::new(
                        Some(LevelSelection::Omit(Levels::Scalar(reference))),
                        None
                    ),
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
        let training = prepare(&[Column::new("x".into(), ["b", "d"]).cast(&dtype)?]);
        let fitted = Fitted::fit(CategoricalDefinition::new(None, None), &training)?;
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
            let fitted = Fitted::fit(CategoricalDefinition::new(None, None), &training)?;
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
                let fitted = Fitted::fit(CategoricalDefinition::new(None, None), &training)?;
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
            let fitted = Fitted::fit(CategoricalDefinition::new(None, None), &training)?;
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
                CategoricalDefinition::new(
                    Some(LevelSelection::Omit(Levels::Scalar(reference))),
                    None,
                ),
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
        let definition = CategoricalDefinition::new(
            Some(LevelSelection::Omit(Levels::Scalar(Scalar::from(1_i64)))),
            None,
        );
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
            let definition = CategoricalDefinition::new(None, None);
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
                Some(1.0_f64),
                Some(f64::NAN),
                Some(f64::INFINITY),
                Some(2.0),
                Some(f64::NEG_INFINITY),
                Some(1.0),
                None,
            ],
        );
        for dtype in [DataType::Float32, DataType::Float64] {
            let inputs = [input.cast(&dtype)?];
            let prepared_inputs = prepare(&inputs);
            let definition = CategoricalDefinition::new(None, None);
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
                    &[Some(true), None, None, Some(false), None, Some(true), None],
                    &[Some(false), None, None, Some(true), None, Some(false), None],
                ],
            );
        }
        Ok(())
    }

    #[test]
    fn transform_enum_delegates_categorical_operations() -> Result<(), TransformError> {
        let input = [Column::new("group".into(), ["a", "b"])];
        let reference = Scalar::from(PlSmallStr::from_static("a"));
        let transform = Transform::C(Fit::Unfitted(CategoricalDefinition::new(
            Some(LevelSelection::Omit(Levels::Scalar(reference))),
            None,
        )));

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

        let fitted_levels = CategoricalDefinition::fitted_levels(&transform.levels).unwrap();
        let names = transform
            .columns
            .iter()
            .enumerate()
            .filter(|(index, _)| !transform.omitted_indices.contains(index))
            .map(|(_, column)| definition.output_name(input[0].name(), &column.key(&fitted_levels)))
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
        let transform = Fit::Unfitted(CategoricalDefinition::new(
            Some(LevelSelection::Omit(Levels::Scalar(reference))),
            None,
        ));
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
        let definition = CategoricalDefinition::new(
            Some(LevelSelection::Omit(Levels::Scalar(Scalar::from(
                PlSmallStr::from_static("a"),
            )))),
            None,
        );
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
        let definition = CategoricalDefinition::new(
            Some(LevelSelection::Omit(Levels::Scalar(Scalar::from(
                PlSmallStr::from_static("a"),
            )))),
            None,
        );
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
    fn categorical_preserves_fitted_order_and_schema() -> Result<(), TransformError> {
        // Exercise small sets and both sides of the eight-level lookup boundary.
        let levels = ["a", "b", "c", "d", "e", "f", "g", "h", "i"];
        for count in [3, 8, 9] {
            let mut values = vec![Some("a"), Some("b"), Some("a"), None];
            values.extend(levels[2..count].iter().copied().map(Some));
            let training = prepare(&[Column::new("group".into(), values)]);
            // Prediction omits fitted levels, but their columns must remain.
            let input = [Column::new(
                "group".into(),
                [Some("a"), Some("b"), Some("unseen"), None],
            )];
            let prepared_input = prepare(&input);
            let definition = CategoricalDefinition::new(None, None);
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
            let names = levels[..count]
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
                CategoricalDefinition::new(
                    Some(LevelSelection::Omit(Levels::Scalar(reference))),
                    None,
                ),
                &training,
            )?;
            let output = fitted.transform(&prepared_input)?;
            let mut expected = vec![
                [Some(false), Some(true), Some(false), None, None],
                [Some(false), Some(false), Some(true), None, None],
            ];
            expected
                .extend((3..count).map(|_| [Some(false), Some(false), Some(false), None, None]));
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
    fn categorical_keeps_only_listed_levels_and_preserves_nulls() -> Result<(), TransformError> {
        let training = prepare(&[Column::new("group".into(), ["a", "b", "c", "d"])]);
        let input = [Column::new(
            "group".into(),
            [
                Some("a"),
                Some("b"),
                Some("c"),
                Some("d"),
                Some("unseen"),
                None,
            ],
        )];
        let prepared_input = prepare(&input);
        let keep = Levels::List(vec![
            Scalar::from(PlSmallStr::from_static("d")),
            Scalar::from(PlSmallStr::from_static("b")),
        ]);
        let fitted = Fitted::fit(
            CategoricalDefinition::new(Some(LevelSelection::Keep(keep)), None),
            &training,
        )?;
        let output = fitted.transform(&prepared_input)?;
        // Columns follow the fitted level order, not the order of `keep`
        assert_eq!(
            output.frame().get_column_names(),
            [
                "C(group, keep=['d', 'b'])[b]",
                "C(group, keep=['d', 'b'])[d]"
            ]
        );
        assert_output(
            &fitted.definition,
            &fitted.state,
            &input,
            output.frame(),
            &[
                &[
                    Some(false),
                    Some(true),
                    Some(false),
                    Some(false),
                    None,
                    None,
                ],
                &[
                    Some(false),
                    Some(false),
                    Some(false),
                    Some(true),
                    None,
                    None,
                ],
            ],
        );
        Ok(())
    }

    #[test]
    fn categorical_rejects_unobserved_kept_level() {
        let input = prepare(&[Column::new("group".into(), ["a", "b"])]);
        let keep = Levels::List(vec![
            Scalar::from(PlSmallStr::from_static("a")),
            Scalar::from(PlSmallStr::from_static("missing")),
        ]);
        let error = CategoricalDefinition::new(Some(LevelSelection::Keep(keep)), None)
            .fit(&input)
            .unwrap_err();

        assert!(matches!(error, TransformError::InvalidParameter(_)));
        assert_eq!(
            error.to_string(),
            "`keep` value is not among the fitted levels"
        );
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
            let definition = CategoricalDefinition::new(
                Some(LevelSelection::Omit(Levels::Scalar(Scalar::from(1_i64)))),
                None,
            );
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
        let definition = CategoricalDefinition::new(
            Some(LevelSelection::Omit(Levels::Scalar(Scalar::from(0.1_f64)))),
            None,
        );
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
        let definition = CategoricalDefinition::new(
            Some(LevelSelection::Omit(Levels::Scalar(Scalar::from(false)))),
            None,
        );
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
        let definition =
            CategoricalDefinition::new(Some(LevelSelection::Omit(Levels::Scalar(reference))), None);

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
            let definition = CategoricalDefinition::new(
                Some(LevelSelection::Omit(Levels::Scalar(reference))),
                None,
            );
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
            let definition = CategoricalDefinition::new(
                Some(LevelSelection::Omit(Levels::Scalar(reference))),
                None,
            );
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
            let definition = CategoricalDefinition::new(
                Some(LevelSelection::Omit(Levels::Scalar(Scalar::from(
                    reference,
                )))),
                None,
            );
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
            let definition = CategoricalDefinition::new(
                reference.map(|reference| LevelSelection::Omit(Levels::Scalar(reference))),
                None,
            );
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
        let definition = CategoricalDefinition::new(None, None);
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

    fn text(value: &'static str) -> Scalar {
        Scalar::from(PlSmallStr::from_static(value))
    }

    #[test]
    fn categorical_bins_merge_levels_at_their_smallest_member() -> Result<(), TransformError> {
        let training = prepare(&[Column::new("group".into(), ["a", "b", "c", "d"])]);
        let input = [Column::new(
            "group".into(),
            [
                Some("a"),
                Some("b"),
                Some("c"),
                Some("d"),
                Some("unseen"),
                None,
            ],
        )];
        let bins = vec![
            // A level repeated within a bin is tolerated
            (
                PlSmallStr::from_static("db"),
                Levels::List(vec![text("d"), text("b"), text("d")]),
            ),
            // A single-member bin renames its level
            (PlSmallStr::from_static("x"), Levels::Scalar(text("c"))),
        ];
        let fitted = Fitted::fit(CategoricalDefinition::new(None, Some(bins)), &training)?;
        let output = fitted.transform(&prepare(&input))?;

        assert_eq!(
            output.frame().get_column_names(),
            [
                "C(group, bin={'db': ['d', 'b', 'd'], 'x': 'c'})[a]",
                "C(group, bin={'db': ['d', 'b', 'd'], 'x': 'c'})[db]",
                "C(group, bin={'db': ['d', 'b', 'd'], 'x': 'c'})[x]",
            ]
        );
        assert_output(
            &fitted.definition,
            &fitted.state,
            &input,
            output.frame(),
            &[
                &[
                    Some(true),
                    Some(false),
                    Some(false),
                    Some(false),
                    None,
                    None,
                ],
                &[Some(false), Some(true), Some(false), Some(true), None, None],
                &[
                    Some(false),
                    Some(false),
                    Some(true),
                    Some(false),
                    None,
                    None,
                ],
            ],
        );
        Ok(())
    }

    #[test]
    fn categorical_selections_name_bins_and_unbinned_levels() -> Result<(), TransformError> {
        let training = prepare(&[Column::new("group".into(), ["a", "b", "c", "d"])]);
        // A bin named after one of its members, as fixest names unnamed bins
        let bins = || {
            Some(vec![(
                PlSmallStr::from_static("a"),
                Levels::List(vec![text("a"), text("b")]),
            )])
        };
        for (selection, expected) in [
            (
                LevelSelection::Omit(Levels::Scalar(text("a"))),
                [
                    "C(group, bin={'a': ['a', 'b']}, omit='a')[c]",
                    "C(group, bin={'a': ['a', 'b']}, omit='a')[d]",
                ],
            ),
            (
                LevelSelection::Keep(Levels::List(vec![text("c"), text("a")])),
                [
                    "C(group, bin={'a': ['a', 'b']}, keep=['c', 'a'])[a]",
                    "C(group, bin={'a': ['a', 'b']}, keep=['c', 'a'])[c]",
                ],
            ),
        ] {
            let fitted = Fitted::fit(
                CategoricalDefinition::new(Some(selection), bins()),
                &training,
            )?;
            let output = fitted.transform(&training)?;
            assert_eq!(output.frame().get_column_names(), expected);
        }

        // Bins covering every level leave a single column
        let all = vec![(
            PlSmallStr::from_static("all"),
            Levels::List(vec![text("a"), text("b"), text("c"), text("d")]),
        )];
        let fitted = Fitted::fit(CategoricalDefinition::new(None, Some(all)), &training)?;
        assert_eq!(fitted.transform(&training)?.frame().width(), 1);
        Ok(())
    }

    #[test]
    fn categorical_bins_on_numeric_levels_are_named_by_string() -> Result<(), TransformError> {
        let training = prepare(&[Column::new("x".into(), [1_i64, 2, 3])]);
        let bins = || {
            Some(vec![(
                PlSmallStr::from_static("1"),
                Levels::List(vec![Scalar::from(1_i64), Scalar::from(2_i64)]),
            )])
        };

        let named = Fitted::fit(
            CategoricalDefinition::new(
                Some(LevelSelection::Omit(Levels::Scalar(text("1")))),
                bins(),
            ),
            &training,
        )?;
        assert_eq!(
            named.transform(&training)?.frame().get_column_names(),
            ["C(x, bin={'1': [1, 2]}, omit='1')[3]"]
        );

        let error = CategoricalDefinition::new(
            Some(LevelSelection::Omit(Levels::Scalar(Scalar::from(1_i64)))),
            bins(),
        )
        .fit(&training)
        .unwrap_err();
        assert_eq!(error.to_string(), "`omit` value is merged into bin '1'");
        Ok(())
    }

    #[test]
    fn categorical_rejects_invalid_bins() {
        let input = prepare(&[Column::new("group".into(), ["a", "b", "c"])]);
        let bin = |name, members: &[&'static str]| {
            (
                PlSmallStr::from_static(name),
                Levels::List(members.iter().copied().map(text).collect()),
            )
        };
        for (bins, selection, message) in [
            (
                vec![bin("ab", &["a", "missing"])],
                None,
                "`bin` value is not among the fitted levels",
            ),
            (
                vec![bin("ab", &["a", "b"]), bin("bc", &["b", "c"])],
                None,
                "level b is in bins 'ab' and 'bc'",
            ),
            (
                vec![bin("c", &["a", "b"])],
                None,
                "bin 'c' has the same name as an unbinned level",
            ),
            (
                vec![bin("ab", &["a", "b"])],
                Some(LevelSelection::Keep(Levels::Scalar(text("b")))),
                "`keep` value is merged into bin 'ab'",
            ),
        ] {
            let error = CategoricalDefinition::new(selection, Some(bins))
                .fit(&input)
                .unwrap_err();
            assert_eq!(error.to_string(), message);
        }
    }
}
