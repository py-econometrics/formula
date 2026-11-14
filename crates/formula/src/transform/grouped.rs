use super::input::{PreparedInput, validate_numeric_field};
use super::{ArgumentCount, TransformDefinition, call_name};
use crate::formula::MaterializedColumns;
use crate::{ComponentKey, TransformError};
use polars::prelude::{Column, DataFrame, DataType, PlSmallStr, Scalar};
use std::collections::BTreeMap;

/// One normalized fixed-effect term with an optional loading and group intercept.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct GroupedTransform {
    intercept: bool,
}

impl TransformDefinition for GroupedTransform {
    fn name(&self) -> &'static str {
        "G"
    }

    fn expands(&self) -> bool {
        true
    }

    fn argument_count(&self) -> ArgumentCount {
        ArgumentCount::AtLeast(1)
    }

    fn parameters(&self) -> BTreeMap<&'static str, Scalar> {
        BTreeMap::from([(Self::INTERCEPT_PARAMETER, Scalar::from(self.intercept))])
    }

    fn call_name(&self, arguments: &[&str]) -> PlSmallStr {
        if self.intercept {
            let intercept = format!("{}=true", Self::INTERCEPT_PARAMETER);
            call_name(
                self.name(),
                arguments.iter().copied().chain([intercept.as_str()]),
            )
        } else {
            call_name(self.name(), arguments.iter().copied())
        }
    }
}

impl GroupedTransform {
    const INTERCEPT_PARAMETER: &'static str = "intercept";
    const GROUP_KEY: &'static str = "group";
    const LOADING_KEY: &'static str = "loading";

    pub(crate) fn new(intercept: bool) -> Self {
        Self { intercept }
    }

    pub(crate) fn intercept(&self) -> bool {
        self.intercept
    }

    fn validate_inputs<'a>(
        &self,
        inputs: &'a PreparedInput,
    ) -> Result<(&'a Column, &'a [Column]), TransformError> {
        let Some((group, loadings)) = inputs.frame.columns().split_first() else {
            return Err(TransformError::InputCountMismatch {
                expected: self.argument_count(),
                actual: inputs.fields.len(),
            });
        };
        let (_, loading_fields) = inputs.fields.split_at(1);
        for field in loading_fields {
            validate_numeric_field(field)?;
        }
        Ok((group, loadings))
    }

    pub(crate) fn transform(
        &self,
        inputs: &PreparedInput,
    ) -> Result<MaterializedColumns<ComponentKey>, TransformError> {
        let (group, loadings) = self.validate_inputs(inputs)?;
        let arguments = inputs
            .fields
            .iter()
            .map(|input| input.name().as_str())
            .collect::<Vec<_>>();
        let name = self.call_name(&arguments);

        let group = group
            .clone()
            .with_name(format!("{name}[{}]", Self::GROUP_KEY).into());
        let height = group.len();
        let mut columns = vec![group];
        let mut keys = vec![ComponentKey::Scalar(Scalar::from(PlSmallStr::from_static(
            Self::GROUP_KEY,
        )))];
        for (index, loading) in loadings.iter().enumerate() {
            let input_position = index + 1;

            columns.push(
                loading
                    .cast(&DataType::Float64)?
                    .with_name(format!("{name}[{}:{input_position}]", Self::LOADING_KEY).into()),
            );

            keys.push(ComponentKey::Tuple(vec![
                ComponentKey::Scalar(Scalar::from(PlSmallStr::from_static(Self::LOADING_KEY))),
                ComponentKey::Scalar(Scalar::from(input_position as u64)),
            ]));
        }

        let frame = DataFrame::new(height, columns)?;

        Ok(MaterializedColumns::new(frame, keys))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fit::FitError;
    use crate::transform::Transform;
    use polars::error::PolarsError;

    fn prepare(inputs: &[Column]) -> PreparedInput {
        PreparedInput::new(inputs, false).unwrap()
    }

    #[test]
    fn intercept_materializes_only_group_codes() -> Result<(), TransformError> {
        for group in [
            Column::new("firm".into(), [Some(2_u32), None, Some(0)]),
            Column::new("firm".into(), Vec::<u32>::new()),
        ] {
            let expected = group
                .clone()
                .with_name("G(firm, intercept=true)[group]".into());
            let output = GroupedTransform::new(true).transform(&prepare(&[group]))?;
            assert!(output.frame().equals_missing(&expected.into_frame()));
        }
        Ok(())
    }

    #[test]
    fn preserves_groups_and_casts_loadings() -> Result<(), TransformError> {
        let inputs = [
            Column::new("firm".into(), [Some(2_u32), None, Some(0)]),
            Column::new("time".into(), [Some(10_i32), Some(20), None]),
            Column::new("exposure".into(), [Some(0.5_f32), None, Some(1.5)]),
        ];
        let prepared_inputs = prepare(&inputs);
        let output = GroupedTransform::new(false).transform(&prepared_inputs)?;
        let frame = output.frame();

        assert_eq!(
            frame.get_column_names(),
            [
                "G(firm, time, exposure)[group]",
                "G(firm, time, exposure)[loading:1]",
                "G(firm, time, exposure)[loading:2]",
            ],
        );
        assert_eq!(
            frame[0].u32()?.iter().collect::<Vec<_>>(),
            vec![Some(2), None, Some(0)],
        );
        assert_eq!(
            frame[1].f64()?.iter().collect::<Vec<_>>(),
            vec![Some(10.0), Some(20.0), None],
        );
        assert_eq!(
            frame[2].f64()?.iter().collect::<Vec<_>>(),
            vec![Some(0.5), None, Some(1.5)],
        );
        assert_eq!(inputs[0].name().as_str(), "firm");
        assert_eq!(inputs[1].name().as_str(), "time");
        assert_eq!(inputs[1].dtype(), &DataType::Int32);
        Ok(())
    }

    #[test]
    fn accepts_numeric_loadings_with_distinct_output_names() -> Result<(), TransformError> {
        for loading in [
            Column::new("x".into(), [2_i32, 3]),
            Column::new("x".into(), [2_i64, 3]),
            Column::new("x".into(), [2_u32, 3]),
            Column::new("x".into(), [2_u64, 3]),
            Column::new("x".into(), [2_f32, 3.0]),
            Column::new("x".into(), [2_f64, 3.0]),
        ] {
            let inputs = prepare(&[Column::new("x".into(), [0_u32, 1]), loading]);
            let output = GroupedTransform::new(false).transform(&inputs)?;
            assert_eq!(
                output.frame()[1].f64()?.iter().collect::<Vec<_>>(),
                vec![Some(2.0), Some(3.0)],
            );
            assert_ne!(output.frame()[0].name(), output.frame()[1].name());
        }
        Ok(())
    }

    #[test]
    fn preserves_empty_and_all_null_inputs() -> Result<(), TransformError> {
        for inputs in [
            [
                Column::new("firm".into(), Vec::<u32>::new()),
                Column::new("time".into(), Vec::<i32>::new()),
            ],
            [
                Column::new("firm".into(), [None::<u32>, None]),
                Column::new("time".into(), [None::<i32>, None]),
            ],
        ] {
            let prepared_inputs = prepare(&inputs);
            let output = GroupedTransform::new(false).transform(&prepared_inputs)?;
            assert_eq!(output.frame().shape(), (inputs[0].len(), 2));
            assert_eq!(output.frame()[0].dtype(), &DataType::UInt32);
            assert_eq!(output.frame()[1].dtype(), &DataType::Float64);
            assert_eq!(output.frame()[0].null_count(), inputs[0].len());
            assert_eq!(output.frame()[1].null_count(), inputs[1].len());
        }
        Ok(())
    }

    #[test]
    fn rejects_wrong_input_counts() {
        for intercept in [true, false] {
            let transform = GroupedTransform::new(intercept);
            let expected = transform.argument_count();
            for count in [0, 1, 2, 3]
                .into_iter()
                .filter(|count| !expected.accepts(*count))
            {
                let inputs = prepare(&vec![Column::new("x".into(), [0_u32]); count]);
                assert!(matches!(
                    transform.transform(&inputs),
                    Err(TransformError::InputCountMismatch {
                        expected: actual_expected,
                        actual,
                    }) if actual == count && actual_expected == expected
                ));
            }
        }
    }

    #[test]
    fn rejects_nonnumeric_loadings() {
        for loading in [
            Column::new("time".into(), ["1", "2"]),
            Column::new("time".into(), [true, false]),
        ] {
            let inputs = prepare(&[
                Column::new("firm".into(), [0_u32, 1]),
                Column::new("valid".into(), [1.0, 2.0]),
                loading,
            ]);
            assert!(matches!(
                GroupedTransform::new(false).transform(&inputs),
                Err(TransformError::NonNumericInput { name, .. }) if name == "time"
            ));
        }
    }

    #[test]
    fn rejects_misaligned_rows() {
        let inputs = [
            Column::new("firm".into(), [0_u32, 1]),
            Column::new("time".into(), [1.0]),
        ];
        assert!(matches!(
            Transform::G(GroupedTransform::new(false)).transform(&inputs),
            Err(FitError::Transform(TransformError::Polars(
                PolarsError::ShapeMismatch(_)
            )))
        ));
    }

    #[test]
    fn enum_delegates_stateless_materialization() -> Result<(), TransformError> {
        let transform = Transform::G(GroupedTransform::new(true));
        assert_eq!(transform.name(), "G");
        assert!(transform.expands());
        assert_eq!(transform.argument_count(), ArgumentCount::AtLeast(1));
        assert_eq!(
            transform.parameters(),
            BTreeMap::from([("intercept", Scalar::from(true))])
        );

        let inputs = [
            Column::new("firm".into(), [0_u32, 1]),
            Column::new("time".into(), [1_i32, 2]),
        ];
        let before_fit = transform.transform(&inputs).unwrap();
        let fitted = transform.fit(&inputs)?;
        assert_eq!(transform, fitted);
        assert!(
            before_fit
                .frame()
                .equals_missing(fitted.transform(&inputs).unwrap().frame())
        );
        assert!(matches!(
            fitted.transform(&[]),
            Err(FitError::Transform(
                TransformError::InputCountMismatch { .. }
            ))
        ));
        Ok(())
    }
}
