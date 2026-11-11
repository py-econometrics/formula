use super::{ArgumentCount, TransformDefinition, validate_numeric_input};
use crate::formula::MaterializedColumns;
use crate::{ComponentKey, TransformError};
use polars::prelude::{Column, DataFrame, DataType, PlSmallStr, Scalar};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct GroupedTransform;

impl TransformDefinition for GroupedTransform {
    fn name(&self) -> &'static str {
        "G"
    }

    fn expands(&self) -> bool {
        true
    }

    fn argument_count(&self) -> ArgumentCount {
        ArgumentCount::Exactly(2)
    }
}

impl GroupedTransform {
    const GROUP_KEY: &'static str = "group";
    const LOADING_KEY: &'static str = "loading";
    fn validate_inputs<'a>(
        &self,
        inputs: &'a [Column],
    ) -> Result<(&'a Column, &'a Column), TransformError> {
        let [group, loading] = inputs else {
            return Err(TransformError::InputCountMismatch {
                expected: self.argument_count(),
                actual: inputs.len(),
            });
        };

        if group.dtype() != &DataType::UInt32 {
            return Err(TransformError::InputTypeMismatch {
                name: group.name().clone(),
                expected: DataType::UInt32.to_string(),
                actual: group.dtype().to_string(),
            });
        }

        validate_numeric_input(loading)?;

        Ok((group, loading))
    }

    pub(crate) fn transform(
        &self,
        inputs: &[Column],
    ) -> Result<MaterializedColumns<ComponentKey>, TransformError> {
        let (group, loading) = self.validate_inputs(inputs)?;
        let name = self.call_name(&[group.name().as_str(), loading.name().as_str()]);

        let group = group
            .clone()
            .with_name(format!("{name}[{}]", Self::GROUP_KEY).into());
        let loading = loading
            .cast(&DataType::Float64)?
            .with_name(format!("{name}[{}]", Self::LOADING_KEY).into());

        let frame = DataFrame::new(group.len(), vec![group, loading])?;
        let keys = vec![
            ComponentKey::Scalar(Scalar::from(PlSmallStr::from_static(Self::GROUP_KEY))),
            ComponentKey::Scalar(Scalar::from(PlSmallStr::from_static(Self::LOADING_KEY))),
        ];

        Ok(MaterializedColumns::new(frame, keys))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fit::FitError;
    use crate::transform::Transform;
    use polars::error::PolarsError;

    #[test]
    fn preserves_groups_and_casts_loading() -> Result<(), TransformError> {
        let inputs = [
            Column::new("firm".into(), [Some(2_u32), None, Some(0)]),
            Column::new("time".into(), [Some(10_i32), Some(20), None]),
        ];
        let output = GroupedTransform.transform(&inputs)?;
        let frame = output.frame();

        assert_eq!(
            frame.get_column_names(),
            ["G(firm, time)[group]", "G(firm, time)[loading]"],
        );
        assert_eq!(
            frame[0].u32()?.iter().collect::<Vec<_>>(),
            vec![Some(2), None, Some(0)],
        );
        assert_eq!(
            frame[1].f64()?.iter().collect::<Vec<_>>(),
            vec![Some(10.0), Some(20.0), None],
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
            let inputs = [Column::new("x".into(), [0_u32, 1]), loading];
            let output = GroupedTransform.transform(&inputs)?;
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
            let output = GroupedTransform.transform(&inputs)?;
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
        for count in [0, 1, 3] {
            let inputs = vec![Column::new("x".into(), [0_u32]); count];
            assert!(matches!(
                GroupedTransform.transform(&inputs),
                Err(TransformError::InputCountMismatch {
                    expected: ArgumentCount::Exactly(2),
                    actual,
                }) if actual == count
            ));
        }
    }

    #[test]
    fn rejects_unencoded_groups() {
        for group in [
            Column::new("firm".into(), [0_i32, 1]),
            Column::new("firm".into(), [0_u64, 1]),
            Column::new("firm".into(), [0_f64, 1.0]),
            Column::new("firm".into(), ["a", "b"]),
        ] {
            let actual_dtype = group.dtype().to_string();
            let inputs = [group, Column::new("time".into(), [1.0, 2.0])];
            assert!(matches!(
                GroupedTransform.transform(&inputs),
                Err(TransformError::InputTypeMismatch { name, expected, actual })
                    if name == "firm" && expected == DataType::UInt32.to_string()
                        && actual == actual_dtype
            ));
        }
    }

    #[test]
    fn rejects_nonnumeric_loadings() {
        for loading in [
            Column::new("time".into(), ["1", "2"]),
            Column::new("time".into(), [true, false]),
        ] {
            let inputs = [Column::new("firm".into(), [0_u32, 1]), loading];
            assert!(matches!(
                GroupedTransform.transform(&inputs),
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
            GroupedTransform.transform(&inputs),
            Err(TransformError::Polars(PolarsError::ShapeMismatch(_)))
        ));
    }

    #[test]
    fn enum_delegates_stateless_materialization() -> Result<(), TransformError> {
        let transform = Transform::G(GroupedTransform);
        assert_eq!(transform.name(), "G");
        assert!(transform.expands());
        assert_eq!(transform.argument_count(), ArgumentCount::Exactly(2));
        assert!(transform.parameters().is_empty());

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
