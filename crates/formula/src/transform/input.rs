use super::TransformError;
use polars::prelude::{Column, DataFrame, Field, PlSmallStr};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArgumentCount {
    Exactly(usize),
    AtLeast(usize),
}

impl ArgumentCount {
    pub(crate) fn accepts(self, actual: usize) -> bool {
        match self {
            Self::Exactly(expected) => actual == expected,
            Self::AtLeast(expected) => actual >= expected,
        }
    }
}

impl std::fmt::Display for ArgumentCount {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Exactly(count) => write!(f, "exactly {count}"),
            Self::AtLeast(count) => write!(f, "at least {count}"),
        }
    }
}

pub(super) fn argument_column_name(index: usize) -> PlSmallStr {
    format!("__arg_{index}").into()
}

#[derive(Debug)]
pub(crate) struct PreparedInput {
    pub(super) frame: DataFrame,
    pub(super) fields: Vec<Field>,
}

impl PreparedInput {
    pub(super) fn new(inputs: &[Column], deduplicate: bool) -> Result<Self, TransformError> {
        if deduplicate {
            let mut seen = std::collections::HashSet::new();
            let unique = inputs
                .iter()
                .filter(|input| seen.insert(input.name()))
                .cloned()
                .collect::<Vec<_>>();

            return Self::new(&unique, false);
        }

        let fields = inputs
            .iter()
            .map(|input| input.field().into_owned())
            .collect();

        let columns = inputs
            .iter()
            .enumerate()
            // Positional names allow repeated arguments and avoid internal collisions.
            .map(|(index, input)| {
                let mut column = input.clone();
                column.rename(argument_column_name(index));
                column
            })
            .collect();

        Ok(Self {
            frame: DataFrame::new(inputs.first().map_or(0, Column::len), columns)?,
            fields,
        })
    }

    pub(super) fn single(&self) -> Result<(&Field, &Column), TransformError> {
        let [field] = self.fields.as_slice() else {
            return Err(TransformError::InputCountMismatch {
                expected: ArgumentCount::Exactly(1),
                actual: self.fields.len(),
            });
        };
        Ok((field, &self.frame.columns()[0]))
    }
}

pub(crate) fn validate_numeric_field(field: &Field) -> Result<(), TransformError> {
    if !field.dtype().is_primitive_numeric() {
        return Err(TransformError::NonNumericInput {
            name: field.name().clone(),
            actual: field.dtype().to_string(),
        });
    }
    Ok(())
}
