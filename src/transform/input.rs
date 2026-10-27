use super::TransformError;
use polars::prelude::{Column, DataFrame, PlSmallStr};

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

pub(super) fn input_frame(inputs: &[Column]) -> Result<DataFrame, TransformError> {
    let Some(first) = inputs.first() else {
        return Err(TransformError::InputCountMismatch {
            expected: ArgumentCount::AtLeast(1),
            actual: inputs.len(),
        });
    };

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

    Ok(DataFrame::new(first.len(), columns)?)
}

pub(super) fn validate_single_input(inputs: &[Column]) -> Result<&Column, TransformError> {
    let [input] = inputs else {
        return Err(TransformError::InputCountMismatch {
            expected: ArgumentCount::Exactly(1),
            actual: inputs.len(),
        });
    };
    Ok(input)
}

pub(crate) fn validate_numeric_input(input: &Column) -> Result<(), TransformError> {
    if !input.dtype().is_primitive_numeric() {
        return Err(TransformError::NonNumericInput {
            name: input.name().clone(),
            actual: input.dtype().to_string(),
        });
    }

    Ok(())
}
