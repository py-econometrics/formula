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

pub(super) fn argument_column_name(index: usize) -> PlSmallStr {
    format!("__arg_{index}").into()
}

pub(super) fn input_frame(inputs: &[Column]) -> Result<DataFrame, TransformError> {
    let Some(first) = inputs.first() else {
        return Err(TransformError::input_count_mismatch(
            ArgumentCount::AtLeast(1),
            inputs.len(),
        ));
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
        return Err(TransformError::input_count_mismatch(
            ArgumentCount::Exactly(1),
            inputs.len(),
        ));
    };
    Ok(input)
}

pub(crate) fn validate_numeric_input(input: &Column) -> Result<(), TransformError> {
    if !input.dtype().is_primitive_numeric() {
        return Err(TransformError::non_numeric_input(input));
    }

    Ok(())
}
