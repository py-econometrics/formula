mod categorical;
mod expression;
mod input;
mod integer;

pub(crate) use categorical::CategoricalTransform;
pub(crate) use expression::ExpressionTransform;
pub use input::ArgumentCount;
pub(crate) use integer::IntegerEncodingTransform;

use polars::error::PolarsError;
use polars::prelude::{DataFrame, PlSmallStr, Series};

fn call_name<'a>(name: &str, arguments: impl IntoIterator<Item = &'a str>) -> PlSmallStr {
    let arguments = arguments.into_iter().collect::<Vec<_>>().join(", ");
    format!("{name}({arguments})").into()
}

#[derive(Clone, Debug, PartialEq)]
pub enum Transform {
    Expression(ExpressionTransform),
    C(CategoricalTransform),
    F(IntegerEncodingTransform),
}

pub(crate) trait TransformOperation {
    fn expands(&self) -> bool;
    fn argument_count(&self) -> ArgumentCount;
    fn fit(&mut self, input: &[Series]) -> Result<(), TransformError>;
    fn transform(&self, input: &[Series]) -> Result<DataFrame, TransformError>;
}

impl TransformOperation for Transform {
    fn expands(&self) -> bool {
        match self {
            Self::Expression(expression) => expression.expands(),
            Self::C(categorical) => categorical.expands(),
            Self::F(integer) => integer.expands(),
        }
    }

    fn argument_count(&self) -> ArgumentCount {
        match self {
            Self::Expression(expression) => expression.argument_count(),
            Self::C(categorical) => categorical.argument_count(),
            Self::F(integer) => integer.argument_count(),
        }
    }

    fn fit(&mut self, inputs: &[Series]) -> Result<(), TransformError> {
        match self {
            Self::Expression(expression) => expression.fit(inputs),
            Self::C(categorical) => categorical.fit(inputs),
            Self::F(integer) => integer.fit(inputs),
        }
    }

    fn transform(&self, inputs: &[Series]) -> Result<DataFrame, TransformError> {
        match self {
            Self::Expression(expression) => expression.transform(inputs),
            Self::C(categorical) => categorical.transform(inputs),
            Self::F(integer) => integer.transform(inputs),
        }
    }
}

#[derive(Debug)]
pub enum TransformError {
    NotFitted,
    InputTypeMismatch,
    InvalidParameter(String),
    Polars(PolarsError),
}

impl From<PolarsError> for TransformError {
    fn from(error: PolarsError) -> Self {
        Self::Polars(error)
    }
}
