use polars::prelude::{DataFrame, Scalar};
use std::collections::BTreeMap;

#[derive(Clone, Debug)]
pub enum StateValue {
    Scalar(Scalar),
    DataFrame(DataFrame),
}

#[derive(Clone, Debug, Default)]
pub struct TransformState {
    fields: BTreeMap<String, StateValue>,
}

impl TransformState {
    pub fn new(fields: impl IntoIterator<Item = (String, StateValue)>) -> Self {
        Self {
            fields: fields.into_iter().collect(),
        }
    }
    pub fn get(&self, name: &str) -> Option<&StateValue> {
        self.fields.get(name)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, &StateValue)> {
        self.fields
            .iter()
            .map(|(name, value)| (name.as_str(), value))
    }
}

pub(crate) trait ExportState {
    fn export_state(&self) -> TransformState;
}
