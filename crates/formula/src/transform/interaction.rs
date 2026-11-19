//! Fixest-style `i(f)`, `i(f, x)` and `i(f, g)`.
//!
//! `i()` is a single factor of its term, but it has one axis per argument: the
//! first is always categorical, and the second is a numeric slope or another
//! categorical. Each categorical axis is encoded like `C()`, with `f::level`
//! column names, so structural coding treats `i(f, g)` as the interaction `f:g`.
//! The call itself only passes its inputs through.

use super::categorical::Bins;
use super::input::PreparedInput;
use super::{
    ArgumentCount, CategoricalDefinition, ParameterValue, TransformDefinition, TransformError,
    call_name, render_parameter, render_scalar,
};
use crate::formula::{ComponentKey, MaterializedColumns};
use polars::prelude::{PlSmallStr, Scalar};
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct InteractionDefinition {
    reference: Option<ParameterValue>,
    reference2: Option<ParameterValue>,
    bins: Option<Bins>,
    bins2: Option<Bins>,
}

impl InteractionDefinition {
    pub(crate) const REFERENCE_PARAMETER: &str = "ref";
    pub(crate) const REFERENCE2_PARAMETER: &str = "ref2";
    pub(crate) const BIN_PARAMETER: &str = "bin";
    pub(crate) const BIN2_PARAMETER: &str = "bin2";

    /// Validates the options of `i(...)` with `arguments` positional arguments.
    pub(crate) fn new(
        arguments: usize,
        reference: Option<ParameterValue>,
        reference2: Option<ParameterValue>,
        bins: Option<ParameterValue>,
        bins2: Option<ParameterValue>,
    ) -> Result<Self, TransformError> {
        let reference = Self::references(Self::REFERENCE_PARAMETER, reference)?;
        let reference2 = Self::references(Self::REFERENCE2_PARAMETER, reference2)?;
        let bins = Self::bins(Self::BIN_PARAMETER, bins)?;
        let bins2 = Self::bins(Self::BIN2_PARAMETER, bins2)?;
        if arguments < 2 && (reference2.is_some() || bins2.is_some()) {
            return Err(TransformError::InvalidParameter(format!(
                "`{}` and `{}` require a second variable",
                Self::REFERENCE2_PARAMETER,
                Self::BIN2_PARAMETER
            )));
        }
        Ok(Self {
            reference,
            reference2,
            bins,
            bins2,
        })
    }

    fn references(
        name: &str,
        value: Option<ParameterValue>,
    ) -> Result<Option<ParameterValue>, TransformError> {
        match value {
            Some(ParameterValue::Map(_)) => Err(TransformError::InvalidParameter(format!(
                "`{name}` must be a value or a list of values"
            ))),
            value => Ok(value),
        }
    }

    fn bins(name: &str, value: Option<ParameterValue>) -> Result<Option<Bins>, TransformError> {
        let bins = match value {
            None => return Ok(None),
            Some(ParameterValue::Map(bins)) => bins,
            Some(_) => {
                return Err(TransformError::InvalidParameter(format!(
                    "`{name}` must be a mapping such as {{'new': ['old']}}"
                )));
            }
        };
        if bins.is_empty() {
            return Err(TransformError::InvalidParameter(format!(
                "`{name}` must not be empty"
            )));
        }
        let mut labels: Vec<&Scalar> = Vec::new();
        let mut seen: Vec<&Scalar> = Vec::new();
        for (label, values) in &bins {
            if labels.contains(&label) {
                return Err(TransformError::InvalidParameter(format!(
                    "`{name}` names the bin label {} more than once",
                    render_scalar(label)
                )));
            }
            labels.push(label);
            if values.is_empty() {
                return Err(TransformError::InvalidParameter(format!(
                    "each bin of `{name}` must list at least one value"
                )));
            }
            for value in values {
                if seen.contains(&value) {
                    return Err(TransformError::InvalidParameter(format!(
                        "`{name}` assigns the value {} to more than one bin",
                        render_scalar(value)
                    )));
                }
                seen.push(value);
            }
        }
        Ok(Some(bins))
    }

    /// Whether the options force a second variable to be categorical.
    pub(crate) fn second_is_categorical(&self) -> bool {
        self.reference2.is_some() || self.bins2.is_some()
    }

    /// The encoding of the first variable. `C(f, omit=...)` as the first variable
    /// supplies its omissions unless `ref` is given.
    pub(crate) fn first(&self, omitted: Option<ParameterValue>) -> CategoricalDefinition {
        CategoricalDefinition::interaction(
            false,
            self.reference.clone().or(omitted),
            self.bins.clone(),
        )
    }

    /// The encoding of a categorical second variable. `C(g, omit=...)` as the
    /// second variable supplies its omissions unless `ref2` is given.
    pub(crate) fn second(&self, omitted: Option<ParameterValue>) -> CategoricalDefinition {
        CategoricalDefinition::interaction(
            true,
            self.reference2.clone().or(omitted),
            self.bins2.clone(),
        )
    }

    /// `i()` contributes its inputs; its term encodes them.
    pub(crate) fn transform(inputs: &PreparedInput) -> MaterializedColumns<ComponentKey> {
        let metadata = vec![ComponentKey::None; inputs.frame.width()];
        MaterializedColumns::new(inputs.frame.clone(), metadata)
    }

    fn named_parameters(&self) -> impl Iterator<Item = (&'static str, ParameterValue)> + '_ {
        [
            (Self::REFERENCE_PARAMETER, self.reference.clone()),
            (Self::REFERENCE2_PARAMETER, self.reference2.clone()),
            (
                Self::BIN_PARAMETER,
                self.bins.clone().map(ParameterValue::Map),
            ),
            (
                Self::BIN2_PARAMETER,
                self.bins2.clone().map(ParameterValue::Map),
            ),
        ]
        .into_iter()
        .filter_map(|(name, value)| Some((name, value?)))
    }
}

impl TransformDefinition for InteractionDefinition {
    fn name(&self) -> &'static str {
        "i"
    }

    fn expands(&self) -> bool {
        true
    }

    fn argument_count(&self) -> ArgumentCount {
        ArgumentCount::Between(1, 2)
    }

    fn parameters(&self) -> BTreeMap<&'static str, ParameterValue> {
        self.named_parameters().collect()
    }

    fn call_name(&self, arguments: &[&str]) -> PlSmallStr {
        let parameters = self
            .named_parameters()
            .map(|(name, value)| format!("{name}={}", render_parameter(&value)))
            .collect::<Vec<_>>();
        call_name(
            self.name(),
            arguments
                .iter()
                .copied()
                .chain(parameters.iter().map(String::as_str)),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use polars::prelude::PlSmallStr;

    #[test]
    fn interaction_calls_describe_themselves() -> Result<(), TransformError> {
        let reference = ParameterValue::List(vec![Scalar::from(-1_i64), Scalar::from(0_i64)]);
        let bins = ParameterValue::Map(vec![(
            Scalar::from(PlSmallStr::from_static("ab")),
            vec![
                Scalar::from(PlSmallStr::from_static("a")),
                Scalar::from(PlSmallStr::from_static("b")),
            ],
        )]);
        let definition =
            InteractionDefinition::new(2, Some(reference.clone()), None, None, Some(bins.clone()))?;
        assert!(definition.expands());
        assert_eq!(definition.argument_count(), ArgumentCount::Between(1, 2));
        assert_eq!(definition.argument_count().to_string(), "between 1 and 2");
        assert!(!definition.argument_count().accepts(3));
        assert_eq!(
            definition.parameters(),
            BTreeMap::from([("ref", reference), ("bin2", bins)])
        );
        assert!(definition.second_is_categorical());
        assert_eq!(
            definition.call_name(&["t", "g"]),
            "i(t, g, ref=[-1, 0], bin2={'ab': ['a', 'b']})"
        );
        Ok(())
    }
}
