use polars::prelude::{Column, Scalar};
use pyo3::create_exception;
use pyo3::exceptions::{PyIndexError, PyKeyError, PySyntaxError, PyUserWarning, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyIterator, PyList, PyTuple};
use pyo3_polars::{PyDataFrame, PySeries};
use std::collections::BTreeMap;
use std::ffi::CString;

create_exception!(
    formula,
    RankDeficiencyWarning,
    PyUserWarning,
    "Terms without columns, because the intercept, fixed effects, or other terms already span them."
);

#[pymodule]
mod _formula {
    #[pymodule_export]
    use super::{
        Factor, FactorContribution, FittedFormula, FittedFormulas, FittedTerm, FixedEffect,
        FixedEffectEncoding, Formula, FormulaColumn, Formulas, RankDeficiencyWarning, Role,
        TransformCall, TransformState, parse, parse_many,
    };
}

#[pyfunction]
fn parse(formulas: &str) -> PyResult<Formulas> {
    formula::parse(formulas)
        .map(|inner| Formulas { inner })
        .map_err(|error| PySyntaxError::new_err(error.to_string()))
}

#[pyfunction]
fn parse_many(formulas: Vec<String>) -> PyResult<Formulas> {
    formula::parse_many(formulas.iter().map(String::as_str))
        .map(|inner| Formulas { inner })
        .map_err(|error| PySyntaxError::new_err(error.to_string()))
}

#[pyclass(frozen, module = "formula")]
struct Formula {
    inner: formula::Formula,
}

#[pymethods]
impl Formula {
    fn __str__(&self) -> String {
        self.inner.to_string()
    }
}

#[pyclass(frozen, module = "formula")]
struct Formulas {
    inner: formula::Formulas,
}

#[pymethods]
impl Formulas {
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __getitem__(&self, key: FormulaKey) -> PyResult<Formula> {
        let inner = match key {
            FormulaKey::Index(index) => {
                let index = if index < 0 {
                    self.inner.len().checked_add_signed(index)
                } else {
                    Some(index as usize)
                };

                index
                    .and_then(|index| self.inner.iter().nth(index))
                    .ok_or_else(|| PyIndexError::new_err("formula index out of range"))?
            }
            FormulaKey::Formula(source) => self
                .inner
                .get_by_str(&source)
                .map_err(|error| PySyntaxError::new_err(error.to_string()))?
                .ok_or_else(|| PyKeyError::new_err(source))?,
        };

        Ok(Formula {
            inner: inner.clone(),
        })
    }

    #[pyo3(signature = (data, *, ensure_full_rank = true))]
    fn fit(
        &self,
        py: Python<'_>,
        data: PyDataFrame,
        ensure_full_rank: bool,
    ) -> PyResult<FittedFormulas> {
        let options = formula::FitOptions { ensure_full_rank };
        let inner = py
            .detach(|| self.inner.fit_with_options(&data.0, options))
            .map_err(|error| PyValueError::new_err(error.to_string()))?;

        // A child formula can be shared by several models; warn about it once.
        let mut messages = Vec::new();
        for model in &inner {
            for deficiency in model.rank_deficiencies() {
                let message = deficiency.to_string();
                if !messages.contains(&message) {
                    messages.push(message);
                }
            }
        }
        let category = py.get_type::<RankDeficiencyWarning>();
        for message in messages {
            let message =
                CString::new(message).map_err(|error| PyValueError::new_err(error.to_string()))?;
            PyErr::warn(py, &category, &message, 1)?;
        }

        Ok(FittedFormulas { inner })
    }
}

#[pyclass(frozen, module = "formula")]
struct FittedFormulas {
    inner: formula::FittedFormulas,
}

#[pymethods]
impl FittedFormulas {
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __getitem__(&self, key: FormulaKey) -> PyResult<FittedFormula> {
        let inner = match key {
            FormulaKey::Index(index) => {
                let index = if index < 0 {
                    self.inner.len().checked_add_signed(index)
                } else {
                    Some(index as usize)
                };

                index
                    .and_then(|index| self.inner.iter().nth(index))
                    .ok_or_else(|| PyIndexError::new_err("model index out of range"))?
            }
            FormulaKey::Formula(source) => self
                .inner
                .get_by_str(&source)
                .map_err(|error| PySyntaxError::new_err(error.to_string()))?
                .ok_or_else(|| PyKeyError::new_err(source))?,
        };

        Ok(FittedFormula {
            inner: inner.clone(),
        })
    }

    fn transform(&self, py: Python<'_>, data: PyDataFrame) -> PyResult<Self> {
        py.detach(|| self.inner.transform(&data.0))
            .map(|inner| Self { inner })
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    fn select(&self, roles: RoleSelection) -> PyResult<Self> {
        let roles = roles.into_roles();

        self.inner
            .select(&roles)
            .map(|inner| Self { inner })
            .map_err(|error| match &error {
                formula::SelectionError::EmptySelection => PyValueError::new_err(error.to_string()),
                formula::SelectionError::MissingRole(_) => PyKeyError::new_err(error.to_string()),
                formula::SelectionError::InvalidQuery(_) => {
                    PySyntaxError::new_err(error.to_string())
                }
                formula::SelectionError::MissingTerm(_) => PyKeyError::new_err(error.to_string()),
                formula::SelectionError::Frame(_) => PyValueError::new_err(error.to_string()),
            })
    }
}

#[pyclass(frozen, module = "formula")]
struct FittedFormula {
    inner: formula::FittedFormula,
}

#[pymethods]
impl FittedFormula {
    #[getter]
    fn formula(&self) -> String {
        self.inner.formula().to_string()
    }

    fn frame(&self) -> PyResult<PyDataFrame> {
        self.inner
            .frame()
            .map(PyDataFrame)
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    fn transform(&self, py: Python<'_>, data: PyDataFrame) -> PyResult<Self> {
        py.detach(|| self.inner.transform(&data.0))
            .map(|inner| Self { inner })
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    fn select(&self, roles: RoleSelection) -> PyResult<Self> {
        let roles = roles.into_roles();

        self.inner
            .select(&roles)
            .map(|inner| Self { inner })
            .map_err(|error| match &error {
                formula::SelectionError::EmptySelection => PyValueError::new_err(error.to_string()),
                formula::SelectionError::MissingRole(_) => PyKeyError::new_err(error.to_string()),
                formula::SelectionError::InvalidQuery(_) => {
                    PySyntaxError::new_err(error.to_string())
                }
                formula::SelectionError::MissingTerm(_) => PyKeyError::new_err(error.to_string()),
                formula::SelectionError::Frame(_) => PyValueError::new_err(error.to_string()),
            })
    }

    fn select_terms(&self, source: &str, role: Role) -> PyResult<Self> {
        self.inner
            .select_terms(source, role.into())
            .map(|inner| Self { inner })
            .map_err(|error| match &error {
                formula::SelectionError::EmptySelection => PyValueError::new_err(error.to_string()),
                formula::SelectionError::MissingRole(_) => PyKeyError::new_err(error.to_string()),
                formula::SelectionError::InvalidQuery(_) => {
                    PySyntaxError::new_err(error.to_string())
                }
                formula::SelectionError::MissingTerm(_) => PyKeyError::new_err(error.to_string()),
                formula::SelectionError::Frame(_) => PyValueError::new_err(error.to_string()),
            })
    }

    fn children(&self, role: Role) -> PyResult<FittedFormulas> {
        self.inner
            .children(role.into())
            .map(|children| FittedFormulas {
                inner: children.iter().cloned().collect(),
            })
            .map_err(|error| PyKeyError::new_err(error.to_string()))
    }

    fn columns(&self, role: Role) -> PyResult<Vec<FormulaColumn>> {
        self.inner
            .columns(role.into())
            .map(|columns| columns.map(FormulaColumn::from).collect())
            .map_err(|error| match &error {
                formula::SelectionError::EmptySelection => PyValueError::new_err(error.to_string()),
                formula::SelectionError::MissingRole(_) => PyKeyError::new_err(error.to_string()),
                formula::SelectionError::InvalidQuery(_) => {
                    PySyntaxError::new_err(error.to_string())
                }
                formula::SelectionError::MissingTerm(_) => PyKeyError::new_err(error.to_string()),
                formula::SelectionError::Frame(_) => PyValueError::new_err(error.to_string()),
            })
    }

    fn terms(&self, role: Role) -> PyResult<Vec<FittedTerm>> {
        self.inner
            .terms(role.into())
            .map(|terms| terms.iter().map(FittedTerm::from).collect())
            .map_err(|error| match &error {
                formula::SelectionError::EmptySelection => PyValueError::new_err(error.to_string()),
                formula::SelectionError::MissingRole(_) => PyKeyError::new_err(error.to_string()),
                formula::SelectionError::InvalidQuery(_) => {
                    PySyntaxError::new_err(error.to_string())
                }
                formula::SelectionError::MissingTerm(_) => PyKeyError::new_err(error.to_string()),
                formula::SelectionError::Frame(_) => PyValueError::new_err(error.to_string()),
            })
    }

    #[getter]
    fn fixed_effects(&self) -> Vec<FixedEffect> {
        self.inner
            .terms(formula::Role::FixedEffect)
            .unwrap_or_default()
            .iter()
            .filter_map(formula::FittedTerm::as_fixed_effect)
            .map(|inner| FixedEffect { inner })
            .collect()
    }

    fn fixed_effect(&self, name: &str) -> PyResult<FixedEffect> {
        self.fixed_effects()
            .into_iter()
            .find(|effect| effect.inner.render() == name)
            .ok_or_else(|| PyKeyError::new_err(name.to_owned()))
    }
}

#[pyclass(frozen, eq, from_py_object, module = "formula")]
#[derive(Clone, Copy, PartialEq, Eq)]
enum Role {
    Dependent,
    Independent,
    FixedEffect,
}

impl From<Role> for formula::Role {
    fn from(role: Role) -> Self {
        match role {
            Role::Dependent => Self::Dependent,
            Role::Independent => Self::Independent,
            Role::FixedEffect => Self::FixedEffect,
        }
    }
}

#[derive(FromPyObject)]
enum RoleSelection {
    Single(Role),
    Multiple(Vec<Role>),
}

impl RoleSelection {
    fn into_roles(self) -> Vec<formula::Role> {
        match self {
            Self::Single(role) => vec![role.into()],
            Self::Multiple(roles) => roles.into_iter().map(Into::into).collect(),
        }
    }
}

#[derive(FromPyObject)]
enum FormulaKey {
    Index(isize),
    Formula(String),
}

#[pyclass(frozen, skip_from_py_object, module = "formula")]
#[derive(Clone, Debug)]
struct TransformCall {
    name: String,
    inputs: Vec<Factor>,
    parameters: BTreeMap<String, formula::ParameterValue>,
    state: Option<formula::TransformState>,
}

impl From<formula::TransformCall<'_>> for TransformCall {
    fn from(call: formula::TransformCall<'_>) -> Self {
        Self {
            name: call.name().to_owned(),
            inputs: call.inputs().map(Factor::from).collect(),
            parameters: call
                .parameters()
                .into_iter()
                .map(|(name, value)| (name.to_owned(), value))
                .collect(),
            state: call.state(),
        }
    }
}

#[pymethods]
impl TransformCall {
    fn __repr__(&self) -> String {
        format!("{self:#?}")
    }

    #[getter]
    fn name(&self) -> &str {
        &self.name
    }

    #[getter]
    fn inputs(&self) -> Vec<Factor> {
        self.inputs.clone()
    }

    #[getter]
    fn parameters<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let parameters = PyDict::new(py);

        for (name, value) in &self.parameters {
            match value {
                formula::ParameterValue::Scalar(value) => {
                    parameters.set_item(name, scalar_to_python(py, value)?)?;
                }
                formula::ParameterValue::List(values) => {
                    parameters.set_item(name, scalars_to_python(py, values)?)?;
                }
                formula::ParameterValue::Map(entries) => {
                    let map = PyDict::new(py);
                    for (label, values) in entries {
                        map.set_item(scalar_to_python(py, label)?, scalars_to_python(py, values)?)?;
                    }
                    parameters.set_item(name, map)?;
                }
            }
        }

        Ok(parameters)
    }

    #[getter]
    fn state(&self) -> Option<TransformState> {
        self.state.clone().map(|inner| TransformState { inner })
    }
}

#[pyclass(frozen, skip_from_py_object, module = "formula")]
#[derive(Clone, Debug)]
struct TransformState {
    inner: formula::TransformState,
}

#[pymethods]
impl TransformState {
    fn __getitem__(&self, py: Python<'_>, name: &str) -> PyResult<Py<PyAny>> {
        let value = self
            .inner
            .get(name)
            .ok_or_else(|| PyKeyError::new_err(name.to_owned()))?;

        match value {
            formula::StateValue::Scalar(value) => scalar_to_python(py, value),
            formula::StateValue::DataFrame(frame) => {
                Ok(PyDataFrame(frame.clone()).into_pyobject(py)?.unbind())
            }
        }
    }

    fn __iter__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyIterator>> {
        let names = self.inner.iter().map(|(name, _)| name).collect::<Vec<_>>();
        PyTuple::new(py, names)?.try_iter()
    }

    fn __len__(&self) -> usize {
        self.inner.iter().count()
    }
}

#[pyclass(frozen, skip_from_py_object, module = "formula")]
#[derive(Clone, Debug)]
struct Factor {
    name: Option<String>,
    value: Option<Scalar>,
    transform: Option<TransformCall>,
}

impl From<&formula::Factor> for Factor {
    fn from(factor: &formula::Factor) -> Self {
        Self {
            name: factor.name().map(str::to_owned),
            value: factor.value().cloned(),
            transform: factor.transform().map(TransformCall::from),
        }
    }
}

#[pymethods]
impl Factor {
    fn __repr__(&self) -> String {
        format!("{self:#?}")
    }

    #[getter]
    fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    #[getter]
    fn value(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        self.value
            .as_ref()
            .map(|value| scalar_to_python(py, value))
            .transpose()
    }

    #[getter]
    fn transform(&self) -> Option<TransformCall> {
        self.transform.clone()
    }
}

#[pyclass(frozen, module = "formula")]
struct FittedTerm {
    inner: formula::FittedTerm,
}

impl From<&formula::FittedTerm> for FittedTerm {
    fn from(term: &formula::FittedTerm) -> Self {
        Self {
            inner: term.clone(),
        }
    }
}

#[pymethods]
impl FittedTerm {
    #[getter]
    fn factors<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let factors = self
            .inner
            .factors()
            .map(Factor::from)
            .map(|factor| Py::new(py, factor))
            .collect::<PyResult<Vec<_>>>()?;

        PyTuple::new(py, factors)
    }

    #[getter]
    fn columns<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let columns = self
            .inner
            .columns()
            .map(FormulaColumn::from)
            .map(|column| Py::new(py, column))
            .collect::<PyResult<Vec<_>>>()?;

        PyTuple::new(py, columns)
    }

    #[getter]
    fn fixed_effect(&self) -> Option<FixedEffect> {
        self.inner
            .as_fixed_effect()
            .map(|inner| FixedEffect { inner })
    }
}

#[pyclass(frozen, module = "formula")]
struct FormulaColumn {
    data: Column,
    term: Vec<Factor>,
    contributions: Vec<FactorContribution>,
}

impl From<formula::FormulaColumn<'_>> for FormulaColumn {
    fn from(column: formula::FormulaColumn<'_>) -> Self {
        Self {
            data: column.data().clone(),
            term: column.term().factors().map(Factor::from).collect(),
            contributions: column
                .contributions()
                .map(|(factor, key)| FactorContribution {
                    factor: Factor::from(factor),
                    key: key.clone(),
                })
                .collect(),
        }
    }
}

#[pyclass(frozen, skip_from_py_object, module = "formula")]
#[derive(Clone, Debug)]
struct FixedEffect {
    inner: formula::FittedFixedEffect,
}

#[pymethods]
impl FixedEffect {
    fn __repr__(&self) -> String {
        self.inner.render()
    }
    #[getter]
    fn codes(&self) -> PySeries {
        PySeries(self.inner.codes().as_materialized_series().clone())
    }

    #[getter]
    fn loadings(&self) -> PyDataFrame {
        PyDataFrame(self.inner.loadings().clone())
    }

    #[getter]
    fn intercept(&self) -> bool {
        self.inner.intercept()
    }

    #[getter]
    fn encoding(&self) -> FixedEffectEncoding {
        FixedEffectEncoding {
            inner: self.inner.encoding().clone(),
        }
    }
}

#[pyclass(frozen, skip_from_py_object, module = "formula")]
#[derive(Clone, Debug)]
struct FixedEffectEncoding {
    inner: formula::IntegerEncodingState,
}

#[pymethods]
impl FixedEffectEncoding {
    #[getter]
    fn codes(&self) -> PyDataFrame {
        PyDataFrame(self.inner.codes())
    }

    #[getter]
    fn levels(&self) -> PyDataFrame {
        PyDataFrame(self.inner.levels())
    }
}

#[pymethods]
impl FormulaColumn {
    #[getter]
    fn data(&self) -> PySeries {
        PySeries(self.data.as_materialized_series().clone())
    }

    #[getter]
    fn term<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let factors = self
            .term
            .iter()
            .cloned()
            .map(|factor| Py::new(py, factor))
            .collect::<PyResult<Vec<_>>>()?;

        PyTuple::new(py, factors)
    }

    #[getter]
    fn contributions<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let contributions = self
            .contributions
            .iter()
            .cloned()
            .map(|contribution| Py::new(py, contribution))
            .collect::<PyResult<Vec<_>>>()?;

        PyTuple::new(py, contributions)
    }
}

#[pyclass(frozen, skip_from_py_object, module = "formula")]
#[derive(Clone, Debug)]
struct FactorContribution {
    factor: Factor,
    key: formula::ComponentKey,
}

#[pymethods]
impl FactorContribution {
    fn __repr__(&self) -> String {
        format!("{self:#?}")
    }

    #[getter]
    fn factor(&self) -> Factor {
        self.factor.clone()
    }

    #[getter]
    fn key(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        component_key_to_python(py, &self.key)
    }
}

fn scalar_to_python(py: Python<'_>, value: &Scalar) -> PyResult<Py<PyAny>> {
    PySeries(value.clone().into_series("".into()))
        .into_pyobject(py)?
        .call_method0("item")
        .map(Bound::unbind)
}

fn scalars_to_python<'py>(py: Python<'py>, values: &[Scalar]) -> PyResult<Bound<'py, PyList>> {
    let values = values
        .iter()
        .map(|value| scalar_to_python(py, value))
        .collect::<PyResult<Vec<_>>>()?;
    PyList::new(py, values)
}

fn component_key_to_python(py: Python<'_>, key: &formula::ComponentKey) -> PyResult<Py<PyAny>> {
    match key {
        formula::ComponentKey::None => Ok(py.None()),
        formula::ComponentKey::Scalar(value) => scalar_to_python(py, value),
        formula::ComponentKey::Tuple(keys) => {
            let items = keys
                .iter()
                .map(|key| component_key_to_python(py, key))
                .collect::<PyResult<Vec<_>>>()?;

            Ok(PyTuple::new(py, items)?.into_any().unbind())
        }
    }
}
