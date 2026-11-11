use polars::prelude::{Column, Scalar};
use pyo3::exceptions::{PyIndexError, PyKeyError, PySyntaxError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use pyo3_polars::{PyDataFrame, PySeries};
use std::collections::BTreeMap;

#[pymodule]
mod _formula {
    #[pymodule_export]
    use super::{
        Factor, FactorContribution, FittedFormula, FittedFormulas, Formula, FormulaColumn,
        Formulas, Role, TransformCall, parse, parse_many,
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

    fn fit(&self, py: Python<'_>, data: PyDataFrame) -> PyResult<FittedFormulas> {
        py.detach(|| self.inner.fit(&data.0))
            .map(|inner| FittedFormulas { inner })
            .map_err(|error| PyValueError::new_err(error.to_string()))
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
    parameters: BTreeMap<String, Scalar>,
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
            parameters.set_item(name, scalar_to_python(py, value)?)?;
        }

        Ok(parameters)
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
