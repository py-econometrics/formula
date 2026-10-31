use pyo3::exceptions::{PyIndexError, PyKeyError, PySyntaxError, PyValueError};
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;

#[pymodule]
mod _formula {
    #[pymodule_export]
    use super::{Model, Models, Role, Specification, parse, parse_many};
}

#[pyfunction]
fn parse(formulas: &str) -> PyResult<Specification> {
    formula::parse(formulas)
        .map(|inner| Specification { inner })
        .map_err(|error| PySyntaxError::new_err(error.to_string()))
}

#[pyfunction]
fn parse_many(formulas: Vec<String>) -> PyResult<Specification> {
    formula::parse_many(formulas.iter().map(String::as_str))
        .map(|inner| Specification { inner })
        .map_err(|error| PySyntaxError::new_err(error.to_string()))
}

#[pyclass(frozen, module = "formula")]
struct Specification {
    inner: formula::Specification,
}

#[pymethods]
impl Specification {
    fn materialize(&self, py: Python<'_>, data: PyDataFrame) -> PyResult<Models> {
        py.detach(|| self.inner.materialize(&data.0))
            .map(|inner| Models { inner })
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }
}

#[pyclass(frozen, module = "formula")]
struct Models {
    inner: formula::Models,
}

#[pymethods]
impl Models {
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __getitem__(&self, key: ModelKey) -> PyResult<Model> {
        let inner = match key {
            ModelKey::Index(index) => {
                let index = if index < 0 {
                    self.inner.len().checked_add_signed(index)
                } else {
                    Some(index as usize)
                };

                index
                    .and_then(|index| self.inner.iter().nth(index))
                    .ok_or_else(|| PyIndexError::new_err("model index out of range"))?
            }
            ModelKey::Formula(source) => self
                .inner
                .get_by_str(&source)
                .map_err(|error| PySyntaxError::new_err(error.to_string()))?
                .ok_or_else(|| PyKeyError::new_err(source))?,
        };

        Ok(Model {
            inner: inner.clone(),
        })
    }

    fn evaluate(&self, py: Python<'_>, data: PyDataFrame) -> PyResult<Self> {
        py.detach(|| self.inner.evaluate(&data.0))
            .map(|inner| Self { inner })
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    fn select(&self, roles: Vec<Role>) -> PyResult<Self> {
        let roles = roles.into_iter().map(Into::into).collect::<Vec<_>>();

        self.inner
            .select(&roles)
            .map(|inner| Self { inner })
            .map_err(|error| match &error {
                formula::SelectionError::EmptySelection => PyValueError::new_err(error.to_string()),
                formula::SelectionError::MissingRole(_) => PyKeyError::new_err(error.to_string()),
            })
    }
}

#[pyclass(frozen, module = "formula")]
struct Model {
    inner: formula::Model,
}

#[pymethods]
impl Model {
    #[getter]
    fn formula(&self) -> String {
        self.inner.formula().to_string()
    }

    fn frame(&self, role: Role) -> Option<PyDataFrame> {
        self.inner
            .expression(role.into())
            .map(|expression| PyDataFrame(expression.frame().clone()))
    }

    fn evaluate(&self, py: Python<'_>, data: PyDataFrame) -> PyResult<Self> {
        py.detach(|| self.inner.evaluate(&data.0))
            .map(|inner| Self { inner })
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    fn select(&self, roles: Vec<Role>) -> PyResult<Self> {
        let roles = roles.into_iter().map(Into::into).collect::<Vec<_>>();

        self.inner
            .select(&roles)
            .map(|inner| Self { inner })
            .map_err(|error| match &error {
                formula::SelectionError::EmptySelection => PyValueError::new_err(error.to_string()),
                formula::SelectionError::MissingRole(_) => PyKeyError::new_err(error.to_string()),
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
enum ModelKey {
    Index(isize),
    Formula(String),
}
