use pyo3::exceptions::{PyIndexError, PySyntaxError, PyValueError};
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;

#[pymodule]
mod _formula {
    #[pymodule_export]
    use super::{Model, Models, Role, Specification, parse, parse_many};
}

#[pyclass(frozen, module = "formula")]
struct Specification {
    inner: formula::Specification,
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
struct Models {
    inner: formula::Models,
}

#[pymethods]
impl Specification {
    fn materialize(&self, py: Python<'_>, data: PyDataFrame) -> PyResult<Models> {
        py.detach(|| self.inner.materialize(&data.0))
            .map(|inner| Models { inner })
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }
}

#[pymethods]
impl Models {
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __getitem__(&self, index: isize) -> PyResult<Model> {
        let index = if index < 0 {
            self.inner.len().checked_add_signed(index)
        } else {
            Some(index as usize)
        };

        index
            .and_then(|index| self.inner.iter().nth(index))
            .map(|(_, inner)| Model {
                inner: inner.clone(),
            })
            .ok_or_else(|| PyIndexError::new_err("model index out of range"))
    }
}

#[pyclass(frozen, module = "formula")]
struct Model {
    inner: formula::Model,
}

#[pymethods]
impl Model {
    fn frame(&self, role: Role) -> Option<PyDataFrame> {
        self.inner
            .expression(role.into())
            .map(|expression| PyDataFrame(expression.frame().clone()))
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
