use pyo3::exceptions::PySyntaxError;
use pyo3::prelude::*;

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

#[pymodule]
mod _formula {
    #[pymodule_export]
    use super::{Specification, parse, parse_many};
}
