use pyo3::prelude::*;

#[pymodule]
fn _formula(_module: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}
