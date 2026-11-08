//! Allocation-sensitive O(days * assets) holding-period simulation.
//! The optimizer, input validation and evaluation remain in Python.

use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyfunction]
fn simulate_holding(
    returns: PyReadonlyArray2<'_, f64>,
    weights: PyReadonlyArray1<'_, f64>,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    let returns = returns.as_array();
    let weights = weights.as_array();
    let (days, assets) = returns.dim();

    if days == 0 || assets == 0 || weights.len() != assets {
        return Err(PyValueError::new_err("invalid holding period dimensions"));
    }
    if weights.iter().any(|&w| !w.is_finite() || w < 0.0) {
        return Err(PyValueError::new_err("weights must be finite and non-negative"));
    }

    // Each asset keeps its drifted balance; no full days-by-assets matrix
    // is allocated. The first daily return compares wealth with 1.0.
    let mut balances = weights.to_vec();
    let mut gross = Vec::with_capacity(days);
    let mut previous = 1.0_f64;

    for daily in returns.outer_iter() {
        for (balance, &r) in balances.iter_mut().zip(daily.iter()) {
            if !r.is_finite() || r < -1.0 {
                return Err(PyValueError::new_err("invalid asset return"));
            }
            *balance *= 1.0 + r;
        }
        let total: f64 = balances.iter().sum();
        if !total.is_finite() || total <= 0.0 {
            return Err(PyValueError::new_err("portfolio value must remain positive"));
        }
        gross.push(total / previous - 1.0);
        previous = total;
    }

    for balance in &mut balances {
        *balance /= previous;
    }
    Ok((gross, balances))
}

#[pymodule]
fn _quantalloc_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(simulate_holding, m)?)?;
    Ok(())
}
