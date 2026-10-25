use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct Results {
    pub rows: usize,
    pub threads: Option<String>,
    pub samples: Vec<Sample>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Operation {
    Parse,
    ParseAndFit,
    Evaluate,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Sample {
    pub benchmark: String,
    pub formula: String,
    pub operation: Operation,
    pub sample: usize,
    pub iterations: usize,
    pub elapsed_ms: f64,
    pub output: Output,
}

#[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Output {
    Parsed { formulas: usize },
    Materialized { models: Vec<ModelShape> },
}

#[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelShape {
    pub left: (usize, usize),
    pub right: (usize, usize),
}
