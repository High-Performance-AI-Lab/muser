//! Minimal fail-closed SafeTensors reader extracted from Ferrite.

mod parser;
mod readers;
mod types;

pub use types::{SafeDtype, SafeTensorsError, SafeTensorsFile, TensorInfo};
