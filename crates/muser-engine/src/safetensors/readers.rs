use std::io::{Read, Seek, SeekFrom};

use super::{SafeDtype, SafeTensorsError, SafeTensorsFile, TensorInfo};

impl SafeTensorsFile {
    pub fn tensor(&self, name: &str) -> Result<&TensorInfo, SafeTensorsError> {
        self.tensors
            .get(name)
            .ok_or_else(|| SafeTensorsError::TensorNotFound { name: name.into() })
    }

    pub fn read_tensor_raw<R: Read + Seek>(
        &self,
        reader: &mut R,
        name: &str,
    ) -> Result<Vec<u8>, SafeTensorsError> {
        let info = self.tensor(name)?;
        let mut bytes = vec![0; info.data_end - info.data_start];
        reader.seek(SeekFrom::Start(self.data_offset + info.data_start as u64))?;
        reader.read_exact(&mut bytes)?;
        Ok(bytes)
    }

    pub fn read_tensor_f32<R: Read + Seek>(
        &self,
        reader: &mut R,
        name: &str,
    ) -> Result<Vec<f32>, SafeTensorsError> {
        let info = self.tensor(name)?;
        let raw = self.read_tensor_raw(reader, name)?;
        match info.dtype {
            SafeDtype::F32 => Ok(raw
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()),
            SafeDtype::F16 => Ok(raw
                .chunks_exact(2)
                .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                .collect()),
            SafeDtype::BF16 => Ok(raw
                .chunks_exact(2)
                .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                .collect()),
            other => Err(SafeTensorsError::UnknownDtype(format!(
                "cannot convert {other} to f32"
            ))),
        }
    }
}
