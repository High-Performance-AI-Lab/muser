use std::io::{self, Read};

const MAX_GGUF_STRING_BYTES: u64 = 16 * 1024 * 1024;

pub(super) fn read_u8<R: Read>(r: &mut R) -> io::Result<u8> {
    let mut buf = [0u8; 1];
    r.read_exact(&mut buf)?;
    Ok(buf[0])
}

pub(super) fn read_u16<R: Read>(r: &mut R) -> io::Result<u16> {
    let mut buf = [0u8; 2];
    r.read_exact(&mut buf)?;
    Ok(u16::from_le_bytes(buf))
}

pub(super) fn read_u32<R: Read>(r: &mut R) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

pub(super) fn read_u64<R: Read>(r: &mut R) -> io::Result<u64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

pub(super) fn read_i8<R: Read>(r: &mut R) -> io::Result<i8> {
    Ok(read_u8(r)? as i8)
}

pub(super) fn read_i16<R: Read>(r: &mut R) -> io::Result<i16> {
    Ok(read_u16(r)? as i16)
}

pub(super) fn read_i32<R: Read>(r: &mut R) -> io::Result<i32> {
    Ok(read_u32(r)? as i32)
}

pub(super) fn read_i64<R: Read>(r: &mut R) -> io::Result<i64> {
    Ok(read_u64(r)? as i64)
}

pub(super) fn read_f32<R: Read>(r: &mut R) -> io::Result<f32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(f32::from_le_bytes(buf))
}

pub(super) fn read_f64<R: Read>(r: &mut R) -> io::Result<f64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(f64::from_le_bytes(buf))
}

pub(super) fn read_string<R: Read>(r: &mut R) -> io::Result<String> {
    let raw_len = read_u64(r)?;
    if raw_len > MAX_GGUF_STRING_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "GGUF string exceeds the bounded parser limit",
        ));
    }
    let len = usize::try_from(raw_len).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "GGUF string length does not fit this platform",
        )
    })?;
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    String::from_utf8(buf).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

pub(super) fn read_bool<R: Read>(r: &mut R) -> io::Result<bool> {
    Ok(read_u8(r)? != 0)
}
