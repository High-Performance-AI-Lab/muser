//! Shared-memory Metal buffers with checked CPU views.

use metal::{Buffer, BufferRef, MTLResourceOptions};

use super::context::{MetalContext, MetalError};

fn shared_tracked() -> MTLResourceOptions {
    // Several accepted Muse paths still cross compute encoders (notably
    // target-hidden prefill/capture). Untracked resources are only valid when
    // every such dependency has an explicit fence/barrier. b9678d4 enabled
    // untracked mode globally before that contract existed and empirically
    // changed DFlash conditioning while leaving final greedy IDs unchanged.
    MTLResourceOptions::StorageModeShared
}

fn page_size() -> usize {
    extern "C" {
        // POSIX `getpagesize()`, a stable libSystem symbol every macOS
        // process already links; declaring it here avoids pulling in the
        // `libc` crate as a new direct dependency just for this constant.
        fn getpagesize() -> std::os::raw::c_int;
    }
    // SAFETY: `getpagesize()` takes no arguments, has no preconditions, and
    // cannot fail.
    unsafe { getpagesize() as usize }
}

#[derive(Clone)]
pub struct GpuBuffer {
    inner: Buffer,
    len: usize,
}

/// Shared Metal storage whose logical elements are IEEE-754 binary16 bits.
///
/// Keeping this distinct from [`GpuBuffer`] prevents an F16 KV plane from
/// being accidentally fingerprinted or indexed as an F32 activation buffer.
#[derive(Clone)]
pub struct GpuHalfBuffer {
    inner: Buffer,
    len: usize,
}

#[derive(Clone)]
pub struct GpuBytes {
    inner: Buffer,
    len: usize,
    _mmap: Option<std::sync::Arc<memmap2::Mmap>>,
}

#[derive(Clone, Copy)]
pub struct GpuByteView<'a> {
    buffer: &'a GpuBytes,
    offset: usize,
    len: usize,
}

impl GpuBytes {
    pub fn zeros(context: &MetalContext, len: usize) -> Result<Self, MetalError> {
        let inner = context.device.new_buffer(len as u64, shared_tracked());
        if len > 0 && (inner.length() as usize != len || inner.contents().is_null()) {
            return Err(MetalError::Allocation(len));
        }
        if len > 0 {
            // SAFETY: this is a new exclusive shared allocation of `len` bytes.
            unsafe { std::ptr::write_bytes(inner.contents() as *mut u8, 0, len) };
        }
        Ok(Self {
            inner,
            len,
            _mmap: None,
        })
    }

    pub fn from_bytes(context: &MetalContext, values: &[u8]) -> Result<Self, MetalError> {
        let inner = context.device.new_buffer_with_data(
            values.as_ptr() as *const std::ffi::c_void,
            values.len() as u64,
            shared_tracked(),
        );
        if !values.is_empty() && inner.length() as usize != values.len() {
            return Err(MetalError::Allocation(values.len()));
        }
        Ok(Self {
            inner,
            len: values.len(),
            _mmap: None,
        })
    }

    pub fn from_mmap(
        context: &MetalContext,
        mmap: std::sync::Arc<memmap2::Mmap>,
    ) -> Result<Self, MetalError> {
        // Metal documents that `newBufferWithBytesNoCopy:length:options:
        // deallocator:` requires a page-aligned length (the pointer is
        // already page-aligned -- POSIX `mmap` always returns one). The raw
        // file length has "worked" here only because `mmap` itself reserves
        // whole pages under the hood and zero-fills the tail of the last
        // one -- that's an undocumented tolerance, not a guarantee, so round
        // the length Metal sees up to the page boundary explicitly. Bytes
        // between the real file length and that boundary are the kernel's
        // own zero-filled mmap tail, so this never reads outside the
        // mapping. `GpuBytes::len()` keeps reporting the exact, unrounded
        // file length; only the Metal-facing allocation grows.
        let mmap_len = mmap.len();
        let rounded_len = if mmap_len == 0 {
            0
        } else {
            let page = page_size();
            mmap_len
                .checked_add(page - 1)
                .map(|padded| padded / page * page)
                .ok_or(MetalError::Allocation(mmap_len))?
        };
        let inner = context.device.new_buffer_with_bytes_no_copy(
            mmap.as_ptr() as *const std::ffi::c_void,
            rounded_len as u64,
            shared_tracked(),
            None,
        );
        if !mmap.is_empty() && inner.length() as usize != rounded_len {
            return Err(MetalError::Allocation(mmap_len));
        }
        Ok(Self {
            inner,
            len: mmap_len,
            _mmap: Some(mmap),
        })
    }

    pub fn metal(&self) -> &BufferRef {
        &self.inner
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn as_mut_bytes(&mut self) -> &mut [u8] {
        if self.len == 0 {
            return &mut [];
        }
        // SAFETY: `&mut self` excludes CPU aliases and callers synchronize
        // completed command buffers before reusing this shared allocation.
        unsafe { std::slice::from_raw_parts_mut(self.inner.contents() as *mut u8, self.len) }
    }

    pub fn view(&self, offset: usize, len: usize) -> Option<GpuByteView<'_>> {
        let end = offset.checked_add(len)?;
        (end <= self.len).then_some(GpuByteView {
            buffer: self,
            offset,
            len,
        })
    }
}

impl GpuByteView<'_> {
    pub fn metal(&self) -> &BufferRef {
        self.buffer.metal()
    }

    pub fn offset(&self) -> usize {
        self.offset
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl GpuBuffer {
    pub fn zeros(context: &MetalContext, len: usize) -> Result<Self, MetalError> {
        let bytes = len
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or(MetalError::Allocation(usize::MAX))?;
        let inner = context.device.new_buffer(bytes as u64, shared_tracked());
        if bytes > 0 && (inner.length() as usize != bytes || inner.contents().is_null()) {
            return Err(MetalError::Allocation(bytes));
        }
        if bytes > 0 {
            // SAFETY: the newly allocated shared buffer owns `bytes` writable
            // bytes and is not visible to a command buffer yet.
            unsafe { std::ptr::write_bytes(inner.contents() as *mut u8, 0, bytes) };
        }
        Ok(Self { inner, len })
    }

    pub fn from_f32(context: &MetalContext, values: &[f32]) -> Result<Self, MetalError> {
        let mut buffer = Self::zeros(context, values.len())?;
        buffer.as_mut_slice().copy_from_slice(values);
        Ok(buffer)
    }

    pub fn metal(&self) -> &BufferRef {
        &self.inner
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn as_slice(&self) -> &[f32] {
        if self.len == 0 {
            return &[];
        }
        // SAFETY: `inner` is a live StorageModeShared allocation created for
        // exactly `len` f32 values and no mutable CPU view exists here.
        unsafe { std::slice::from_raw_parts(self.inner.contents() as *const f32, self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        if self.len == 0 {
            return &mut [];
        }
        // SAFETY: `&mut self` provides exclusive CPU access to the live shared
        // allocation. Callers synchronize completed GPU writes before use.
        unsafe { std::slice::from_raw_parts_mut(self.inner.contents() as *mut f32, self.len) }
    }
}

impl GpuHalfBuffer {
    /// CPU-zeroing allocation. Default choice for anything whose bytes might
    /// be read before every element has an explicit writer.
    pub fn zeros(context: &MetalContext, len: usize) -> Result<Self, MetalError> {
        let mut buffer = Self::uninitialized(context, len)?;
        if buffer.len > 0 {
            buffer.as_mut_bits().fill(0);
        }
        Ok(buffer)
    }

    /// Allocate without CPU-touching the backing bytes.
    ///
    /// This is for multi-gigabyte KV planes ONLY. Their ring/logical
    /// metadata (`MetalKvPlane::origin_logical`/`origin_physical`/`len`)
    /// guarantees every row is written by a `store_kv_*` dispatch before any
    /// row within `[origin, origin + len)` is ever read back, so the CPU
    /// zero-fill `zeros()` performs is pure startup-time cost on an
    /// allocation that can be many gigabytes. Every other caller must keep
    /// using `zeros()` -- this path leaves stale bytes behind in release
    /// builds.
    ///
    /// In debug/test builds the bytes are poisoned (never plain zero)
    /// instead of left stale, so a bug that lets a read reach a
    /// not-yet-written row is conspicuous rather than silently reading
    /// zeros; see `kv_uninitialized_write_then_read_round_trips` below.
    pub fn uninitialized(context: &MetalContext, len: usize) -> Result<Self, MetalError> {
        let bytes = len
            .checked_mul(std::mem::size_of::<u16>())
            .ok_or(MetalError::Allocation(usize::MAX))?;
        let inner = context.device.new_buffer(bytes as u64, shared_tracked());
        if bytes > 0 && (inner.length() as usize != bytes || inner.contents().is_null()) {
            return Err(MetalError::Allocation(bytes));
        }
        let buffer = Self { inner, len };
        #[cfg(debug_assertions)]
        let mut buffer = buffer;
        #[cfg(debug_assertions)]
        if buffer.len > 0 {
            buffer.as_mut_bits().fill(0xDEAD);
        }
        Ok(buffer)
    }

    pub fn from_bits(context: &MetalContext, values: &[u16]) -> Result<Self, MetalError> {
        let mut buffer = Self::zeros(context, values.len())?;
        buffer.as_mut_bits().copy_from_slice(values);
        Ok(buffer)
    }

    pub fn metal(&self) -> &BufferRef {
        &self.inner
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn as_bits(&self) -> &[u16] {
        if self.len == 0 {
            return &[];
        }
        // SAFETY: `inner` is a live StorageModeShared allocation created for
        // exactly `len` u16 values and completed GPU writes are synchronized
        // before snapshots obtain this view.
        unsafe { std::slice::from_raw_parts(self.inner.contents() as *const u16, self.len) }
    }

    pub fn as_mut_bits(&mut self) -> &mut [u16] {
        if self.len == 0 {
            return &mut [];
        }
        // SAFETY: `&mut self` provides exclusive CPU access to the live shared
        // allocation. Callers synchronize completed GPU writes before use.
        unsafe { std::slice::from_raw_parts_mut(self.inner.contents() as *mut u16, self.len) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zeros_is_still_all_zero() {
        let context = MetalContext::new().expect("Metal context");
        let buffer = GpuHalfBuffer::zeros(&context, 64).expect("zeros");
        assert!(buffer.as_bits().iter().all(|&bit| bit == 0));
    }

    #[test]
    fn kv_uninitialized_write_then_read_round_trips() {
        let context = MetalContext::new().expect("Metal context");
        let mut buffer = GpuHalfBuffer::uninitialized(&context, 8).expect("uninitialized");
        // Debug/test builds poison instead of leaving genuinely stale bytes,
        // so a read that reaches a row before `MetalKvPlane` has written it
        // (violating the origin/len invariant `uninitialized` relies on) is
        // conspicuous rather than a silent, misleadingly-plausible zero.
        #[cfg(debug_assertions)]
        assert!(
            buffer.as_bits().iter().all(|&bit| bit == 0xDEAD),
            "debug builds must poison, not silently zero, unwritten KV rows"
        );
        let values: Vec<u16> = (0..8).collect();
        buffer.as_mut_bits().copy_from_slice(&values);
        assert_eq!(buffer.as_bits(), values.as_slice());
    }

    #[test]
    fn from_mmap_rounds_metal_length_up_to_the_page_boundary() {
        let context = MetalContext::new().expect("Metal context");
        let mut path = std::env::temp_dir();
        path.push(format!(
            "muser-buffer-page-round-test-{}-{}",
            std::process::id(),
            line!()
        ));
        std::fs::write(&path, vec![7u8; 10]).expect("write temp file");
        let file = std::fs::File::open(&path).expect("open temp file");
        // SAFETY: the temp file above is not concurrently modified by
        // another process for the lifetime of this mapping.
        let mmap = unsafe { memmap2::MmapOptions::new().map(&file) }.expect("mmap temp file");
        let _ = std::fs::remove_file(&path);

        let bytes = GpuBytes::from_mmap(&context, std::sync::Arc::new(mmap)).expect("from_mmap");

        assert_eq!(
            bytes.len(),
            10,
            "GpuBytes::len() stays the exact, unrounded file length"
        );
        let page = page_size();
        assert_eq!(
            bytes.metal().length() as usize % page,
            0,
            "the Metal-facing allocation must be page aligned per Apple's documented contract"
        );
        assert!(bytes.metal().length() as usize >= 10);
    }
}
