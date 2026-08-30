//! `Read + Write` wrapper over `native/melon_rdma/melon_rdma_pipe.c`
//! (FFI, compiled by `build.rs` only when the `melon-rdma` feature is on).
//!
//! Bootstraps an RC QP over an already-connected TCP socket (qpn/psn/gid
//! exchange), then activates it — the same RESET->INIT->RTR->RTS sequence
//! and RoCEv1/GID-index addressing already proven on this hardware in
//! MelonDMA's own `ggml-rpc/transport.cpp`. After `open()` returns, all
//! further I/O goes over RDMA SEND/RECV, not the bootstrap socket.

use std::ffi::{c_char, c_void, CStr, CString};
use std::io::{self, Read, Write};
use std::os::fd::RawFd;

#[allow(non_camel_case_types)]
enum melon_rdma_pipe_t {}

extern "C" {
    fn melon_rdma_pipe_open(
        bootstrap_fd: i32,
        dev_name: *const c_char,
        gid_index: i32,
    ) -> *mut melon_rdma_pipe_t;
    fn melon_rdma_pipe_send(pipe: *mut melon_rdma_pipe_t, data: *const c_void, len: usize) -> i32;
    fn melon_rdma_pipe_recv(pipe: *mut melon_rdma_pipe_t, buf: *mut c_void, len: usize) -> isize;
    fn melon_rdma_pipe_close(pipe: *mut melon_rdma_pipe_t);
    fn melon_rdma_pipe_last_error() -> *const c_char;
}

fn last_error() -> String {
    unsafe {
        let ptr = melon_rdma_pipe_last_error();
        if ptr.is_null() {
            "unknown melon_rdma_pipe error".to_string()
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    }
}

pub struct MelonRdmaStream {
    handle: *mut melon_rdma_pipe_t,
}

// The C side touches only its own heap-allocated state through `handle`;
// nothing here is shared across threads concurrently.
unsafe impl Send for MelonRdmaStream {}

impl MelonRdmaStream {
    /// Takes ownership of `bootstrap_fd` — an already TCP-connected socket
    /// (e.g. from `TcpStream::into_raw_fd()`) — and bootstraps + activates
    /// an RC QP over it. `dev_name` is the local RDMA device (e.g.
    /// `"rocep1s0f1"` on Linux, `"mlx5_0"` on macOS via the compat shim).
    ///
    /// Re-verify `gid_index` against `ibv_devinfo -v -d <dev_name>` before
    /// trusting a previously-known value: this exact GID table has already
    /// drifted once on the paired hardware (a bonded interface moved the
    /// expected RoCEv1 entry from index 4 to index 2).
    pub fn open(bootstrap_fd: RawFd, dev_name: &str, gid_index: i32) -> io::Result<Self> {
        let c_dev = CString::new(dev_name)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;
        let handle = unsafe { melon_rdma_pipe_open(bootstrap_fd, c_dev.as_ptr(), gid_index) };
        if handle.is_null() {
            return Err(io::Error::other(format!(
                "melon_rdma_pipe_open({dev_name}, gid_index={gid_index}) failed: {}",
                last_error()
            )));
        }
        Ok(Self { handle })
    }
}

impl Read for MelonRdmaStream {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let n =
            unsafe { melon_rdma_pipe_recv(self.handle, buf.as_mut_ptr().cast(), buf.len()) };
        if n < 0 {
            return Err(io::Error::other(format!(
                "melon_rdma_pipe_recv failed: {}",
                last_error()
            )));
        }
        Ok(n as usize)
    }
}

impl Write for MelonRdmaStream {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let rc = unsafe { melon_rdma_pipe_send(self.handle, buf.as_ptr().cast(), buf.len()) };
        if rc != 0 {
            return Err(io::Error::other(format!(
                "melon_rdma_pipe_send failed: {}",
                last_error()
            )));
        }
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl Drop for MelonRdmaStream {
    fn drop(&mut self) {
        unsafe { melon_rdma_pipe_close(self.handle) };
    }
}
