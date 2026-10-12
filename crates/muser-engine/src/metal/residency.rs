//! Minimal `MTLResidencySet` owner extracted from Ferrite's Metal substrate.
//!
//! The immutable GGUF arena is bound by every projection in every command
//! buffer. Attaching it once lets Metal skip repeating residency work for the
//! 16+ GiB allocation on every token. The Objective-C surface is public on
//! macOS 15+, and absence fails open to the ordinary Metal residency path.

use metal::foreign_types::{ForeignType, ForeignTypeRef};
use metal::{BufferRef, CommandQueue, Device};
use objc::runtime::Object;
use objc::{msg_send, sel, sel_impl};
use std::ffi::c_void;

pub struct ResidencySet {
    handles: std::sync::Mutex<ResidencyHandles>,
}

struct ResidencyHandles {
    set: *mut Object,
    queue: *mut Object,
}

// Both pointers are owned Objective-C references and all messages are
// serialized through the mutex.
unsafe impl Send for ResidencyHandles {}

impl Drop for ResidencySet {
    fn drop(&mut self) {
        let handles = self
            .handles
            .get_mut()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if !handles.set.is_null() {
            unsafe {
                if !handles.queue.is_null() {
                    let _: () = msg_send![handles.queue, removeResidencySet: handles.set];
                }
                let _: () = msg_send![handles.set, release];
            }
        }
        if !handles.queue.is_null() {
            unsafe {
                let _: () = msg_send![handles.queue, release];
            }
        }
        handles.set = std::ptr::null_mut();
        handles.queue = std::ptr::null_mut();
    }
}

impl ResidencySet {
    pub fn request_residency(&self) {
        let handles = self
            .handles
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if !handles.set.is_null() {
            unsafe {
                let _: () = msg_send![handles.set, requestResidency];
            }
        }
    }
}

pub fn create_and_attach(
    device: &Device,
    queue: &CommandQueue,
    buffers: &[&BufferRef],
) -> Option<ResidencySet> {
    if buffers.is_empty() {
        return None;
    }
    unsafe {
        let descriptor_class = objc::runtime::Class::get("MTLResidencySetDescriptor")?;
        let descriptor: *mut Object = msg_send![descriptor_class, alloc];
        let descriptor: *mut Object = msg_send![descriptor, init];
        if descriptor.is_null() {
            return None;
        }
        let _: () = msg_send![descriptor, setInitialCapacity: buffers.len() as u64];
        let device_ptr = device.as_ptr() as *mut Object;
        let queue_ptr = queue.as_ptr() as *mut Object;
        let mut error: *mut Object = std::ptr::null_mut();
        let set: *mut Object = msg_send![
            device_ptr,
            newResidencySetWithDescriptor: descriptor
            error: (&mut error as *mut *mut Object)
        ];
        let _: () = msg_send![descriptor, release];
        if set.is_null() {
            return None;
        }
        for buffer in buffers {
            let pointer: *const c_void = buffer.as_ptr() as *const c_void;
            let _: () = msg_send![set, addAllocation: pointer];
        }
        let _: () = msg_send![set, commit];
        let _: () = msg_send![set, requestResidency];
        let _: () = msg_send![queue_ptr, addResidencySet: set];
        let queue_owned: *mut Object = msg_send![queue_ptr, retain];
        Some(ResidencySet {
            handles: std::sync::Mutex::new(ResidencyHandles {
                set,
                queue: queue_owned,
            }),
        })
    }
}
