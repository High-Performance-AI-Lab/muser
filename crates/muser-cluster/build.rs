// Compiles native/melon_rdma/melon_rdma_bulk.c into muser-cluster only when
// the `melon-rdma` feature is enabled. A stock `cargo build` (no feature)
// never touches this — no MelonDMA checkout or RDMA NIC required.
//
// On macOS this links against MelonDMA's `libibverbs_compat` shim (no real
// libibverbs exists on macOS — the DriverKit extension owns the NIC), found
// via the MELONDMA_DEXT_DIR env var pointing at a MelonDMA checkout's
// src/dext. On Linux it links the system's real libibverbs directly.

fn main() {
    if std::env::var("CARGO_FEATURE_MELON_RDMA").is_err() {
        return;
    }

    let src = "../../native/melon_rdma/melon_rdma_bulk.c";
    println!("cargo:rerun-if-changed={src}");
    println!("cargo:rerun-if-changed=../../native/melon_rdma/melon_rdma_bulk.h");
    println!("cargo:rerun-if-env-changed=MELONDMA_DEXT_DIR");

    let mut build = cc::Build::new();
    build.file(src).warnings(true);

    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os == "macos" {
        let dext_dir = std::env::var("MELONDMA_DEXT_DIR").expect(
            "MELONDMA_DEXT_DIR must point at a MelonDMA checkout's src/dext \
             (for its libibverbs_compat shim) to build the melon-rdma feature on macOS",
        );
        build.include(format!("{dext_dir}/usermode/libibverbs_compat/include"));
        println!("cargo:rustc-link-search=native={dext_dir}/build");
        // MelonDMA's header and library are a moving ABI: they live in another
        // checkout and are rebuilt on their own schedule. Cargo tracks neither
        // unless told, and an object compiled against a smaller struct than
        // the library now zeroes is a stack smash inside a verbs call that
        // reads as a bug in this code. Rebuild whenever either one moves.
        println!(
            "cargo:rerun-if-changed={dext_dir}/usermode/libibverbs_compat/include/infiniband/verbs.h"
        );
        println!("cargo:rerun-if-changed={dext_dir}/build/libibverbs.dylib");
        // The link search path is a build-time lookup; without an rpath the
        // linked binary has no LC_RPATH at all and dyld fails the load with
        // "Library not loaded: @rpath/libibverbs.dylib" the moment it runs.
        // MelonDMA's shim is not installed into a system library directory --
        // it lives in the checkout -- so the same directory has to be recorded
        // for runtime too, exactly as MelonDMA's own tools do it.
        println!("cargo:rustc-link-arg=-Wl,-rpath,{dext_dir}/build");
    }
    println!("cargo:rustc-link-lib=dylib=ibverbs");

    build.compile("melon_rdma_bulk");
}
