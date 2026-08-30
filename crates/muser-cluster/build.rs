// Compiles native/melon_rdma/melon_rdma_pipe.c into muser-cluster only when
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

    let src = "../../native/melon_rdma/melon_rdma_pipe.c";
    println!("cargo:rerun-if-changed={src}");
    println!("cargo:rerun-if-changed=../../native/melon_rdma/melon_rdma_pipe.h");
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
    }
    println!("cargo:rustc-link-lib=dylib=ibverbs");

    build.compile("melon_rdma_pipe");
}
