fn main() {
    // objc 0.2's exported macros still probe this historical feature in the
    // destination crate. Declare the cfg value to rustc without exposing a
    // meaningless downstream Cargo feature.
    println!("cargo:rustc-check-cfg=cfg(feature, values(\"cargo-clippy\"))");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_ANE_COREML");
    if std::env::var_os("CARGO_FEATURE_ANE_COREML").is_some()
        && std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos")
    {
        println!("cargo:rustc-link-lib=framework=CoreML");
        println!("cargo:rustc-link-lib=framework=Foundation");
    }
}
