# Muser metal patch

This directory is the crates.io `metal` 0.33.0 source (MIT OR Apache-2.0).
Muser carries one dependency-only patch: the archived `paste` macro crate is
replaced by its maintained, drop-in `pastey` successor under the existing
`paste` crate alias. No Metal API or implementation source is changed.

The patch exists so the release dependency audit can deny every RustSec
warning without an accepted-risk exception.
