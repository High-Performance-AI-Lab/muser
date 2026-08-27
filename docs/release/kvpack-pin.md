# kvpack source pin

Muser consumes the minimal upstream kvpack workspace from
`third_party/kvpack` only.

- tag: `kvpack-v0.1.0-alpha.2-rc1`
- commit: `70c34c7d790dbfc9c1271727dd34ea0e863404d2`
- workspace members: `kvpack-core`, `kvpack`, `kvpack-handoff`
- patches: none
- licenses: MIT OR Apache-2.0

`third_party/kvpack/provenance.json` binds the upstream root tree and SHA-256
of every retained file. Run:

```sh
python3 scripts/audit_vendored_kvpack.py
```

The audit fails on a missing, extra, changed, symlinked, or special entry.
`Cargo.toml` points directly into this tree, and the source audit rejects any
path package whose manifest resolves outside the Muser repository. A sibling
checkout is never a build input.
