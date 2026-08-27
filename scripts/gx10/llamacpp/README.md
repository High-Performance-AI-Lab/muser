# GX10 llama.cpp prefill producer

`muser_prefilld.py` is the resident, single-flight prefill producer: it holds
the GX10 GPU lease, keeps a warm `spark_kv_export` container serving jobs over
a FIFO, and answers TLS control requests from the Mac receiver. See the
module docstring and `muser_prefilld.py:main` for the full startup sequence
(GPU lease, warm exporter, control listener).

## Running as a service

`muser-prefilld.service` runs the daemon under systemd with restart-with-
backoff. Install it on the GX10 host:

```
sudo cp muser-prefilld.service /etc/systemd/system/
sudo mkdir -p /etc/muser
# MUSER_GX10_MODEL=/path/to/model.gguf
# MUSER_GX10_HANDOFF_CONFIG=/path/to/handoff-config.json
sudoedit /etc/muser/gx10-prefilld.env
sudo systemctl daemon-reload
sudo systemctl enable --now muser-prefilld.service
```

`--mmproj` and `--dflash` (vision and DFlash handoff schemas) are not wired
into the unit's `ExecStart` yet; add them to the `EnvironmentFile` and the
`ExecStart` line together if the armed handoff config requires them
(`config_has_vision` / `config_has_dflash` in `muser_prefilld.py` decide
which schema versions do).

A per-job producer failure (consumer/pipe drop, a bad request) is retry-armed
inside the daemon and never reaches systemd. The unit's `Restart=on-failure`
exists for the deliberate fail-stop path — the warm exporter container itself
died — and for outright crashes; `StartLimitBurst` caps runaway restarts so a
genuinely wedged GX10 pages instead of looping forever.

## Building the exporter container

Build `spark_kv_export` from `Dockerfile`, which applies the full 3-patch
adapter set (`muser_streaming_kv.patch`, `muser_logical_swa.patch`,
`muser_cuda_metal_compat.patch`) against a CUDA base image pinned by digest
and produces the `container_receipt` schema `muser_prefilld.py` verifies
before arming an exporter identity.

`install_on_gx10.sh` is a deprecated host-build fallback (2 of 3 patches, no
digest pin) kept for local debugging only; it refuses to run unless
`FORCE_HOST_BUILD=1` is set.
