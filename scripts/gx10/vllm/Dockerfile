FROM muser-gx10-prefill:nvfp4-dflash-rope-v96 AS dflash-exporter

FROM nvcr.io/nvidia/vllm:26.07-py3@sha256:95c498a475142c20c989c65e5d223348c09fed83ba17ddf44f117610c0bd3268

ARG VLLM_COMMIT=6adad08767583f52eb4d2122111af0bf638ed5e6
ARG VLLM_WHEEL_SHA256=230d876ef3d90718ce8b42bef3b24c4384a714d623db7a2eb9c27c59d138066c
ARG VLLM_WHEEL=vllm-0.27.2rc1.dev76+g6adad0876-cp38-abi3-manylinux_2_28_aarch64.whl
ARG VLLM_WHEEL_URL=vllm-0.27.2rc1.dev76%2Bg6adad0876-cp38-abi3-manylinux_2_28_aarch64.whl

RUN test "$VLLM_COMMIT" = "6adad08767583f52eb4d2122111af0bf638ed5e6" \
    && curl --fail --location --retry 3 \
      --output "/tmp/${VLLM_WHEEL}" \
      "https://wheels.vllm.ai/${VLLM_COMMIT}/${VLLM_WHEEL_URL}" \
    && printf '%s  %s\n' "$VLLM_WHEEL_SHA256" "/tmp/${VLLM_WHEEL}" \
      | sha256sum --check --strict \
    && python3 -m pip install --no-cache-dir --no-deps --force-reinstall \
      "/tmp/${VLLM_WHEEL}" \
    && rm "/tmp/${VLLM_WHEEL}" \
    && python3 -c 'import importlib.util,torch,vllm; assert torch.version.cuda == "13.3"; assert vllm.__version__.startswith("0.27.2rc1.dev76"); assert importlib.util.find_spec("vllm.model_executor.models.muse_glimmer") is not None'

ENV PYTHONHASHSEED=0 \
    VLLM_ATTENTION_BACKEND=FLASH_ATTN \
    VLLM_ENABLE_V1_MULTIPROCESSING=0 \
    VLLM_USE_FLASHINFER_SAMPLER=0 \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    PYTHONPATH=/opt/muser/scripts/gx10/vllm \
    LD_LIBRARY_PATH=/opt/muser/lib:${LD_LIBRARY_PATH}

COPY --from=dflash-exporter /opt/muser/bin/spark_kv_export /opt/muser/bin/spark_kv_export
COPY --from=dflash-exporter /opt/muser/lib /opt/muser/lib

COPY scripts/gx10/llamacpp/llamacpp_session_send.py /opt/muser/scripts/gx10/llamacpp/llamacpp_session_send.py
COPY scripts/gx10/llamacpp/protocol.py /opt/muser/scripts/gx10/llamacpp/protocol.py
COPY scripts/gx10/llamacpp/muser_v2_send.py /opt/muser/scripts/gx10/llamacpp/muser_v2_send.py
COPY scripts/gx10/vllm/muser_vllm /opt/muser/scripts/gx10/vllm/muser_vllm
COPY scripts/gx10/vllm/benchmark_native_prefill.py /opt/muser/scripts/gx10/vllm/benchmark_native_prefill.py
COPY scripts/gx10/vllm/request_producer.py /opt/muser/scripts/gx10/vllm/request_producer.py
COPY scripts/gx10/vllm/score_nvfp4_drift.py /opt/muser/scripts/gx10/vllm/score_nvfp4_drift.py
COPY scripts/gx10/vllm/resident_producer.py /opt/muser/scripts/gx10/vllm/resident_producer.py

ENTRYPOINT ["python3", "/opt/muser/scripts/gx10/vllm/resident_producer.py"]
