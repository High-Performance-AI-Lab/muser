ARG CUDA_IMAGE=nvcr.io/nvidia/cuda@sha256:7d2f6a8c2071d911524f95061a0db363e24d27aa51ec831fcccf9e76eb72bc92
FROM ${CUDA_IMAGE} AS build

ARG LLAMA_COMMIT
ARG ADAPTER_DIGEST
ARG CUDA_MATMUL=default
RUN apt-get update -qq \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
       cmake g++ git ninja-build >/dev/null \
    && rm -rf /var/lib/apt/lists/*
COPY llama/ /src/
COPY adapter/ /adapter/
WORKDIR /src
# Copy both the real versioned libraries and their SONAME symlinks. A
# regular-file-only find leaves libmtmd.so.0/libllama.so.0 unresolved in the
# runtime stage even though their .0.0.0 targets are present.
RUN set -eu \
    && case "$CUDA_MATMUL" in \
         default)      force_cublas=OFF; force_mmq=OFF ;; \
         force-cublas) force_cublas=ON;  force_mmq=OFF ;; \
         force-mmq)    force_cublas=OFF; force_mmq=ON  ;; \
         *) echo "invalid CUDA_MATMUL=$CUDA_MATMUL" >&2; exit 2 ;; \
       esac \
    && git apply --check /adapter/muser_streaming_kv.patch \
    && git apply /adapter/muser_streaming_kv.patch \
    && git apply --check /adapter/muser_logical_swa.patch \
    && git apply /adapter/muser_logical_swa.patch \
    && git apply --check /adapter/muser_cuda_metal_compat.patch \
    && git apply /adapter/muser_cuda_metal_compat.patch \
    && git apply --check /adapter/muser_dflash_rope_nco.patch \
    && git apply /adapter/muser_dflash_rope_nco.patch \
    && cmake -S . -B build -G Ninja \
       -DCMAKE_BUILD_TYPE=Release \
       -DGGML_CUDA=ON \
       -DCMAKE_CUDA_ARCHITECTURES=121a-real \
       -DGGML_CUDA_FORCE_CUBLAS="$force_cublas" \
       -DGGML_CUDA_FORCE_MMQ="$force_mmq" \
       -DGGML_NATIVE=OFF \
       -DBUILD_SHARED_LIBS=ON \
       -DLLAMA_BUILD_TESTS=OFF \
       -DLLAMA_BUILD_EXAMPLES=OFF \
       -DLLAMA_BUILD_SERVER=OFF \
    && cmake --build build --target mtmd -j10 \
    && g++ -O2 -std=c++17 -pthread /adapter/spark_kv_export.cpp \
       -I/src/include -I/src/src -I/src/ggml/include -I/src/tools/mtmd \
       -L/src/build/bin -lmtmd -lllama -lggml -lggml-base -lggml-cpu -lggml-cuda \
       -L/usr/local/cuda/lib64/stubs -lcuda \
       -Wl,-rpath,'$ORIGIN/../lib' \
       -o /src/build/bin/spark_kv_export \
    && mkdir -p /opt/muser/bin /opt/muser/lib \
    && cp /src/build/bin/spark_kv_export /opt/muser/bin/ \
    && cp -a /src/build/bin/lib*.so* /opt/muser/lib/

FROM ${CUDA_IMAGE}
ARG LLAMA_COMMIT
ARG ADAPTER_DIGEST
ARG CUDA_MATMUL=default
LABEL org.opencontainers.image.title="Muser GX10 prefill exporter" \
      org.opencontainers.image.version="0.1.0-beta.1" \
      org.opencontainers.image.revision="${LLAMA_COMMIT}" \
      io.muser.adapter.sha256="${ADAPTER_DIGEST}" \
      io.muser.cuda.matmul="${CUDA_MATMUL}"
COPY --from=build /opt/muser/ /opt/muser/
ENV LD_LIBRARY_PATH=/opt/muser/lib
ENTRYPOINT ["/opt/muser/bin/spark_kv_export"]
