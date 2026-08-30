/* melon_rdma_pipe.h — minimal RDMA byte-stream pipe for muser's Linux
 * (GX10) side, mirroring the RESET->INIT->RTR->RTS activation sequence
 * already proven today on this exact hardware in MelonDMA's ggml-rpc
 * transport.cpp (GID index 2, RoCEv1, path MTU 4096). Exposes a plain
 * byte-stream API (send/recv, like a TCP socket) so it can sit underneath
 * an ssl.MemoryBIO-terminated TLS session without muser's protocol code
 * changing at all.
 *
 * Bootstrap: QP parameters (qpn/psn/gid) are exchanged over an
 * already-connected plain TCP socket (the same one muser_v2_send.py
 * already opens via socket.create_connection()) before the RDMA QP is
 * activated — the same pattern ggml-rpc's transport.cpp uses. After
 * activation the TCP socket is no longer used for data; only close it
 * once the RDMA pipe itself is closed.
 */
#pragma once

#include <stddef.h>
#include <sys/types.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct melon_rdma_pipe melon_rdma_pipe_t;

/* Bootstraps and activates an RC QP using `bootstrap_fd` (an already
 * TCP-connected socket, blocking) for the qpn/psn/gid exchange, then
 * returns a handle for send/recv. `dev_name` e.g. "rocep1s0f1";
 * `gid_index` the RoCEv1 GID table index for the desired local address
 * (re-verify with `ibv_devinfo -v` before assuming a prior value still
 * holds — this exact table has drifted once already on this hardware).
 * Returns NULL on failure; see melon_rdma_pipe_last_error(). */
melon_rdma_pipe_t *melon_rdma_pipe_open(int bootstrap_fd, const char *dev_name, int gid_index);

/* Blocking send of exactly `len` bytes. Returns 0 on success, -1 on error. */
int melon_rdma_pipe_send(melon_rdma_pipe_t *pipe, const void *data, size_t len);

/* Blocking receive of up to `len` bytes into `buf`. Returns the number of
 * bytes actually read (1..len, may be less than len like a socket recv()),
 * 0 if the peer closed cleanly, or -1 on error. */
ssize_t melon_rdma_pipe_recv(melon_rdma_pipe_t *pipe, void *buf, size_t len);

/* Closes the RDMA resources and the bootstrap socket. */
void melon_rdma_pipe_close(melon_rdma_pipe_t *pipe);

/* Thread-unsafe (single global buffer); fine for this single-threaded
 * sender/receiver use. */
const char *melon_rdma_pipe_last_error(void);

#ifdef __cplusplus
}
#endif
