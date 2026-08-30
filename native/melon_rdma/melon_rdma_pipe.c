/* melon_rdma_pipe.c — see melon_rdma_pipe.h.
 *
 * First-correctness-pass implementation: unpipelined (one outstanding SEND
 * at a time, one outstanding RECV slot at a time). Matches this project's
 * own established discipline (see MelonDMA notes/40-59): prove correctness
 * on real hardware first, add pipelining/windowing only after that holds.
 *
 * Activation sequence (RESET->INIT->RTR->RTS), the RoCEv1/GID-index
 * addressing, and the byte-stream-over-message-oriented-SEND/RECV framing
 * (deliver wc.byte_len bytes, remember a partial-consumption cursor across
 * recv() calls, repost) all mirror ggml-rpc/transport.cpp's already
 * hardware-proven design in the same repo (MelonDMA), rather than
 * reinventing verbs usage from scratch.
 */
#include "melon_rdma_pipe.h"

#include <arpa/inet.h>
#include <errno.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#include <infiniband/verbs.h>

#define MELON_RDMA_CHUNK (256 * 1024)
/* RX ring is one single MR covering RX_DEPTH*CHUNK bytes. Both 16*256KiB=4MiB
 * and 8*256KiB=2MiB hit a registration ceiling in the macOS DriverKit compat
 * shim (reg_mr failed, kr=0xe00002c2) that 4*256KiB=1MiB does not — looks
 * like a DMA/IOMemoryDescriptor mapping constraint tied to allocation size
 * or layout, not a simple byte-count cap (root cause not yet isolated;
 * worth a follow-up). Left at the proven-safe depth. TX generations are
 * registered as separate 256KiB MRs each regardless of window depth, so
 * TX_WINDOW's much deeper pipelining below is not constrained by this same
 * ceiling — and TX depth is what the real streaming-send bottleneck needs. */
#define MELON_RDMA_RX_DEPTH 4
#define MELON_RDMA_IB_PORT 1

/* Deep-pipelined TX, matching MelonDMA's own ggml-rpc/transport.cpp fix
 * (notes/59, "RDMA send pipelining fix"): the naive one-outstanding-send
 * design serializes each chunk's full network round-trip in front of the
 * next chunk's memcpy+post, which is the textbook doorbell/round-trip
 * accumulation problem for RDMA throughput (Kalia et al., "Design
 * Guidelines for High Performance RDMA Systems"). With N generations,
 * send() only blocks once every N calls — it posts generation G before
 * waiting for generation (G - N)'s completion, so up to N operations'
 * network time overlaps in flight instead of each one following the last.
 *
 * Widened from 2 to 16 after muser's own real streaming workload
 * (DeferredHandoffV2Sender in muser_v2_send.py, called from
 * muser_vllm/connector.py) exposed exactly the case a shallow window
 * doesn't help: many small per-layer/per-tile segments sent back-to-back
 * during prefill, not one large bulk transfer. A depth-2 window still
 * forces a wait roughly every other one of those small sends; at 16 a
 * whole request's worth of small segments can be in flight before send()
 * ever has to block, matching a TCP kernel send buffer's ability to
 * absorb a burst of small writes without per-write synchronization. */
#define MELON_RDMA_TX_WINDOW 16

/* Send-side wr_id values are tagged with this high bit so they can never
 * collide with an RX slot index (0..MELON_RDMA_RX_DEPTH-1) on the shared CQ. */
#define MELON_TX_WRID_TAG 0x8000000000000000ULL

static char g_last_error[256];

static void set_error(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_last_error, sizeof(g_last_error), fmt, ap);
    va_end(ap);
}

const char *melon_rdma_pipe_last_error(void) { return g_last_error; }

struct melon_rdma_pipe {
    int bootstrap_fd;

    struct ibv_context *ctx;
    struct ibv_pd *pd;
    struct ibv_cq *cq;
    struct ibv_qp *qp;

    uint8_t local_gid[16];
    int gid_index;
    uint8_t ib_port;
    enum ibv_mtu path_mtu;
    uint32_t qpn;
    uint32_t psn;

    /* TX: MELON_RDMA_TX_WINDOW staging-buffer generations, double-buffered
     * (see MELON_RDMA_TX_WINDOW above). */
    void *tx_buf[MELON_RDMA_TX_WINDOW];
    struct ibv_mr *tx_mr[MELON_RDMA_TX_WINDOW];
    int tx_outstanding[MELON_RDMA_TX_WINDOW]; /* 1 while generation g's send has no completion yet */
    int tx_next_gen;

    /* RX: ring of MELON_RDMA_RX_DEPTH chunk-sized buffers, one MR. */
    void *rx_buf;
    struct ibv_mr *rx_mr;
    int rx_next_slot; /* next slot to post/consume */

    /* Partial-delivery cursor into the currently-landed RX slot. */
    int rx_pending_slot;   /* -1 if nothing pending */
    size_t rx_pending_offset;
    size_t rx_pending_length;
};

static int post_rx_slot(melon_rdma_pipe_t *p, int slot) {
    struct ibv_sge sge = {0};
    sge.addr = (uintptr_t)((uint8_t *)p->rx_buf + (size_t)slot * MELON_RDMA_CHUNK);
    sge.length = MELON_RDMA_CHUNK;
    sge.lkey = p->rx_mr->lkey;

    struct ibv_recv_wr wr = {0};
    wr.wr_id = (uint64_t)slot;
    wr.sg_list = &sge;
    wr.num_sge = 1;

    struct ibv_recv_wr *bad = NULL;
    if (ibv_post_recv(p->qp, &wr, &bad) != 0) {
        set_error("ibv_post_recv failed: %s", strerror(errno));
        return -1;
    }
    return 0;
}

static int send_exact(int fd, const void *buf, size_t len) {
    const uint8_t *p = buf;
    size_t sent = 0;
    while (sent < len) {
        ssize_t n = send(fd, p + sent, len - sent, 0);
        if (n <= 0) {
            if (n < 0 && errno == EINTR) continue;
            return -1;
        }
        sent += (size_t)n;
    }
    return 0;
}

static int recv_exact(int fd, void *buf, size_t len) {
    uint8_t *p = buf;
    size_t got = 0;
    while (got < len) {
        ssize_t n = recv(fd, p + got, len - got, 0);
        if (n <= 0) {
            if (n < 0 && errno == EINTR) continue;
            return -1;
        }
        got += (size_t)n;
    }
    return 0;
}

/* Wire format for the bootstrap exchange: 4 + 4 + 1 + 1 + 16 bytes,
 * fixed-width, network byte order for the two u32s. */
struct bootstrap_wire {
    uint32_t qpn;
    uint32_t psn;
    uint8_t gid_index; /* informational only; the peer's own local index is what matters */
    uint8_t path_mtu;  /* enum ibv_mtu value, min() of both sides is used */
    uint8_t gid[16];
} __attribute__((packed));

melon_rdma_pipe_t *melon_rdma_pipe_open(int bootstrap_fd, const char *dev_name, int gid_index) {
    melon_rdma_pipe_t *p = calloc(1, sizeof(*p));
    if (!p) {
        set_error("out of memory");
        return NULL;
    }
    p->bootstrap_fd = bootstrap_fd;
    p->gid_index = gid_index;
    p->ib_port = MELON_RDMA_IB_PORT;
    p->rx_pending_slot = -1;

    int num_devices = 0;
    struct ibv_device **devices = ibv_get_device_list(&num_devices);
    if (!devices) {
        set_error("ibv_get_device_list failed: %s", strerror(errno));
        goto fail;
    }
    struct ibv_device *dev = NULL;
    for (int i = 0; i < num_devices; i++) {
        if (strcmp(ibv_get_device_name(devices[i]), dev_name) == 0) {
            dev = devices[i];
            break;
        }
    }
    if (!dev) {
        set_error("RDMA device '%s' not found", dev_name);
        ibv_free_device_list(devices);
        goto fail;
    }
    p->ctx = ibv_open_device(dev);
    ibv_free_device_list(devices);
    if (!p->ctx) {
        set_error("ibv_open_device('%s') failed: %s", dev_name, strerror(errno));
        goto fail;
    }

    struct ibv_port_attr port_attr;
    if (ibv_query_port(p->ctx, p->ib_port, &port_attr) != 0) {
        set_error("ibv_query_port failed: %s", strerror(errno));
        goto fail;
    }
    p->path_mtu = port_attr.active_mtu;

    union ibv_gid local_gid;
    if (ibv_query_gid(p->ctx, p->ib_port, p->gid_index, &local_gid) != 0) {
        set_error("ibv_query_gid(index=%d) failed: %s", p->gid_index, strerror(errno));
        goto fail;
    }
    memcpy(p->local_gid, local_gid.raw, 16);

    p->pd = ibv_alloc_pd(p->ctx);
    if (!p->pd) {
        set_error("ibv_alloc_pd failed: %s", strerror(errno));
        goto fail;
    }
    /* Shared send+recv CQ: must hold up to MELON_RDMA_TX_WINDOW outstanding
     * send completions and MELON_RDMA_RX_DEPTH outstanding recv completions
     * at once, not just whichever is larger. */
    p->cq = ibv_create_cq(p->ctx, MELON_RDMA_TX_WINDOW + MELON_RDMA_RX_DEPTH + 4, NULL, NULL, 0);
    if (!p->cq) {
        set_error("ibv_create_cq failed: %s", strerror(errno));
        goto fail;
    }

    struct ibv_qp_init_attr qp_init = {0};
    qp_init.send_cq = p->cq;
    qp_init.recv_cq = p->cq;
    qp_init.qp_type = IBV_QPT_RC;
    qp_init.cap.max_send_wr = MELON_RDMA_TX_WINDOW + 2;
    qp_init.cap.max_recv_wr = MELON_RDMA_RX_DEPTH;
    qp_init.cap.max_send_sge = 1;
    qp_init.cap.max_recv_sge = 1;
    p->qp = ibv_create_qp(p->pd, &qp_init);
    if (!p->qp) {
        set_error("ibv_create_qp failed: %s", strerror(errno));
        goto fail;
    }
    p->qpn = p->qp->qp_num;
    p->psn = p->qp->qp_num & 0xffffff;

    for (int i = 0; i < MELON_RDMA_TX_WINDOW; i++) {
        p->tx_buf[i] = malloc(MELON_RDMA_CHUNK);
        if (!p->tx_buf[i]) {
            set_error("out of memory allocating RDMA TX buffer %d", i);
            goto fail;
        }
        p->tx_mr[i] = ibv_reg_mr(p->pd, p->tx_buf[i], MELON_RDMA_CHUNK, IBV_ACCESS_LOCAL_WRITE);
        if (!p->tx_mr[i]) {
            set_error("ibv_reg_mr (tx %d) failed: %s", i, strerror(errno));
            goto fail;
        }
    }
    p->rx_buf = malloc((size_t)MELON_RDMA_RX_DEPTH * MELON_RDMA_CHUNK);
    if (!p->rx_buf) {
        set_error("out of memory allocating RDMA RX buffer");
        goto fail;
    }
    p->rx_mr = ibv_reg_mr(p->pd, p->rx_buf, (size_t)MELON_RDMA_RX_DEPTH * MELON_RDMA_CHUNK,
                           IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    if (!p->rx_mr) {
        set_error("ibv_reg_mr (rx) failed: %s", strerror(errno));
        goto fail;
    }

    /* --- Bootstrap: exchange qpn/psn/gid/mtu over the plain TCP socket --- */
    struct bootstrap_wire local_wire = {0};
    local_wire.qpn = htonl(p->qpn);
    local_wire.psn = htonl(p->psn);
    local_wire.gid_index = (uint8_t)p->gid_index;
    local_wire.path_mtu = (uint8_t)p->path_mtu;
    memcpy(local_wire.gid, p->local_gid, 16);

    if (send_exact(p->bootstrap_fd, &local_wire, sizeof(local_wire)) != 0) {
        set_error("bootstrap send failed: %s", strerror(errno));
        goto fail;
    }
    struct bootstrap_wire remote_wire;
    if (recv_exact(p->bootstrap_fd, &remote_wire, sizeof(remote_wire)) != 0) {
        set_error("bootstrap recv failed: %s", strerror(errno));
        goto fail;
    }
    uint32_t remote_qpn = ntohl(remote_wire.qpn);
    uint32_t remote_psn = ntohl(remote_wire.psn);
    enum ibv_mtu remote_mtu = (enum ibv_mtu)remote_wire.path_mtu;
    if (remote_mtu < p->path_mtu) {
        p->path_mtu = remote_mtu;
    }

    /* RESET -> INIT. Must happen before post_rx: ibv_post_recv is only
     * valid from INIT state onward, not on a QP still in RESET. */
    {
        struct ibv_qp_attr a = {0};
        a.qp_state = IBV_QPS_INIT;
        a.port_num = p->ib_port;
        a.pkey_index = 0;
        a.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE;
        int rc = ibv_modify_qp(p->qp, &a,
                                IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);
        if (rc != 0) {
            set_error("RESET->INIT failed: %s", strerror(rc));
            goto fail;
        }
    }

    /* Pre-post the full RX ring before going to RTR. */
    for (int i = 0; i < MELON_RDMA_RX_DEPTH; i++) {
        if (post_rx_slot(p, i) != 0) goto fail;
    }
    p->rx_next_slot = 0;

    /* INIT -> RTR */
    {
        struct ibv_qp_attr a = {0};
        a.qp_state = IBV_QPS_RTR;
        a.path_mtu = p->path_mtu;
        a.dest_qp_num = remote_qpn;
        a.rq_psn = remote_psn;
        a.max_dest_rd_atomic = 1;
        a.min_rnr_timer = 1;
        a.ah_attr.is_global = 1;
        memcpy(&a.ah_attr.grh.dgid, remote_wire.gid, 16);
        a.ah_attr.grh.hop_limit = 1;
        a.ah_attr.grh.sgid_index = (uint8_t)p->gid_index;
        a.ah_attr.dlid = 0;
        a.ah_attr.port_num = p->ib_port;
        int rc = ibv_modify_qp(p->qp, &a,
                                IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                                IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
        if (rc != 0) {
            set_error("INIT->RTR failed: %s (path_mtu=%d dest_qpn=%u rq_psn=%u gid_index=%d)",
                       strerror(rc), (int)a.path_mtu, a.dest_qp_num, a.rq_psn, p->gid_index);
            goto fail;
        }
    }
    /* RTR -> RTS */
    {
        struct ibv_qp_attr a = {0};
        a.qp_state = IBV_QPS_RTS;
        a.timeout = 14;
        a.retry_cnt = 7;
        a.rnr_retry = 7;
        a.sq_psn = p->psn;
        a.max_rd_atomic = 1;
        int rc = ibv_modify_qp(p->qp, &a,
                                IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                                IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC);
        if (rc != 0) {
            set_error("RTR->RTS failed: %s", strerror(rc));
            goto fail;
        }
    }

    return p;

fail:
    melon_rdma_pipe_close(p);
    return NULL;
}

/* Polls the shared CQ until exactly one completion has been classified:
 * either it is generation `want_tx_gen`'s TX completion (returns 1), or it
 * is something else — another TX generation, or an RX completion, which
 * gets filed into `p->tx_outstanding`/`p->rx_pending_*` respectively — in
 * which case this keeps polling. Pass want_tx_gen = -1 to instead return as
 * soon as *any* RX completion has been filed (used by recv()). */
static int drain_until(melon_rdma_pipe_t *p, int want_tx_gen) {
    for (;;) {
        if (want_tx_gen >= 0 && !p->tx_outstanding[want_tx_gen]) {
            return 0; /* already satisfied by an earlier poll in this call */
        }
        if (want_tx_gen < 0 && p->rx_pending_slot >= 0) {
            return 0;
        }
        struct ibv_wc wc;
        int n = ibv_poll_cq(p->cq, 1, &wc);
        if (n < 0) {
            set_error("ibv_poll_cq failed");
            return -1;
        }
        if (n == 0) continue;
        if (wc.wr_id & MELON_TX_WRID_TAG) {
            int gen = (int)(wc.wr_id & ~MELON_TX_WRID_TAG);
            if (wc.status != IBV_WC_SUCCESS) {
                set_error("TX CQE error: status=%d vendor_err=0x%x gen=%d", wc.status, wc.vendor_err, gen);
                return -1;
            }
            p->tx_outstanding[gen] = 0;
        } else {
            if (wc.status != IBV_WC_SUCCESS) {
                set_error("RX CQE error: status=%d vendor_err=0x%x", wc.status, wc.vendor_err);
                return -1;
            }
            /* Only one RX completion can be "pending/unconsumed" at a time
             * in this simple byte-stream cursor; with RX_DEPTH pre-posted
             * slots this call is never asked to juggle more than one before
             * the caller drains it via recv(). */
            p->rx_pending_slot = (int)wc.wr_id;
            p->rx_pending_offset = 0;
            p->rx_pending_length = wc.byte_len;
        }
    }
}

int melon_rdma_pipe_send(melon_rdma_pipe_t *p, const void *data, size_t len) {
    const uint8_t *src = data;
    size_t sent = 0;
    while (sent < len) {
        size_t chunk = len - sent;
        if (chunk > MELON_RDMA_CHUNK) chunk = MELON_RDMA_CHUNK;

        int gen = p->tx_next_gen;
        /* This generation's buffer is still owned by the NIC until its own
         * previous send completes — wait for that one completion (not this
         * chunk's), which is the whole point: generation (1-gen)'s send,
         * posted last iteration, gets its network round-trip time to
         * overlap with this iteration's memcpy+post instead of blocking it. */
        if (drain_until(p, gen) != 0) {
            return -1;
        }

        memcpy(p->tx_buf[gen], src + sent, chunk);

        struct ibv_sge sge = {0};
        sge.addr = (uintptr_t)p->tx_buf[gen];
        sge.length = (uint32_t)chunk;
        sge.lkey = p->tx_mr[gen]->lkey;

        struct ibv_send_wr wr = {0};
        wr.wr_id = MELON_TX_WRID_TAG | (uint64_t)gen;
        wr.sg_list = &sge;
        wr.num_sge = 1;
        wr.opcode = IBV_WR_SEND;
        wr.send_flags = IBV_SEND_SIGNALED;

        struct ibv_send_wr *bad = NULL;
        if (ibv_post_send(p->qp, &wr, &bad) != 0) {
            set_error("ibv_post_send failed: %s", strerror(errno));
            return -1;
        }
        p->tx_outstanding[gen] = 1;
        p->tx_next_gen = (gen + 1) % MELON_RDMA_TX_WINDOW;
        sent += chunk;
    }
    return 0;
}

/* Drains every generation's outstanding send. Called before close() and
 * usable by callers that want a synchronous "fully flushed" barrier. */
static int drain_all_tx(melon_rdma_pipe_t *p) {
    for (int gen = 0; gen < MELON_RDMA_TX_WINDOW; gen++) {
        if (drain_until(p, gen) != 0) return -1;
    }
    return 0;
}

ssize_t melon_rdma_pipe_recv(melon_rdma_pipe_t *p, void *buf, size_t len) {
    if (p->rx_pending_slot < 0) {
        if (drain_until(p, -1) != 0) {
            return -1;
        }
    }

    int slot = p->rx_pending_slot;
    size_t available = p->rx_pending_length - p->rx_pending_offset;
    size_t take = available < len ? available : len;
    const uint8_t *slot_buf = (const uint8_t *)p->rx_buf + (size_t)slot * MELON_RDMA_CHUNK;
    memcpy(buf, slot_buf + p->rx_pending_offset, take);
    p->rx_pending_offset += take;

    if (p->rx_pending_offset >= p->rx_pending_length) {
        /* Fully consumed: repost this slot for the next incoming message. */
        p->rx_pending_slot = -1;
        p->rx_pending_offset = 0;
        p->rx_pending_length = 0;
        if (post_rx_slot(p, slot) != 0) {
            return -1;
        }
    }
    return (ssize_t)take;
}

void melon_rdma_pipe_close(melon_rdma_pipe_t *p) {
    if (!p) return;
    if (p->qp) {
        /* Best-effort: let the last posted send(s) actually land before
         * tearing down the QP, so close() right after send() cannot
         * silently drop the final in-flight chunk. Ignore failures here —
         * we are closing regardless. */
        drain_all_tx(p);
    }
    for (int i = 0; i < MELON_RDMA_TX_WINDOW; i++) {
        if (p->tx_mr[i]) ibv_dereg_mr(p->tx_mr[i]);
        free(p->tx_buf[i]);
    }
    if (p->rx_mr) ibv_dereg_mr(p->rx_mr);
    if (p->qp) ibv_destroy_qp(p->qp);
    if (p->cq) ibv_destroy_cq(p->cq);
    if (p->pd) ibv_dealloc_pd(p->pd);
    if (p->ctx) ibv_close_device(p->ctx);
    free(p->rx_buf);
    if (p->bootstrap_fd >= 0) close(p->bootstrap_fd);
    free(p);
}
