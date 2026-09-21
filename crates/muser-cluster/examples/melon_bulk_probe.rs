//! Creates the receiving end of the bulk lane from a cluster config and
//! prints what it resolved. Reproduces lane bring-up on its own, without a
//! model, a producer or a handoff.
//!
//! Usage: melon_bulk_probe <cluster.json>
use muser_cluster::config::ReceiverConfigV2;

fn main() {
    let path = std::env::args().nth(1).expect("usage: melon_bulk_probe <cluster.json>");
    let config = ReceiverConfigV2::load(std::path::Path::new(&path)).expect("load cluster config");
    let rdma = config.rdma.as_ref().expect("cluster config has no rdma section");
    eprintln!("creating: {rdma:?}");
    let mut receiver = muser_cluster::melon_rdma::BulkReceiver::create(rdma).expect("create");
    eprintln!(
        "created: gid_index={} max_put={} endpoint={}",
        receiver.gid_index(),
        receiver.max_put(),
        receiver.endpoint().iter().map(|b| format!("{b:02x}")).collect::<String>()
    );
}
