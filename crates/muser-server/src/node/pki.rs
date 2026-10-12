//! Lab PKI, minted with `openssl` shell-outs.
//!
//! One CA per Mac (`~/.muser/ca`, 10 years, created exactly once), one leaf
//! per side per node. Both leaves carry `serverAuth` **and** `clientAuth`:
//! the Mac is the TLS server for the handoff stream and the client for the
//! producer control channel, and the node is the mirror image of that.
//!
//! Both peers pin the leaf by SHA-256 over its DER, so a leaf's subject
//! alternative name still has to be right — `rustls` and Python's
//! `ssl.PROTOCOL_TLS_CLIENT` both verify the name before the pin is even
//! consulted. `receiver_server_name` / `producer_control.server_name` are
//! the names minted here.

use std::path::{Path, PathBuf};
use std::process::Command;

use sha2::{Digest, Sha256};

use super::registry::{create_private_dir, write_private};
use super::Result;

/// The name the node uses for the Mac receiver, and the Mac for the node.
pub const RECEIVER_SERVER_NAME: &str = "muser-receiver";
pub const PRODUCER_SERVER_NAME: &str = "muser-prefilld";

/// Ten years: this CA outlives the lab it is minted for, and rotating it
/// means re-enrolling every node, which is exactly the intended cost.
const CA_DAYS: &str = "3650";
/// Leaf lifetime, inside the 825-day ceiling clients enforce.
const LEAF_DAYS: &str = "825";

#[derive(Debug, Clone)]
pub struct Ca {
    pub dir: PathBuf,
    pub key: PathBuf,
    pub cert: PathBuf,
}

impl Ca {
    pub fn paths(home: &Path) -> Self {
        let dir = home.join("ca");
        Self {
            key: dir.join("ca.key.pem"),
            cert: dir.join("ca.cert.pem"),
            dir,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Leaf {
    pub key: PathBuf,
    pub cert: PathBuf,
    /// SHA-256 over the certificate's DER encoding — the wire pin.
    pub pin: String,
}

/// Create the lab CA if it does not exist, or return the existing one.
///
/// The key file is claimed with `create_new` (O_EXCL): two `muser node add`
/// runs racing on a fresh Mac cannot both believe they minted the CA, and
/// the loser fails loudly instead of overwriting a CA that nodes are already
/// enrolled against.
pub fn ensure_ca(home: &Path) -> Result<(Ca, bool)> {
    let ca = Ca::paths(home);
    if ca.cert.is_file() && ca.key.is_file() {
        return Ok((ca, false));
    }
    create_private_dir(&ca.dir)?;
    claim_exclusively(&ca.key)?;
    let result = (|| -> Result<()> {
        openssl(&[
            "genpkey",
            "-algorithm",
            "EC",
            "-pkeyopt",
            "ec_paramgen_curve:P-256",
            "-out",
            &ca.key.display().to_string(),
        ])?;
        set_private(&ca.key)?;
        openssl(&[
            "req",
            "-x509",
            "-new",
            "-key",
            &ca.key.display().to_string(),
            "-sha256",
            "-days",
            CA_DAYS,
            "-subj",
            "/CN=muser-lab-ca",
            "-addext",
            "basicConstraints=critical,CA:TRUE,pathlen:0",
            "-addext",
            "keyUsage=critical,keyCertSign,cRLSign",
            "-out",
            &ca.cert.display().to_string(),
        ])
    })();
    if let Err(error) = result {
        // A half-minted CA must not be mistaken for a usable one next run.
        let _ = std::fs::remove_file(&ca.key);
        let _ = std::fs::remove_file(&ca.cert);
        return Err(error);
    }
    Ok((ca, true))
}

/// Mint a leaf keypair and have the lab CA sign it for both TLS roles.
pub fn issue_leaf(ca: &Ca, dir: &Path, stem: &str, common_name: &str) -> Result<Leaf> {
    create_private_dir(dir)?;
    let key = dir.join(format!("{stem}.key.pem"));
    let cert = dir.join(format!("{stem}.cert.pem"));
    let csr = dir.join(format!("{stem}.csr.pem"));
    let extensions = dir.join(format!("{stem}.ext.cnf"));
    write_private(
        &extensions,
        format!(
            "[leaf]\n\
             basicConstraints=critical,CA:FALSE\n\
             keyUsage=critical,digitalSignature,keyEncipherment\n\
             extendedKeyUsage=serverAuth,clientAuth\n\
             subjectAltName=DNS:{common_name}\n"
        )
        .as_bytes(),
    )?;
    openssl(&[
        "genpkey",
        "-algorithm",
        "EC",
        "-pkeyopt",
        "ec_paramgen_curve:P-256",
        "-out",
        &key.display().to_string(),
    ])?;
    set_private(&key)?;
    openssl(&[
        "req",
        "-new",
        "-key",
        &key.display().to_string(),
        "-subj",
        &format!("/CN={common_name}"),
        "-out",
        &csr.display().to_string(),
    ])?;
    openssl(&[
        "x509",
        "-req",
        "-in",
        &csr.display().to_string(),
        "-CA",
        &ca.cert.display().to_string(),
        "-CAkey",
        &ca.key.display().to_string(),
        "-CAcreateserial",
        "-days",
        LEAF_DAYS,
        "-sha256",
        "-extfile",
        &extensions.display().to_string(),
        "-extensions",
        "leaf",
        "-out",
        &cert.display().to_string(),
    ])?;
    let _ = std::fs::remove_file(&csr);
    let pin = der_pin(&cert)?;
    Ok(Leaf { key, cert, pin })
}

/// Validate and sign a CSR whose private key was generated on the machine
/// where it will remain. No private material crosses this API boundary.
pub fn sign_csr(ca: &Ca, dir: &Path, stem: &str, common_name: &str, csr: &Path) -> Result<Leaf> {
    create_private_dir(dir)?;
    openssl(&[
        "req",
        "-in",
        &csr.display().to_string(),
        "-noout",
        "-verify",
    ])?;
    let subject = openssl_output(&[
        "req",
        "-in",
        &csr.display().to_string(),
        "-noout",
        "-subject",
        "-nameopt",
        "RFC2253",
    ])?;
    if subject.trim() != format!("subject=CN={common_name}") {
        return Err(format!(
            "node CSR subject is {:?}, expected CN={common_name}",
            subject.trim()
        ));
    }
    let cert = dir.join(format!("{stem}.cert.pem"));
    let extensions = dir.join(format!("{stem}.ext.cnf"));
    write_private(
        &extensions,
        format!(
            "[leaf]\n\
             basicConstraints=critical,CA:FALSE\n\
             keyUsage=critical,digitalSignature,keyEncipherment\n\
             extendedKeyUsage=serverAuth,clientAuth\n\
             subjectAltName=DNS:{common_name}\n"
        )
        .as_bytes(),
    )?;
    openssl(&[
        "x509",
        "-req",
        "-in",
        &csr.display().to_string(),
        "-CA",
        &ca.cert.display().to_string(),
        "-CAkey",
        &ca.key.display().to_string(),
        "-CAcreateserial",
        "-days",
        LEAF_DAYS,
        "-sha256",
        "-extfile",
        &extensions.display().to_string(),
        "-extensions",
        "leaf",
        "-out",
        &cert.display().to_string(),
    ])?;
    let pin = der_pin(&cert)?;
    Ok(Leaf {
        key: PathBuf::new(),
        cert,
        pin,
    })
}

/// SHA-256 over the certificate's DER encoding — what both peers compare
/// the presented leaf against.
pub fn der_pin(cert: &Path) -> Result<String> {
    let der = cert.with_extension("der");
    openssl(&[
        "x509",
        "-in",
        &cert.display().to_string(),
        "-outform",
        "DER",
        "-out",
        &der.display().to_string(),
    ])?;
    let bytes = std::fs::read(&der).map_err(|error| format!("read {}: {error}", der.display()))?;
    let _ = std::fs::remove_file(&der);
    Ok(format!("{:x}", Sha256::digest(&bytes)))
}

/// A fresh 32-byte HMAC key straight from the kernel CSPRNG. The key never
/// enters the registry; only its id and epoch do.
pub fn mint_hmac_key(path: &Path) -> Result<()> {
    use std::io::Read;
    // Exactly 32 bytes, by `read_exact`: /dev/urandom never reports EOF, so
    // a read-to-end here would never return.
    let mut file = std::fs::File::open("/dev/urandom")
        .map_err(|error| format!("open /dev/urandom: {error}"))?;
    let mut bytes = [0u8; 32];
    file.read_exact(&mut bytes)
        .map_err(|error| format!("read /dev/urandom: {error}"))?;
    if bytes.iter().all(|byte| *byte == 0) {
        return Err("HMAC key material is not random".into());
    }
    write_private(path, &bytes)
}

/// Reserve a path before anything writes to it. `create_new` is the O_EXCL
/// the CA's "create once" rule needs.
fn claim_exclusively(path: &Path) -> Result<()> {
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    match options.open(path) {
        Ok(_) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => Err(format!(
            "{} already exists — another enrolment owns this lab CA",
            path.display()
        )),
        Err(error) => Err(format!("claim {}: {error}", path.display())),
    }
}

fn set_private(path: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))
            .map_err(|error| format!("chmod {}: {error}", path.display()))?;
    }
    #[cfg(not(unix))]
    {
        let _ = path;
    }
    Ok(())
}

pub fn openssl_argv(args: &[&str]) -> Vec<String> {
    std::iter::once("openssl".to_string())
        .chain(args.iter().map(|value| value.to_string()))
        .collect()
}

fn openssl(args: &[&str]) -> Result<()> {
    let output = Command::new("openssl")
        .args(args)
        .output()
        .map_err(|error| format!("spawn openssl: {error}"))?;
    if output.status.success() {
        return Ok(());
    }
    Err(format!(
        "openssl {} failed: {}",
        args.first().copied().unwrap_or(""),
        String::from_utf8_lossy(&output.stderr).trim()
    ))
}

fn openssl_output(args: &[&str]) -> Result<String> {
    let output = Command::new("openssl")
        .args(args)
        .output()
        .map_err(|error| format!("spawn openssl: {error}"))?;
    if output.status.success() {
        return Ok(String::from_utf8_lossy(&output.stdout).into_owned());
    }
    Err(format!(
        "openssl {} failed: {}",
        args.first().copied().unwrap_or(""),
        String::from_utf8_lossy(&output.stderr).trim()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_node_generated_csr_can_be_signed_without_importing_its_key() {
        let home = std::env::temp_dir().join(format!("muser-csr-sign-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&home);
        create_private_dir(&home).unwrap();
        let (ca, _) = ensure_ca(&home).unwrap();
        let remote = home.join("remote");
        create_private_dir(&remote).unwrap();
        let key = remote.join("node.key.pem");
        let csr = remote.join("node.csr.pem");
        openssl(&[
            "genpkey",
            "-algorithm",
            "EC",
            "-pkeyopt",
            "ec_paramgen_curve:P-256",
            "-out",
            &key.display().to_string(),
        ])
        .unwrap();
        openssl(&[
            "req",
            "-new",
            "-key",
            &key.display().to_string(),
            "-subj",
            "/CN=muser-prefilld",
            "-out",
            &csr.display().to_string(),
        ])
        .unwrap();
        let local = home.join("local");
        let leaf = sign_csr(&ca, &local, "node", PRODUCER_SERVER_NAME, &csr).unwrap();
        assert!(
            leaf.key.as_os_str().is_empty(),
            "no imported private-key path may be returned"
        );
        assert!(leaf.cert.is_file());
        assert_eq!(leaf.pin.len(), 64);
        assert!(
            key.is_file(),
            "the private key remains on the simulated node"
        );
        let _ = std::fs::remove_dir_all(&home);
    }
}
