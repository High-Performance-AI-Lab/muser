//! Local-CA operator workflow for native LAN TLS.

use std::fs::{self, OpenOptions};
use std::io::{self, Write as _};
use std::path::{Path, PathBuf};

use rcgen::{
    BasicConstraints, CertificateParams, DnType, ExtendedKeyUsagePurpose, IsCa, Issuer, KeyPair,
    KeyUsagePurpose,
};

use crate::cli::{TlsArgs, TlsCommand, TlsInitArgs, TlsIssueArgs};

pub fn run(args: TlsArgs) -> Result<(), String> {
    match args.command {
        TlsCommand::Init(args) => init(args),
        TlsCommand::Issue(args) => issue(args),
    }
}

fn pki_dir(explicit: Option<PathBuf>) -> Result<PathBuf, String> {
    if let Some(path) = explicit {
        return Ok(path);
    }
    if let Some(root) = std::env::var_os("MUSER_HOME").filter(|value| !value.is_empty()) {
        return Ok(PathBuf::from(root).join("pki"));
    }
    let home = std::env::var_os("HOME")
        .filter(|value| !value.is_empty())
        .ok_or("TLS helper needs --dir, $MUSER_HOME, or $HOME")?;
    Ok(PathBuf::from(home).join(".muser/pki"))
}

fn init(args: TlsInitArgs) -> Result<(), String> {
    let directory = pki_dir(args.dir)?;
    create_private_dir(&directory)?;
    let certificate_path = directory.join("ca.pem");
    let key_path = directory.join("ca-key.pem");
    refuse_existing(&[&certificate_path, &key_path])?;

    let mut params = CertificateParams::new(Vec::<String>::new()).map_err(display)?;
    params
        .distinguished_name
        .push(DnType::CommonName, "Muser Local CA");
    params.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
    params.key_usages = vec![
        KeyUsagePurpose::KeyCertSign,
        KeyUsagePurpose::CrlSign,
        KeyUsagePurpose::DigitalSignature,
    ];
    let key = KeyPair::generate().map_err(display)?;
    let certificate = params.self_signed(&key).map_err(display)?;
    atomic_write(&key_path, key.serialize_pem().as_bytes(), 0o600)?;
    atomic_write(&certificate_path, certificate.pem().as_bytes(), 0o644)?;
    println!("Muser local CA created: {}", certificate_path.display());
    println!(
        "Install this CA certificate on each client; keep {} private.",
        key_path.display()
    );
    Ok(())
}

fn issue(args: TlsIssueArgs) -> Result<(), String> {
    if !valid_name(&args.name) {
        return Err(
            "certificate name must use only ASCII letters, digits, '.', '_', or '-'".into(),
        );
    }
    if args.san.is_empty() || args.san.iter().any(|san| san.trim().is_empty()) {
        return Err("at least one nonempty explicit --san is required".into());
    }
    let directory = pki_dir(args.dir)?;
    let ca_pem = fs::read_to_string(directory.join("ca.pem")).map_err(display)?;
    let ca_key_pem = fs::read_to_string(directory.join("ca-key.pem")).map_err(display)?;
    let ca_key = KeyPair::from_pem(&ca_key_pem).map_err(display)?;
    let issuer = Issuer::from_ca_cert_pem(&ca_pem, ca_key).map_err(display)?;

    let output = args
        .out_dir
        .unwrap_or_else(|| directory.join("issued").join(&args.name));
    create_private_dir(&output)?;
    let certificate_path = output.join("server.pem");
    let key_path = output.join("server-key.pem");
    refuse_existing(&[&certificate_path, &key_path])?;

    let mut params = CertificateParams::new(args.san.clone()).map_err(display)?;
    params
        .distinguished_name
        .push(DnType::CommonName, args.name.as_str());
    params.use_authority_key_identifier_extension = true;
    params.key_usages = vec![KeyUsagePurpose::DigitalSignature];
    params.extended_key_usages = vec![ExtendedKeyUsagePurpose::ServerAuth];
    let key = KeyPair::generate().map_err(display)?;
    let certificate = params.signed_by(&key, &issuer).map_err(display)?;
    atomic_write(&key_path, key.serialize_pem().as_bytes(), 0o600)?;
    let chain = format!("{}{}", certificate.pem(), ca_pem);
    atomic_write(&certificate_path, chain.as_bytes(), 0o644)?;
    println!(
        "Issued {} for SANs: {}",
        certificate_path.display(),
        args.san.join(", ")
    );
    println!(
        "Serve with --tls-cert {} --tls-key {}",
        certificate_path.display(),
        key_path.display()
    );
    Ok(())
}

fn valid_name(name: &str) -> bool {
    !name.is_empty()
        && name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
}

fn refuse_existing(paths: &[&Path]) -> Result<(), String> {
    if let Some(path) = paths.iter().find(|path| path.exists()) {
        return Err(format!(
            "refusing to overwrite existing TLS material {}",
            path.display()
        ));
    }
    Ok(())
}

fn create_private_dir(path: &Path) -> Result<(), String> {
    fs::create_dir_all(path).map_err(display)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        fs::set_permissions(path, fs::Permissions::from_mode(0o700)).map_err(display)?;
    }
    Ok(())
}

fn atomic_write(path: &Path, bytes: &[u8], mode: u32) -> Result<(), String> {
    let parent = path
        .parent()
        .ok_or_else(|| format!("{} has no parent directory", path.display()))?;
    let temporary = parent.join(format!(
        ".{}.{}.tmp",
        path.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("tls"),
        std::process::id()
    ));
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        options.mode(mode);
    }
    let result = (|| -> io::Result<()> {
        let mut file = options.open(&temporary)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        fs::rename(&temporary, path)?;
        fs::File::open(parent)?.sync_all()
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result.map_err(display)
}

fn display(error: impl std::fmt::Display) -> String {
    error.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn certificate_names_are_path_safe() {
        assert!(valid_name("muser.local-1"));
        assert!(!valid_name("../escape"));
        assert!(!valid_name(""));
    }
}
