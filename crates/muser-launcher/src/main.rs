use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

const HELP: &str = "Muser.app launcher\n\nOpen Muser.app from Finder to start Muser in Terminal through its signed native helper.\n";
const TERMINAL_HELP: &str =
    "Muser.app Terminal helper\n\nThis signed internal executable starts `muser up`.\n";

struct BundlePaths {
    muser: PathBuf,
    runtime: PathBuf,
    terminal: PathBuf,
}

fn bundle_paths(executable: &Path) -> Result<BundlePaths, String> {
    let executable_directory = executable
        .parent()
        .filter(|path| {
            path.file_name()
                .is_some_and(|name| name == "MacOS" || name == "Helpers")
        })
        .ok_or_else(|| {
            "launcher is not inside Muser.app/Contents/MacOS or Contents/Helpers".to_owned()
        })?;
    let contents = executable_directory
        .parent()
        .filter(|path| path.file_name().is_some_and(|name| name == "Contents"))
        .ok_or_else(|| "launcher is not inside an application bundle".to_owned())?;
    let resources = contents.join("Resources");
    let helpers = contents.join("Helpers");
    Ok(BundlePaths {
        muser: helpers.join("muser"),
        runtime: resources.join("muser"),
        terminal: helpers.join("muser-terminal"),
    })
}

fn require_executable(path: &Path, label: &str) -> Result<(), String> {
    let metadata = path
        .metadata()
        .map_err(|error| format!("{label} is unavailable: {error}"))?;
    if !metadata.is_file() {
        return Err(format!("{label} is not a regular file"));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if metadata.permissions().mode() & 0o111 == 0 {
            return Err(format!("{label} is not executable"));
        }
    }
    Ok(())
}

fn require_directory(path: &Path, label: &str) -> Result<(), String> {
    let metadata = path
        .metadata()
        .map_err(|error| format!("{label} is unavailable: {error}"))?;
    if !metadata.is_dir() {
        return Err(format!("{label} is not a directory"));
    }
    Ok(())
}

fn validate_bundle(executable: &Path) -> Result<BundlePaths, String> {
    let paths = bundle_paths(executable)?;
    require_executable(&paths.muser, "Muser runtime")?;
    require_executable(&paths.terminal, "signed Terminal helper")?;
    require_directory(&paths.runtime, "Muser runtime resources")?;
    Ok(paths)
}

fn is_terminal_helper(executable: &Path) -> bool {
    executable
        .file_name()
        .is_some_and(|name| name == "muser-terminal")
        && executable
            .parent()
            .and_then(Path::file_name)
            .is_some_and(|name| name == "Helpers")
}

fn show_start_error() {
    let _ = Command::new("/usr/bin/osascript")
        .args([
            "-e",
            "display alert \"Muser could not start\" message \"The signed application bundle is incomplete or Terminal could not be opened. Download Muser again.\" as critical",
        ])
        .status();
}

fn muser_command(paths: &BundlePaths) -> Command {
    let mut command = Command::new(&paths.muser);
    command
        .arg("up")
        .current_dir(&paths.runtime)
        .env("MUSER_REPO_ROOT", &paths.runtime);
    command
}

fn exec_muser(paths: &BundlePaths) -> Result<(), String> {
    let mut command = muser_command(paths);
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        let error = command.exec();
        Err(format!("start Muser runtime: {error}"))
    }
    #[cfg(not(unix))]
    {
        let _ = command;
        Err("Muser.app can only launch on macOS".to_owned())
    }
}

fn run() -> Result<(), String> {
    let executable =
        std::env::current_exe().map_err(|error| format!("locate launcher: {error}"))?;
    let paths = validate_bundle(&executable)?;
    let mut arguments = std::env::args_os().skip(1);
    let argument = arguments.next();
    if arguments.next().is_some() {
        return Err("the Muser.app launcher accepts at most one option".to_owned());
    }

    if is_terminal_helper(&executable) {
        match argument.as_deref() {
            Some(value) if value == "--help" || value == "-h" => {
                print!("{TERMINAL_HELP}");
                return Ok(());
            }
            Some(value) if value == "--check-bundle" => return Ok(()),
            Some(_) => return Err("unknown Muser.app Terminal-helper option".to_owned()),
            None => return exec_muser(&paths),
        }
    }

    match argument.as_deref() {
        Some(value) if value == "--help" || value == "-h" => {
            print!("{HELP}");
            return Ok(());
        }
        Some(value) if value == "--check-bundle" => return Ok(()),
        Some(_) => return Err("unknown Muser.app launcher option".to_owned()),
        None => {}
    }

    if !cfg!(target_os = "macos") {
        return Err("Muser.app can only launch on macOS".to_owned());
    }
    let status = Command::new("/usr/bin/open")
        .arg("-a")
        .arg("Terminal")
        .arg(&paths.terminal)
        .status()
        .map_err(|error| format!("open Terminal: {error}"))?;
    if !status.success() {
        return Err(format!("Terminal returned {status}"));
    }
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("muser-launcher: {error}");
            show_start_error();
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{bundle_paths, is_terminal_helper, muser_command};
    use std::ffi::OsStr;
    use std::path::Path;

    #[test]
    fn application_layout_resolves_the_embedded_runtime() {
        let paths =
            bundle_paths(Path::new("/Volumes/Muser/Muser.app/Contents/MacOS/Muser")).unwrap();
        assert_eq!(
            paths.terminal,
            Path::new("/Volumes/Muser/Muser.app/Contents/Helpers/muser-terminal")
        );
        assert_eq!(
            paths.muser,
            Path::new("/Volumes/Muser/Muser.app/Contents/Helpers/muser")
        );
        assert_eq!(
            paths.runtime,
            Path::new("/Volumes/Muser/Muser.app/Contents/Resources/muser")
        );
    }

    #[test]
    fn terminal_helper_resolves_the_same_application() {
        let executable = Path::new("/Volumes/Muser/Muser.app/Contents/Helpers/muser-terminal");
        let paths = bundle_paths(executable).unwrap();
        assert!(is_terminal_helper(executable));
        assert_eq!(
            paths.muser,
            Path::new("/Volumes/Muser/Muser.app/Contents/Helpers/muser")
        );
        let command = muser_command(&paths);
        assert_eq!(
            command.get_program(),
            OsStr::new("/Volumes/Muser/Muser.app/Contents/Helpers/muser")
        );
        assert_eq!(command.get_args().collect::<Vec<_>>(), [OsStr::new("up")]);
        assert_eq!(
            command.get_current_dir(),
            Some(Path::new(
                "/Volumes/Muser/Muser.app/Contents/Resources/muser"
            ))
        );
        assert!(command.get_envs().any(|(name, value)| {
            name == "MUSER_REPO_ROOT"
                && value
                    == Some(OsStr::new(
                        "/Volumes/Muser/Muser.app/Contents/Resources/muser",
                    ))
        }));
    }

    #[test]
    fn a_loose_binary_is_not_misrepresented_as_an_application() {
        let error = bundle_paths(Path::new("/tmp/muser-launcher"))
            .err()
            .expect("loose launcher must fail");
        assert!(error.contains("Muser.app/Contents/MacOS"));
    }
}
