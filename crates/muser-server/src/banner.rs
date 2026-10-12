//! The `muser up` startup banner. Purely cosmetic — but "genuinely nice
//! terminal output" is the whole point of Deliverable A, and a stranger's
//! first few lines of output are the first impression of the product.
//! Respects `NO_COLOR` / non-tty output automatically via the `console`
//! crate (same one `indicatif` uses for its own styling).

use console::style;

const WIDTH: usize = 64;

pub fn print_banner() {
    let top = format!("╭{}╮", "─".repeat(WIDTH));
    let bottom = format!("╰{}╯", "─".repeat(WIDTH));

    println!();
    println!("{}", style(&top).cyan());
    blank_line();
    wordmark_line();
    center_line("inference you can see through", false);
    blank_line();
    center_line("Muse Glimmer-30B", false);
    center_line("M3 Ultra decode + 1x GX10 prefill/storage + kvpack", true);
    blank_line();
    println!("{}", style(&bottom).cyan());
    println!();
}

fn blank_line() {
    println!(
        "{}{}{}",
        style("│").cyan(),
        " ".repeat(WIDTH),
        style("│").cyan()
    );
}

fn wordmark_line() {
    let text = "muser";
    let pad = WIDTH.saturating_sub(text.chars().count());
    let left = pad / 2;
    let right = pad - left;
    println!(
        "{}{}{}{}{}",
        style("│").cyan(),
        " ".repeat(left),
        style(text).bold().cyan(),
        " ".repeat(right),
        style("│").cyan()
    );
}

fn center_line(text: &str, dim: bool) {
    let pad = WIDTH.saturating_sub(text.chars().count());
    let left = pad / 2;
    let right = pad - left;
    let rendered = if dim {
        style(text).dim().to_string()
    } else {
        text.to_string()
    };
    println!(
        "{}{}{}{}{}",
        style("│").cyan(),
        " ".repeat(left),
        rendered,
        " ".repeat(right),
        style("│").cyan()
    );
}
