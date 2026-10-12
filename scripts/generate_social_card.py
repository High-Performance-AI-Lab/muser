#!/usr/bin/env python3
"""Generate Muser's deterministic 1200x630 social share card."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "docs" / "assets" / "muser-social-card.svg"

# Public summary and retained receipt IDs for every number rendered below.
SOURCES = (
    "docs/benchmarks.md#3-disaggregated-prefill-gb10-nvfp4--mac",
    "receipt-id: phase4-disagg-20260820",
    "ledger: Phase 4 disaggregated GX10→Mac context matrix (2026-08-20)",
)


def render() -> str:
    source = "benchmarks §3  •  receipt phase4-disagg-20260820"
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="630" viewBox="0 0 1200 630" role="img" aria-labelledby="title description">
  <title id="title">Muser measured disaggregated prefill</title>
  <desc id="description">Muser measured a 3.75 to 4.26 times TTFT payoff across six exactness-gated prompt depths.</desc>
  <defs>
    <linearGradient id="background" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#090b16"/>
      <stop offset="0.55" stop-color="#15153a"/>
      <stop offset="1" stop-color="#092c35"/>
    </linearGradient>
    <radialGradient id="glow" cx="0.78" cy="0.18" r="0.72">
      <stop offset="0" stop-color="#55e6c1" stop-opacity="0.28"/>
      <stop offset="1" stop-color="#55e6c1" stop-opacity="0"/>
    </radialGradient>
  </defs>
  <rect width="1200" height="630" rx="32" fill="url(#background)"/>
  <rect width="1200" height="630" rx="32" fill="url(#glow)"/>
  <path d="M0 500 C260 420 390 590 655 504 C880 431 1015 470 1200 408 V630 H0 Z" fill="#55e6c1" opacity="0.055"/>
  <g transform="translate(72 62)">
    <rect width="62" height="62" rx="17" fill="#55e6c1"/>
    <path d="M15 45 V17 H23 L31 31 L39 17 H47 V45 H39 V29 L31 42 L23 29 V45 Z" fill="#07161a"/>
    <text x="82" y="43" fill="#f6f3ff" font-family="-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif" font-size="38" font-weight="720" letter-spacing="1">MUSER</text>
  </g>
  <text x="72" y="235" fill="#a7a3bf" font-family="-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif" font-size="28" font-weight="560" letter-spacing="2">DISAGGREGATED PREFILL</text>
  <text x="65" y="390" fill="#f6f3ff" font-family="-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif" font-size="132" font-weight="780" letter-spacing="-6">3.75–4.26×</text>
  <text x="72" y="447" fill="#55e6c1" font-family="-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif" font-size="34" font-weight="650">measured time-to-first-token payoff</text>
  <g transform="translate(72 494)" font-family="-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif" font-size="22" font-weight="560">
    <rect width="152" height="46" rx="23" fill="#ffffff" opacity="0.09"/>
    <text x="76" y="30" fill="#ddd9ed" text-anchor="middle">six depths</text>
    <rect x="168" width="214" height="46" rx="23" fill="#ffffff" opacity="0.09"/>
    <text x="275" y="30" fill="#ddd9ed" text-anchor="middle">exactness-gated</text>
    <rect x="398" width="262" height="46" rx="23" fill="#ffffff" opacity="0.09"/>
    <text x="529" y="30" fill="#ddd9ed" text-anchor="middle">GB10 → Apple Silicon</text>
  </g>
  <text x="72" y="588" fill="#77758e" font-family="ui-monospace,'SFMono-Regular',Menlo,monospace" font-size="17">{source}</text>
</svg>
'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--png-output", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = render()
    if args.check:
        if not args.output.is_file() or args.output.read_text(encoding="utf-8") != expected:
            raise SystemExit(f"social card is stale: {args.output}")
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(expected, encoding="utf-8")
    if args.png_output is not None:
        args.png_output.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "sips",
                "-s",
                "format",
                "png",
                str(args.output),
                "--out",
                str(args.png_output),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
