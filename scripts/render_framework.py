"""Render assets/framework.svg to assets/framework.png at 1600 px wide.

The SVG in `assets/framework.svg` is the editable source-of-truth for the
EGTR-Review framework figure (Figure 2 in the paper). This script converts
it to a metadata-stripped PNG using whichever rasterizer is available
locally; preference order is documented in TRY_BACKENDS below.

Anonymity
---------
After rasterizing, the script strips PNG metadata in-place using either
`exiftool` (preferred) or Python's `Pillow`, both of which leave the
visible pixels untouched but remove `Creator` / `Software` / `tIME` /
EXIF-style fields a renderer may have written.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def _ok(cmd: list[str]) -> bool:
    """Run a command, return True on exit code 0."""
    try:
        return subprocess.run(cmd, check=False, capture_output=True).returncode == 0
    except FileNotFoundError:
        return False


def render_with_cairosvg(svg: Path, png: Path, width: int) -> bool:
    try:
        import cairosvg
    except ImportError:
        return False
    cairosvg.svg2png(url=str(svg), write_to=str(png), output_width=width)
    return True


def render_with_rsvg(svg: Path, png: Path, width: int) -> bool:
    if not shutil.which("rsvg-convert"):
        return False
    return _ok(["rsvg-convert", "-w", str(width), "-o", str(png), str(svg)])


def render_with_inkscape(svg: Path, png: Path, width: int) -> bool:
    if not shutil.which("inkscape"):
        return False
    return _ok([
        "inkscape", str(svg), "--export-type=png",
        f"--export-filename={png}", f"--export-width={width}",
    ])


def render_with_magick(svg: Path, png: Path, width: int) -> bool:
    for tool in ("magick", "convert"):
        if shutil.which(tool):
            return _ok([
                tool, "-density", "300", "-background", "white",
                str(svg), "-resize", f"{width}x", str(png),
            ])
    return False


TRY_BACKENDS = [
    ("cairosvg",      render_with_cairosvg),
    ("rsvg-convert",  render_with_rsvg),
    ("inkscape",      render_with_inkscape),
    ("imagemagick",   render_with_magick),
]


def strip_png_metadata(png: Path) -> None:
    """Best-effort metadata strip: try exiftool, fall back to Pillow."""
    if shutil.which("exiftool"):
        subprocess.run(
            ["exiftool", "-overwrite_original", "-all=", str(png)],
            check=False,
            capture_output=True,
        )
        # remove backup file if exiftool wrote one (e.g. on older versions)
        backup = png.with_suffix(png.suffix + "_original")
        if backup.exists():
            backup.unlink()
        return
    try:
        from PIL import Image
        img = Image.open(png)
        clean = Image.new(img.mode, img.size)
        clean.putdata(list(img.getdata()))
        clean.save(png, "PNG", optimize=True)
    except ImportError:
        print(
            "[render_framework] WARNING: neither exiftool nor Pillow available; "
            "PNG metadata was NOT stripped. Install one before publishing.",
            file=sys.stderr,
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--svg", default="assets/framework.svg",
                    help="Editable SVG source.")
    ap.add_argument("--png", default="assets/framework.png",
                    help="Output PNG path.")
    ap.add_argument("--width", type=int, default=1600,
                    help="Output PNG width in pixels (default: 1600).")
    args = ap.parse_args()

    svg = Path(args.svg)
    png = Path(args.png)
    if not svg.exists():
        print(f"[render_framework] missing source: {svg}", file=sys.stderr)
        return 2

    for name, fn in TRY_BACKENDS:
        try:
            ok = fn(svg, png, args.width)
        except Exception as e:
            print(f"[render_framework] {name} failed: {e}", file=sys.stderr)
            ok = False
        if ok and png.exists() and png.stat().st_size > 0:
            print(f"[render_framework] rasterized via {name} -> {png}")
            strip_png_metadata(png)
            print(f"[render_framework] PNG metadata stripped ({png.stat().st_size} bytes)")
            return 0

    print(
        "[render_framework] No usable SVG rasterizer found. Install one of:\n"
        "  pip install cairosvg   (recommended; pure-pip)\n"
        "  apt install librsvg2-bin       (rsvg-convert)\n"
        "  apt install inkscape\n"
        "  apt install imagemagick",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
