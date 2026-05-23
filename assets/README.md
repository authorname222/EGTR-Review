# Assets

The framework figure referenced by the top-level README:

```
framework.svg   editable source (committed)
framework.png   1600 px PNG render (generated; metadata stripped)
```

`framework.png` is **not committed** — it is regenerated from `framework.svg`
by `scripts/render_framework.py`.

## How to (re)generate `framework.png`

```bash
# pick one rasterizer if none is available yet
pip install cairosvg            # pure-Python; recommended
# or:  apt install librsvg2-bin  (rsvg-convert)
# or:  apt install inkscape
# or:  apt install imagemagick

python scripts/render_framework.py
```

The script tries `cairosvg → rsvg-convert → inkscape → magick/convert` in
order and writes `assets/framework.png` at 1600 px wide. After rasterising,
it strips PNG metadata using `exiftool` if available, falling back to
Pillow. The visible pixels are untouched; `Creator`, `Software`, `tIME`,
and EXIF-style fields are removed.

## Anonymity checklist (before any public upload)

1. Re-run `python scripts/render_framework.py` from a clean checkout.
2. Verify metadata is empty:
   ```bash
   exiftool assets/framework.png | grep -Ei 'Author|Creator|Software|GPS|Camera|User'
   ```
   The command should return nothing.
3. Do not commit screenshots from the authors' machines: they typically
   include OS user names in window titles, file paths visible in terminals,
   or institution wallpapers / themes. The SVG/PNG pipeline is metadata-
   free by construction.

## Editing the figure

`framework.svg` is hand-written XML — open it in any text editor, in
Inkscape, in drawio (File → Import → SVG), or in any vector editor. The
SVG header documents the colour conventions used by the figure:

| Region                | Fill       |
|-----------------------|-----------|
| Teacher block         | `#dbeafe`  |
| Per-agent box         | `#ffffff` border `#1d4ed8` |
| External retrieval    | `#dcfce7`  |
| Intermediate symbol   | `#fef3c7`  |
| Distillation arrow    | `#f97316`  |
| Student block         | `#ede9fe`  |
| Inference highlight   | `#fff7ed`  |

Re-export the PNG after any edit by running `render_framework.py` again.
