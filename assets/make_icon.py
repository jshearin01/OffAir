#!/usr/bin/env python3
"""Regenerate the OffAir application icon (source PNG + Windows .ico).

Design — a bold mark that still reads at 16 px, in the app's indigo
accent (#4F46E5, see app/ui/theme.py):

* Indigo rounded-square tile (gradient #4F46E5 -> #4338CA), the
  "broadcast" tile.
* A white microphone in the middle: capsule body with two grille
  slits, a U-shaped cradle arc, a stem and a base.
* One short white soundwave bar on each side of the mic.

The mark is deliberately simple — no text, no fine detail — so the
16 px .ico entry stays legible (tile + white mic blob). Run it with
the system Python (Pillow is installed there)::

    python3 assets/make_icon.py

Outputs (repo root ``assets/``):
``offair-icon.png`` — 1024x1024 master source, ``offair.ico`` —
multi-size Windows icon (16/32/48/256), the file wired into
OffAir.spec, OffAir.iss and the app itself (see docs/PACKAGING.md).
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

SIZE = 1024
INDIGO_TOP = (79, 70, 229)      # #4F46E5 — the app accent (theme.py)
INDIGO_BOTTOM = (67, 56, 202)   # #4338CA — the accent hover
WHITE = (255, 255, 255, 255)
TRANSPARENT = (0, 0, 0, 0)

ASSETS = Path(__file__).resolve().parent
PNG_PATH = ASSETS / "offair-icon.png"
ICO_PATH = ASSETS / "offair.ico"
ICO_SIZES = [(16, 16), (32, 32), (48, 48), (256, 256)]


def _vertical_gradient(size: int, top: tuple, bottom: tuple) -> Image.Image:
    """An RGB vertical gradient, ``top`` -> ``bottom``."""
    gradient = Image.new("RGB", (1, size))
    pixels = gradient.load()
    for y in range(size):
        t = y / max(size - 1, 1)
        pixels[0, y] = tuple(
            round(top[i] + (bottom[i] - top[i]) * t) for i in range(3))
    return gradient.resize((size, size))


def draw_icon(size: int = SIZE) -> Image.Image:
    """Draw the icon at ``size`` px square; returns an RGBA image."""
    s = size / SIZE  # scale factor for every coordinate below
    img = Image.new("RGBA", (size, size), TRANSPARENT)

    # Indigo rounded-square tile.
    tile = _vertical_gradient(size, INDIGO_TOP, INDIGO_BOTTOM).convert("RGBA")
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        [int(48 * s), int(48 * s), int(976 * s), int(976 * s)],
        radius=int(224 * s), fill=255)
    img.paste(tile, (0, 0), mask)

    draw = ImageDraw.Draw(img)

    # Soundwave bars flanking the mic (outside the cradle arc).
    for x0 in (300, 692):
        draw.rectangle([int(x0 * s), int(330 * s),
                        int((x0 + 32) * s), int(550 * s)], fill=WHITE)

    # Microphone capsule.
    draw.rounded_rectangle(
        [int(452 * s), int(296 * s), int(572 * s), int(560 * s)],
        radius=int(60 * s), fill=WHITE)
    # Grille slits across the capsule (indigo, show at 32 px and up).
    for y0 in (384, 444):
        draw.rectangle([int(468 * s), int(y0 * s),
                        int(556 * s), int((y0 + 16) * s)],
                       fill=INDIGO_TOP + (255,))

    # Cradle arc under the capsule: the bottom half of a ring centred
    # on the mic (Pillow angles run clockwise from 3 o'clock, so
    # 20..160 deg is the bottom arc).
    draw.arc([int(364 * s), int(320 * s), int(660 * s), int(616 * s)],
             start=20, end=160, fill=WHITE, width=int(38 * s))
    # Stem + base.
    draw.rectangle([int(494 * s), int(610 * s),
                    int(530 * s), int(690 * s)], fill=WHITE)
    draw.rounded_rectangle(
        [int(404 * s), int(688 * s), int(620 * s), int(732 * s)],
        radius=int(22 * s), fill=WHITE)
    return img


def main() -> None:
    icon = draw_icon()
    icon.save(PNG_PATH)
    # Pillow stores the 256 px entry PNG-compressed and the smaller
    # ones as BMP, which is what Windows expects in a .ico.
    icon.save(ICO_PATH, sizes=ICO_SIZES)
    print(f"Wrote {PNG_PATH} ({icon.size[0]}x{icon.size[1]})")
    print(f"Wrote {ICO_PATH} (sizes: "
          + ", ".join(f"{w}x{h}" for w, h in ICO_SIZES) + ")")


if __name__ == "__main__":
    main()
