#!/usr/bin/env python3
"""8-bit NES-style music video generator.

Pixel art office scenes, sprite characters, dialogue boxes, and chiptune aesthetics
rendered at NES resolution (256x240) and scaled up with nearest-neighbor.

Usage:
    python nes_video.py --audio song.mp3 --lyrics lyrics.txt --output nes_video.mp4

Requirements:
    pip install moviepy pillow numpy openai-whisper pydub
"""

import argparse
import math
import random
import re
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("nes_video")

# ── NES Constants ─────────────────────────────────────────────────────

NES_W, NES_H = 256, 240        # native NES resolution
OUT_W, OUT_H = 1920, 1080      # output resolution
SCALE = OUT_W // NES_W         # 7.5 → we'll use 7 and center
SCALE_X = 7
SCALE_Y = 4                    # adjusted below
FPS = 30

# We render at 256x240, scale to 1792x1080 (7x4.5), then pad to 1920x1080
RENDER_W = NES_W * SCALE_X     # 1792
RENDER_H = NES_H * (OUT_H // NES_H)  # 240 * 4 = 960... let's just scale uniformly

# Actually: scale both axes by the same factor to keep square pixels
PIXEL_SCALE = OUT_H // NES_H   # 1080 // 240 = 4 (with 120px left over)
# 240 * 4 = 960, 256 * 4 = 1024. We'll render at 256x240, scale 4x to 1024x960, pad to 1920x1080
# OR: use 270p: 256x240 * 4.5 = 1152x1080... not integer. Use 4x + padding.
# Cleanest: render 256x240, scale to fill width: 1920/256 = 7.5 (not integer)
# Best: render 320x240 (common NES-era), scale 4x = 1280x960, pad to 1920x1080
# Or render 256x240, scale with nearest to 1920x1080 directly (non-square pixels are fine for the aesthetic)

# Simplest approach: render at 256x240, PIL resize nearest to 1920x1080
# Pixels won't be perfectly square but that's authentic to how NES looked on CRT anyway

# ── NES Color Palette (2C02) ─────────────────────────────────────────
# Subset of the 64-color NES palette — the most useful colors

NES_PALETTE = {
    "black":        (0, 0, 0),
    "dark_gray":    (79, 79, 79),
    "gray":         (133, 133, 133),
    "light_gray":   (196, 196, 196),
    "white":        (254, 254, 254),

    "dark_blue":    (0, 0, 131),
    "blue":         (0, 59, 214),
    "light_blue":   (83, 167, 255),
    "sky_blue":     (174, 219, 255),

    "dark_red":     (165, 0, 0),
    "red":          (214, 29, 29),
    "light_red":    (255, 123, 108),
    "pink":         (255, 194, 188),

    "dark_green":   (0, 107, 0),
    "green":        (0, 161, 0),
    "light_green":  (92, 219, 75),
    "pale_green":   (179, 255, 171),

    "brown":        (107, 71, 0),
    "orange":       (214, 131, 0),
    "yellow":       (255, 214, 29),
    "light_yellow": (255, 242, 147),

    "dark_purple":  (79, 0, 107),
    "purple":       (131, 29, 165),
    "light_purple": (196, 108, 229),
    "lavender":     (234, 188, 255),

    "dark_cyan":    (0, 107, 107),
    "cyan":         (0, 167, 167),
    "light_cyan":   (83, 229, 229),

    "skin_light":   (255, 196, 147),
    "skin_dark":    (214, 131, 79),
    "hair_brown":   (107, 59, 0),
    "hair_blond":   (255, 214, 29),

    "floor_brown":  (131, 79, 29),
    "wall_beige":   (196, 167, 131),
    "desk_gray":    (133, 133, 133),
}
# Convert to list for indexed access
PAL = NES_PALETTE

# ── Pixel Font (4x6 bitmap glyphs) ──────────────────────────────────
# Each glyph is 4 wide x 6 tall, stored as list of 6 rows (each row is 4 bits)
# 1 = pixel on, 0 = pixel off

FONT_4X6 = {
    'A': [0b0110, 0b1001, 0b1001, 0b1111, 0b1001, 0b1001],
    'B': [0b1110, 0b1001, 0b1110, 0b1001, 0b1001, 0b1110],
    'C': [0b0111, 0b1000, 0b1000, 0b1000, 0b1000, 0b0111],
    'D': [0b1110, 0b1001, 0b1001, 0b1001, 0b1001, 0b1110],
    'E': [0b1111, 0b1000, 0b1110, 0b1000, 0b1000, 0b1111],
    'F': [0b1111, 0b1000, 0b1110, 0b1000, 0b1000, 0b1000],
    'G': [0b0111, 0b1000, 0b1000, 0b1011, 0b1001, 0b0111],
    'H': [0b1001, 0b1001, 0b1111, 0b1001, 0b1001, 0b1001],
    'I': [0b1110, 0b0100, 0b0100, 0b0100, 0b0100, 0b1110],
    'J': [0b0111, 0b0010, 0b0010, 0b0010, 0b1010, 0b0100],
    'K': [0b1001, 0b1010, 0b1100, 0b1100, 0b1010, 0b1001],
    'L': [0b1000, 0b1000, 0b1000, 0b1000, 0b1000, 0b1111],
    'M': [0b1001, 0b1111, 0b1111, 0b1001, 0b1001, 0b1001],
    'N': [0b1001, 0b1101, 0b1111, 0b1011, 0b1001, 0b1001],
    'O': [0b0110, 0b1001, 0b1001, 0b1001, 0b1001, 0b0110],
    'P': [0b1110, 0b1001, 0b1001, 0b1110, 0b1000, 0b1000],
    'Q': [0b0110, 0b1001, 0b1001, 0b1001, 0b0110, 0b0011],
    'R': [0b1110, 0b1001, 0b1001, 0b1110, 0b1010, 0b1001],
    'S': [0b0111, 0b1000, 0b0110, 0b0001, 0b0001, 0b1110],
    'T': [0b1111, 0b0100, 0b0100, 0b0100, 0b0100, 0b0100],
    'U': [0b1001, 0b1001, 0b1001, 0b1001, 0b1001, 0b0110],
    'V': [0b1001, 0b1001, 0b1001, 0b1001, 0b0110, 0b0100],
    'W': [0b1001, 0b1001, 0b1001, 0b1111, 0b1111, 0b1001],
    'X': [0b1001, 0b1001, 0b0110, 0b0110, 0b1001, 0b1001],
    'Y': [0b1001, 0b1001, 0b0110, 0b0100, 0b0100, 0b0100],
    'Z': [0b1111, 0b0001, 0b0010, 0b0100, 0b1000, 0b1111],
    '0': [0b0110, 0b1001, 0b1011, 0b1101, 0b1001, 0b0110],
    '1': [0b0100, 0b1100, 0b0100, 0b0100, 0b0100, 0b1110],
    '2': [0b0110, 0b1001, 0b0010, 0b0100, 0b1000, 0b1111],
    '3': [0b1110, 0b0001, 0b0110, 0b0001, 0b0001, 0b1110],
    '4': [0b1001, 0b1001, 0b1111, 0b0001, 0b0001, 0b0001],
    '5': [0b1111, 0b1000, 0b1110, 0b0001, 0b0001, 0b1110],
    '6': [0b0111, 0b1000, 0b1110, 0b1001, 0b1001, 0b0110],
    '7': [0b1111, 0b0001, 0b0010, 0b0100, 0b0100, 0b0100],
    '8': [0b0110, 0b1001, 0b0110, 0b1001, 0b1001, 0b0110],
    '9': [0b0110, 0b1001, 0b0111, 0b0001, 0b0001, 0b1110],
    ' ': [0b0000, 0b0000, 0b0000, 0b0000, 0b0000, 0b0000],
    '!': [0b0100, 0b0100, 0b0100, 0b0100, 0b0000, 0b0100],
    '?': [0b0110, 0b1001, 0b0010, 0b0100, 0b0000, 0b0100],
    '.': [0b0000, 0b0000, 0b0000, 0b0000, 0b0000, 0b0100],
    ',': [0b0000, 0b0000, 0b0000, 0b0000, 0b0100, 0b1000],
    '-': [0b0000, 0b0000, 0b1111, 0b0000, 0b0000, 0b0000],
    "'": [0b0100, 0b0100, 0b0000, 0b0000, 0b0000, 0b0000],
    '"': [0b1010, 0b1010, 0b0000, 0b0000, 0b0000, 0b0000],
    ':': [0b0000, 0b0100, 0b0000, 0b0000, 0b0100, 0b0000],
    ';': [0b0000, 0b0100, 0b0000, 0b0000, 0b0100, 0b1000],
    '/': [0b0001, 0b0010, 0b0010, 0b0100, 0b0100, 0b1000],
    '(': [0b0010, 0b0100, 0b0100, 0b0100, 0b0100, 0b0010],
    ')': [0b0100, 0b0010, 0b0010, 0b0010, 0b0010, 0b0100],
    '*': [0b0000, 0b1001, 0b0110, 0b1001, 0b0000, 0b0000],
}

GLYPH_W, GLYPH_H = 4, 6
GLYPH_SPACING = 1  # 1px between chars


def draw_text(img, x, y, text, color, scale=1):
    """Draw text using the 4x6 bitmap font onto a PIL Image."""
    pixels = img.load()
    w, h = img.size
    cx = x
    for ch in text.upper():
        glyph = FONT_4X6.get(ch)
        if glyph is None:
            cx += (GLYPH_W + GLYPH_SPACING) * scale
            continue
        for row_i, row_bits in enumerate(glyph):
            for col_i in range(GLYPH_W):
                if row_bits & (1 << (GLYPH_W - 1 - col_i)):
                    for sy in range(scale):
                        for sx in range(scale):
                            px = cx + col_i * scale + sx
                            py = y + row_i * scale + sy
                            if 0 <= px < w and 0 <= py < h:
                                pixels[px, py] = color
        cx += (GLYPH_W + GLYPH_SPACING) * scale
    return cx  # return end x


def text_width(text, scale=1):
    """Calculate pixel width of text string."""
    return len(text) * (GLYPH_W + GLYPH_SPACING) * scale - GLYPH_SPACING * scale


def draw_text_centered(img, y, text, color, scale=1):
    """Draw text centered horizontally."""
    tw = text_width(text, scale)
    x = (img.size[0] - tw) // 2
    draw_text(img, x, y, text, color, scale)


# ── Sprite System ────────────────────────────────────────────────────
# Sprites are 16x16 pixel arrays (or 16x24 for tall chars)
# Stored as list of rows, each row is a string where each char maps to a color

SPRITE_COLORS = {
    '.': None,                        # transparent
    'K': PAL["black"],
    'W': PAL["white"],
    'S': PAL["skin_light"],
    'D': PAL["skin_dark"],
    'B': PAL["blue"],
    'b': PAL["dark_blue"],
    'R': PAL["red"],
    'r': PAL["dark_red"],
    'G': PAL["green"],
    'g': PAL["dark_green"],
    'Y': PAL["yellow"],
    'O': PAL["orange"],
    'H': PAL["hair_brown"],
    'h': PAL["hair_blond"],
    'P': PAL["purple"],
    'L': PAL["light_gray"],
    'l': PAL["gray"],
    'T': PAL["dark_gray"],
    'C': PAL["cyan"],
    'F': PAL["floor_brown"],
    'N': PAL["pink"],
}

# Protagonist — office worker in white shirt, blue tie
SPRITE_PROTAGONIST_STAND = [
    "....HHHH....",
    "...HHHHHH...",
    "..HSSSSSSH..",
    "..SSSSSSS...",
    "..S.KSK.S...",
    "..SSSSSSSS..",
    "..SS.SS.SS..",
    "...SSSSSS...",
    "..KWWWWWK..",
    ".KWWBWWWWK.",
    ".KWWBWWWWK.",
    ".KSWBWWSK..",
    "..KWWWWK...",
    "..KLLLLK...",
    "..KL..LK...",
    "..KK..KK...",
]

SPRITE_PROTAGONIST_WALK1 = [
    "....HHHH....",
    "...HHHHHH...",
    "..HSSSSSSH..",
    "..SSSSSSS...",
    "..S.KSK.S...",
    "..SSSSSSSS..",
    "..SS.SS.SS..",
    "...SSSSSS...",
    "..KWWWWWK..",
    ".KWWBWWWWK.",
    ".KSWBWWWWK.",
    "..KWBWWSK..",
    "..KWWWWK...",
    "..KL..LK...",
    ".KL....KK..",
    ".KK..KK....",
]

SPRITE_PROTAGONIST_WALK2 = [
    "....HHHH....",
    "...HHHHHH...",
    "..HSSSSSSH..",
    "..SSSSSSS...",
    "..S.KSK.S...",
    "..SSSSSSSS..",
    "..SS.SS.SS..",
    "...SSSSSS...",
    "..KWWWWWK..",
    ".KWWBWWWWK.",
    ".KWWBWWSK..",
    "..KSBWWK...",
    "..KWWWWK...",
    "..KL..LK...",
    "..KK..LK...",
    "......KK...",
]

# Boss — suit, red tie, bigger
SPRITE_BOSS_STAND = [
    "...hhhhh....",
    "..hhhhhhh...",
    "..hSSSSSh...",
    "..SSSSSSS...",
    "..S.KSK.S...",
    "..SSSSSSSS..",
    "..SSKSKSSS..",
    "...SSSSSS...",
    ".KTTTTTTK..",
    "KTTTRTTTTK.",
    "KTTTRTTTTK.",
    "KSTTRTTTSK.",
    ".KTTTTTTK..",
    "..KTTTTK...",
    "..KT..TK...",
    "..KK..KK...",
]

# Corporate choir — small identical drones
SPRITE_CHOIR = [
    "..llll..",
    ".llllll.",
    ".lSSSSl.",
    "..SSSS..",
    "..S.KS..",
    "..SSSS..",
    ".KLLLLK.",
    "KLLLLLK.",
    ".KLLLLK.",
    "..KLLK..",
    "..K..K..",
    "..KK.KK.",
]


def parse_sprite(rows):
    """Parse sprite string rows into a list of (row, col, color) pixels."""
    pixels = []
    max_w = max(len(r) for r in rows)
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            c = SPRITE_COLORS.get(ch)
            if c is not None:
                pixels.append((x, y, c))
    return pixels, max_w, len(rows)


def draw_sprite(img, sprite_data, sx, sy, scale=1, flip_h=False):
    """Draw a parsed sprite at position (sx, sy) with optional scale and horizontal flip."""
    pixels_list, sw, sh = sprite_data
    pix = img.load()
    w, h = img.size
    for px, py, color in pixels_list:
        dx = (sw - 1 - px) if flip_h else px
        for oy in range(scale):
            for ox in range(scale):
                fx = sx + dx * scale + ox
                fy = sy + py * scale + oy
                if 0 <= fx < w and 0 <= fy < h:
                    pix[fx, fy] = color


# Pre-parse sprites
SPR_PROTAG_STAND = parse_sprite(SPRITE_PROTAGONIST_STAND)
SPR_PROTAG_WALK1 = parse_sprite(SPRITE_PROTAGONIST_WALK1)
SPR_PROTAG_WALK2 = parse_sprite(SPRITE_PROTAGONIST_WALK2)
SPR_BOSS = parse_sprite(SPRITE_BOSS_STAND)
SPR_CHOIR = parse_sprite(SPRITE_CHOIR)


# ── Background Tile System ──────────────────────────────────────────

def fill_rect(img, x, y, w, h, color):
    """Fill a rectangle on the image."""
    draw = ImageDraw.Draw(img)
    draw.rectangle([x, y, x + w - 1, y + h - 1], fill=color)


def draw_tile_row(img, y, pattern, tile_w=8):
    """Draw a repeating pattern row of tiles."""
    w = img.size[0]
    for x in range(0, w, tile_w):
        idx = (x // tile_w) % len(pattern)
        fill_rect(img, x, y, tile_w, tile_w, pattern[idx])


# ── Scene Backgrounds ───────────────────────────────────────────────

def draw_office_bg(img, scroll_x=0):
    """Generic open office background."""
    w, h = img.size
    # Ceiling
    fill_rect(img, 0, 0, w, 40, PAL["light_gray"])
    # Fluorescent lights
    for lx in range(20, w, 60):
        fill_rect(img, lx, 35, 30, 3, PAL["white"])
        fill_rect(img, lx + 2, 38, 26, 1, PAL["light_yellow"])
    # Walls
    fill_rect(img, 0, 40, w, 100, PAL["wall_beige"])
    # Motivational poster
    fill_rect(img, 50, 55, 30, 24, PAL["white"])
    fill_rect(img, 52, 57, 26, 14, PAL["sky_blue"])
    draw_text(img, 53, 73, "SYNERGY", PAL["black"], scale=1)
    # Window
    fill_rect(img, 150, 50, 40, 40, PAL["sky_blue"])
    fill_rect(img, 148, 48, 44, 2, PAL["dark_gray"])
    fill_rect(img, 148, 90, 44, 2, PAL["dark_gray"])
    fill_rect(img, 168, 50, 2, 40, PAL["dark_gray"])
    # Floor
    fill_rect(img, 0, 140, w, 100, PAL["floor_brown"])
    # Floor tiles
    for fx in range(0, w, 16):
        fill_rect(img, fx, 140, 1, 100, PAL["brown"])
    # Cubicle walls
    for cx in range(30 - (scroll_x % 80), w, 80):
        fill_rect(img, cx, 120, 4, 60, PAL["gray"])
        fill_rect(img, cx, 118, 60, 4, PAL["gray"])
    # Desks in cubicles
    for cx in range(40 - (scroll_x % 80), w, 80):
        fill_rect(img, cx, 155, 40, 4, PAL["desk_gray"])
        # Computer monitor
        fill_rect(img, cx + 12, 142, 16, 12, PAL["dark_gray"])
        fill_rect(img, cx + 14, 144, 12, 8, PAL["dark_blue"])
        fill_rect(img, cx + 18, 154, 4, 2, PAL["dark_gray"])


def draw_boardroom_bg(img):
    """Boardroom scene — long table, big screen."""
    w, h = img.size
    fill_rect(img, 0, 0, w, 50, PAL["dark_gray"])
    # Ceiling lights
    fill_rect(img, 60, 45, 140, 3, PAL["white"])
    # Walls
    fill_rect(img, 0, 50, w, 90, PAL["wall_beige"])
    # Big presentation screen
    fill_rect(img, 70, 55, 120, 70, PAL["black"])
    fill_rect(img, 74, 59, 112, 62, PAL["dark_blue"])
    draw_text(img, 82, 75, "Q4 SYNERGY", PAL["yellow"], scale=1)
    draw_text(img, 82, 85, "METRICS UP!", PAL["green"], scale=1)
    # Floor
    fill_rect(img, 0, 140, w, 100, PAL["floor_brown"])
    # Conference table
    fill_rect(img, 40, 160, 180, 20, PAL["brown"])
    fill_rect(img, 40, 160, 180, 3, PAL["orange"])
    # Chairs (simplified as colored blocks)
    for cx in [50, 80, 110, 140, 170, 200]:
        fill_rect(img, cx, 182, 10, 10, PAL["dark_gray"])
        fill_rect(img, cx + 1, 150, 8, 10, PAL["dark_gray"])


def draw_breakroom_bg(img):
    """Breakroom — coffee machine, sad plant."""
    w, h = img.size
    fill_rect(img, 0, 0, w, 50, PAL["light_gray"])
    fill_rect(img, 80, 45, 40, 3, PAL["white"])
    fill_rect(img, 0, 50, w, 90, PAL["wall_beige"])
    # Vending machine
    fill_rect(img, 20, 60, 30, 70, PAL["red"])
    fill_rect(img, 24, 64, 22, 30, PAL["dark_gray"])
    fill_rect(img, 26, 66, 18, 26, PAL["black"])
    draw_text(img, 28, 70, "SODA", PAL["white"], scale=1)
    # Coffee machine
    fill_rect(img, 180, 100, 20, 30, PAL["dark_gray"])
    fill_rect(img, 184, 104, 12, 10, PAL["black"])
    fill_rect(img, 188, 120, 6, 8, PAL["brown"])
    # Sad plant
    fill_rect(img, 120, 110, 8, 20, PAL["brown"])
    fill_rect(img, 116, 100, 16, 12, PAL["dark_green"])
    fill_rect(img, 118, 96, 12, 6, PAL["green"])
    # Floor
    fill_rect(img, 0, 140, w, 100, PAL["light_gray"])
    # Table
    fill_rect(img, 80, 155, 80, 4, PAL["desk_gray"])
    fill_rect(img, 85, 159, 4, 20, PAL["desk_gray"])
    fill_rect(img, 150, 159, 4, 20, PAL["desk_gray"])


def draw_stage_bg(img, flash=False):
    """Concert stage for chorus sections."""
    w, h = img.size
    # Dark background
    fill_rect(img, 0, 0, w, h, PAL["black"])
    # Stage floor
    fill_rect(img, 0, 170, w, 70, PAL["dark_red"] if not flash else PAL["red"])
    fill_rect(img, 0, 170, w, 3, PAL["yellow"] if flash else PAL["orange"])
    # Stage lights
    colors = [PAL["red"], PAL["blue"], PAL["green"], PAL["yellow"], PAL["purple"]]
    for i, lx in enumerate(range(20, w, 50)):
        c = colors[i % len(colors)]
        if flash:
            # Light beams
            for by in range(0, 170, 2):
                bw = 2 + (by * 20 // 170)
                fill_rect(img, lx + 5 - bw // 2, by, bw, 1, c)
        fill_rect(img, lx, 0, 12, 8, c)
    # Speakers
    fill_rect(img, 5, 140, 20, 30, PAL["dark_gray"])
    fill_rect(img, 231, 140, 20, 30, PAL["dark_gray"])


def draw_elevator_bg(img):
    """Elevator interior."""
    w, h = img.size
    fill_rect(img, 0, 0, w, h, PAL["gray"])
    # Elevator doors
    fill_rect(img, 60, 30, 60, 170, PAL["light_gray"])
    fill_rect(img, 120, 30, 60, 170, PAL["light_gray"])
    fill_rect(img, 118, 30, 4, 170, PAL["dark_gray"])
    # Door frame
    fill_rect(img, 56, 26, 132, 4, PAL["dark_gray"])
    fill_rect(img, 56, 30, 4, 174, PAL["dark_gray"])
    fill_rect(img, 184, 30, 4, 174, PAL["dark_gray"])
    # Floor indicator
    fill_rect(img, 105, 10, 30, 14, PAL["black"])
    draw_text(img, 110, 12, "B2", PAL["red"], scale=1)
    # Floor
    fill_rect(img, 0, 200, w, 40, PAL["floor_brown"])
    # Handrail
    fill_rect(img, 10, 150, 40, 3, PAL["dark_gray"])
    fill_rect(img, 194, 150, 40, 3, PAL["dark_gray"])


def draw_parking_lot_bg(img, time_cycle=0):
    """Parking lot — protagonist walking out."""
    w, h = img.size
    # Sky gradient (simplified)
    fill_rect(img, 0, 0, w, 60, PAL["sky_blue"])
    fill_rect(img, 0, 60, w, 30, PAL["light_blue"])
    # Sun/moon
    fill_rect(img, 200, 15, 16, 16, PAL["yellow"])
    fill_rect(img, 202, 13, 12, 2, PAL["light_yellow"])
    # Building in background
    fill_rect(img, 20, 40, 80, 60, PAL["gray"])
    for wy in range(44, 90, 12):
        for wx in range(26, 90, 14):
            fill_rect(img, wx, wy, 8, 8, PAL["light_yellow"] if random.random() > 0.3 else PAL["dark_blue"])
    # Ground / parking lot
    fill_rect(img, 0, 100, w, 140, PAL["dark_gray"])
    # Parking lines
    for lx in range(20, w, 30):
        fill_rect(img, lx, 130, 2, 40, PAL["yellow"])
    # Cars (simple blocks)
    fill_rect(img, 30, 140, 20, 12, PAL["red"])
    fill_rect(img, 34, 136, 12, 6, PAL["light_red"])
    fill_rect(img, 90, 140, 20, 12, PAL["blue"])
    fill_rect(img, 94, 136, 12, 6, PAL["light_blue"])
    fill_rect(img, 180, 140, 20, 12, PAL["dark_green"])


def draw_rooftop_bg(img):
    """Rooftop scene — city skyline."""
    w, h = img.size
    # Dark sky
    fill_rect(img, 0, 0, w, 120, PAL["dark_blue"])
    # Stars
    pix = img.load()
    rng = random.Random(42)
    for _ in range(30):
        sx, sy = rng.randint(0, w - 1), rng.randint(0, 80)
        pix[sx, sy] = PAL["white"]
    # Skyline
    buildings = [(10, 60), (30, 80), (55, 50), (80, 90), (110, 70),
                 (140, 55), (160, 85), (190, 65), (220, 75)]
    for bx, bh in buildings:
        top = 120 - bh
        fill_rect(img, bx, top, 18, bh, PAL["dark_gray"])
        # Windows
        for wy in range(top + 4, 120, 8):
            for wx in range(bx + 3, bx + 16, 6):
                fill_rect(img, wx, wy, 3, 4,
                          PAL["light_yellow"] if rng.random() > 0.4 else PAL["black"])
    # Rooftop floor
    fill_rect(img, 0, 120, w, 120, PAL["gray"])
    fill_rect(img, 0, 120, w, 2, PAL["dark_gray"])
    # AC unit
    fill_rect(img, 200, 130, 24, 16, PAL["dark_gray"])
    fill_rect(img, 204, 132, 8, 4, PAL["black"])


def draw_cubicle_maze_bg(img, scroll_x=0):
    """Surreal repeating cubicle maze."""
    w, h = img.size
    fill_rect(img, 0, 0, w, 40, PAL["light_gray"])
    # Flickering lights
    for lx in range(10, w, 40):
        fill_rect(img, lx, 35, 20, 3,
                  PAL["white"] if random.random() > 0.2 else PAL["light_yellow"])
    fill_rect(img, 0, 40, w, 100, PAL["wall_beige"])
    fill_rect(img, 0, 140, w, 100, PAL["floor_brown"])

    # Dense cubicle grid, scrolling
    for cx in range(-scroll_x % 40 - 40, w + 40, 40):
        fill_rect(img, cx, 110, 3, 80, PAL["gray"])
        fill_rect(img, cx, 108, 36, 3, PAL["gray"])
        # Tiny monitor
        fill_rect(img, cx + 10, 130, 10, 8, PAL["dark_gray"])
        fill_rect(img, cx + 12, 132, 6, 4, PAL["dark_blue"])
        # Tiny desk
        fill_rect(img, cx + 5, 145, 26, 3, PAL["desk_gray"])


# ── Dialogue Box ─────────────────────────────────────────────────────

def draw_dialogue_box(img, text, y_pos=190, speaker=None):
    """Draw an NES-style dialogue box with text."""
    w = img.size[0]
    box_x, box_w = 8, w - 16
    box_y, box_h = y_pos, 44

    # Box background
    fill_rect(img, box_x, box_y, box_w, box_h, PAL["black"])
    # Border
    draw = ImageDraw.Draw(img)
    draw.rectangle([box_x, box_y, box_x + box_w - 1, box_y + box_h - 1],
                   outline=PAL["white"])
    draw.rectangle([box_x + 1, box_y + 1, box_x + box_w - 2, box_y + box_h - 2],
                   outline=PAL["white"])

    if speaker:
        draw_text(img, box_x + 6, box_y + 4, speaker + ":", PAL["yellow"], scale=1)
        draw_text(img, box_x + 6, box_y + 14, text, PAL["white"], scale=1)
    else:
        # Center the text vertically in the box
        draw_text(img, box_x + 6, box_y + 8, text, PAL["white"], scale=1)


def draw_big_text(img, text, y, color, flash_frame=0, scale=2):
    """Draw large text, optionally flashing."""
    if flash_frame > 0 and (flash_frame // 4) % 2 == 0:
        return  # flash off
    draw_text_centered(img, y, text, color, scale=scale)


# ── Scene Definitions ────────────────────────────────────────────────
# Each scene is a function that takes (img, t_local, frame_num, line_text)
# t_local is time within the scene (0.0 to scene_duration)

@dataclass
class TimedLine:
    text: str
    start: float
    end: float


@dataclass
class Scene:
    name: str
    start: float
    end: float
    bg_func: str       # name of background function
    action: str        # type of action/animation


# Song structure (approximate timestamps from Whisper regions)
SONG_SECTIONS = [
    # (name, start, end, background, action)
    ("intro",           0.0,   16.0,  "office",       "walk_in"),
    ("verse1",         16.0,   40.0,  "office",       "cubicle_life"),
    ("prechorus1",     40.0,   50.0,  "boardroom",    "meeting"),
    ("chorus1",        50.0,   68.0,  "stage",        "rally"),
    ("verse2",         68.0,   92.0,  "cubicle_maze", "maze_wander"),
    ("prechorus2",     92.0,  102.0,  "elevator",     "elevator_ride"),
    ("chorus2",       102.0,  120.0,  "stage",        "rally_intense"),
    ("bridge",        120.0,  135.0,  "breakroom",    "existential"),
    ("guitar_solo",   135.0,  160.0,  "rooftop",      "guitar_solo"),
    ("breakdown",     160.0,  172.0,  "cubicle_maze", "breakdown"),
    ("final_chorus",  172.0,  195.0,  "stage",        "final_rally"),
    ("outro",         195.0,  230.0,  "parking_lot",  "walk_out"),
]


def get_scene_at(t):
    """Get the scene active at time t."""
    for name, start, end, bg, action in SONG_SECTIONS:
        if start <= t < end:
            return Scene(name, start, end, bg, action)
    return Scene("black", 0, 999, "black", "none")


def walk_cycle_sprite(frame_num, speed=8):
    """Return the right walk cycle sprite based on frame."""
    phase = (frame_num // speed) % 4
    if phase == 0:
        return SPR_PROTAG_STAND
    elif phase == 1:
        return SPR_PROTAG_WALK1
    elif phase == 2:
        return SPR_PROTAG_STAND
    else:
        return SPR_PROTAG_WALK2


# ── Main Frame Renderer ─────────────────────────────────────────────

def render_nes_frame(t, frame_num, active_lines):
    """Render a single frame at NES resolution."""
    img = Image.new("RGB", (NES_W, NES_H), PAL["black"])
    scene = get_scene_at(t)
    t_local = t - scene.start
    scene_dur = scene.end - scene.start
    progress = t_local / scene_dur if scene_dur > 0 else 0

    # Draw background
    if scene.bg_func == "office":
        scroll = int(t_local * 15) if scene.action == "walk_in" else 0
        draw_office_bg(img, scroll_x=scroll)
    elif scene.bg_func == "boardroom":
        draw_boardroom_bg(img)
    elif scene.bg_func == "breakroom":
        draw_breakroom_bg(img)
    elif scene.bg_func == "stage":
        flash = (frame_num // 3) % 2 == 0 if "rally" in scene.action else False
        draw_stage_bg(img, flash=flash)
    elif scene.bg_func == "elevator":
        draw_elevator_bg(img)
    elif scene.bg_func == "parking_lot":
        draw_parking_lot_bg(img)
    elif scene.bg_func == "rooftop":
        draw_rooftop_bg(img)
    elif scene.bg_func == "cubicle_maze":
        scroll = int(t_local * 30)
        draw_cubicle_maze_bg(img, scroll_x=scroll)

    # Draw action / sprites
    if scene.action == "walk_in":
        # Protagonist walks in from left
        px = min(int(t_local * 20), 120)
        spr = walk_cycle_sprite(frame_num) if px < 120 else SPR_PROTAG_STAND
        draw_sprite(img, spr, px, 160)

    elif scene.action == "cubicle_life":
        # Protagonist at desk, typing animation
        draw_sprite(img, SPR_PROTAG_STAND, 100, 150)
        # Boss walking by periodically
        boss_x = int((t_local * 12) % 300) - 30
        if 0 <= boss_x < 256:
            draw_sprite(img, SPR_BOSS, boss_x, 155)

    elif scene.action == "meeting":
        # Boardroom scene: boss presenting, choir sitting
        draw_sprite(img, SPR_BOSS, 120, 100, scale=1)
        # Choir members around table
        for i, cx in enumerate([55, 85, 145, 175]):
            draw_sprite(img, SPR_CHOIR, cx, 150)
        draw_sprite(img, SPR_PROTAG_STAND, 105, 150)

    elif scene.action in ("rally", "rally_intense", "final_rally"):
        # Stage scene: everyone on stage, flashing text
        # Choir lined up
        for i in range(6):
            cx = 20 + i * 36
            draw_sprite(img, SPR_CHOIR, cx, 160)
        # Boss center stage
        draw_sprite(img, SPR_BOSS, 116, 145)
        # Protagonist off to side
        px = 40 if scene.action != "final_rally" else 116
        draw_sprite(img, SPR_PROTAG_STAND, px, 150)

        # Big flashing lyrics for chorus
        if scene.action == "final_rally":
            # Extra sprites crowd the stage
            for i in range(4):
                draw_sprite(img, SPR_CHOIR, 10 + i * 60, 170)

    elif scene.action == "maze_wander":
        # Walking through cubicle maze
        spr = walk_cycle_sprite(frame_num, speed=6)
        draw_sprite(img, spr, 120, 155)

    elif scene.action == "elevator_ride":
        # Standing in elevator
        draw_sprite(img, SPR_PROTAG_STAND, 110, 170)
        draw_sprite(img, SPR_BOSS, 140, 168)
        # Awkward standing

    elif scene.action == "existential":
        # Breakroom — protagonist alone, staring at coffee
        draw_sprite(img, SPR_PROTAG_STAND, 100, 155)
        # Coffee cup on table
        fill_rect(img, 130, 150, 6, 5, PAL["white"])
        fill_rect(img, 132, 148, 2, 3, PAL["white"])

    elif scene.action == "guitar_solo":
        # Rooftop — protagonist with arms raised, city behind
        # Epic pose: use stand sprite but draw "guitar" pixels
        draw_sprite(img, SPR_PROTAG_STAND, 120, 135)
        # Guitar (crude pixels)
        pix = img.load()
        for gx in range(130, 145):
            gy = 145 + (gx - 130) // 2
            if 0 <= gx < NES_W and 0 <= gy < NES_H:
                pix[gx, gy] = PAL["brown"]
                if gx < 135:
                    pix[gx, gy - 1] = PAL["brown"]
        # Wind effect on other sprites
        if (frame_num // 10) % 2 == 0:
            # Stars twinkle
            for _ in range(3):
                sx = random.randint(0, NES_W - 1)
                sy = random.randint(0, 60)
                pix[sx, sy] = PAL["white"]

    elif scene.action == "breakdown":
        # Rapid scene cuts simulated by changing sprites quickly
        phase = (frame_num // 8) % 4
        if phase == 0:
            draw_sprite(img, SPR_PROTAG_STAND, 120, 155)
        elif phase == 1:
            draw_sprite(img, SPR_BOSS, 120, 155)
        elif phase == 2:
            for i in range(6):
                draw_sprite(img, SPR_CHOIR, 20 + i * 38, 155)
        else:
            spr = walk_cycle_sprite(frame_num, speed=3)
            draw_sprite(img, spr, 120, 155)

    elif scene.action == "walk_out":
        # Protagonist walks toward camera / right edge
        px = 60 + int(t_local * 10)
        if px < 280:
            spr = walk_cycle_sprite(frame_num, speed=6)
            draw_sprite(img, spr, min(px, 240), 155)

    # ── Draw active lyrics ──
    for line in active_lines:
        line_text = line.text.strip()
        if not line_text:
            continue

        line_progress = (t - line.start) / (line.end - line.start) if line.end > line.start else 0
        line_progress = max(0, min(1, line_progress))

        # Typewriter reveal for dialogue box text
        reveal_count = int(len(line_text) * min(1.0, line_progress * 3))
        revealed = line_text[:reveal_count]

        is_chorus = any(w in line_text.upper() for w in ["SYNERGY", "LEAN IN", "AGILE", "ROCKSTAR", "NINJA", "PIVOT", "BLUE SKY"])
        is_section_header = line_text.startswith("[")

        if is_section_header:
            # Section headers as centered title cards
            clean = line_text.strip("[]")
            draw_big_text(img, clean, 100, PAL["yellow"], scale=2)
        elif is_chorus:
            # Chorus lines as big flashing text
            draw_big_text(img, revealed, 20, PAL["yellow"],
                         flash_frame=frame_num if "SYNERGY" in line_text.upper() else 0,
                         scale=2)
            # Also show in smaller text below
            if len(revealed) > 20:
                draw_text_centered(img, 50, revealed[:20], PAL["white"], scale=1)
                draw_text_centered(img, 60, revealed[20:], PAL["white"], scale=1)
        else:
            # Normal lines in dialogue box
            # Truncate to fit box (about 46 chars at scale 1)
            max_chars = 46
            if len(revealed) > max_chars:
                # Split into two lines
                mid = revealed[:max_chars].rfind(' ')
                if mid < 0:
                    mid = max_chars
                line1 = revealed[:mid]
                line2 = revealed[mid:].strip()
                draw_dialogue_box(img, line1, y_pos=190)
                # Draw second line inside the box
                draw_text(img, 14, 204, line2, PAL["white"], scale=1)
            else:
                draw_dialogue_box(img, revealed, y_pos=190)

    # ── Scene transition effects ──
    # Fade in from black at scene start
    if t_local < 0.5:
        alpha = t_local / 0.5
        arr = np.array(img)
        arr = (arr * alpha).astype(np.uint8)
        img = Image.fromarray(arr)

    # Fade out to black at scene end
    time_left = scene.end - t
    if time_left < 0.3 and scene.name != "outro":
        alpha = time_left / 0.3
        arr = np.array(img)
        arr = (arr * max(0, alpha)).astype(np.uint8)
        img = Image.fromarray(arr)

    return img


# ── Whisper Alignment (reused from lyric_video.py) ───────────────────

def parse_lyrics_file(lyrics_path):
    """Parse lyrics file, keeping section headers and content lines."""
    lines = []
    for raw in Path(lyrics_path).read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        lines.append(line)
    return lines


def align_lyrics_with_whisper(audio_path, lyrics_lines):
    """Detect vocal regions with Whisper, place lyrics proportionally."""
    import whisper

    log.info("Loading Whisper model (base)...")
    model = whisper.load_model("base")
    log.info("Transcribing audio for timing...")
    result = model.transcribe(str(audio_path), word_timestamps=True, language="en")

    segments = result.get("segments", [])
    if not segments:
        log.warning("Whisper returned no segments — falling back to even spacing")
        return evenly_space_lyrics(audio_path, lyrics_lines)

    GAP_THRESHOLD = 3.0
    vocal_regions = []
    cur_start = segments[0]["start"]
    cur_end = segments[0]["end"]
    for seg in segments[1:]:
        if seg["start"] - cur_end < GAP_THRESHOLD:
            cur_end = max(cur_end, seg["end"])
        else:
            vocal_regions.append((cur_start, cur_end))
            cur_start = seg["start"]
            cur_end = seg["end"]
    vocal_regions.append((cur_start, cur_end))

    total_vocal = sum(end - start for start, end in vocal_regions)
    n_lines = len(lyrics_lines)
    log.info("Found %d vocal regions (%.1fs total):", len(vocal_regions), total_vocal)
    for i, (rs, re) in enumerate(vocal_regions):
        log.info("  Region %d: %.1f-%.1fs (%.1fs)", i + 1, rs, re, re - rs)

    region_line_counts = []
    allocated = 0
    for rs, re in vocal_regions:
        share = (re - rs) / total_vocal
        count = round(share * n_lines)
        region_line_counts.append(count)
        allocated += count

    diff = n_lines - allocated
    if diff != 0:
        sorted_idx = sorted(range(len(vocal_regions)),
                            key=lambda i: vocal_regions[i][1] - vocal_regions[i][0],
                            reverse=(diff > 0))
        for idx in sorted_idx:
            if diff == 0:
                break
            if diff > 0:
                region_line_counts[idx] += 1
                diff -= 1
            elif region_line_counts[idx] > 0:
                region_line_counts[idx] -= 1
                diff += 1

    log.info("Line allocation: %s (total %d)", region_line_counts, sum(region_line_counts))

    timed_lines = []
    line_idx = 0
    for (rs, re), count in zip(vocal_regions, region_line_counts):
        if count == 0 or line_idx >= n_lines:
            continue
        per_line = (re - rs) / count
        for j in range(count):
            if line_idx >= n_lines:
                break
            start = rs + j * per_line
            end = start + per_line - 0.15
            timed_lines.append(TimedLine(text=lyrics_lines[line_idx], start=start, end=end))
            line_idx += 1

    return timed_lines


def evenly_space_lyrics(audio_path, lyrics_lines):
    from pydub import AudioSegment
    audio = AudioSegment.from_file(str(audio_path))
    total_duration = len(audio) / 1000
    usable_start = 2.0
    usable_end = total_duration - 2.0
    per_line = (usable_end - usable_start) / len(lyrics_lines)
    return [
        TimedLine(text=line, start=usable_start + i * per_line,
                  end=usable_start + (i + 1) * per_line - 0.1)
        for i, line in enumerate(lyrics_lines)
    ]


# ── Render Pipeline ──────────────────────────────────────────────────

def render_video(audio_path, timed_lines, output_path):
    from moviepy import AudioFileClip, VideoClip

    audio = AudioFileClip(str(audio_path))
    total_duration = audio.duration
    total_frames = int(total_duration * FPS)
    log.info("Rendering %d frames (%.1fs @ %dfps)...", total_frames, total_duration, FPS)

    rendered = [0]

    def make_frame(t):
        frame_num = int(t * FPS)
        if rendered[0] % 150 == 0:
            log.info("  Frame %d/%d (%.1fs)", rendered[0], total_frames, t)
        rendered[0] += 1

        # Find active lyrics
        active = [l for l in timed_lines if l.start <= t <= l.end]

        # Render at NES resolution
        nes_img = render_nes_frame(t, frame_num, active)

        # Scale up to output resolution with nearest-neighbor (crisp pixels)
        out_img = nes_img.resize((OUT_W, OUT_H), Image.NEAREST)

        return np.array(out_img)

    video = VideoClip(make_frame, duration=total_duration)
    video = video.with_audio(audio)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    video.write_videofile(
        str(output_path), fps=FPS, codec="libx264",
        audio_codec="aac", audio_bitrate="192k",
        preset="medium", threads=4, logger=None,
    )
    video.close()
    audio.close()
    log.info("Done! Output: %s (%.1f MB)", output_path, output_path.stat().st_size / 1024 / 1024)


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="8-bit NES-style music video generator")
    parser.add_argument("--audio", required=True, help="Path to audio file")
    parser.add_argument("--lyrics", required=True, help="Path to lyrics text file")
    parser.add_argument("--output", default="nes_video.mp4", help="Output video path")
    parser.add_argument("--no-whisper", action="store_true", help="Skip Whisper, evenly space lyrics")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)

    audio_path = Path(args.audio)
    lyrics_path = Path(args.lyrics)
    output_path = Path(args.output)

    if not audio_path.exists():
        log.error("Audio file not found: %s", audio_path)
        return
    if not lyrics_path.exists():
        log.error("Lyrics file not found: %s", lyrics_path)
        return

    log.info("Parsing lyrics from %s", lyrics_path)
    lyrics_lines = parse_lyrics_file(lyrics_path)
    log.info("  %d lines", len(lyrics_lines))

    if args.no_whisper:
        timed_lines = evenly_space_lyrics(audio_path, lyrics_lines)
    else:
        timed_lines = align_lyrics_with_whisper(audio_path, lyrics_lines)

    render_video(audio_path, timed_lines, output_path)


if __name__ == "__main__":
    main()
