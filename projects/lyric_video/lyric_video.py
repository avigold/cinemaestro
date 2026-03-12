#!/usr/bin/env python3
"""Gritty kinetic typography lyric video generator.

VHS distortion, film grain, screen corruption, aggressive typography,
particle systems, shockwaves, fire, lightning — Se7en meets Fight Club.

Usage:
    python lyric_video.py --audio song.mp3 --lyrics lyrics.txt --output video.mp4

Requirements:
    pip install moviepy pillow numpy openai-whisper pydub
"""

import argparse
import math
import random
import re
import string
import logging
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("lyric_video")

# ── Configuration ──────────────────────────────────────────────────────

WIDTH, HEIGHT = 1920, 1080
FPS = 30
BG_COLOR = (6, 3, 3)  # near-black, warm

# Gritty palette — harsh, high contrast, raw
COLORS = [
    (220, 210, 195),  # dirty white
    (200, 25, 25),    # blood red
    (210, 180, 30),   # bile yellow
    (190, 80, 15),    # rust
    (255, 255, 255),  # stark white (rare contrast)
    (120, 170, 40),   # sickly green
    (160, 140, 130),  # concrete
    (255, 60, 30),    # hot orange-red
    (180, 160, 60),   # ochre
]

FONT_SIZES = [88, 104, 120, 140, 160, 180]
FONT_PATHS = [
    "/System/Library/Fonts/Supplemental/Impact.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/SFNSDisplay.ttf",
]

# Text placement zones — mostly centered, occasional variation
POSITION_ZONES = [
    (0.50, 0.50),   # dead center
    (0.50, 0.45),   # center, slightly high
    (0.50, 0.55),   # center, slightly low
    (0.50, 0.42),   # center, high
    (0.50, 0.50),   # dead center again (weighted)
    (0.38, 0.50),   # left-of-center
    (0.50, 0.50),   # dead center
    (0.50, 0.58),   # center, low
    (0.62, 0.50),   # right-of-center
    (0.50, 0.50),   # dead center
]


# ── Data Types ─────────────────────────────────────────────────────────

@dataclass
class TimedLine:
    text: str
    start: float
    end: float

@dataclass
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    life: float
    size: float
    color: tuple
    gravity: float = 0.0
    drag: float = 0.98

@dataclass
class AnimatedLine:
    line: TimedLine
    effect: str
    color: tuple
    font: ImageFont.FreeTypeFont
    font_size: int
    position: tuple = (0, 0)  # pre-computed aggressive position
    particles: list = field(default_factory=list)


# ── Easing Functions ──────────────────────────────────────────────────

def ease_out_cubic(t):
    return 1 - (1 - t) ** 3

def ease_out_bounce(t):
    if t < 1 / 2.75:
        return 7.5625 * t * t
    elif t < 2 / 2.75:
        t -= 1.5 / 2.75
        return 7.5625 * t * t + 0.75
    elif t < 2.5 / 2.75:
        t -= 2.25 / 2.75
        return 7.5625 * t * t + 0.9375
    else:
        t -= 2.625 / 2.75
        return 7.5625 * t * t + 0.984375

def ease_out_elastic(t):
    if t == 0 or t == 1:
        return t
    return 2 ** (-10 * t) * math.sin((t - 0.075) * (2 * math.pi) / 0.3) + 1

def ease_out_expo(t):
    return 1 if t == 1 else 1 - 2 ** (-10 * t)

def ease_in_expo(t):
    return 0 if t == 0 else 2 ** (10 * (t - 1))

def ease_in_out_cubic(t):
    return 4 * t * t * t if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2

def ease_out_back(t):
    c1 = 1.70158
    c3 = c1 + 1
    return 1 + c3 * (t - 1) ** 3 + c1 * (t - 1) ** 2


# ── Font Loading ──────────────────────────────────────────────────────

def load_font(size):
    for path in FONT_PATHS:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()

def get_text_size(draw, text, font):
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


# ── Position System ──────────────────────────────────────────────────

MARGIN_X = 100
MARGIN_Y = 120  # extra vertical room for wave/shake offsets


def compute_position(tw, th, line_idx):
    """Pick a position, mostly centered, with safe margins."""
    zone = POSITION_ZONES[line_idx % len(POSITION_ZONES)]
    x = int(zone[0] * WIDTH) - tw // 2
    y = int(zone[1] * HEIGHT) - th // 2
    x = max(MARGIN_X, min(x, WIDTH - tw - MARGIN_X))
    y = max(MARGIN_Y, min(y, HEIGHT - th - MARGIN_Y))
    return (x, y)


def clamp_to_screen(x, y, tw, th):
    """Clamp a position so text stays fully on-screen."""
    x = max(MARGIN_X, min(x, WIDTH - tw - MARGIN_X))
    y = max(MARGIN_Y, min(y, HEIGHT - th - MARGIN_Y))
    return x, y


def get_pos(anim, tw, th, frame_num=0):
    """Get position with per-frame micro-jitter, clamped to screen."""
    bx, by = anim.position
    jx = random.randint(-4, 4)
    jy = random.randint(-3, 3)
    x, y = bx + jx, by + jy
    # Re-clamp for the actual text size (may differ from pre-computed due to scaling)
    return clamp_to_screen(x, y, tw, th)


# ── Particle System ──────────────────────────────────────────────────

def spawn_particles(cx, cy, count, speed_range, color, size_range=(2, 6),
                    gravity=0.0, spread=2 * math.pi, angle_offset=0, drag=0.98):
    particles = []
    for _ in range(count):
        angle = angle_offset + random.uniform(-spread / 2, spread / 2)
        speed = random.uniform(*speed_range)
        particles.append(Particle(
            x=cx + random.uniform(-5, 5),
            y=cy + random.uniform(-5, 5),
            vx=math.cos(angle) * speed,
            vy=math.sin(angle) * speed,
            life=1.0,
            size=random.uniform(*size_range),
            color=color,
            gravity=gravity,
            drag=drag,
        ))
    return particles

def update_particles(particles, dt=1/30, decay=0.02):
    alive = []
    for p in particles:
        p.x += p.vx * dt * 60
        p.y += p.vy * dt * 60
        p.vy += p.gravity * dt * 60
        p.vx *= p.drag
        p.vy *= p.drag
        p.life -= decay
        if p.life > 0:
            alive.append(p)
    return alive

def draw_particles(img, particles):
    arr = np.array(img)
    for p in particles:
        x, y = int(p.x), int(p.y)
        s = max(1, int(p.size * p.life))
        if 0 <= x < WIDTH and 0 <= y < HEIGHT:
            alpha = max(0, min(1.0, p.life))
            y1, y2 = max(0, y - s), min(HEIGHT, y + s)
            x1, x2 = max(0, x - s), min(WIDTH, x + s)
            for c in range(3):
                region = arr[y1:y2, x1:x2, c].astype(float)
                region += p.color[c] * alpha * 0.8
                arr[y1:y2, x1:x2, c] = np.clip(region, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


# ── Gritty Post-Processing ──────────────────────────────────────────

def apply_film_grain(img, intensity=0.12):
    """Add photographic film grain noise."""
    arr = np.array(img).astype(np.int16)
    noise = np.random.normal(0, intensity * 60, arr.shape).astype(np.int16)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def apply_vhs_tracking(img, frame_num, severity=0.5):
    """VHS tracking errors — horizontal band displacement and color bleed."""
    if random.random() > 0.25:  # only ~25% of frames
        return img
    arr = np.array(img)
    h = arr.shape[0]
    # 1-3 horizontal error bands
    num_bands = random.randint(1, 3)
    for _ in range(num_bands):
        band_y = random.randint(0, h - 30)
        band_h = random.randint(3, int(25 * severity))
        band_y2 = min(h, band_y + band_h)
        shift = random.randint(int(-50 * severity), int(50 * severity))
        arr[band_y:band_y2] = np.roll(arr[band_y:band_y2], shift, axis=1)
        # Color bleed in band
        if random.random() < 0.4:
            arr[band_y:band_y2, :, random.randint(0, 2)] = np.clip(
                arr[band_y:band_y2, :, random.randint(0, 2)].astype(int) + 40, 0, 255
            ).astype(np.uint8)
    return Image.fromarray(arr)

def apply_screen_corruption(img, frame_num, intensity=0.3):
    """Random block corruption, horizontal tears, dropout lines."""
    if random.random() > 0.18:  # ~18% of frames
        return img
    arr = np.array(img)
    rng = random.Random(frame_num)

    # Horizontal tear
    tear_y = rng.randint(0, HEIGHT - 10)
    tear_h = rng.randint(2, 12)
    tear_shift = rng.randint(-80, 80)
    y2 = min(HEIGHT, tear_y + tear_h)
    arr[tear_y:y2] = np.roll(arr[tear_y:y2], tear_shift, axis=1)

    # Dropout lines (black)
    if rng.random() < 0.5:
        for _ in range(rng.randint(1, 3)):
            ly = rng.randint(0, HEIGHT - 1)
            arr[ly:ly + 1, :] = 0

    # Random noise block
    if rng.random() < 0.4:
        bx = rng.randint(0, WIDTH - 100)
        by = rng.randint(0, HEIGHT - 30)
        bw = rng.randint(30, 150)
        bh = rng.randint(3, 15)
        bx2 = min(bx + bw, WIDTH)
        by2 = min(by + bh, HEIGHT)
        actual_w = bx2 - bx
        actual_h = by2 - by
        if actual_w > 0 and actual_h > 0:
            arr[by:by2, bx:bx2] = np.random.randint(0, 80, (actual_h, actual_w, 3), dtype=np.uint8)

    return Image.fromarray(arr)

def apply_camera_shake(img, frame_num, intensity=6):
    """Constant low-level camera shake with occasional violent spikes."""
    # Base oscillation
    t = frame_num / FPS
    ox = int(intensity * math.sin(t * 7.3) + random.uniform(-2, 2))
    oy = int(intensity * math.sin(t * 5.1 + 1.7) + random.uniform(-2, 2))
    # Occasional violent spike
    if random.random() < 0.04:
        ox += random.randint(-20, 20)
        oy += random.randint(-15, 15)
    arr = np.array(img)
    arr = np.roll(arr, ox, axis=1)
    arr = np.roll(arr, oy, axis=0)
    return Image.fromarray(arr)

def apply_chromatic_aberration(img, amount=5):
    arr = np.array(img)
    result = np.zeros_like(arr)
    if amount > 0:
        result[:, :max(0, WIDTH - amount), 0] = arr[:, min(WIDTH, amount):, 0]
        result[:, :, 1] = arr[:, :, 1]
        result[:, min(WIDTH, amount):, 2] = arr[:, :max(0, WIDTH - amount), 2]
    else:
        result[:, :, :] = arr
    return Image.fromarray(result)

def apply_scanlines(img, intensity=0.12, spacing=2):
    arr = np.array(img).astype(float)
    for y in range(0, HEIGHT, spacing):
        arr[y, :, :] *= (1.0 - intensity)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

def apply_vignette(img, strength=0.5):
    arr = np.array(img).astype(float)
    y_coords = np.linspace(-1, 1, HEIGHT)[:, None]
    x_coords = np.linspace(-1, 1, WIDTH)[None, :]
    dist = np.sqrt(x_coords ** 2 + y_coords ** 2)
    vignette = 1.0 - np.clip((dist - 0.4) * strength * 2, 0, 1)
    arr *= vignette[:, :, None]
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

def apply_shockwave(img, cx, cy, radius, thickness=40, intensity=20):
    arr = np.array(img)
    h, w = arr.shape[:2]
    y_grid, x_grid = np.mgrid[0:h, 0:w]
    dx = x_grid - cx
    dy = y_grid - cy
    dist = np.sqrt(dx ** 2 + dy ** 2)
    ring_mask = np.exp(-((dist - radius) ** 2) / (2 * (thickness / 2) ** 2))
    angle = np.arctan2(dy, dx)
    disp_x = (np.cos(angle) * ring_mask * intensity).astype(int)
    disp_y = (np.sin(angle) * ring_mask * intensity).astype(int)
    src_x = np.clip(x_grid - disp_x, 0, w - 1)
    src_y = np.clip(y_grid - disp_y, 0, h - 1)
    return Image.fromarray(arr[src_y, src_x])


# ── Text rendering helpers ──────────────────────────────────────────

def render_text_gritty(img, text, font, x, y, color, alpha=255, glow=False):
    """Render text with dirty glow and rough edges."""
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)

    if glow:
        # Single dirty glow pass
        glow_layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow_layer)
        gd.text((x, y), text, fill=(*color, int(alpha * 0.4)), font=font)
        glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(radius=15))
        overlay = Image.alpha_composite(overlay, glow_layer)
        od = ImageDraw.Draw(overlay)

    # Main text
    od.text((x, y), text, fill=(*color, alpha), font=font)
    result = Image.alpha_composite(img.convert("RGBA"), overlay)
    return result


def draw_lightning_bolt(draw, x1, y1, x2, y2, color, width=3, depth=0, max_depth=3):
    dx = x2 - x1
    dy = y2 - y1
    length = math.sqrt(dx * dx + dy * dy)
    if length < 5 or depth > max_depth:
        draw.line([(x1, y1), (x2, y2)], fill=color, width=max(1, width))
        return
    mx = (x1 + x2) / 2 + random.uniform(-length * 0.3, length * 0.3)
    my = (y1 + y2) / 2 + random.uniform(-length * 0.3, length * 0.3)
    draw_lightning_bolt(draw, x1, y1, mx, my, color, width, depth + 1, max_depth)
    draw_lightning_bolt(draw, mx, my, x2, y2, color, width, depth + 1, max_depth)
    if depth < 2 and random.random() < 0.4:
        bx = mx + random.uniform(-length * 0.4, length * 0.4)
        by = my + random.uniform(0, length * 0.3)
        draw_lightning_bolt(draw, mx, my, bx, by, tuple(max(0, c - 60) for c in color[:3]) + (color[3],),
                           max(1, width - 1), depth + 1, max_depth)


# ── Effect Renderers ──────────────────────────────────────────────────

def effect_slam_shockwave(img, anim, progress, frame_num):
    """Word SLAMS in with shockwave + screen corruption."""
    text = anim.line.text.upper()
    impact = 0.12

    if progress < impact:
        t = progress / impact
        t_eased = ease_out_expo(t)
        scale = 5.0 - 4.0 * t_eased
        alpha = int(255 * min(t * 5, 1.0))
    elif progress < 0.82:
        scale = 1.0
        alpha = 255
    else:
        t = (progress - 0.82) / 0.18
        scale = 1.0
        alpha = int(255 * (1 - ease_in_expo(min(t, 1.0))))

    font_size = max(12, min(int(anim.font_size * scale), 700))
    font = load_font(font_size)
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    tw, th = get_text_size(od, text, font)
    x, y = get_pos(anim, tw, th, frame_num)

    # Violent shake on impact
    if impact * 0.7 < progress < impact + 0.15:
        shake = int(30 * (1 - (progress - impact * 0.7) / 0.2))
        x += random.randint(-shake, shake)
        y += random.randint(-shake, shake)

    img_rgba = render_text_gritty(img, text, font, x, y, anim.color, alpha, glow=True)

    # Shockwave
    if impact - 0.02 < progress < impact + 0.3:
        wave_t = (progress - impact + 0.02) / 0.32
        radius = wave_t * max(WIDTH, HEIGHT) * 0.7
        intensity = 35 * (1 - wave_t)
        cx = x + tw // 2
        cy = y + th // 2
        img_rgba = apply_shockwave(img_rgba.convert("RGB"), cx, cy,
                                    radius, thickness=60, intensity=int(intensity))
        img_rgba = img_rgba.convert("RGBA")

    # Impact particles
    if abs(progress - impact) < 0.03:
        cx = anim.position[0] + tw // 2
        cy = anim.position[1] + th // 2
        anim.particles.extend(spawn_particles(cx, cy, 60, (5, 20), anim.color, (2, 8), 0.3, drag=0.96))

    anim.particles = update_particles(anim.particles, decay=0.025)
    result = draw_particles(img_rgba.convert("RGB"), anim.particles)

    if progress < impact + 0.1:
        result = apply_chromatic_aberration(result, int(20 * max(0, 1 - progress / (impact + 0.1))))

    return result.convert("RGB")


def effect_explode_particles(img, anim, progress, frame_num):
    """Word DETONATES into debris."""
    text = anim.line.text.upper()
    hold_end = 0.35

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    font = anim.font

    tw, th = get_text_size(od, text, font)
    bx, by = anim.position

    if progress < 0.05:
        t = progress / 0.05
        alpha = int(255 * ease_out_expo(t))
        # White flash
        flash_alpha = int(180 * (1 - t))
        flash = Image.new("RGBA", img.size, (255, 255, 255, flash_alpha))
        overlay = Image.alpha_composite(overlay, flash)
        od = ImageDraw.Draw(overlay)
        od.text((bx, by), text, fill=(*anim.color, alpha), font=font)
    elif progress < hold_end:
        shake = int(5 * math.sin(progress * 80))
        od.text((bx + shake, by), text, fill=(*anim.color, 255), font=font)
    else:
        t = (progress - hold_end) / (1.0 - hold_end)
        t_eased = ease_out_cubic(t)

        if abs(progress - hold_end) < 0.04 and len(anim.particles) < 200:
            cx = bx + tw // 2
            cy = by + th // 2
            anim.particles.extend(spawn_particles(cx, cy, 150, (3, 25), anim.color, (1, 7), 0.15, drag=0.97))
            anim.particles.extend(spawn_particles(cx, cy, 80, (2, 15), (255, 180, 40), (2, 10), -0.05, drag=0.99))

        rng = random.Random(frame_num + hash(text))
        current_x = bx
        for char in text:
            char_w, char_h = get_text_size(od, char, font)
            if char.strip():
                angle = rng.uniform(0, 2 * math.pi)
                distance = t_eased * rng.uniform(500, 1200)
                dx = math.cos(angle) * distance
                dy = math.sin(angle) * distance
                alpha = int(255 * max(0, (1 - t_eased * 1.3)))

                letter_img = Image.new("RGBA", (char_w + 30, char_h + 30), (0, 0, 0, 0))
                ld = ImageDraw.Draw(letter_img)
                ld.text((15, 15), char, fill=(*anim.color, alpha), font=font)
                rotation = t_eased * rng.uniform(-360, 360)
                letter_img = letter_img.rotate(rotation, expand=True, resample=Image.BICUBIC)
                paste_x = int(current_x + dx) - letter_img.width // 2
                paste_y = int(by + dy) - letter_img.height // 2
                if -letter_img.width < paste_x < WIDTH and -letter_img.height < paste_y < HEIGHT:
                    overlay.paste(letter_img, (paste_x, paste_y), letter_img)
            current_x += char_w

    img_rgba = Image.alpha_composite(img.convert("RGBA"), overlay)
    anim.particles = update_particles(anim.particles, decay=0.015)
    result = draw_particles(img_rgba.convert("RGB"), anim.particles)

    if hold_end < progress < hold_end + 0.2:
        wave_t = (progress - hold_end) / 0.2
        cx = bx + tw // 2
        cy = by + th // 2
        result = apply_shockwave(result, cx, cy, wave_t * 800, 50, int(25 * (1 - wave_t)))

    return result.convert("RGB")


def effect_lightning_strike(img, anim, progress, frame_num):
    """Lightning bolt strikes, text appears at impact."""
    text = anim.line.text.upper()
    font = anim.font

    dummy = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    tw, th = get_text_size(dummy, text, font)
    bx, by = anim.position
    cx = bx + tw // 2
    cy = by + th // 2

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)

    strike_time = 0.15

    if progress < strike_time:
        t = progress / strike_time
        bolt_end_y = int(cy * t)
        random.seed(frame_num // 2)
        for _ in range(3):
            bolt_x = cx + random.randint(-100, 100)
            draw_lightning_bolt(od, bolt_x, 0, cx, bolt_end_y, (180, 200, 255, int(200 * t)), 3)
    elif progress < strike_time + 0.05:
        t = (progress - strike_time) / 0.05
        flash_alpha = int(220 * (1 - t))
        flash = Image.new("RGBA", img.size, (220, 230, 255, flash_alpha))
        overlay = Image.alpha_composite(overlay, flash)
        od = ImageDraw.Draw(overlay)
        alpha = int(255 * ease_out_expo(t))
        od.text((bx, by), text, fill=(*anim.color, alpha), font=font)
        if len(anim.particles) < 100:
            anim.particles.extend(spawn_particles(cx, cy, 80, (5, 20), (180, 200, 255), (1, 5), 0.1, drag=0.95))
    elif progress < 0.8:
        img_rgba = render_text_gritty(img, text, font, bx, by, anim.color, 255, glow=True)
        if random.random() < 0.3:
            arc_overlay = Image.new("RGBA", img_rgba.size, (0, 0, 0, 0))
            ad = ImageDraw.Draw(arc_overlay)
            sx = bx + random.randint(0, tw)
            sy = by + random.randint(-10, th + 10)
            draw_lightning_bolt(ad, sx, sy, sx + random.randint(-80, 80),
                              sy + random.randint(-80, 80), (150, 180, 255, 180), 2, max_depth=2)
            img_rgba = Image.alpha_composite(img_rgba.convert("RGBA"), arc_overlay)
        anim.particles = update_particles(anim.particles, decay=0.03)
        return draw_particles(img_rgba.convert("RGB"), anim.particles).convert("RGB")
    else:
        t = (progress - 0.8) / 0.2
        alpha = int(255 * (1 - ease_in_expo(t)))
        od.text((bx, by), text, fill=(*anim.color, alpha), font=font)

    img_rgba = Image.alpha_composite(img.convert("RGBA"), overlay)
    anim.particles = update_particles(anim.particles, decay=0.03)
    return draw_particles(img_rgba.convert("RGB"), anim.particles).convert("RGB")


def effect_fire_text(img, anim, progress, frame_num):
    """Text with fire particles rising from letters."""
    text = anim.line.text.upper()
    font = anim.font

    dummy = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    tw, th = get_text_size(dummy, text, font)
    bx, by = anim.position

    if progress < 0.06:
        alpha = int(255 * (progress / 0.06))
    elif progress > 0.87:
        alpha = int(255 * (1 - (progress - 0.87) / 0.13))
    else:
        alpha = 255

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    core_color = (255, 220, 100)
    od.text((bx, by), text, fill=(*core_color, alpha), font=font)

    glow_layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow_layer)
    gd.text((bx, by), text, fill=(255, 100, 20, int(alpha * 0.6)), font=font)
    glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(radius=12))
    overlay = Image.alpha_composite(glow_layer, overlay)

    img_rgba = Image.alpha_composite(img.convert("RGBA"), overlay)

    if 0.04 < progress < 0.89 and frame_num % 2 == 0:
        fire_colors = [(255, 200, 50), (255, 140, 20), (255, 80, 10), (255, 60, 0)]
        current_x = bx
        for char in text:
            cw, _ = get_text_size(od, char, font)
            if char.strip():
                anim.particles.extend(spawn_particles(
                    current_x + cw // 2, by - 5, 4, (1, 5),
                    random.choice(fire_colors), (2, 8), -0.2,
                    spread=1.0, angle_offset=-math.pi / 2, drag=0.97))
            current_x += cw

    anim.particles = update_particles(anim.particles, decay=0.03)
    return draw_particles(img_rgba.convert("RGB"), anim.particles).convert("RGB")


def effect_glitch_distort(img, anim, progress, frame_num):
    """Extreme glitch — RGB split, block corruption, scanline displacement."""
    text = anim.line.text.upper()
    font = anim.font

    if progress < 0.05:
        alpha = int(255 * (progress / 0.05))
    elif progress > 0.9:
        alpha = int(255 * (1 - (progress - 0.9) / 0.1))
    else:
        alpha = 255

    tw, th = get_text_size(ImageDraw.Draw(img), text, font)
    bx, by = anim.position

    rng = random.Random(frame_num)
    glitch_wave = abs(math.sin(progress * 15)) * 0.7 + 0.3
    if rng.random() < 0.2:
        glitch_wave = 1.0

    split = int(25 * glitch_wave)
    layers = []
    for color, ox, oy in [
        ((255, 0, 0), -split, rng.randint(-5, 5)),
        ((0, 255, 0), rng.randint(-5, 5), rng.randint(-5, 5)),
        ((0, 0, 255), split, rng.randint(-5, 5)),
    ]:
        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        ImageDraw.Draw(layer).text((bx + ox, by + oy), text, fill=(*color, int(alpha * 0.5)), font=font)
        layers.append(layer)

    main = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(main).text((bx, by), text, fill=(*anim.color, alpha), font=font)

    result = img.convert("RGBA")
    for layer in layers:
        result = Image.alpha_composite(result, layer)
    result = Image.alpha_composite(result, main)

    arr = np.array(result)
    num_slices = rng.randint(4, 10) if glitch_wave > 0.6 else rng.randint(2, 5)
    for _ in range(num_slices):
        sy = rng.randint(max(0, by - 40), min(HEIGHT - 1, by + th + 40))
        sh = rng.randint(2, 25)
        sy = max(0, min(sy, HEIGHT - sh))
        arr[sy:sy + sh] = np.roll(arr[sy:sy + sh], rng.randint(-80, 80), axis=1)

    if glitch_wave > 0.5:
        for _ in range(rng.randint(2, 7)):
            bbx = rng.randint(max(0, bx - 60), min(WIDTH - 10, bx + tw + 60))
            bby = rng.randint(max(0, by - 20), min(HEIGHT - 5, by + th + 20))
            bw = rng.randint(20, 150)
            bh = rng.randint(3, 18)
            bbx = max(0, min(bbx, WIDTH - bw))
            bby = max(0, min(bby, HEIGHT - bh))
            arr[bby:bby + bh, bbx:bbx + bw] = [rng.randint(0, 255) for _ in range(3)] + [int(alpha * 0.5)]

    return Image.fromarray(arr).convert("RGB")


def effect_bass_drop(img, anim, progress, frame_num):
    """Screen warps and text crashes down."""
    text = anim.line.text.upper()
    font = anim.font

    dummy = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    tw, th = get_text_size(dummy, text, font)
    bx, by = anim.position

    drop_time = 0.2

    if progress < drop_time:
        t = progress / drop_time
        scale = 0.5 + 0.5 * t
        y = -th + (by + th) * ease_out_bounce(t)
        alpha = int(255 * min(t * 3, 1.0))

        fs = max(12, int(anim.font_size * scale))
        f = load_font(fs)
        tw2, th2 = get_text_size(dummy, text, f)

        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        ImageDraw.Draw(overlay).text((bx, int(y)), text, fill=(*anim.color, alpha), font=f)
        result = Image.alpha_composite(img.convert("RGBA"), overlay)

        if t > 0.8:
            shake = int(25 * (1 - (t - 0.8) / 0.2))
            arr = np.array(result)
            arr = np.roll(arr, random.randint(-shake, shake), axis=1)
            arr = np.roll(arr, random.randint(-shake, shake), axis=0)
            result = Image.fromarray(arr)

        if abs(t - 1.0) < 0.1 and len(anim.particles) < 100:
            anim.particles.extend(spawn_particles(
                bx + tw // 2, by + th, 50, (3, 15), anim.color, (2, 6), 0.2,
                spread=math.pi, angle_offset=-math.pi / 2, drag=0.96))

    elif progress < 0.8:
        result = render_text_gritty(img, text, font, bx, by, anim.color, 255, glow=True)
    else:
        t = (progress - 0.8) / 0.2
        alpha = int(255 * (1 - ease_in_expo(t)))
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        ImageDraw.Draw(overlay).text((bx, by), text, fill=(*anim.color, alpha), font=font)
        result = Image.alpha_composite(img.convert("RGBA"), overlay)

    result = result.convert("RGB")
    anim.particles = update_particles(anim.particles, decay=0.02)
    result = draw_particles(result, anim.particles)
    return result.convert("RGB")


def effect_vortex_spiral(img, anim, progress, frame_num):
    """Letters spiral inward from a vortex."""
    text = anim.line.text.upper()
    font = anim.font

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)

    dummy = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    total_w, total_h = get_text_size(dummy, text, font)
    bx, by = anim.position

    entry_end = 0.4
    exit_start = 0.8
    cx_center = bx + total_w // 2
    cy_center = by + total_h // 2

    rng = random.Random(hash(text))
    current_x = bx
    for i, char in enumerate(text):
        char_w, char_h = get_text_size(od, char, font)
        if not char.strip():
            current_x += char_w
            continue

        start_angle = rng.uniform(0, 2 * math.pi)
        start_radius = rng.uniform(400, 800)

        if progress < entry_end:
            t = ease_out_expo(progress / entry_end)
            radius = start_radius * (1 - t)
            angle = start_angle + t * 4 * math.pi
            spiral_x = cx_center + math.cos(angle) * radius
            spiral_y = cy_center + math.sin(angle) * radius
            cx = spiral_x + (current_x - spiral_x) * t
            cy = spiral_y + (by - spiral_y) * t
            alpha = int(255 * min(t * 2, 1.0))
        elif progress < exit_start:
            cx = current_x
            cy = by
            alpha = 255
        else:
            t = (progress - exit_start) / (1.0 - exit_start)
            radius = start_radius * ease_in_expo(t) * 0.5
            angle = start_angle - t * 3 * math.pi
            cx = current_x + math.cos(angle) * radius
            cy = by + math.sin(angle) * radius
            alpha = int(255 * (1 - t))

        letter_size = char_w + 40
        letter_img = Image.new("RGBA", (letter_size, letter_size), (0, 0, 0, 0))
        ImageDraw.Draw(letter_img).text((20, 20 - char_h // 4), char,
                                        fill=(*anim.color, alpha), font=font)
        if progress < entry_end:
            rot = (1 - ease_out_expo(progress / entry_end)) * 720
            letter_img = letter_img.rotate(rot, expand=True, resample=Image.BICUBIC)

        px = int(cx) - letter_img.width // 2
        py = int(cy) - letter_img.height // 2
        if -letter_img.width < px < WIDTH and -letter_img.height < py < HEIGHT:
            overlay.paste(letter_img, (px, py), letter_img)
        current_x += char_w

    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")


def effect_scatter_assemble(img, anim, progress, frame_num):
    """Letters scattered then assemble with trails."""
    text = anim.line.text.upper()
    font = anim.font

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)

    dummy = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    total_w, total_h = get_text_size(dummy, text, font)
    bx, by = anim.position

    assemble_end = 0.35
    exit_start = 0.8
    rng = random.Random(hash(text))

    current_x = bx
    for char in text:
        char_w, _ = get_text_size(od, char, font)
        scatter_x = rng.randint(50, WIDTH - 100)
        scatter_y = rng.randint(50, HEIGHT - 100)

        if progress < assemble_end:
            t = ease_out_elastic(progress / assemble_end)
            cx = scatter_x + (current_x - scatter_x) * t
            cy = scatter_y + (by - scatter_y) * t
            alpha = int(100 + 155 * t)
            # Motion trail
            if t < 0.9:
                for trail in range(4):
                    tt = max(0, t - trail * 0.05)
                    tx = scatter_x + (current_x - scatter_x) * tt
                    ty = scatter_y + (by - scatter_y) * tt
                    ta = int(alpha * 0.12 * (1 - trail / 4))
                    if char.strip():
                        od.text((int(tx), int(ty)), char, fill=(*anim.color, ta), font=font)
        elif progress < exit_start:
            cx, cy, alpha = current_x, by, 255
        else:
            t = (progress - exit_start) / (1.0 - exit_start)
            cx, cy = current_x, by
            alpha = int(255 * (1 - ease_in_expo(t)))

        if char.strip():
            od.text((int(cx), int(cy)), char, fill=(*anim.color, alpha), font=font)
        current_x += char_w

    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")


def effect_flash_strobe(img, anim, progress, frame_num):
    """Rapid strobing flash cuts with position jumps."""
    text = anim.line.text.upper()
    font = anim.font

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)

    tw, th = get_text_size(od, text, font)
    bx, by = anim.position
    rng = random.Random(frame_num)

    # Multiple positions including assigned position
    positions = [
        (bx, by),
        (30, 50),
        (WIDTH - tw - 30, HEIGHT - th - 50),
        (30, HEIGHT - th - 50),
        (WIDTH - tw - 30, 50),
    ]

    if progress < 0.2:
        pos_idx = int(progress * 30) % len(positions)
        x, y = positions[pos_idx]
        visible = int(progress * 40) % 3 != 0
        if visible:
            if int(progress * 20) % 2 == 0:
                od.text((x, y), text, fill=(*anim.color, 255), font=font)
            else:
                inv = tuple(255 - c for c in anim.color)
                od.text((x, y), text, fill=(*inv, 255), font=font)
                flash = Image.new("RGBA", img.size, (255, 255, 255, 100))
                overlay = Image.alpha_composite(flash, overlay)
    elif progress < 0.8:
        jitter = random.randint(-8, 8) if rng.random() < 0.15 else 0
        od.text((bx + jitter, by + jitter), text, fill=(*anim.color, 255), font=font)
    else:
        t = (progress - 0.8) / 0.2
        pos_idx = int(progress * 25) % len(positions)
        x, y = positions[pos_idx]
        if rng.random() > t * 0.8:
            od.text((x, y), text, fill=(*anim.color, int(255 * (1 - t))), font=font)

    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")


def effect_gravity_bounce(img, anim, progress, frame_num):
    """Letters drop from above with staggered physics."""
    text = anim.line.text.upper()
    font = anim.font

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)

    dummy = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    total_w, total_h = get_text_size(dummy, text, font)
    bx, by = anim.position

    entry_end = 0.4
    exit_start = 0.82
    rng = random.Random(hash(text))

    current_x = bx
    for i, char in enumerate(text):
        char_w, _ = get_text_size(od, char, font)
        if not char.strip():
            current_x += char_w
            continue

        delay = i * 0.02
        local_progress = max(0, progress - delay)
        local_t = min(1.0, local_progress / entry_end) if entry_end > 0 else 1.0

        if local_progress < entry_end and progress < exit_start:
            bounce_t = ease_out_bounce(local_t)
            y = -total_h * 2 + (by + total_h * 2) * bounce_t
            alpha = int(255 * min(local_t * 3, 1.0))
            if abs(bounce_t - 1.0) < 0.05 and len(anim.particles) < 300:
                anim.particles.extend(spawn_particles(
                    current_x + char_w // 2, by + total_h, 5,
                    (2, 8), anim.color, (1, 4), 0.3,
                    spread=math.pi, angle_offset=-math.pi / 2))
        elif progress < exit_start:
            y, alpha = by, 255
        else:
            t = (progress - exit_start) / (1.0 - exit_start)
            y = by + HEIGHT * ease_in_expo(t)
            alpha = int(255 * (1 - t))

        od.text((current_x, int(y)), char, fill=(*anim.color, alpha), font=font)
        current_x += char_w

    img_rgba = Image.alpha_composite(img.convert("RGBA"), overlay)
    anim.particles = update_particles(anim.particles, decay=0.04)
    return draw_particles(img_rgba.convert("RGB"), anim.particles).convert("RGB")


def effect_typewriter_glitch(img, anim, progress, frame_num):
    """Typewriter with glitch corruption."""
    text = anim.line.text
    font = anim.font

    type_end = 0.6
    chars_visible = int(len(text) * min(progress / type_end, 1.0))
    visible_text = text[:chars_visible]

    dummy = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    tw, th = get_text_size(dummy, text, font)
    bx, by = anim.position

    alpha = int(255 * (1 - (progress - 0.88) / 0.12)) if progress > 0.88 else 255

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    od.text((bx, by), visible_text, fill=(*anim.color, alpha), font=font)

    if progress < type_end:
        cursor_w, _ = get_text_size(od, visible_text, font)
        if int(progress * 12) % 2 == 0:
            od.rectangle([bx + cursor_w + 4, by, bx + cursor_w + 8, by + th],
                        fill=(*anim.color, alpha))

        # Glitch on new chars
        if chars_visible > 0 and frame_num % 3 == 0 and random.random() < 0.5:
            result_arr = np.array(Image.alpha_composite(img.convert("RGBA"), overlay))
            sy = random.randint(max(0, by - 5), min(HEIGHT - 1, by + th + 5))
            sh = random.randint(2, 10)
            sy = max(0, min(sy, HEIGHT - sh))
            result_arr[sy:sy + sh] = np.roll(result_arr[sy:sy + sh], random.randint(-40, 40), axis=1)
            return Image.fromarray(result_arr).convert("RGB")

    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")


def effect_zoom_warp(img, anim, progress, frame_num):
    """Text zooms from infinity with speed lines."""
    text = anim.line.text.upper()
    bx, by = anim.position

    if progress < 0.5:
        t = progress / 0.5
        scale = 0.02 + 2.0 * ease_out_expo(t)
        alpha = int(255 * min(t * 3, 1.0))
    elif progress < 0.75:
        scale, alpha = 2.0, 255
    else:
        t = (progress - 0.75) / 0.25
        scale = 2.0 + 8.0 * ease_in_expo(t)
        alpha = int(255 * (1 - ease_in_expo(t)))

    fs = max(8, min(int(anim.font_size * scale), 900))
    font = load_font(fs)

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    tw, th = get_text_size(od, text, font)
    # Center the zoom on the assigned position
    x = bx + (anim.font_size - fs) // 2 if scale < 2.5 else (WIDTH - tw) // 2
    y = by + (anim.font_size - fs) // 2 if scale < 2.5 else (HEIGHT - th) // 2

    od.text((x, y), text, fill=(*anim.color, alpha), font=font)

    # Speed lines
    if progress < 0.5:
        t = progress / 0.5
        lines_overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        ld = ImageDraw.Draw(lines_overlay)
        rng = random.Random(frame_num // 2)
        cx = bx + tw // 4
        cy = by + th // 2
        for _ in range(int(30 * (1 - t))):
            angle = rng.uniform(0, 2 * math.pi)
            sx = cx + math.cos(angle) * 50
            sy = cy + math.sin(angle) * 50
            ex = cx + math.cos(angle) * rng.uniform(300, 800)
            ey = cy + math.sin(angle) * rng.uniform(300, 800)
            ld.line([(sx, sy), (ex, ey)], fill=(*anim.color, int(80 * (1 - t))), width=2)
        overlay = Image.alpha_composite(lines_overlay, overlay)

    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")


def effect_wave_distort(img, anim, progress, frame_num):
    """Letters wave with color cycling."""
    text = anim.line.text.upper()
    font = anim.font

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)

    dummy = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    total_w, total_h = get_text_size(dummy, text, font)
    bx, by = anim.position

    if progress < 0.08:
        alpha = int(255 * (progress / 0.08))
    elif progress > 0.88:
        alpha = int(255 * (1 - (progress - 0.88) / 0.12))
    else:
        alpha = 255

    current_x = bx
    char_h = total_h
    for i, char in enumerate(text):
        char_w, _ = get_text_size(od, char, font)
        if not char.strip():
            current_x += char_w
            continue
        phase = progress * 10 - i * 0.6
        wave_y = math.sin(phase) * 45
        # Clamp vertical position to stay on-screen
        cy = int(by + wave_y)
        cy = max(MARGIN_Y, min(cy, HEIGHT - char_h - MARGIN_Y))
        # Harsh color cycling
        hue = (progress * 3 + i * 0.3) % 1.0
        r = int(128 + 127 * math.sin(hue * 2 * math.pi))
        g = int(80 + 80 * math.sin(hue * 2 * math.pi + 2.1))
        b = int(60 + 60 * math.sin(hue * 2 * math.pi + 4.2))
        od.text((current_x, cy), char, fill=(r, g, b, alpha), font=font)
        current_x += char_w

    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")


def effect_spin_in(img, anim, progress, frame_num):
    """Word spins in with elastic overshoot."""
    text = anim.line.text.upper()
    font = anim.font

    overlay = Image.new("RGBA", (WIDTH * 2, HEIGHT * 2), (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)

    entry = 0.25
    exit_start = 0.8

    if progress < entry:
        t = ease_out_elastic(progress / entry)
        rotation = 720 * (1 - t)
        scale = max(0.1, t)
        alpha = int(255 * min(progress / entry * 3, 1.0))
    elif progress < exit_start:
        rotation, scale, alpha = 0, 1.0, 255
    else:
        t = (progress - exit_start) / (1.0 - exit_start)
        rotation = -180 * ease_in_expo(t)
        scale = max(0.1, 1.0 - 0.8 * t)
        alpha = int(255 * (1 - t))

    fs = max(12, int(anim.font_size * scale))
    sf = load_font(fs)
    tw, th = get_text_size(od, text, sf)

    # Map position to double-size canvas
    bx, by = anim.position
    x = bx + WIDTH // 2
    y = by + HEIGHT // 2
    od.text((x, y), text, fill=(*anim.color, alpha), font=sf)
    overlay = overlay.rotate(rotation, center=(bx + tw // 2 + WIDTH // 2, by + th // 2 + HEIGHT // 2),
                            resample=Image.BICUBIC)
    overlay = overlay.crop((WIDTH // 2, HEIGHT // 2, WIDTH // 2 + WIDTH, HEIGHT // 2 + HEIGHT))

    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")


def effect_matrix_rain(img, anim, progress, frame_num):
    """Matrix code rain assembles into the word."""
    text = anim.line.text.upper()
    font = anim.font

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)

    dummy = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    total_w, total_h = get_text_size(dummy, text, font)
    bx, by = anim.position

    assemble_start = 0.35
    assemble_end = 0.55
    exit_start = 0.85

    matrix_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789@#$%&*"
    small_font = load_font(24)

    if progress < assemble_start:
        rng = random.Random(frame_num)
        for col in range(40):
            col_x = int(col * WIDTH / 40) + rng.randint(-10, 10)
            for row in range(15):
                char_y = (frame_num * 8 + row * 70 + col * 37) % (HEIGHT + 200) - 100
                char = rng.choice(matrix_chars)
                brightness = max(0, 255 - row * 25)
                od.text((col_x, char_y), char, fill=(0, brightness, 0, int(brightness * 0.7)), font=small_font)

        t = progress / assemble_start
        if t > 0.5:
            h_alpha = int(200 * (t - 0.5) * 2)
            current_x = bx
            for char in text:
                cw, _ = get_text_size(od, char, font)
                if char.strip():
                    od.text((current_x, by), char, fill=(0, 255, 0, h_alpha), font=font)
                current_x += cw

    elif progress < assemble_end:
        t = ease_out_back((progress - assemble_start) / (assemble_end - assemble_start))
        rng = random.Random(hash(text))
        current_x = bx
        for char in text:
            cw, _ = get_text_size(od, char, font)
            if char.strip():
                sx = rng.randint(50, WIDTH - 100)
                sy = rng.randint(-200, HEIGHT + 200)
                cx = sx + (current_x - sx) * t
                cy = sy + (by - sy) * t
                r = int(anim.color[0] * t)
                g = int(255 + (anim.color[1] - 255) * t)
                b = int(anim.color[2] * t)
                od.text((int(cx), int(cy)), char, fill=(r, g, b, 255), font=font)
            current_x += cw

    elif progress < exit_start:
        od.text((bx, by), text, fill=(*anim.color, 255), font=font)
    else:
        t = (progress - exit_start) / (1.0 - exit_start)
        od.text((bx, by), text, fill=(*anim.color, int(255 * (1 - t))), font=font)

    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")


def effect_trail_afterimage(img, anim, progress, frame_num):
    """Text sweeps across leaving color-shifted trails."""
    text = anim.line.text.upper()
    font = anim.font

    dummy = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    tw, th = get_text_size(dummy, text, font)
    bx, by = anim.position

    if progress < 0.6:
        t = ease_in_out_cubic(progress / 0.6)
        x = -tw + (bx + tw) * t
        y = by + int(60 * math.sin(t * math.pi))
        alpha = int(255 * min(progress * 5, 1.0))
    elif progress < 0.85:
        x, y, alpha = bx, by, 255
    else:
        t = (progress - 0.85) / 0.15
        x, y = bx, by
        alpha = int(255 * (1 - t))

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)

    if progress < 0.6:
        trail_colors = [(200, 40, 40), (40, 200, 40), (40, 40, 200), (200, 200, 40)]
        for i, offset in enumerate([0.08, 0.06, 0.04, 0.02]):
            tp = max(0, progress - offset)
            tt = ease_in_out_cubic(tp / 0.6) if tp < 0.6 else 1.0
            tx = int(-tw + (bx + tw) * tt)
            ty = by + int(60 * math.sin(tt * math.pi))
            ta = int(alpha * 0.2 * (1 - i / len(trail_colors)))
            od.text((tx, ty), text, fill=(*trail_colors[i], ta), font=font)

    od.text((int(x), int(y)), text, fill=(*anim.color, alpha), font=font)
    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")


def effect_mirror_kaleidoscope(img, anim, progress, frame_num):
    """Text with kaleidoscope mirror effect."""
    text = anim.line.text.upper()
    font = anim.font

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    tw, th = get_text_size(od, text, font)
    bx, by = anim.position

    if progress < 0.1:
        alpha = int(255 * (progress / 0.1))
    elif progress > 0.85:
        alpha = int(255 * (1 - (progress - 0.85) / 0.15))
    else:
        alpha = 255

    od.text((bx, by), text, fill=(*anim.color, alpha), font=font)

    h_flip = overlay.transpose(Image.FLIP_LEFT_RIGHT)
    v_flip = overlay.transpose(Image.FLIP_TOP_BOTTOM)
    rotation = progress * 30
    h_flip = h_flip.rotate(rotation, center=(WIDTH // 2, HEIGHT // 2), resample=Image.BICUBIC)
    v_flip = v_flip.rotate(-rotation, center=(WIDTH // 2, HEIGHT // 2), resample=Image.BICUBIC)

    result = img.convert("RGBA")
    for mirror in [h_flip, v_flip]:
        arr = np.array(mirror).astype(float)
        arr[:, :, 3] = np.clip(arr[:, :, 3] * 0.2, 0, 255)
        result = Image.alpha_composite(result, Image.fromarray(arr.astype(np.uint8)))

    glow = overlay.filter(ImageFilter.GaussianBlur(radius=10))
    result = Image.alpha_composite(result, glow)
    result = Image.alpha_composite(result, overlay)
    return result.convert("RGB")


def effect_3d_flip(img, anim, progress, frame_num):
    """Text flips in from behind the screen."""
    text = anim.line.text.upper()
    font = load_font(anim.font_size)

    dummy = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    tw, th = get_text_size(dummy, text, font)
    bx, by = anim.position

    text_img = Image.new("RGBA", (tw + 40, th + 40), (0, 0, 0, 0))
    ImageDraw.Draw(text_img).text((20, 20), text, fill=(*anim.color, 255), font=font)

    if progress < 0.3:
        t = ease_out_back(progress / 0.3)
        y_scale = max(0.01, t)
    elif progress < 0.75:
        y_scale = 1.0
    else:
        t = ease_in_expo((progress - 0.75) / 0.25)
        y_scale = max(0.01, 1 - t)

    new_h = max(1, int(text_img.height * y_scale))
    scaled = text_img.resize((text_img.width, new_h), Image.BICUBIC)

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    paste_y = by + (th - new_h) // 2
    overlay.paste(scaled, (bx, paste_y), scaled)

    glow = overlay.filter(ImageFilter.GaussianBlur(radius=8))
    result = Image.alpha_composite(img.convert("RGBA"), glow)
    result = Image.alpha_composite(result, overlay)
    return result.convert("RGB")


def effect_neon_pulse(img, anim, progress, frame_num):
    """Pulsing neon sign with flicker."""
    text = anim.line.text.upper()
    font = anim.font

    dummy = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    tw, th = get_text_size(dummy, text, font)
    bx, by = anim.position

    rng = random.Random(frame_num)
    flicker = 1.0
    if progress < 0.15:
        flicker = 0.3 + 0.7 * (1 if int(progress * 40) % 3 != 0 else 0.2)
    elif rng.random() < 0.06:
        flicker = 0.2

    alpha = int(255 * flicker)
    if progress > 0.85:
        alpha = int(alpha * (1 - (progress - 0.85) / 0.15))

    result = render_text_gritty(img, text, font, bx, by, anim.color, alpha, glow=True)
    return result.convert("RGB")


# ── Effect Registry ───────────────────────────────────────────────────

EFFECTS = {
    "slam_shockwave": effect_slam_shockwave,
    "explode_particles": effect_explode_particles,
    "lightning_strike": effect_lightning_strike,
    "fire_text": effect_fire_text,
    "glitch_distort": effect_glitch_distort,
    "bass_drop": effect_bass_drop,
    "vortex_spiral": effect_vortex_spiral,
    "scatter_assemble": effect_scatter_assemble,
    "flash_strobe": effect_flash_strobe,
    "gravity_bounce": effect_gravity_bounce,
    "typewriter_glitch": effect_typewriter_glitch,
    "zoom_warp": effect_zoom_warp,
    "wave_distort": effect_wave_distort,
    "spin_in": effect_spin_in,
    "matrix_rain": effect_matrix_rain,
    "trail_afterimage": effect_trail_afterimage,
    "mirror_kaleidoscope": effect_mirror_kaleidoscope,
    "3d_flip": effect_3d_flip,
    "neon_pulse": effect_neon_pulse,
}

HIGH_ENERGY = ["slam_shockwave", "explode_particles", "lightning_strike", "bass_drop",
               "flash_strobe", "gravity_bounce"]
MEDIUM_ENERGY = ["fire_text", "vortex_spiral", "glitch_distort", "3d_flip",
                 "zoom_warp", "trail_afterimage", "spin_in"]
LOW_ENERGY = ["neon_pulse", "matrix_rain", "wave_distort", "scatter_assemble",
              "typewriter_glitch", "mirror_kaleidoscope"]


# ── Semantic Effect Matching ─────────────────────────────────────────
# Maps keyword patterns in lyrics to effects that make thematic sense.

SEMANTIC_RULES = [
    # (keyword patterns, effect, reason)
    (["circle back", "loop", "close the loop"], "vortex_spiral", "circular motion"),
    (["slam", "smash", "hit", "crash"], "slam_shockwave", "impact"),
    (["explode", "blow", "burst", "bang", "boom"], "explode_particles", "explosion"),
    (["synergy", "power", "loud", "scream"], "slam_shockwave", "aggressive emphasis"),
    (["move the needle", "pivot", "fast", "agile"], "zoom_warp", "speed/motion"),
    (["low-hanging", "drop", "gravity", "fall", "heavy lifting"], "gravity_bounce", "weight/falling"),
    (["blue sky", "sky", "rise", "rising", "glowing", "dream"], "fire_text", "upward/aspiration"),
    (["lightning", "electric", "spark", "ignite", "light"], "lightning_strike", "electrical"),
    (["ping", "flash", "quick", "check", "temperature"], "flash_strobe", "quick/instant"),
    (["ninja", "rockstar", "thought-leader"], "spin_in", "dramatic entrance"),
    (["lean in", "lean"], "bass_drop", "forward pressure"),
    (["glitch", "bandwidth", "offline", "constraints"], "glitch_distort", "technical/broken"),
    (["unpack", "deep dive", "dive"], "scatter_assemble", "pulling apart"),
    (["it is what it is", "needful", "per my last", "kindly"], "typewriter_glitch", "deadpan/ironic"),
    (["pin in it", "stop", "hard stop"], "slam_shockwave", "abrupt halt"),
    (["ball rolling", "rolling", "track"], "trail_afterimage", "rolling motion"),
    (["holistic", "vision", "alignment"], "mirror_kaleidoscope", "wholeness/reflection"),
    (["ideation", "thinking", "mindset", "plan"], "matrix_rain", "thought/computation"),
    (["stage", "crowd", "hear", "bow", "everyone", "great meeting"], "3d_flip", "performance/reveal"),
    (["fire", "burn", "hot", "heat"], "fire_text", "heat"),
    (["through", "corporate age", "crew"], "zoom_warp", "pushing through"),
    (["wave", "flow"], "wave_distort", "wave motion"),
    (["email", "needful", "eod"], "typewriter_glitch", "office drudgery"),
]


def pick_effect_semantic(lyric_text, last_effect=None, used_for_line=None):
    """Pick an effect based on lyric content. Falls back to energy-aware random."""
    text_lower = lyric_text.lower()

    # Check semantic rules
    candidates = []
    for keywords, effect, reason in SEMANTIC_RULES:
        for kw in keywords:
            if kw in text_lower:
                candidates.append((effect, reason, kw))
                break

    # Filter out last effect to avoid repetition
    if candidates:
        filtered = [c for c in candidates if c[0] != last_effect]
        if filtered:
            choice = filtered[0]
        else:
            choice = candidates[0]
        return choice[0]

    # ALL CAPS = high energy
    if lyric_text == lyric_text.upper() and len(lyric_text) > 3:
        pool = [e for e in HIGH_ENERGY if e != last_effect]
        return random.choice(pool) if pool else random.choice(HIGH_ENERGY)

    # Ellipsis = contemplative/low energy
    if "…" in lyric_text or "..." in lyric_text:
        pool = [e for e in LOW_ENERGY if e != last_effect]
        return random.choice(pool) if pool else random.choice(LOW_ENERGY)

    # Fallback: energy-aware random
    all_effects = list(EFFECTS.keys())
    pool = [e for e in all_effects if e != last_effect]
    return random.choice(pool)


# ── Lyrics Parsing & Whisper Alignment ────────────────────────────────

def parse_lyrics_file(lyrics_path):
    lines = []
    for line in lyrics_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            continue
        lines.append(line)
    return lines


def _normalize_text(text):
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def align_lyrics_with_whisper(audio_path, lyrics_lines):
    """Use Whisper to detect vocal regions, then place lyrics only where vocals exist.

    Instrumental gaps (guitar solos, breaks) get no text overlay.
    """
    import whisper

    log.info("Loading Whisper model (base)...")
    model = whisper.load_model("base")

    log.info("Transcribing audio for timing...")
    result = model.transcribe(str(audio_path), word_timestamps=True, language="en")

    segments = result.get("segments", [])
    if not segments:
        log.warning("Whisper returned no segments — falling back to even spacing")
        return evenly_space_lyrics(audio_path, lyrics_lines)

    log.info("Whisper found %d segments covering %.1f-%.1fs",
             len(segments), segments[0]["start"], segments[-1]["end"])

    # ── Step 1: merge adjacent Whisper segments into vocal regions ──
    # If the gap between two segments is < GAP_THRESHOLD, they're part of the
    # same vocal region.  Larger gaps are instrumental breaks.
    GAP_THRESHOLD = 3.0  # seconds — gaps longer than this are instrumentals

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
    log.info("Identified %d vocal regions (%.1fs total vocal):",
             len(vocal_regions), total_vocal)
    for i, (rs, re) in enumerate(vocal_regions):
        log.info("  Region %d: %.1f-%.1fs (%.1fs)", i + 1, rs, re, re - rs)

    # ── Step 2: allocate lines to regions proportionally ──
    # Each vocal region gets a share of lines proportional to its duration.
    # Lines are evenly spaced within their region — no line ever spans a gap.

    n_lines = len(lyrics_lines)

    # Compute how many lines each region gets
    region_line_counts = []
    allocated = 0
    for rs, re in vocal_regions:
        share = (re - rs) / total_vocal
        count = round(share * n_lines)
        region_line_counts.append(count)
        allocated += count

    # Fix rounding: add/remove from the largest/smallest regions
    diff = n_lines - allocated
    if diff != 0:
        # Sort regions by duration (descending for surplus, ascending for deficit)
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

    log.info("Line allocation per region: %s (total %d)",
             region_line_counts, sum(region_line_counts))

    # Distribute lines evenly within each region
    timed_lines = []
    line_idx = 0
    for (rs, re), count in zip(vocal_regions, region_line_counts):
        if count == 0 or line_idx >= n_lines:
            continue
        region_dur = re - rs
        per_line = region_dur / count
        for j in range(count):
            if line_idx >= n_lines:
                break
            start = rs + j * per_line
            end = start + per_line - 0.15
            timed_lines.append(TimedLine(
                text=lyrics_lines[line_idx], start=start, end=end))
            log.info("  [%.1f-%.1f] %s", start, end, lyrics_lines[line_idx][:50])
            line_idx += 1

    return timed_lines


def evenly_space_lyrics(audio_path, lyrics_lines):
    from pydub import AudioSegment
    audio = AudioSegment.from_file(str(audio_path))
    total_duration = len(audio) / 1000
    usable_start = 2.0
    usable_end = total_duration - 2.0
    usable_duration = usable_end - usable_start
    n_lines = len(lyrics_lines)
    line_duration = usable_duration / n_lines
    timed_lines = []
    for i, text in enumerate(lyrics_lines):
        start = usable_start + i * line_duration
        timed_lines.append(TimedLine(text=text, start=start, end=start + line_duration - 0.1))
    return timed_lines


# ── Video Assembly ────────────────────────────────────────────────────

def assign_effects(timed_lines):
    """Assign effects semantically based on lyric content + aggressive positions."""
    animated = []
    last_effect = None

    for i, line in enumerate(timed_lines):
        effect = pick_effect_semantic(line.text, last_effect)
        last_effect = effect

        color = random.choice(COLORS)
        font_size = random.choice(FONT_SIZES)
        font = load_font(font_size)

        dummy = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        tw, _ = get_text_size(dummy, line.text.upper(), font)
        while tw > WIDTH - 100 and font_size > 48:
            font_size -= 8
            font = load_font(font_size)
            tw, _ = get_text_size(dummy, line.text.upper(), font)

        tw, th = get_text_size(dummy, line.text.upper(), font)
        position = compute_position(tw, th, i)

        animated.append(AnimatedLine(
            line=line, effect=effect, color=color,
            font=font, font_size=font_size, position=position,
        ))

    return animated


def render_frame(animated_lines, time_s, frame_num):
    """Render a single frame with full gritty post-processing."""
    img = Image.new("RGB", (WIDTH, HEIGHT), BG_COLOR)

    # Occasional deep red background flash
    if random.random() < 0.03:
        arr = np.array(img)
        arr[:, :, 0] = np.clip(arr[:, :, 0].astype(int) + random.randint(10, 30), 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

    for anim in animated_lines:
        if anim.line.start <= time_s <= anim.line.end:
            duration = anim.line.end - anim.line.start
            progress = (time_s - anim.line.start) / duration if duration > 0 else 0
            progress = max(0, min(1, progress))
            img = EFFECTS[anim.effect](img, anim, progress, frame_num)

    # ── Gritty post-processing pipeline ──
    img = apply_vignette(img, strength=0.5)
    img = apply_film_grain(img, intensity=0.14)
    img = apply_camera_shake(img, frame_num, intensity=5)
    img = apply_vhs_tracking(img, frame_num, severity=0.5)
    img = apply_screen_corruption(img, frame_num, intensity=0.3)
    img = apply_scanlines(img, intensity=0.10, spacing=2)
    img = apply_chromatic_aberration(img, amount=4)

    return np.array(img)


def render_video(audio_path, animated_lines, output_path):
    from moviepy import AudioFileClip, VideoClip

    audio = AudioFileClip(str(audio_path))
    total_duration = audio.duration
    total_frames = int(total_duration * FPS)
    log.info("Rendering %d frames (%.1fs @ %dfps)...", total_frames, total_duration, FPS)

    rendered = [0]

    def make_frame(t):
        frame_num = int(t * FPS)
        if rendered[0] % 100 == 0:
            log.info("  Frame %d/%d (%.1fs)", rendered[0], total_frames, t)
        rendered[0] += 1
        return render_frame(animated_lines, t, frame_num)

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


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Gritty kinetic typography lyric video generator")
    parser.add_argument("--audio", required=True, help="Path to audio file (mp3/wav)")
    parser.add_argument("--lyrics", required=True, help="Path to lyrics text file")
    parser.add_argument("--output", default="lyric_video.mp4", help="Output video path")
    parser.add_argument("--no-whisper", action="store_true", help="Skip Whisper, evenly space lyrics")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()

    if args.seed is not None:
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

    animated_lines = assign_effects(timed_lines)
    for al in animated_lines:
        log.info("  %-22s → %s (%s)", al.effect, al.line.text[:40], al.color)

    render_video(audio_path, animated_lines, output_path)


if __name__ == "__main__":
    main()
