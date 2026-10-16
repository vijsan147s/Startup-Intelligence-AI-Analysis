"""Render the LinkedIn video for Startup Intelligence AI.

Frames are drawn with Pillow and piped as raw RGB into ffmpeg, so the only
extra dependency is ffmpeg itself.

    python make_video.py                     # both cuts
    python make_video.py --only vertical
    python make_video.py --draft             # 8 fps, for checking timing

The video is silent. Captions are burned in, and this script also writes
docs/video/linkedin_voiceover.srt plus a per-scene timing sheet so the
voiceover can be recorded against the finished cut.

Every figure on screen is read from models/*.json, data/processed/*, the SQLite
database and src/config.py by video_facts.py, so the render cannot drift out of
sync with the pipeline.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))

from video_facts import build_facts, build_scenes, corrections

OUT_DIR = Path(__file__).resolve().parent / "docs" / "video"
DURATION = 90.0

BG_TOP = (11, 16, 32)
BG_BOTTOM = (22, 28, 48)
INK = (241, 245, 249)
MUTED = (148, 163, 184)
DIM = (100, 116, 139)
ACCENT = (74, 222, 128)
WARN = (248, 113, 113)
GOLD = (250, 204, 21)
PANEL = (17, 24, 43)
EDGE = (38, 50, 74)

FONT_DIR = Path(r"C:\Windows\Fonts")
FACES = {
    "sans": FONT_DIR / "segoeui.ttf",
    "sans_b": FONT_DIR / "seguisb.ttf",
    "mono": FONT_DIR / "consola.ttf",
}


def font(kind, size):
    return ImageFont.truetype(str(FACES[kind]), int(size))


def clamp01(x):
    return max(0.0, min(1.0, x))


def seg(t, start, dur=0.45):
    """Eased 0..1 ramp for an animation that begins at `start`."""
    if dur <= 0:
        return 1.0
    return 1.0 - (1.0 - clamp01((t - start) / dur)) ** 3


def wrap(d, s, fnt, max_w):
    """Greedy word wrap. `d` is a Draw for its text-length measurements."""
    out = []
    for para in s.split("\n"):
        cur = ""
        for word in para.split():
            trial = f"{cur} {word}".strip()
            if d.textlength(trial, font=fnt) <= max_w or not cur:
                cur = trial
            else:
                out.append(cur)
                cur = word
        out.append(cur)
    return "\n".join(out)


class Frame:
    """One frame's buffer plus the vertical budget the scenes lay out against.

    The caption panel is sized once from the worst-case voiceover line count, so
    the content band does not shift between frames or scenes.
    """

    def __init__(self, w, h, cap_lines):
        self.w, self.h = w, h
        self.wide = w > h
        s = h / (1080.0 if self.wide else 1920.0)

        self.m = int(w * (0.05 if self.wide else 0.075))
        self.pad = int(w * 0.026)

        self.f_kicker = font("sans_b", 26 * s)
        self.f_title = font("sans_b", 52 * s if self.wide else 62 * s)
        self.f_hero = font("sans_b", 74 * s if self.wide else 96 * s)
        self.f_stat = font("sans_b", 44 * s if self.wide else 52 * s)
        self.f_body = font("sans", 27 * s if self.wide else 30 * s)
        self.f_small = font("sans", 23 * s if self.wide else 25 * s)
        self.f_mono = font("mono", 23 * s if self.wide else 27 * s)
        self.f_cap = font("sans_b", 30 * s if self.wide else 36 * s)

        cap_lh = int(self.f_cap.size * 1.34)
        cap_h = cap_lines * cap_lh + 2 * self.pad
        cap_bot = h - int(h * (0.075 if self.wide else 0.055))
        self.cap_top = cap_bot - cap_h
        self.cap_bot = cap_bot
        self.cap_lh = cap_lh
        self.cap_lines = cap_lines

        self.band_top = int(h * (0.26 if self.wide else 0.30))
        self.band_bot = self.cap_top - int(h * 0.028)
        self.band_h = self.band_bot - self.band_top

        self.meas = ImageDraw.Draw(Image.new("RGB", (8, 8)))
        self.mono_arrow_w = self.meas.textlength("->", font=self.f_mono) + self.pad
        self.base = self._gradient(w, h)
        self.img = self.base.copy()

    @staticmethod
    def _gradient(w, h):
        ramp = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
        top = np.array(BG_TOP, dtype=np.float32)
        bottom = np.array(BG_BOTTOM, dtype=np.float32)
        col = top + (bottom - top) * ramp
        return Image.fromarray(
            np.repeat(col[:, None, :], w, axis=1).astype(np.uint8), "RGB"
        )

    # -- drawing helpers ------------------------------------------------
    def panel(self, d, box, fill=PANEL, radius=18, outline=None, width=1):
        d.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)

    def wrap(self, s, fnt, max_w):
        return wrap(self.meas, s, fnt, max_w)

    def chrome(self, d, t, i, total, scene):
        """Kicker, title, scene counter, progress bar."""
        h, m = self.h, self.m
        ky = int(h * (0.075 if self.wide else 0.085))
        d.text((m, ky), scene["kicker"], font=self.f_kicker, fill=ACCENT)
        rx = m + self.meas.textlength(scene["kicker"], font=self.f_kicker) + 26
        d.line([(rx, ky + self.f_kicker.size // 2), (self.w - m, ky + self.f_kicker.size // 2)],
               fill=EDGE, width=2)

        ty = ky + int(self.f_kicker.size * 2.1)
        for line in scene["title"].split("\n"):
            d.text((m, ty), line, font=self.f_title, fill=INK)
            ty += int(self.f_title.size * 1.18)

        d.text((self.w - m, ky - int(self.f_kicker.size * 1.3)),
               f"{i + 1}/{total}", font=self.f_kicker, fill=DIM, anchor="ra")

        bar_y = int(h * 0.955)
        bar_h = max(4, int(h * 0.004))
        d.rounded_rectangle([m, bar_y, self.w - m, bar_y + bar_h],
                            radius=bar_h // 2, fill=(30, 41, 59))
        end = m + int((self.w - 2 * m) * clamp01(t / DURATION))
        d.rounded_rectangle([m, bar_y, end, bar_y + bar_h], radius=bar_h // 2, fill=ACCENT)

    def captions(self, d, t, lines):
        active = [txt for a, b, txt in lines if a <= t < b]
        if not active:
            return
        # Wrap to the panel's inner width; the reserved box was sized from the
        # worst-case wrapped voiceover line, so this can never overflow it.
        body = self.wrap(
            "\n".join(active), self.f_cap, self.w - 2 * self.m - 2 * self.pad
        )
        if body.count("\n") + 1 > self.cap_lines:
            raise ValueError(
                f"caption overflows the reserved {self.cap_lines} lines: {body!r}"
            )
        self.panel(d, [self.m, self.cap_top, self.w - self.m, self.cap_bot],
                   fill=(13, 19, 35), radius=20)
        d.line([(self.m, self.cap_top), (self.m, self.cap_bot)], fill=ACCENT, width=5)
        d.multiline_text(
            (self.w // 2, (self.cap_top + self.cap_bot) // 2), body,
            font=self.f_cap, fill=INK, anchor="mm",
            spacing=int(self.f_cap.size * 0.30), align="center",
        )


def _rows(n, top, height, gap=0.0, row_h_override=None):
    """Split `height` into n rows with `gap` between them, returning tops and row height."""
    if row_h_override is not None:
        row = row_h_override
        return [top + i * row for i in range(n)], row
    row = (height - gap * (n - 1)) / n
    return [top + i * (row + gap) for i in range(n)], row


def scene_one(f, d, sc, t, lt):
    ys, row = _rows(len(sc["cards"]), f.band_top, f.band_h, 0)
    for i, (value, label) in enumerate(sc["cards"]):
        if lt < sc["stagger"][i]:
            break
        a = seg(lt, sc["stagger"][i], 0.5)
        y = ys[i] + row / 2 - f.f_hero.size * 0.62
        d.text((f.m, y), value, font=f.f_hero, fill=(*INK, int(255 * a)))
        vx = f.m + f.meas.textlength(value, font=f.f_hero) + f.pad
        d.text((vx, y + f.f_hero.size * 0.44), label, font=f.f_body,
               fill=(*MUTED, int(255 * a)))


def scene_two(f, d, sc, t, lt):
    n = len(sc["rows"])
    usable = f.band_h * 0.78
    ys, row = _rows(n, f.band_top, usable, f.band_h * 0.035)
    for i, (count, label, is_bad) in enumerate(sc["rows"]):
        a = seg(lt, 0.15 + i * 0.5, 0.45)
        if a <= 0:
            continue
        dy = int((1 - a) * 26)
        f.panel(d, [f.m, ys[i] + dy, f.w - f.m, ys[i] + dy + row], radius=16)
        mid = ys[i] + row / 2 + dy
        d.text((f.m + f.pad * 2, mid), count, font=f.f_stat,
               fill=(*(WARN if is_bad else MUTED), int(255 * a)), anchor="lm")
        d.text((f.w - f.m - f.pad * 2, mid), label, font=f.f_body,
               fill=(*MUTED, int(255 * a)), anchor="rm")
    a = seg(lt, 2.0, 0.5)
    if a > 0:
        d.text((f.m, f.band_top + usable + f.band_h * 0.07), sc["footnote"],
               font=f.f_body, fill=(*GOLD, int(255 * a)))


def scene_three(f, d, sc, t, lt):
    ys, row = _rows(len(sc["transforms"]), f.band_top, f.band_h * 0.50, f.band_h * 0.02)
    for i, (src, dst) in enumerate(sc["transforms"]):
        a = seg(lt, 0.1 + i * 0.55, 0.45)
        if a <= 0:
            continue
        y = ys[i] + row * 0.28
        d.text((f.m, y), src, font=f.f_mono, fill=(*DIM, int(255 * a)))
        ax = f.m + f.meas.textlength(src, font=f.f_mono) + f.pad
        d.text((ax, y), "->", font=f.f_mono, fill=(*GOLD, int(255 * a)))
        d.text((ax + f.mono_arrow_w, y), dst, font=f.f_mono, fill=(*ACCENT, int(255 * a)))

    a = seg(lt, 2.1, 0.5)
    if a > 0:
        ys2, row2 = _rows(len(sc["counts"]), f.band_top + f.band_h * 0.58,
                           f.band_h * 0.40, f.band_h * 0.02)
        for i, c in enumerate(sc["counts"]):
            d.text((f.m, ys2[i] + row2 * 0.2), c, font=f.f_body,
                   fill=(*MUTED, int(255 * a)))


def scene_four(f, d, sc, t, lt):
    ys, row = _rows(len(sc["stack"]), f.band_top, f.band_h * 0.46, f.band_h * 0.012)
    for i, line in enumerate(sc["stack"]):
        a = seg(lt, 0.1 + i * 0.32, 0.4)
        if a <= 0:
            continue
        d.text((f.m + (f.pad if i else 0), ys[i] + row * 0.2), line,
               font=f.f_mono, fill=(*(INK if i == 0 else MUTED), int(255 * a)))

    a = seg(lt, 1.9, 0.5)
    if a > 0:
        top = f.band_top + f.band_h * 0.52
        h = f.band_h * 0.44
        if f.wide:
            n = len(sc["stats"])
            bw = (f.w - 2 * f.m - f.pad * (n - 1)) / n
            for i, (num, label) in enumerate(sc["stats"]):
                x = f.m + i * (bw + f.pad)
                f.panel(d, [x, top, x + bw, top + h], radius=16)
                d.text((x + bw / 2, top + h / 2), num, font=f.f_stat,
                       fill=(*ACCENT, int(255 * a)), anchor="mm")
                d.text((x + bw / 2, top + h * 0.76), label, font=f.f_small,
                       fill=(*MUTED, int(255 * a)), anchor="ma")
        else:
            ys2, row2 = _rows(n_ := len(sc["stats"]), top, h, f.pad)
            for i, (num, label) in enumerate(sc["stats"]):
                f.panel(d, [f.m, ys2[i], f.w - f.m, ys2[i] + row2], radius=16)
                d.text((f.m + f.pad * 2, ys2[i] + row2 * 0.5), num, font=f.f_stat,
                       fill=(*ACCENT, int(255 * a)), anchor="lm")
                d.text((f.w - f.m - f.pad * 2, ys2[i] + row2 * 0.5), label, font=f.f_body,
                       fill=(*MUTED, int(255 * a)), anchor="rm")


def scene_five(f, d, sc, t, lt):
    """CV RMSLE bars, then the held-out test metrics."""
    vmax = max(v for _, v, _ in sc["bars"])
    label_h = f.f_small.size * 1.35
    gap = f.band_h * 0.028
    bar_h = f.band_h * 0.038
    track = f.w - 2 * f.m - f.pad * 2

    for i, (label, value, selected) in enumerate(sc["bars"]):
        a = seg(lt, 0.15 + i * 0.4, 0.5)
        if a <= 0:
            continue
        y = f.band_top + i * (label_h + bar_h + gap)
        col = ACCENT if selected else DIM
        d.text((f.m, y), label, font=f.f_small, fill=(*MUTED, int(255 * a)))
        d.text((f.w - f.m, y), f"{value:.4f}", font=f.f_small,
               fill=(*col, int(255 * a)), anchor="ra")
        by = y + label_h
        f.panel(d, [f.m, by, f.m + track, by + bar_h], fill=(26, 35, 56), radius=bar_h // 2)
        w = int(track * (value / vmax) * a)
        if w > bar_h:
            d.rounded_rectangle([f.m, by, f.m + w, by + bar_h],
                                radius=bar_h // 2, fill=col)

    y = f.band_top + 3 * (label_h + bar_h + gap) + f.band_h * 0.01
    a = seg(lt, 1.5, 0.5)
    if a > 0:
        d.text((f.m, y), sc["bar_note"], font=f.f_small, fill=(*DIM, int(255 * a)))

        my = y + f.band_h * 0.07
        mh = f.band_h * 0.26
        n = len(sc["metrics"])
        bw = (f.w - 2 * f.m - f.pad * (n - 1)) / n
        for i, (label, value) in enumerate(sc["metrics"]):
            b = seg(lt, 1.6 + i * 0.12, 0.5)
            x = f.m + i * (bw + f.pad)
            f.panel(d, [x, my, x + bw, my + mh], radius=16)
            d.text((x + bw / 2, my + mh * 0.24), label, font=f.f_small,
                   fill=(*MUTED, int(255 * b)), anchor="ma")
            d.text((x + bw / 2, my + mh * 0.52), value, font=f.f_stat,
                   fill=(*(WARN if label == "R2" else INK), int(255 * b)), anchor="ma")

        d.text((f.m, my + mh + f.band_h * 0.05), sc["metrics_note"],
               font=f.f_body, fill=(*GOLD, int(255 * a)))


def scene_six(f, d, sc, t, lt):
    n = len(sc["choice"])
    ph = f.band_h * 0.28
    ys, _ = _rows(n, f.band_top, ph * n + f.band_h * 0.05, f.band_h * 0.04)
    for i, (name, value, tag, shipped) in enumerate(sc["choice"]):
        a = seg(lt, 0.2 + i * 0.7, 0.5)
        if a <= 0:
            continue
        # The shipped model is the one highlighted; the higher-scoring model is
        # deliberately the muted one, since it was rejected.
        col = ACCENT if shipped else DIM
        f.panel(d, [f.m, ys[i], f.w - f.m, ys[i] + ph], radius=18,
                outline=col if shipped else None, width=3)
        d.text((f.m + f.pad * 2, ys[i] + ph * 0.26), name, font=f.f_body,
               fill=(*INK, int(255 * a)))
        d.text((f.m + f.pad * 2, ys[i] + ph * 0.60), tag, font=f.f_small,
               fill=(*col, int(255 * a)))
        d.text((f.w - f.m - f.pad * 2, ys[i] + ph / 2), value, font=f.f_stat,
               fill=(*col, int(255 * a)), anchor="rm")

    a = seg(lt, 1.7, 0.5)
    if a > 0:
        yy = ys[-1] + ph + f.band_h * 0.05
        f.panel(d, [f.m, yy, f.w - f.m, yy + f.band_h * 0.11],
                fill=PANEL, radius=14, outline=GOLD, width=2)
        d.text((f.w / 2, yy + f.band_h * 0.055), sc["cv_note"], font=f.f_mono,
               fill=(*GOLD, int(255 * a)), anchor="mm")
        body = f.wrap(sc["choice_note"], f.f_body, f.w - 2 * f.m - 4 * f.pad)
        d.multiline_text((f.w / 2, yy + f.band_h * 0.17), body, font=f.f_body,
                         fill=(*MUTED, int(255 * a)), anchor="ma", align="center")


def scene_seven(f, d, sc, t, lt):
    side = f.wide
    n = len(sc["panels"])
    gap = f.band_h * 0.05
    if side:
        bw = (f.w - 2 * f.m - gap) / 2
        ph = f.band_h
        pos = [(f.m, f.band_top), (f.m + bw + gap, f.band_top)]
    else:
        bw = f.w - 2 * f.m
        ph = (f.band_h - gap) / n
        pos = [(f.m, f.band_top + i * (ph + gap)) for i in range(n)]

    for i, panel in enumerate(sc["panels"]):
        a = seg(lt, 0.2 + i * 0.6, 0.5)
        if a <= 0:
            continue
        x, y = pos[i]
        f.panel(d, [x, y, x + bw, y + ph], radius=18, outline=EDGE, width=2)
        d.text((x + f.pad * 2, y + f.pad), panel["label"], font=f.f_small,
               fill=(*ACCENT, int(255 * a)))

        # Reserve the label strip and the note strip, then split what's left
        # evenly between the data lines so nothing can overlap either.
        top = y + f.pad + f.f_small.size * 1.9
        bottom = y + ph - f.pad * 2 - f.f_small.size * 1.9
        lines = panel["lines"]
        ys, row = _rows(len(lines), top, bottom - top, gap=f.band_h * 0.03)
        for j, line in enumerate(lines):
            body = f.wrap(line, f.f_mono, bw - 4 * f.pad)
            d.multiline_text((x + f.pad * 2, ys[j]), body, font=f.f_mono,
                             fill=(*INK, int(255 * a)),
                             spacing=int(f.f_mono.size * 1.5))
        d.multiline_text((x + f.pad * 2, y + ph - f.pad * 2),
                         f.wrap(panel["note"], f.f_small, bw - 4 * f.pad),
                         font=f.f_small, fill=(*DIM, int(255 * a)))


def scene_eight(f, d, sc, t, lt):
    """Permutation importance as a signed, zero-centred bar chart."""
    rows = sc["importance"]
    top = abs(max(v for _, v in rows)) or 1.0
    usable = f.band_h * 0.66
    ys, row = _rows(len(rows), f.band_top, usable, f.band_h * 0.012)

    name_w = f.w * (0.30 if not f.wide else 0.26)
    val_w = f.meas.textlength("1,138,043", font=f.f_small) + f.pad
    left = f.m + name_w
    right = f.w - f.m - val_w
    # Centre the zero axis and scale so the longest bar fits on either side.
    # The value column is reserved separately, so a bar can never run into its
    # own label.
    mid = (left + right) / 2
    half = (right - left) / 2 * 0.92

    d.line([(mid, ys[0] - f.band_h * 0.012), (mid, ys[-1] + row * 0.8)],
           fill=EDGE, width=2)

    for i, (name, value) in enumerate(rows):
        b = seg(lt, 0.2 + i * 0.12, 0.4)
        if b <= 0:
            continue
        y = ys[i] + row * 0.22
        bh = row * 0.52
        col = ACCENT if value > 0 else (MUTED if value < 0 else GOLD)
        d.text((f.m, y), name, font=f.f_small, fill=(*DIM, int(255 * b)))
        d.text((f.w - f.m, y), f"{value:,.0f}", font=f.f_small,
               fill=(*col, int(255 * b)), anchor="ra")
        w = int(abs(value) / top * half * b)
        if w > 1:
            box = [mid, y, mid + w, y + bh] if value >= 0 else [mid - w, y, mid, y + bh]
            d.rounded_rectangle(box, radius=4, fill=col)

    a = seg(lt, 1.3, 0.5)
    if a > 0:
        d.text((f.m, ys[-1] + row * 0.9), sc["imp_note"], font=f.f_small,
               fill=(*MUTED, int(255 * a)))

    b = seg(lt, 1.7, 0.5)
    if b > 0:
        yy = f.band_top + f.band_h * 0.74
        f.panel(d, [f.m, yy, f.w - f.m, f.band_bot], fill=PANEL, radius=14,
                outline=GOLD, width=2)
        d.multiline_text(
            (f.w / 2, (yy + f.band_bot) / 2),
            f.wrap(sc["verdict"], f.f_body, f.w - 2 * f.m - 4 * f.pad),
            font=f.f_body, fill=(*GOLD, int(255 * b)), anchor="mm", align="center",
        )


def scene_nine(f, d, sc, t, lt):
    fade = clamp01((lt - 14.5) / 0.9)
    if fade <= 0:
        a = seg(lt, 0.2, 0.5)
        cards = sc["cards"]
        gap = f.band_h * 0.03
        h = min(f.band_h * 0.86, (f.band_h * 0.62) / len(cards)) if not f.wide \
            else f.band_h * 0.30
        if f.wide:
            n = len(cards)
            bw = (f.w - 2 * f.m - gap * (n - 1)) / n
            pos = [(f.m + i * (bw + gap), f.band_top) for i in range(n)]
            bh = h
        else:
            bw = f.w - 2 * f.m
            bh = (f.band_h * 0.72) / len(cards)
            pos = [(f.m, f.band_top + i * (bh + gap)) for i in range(len(cards))]

        for i, (value, label) in enumerate(cards):
            b = seg(lt, 0.3 + i * 0.4, 0.45)
            if b <= 0:
                continue
            x, y = pos[i]
            f.panel(d, [x, y, x + bw, y + bh], radius=16)
            if f.wide:
                # Narrow cards, so stack the value over its label.
                d.text((x + f.pad * 2, y + bh * 0.26), value, font=f.f_stat,
                       fill=(*ACCENT, int(255 * b)), anchor="lm")
                d.text((x + f.pad * 2, y + bh * 0.68), label, font=f.f_small,
                       fill=(*MUTED, int(255 * b)), anchor="lm")
            else:
                d.text((x + f.pad * 2, y + bh / 2), value, font=f.f_stat,
                       fill=(*ACCENT, int(255 * b)), anchor="lm")
                d.text((f.w - f.m - f.pad * 2, y + bh / 2), label, font=f.f_body,
                       fill=(*MUTED, int(255 * b)), anchor="rm")
        a2 = seg(lt, 2.1, 0.5)
        if a2 > 0:
            d.text((f.w / 2, f.band_bot - f.band_h * 0.05), sc["runtime"],
                   font=f.f_mono, fill=(*DIM, int(255 * a2)), anchor="mm")
        return

    card = "\n".join(sc["final_card"])
    card_h = card.count("\n") + 1
    card_h *= f.f_hero.size * 1.34
    card_cy = f.band_top + card_h * 0.75
    d.multiline_text((f.w / 2, card_cy), card, font=f.f_hero,
                     fill=(*INK, int(255 * fade)), anchor="mm", align="center",
                     spacing=int(f.f_hero.size * 0.34))
    b = seg(lt, 16.0, 0.8)
    if b > 0:
        # Anchor the closing line below the measured card block so the two
        # cannot collide in either orientation.
        close_y = card_cy + card_h * 0.62 + f.f_body.size * 1.5
        close_y = min(close_y, f.band_bot - f.f_body.size)
        d.text((f.w / 2, close_y), sc["closing"], font=f.f_body,
               fill=(*ACCENT, int(255 * b)), anchor="mm")


RENDERERS = [scene_one, scene_two, scene_three, scene_four, scene_five,
             scene_six, scene_seven, scene_eight, scene_nine]


class Renderer:
    def __init__(self, scenes, w, h):
        self.scenes = scenes
        probe = Frame(w, h, 3)
        cap_lines = max(
            probe.wrap(txt, probe.f_cap, w - 2 * probe.m - 2 * probe.pad).count("\n") + 1
            for sc in scenes for _, _, txt in sc["vo"]
        )
        self.cap_lines = cap_lines
        self.frame = Frame(w, h, cap_lines)
        self.f = self.frame
        self.w, self.h = w, h

    def _at(self, t):
        idx = next((i for i, s in enumerate(self.scenes) if s["start"] <= t < s["end"]),
                   len(self.scenes) - 1)
        return idx, self.scenes[idx], t - self.scenes[idx]["start"]

    def render(self, t):
        f = self.frame
        f.img = f.base.copy()
        layer = Image.new("RGBA", (f.w, f.h), (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)

        idx, scene, lt = self._at(t)
        f.chrome(d, t, idx, len(self.scenes), scene)
        RENDERERS[idx](f, d, scene, t, lt)
        f.captions(d, t, scene["vo"])

        f.img = Image.alpha_composite(f.img.convert("RGBA"), layer).convert("RGB")
        return f.img


def encode(renderer, out_path, fps, total_frames):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{renderer.w}x{renderer.h}", "-r", str(fps), "-i", "-",
        "-c:v", "libx264", "-preset", "slow", "-crf", "19",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(out_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(total_frames):
        proc.stdin.write(renderer.render(i / fps).tobytes())
        if i % 300 == 0:
            print(f"    frame {i}/{total_frames}", flush=True)
    proc.stdin.close()
    if proc.wait() != 0:
        raise SystemExit(f"ffmpeg failed for {out_path}")


def _ts(v):
    h, rem = divmod(v, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:06.3f}".replace(".", ",")


def write_srt(scenes, path):
    cues = []
    for scene in scenes:
        for start, end, text in scene["vo"]:
            cues.append(f"{len(cues) + 1}\n{_ts(start)} --> {_ts(end)}\n{text}\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(cues), encoding="utf-8")


def write_timing_sheet(scenes, facts, path):
    fixes = corrections(facts)
    words = sum(len(txt.split()) for s in scenes for _, _, txt in s["vo"])
    # Proper-cased scene names; .title() would mangle acronyms like BI.
    labels = {
        "THE DATASET": "The Dataset",
        "THE PROBLEM": "The Problem",
        "CLEANING": "Cleaning",
        "STORAGE AND BI": "Storage and BI",
        "REGRESSION": "Regression",
        "THE ANTI-LEAKAGE CALL": "The Anti-Leakage Call",
        "CLASSIFICATION AND CLUSTERING": "Classification and Clustering",
        "EXPLAINABILITY": "Explainability",
        "THE APP": "The App",
    }
    lines = [
        "# Voiceover timing sheet", "",
        "The render is silent. Record against these timings; the burned-in",
        "captions follow the same beats. Figures are read from the live artifacts",
        "by `video_facts.py`, so this sheet regenerates rather than drifting.", "",
        "| Scene | In | Out | Dur | Voiceover |", "|---|---|---|---|---|",
    ]
    for i, s in enumerate(scenes, 1):
        vo = " ".join(txt for _, _, txt in s["vo"]).replace("\n", " ")
        lines.append(f"| {i}. {labels.get(s['kicker'], s['kicker'].title())} | "
                     f"{s['start']:.1f}s | {s['end']:.1f}s | "
                     f"{s['end'] - s['start']:.1f}s | {vo} |")
    rf, lr, rf_c, cl = facts["rf"], facts["lr"], facts["rf_c"], facts["clusters"]
    lines += [
        "", f"Total **{DURATION:.0f}s**, {words} spoken words "
        f"(~{words / (DURATION / 60):.0f} wpm).", "",
        "## Figures shown on screen", "",
        f"- Funding events: {facts['total_rows']:,}",
        f"- Total recorded funding: {facts['total_funding'] / 1e9:.2f}B USD",
        f"- Amounts missing: {facts['missing_amounts']:,}   invalid dates: {facts['invalid_dates']}",
        f"- Cities / industries / types normalized: "
        f"{facts['n_cities']} / {facts['n_industries']} / {facts['n_types']}",
        f"- Tests collected by pytest: {facts['n_tests']}",
        f"- RF CV RMSLE {rf['cv_rmsle']:.4f} -> test RMSLE {rf['rmsle']:.4f}, "
        f"MAE {rf['mae'] / 1e6:.1f}M, R2 {rf['r2']:.4f}",
        f"- Classifier: LogReg shipped (CV F1 {lr['cv_f1']:.4f}, test F1 "
        f"{lr['f1_weighted']:.4f}); RF test F1 {rf_c['f1_weighted']:.4f} not shipped",
        f"- Clustering: k={cl['optimal_k']}, silhouette {cl['silhouette']:.4f}, "
        f"{cl['events_clustered']:,} events", "",
        "## Corrections against the original package", "",
    ]
    lines += [f"- {c}" for c in fixes]
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Render the LinkedIn video.")
    ap.add_argument("--only", choices=["vertical", "landscape"], action="append")
    ap.add_argument("--draft", action="store_true", help="8 fps, for checking timing")
    ap.add_argument("--meta-only", action="store_true",
                    help="regenerate the SRT and timing sheet without rendering")
    args = ap.parse_args()

    facts = build_facts()
    scenes = build_scenes(facts)

    print("Figures read live from the artifacts.")
    print("Corrections against docs/linkedin_video_package.md:")
    for c in corrections(facts):
        print(f"  - {c}")

    write_srt(scenes, OUT_DIR / "linkedin_voiceover.srt")
    write_timing_sheet(scenes, facts, OUT_DIR / "VOICEOVER_TIMING.md")
    print(f"Wrote {OUT_DIR / 'linkedin_voiceover.srt'}")
    print(f"Wrote {OUT_DIR / 'VOICEOVER_TIMING.md'}")

    if args.meta_only:
        return

    targets = {"vertical": (1080, 1920, "linkedin_vertical_1080x1920.mp4"),
               "landscape": (1920, 1080, "linkedin_landscape_1920x1080.mp4")}
    fps = 8 if args.draft else 30
    total_frames = int(DURATION * fps)

    for name in args.only or ["vertical", "landscape"]:
        w, h, filename = targets[name]
        print(f"\nRendering {name} {w}x{h} at {fps} fps ({total_frames} frames)")
        renderer = Renderer(scenes, w, h)
        out = OUT_DIR / filename
        encode(renderer, out, fps, total_frames)
        print(f"  -> {out}")


if __name__ == "__main__":
    main()
