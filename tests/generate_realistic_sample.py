"""Generate a more realistic synthetic sample with shirorekha-like headlines."""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import cv2


def add_word_with_headline(img, x, y, w, h, headline_thickness=3):
    """Draw a word-like shape with a headline bar across the top."""
    cv2.line(img, (x, y), (x + w, y), (15, 15, 15), headline_thickness)

    n_strokes = max(2, w // 25)
    rng = np.random.RandomState(x * 100 + y)
    for _ in range(n_strokes):
        sx = x + rng.randint(5, max(6, w - 5))
        sw = rng.randint(8, max(9, min(25, w // 3)))
        sh = rng.randint(h // 2, h)
        cv2.rectangle(img, (sx, y + 2), (sx + sw, y + sh), (20, 20, 20), -1)
        if rng.rand() > 0.5:
            cv2.circle(img, (sx + sw // 2, y + sh - 3), 4, (20, 20, 20), -1)


def make_realistic_sample(path="data/samples/hindi_like.jpg"):
    width, height = 1000, 700
    img = np.ones((height, width, 3), dtype=np.uint8) * 235

    noise = np.random.RandomState(42).randint(0, 15, img.shape, dtype=np.uint8)
    img = cv2.subtract(img, noise)

    line_configs = [
        {"y": 60,  "h": 55, "words": [(40, 180), (250, 140), (420, 200), (660, 120), (810, 150)]},
        {"y": 160, "h": 50, "words": [(30, 200), (270, 170), (480, 220), (740, 180)]},
        {"y": 260, "h": 58, "words": [(50, 160), (250, 190), (480, 150), (670, 140), (850, 110)]},
        {"y": 370, "h": 52, "words": [(60, 230), (330, 180), (550, 200), (790, 160)]},
        {"y": 470, "h": 48, "words": [(40, 170), (250, 210), (500, 160), (700, 130), (870, 90)]},
        {"y": 570, "h": 55, "words": [(50, 190), (280, 220), (540, 180), (760, 190)]},
    ]

    for lc in line_configs:
        y = lc["y"]
        h = lc["h"]
        for (x, w) in lc["words"]:
            add_word_with_headline(img, x, y, w, h)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img)
    print(f"Saved realistic sample to {path}  ({len(line_configs)} lines, "
          f"{sum(len(l['words']) for l in line_configs)} words)")
    return path, len(line_configs), sum(len(l["words"]) for l in line_configs)


if __name__ == "__main__":
    make_realistic_sample()
