"""Generate a synthetic sample JPG for CLI testing."""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import cv2


def make_sample(path: str = "data/samples/synthetic_test.jpg") -> None:
    width, height = 900, 500
    img = np.ones((height, width, 3), dtype=np.uint8) * 240

    lines_data = [
        [(50, 60, 200, 30), (270, 60, 180, 30), (480, 60, 220, 30), (730, 60, 120, 30)],
        [(40, 140, 160, 35), (230, 140, 250, 35), (520, 140, 170, 35)],
        [(60, 230, 190, 28), (290, 230, 210, 28), (540, 230, 150, 28), (720, 230, 130, 28)],
        [(30, 330, 300, 32), (370, 330, 200, 32), (610, 330, 250, 32)],
        [(70, 420, 170, 30), (280, 420, 230, 30), (550, 420, 190, 30), (770, 420, 90, 30)],
    ]

    for line in lines_data:
        for (x, y, w, h) in line:
            # Draw word-like blob with slight randomness
            cv2.rectangle(img, (x, y), (x + w, y + h), (15, 15, 15), -1)
            # Add a "headline" bar across the top (like shirorekha)
            cv2.line(img, (x, y + 3), (x + w, y + 3), (10, 10, 10), 2)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img)
    print(f"Saved synthetic sample to {path}")


if __name__ == "__main__":
    make_sample()
