"""End-to-end smoke test using a synthetic image with horizontal text bands."""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import cv2

from segmentation.preprocess import preprocess
from segmentation.lines import detect_lines, crop_lines
from segmentation.headline import attenuate_headline
from segmentation.words import detect_words, crop_words
from segmentation.visualize import annotate_image


def make_synthetic_image(
    width: int = 800,
    height: int = 400,
    n_lines: int = 4,
    words_per_line: int = 3,
) -> np.ndarray:
    """Create a white image with black rectangular 'words' arranged in lines."""
    img = np.ones((height, width, 3), dtype=np.uint8) * 245

    line_height = height // (n_lines * 2)
    word_width = width // (words_per_line * 2 + 1)

    for line_idx in range(n_lines):
        y0 = int(height * (2 * line_idx + 1) / (2 * n_lines + 1))
        for word_idx in range(words_per_line):
            x0 = int(width * (2 * word_idx + 1) / (2 * words_per_line + 1))
            cv2.rectangle(
                img,
                (x0, y0),
                (x0 + word_width, y0 + line_height),
                (20, 20, 20),
                -1,
            )
    return img


def test_full_pipeline():
    img = make_synthetic_image()
    assert img.shape == (400, 800, 3)

    binary, gray, y_off, x_off = preprocess(img, do_deskew=False)
    assert binary.shape[0] > 0 and binary.shape[1] > 0
    print(f"  Binary shape: {binary.shape}, offsets: y={y_off} x={x_off}")

    line_bounds, cc_labels, line_ccs = detect_lines(binary)
    print(f"  Detected {len(line_bounds)} lines: {line_bounds}")
    assert len(line_bounds) >= 3, f"Expected >= 3 lines, got {len(line_bounds)}"

    line_crops = crop_lines(binary, line_bounds, cc_labels, line_ccs)
    assert len(line_crops) == len(line_bounds)

    all_word_bounds: list[list[tuple[int, int]]] = []
    total_words = 0
    for i, lc in enumerate(line_crops):
        wb = detect_words(lc, remove_headline=False)
        all_word_bounds.append(wb)
        total_words += len(wb)
        print(f"    Line {i}: {len(wb)} words")

    assert total_words >= 6, f"Expected >= 6 words total, got {total_words}"

    annotated = annotate_image(img, line_bounds, all_word_bounds, y_off, x_off)
    assert annotated.shape[:2] == img.shape[:2]
    print(f"  Annotated image shape: {annotated.shape}")

    print("ALL TESTS PASSED")


if __name__ == "__main__":
    test_full_pipeline()
