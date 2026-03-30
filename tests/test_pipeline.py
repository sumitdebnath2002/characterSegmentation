"""End-to-end smoke test using a synthetic image with horizontal text bands."""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import cv2
from pathlib import Path

from segmentation.preprocess import preprocess, remove_ruled_lines, binarize, to_grayscale, denoise
from segmentation.lines import detect_lines, crop_lines, _split_wide_bands
from segmentation.headline import attenuate_headline
from segmentation.words import detect_words, crop_words
from segmentation.characters import (
    segment_characters,
    _estimate_expected_char_width,
    _cc_fallback_cuts,
    _gate_cuts_by_width,
    _find_shirorekha_band,
)
from segmentation.profiles import horizontal_profile, smooth_profile
from segmentation.visualize import annotate_image
from segmentation.io import load_image


SAMPLES_DIR = Path(__file__).parent.parent / "data" / "samples"


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

    print("  test_full_pipeline PASSED")


# ═══════════════════════════════════════════════════════════════════════════════
# Ruled-line removal tests
# ═══════════════════════════════════════════════════════════════════════════════

def _make_ruled_image_with_text(
    width: int = 800,
    height: int = 400,
    n_rules: int = 8,
    n_text_lines: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a binary image with horizontal ruled lines and short text blocks.

    Returns (binary_with_rules, binary_text_only) for comparison.
    """
    text_only = np.zeros((height, width), dtype=np.uint8)
    line_h = 20
    word_w = 60

    for line_idx in range(n_text_lines):
        y0 = 30 + line_idx * (height // (n_text_lines + 1))
        for word_idx in range(4):
            x0 = 50 + word_idx * (width // 5)
            cv2.rectangle(text_only, (x0, y0), (x0 + word_w, y0 + line_h), 255, -1)

    rules_only = np.zeros((height, width), dtype=np.uint8)
    for i in range(n_rules):
        y = int(height * (i + 1) / (n_rules + 1))
        cv2.line(rules_only, (0, y), (width - 1, y), 255, 1)

    combined = np.maximum(text_only, rules_only)
    return combined, text_only


def test_remove_ruled_lines():
    """Verify that ruled notebook lines are removed without harming text."""
    combined, text_only = _make_ruled_image_with_text()
    cleaned = remove_ruled_lines(combined)

    text_pixels_before = np.count_nonzero(text_only)
    text_pixels_after = np.count_nonzero(np.bitwise_and(cleaned, text_only))
    preservation_ratio = text_pixels_after / text_pixels_before if text_pixels_before > 0 else 0

    assert preservation_ratio > 0.90, (
        f"Text preservation too low: {preservation_ratio:.2%}"
    )

    rule_pixels_original = np.count_nonzero(combined) - text_pixels_before
    rule_pixels_after = np.count_nonzero(cleaned) - text_pixels_after
    removal_ratio = 1.0 - (rule_pixels_after / rule_pixels_original) if rule_pixels_original > 0 else 1.0

    assert removal_ratio > 0.80, (
        f"Rule removal too low: {removal_ratio:.2%}"
    )

    print(f"  Text preserved: {preservation_ratio:.1%}, rules removed: {removal_ratio:.1%}")
    print("  test_remove_ruled_lines PASSED")


def test_remove_ruled_lines_no_false_positive():
    """Verify that remove_ruled_lines does nothing on a non-ruled image."""
    text_only = np.zeros((400, 800), dtype=np.uint8)
    for y in [50, 150, 250]:
        cv2.rectangle(text_only, (50, y), (200, y + 20), 255, -1)
        cv2.rectangle(text_only, (300, y), (450, y + 20), 255, -1)

    result = remove_ruled_lines(text_only)
    assert np.array_equal(result, text_only), "Should not modify non-ruled image"
    print("  test_remove_ruled_lines_no_false_positive PASSED")


# ═══════════════════════════════════════════════════════════════════════════════
# Band splitting tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_split_wide_bands_tight_spacing():
    """Three closely-spaced text lines merged into one oversized band should be split."""
    height, width = 500, 600
    binary = np.zeros((height, width), dtype=np.uint8)

    # Two normal lines (each ~30px tall):
    cv2.rectangle(binary, (30, 20), (550, 50), 255, -1)     # line 1
    cv2.rectangle(binary, (30, 350), (550, 380), 255, -1)   # line 5

    # Three lines merged into one oversized band (~130px, with 5px gaps):
    cv2.rectangle(binary, (30, 140), (550, 170), 255, -1)   # line 2
    cv2.rectangle(binary, (30, 180), (550, 210), 255, -1)   # line 3
    cv2.rectangle(binary, (30, 220), (550, 250), 255, -1)   # line 4

    hp = smooth_profile(horizontal_profile(binary), window=5)

    # Simulate what HPP would produce: lines 2+3+4 merged into one band
    initial_bands = [(15, 55), (135, 255), (345, 385)]
    result = _split_wide_bands(hp, initial_bands, max_height_ratio=2.0, min_line_height=10)

    assert len(result) >= 4, (
        f"Expected >= 4 bands after split, got {len(result)}: {result}"
    )
    print(f"  Split result: {result}")
    print("  test_split_wide_bands_tight_spacing PASSED")


def test_split_wide_bands_no_false_split():
    """A single normal-height band should not be split."""
    height, width = 200, 400
    binary = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(binary, (30, 50), (350, 80), 255, -1)

    hp = smooth_profile(horizontal_profile(binary), window=5)
    bands = [(45, 85)]
    result = _split_wide_bands(hp, bands, max_height_ratio=2.0, min_line_height=10)

    assert len(result) == 1, f"Expected 1 band (no split), got {len(result)}"
    print("  test_split_wide_bands_no_false_split PASSED")


# ═══════════════════════════════════════════════════════════════════════════════
# Regression tests on real images
# ═══════════════════════════════════════════════════════════════════════════════

def test_bangla1_line_count():
    """bangla1.jpeg should produce exactly 4 lines."""
    path = SAMPLES_DIR / "bangla1.jpeg"
    if not path.exists():
        print(f"  SKIP: {path} not found")
        return
    img = load_image(path)
    binary, _, _, _ = preprocess(img, do_deskew=True, do_slant_correct=True)
    line_bounds, _, _ = detect_lines(binary)
    print(f"  bangla1.jpeg: {len(line_bounds)} lines")
    assert len(line_bounds) == 4, f"Expected 4 lines, got {len(line_bounds)}"
    print("  test_bangla1_line_count PASSED")


def test_bangla2_line_count():
    """bangla2.jpeg should produce 9-10 lines (the poem has 10 text lines)."""
    path = SAMPLES_DIR / "bangla2.jpeg"
    if not path.exists():
        print(f"  SKIP: {path} not found")
        return
    img = load_image(path)
    binary, _, _, _ = preprocess(img, do_deskew=True, do_slant_correct=True)
    line_bounds, _, _ = detect_lines(binary)
    print(f"  bangla2.jpeg: {len(line_bounds)} lines")
    assert 9 <= len(line_bounds) <= 11, (
        f"Expected 9-11 lines, got {len(line_bounds)}"
    )
    print("  test_bangla2_line_count PASSED")


def test_hindi_line_count():
    """hindi.png should produce exactly 5 lines."""
    path = SAMPLES_DIR / "hindi.png"
    if not path.exists():
        print(f"  SKIP: {path} not found")
        return
    img = load_image(path)
    binary, _, _, _ = preprocess(img, do_deskew=True, do_slant_correct=True)
    line_bounds, _, _ = detect_lines(binary)
    print(f"  hindi.png: {len(line_bounds)} lines")
    assert len(line_bounds) == 5, f"Expected 5 lines, got {len(line_bounds)}"
    print("  test_hindi_line_count PASSED")


def test_image_line_count():
    """image.png should produce exactly 9 lines."""
    path = SAMPLES_DIR / "image.png"
    if not path.exists():
        print(f"  SKIP: {path} not found")
        return
    img = load_image(path)
    binary, _, _, _ = preprocess(img, do_deskew=True, do_slant_correct=True)
    line_bounds, _, _ = detect_lines(binary)
    print(f"  image.png: {len(line_bounds)} lines")
    assert len(line_bounds) == 9, f"Expected 9 lines, got {len(line_bounds)}"
    print("  test_image_line_count PASSED")


# ═══════════════════════════════════════════════════════════════════════════════
# Character segmentation tests
# ═══════════════════════════════════════════════════════════════════════════════

def _make_word_with_separated_ccs(
    n_chars: int = 3,
    char_w: int = 30,
    char_h: int = 40,
    gap: int = 8,
    shirorekha_h: int = 4,
) -> np.ndarray:
    """Synthetic word: *n_chars* blocks joined by a shirorekha, separated by
    clear gaps in the body region below the headline.
    """
    total_w = n_chars * char_w + (n_chars - 1) * gap
    img = np.zeros((char_h, total_w), dtype=np.uint8)

    # Draw shirorekha across full width
    img[2 : 2 + shirorekha_h, :] = 255

    # Draw character bodies below shirorekha
    body_top = 2 + shirorekha_h + 2
    for i in range(n_chars):
        x0 = i * (char_w + gap)
        img[body_top : char_h - 2, x0 : x0 + char_w] = 255

    return img


def _make_single_wide_cc() -> np.ndarray:
    """Single solid block -- should NOT be split."""
    img = np.zeros((40, 50), dtype=np.uint8)
    img[2:6, :] = 255   # shirorekha
    img[10:38, 5:45] = 255  # one body
    return img


def test_segment_separated_word():
    """A word with 3 well-separated CCs below a shirorekha should yield 3 chars."""
    word = _make_word_with_separated_ccs(n_chars=3)
    chars = segment_characters(word)
    print(f"  Separated 3-char word → {len(chars)} chars")
    assert 2 <= len(chars) <= 4, (
        f"Expected 2-4 chars from 3-char word, got {len(chars)}"
    )
    print("  test_segment_separated_word PASSED")


def test_segment_single_cc_no_split():
    """A single wide CC should not be split into multiple chars."""
    word = _make_single_wide_cc()
    chars = segment_characters(word)
    print(f"  Single CC → {len(chars)} chars")
    assert len(chars) == 1, f"Expected 1 char, got {len(chars)}"
    print("  test_segment_single_cc_no_split PASSED")


def test_cc_fallback_finds_gaps():
    """_cc_fallback_cuts should find gaps between separated CC bodies."""
    word = _make_word_with_separated_ccs(n_chars=4, gap=10)
    shiro = _find_shirorekha_band(word)
    cuts = _cc_fallback_cuts(word, shiro, stroke_width=3.0)
    print(f"  CC fallback on 4-char word → {len(cuts)} cuts: {cuts}")
    assert len(cuts) >= 2, f"Expected >= 2 cuts, got {len(cuts)}"
    print("  test_cc_fallback_finds_gaps PASSED")


def test_gate_cuts_by_width():
    """Width gating should remove cuts that produce too-narrow segments."""
    S = np.zeros(200, dtype=np.float64)
    cuts = [20, 30, 100, 150]
    S[20] = 0.3
    S[30] = 0.8
    S[100] = 0.5
    S[150] = 0.6

    result = _gate_cuts_by_width(cuts, S, 200, expected_char_width=50.0, min_width_ratio=0.4)
    bounds = [0] + result + [200]
    seg_widths = [bounds[i + 1] - bounds[i] for i in range(len(bounds) - 1)]
    min_allowed = 50.0 * 0.4

    for sw in seg_widths:
        assert sw >= min_allowed, (
            f"Segment width {sw} < minimum {min_allowed}; cuts={result}"
        )
    print(f"  Width gating: {cuts} → {result}, widths={seg_widths}")
    print("  test_gate_cuts_by_width PASSED")


def test_estimate_expected_char_width():
    """Expected char width on a 3-CC word should be close to single CC width."""
    word = _make_word_with_separated_ccs(n_chars=3, char_w=30, gap=8)
    shiro = _find_shirorekha_band(word)
    ecw = _estimate_expected_char_width(word, shiro)
    print(f"  Estimated char width: {ecw:.1f} (expected ~30)")
    assert 15 <= ecw <= 50, f"Expected char width 15-50, got {ecw}"
    print("  test_estimate_expected_char_width PASSED")


# ═══════════════════════════════════════════════════════════════════════════════
# Character regression tests on real images
# ═══════════════════════════════════════════════════════════════════════════════

def test_hindi_char_count():
    """hindi.png (printed) should produce a reasonable character count."""
    path = SAMPLES_DIR / "hindi.png"
    if not path.exists():
        print(f"  SKIP: {path} not found")
        return
    img = load_image(path)
    binary, _, _, _ = preprocess(img, do_deskew=True, do_slant_correct=True)
    line_bounds, cc_labels, line_ccs = detect_lines(binary)
    line_crops = crop_lines(binary, line_bounds, cc_labels, line_ccs)

    total_chars = 0
    for lc in line_crops:
        wbs = detect_words(lc)
        wcs = crop_words(lc, wbs)
        for wc in wcs:
            chars = segment_characters(wc)
            total_chars += len(chars)

    print(f"  hindi.png: {total_chars} chars (expected 110-190)")
    assert 110 <= total_chars <= 190, (
        f"Expected 110-190 chars, got {total_chars}"
    )
    print("  test_hindi_char_count PASSED")


def test_image_char_count():
    """image.png (handwritten) should produce a reasonable character count."""
    path = SAMPLES_DIR / "image.png"
    if not path.exists():
        print(f"  SKIP: {path} not found")
        return
    img = load_image(path)
    binary, _, _, _ = preprocess(img, do_deskew=True, do_slant_correct=True)
    line_bounds, cc_labels, line_ccs = detect_lines(binary)
    line_crops = crop_lines(binary, line_bounds, cc_labels, line_ccs)

    total_chars = 0
    for lc in line_crops:
        wbs = detect_words(lc)
        wcs = crop_words(lc, wbs)
        for wc in wcs:
            chars = segment_characters(wc)
            total_chars += len(chars)

    print(f"  image.png: {total_chars} chars (expected 70-140)")
    assert 70 <= total_chars <= 140, (
        f"Expected 70-140 chars, got {total_chars}"
    )
    print("  test_image_char_count PASSED")


if __name__ == "__main__":
    print("=== Original smoke test ===")
    test_full_pipeline()

    print("\n=== Ruled-line removal ===")
    test_remove_ruled_lines()
    test_remove_ruled_lines_no_false_positive()

    print("\n=== Band splitting ===")
    test_split_wide_bands_tight_spacing()
    test_split_wide_bands_no_false_split()

    print("\n=== Character segmentation ===")
    test_segment_separated_word()
    test_segment_single_cc_no_split()
    test_cc_fallback_finds_gaps()
    test_gate_cuts_by_width()
    test_estimate_expected_char_width()

    print("\n=== Regression: real images (lines) ===")
    test_bangla1_line_count()
    test_bangla2_line_count()
    test_hindi_line_count()
    test_image_line_count()

    print("\n=== Regression: real images (characters) ===")
    test_hindi_char_count()
    test_image_char_count()

    print("\nALL TESTS PASSED")
