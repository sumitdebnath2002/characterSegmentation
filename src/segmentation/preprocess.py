"""Preprocessing pipeline: grayscale, binarisation, denoise, deskew, slant correction, margin crop."""

from __future__ import annotations

import cv2
import numpy as np


def to_grayscale(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def denoise(gray: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Light median-filter denoising."""
    return cv2.medianBlur(gray, ksize)


def binarize(gray: np.ndarray, method: str = "otsu") -> np.ndarray:
    """Return a binary ink mask (ink=255, bg=0).

    *method*: ``"otsu"`` or ``"adaptive"``.
    """
    if method == "adaptive":
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15
        )
    else:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def remove_small_noise(binary: np.ndarray, min_area: int = 30) -> np.ndarray:
    """Remove connected components smaller than *min_area* pixels."""
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    clean = np.zeros_like(binary)
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == i] = 255
    return clean


# ── Ruled-line suppression ────────────────────────────────────────────────────

def remove_ruled_lines(
    binary: np.ndarray,
    min_line_fraction: float = 0.6,
    min_rule_count: int = 3,
) -> np.ndarray:
    """Remove horizontal ruled notebook lines from a binary image.

    Uses morphological opening with a very wide horizontal kernel to
    isolate strokes that span a large fraction of the page width.  These
    are much longer than any text shirorekha (which is word-length), so
    the distinction is reliable.

    The removal is only applied when at least *min_rule_count* such long
    strokes are found, to avoid false positives on non-ruled images.
    """
    h, w = binary.shape
    kw = max(int(w * min_line_fraction), 50)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 1))
    rules = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    if not np.any(rules):
        return binary

    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(rules, 8)
    long_count = sum(
        1 for i in range(1, n_labels)
        if stats[i, cv2.CC_STAT_WIDTH] > w * 0.4
    )
    if long_count >= min_rule_count:
        return cv2.subtract(binary, rules)
    return binary


# ── Skew (whole-page rotation) ──────────────────────────────────────────────

def estimate_skew(binary: np.ndarray, angle_range: float = 10.0, steps: int = 181) -> float:
    """Estimate page skew by maximising horizontal-profile variance.

    Two-pass approach: coarse sweep then fine refinement around the best
    candidate.  Searches ``[-angle_range, +angle_range]`` degrees.
    """
    h, w = binary.shape

    def _score(angle: float) -> float:
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        rotated = cv2.warpAffine(binary, M, (w, h), flags=cv2.INTER_NEAREST)
        profile = np.sum(rotated, axis=1).astype(np.float64)
        return float(np.var(profile))

    # coarse pass
    coarse_angles = np.linspace(-angle_range, angle_range, steps)
    scores = [_score(a) for a in coarse_angles]
    best_idx = int(np.argmax(scores))
    coarse_best = coarse_angles[best_idx]

    # fine pass: ±1° around coarse best with 0.05° steps
    fine_angles = np.linspace(coarse_best - 1.0, coarse_best + 1.0, 41)
    fine_scores = [_score(a) for a in fine_angles]
    best_angle = fine_angles[int(np.argmax(fine_scores))]
    return float(best_angle)


def deskew(binary: np.ndarray, angle: float | None = None) -> np.ndarray:
    """Rotate *binary* to remove skew.  If *angle* is ``None`` it is estimated."""
    if angle is None:
        angle = estimate_skew(binary)
    if abs(angle) < 0.05:
        return binary
    h, w = binary.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(binary, M, (w, h), flags=cv2.INTER_NEAREST)


# ── Slant correction (character lean) ────────────────────────────────────────

def _suppress_horizontal(binary: np.ndarray) -> np.ndarray:
    """Zero-out long horizontal strokes (shirorekha) that would bias
    gradient-based slant estimation."""
    h_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (max(binary.shape[1] // 12, 7), 1)
    )
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
    return cv2.subtract(binary, horizontal)


def estimate_slant(binary: np.ndarray) -> float:
    """Estimate the shear angle that straightens handwriting.

    Removes shirorekha, then sweeps horizontal shear angles and picks the
    one that maximises the *sharpness* (sum of absolute first-differences)
    of the vertical projection profile.  This metric is immune to
    border-padding effects that plague raw variance.

    Two-pass: coarse (2° steps) then fine (0.25° steps).

    Returns the correction shear angle in degrees.
    """
    cleaned = _suppress_horizontal(binary)
    if np.count_nonzero(cleaned) < 100:
        return 0.0

    h, w = cleaned.shape

    def _vp_sharpness(angle_deg: float) -> float:
        shear_k = np.tan(np.radians(angle_deg))
        offset = -shear_k * h / 2
        M = np.float32([[1, shear_k, offset], [0, 1, 0]])
        new_w = int(w + abs(shear_k) * h) + 2
        shifted = cv2.warpAffine(cleaned, M, (new_w, h), flags=cv2.INTER_NEAREST)
        vp = np.sum(shifted > 0, axis=0).astype(np.float64)
        # crop VP to the tight ink bounding box to avoid border inflation
        nonzero = np.nonzero(vp > 0)[0]
        if len(nonzero) < 2:
            return 0.0
        vp = vp[nonzero[0]:nonzero[-1] + 1]
        return float(np.sum(np.abs(np.diff(vp))))

    # coarse sweep
    coarse_angles = np.arange(-30, 31, 2.0)
    coarse_scores = [_vp_sharpness(a) for a in coarse_angles]
    best_coarse = coarse_angles[int(np.argmax(coarse_scores))]

    # fine sweep around coarse best
    fine_angles = np.arange(best_coarse - 2, best_coarse + 2.25, 0.25)
    fine_scores = [_vp_sharpness(a) for a in fine_angles]
    best_angle = fine_angles[int(np.argmax(fine_scores))]

    return float(best_angle)


def correct_slant(binary: np.ndarray, angle: float | None = None) -> np.ndarray:
    """Apply a horizontal shear to straighten slanted handwriting.

    If *angle* is ``None`` it is estimated automatically.
    """
    if angle is None:
        angle = estimate_slant(binary)
    if abs(angle) < 1.5:
        return binary

    h, w = binary.shape
    shear_k = np.tan(np.radians(angle))
    offset = -shear_k * h / 2
    M = np.float32([[1, shear_k, offset], [0, 1, 0]])
    new_w = int(w + abs(shear_k) * h)
    result = cv2.warpAffine(binary, M, (new_w, h), flags=cv2.INTER_NEAREST)
    return result


# ── Margin crop ──────────────────────────────────────────────────────────────

def crop_margins(binary: np.ndarray, pad: int = 10) -> tuple[np.ndarray, int, int]:
    """Crop to the tight bounding box of ink plus *pad* pixels on each side.

    Returns ``(cropped, y_offset, x_offset)`` so coordinates can be mapped
    back to the original image.
    """
    coords = cv2.findNonZero(binary)
    if coords is None:
        return binary, 0, 0
    x, y, w, h = cv2.boundingRect(coords)
    y0 = max(y - pad, 0)
    y1 = min(y + h + pad, binary.shape[0])
    x0 = max(x - pad, 0)
    x1 = min(x + w + pad, binary.shape[1])
    return binary[y0:y1, x0:x1], y0, x0


# ── Full pipeline ────────────────────────────────────────────────────────────

def preprocess(
    img: np.ndarray,
    binarize_method: str = "otsu",
    do_deskew: bool = True,
    do_slant_correct: bool = True,
    do_crop: bool = True,
    do_remove_rules: bool = True,
    noise_min_area: int = 30,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Full preprocessing pipeline.

    Returns ``(binary_cropped, gray, y_offset, x_offset)``.
    """
    gray = to_grayscale(img)
    gray = denoise(gray)
    binary = binarize(gray, method=binarize_method)
    binary = remove_small_noise(binary, min_area=noise_min_area)
    if do_remove_rules:
        binary = remove_ruled_lines(binary)
    if do_deskew:
        binary = deskew(binary)
    if do_slant_correct:
        binary = correct_slant(binary)
    y_off, x_off = 0, 0
    if do_crop:
        binary, y_off, x_off = crop_margins(binary)
    return binary, gray, y_off, x_off
