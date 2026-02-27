"""Headline (shirorekha / matra) detection and attenuation.

Devanagari and Bangla scripts use a horizontal headline that connects
characters within a word.  This headline creates a strong horizontal
component that can bridge vertical gaps between words, making vertical
projection-based word segmentation unreliable.  Attenuating it before
computing the vertical profile greatly improves word boundary detection.
"""

from __future__ import annotations

import cv2
import numpy as np

from .profiles import horizontal_profile


def detect_headline_band(
    line_binary: np.ndarray, upper_fraction: float = 0.45
) -> tuple[int, int] | None:
    """Find the headline band in a single-line binary crop.

    Looks for the row with the strongest ink density in the upper
    *upper_fraction* of the line.  Returns ``(row_start, row_end)``
    of the headline band, or ``None`` if no clear headline is found.
    """
    h = line_binary.shape[0]
    search_h = max(int(h * upper_fraction), 1)
    hp = horizontal_profile(line_binary[:search_h])
    if hp.max() == 0:
        return None

    peak_row = int(np.argmax(hp))
    peak_val = hp[peak_row]

    # Expand around the peak while the profile stays above 60 % of peak
    threshold = peak_val * 0.6
    r0 = peak_row
    while r0 > 0 and hp[r0 - 1] >= threshold:
        r0 -= 1
    r1 = peak_row
    while r1 < search_h - 1 and hp[r1 + 1] >= threshold:
        r1 += 1
    r1 += 1  # exclusive

    band_height = r1 - r0
    if band_height < 2 or band_height > h * 0.4:
        return None
    return (r0, r1)


def attenuate_headline(
    line_binary: np.ndarray,
    headline_band: tuple[int, int] | None = None,
    kernel_width_fraction: float = 0.10,
) -> np.ndarray:
    """Return a copy of *line_binary* with the headline stroke removed.

    Long horizontal components inside the headline band are detected via
    morphological opening with a wide horizontal kernel and then subtracted.
    """
    if headline_band is None:
        headline_band = detect_headline_band(line_binary)
    if headline_band is None:
        return line_binary.copy()

    r0, r1 = headline_band
    w = line_binary.shape[1]
    kw = max(int(w * kernel_width_fraction), 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 1))

    band = line_binary[r0:r1, :].copy()
    horizontal_strokes = cv2.morphologyEx(band, cv2.MORPH_OPEN, kernel)

    result = line_binary.copy()
    result[r0:r1, :] = cv2.subtract(band, horizontal_strokes)
    return result
