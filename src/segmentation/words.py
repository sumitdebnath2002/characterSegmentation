"""Word segmentation via vertical projection profiling."""

from __future__ import annotations

import cv2
import numpy as np

from .headline import attenuate_headline
from .profiles import vertical_profile, smooth_profile, profile_to_runs


def _expand_bounds_with_ccs(
    line_binary: np.ndarray,
    word_bounds: list[tuple[int, int]],
    pad: int = 3,
) -> list[tuple[int, int]]:
    """Expand each word box to include any connected component in
    *line_binary* that overlaps with it.

    This captures matras (ि, ी, ु, ू, etc.) that are connected to
    characters through the shirorekha or main strokes but whose pixels
    extend beyond the vertical-projection boundaries.

    A small *pad* is added on each side after expansion.
    """
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        line_binary, connectivity=8,
    )
    h, w = line_binary.shape

    expanded: list[tuple[int, int]] = []
    for wx0, wx1 in word_bounds:
        x0, x1 = wx0, wx1
        # Find all CC labels that have at least one pixel inside [wx0, wx1)
        region_labels = np.unique(labels[:, wx0:wx1])
        for lbl in region_labels:
            if lbl == 0:
                continue
            cc_x = stats[lbl, cv2.CC_STAT_LEFT]
            cc_w = stats[lbl, cv2.CC_STAT_WIDTH]
            x0 = min(x0, cc_x)
            x1 = max(x1, cc_x + cc_w)

        x0 = max(0, x0 - pad)
        x1 = min(w, x1 + pad)
        expanded.append((x0, x1))

    # Merge any boxes that now overlap
    if len(expanded) < 2:
        return expanded

    expanded.sort()
    merged: list[tuple[int, int]] = [expanded[0]]
    for x0, x1 in expanded[1:]:
        if x0 <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], x1))
        else:
            merged.append((x0, x1))
    return merged


def detect_words(
    line_binary: np.ndarray,
    smooth_window: int = 5,
    threshold: float | None = None,
    min_word_width: int = 8,
    min_gap: int = 4,
    remove_headline: bool = True,
    expand_with_ccs: bool = True,
    pad: int = 3,
) -> list[tuple[int, int]]:
    """Detect word boundaries inside a single-line binary crop.

    Returns a list of ``(x_start, x_end)`` pairs (end exclusive).

    When *expand_with_ccs* is True (default), each initial word box is
    expanded to include every connected component in the original
    *line_binary* that overlaps with it.  This prevents matras from being
    clipped.
    """
    if remove_headline:
        mask = attenuate_headline(line_binary)
    else:
        mask = line_binary

    vp = vertical_profile(mask)
    vp = smooth_profile(vp, window=smooth_window)
    runs = profile_to_runs(vp, threshold=threshold, min_run=min_word_width, min_gap=min_gap)

    if expand_with_ccs and runs:
        runs = _expand_bounds_with_ccs(line_binary, runs, pad=pad)

    return runs


def crop_words(
    line_binary: np.ndarray,
    word_bounds: list[tuple[int, int]],
    original_line: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Crop word regions from a line image.

    If *original_line* is given the crops are taken from it (for saving
    original-quality word images).
    """
    src = original_line if original_line is not None else line_binary
    crops: list[np.ndarray] = []
    for x0, x1 in word_bounds:
        crops.append(src[:, x0:x1].copy())
    return crops
