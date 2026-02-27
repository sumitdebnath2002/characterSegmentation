"""Compute and smooth 1-D projection profiles."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d


def horizontal_profile(binary: np.ndarray) -> np.ndarray:
    """Row-wise sum of ink pixels → shape ``(height,)``."""
    return np.sum(binary > 0, axis=1).astype(np.float64)


def vertical_profile(binary: np.ndarray) -> np.ndarray:
    """Column-wise sum of ink pixels → shape ``(width,)``."""
    return np.sum(binary > 0, axis=0).astype(np.float64)


def smooth_profile(profile: np.ndarray, window: int = 5) -> np.ndarray:
    """Smooth a 1-D profile with a uniform (box) filter."""
    if window < 2:
        return profile.copy()
    return uniform_filter1d(profile, size=window, mode="nearest")


def profile_to_runs(
    profile: np.ndarray,
    threshold: float | None = None,
    min_run: int = 5,
    min_gap: int = 3,
) -> list[tuple[int, int]]:
    """Convert a 1-D profile into contiguous runs of "text" indices.

    A position is considered text when ``profile[i] > threshold``.
    Runs shorter than *min_run* are discarded.
    Gaps shorter than *min_gap* are bridged (merged into adjacent runs).

    Returns a list of ``(start, end)`` pairs (end exclusive).
    """
    if threshold is None:
        peak = np.max(profile)
        threshold = peak * 0.1 if peak > 0 else 0

    text_mask = profile > threshold

    runs: list[tuple[int, int]] = []
    in_run = False
    start = 0
    for i, v in enumerate(text_mask):
        if v and not in_run:
            start = i
            in_run = True
        elif not v and in_run:
            runs.append((start, i))
            in_run = False
    if in_run:
        runs.append((start, len(text_mask)))

    # bridge small gaps
    merged: list[tuple[int, int]] = []
    for s, e in runs:
        if merged and (s - merged[-1][1]) < min_gap:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))

    # discard tiny runs
    merged = [(s, e) for s, e in merged if (e - s) >= min_run]
    return merged
