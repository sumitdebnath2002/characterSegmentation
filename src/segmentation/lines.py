"""Line segmentation via horizontal projection profiling + CCA refinement."""

from __future__ import annotations

import cv2
import numpy as np

from .profiles import horizontal_profile, smooth_profile, profile_to_runs


def _core_line_bands(
    binary: np.ndarray,
    smooth_window: int = 5,
    threshold: float | None = None,
    min_line_height: int = 10,
    min_gap: int = 3,
) -> list[tuple[int, int]]:
    """Find raw text-line bands using horizontal projection profiling.

    Returns ``(y_start, y_end)`` pairs -- tight around the ink density
    peaks, with no padding.  These serve as "core" bands for CCA
    assignment.
    """
    hp = horizontal_profile(binary)
    hp = smooth_profile(hp, window=smooth_window)
    return profile_to_runs(hp, threshold=threshold, min_run=min_line_height, min_gap=min_gap)


def _assign_ccs_to_lines(
    binary: np.ndarray,
    core_bands: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray, list[list[int]]]:
    """Assign every connected component to the nearest core line band.

    Returns
    -------
    labels : np.ndarray
        The label map from ``cv2.connectedComponentsWithStats``.
    stats : np.ndarray
        CC stats array (n_labels x 5).
    line_ccs : list[list[int]]
        ``line_ccs[i]`` is the list of CC label ids assigned to line *i*.
    """
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8,
    )

    # midpoint of each core band
    band_mids = [(y0 + y1) / 2.0 for y0, y1 in core_bands]

    # maximum distance a CC centroid may be from a band centre to be
    # assigned to it -- use 1.5x the average line height as the cutoff
    avg_line_h = sum(y1 - y0 for y0, y1 in core_bands) / len(core_bands)
    max_dist = avg_line_h * 1.5
    # CCs taller than this are likely borders / noise, not text
    max_cc_height = avg_line_h * 3.0
    img_h, img_w = binary.shape

    line_ccs: list[list[int]] = [[] for _ in core_bands]

    for lbl in range(1, n_labels):  # skip background (0)
        cc_h = stats[lbl, cv2.CC_STAT_HEIGHT]
        cc_w = stats[lbl, cv2.CC_STAT_WIDTH]

        # skip oversized CCs (image borders, large blobs)
        if cc_h > max_cc_height or cc_w > img_w * 0.9:
            continue

        cy = centroids[lbl][1]  # vertical centroid of the CC

        # find the core band whose centre is closest
        best_line = 0
        best_dist = abs(cy - band_mids[0])
        for idx, mid in enumerate(band_mids[1:], 1):
            d = abs(cy - mid)
            if d < best_dist:
                best_dist = d
                best_line = idx

        # ignore CCs that are too far from any line (noise, captions, etc.)
        if best_dist > max_dist:
            continue

        line_ccs[best_line].append(lbl)

    return labels, stats, line_ccs


def detect_lines(
    binary: np.ndarray,
    smooth_window: int = 5,
    threshold: float | None = None,
    min_line_height: int = 10,
    min_gap: int = 3,
) -> tuple[list[tuple[int, int]], np.ndarray, list[list[int]]]:
    """Detect text lines using HPP for core bands + CCA for full coverage.

    Returns
    -------
    line_bounds : list[tuple[int, int]]
        ``(y_start, y_end)`` pairs expanded to cover every CC assigned to
        the line, sorted top-to-bottom.
    labels : np.ndarray
        CC label map for the full binary image.
    line_ccs : list[list[int]]
        CC label ids assigned to each line.
    """
    core_bands = _core_line_bands(
        binary, smooth_window, threshold, min_line_height, min_gap,
    )
    if not core_bands:
        empty_labels = np.zeros(binary.shape, dtype=np.int32)
        return [], empty_labels, []

    labels, stats, line_ccs = _assign_ccs_to_lines(binary, core_bands)

    h = binary.shape[0]
    line_bounds: list[tuple[int, int]] = []
    for i, cc_ids in enumerate(line_ccs):
        if not cc_ids:
            # fallback to the core band if no CCs were assigned
            line_bounds.append(core_bands[i])
            continue
        y0 = h
        y1 = 0
        for lbl in cc_ids:
            cy = stats[lbl, cv2.CC_STAT_TOP]
            ch = stats[lbl, cv2.CC_STAT_HEIGHT]
            y0 = min(y0, cy)
            y1 = max(y1, cy + ch)
        line_bounds.append((max(0, y0), min(h, y1)))

    return line_bounds, labels, line_ccs


def crop_lines(
    binary: np.ndarray,
    line_bounds: list[tuple[int, int]],
    labels: np.ndarray | None = None,
    line_ccs: list[list[int]] | None = None,
) -> list[np.ndarray]:
    """Crop line strips from *binary*.

    If *labels* and *line_ccs* are provided (from :func:`detect_lines`),
    pixels that belong to CCs assigned to a **different** line are masked
    out.  This prevents modifier fragments from neighbouring lines from
    leaking into the crop.
    """
    crops: list[np.ndarray] = []
    for i, (y0, y1) in enumerate(line_bounds):
        strip = binary[y0:y1, :].copy()

        if labels is not None and line_ccs is not None:
            # build a mask: keep only pixels whose CC label is in this line
            label_strip = labels[y0:y1, :]
            keep = np.zeros_like(strip, dtype=bool)
            for lbl in line_ccs[i]:
                keep |= (label_strip == lbl)
            strip[~keep] = 0

        crops.append(strip)
    return crops
