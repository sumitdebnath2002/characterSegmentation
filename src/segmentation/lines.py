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


def _split_wide_bands(
    hp: np.ndarray,
    bands: list[tuple[int, int]],
    max_height_ratio: float = 2.0,
    valley_ratio: float = 0.5,
    min_line_height: int = 10,
) -> list[tuple[int, int]]:
    """Split any band taller than *max_height_ratio* * median band height.

    Within an oversized band, local minima of the horizontal profile that
    dip below *valley_ratio* * the local peak are used as split points.
    Splitting repeats until no oversized bands remain (or no valid split
    point is found).
    """
    if len(bands) < 1:
        return bands

    heights = [y1 - y0 for y0, y1 in bands]
    median_h = float(np.median(heights)) if len(heights) > 1 else heights[0] / 2.0
    threshold_h = max(median_h * max_height_ratio, min_line_height * 3)

    result: list[tuple[int, int]] = []
    for y0, y1 in bands:
        band_h = y1 - y0
        if band_h <= threshold_h:
            result.append((y0, y1))
            continue

        sub_bands = _try_split_band(hp, y0, y1, median_h, valley_ratio, min_line_height)
        result.extend(sub_bands)

    return sorted(result)


def _try_split_band(
    hp: np.ndarray,
    y0: int,
    y1: int,
    expected_h: float,
    valley_ratio: float,
    min_line_height: int,
) -> list[tuple[int, int]]:
    """Recursively split a single band using local minima."""
    band_h = y1 - y0
    if band_h < min_line_height * 2:
        return [(y0, y1)]

    segment = hp[y0:y1]
    local_peak = float(np.max(segment))
    if local_peak == 0:
        return [(y0, y1)]

    # Find all local minima within the band
    minima: list[tuple[float, int]] = []
    margin = max(int(expected_h * 0.3), 3)
    for i in range(margin, len(segment) - margin):
        if segment[i] <= segment[i - 1] and segment[i] <= segment[i + 1]:
            if segment[i] < local_peak * valley_ratio:
                minima.append((float(segment[i]), i))

    if not minima:
        return [(y0, y1)]

    # Estimate how many lines this band should contain
    n_expected = max(1, round(band_h / expected_h))
    if n_expected <= 1:
        return [(y0, y1)]

    # We need (n_expected - 1) split points; pick the deepest minima
    # that are sufficiently spaced apart
    minima.sort(key=lambda x: x[0])

    split_rows: list[int] = []
    min_spacing = max(int(expected_h * 0.4), min_line_height)
    for _, local_idx in minima:
        abs_row = y0 + local_idx
        too_close = any(abs(abs_row - s) < min_spacing for s in split_rows)
        if not too_close:
            split_rows.append(abs_row)
        if len(split_rows) >= n_expected - 1:
            break

    if not split_rows:
        return [(y0, y1)]

    split_rows.sort()
    boundaries = [y0] + split_rows + [y1]
    sub_bands = []
    for i in range(len(boundaries) - 1):
        s, e = boundaries[i], boundaries[i + 1]
        if e - s >= min_line_height:
            sub_bands.append((s, e))

    return sub_bands if sub_bands else [(y0, y1)]


def _assign_ccs_to_lines(
    binary: np.ndarray,
    core_bands: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[list[int]]]:
    """Assign every connected component to the nearest core line band.

    Returns
    -------
    labels : np.ndarray
        The label map from ``cv2.connectedComponentsWithStats``.
    stats : np.ndarray
        CC stats array (n_labels x 5).
    centroids : np.ndarray
        CC centroid array (n_labels x 2).
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

    return labels, stats, centroids, line_ccs


def _validate_and_split_lines(
    line_bounds: list[tuple[int, int]],
    labels: np.ndarray,
    stats: np.ndarray,
    centroids: np.ndarray,
    line_ccs: list[list[int]],
    max_height_ratio: float = 2.0,
    min_line_height: int = 10,
) -> tuple[list[tuple[int, int]], list[list[int]]]:
    """Split any final line whose height >> expected using CC centroid clustering.

    After CCA expands line bounds, some lines may still be oversized
    because the HPP bands themselves were too coarse.  This function
    detects oversized lines by comparing each line's height to the
    median, then splits them by finding the largest gap(s) in the
    vertical centroid distribution of their assigned CCs.
    """
    if len(line_bounds) < 1:
        return line_bounds, line_ccs

    heights = [y1 - y0 for y0, y1 in line_bounds]
    median_h = float(np.median(heights))
    threshold_h = max(median_h * max_height_ratio, min_line_height * 3)

    new_bounds: list[tuple[int, int]] = []
    new_ccs: list[list[int]] = []

    img_h = labels.shape[0]

    for i, (y0, y1) in enumerate(line_bounds):
        cc_ids = line_ccs[i]
        band_h = y1 - y0

        if band_h <= threshold_h or len(cc_ids) < 2:
            new_bounds.append((y0, y1))
            new_ccs.append(cc_ids)
            continue

        n_expected = max(2, round(band_h / median_h))

        cy_list = [(centroids[lbl][1], lbl) for lbl in cc_ids]
        cy_list.sort()

        # Find the (n_expected - 1) largest gaps between sorted centroids
        gaps: list[tuple[float, int]] = []
        for j in range(1, len(cy_list)):
            gap = cy_list[j][0] - cy_list[j - 1][0]
            gaps.append((gap, j))

        gaps.sort(reverse=True)
        min_gap_size = median_h * 0.3
        split_indices: list[int] = []
        for gap_size, gap_idx in gaps:
            if gap_size < min_gap_size:
                break
            split_indices.append(gap_idx)
            if len(split_indices) >= n_expected - 1:
                break

        if not split_indices:
            new_bounds.append((y0, y1))
            new_ccs.append(cc_ids)
            continue

        split_indices.sort()
        partitions = [0] + split_indices + [len(cy_list)]
        for k in range(len(partitions) - 1):
            part_lbls = [cy_list[j][1] for j in range(partitions[k], partitions[k + 1])]
            if not part_lbls:
                continue
            sub_y0 = img_h
            sub_y1 = 0
            for lbl in part_lbls:
                ct = stats[lbl, cv2.CC_STAT_TOP]
                ch = stats[lbl, cv2.CC_STAT_HEIGHT]
                sub_y0 = min(sub_y0, ct)
                sub_y1 = max(sub_y1, ct + ch)
            new_bounds.append((max(0, sub_y0), min(img_h, sub_y1)))
            new_ccs.append(part_lbls)

    # Sort by y0
    order = sorted(range(len(new_bounds)), key=lambda k: new_bounds[k][0])
    return [new_bounds[k] for k in order], [new_ccs[k] for k in order]


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

    # Split oversized bands that likely contain multiple merged lines
    hp = smooth_profile(horizontal_profile(binary), window=smooth_window)
    core_bands = _split_wide_bands(hp, core_bands, min_line_height=min_line_height)

    labels, stats, centroids, line_ccs = _assign_ccs_to_lines(binary, core_bands)

    h = binary.shape[0]
    line_bounds: list[tuple[int, int]] = []
    for i, cc_ids in enumerate(line_ccs):
        if not cc_ids:
            line_bounds.append(core_bands[i])
            continue
        y0 = h
        y1 = 0
        for lbl in cc_ids:
            ct = stats[lbl, cv2.CC_STAT_TOP]
            ch = stats[lbl, cv2.CC_STAT_HEIGHT]
            y0 = min(y0, ct)
            y1 = max(y1, ct + ch)
        line_bounds.append((max(0, y0), min(h, y1)))

    # Safety net: split any final line that is still oversized
    line_bounds, line_ccs = _validate_and_split_lines(
        line_bounds, labels, stats, centroids, line_ccs,
        min_line_height=min_line_height,
    )

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
