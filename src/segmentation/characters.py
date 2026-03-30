"""Character segmentation for Hindi / Bengali script.

Implements the three-phase algorithm:
  Phase 1 – Separation of region above the headline (skeleton-based)
  Phase 2 – Fuzzy column-score computation (four features per column)
  Phase 3 – Savitzky-Golay smoothing, peak detection, false-peak removal
  Phase 3b – CC-aware cut filtering + shirorekha-gap check
  Final assembly – cut word image, reattach top components, merge tiny segments
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.signal import savgol_filter
from skimage.morphology import skeletonize


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 1 – Separation of region above headline
# ═════════════════════════════════════════════════════════════════════════════

def _row_transitions(img: np.ndarray) -> np.ndarray:
    """Count 0→1 transitions per row (vectorised)."""
    return np.sum(np.diff(img.astype(np.int8), axis=1) == 1, axis=1)


def _has_above_headline_region(img: np.ndarray, k: float = 0.6) -> bool:
    """Steps 1-3: decide whether a region above the headline exists."""
    H = img.shape[0]
    if H < 8:
        return False
    T = _row_transitions(img)
    bh = H // 4
    if bh == 0:
        return False
    bands = [
        float(np.mean(T[:bh])),
        float(np.mean(T[bh : 2 * bh])),
        float(np.mean(T[2 * bh : 3 * bh])),
        float(np.mean(T[3 * bh :])),
    ]
    mx = max(bands[1], bands[2], bands[3])
    return mx > 0 and bands[0] < k * mx


def _find_topmost_pixel(skel: np.ndarray) -> tuple[int, int] | None:
    rows, cols = np.nonzero(skel)
    if len(rows) == 0:
        return None
    idx = int(np.argmin(rows))
    return int(rows[idx]), int(cols[idx])


def _traverse_skeleton_downward(
    skel: np.ndarray, sr: int, sc: int,
) -> tuple[int, int] | None:
    """Follow the skeleton from *(sr, sc)* downward.

    Returns the first junction pixel (>3 black pixels in 3×3 window)
    or ``None`` if no junction is encountered.
    """
    H, W = skel.shape
    visited = {(sr, sc)}
    r, c = sr, sc

    for _ in range(H + W):
        r0, r1 = max(0, r - 1), min(H, r + 2)
        c0, c1 = max(0, c - 1), min(W, c + 2)
        if int(np.sum(skel[r0:r1, c0:c1])) > 3:
            return (r, c)

        best: tuple[int, int] | None = None
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < H
                    and 0 <= nc < W
                    and skel[nr, nc]
                    and (nr, nc) not in visited
                ):
                    if best is None or nr > best[0] or (
                        nr == best[0] and abs(nc - c) < abs(best[1] - c)
                    ):
                        best = (nr, nc)

        if best is None:
            return None
        visited.add(best)
        r, c = best

    return None


def separate_above_headline(
    word_binary: np.ndarray, k: float = 0.6,
) -> tuple[np.ndarray, list[tuple[np.ndarray, tuple[int, int]]]]:
    """Phase 1: iteratively remove above-headline components.

    Returns
    -------
    modified_binary : uint8 array (0/255) with top components removed.
    separated : list of ``(full_size_mask_0_255, (x_start, x_end))``
    """
    img = (word_binary > 0).astype(np.uint8)
    H, W = img.shape
    separated: list[tuple[np.ndarray, tuple[int, int]]] = []

    for _ in range(10):
        if not _has_above_headline_region(img, k):
            break

        skel = skeletonize(img.astype(bool)).astype(np.uint8)
        top = _find_topmost_pixel(skel)
        if top is None:
            break

        junction = _traverse_skeleton_downward(skel, top[0], top[1])

        # Fallback: top component may already be disconnected
        if junction is None:
            n_lab, labels, stats, _ = cv2.connectedComponentsWithStats(img, 8)
            lbl = int(labels[top[0], top[1]])
            if lbl > 0:
                ch = stats[lbl, cv2.CC_STAT_HEIGHT]
                ct = stats[lbl, cv2.CC_STAT_TOP]
                if ch < H // 2 and ct + ch < H // 2:
                    mask = (labels == lbl).astype(np.uint8) * 255
                    cx = int(stats[lbl, cv2.CC_STAT_LEFT])
                    cw = int(stats[lbl, cv2.CC_STAT_WIDTH])
                    separated.append((mask, (cx, cx + cw)))
                    img[labels == lbl] = 0
                    continue
            break

        jr, jc = junction
        if jr >= H // 2:
            break

        cut = img.copy()
        cut[max(0, jr - 1) : min(H, jr + 2), max(0, jc - 1) : min(W, jc + 2)] = 0
        n_lab, labels, stats, _ = cv2.connectedComponentsWithStats(cut, 8)

        best_lbl, best_top = -1, H
        for lbl in range(1, n_lab):
            ct = stats[lbl, cv2.CC_STAT_TOP]
            ch = stats[lbl, cv2.CC_STAT_HEIGHT]
            if ch < H // 2 and ct + ch <= jr + 2 and ct < best_top:
                best_top, best_lbl = ct, lbl

        if best_lbl == -1:
            break

        mask = (labels == best_lbl).astype(np.uint8) * 255
        cx = int(stats[best_lbl, cv2.CC_STAT_LEFT])
        cw = int(stats[best_lbl, cv2.CC_STAT_WIDTH])
        separated.append((mask, (cx, cx + cw)))
        img[labels == best_lbl] = 0

    return (img * 255).astype(np.uint8), separated


# ═════════════════════════════════════════════════════════════════════════════
# Adaptive stroke-width estimation
# ═════════════════════════════════════════════════════════════════════════════

def _estimate_stroke_width(binary: np.ndarray) -> float:
    """Estimate median stroke width via distance transform."""
    img = (binary > 0).astype(np.uint8)
    if not np.any(img):
        return 3.0
    dt = cv2.distanceTransform(img, cv2.DIST_L2, 3)
    skel = skeletonize(img.astype(bool))
    vals = dt[skel]
    if len(vals) == 0:
        return 3.0
    return float(np.median(vals)) * 2.0


# ═════════════════════════════════════════════════════════════════════════════
# Shirorekha (headline) detection
# ═════════════════════════════════════════════════════════════════════════════

def _find_shirorekha_band(binary: np.ndarray) -> tuple[int, int] | None:
    """Find the row band occupied by the shirorekha.

    Returns (row_start, row_end) of the densest horizontal ink band in the
    upper half of the image, or None if no clear headline is found.
    """
    img = (binary > 0).astype(np.uint8)
    H, W = img.shape
    hp = np.sum(img, axis=1).astype(np.float64)
    if np.max(hp) == 0:
        return None

    upper_half = hp[: max(1, H * 2 // 3)]
    peak_row = int(np.argmax(upper_half))

    threshold = hp[peak_row] * 0.5
    r0 = peak_row
    while r0 > 0 and hp[r0 - 1] >= threshold:
        r0 -= 1
    r1 = peak_row
    while r1 < H - 1 and hp[r1 + 1] >= threshold:
        r1 += 1

    band_h = r1 - r0 + 1
    if band_h < 1 or band_h > H // 2:
        return None
    return (r0, r1 + 1)


# ═════════════════════════════════════════════════════════════════════════════
# Expected character width estimation
# ═════════════════════════════════════════════════════════════════════════════

def _estimate_expected_char_width(
    binary: np.ndarray,
    shiro_band: tuple[int, int] | None,
) -> float:
    """Estimate the expected width of a single character (akshar).

    Uses the median width of connected components in the body region
    (below the shirorekha) as a proxy.  Falls back to word_width / 3
    if no usable CCs are found.
    """
    img = (binary > 0).astype(np.uint8)
    H, W = img.shape

    body_top = shiro_band[1] if shiro_band else 0
    body = img[body_top:, :] if body_top < H else img

    if not np.any(body):
        return max(W / 3.0, 10.0)

    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(body, 8)
    widths = []
    for lbl in range(1, n_labels):
        cc_w = stats[lbl, cv2.CC_STAT_WIDTH]
        cc_h = stats[lbl, cv2.CC_STAT_HEIGHT]
        cc_area = stats[lbl, cv2.CC_STAT_AREA]
        if cc_area < 20 or cc_h < H * 0.15:
            continue
        widths.append(cc_w)

    if not widths:
        return max(W / 3.0, 10.0)

    return float(np.median(widths))


def _gate_cuts_by_width(
    cuts: list[int],
    S: np.ndarray,
    word_width: int,
    expected_char_width: float,
    min_width_ratio: float = 0.4,
) -> list[int]:
    """Remove weakest cuts until all segments are at least
    *min_width_ratio* * *expected_char_width* wide.
    """
    if not cuts:
        return cuts

    min_seg_width = max(5, expected_char_width * min_width_ratio)
    result = sorted(cuts)

    changed = True
    while changed and len(result) > 0:
        changed = False
        bounds = [0] + result + [word_width]
        seg_widths = [bounds[i + 1] - bounds[i] for i in range(len(bounds) - 1)]
        narrowest_idx = int(np.argmin(seg_widths))

        if seg_widths[narrowest_idx] < min_seg_width:
            # Remove the cut adjacent to this narrow segment that has
            # the lower S value
            if narrowest_idx == 0:
                result.pop(0)
            elif narrowest_idx == len(result):
                result.pop(-1)
            else:
                left_cut = result[narrowest_idx - 1]
                right_cut = result[narrowest_idx]
                if S[left_cut] <= S[right_cut]:
                    result.pop(narrowest_idx - 1)
                else:
                    result.pop(narrowest_idx)
            changed = True

    return result


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 2 – Fuzzy character-segmentation column scores
# ═════════════════════════════════════════════════════════════════════════════

def _compute_column_scores(
    binary: np.ndarray,
    stroke_thin: float = 5.0,
    stroke_thick: float = 15.0,
) -> np.ndarray:
    """Compute the fuzzy possibility score S[j] for every column *j*.

    """
    img = (binary > 0).astype(np.uint8)
    H, W = img.shape
    S = np.zeros(W, dtype=np.float64)
    if H == 0 or W == 0:
        return S

    norm_run = H * (H + 1) / 2.0

    for j in range(W):
        col = img[:, j]

        # Feature X1 – first black pixel location
        blacks = np.nonzero(col)[0]
        if len(blacks) == 0:
            continue
        p1 = int(blacks[0])
        mu1 = np.clip(1.0 - p1 / H, 0.0, 1.0)

        # Feature X2 – thickness of first stroke
        whites_after = np.where(col[p1:] == 0)[0]
        p2 = (p1 + int(whites_after[0])) if len(whites_after) else H
        X2 = p2 - p1
        if X2 <= stroke_thin:
            mu2 = 1.0
        elif X2 >= stroke_thick:
            mu2 = 0.0
        else:
            mu2 = (stroke_thick - X2) / (stroke_thick - stroke_thin)

        # Feature X3 – white pixel count below stroke
        mu3 = (float(np.sum(col[p2:] == 0)) / H) if p2 < H else 0.0

        # Feature X4 – vertical white-run sum
        X4 = 0.0
        if p2 < H:
            below = col[p2:]
            padded = np.concatenate(([1], below, [1]))
            d = np.diff(padded.astype(np.int8))
            starts = np.where(d == -1)[0]
            ends = np.where(d == 1)[0]
            n_runs = min(len(starts), len(ends))
            if n_runs > 0:
                lengths = ends[:n_runs] - starts[:n_runs]
                X4 = float(np.sum(lengths * (lengths + 1) / 2.0))
        mu4 = X4 / norm_run if norm_run > 0 else 0.0

        S[j] = mu1 * mu2 * mu3 * mu4

    return S


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 3 – Smoothing & cut detection
# ═════════════════════════════════════════════════════════════════════════════

def _smooth_scores(
    S: np.ndarray, poly_order: int = 1, window: int = 7,
) -> np.ndarray:
    n = len(S)
    if n < 4:
        return S.copy()
    # Scale window with word width so wider words get more smoothing
    adaptive_w = max(window, n // 15)
    w = min(adaptive_w, n)
    if w % 2 == 0:
        w -= 1
    w = max(w, 3)
    po = min(poly_order, w - 1)
    return savgol_filter(S, window_length=w, polyorder=po)


def _detect_peaks(S_smooth: np.ndarray) -> list[int]:
    peaks: list[int] = []
    for j in range(1, len(S_smooth) - 1):
        if S_smooth[j] > S_smooth[j - 1] and S_smooth[j] > S_smooth[j + 1]:
            peaks.append(j)
    return peaks


def _verify_peaks(
    peaks: list[int],
    S: np.ndarray,
    S_smooth: np.ndarray,
    binary: np.ndarray,
    expected_char_width: float = 0.0,
    min_width_ratio: float = 0.4,
) -> list[int]:
    """Remove false peaks based on segment height, width, and prominence."""
    if len(peaks) < 2:
        return list(peaks)

    H, W = binary.shape
    img = (binary > 0).astype(np.uint8)
    verified = list(peaks)

    min_seg_w = max(5, expected_char_width * min_width_ratio) if expected_char_width > 0 else 5

    # Pass 1: prominence filter -- a peak must rise meaningfully above
    # the average of its neighboring valleys
    if len(S_smooth) > 2:
        prominent: list[int] = []
        for j in verified:
            left_min = float(np.min(S_smooth[max(0, j - 5):j])) if j > 0 else 0.0
            right_min = float(np.min(S_smooth[j + 1:min(len(S_smooth), j + 6)])) if j < len(S_smooth) - 1 else 0.0
            valley_avg = (left_min + right_min) / 2.0
            prominence = S_smooth[j] - valley_avg
            if prominence > 0.02 or S_smooth[j] > 0.15:
                prominent.append(j)
        verified = prominent if prominent else verified

    # Pass 2: height-based verification (original logic)
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(verified) - 1:
            c1, c2 = verified[i], verified[i + 1]
            seg = img[:, c1:c2]
            ink_rows = np.any(seg, axis=1)
            if np.any(ink_rows):
                idx = np.nonzero(ink_rows)[0]
                seg_h = int(idx[-1] - idx[0] + 1)
            else:
                seg_h = 0

            if seg_h < H / 2:
                if S[c1] <= S[c2]:
                    verified.pop(i)
                else:
                    verified.pop(i + 1)
                changed = True
            else:
                i += 1

    # Pass 3: minimum width check
    changed = True
    while changed and len(verified) > 0:
        changed = False
        bounds = [0] + verified + [W]
        for i in range(len(bounds) - 1):
            seg_w = bounds[i + 1] - bounds[i]
            if seg_w < min_seg_w and len(verified) > 0:
                # Remove the adjacent cut with lower S value
                if i == 0 and len(verified) > 0:
                    verified.pop(0)
                elif i >= len(verified) and len(verified) > 0:
                    verified.pop(-1)
                elif i - 1 < len(verified) and i < len(verified):
                    if S[verified[i - 1]] <= S[verified[i]]:
                        verified.pop(i - 1)
                    else:
                        verified.pop(i)
                elif i - 1 < len(verified):
                    verified.pop(i - 1)
                changed = True
                break

    return verified


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 3b – CC-aware + shirorekha-gap filtering of cuts
# ═════════════════════════════════════════════════════════════════════════════

def _filter_cuts_by_cc(
    cuts: list[int],
    binary: np.ndarray,
    shiro_band: tuple[int, int] | None,
) -> list[int]:
    """Remove cuts that slice through large character bodies below the
    shirorekha, or that lack a shirorekha gap.

    In Devanagari the shirorekha often connects all aksharas into one
    huge CC, so we cannot reject every cut that touches a CC.  Instead
    we focus on the region **below** the headline:

      1. If a cut column has ink from a CC whose body (below shirorekha)
         spans across the cut on both sides, the cut is rejected -- it
         would slice a character body.
      2. If a shirorekha band was detected and the cut column is solid
         ink across the entire headline band (no gap at all), the cut
         is rejected.
    """
    if not cuts:
        return cuts

    img = (binary > 0).astype(np.uint8)
    H, W = img.shape
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(img, 8)

    # Determine the "body region" = rows below the shirorekha
    body_top = shiro_band[1] if shiro_band else 0

    kept: list[int] = []
    for j in cuts:
        col = img[:, j]

        # condition 1: column has no ink at all → always keep
        if not np.any(col):
            kept.append(j)
            continue

        # condition 2: check the body region (below headline) for
        # CCs that would be sliced
        body_col = img[body_top:, j] if body_top < H else col
        slices_body = False
        if np.any(body_col):
            body_labels = labels[body_top:, j] if body_top < H else labels[:, j]
            body_lbls = set(int(l) for l in body_labels if l > 0)
            for lbl in body_lbls:
                x0 = stats[lbl, cv2.CC_STAT_LEFT]
                w = stats[lbl, cv2.CC_STAT_WIDTH]
                cc_area = stats[lbl, cv2.CC_STAT_AREA]
                if cc_area < 20:
                    continue
                if x0 < j and x0 + w - 1 > j:
                    slices_body = True
                    break

        if slices_body:
            continue

        # condition 3: shirorekha gap check -- at least one pixel in
        # the headline band must be background
        if shiro_band is not None:
            sr0, sr1 = shiro_band
            shiro_col = col[sr0:sr1]
            if len(shiro_col) > 0 and np.all(shiro_col):
                continue

        kept.append(j)

    return kept


# ═════════════════════════════════════════════════════════════════════════════
# CC-based fallback for under-segmented words
# ═════════════════════════════════════════════════════════════════════════════

def _cc_fallback_cuts(
    binary: np.ndarray,
    shiro_band: tuple[int, int] | None,
    stroke_width: float,
) -> list[int]:
    """Find cut points using gaps between CC bounding boxes.

    Used as a fallback when the fuzzy pipeline produces too few cuts
    for a word that is clearly wider than a single character.
    """
    img = (binary > 0).astype(np.uint8)
    H, W = img.shape

    body_top = shiro_band[1] if shiro_band else 0
    body = img[body_top:, :] if body_top < H else img

    if not np.any(body):
        return []

    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(body, 8)

    boxes: list[tuple[int, int]] = []
    for lbl in range(1, n_labels):
        cc_area = stats[lbl, cv2.CC_STAT_AREA]
        cc_h = stats[lbl, cv2.CC_STAT_HEIGHT]
        if cc_area < 15 or cc_h < max(3, H * 0.08):
            continue
        x0 = stats[lbl, cv2.CC_STAT_LEFT]
        cc_w = stats[lbl, cv2.CC_STAT_WIDTH]
        boxes.append((x0, x0 + cc_w))

    if len(boxes) < 2:
        return []

    boxes.sort()

    merged_boxes: list[tuple[int, int]] = [boxes[0]]
    for x0, x1 in boxes[1:]:
        prev_x0, prev_x1 = merged_boxes[-1]
        if x0 <= prev_x1 + 1:
            merged_boxes[-1] = (prev_x0, max(prev_x1, x1))
        else:
            merged_boxes.append((x0, x1))

    if len(merged_boxes) < 2:
        return []

    all_gaps = []
    for i in range(len(merged_boxes) - 1):
        all_gaps.append(merged_boxes[i + 1][0] - merged_boxes[i][1])
    median_gap = float(np.median(all_gaps)) if all_gaps else 0
    gap_threshold = max(2, min(stroke_width * 1.5, median_gap * 0.5))

    cuts: list[int] = []
    for i in range(len(merged_boxes) - 1):
        gap_start = merged_boxes[i][1]
        gap_end = merged_boxes[i + 1][0]
        gap = gap_end - gap_start
        if gap >= gap_threshold:
            cuts.append((gap_start + gap_end) // 2)

    return cuts


# ═════════════════════════════════════════════════════════════════════════════
# Skeleton-based projection profile for character segmentation
# ═════════════════════════════════════════════════════════════════════════════

def _skeleton_projection_cuts(
    binary: np.ndarray,
    expected_char_width: float,
    min_width_ratio: float = 0.4,
) -> list[int]:
    """Find cut points using vertical projection on the **skeletonised** image.

    Sahare & Dhok (2015) showed that projection profiles on thinned images
    produce much clearer inter-character valleys than on the original binary,
    because thick strokes (especially the shirorekha) are reduced to 1px
    width, revealing gaps that were hidden by stroke thickness.

    The skeleton VPP is combined with the binary VPP: a candidate cut must
    be a valley in the skeleton profile AND not pass through a high-density
    region in the binary profile.  This prevents false cuts inside thick
    conjuncts while still finding gaps in handwritten text.
    """
    img = (binary > 0).astype(np.uint8)
    H, W = img.shape

    if W < 10 or H < 5:
        return []

    skel = skeletonize(img.astype(bool)).astype(np.uint8)
    vp_skel = np.sum(skel, axis=0).astype(np.float64)
    vp_bin = np.sum(img, axis=0).astype(np.float64)

    if np.max(vp_skel) == 0:
        return []

    # Smooth the skeleton VPP to suppress single-pixel noise
    sw = max(3, W // 20)
    if sw % 2 == 0:
        sw += 1
    sw = min(sw, W if W % 2 == 1 else W - 1)
    if W >= 4:
        vp_skel_s = savgol_filter(vp_skel, window_length=sw, polyorder=1)
    else:
        vp_skel_s = vp_skel.copy()

    # Also smooth binary VPP for comparison
    if W >= 4:
        vp_bin_s = savgol_filter(vp_bin, window_length=sw, polyorder=1)
    else:
        vp_bin_s = vp_bin.copy()

    # A column is a "skeleton valley" if it falls below 20% of the skeleton peak
    skel_peak = np.max(vp_skel_s)
    skel_thresh = skel_peak * 0.20
    is_skel_valley = vp_skel_s < skel_thresh

    # Find contiguous valley runs in the skeleton profile
    padded = np.concatenate(([False], is_skel_valley, [False]))
    diff = np.diff(padded.astype(np.int8))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    if len(starts) == 0:
        return []

    min_seg_w = max(5, expected_char_width * min_width_ratio)
    bin_peak = np.max(vp_bin_s) if np.max(vp_bin_s) > 0 else 1.0

    candidates: list[tuple[int, float]] = []
    for s, e in zip(starts, ends):
        # Pick the column within the valley with the lowest skeleton density
        valley_region = vp_skel_s[s:e]
        best_offset = int(np.argmin(valley_region))
        col = s + best_offset

        if col < min_seg_w or col > W - min_seg_w:
            continue

        # Reject if the binary VPP at this column is still high
        # (would mean we're cutting through a thick stroke, not a real gap)
        bin_density = vp_bin_s[col] / bin_peak
        if bin_density > 0.6:
            continue

        # Score: lower skeleton density + lower binary density = better cut
        skel_density = vp_skel_s[col] / skel_peak if skel_peak > 0 else 0
        score = skel_density + bin_density
        candidates.append((col, score))

    if not candidates:
        return []

    # Sort by column position
    candidates.sort(key=lambda x: x[0])
    cuts = [c for c, _ in candidates]

    # Remove cuts that produce too-narrow segments, preferring cuts with
    # lower score (= deeper valley)
    changed = True
    while changed and len(cuts) > 0:
        changed = False
        bounds = [0] + cuts + [W]
        seg_widths = [bounds[i + 1] - bounds[i] for i in range(len(bounds) - 1)]
        narrowest_idx = int(np.argmin(seg_widths))

        if seg_widths[narrowest_idx] < min_seg_w:
            # Find which adjacent cut to remove (the one with higher score)
            if narrowest_idx == 0:
                cuts.pop(0)
            elif narrowest_idx == len(cuts):
                cuts.pop(-1)
            else:
                left_score = next(sc for c, sc in candidates if c == cuts[narrowest_idx - 1])
                right_score = next(sc for c, sc in candidates if c == cuts[narrowest_idx])
                if left_score >= right_score:
                    cuts.pop(narrowest_idx - 1)
                else:
                    cuts.pop(narrowest_idx)
            changed = True

    return cuts


# ═════════════════════════════════════════════════════════════════════════════
# FINAL ASSEMBLY
# ═════════════════════════════════════════════════════════════════════════════

def _crop_to_content(binary: np.ndarray, pad: int = 2) -> np.ndarray:
    coords = cv2.findNonZero(binary)
    if coords is None:
        return binary
    x, y, w, h = cv2.boundingRect(coords)
    y0, y1 = max(0, y - pad), min(binary.shape[0], y + h + pad)
    x0, x1 = max(0, x - pad), min(binary.shape[1], x + w + pad)
    return binary[y0:y1, x0:x1].copy()


def _assemble_characters(
    modified: np.ndarray,
    cuts: list[int],
    separated: list[tuple[np.ndarray, tuple[int, int]]],
) -> list[np.ndarray]:
    """Cut the modified binary at *cuts* and paste back top components."""
    H, W = modified.shape
    bounds = [0] + sorted(cuts) + [W]

    chars: list[np.ndarray] = []
    for idx in range(len(bounds) - 1):
        x0, x1 = bounds[idx], bounds[idx + 1]
        if x1 <= x0:
            continue

        seg = modified[:, x0:x1].copy()

        for comp_mask, (cx0, cx1) in separated:
            ol = max(x0, cx0)
            or_ = min(x1, cx1)
            if ol < or_:
                sl = ol - x0
                sr = or_ - x0
                seg[:, sl:sr] = np.maximum(
                    seg[:, sl:sr], comp_mask[:, ol:or_]
                )

        if np.any(seg > 0):
            chars.append(seg)

    return chars


def _merge_tiny_segments(
    chars: list[np.ndarray],
    min_width_ratio: float = 0.25,
    min_area: int = 50,
    min_ink_ratio: float = 0.10,
    stroke_width: float = 3.0,
) -> list[np.ndarray]:
    """Merge character segments that are too small to be real characters.

    A segment is "tiny" if:
    - its content width is below *min_width_ratio* of the **median** width
    - its ink area is below *min_area*
    - its ink area is < *min_ink_ratio* of the largest segment's ink area
    - its aspect ratio (w/h) < 0.3  (vertical sliver)
    """
    if len(chars) <= 1:
        return chars

    def _ink_area(ch: np.ndarray) -> int:
        return int(np.count_nonzero(ch))

    def _content_dims(ch: np.ndarray) -> tuple[int, int]:
        coords = cv2.findNonZero((ch > 0).astype(np.uint8))
        if coords is None:
            return 0, 0
        _, _, w, h = cv2.boundingRect(coords)
        return w, h

    def _merge_pair(merged: list[np.ndarray], i: int, nb: int) -> None:
        left = min(i, nb)
        right = max(i, nb)
        a, b = merged[left], merged[right]
        h = max(a.shape[0], b.shape[0])
        if a.shape[0] < h:
            pad_a = np.zeros((h - a.shape[0], a.shape[1]), dtype=a.dtype)
            a = np.vstack([a, pad_a])
        if b.shape[0] < h:
            pad_b = np.zeros((h - b.shape[0], b.shape[1]), dtype=b.dtype)
            b = np.vstack([b, pad_b])
        merged[left] = np.hstack([a, b])
        merged.pop(right)

    merged: list[np.ndarray] = list(chars)

    # Absolute floor: a character should be at least a few pixels wide,
    # but not wider than reasonable for the stroke thickness
    abs_min_w = max(6, min(stroke_width * 1.5, 15))

    changed = True
    while changed and len(merged) > 1:
        changed = False
        ink_areas = [_ink_area(ch) for ch in merged]
        max_ink = max(ink_areas) if ink_areas else 1
        ink_threshold = max(min_area, max_ink * min_ink_ratio)

        # Use median (not mean) so slivers don't drag the reference down
        widths = sorted(ch.shape[1] for ch in merged)
        median_w = widths[len(widths) // 2] if widths else 1
        threshold_w = max(abs_min_w, median_w * min_width_ratio)

        for i in range(len(merged)):
            ch = merged[i]
            cw, ch_h = _content_dims(ch)
            ca = ink_areas[i]
            ar = cw / max(ch_h, 1) if ch_h > 0 else 0

            is_tiny = (
                cw < threshold_w
                or ca < min_area
                or ca < ink_threshold
                or ar < 0.35
                or cw < 8
            )
            if is_tiny:
                if i == 0:
                    nb = 1
                elif i == len(merged) - 1:
                    nb = i - 1
                else:
                    nb = i - 1 if merged[i - 1].shape[1] >= merged[i + 1].shape[1] else i + 1
                _merge_pair(merged, i, nb)
                changed = True
                break

    return merged


# ═════════════════════════════════════════════════════════════════════════════
# Post-segmentation re-split of oversized segments
# ═════════════════════════════════════════════════════════════════════════════

def _resplit_oversized(
    chars: list[np.ndarray],
    median_char_width: float,
    max_ar: float = 2.0,
    width_mult: float = 1.8,
) -> list[np.ndarray]:
    """Re-segment any character crop that is suspiciously wide.

    A crop is considered "oversized" if its aspect ratio (w/h) exceeds
    *max_ar* OR its width exceeds *width_mult* * *median_char_width*.
    Such crops are re-segmented using a simplified skeleton-projection
    approach with relaxed parameters (no CC filtering).
    """
    if not chars or median_char_width < 5:
        return chars

    result: list[np.ndarray] = []
    for ch in chars:
        h, w = ch.shape[:2]
        ar = w / max(h, 1)
        if (ar > max_ar or w > width_mult * median_char_width) and w > 20:
            sub_cuts = _skeleton_projection_cuts(
                ch, median_char_width, min_width_ratio=0.35,
            )
            if len(sub_cuts) > 0:
                bounds = [0] + sorted(sub_cuts) + [w]
                for i in range(len(bounds) - 1):
                    x0, x1 = bounds[i], bounds[i + 1]
                    sub = ch[:, x0:x1].copy()
                    if np.any(sub > 0):
                        cropped = _crop_to_content(sub)
                        if cropped.shape[0] >= 4 and cropped.shape[1] >= 4:
                            result.append(cropped)
                continue
        result.append(ch)

    return result


# ═════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═════════════════════════════════════════════════════════════════════════════

def segment_characters(
    word_binary: np.ndarray,
    *,
    k: float = 0.6,
    sg_window: int = 7,
    sg_poly: int = 1,
    min_char_width_ratio: float = 0.4,
    return_debug: bool = False,
) -> list[np.ndarray] | tuple[list[np.ndarray], dict]:
    """Segment a binary word image into individual characters.

    Parameters
    ----------
    word_binary : H×W uint8 array (ink=255, bg=0).
    k : Threshold ratio for above-headline detection (Phase 1).
    sg_window : Savitzky-Golay smoothing window (Phase 3, must be odd).
    sg_poly : Savitzky-Golay polynomial order.
    min_char_width_ratio : Minimum segment width as a fraction of the
        estimated character width. Cuts producing narrower segments are
        removed.
    return_debug : Also return a dict of intermediate arrays.

    Returns
    -------
    List of cropped binary character images (uint8, 0/255).
    With *return_debug*: ``(characters, debug_dict)``.
    """
    empty_debug: dict = {
        "S": np.array([]),
        "S_smooth": np.array([]),
        "peaks": [],
        "cuts": [],
        "modified_binary": word_binary,
        "separated": [],
    }

    if word_binary.size == 0:
        return ([], empty_debug) if return_debug else []

    if len(word_binary.shape) == 3:
        word_binary = cv2.cvtColor(word_binary, cv2.COLOR_BGR2GRAY)
    _, word_binary = cv2.threshold(word_binary, 127, 255, cv2.THRESH_BINARY)

    H, W = word_binary.shape
    if H < 8 or W < 5:
        out = [word_binary]
        empty_debug["S"] = np.zeros(W)
        empty_debug["S_smooth"] = np.zeros(W)
        return (out, empty_debug) if return_debug else out

    # Phase 1
    modified, separated = separate_above_headline(word_binary, k=k)

    # Adaptive stroke thresholds
    sw = _estimate_stroke_width(modified)
    stroke_thin = max(2.0, sw * 0.8)
    stroke_thick = max(stroke_thin + 4, sw * 2.5)

    # Detect shirorekha band
    shiro_band = _find_shirorekha_band(modified)

    # Estimate expected character width
    expected_cw = _estimate_expected_char_width(modified, shiro_band)

    # Phase 2 – with adaptive thresholds
    S = _compute_column_scores(modified, stroke_thin, stroke_thick)

    # Phase 3 – smoothing, peak detection, verification with width + prominence
    S_smooth = _smooth_scores(S, poly_order=sg_poly, window=sg_window)
    peaks = _detect_peaks(S_smooth)
    cuts = _verify_peaks(
        peaks, S, S_smooth, modified,
        expected_char_width=expected_cw,
        min_width_ratio=min_char_width_ratio,
    )

    # Phase 3b – CC-aware + shirorekha-gap filtering
    cuts = _filter_cuts_by_cc(cuts, modified, shiro_band)

    # Width gating: remove weakest peaks until segments are wide enough
    cuts = _gate_cuts_by_width(
        cuts, S, W,
        expected_char_width=expected_cw,
        min_width_ratio=min_char_width_ratio,
    )

    # Fallback chain for under-segmented words:
    #   1. CC gap analysis
    #   2. Skeleton-based projection profile (Sahare & Dhok approach)
    n_expected_chars = max(1, round(W / expected_cw))
    too_few_cuts = len(cuts) < max(1, n_expected_chars - 1)
    word_clearly_multi = W > 1.5 * expected_cw

    if too_few_cuts and word_clearly_multi:
        cc_cuts = _cc_fallback_cuts(modified, shiro_band, sw)
        if len(cc_cuts) > len(cuts):
            cuts = cc_cuts

    # If still under-segmented, try skeleton projection fallback
    too_few_cuts = len(cuts) < max(1, n_expected_chars - 1)
    if too_few_cuts and word_clearly_multi:
        skel_cuts = _skeleton_projection_cuts(modified, expected_cw, min_char_width_ratio)
        if len(skel_cuts) > len(cuts):
            cuts = skel_cuts

    # Assembly
    chars = _assemble_characters(modified, cuts, separated)

    # Merge tiny fragments (slivers, matra fragments) into neighbors
    chars = _merge_tiny_segments(chars, stroke_width=sw)

    result = [
        _crop_to_content(ch)
        for ch in chars
        if ch.size > 0 and np.any(ch > 0)
    ]

    # Filter degenerate crops (slivers, tiny dots)
    result = [
        ch for ch in result
        if ch.shape[0] >= 4 and ch.shape[1] >= 4
        and ch.shape[0] / max(ch.shape[1], 1) < 5
        and ch.shape[1] / max(ch.shape[0], 1) < 5
    ]

    # Re-split oversized segments that the initial pass missed
    if len(result) > 1:
        seg_widths = sorted(ch.shape[1] for ch in result)
        median_seg_w = seg_widths[len(seg_widths) // 2]
    else:
        median_seg_w = expected_cw
    result = _resplit_oversized(result, median_seg_w)

    # Second merge pass to clean up any new slivers from re-splitting
    result = _merge_tiny_segments(result, stroke_width=sw)
    result = [
        _crop_to_content(ch)
        for ch in result
        if ch.size > 0 and np.any(ch > 0)
    ]
    result = [
        ch for ch in result
        if ch.shape[0] >= 4 and ch.shape[1] >= 4
        and ch.shape[0] / max(ch.shape[1], 1) < 5
        and ch.shape[1] / max(ch.shape[0], 1) < 5
    ]

    if not result:
        result = [word_binary]

    if return_debug:
        debug = {
            "S": S,
            "S_smooth": S_smooth,
            "peaks": peaks,
            "cuts": cuts,
            "modified_binary": modified,
            "separated": separated,
        }
        return result, debug
    return result
