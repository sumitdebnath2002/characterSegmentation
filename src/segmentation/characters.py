"""Character segmentation for Hindi / Bengali script.

Implements the three-phase algorithm:
  Phase 1 – Separation of region above the headline (skeleton-based)
  Phase 2 – Fuzzy column-score computation (four features per column)
  Phase 3 – Savitzky-Golay smoothing, peak detection, false-peak removal
  Final assembly – cut word image and reattach separated top components
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
# PHASE 2 – Fuzzy character-segmentation column scores
# ═════════════════════════════════════════════════════════════════════════════

_STROKE_THIN = 5
_STROKE_THICK = 15


def _compute_column_scores(binary: np.ndarray) -> np.ndarray:
    """Compute the fuzzy possibility score S[j] for every column *j*."""
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
        if X2 <= _STROKE_THIN:
            mu2 = 1.0
        elif X2 >= _STROKE_THICK:
            mu2 = 0.0
        else:
            mu2 = (_STROKE_THICK - X2) / (_STROKE_THICK - _STROKE_THIN)

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
    w = min(window, n)
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
    peaks: list[int], S: np.ndarray, binary: np.ndarray,
) -> list[int]:
    """Remove false peaks whose inter-cut segment height < H/2."""
    if len(peaks) < 2:
        return list(peaks)

    H = binary.shape[0]
    img = (binary > 0).astype(np.uint8)
    verified = list(peaks)

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

    return verified


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


# ═════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═════════════════════════════════════════════════════════════════════════════

def segment_characters(
    word_binary: np.ndarray,
    *,
    k: float = 0.6,
    sg_window: int = 7,
    sg_poly: int = 1,
    return_debug: bool = False,
) -> list[np.ndarray] | tuple[list[np.ndarray], dict]:
    """Segment a binary word image into individual characters.

    Parameters
    ----------
    word_binary : H×W uint8 array (ink=255, bg=0).
    k : Threshold ratio for above-headline detection (Phase 1).
    sg_window : Savitzky-Golay smoothing window (Phase 3, must be odd).
    sg_poly : Savitzky-Golay polynomial order.
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

    # Phase 2
    S = _compute_column_scores(modified)

    # Phase 3
    S_smooth = _smooth_scores(S, poly_order=sg_poly, window=sg_window)
    peaks = _detect_peaks(S_smooth)
    cuts = _verify_peaks(peaks, S, modified)

    # Assembly
    chars = _assemble_characters(modified, cuts, separated)
    result = [
        _crop_to_content(ch)
        for ch in chars
        if ch.size > 0 and np.any(ch > 0)
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
