"""Visualization helpers: annotated images and projection profile plots."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .profiles import horizontal_profile, vertical_profile, smooth_profile


# ── Colours (BGR) ───────────────────────────────────────────────────────────

LINE_COLOR = (0, 200, 0)       # green for line boxes
WORD_COLOR = (255, 80, 80)     # blue-ish for word boxes
LINE_THICKNESS = 2
WORD_THICKNESS = 1


# ── Annotated image ─────────────────────────────────────────────────────────

def draw_line_boxes(
    img: np.ndarray,
    line_bounds: list[tuple[int, int]],
    y_offset: int = 0,
    x_offset: int = 0,
) -> np.ndarray:
    """Draw green rectangles for each detected line on a copy of *img*."""
    canvas = img.copy()
    if len(canvas.shape) == 2:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    h, w = canvas.shape[:2]
    for idx, (y0, y1) in enumerate(line_bounds):
        ay0, ay1 = y0 + y_offset, y1 + y_offset
        cv2.rectangle(canvas, (x_offset, ay0), (w - 1, ay1), LINE_COLOR, LINE_THICKNESS)
        cv2.putText(
            canvas, f"L{idx}", (x_offset + 2, ay0 + 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, LINE_COLOR, 1,
        )
    return canvas


def draw_word_boxes(
    img: np.ndarray,
    line_bounds: list[tuple[int, int]],
    words_per_line: list[list[tuple[int, int]]],
    y_offset: int = 0,
    x_offset: int = 0,
) -> np.ndarray:
    """Draw red rectangles for each word, overlaid on *img*."""
    canvas = img.copy()
    if len(canvas.shape) == 2:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    for line_idx, ((ly0, ly1), word_list) in enumerate(zip(line_bounds, words_per_line)):
        ay0, ay1 = ly0 + y_offset, ly1 + y_offset
        for w_idx, (wx0, wx1) in enumerate(word_list):
            ax0 = wx0 + x_offset
            ax1 = wx1 + x_offset
            cv2.rectangle(canvas, (ax0, ay0), (ax1, ay1), WORD_COLOR, WORD_THICKNESS)
    return canvas


def annotate_image(
    img: np.ndarray,
    line_bounds: list[tuple[int, int]],
    words_per_line: list[list[tuple[int, int]]],
    y_offset: int = 0,
    x_offset: int = 0,
) -> np.ndarray:
    """Draw both line and word boxes on a copy of the original image."""
    canvas = draw_line_boxes(img, line_bounds, y_offset, x_offset)
    canvas = draw_word_boxes(canvas, line_bounds, words_per_line, y_offset, x_offset)
    return canvas


# ── Profile plots ────────────────────────────────────────────────────────────

def plot_horizontal_profile(
    binary: np.ndarray,
    line_bounds: list[tuple[int, int]],
    save_path: str | Path | None = None,
    smooth_window: int = 5,
) -> None:
    """Plot horizontal projection profile with detected line boundaries."""
    hp = horizontal_profile(binary)
    hp_smooth = smooth_profile(hp, smooth_window)
    fig, axes = plt.subplots(1, 2, figsize=(14, max(6, binary.shape[0] / 80)))

    axes[0].imshow(binary, cmap="gray")
    axes[0].set_title("Binary image")
    for y0, y1 in line_bounds:
        axes[0].axhline(y0, color="lime", linewidth=0.7)
        axes[0].axhline(y1, color="red", linewidth=0.7)

    axes[1].plot(hp_smooth, np.arange(len(hp_smooth)), color="steelblue")
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Ink pixel count")
    axes[1].set_ylabel("Row (y)")
    axes[1].set_title("Horizontal projection profile")
    for y0, y1 in line_bounds:
        axes[1].axhline(y0, color="lime", linewidth=0.7, linestyle="--")
        axes[1].axhline(y1, color="red", linewidth=0.7, linestyle="--")

    plt.tight_layout()
    if save_path:
        fig.savefig(str(save_path), dpi=150)
    plt.close(fig)


def plot_vertical_profile(
    line_binary: np.ndarray,
    word_bounds: list[tuple[int, int]],
    line_idx: int = 0,
    save_path: str | Path | None = None,
    smooth_window: int = 5,
) -> None:
    """Plot vertical projection profile for one line with word boundaries."""
    vp = vertical_profile(line_binary)
    vp_smooth = smooth_profile(vp, smooth_window)
    fig, axes = plt.subplots(2, 1, figsize=(max(8, line_binary.shape[1] / 60), 5))

    axes[0].imshow(line_binary, cmap="gray")
    axes[0].set_title(f"Line {line_idx}")
    for x0, x1 in word_bounds:
        axes[0].axvline(x0, color="cyan", linewidth=0.7)
        axes[0].axvline(x1, color="orange", linewidth=0.7)

    axes[1].plot(vp_smooth, color="steelblue")
    axes[1].set_xlabel("Column (x)")
    axes[1].set_ylabel("Ink pixel count")
    axes[1].set_title("Vertical projection profile")
    for x0, x1 in word_bounds:
        axes[1].axvline(x0, color="cyan", linewidth=0.7, linestyle="--")
        axes[1].axvline(x1, color="orange", linewidth=0.7, linestyle="--")

    plt.tight_layout()
    if save_path:
        fig.savefig(str(save_path), dpi=150)
    plt.close(fig)


CHAR_CUT_COLOR = (0, 255, 255)  # cyan for character cuts
CHAR_CUT_THICKNESS = 1


def draw_character_cuts(
    img: np.ndarray,
    line_bounds: list[tuple[int, int]],
    words_per_line: list[list[tuple[int, int]]],
    chars_cuts_per_line: list[list[list[int]]],
    y_offset: int = 0,
    x_offset: int = 0,
) -> np.ndarray:
    """Draw cyan vertical lines at character cut positions."""
    canvas = img.copy()
    if len(canvas.shape) == 2:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    for line_idx, ((ly0, ly1), wl) in enumerate(zip(line_bounds, words_per_line)):
        if line_idx >= len(chars_cuts_per_line):
            continue
        for w_idx, (wx0, _wx1) in enumerate(wl):
            if w_idx >= len(chars_cuts_per_line[line_idx]):
                continue
            for cx in chars_cuts_per_line[line_idx][w_idx]:
                abs_x = wx0 + cx + x_offset
                cv2.line(
                    canvas,
                    (abs_x, ly0 + y_offset),
                    (abs_x, ly1 + y_offset),
                    CHAR_CUT_COLOR,
                    CHAR_CUT_THICKNESS,
                )
    return canvas


def annotate_with_characters(
    img: np.ndarray,
    line_bounds: list[tuple[int, int]],
    words_per_line: list[list[tuple[int, int]]],
    chars_cuts_per_line: list[list[list[int]]],
    y_offset: int = 0,
    x_offset: int = 0,
) -> np.ndarray:
    """Draw line boxes, word boxes, and character-cut lines."""
    canvas = annotate_image(img, line_bounds, words_per_line, y_offset, x_offset)
    canvas = draw_character_cuts(
        canvas, line_bounds, words_per_line,
        chars_cuts_per_line, y_offset, x_offset,
    )
    return canvas


def plot_character_scores(
    word_binary: np.ndarray,
    S: np.ndarray,
    S_smooth: np.ndarray,
    cuts: list[int],
    line_idx: int = 0,
    word_idx: int = 0,
    save_path: str | Path | None = None,
) -> None:
    """Plot fuzzy column scores and detected character cuts for one word."""
    fig, axes = plt.subplots(2, 1, figsize=(max(8, word_binary.shape[1] / 40), 5))

    axes[0].imshow(word_binary, cmap="gray")
    axes[0].set_title(f"Line {line_idx} – Word {word_idx}")
    for c in cuts:
        axes[0].axvline(c, color="red", linewidth=1, linestyle="--")

    axes[1].plot(S, color="lightsteelblue", alpha=0.6, label="Raw S")
    axes[1].plot(S_smooth, color="steelblue", label="Smoothed S")
    for c in cuts:
        axes[1].axvline(c, color="red", linewidth=0.7, linestyle="--")
    axes[1].set_xlabel("Column (x)")
    axes[1].set_ylabel("Fuzzy score")
    axes[1].set_title("Column possibility scores")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    if save_path:
        fig.savefig(str(save_path), dpi=150)
    plt.close(fig)


def plot_preprocess_stages(
    original: np.ndarray,
    gray: np.ndarray,
    binary: np.ndarray,
    save_path: str | Path | None = None,
) -> None:
    """Side-by-side plot: original → grayscale → binary."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original")
    axes[1].imshow(gray, cmap="gray")
    axes[1].set_title("Grayscale + denoised")
    axes[2].imshow(binary, cmap="gray")
    axes[2].set_title("Binary ink mask")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    if save_path:
        fig.savefig(str(save_path), dpi=150)
    plt.close(fig)
