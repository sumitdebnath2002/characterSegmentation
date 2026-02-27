"""Image I/O utilities: loading, saving, and output folder management."""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np


def load_image(path: str | Path) -> np.ndarray:
    """Load an image from *path* and return it as a BGR NumPy array.

    Raises ``FileNotFoundError`` if the file does not exist and
    ``ValueError`` if OpenCV cannot decode it.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not decode image: {path}")
    return img


def save_image(img: np.ndarray, path: str | Path) -> None:
    """Save *img* to *path*, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def prepare_output_dirs(base: str | Path) -> dict[str, Path]:
    """Create and return a dict of output sub-directories under *base*.

    Keys: ``lines``, ``words``, ``debug``.
    """
    base = Path(base)
    dirs: dict[str, Path] = {}
    for name in ("lines", "words", "characters", "debug"):
        d = base / name
        d.mkdir(parents=True, exist_ok=True)
        dirs[name] = d
    return dirs
