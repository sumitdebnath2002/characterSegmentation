"""Command-line interface for Hindi/Bangla text segmentation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .io import load_image, save_image, prepare_output_dirs
from .preprocess import preprocess, to_grayscale, denoise
from .lines import detect_lines, crop_lines
from .words import detect_words, crop_words
from .characters import segment_characters
from .visualize import (
    annotate_image,
    annotate_with_characters,
    plot_horizontal_profile,
    plot_vertical_profile,
    plot_character_scores,
    plot_preprocess_stages,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="segmentation",
        description="Segment handwritten Hindi/Bangla text into lines and words "
        "using projection profiling.",
    )
    p.add_argument("--input", "-i", required=True, help="Path to input JPG image.")
    p.add_argument(
        "--output", "-o", default="outputs/run",
        help="Base directory for outputs (default: outputs/run).",
    )
    p.add_argument(
        "--binarize", choices=["otsu", "adaptive"], default="otsu",
        help="Binarization method (default: otsu).",
    )
    p.add_argument(
        "--no-deskew", action="store_true",
        help="Disable automatic deskew.",
    )
    p.add_argument(
        "--no-slant", action="store_true",
        help="Disable automatic slant correction.",
    )
    p.add_argument(
        "--no-headline", action="store_true",
        help="Disable headline (shirorekha) attenuation before word segmentation.",
    )
    p.add_argument(
        "--no-rule-removal", action="store_true",
        help="Disable automatic ruled notebook line removal.",
    )
    p.add_argument(
        "--debug", action="store_true",
        help="Save debug visualizations (projection plots, preprocessing stages).",
    )
    p.add_argument(
        "--line-smooth", type=int, default=5,
        help="Smoothing window for horizontal profile (default: 5).",
    )
    p.add_argument(
        "--word-smooth", type=int, default=9,
        help="Smoothing window for vertical profile (default: 9).",
    )
    p.add_argument(
        "--min-line-height", type=int, default=10,
        help="Minimum line height in pixels (default: 10).",
    )
    p.add_argument(
        "--min-word-width", type=int, default=15,
        help="Minimum word width in pixels (default: 15).",
    )
    p.add_argument(
        "--min-line-gap", type=int, default=3,
        help="Minimum gap between lines to not merge (default: 3).",
    )
    p.add_argument(
        "--min-word-gap", type=int, default=10,
        help="Minimum gap between words to not merge (default: 10).",
    )
    p.add_argument(
        "--no-char-seg", action="store_true",
        help="Disable character segmentation.",
    )
    p.add_argument(
        "--sg-window", type=int, default=7,
        help="Savitzky-Golay window for character score smoothing (default: 7).",
    )
    p.add_argument(
        "--min-char-width-ratio", type=float, default=0.4,
        help="Minimum segment width as a fraction of estimated character width. "
        "Cuts producing narrower segments are removed (default: 0.4).",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    input_path = Path(args.input)
    dirs = prepare_output_dirs(args.output)

    # ── Load ─────────────────────────────────────────────────────────────
    print(f"Loading {input_path} ...")
    img = load_image(input_path)

    # ── Preprocess ───────────────────────────────────────────────────────
    print("Preprocessing ...")
    binary, gray, y_off, x_off = preprocess(
        img,
        binarize_method=args.binarize,
        do_deskew=not args.no_deskew,
        do_slant_correct=not args.no_slant,
        do_remove_rules=not args.no_rule_removal,
    )

    if args.debug:
        plot_preprocess_stages(
            img, gray, binary,
            save_path=dirs["debug"] / "preprocess_stages.png",
        )

    # ── Line segmentation ────────────────────────────────────────────────
    print("Detecting lines (horizontal projection + CCA) ...")
    line_bounds, cc_labels, line_ccs = detect_lines(
        binary,
        smooth_window=args.line_smooth,
        min_line_height=args.min_line_height,
        min_gap=args.min_line_gap,
    )
    print(f"  Found {len(line_bounds)} lines.")

    line_crops_bin = crop_lines(binary, line_bounds, cc_labels, line_ccs)

    for idx, crop in enumerate(line_crops_bin):
        save_image(crop, dirs["lines"] / f"line_{idx:03d}.png")

    if args.debug:
        plot_horizontal_profile(
            binary, line_bounds,
            save_path=dirs["debug"] / "horizontal_profile.png",
            smooth_window=args.line_smooth,
        )

    # ── Word segmentation ────────────────────────────────────────────────
    print("Detecting words (vertical projection per line) ...")
    words_per_line: list[list[tuple[int, int]]] = []
    all_word_crops: list[list[np.ndarray]] = []
    word_count = 0

    for line_idx, line_bin in enumerate(tqdm(line_crops_bin, desc="  Lines")):
        wb = detect_words(
            line_bin,
            smooth_window=args.word_smooth,
            min_word_width=args.min_word_width,
            min_gap=args.min_word_gap,
            remove_headline=not args.no_headline,
        )
        words_per_line.append(wb)

        word_crops = crop_words(line_bin, wb)
        all_word_crops.append(word_crops)
        for w_idx, wc in enumerate(word_crops):
            save_image(wc, dirs["words"] / f"line_{line_idx:03d}_word_{w_idx:03d}.png")
        word_count += len(wb)

        if args.debug and line_idx < 5:
            plot_vertical_profile(
                line_bin, wb, line_idx=line_idx,
                save_path=dirs["debug"] / f"vertical_profile_line_{line_idx:03d}.png",
                smooth_window=args.word_smooth,
            )

    print(f"  Found {word_count} words across {len(line_bounds)} lines.")

    # ── Character segmentation ───────────────────────────────────────────
    chars_cuts_per_line: list[list[list[int]]] = []

    if not args.no_char_seg:
        print("Segmenting characters (fuzzy scoring per word) ...")
        char_count = 0

        for line_idx, word_crops in enumerate(
            tqdm(all_word_crops, desc="  Lines"),
        ):
            line_cuts: list[list[int]] = []
            for w_idx, wc in enumerate(word_crops):
                chars, dbg = segment_characters(
                    wc,
                    sg_window=args.sg_window,
                    min_char_width_ratio=args.min_char_width_ratio,
                    return_debug=True,
                )
                line_cuts.append(dbg["cuts"])

                for c_idx, ch in enumerate(chars):
                    save_image(
                        ch,
                        dirs["characters"]
                        / f"line_{line_idx:03d}_word_{w_idx:03d}_char_{c_idx:03d}.png",
                    )
                char_count += len(chars)

                if args.debug and line_idx < 3 and w_idx < 5:
                    plot_character_scores(
                        wc,
                        dbg["S"],
                        dbg["S_smooth"],
                        dbg["cuts"],
                        line_idx=line_idx,
                        word_idx=w_idx,
                        save_path=dirs["debug"]
                        / f"char_scores_L{line_idx:03d}_W{w_idx:03d}.png",
                    )

            chars_cuts_per_line.append(line_cuts)

        print(f"  Segmented {char_count} characters from {word_count} words.")

    # ── Annotated output ─────────────────────────────────────────────────
    print("Saving annotated image ...")
    if chars_cuts_per_line:
        annotated = annotate_with_characters(
            binary, line_bounds, words_per_line, chars_cuts_per_line,
            y_offset=0, x_offset=0,
        )
    else:
        annotated = annotate_image(
            binary, line_bounds, words_per_line,
            y_offset=0, x_offset=0,
        )
    save_image(annotated, Path(args.output) / "annotated.png")

    print(f"Done. Results saved to {args.output}/")


if __name__ == "__main__":
    main()
