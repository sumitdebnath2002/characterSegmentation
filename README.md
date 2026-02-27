# Hindi / Bangla Handwritten Text Segmentation

Line, word, and character segmentation of handwritten Devanagari (Hindi)
and Bangla text using projection profiling and fuzzy feature-based
character segmentation.


## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.9+.


## Usage

```bash
PYTHONPATH=src python3 -m segmentation \
    --input data/samples/image.png \
    --output outputs/run1 \
    --debug
```

### Key flags

| Flag | Default | Description |
|---|---|---|
| `--input` / `-i` | *(required)* | Path to input image (PNG, JPG, etc.) |
| `--output` / `-o` | `outputs/run` | Output directory |
| `--binarize` | `otsu` | `otsu` or `adaptive` |
| `--no-deskew` | off | Disable automatic skew correction |
| `--no-slant` | off | Disable automatic slant correction |
| `--no-headline` | off | Disable shirorekha/matra attenuation |
| `--no-char-seg` | off | Disable character segmentation |
| `--debug` | off | Save debug plots (profiles, preprocessing stages, fuzzy scores) |
| `--line-smooth` | 5 | Horizontal profile smoothing window |
| `--word-smooth` | 9 | Vertical profile smoothing window |
| `--sg-window` | 7 | Savitzky-Golay window for character score smoothing (must be odd) |
| `--min-line-height` | 10 | Discard lines shorter than N pixels |
| `--min-word-width` | 15 | Discard words narrower than N pixels |
| `--min-line-gap` | 3 | Gaps smaller than N rows are merged |
| `--min-word-gap` | 10 | Gaps smaller than N columns are merged |

Tip: defaults are tuned for handwritten text. For printed/typeset text,
try smaller values: `--word-smooth 5 --min-word-gap 4 --min-word-width 8`.


## Output structure

```
outputs/run1/
  annotated.png     # full image with line (green), word (blue), character cut (cyan) overlays
  lines/            # one PNG per detected text line
  words/            # one PNG per detected word  (line_NNN_word_NNN.png)
  characters/       # one PNG per segmented character (line_NNN_word_NNN_char_NNN.png)
  debug/            # projection profiles, preprocessing stages, per-word fuzzy score plots
```


## Full Pipeline Flow

The pipeline takes a raw document image as input and produces individually
cropped character images as final output. Every intermediate result (lines,
words) is also saved. The five major stages are described below.


### Stage 1 -- Preprocessing

Source: `src/segmentation/preprocess.py`

1. **Grayscale conversion.** The input BGR image is converted to a
   single-channel grayscale image.
2. **Denoising.** A median blur with kernel size 3 removes salt-and-pepper
   noise.
3. **Binarization.** The grayscale image is converted to a binary ink mask
   (ink = 255, background = 0) using either Otsu global thresholding or
   adaptive Gaussian thresholding, selectable via `--binarize`.
4. **Small noise removal.** Connected components with an area smaller than
   30 pixels are removed as isolated specks.
5. **Deskew.** Whole-page rotation is estimated by sweeping rotation angles
   in the range -10 to +10 degrees and selecting the angle that maximises
   the variance of the horizontal projection profile. A coarse sweep (1
   degree steps) is followed by a fine sweep (0.05 degree steps) around the
   best candidate.
6. **Slant correction.** Character lean is estimated by suppressing
   horizontal strokes (shirorekha) via morphological opening and then
   sweeping horizontal shear angles. The angle that maximises the sharpness
   (sum of absolute first-differences) of the vertical projection profile
   is chosen. A two-pass coarse/fine sweep is used.
7. **Margin cropping.** The binary image is cropped to the tight bounding
   box of ink with a 10-pixel pad on each side.


### Stage 2 -- Line Segmentation

Source: `src/segmentation/lines.py`

1. Compute the horizontal projection profile H(y): the row-wise sum of ink
   pixels.
2. Smooth the profile with a uniform box filter (configurable window).
3. Threshold the profile (default: 10 percent of the peak value) to produce
   a binary text/gap mask.
4. Extract contiguous "text" runs. Small gaps shorter than `--min-line-gap`
   are bridged; runs shorter than `--min-line-height` are discarded.
5. Each surviving run defines a line boundary (y_start, y_end). The binary
   image is cropped at these boundaries to produce individual line images
   saved to `lines/`.


### Stage 3 -- Word Segmentation

Source: `src/segmentation/words.py`, `src/segmentation/headline.py`

For each line image:

1. **Headline (shirorekha) attenuation.** The horizontal headline that
   connects characters in Devanagari and Bangla is detected by finding the
   row with peak ink density in the upper 45 percent of the line. A band
   around this peak is identified, and long horizontal strokes within it are
   removed via morphological opening with a wide horizontal kernel. This
   prevents the headline from bridging inter-word gaps in the vertical
   profile.
2. **Vertical projection profiling.** Compute the column-wise sum of ink
   pixels on the headline-attenuated image. Smooth and threshold as in line
   segmentation, using `--word-smooth` and `--min-word-gap`.
3. **Connected component expansion.** Each initial word box is expanded to
   include every connected component in the original (non-attenuated) line
   image that overlaps with it. This captures matras and modifiers (such as
   the short-i matra, long-i matra, u-matra, uu-matra) whose pixels extend
   beyond the vertical-projection boundaries. Overlapping boxes are merged.
4. The line image is cropped at the resulting word boundaries. Word images
   are saved to `words/`.


### Stage 4 -- Character Segmentation

Source: `src/segmentation/characters.py`

For each word image, individual characters are segmented using a
three-phase fuzzy algorithm. The word image is expected to be binary
(ink = 255, background = 0).

#### Phase 1 -- Separation of region above headline

Some matras and modifiers in Devanagari and Bangla sit above the headline
and span across character boundaries. These are temporarily separated
before computing per-column scores so they do not interfere with
segmentation.

1. **Row transition counting.** For each row, count the number of 0-to-1
   (background-to-ink) transitions.
2. **Band analysis.** Divide the image height into four equal horizontal
   bands. Compute the average transition count for each band (T1 through
   T4).
3. **Detection.** If T1 (top band) is significantly lower than the maximum
   of T2, T3, T4 -- specifically, T1 < 0.6 * max(T2, T3, T4) -- then a
   sparse region above the headline is deemed to exist.
4. **Skeleton-based separation.** If a region is detected:
   a. Skeletonize the binary image using morphological thinning.
   b. Locate the topmost black pixel in the skeleton.
   c. Traverse the skeleton downward from that pixel, following connected
      neighbours and preferring downward movement.
   d. At each pixel, count black pixels in a 3x3 window. If the count
      exceeds 3, a junction is detected (the point where the top component
      meets the main body).
   e. Cut the connection at the junction pixel. Find connected components
      in the resulting image. The component above the junction is the
      candidate top component.
   f. Validate: the component height must be less than H/2 and the
      junction row must be in the upper half of the image. If valid, store
      the component and its x-coordinate range; remove it from the working
      image.
   g. Repeat from step 1 until no above-headline region is detected (up to
      10 iterations).
5. **Fallback.** If skeleton traversal finds no junction, a connected
   component analysis is used to check if the topmost component is small
   and entirely within the upper half. If so, it is separated directly.

The output of Phase 1 is a modified binary image (top components removed)
and a list of separated components with their x-ranges.

#### Phase 2 -- Fuzzy column scoring

For each column j of the modified word image, four features are computed
and combined into a single possibility score:

1. **Feature X1 -- first black pixel location.** Scan downward from the top
   of the column to find the first ink pixel at row p1.
   Fuzzy score: mu1 = 1 - (p1 / H), clamped to [0, 1].
   Columns where ink starts near the top (at the headline) score high.

2. **Feature X2 -- thickness of the first stroke.** From p1, scan downward
   to the first white pixel at row p2. The stroke thickness is X2 = p2 - p1.
   Fuzzy score: mu2 = 1 if X2 <= 5; mu2 = 0 if X2 >= 15; linearly
   interpolated in between.
   Columns with thin strokes (just the headline) score high; columns
   passing through the body of a character score low.

3. **Feature X3 -- white pixel count below the stroke.** Count white pixels
   from p2 to the bottom of the column.
   Fuzzy score: mu3 = X3 / H.
   More white space below the headline means a better cut candidate.

4. **Feature X4 -- vertical white run sum.** Scan from p2 to the bottom.
   For each continuous run of white pixels with length n, accumulate
   n * (n + 1) / 2. This triangular-number weighting gives disproportionate
   credit to long uninterrupted white runs.
   Fuzzy score: mu4 = X4 / (H * (H + 1) / 2).

The column possibility score is: S[j] = mu1 * mu2 * mu3 * mu4.

Columns in the gaps between characters -- where there is a thin headline
and large white space below -- receive high scores. Columns passing through
the body of a character receive scores near zero.

#### Phase 3 -- Smoothing and cut detection

1. **Smoothing.** Apply a Savitzky-Golay filter (polynomial degree 1,
   configurable window size via `--sg-window`) to the raw score array. This
   removes high-frequency noise while preserving the positions of true
   peaks.
2. **Peak detection.** Identify local maxima in the smoothed score array:
   columns where S_smooth[j] is greater than both its immediate neighbours.
3. **False peak verification.** For each pair of adjacent peak candidates,
   extract the image segment between them and measure the vertical extent
   of ink in that segment. If the ink height is less than H/2, the segment
   is too small to be a valid character; the peak with the lower raw S
   value is removed. This process repeats until no more false peaks remain.

#### Final assembly

1. The word image is cut at the verified peak positions to produce
   character segments.
2. For each segment, any above-headline components (separated in Phase 1)
   whose x-range overlaps with the segment are pasted back. This restores
   matras and modifiers to the correct character.
3. Each character image is cropped to its tight bounding box with a
   2-pixel pad and saved to `characters/`.


### Stage 5 -- Output and Visualisation

Source: `src/segmentation/visualize.py`, `src/segmentation/io.py`

- **annotated.png** shows the full binary image with green rectangles for
  line boundaries, blue rectangles for word boundaries, and cyan vertical
  lines at character cut positions.
- When `--debug` is set, additional plots are saved:
  - `preprocess_stages.png` -- side-by-side original, grayscale, and binary
    images.
  - `horizontal_profile.png` -- the horizontal projection profile with line
    boundaries marked.
  - `vertical_profile_line_NNN.png` -- vertical projection profiles for
    each line with word boundaries marked.
  - `char_scores_LNNN_WNNN.png` -- per-word fuzzy score plots showing the
    raw column scores, the Savitzky-Golay smoothed scores, and the final
    character cut positions.


## Project Structure

```
src/segmentation/
  __init__.py       # package marker
  __main__.py       # entry point for python -m segmentation
  cli.py            # argument parsing and pipeline orchestration
  preprocess.py     # grayscale, binarize, denoise, deskew, slant correction
  lines.py          # line segmentation via horizontal projection profiling
  headline.py       # headline (shirorekha) detection and attenuation
  words.py          # word segmentation via vertical projection profiling
  characters.py     # character segmentation (fuzzy scoring, skeleton analysis)
  profiles.py       # 1-D projection profile utilities (compute, smooth, threshold)
  visualize.py      # annotated images and debug plots
  io.py             # image loading, saving, output directory management
```


## Dependencies

- opencv-python
- numpy
- matplotlib
- scikit-image (skeletonization in character segmentation)
- scipy (Savitzky-Golay filter, uniform filter)
- tqdm


## Limitations

- Severe page skew (beyond about 7 degrees) may not be fully corrected.
- Touching or overlapping lines will be merged into a single line.
- Very cursive writing with no clear inter-word gaps can under-segment
  words.
- Heavy background noise or coloured paper may require adaptive
  binarization (`--binarize adaptive`).
- Character segmentation assumes the presence of a headline (shirorekha).
  Scripts without a headline may not segment correctly.
- Very small or heavily connected characters may be merged into a single
  segment.
