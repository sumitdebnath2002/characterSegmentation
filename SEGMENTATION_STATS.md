# Segmentation Pipeline: Stats & Version Evolution

## Test Image
- **File:** `data/samples/image.png`
- **Content:** 9 lines of handwritten Hindi (Devanagari) text
- **Ground truth:** 9 lines, ~39–43 words (exact count depends on punctuation handling)

---

## Detection Counts by Version

| Output Folder | Lines | Words | Characters | Notes |
|---------------|-------|-------|------------|-------|
| `test_run` | 5 | 18 | — | HPP missed lines; default thresholds too aggressive |
| `hindi_run` | 9 | 70 | — | Massive word over-segmentation |
| `slant_corrected` | 9 | 47 | — | Slant correction helped; words still over-segmented |
| `corrected_run` | 9 | 41 | — | Parameter tuning improved word count |
| `hindi_handwritten_tuned` | 9 | 41 | — | Smoothing/gap tuning |
| `handwritten_fixed2` | 9 | 43 | — | Headline attenuation added |
| `matra_fixed` | 9 | 43 | — | Padding added to avoid clipping matras |
| `char_test` | 9 | 43 | 187 | Character segmentation added; over-segmented chars |
| `cca_final` | 9 | 40 | 128 | CCA-based lines; cleaner crops |
| **`image_improved`** | **9** | **40** | **73** | + Improved char seg (all 4 fixes below) |

### hindi.png (printed Hindi, 5 lines)

| Output Folder | Lines | Words | Characters | Notes |
|---------------|-------|-------|------------|-------|
| `hindi_final` | 5 | 47 | 188 | CCA lines, old char seg (heavy over-seg) |
| **`hindi_improved2`** | **5** | **47** | **90** | + Improved char seg (all 4 fixes below) |

---

## Pipeline Version Evolution: Problems & Fixes

| Version | Problem Identified | Change Made | Impact |
|---------|-------------------|-------------|--------|
| **v1 (baseline HPP)** | Only 5/9 lines detected | Tuned `min_line_height`, `min_gap`, `smooth_window` | 9/9 lines detected |
| **v2 (word seg)** | 70 words (massive over-segmentation) | Added headline (shirorekha) attenuation before VPP | Reduced to ~43 words |
| **v3 (slant fix)** | Skewed text caused uneven profiles | Added deskew + slant correction in preprocessing | Cleaner projection profiles |
| **v4 (matra fix)** | Matras clipped at line boundaries | Added `pad_y` gap-aware padding | Fewer broken glyphs at line edges |
| **v5 (char seg)** | 187 chars (over-segmented) | Added Savitzky-Golay smoothing to fuzzy scores | Reduced to 128 chars |
| **v6 (CCA lines)** | Padding still cut matras; border CC noise inflated bounds | CCA-based line assignment + CC masking + noise filters | Clean crops, 40 words, 128 chars |
| **v7 (char seg v2)** | Shirorekha thin-spots → false cuts; tiny dot fragments; cuts slicing through CCs | Adaptive stroke thresholds + shirorekha gap check + CC-aware cuts + tiny-segment merge | image.png: 128→73 chars; hindi.png: 188→90 chars |

---

## How CCA Helps

### 1. Adaptive Line Bounds (no more guesswork)

- **Old:** Fixed `pad_y = 4px` on each side of the HPP core band.
- **New:** Each connected component (CC) is assigned to the line whose core-band center is closest to the CC's centroid. Line bounds expand to cover *all* assigned CCs.
- **Effect:** Matras extending 12+ px above/below the core band are no longer clipped.

### 2. CC Masking in Crops (no cross-line leakage)

- **Old:** When line bounds overlapped, neighboring ink leaked into the crop.
- **New:** Each pixel is checked against the label map; only pixels whose CC belongs to that line are kept.
- **Effect:** Overlapping line regions yield clean, non-contaminated crops.

### 3. Noise Filtering (no border/caption blowup)

- **Height filter:** CCs taller than 3× avg line height → discarded (image borders).
- **Width filter:** CCs wider than 90% of image width → discarded (horizontal borders).
- **Distance filter:** CCs whose centroid is >1.5× avg line height from nearest line → discarded (captions, noise).
- **Effect:** Border rectangles and caption text no longer inflate line bounds.

### 4. Cascading Effect on Word & Character Segmentation

- Cleaner line crops → better vertical projection for words → fewer spurious splits.
- Complete glyphs (no clipped matras) → better fuzzy scoring for characters → fewer false boundaries.
- **Word count:** 43 → 40 (closer to ground truth ~39).
- **Character count:** 187 → 128 (less over-segmentation).

---

## Final Results

### image.png → `outputs/image_improved/`

| Stage | Count |
|-------|-------|
| Lines | 9 |
| Words | 40 |
| Characters | 73 |

### hindi.png → `outputs/hindi_improved2/`

| Stage | Count |
|-------|-------|
| Lines | 5 |
| Words | 47 |
| Characters | 90 |

### Output Structure (same for both)

| Output Path | Contents |
|-------------|----------|
| `annotated.png` | Full image with line (green), word (blue), character (yellow) boundaries |
| `lines/` | Line crop images |
| `words/` | Word crop images |
| `characters/` | Character crop images |
| `debug/` | Horizontal profile, vertical profiles, char scores, preprocessing stages |

---

## Character Segmentation v7: What Changed

### 1. Adaptive Stroke Thresholds
- **Old:** Fixed constants `_STROKE_THIN = 5`, `_STROKE_THICK = 15`.
- **New:** Median stroke width estimated via distance transform on the skeleton; thresholds derived as `0.8×` and `2.5×` that width.
- **Effect:** Adapts to both thin handwriting and thick printed text.

### 2. Shirorekha Gap Check
- **New:** Detects the headline (shirorekha) row band via horizontal projection peak. A cut is only valid if the headline has a gap (at least one background pixel) at that column.
- **Effect:** Prevents false cuts at thin spots *within* the shirorekha where body characters are joined.

### 3. CC-Aware Cut Filtering
- **New:** After peak detection, each candidate cut is checked against the connected-component label map. If a CC body *below* the shirorekha spans both sides of the cut column, the cut is rejected.
- **Effect:** Prevents cuts that would slice through a character body, especially for conjuncts and matras.

### 4. Tiny Segment Merge
- **New:** After assembly, any character segment narrower than 15% of the average segment width or with <30 ink pixels is merged into its nearest neighbor.
- **Effect:** Eliminates dot/bindu fragments and shirorekha slivers that were being output as standalone "characters".
