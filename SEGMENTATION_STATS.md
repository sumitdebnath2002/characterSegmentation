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
| **`cca_final`** | **9** | **40** | **128** | CCA-based lines; cleaner crops; best overall |

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

## Final Results (`outputs/cca_final/`)

| Stage | Count |
|-------|-------|
| Lines | 9 |
| Words | 40 |
| Characters | 128 |

| Output Path | Contents |
|-------------|----------|
| `annotated.png` | Full image with line (green), word (blue), character (yellow) boundaries |
| `lines/` | 9 line crop images |
| `words/` | 40 word crop images |
| `characters/` | 128 character crop images |
| `debug/` | Horizontal profile, vertical profiles, char scores, preprocessing stages |
