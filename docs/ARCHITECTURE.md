# ARCHITECTURE

## Product
Single Rust crate, one binary `pixelshift`:
- `pixelshift` (no args) launches the egui GUI;
- `pixelshift merge <inputs...>` merges headless;
- `pixelshift inspect <inputs...>` prints metadata, detected mode, measured shifts;
- `pixelshift gui [--selftest]` explicit GUI entry, `--selftest` runs the GUI state
  machine (load -> align -> merge -> preview -> save) without a window and prints
  a report (used for acceptance in a headless environment).

Inputs may be a directory (series auto-detected) or explicit NEF files.

## Pipeline
1. **rawio** (WP1): decode every NEF with LibRaw through a tiny C shim compiled by
   build.rs. Output: `RawFrame` — active area only (6048x4032), camera native
   units, plus black/white, CFA, as-shot WB multipliers, `cam_xyz` (XYZ->camera)
   and `rgb_cam` (camera->sRGB D65), EXIF basics.
2. **shift** (WP2): estimate the sub-pixel rigid transform of each frame
   relative to the first frame: translation in sensor pixels (phase correlation
   for the integer part, gradient based refinement for the fraction) plus a
   small rotation about the image centre. Measured data show that the residual
   displacement is NOT constant over the field: on series 104 frame 1 dx varies
   from -0.69 px to -1.26 px across the frame, which fits a rotation of
   7..9e-5 rad (14..18 arc sec) with a ~0.05 px RMS residual. Dropping the
   rotation would blur the corners by ~0.6 px, i.e. > 1 px on a 2x grid, so the
   contract maps a sample through translation AND rotation (Shift::map in
   src/types.rs). Target accuracy: <= 0.02 px in the centre, <= 0.1 px at the
   corners.
3. **merge** (WP2): normalised per-colour splatting.
   - Output grid density `scale` (1 for modes 4/8, 2 for 16/32). Output pixel
     (X,Y) of a `scale = k` result corresponds to scene coordinate (X/k, Y/k) of
     the reference frame.
   - For every frame f and every raw sample (x,y) of colour c, the sample belongs
     to output position (k*(x + dx_f), k*(y + dy_f)) where (dx_f, dy_f) is the
     measured shift of frame f. Distribute the black-subtracted value to the four
     surrounding output pixels with bilinear weights into accumulator[c] and
     accumulator weight[c]. Finally out[c] = acc[c]/w[c].
   - Because shifts land on exact half-pixel positions in the ideal case, a
     correct measurement produces a delta kernel (no blur). Redundant positions
     of the 8/32 frame modes average automatically (that is the noise reduction).
   - Pixels with total weight below `min_coverage` are rebuilt by iterated
     neighbour averaging inside the same channel (`filled_pixels` statistic).
   - `denoise = false` repeats: use only the first occurrence of each position.
4. **write** (WP1): linear DNG (default) or 16-bit RGB TIFF; optional 2x -> 1x
   downscale (`merge::downsample_1x`).

## Why 4x pixel count, not 4x linear resolution
Mode 16 samples the scene on a half-pixel lattice in both axes, so the output
grid is 2x denser per axis (4x pixels). Real resolution gain over a single frame
is limited by the reconstruction kernel and by the ~1 px optical PSF; expect a
clear gain near the reference Nyquist frequency rather than a free 2x. Modes 4/8
keep the sensor grid and gain full colour resolution per pixel instead.

## Verified sampling coverage (checked numerically, tools/coverage_check.py)
- 4 frames, scale 1, offsets {0,1}x{0,1}: every interior output pixel receives
  exactly one R, one B and two G samples -> true RGB per pixel, no interpolation.
- 16 frames, scale 2, offsets {0,0.5,1,1.5}^2: every interior output pixel
  receives exactly one R, one B and two G samples as well (4 samples/pixel,
  no holes, no colour interpolation needed). The 2x result is a full colour
  image, not a mosaic.
- 8 / 32 frames (duplicated positions): 8 samples/pixel (R,R,G,G,G,G,B,B), i.e.
  the same coverage with a factor sqrt(2) noise reduction.
- Consequence for the algorithm: normalised splatting is sufficient; extra
  per-colour interpolation is only needed to fill pixels whose coverage is
  incomplete (real, imperfect shifts, image borders).

## Motion compensation (src/motion.rs)

Opt-in (`MergeParams::motion`, CLI `--motion [--motion-sensitivity F]`, GUI
checkbox + slider). After the rigid alignment, every non-reference frame is
resampled through its Shift onto the CFA phase invariant lattice (the same
`luma_plane` the alignment uses) with the six-tap Lanczos sampler; the
difference against the reference is accumulated per 32x32 sensor px block and
compared to a locally median block energy (7x7 block window). Motion occupies
a minority of blocks, so the median estimates the noise + interpolation floor;
a block above threshold x floor is moving. Isolated single blocks are dropped
as noise spikes, the mask is dilated by one block, and the reference frame is
never masked - fully masked regions fall back to its own (demosaiced) samples,
so a moving object stays sharp instead of averaging into ghosts.

Measured on the 109NZ6_3 experiment set (9 x 32 frames, Z6 III):
- tripod static series: masks 2-4 % of blocks, quality metrics unchanged
  within noise on one series; the other carries a bright clipped region where
  our merge differs from NX Studio with or without masks (global metrics on
  that scene swing several dB with the validation window placement).
- fan at 50 rpm: in the fan region the masked merge is ~25 % closer to the
  reference-time single frame than the plain merge, the background is bit
  identical to the plain merge, and the NX Studio reference ghosts with a 3-6x
  larger deviation from the single frame.
- handheld IBIS series: the rigid aligner itself fails (inter-frame shifts of
  5-22 px exceed its search design, residuals up to 9 px) - large-motion
  alignment is the prerequisite before motion compensation can help there.

The saturation guard treats blocks whose luma is within 2 % of the block-sum
maximum as no data: clipped blocks disagree nonlinearly and dwarf any
threshold.

## Output format decisions (current)

Layout (all DNG flavors): the main image is **IFD0** (`NewSubfileType = 0`), the
previews are following IFDs via the next pointer: IFD1 = 8-bit RGB (`1024` px long
side, `NewSubfileType = 1`), IFD2 = baseline JPEG (`1600` px long side,
`Compression = 7`, `PhotometricInterpretation = 6`). No SubIFDs are written.

Compression (`WriteOpts::compress`, CLI `--compression lossless`): the DNG main
image is written as lossless JPEG (`Compression = 7`, SOF3, predictor 1, point
transform 0) in ONE strip spanning the image; the entropy data is streamed to the
file row by row after a first pass collects the difference histogram (peak memory
stays at one row). The single strip is deliberate: LibRaw's
`lossless_dng_load_raw` follows multi-strip lossless JPEG through a tile-offset
walk that does not work for strip layouts, so a per-strip stream format would
decode only the first strip. Differences are modular (mod 2^16, category 16 is
the fixed -32768 sentinel without amplitude bits) and the first sample of every
line after the first predicts from the first sample of the line above - both
quirks of the Adobe/LibRaw readers, verified by bit-identical `dcraw_emu` decodes
of compressed and uncompressed files (tests/writers.rs). The TIFF writer stores
AdobeDeflate (`Compression = 8`) with the horizontal-differencing predictor
(`Predictor = 2`): per component, each pixel keeps its difference to the same
component of the previous pixel.
Why: a preview in IFD0 next to the raw image in a SubIFD made macOS ImageIO
(Preview/sips) treat the linear RGB data as a CFA mosaic and render coloured
blocks; with the main image in IFD0 the same file renders cleanly.

Tags of the main image: `BitsPerSample` 16, `SamplesPerPixel` 3 (1 for the cfa
flavor), `PlanarConfiguration` 1, `SampleFormat` 1, `Compression` 1 (16-bit Deflate
is not readable by LibRaw 0.22, so it is not the default), strips of 16 rows,
`BlackLevel` and `WhiteLevel` with **one value per sample** (a single `BlackLevel`
next to `SamplesPerPixel = 3` made rawler, the decoder of RapidRAW, panic),
`ColorMatrix1` (XYZ D50 -> camera native, LibRaw `cam_xyz`), `CalibrationIlluminant1`
= 21 (D65), `AsShotNeutral` = normalised (1/wb_r, 1/wb_g, 1/wb_b) with the pixels
stored WITHOUT white balance, `DefaultCropOrigin/Size`, `ActiveArea`,
`BaselineExposure`, `DNGVersion` 1.4.0.0, `DNGBackwardVersion` 1.1.0.0,
`UniqueCameraModel`, `Make`/`Model` verbatim from the NEF (`NIKON CORPORATION`,
`NIKON Z6_3`), EXIF basics, `Software`.

Flavors (`--dng-flavor`):
- `linearraw` (default): `PhotometricInterpretation = 34892`, 3 samples, full colour
  resolution, no demosaicing in the editor. Read by LibRaw, rawler, ImageMagick,
  ImageIO (with the IFD0 layout) and Adobe's DNG pipeline.
- `cfa`: `PhotometricInterpretation = 32803`, 1 sample, `CFARepeatPatternDim` 2x2,
  `CFAPattern` 0,1,1,2 (RGGB), `CFAPlaneColor`, `CFALayout` 1. The merged RGB is
  re-mosaiced back to a Bayer image so that every raw editor treats it as a normal
  camera raw and demosaics it itself. Universal compatibility, but the editor throws
  away two thirds of the colour samples again (luminance detail is kept).
- `rgb`: `PhotometricInterpretation = 2`, 16-bit RGB. Experimental: of the readers
  tested locally none accepts it (LibRaw, rawler, ImageIO and the ImageMagick LibRaw
  delegate all refuse).
- TIFF (`--format tiff`): 16-bit sRGB by default (`--tiff-render`: white balance,
  camera matrix, gamma), `--tiff-linear` for camera native linear data. Rendered
  TIFF is the universal fallback (Capture One, Preview, any editor).

Reconstruction: `MergeParams::deconvolve` (default 1.0) compensates the known blur
of the splat kernel with a separable [-k, 1+2k, -k] filter, k calibrated from the
measured kernel response at half the output Nyquist frequency, with a noise gate
(correction shrunk where the local high pass is comparable to noise) and a ringing
limiter (clamped to the local 3x3 range +/- max(2 % of that range, 0.25 % of the
white span)). It is skipped bit-exactly on an ideal lattice. `deconvolve = 0`
restores pure normalised splatting. Measured on series 106: high frequency energy
0.44 -> 0.57 of the NX Studio reference, gain over a single frame upscaled 2x
2.36 -> 3.18, flat-area noise +15 %.

The 5x5 residual correction grid (`ShiftGrid`, `estimate_series_with_grids`) is
implemented and tested on synthetic drift, but the merge leaves `params.grids`
empty: on the real series it improves the geometric residual while lowering the
detail metric (0.475 -> 0.400 at deconv 0), i.e. the corrected per-frame phases no
longer add up on the 2x grid.
## Verified sampling coverage (checked numerically, tools/coverage_check.py)
- 4 frames, scale 1, offsets {0,1}x{0,1}: every interior output pixel receives
  exactly one R, one B and two G samples -> true RGB per pixel, no interpolation.
- 16 frames, scale 2, offsets {0,0.5,1,1.5}^2: every interior output pixel
  receives exactly one R, one B and two G samples as well (4 samples/pixel,
  no holes, no colour interpolation needed). The 2x result is a full colour
  image, not a mosaic.
- 8 / 32 frames (duplicated positions): 8 samples/pixel (R,R,G,G,G,G,B,B), i.e.
  the same coverage with a factor sqrt(2) noise reduction.
- Consequence for the algorithm: normalised splatting is sufficient; extra
  per-colour interpolation is only needed to fill pixels whose coverage is
  incomplete (real, imperfect shifts, image borders).

## Output format decisions
- Stored data: camera native linear RGB, black subtracted, **white balance NOT
  applied**, range 0..(white - black) = 0..14303 for this camera.
- DNG (verified against DNG spec 1.7 and LibRaw 0.22 behaviour):
  IFD0 = 8-bit RGB preview (PhotometricInterpretation 2 is allowed for previews
  only) + metadata + `SubIFDs` -> main image IFD with PhotometricInterpretation
  34892 (LinearRaw; PI 2 is rejected by LibRaw for the main image),
  SamplesPerPixel 3, BitsPerSample 16, PlanarConfiguration 1, SampleFormat 1,
  Compression 1 by default, lossless JPEG (7) with `--compression lossless`
  (one strip, streamed; LibRaw does not walk a multi-strip LJ layout), strips of
  16 rows when uncompressed, WhiteLevel = white - black, BlackLevel = 0,
  DNGVersion 1.4.0.0, DNGBackwardVersion 1.1.0.0, UniqueCameraModel,
  ColorMatrix1 = LibRaw cam_xyz (XYZ D50 -> camera native), CalibrationIlluminant1
  = 21 (D65), AsShotNeutral = normalised (1/wb_r, 1/wb_g, 1/wb_b) with pixels
  stored WITHOUT white balance, DefaultCropOrigin/Size, ActiveArea,
  BaselineExposure, EXIF basics, Software.
  Size consequence: 12096x8064x3x2 B = 558 MiB per uncompressed DNG.
- TIFF: baseline 16-bit RGB, little-endian, strips; optionally AdobeDeflate + Predictor 2.
- Both writers must be able to stream row bands (peak memory well below 8 GB).

## CLI conventions
- Default output path: next to the first input file,
  `<first-stem>_pixelshift_<scale>x.dng` (e.g. `_DSC0534_pixelshift_2x.dng`).
  With `--also-1x` additionally `<first-stem>_pixelshift_1x.dng`.
- `--format dng|tiff`, `--scale auto|1|2` (auto: 4/8 frames -> 1, 16/32 -> 2),
  `--no-compress`, `--no-align`, `--threads N`, `--quiet`.
- Progress lines on stderr, summary on stdout, exit code 1 on error.
- `inspect` prints: camera, dimensions, CFA, black/white, WB, detected mode and
  suggested scale, per frame shifts (dx, dy, confidence), and the expected output
  size in megapixels and bytes.

## GUI (WP3)
- Drop / open a folder or a list of NEFs; series auto-detected (`rawio::find_series`).
- Left panel: series and frames with shifts, ISO/shutter; options panel (scale,
  format, compression, also-1x, alignment, threads).
- Central canvas: merged result preview with zoom/pan and an optional 1:1 crop;
  a "detail" toggle showing the reference frame's own crop for comparison.
- Bottom: progress bars (decoding / aligning / merging / writing) and statistics
  (frames used, samples per channel, filled pixels, elapsed).
- Save button opens a native file dialog (rfd); batch list to merge several
  series sequentially.
- UI strings in English; docs in Russian.

## Performance targets
- 16 frame merge at scale 2 on M4 Pro: <= 60 s wall clock, <= 6 GB peak RSS.
- 32 frame merge: <= 120 s. Decoding is parallelised with rayon, merge row bands
  with rayon, writing streams compressed strips.

## Known risks
- DNG colour metadata semantics (WB applied or not) — verified with LibRaw in
  tools/validate.py against the NX Studio NEFX reference.
- Sub-pixel accuracy of alignment on noisy frames; residual blur must stay
  below ~0.05 px.
- Memory: never materialise all frames plus the 96 MP result as f32; use f32
  accumulators per channel band only if needed, otherwise u32 sums plus weights.
## Measured properties of Nikon pixel shift bursts (independent measurement, all 60 frames)

Source: .scratch/measure/report.md + .scratch/measure/shifts.json (3x3 windows per frame,
1024x1024 each, phase correlation initialisation + Gauss-Newton refinement, validated on
synthetic shifts to 0.002 px; active area 6048x4032 at left=8/top=4, black 1008, CFA RGGB).

- The realised offsets are NOT on the ideal lattice: deviation from the nearest 0.5 px
  multiple is RMS 0.17..0.22 px, max 0.23..0.34 px (series 104/105/106/107), i.e. 20..100x
  the estimator noise and systematic (snapping to the grid always increases the residual).
  Nikon's mechanism is accurate to ~0.1..0.25 px, so a merge must interpolate rather than
  reassemble a perfect lattice.
- The motion is rigid but not a pure translation: a rotation about the image centre of
  1e-5..1.4e-3 rad (3..320 arc sec) is present, it grows within a burst (series 107:
  1.8e-4 -> 5.7e-4 rad over 32 frames) and dropping it raises the full-frame residual by
  40..56 % on series 106.
- There is drift inside a burst: series 107 moves from dy = -0.6 px to -4.5 px over 4 s.
- A rigid model leaves up to 1.5 px error at the corners of series 106/107, while the
  centre is fitted to ~0.05 px. A 3x3 residual correction grid (ShiftGrid) is therefore part
  of the contract: the merge adds a bilinearly interpolated correction to the rigid map.
- An isotropic scale term is not needed (it changes the residual by <= 2 % and makes 106
  worse; fitted scale-1 <= 3.5e-5). A full affine model halves the 106 residual (0.78 -> 0.42 px)
  but a local grid does better.
- The duplicate positions of the 8/32 frame modes are real (series 107 alternates dx ~0 /
  ~0.95 px with a two frame period), but the duplicates differ by 0.02..0.5 px because of drift,
  so they must still be aligned individually.