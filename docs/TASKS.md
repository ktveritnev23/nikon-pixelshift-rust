# WORK PACKAGES

## WP1 — raw I/O and output writers
Files: src/rawio/**, src/write/**, shim/**, tests/rawio.rs, tests/writers.rs.
Acceptance:
- `rawio::open` on 104NZ6_3/_DSC0522.NEF yields width=6048 height=4032,
  cfa pattern "RGGB", black=1008, white=16383 or 15311 as LibRaw reports,
  WB ~[1.244, 1.0, 2.010], sane cam_xyz/rgb_cam, EXIF ISO/shutter from the file.
- The whole 4/8/16/32 series decode in parallel; identical metadata across frames.
- `rawio::find_series` detects 4 series with modes 4, 8, 16, 32 in the workspace.
- `write_dng` produces a file that `raw-identify` recognises as a raw image with
  the expected dimensions and that `dcraw_emu -T -4 -o 0 <file>` decodes without
  error; `tiffinfo` shows the expected tags; the decoded pixel statistics match
  the in-memory image (compare means/percentiles).
- `write_tiff16` produces a file that ImageMagick/PIL read with correct geometry
  and identical pixel values.
Evidence: command lines + outputs, test names and pass/fail.

## WP2 — alignment and merge core
Files: src/shift.rs, src/merge.rs, tests/alignment.rs, tests/merge_synthetic.rs.
Acceptance:
- Synthetic ground truth test: render a known RGB image as 4 (or 16) Bayer frames
  with exact half/integer pixel shifts and no noise; merge must reconstruct the
  original colour image at the right scale with PSNR > 45 dB (4 frames, scale 1)
  and > 40 dB (16 frames, scale 2) inside the valid area.
- Shift estimation on those synthetic frames recovers the injected shifts to
  within 0.02 px, including 0.5 px cases, with and without additive noise.
- Real data: for 104NZ6_3 (4 frames) shifts must be close to integer values, for
  106NZ6_3 (16 frames) close to multiples of 0.5 px; cross-check with
  .scratch/measure/shifts.json.
- Merge of a real series runs within the performance targets and produces no
  visible holes (filled_pixels share < 0.1 %); `preview` and `preview_crop`
  produce sane sRGB images (non-flat, no NaN, correct aspect).
Evidence: test output, timing, statistics, and a short note on residual blur.

## WP3 — CLI and GUI
Files: src/cli.rs, src/app/**, tests/cli.rs.
Acceptance:
- `pixelshift --help`, `inspect` and `merge` work as specified in
  docs/ARCHITECTURE.md; `merge` on a real series writes a DNG and prints a summary.
- `pixelshift gui --selftest` exercises load/align/merge/preview/save and prints
  timings without opening a window; `pixelshift` launches the real eframe window
  (verify it starts and stays alive for a few seconds).
- A fabricated MergeResult drives the preview widgets (unit test), so the GUI can
  be validated before WP1/WP2 exist.
Evidence: command lines + outputs, screenshots if possible.

## WP4 — validation and documentation (dispatcher)
tools/validate.py, docs/VERIFICATION.md, README.md (Russian), release build,
comparison against the NX Studio NEFX reference for all four series.
## Status (dispatcher log)

- WP1 (raw I/O + writers): DONE. Evidence: tests/rawio.rs 4/4, tests/writers.rs 4/4
  (+1 ignored 96 MP streaming test that passed on its own), raw-identify/tiffinfo/
  dcraw_emu round trips, peak RSS of the DNG writer ~595 MB for a 585 MB file.
  Two blockers fixed by the dispatcher: build.rs (brew .pc links -lstdc++, which
  does not exist on macOS) and WriteOpts::default().compress = false.
- WP3 (CLI + GUI): DONE. Evidence: 11 CLI tests, GUI state machine selftest on real
  data, window liveness check, real DNG merges of series 104 and 106.
- WP2 (alignment + merge core): IN PROGRESS. Synthetic round trip tests pass, real
  data quality is not acceptable yet: with translation-only alignment the 2x result
  has gain_vs_single = 1.09 (a 2x upscale of a single frame has laplacian energy
  360, our merge 391, the NX reference 1816), filled share 12..72 %. A rigid model
  (translation + rotation) is being implemented; the first attempt produced a
  degenerate fit (rot up to -318 arcsec and a constant 0.272 px residual).
- Independent shift measurement (fact finding subagent): still running; interim
  result for series 104 frame 1: dx -0.69..-1.26 px across the field, dy -0.07..0.31,
  i.e. a real rotation of ~7..9e-5 rad (14..18 arcsec) with a ~0.05 px residual.
- Dispatcher verification assets: tools/validate.py (DNG vs NX reference: alignment,
  per-channel gain/correlation/PSNR, detail laplacian ratio), tools/resolution.py
  (single frame 2x upscale vs merge vs NX reference), tools/coverage_check.py and
  tools/coverage_measured.py (sampling coverage of a pixel shift pattern),
  tools/pattern_probe2.py (sub-pixel shift probe on the colour invariant 2x2 box
  projection, self tested to ~0.05 px).
## Final status

- WP1 DONE, WP2 DONE, WP3 DONE, WP4 (validation + docs) DONE by the dispatcher.
- cargo test: lib 8, alignment 8, cli 11 (+1 ignored), merge_synthetic 10 (+2 ignored),
  rawio 4, writers 4 (+1 ignored) -> all green.
- Full acceptance run (tools/accept.sh): see docs/VERIFICATION.md. Merge times 1.5-6.5 s per
  series, peak RSS 0.94-3.74 GB, DNG validated by raw-identify/tiffinfo/dcraw_emu.
- Known limitation, measured: the 16 frame series 106 has a large residual rotation
  (-332 arc sec) and intra-burst drift, so its 2x output reaches 0.44 of the NX Studio
  detail energy (86 % of it for series 107). Closing that gap needs reconstruction-kernel
  compensation (deconvolution), which was rejected for the first release: it would trade
  faithful linear data for a sharpened look that the raw editor can apply anyway.
- ShiftGrid (local residual correction) is implemented and tested on synthetic drift but
  is not wired into the CLI: on the real series its gain is below the noise of the anchor
  estimate (detail 0.4431 -> 0.4424), so enabling it would only add code paths.
