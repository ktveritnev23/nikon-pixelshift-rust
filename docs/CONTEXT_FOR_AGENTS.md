# CONTEXT FOR IMPLEMENTATION AGENTS (read first)

## Project
Rust + egui application called **pixelshift**: merges Nikon Pixel Shift raw series
(Nikon Z6 III, 4 / 8 / 16 / 32 frames, CFA RGGB, 14-bit, 6048x4032 active area)
into a high resolution **linear RGB** image and writes it as a **linear DNG**
(default) or 16-bit TIFF, so that it can be processed in Lightroom/ACR or
Capture One.

Reference material (read as needed, do not modify):
- ../104NZ6_3, ../105NZ6_3, ../106NZ6_3, ../107NZ6_3 : 4/8/16/32 frame NEF series plus
  *_merged.NEFX (NX Studio result: linear RGB 14-bit, 6048x4032 for 4/8 frames,
  12096x8064 for 16/32 frames).
- docs/ARCHITECTURE.md : pipeline, geometry/algorithm spec, DNG recipe, CLI spec.
- docs/TASKS.md : work package boundaries and acceptance criteria.
- .scratch/measure/ : measured frame shifts (from another work stream).
- .scratch/research/ : format/algorithm research notes.

## Environment (macOS, Apple M4 Pro, 24 GB RAM)
- Rust 1.99, cargo. **Always export**:
    export CARGO_HOME=/Users/mihail/Pictures/experimental_merge/.cargo-home
    export CARGO_TARGET_DIR=<assigned target dir, see your task>
  Writing to ~/.cargo is blocked by the sandbox and the build fails with
  "Operation not permitted".
- Network access to crates.io works. Cargo.toml is frozen: if you need a new
  crate or a feature flag, report it to the dispatcher instead of editing it.
- LibRaw 0.22.2 (brew): headers in /opt/homebrew/include/libraw, pkg-config
  name "libraw_r". NOTE: the brew .pc file lists "-lstdc++", which does not exist
  on macOS. build.rs already filters it out (pkg_config with cargo_metadata(false)
  and manual link flags), so no LIBRARY_PATH workaround is needed. Command line tools usable for verification:
  unprocessed_raw, dcraw_emu, raw-identify, 4channels, tiffinfo, tiffdump, magick.
- python3 with numpy/scipy/PIL is available for quick analysis (do not install
  system packages; a venv inside the workspace is fine).
- The sandbox is workspace-write: /tmp is NOT writable (this includes /var/folders
  temp dirs). Use directories inside the project, e.g. .scratch/<your-wp>/.
- **Disk space is tight (~5 GB free).** Delete large intermediates immediately.
  A 96 MP 16-bit RGB image is 580 MB in memory, 146 MB as a 1x 16-bit RGB TIFF.

## Rules
- Stay inside your work package boundary (files listed in your task). Never edit
  another work package's files, src/types.rs, src/lib.rs, src/main.rs, build.rs
  or Cargo.toml. If a shared contract change is needed, report it and work
  around it locally.
- src/types.rs is the frozen data contract. Read it before writing code.
- Every public item you add must have a doc comment only where it explains
  something non obvious; do not write redundant comments.
- Verify claims with real command output, not assumptions. Tests must run
  against the real NEF files in this workspace where possible.
- Report at the end: what you built, the exact commands you ran, their results,
  and anything you could not do. Evidence beats prose.
