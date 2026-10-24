//! CLI acceptance tests: argument parsing, default output names, --help and the
//! headless GUI selftest.

use clap::Parser;
use pixelshift::app::model::{Compression, OutFormat, ScaleMode};
use pixelshift::app::pipeline;
use pixelshift::cli::{Cli, Command, CompressionArg, FlavorArg, FormatArg, ScaleArg};
use pixelshift::types::MergeParams;
use pixelshift::write::DngFlavor;
use std::path::PathBuf;
use std::process::{Command as Process, Output};

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_pixelshift")
}

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn run(args: &[&str]) -> Output {
    Process::new(bin())
        .args(args)
        .current_dir(root())
        .output()
        .expect("run pixelshift")
}

fn stdout(output: &Output) -> String {
    String::from_utf8_lossy(&output.stdout).into_owned()
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

#[test]
fn help_lists_every_subcommand() {
    let output = run(&["--help"]);
    assert!(output.status.success(), "{}", stderr(&output));
    let text = stdout(&output);
    for needle in ["merge", "inspect", "gui"] {
        assert!(text.contains(needle), "--help misses {needle}:\n{text}");
    }
    assert!(text.contains("Pixel Shift"), "{}", text);
}

#[test]
fn merge_help_documents_the_flags() {
    let output = run(&["merge", "--help"]);
    assert!(output.status.success(), "{}", stderr(&output));
    let text = stdout(&output);
    for needle in [
        "--out",
        "--scale",
        "--format",
        "--dng-flavor",
        "--tiff-render",
        "--tiff-linear",
        "--deconvolve",
        "--distort",
        "--compression",
        "--also-1x",
        "--no-align",
        "--threads",
        "--quiet",
    ] {
        assert!(text.contains(needle), "merge --help misses {needle}:\n{text}");
    }
}

#[test]
fn merge_defaults_are_headless_friendly() {
    let cli = Cli::try_parse_from(["pixelshift", "merge", "series"]).expect("parse");
    let Command::Merge(args) = cli.command else {
        panic!("expected the merge subcommand");
    };
    assert_eq!(args.inputs, vec![PathBuf::from("series")]);
    assert_eq!(args.scale, ScaleArg::Auto);
    assert_eq!(args.format, FormatArg::Dng);
    assert_eq!(args.dng_flavor, FlavorArg::LinearRaw);
    assert_eq!(args.compression, CompressionArg::None);
    assert!(!args.tiff_render && !args.tiff_linear);
    assert_eq!(args.deconvolve, None);
    assert!(!args.also_1x && !args.no_align && !args.quiet && !args.no_compress);
    assert_eq!(args.threads, None);
    assert_eq!(args.out, None);
    let options = args.options();
    assert_eq!(options.scale, ScaleMode::Auto);
    assert_eq!(options.format, OutFormat::Dng);
    assert_eq!(options.compression, Compression::None);
    assert_eq!(options.flavor, DngFlavor::LinearRaw);
    assert!(options.tiff_render, "TIFF is rendered as sRGB by default");
    assert_eq!(options.deconvolve, MergeParams::default().deconvolve);
    assert!(options.align);
}

#[test]
fn merge_parses_every_flag() {
    let cli = Cli::try_parse_from([
        "pixelshift",
        "merge",
        "104NZ6_3",
        "extra.nef",
        "--out",
        "out/result.dng",
        "--scale",
        "2",
        "--format",
        "tiff",
        "--dng-flavor",
        "cfa",
        "--tiff-linear",
        "--deconvolve",
        "0.75",
        "--compression",
        "lossless",
        "--also-1x",
        "--no-align",
        "--threads",
        "6",
        "--quiet",
    ])
    .expect("parse");
    let Command::Merge(args) = cli.command else {
        panic!("expected the merge subcommand");
    };
    assert_eq!(
        args.inputs,
        vec![PathBuf::from("104NZ6_3"), PathBuf::from("extra.nef")]
    );
    assert_eq!(args.out, Some(PathBuf::from("out/result.dng")));
    assert_eq!(args.scale, ScaleArg::X2);
    assert_eq!(args.format, FormatArg::Tiff);
    assert_eq!(args.dng_flavor, FlavorArg::Cfa);
    assert!(args.tiff_linear);
    assert_eq!(args.deconvolve, Some(0.75));
    assert_eq!(args.compression, CompressionArg::Lossless);
    assert!(args.also_1x && args.no_align && args.quiet);
    assert_eq!(args.threads, Some(6));

    let options = args.options();
    assert_eq!(options.scale, ScaleMode::Two);
    assert_eq!(options.format, OutFormat::Tiff);
    assert_eq!(options.compression, Compression::Lossless);
    assert_eq!(options.flavor, DngFlavor::Cfa);
    assert!(!options.tiff_render, "--tiff-linear disables the render");
    assert_eq!(options.deconvolve, 0.75);
    assert_eq!(
        options.merge_params(1, Vec::new()).deconvolve,
        0.75,
        "--deconvolve must reach MergeParams"
    );
    assert!(!options.align);
    assert_eq!(options.threads, Some(6));
}

#[test]
fn dng_flavor_accepts_the_three_compositions() {
    for (text, expected) in [
        ("linearraw", FlavorArg::LinearRaw),
        ("rgb", FlavorArg::Rgb),
        ("cfa", FlavorArg::Cfa),
    ] {
        let cli = Cli::try_parse_from(["pixelshift", "merge", "in", "--dng-flavor", text])
            .unwrap_or_else(|error| panic!("--dng-flavor {text} was rejected: {error}"));
        let Command::Merge(args) = cli.command else {
            unreachable!()
        };
        assert_eq!(args.dng_flavor, expected);
        assert_eq!(args.options().flavor, expected.flavor());
    }
    assert!(Cli::try_parse_from(["pixelshift", "merge", "in", "--dng-flavor", "yuv"]).is_err());
}

#[test]
fn tiff_render_is_the_default_and_can_be_turned_off() {
    let cli = Cli::try_parse_from(["pixelshift", "merge", "in", "--format", "tiff", "--tiff-render"])
        .expect("parse");
    let Command::Merge(args) = cli.command else {
        unreachable!()
    };
    assert!(args.options().tiff_render);

    assert!(
        Cli::try_parse_from([
            "pixelshift", "merge", "in", "--tiff-render", "--tiff-linear"
        ])
        .is_err(),
        "--tiff-render and --tiff-linear are mutually exclusive"
    );
}

#[test]
fn deconvolve_is_limited_to_the_documented_range() {
    for text in ["0", "0.5", "1", "2"] {
        let cli = Cli::try_parse_from(["pixelshift", "merge", "in", "--deconvolve", text])
            .unwrap_or_else(|error| panic!("--deconvolve {text} was rejected: {error}"));
        let Command::Merge(args) = cli.command else {
            unreachable!()
        };
        assert_eq!(args.options().deconvolve, text.parse::<f32>().unwrap());
    }
    for text in ["-0.1", "2.5", "x"] {
        assert!(
            Cli::try_parse_from(["pixelshift", "merge", "in", "--deconvolve", text]).is_err(),
            "--deconvolve {text} must be rejected"
        );
    }
}

#[test]
fn distort_takes_k1_and_an_optional_k2() {
    let cli = Cli::try_parse_from(["pixelshift", "merge", "in"]).expect("parse");
    let Command::Merge(args) = cli.command else {
        unreachable!()
    };
    assert!(args.options().distort.is_identity(), "0 must be the default");

    let cli = Cli::try_parse_from(["pixelshift", "merge", "in", "--distort", "0.01"])
        .expect("parse k1");
    let Command::Merge(args) = cli.command else {
        unreachable!()
    };
    assert_eq!(args.options().distort, pixelshift::distort::Distortion::new(0.01, 0.0));

    let cli = Cli::try_parse_from(["pixelshift", "merge", "in", "--distort", "-0.002,0.0001"])
        .expect("parse k1,k2");
    let Command::Merge(args) = cli.command else {
        unreachable!()
    };
    assert_eq!(
        args.options().distort,
        pixelshift::distort::Distortion::new(-0.002, 0.0001)
    );

    let cli = Cli::try_parse_from(["pixelshift", "merge", "in", "--distort", "0.01,-0.002,0.0001"])
        .expect("parse k1,k2,k3");
    let Command::Merge(args) = cli.command else {
        unreachable!()
    };
    assert_eq!(
        args.options().distort,
        pixelshift::distort::Distortion::new3(0.01, -0.002, 0.0001)
    );

    for text in ["0.5", "x", "0.001,0.002,0.003,0.004", "0.07", "-0.07"] {
        assert!(
            Cli::try_parse_from(["pixelshift", "merge", "in", "--distort", text]).is_err(),
            "--distort {text} must be rejected"
        );
    }

    let cli = Cli::try_parse_from(["pixelshift", "inspect", "in", "--distort", "0.004"])
        .expect("parse inspect --distort");
    let Command::Inspect(args) = cli.command else {
        unreachable!()
    };
    assert_eq!(args.distort.k1, 0.004);
}

/// A 24 mm zoom needs about 3 % of barrel distortion, so the command line has
/// to accept it; anything past the limit must still be refused.
#[test]
fn distort_accepts_three_percent_of_barrel() {
    let cli = Cli::try_parse_from(["pixelshift", "merge", "in", "--distort", "0.03"])
        .expect("parse --distort 0.03");
    let Command::Merge(args) = cli.command else {
        unreachable!()
    };
    assert_eq!(
        args.options().distort,
        pixelshift::distort::Distortion::new(0.03, 0.0)
    );

    let cli = Cli::try_parse_from(["pixelshift", "merge", "in", "--distort", "-0.03"])
        .expect("parse --distort -0.03");
    let Command::Merge(args) = cli.command else {
        unreachable!()
    };
    assert_eq!(args.options().distort.k1, -0.03);

    let cli = Cli::try_parse_from(["pixelshift", "merge", "in", "--distort", "0.06"])
        .expect("the new limit itself must be accepted");
    let Command::Merge(args) = cli.command else {
        unreachable!()
    };
    assert_eq!(args.options().distort.k1, 0.06);
}

#[test]
fn scale_accepts_auto_1_and_2_only() {
    for (text, expected) in [
        ("auto", ScaleArg::Auto),
        ("1", ScaleArg::X1),
        ("2", ScaleArg::X2),
    ] {
        let cli = Cli::try_parse_from(["pixelshift", "merge", "in", "--scale", text])
            .unwrap_or_else(|error| panic!("--scale {text} was rejected: {error}"));
        let Command::Merge(args) = cli.command else {
            unreachable!()
        };
        assert_eq!(args.scale, expected);
    }
    assert!(Cli::try_parse_from(["pixelshift", "merge", "in", "--scale", "3"]).is_err());
    assert!(Cli::try_parse_from(["pixelshift", "merge"]).is_err());
}

#[test]
fn inspect_and_gui_parse() {
    let cli = Cli::try_parse_from(["pixelshift", "inspect", "104NZ6_3", "a.nef"]).expect("parse");
    let Command::Inspect(args) = cli.command else {
        panic!("expected the inspect subcommand");
    };
    assert_eq!(args.inputs.len(), 2);

    let cli = Cli::try_parse_from(["pixelshift", "gui", "--selftest", "--demo", "105NZ6_3"])
        .expect("parse");
    let Command::Gui(args) = cli.command else {
        panic!("expected the gui subcommand");
    };
    assert!(args.selftest && args.demo);
    assert_eq!(args.inputs, vec![PathBuf::from("105NZ6_3")]);

    let cli = Cli::try_parse_from(["pixelshift", "gui"]).expect("parse");
    let Command::Gui(args) = cli.command else {
        panic!("expected the gui subcommand");
    };
    assert!(!args.selftest && !args.demo && args.inputs.is_empty());
}

#[test]
fn default_output_names_follow_the_documented_pattern() {
    let input = PathBuf::from("/photos/104NZ6_3/_DSC0522.NEF");
    assert_eq!(
        pipeline::default_output_path(&input, 1, "dng"),
        PathBuf::from("/photos/104NZ6_3/_DSC0522_pixelshift_1x.dng")
    );
    assert_eq!(
        pipeline::default_output_path(&input, 2, "tiff"),
        PathBuf::from("/photos/104NZ6_3/_DSC0522_pixelshift_2x.tiff")
    );
    assert_eq!(
        pipeline::sibling_1x_path(&PathBuf::from("/photos/a/_DSC0522_pixelshift_2x.dng")),
        PathBuf::from("/photos/a/_DSC0522_pixelshift_1x.dng")
    );
    assert_eq!(
        pipeline::sibling_1x_path(&PathBuf::from("/photos/a/custom.dng")),
        PathBuf::from("/photos/a/custom_1x.dng")
    );
}

#[test]
fn series_detects_the_capture_mode_from_the_frame_count() {
    let files: Vec<PathBuf> = (0..8)
        .map(|i| PathBuf::from(format!("/x/_DSC{}.NEF", 520 + i)))
        .collect();
    let series = pipeline::series_from_files(&files).expect("series");
    assert_eq!(series.mode, 8);
    assert_eq!(series.suggested_scale, 1);
    assert_eq!(series.name(), "x");
    assert!(series.label().contains("mode 8"));
    assert!(!series.unexpected_mode());

    for (count, mode, scale) in [(4usize, 4u32, 1u32), (16, 16, 2), (32, 32, 2)] {
        let files: Vec<PathBuf> = (0..count)
            .map(|i| PathBuf::from(format!("/x/_DSC{}.NEF", 520 + i)))
            .collect();
        let series = pipeline::series_from_files(&files).expect("series");
        assert_eq!((series.mode, series.suggested_scale), (mode, scale));
    }

    let odd: Vec<PathBuf> = (0..7)
        .map(|i| PathBuf::from(format!("/x/_DSC{}.NEF", 520 + i)))
        .collect();
    let series = pipeline::series_from_files(&odd).expect("series");
    assert!(series.unexpected_mode());
    assert_eq!(series.suggested_scale, 1);
}

#[test]
fn expected_output_size_is_the_uncompressed_payload() {
    let one = pipeline::expected_output(6048, 4032, 1);
    assert_eq!((one.width, one.height), (6048, 4032));
    assert_eq!(one.bytes(), 6048 * 4032 * 3 * 2);
    assert!((one.mib() - 139.5).abs() < 0.5, "{}", one.mib());

    let two = pipeline::expected_output(6048, 4032, 2);
    assert_eq!((two.width, two.height), (12096, 8064));
    assert!((two.mib() - 558.0).abs() < 1.0, "{}", two.mib());
    assert!((two.megapixels() - 97.5).abs() < 0.1);
}

#[test]
fn gui_selftest_runs_without_a_window() {
    let output = run(&["gui", "--selftest"]);
    let text = stdout(&output);
    assert!(
        output.status.success(),
        "gui --selftest failed\nstdout:\n{text}\nstderr:\n{}",
        stderr(&output)
    );
    for needle in [
        "demo state machine",
        "load: frames decoded",
        "align: shifts estimated",
        "merge: result produced",
        "preview: full image",
        "zoom: lod decision",
        "preview: zoom -> detail crop",
        "preview: distortion k1 live",
        "save: file written",
        " ms",
        "PASS",
    ] {
        assert!(
            text.contains(needle),
            "selftest report misses {needle}:\n{text}"
        );
    }
    let saved = root().join(".scratch/wp3/selftest");
    assert!(saved.is_dir(), "selftest output directory missing");
}

#[test]
fn inspect_reports_missing_modules_without_panicking() {
    let output = run(&["inspect", "__no_such_series__"]);
    let combined = format!("{}{}", stdout(&output), stderr(&output));
    assert!(
        !combined.contains("panicked at"),
        "inspect panicked instead of reporting an error:\n{combined}"
    );
    assert!(
        !output.status.success(),
        "inspect of a missing input must fail:\n{combined}"
    );
    assert!(
        combined.to_lowercase().contains("error") || combined.contains("does not exist"),
        "inspect failed without a readable reason:\n{combined}"
    );
    assert_eq!(output.status.code(), Some(1), "exit code must be 1 on error");
}

#[test]
#[ignore = "heavy: writes ~700 MB of DNG output; run with --ignored"]
fn e2e_merge_real_series_104_and_106() {
    let out_dir = root().join(".scratch/wp3/e2e");
    std::fs::create_dir_all(&out_dir).expect("create e2e directory");

    for series in ["104NZ6_3", "106NZ6_3"] {
        let out = out_dir.join(format!("{series}_pixelshift.dng"));
        let _ = std::fs::remove_file(&out);
        let output = Process::new(bin())
            .current_dir(root())
            .args(["merge", series, "--out"])
            .arg(&out)
            .output()
            .expect("run merge");
        println!("merge {series}:\n{}", stdout(&output));
        eprintln!("{}", stderr(&output));
        assert!(output.status.success(), "merge {series} failed");
        let bytes = std::fs::metadata(&out).expect("output file").len();
        assert!(bytes > 10_000_000, "{series}: output only {bytes} bytes");
        println!("{series}: wrote {} ({} bytes)", out.display(), bytes);
    }
}
