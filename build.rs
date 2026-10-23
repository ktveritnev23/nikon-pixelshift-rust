fn main() {
    println!("cargo:rerun-if-changed=shim/libraw_shim.c");
    println!("cargo:rerun-if-changed=shim/libraw_shim.h");
    if !std::path::Path::new("shim/libraw_shim.c").exists() {
        println!("cargo:warning=shim/libraw_shim.c not found; raw decoding will fail at runtime");
        return;
    }
    // Do not let pkg-config emit link flags directly: the brew .pc file carries
    // "-lstdc++", which does not exist on macOS (libc++ is linked by default).
    let lib = pkg_config::Config::new()
        .cargo_metadata(false)
        .probe("libraw_r")
        .expect("LibRaw not found. Install it first: brew install libraw");

    for path in &lib.link_paths {
        println!("cargo:rustc-link-search=native={}", path.display());
    }
    for name in &lib.libs {
        if name == "stdc++" || name == "c++" {
            continue;
        }
        println!("cargo:rustc-link-lib={}", name);
    }

    let mut build = cc::Build::new();
    build.file("shim/libraw_shim.c").warnings(false).opt_level(2);
    for path in &lib.include_paths {
        build.include(path);
    }
    build.compile("pixelshift_libraw_shim");
}
