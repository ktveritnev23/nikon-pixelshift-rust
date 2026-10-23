/* Thin POD interface over LibRaw for pixelshift.
 *
 * Owned by work package WP1. The shim exists so that the Rust side never has
 * to mirror LibRaw's large and version dependent C++ structs: it copies out
 * only the fields the pipeline needs and hands back a malloc'ed buffer with
 * the active (visible) pixel area.
 */
#ifndef PIXELSHIFT_LIBRAW_SHIM_H
#define PIXELSHIFT_LIBRAW_SHIM_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PS_ERROR_LEN 512

/* Decoded frame description. POD on purpose: layout is mirrored by
 * #[repr(C)] PsFrame in src/rawio/shim.rs. */
typedef struct ps_frame {
    /* Active area (visible pixels), the only pixels that are copied out. */
    int width;
    int height;
    /* Full raw sensor geometry as LibRaw reports it. */
    int raw_width;
    int raw_height;
    /* Origin of the active area inside the raw image. */
    int active_left;
    int active_top;
    int left_margin;
    int top_margin;
    int iwidth;
    int iheight;
    int colors;
    unsigned filters;
    /* 2x2 CFA of the active area, row major: cfa[y*2+x], 0=R 1=G 2=B.
     * LibRaw's fourth colour (3 = G2) is folded into G. */
    int cfa[4];
    int black;
    unsigned cblack[4];
    int maximum;
    unsigned linear_max[4];
    float cam_mul[4];
    /* LibRaw color.cam_xyz, row major [4][3]: rows = camera R,G,B,G2,
     * columns = X,Y,Z (XYZ -> camera). */
    float cam_xyz[12];
    /* LibRaw color.rgb_cam transposed to camera-major, row major [4][3]:
     * rows = camera R,G,B,G2, columns = sRGB R,G,B (camera -> sRGB D65). */
    float rgb_cam[12];
    float iso_speed;
    float shutter;
    float aperture;
    float focal_len;
    /* Lens identification from LibRaw lens.*: a raw editor matches its own lens
     * profile on these strings, which is what makes distortion and vignetting
     * correction available at all. Empty strings when the file has none. */
    char lens_make[64];
    char lens_model[128];
    /* Lens and body serial numbers as recorded in the file
     * (LibRaw lens.LensSerial / shootinginfo.BodySerial), "" when unknown. */
    char lens_serial[128];
    char body_serial[64];
    /* Focal length in 35 mm terms (LibRaw lens.FocalLengthIn35mmFormat),
     * 0 when unknown. */
    int focal_35mm;
    /* Maximum aperture as an f number (EXIF_MaxAp, falling back to the maker
     * note and the DNG lens record), 0 when unknown. */
    float max_ap;
    /* Lens specification (LibRaw lens.MinFocal/MaxFocal/MaxAp4MinFocal/
     * MaxAp4MaxFocal), all 0 when unknown. */
    float min_focal;
    float max_focal;
    float max_ap4_min_focal;
    float max_ap4_max_focal;
    /* EXIF timestamp as a Unix epoch (LibRaw other.timestamp), 0 if unset. */
    long long timestamp;
    char make[64];
    char model[64];
    char software[64];
    /* EXIF style capture time in local time, "" when unknown. */
    char datetime[24];
    /* Active-area samples, width*height, owned by this struct; release with
     * ps_free(). NULL when decoding failed. */
    uint16_t *pixels;
    /* Human readable reason, valid when ps_decode() != 0. */
    char error[PS_ERROR_LEN];
} ps_frame;

/* Decode one raw file. Returns 0 on success, a LibRaw error code (< 0) or a
 * shim specific error otherwise; the message is placed in out->error. */
int ps_decode(const char *path, ps_frame *out);

/* Release the pixel buffer of a frame (safe on zeroed and already freed
 * structs). */
void ps_free(ps_frame *f);

/* LibRaw version string, e.g. "0.22.2". */
const char *ps_version(void);

/* Demosaiced 8-bit sRGB rendering of one file, used as the reference image of
 * the distortion fit. POD on purpose: layout is mirrored by #[repr(C)] PsRgb
 * in src/rawio/shim.rs. */
typedef struct ps_rgb {
    int width;
    int height;
    /* Channels in data (3 = RGB). */
    int colors;
    /* Bits per sample (the shim always renders 8). */
    int bits;
    /* width * height * colors bytes with the camera orientation of the file
     * ignored (LibRaw user_flip = 0), so the pixels line up with the merged
     * frame. Owned by this struct; release with ps_free_rgb(). */
    unsigned char *data;
    char error[PS_ERROR_LEN];
} ps_rgb;

/* Render one raw file through libraw_dcraw_process + libraw_make_mem_image.
 * Returns 0 on success, a LibRaw error code (< 0) or a shim specific error
 * otherwise; the message is placed in out->error. */
int ps_render_rgb(const char *path, ps_rgb *out);

/* Release the pixel buffer of a rendering (safe on zeroed and already freed
 * structs). */
void ps_free_rgb(ps_rgb *r);

#ifdef __cplusplus
}
#endif

#endif /* PIXELSHIFT_LIBRAW_SHIM_H */
