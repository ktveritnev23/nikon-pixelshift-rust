#include "libraw_shim.h"

#include <libraw/libraw.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* dcraw's classic fcol(): color of the sensor pixel (row, col) in raw
 * coordinates. Values: 0=R 1=G 2=B 3=G2. */
static int ps_fcol(unsigned filters, int row, int col) {
    int shift = ((((row << 1) & 14) | (col & 1)) << 1);
    return (int)((filters >> shift) & 3u);
}

static void ps_set_error(ps_frame *out, int code, const char *prefix) {
    const char *msg = libraw_strerror(code);
    if (!msg) msg = "unknown LibRaw error";
    snprintf(out->error, PS_ERROR_LEN, "%s: %s (LibRaw code %d)", prefix, msg, code);
}

static void ps_set_message(ps_frame *out, const char *msg) {
    snprintf(out->error, PS_ERROR_LEN, "%s", msg);
}

static void ps_copy_str(char *dst, size_t cap, const char *src) {
    if (cap == 0) return;
    if (!src) {
        dst[0] = 0;
        return;
    }
    strncpy(dst, src, cap - 1);
    dst[cap - 1] = 0;
}

int ps_decode(const char *path, ps_frame *out) {
    libraw_data_t *lr;
    libraw_image_sizes_t *sz;
    const unsigned short *src;
    uint16_t *buf;
    size_t stride, width, height, y;
    int rc, ax, ay, aw, ah;

    if (!out) return -1;
    memset(out, 0, sizeof(*out));
    if (!path) {
        ps_set_message(out, "ps_decode: path is NULL");
        return -1;
    }

    lr = libraw_init(0);
    if (!lr) {
        ps_set_message(out, "libraw_init failed (out of memory)");
        return -1;
    }

    rc = libraw_open_file(lr, path);
    if (rc != LIBRAW_SUCCESS) {
        ps_set_error(out, rc, "libraw_open_file");
        libraw_close(lr);
        return rc ? rc : -1;
    }

    rc = libraw_unpack(lr);
    if (rc != LIBRAW_SUCCESS) {
        ps_set_error(out, rc, "libraw_unpack");
        libraw_close(lr);
        return rc ? rc : -1;
    }

    sz = &lr->sizes;
    if (lr->idata.colors < 3) {
        ps_set_message(out, "ps_decode: not an RGB (three colour) raw file");
        libraw_close(lr);
        return -1;
    }
    if (lr->idata.filters == 0) {
        ps_set_message(out, "ps_decode: file has no Bayer filter pattern (unsupported CFA layout)");
        libraw_close(lr);
        return -1;
    }
    if (!lr->rawdata.raw_image) {
        ps_set_message(out, "ps_decode: LibRaw did not produce a mono raw image");
        libraw_close(lr);
        return -1;
    }

    /* Active (visible) area. sizes.width/height is what LibRaw considers the
     * visible frame, but some cameras (the Nikon Z6 III among them) report the
     * whole stored raw there and describe the really usable area with
     * raw_inset_crops[0] (the DNG ActiveArea). Both use raw coordinates, so the
     * inset wins when present and self consistent. */
    ax = (int)sz->left_margin;
    ay = (int)sz->top_margin;
    aw = (int)sz->width;
    ah = (int)sz->height;
    {
        const libraw_raw_inset_crop_t *ins = &sz->raw_inset_crops[0];
        if (ins->cwidth && ins->cheight &&
            (int)ins->cleft + (int)ins->cwidth <= (int)sz->raw_width &&
            (int)ins->ctop + (int)ins->cheight <= (int)sz->raw_height) {
            ax = (int)ins->cleft;
            ay = (int)ins->ctop;
            aw = (int)ins->cwidth;
            ah = (int)ins->cheight;
        }
    }
    if (aw <= 0 || ah <= 0 || ax < 0 || ay < 0 || ax + aw > (int)sz->raw_width ||
        ay + ah > (int)sz->raw_height) {
        ps_set_message(out, "ps_decode: active area lies outside the raw image");
        libraw_close(lr);
        return -1;
    }

    width = (size_t)aw;
    height = (size_t)ah;
    buf = (uint16_t *)malloc(width * height * sizeof(uint16_t));
    if (!buf) {
        ps_set_message(out, "ps_decode: out of memory for the pixel buffer");
        libraw_close(lr);
        return -1;
    }

    /* raw_pitch is in bytes; fall back to raw_width samples when unset. */
    stride = sz->raw_pitch ? (size_t)(sz->raw_pitch / sizeof(unsigned short))
                           : (size_t)sz->raw_width;
    src = lr->rawdata.raw_image;
    for (y = 0; y < height; y++) {
        const unsigned short *line = src + ((size_t)ay + y) * stride + (size_t)ax;
        memcpy(buf + y * width, line, width * sizeof(uint16_t));
    }

    out->width = aw;
    out->height = ah;
    out->active_left = ax;
    out->active_top = ay;
    out->raw_width = (int)sz->raw_width;
    out->raw_height = (int)sz->raw_height;
    out->left_margin = (int)sz->left_margin;
    out->top_margin = (int)sz->top_margin;
    out->iwidth = (int)sz->iwidth;
    out->iheight = (int)sz->iheight;
    out->colors = (int)lr->idata.colors;
    out->filters = lr->idata.filters;

    {
        int dx, dy;
        for (dy = 0; dy < 2; dy++) {
            for (dx = 0; dx < 2; dx++) {
                int c = ps_fcol(out->filters, ay + dy, ax + dx);
                if (c == 3) c = 1; /* G2 folds into G */
                out->cfa[dy * 2 + dx] = c;
            }
        }
    }

    out->black = (int)lr->color.black;
    out->maximum = (int)lr->color.maximum;
    {
        int i;
        for (i = 0; i < 4; i++) {
            out->cblack[i] = lr->color.cblack[i];
            out->linear_max[i] = lr->color.linear_max[i];
            out->cam_mul[i] = lr->color.cam_mul[i];
        }
        for (i = 0; i < 12; i++) {
            out->cam_xyz[i] = (&lr->color.cam_xyz[0][0])[i];
        }
        /* LibRaw stores rgb_cam as [rgb][camera channel]; transpose it. */
        for (i = 0; i < 4; i++) {
            int j;
            for (j = 0; j < 3; j++) {
                out->rgb_cam[i * 3 + j] = lr->color.rgb_cam[j][i];
            }
        }
    }

    out->iso_speed = lr->other.iso_speed;
    out->shutter = lr->other.shutter;
    out->aperture = lr->other.aperture;
    out->focal_len = lr->other.focal_len;
    out->focal_35mm = (int)lr->lens.FocalLengthIn35mmFormat;
    ps_copy_str(out->lens_make, sizeof(out->lens_make), lr->lens.LensMake);
    ps_copy_str(out->lens_model, sizeof(out->lens_model), lr->lens.Lens);
    ps_copy_str(out->lens_serial, sizeof(out->lens_serial), lr->lens.LensSerial);
    ps_copy_str(out->body_serial, sizeof(out->body_serial), lr->shootinginfo.BodySerial);
    /* The camera writes MaxAp4MinFocal/MaxAp4MaxFocal reliably, EXIF_MaxAp and
     * the maker note aperture often stay 0 (verified on a Z6 III with a
     * Viltrox 135/1.8), so the lens record is the last resort. */
    out->max_ap = lr->lens.EXIF_MaxAp;
    if (!(out->max_ap > 0.01f)) out->max_ap = lr->lens.makernotes.MaxAp;
    if (!(out->max_ap > 0.01f)) out->max_ap = lr->lens.dng.MaxAp4MaxFocal;
    if (!(out->max_ap > 0.01f)) out->max_ap = lr->lens.MaxAp4MaxFocal;
    if (!(out->max_ap > 0.01f)) out->max_ap = lr->lens.MaxAp4MinFocal;
    if (!(out->max_ap > 0.01f)) out->max_ap = 0.0f;
    out->min_focal = lr->lens.MinFocal;
    out->max_focal = lr->lens.MaxFocal;
    out->max_ap4_min_focal = lr->lens.MaxAp4MinFocal;
    out->max_ap4_max_focal = lr->lens.MaxAp4MaxFocal;
    out->timestamp = (long long)lr->other.timestamp;

    ps_copy_str(out->make, sizeof(out->make), lr->idata.make);
    ps_copy_str(out->model, sizeof(out->model), lr->idata.model);
    ps_copy_str(out->software, sizeof(out->software), lr->idata.software);

    if (lr->other.timestamp > 0) {
        time_t t = (time_t)lr->other.timestamp;
        struct tm tmv;
        if (localtime_r(&t, &tmv) != NULL) {
            strftime(out->datetime, sizeof(out->datetime), "%Y:%m:%d %H:%M:%S", &tmv);
        }
    }

    libraw_close(lr);

    /* LibRaw leaves maximum at 0 for some formats; fall back to the
     * highlight linearity limit or to a full 16 bit range. */
    if (out->maximum <= 0) {
        out->maximum = (int)(out->linear_max[0] ? out->linear_max[0] : 65535u);
    }

    out->pixels = buf;
    return 0;
}

void ps_free(ps_frame *f) {
    if (!f) return;
    free(f->pixels);
    f->pixels = NULL;
}

static void ps_set_rgb_error(ps_rgb *out, int code, const char *prefix) {
    const char *msg = libraw_strerror(code);
    if (!msg) msg = "unknown LibRaw error";
    snprintf(out->error, PS_ERROR_LEN, "%s: %s (LibRaw code %d)", prefix, msg, code);
}

int ps_render_rgb(const char *path, ps_rgb *out) {
    libraw_data_t *lr;
    libraw_processed_image_t *img;
    size_t bytes;
    int rc, code;

    if (!out) return -1;
    memset(out, 0, sizeof(*out));
    if (!path) {
        snprintf(out->error, PS_ERROR_LEN, "ps_render_rgb: path is NULL");
        return -1;
    }

    lr = libraw_init(0);
    if (!lr) {
        snprintf(out->error, PS_ERROR_LEN, "libraw_init failed (out of memory)");
        return -1;
    }
    rc = libraw_open_file(lr, path);
    if (rc != LIBRAW_SUCCESS) {
        ps_set_rgb_error(out, rc, "libraw_open_file");
        libraw_close(lr);
        return rc ? rc : -1;
    }
    rc = libraw_unpack(lr);
    if (rc != LIBRAW_SUCCESS) {
        ps_set_rgb_error(out, rc, "libraw_unpack");
        libraw_close(lr);
        return rc ? rc : -1;
    }

    /* 8 bit sRGB, no camera orientation, no auto brightening: the reference is
     * only ever compared structurally with the merged frame, and a rotation the
     * merge does not apply would shift every block. */
    lr->params.user_flip = 0;
    lr->params.output_bps = 8;
    lr->params.output_color = 1;
    lr->params.no_auto_bright = 1;
    lr->params.gamm[0] = 1.0 / 2.4;
    lr->params.gamm[1] = 12.92;

    code = libraw_dcraw_process(lr);
    if (code != LIBRAW_SUCCESS) {
        ps_set_rgb_error(out, code, "libraw_dcraw_process");
        libraw_close(lr);
        return code ? code : -1;
    }
    img = libraw_dcraw_make_mem_image(lr, &rc);
    if (!img || rc != LIBRAW_SUCCESS) {
        snprintf(out->error, PS_ERROR_LEN, "libraw_dcraw_make_mem_image failed (code %d)", rc);
        if (img) libraw_dcraw_clear_mem(img);
        libraw_close(lr);
        return -1;
    }
    if (img->type != LIBRAW_IMAGE_BITMAP || img->colors < 3) {
        snprintf(out->error, PS_ERROR_LEN, "ps_render_rgb: LibRaw produced no RGB bitmap");
        libraw_dcraw_clear_mem(img);
        libraw_close(lr);
        return -1;
    }

    bytes = (size_t)img->width * (size_t)img->height * (size_t)img->colors;
    out->data = (unsigned char *)malloc(bytes);
    if (!out->data) {
        snprintf(out->error, PS_ERROR_LEN, "ps_render_rgb: out of memory (%zu bytes)", bytes);
        libraw_dcraw_clear_mem(img);
        libraw_close(lr);
        return -1;
    }
    memcpy(out->data, img->data, bytes);
    out->width = (int)img->width;
    out->height = (int)img->height;
    out->colors = (int)img->colors;
    out->bits = (int)img->bits;
    libraw_dcraw_clear_mem(img);
    libraw_close(lr);
    return 0;
}

void ps_free_rgb(ps_rgb *r) {
    if (!r) return;
    if (r->data) free(r->data);
    r->data = NULL;
}

const char *ps_version(void) { return libraw_version(); }
