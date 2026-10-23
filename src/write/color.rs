//! Colour transform shared by the DNG preview and the rendered TIFF.
//!
//! The merged planes are camera native, black subtracted, white balance NOT
//! applied. Rendering means: divide by the as-shot neutral multiplier, apply
//! XYZ (D50) -> camera and camera -> linear sRGB, clip and encode.

/// XYZ (D50) -> linear sRGB (D65, Bradford adapted), the standard matrix.
pub const XYZ_D50_TO_SRGB: [[f32; 3]; 3] = [
    [3.133_856_1, -1.616_866_7, -0.490_614_6],
    [-0.978_768_4, 1.916_141_5, 0.033_454_0],
    [0.071_945_3, -0.228_991_4, 1.405_242_7],
];

pub fn mul3(m: &[[f32; 3]; 3], v: &[f32; 3]) -> [f32; 3] {
    let mut out = [0.0f32; 3];
    for (i, row) in m.iter().enumerate() {
        out[i] = row[0] * v[0] + row[1] * v[1] + row[2] * v[2];
    }
    out
}

fn mat3_mul(a: &[[f32; 3]; 3], b: &[[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let mut out = [[0.0f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    out
}

fn invert3(m: &[[f32; 3]; 3]) -> Option<[[f32; 3]; 3]> {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    if !det.is_finite() || det.abs() < 1e-9 {
        return None;
    }
    let inv_det = 1.0 / det;
    let mut out = [[0.0f32; 3]; 3];
    out[0][0] = (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv_det;
    out[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv_det;
    out[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det;
    out[1][0] = (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv_det;
    out[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det;
    out[1][2] = (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv_det;
    out[2][0] = (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv_det;
    out[2][1] = (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv_det;
    out[2][2] = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_det;
    Some(out)
}

/// Camera (already white balanced) -> linear sRGB.
///
/// The rows are normalised so that a white balanced neutral maps to exactly
/// (1, 1, 1): camera and sRGB white points differ (D50 vs the as-shot
/// illuminant), and a preview is worthless if its greys are tinted.
pub fn camera_to_srgb(color_matrix: &[[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let Some(xyz_to_cam_inv) = invert3(color_matrix) else {
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    };
    let mut m = mat3_mul(&XYZ_D50_TO_SRGB, &xyz_to_cam_inv);
    for row in m.iter_mut() {
        let sum = row[0] + row[1] + row[2];
        if sum.is_finite() && sum.abs() > 1e-6 {
            for v in row.iter_mut() {
                *v /= sum;
            }
        }
    }
    m
}

/// sRGB transfer function, linear 0..1 -> 0..1.
pub fn encode_srgb_linear(linear: f32) -> f32 {
    let v = linear.clamp(0.0, 1.0);
    if v <= 0.003_130_8 {
        12.92 * v
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
}

pub fn encode_srgb_u8(linear: f32) -> u8 {
    (encode_srgb_linear(linear) * 255.0).round().clamp(0.0, 255.0) as u8
}

pub fn encode_srgb_u16(linear: f32) -> u16 {
    (encode_srgb_linear(linear) * 65535.0).round().clamp(0.0, 65535.0) as u16
}
