//! Software drawing onto a `wl_shm` canvas: colours, rectangles, text.
//!
//! Buffers are `Argb8888`, which is a packed little-endian `0xAARRGGBB` — the
//! four bytes in memory are `[B, G, R, A]` — and the compositor expects
//! **premultiplied** alpha. [`Color`] is straight (non-premultiplied) and gets
//! premultiplied at write time, the same convention the output picker uses.
//!
//! Everything clips: a [`Canvas`] carries a clip rectangle so a scrolled list
//! can draw rows that run past the viewport without the caller doing the
//! arithmetic. Out-of-bounds writes are dropped rather than wrapped.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    reason = "pixel geometry: every value here is a surface- or image-sized non-negative integer, and the conversions between i32/u32/usize/f32 are all inside that range. Checked conversions at each site would be noise around arithmetic that cannot overflow."
)]

use super::text::Fonts;

/// A straight (non-premultiplied) RGBA colour.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl Color {
    pub const fn rgb(value: u32) -> Self {
        let [_, r, g, b] = value.to_be_bytes();
        Self { r, g, b, a: 255 }
    }

    pub const fn rgba(value: u32) -> Self {
        let [a, r, g, b] = value.to_be_bytes();
        Self { r, g, b, a }
    }

    /// The same colour at `alpha` (0.0–1.0) of its opacity.
    pub fn with_alpha(self, alpha: f32) -> Self {
        Self {
            a: (f32::from(self.a) * alpha.clamp(0.0, 1.0)) as u8,
            ..self
        }
    }

    /// Linear blend towards `other`. Used for hover/pressed shading so the
    /// theme only has to name a handful of base colours.
    pub fn mix(self, other: Self, t: f32) -> Self {
        let t = t.clamp(0.0, 1.0);
        let lerp = |a: u8, b: u8| (f32::from(a) + (f32::from(b) - f32::from(a)) * t) as u8;
        Self {
            r: lerp(self.r, other.r),
            g: lerp(self.g, other.g),
            b: lerp(self.b, other.b),
            a: lerp(self.a, other.a),
        }
    }
}

/// An axis-aligned rectangle in canvas (device) pixels.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Rect {
    pub x: i32,
    pub y: i32,
    pub w: i32,
    pub h: i32,
}

impl Rect {
    pub const fn new(x: i32, y: i32, w: i32, h: i32) -> Self {
        Self { x, y, w, h }
    }

    pub const fn right(self) -> i32 {
        self.x + self.w
    }

    pub const fn bottom(self) -> i32 {
        self.y + self.h
    }

    pub fn contains(self, x: f64, y: f64) -> bool {
        x >= f64::from(self.x)
            && y >= f64::from(self.y)
            && x < f64::from(self.right())
            && y < f64::from(self.bottom())
    }

    /// Shrink by `d` on every side (negative grows).
    pub const fn inset(self, d: i32) -> Self {
        Self {
            x: self.x + d,
            y: self.y + d,
            w: self.w - 2 * d,
            h: self.h - 2 * d,
        }
    }

    pub fn intersect(self, other: Self) -> Self {
        let x = self.x.max(other.x);
        let y = self.y.max(other.y);
        let right = self.right().min(other.right());
        let bottom = self.bottom().min(other.bottom());
        Self {
            x,
            y,
            w: (right - x).max(0),
            h: (bottom - y).max(0),
        }
    }

    pub const fn is_empty(self) -> bool {
        self.w <= 0 || self.h <= 0
    }
}

/// A borrowed BGRA/BGRX image, as [`Canvas::blit`] consumes it. This is the
/// shape captured frames arrive in (see `crate::capture`).
pub struct Image<'a> {
    pub data: &'a [u8],
    pub width: i32,
    pub height: i32,
    /// Bytes per row, which is not always `width * 4`.
    pub stride: usize,
}

/// A mutable view over one `wl_shm` buffer.
///
/// **Everything public here is in logical pixels.** Views lay out as if the
/// display were 1×; the canvas multiplies by the surface's buffer scale on the
/// way to the framebuffer, so the same layout code is sharp on a `HiDPI` monitor
/// and hit-tests directly against the logical pointer coordinates Wayland
/// delivers. Scale is an integer (we use `wl_surface.set_buffer_scale`), so
/// the mapping is exact — no half-pixel seams.
pub struct Canvas<'a> {
    buf: &'a mut [u8],
    /// Device-pixel dimensions of the buffer.
    width: i32,
    height: i32,
    stride: i32,
    /// Clip rectangle, in device pixels.
    clip: Rect,
    scale: i32,
}

impl<'a> Canvas<'a> {
    pub fn new(buf: &'a mut [u8], width: i32, height: i32, stride: i32, scale: i32) -> Self {
        Self {
            buf,
            width,
            height,
            stride,
            clip: Rect::new(0, 0, width, height),
            scale: scale.max(1),
        }
    }

    /// Canvas size in logical pixels — the coordinate space views lay out in.
    pub const fn size(&self) -> (i32, i32) {
        (self.width / self.scale, self.height / self.scale)
    }

    /// Logical rect → device rect.
    const fn dev(&self, r: Rect) -> Rect {
        Rect::new(
            r.x * self.scale,
            r.y * self.scale,
            r.w * self.scale,
            r.h * self.scale,
        )
    }

    /// Run `body` with the clip narrowed to `rect` (logical), restoring it
    /// afterwards.
    pub fn clipped<R>(&mut self, rect: Rect, body: impl FnOnce(&mut Self) -> R) -> R {
        let saved = self.clip;
        self.clip = saved.intersect(self.dev(rect));
        let out = body(self);
        self.clip = saved;
        out
    }

    /// Blend one straight-alpha pixel at `(x, y)` with `coverage` (0–255)
    /// modulating its alpha. Source-over onto premultiplied destination.
    fn blend(&mut self, x: i32, y: i32, color: Color, coverage: u8) {
        if x < self.clip.x || y < self.clip.y || x >= self.clip.right() || y >= self.clip.bottom() {
            return;
        }
        let alpha = u32::from(color.a) * u32::from(coverage) / 255;
        if alpha == 0 {
            return;
        }
        let offset = (y * self.stride + x * 4) as usize;
        let Some(px) = self.buf.get_mut(offset..offset + 4) else {
            return;
        };
        if alpha == 255 {
            px.copy_from_slice(&[color.b, color.g, color.r, 255]);
            return;
        }
        // src premultiplied by its own alpha, dst scaled by (1 - alpha).
        let inv = 255 - alpha;
        let over = |src: u8, dst: u8| -> u8 {
            ((u32::from(src) * alpha + u32::from(dst) * inv) / 255) as u8
        };
        px[0] = over(color.b, px[0]);
        px[1] = over(color.g, px[1]);
        px[2] = over(color.r, px[2]);
        px[3] = (alpha + u32::from(px[3]) * inv / 255) as u8;
    }

    /// Overwrite the whole canvas (ignoring the clip) with an opaque colour.
    pub fn clear(&mut self, color: Color) {
        let px = [color.b, color.g, color.r, color.a];
        for chunk in self.buf.chunks_exact_mut(4) {
            chunk.copy_from_slice(&px);
        }
    }

    pub fn fill(&mut self, rect: Rect, color: Color) {
        let area = self.dev(rect).intersect(self.clip);
        if area.is_empty() {
            return;
        }
        for y in area.y..area.bottom() {
            for x in area.x..area.right() {
                self.blend(x, y, color, 255);
            }
        }
    }

    /// A filled rectangle with anti-aliased rounded corners.
    ///
    /// Coverage in the corner arcs comes from the distance to the arc centre,
    /// clamped to a one-pixel band — cheap, and visually indistinguishable
    /// from proper analytic coverage at the radii a UI uses.
    pub fn fill_rounded(&mut self, rect: Rect, radius: i32, color: Color) {
        let rect = self.dev(rect);
        let radius = (radius * self.scale).min(rect.w / 2).min(rect.h / 2).max(0);
        if radius == 0 {
            let area = rect.intersect(self.clip);
            for y in area.y..area.bottom() {
                for x in area.x..area.right() {
                    self.blend(x, y, color, 255);
                }
            }
            return;
        }
        let area = rect.intersect(self.clip);
        if area.is_empty() {
            return;
        }
        let r = radius as f32;
        for y in area.y..area.bottom() {
            for x in area.x..area.right() {
                // Distance from this pixel's centre to the nearest corner
                // centre, but only inside the corner squares — the straight
                // edges and the middle are always fully covered.
                let dx = if x < rect.x + radius {
                    (rect.x + radius) as f32 - (x as f32 + 0.5)
                } else if x >= rect.right() - radius {
                    (x as f32 + 0.5) - (rect.right() - radius) as f32
                } else {
                    0.0
                };
                let dy = if y < rect.y + radius {
                    (rect.y + radius) as f32 - (y as f32 + 0.5)
                } else if y >= rect.bottom() - radius {
                    (y as f32 + 0.5) - (rect.bottom() - radius) as f32
                } else {
                    0.0
                };
                let coverage = if dx == 0.0 || dy == 0.0 {
                    255
                } else {
                    let d = dx.hypot(dy);
                    // 1px feather from just inside the arc to just outside.
                    let c = ((r + 0.5 - d) * 255.0).clamp(0.0, 255.0);
                    c as u8
                };
                if coverage > 0 {
                    self.blend(x, y, color, coverage);
                }
            }
        }
    }

    /// A rounded outline `thickness` px wide, drawn inside `rect`.
    pub fn stroke_rounded(&mut self, rect: Rect, radius: i32, thickness: i32, color: Color) {
        if thickness <= 0 || rect.is_empty() {
            return;
        }
        // Outline = outer rounded rect minus inner one. Drawing it as two
        // fills would blend the inner one over the outer; instead clip to the
        // four edge bands so only the ring is touched.
        let inner = rect.inset(thickness);
        let bands = [
            Rect::new(rect.x, rect.y, rect.w, thickness),
            Rect::new(rect.x, inner.bottom(), rect.w, thickness),
            Rect::new(rect.x, inner.y, thickness, inner.h),
            Rect::new(inner.right(), inner.y, thickness, inner.h),
        ];
        for band in bands {
            self.clipped(band, |c| c.fill_rounded(rect, radius, color));
        }
    }

    /// Copy a region of an external BGRA image into `dst` (logical), scaling
    /// with nearest-neighbour sampling.
    ///
    /// Used for the frozen screen behind the region selector and for the
    /// colour picker's magnifier, where nearest-neighbour is not a compromise
    /// but the point: you want to see the actual pixel grid.
    pub fn blit(&mut self, dst: Rect, src: &Image<'_>, src_rect: Rect) {
        let target = self.dev(dst);
        let area = target.intersect(self.clip);
        if area.is_empty() || src_rect.is_empty() {
            return;
        }
        let horizontal_step = src_rect.w as f32 / target.w as f32;
        let vertical_step = src_rect.h as f32 / target.h as f32;
        for y in area.y..area.bottom() {
            let sy = src_rect.y + ((y - target.y) as f32 * vertical_step) as i32;
            if sy < 0 || sy >= src.height {
                continue;
            }
            for x in area.x..area.right() {
                let sx = src_rect.x + ((x - target.x) as f32 * horizontal_step) as i32;
                if sx < 0 || sx >= src.width {
                    continue;
                }
                let offset = (sy as usize) * src.stride + (sx as usize) * 4;
                let Some(px) = src.data.get(offset..offset + 4) else {
                    continue;
                };
                // Source is opaque BGRX/BGRA; write it straight through rather
                // than blending, which is both faster and correct for a
                // background layer.
                let dst_offset = (y * self.stride + x * 4) as usize;
                if let Some(slot) = self.buf.get_mut(dst_offset..dst_offset + 4) {
                    slot.copy_from_slice(&[px[0], px[1], px[2], 255]);
                }
            }
        }
    }

    /// Width of `text` in logical pixels. Measured at the device size the
    /// glyphs will actually be rasterized at, so layout and drawing agree.
    pub fn measure(&self, fonts: &Fonts, text: &str, px: f32, bold: bool) -> f32 {
        let s = self.scale as f32;
        fonts.measure(text, px * s, bold) / s
    }

    /// Draw `text` with its baseline at `baseline_y`, pen starting at `x`.
    /// Returns the pen's end position.
    #[allow(
        clippy::too_many_arguments,
        reason = "a text call is a pen position, a size, a weight, a colour and a string; bundling them into a struct would make every call site longer, not clearer"
    )]
    pub fn text(
        &mut self,
        fonts: &Fonts,
        x: f32,
        baseline_y: f32,
        px: f32,
        bold: bool,
        color: Color,
        text: &str,
    ) -> f32 {
        let s = self.scale as f32;
        fonts.layout(
            text,
            px * s,
            bold,
            x * s,
            baseline_y * s,
            |gx, gy, w, h, coverage| {
                for row in 0..h {
                    for col in 0..w {
                        let value = coverage[row * w + col];
                        if value > 0 {
                            self.blend(gx + col as i32, gy + row as i32, color, value);
                        }
                    }
                }
            },
        );
        x + self.measure(fonts, text, px, bold)
    }

    /// Draw `text` centred horizontally in `rect` and vertically about its
    /// cap height.
    pub fn text_centered(
        &mut self,
        fonts: &Fonts,
        rect: Rect,
        px: f32,
        bold: bool,
        color: Color,
        text: &str,
    ) {
        let width = self.measure(fonts, text, px, bold);
        let x = rect.x as f32 + (rect.w as f32 - width) / 2.0;
        // Centre on the cap height rather than the line box: text looks
        // vertically centred when its caps are, not when its descenders are.
        let baseline = rect.y as f32 + f32::midpoint(rect.h as f32, px * 0.72);
        self.clipped(rect, |c| {
            c.text(fonts, x, baseline, px, bold, color, text);
        });
    }

    /// Draw `text` left-aligned and clipped to `rect`, ellipsized if it
    /// doesn't fit, with its baseline vertically centred.
    pub fn text_in(
        &mut self,
        fonts: &Fonts,
        rect: Rect,
        px: f32,
        bold: bool,
        color: Color,
        text: &str,
    ) {
        let s = self.scale as f32;
        let shown = fonts.ellipsize(text, px * s, bold, rect.w as f32 * s);
        let baseline = rect.y as f32 + f32::midpoint(rect.h as f32, px * 0.72);
        self.clipped(rect, |c| {
            c.text(fonts, rect.x as f32, baseline, px, bold, color, &shown);
        });
    }
}

/// Colours for one appearance. Two instances exist ([`Theme::DARK`] and
/// [`Theme::LIGHT`]); which one a dialog uses follows the same
/// `org.freedesktop.appearance color-scheme` value we serve to apps through
/// the Settings portal, so the portal's own windows match what it tells
/// everyone else the desktop looks like.
#[derive(Clone, Copy)]
pub struct Theme {
    pub bg: Color,
    pub header: Color,
    pub surface: Color,
    pub raised: Color,
    pub border: Color,
    pub text: Color,
    pub dim: Color,
    pub accent: Color,
    pub on_accent: Color,
    pub danger: Color,
}

impl Theme {
    pub const DARK: Self = Self {
        bg: Color::rgb(0x1B_1D_22),
        header: Color::rgb(0x22_25_2B),
        surface: Color::rgb(0x24_27_2E),
        raised: Color::rgb(0x2E_32_3B),
        border: Color::rgb(0x3A_3F_4A),
        text: Color::rgb(0xE8_EA_ED),
        dim: Color::rgb(0x9A_A0_AB),
        // The same azure the output picker highlights with, so the desktop's
        // own chrome reads as one product.
        accent: Color::rgb(0x4A_9E_FF),
        on_accent: Color::rgb(0x0B_12_1C),
        danger: Color::rgb(0xE0_5A_5A),
    };

    pub const LIGHT: Self = Self {
        bg: Color::rgb(0xF6_F7_F9),
        header: Color::rgb(0xEC_EE_F2),
        surface: Color::rgb(0xFF_FF_FF),
        raised: Color::rgb(0xE7_EA_EF),
        border: Color::rgb(0xD2_D7_DE),
        text: Color::rgb(0x1B_1D_22),
        dim: Color::rgb(0x5E_66_72),
        accent: Color::rgb(0x1B_6F_D9),
        on_accent: Color::rgb(0xFF_FF_FF),
        danger: Color::rgb(0xC0_3A_3A),
    };
}

#[cfg(test)]
mod tests {
    use super::{Canvas, Color, Rect};

    /// `clear` must write premultiplied bytes in `wl_shm`'s B,G,R,A order —
    /// the translucent wash the pickers draw over the desktop depends on the
    /// alpha actually landing in the buffer.
    #[test]
    fn clear_writes_premultiplied_bgra() {
        let mut buf = vec![0u8; 4 * 4];
        let mut canvas = Canvas::new(&mut buf, 2, 2, 8, 1);
        canvas.clear(Color::rgba(0x8C_00_00_00));
        assert_eq!(&buf[..4], &[0, 0, 0, 0x8C]);
        assert_eq!(&buf[12..], &[0, 0, 0, 0x8C]);
    }

    /// A logical rect is scaled to device pixels, and only that rect is
    /// touched.
    #[test]
    fn fill_maps_logical_to_device_pixels() {
        let mut buf = vec![0u8; 4 * 16];
        let mut canvas = Canvas::new(&mut buf, 4, 4, 16, 2);
        canvas.fill(Rect::new(0, 0, 1, 1), Color::rgb(0xFF_00_00));
        // 1×1 logical at scale 2 covers the top-left 2×2 device block.
        assert_eq!(&buf[0..4], &[0, 0, 255, 255]);
        assert_eq!(&buf[4..8], &[0, 0, 255, 255]);
        assert_eq!(&buf[16..20], &[0, 0, 255, 255]);
        // ...and nothing else.
        assert_eq!(&buf[8..12], &[0, 0, 0, 0]);
    }
}
