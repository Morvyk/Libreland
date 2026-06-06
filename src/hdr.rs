//! HDR signalling helpers for DRM/KMS outputs.
//!
//! The connector properties an HDR signal needs (`Colorspace`,
//! `max bpc`, `HDR_OUTPUT_METADATA`) require a modeset and must ride in
//! the *same* atomic commit that sets the mode and plane — a separate
//! side-channel commit on an already-settled pipe wedges the display
//! (verified on real 4K@240 HDR hardware). smithay 0.7 has no API for
//! this, so Libreland vendors a patched smithay (`vendor/smithay`) that
//! adds [`DrmSurface::set_hdr`]; this module resolves the concrete
//! property values into a [`HdrMetadata`] for that call.
//!
//! This is the DRM *signal* half of HDR only: it makes the link carry a
//! 10-bit Rec.2020 / PQ stream. Colour-correct compositing (decoding
//! each source to linear light, tonemapping for SDR outputs and
//! screenshots) is a separate pipeline still to come — until it lands,
//! content looks washed-out/wrong on an HDR output even though the wire
//! signal is valid.

use anyhow::{Context as _, Result};
use smithay::backend::drm::{DrmSurface, HdrMetadata};
use smithay::reexports::drm::control::property::ValueType;
use smithay::reexports::drm::control::{Device as ControlDevice, connector, property};

/// SMPTE ST 2084 (PQ) selector for the infoframe `eotf` field
/// (`enum hdmi_eotf`, CTA-861.G). 0 = SDR gamma, 1 = traditional HDR
/// gamma, 2 = PQ, 3 = HLG.
const EOTF_SMPTE_ST2084: u8 = 2;

/// `Static_Metadata_Descriptor_ID` value selecting "Static Metadata
/// Type 1" — the only descriptor the kernel uABI defines. Used for both
/// the outer [`HdrOutputMetadata::metadata_type`] and the infoframe's.
const STATIC_METADATA_TYPE_1: u8 = 0;

/// CIE 1931 chromaticity pair in CTA-861 units of `0.00002`
/// (`0xC350` == `1.0`). Mirrors the kernel's anonymous
/// `struct { __u16 x, y; }` inside `hdr_metadata_infoframe`.
#[repr(C)]
#[derive(Clone, Copy)]
#[allow(
    dead_code,
    reason = "fields define the C blob layout serialized via create_property_blob; never read field-wise in Rust"
)]
struct Chromaticity {
    x: u16,
    y: u16,
}

/// HDR Metadata Infoframe, CTA-861.G — byte-for-byte the kernel's
/// `struct hdr_metadata_infoframe` (passed as a property blob).
#[repr(C)]
#[derive(Clone, Copy)]
#[allow(
    dead_code,
    reason = "fields define the C blob layout serialized via create_property_blob; never read field-wise in Rust"
)]
struct HdrMetadataInfoframe {
    /// Electro-Optical Transfer Function used in the stream.
    eotf: u8,
    /// `Static_Metadata_Descriptor_ID`.
    metadata_type: u8,
    /// Mastering display primaries, R/G/B order.
    display_primaries: [Chromaticity; 3],
    white_point: Chromaticity,
    /// Max mastering display luminance, units of 1 cd/m².
    max_display_mastering_luminance: u16,
    /// Min mastering display luminance, units of 0.0001 cd/m².
    min_display_mastering_luminance: u16,
    /// Max Content Light Level, units of 1 cd/m² (0 = unknown).
    max_cll: u16,
    /// Max Frame-Average Light Level, units of 1 cd/m² (0 = unknown).
    max_fall: u16,
}

/// Kernel `struct hdr_output_metadata` — the `HDR_OUTPUT_METADATA`
/// property's blob payload.
#[repr(C)]
#[derive(Clone, Copy)]
#[allow(
    dead_code,
    reason = "fields define the C blob layout serialized via create_property_blob; never read field-wise in Rust"
)]
struct HdrOutputMetadata {
    /// `Static_Metadata_Descriptor_ID` (HDMI Static Metadata Type 1).
    metadata_type: u32,
    hdmi_metadata_type1: HdrMetadataInfoframe,
}

/// Default PQ / Rec.2020 mastering metadata.
///
/// Placeholder mastering volume for the DRM bring-up: full Rec.2020
/// primaries + D65 white, 1000 cd/m² peak. A later phase will read the
/// connector's EDID HDR static-metadata block and substitute the
/// panel's real luminance/primaries; content light levels stay 0
/// (unknown) until clients supply them via the colour-management
/// protocol. CIE coordinates are pre-scaled by 50000 (i.e. ÷ 0.00002).
fn pq_rec2020_metadata() -> HdrOutputMetadata {
    HdrOutputMetadata {
        metadata_type: u32::from(STATIC_METADATA_TYPE_1),
        hdmi_metadata_type1: HdrMetadataInfoframe {
            eotf: EOTF_SMPTE_ST2084,
            metadata_type: STATIC_METADATA_TYPE_1,
            display_primaries: [
                Chromaticity { x: 35400, y: 14600 }, // R 0.708, 0.292
                Chromaticity { x: 8500, y: 39850 },  // G 0.170, 0.797
                Chromaticity { x: 6550, y: 2300 },   // B 0.131, 0.046
            ],
            white_point: Chromaticity { x: 15635, y: 16450 }, // D65 0.3127, 0.3290
            max_display_mastering_luminance: 1000,
            min_display_mastering_luminance: 1, // 0.0001 cd/m²
            max_cll: 0,
            max_fall: 0,
        },
    }
}

/// Resolve a connector's HDR-on properties into a [`HdrMetadata`] for
/// [`DrmSurface::set_hdr`], folding them into smithay's own modeset:
/// `Colorspace=BT2020_RGB`, `max bpc=10`, and a PQ / Rec.2020
/// `HDR_OUTPUT_METADATA` blob.
///
/// Returns `Ok(None)` when the connector exposes no
/// `HDR_OUTPUT_METADATA` property at all (it cannot do HDR; leave it
/// untouched). `Colorspace` / `max bpc` are included only when present —
/// some drivers signal via metadata alone. The metadata blob is created
/// on `surface`'s own fd (required: the modeset that consumes it runs on
/// that fd).
pub fn hdr_metadata(
    surface: &DrmSurface,
    connector: connector::Handle,
) -> Result<Option<HdrMetadata>> {
    let props = surface
        .get_properties(connector)
        .context("reading connector properties")?;
    let (handles, _values) = props.as_props_and_values();

    let mut has_metadata = false;
    let mut has_max_bpc = false;
    let mut colorspace = None;

    for &handle in handles {
        let Ok(info) = surface.get_property(handle) else {
            continue;
        };
        match info.name().to_str() {
            Ok("HDR_OUTPUT_METADATA") => has_metadata = true,
            Ok("max bpc") => has_max_bpc = true,
            Ok("Colorspace") => {
                if let ValueType::Enum(values) = info.value_type() {
                    let (_, enums) = values.values();
                    colorspace = enums
                        .iter()
                        .find(|entry| entry.name().to_str() == Ok("BT2020_RGB"))
                        .map(property::EnumValue::value);
                }
            }
            _ => {}
        }
    }

    if !has_metadata {
        return Ok(None);
    }

    let metadata = surface
        .create_property_blob(&pq_rec2020_metadata())
        .context("creating HDR_OUTPUT_METADATA blob")?;

    Ok(Some(HdrMetadata {
        colorspace,
        max_bpc: if has_max_bpc { Some(10) } else { None },
        metadata,
    }))
}
