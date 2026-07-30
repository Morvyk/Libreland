//! `org.freedesktop.impl.portal.Lockdown` — administrative "you may not"
//! switches.
//!
//! Each property turns a whole portal off for every app: printing, saving to
//! disk, launching handlers, location, camera, microphone, sound output. The
//! frontend reads them and refuses the matching requests before they ever
//! reach a backend.
//!
//! GNOME stores these in `GSettings`; ours live in the same
//! `$XDG_CONFIG_HOME/libreland/portal.conf` as the appearance settings, under
//! a `[lockdown]` section:
//!
//! ```ini
//! [lockdown]
//! disable-camera = true
//! disable-microphone = true
//! ```
//!
//! The properties are declared writable because the interface says so, but a
//! write is refused: the point of a lockdown is that an app can't undo it, and
//! nothing in this desktop has a settings UI that would legitimately set them.

use zbus::{fdo, interface};

use super::settings;

pub struct Lockdown;

impl Lockdown {
    pub const fn new() -> Self {
        Self
    }

    /// Read one `[lockdown]` flag from the settings keyfile.
    fn flag(key: &str) -> bool {
        let state = settings::state();
        state.bool_in("lockdown", key)
    }

    fn refuse() -> fdo::Result<()> {
        Err(fdo::Error::PropertyReadOnly(
            "lockdown flags are set in libreland/portal.conf, not over D-Bus".into(),
        ))
    }
}

#[interface(name = "org.freedesktop.impl.portal.Lockdown")]
impl Lockdown {
    #[zbus(property, name = "disable-printing")]
    fn disable_printing(&self) -> bool {
        Self::flag("disable-printing")
    }
    #[zbus(property, name = "disable-printing")]
    fn set_disable_printing(&self, _value: bool) -> fdo::Result<()> {
        Self::refuse()
    }

    #[zbus(property, name = "disable-save-to-disk")]
    fn disable_save_to_disk(&self) -> bool {
        Self::flag("disable-save-to-disk")
    }
    #[zbus(property, name = "disable-save-to-disk")]
    fn set_disable_save_to_disk(&self, _value: bool) -> fdo::Result<()> {
        Self::refuse()
    }

    #[zbus(property, name = "disable-application-handlers")]
    fn disable_application_handlers(&self) -> bool {
        Self::flag("disable-application-handlers")
    }
    #[zbus(property, name = "disable-application-handlers")]
    fn set_disable_application_handlers(&self, _value: bool) -> fdo::Result<()> {
        Self::refuse()
    }

    #[zbus(property, name = "disable-location")]
    fn disable_location(&self) -> bool {
        Self::flag("disable-location")
    }
    #[zbus(property, name = "disable-location")]
    fn set_disable_location(&self, _value: bool) -> fdo::Result<()> {
        Self::refuse()
    }

    #[zbus(property, name = "disable-camera")]
    fn disable_camera(&self) -> bool {
        Self::flag("disable-camera")
    }
    #[zbus(property, name = "disable-camera")]
    fn set_disable_camera(&self, _value: bool) -> fdo::Result<()> {
        Self::refuse()
    }

    #[zbus(property, name = "disable-microphone")]
    fn disable_microphone(&self) -> bool {
        Self::flag("disable-microphone")
    }
    #[zbus(property, name = "disable-microphone")]
    fn set_disable_microphone(&self, _value: bool) -> fdo::Result<()> {
        Self::refuse()
    }

    #[zbus(property, name = "disable-sound-output")]
    fn disable_sound_output(&self) -> bool {
        Self::flag("disable-sound-output")
    }
    #[zbus(property, name = "disable-sound-output")]
    fn set_disable_sound_output(&self, _value: bool) -> fdo::Result<()> {
        Self::refuse()
    }

    #[zbus(property, name = "version")]
    fn version(&self) -> u32 {
        1
    }
}
