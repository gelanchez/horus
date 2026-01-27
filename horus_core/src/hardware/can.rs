//! SocketCAN interface discovery for HORUS (Linux only).
//!
//! Scans for available CAN bus interfaces using the SocketCAN subsystem.

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

/// CAN interface state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CanState {
    /// Interface is down/stopped
    Stopped,
    /// Interface is up and running
    Running,
    /// Interface is in error state
    Error,
    /// Bus-off state (too many errors)
    BusOff,
    /// Unknown state
    Unknown,
}

impl From<&str> for CanState {
    fn from(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "stopped" => CanState::Stopped,
            "error-active" | "error-warning" | "error-passive" => CanState::Running,
            "bus-off" => CanState::BusOff,
            "sleeping" => CanState::Stopped,
            _ => CanState::Unknown,
        }
    }
}

/// CAN interface type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CanInterfaceType {
    /// Physical CAN hardware
    Physical,
    /// Virtual CAN interface (vcan)
    Virtual,
    /// CAN over Serial Line (slcan)
    SlCan,
    /// Peak PCAN USB adapters
    Peak,
    /// Kvaser adapters
    Kvaser,
    /// CANable/candleLight USB adapters
    GsUsb,
    /// MCP251x SPI CAN controllers
    Mcp251x,
    /// Other/unknown type
    Other,
}

/// CAN interface information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CanInterface {
    /// Interface name (e.g., "can0", "vcan0")
    pub name: String,
    /// Interface type
    pub interface_type: CanInterfaceType,
    /// Current state
    pub state: CanState,
    /// Configured bitrate (if available)
    pub bitrate: Option<u32>,
    /// Sample point (percentage)
    pub sample_point: Option<f32>,
    /// TX queue length
    pub txqueuelen: Option<u32>,
    /// Driver name
    pub driver: Option<String>,
    /// Hardware description
    pub description: Option<String>,
    /// Whether the interface is up
    pub is_up: bool,
    /// Whether loopback is enabled
    pub loopback: bool,
    /// Whether listen-only mode is enabled
    pub listen_only: bool,
    /// CAN FD (Flexible Data-rate) support
    pub fd_support: bool,
    /// RX error count
    pub rx_errors: u64,
    /// TX error count
    pub tx_errors: u64,
}

impl CanInterface {
    /// Get device specification string
    pub fn device_spec(&self) -> String {
        self.name.clone()
    }

    /// Check if this is a virtual interface
    pub fn is_virtual(&self) -> bool {
        matches!(self.interface_type, CanInterfaceType::Virtual)
    }
}

/// CAN interface discovery
pub struct CanDiscovery {
    /// Discovered interfaces
    interfaces: Vec<CanInterface>,
}

impl CanDiscovery {
    /// Create a new CAN discovery instance
    pub fn new() -> Self {
        Self {
            interfaces: Vec::new(),
        }
    }

    /// Enumerate CAN interfaces
    pub fn enumerate_interfaces(&mut self) -> Vec<CanInterface> {
        self.interfaces.clear();

        // Scan /sys/class/net for CAN interfaces
        if let Ok(entries) = fs::read_dir("/sys/class/net") {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().to_string();

                // Check if this is a CAN interface
                if self.is_can_interface(&name) {
                    if let Some(interface) = self.probe_interface(&name) {
                        self.interfaces.push(interface);
                    }
                }
            }
        }

        // Sort by name
        self.interfaces.sort_by(|a, b| {
            // Sort virtual interfaces last
            match (a.is_virtual(), b.is_virtual()) {
                (true, false) => std::cmp::Ordering::Greater,
                (false, true) => std::cmp::Ordering::Less,
                _ => a.name.cmp(&b.name),
            }
        });

        self.interfaces.clone()
    }

    /// Get cached interfaces
    pub fn interfaces(&self) -> &[CanInterface] {
        &self.interfaces
    }

    /// Get physical (non-virtual) interfaces
    pub fn physical_interfaces(&self) -> Vec<&CanInterface> {
        self.interfaces.iter().filter(|i| !i.is_virtual()).collect()
    }

    /// Get interface count
    pub fn interface_count(&self) -> usize {
        self.interfaces.len()
    }

    fn is_can_interface(&self, name: &str) -> bool {
        // Check if it's a CAN interface by looking at the type file
        let type_path = format!("/sys/class/net/{}/type", name);
        if let Ok(type_str) = fs::read_to_string(&type_path) {
            // ARPHRD_CAN = 280
            if type_str.trim() == "280" {
                return true;
            }
        }

        // Also check by name pattern
        name.starts_with("can")
            || name.starts_with("vcan")
            || name.starts_with("slcan")
            || name.starts_with("peak")
    }

    fn probe_interface(&self, name: &str) -> Option<CanInterface> {
        let base_path = format!("/sys/class/net/{}", name);

        // Determine interface type
        let interface_type = self.detect_interface_type(name);

        // Get state
        let state = self.get_state(name);

        // Get bitrate
        let bitrate = self.get_bitrate(name);

        // Get sample point
        let sample_point = self.get_sample_point(name);

        // Get TX queue length
        let txqueuelen = self.read_sysfs_u32(&format!("{}/tx_queue_len", base_path));

        // Get driver
        let driver = self.get_driver(name);

        // Get flags
        let flags = self.read_sysfs_string(&format!("{}/flags", base_path));
        let is_up = flags
            .as_ref()
            .map(|f| {
                // Parse hex flags, check IFF_UP (0x1)
                u32::from_str_radix(f.trim_start_matches("0x"), 16)
                    .map(|v| v & 0x1 != 0)
                    .unwrap_or(false)
            })
            .unwrap_or(false);

        // Get CAN-specific settings
        let (loopback, listen_only, fd_support) = self.get_can_settings(name);

        // Get error counters
        let rx_errors = self
            .read_sysfs_u64(&format!("{}/statistics/rx_errors", base_path))
            .unwrap_or(0);
        let tx_errors = self
            .read_sysfs_u64(&format!("{}/statistics/tx_errors", base_path))
            .unwrap_or(0);

        // Get description
        let description = self.get_description(name, &interface_type, &driver);

        Some(CanInterface {
            name: name.to_string(),
            interface_type,
            state,
            bitrate,
            sample_point,
            txqueuelen,
            driver,
            description,
            is_up,
            loopback,
            listen_only,
            fd_support,
            rx_errors,
            tx_errors,
        })
    }

    fn detect_interface_type(&self, name: &str) -> CanInterfaceType {
        // Check by name pattern first
        if name.starts_with("vcan") {
            return CanInterfaceType::Virtual;
        }
        if name.starts_with("slcan") {
            return CanInterfaceType::SlCan;
        }

        // Check driver
        let driver_link = format!("/sys/class/net/{}/device/driver", name);
        if let Ok(target) = fs::read_link(&driver_link) {
            let driver_name = target
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_default();

            return match driver_name.as_str() {
                "peak_usb" | "pcan_usb" => CanInterfaceType::Peak,
                "kvaser_usb" => CanInterfaceType::Kvaser,
                "gs_usb" => CanInterfaceType::GsUsb,
                "mcp251x" | "mcp251xfd" => CanInterfaceType::Mcp251x,
                "vcan" => CanInterfaceType::Virtual,
                "slcan" => CanInterfaceType::SlCan,
                _ => CanInterfaceType::Physical,
            };
        }

        // Check modalias for USB devices
        let modalias_path = format!("/sys/class/net/{}/device/modalias", name);
        if let Ok(modalias) = fs::read_to_string(&modalias_path) {
            if modalias.contains("v0C72") || modalias.contains("peak") {
                return CanInterfaceType::Peak;
            }
            if modalias.contains("v0BFD") || modalias.contains("kvaser") {
                return CanInterfaceType::Kvaser;
            }
            if modalias.contains("v1D50") {
                // candleLight/CANable
                return CanInterfaceType::GsUsb;
            }
        }

        CanInterfaceType::Physical
    }

    fn get_state(&self, name: &str) -> CanState {
        // Try to read CAN state
        let state_path = format!("/sys/class/net/{}/can_state", name);
        if let Ok(state_str) = fs::read_to_string(&state_path) {
            return CanState::from(state_str.trim());
        }

        // Fallback to operstate
        let oper_path = format!("/sys/class/net/{}/operstate", name);
        if let Ok(oper_str) = fs::read_to_string(&oper_path) {
            return match oper_str.trim() {
                "up" => CanState::Running,
                "down" => CanState::Stopped,
                _ => CanState::Unknown,
            };
        }

        CanState::Unknown
    }

    fn get_bitrate(&self, name: &str) -> Option<u32> {
        let bitrate_path = format!("/sys/class/net/{}/can_bittiming/bitrate", name);
        self.read_sysfs_u32(&bitrate_path)
    }

    fn get_sample_point(&self, name: &str) -> Option<f32> {
        let sp_path = format!("/sys/class/net/{}/can_bittiming/sample_point", name);
        self.read_sysfs_u32(&sp_path).map(|v| v as f32 / 1000.0)
    }

    fn get_driver(&self, name: &str) -> Option<String> {
        let driver_link = format!("/sys/class/net/{}/device/driver", name);
        fs::read_link(&driver_link)
            .ok()
            .and_then(|p| p.file_name().map(|n| n.to_string_lossy().to_string()))
    }

    fn get_can_settings(&self, name: &str) -> (bool, bool, bool) {
        let ctrlmode_path = format!("/sys/class/net/{}/can_ctrlmode", name);

        // Default values
        let mut loopback = false;
        let mut listen_only = false;
        let mut fd_support = false;

        // Try to read control mode flags
        if let Ok(content) = fs::read_to_string(&ctrlmode_path) {
            loopback = content.contains("LOOPBACK");
            listen_only = content.contains("LISTENONLY");
            fd_support = content.contains("FD");
        }

        // Also check for FD in bittiming_const
        let fd_path = format!("/sys/class/net/{}/can_bittiming_const_fd", name);
        if PathBuf::from(&fd_path).exists() {
            fd_support = true;
        }

        (loopback, listen_only, fd_support)
    }

    fn get_description(
        &self,
        _name: &str,
        itype: &CanInterfaceType,
        driver: &Option<String>,
    ) -> Option<String> {
        match itype {
            CanInterfaceType::Virtual => Some("Virtual CAN interface for testing".to_string()),
            CanInterfaceType::SlCan => Some("CAN over Serial Line (LAWICEL protocol)".to_string()),
            CanInterfaceType::Peak => Some("PEAK-System PCAN USB adapter".to_string()),
            CanInterfaceType::Kvaser => Some("Kvaser CAN adapter".to_string()),
            CanInterfaceType::GsUsb => Some("CANable/candleLight gs_usb adapter".to_string()),
            CanInterfaceType::Mcp251x => Some("Microchip MCP251x SPI CAN controller".to_string()),
            CanInterfaceType::Physical => driver.as_ref().map(|d| format!("CAN interface ({})", d)),
            CanInterfaceType::Other => None,
        }
    }

    fn read_sysfs_string(&self, path: &str) -> Option<String> {
        fs::read_to_string(path).ok().map(|s| s.trim().to_string())
    }

    fn read_sysfs_u32(&self, path: &str) -> Option<u32> {
        fs::read_to_string(path)
            .ok()
            .and_then(|s| s.trim().parse().ok())
    }

    fn read_sysfs_u64(&self, path: &str) -> Option<u64> {
        fs::read_to_string(path)
            .ok()
            .and_then(|s| s.trim().parse().ok())
    }
}

impl Default for CanDiscovery {
    fn default() -> Self {
        Self::new()
    }
}

/// Format bitrate for display
pub fn format_bitrate(bitrate: u32) -> String {
    if bitrate >= 1_000_000 {
        format!("{} Mbit/s", bitrate / 1_000_000)
    } else if bitrate >= 1_000 {
        format!("{} kbit/s", bitrate / 1_000)
    } else {
        format!("{} bit/s", bitrate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_enumerate_interfaces() {
        let mut discovery = CanDiscovery::new();
        let interfaces = discovery.enumerate_interfaces();

        println!("Found {} CAN interfaces:", interfaces.len());
        for iface in &interfaces {
            println!(
                "  {}: {:?} ({:?})",
                iface.name, iface.interface_type, iface.state
            );
            if let Some(bitrate) = iface.bitrate {
                println!("    Bitrate: {}", format_bitrate(bitrate));
            }
            if let Some(ref desc) = iface.description {
                println!("    Description: {}", desc);
            }
        }
    }

    #[test]
    fn test_can_state_from_string() {
        assert_eq!(CanState::from("stopped"), CanState::Stopped);
        assert_eq!(CanState::from("error-active"), CanState::Running);
        assert_eq!(CanState::from("bus-off"), CanState::BusOff);
        assert_eq!(CanState::from("unknown"), CanState::Unknown);
    }

    #[test]
    fn test_format_bitrate() {
        assert_eq!(format_bitrate(500_000), "500 kbit/s");
        assert_eq!(format_bitrate(1_000_000), "1 Mbit/s");
        assert_eq!(format_bitrate(125_000), "125 kbit/s");
    }
}
