//! SPI bus and device discovery for HORUS (Linux only).
//!
//! Scans for available SPI buses and chip selects.

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

/// SPI bus information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpiBus {
    /// Bus number (e.g., 0 for spidev0.x)
    pub bus_number: u8,
    /// Device path base (e.g., /dev/spidev0)
    pub device_path: PathBuf,
    /// Controller name from sysfs
    pub controller_name: Option<String>,
    /// Maximum speed supported (Hz)
    pub max_speed_hz: Option<u32>,
    /// Available chip selects
    pub chip_selects: Vec<SpiChipSelect>,
}

impl SpiBus {
    /// Get all device paths for this bus
    pub fn device_paths(&self) -> Vec<PathBuf> {
        self.chip_selects
            .iter()
            .map(|cs| cs.device_path.clone())
            .collect()
    }
}

/// SPI chip select (individual device endpoint)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpiChipSelect {
    /// Bus number
    pub bus_number: u8,
    /// Chip select number
    pub cs_number: u8,
    /// Full device path (e.g., /dev/spidev0.0)
    pub device_path: PathBuf,
    /// Whether the device file exists and is accessible
    pub accessible: bool,
}

impl SpiChipSelect {
    /// Get device specification string
    pub fn device_spec(&self) -> String {
        format!("spidev{}.{}", self.bus_number, self.cs_number)
    }
}

/// SPI discovery
pub struct SpiDiscovery {
    /// Discovered buses
    buses: Vec<SpiBus>,
}

impl SpiDiscovery {
    /// Create a new SPI discovery instance
    pub fn new() -> Self {
        Self { buses: Vec::new() }
    }

    /// Enumerate SPI buses and chip selects
    pub fn enumerate_buses(&mut self) -> Vec<SpiBus> {
        self.buses.clear();

        // Track found bus/cs pairs
        let mut found: std::collections::HashMap<u8, Vec<SpiChipSelect>> =
            std::collections::HashMap::new();

        // Scan /dev for spidev* devices
        if let Ok(entries) = fs::read_dir("/dev") {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                if name.starts_with("spidev") {
                    // Parse spidevX.Y format
                    if let Some((bus, cs)) = self.parse_spidev_name(&name) {
                        let device_path = entry.path();
                        let accessible = device_path.exists()
                            && fs::OpenOptions::new().read(true).open(&device_path).is_ok();

                        let chip_select = SpiChipSelect {
                            bus_number: bus,
                            cs_number: cs,
                            device_path,
                            accessible,
                        };

                        found.entry(bus).or_default().push(chip_select);
                    }
                }
            }
        }

        // Also check sysfs for SPI controllers
        if let Ok(entries) = fs::read_dir("/sys/class/spi_master") {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                if let Some(suffix) = name.strip_prefix("spi") {
                    if let Ok(bus_num) = suffix.parse::<u8>() {
                        // Ensure bus entry exists
                        found.entry(bus_num).or_default();
                    }
                }
            }
        }

        // Build bus list
        for (bus_num, chip_selects) in found {
            let controller_name = self.get_controller_name(bus_num);
            let max_speed = self.get_max_speed(bus_num);

            let mut cs_list = chip_selects;
            cs_list.sort_by_key(|cs| cs.cs_number);

            self.buses.push(SpiBus {
                bus_number: bus_num,
                device_path: PathBuf::from(format!("/dev/spidev{}", bus_num)),
                controller_name,
                max_speed_hz: max_speed,
                chip_selects: cs_list,
            });
        }

        // Sort by bus number
        self.buses.sort_by_key(|b| b.bus_number);
        self.buses.clone()
    }

    /// Get cached buses
    pub fn buses(&self) -> &[SpiBus] {
        &self.buses
    }

    /// Get total chip select count
    pub fn chip_select_count(&self) -> usize {
        self.buses.iter().map(|b| b.chip_selects.len()).sum()
    }

    fn parse_spidev_name(&self, name: &str) -> Option<(u8, u8)> {
        // Parse spidevX.Y
        let stripped = name.strip_prefix("spidev")?;
        let parts: Vec<&str> = stripped.split('.').collect();
        if parts.len() == 2 {
            let bus = parts[0].parse::<u8>().ok()?;
            let cs = parts[1].parse::<u8>().ok()?;
            Some((bus, cs))
        } else {
            None
        }
    }

    fn get_controller_name(&self, bus_number: u8) -> Option<String> {
        // Try to read controller name from sysfs
        let paths = [
            format!(
                "/sys/class/spi_master/spi{}/device/of_node/name",
                bus_number
            ),
            format!("/sys/class/spi_master/spi{}/device/driver/name", bus_number),
        ];

        for path in &paths {
            if let Ok(name) = fs::read_to_string(path) {
                let name = name.trim().trim_end_matches('\0').to_string();
                if !name.is_empty() {
                    return Some(self.friendly_controller_name(&name));
                }
            }
        }

        // Try to identify from compatible string
        let compat_path = format!(
            "/sys/class/spi_master/spi{}/device/of_node/compatible",
            bus_number
        );
        if let Ok(compat) = fs::read_to_string(compat_path) {
            return Some(self.friendly_controller_name(compat.trim()));
        }

        None
    }

    fn friendly_controller_name(&self, raw: &str) -> String {
        if raw.contains("bcm2835") || raw.contains("bcm2711") {
            "Raspberry Pi SPI".to_string()
        } else if raw.contains("tegra") {
            "NVIDIA Tegra SPI".to_string()
        } else if raw.contains("omap") || raw.contains("ti,") {
            "TI OMAP SPI".to_string()
        } else if raw.contains("imx") || raw.contains("fsl,") {
            "NXP i.MX SPI".to_string()
        } else if raw.contains("rockchip") {
            "Rockchip SPI".to_string()
        } else if raw.contains("allwinner") || raw.contains("sun") {
            "Allwinner SPI".to_string()
        } else if raw.contains("designware") || raw.contains("dw-apb") {
            "DesignWare SPI".to_string()
        } else {
            raw.to_string()
        }
    }

    fn get_max_speed(&self, bus_number: u8) -> Option<u32> {
        // Try to read max speed from sysfs
        let speed_path = format!("/sys/class/spi_master/spi{}/max_speed_hz", bus_number);
        fs::read_to_string(speed_path)
            .ok()
            .and_then(|s| s.trim().parse().ok())
    }
}

impl Default for SpiDiscovery {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_enumerate_buses() {
        let mut discovery = SpiDiscovery::new();
        let buses = discovery.enumerate_buses();

        println!("Found {} SPI buses:", buses.len());
        for bus in &buses {
            println!(
                "  SPI{}: {} ({} chip selects)",
                bus.bus_number,
                bus.controller_name.as_deref().unwrap_or("Unknown"),
                bus.chip_selects.len()
            );
            for cs in &bus.chip_selects {
                println!(
                    "    CS{}: {} (accessible: {})",
                    cs.cs_number,
                    cs.device_path.display(),
                    cs.accessible
                );
            }
        }
    }

    #[test]
    fn test_parse_spidev_name() {
        let discovery = SpiDiscovery::new();

        assert_eq!(discovery.parse_spidev_name("spidev0.0"), Some((0, 0)));
        assert_eq!(discovery.parse_spidev_name("spidev0.1"), Some((0, 1)));
        assert_eq!(discovery.parse_spidev_name("spidev1.0"), Some((1, 0)));
        assert_eq!(discovery.parse_spidev_name("spidev32.3"), Some((32, 3)));
        assert_eq!(discovery.parse_spidev_name("spidev"), None);
        assert_eq!(discovery.parse_spidev_name("spi0.0"), None);
    }
}
