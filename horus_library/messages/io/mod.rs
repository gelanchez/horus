//! Industrial I/O and communication message types for robotics
//!
//! This module provides messages for digital/analog I/O, industrial protocols,
//! and integration with PLCs, SCADA systems, and factory automation.

mod analog;
mod can;
mod digital;
mod ethernetip;
mod i2c;
mod modbus;
mod network;
mod safety;
mod serial;
mod spi;

pub use analog::AnalogIO;
pub use can::CanFrame;
pub use digital::DigitalIO;
pub use ethernetip::EtherNetIPMessage;
pub use i2c::I2cMessage;
pub use modbus::ModbusMessage;
pub use network::NetworkStatus;
pub use safety::SafetyRelayStatus;
pub use serial::SerialData;
pub use spi::SpiMessage;

#[cfg(test)]
mod tests;
