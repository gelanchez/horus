//! Zenoh transport configuration
//!
//! Configuration for Zenoh backend connections, including:
//! - Connection endpoints (routers, peers)
//! - Namespace and topic mapping
//! - ROS2 compatibility settings
//! - QoS and reliability options

use std::path::PathBuf;
use std::time::Duration;

/// Configuration for Zenoh backend
#[derive(Debug, Clone)]
pub struct ZenohConfig {
    /// Path to zenoh config file (optional)
    /// If provided, loads full zenoh configuration from file
    pub config_path: Option<PathBuf>,

    /// Namespace prefix for all topics
    /// Default: "horus"
    /// Topics will be published as: {namespace}/{topic_name}
    pub namespace: Option<String>,

    /// Enable ROS2 compatibility mode
    /// When true, uses ROS2-compatible topic naming and message formats
    pub ros2_mode: bool,

    /// ROS2 domain ID (0-232)
    /// Only used when ros2_mode is true
    pub ros2_domain_id: u32,

    /// Connection mode
    pub mode: ZenohMode,

    /// Endpoints to connect to (routers or peers)
    /// Format: "tcp/192.168.1.1:7447" or "udp/192.168.1.1:7447"
    pub connect: Vec<String>,

    /// Endpoints to listen on
    /// Format: "tcp/0.0.0.0:7447"
    pub listen: Vec<String>,

    /// Session timeout
    pub session_timeout: Duration,

    /// Enable shared memory transport for local communication
    pub shared_memory: bool,

    /// QoS settings
    pub qos: ZenohQos,

    /// Serialization format
    pub serialization: SerializationFormat,
}

impl Default for ZenohConfig {
    fn default() -> Self {
        Self {
            config_path: None,
            namespace: Some("horus".to_string()),
            ros2_mode: false,
            ros2_domain_id: 0,
            mode: ZenohMode::Peer,
            connect: Vec::new(),
            listen: Vec::new(),
            session_timeout: Duration::from_secs(10),
            shared_memory: true,
            qos: ZenohQos::default(),
            serialization: SerializationFormat::Bincode,
        }
    }
}

impl ZenohConfig {
    /// Create a new default configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Create configuration for ROS2 interop
    pub fn ros2(domain_id: u32) -> Self {
        Self {
            ros2_mode: true,
            ros2_domain_id: domain_id,
            namespace: None,                         // Use ROS2 native naming
            serialization: SerializationFormat::Cdr, // ROS2 uses CDR
            ..Default::default()
        }
    }

    /// Create configuration for cloud/remote connection
    pub fn cloud(router_endpoint: &str) -> Self {
        Self {
            mode: ZenohMode::Client,
            connect: vec![router_endpoint.to_string()],
            ..Default::default()
        }
    }

    /// Create configuration for local mesh
    pub fn local_mesh() -> Self {
        Self {
            mode: ZenohMode::Peer,
            shared_memory: true,
            ..Default::default()
        }
    }

    /// Set the namespace
    pub fn with_namespace(mut self, namespace: &str) -> Self {
        self.namespace = Some(namespace.to_string());
        self
    }

    /// Add a connection endpoint
    pub fn connect_to(mut self, endpoint: &str) -> Self {
        self.connect.push(endpoint.to_string());
        self
    }

    /// Add a listen endpoint
    pub fn listen_on(mut self, endpoint: &str) -> Self {
        self.listen.push(endpoint.to_string());
        self
    }

    /// Enable ROS2 mode
    pub fn with_ros2(mut self, domain_id: u32) -> Self {
        self.ros2_mode = true;
        self.ros2_domain_id = domain_id;
        self
    }

    /// Load from config file
    pub fn from_file(path: impl Into<PathBuf>) -> Self {
        Self {
            config_path: Some(path.into()),
            ..Default::default()
        }
    }

    /// Get the full key expression for a topic
    pub fn topic_to_key_expr(&self, topic: &str) -> String {
        if self.ros2_mode {
            // ROS2 naming: rt/{topic} for topics, rq/{service}/Request for services
            if topic.starts_with('/') {
                format!("rt{}", topic)
            } else {
                format!("rt/{}", topic)
            }
        } else {
            // HORUS naming: {namespace}/{topic}
            match &self.namespace {
                Some(ns) => format!("{}/{}", ns, topic),
                None => topic.to_string(),
            }
        }
    }

    /// Parse a HORUS topic from a Zenoh key expression
    pub fn key_expr_to_topic(&self, key_expr: &str) -> Option<String> {
        if self.ros2_mode {
            // Strip "rt/" prefix
            key_expr
                .strip_prefix("rt/")
                .or_else(|| key_expr.strip_prefix("rt"))
                .map(|s| s.to_string())
        } else {
            // Strip namespace prefix
            match &self.namespace {
                Some(ns) => {
                    let prefix = format!("{}/", ns);
                    key_expr.strip_prefix(&prefix).map(|s| s.to_string())
                }
                None => Some(key_expr.to_string()),
            }
        }
    }
}

/// Zenoh session mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZenohMode {
    /// Peer mode: discover and connect to other peers
    Peer,
    /// Client mode: connect only to routers
    Client,
    /// Router mode: accept connections from clients and peers
    Router,
}

impl Default for ZenohMode {
    fn default() -> Self {
        Self::Peer
    }
}

/// QoS settings for Zenoh
///
/// This struct provides comprehensive QoS configuration compatible with ROS2 DDS policies.
/// It maps ROS2 QoS settings to Zenoh equivalents for interoperability.
#[derive(Debug, Clone)]
pub struct ZenohQos {
    /// Reliability: best-effort or reliable
    pub reliability: Reliability,
    /// Congestion control
    pub congestion: CongestionControl,
    /// Priority (0-7, higher = more important)
    pub priority: u8,
    /// Express mode (skip batching for low latency)
    pub express: bool,
    /// History policy: how many samples to keep
    pub history: HistoryPolicy,
    /// Durability: whether late joiners receive historical data
    pub durability: Durability,
    /// Deadline: expected maximum time between messages
    pub deadline: Option<Duration>,
    /// Lifespan: how long messages remain valid
    pub lifespan: Option<Duration>,
    /// Liveliness: how entities assert their presence
    pub liveliness: Liveliness,
    /// Liveliness lease duration (how long before entity is considered not alive)
    pub liveliness_lease_duration: Option<Duration>,
}

impl Default for ZenohQos {
    fn default() -> Self {
        Self {
            reliability: Reliability::BestEffort,
            congestion: CongestionControl::Drop,
            priority: 5,
            express: false,
            history: HistoryPolicy::KeepLast(1),
            durability: Durability::Volatile,
            deadline: None,
            lifespan: None,
            liveliness: Liveliness::Automatic,
            liveliness_lease_duration: None,
        }
    }
}

impl ZenohQos {
    /// Create QoS for real-time data (low latency, may drop)
    pub fn realtime() -> Self {
        Self {
            reliability: Reliability::BestEffort,
            congestion: CongestionControl::Drop,
            priority: 7,
            express: true,
            history: HistoryPolicy::KeepLast(1),
            durability: Durability::Volatile,
            deadline: None,
            lifespan: None,
            liveliness: Liveliness::Automatic,
            liveliness_lease_duration: None,
        }
    }

    /// Create QoS for reliable data (no loss, may delay)
    pub fn reliable() -> Self {
        Self {
            reliability: Reliability::Reliable,
            congestion: CongestionControl::Block,
            priority: 5,
            express: false,
            history: HistoryPolicy::KeepLast(10),
            durability: Durability::Volatile,
            deadline: None,
            lifespan: None,
            liveliness: Liveliness::Automatic,
            liveliness_lease_duration: None,
        }
    }

    /// Create QoS for bulk data (throughput optimized)
    pub fn bulk() -> Self {
        Self {
            reliability: Reliability::Reliable,
            congestion: CongestionControl::Block,
            priority: 1,
            express: false,
            history: HistoryPolicy::KeepLast(100),
            durability: Durability::Volatile,
            deadline: None,
            lifespan: None,
            liveliness: Liveliness::Automatic,
            liveliness_lease_duration: None,
        }
    }

    /// Create QoS for sensor data (common ROS2 pattern)
    /// Best-effort, keep last sample, volatile
    pub fn sensor_data() -> Self {
        Self {
            reliability: Reliability::BestEffort,
            congestion: CongestionControl::Drop,
            priority: 5,
            express: false,
            history: HistoryPolicy::KeepLast(5),
            durability: Durability::Volatile,
            deadline: Some(Duration::from_millis(100)), // Expect data every 100ms
            lifespan: Some(Duration::from_millis(200)), // Data valid for 200ms
            liveliness: Liveliness::Automatic,
            liveliness_lease_duration: Some(Duration::from_secs(1)),
        }
    }

    /// Create QoS for parameters (ROS2 pattern)
    /// Reliable, transient local durability
    pub fn parameters() -> Self {
        Self {
            reliability: Reliability::Reliable,
            congestion: CongestionControl::Block,
            priority: 5,
            express: false,
            history: HistoryPolicy::KeepLast(1),
            durability: Durability::TransientLocal,
            deadline: None,
            lifespan: None,
            liveliness: Liveliness::Automatic,
            liveliness_lease_duration: None,
        }
    }

    /// Create QoS for services (ROS2 pattern)
    /// Reliable, volatile, keep all history
    pub fn services() -> Self {
        Self {
            reliability: Reliability::Reliable,
            congestion: CongestionControl::Block,
            priority: 6,
            express: false,
            history: HistoryPolicy::KeepAll,
            durability: Durability::Volatile,
            deadline: None,
            lifespan: None,
            liveliness: Liveliness::Automatic,
            liveliness_lease_duration: None,
        }
    }

    /// Create QoS for actions (ROS2 pattern)
    /// Reliable with feedback history
    pub fn actions() -> Self {
        Self {
            reliability: Reliability::Reliable,
            congestion: CongestionControl::Block,
            priority: 5,
            express: false,
            history: HistoryPolicy::KeepLast(10),
            durability: Durability::Volatile,
            deadline: None,
            lifespan: None,
            liveliness: Liveliness::Automatic,
            liveliness_lease_duration: None,
        }
    }

    /// Create QoS for system default (matches ROS2 default)
    pub fn system_default() -> Self {
        Self {
            reliability: Reliability::Reliable,
            congestion: CongestionControl::Block,
            priority: 5,
            express: false,
            history: HistoryPolicy::KeepLast(10),
            durability: Durability::Volatile,
            deadline: None,
            lifespan: None,
            liveliness: Liveliness::Automatic,
            liveliness_lease_duration: None,
        }
    }

    // Builder methods for customization

    /// Set reliability
    pub fn with_reliability(mut self, reliability: Reliability) -> Self {
        self.reliability = reliability;
        self
    }

    /// Set history policy
    pub fn with_history(mut self, history: HistoryPolicy) -> Self {
        self.history = history;
        self
    }

    /// Set durability
    pub fn with_durability(mut self, durability: Durability) -> Self {
        self.durability = durability;
        self
    }

    /// Set deadline
    pub fn with_deadline(mut self, deadline: Duration) -> Self {
        self.deadline = Some(deadline);
        self
    }

    /// Set lifespan
    pub fn with_lifespan(mut self, lifespan: Duration) -> Self {
        self.lifespan = Some(lifespan);
        self
    }

    /// Set liveliness policy
    pub fn with_liveliness(
        mut self,
        liveliness: Liveliness,
        lease_duration: Option<Duration>,
    ) -> Self {
        self.liveliness = liveliness;
        self.liveliness_lease_duration = lease_duration;
        self
    }

    /// Set priority
    pub fn with_priority(mut self, priority: u8) -> Self {
        self.priority = priority.min(7);
        self
    }

    /// Set express mode
    pub fn with_express(mut self, express: bool) -> Self {
        self.express = express;
        self
    }

    /// Check if deadline has been missed
    pub fn check_deadline(&self, last_message_time: std::time::Instant) -> bool {
        if let Some(deadline) = self.deadline {
            last_message_time.elapsed() > deadline
        } else {
            false
        }
    }

    /// Check if message has expired based on lifespan
    pub fn is_message_expired(&self, message_time: std::time::Instant) -> bool {
        if let Some(lifespan) = self.lifespan {
            message_time.elapsed() > lifespan
        } else {
            false
        }
    }
}

/// History policy for QoS
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HistoryPolicy {
    /// Keep last N samples
    KeepLast(usize),
    /// Keep all samples (bounded by resource limits)
    KeepAll,
}

impl Default for HistoryPolicy {
    fn default() -> Self {
        Self::KeepLast(1)
    }
}

impl HistoryPolicy {
    /// Get the depth (number of samples to keep)
    pub fn depth(&self) -> Option<usize> {
        match self {
            Self::KeepLast(n) => Some(*n),
            Self::KeepAll => None,
        }
    }
}

/// Durability policy for QoS
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Durability {
    /// No historical data for late joiners
    Volatile,
    /// Publisher keeps historical data for late-joining subscribers
    TransientLocal,
}

impl Default for Durability {
    fn default() -> Self {
        Self::Volatile
    }
}

/// Liveliness policy for QoS
///
/// Configures how entities assert their presence in the system.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Liveliness {
    /// System automatically asserts liveliness
    Automatic,
    /// User must manually assert liveliness on the participant
    ManualByParticipant,
    /// User must manually assert liveliness on each topic
    ManualByTopic,
}

impl Default for Liveliness {
    fn default() -> Self {
        Self::Automatic
    }
}

impl Liveliness {
    /// Check if manual assertion is required
    pub fn requires_manual_assertion(&self) -> bool {
        !matches!(self, Self::Automatic)
    }
}

/// Reliability mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reliability {
    /// Best-effort delivery (may lose messages)
    BestEffort,
    /// Reliable delivery (guarantees delivery)
    Reliable,
}

/// Congestion control mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CongestionControl {
    /// Drop messages when congested
    Drop,
    /// Block sender when congested
    Block,
}

/// Serialization format for messages
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SerializationFormat {
    /// Bincode (HORUS native, fastest)
    Bincode,
    /// CDR (ROS2 compatible)
    Cdr,
    /// JSON (human-readable, debugging)
    Json,
    /// MessagePack (compact, cross-language)
    MessagePack,
}

impl Default for SerializationFormat {
    fn default() -> Self {
        Self::Bincode
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = ZenohConfig::default();
        assert_eq!(config.namespace, Some("horus".to_string()));
        assert!(!config.ros2_mode);
        assert!(config.shared_memory);
    }

    #[test]
    fn test_topic_to_key_expr() {
        let config = ZenohConfig::default();
        assert_eq!(config.topic_to_key_expr("odom"), "horus/odom");
        assert_eq!(
            config.topic_to_key_expr("sensors/lidar"),
            "horus/sensors/lidar"
        );
    }

    #[test]
    fn test_ros2_topic_naming() {
        let config = ZenohConfig::ros2(0);
        assert_eq!(config.topic_to_key_expr("/cmd_vel"), "rt/cmd_vel");
        assert_eq!(config.topic_to_key_expr("odom"), "rt/odom");
    }

    #[test]
    fn test_key_expr_to_topic() {
        let config = ZenohConfig::default();
        assert_eq!(
            config.key_expr_to_topic("horus/odom"),
            Some("odom".to_string())
        );

        let ros2_config = ZenohConfig::ros2(0);
        assert_eq!(
            ros2_config.key_expr_to_topic("rt/cmd_vel"),
            Some("cmd_vel".to_string())
        );
    }

    #[test]
    fn test_cloud_config() {
        let config = ZenohConfig::cloud("tcp/cloud.example.com:7447");
        assert_eq!(config.mode, ZenohMode::Client);
        assert_eq!(config.connect, vec!["tcp/cloud.example.com:7447"]);
    }

    #[test]
    fn test_builder_pattern() {
        let config = ZenohConfig::new()
            .with_namespace("robot1")
            .connect_to("tcp/192.168.1.100:7447")
            .listen_on("tcp/0.0.0.0:7447");

        assert_eq!(config.namespace, Some("robot1".to_string()));
        assert_eq!(config.connect.len(), 1);
        assert_eq!(config.listen.len(), 1);
    }

    #[test]
    fn test_qos_presets() {
        let rt_qos = ZenohQos::realtime();
        assert!(rt_qos.express);
        assert_eq!(rt_qos.priority, 7);

        let reliable_qos = ZenohQos::reliable();
        assert_eq!(reliable_qos.reliability, Reliability::Reliable);
    }

    #[test]
    fn test_qos_deadline() {
        let qos = ZenohQos::default().with_deadline(std::time::Duration::from_millis(100));
        assert_eq!(qos.deadline, Some(std::time::Duration::from_millis(100)));
    }

    #[test]
    fn test_qos_lifespan() {
        let qos = ZenohQos::default().with_lifespan(std::time::Duration::from_secs(60));
        assert_eq!(qos.lifespan, Some(std::time::Duration::from_secs(60)));
    }

    #[test]
    fn test_qos_liveliness() {
        let qos = ZenohQos::default().with_liveliness(
            Liveliness::ManualByTopic,
            Some(std::time::Duration::from_secs(1)),
        );
        assert_eq!(qos.liveliness, Liveliness::ManualByTopic);
        assert_eq!(
            qos.liveliness_lease_duration,
            Some(std::time::Duration::from_secs(1))
        );
    }

    #[test]
    fn test_qos_history() {
        let keep_last = ZenohQos::default().with_history(HistoryPolicy::KeepLast(10));
        assert_eq!(keep_last.history, HistoryPolicy::KeepLast(10));

        let keep_all = ZenohQos::default().with_history(HistoryPolicy::KeepAll);
        assert_eq!(keep_all.history, HistoryPolicy::KeepAll);
    }

    #[test]
    fn test_qos_durability() {
        let volatile = ZenohQos::default().with_durability(Durability::Volatile);
        assert_eq!(volatile.durability, Durability::Volatile);

        let transient = ZenohQos::default().with_durability(Durability::TransientLocal);
        assert_eq!(transient.durability, Durability::TransientLocal);
    }

    #[test]
    fn test_qos_sensor_preset() {
        let qos = ZenohQos::sensor_data();
        assert_eq!(qos.reliability, Reliability::BestEffort);
        assert_eq!(qos.history, HistoryPolicy::KeepLast(5));
        assert!(qos.deadline.is_some()); // Has deadline constraint
        assert!(qos.lifespan.is_some()); // Has lifespan constraint
    }

    #[test]
    fn test_qos_parameters_preset() {
        let qos = ZenohQos::parameters();
        assert_eq!(qos.reliability, Reliability::Reliable);
        assert_eq!(qos.durability, Durability::TransientLocal);
    }

    #[test]
    fn test_qos_services_preset() {
        let qos = ZenohQos::services();
        assert_eq!(qos.reliability, Reliability::Reliable);
        assert_eq!(qos.history, HistoryPolicy::KeepAll); // Services need all messages
    }

    #[test]
    fn test_qos_actions_preset() {
        let qos = ZenohQos::actions();
        assert_eq!(qos.reliability, Reliability::Reliable);
        assert_eq!(qos.history, HistoryPolicy::KeepLast(10)); // Actions use bounded history
    }
}
