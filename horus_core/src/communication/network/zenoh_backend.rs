//! Zenoh backend for HORUS Hub/Link communication
//!
//! Provides Zenoh-based transport for:
//! - Multi-robot mesh networking
//! - Cloud connectivity
//! - ROS2 interoperability
//!
//! Feature-gated: requires `zenoh-transport` feature

use parking_lot::Mutex;
use std::collections::VecDeque;
use std::marker::PhantomData;
use std::sync::Arc;

use super::zenoh_config::{SerializationFormat, ZenohConfig};
use crate::error::{HorusError, HorusResult};

/// Serialize a message using the specified format
#[cfg(feature = "zenoh-transport")]
fn serialize_message<T: serde::Serialize>(
    msg: &T,
    format: SerializationFormat,
) -> HorusResult<Vec<u8>> {
    match format {
        SerializationFormat::Bincode => bincode::serialize(msg)
            .map_err(|e| HorusError::Communication(format!("Bincode serialize error: {}", e))),
        SerializationFormat::Json => serde_json::to_vec(msg)
            .map_err(|e| HorusError::Communication(format!("JSON serialize error: {}", e))),
        #[cfg(feature = "cdr-encoding")]
        SerializationFormat::Cdr => {
            // ROS2/DDS typically uses little-endian CDR
            cdr_encoding::to_vec::<_, byteorder::LittleEndian>(msg)
                .map_err(|e| HorusError::Communication(format!("CDR serialize error: {}", e)))
        }
        #[cfg(not(feature = "cdr-encoding"))]
        SerializationFormat::Cdr => {
            log::warn!("CDR format requires 'zenoh-ros2' feature, falling back to bincode");
            bincode::serialize(msg)
                .map_err(|e| HorusError::Communication(format!("Bincode serialize error: {}", e)))
        }
        SerializationFormat::MessagePack => rmp_serde::to_vec(msg)
            .map_err(|e| HorusError::Communication(format!("MessagePack serialize error: {}", e))),
    }
}

/// Zenoh backend for HORUS Hub/Link communication
///
/// Provides pub/sub communication over Zenoh protocol.
/// Supports both HORUS-native (bincode) and ROS2-compatible (CDR) serialization.
#[cfg(feature = "zenoh-transport")]
pub struct ZenohBackend<T> {
    /// Zenoh session (shared across publishers/subscribers)
    session: Arc<zenoh::Session>,
    /// Publisher for this topic (stored as boxed to handle lifetime)
    publisher: Option<Box<zenoh::pubsub::Publisher<'static>>>,
    /// Receive buffer for incoming messages
    recv_buffer: Arc<Mutex<VecDeque<T>>>,
    /// Key expression for this topic
    key_expr: String,
    /// Configuration
    config: ZenohConfig,
    /// Phantom data for type parameter
    _phantom: PhantomData<T>,
}

#[cfg(feature = "zenoh-transport")]
impl<T> ZenohBackend<T>
where
    T: serde::Serialize + serde::de::DeserializeOwned + Send + Sync + Clone + 'static,
{
    /// Create a new Zenoh backend for a topic
    ///
    /// This is an async function that must be called from within a tokio runtime.
    pub async fn new(topic: &str, config: ZenohConfig) -> HorusResult<Self> {
        // Build zenoh config - use default for now
        // In zenoh 1.2+, configuration is done differently
        let zenoh_config = zenoh::Config::default();

        // Log mode for debugging
        log::debug!("Creating Zenoh backend with mode: {:?}", config.mode);

        // Add connect endpoints logging
        if !config.connect.is_empty() {
            for endpoint in &config.connect {
                log::debug!("Zenoh will connect to: {}", endpoint);
            }
        }

        // Open session
        let session = zenoh::open(zenoh_config)
            .await
            .map_err(|e| HorusError::Communication(format!("Zenoh open failed: {}", e)))?;

        let key_expr = config.topic_to_key_expr(topic);

        Ok(Self {
            session: Arc::new(session),
            publisher: None,
            recv_buffer: Arc::new(Mutex::new(VecDeque::with_capacity(1024))),
            key_expr,
            config,
            _phantom: PhantomData,
        })
    }

    /// Create a blocking version (for non-async contexts)
    pub fn new_blocking(topic: &str, config: ZenohConfig) -> HorusResult<Self> {
        // Use tokio runtime to run async code
        let rt = tokio::runtime::Handle::try_current()
            .or_else(|_| {
                // Create a new runtime if not in one
                tokio::runtime::Runtime::new().map(|rt| rt.handle().clone())
            })
            .map_err(|e| {
                HorusError::Communication(format!("Failed to get tokio runtime: {}", e))
            })?;

        rt.block_on(Self::new(topic, config))
    }

    /// Initialize publisher for sending messages
    pub async fn init_publisher(&mut self) -> HorusResult<()> {
        if self.publisher.is_some() {
            return Ok(());
        }

        let publisher = self
            .session
            .declare_publisher(&self.key_expr)
            .await
            .map_err(|e| HorusError::Communication(format!("Zenoh publisher error: {}", e)))?;

        // Store publisher - use unsafe transmute to handle lifetime
        // This is safe because the session outlives the publisher
        let static_publisher: zenoh::pubsub::Publisher<'static> =
            unsafe { std::mem::transmute(publisher) };
        self.publisher = Some(Box::new(static_publisher));
        Ok(())
    }

    /// Initialize subscriber for receiving messages
    pub async fn init_subscriber(&mut self) -> HorusResult<()> {
        let recv_buffer = self.recv_buffer.clone();
        let serialization = self.config.serialization;

        let subscriber = self
            .session
            .declare_subscriber(&self.key_expr)
            .await
            .map_err(|e| HorusError::Communication(format!("Zenoh subscriber error: {}", e)))?;

        // Spawn a task to receive messages
        let key_expr = self.key_expr.clone();
        tokio::spawn(async move {
            loop {
                match subscriber.recv_async().await {
                    Ok(sample) => {
                        let payload = sample.payload().to_bytes();
                        let msg: Option<T> = match serialization {
                            SerializationFormat::Bincode => bincode::deserialize(&payload).ok(),
                            SerializationFormat::Json => serde_json::from_slice(&payload).ok(),
                            #[cfg(feature = "cdr-encoding")]
                            SerializationFormat::Cdr => {
                                // ROS2/DDS uses little-endian CDR, returns (value, bytes_read)
                                cdr_encoding::from_bytes::<_, byteorder::LittleEndian>(&payload)
                                    .ok()
                                    .map(|(msg, _bytes_read)| msg)
                            }
                            #[cfg(not(feature = "cdr-encoding"))]
                            SerializationFormat::Cdr => {
                                log::warn!("CDR format requires 'zenoh-ros2' feature, falling back to bincode");
                                bincode::deserialize(&payload).ok()
                            }
                            SerializationFormat::MessagePack => {
                                rmp_serde::from_slice(&payload).ok()
                            }
                        };

                        if let Some(msg) = msg {
                            let mut buffer = recv_buffer.lock();
                            // Keep buffer bounded
                            if buffer.len() >= 10000 {
                                buffer.pop_front();
                            }
                            buffer.push_back(msg);
                        }
                    }
                    Err(e) => {
                        log::warn!("Zenoh subscriber error on {}: {}", key_expr, e);
                        break;
                    }
                }
            }
        });

        Ok(())
    }

    /// Send a message (synchronous wrapper)
    pub fn send(&self, msg: &T) -> HorusResult<()> {
        let publisher = self
            .publisher
            .as_ref()
            .ok_or_else(|| HorusError::Communication("Publisher not initialized".into()))?;

        // Serialize using configured format
        let payload = serialize_message(msg, self.config.serialization)?;

        // Use blocking runtime for sync API compatibility
        let rt = tokio::runtime::Handle::try_current()
            .map_err(|e| HorusError::Communication(format!("No tokio runtime: {}", e)))?;

        rt.block_on(async { publisher.put(payload).await })
            .map_err(|e| HorusError::Communication(format!("Zenoh put error: {}", e)))?;

        Ok(())
    }

    /// Send a message asynchronously
    pub async fn send_async(&self, msg: &T) -> HorusResult<()> {
        let publisher = self
            .publisher
            .as_ref()
            .ok_or_else(|| HorusError::Communication("Publisher not initialized".into()))?;

        let payload = serialize_message(msg, self.config.serialization)?;

        publisher
            .put(payload)
            .await
            .map_err(|e| HorusError::Communication(format!("Zenoh put error: {}", e)))?;

        Ok(())
    }

    /// Receive a message (non-blocking)
    pub fn recv(&self) -> Option<T> {
        let mut buffer = self.recv_buffer.lock();
        buffer.pop_front()
    }

    /// Try to receive a message with timeout
    pub fn recv_timeout(&self, timeout: std::time::Duration) -> Option<T> {
        let start = std::time::Instant::now();
        while start.elapsed() < timeout {
            if let Some(msg) = self.recv() {
                return Some(msg);
            }
            std::thread::sleep(std::time::Duration::from_micros(100));
        }
        None
    }

    /// Get the key expression for this backend
    pub fn key_expr(&self) -> &str {
        &self.key_expr
    }

    /// Get the configuration
    pub fn config(&self) -> &ZenohConfig {
        &self.config
    }

    /// Get the number of pending messages
    pub fn pending_count(&self) -> usize {
        self.recv_buffer.lock().len()
    }

    /// Clear the receive buffer
    pub fn clear_buffer(&self) {
        self.recv_buffer.lock().clear();
    }

    /// Get session info (for debugging)
    pub fn session_info(&self) -> ZenohSessionInfo {
        ZenohSessionInfo {
            key_expr: self.key_expr.clone(),
            has_publisher: self.publisher.is_some(),
            pending_messages: self.pending_count(),
        }
    }
}

#[cfg(feature = "zenoh-transport")]
impl<T> Clone for ZenohBackend<T> {
    fn clone(&self) -> Self {
        Self {
            session: self.session.clone(),
            publisher: None, // Publishers are not cloned, must re-init
            recv_buffer: self.recv_buffer.clone(),
            key_expr: self.key_expr.clone(),
            config: self.config.clone(),
            _phantom: PhantomData,
        }
    }
}

#[cfg(feature = "zenoh-transport")]
impl<T> std::fmt::Debug for ZenohBackend<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ZenohBackend")
            .field("key_expr", &self.key_expr)
            .field("has_publisher", &self.publisher.is_some())
            .field("pending_messages", &self.recv_buffer.lock().len())
            .finish()
    }
}

/// Session information for debugging
#[derive(Debug, Clone)]
pub struct ZenohSessionInfo {
    pub key_expr: String,
    pub has_publisher: bool,
    pub pending_messages: usize,
}

/// Stub implementation when zenoh feature is not enabled
#[cfg(not(feature = "zenoh-transport"))]
pub struct ZenohBackend<T> {
    _phantom: PhantomData<T>,
}

#[cfg(not(feature = "zenoh-transport"))]
impl<T> ZenohBackend<T> {
    pub fn new_blocking(_topic: &str, _config: ZenohConfig) -> HorusResult<Self> {
        Err(HorusError::Communication(
            "Zenoh transport not enabled. Compile with --features zenoh-transport".into(),
        ))
    }

    pub fn send(&self, _msg: &T) -> HorusResult<()> {
        Err(HorusError::Communication("Zenoh not enabled".into()))
    }

    pub fn recv(&self) -> Option<T> {
        None
    }
}

#[cfg(not(feature = "zenoh-transport"))]
impl<T> std::fmt::Debug for ZenohBackend<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ZenohBackend")
            .field("enabled", &false)
            .finish()
    }
}

#[cfg(test)]
#[cfg(feature = "zenoh-transport")]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    struct TestMessage {
        value: i32,
        text: String,
    }

    #[tokio::test]
    async fn test_zenoh_config_topic_mapping() {
        let config = ZenohConfig::default();
        assert_eq!(config.topic_to_key_expr("test"), "horus/test");

        let ros2_config = ZenohConfig::ros2(0);
        assert_eq!(ros2_config.topic_to_key_expr("/cmd_vel"), "rt/cmd_vel");
    }

    // Note: Full integration tests require a running Zenoh router
    // These are marked as ignored and can be run manually
    #[tokio::test]
    #[ignore]
    async fn test_zenoh_pub_sub() {
        let config = ZenohConfig::default();

        let mut publisher = ZenohBackend::<TestMessage>::new("test_topic", config.clone())
            .await
            .expect("Failed to create publisher");
        publisher
            .init_publisher()
            .await
            .expect("Failed to init publisher");

        let mut subscriber = ZenohBackend::<TestMessage>::new("test_topic", config)
            .await
            .expect("Failed to create subscriber");
        subscriber
            .init_subscriber()
            .await
            .expect("Failed to init subscriber");

        // Give time for discovery
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        let msg = TestMessage {
            value: 42,
            text: "hello".to_string(),
        };

        publisher.send_async(&msg).await.expect("Failed to send");

        // Wait for message
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        let received = subscriber.recv();
        assert_eq!(received, Some(msg));
    }

    /// Test that verifies ZenohBackend can be instantiated and session info works
    /// This test doesn't require network connectivity
    /// Note: Zenoh requires multi_thread runtime
    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn test_zenoh_backend_instantiation() {
        let config = ZenohConfig::default();

        // Create a backend - this should work as Zenoh uses peer mode by default
        let backend = ZenohBackend::<TestMessage>::new("instantiation_test", config)
            .await
            .expect("Failed to create backend");

        // Verify session info
        let info = backend.session_info();
        assert_eq!(info.key_expr, "horus/instantiation_test");
        assert!(!info.has_publisher);
        assert_eq!(info.pending_messages, 0);
    }

    /// Test the blocking API wrapper
    #[test]
    fn test_zenoh_blocking_api() {
        // Create a multi-thread tokio runtime (required by Zenoh)
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .enable_all()
            .build()
            .expect("Failed to create runtime");

        rt.block_on(async {
            let config = ZenohConfig::default();
            let backend = ZenohBackend::<TestMessage>::new("blocking_test", config)
                .await
                .expect("Failed to create backend");

            assert_eq!(backend.key_expr(), "horus/blocking_test");
            assert_eq!(backend.pending_count(), 0);
        });
    }

    /// Test buffer management
    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn test_zenoh_buffer_management() {
        let config = ZenohConfig::default();
        let backend = ZenohBackend::<TestMessage>::new("buffer_test", config)
            .await
            .expect("Failed to create backend");

        // Initially empty
        assert_eq!(backend.pending_count(), 0);
        assert!(backend.recv().is_none());

        // Clear should work on empty buffer
        backend.clear_buffer();
        assert_eq!(backend.pending_count(), 0);
    }

    /// Test Debug impl
    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn test_zenoh_debug_impl() {
        let config = ZenohConfig::default();
        let backend = ZenohBackend::<TestMessage>::new("debug_test", config)
            .await
            .expect("Failed to create backend");

        let debug_str = format!("{:?}", backend);
        assert!(debug_str.contains("ZenohBackend"));
        assert!(debug_str.contains("horus/debug_test"));
    }

    /// Test Clone impl
    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn test_zenoh_clone() {
        let config = ZenohConfig::default();
        let backend = ZenohBackend::<TestMessage>::new("clone_test", config)
            .await
            .expect("Failed to create backend");

        let cloned = backend.clone();

        // Cloned backend shares the same key expression
        assert_eq!(cloned.key_expr(), backend.key_expr());
        // But publisher is not cloned
        assert!(!cloned.session_info().has_publisher);
    }

    /// Test publisher initialization
    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn test_zenoh_publisher_init() {
        let config = ZenohConfig::default();
        let mut backend = ZenohBackend::<TestMessage>::new("pub_init_test", config)
            .await
            .expect("Failed to create backend");

        assert!(!backend.session_info().has_publisher);

        backend
            .init_publisher()
            .await
            .expect("Failed to init publisher");

        assert!(backend.session_info().has_publisher);

        // Calling init again should be a no-op
        backend
            .init_publisher()
            .await
            .expect("Failed to init publisher again");
        assert!(backend.session_info().has_publisher);
    }

    /// Test subscriber initialization
    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn test_zenoh_subscriber_init() {
        let config = ZenohConfig::default();
        let mut backend = ZenohBackend::<TestMessage>::new("sub_init_test", config)
            .await
            .expect("Failed to create backend");

        // Should start empty
        assert_eq!(backend.pending_count(), 0);

        // Initialize subscriber
        backend
            .init_subscriber()
            .await
            .expect("Failed to init subscriber");

        // Still empty (no messages sent yet)
        assert_eq!(backend.pending_count(), 0);
    }

    /// Test in-process pub/sub (peer-to-peer mode)
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_zenoh_inproc_pub_sub() {
        let config = ZenohConfig::default();

        // Create publisher backend
        let mut publisher = ZenohBackend::<TestMessage>::new("inproc_test", config.clone())
            .await
            .expect("Failed to create publisher");
        publisher
            .init_publisher()
            .await
            .expect("Failed to init publisher");

        // Create subscriber backend (shares the same topic)
        let mut subscriber = ZenohBackend::<TestMessage>::new("inproc_test", config)
            .await
            .expect("Failed to create subscriber");
        subscriber
            .init_subscriber()
            .await
            .expect("Failed to init subscriber");

        // Give time for Zenoh discovery
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;

        // Send a message
        let msg = TestMessage {
            value: 123,
            text: "test message".to_string(),
        };
        publisher
            .send_async(&msg)
            .await
            .expect("Failed to send message");

        // Wait for message to be received
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;

        // Check if message was received
        let received = subscriber.recv();
        assert_eq!(received, Some(msg));
    }

    /// Test MessagePack serialization format
    #[test]
    fn test_messagepack_serialization() {
        let msg = TestMessage {
            value: 42,
            text: "msgpack test".to_string(),
        };

        // Serialize with MessagePack
        let payload = serialize_message(&msg, SerializationFormat::MessagePack)
            .expect("MessagePack serialize failed");

        // Deserialize with MessagePack
        let deserialized: TestMessage =
            rmp_serde::from_slice(&payload).expect("MessagePack deserialize failed");

        assert_eq!(msg, deserialized);
    }

    /// Test CDR serialization format (ROS2 compatible)
    #[cfg(feature = "cdr-encoding")]
    #[test]
    fn test_cdr_serialization() {
        let msg = TestMessage {
            value: 123,
            text: "cdr test".to_string(),
        };

        // Serialize with CDR
        let payload =
            serialize_message(&msg, SerializationFormat::Cdr).expect("CDR serialize failed");

        // Deserialize with CDR
        let (deserialized, _bytes_read): (TestMessage, usize) =
            cdr_encoding::from_bytes::<_, byteorder::LittleEndian>(&payload)
                .expect("CDR deserialize failed");

        assert_eq!(msg, deserialized);
    }

    /// Test JSON serialization format
    #[test]
    fn test_json_serialization() {
        let msg = TestMessage {
            value: 99,
            text: "json test".to_string(),
        };

        // Serialize with JSON
        let payload =
            serialize_message(&msg, SerializationFormat::Json).expect("JSON serialize failed");

        // Deserialize with JSON
        let deserialized: TestMessage =
            serde_json::from_slice(&payload).expect("JSON deserialize failed");

        assert_eq!(msg, deserialized);
    }

    /// Test Bincode serialization format (default)
    #[test]
    fn test_bincode_serialization() {
        let msg = TestMessage {
            value: 77,
            text: "bincode test".to_string(),
        };

        // Serialize with Bincode
        let payload = serialize_message(&msg, SerializationFormat::Bincode)
            .expect("Bincode serialize failed");

        // Deserialize with Bincode
        let deserialized: TestMessage =
            bincode::deserialize(&payload).expect("Bincode deserialize failed");

        assert_eq!(msg, deserialized);
    }

    // ========== CONFLICT PREVENTION TESTS ==========
    // These tests ensure no namespace/topic conflicts between HORUS, ROS2, and Zenoh

    /// Test that HORUS native topics and ROS2 topics are properly isolated
    /// HORUS uses: horus/{topic}
    /// ROS2 uses: rt/{topic}
    /// These namespaces should NEVER overlap
    #[test]
    fn test_horus_ros2_namespace_isolation() {
        let horus_config = ZenohConfig::default();
        let ros2_config = ZenohConfig::ros2(0);

        // Same logical topic name should produce different key expressions
        let topic = "cmd_vel";

        let horus_key = horus_config.topic_to_key_expr(topic);
        let ros2_key = ros2_config.topic_to_key_expr(topic);

        // CRITICAL: These must be different to prevent cross-system contamination
        assert_ne!(
            horus_key, ros2_key,
            "HORUS and ROS2 key expressions must differ for the same topic"
        );

        // Verify the prefixes are different
        assert!(
            horus_key.starts_with("horus/"),
            "HORUS topics must be prefixed with 'horus/'"
        );
        assert!(
            ros2_key.starts_with("rt/"),
            "ROS2 topics must be prefixed with 'rt/'"
        );
    }

    /// Test that multi-robot namespaces are properly isolated
    #[test]
    fn test_multi_robot_namespace_isolation() {
        let robot1_config = ZenohConfig::new().with_namespace("robot1");
        let robot2_config = ZenohConfig::new().with_namespace("robot2");
        let robot3_config = ZenohConfig::new().with_namespace("fleet/robot3");

        let topic = "odom";

        let robot1_key = robot1_config.topic_to_key_expr(topic);
        let robot2_key = robot2_config.topic_to_key_expr(topic);
        let robot3_key = robot3_config.topic_to_key_expr(topic);

        // All robots must have different key expressions
        assert_ne!(
            robot1_key, robot2_key,
            "Robot1 and Robot2 must have different namespaces"
        );
        assert_ne!(
            robot1_key, robot3_key,
            "Robot1 and Robot3 must have different namespaces"
        );
        assert_ne!(
            robot2_key, robot3_key,
            "Robot2 and Robot3 must have different namespaces"
        );

        // Verify expected patterns
        assert_eq!(robot1_key, "robot1/odom");
        assert_eq!(robot2_key, "robot2/odom");
        assert_eq!(robot3_key, "fleet/robot3/odom");
    }

    /// Test that ROS2 domain IDs don't affect Zenoh key expressions
    /// (Domain IDs are handled at the DDS bridge level, not in key expressions)
    #[test]
    fn test_ros2_domain_id_independence() {
        let domain0 = ZenohConfig::ros2(0);
        let domain1 = ZenohConfig::ros2(1);
        let domain42 = ZenohConfig::ros2(42);

        let topic = "/sensor/lidar";

        // Key expressions should be the same regardless of domain ID
        // (Domain isolation is handled by zenoh-bridge-ros2dds, not key expressions)
        let key0 = domain0.topic_to_key_expr(topic);
        let key1 = domain1.topic_to_key_expr(topic);
        let key42 = domain42.topic_to_key_expr(topic);

        assert_eq!(key0, key1);
        assert_eq!(key1, key42);
        assert_eq!(key0, "rt/sensor/lidar");
    }

    /// Test that empty/None namespace produces raw topic names
    #[test]
    fn test_empty_namespace_behavior() {
        let mut config = ZenohConfig::default();
        config.namespace = None;

        let topic = "test_topic";
        let key = config.topic_to_key_expr(topic);

        // With no namespace, topic should be used directly
        assert_eq!(key, "test_topic");
    }

    /// Test key expression parsing (reverse of topic_to_key_expr)
    #[test]
    fn test_key_expr_parsing_isolation() {
        let horus_config = ZenohConfig::default();
        let ros2_config = ZenohConfig::ros2(0);

        // HORUS should parse HORUS keys, not ROS2 keys
        assert_eq!(
            horus_config.key_expr_to_topic("horus/odom"),
            Some("odom".to_string())
        );
        assert_eq!(horus_config.key_expr_to_topic("rt/odom"), None); // Can't parse ROS2 key

        // ROS2 should parse ROS2 keys, not HORUS keys
        assert_eq!(
            ros2_config.key_expr_to_topic("rt/odom"),
            Some("odom".to_string())
        );
        assert_eq!(ros2_config.key_expr_to_topic("horus/odom"), None); // Can't parse HORUS key
    }

    /// Test that HORUS and ROS2 can coexist on the same Zenoh network
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_horus_ros2_coexistence() {
        // Create HORUS backend
        let horus_config = ZenohConfig::default();
        let horus_backend = ZenohBackend::<TestMessage>::new("coexist_test", horus_config.clone())
            .await
            .expect("Failed to create HORUS backend");

        // Create ROS2 backend for same logical topic
        let ros2_config = ZenohConfig::ros2(0);
        let ros2_backend = ZenohBackend::<TestMessage>::new("coexist_test", ros2_config.clone())
            .await
            .expect("Failed to create ROS2 backend");

        // Verify they're using different key expressions
        let horus_key = horus_backend.key_expr();
        let ros2_key = ros2_backend.key_expr();

        assert_ne!(horus_key, ros2_key);
        assert!(horus_key.contains("horus/"));
        assert!(ros2_key.contains("rt/"));
    }

    /// Test serialization format isolation
    /// Different formats should NOT be able to deserialize each other's data
    #[test]
    fn test_serialization_format_isolation() {
        let msg = TestMessage {
            value: 42,
            text: "isolation test".to_string(),
        };

        // Serialize with different formats
        let bincode_data = serialize_message(&msg, SerializationFormat::Bincode).unwrap();
        let msgpack_data = serialize_message(&msg, SerializationFormat::MessagePack).unwrap();
        let json_data = serialize_message(&msg, SerializationFormat::Json).unwrap();

        // Verify each format produces different bytes (no accidental compatibility)
        assert_ne!(bincode_data, msgpack_data);
        assert_ne!(bincode_data, json_data);
        assert_ne!(msgpack_data, json_data);

        // Verify cross-format deserialization fails gracefully
        // Bincode data should NOT deserialize as MessagePack
        let cross_deser: Result<TestMessage, _> = rmp_serde::from_slice(&bincode_data);
        assert!(
            cross_deser.is_err(),
            "Bincode data should not deserialize as MessagePack"
        );

        // MessagePack data should NOT deserialize as Bincode
        let cross_deser: Result<TestMessage, _> = bincode::deserialize(&msgpack_data);
        assert!(
            cross_deser.is_err(),
            "MessagePack data should not deserialize as Bincode"
        );
    }

    /// Test that hierarchical topic namespaces work correctly
    #[test]
    fn test_hierarchical_topic_namespaces() {
        let config = ZenohConfig::default();

        // Nested topics should preserve hierarchy
        assert_eq!(
            config.topic_to_key_expr("sensors/lidar/scan"),
            "horus/sensors/lidar/scan"
        );
        assert_eq!(
            config.topic_to_key_expr("actuators/wheel/left"),
            "horus/actuators/wheel/left"
        );

        // Parse back correctly
        assert_eq!(
            config.key_expr_to_topic("horus/sensors/lidar/scan"),
            Some("sensors/lidar/scan".to_string())
        );
    }

    /// Test ROS2 topic naming with leading slashes
    #[test]
    fn test_ros2_leading_slash_handling() {
        let ros2_config = ZenohConfig::ros2(0);

        // Topics with and without leading slash should be normalized
        assert_eq!(ros2_config.topic_to_key_expr("/cmd_vel"), "rt/cmd_vel");
        assert_eq!(ros2_config.topic_to_key_expr("cmd_vel"), "rt/cmd_vel");

        // Hierarchical ROS2 topics
        assert_eq!(
            ros2_config.topic_to_key_expr("/robot/sensor/imu"),
            "rt/robot/sensor/imu"
        );
    }

    /// Test that config cloning preserves isolation properties
    #[test]
    fn test_config_clone_isolation() {
        let original = ZenohConfig::default();
        let mut cloned = original.clone();

        // Modify cloned config
        cloned.namespace = Some("modified".to_string());

        // Original should be unchanged
        assert_eq!(original.namespace, Some("horus".to_string()));
        assert_eq!(cloned.namespace, Some("modified".to_string()));

        // Key expressions should differ
        assert_ne!(
            original.topic_to_key_expr("test"),
            cloned.topic_to_key_expr("test")
        );
    }
}
