/// Network communication backends for HORUS
///
/// This module provides network-based communication in addition to the local
/// shared memory backend. It includes:
/// - Endpoint parsing for network addresses
/// - Binary protocol for efficient serialization
/// - UDP direct connections (no discovery)
/// - Unix domain sockets (localhost optimization)
/// - Multicast discovery
/// - Message batching for efficiency
/// - Congestion control with drop policies
/// - Compression (LZ4/Zstd)
/// - Query/Response patterns
/// - Topic caching with TTL
///
/// Network v2 high-performance backends:
/// - Batch UDP with sendmmsg/recvmmsg (200K+ packets/sec)
/// - Real io_uring zero-copy (3-5µs latency)
/// - QUIC transport (0-RTT, reliable)
/// - Smart transport selection (auto-picks best backend)
pub mod backend;
pub mod batching;
pub mod caching;
pub mod compression;
pub mod congestion;
pub mod direct;
pub mod discovery;
pub mod endpoint;
pub mod fragmentation;
pub mod protocol;
pub mod queryable;
pub mod reconnect;
pub mod router;
pub mod udp_direct;
pub mod udp_multicast;

// Network v2 high-performance modules
pub mod batch_udp;
pub mod smart_copy;
pub mod smart_transport;

// Unix domain sockets are only available on Unix-like systems
#[cfg(unix)]
pub mod unix_socket;

#[cfg(feature = "tls")]
pub mod tls;

// QUIC transport (requires quic feature)
#[cfg(feature = "quic")]
pub mod quic;

// io_uring backend (real implementation using io-uring crate, Linux only)
// Requires the `io-uring-net` feature flag
#[cfg(target_os = "linux")]
pub mod io_uring;

// Zenoh transport for multi-robot mesh, cloud connectivity, and ROS2 interop
// Requires the `zenoh-transport` feature flag
#[cfg(feature = "zenoh-transport")]
pub mod zenoh_backend;
pub mod zenoh_config;

// ROS2 service protocol for Zenoh (request/response over rq/rs topics)
// Requires the `zenoh-transport` feature flag
#[cfg(feature = "zenoh-transport")]
pub mod zenoh_ros2_services;

// ROS2 action protocol for Zenoh (goal/cancel/result/feedback/status topics)
// Requires the `zenoh-transport` feature flag
#[cfg(feature = "zenoh-transport")]
pub mod zenoh_ros2_actions;

// ROS2 parameter server protocol for Zenoh (list/get/set/describe parameters)
// Requires the `zenoh-transport` feature flag
#[cfg(feature = "zenoh-transport")]
pub mod zenoh_ros2_params;

// Re-export commonly used types
pub use backend::NetworkBackend;
pub use direct::{DirectBackend, DirectRole};
pub use discovery::{DiscoveryService, PeerInfo};
pub use endpoint::{parse_endpoint, Endpoint, DEFAULT_PORT, MULTICAST_ADDR, MULTICAST_PORT};
pub use fragmentation::{Fragment, FragmentManager};
pub use protocol::{HorusPacket, MessageType};
pub use reconnect::{ConnectionHealth, ReconnectContext, ReconnectStrategy};
pub use router::RouterBackend;
pub use udp_direct::UdpDirectBackend;
pub use udp_multicast::UdpMulticastBackend;

// Re-export new modules
pub use batching::{BatchConfig, BatchReceiver, MessageBatch, MessageBatcher, SharedBatcher};
pub use caching::{CacheConfig, CacheStats, SharedCache, TopicCache};
pub use compression::{
    CompressedData, CompressedPacket, CompressionAlgo, CompressionConfig, Compressor,
};
pub use congestion::{
    CongestionConfig, CongestionController, CongestionResult, DropPolicy,
    SharedCongestionController,
};
pub use queryable::{
    QueryClient, QueryConfig, QueryError, QueryHandler, QueryRequest, QueryResponse, QueryServer,
    ResponseStatus,
};

#[cfg(unix)]
pub use unix_socket::UnixSocketBackend;

#[cfg(feature = "tls")]
pub use tls::{TlsCertConfig, TlsStream};

// Network v2 re-exports
pub use batch_udp::{
    BatchUdpConfig, BatchUdpReceiver, BatchUdpSender, BatchUdpStats, ReceivedPacket,
    ScalableUdpBackend,
};
pub use smart_copy::{
    BufferPool, CopyStrategy, RegisteredBuffer, SmartCopyConfig, SmartCopySender, SmartCopyStats,
};
pub use smart_transport::{
    NetworkLocation, TransportBuilder, TransportPreferences, TransportSelector,
    TransportSelectorStats, TransportType,
};

// io_uring re-exports (real implementation)
#[cfg(all(target_os = "linux", feature = "io-uring-net"))]
pub use io_uring::{
    is_real_io_uring_available, is_sqpoll_available, CompletionResult, RealIoUringBackend,
    RealIoUringConfig, RealIoUringStats,
};

#[cfg(feature = "quic")]
pub use quic::{generate_self_signed_cert, QuicConfig, QuicStats, QuicTransport};

// Zenoh re-exports
#[cfg(feature = "zenoh-transport")]
pub use zenoh_backend::{ZenohBackend, ZenohSessionInfo};
pub use zenoh_config::{
    CongestionControl, Durability, HistoryPolicy, Liveliness, Reliability, SerializationFormat,
    ZenohConfig, ZenohMode, ZenohQos,
};

// ROS2 services re-exports
#[cfg(feature = "zenoh-transport")]
pub use zenoh_ros2_services::{
    parse_service_topic, service_to_request_topic, service_to_response_topic, EmptyRequest,
    EmptyResponse, RequestTracker, Ros2RequestHeader, Ros2ServiceConfig, Ros2ServiceError,
    Ros2ServiceRequest, Ros2ServiceResponse, ServiceRegistry, ServiceStats, SetBoolRequest,
    SetBoolResponse, TriggerRequest, TriggerResponse,
};

// ROS2 actions re-exports
#[cfg(feature = "zenoh-transport")]
pub use zenoh_ros2_actions::{
    action_to_cancel_goal_topic,
    action_to_feedback_topic,
    action_to_get_result_topic,
    // Topic naming
    action_to_send_goal_topic,
    action_to_status_topic,
    apply_namespace_to_action,
    parse_action_topic,
    ActionClientTopics,
    ActionServerTopics,
    ActionStats,
    ActionTopicType,
    // Trait
    ActionType,
    CancelCallback,
    CancelGoalErrorCode,
    CancelGoalRequest,
    CancelGoalResponse,
    // Goal handles
    ClientGoalHandle,
    EmptyFeedback,
    // Common action types
    EmptyGoal,
    EmptyResult,
    ExecuteCallback,
    FeedbackMessage,
    GetResultRequest,
    GetResultResponse,
    GoalCallback,
    // Core types
    GoalId,
    GoalInfo,
    GoalStatus,
    GoalStatusArray,
    GoalStatusInfo,
    ProgressFeedback,
    // Client/Server
    Ros2ActionClient,
    Ros2ActionConfig,
    Ros2ActionError,
    Ros2ActionServer,
    Ros2Time,
    // Request/Response types
    SendGoalRequest,
    SendGoalResponse,
    ServerGoalHandle,
};

// ROS2 parameters re-exports
#[cfg(feature = "zenoh-transport")]
pub use zenoh_ros2_params::{
    param_to_describe_parameters_topic,
    param_to_events_topic,
    param_to_get_parameter_types_topic,
    param_to_get_parameters_topic,
    // Topic naming
    param_to_list_parameters_topic,
    param_to_set_parameters_atomically_topic,
    param_to_set_parameters_topic,
    DescribeParametersRequest,
    DescribeParametersResponse,
    FloatingPointRange,
    GetParameterTypesRequest,
    GetParameterTypesResponse,
    GetParametersRequest,
    GetParametersResponse,
    IntegerRange,
    // Request/Response types
    ListParametersRequest,
    ListParametersResponse,
    ListParametersResult,
    // Store and Client/Server
    LocalParameterStore,
    Parameter,
    ParameterClientTopics,
    ParameterDescriptor,
    ParameterEvent,
    ParameterEventType,
    ParameterQos,
    ParameterServerTopics,
    ParameterStats,
    ParameterTopicType,
    // Core types
    ParameterType,
    ParameterValue,
    Ros2ParamTime,
    Ros2ParameterClient,
    Ros2ParameterConfig,
    Ros2ParameterError,
    Ros2ParameterServer,
    SetParameterResult,
    SetParametersAtomicallyRequest,
    SetParametersAtomicallyResponse,
    SetParametersRequest,
    SetParametersResponse,
};
