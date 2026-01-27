//! Core types for HORUS State Machines.
//!
//! This module provides the foundational types for building hierarchical
//! finite state machines (HFSM) for robot mode management.
//!
//! # Key Concepts
//!
//! - **State**: A discrete mode the system can be in (e.g., Idle, Moving, Error)
//! - **Event**: An input that may trigger a transition
//! - **Transition**: A rule for moving from one state to another
//! - **Guard**: A condition that must be true for a transition to occur
//! - **Action**: Code executed on state entry, exit, or during transitions

use serde::{Deserialize, Serialize};
use std::fmt;
use std::hash::Hash;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Unique identifier for a state.
///
/// States can be identified by string names or numeric IDs for performance.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StateId {
    /// Named state identifier (user-friendly)
    Named(String),
    /// Numeric state identifier (performance-optimized)
    Numeric(u32),
}

impl StateId {
    /// Create a named state ID.
    pub fn named(name: impl Into<String>) -> Self {
        StateId::Named(name.into())
    }

    /// Create a numeric state ID.
    pub fn numeric(id: u32) -> Self {
        StateId::Numeric(id)
    }

    /// Get the name of the state (for Named) or formatted number (for Numeric).
    pub fn as_str(&self) -> String {
        match self {
            StateId::Named(name) => name.clone(),
            StateId::Numeric(id) => format!("state_{}", id),
        }
    }
}

impl fmt::Display for StateId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StateId::Named(name) => write!(f, "{}", name),
            StateId::Numeric(id) => write!(f, "State({})", id),
        }
    }
}

impl From<&str> for StateId {
    fn from(s: &str) -> Self {
        StateId::Named(s.to_string())
    }
}

impl From<String> for StateId {
    fn from(s: String) -> Self {
        StateId::Named(s)
    }
}

impl From<u32> for StateId {
    fn from(id: u32) -> Self {
        StateId::Numeric(id)
    }
}

/// An event that can trigger state transitions.
///
/// Events represent inputs to the state machine that may cause it to
/// change state. Events can carry optional payload data.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Event {
    /// Event type identifier
    pub event_type: String,
    /// Optional priority for event processing
    pub priority: EventPriority,
    /// Timestamp when the event was created
    #[serde(skip)]
    pub timestamp: Option<Instant>,
}

impl Event {
    /// Create a new event.
    pub fn new(event_type: impl Into<String>) -> Self {
        Self {
            event_type: event_type.into(),
            priority: EventPriority::Normal,
            timestamp: Some(Instant::now()),
        }
    }

    /// Create a high-priority event.
    pub fn high_priority(event_type: impl Into<String>) -> Self {
        Self {
            event_type: event_type.into(),
            priority: EventPriority::High,
            timestamp: Some(Instant::now()),
        }
    }

    /// Create a critical event (processed immediately).
    pub fn critical(event_type: impl Into<String>) -> Self {
        Self {
            event_type: event_type.into(),
            priority: EventPriority::Critical,
            timestamp: Some(Instant::now()),
        }
    }

    /// Get the event type.
    pub fn event_type(&self) -> &str {
        &self.event_type
    }
}

impl fmt::Display for Event {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Event({})", self.event_type)
    }
}

impl From<&str> for Event {
    fn from(s: &str) -> Self {
        Event::new(s)
    }
}

/// Priority level for events.
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord, Default,
)]
pub enum EventPriority {
    /// Low priority - processed after normal events
    Low = 0,
    /// Normal priority - default processing order
    #[default]
    Normal = 1,
    /// High priority - processed before normal events
    High = 2,
    /// Critical - processed immediately, bypasses queue
    Critical = 3,
}

/// Result of attempting a state transition.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TransitionResult {
    /// Transition succeeded - now in new state
    Success {
        from: StateId,
        to: StateId,
        via_event: Option<String>,
    },
    /// Transition blocked by guard condition
    GuardFailed {
        from: StateId,
        to: StateId,
        reason: String,
    },
    /// No matching transition for this event
    NoTransition { current: StateId, event: String },
    /// State not found
    StateNotFound(StateId),
    /// Transition to same state (self-loop)
    SelfTransition(StateId),
    /// Error during transition
    Error(String),
}

impl TransitionResult {
    /// Check if the transition was successful.
    pub fn is_success(&self) -> bool {
        matches!(self, TransitionResult::Success { .. })
    }

    /// Check if this was a self-transition.
    pub fn is_self_transition(&self) -> bool {
        matches!(self, TransitionResult::SelfTransition(_))
    }

    /// Get the destination state if transition succeeded.
    pub fn destination(&self) -> Option<&StateId> {
        match self {
            TransitionResult::Success { to, .. } => Some(to),
            TransitionResult::SelfTransition(state) => Some(state),
            _ => None,
        }
    }
}

/// Configuration for a state within the machine.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StateConfig {
    /// Unique identifier for this state
    pub id: StateId,
    /// Human-readable name
    pub name: String,
    /// Optional description
    pub description: Option<String>,
    /// Whether this is an initial state
    pub is_initial: bool,
    /// Whether this is a final/terminal state
    pub is_final: bool,
    /// Parent state for hierarchical machines (None if top-level)
    pub parent: Option<StateId>,
    /// Timeout - auto-transition if exceeded
    pub timeout: Option<Duration>,
    /// State to transition to on timeout
    pub timeout_target: Option<StateId>,
    /// Custom metadata
    pub metadata: std::collections::HashMap<String, String>,
}

impl StateConfig {
    /// Create a new state configuration.
    pub fn new(id: impl Into<StateId>) -> Self {
        let id = id.into();
        let name = id.as_str();
        Self {
            id,
            name,
            description: None,
            is_initial: false,
            is_final: false,
            parent: None,
            timeout: None,
            timeout_target: None,
            metadata: std::collections::HashMap::new(),
        }
    }

    /// Mark this as an initial state.
    pub fn initial(mut self) -> Self {
        self.is_initial = true;
        self
    }

    /// Mark this as a final state.
    pub fn final_state(mut self) -> Self {
        self.is_final = true;
        self
    }

    /// Set the parent state (for hierarchical FSM).
    pub fn with_parent(mut self, parent: impl Into<StateId>) -> Self {
        self.parent = Some(parent.into());
        self
    }

    /// Set a timeout for this state.
    pub fn with_timeout(mut self, timeout: Duration, target: impl Into<StateId>) -> Self {
        self.timeout = Some(timeout);
        self.timeout_target = Some(target.into());
        self
    }

    /// Add custom metadata.
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Set the description.
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }
}

/// Configuration for a transition between states.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TransitionConfig {
    /// Source state
    pub from: StateId,
    /// Target state
    pub to: StateId,
    /// Event that triggers this transition (None for automatic)
    pub event: Option<String>,
    /// Priority when multiple transitions match
    pub priority: i32,
    /// Human-readable description
    pub description: Option<String>,
}

impl TransitionConfig {
    /// Create a new transition configuration.
    pub fn new(from: impl Into<StateId>, to: impl Into<StateId>) -> Self {
        Self {
            from: from.into(),
            to: to.into(),
            event: None,
            priority: 0,
            description: None,
        }
    }

    /// Set the triggering event.
    pub fn on_event(mut self, event: impl Into<String>) -> Self {
        self.event = Some(event.into());
        self
    }

    /// Set the transition priority.
    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }

    /// Set the description.
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }
}

/// Type alias for guard functions.
///
/// Guards are predicates that must return true for a transition to occur.
/// They receive the current context and return whether the transition is allowed.
pub type GuardFn<C> = Arc<dyn Fn(&C) -> bool + Send + Sync>;

/// Type alias for action functions.
///
/// Actions are executed on state entry, exit, or during transitions.
pub type ActionFn<C> = Arc<dyn Fn(&mut C) + Send + Sync>;

/// Type alias for async action functions.
pub type AsyncActionFn<C> = Arc<
    dyn Fn(&mut C) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send>> + Send + Sync,
>;

/// Represents a complete transition with guard and actions.
pub struct Transition<C> {
    /// Configuration for this transition
    pub config: TransitionConfig,
    /// Optional guard condition
    pub guard: Option<GuardFn<C>>,
    /// Actions to execute during transition
    pub actions: Vec<ActionFn<C>>,
}

// Manual Clone implementation that doesn't require C: Clone
// since Arc<dyn Fn...> is Clone without requiring the type parameter
impl<C> Clone for Transition<C> {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            guard: self.guard.clone(),
            actions: self.actions.clone(),
        }
    }
}

impl<C> Transition<C> {
    /// Create a new transition.
    pub fn new(from: impl Into<StateId>, to: impl Into<StateId>) -> Self {
        Self {
            config: TransitionConfig::new(from, to),
            guard: None,
            actions: Vec::new(),
        }
    }

    /// Set the triggering event.
    pub fn on_event(mut self, event: impl Into<String>) -> Self {
        self.config.event = Some(event.into());
        self
    }

    /// Add a guard condition.
    pub fn with_guard<F>(mut self, guard: F) -> Self
    where
        F: Fn(&C) -> bool + Send + Sync + 'static,
    {
        self.guard = Some(Arc::new(guard));
        self
    }

    /// Add an action to execute during the transition.
    pub fn with_action<F>(mut self, action: F) -> Self
    where
        F: Fn(&mut C) + Send + Sync + 'static,
    {
        self.actions.push(Arc::new(action));
        self
    }

    /// Check if the guard allows this transition.
    pub fn check_guard(&self, context: &C) -> bool {
        match &self.guard {
            Some(guard) => guard(context),
            None => true,
        }
    }

    /// Execute all transition actions.
    pub fn execute_actions(&self, context: &mut C) {
        for action in &self.actions {
            action(context);
        }
    }
}

impl<C> std::fmt::Debug for Transition<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Transition")
            .field("from", &self.config.from)
            .field("to", &self.config.to)
            .field("event", &self.config.event)
            .field("has_guard", &self.guard.is_some())
            .field("action_count", &self.actions.len())
            .finish()
    }
}

/// Represents a state with its callbacks.
pub struct State<C> {
    /// Configuration for this state
    pub config: StateConfig,
    /// Action executed when entering this state
    pub on_entry: Option<ActionFn<C>>,
    /// Action executed when exiting this state
    pub on_exit: Option<ActionFn<C>>,
    /// Action executed on each tick while in this state
    pub on_tick: Option<ActionFn<C>>,
    /// Child states (for hierarchical FSM)
    pub children: Vec<StateId>,
}

impl<C> State<C> {
    /// Create a new state.
    pub fn new(id: impl Into<StateId>) -> Self {
        Self {
            config: StateConfig::new(id),
            on_entry: None,
            on_exit: None,
            on_tick: None,
            children: Vec::new(),
        }
    }

    /// Mark this as an initial state.
    pub fn initial(mut self) -> Self {
        self.config.is_initial = true;
        self
    }

    /// Mark this as a final state.
    pub fn final_state(mut self) -> Self {
        self.config.is_final = true;
        self
    }

    /// Set the entry action.
    pub fn on_entry<F>(mut self, action: F) -> Self
    where
        F: Fn(&mut C) + Send + Sync + 'static,
    {
        self.on_entry = Some(Arc::new(action));
        self
    }

    /// Set the exit action.
    pub fn on_exit<F>(mut self, action: F) -> Self
    where
        F: Fn(&mut C) + Send + Sync + 'static,
    {
        self.on_exit = Some(Arc::new(action));
        self
    }

    /// Set the tick action (executed each update while in this state).
    pub fn on_tick<F>(mut self, action: F) -> Self
    where
        F: Fn(&mut C) + Send + Sync + 'static,
    {
        self.on_tick = Some(Arc::new(action));
        self
    }

    /// Set the parent state.
    pub fn with_parent(mut self, parent: impl Into<StateId>) -> Self {
        self.config.parent = Some(parent.into());
        self
    }

    /// Add a child state.
    pub fn with_child(mut self, child: impl Into<StateId>) -> Self {
        self.children.push(child.into());
        self
    }

    /// Set a timeout for this state.
    pub fn with_timeout(mut self, timeout: Duration, target: impl Into<StateId>) -> Self {
        self.config.timeout = Some(timeout);
        self.config.timeout_target = Some(target.into());
        self
    }

    /// Execute the entry action.
    pub fn enter(&self, context: &mut C) {
        if let Some(action) = &self.on_entry {
            action(context);
        }
    }

    /// Execute the exit action.
    pub fn exit(&self, context: &mut C) {
        if let Some(action) = &self.on_exit {
            action(context);
        }
    }

    /// Execute the tick action.
    pub fn tick(&self, context: &mut C) {
        if let Some(action) = &self.on_tick {
            action(context);
        }
    }
}

impl<C> std::fmt::Debug for State<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("State")
            .field("config", &self.config)
            .field("has_on_entry", &self.on_entry.is_some())
            .field("has_on_exit", &self.on_exit.is_some())
            .field("has_on_tick", &self.on_tick.is_some())
            .field("children", &self.children)
            .finish()
    }
}

/// Error type for state machine operations.
#[derive(Clone, Debug, thiserror::Error)]
pub enum StateMachineError {
    #[error("State not found: {0}")]
    StateNotFound(StateId),

    #[error("No initial state defined")]
    NoInitialState,

    #[error("Multiple initial states defined: {0:?}")]
    MultipleInitialStates(Vec<StateId>),

    #[error("Transition blocked by guard: {from} -> {to}, reason: {reason}")]
    GuardFailed {
        from: StateId,
        to: StateId,
        reason: String,
    },

    #[error("No transition found for event '{event}' in state '{state}'")]
    NoTransition { state: StateId, event: String },

    #[error("Invalid state hierarchy: {0}")]
    InvalidHierarchy(String),

    #[error("Timeout in state '{0}'")]
    StateTimeout(StateId),

    #[error("State machine not started")]
    NotStarted,

    #[error("State machine already started")]
    AlreadyStarted,

    #[error("Internal error: {0}")]
    Internal(String),
}

/// Metrics for state machine monitoring.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct StateMachineMetrics {
    /// Total number of transitions
    pub transition_count: u64,
    /// Number of successful transitions
    pub successful_transitions: u64,
    /// Number of guard failures
    pub guard_failures: u64,
    /// Number of events processed
    pub events_processed: u64,
    /// Number of events dropped (queue overflow)
    pub events_dropped: u64,
    /// Time spent in each state (nanoseconds)
    pub state_time_ns: std::collections::HashMap<String, u64>,
    /// Count of visits to each state
    pub state_visit_count: std::collections::HashMap<String, u64>,
    /// Average transition time (nanoseconds)
    pub avg_transition_time_ns: u64,
}

impl StateMachineMetrics {
    /// Create new empty metrics.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a successful transition.
    pub fn record_transition(&mut self, _from: &StateId, to: &StateId, duration_ns: u64) {
        self.transition_count += 1;
        self.successful_transitions += 1;

        // Update average transition time
        let total = self.avg_transition_time_ns * (self.successful_transitions - 1) + duration_ns;
        self.avg_transition_time_ns = total / self.successful_transitions;

        // Update state visit count
        *self.state_visit_count.entry(to.as_str()).or_insert(0) += 1;
    }

    /// Record a guard failure.
    pub fn record_guard_failure(&mut self, _from: &StateId, _to: &StateId) {
        self.guard_failures += 1;
    }

    /// Record time spent in a state.
    pub fn record_state_time(&mut self, state: &StateId, duration_ns: u64) {
        *self.state_time_ns.entry(state.as_str()).or_insert(0) += duration_ns;
    }

    /// Record an event processed.
    pub fn record_event(&mut self) {
        self.events_processed += 1;
    }

    /// Record an event dropped.
    pub fn record_event_dropped(&mut self) {
        self.events_dropped += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_id_creation() {
        let named = StateId::named("idle");
        assert_eq!(named.as_str(), "idle");

        let numeric = StateId::numeric(42);
        assert_eq!(numeric.as_str(), "state_42");

        let from_str: StateId = "moving".into();
        assert_eq!(from_str.as_str(), "moving");
    }

    #[test]
    fn test_event_creation() {
        let event = Event::new("button_pressed");
        assert_eq!(event.event_type(), "button_pressed");
        assert_eq!(event.priority, EventPriority::Normal);

        let critical = Event::critical("emergency_stop");
        assert_eq!(critical.priority, EventPriority::Critical);
    }

    #[test]
    fn test_state_config() {
        let config = StateConfig::new("idle")
            .initial()
            .with_description("Initial idle state")
            .with_timeout(Duration::from_secs(30), "timeout_state");

        assert!(config.is_initial);
        assert!(!config.is_final);
        assert_eq!(config.timeout, Some(Duration::from_secs(30)));
    }

    #[test]
    fn test_transition_result() {
        let success = TransitionResult::Success {
            from: "idle".into(),
            to: "moving".into(),
            via_event: Some("start".into()),
        };
        assert!(success.is_success());
        assert_eq!(success.destination(), Some(&StateId::named("moving")));

        let guard_failed = TransitionResult::GuardFailed {
            from: "idle".into(),
            to: "moving".into(),
            reason: "battery low".into(),
        };
        assert!(!guard_failed.is_success());
    }

    #[test]
    fn test_state_with_callbacks() {
        struct TestContext {
            entered: bool,
            exited: bool,
            ticked: bool,
        }

        let state = State::<TestContext>::new("test")
            .on_entry(|ctx| ctx.entered = true)
            .on_exit(|ctx| ctx.exited = true)
            .on_tick(|ctx| ctx.ticked = true);

        let mut ctx = TestContext {
            entered: false,
            exited: false,
            ticked: false,
        };

        state.enter(&mut ctx);
        assert!(ctx.entered);

        state.tick(&mut ctx);
        assert!(ctx.ticked);

        state.exit(&mut ctx);
        assert!(ctx.exited);
    }

    #[test]
    fn test_transition_with_guard() {
        struct TestContext {
            allowed: bool,
        }

        let transition = Transition::<TestContext>::new("idle", "moving")
            .on_event("start")
            .with_guard(|ctx| ctx.allowed);

        let allowed_ctx = TestContext { allowed: true };
        assert!(transition.check_guard(&allowed_ctx));

        let denied_ctx = TestContext { allowed: false };
        assert!(!transition.check_guard(&denied_ctx));
    }

    #[test]
    fn test_metrics() {
        let mut metrics = StateMachineMetrics::new();

        metrics.record_transition(&"idle".into(), &"moving".into(), 1000);
        assert_eq!(metrics.successful_transitions, 1);
        assert_eq!(metrics.avg_transition_time_ns, 1000);

        metrics.record_transition(&"moving".into(), &"stopped".into(), 2000);
        assert_eq!(metrics.successful_transitions, 2);
        assert_eq!(metrics.avg_transition_time_ns, 1500);

        metrics.record_guard_failure(&"stopped".into(), &"moving".into());
        assert_eq!(metrics.guard_failures, 1);
    }
}
