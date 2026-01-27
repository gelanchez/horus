//! State Machine implementation for HORUS.
//!
//! This module provides a flexible, type-safe state machine that supports:
//! - Hierarchical states (nested state machines)
//! - Guard conditions on transitions
//! - Entry/exit/tick actions on states
//! - Event-driven and automatic transitions
//! - Timeout-based transitions
//! - Comprehensive metrics and logging

use super::types::*;
use parking_lot::RwLock;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// A finite state machine with support for hierarchical states.
///
/// The `StateMachine<C>` is parameterized by a context type `C` that holds
/// the application state accessible to guards and actions.
///
/// # Example
///
/// ```rust,ignore
/// use horus_core::state_machines::{StateMachine, State, Transition};
///
/// struct RobotContext {
///     battery_level: f32,
///     is_moving: bool,
/// }
///
/// let mut fsm = StateMachine::<RobotContext>::new("robot_fsm")
///     .add_state(State::new("idle").initial())
///     .add_state(State::new("moving"))
///     .add_state(State::new("charging"))
///     .add_transition(
///         Transition::new("idle", "moving")
///             .on_event("start")
///             .with_guard(|ctx| ctx.battery_level > 0.2)
///     )
///     .add_transition(
///         Transition::new("moving", "idle")
///             .on_event("stop")
///     )
///     .build()?;
///
/// let mut ctx = RobotContext { battery_level: 0.8, is_moving: false };
/// fsm.start(&mut ctx)?;
/// fsm.process_event(&Event::new("start"), &mut ctx)?;
/// ```
pub struct StateMachine<C> {
    /// Name of this state machine
    name: String,
    /// All states in the machine
    states: HashMap<StateId, State<C>>,
    /// All transitions indexed by source state
    transitions: HashMap<StateId, Vec<Transition<C>>>,
    /// Current active state
    current_state: Option<StateId>,
    /// Stack of active states (for hierarchical FSM)
    state_stack: Vec<StateId>,
    /// Event queue for deferred processing
    event_queue: VecDeque<Event>,
    /// Maximum event queue size
    max_queue_size: usize,
    /// Time when current state was entered
    state_entered_at: Option<Instant>,
    /// Whether the machine is running
    is_running: bool,
    /// Metrics collector
    metrics: StateMachineMetrics,
    /// History of state transitions (limited size)
    history: VecDeque<TransitionRecord>,
    /// Maximum history size
    max_history_size: usize,
}

/// Record of a state transition for history tracking.
#[derive(Clone, Debug)]
pub struct TransitionRecord {
    /// Source state
    pub from: StateId,
    /// Destination state
    pub to: StateId,
    /// Triggering event (if any)
    pub event: Option<String>,
    /// Timestamp of the transition
    pub timestamp: Instant,
    /// Duration of the transition
    pub duration: Duration,
}

impl<C> StateMachine<C> {
    /// Create a new state machine with the given name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            states: HashMap::new(),
            transitions: HashMap::new(),
            current_state: None,
            state_stack: Vec::new(),
            event_queue: VecDeque::new(),
            max_queue_size: 100,
            state_entered_at: None,
            is_running: false,
            metrics: StateMachineMetrics::new(),
            history: VecDeque::new(),
            max_history_size: 50,
        }
    }

    /// Get the name of the state machine.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Add a state to the machine.
    pub fn add_state(mut self, state: State<C>) -> Self {
        let id = state.config.id.clone();
        self.states.insert(id, state);
        self
    }

    /// Add a transition to the machine.
    pub fn add_transition(mut self, transition: Transition<C>) -> Self {
        let from = transition.config.from.clone();
        self.transitions.entry(from).or_default().push(transition);
        self
    }

    /// Set the maximum event queue size.
    pub fn with_max_queue_size(mut self, size: usize) -> Self {
        self.max_queue_size = size;
        self
    }

    /// Set the maximum history size.
    pub fn with_max_history_size(mut self, size: usize) -> Self {
        self.max_history_size = size;
        self
    }

    /// Build and validate the state machine.
    ///
    /// Returns an error if validation fails (e.g., no initial state).
    pub fn build(self) -> Result<Self, StateMachineError> {
        // Find initial state
        let initial_states: Vec<_> = self
            .states
            .values()
            .filter(|s| s.config.is_initial)
            .map(|s| s.config.id.clone())
            .collect();

        match initial_states.len() {
            0 => Err(StateMachineError::NoInitialState),
            1 => Ok(self),
            _ => Err(StateMachineError::MultipleInitialStates(initial_states)),
        }
    }

    /// Start the state machine, entering the initial state.
    pub fn start(&mut self, context: &mut C) -> Result<(), StateMachineError> {
        if self.is_running {
            return Err(StateMachineError::AlreadyStarted);
        }

        // Find initial state
        let initial_state = self
            .states
            .values()
            .find(|s| s.config.is_initial)
            .map(|s| s.config.id.clone())
            .ok_or(StateMachineError::NoInitialState)?;

        // Enter the initial state
        self.enter_state(&initial_state, context)?;
        self.is_running = true;

        log::info!(
            "State machine '{}' started in state '{}'",
            self.name,
            initial_state
        );

        Ok(())
    }

    /// Stop the state machine.
    pub fn stop(&mut self, context: &mut C) -> Result<(), StateMachineError> {
        if !self.is_running {
            return Ok(());
        }

        // Exit current state
        if let Some(current) = self.current_state.clone() {
            self.exit_state(&current, context)?;
        }

        self.is_running = false;
        self.current_state = None;
        self.state_stack.clear();

        log::info!("State machine '{}' stopped", self.name);

        Ok(())
    }

    /// Check if the state machine is running.
    pub fn is_running(&self) -> bool {
        self.is_running
    }

    /// Get the current state.
    pub fn current_state(&self) -> Option<&StateId> {
        self.current_state.as_ref()
    }

    /// Get the state stack (for hierarchical FSM).
    pub fn state_stack(&self) -> &[StateId] {
        &self.state_stack
    }

    /// Check if currently in a specific state.
    pub fn is_in_state(&self, state: &StateId) -> bool {
        self.current_state.as_ref() == Some(state) || self.state_stack.contains(state)
    }

    /// Queue an event for processing.
    pub fn queue_event(&mut self, event: Event) -> Result<(), StateMachineError> {
        if self.event_queue.len() >= self.max_queue_size {
            self.metrics.record_event_dropped();
            log::warn!("Event queue full, dropping event: {}", event.event_type);
            return Ok(());
        }

        // Insert based on priority
        if event.priority == EventPriority::Critical {
            self.event_queue.push_front(event);
        } else {
            self.event_queue.push_back(event);
        }

        Ok(())
    }

    /// Process a single event immediately.
    pub fn process_event(
        &mut self,
        event: &Event,
        context: &mut C,
    ) -> Result<TransitionResult, StateMachineError> {
        if !self.is_running {
            return Err(StateMachineError::NotStarted);
        }

        let current = self
            .current_state
            .clone()
            .ok_or(StateMachineError::NotStarted)?;

        self.metrics.record_event();

        // Find matching transition - collect data without calling mutable methods
        let (matching_transition, guard_failures): (
            Option<Transition<C>>,
            Vec<(StateId, StateId)>,
        ) = {
            if let Some(transitions) = self.transitions.get(&current) {
                // Sort by priority (highest first)
                let mut matching: Vec<_> = transitions
                    .iter()
                    .filter(|t| t.config.event.as_ref() == Some(&event.event_type))
                    .collect();
                matching.sort_by_key(|t| -t.config.priority);

                let mut found = None;
                let mut failures = Vec::new();
                for transition in matching {
                    // Check guard
                    if !transition.check_guard(context) {
                        // Collect guard failure info - will record after borrow ends
                        failures.push((current.clone(), transition.config.to.clone()));
                        log::debug!(
                            "Guard failed for transition {} -> {}",
                            current,
                            transition.config.to
                        );
                        continue;
                    }
                    found = Some((*transition).clone());
                    break;
                }
                (found, failures)
            } else {
                (None, Vec::new())
            }
        };

        // Record guard failures after immutable borrow ends
        for (from, to) in guard_failures {
            self.metrics.record_guard_failure(&from, &to);
        }

        if let Some(transition) = matching_transition {
            let result = self.execute_transition(&transition, context)?;
            return Ok(result);
        }

        // Check parent states for hierarchical FSM - collect parent stack first
        let parent_stack: Vec<StateId> = self.state_stack.iter().rev().cloned().collect();

        let parent_transition: Option<Transition<C>> = {
            let mut found = None;
            for parent in &parent_stack {
                if let Some(transitions) = self.transitions.get(parent) {
                    for transition in transitions {
                        if transition.config.event.as_ref() == Some(&event.event_type)
                            && transition.check_guard(context)
                        {
                            found = Some((*transition).clone());
                            break;
                        }
                    }
                }
                if found.is_some() {
                    break;
                }
            }
            found
        };

        if let Some(transition) = parent_transition {
            return self.execute_transition(&transition, context);
        }

        Ok(TransitionResult::NoTransition {
            current,
            event: event.event_type.clone(),
        })
    }

    /// Process all queued events.
    pub fn process_queue(
        &mut self,
        context: &mut C,
    ) -> Result<Vec<TransitionResult>, StateMachineError> {
        let mut results = Vec::new();

        while let Some(event) = self.event_queue.pop_front() {
            let result = self.process_event(&event, context)?;
            results.push(result);
        }

        Ok(results)
    }

    /// Tick the state machine (call each update cycle).
    ///
    /// This processes queued events, checks timeouts, and runs tick actions.
    pub fn tick(&mut self, context: &mut C) -> Result<(), StateMachineError> {
        if !self.is_running {
            return Err(StateMachineError::NotStarted);
        }

        // Process queued events
        let _ = self.process_queue(context)?;

        // Check for timeout
        if let Some(current) = self.current_state.clone() {
            if let Some(state) = self.states.get(&current) {
                if let (Some(timeout), Some(target), Some(entered_at)) = (
                    state.config.timeout,
                    state.config.timeout_target.clone(),
                    self.state_entered_at,
                ) {
                    if entered_at.elapsed() >= timeout {
                        log::info!(
                            "State '{}' timed out after {:?}, transitioning to '{}'",
                            current,
                            timeout,
                            target
                        );
                        self.force_transition(&target, context)?;
                        return Ok(());
                    }
                }

                // Execute tick action
                state.tick(context);
            }
        }

        Ok(())
    }

    /// Force a transition to a specific state (bypasses guards).
    pub fn force_transition(
        &mut self,
        target: &StateId,
        context: &mut C,
    ) -> Result<TransitionResult, StateMachineError> {
        if !self.states.contains_key(target) {
            return Ok(TransitionResult::StateNotFound(target.clone()));
        }

        let from = self
            .current_state
            .clone()
            .ok_or(StateMachineError::NotStarted)?;

        let start_time = Instant::now();

        // Exit current state
        self.exit_state(&from, context)?;

        // Enter new state
        self.enter_state(target, context)?;

        let duration = start_time.elapsed();
        self.metrics
            .record_transition(&from, target, duration.as_nanos() as u64);

        // Record in history
        self.record_transition(&from, target, None, duration);

        Ok(TransitionResult::Success {
            from,
            to: target.clone(),
            via_event: None,
        })
    }

    /// Execute a transition.
    fn execute_transition(
        &mut self,
        transition: &Transition<C>,
        context: &mut C,
    ) -> Result<TransitionResult, StateMachineError> {
        let from = transition.config.from.clone();
        let to = transition.config.to.clone();

        // Self-transition handling
        if from == to {
            // Execute exit then entry for self-transitions
            if let Some(state) = self.states.get(&from) {
                state.exit(context);
                transition.execute_actions(context);
                state.enter(context);
            }
            return Ok(TransitionResult::SelfTransition(from));
        }

        let start_time = Instant::now();

        // Exit current state
        self.exit_state(&from, context)?;

        // Execute transition actions
        transition.execute_actions(context);

        // Enter new state
        self.enter_state(&to, context)?;

        let duration = start_time.elapsed();
        self.metrics
            .record_transition(&from, &to, duration.as_nanos() as u64);

        // Record in history
        self.record_transition(&from, &to, transition.config.event.clone(), duration);

        log::debug!(
            "Transition: {} -> {} (event: {:?})",
            from,
            to,
            transition.config.event
        );

        Ok(TransitionResult::Success {
            from,
            to,
            via_event: transition.config.event.clone(),
        })
    }

    /// Enter a state.
    fn enter_state(
        &mut self,
        state_id: &StateId,
        context: &mut C,
    ) -> Result<(), StateMachineError> {
        // Clone children list and check for initial child to avoid borrow conflicts
        let (children, has_children) = {
            let state = self
                .states
                .get(state_id)
                .ok_or_else(|| StateMachineError::StateNotFound(state_id.clone()))?;
            (state.children.clone(), !state.children.is_empty())
        };

        // Track time in previous state
        if let (Some(prev), Some(entered_at)) = (&self.current_state, self.state_entered_at) {
            self.metrics
                .record_state_time(prev, entered_at.elapsed().as_nanos() as u64);
        }

        // Execute entry action
        if let Some(state) = self.states.get(state_id) {
            state.enter(context);
        }

        self.current_state = Some(state_id.clone());
        self.state_entered_at = Some(Instant::now());

        // Handle hierarchical states - enter default child if any
        if has_children {
            // Find initial child state
            let initial_child = children
                .iter()
                .find(|child_id| {
                    self.states
                        .get(*child_id)
                        .map(|c| c.config.is_initial)
                        .unwrap_or(false)
                })
                .cloned();

            if let Some(child_id) = initial_child {
                self.state_stack.push(state_id.clone());
                return self.enter_state(&child_id, context);
            }
        }

        Ok(())
    }

    /// Exit a state.
    fn exit_state(&mut self, state_id: &StateId, context: &mut C) -> Result<(), StateMachineError> {
        // Exit nested states first (for hierarchical FSM)
        while self
            .state_stack
            .last()
            .map(|s| s != state_id)
            .unwrap_or(false)
        {
            if let Some(nested) = self.state_stack.pop() {
                if let Some(state) = self.states.get(&nested) {
                    state.exit(context);
                }
            }
        }

        let state = self
            .states
            .get(state_id)
            .ok_or_else(|| StateMachineError::StateNotFound(state_id.clone()))?;

        state.exit(context);

        // Track time in state
        if let Some(entered_at) = self.state_entered_at {
            self.metrics
                .record_state_time(state_id, entered_at.elapsed().as_nanos() as u64);
        }

        Ok(())
    }

    /// Record a transition in history.
    fn record_transition(
        &mut self,
        from: &StateId,
        to: &StateId,
        event: Option<String>,
        duration: Duration,
    ) {
        let record = TransitionRecord {
            from: from.clone(),
            to: to.clone(),
            event,
            timestamp: Instant::now(),
            duration,
        };

        self.history.push_back(record);

        // Trim history if needed
        while self.history.len() > self.max_history_size {
            self.history.pop_front();
        }
    }

    /// Get the transition history.
    pub fn history(&self) -> &VecDeque<TransitionRecord> {
        &self.history
    }

    /// Get metrics for this state machine.
    pub fn metrics(&self) -> &StateMachineMetrics {
        &self.metrics
    }

    /// Get a state by ID.
    pub fn get_state(&self, id: &StateId) -> Option<&State<C>> {
        self.states.get(id)
    }

    /// Get all state IDs.
    pub fn state_ids(&self) -> Vec<StateId> {
        self.states.keys().cloned().collect()
    }

    /// Get all transitions from a state.
    pub fn get_transitions(&self, from: &StateId) -> Option<&Vec<Transition<C>>> {
        self.transitions.get(from)
    }

    /// Check if a state is a final state.
    pub fn is_final_state(&self, state: &StateId) -> bool {
        self.states
            .get(state)
            .map(|s| s.config.is_final)
            .unwrap_or(false)
    }

    /// Get the time spent in the current state.
    pub fn time_in_current_state(&self) -> Option<Duration> {
        self.state_entered_at.map(|t| t.elapsed())
    }

    /// Reset the state machine to initial state.
    pub fn reset(&mut self, context: &mut C) -> Result<(), StateMachineError> {
        self.stop(context)?;
        self.event_queue.clear();
        self.history.clear();
        self.metrics = StateMachineMetrics::new();
        self.start(context)
    }
}

impl<C> std::fmt::Debug for StateMachine<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StateMachine")
            .field("name", &self.name)
            .field("current_state", &self.current_state)
            .field("is_running", &self.is_running)
            .field("state_count", &self.states.len())
            .field(
                "transition_count",
                &self.transitions.values().map(|v| v.len()).sum::<usize>(),
            )
            .field("queue_size", &self.event_queue.len())
            .finish()
    }
}

/// Thread-safe wrapper around StateMachine.
///
/// Provides concurrent access to a state machine with interior mutability.
pub struct SharedStateMachine<C> {
    inner: Arc<RwLock<StateMachine<C>>>,
}

impl<C> SharedStateMachine<C> {
    /// Create a new shared state machine.
    pub fn new(machine: StateMachine<C>) -> Self {
        Self {
            inner: Arc::new(RwLock::new(machine)),
        }
    }

    /// Start the state machine.
    pub fn start(&self, context: &mut C) -> Result<(), StateMachineError> {
        self.inner.write().start(context)
    }

    /// Stop the state machine.
    pub fn stop(&self, context: &mut C) -> Result<(), StateMachineError> {
        self.inner.write().stop(context)
    }

    /// Queue an event.
    pub fn queue_event(&self, event: Event) -> Result<(), StateMachineError> {
        self.inner.write().queue_event(event)
    }

    /// Process an event.
    pub fn process_event(
        &self,
        event: &Event,
        context: &mut C,
    ) -> Result<TransitionResult, StateMachineError> {
        self.inner.write().process_event(event, context)
    }

    /// Tick the state machine.
    pub fn tick(&self, context: &mut C) -> Result<(), StateMachineError> {
        self.inner.write().tick(context)
    }

    /// Get the current state.
    pub fn current_state(&self) -> Option<StateId> {
        self.inner.read().current_state().cloned()
    }

    /// Check if running.
    pub fn is_running(&self) -> bool {
        self.inner.read().is_running()
    }

    /// Get metrics.
    pub fn metrics(&self) -> StateMachineMetrics {
        self.inner.read().metrics().clone()
    }

    /// Clone the Arc for sharing.
    pub fn clone_shared(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<C> Clone for SharedStateMachine<C> {
    fn clone(&self) -> Self {
        self.clone_shared()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestContext {
        value: i32,
        entry_count: u32,
        exit_count: u32,
    }

    fn create_test_fsm() -> StateMachine<TestContext> {
        StateMachine::new("test_fsm")
            .add_state(
                State::new("idle")
                    .initial()
                    .on_entry(|ctx: &mut TestContext| ctx.entry_count += 1)
                    .on_exit(|ctx: &mut TestContext| ctx.exit_count += 1),
            )
            .add_state(
                State::new("running")
                    .on_entry(|ctx: &mut TestContext| ctx.entry_count += 1)
                    .on_exit(|ctx: &mut TestContext| ctx.exit_count += 1),
            )
            .add_state(
                State::new("stopped")
                    .final_state()
                    .on_entry(|ctx: &mut TestContext| ctx.entry_count += 1),
            )
            .add_transition(Transition::new("idle", "running").on_event("start"))
            .add_transition(Transition::new("running", "idle").on_event("pause"))
            .add_transition(Transition::new("running", "stopped").on_event("stop"))
            .add_transition(
                Transition::new("idle", "running")
                    .on_event("force_start")
                    .with_guard(|ctx: &TestContext| ctx.value > 0),
            )
    }

    #[test]
    fn test_fsm_creation() {
        let fsm = create_test_fsm().build().unwrap();
        assert_eq!(fsm.name(), "test_fsm");
        assert!(!fsm.is_running());
    }

    #[test]
    fn test_fsm_start_stop() {
        let mut fsm = create_test_fsm().build().unwrap();
        let mut ctx = TestContext {
            value: 0,
            entry_count: 0,
            exit_count: 0,
        };

        fsm.start(&mut ctx).unwrap();
        assert!(fsm.is_running());
        assert_eq!(fsm.current_state(), Some(&StateId::named("idle")));
        assert_eq!(ctx.entry_count, 1); // Entered idle

        fsm.stop(&mut ctx).unwrap();
        assert!(!fsm.is_running());
        assert_eq!(ctx.exit_count, 1); // Exited idle
    }

    #[test]
    fn test_transitions() {
        let mut fsm = create_test_fsm().build().unwrap();
        let mut ctx = TestContext {
            value: 0,
            entry_count: 0,
            exit_count: 0,
        };

        fsm.start(&mut ctx).unwrap();

        // Transition idle -> running
        let result = fsm.process_event(&Event::new("start"), &mut ctx).unwrap();
        assert!(result.is_success());
        assert_eq!(fsm.current_state(), Some(&StateId::named("running")));
        assert_eq!(ctx.entry_count, 2); // idle + running
        assert_eq!(ctx.exit_count, 1); // exited idle

        // Transition running -> stopped
        let result = fsm.process_event(&Event::new("stop"), &mut ctx).unwrap();
        assert!(result.is_success());
        assert_eq!(fsm.current_state(), Some(&StateId::named("stopped")));
        assert!(fsm.is_final_state(&StateId::named("stopped")));
    }

    #[test]
    fn test_guard_conditions() {
        let mut fsm = create_test_fsm().build().unwrap();
        let mut ctx = TestContext {
            value: 0,
            entry_count: 0,
            exit_count: 0,
        };

        fsm.start(&mut ctx).unwrap();

        // Guard should fail (value = 0)
        let result = fsm
            .process_event(&Event::new("force_start"), &mut ctx)
            .unwrap();
        match result {
            TransitionResult::NoTransition { .. } => {}
            _ => panic!("Expected NoTransition, got {:?}", result),
        }

        // Update context to pass guard
        ctx.value = 10;
        let result = fsm
            .process_event(&Event::new("force_start"), &mut ctx)
            .unwrap();
        assert!(result.is_success());
    }

    #[test]
    fn test_no_initial_state() {
        let fsm = StateMachine::<TestContext>::new("no_initial")
            .add_state(State::new("state1"))
            .add_state(State::new("state2"));

        let result = fsm.build();
        assert!(matches!(result, Err(StateMachineError::NoInitialState)));
    }

    #[test]
    fn test_event_queue() {
        let mut fsm = create_test_fsm().build().unwrap();
        let mut ctx = TestContext {
            value: 0,
            entry_count: 0,
            exit_count: 0,
        };

        fsm.start(&mut ctx).unwrap();

        // Queue multiple events
        fsm.queue_event(Event::new("start")).unwrap();
        fsm.queue_event(Event::new("pause")).unwrap();

        // Process queue
        let results = fsm.process_queue(&mut ctx).unwrap();
        assert_eq!(results.len(), 2);

        // Should be back in idle
        assert_eq!(fsm.current_state(), Some(&StateId::named("idle")));
    }

    #[test]
    fn test_force_transition() {
        let mut fsm = create_test_fsm().build().unwrap();
        let mut ctx = TestContext {
            value: 0,
            entry_count: 0,
            exit_count: 0,
        };

        fsm.start(&mut ctx).unwrap();

        // Force transition to stopped
        let result = fsm
            .force_transition(&StateId::named("stopped"), &mut ctx)
            .unwrap();
        assert!(result.is_success());
        assert_eq!(fsm.current_state(), Some(&StateId::named("stopped")));
    }

    #[test]
    fn test_metrics() {
        let mut fsm = create_test_fsm().build().unwrap();
        let mut ctx = TestContext {
            value: 0,
            entry_count: 0,
            exit_count: 0,
        };

        fsm.start(&mut ctx).unwrap();
        fsm.process_event(&Event::new("start"), &mut ctx).unwrap();
        fsm.process_event(&Event::new("stop"), &mut ctx).unwrap();

        let metrics = fsm.metrics();
        assert_eq!(metrics.successful_transitions, 2);
        assert_eq!(metrics.events_processed, 2);
    }

    #[test]
    fn test_history() {
        let mut fsm = create_test_fsm().build().unwrap();
        let mut ctx = TestContext {
            value: 0,
            entry_count: 0,
            exit_count: 0,
        };

        fsm.start(&mut ctx).unwrap();
        fsm.process_event(&Event::new("start"), &mut ctx).unwrap();

        assert_eq!(fsm.history().len(), 1);
        let record = fsm.history().front().unwrap();
        assert_eq!(record.from, StateId::named("idle"));
        assert_eq!(record.to, StateId::named("running"));
        assert_eq!(record.event, Some("start".to_string()));
    }
}
